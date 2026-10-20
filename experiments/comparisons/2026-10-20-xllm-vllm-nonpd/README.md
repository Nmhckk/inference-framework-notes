# xLLM 与 vLLM-Ascend 非 PD 对比：GLM-5.3-Flash（2026-10-20）

## 结论

**当前不能得出“xLLM 比 vLLM 好”。** 已验证的 xLLM 四实例非 PD 结果为 82.3239 output token/s；用户提供的 vLLM 非 PD 参考结果为 426.77 output token/s。若两组确实使用同一模型、四台 910C、32 个逻辑设备和同一数据集，vLLM 的输出吞吐是 xLLM 的 **5.18 倍**，TTFT 更低，TPOT 也低一个数量级。

vLLM 记录仍缺服务端启动命令、框架提交、实际设备数和原始结果文件，因此只能作为方向性对比，不能写成严格的框架排名。xLLM 的 48 条请求原始结果、运行参数和缓存/MTP 计数均已保存，证据强度高于只有截图的 vLLM 记录。

| 指标 | xLLM 四实例非 PD | vLLM 非 PD 参考 | 方向性差距 |
| --- | ---: | ---: | ---: |
| TTFT 均值 | 23.268 s | 10.624 s | xLLM 为 2.19× |
| TPOT 均值 | 144.600 ms | 16.222 ms | xLLM 为 8.91× |
| 输出吞吐 | 82.3239 token/s | 426.77 token/s | vLLM 为 5.18× |
| 请求数 / 成功数 | 48 / 48 | 未提供 | 不可核对 |
| 设备数 | 4×TP8，32 NPU | 未提供 | 不可归一化 |

若 vLLM 同样使用 32 个逻辑设备，则条件性归一化结果为 xLLM 2.573、vLLM 13.337 output token/s/NPU。这个换算只用于指出需要核对资源，不能替代同机复测。

## 为什么当前 xLLM 没有跑赢

### 1. xLLM 的 MTP 会关闭 P/D 混批

本次 xLLM 参数同时包含 `--num_speculative_tokens=1`、`--enable_mix_batch=false` 和 `--enable_schedule_overlap=true`。xLLM 的策略工厂还会在 MTP 大于 0 时强制把 `enable_mix_batch` 设为 false，随后选择 `PrefillFirstPolicy`：

- [MTP/CP/KV 分片会关闭混批](https://github.com/xLLM-AI/xllm/blob/ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb/xllm/core/scheduler/scheduler_policy.cpp#L1061-L1091)
- [关闭混批时选择 PrefillFirstPolicy](https://github.com/xLLM-AI/xllm/blob/ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb/xllm/core/scheduler/scheduler_policy.cpp#L1096-L1105)

因此这里的 `schedule_overlap` 是“当前 NPU step 与上一批结果处理/下一批准备重叠”，并不表示 Prefill token 和 Decode token 会进入同一个执行批次。实现中的 producer-consumer 流水见 [continuous_scheduler.cpp](https://github.com/xLLM-AI/xllm/blob/ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb/xllm/core/scheduler/continuous_scheduler.cpp#L525-L597)。在 80K 长输入场景，即使前缀命中约 90%，剩余 Prefill chunk 仍会与 Decode 争用实例，并可能造成 Decode 间隔拉长。

vLLM V1 调度器不把请求硬编码成互斥的 Prefill/Decode 阶段，而是用 `num_computed_tokens` 和统一 token budget 表达 chunked prefill、prefix cache 与 speculative decode；每一步先处理 running requests，再分配剩余预算。这套控制流见 [scheduler.py](https://github.com/vllm-project/vllm/blob/fc8e08a2a607968226376c53a30cfa6337578f79/vllm/v1/core/sched/scheduler.py#L591-L711) 和 [KV 分配/抢占](https://github.com/vllm-project/vllm/blob/fc8e08a2a607968226376c53a30cfa6337578f79/vllm/v1/core/sched/scheduler.py#L775-L858)。它更容易在长 Prefill 下保持 Decode 连续性，但最终效果仍取决于模型后端和参数。

### 2. xLLM 当前为 eager，vLLM 公开配置启用了 Decode 图

xLLM 的图模式候选曾把本项目短基准从 22.10 提到 34.90 token/s，但压力测试后 1K 检索回归为 0/4，故不能作为有效结果，最终稳定配置为 `--enable_graph=false`。

vLLM-Ascend 的 GLM-5.3-Flash 官方配置使用 `FULL_DECODE_ONLY`，明确用于降低逐 token Decode 的 kernel launch 开销；A3 配置还让 MTP draft 进入图模式。[官方部署参数与说明](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/GLM5.3-Flash.md#online-service-deployment)

Decode 是大量形状相近的小步执行。可把单步时间粗略拆为：

```text
T_step = T_operator + T_collective + T_launch + T_host_gap
```

图模式主要压低 `T_launch + T_host_gap`。xLLM 当前每个 Decode step 都走 eager，而 vLLM 参考配置很可能使用 Decode 图，因此这比调度器语言是 C++ 还是 Python 更能直接影响 TPOT。

### 3. MTP 深度不对齐

xLLM 的 GLM-5.3-Flash 文档说明当前仅支持一个 NextN 层；本次也只开了一个 speculative token。[xLLM MTP 权重加载与限制](https://github.com/xLLM-AI/xllm/blob/ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb/docs/src/content/docs/en/cookbook/autoregressive_models/glm/glm_5_3_flash.md#load-mtp-weights-directly)

本次原生日志中的 MTP1 接受率为 86.82%。在忽略边界和拒绝修复开销的近似下，每次 target verify 的期望提交量为：

```text
E[committed tokens / verify] ≈ 1 + acceptance = 1.8682
```

vLLM-Ascend 的公开 A2 非 PD 示例使用 MTP3，A3 示例使用 MTP5；其 KDA 实现允许最多七个 speculative tokens，并把 Q/K/V/b/f_a/g_a 六个投影合成一次 GEMM。[KDA MTP 上限与 6→1 投影融合](https://github.com/vllm-project/vllm-ascend/blob/1026eedfd2a13662d54f78e1c45566989a7157af/vllm_ascend/models/glm5next/kda.py#L149-L186)

若接受率足够高，MTP3 每次 verify 最多提交 4 token，MTP5 最多提交 6 token；代价是 draft 和 verify 计算增加。没有 vLLM 原生日志中的接受率，不能计算它的真实 MTP 收益，但 MTP1 与 MTP3/5 本身已不是同一算法配置。

### 4. vLLM 的 GLM 专用图与混合缓存路径更完整

GLM-5.3-Flash 同时包含 KDA recurrent/conv state、MLA KV 和 KeyPool/indexer state，不能只用普通 PagedAttention 解释性能。vLLM-Ascend 已给 KeyPool tail metadata 声明 uniform-batch 图支持，避免该缓存组把主模型的 `FULL_DECODE_ONLY` 降级。[indexer_kpool.py](https://github.com/vllm-project/vllm-ascend/blob/1026eedfd2a13662d54f78e1c45566989a7157af/vllm_ascend/attention/indexer_kpool.py#L321-L333)

xLLM 的通用 xTensor 虚拟连续 KV 是有价值的设计，但当前代码明确拒绝 index cache、conv cache、SSM cache 和 linear attention，因此 **xTensor 优势不能用于解释本次 GLM-5.3-Flash**。[kv_cache.cpp](https://github.com/xLLM-AI/xllm/blob/ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb/xllm/core/kv_cache/storage/kv_cache.cpp#L347-L363)

本项目尝试的 Python 侧 KPool 私有池复用也未形成端到端收益：3P1D 从 22.10 降到 20.05 token/s，已关闭。因而当前 82.32 token/s 不能包含“KPool 优化已经生效”的表述。

### 5. 两边使用的执行参数很可能不是同一档

xLLM 当前为每实例 `max_seqs=8`、MTP1、eager、无 P/D 混批、显存利用率 0.80。vLLM-Ascend 官方 A2 非 PD 示例是每节点 DP rank + TP8，`max_num_seqs=32`、MTP3、Decode 图、EP 和 0.85 显存利用率；A3 示例还启用 MTP5 和 shared-expert multistream overlap。[GLM-5.3-Flash 官方教程](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/GLM5.3-Flash.md)

这些差异会同时改变批量上限、每步提交 token 数、Host launch 开销、MoE 通信重叠和可用缓存容量。当前差距更像“成熟优化组合的差距”，还不是 `xLLM scheduler` 对 `vLLM scheduler` 的单变量结论。

## xLLM 真正有依据的优势

### 已在本项目生效

1. **C++ 调度和 Host/NPU 流水已生效。** `schedule_overlap` 的 producer-consumer 路径将当前 step 与上一批输出处理重叠。它降低 Host 空泡，但无法弥补 eager Decode 算子/图路径的差距。
2. **MTP1 与长前缀缓存组合已稳定跑完性能负载。** 48/48 请求完成，输出长度均为 600；日志记录 MTP 接受率 86.82%，前缀命中 token 比例约 89.99%。这是当前最强的正向实证。
3. **四实例路由均衡。** 含一条主预热请求的分布为 13/12/12/12，说明 service router 没有把负载集中到单机。它解释了四实例能获得 3P1D 的 3.72 倍输出吞吐。
4. **原权重可直接加载 MTP draft。** 同一 GLM-5.3-Flash W8A8 目录同时作为 target 与 draft，无需改写权重，部署路径较短。

### 框架层面有潜力，但本轮没有转化成 GLM 性能优势

1. **SLO 感知的服务层。** xLLM-Service 的源码根据预测 TPOT 选择 Decode 实例、根据预测 TTFT 选择 Prefill 实例，并可在负载变化时切换角色。[instance_mgr.cpp](https://github.com/xLLM-AI/xllm-service/blob/b232b7094f17fbab68f010290554b012ee384a0b/xllm_service/scheduler/managers/instance_mgr.cpp#L915-L1028) 这对混合负载和动态 PD 有研究价值，但本轮是四个固定 DEFAULT 实例，主要只用了负载分发。
2. **可替换的调度策略。** xLLM 明确实现 Prefill-first、Decode-first 和 unified 多 SLO 策略，便于做策略实验；本轮因 MTP 限制只走 Prefill-first。
3. **xTensor 的连续虚拟地址。** 理论上可兼顾动态物理页分配与连续算子接口，减少普通 Attention 路径的页表访问。当前 GLM 混合状态不支持，需换普通 Attention 模型验证。
4. **论文中的其他模型结果。** xLLM 技术报告在 Qwen3、2048/2048、TPOT 50 ms 约束下报告 910C 吞吐最高为 vLLM-Ascend 的 2.2 倍。[xLLM Technical Report](https://arxiv.org/abs/2510.14686) 该结果使用旧版 vLLM-Ascend，且模型、长度和优化路径均不同，不能外推到 GLM-5.3-Flash。

## 理论上如何判断谁更好

非 PD 下，Prefill 与 Decode 共享同一实例。单实例 Decode 吞吐可近似写为：

```text
decode_tps ≈ active_sequences × E[committed_tokens_per_step] / T_step
```

其中：

- `active_sequences` 受 KV/混合状态容量和调度准入限制；
- `E[committed_tokens_per_step]` 主要受 MTP 深度和接受率影响；
- `T_step` 由模型算子、TP/EP 通信、图 launch、Host 间隙及 Prefill 干扰组成。

xLLM 当前的前缀命中和 MTP 接受率都较好，但只有 MTP1，图关闭，且 MTP 强制关闭 P/D 混批。这三个限制分别压低分子、抬高分母并增加 Decode 干扰。其单用户生成速度约为 `1000 / 144.6 = 6.92 token/s`；vLLM 参考为 `1000 / 16.222 = 61.64 token/s`。当前主要瓶颈已经落在 Decode step，而不是请求路由或前缀命中。

## 可以正式采用的项目判断

- **GLM-5.3-Flash、80K/600、并发 16、非 PD：当前选择 vLLM-Ascend 作为性能领先候选，xLLM 作为稳定可运行且便于改调度代码的研究基线。**
- xLLM 的优势应表述为 C++ 调度流水、集成式 SLO 路由、策略可替换和国产硬件适配潜力；不能表述为当前 GLM 性能更快。
- vLLM 当前领先最可能来自 Decode 图、较深 MTP、成熟的 GLM KeyPool/KDA 路径和更高批量，而不是单一 PagedAttention 机制。
- 要形成严格排名，必须补齐 vLLM 的启动命令、框架/镜像版本、32 NPU 资源证明、MTP 接受率、prefix hit、成功请求数及原始 AISBench JSON，并使用与 xLLM 完全相同的数据集哈希。

## 证据

- [xLLM 四实例非 PD 完整结果](../../xllm/2026-10-20-nonpd/README.md)
- [xLLM 原始性能摘要](../../xllm/2026-10-20-nonpd/performance-summary.json)
- [xLLM 原生缓存与 MTP 计数](../../xllm/2026-10-20-nonpd/native-runtime-evidence.json)
- [vLLM PagedAttention 论文](https://arxiv.org/abs/2309.06180)
- [vLLM-Ascend GLM-5.3-Flash 教程](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/GLM5.3-Flash.md)
- [vLLM-Ascend GLM-5.3-Flash 适配路线与未完成验证](https://github.com/vllm-project/vllm-ascend/issues/15665)
- [xLLM 技术报告](https://arxiv.org/abs/2510.14686)

