# xLLM 与 vLLM-Ascend 非 PD 首轮对照（2026-10-20）

## 结论与更正

用户提供的 vLLM 数据（TTFT 10.624 秒、TPOT 16.222 毫秒、输出 426.77 token/s）实际启用了 PD 分离，不能与 xLLM 四实例非 PD 结果计算倍率，也不能用于判断非 PD 下哪个框架更快。此前基于该截图计算的 2.19×、8.91×、5.18× 差距全部撤回。

同日已完成严格程度更高的同机复测：两边使用同一台主机、同一份权重、同一批数据、16个逻辑设备和同一个客户端；xLLM 为两个 TP8 副本，vLLM 为 DP2×TP8。两边均48/48成功，当前已验证性能配置下 vLLM 输出吞吐为 xLLM 的3.95倍，xLLM TTFT/TPOT分别为 vLLM 的2.96/4.77倍。xLLM 在压测结束约12分钟后 API 无响应并需要重启，故稳定性门禁未通过。[完整同机结果](../../xllm/2026-10-20-samehost-tp8x2/README.md)。

这仍是“最佳已验证配置”对比而非纯框架内核消融：vLLM 启用INT8 KV、Decode图和MTP5，xLLM使用BF16/auto KV、eager和MTP1。差距说明这些能力在当前GLM路径中的可用性和成熟度不同，不能直接归因于某一个组件。

## 网上可以采用的 vLLM 证据

### 官方非 PD 配置

vLLM-Ascend 的 GLM-5.3-Flash 教程基于 `v0.30.0RC1`。A2 示例是两节点共置式非 PD：每节点一个 DP rank、每 rank TP8，即 DP2×TP8；主要优化为：

- `--enable-expert-parallel`
- `--speculative-config` 使用 DeepSeek MTP3
- `FULL_DECODE_ONLY` 图模式
- `--max-num-seqs 32`
- `--max-num-batched-tokens 8192`
- W8A8 使用 `--quantization ascend`

来源：[vLLM-Ascend GLM-5.3-Flash 官方教程](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/GLM5.3-Flash.md#52-multi-node-colocated-deployment)。该教程给出了可执行配置，但没有公开对应吞吐、TTFT 和 TPOT。

### 公开性能数据的边界

- [KPool TopK 性能问题 #18484](https://github.com/vllm-project/vllm-ascend/issues/18484) 给出 4096/2048、并发80、3440.50 output token/s、17.7 ms TPOT，但拓扑为 DP2TP8 Prefill + DP8TP2 Decode，属于 PD，不能用于本次非 PD 对照。
- [动态 slot-mapping 问题 #17527](https://github.com/vllm-project/vllm-ascend/issues/17527) 提到 GLM-5.3-Flash 并发32修复后 P99 ITL 约120 ms，但没有完整负载、设备数和输出吞吐，只能说明运行时重编译会显著伤害尾延迟。
- [A2 长上下文问题 #15984](https://github.com/vllm-project/vllm-ascend/issues/15984) 使用单节点 TP8+EP，但报告的是旧镜像长请求挂起问题，没有可用性能基线。
- NVIDIA/AMD 上的公开结果硬件、量化和算子后端均不同，不用于昇腾框架排名。

## 代码和理论判断

非 PD 的近似 Decode 吞吐为：

```text
decode_tps ≈ active_sequences × E[committed_tokens_per_step] / T_step
T_step = T_operator + T_collective + T_launch + T_host_gap + T_prefill_interference
```

vLLM 当前优势来自已有运行证据的实现组合：

1. 本轮 5-token MTP 比 xLLM 当前 MTP1 有更高的每步提交 token 上限；实测 draft token 接受率为51.49%。
2. `FULL_DECODE_ONLY` 可减少逐 token 的 Host/Kernel launch 开销；xLLM 当前有效结果关闭图模式。
3. vLLM-Ascend 已为 GLM 的 KDA、KeyPool、NoPE sparse MLA 和混合缓存提供专用实现，适配路线见 [#15665](https://github.com/vllm-project/vllm-ascend/issues/15665)。
4. vLLM 统一 token budget 调度 chunked prefill、prefix cache 和 Decode；xLLM 当前 MTP 会关闭 P/D 混批并选择 Prefill-first，见 [xLLM scheduler_policy.cpp](https://github.com/xLLM-AI/xllm/blob/ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb/xllm/core/scheduler/scheduler_policy.cpp#L1061-L1105)。

首轮实测支持这些机制的组合效果，但还需在同机上做逐项消融，才能分摊每一项的独立收益。

## 首轮非 PD 实测

2026-10-20 已在一台空闲 910C 服务器上完成 vLLM-Ascend 非 PD 混部实测：单机16个逻辑设备、DP2×TP8、W8A8 + INT8 KV + prefix cache + 5-token MTP + `FULL_DECODE_ONLY` 图。48/48请求成功，TTFT均值17.577秒、TPOT均值49.547毫秒、输出吞吐185.338 token/s。[完整结果](../../vllm/2026-10-20-nonpd-run/README.md)。

与 xLLM 四实例非 PD 对比，vLLM 在只使用一半活跃逻辑设备的情况下，集群输出吞吐为2.25倍、单活跃逻辑设备吞吐为4.50倍，TTFT降低24.5%，TPOT降低65.7%。两者的数据集、长度、请求数、并发和请求速率相同，但服务器与客户端尚未完全统一，因此定位为强预备结论，不把全部差距归因于单一机制。

## 后续消融

同机与统一客户端已经完成。下一轮在同一拓扑上依次关闭 vLLM 的图、MTP、INT8 KV 和前缀缓存，并在 xLLM 输出正确性通过后重测图路径，用消融实验确定每个机制的独立收益。

必须保存镜像 digest、vLLM/vLLM-Ascend/CANN 版本、权重哈希、设备映射、启动日志、MTP 接受率、prefix hit、原始请求结果和 NPU 利用率，之后才能给出框架结论。

## 证据

- [xLLM 四实例非 PD 完整结果](../../xllm/2026-10-20-nonpd/README.md)
- [xLLM 与 vLLM 同机对照](../../xllm/2026-10-20-samehost-tp8x2/README.md)
- [xLLM 原始性能摘要](../../xllm/2026-10-20-nonpd/performance-summary.json)
- [xLLM 原生缓存与 MTP 计数](../../xllm/2026-10-20-nonpd/native-runtime-evidence.json)
- [vLLM 非 PD 完整结果](../../vllm/2026-10-20-nonpd-run/README.md)
- [vLLM 结构化性能摘要](../../vllm/2026-10-20-nonpd-run/performance-summary.json)
- [vLLM-Ascend GLM-5.3-Flash 教程](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/GLM5.3-Flash.md)
- [vLLM-Ascend GLM-5.3-Flash 适配路线](https://github.com/vllm-project/vllm-ascend/issues/15665)

