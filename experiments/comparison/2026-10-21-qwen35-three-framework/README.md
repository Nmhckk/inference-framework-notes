# Qwen3.5-4B：xLLM、SGLang、vLLM 同卡对照

## 结论

在同一台 Ascend 910C-64G 的同一张物理卡、同一份 Qwen3.5-4B BF16 权重、同一数据和客户端下，2033–2034 输入、2048 输出、48 请求、16 并发的三轮结果为：

| 配置 | 输出吞吐（tok/s） | TTFT P90（s） | TPOT 均值（ms） |
| --- | ---: | ---: | ---: |
| **xLLM，Graph 关，Overlap 开** | **1099.10 ± 0.91** | **1.313** | **14.118** |
| SGLang，Decode Full Graph，Overlap 开 | 1057.02 ± 14.49 | 2.381¹ | 14.371 |
| vLLM-Ascend rc2，FULL_DECODE_ONLY，Async 开 | 951.61 ± 1.54 | 1.530 | 16.304 |
| SGLang，Decode Full Graph，Overlap 关 | 937.22 ± 9.13 | 2.318² | 16.455 |

每个框架均为 3 轮、每轮 48/48 请求成功并完整输出 2048 token。相对 SGLang 三轮均值，xLLM 吞吐高 **3.98%**、TPOT 低 **1.76%**；SGLang 首轮存在冷态 TTFT 扰动，若只看稳定的第 2、3 轮，xLLM 吞吐仍高 **3.16%**。相对当前 vLLM-Ascend rc2，xLLM 吞吐高 **15.50%**、TPOT 低 **13.40%**。

¹ SGLang Overlap 开启的第 1 轮 TTFT P90 为 3.837 秒，第 2、3 轮为 1.659/1.648 秒；三轮 P90 均值受首轮冷态影响。  
² SGLang Overlap 关闭的第 3 轮 TTFT 受 Host 抖动影响，吞吐也由前两轮 941–944 降至 927 tok/s；本表不删除该轮。

## 已经证实的机制

### 1. SGLang 的 Overlap 是主要性能来源

只改变 `--disable-overlap-schedule`，Decode Graph、模型、卡和客户端均不变：

- 吞吐 937.22 → 1057.02 tok/s，**+12.78%**；
- TPOT 16.455 → 14.371 ms，**-12.66%**。

[SGLang scheduler](https://github.com/sgl-project/sglang/blob/e9b22a0818292eedc0630b2b72cdfcb4ae81bf85/python/sglang/srt/managers/scheduler.py)默认启用 event-loop overlap，在设备执行当前 batch 时处理上一 batch 的结果和下一批调度。它也解释了 SGLang 相对本轮 vLLM 基线的 **11.08%** 吞吐优势。xLLM 自身的 Schedule Overlap 消融为 **+3.46%**；两者实现方式不同，不能把百分比直接等同为调度器优劣。

### 2. xLLM 对 SGLang 的领先不是“独占融合算子”

xLLM 日志确认 8 个 Full-Attention 层使用 `split_qkv_rmsnorm_mrope`。继续核查后发现，本次 SGLang 镜像提交 `e9b22a081` 的 [Qwen3.5 NPU Decode 路径](https://github.com/sgl-project/sglang/blob/e9b22a0818292eedc0630b2b72cdfcb4ae81bf85/python/sglang/srt/models/qwen3_5.py#L1541-L1559)同样调用 `split_qkvgate_gemma_rmsnorm_rope`，并成功捕获 batch size 1/2/4/8/12/16 的 Decode Full Graph。

因此，当前能确认的是：两个框架都具备 Qwen3.5 专用融合、Decode 图和异步调度时，xLLM 在这组 Decode 较长的负载下仍有约 **3%–4%** 的端到端优势。剩余差异可能来自 GDN kernel、状态缓存布局、Host 发射和服务链路，但尚无逐算子 profiler，不能继续拆给单一模块。

### 3. xLLM 对当前 vLLM 镜像的性能优势成立，算子归因仍缺运行证据

xLLM 相对固定镜像 `v0.30.0rc2-a3` 的 **15.50%** 端到端吞吐优势由三轮逐请求数据支持；xLLM 日志也证明 fused QKV/RMSNorm/mRoPE NPU kernel 和 Schedule Overlap 生效。不过归档的 `vllm-server.log` 是空文件，无法从当前证据包独立复核此前观察到的 `enable_qk_norm_rope_fusion=False` 与 NPU `causal_conv1d_update` fallback。因此这两项降级为待补运行证据，暂不用于定量解释 15.50% 差距。

vLLM-Ascend 当前主线源码仍记录了同步 PyTorch causal-conv fallback 的风险，但主线不能替代被测镜像的源码 SHA 或服务日志。详细证据分级见[代码级性能归因](../2026-10-21-qwen35-code-attribution/README.md)。

### 4. SGLang 在 910C 上存在默认配置兼容门槛

SGLang 按模型配置默认使用 FP32 Mamba/GDN state，首个请求报错：`FP32 initial_state is only supported on Ascend950 (arch35)`。显式增加 `--mamba-ssm-dtype bfloat16` 后，功能门禁 4/4 和全部性能请求通过。它是当前镜像在 910C 上的部署风险；正确配置后的性能则接近 xLLM。

## 测试条件

- 主机：`bm-jn-zs-zone1-910C-64G-10-171`，三框架顺序独占物理 NPU 0；只操作带 `hzy.codex.owner=inference-framework` 标签的测试容器，测试后已删除，K8s 未改。
- 模型：同一份 Qwen3.5-4B BF16 HF 权重；实际输入 2033–2034 token，固定输出 2048 token。
- 客户端：OpenAI streaming，48 请求、并发 16、`request_rate=0`、2 个 warmup、`temperature=0`、`ignore_eos=true`；Prefix Cache 关闭。
- xLLM：0.11.0，源码快照 `53f22bb603556dfb933f3f14a993cfbd20ce2e26`；Graph 关闭、Schedule Overlap 开启。
- SGLang：`0.5.23.dev315+ge9b22a081`，镜像 `quay.io/ascend/sglang:main-cann9.1.0-a3`；Decode Full Graph、Overlap、BF16 state 开启，Radix Cache 关闭。
- vLLM-Ascend：镜像 `quay.io/ascend/vllm-ascend:v0.30.0rc2-a3`，运行时 `0.30.0+empty`；Async Scheduling 和 `FULL_DECODE_ONLY` 开启。
- 三者均未使用 MTP、PD、量化或 Prefix Cache。

客户端验证了 API 状态、实际 token 数、流式时序和输出摘要；这属于性能与基本输出门禁，不替代任务级精度评测。

## 对项目方向的含义

- **xLLM 已验证优势**：在 Qwen3.5 Decode 较长、单卡饱和负载中，当前 xLLM 集成的模型专用算子和轻量 runtime 优于所测 vLLM 版本，并小幅领先同样高度优化的 SGLang。
- **SGLang 值得吸收的设计**：本轮 Overlap 独立贡献 12.78%；Radix Cache 被刻意关闭，因此还没有测到它在重复前缀、Agent 和多轮场景的核心优势。
- **vLLM 的价值**：功能面、配置体系和持续优化速度强；此前 80K/600 长输入测试中它仍领先 xLLM。本轮结果说明框架胜负依赖模型、版本与负载区间。
- **自研框架候选组合**：采用 xLLM 风格的模型专用 NPU kernel 与紧凑执行链路，同时吸收 SGLang 的 overlap 与 radix-aware 调度，再保留 vLLM 的统一 token budget、可观测性和广泛模型适配。

## 证据与复现

- [三框架聚合摘要](summary.csv)与[完整 JSON](comparison-summary.json)
- [聚合脚本](summarize.py)
- [SGLang 启动参数](launch_sglang.sh)；追加 `--disable-overlap-schedule` 即为消融配置
- [SGLang 完整证据包](sglang-qwen35-evidence.tar.gz)，SHA-256：`e2a9437c465ecaeb17704200811bccd4a9676493d8e6adbab94186ab7e8f479e`
- xLLM/vLLM 的启动配置、三轮逐请求结果和日志沿用[上一组同卡证据](../2026-10-21-qwen35-paper-slo/README.md)

证据包包含 SGLang 六轮逐请求结果、服务与错误日志、容器配置、运行时版本、固定提交源码、启动脚本及测试前后 NPU 状态。下一步用 profiler 比较三框架的 Host gap、GDN 时间与图 replay，再做 offered-QPS/TPOT-SLO 曲线，避免只凭单个饱和点归因。
