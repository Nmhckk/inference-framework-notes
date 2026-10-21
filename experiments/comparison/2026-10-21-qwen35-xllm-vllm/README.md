# Qwen3.5-4B：xLLM 与 vLLM-Ascend 同机实测

日期：2026-10-21

## 结论

Qwen3.5 是当前更适合继续评价 xLLM 的模型族：官方有独立部署指南，Ascend 路径覆盖混合 GDN/全注意力、ACL Graph、Schedule Overlap 和 Qwen3.5 MTP；源码中还有独立的模型、GDN 元数据和 NPU 层实现。本轮先选 4B BF16 单卡版，以排除 TP/HCCL 通信差异。

在同一台 910C 64GB、同一张 NPU、同一份权重和同一请求流下，vLLM-Ascend 的总吞吐和 Decode 延迟更好；xLLM 仅在 TTFT P90 上领先。本轮不能得出“xLLM 整体更快”的结论。

| 框架 | 成功请求 | 输出吞吐 | TTFT mean | TTFT P50 | TTFT P90 | TPOT mean | TPOT P50 | TPOT P90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| xLLM 0.11.0 | 48/48 | 65.52 tok/s | 27.95 s | 17.89 s | **65.61 s** | 192.28 ms | 220.05 ms | 222.57 ms |
| vLLM-Ascend 0.30.0rc2 | 48/48 | **69.12 tok/s** | **25.94 s** | **12.86 s** | 69.62 s | **183.04 ms** | **208.88 ms** | **209.41 ms** |
| xLLM 相对 vLLM | — | -5.21% | +7.74% | +39.14% | **-5.75%** | +5.05% | +5.35% | +6.28% |

低延迟指标越低越好。xLLM 的 TTFT P90 优势来自首批长 Prefill 请求更早得到服务；进入稳态后，vLLM 的 Decode 更快，因此最终吞吐高 5.5%。

## 测试条件

- 服务器：`bm-jn-zs-zone1-910C-64G-10-171`
- 设备：Ascend 910C 64GB，物理 NPU 0，单卡 TP1
- 模型：Qwen3.5-4B BF16，约 8.6 GiB 权重
- 输入：48 条请求，实际 81,798～82,558 token
- 输出：每条固定 600 token，`ignore_eos=true`
- 并发：16；到达率：0.93 request/s
- Prefix Cache：两侧关闭
- Prefill chunk：两侧 8,192 token
- Decode 图：两侧捕获 1/2/4/8/16 batch
- xLLM：ACL Graph、Schedule Overlap、Chunked Prefill
- vLLM：`FULL_DECODE_ONLY`、Async Scheduling、Chunked Prefill

服务启动前进行了短请求和两条 80K 请求预热；两侧输出均可读，正式请求均返回 HTTP 200 且生成满 600 token。测试结束后已删除本项目容器，未修改 K8s 任务。

## xLLM 目前能确认的优点

### 1. 高 KV 压力下的 Zero-Evict 接纳策略

xLLM 的 `BlockCapacityGuard` 在接纳 Prefill 请求前模拟后续 Decode 的块需求；若未来容量不足，就暂缓新请求，而不是先接纳再驱逐运行中的请求。对应实现位于 [`zero_eviction_scheduler.cpp`](https://github.com/xLLM-AI/xllm/blob/b612303512c9d4c7304f7d558f250cf0c3e88918/xllm/core/scheduler/zero_eviction_scheduler.cpp#L199-L250) 和[接纳判断](https://github.com/xLLM-AI/xllm/blob/b612303512c9d4c7304f7d558f250cf0c3e88918/xllm/core/scheduler/zero_eviction_scheduler.cpp#L317-L326)。官方给出的 Qwen3-8B 结果是在 E2E SLO 下 TPOT 降低 27%，但这属于官方数据，本轮尚未独立复现。

当前 vLLM 调度器在 KV 分配失败时仍可能选择运行中请求进行 preempt，再重新放回等待队列，见 [`scheduler.py`](https://github.com/vllm-project/vllm/blob/e632d66433e3e4fe064420bd12973f31e75aa84e/vllm/v1/core/sched/scheduler.py#L830-L903)。不过 vLLM 已加入 KV watermark，见 [`kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/e632d66433e3e4fe064420bd12973f31e75aa84e/vllm/v1/core/kv_cache_manager.py#L205-L208)，所以不能把它描述成“完全没有防抢占机制”。

### 2. C++ 调度重叠路径更直接

xLLM 使用 C++ thread pool、future/promise 和上一拍输出，把第 i+1 步输入准备与第 i 步设备执行重叠，具体代码见 [`worker_impl.cpp`](https://github.com/xLLM-AI/xllm/blob/b612303512c9d4c7304f7d558f250cf0c3e88918/xllm/core/runtime/worker_impl.cpp#L1846-L1903)。官方文档报告设备步间空隙约 200 μs，并在一个小型 Qwen 模型上得到 17% 吞吐提升。

这项能力在本轮已确认启用，但没有转化为整体胜出。原因是 80K Prefill 和模型算子占主导，框架调度气泡只占总成本的一部分。

### 3. 本轮 KV 可用空间略高

xLLM 服务日志报告 45.02 GiB KV Cache，并为 GDN 状态预留 18 个 slot；vLLM 报告 44.66 GiB KV Cache。xLLM 多出约 0.36 GiB（0.8%），说明其单卡运行时预留略轻。差距很小，尚不足以改变 16 并发结果。

### 4. Qwen3.5 适配是独立实现，不只是通用 Transformers 回退

xLLM 源码注册了 `Qwen3_5ForConditionalGeneration`，包含 Qwen3.5 GDN 元数据、NPU decoder layer 和 MTP 状态路径。本轮启动日志还确认 8 个全注意力层启用了 `split_qkv_rmsnorm_mrope` 融合，并完成五档 ACL Graph 捕获。

## vLLM 在本轮更强的地方

- 稳态 Decode：TPOT P50 比 xLLM 低 5.1%，P90 低 5.9%。
- 总输出吞吐：比 xLLM 高 5.5%。
- 成熟的 PagedAttention/KV block 管理和更广的工具生态，使部署、基准工具和可观测性更完整。
- 即使镜像提示原生 NPU `causal_conv1d_update` 不可用、回退 PyTorch 路径，vLLM 仍完成图捕获并在本轮赢得 Decode，说明不能仅凭某个算子告警推断最终性能。

## 后续最有价值的测试

1. Qwen3.5-27B W8A8 + MTP、TP2：两侧官方文档都提供了对应路径，能验证量化、通信和推测解码后的结论。
2. 24/32 并发高 KV 压力测试：打开 xLLM Zero-Evict，并统计 vLLM preemption 次数，验证 xLLM 的核心调度优势。
3. 固定 TPOT SLO 扫描到达率，绘制 throughput/TTFT/TPOT 帕累托前沿；单点数据不适合评价在线服务能力。
4. 重复前缀负载：两侧同时打开 Prefix Cache，分别统计命中率、TTFT 和 HBM 占用。

## 复现材料

- [`benchmark_stream.py`](./benchmark_stream.py)：框架无关的 OpenAI 流式压测器
- [`launch_xllm.sh`](./launch_xllm.sh)：xLLM 启动参数
- [`launch_vllm.sh`](./launch_vllm.sh)：vLLM-Ascend 启动参数
- [`results/xllm-80k-samehost/summary.json`](./results/xllm-80k-samehost/summary.json)
- [`results/vllm-80k-samehost/summary.json`](./results/vllm-80k-samehost/summary.json)
- `details.jsonl`：逐请求 TTFT、TPOT、token 数、状态码和输出哈希
- `evidence/*-server.log`：模型识别、权重加载、KV 容量和图捕获证据

## 资料

- [xLLM Qwen3.5 部署指南](https://docs.xllm-ai.com/en/cookbook/autoregressive_models/qwen/qwen3_5/)
- [xLLM 支持模型矩阵](https://docs.xllm-ai.com/en/supported_models/)
- [xLLM Async Schedule](https://docs.xllm-ai.com/en/features/async_schedule/)
- [xLLM Zero Evict Scheduler](https://docs.xllm-ai.com/en/features/zero_evict_scheduler/)
- [xLLM Technical Report](https://arxiv.org/abs/2510.14686)
- [vLLM-Ascend Qwen3.5/Qwen3.6 指南](https://docs.vllm.ai/projects/ascend/en/v0.30.0rc1/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.html)
- [vLLM PagedAttention 论文](https://arxiv.org/abs/2309.06180)
