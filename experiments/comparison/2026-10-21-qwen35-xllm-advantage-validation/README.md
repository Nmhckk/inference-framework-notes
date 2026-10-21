# Qwen3.5-4B：xLLM 优势验证

日期：2026-10-21

## 结论

本轮在同一台 Ascend 910C 64GB、同一张物理 NPU、同一份 Qwen3.5-4B BF16 权重和同一请求流上，分别验证 xLLM 的 Schedule Overlap、Zero-Evict 和 Prefix Cache。

- **Schedule Overlap 有效但收益小**：输出吞吐提高 0.48%，TPOT P50 降低 1.33%。单次结果可能包含运行波动，暂按约 1% 的方向性收益处理。
- **Zero-Evict 未产生收益**：32 并发高 KV 压力下，吞吐、TTFT 和 TPOT 的差异都低于 0.2%。当前负载没有证据表明默认调度发生了可被 Zero-Evict 避免的驱逐。
- **xLLM Prefix Cache 更偏向低 TPOT**：TPOT P50 为 11.43 ms，比 vLLM 的 57.33 ms 低 80.1%；但它串行化了大量 Prefill，TTFT P50 达 111.01 s，总吞吐只有 80.29 token/s。
- **vLLM Prefix Cache 的综合性能明显更强**：TTFT P50 1.04 s、总吞吐 261.21 token/s，分别比 xLLM 低约 99.1%、高 225.3%。因此本轮不能把 Prefix Cache 作为 xLLM 的整体优势。

xLLM 当前能确认的优势是：在请求已经进入 Decode 后，可以用更激进的服务顺序换取很低的单用户 TPOT；C++ Schedule Overlap 也有小幅稳态收益。它尚未在总吞吐、首 token 延迟或高 KV 压力稳定性上战胜 vLLM。

## 公共测试条件

- 服务器：`bm-jn-zs-zone1-910C-64G-10-171`
- 设备：物理 NPU 0，单卡 TP1
- 模型：Qwen3.5-4B BF16
- 请求：48 条，实际输入 81,798～82,558 token，每条固定输出 600 token
- 并发/到达率：16 或 32；0.93 request/s
- Prefill chunk：8,192 token
- Decode 图：1/2/4/8/16 batch
- 正确性门槛：HTTP 200、生成满 600 token；所有组均 48/48 成功

## 1. Schedule Overlap 消融

两组均关闭 Prefix Cache，保持 16 并发、ACL Graph、Chunked Prefill 和相同 KV 容量。

| xLLM 配置 | 输出吞吐 | TTFT P50 | TTFT P90 | TPOT P50 | TPOT P90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overlap 关闭 | 65.21 tok/s | 16.88 s | 65.79 s | 223.01 ms | 224.68 ms |
| Overlap 开启 | **65.52 tok/s** | 17.89 s | **65.61 s** | **220.05 ms** | **222.57 ms** |
| 开启相对关闭 | **+0.48%** | +5.95% | -0.27% | **-1.33%** | **-0.94%** |

源码中，xLLM 使用 thread pool、future/promise 和上一拍输出，把下一步输入准备与当前设备执行重叠，见 [`worker_impl.cpp`](https://github.com/xLLM-AI/xllm/blob/b612303512c9d4c7304f7d558f250cf0c3e88918/xllm/core/runtime/worker_impl.cpp#L1846-L1903)。本轮 80K Prefill 占主导，调度间隙只占很小比例，所以端到端收益有限。

## 2. Zero-Evict 高 KV 压力消融

将 `max_seqs_per_batch` 和客户端并发都提高到 32，保持 Prefix Cache 关闭。Zero-Evict 组额外设置 `--use_zero_evict=true --max_decode_token_per_sequence=600`。两组 KV 容量均为 44.97 GiB，GDN 状态槽均为 34。

| xLLM 调度器 | 输出吞吐 | TTFT P50 | TTFT P90 | TPOT P50 | TPOT P90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 默认 Continuous | **64.71 tok/s** | **166.84 s** | **198.06 s** | **208.49 ms** | **222.23 ms** |
| Zero-Evict | 64.66 tok/s | 166.95 s | 198.28 s | 208.65 ms | 222.36 ms |
| Zero-Evict 相对默认 | -0.09% | +0.07% | +0.11% | +0.08% | +0.06% |

Zero-Evict 会在接纳 Prefill 前模拟后续 Decode 的块需求，具体见 [`zero_eviction_scheduler.cpp`](https://github.com/xLLM-AI/xllm/blob/b612303512c9d4c7304f7d558f250cf0c3e88918/xllm/core/scheduler/zero_eviction_scheduler.cpp#L199-L250)。本轮虽然 TTFT P90 被压力放大到约 198 秒，但两组逐请求轨迹几乎重合，日志中也没有驱逐、重算或 OOM 证据。由此只能得出“保护机制没有在当前负载形成收益”，不能否定它在更长输出、显式抢占或严格 SLO 下的价值。

## 3. Prefix Cache 跨框架对照

数据集约 93% 前缀重复。两侧均使用 16 并发，分别对照各自关闭 Prefix Cache 的同机结果。

| 框架/配置 | 输出吞吐 | TTFT P50 | TTFT P90 | TPOT P50 | TPOT P90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| xLLM，缓存关闭 | 65.52 tok/s | 17.89 s | 65.61 s | 220.05 ms | 222.57 ms |
| xLLM，缓存开启/自动状态槽 | 80.29 tok/s | 111.01 s | 116.57 s | **11.43 ms** | **11.45 ms** |
| xLLM，缓存开启/64 状态槽 | 80.22 tok/s | 111.14 s | 116.68 s | 11.45 ms | 11.46 ms |
| vLLM，缓存关闭 | 69.12 tok/s | 12.86 s | 69.62 s | 208.88 ms | 209.41 ms |
| vLLM，缓存开启 | **261.21 tok/s** | **1.04 s** | **1.06 s** | 57.33 ms | 57.74 ms |

xLLM 开启混合 GDN Prefix Cache 后，默认把约 47% KV 预算分给线性状态槽：状态槽从 18 增至 444，预留 21.30 GiB，Full-Attention KV 块从 11,303 降至 6,072。源码见 [`kv_cache_estimation.cpp`](https://github.com/xLLM-AI/xllm/blob/b612303512c9d4c7304f7d558f250cf0c3e88918/xllm/core/kv_cache/layout/kv_cache_estimation.cpp#L288-L327)。

把 `max_linear_state_cache_slots` 限制为 64 后，状态预留降至 3.17 GiB、KV 块回升到 10,714，但端到端指标基本不变。因此主要瓶颈不是默认显存比例，而是请求接纳和 Prefill/Decode 服务顺序。xLLM 的 11 ms TPOT来自更少请求同时进入 Decode；它是延迟资源分配策略，不代表总算力高于 vLLM。

vLLM 对 Qwen3.5 自动使用 Mamba/GDN `align` 缓存模式。即使日志提示原生 NPU `causal_conv1d_update` 不可用并回退 PyTorch，它仍实现 1.04 s TTFT P50 和 261.21 token/s，说明其混合模型 Prefix Cache 与调度组合目前更成熟。

## 对自研框架的启示

1. 保留 xLLM 的 C++ Schedule Overlap，但预期收益应按负载评估，不能把官方微基准提升直接外推到长 Prefill。
2. Zero-Evict 应做成 SLO/抢占触发的可选接纳策略；先确认默认路径确实发生驱逐，再启用保护。
3. Prefix Cache 需要把命中收益、状态显存和调度公平性一起优化。只压低 TPOT而让 TTFT增长百倍，不适合作为默认在线策略。
4. 后续优先在 Qwen3.5-27B W8A8＋MTP、TP2 上重复测试，并记录缓存命中率、preemption 次数和每阶段 NPU 时间。

## 复现材料

- `scripts/`：所有启动参数，包括 Overlap、Zero-Evict、Prefix Cache 和 64 状态槽变体
- `results/*/summary.json`：汇总指标
- `results/*/details.jsonl`：48 条逐请求状态、TTFT、TPOT、token 数和输出哈希
- `evidence/*-server.log`：模型、KV/GDN 容量、图捕获和配置证据
- `compare_results.py`：从原始 JSON 重算本文表格和百分比

测试结束后已删除本项目容器，NPU0 恢复空闲；未修改 K8s、其他容器或其他用户任务。
