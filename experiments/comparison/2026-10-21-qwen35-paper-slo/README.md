# Qwen3.5-4B：xLLM 优势区间与机制消融

## 结论

在同一台 Ascend 910C-64G 的同一张物理卡、同一份 Qwen3.5-4B BF16 权重和同一客户端下，2033–2034 输入、2048 输出、48 请求、16 并发的三轮结果为：

| 配置 | 输出吞吐（tok/s） | TTFT P90（s） | TTFT P99（s） | TPOT 均值（ms） | TPOT P99（ms） |
| --- | ---: | ---: | ---: | ---: | ---: |
| **xLLM，Graph 关，Schedule Overlap 开** | **1099.10 ± 0.91** | **1.313** | **1.580** | **14.118** | **14.480** |
| xLLM，Graph 关，Schedule Overlap 关 | 1062.39 ± 1.36 | 1.292 | 1.600 | 14.684 | 14.984 |
| xLLM，Graph 开，Schedule Overlap 开 | 1094.96 ± 3.83 | 1.290 | 1.577 | 14.156 | 14.509 |
| vLLM-Ascend rc2，FULL_DECODE_ONLY Graph＋Async Scheduling | 951.61 ± 1.54 | 1.530 | 1.817 | 16.304 | 16.728 |

每组 3 轮、每轮 48/48 请求成功并完整输出 2048 token。相对当前 vLLM 基线，xLLM 最佳配置：

- 输出吞吐 **+15.50%**；
- TTFT P90 **-14.21%**、P99 **-13.02%**；
- TPOT 均值 **-13.40%**、P99 **-13.44%**。

这是目前项目中第一组可重复的 xLLM 跨框架优势。结论只覆盖该模型、单卡和 Decode 较长的饱和负载，不外推到 80K 长输入、PD、多卡或其他模型。

## 优势来自哪里

### 1. Schedule Overlap 已通过 A/B 验证

只改变 `--enable_schedule_overlap`，Graph 均关闭：

- 吞吐 1062.39 → 1099.10 tok/s，**+3.46%**；
- TPOT 均值 14.684 → 14.118 ms，**-3.85%**；
- TTFT P90 增加 1.61%，说明它改善持续 Decode 流水线，未同时改善所有延迟指标。

[xLLM 调度器](https://github.com/xLLM-AI/xllm/blob/53f22bb603556dfb933f3f14a993cfbd20ce2e26/xllm/core/scheduler/continuous_scheduler.cpp)的 overlap 路径先提交当前 batch，再回收上一 batch 的结果，形成只前推一步的 producer-consumer 流水；[Worker 异步执行](https://github.com/xLLM-AI/xllm/blob/53f22bb603556dfb933f3f14a993cfbd20ce2e26/xllm/core/runtime/worker_impl.cpp)保存上一轮设备输出，用它修正下一轮 Decode 输入。这会把 Host 侧调度、结果处理和下一轮 NPU 计算重叠起来。2048-token Decode 会重复这一过程两千余次，因此比此前 80K/600 测试更容易显现收益。

### 2. Qwen3.5 专用融合算子已经实际命中

服务日志在 8 个 Full-Attention 层均记录 `using fused split_qkv_rmsnorm_mrope kernel`。对应的 [Qwen3NextAttention NPU 实现](https://github.com/xLLM-AI/xllm/blob/53f22bb603556dfb933f3f14a993cfbd20ce2e26/xllm/core/layers/npu_torch/qwen3_next_attention.cpp)把 QKV 拆分、Q/K RMSNorm、mRoPE gather 和门控准备合入 TileLang NPU kernel，减少中间张量和 kernel launch。日志还确认混合架构分配了 18 个 GDN linear-state slot。

Schedule Overlap 只能解释约 3.5 个百分点；即便关闭它，xLLM 吞吐仍比 vLLM 高约 11.6%。剩余差异与 Qwen3.5 专用融合算子、混合状态管理和 runtime 开销一致，但本轮没有对 vLLM 和 xLLM 的各个算子逐一禁用，不能把剩余 11.6% 全部归因于某一个 kernel。下一步需要采集 Host gap、AICore 时间和逐算子 trace。

### 3. Graph 不是本负载下的优势

在 Schedule Overlap 开启时，xLLM Graph 开启后的三轮均值相对关闭：吞吐 **-0.38%**、TPOT 均值 **+0.26%**，TTFT P90 **-1.74%**。差异接近运行波动，当前结论是“无稳定吞吐增益”。不能把论文中的 Adaptive Graph 收益直接迁移到 Qwen3.5-4B、并发 16；后续仍应在低并发和 offered-QPS/SLO 扫描中验证。

## 测试条件

- 硬件：同一台 910C-64G 超节点，顺序独占物理 NPU 0；测试前后其余 NPU 无运行进程。
- 模型：同一份 Qwen3.5-4B BF16 HF 权重；实际输入 2033–2034 token，固定输出 2048 token。
- 客户端：OpenAI streaming；48 请求、并发 16、`request_rate=0`、2 个 warmup；prefix cache 关闭。
- xLLM：0.11.0，源码快照 `53f22bb603556dfb933f3f14a993cfbd20ce2e26`，`max_num_seqs=16`、`max_num_batched_tokens=8192`、chunked prefill 开启。
- vLLM：镜像 `v0.30.0rc2-a3`，运行时报告 `0.30.0+empty`，Async Scheduling 和 `FULL_DECODE_ONLY` 图开启，capture size 为 1/2/4/8/16；批量上限与 xLLM 对齐。
- 没有 MTP draft 权重，因此双方均未启用 MTP；没有启用 prefix cache、PD 或量化。

该客户端验证了 API 成功率、实际 token 数、streaming 时序和输出摘要。temperature=0 下两个框架的前 160 字符有 46/48 一致，但完整输出 hash 不一致；本轮属于性能与基本输出门禁，不是完整精度评测。

## 证据与复现

- [三轮聚合摘要](summary.csv)
- [完整聚合与 P99](export/comparison-summary.json)
- [逐请求结果](export/results/)
- [xLLM 服务日志](export/logs/exact-clean-server.log)与[Graph 日志](export/logs/xllm-graph-on-server.log)
- [xLLM 启动配置](export/input/launch_xllm_exact.sh)、[vLLM 启动配置](export/launch_vllm.sh)和[统一客户端](export/input/benchmark_stream.py)
- [镜像、源码和数据哈希](export/runtime-versions.txt)
- [容器配置](export/docker-inspect.json)、[测试后 NPU 状态](export/npu-smi-after-cleanup.txt)
- [一次误标运行的排除说明](export/EXCLUSIONS.md)

## 仍需验证

1. 按 xLLM 论文方法扫描 offered QPS，比较 TPOT P99≤50 ms 下的最高 goodput，而不是只测一个饱和并发点。
2. 获取 Qwen3.5-27B HF 权重与 `mtp_layer_parameters.safetensors` 后，做 GDN Decode SuperOp 和 MTP0/3 消融；当前共享目录只有训练检查点，不能直接作为推理权重。
3. 采集两框架 profiler，拆出 Host gap、融合 kernel、GDN 状态读写和图重放时间，解释剩余约 11.6% 的跨框架差异。
4. 再补 SGLang 相同单卡基线，然后进行 TP1/2/4 扩展效率比较。

方法依据：[xLLM Technical Report](https://arxiv.org/abs/2510.14686)；测试方向与公开 Issue 边界见[优势验证方向校准](../../../research/2026-10-21-xllm-advantage-validation-direction.md)。
