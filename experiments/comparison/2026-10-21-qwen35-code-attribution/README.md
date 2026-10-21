# Qwen3.5 三框架代码级性能归因

## 结论

同卡实测中，xLLM 为 1099.10 token/s，SGLang 为 1057.02 token/s，vLLM-Ascend rc2 为 951.61 token/s。进一步对固定提交和运行日志做代码门禁后，能严谨成立的结论是：

1. **调度重叠是已被 A/B 证实的收益。** xLLM 开启 Schedule Overlap 后吞吐提升 3.46%；SGLang 开启 Overlap 后提升 12.78%。xLLM 的 C++ scheduler 先发射当前 batch，再消费上一 batch 的结果；Worker 用 NPU `replace_token(..., synchronize_stream=false)` 把上一轮 token 写回下一轮输入，减少 Host 等待。SGLang 使用 Python event loop、forward stream 和 result queue 实现同一类流水。
2. **xLLM 的 Qwen3.5 GDN Decode 是最值得继续验证的算子优势。** xLLM 把 causal-conv、门控、递归状态读写和 norm 收进单次 `mega_gdn_decode` 调用；SGLang 当前 NPU 路径先调用 `torch.ops.npu.causal_conv1d`，再进入 `kernel_dispatcher.decode`。这会影响 kernel launch 数、临时张量和状态访存，但本轮没有逐算子 profiler，因此它是强候选，不是已量化的 3.98% 因果解释。
3. **Full-Attention 融合不是 xLLM 独占优势。** xLLM 的 `split_qkv_rmsnorm_mrope` 和 SGLang 的 `split_qkvgate_gemma_rmsnorm_rope` 都融合 QKV 拆分、Q/K RMSNorm 和 RoPE。运行日志确认 xLLM 的 8 个 Full-Attention 层全部命中融合；SGLang 固定提交也走 NPU 融合路径。
4. **Graph 不是本轮 xLLM 领先来源。** xLLM 最优结果为 Graph 关闭，SGLang 运行日志确认使用 Decode Graph，vLLM 启动配置要求 `FULL_DECODE_ONLY`，但其运行日志缺失。xLLM Graph A/B 没有稳定收益，说明至少对 xLLM 而言，Graph 不能解释领先。
5. **混合状态缓存设计正确，但三家都在做。** xLLM 运行时把 Full-Attention KV 分配为 11,385 个 paged blocks，把 GDN 固定状态单独分配为 18 个 state slots，共 884.25 MiB；这避免把按请求固定的 GDN state 按 token 分页。SGLang 的 `MambaPool` 和 vLLM 的 hybrid Mamba cache 也遵循类似思想，因此它是现代混合模型的必要设计，不能单独解释框架胜负。

## 为什么 xLLM 在这组负载更快

### 已证实：一拍流水隐藏 Host 调度

xLLM [`continuous_scheduler.cpp`](https://github.com/xLLM-AI/xllm/blob/53f22bb603556dfb933f3f14a993cfbd20ce2e26/xllm/core/scheduler/continuous_scheduler.cpp#L568-L619)中的顺序是：

```cpp
BatchGroup batch = schedule_request(timeout);
step_callback_(batch);                 // 发射第 i 拍
result_callback_(last_batch_);         // 消费第 i-1 拍
process_batch_output(true);
last_batch_ = std::move(batch);
```

Worker 随后在设备侧用上一步输出替换负占位 token；未开 overlap 时则立即 `future.get()` 等待结果。这个差异与 xLLM 自身 **+3.46%** 的 A/B 结果直接对应，属于已验证因果。

SGLang [`scheduler.py`](https://github.com/sgl-project/sglang/blob/e9b22a0818292eedc0630b2b72cdfcb4ae81bf85/python/sglang/srt/managers/scheduler.py#L2008-L2078)把当前 `run_batch` 结果压入 `result_queue`，然后处理上一 batch。其 A/B 收益更大，说明未开启 overlap 时 Python scheduler、采样和结果处理造成的空隙更多；不能据此说 xLLM 调度器优于 SGLang，因为双方“关闭态”的基线成本不同。

vLLM 的 `AsyncScheduler` 同样给尚未返回的输出放置占位 token，并允许调度器提前推进；本轮也显式开启 `--async-scheduling`。因此“一拍流水”是三家共同方向，xLLM 的已证实结论是其实现对自身基线有效，不能仅凭 C++/Python 语言差异解释跨框架胜负。

### 强候选：xLLM 的 MegaGDN 缩短 Decode 算子链

xLLM 固定提交的 [`gated_delta_net.py`](https://github.com/xLLM-AI/xllm/blob/53f22bb603556dfb933f3f14a993cfbd20ce2e26/xllm/python/layers/npu/qwen3_5/gated_delta_net.py#L155-L189)在两个输入投影后，以一次 `mega_gdn_decode` 处理单 token GDN。接口同时接收 conv state、SSM state、读写 slot、gate 参数和 norm 权重，源码注释明确说明它会更新两类状态缓存。

SGLang 固定提交的 [`ascend_gdn_backend.py`](https://github.com/sgl-project/sglang/blob/e9b22a0818292eedc0630b2b72cdfcb4ae81bf85/python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py#L114-L168)先执行 NPU causal-conv，再 split/view QKV，再调 GDN decode kernel。两边最终都使用 NPU kernel，但 xLLM 的融合边界更大，理论上减少：

- causal-conv 与 recurrence 之间的独立发射；
- Q/K/V 中间张量的落盘与再次读取；
- conv/SSM 状态索引的重复准备；
- Python 层逐算子调度开销。

这与 xLLM 比 SGLang 高 3.98% 的方向一致。要确认占比，需要在相同 batch 16 下采集两边逐算子 timeline，比较每 token 的 GDN kernel 数、总 GDN device time、Host gap 和 H2D/D2H 时间。

### 已排除：独占 QKV 融合、Graph、MTP、Prefix Cache、PD

- xLLM [`attention.py`](https://github.com/xLLM-AI/xllm/blob/53f22bb603556dfb933f3f14a993cfbd20ce2e26/xllm/python/layers/npu/qwen3_5/attention.py#L97-L138)和 SGLang [`qwen3_5.py`](https://github.com/sgl-project/sglang/blob/e9b22a0818292eedc0630b2b72cdfcb4ae81bf85/python/sglang/srt/models/qwen3_5.py#L1541-L1588)都有 Decode 专用融合。
- xLLM 本轮 `enable_graph=0`；SGLang 日志显示 `npu graph: True`，vLLM 启动项为 `FULL_DECODE_ONLY`。
- 三框架均未启用 MTP、Prefix Cache、PD 或量化，这些机制不能用于解释本轮差距。

## vLLM 归因的证据修正

vLLM 的 951.61 token/s 性能结果、启动命令、逐请求数据和容器摘要都完整；但归档的 `vllm-server.log` 是 0 字节。此前报告中“该固定镜像日志显示 `enable_qk_norm_rope_fusion=False`，并发生 causal-conv fallback”的运行时归因，**无法从当前 GitHub 证据包独立复核，因此降级为待补证据**。

vLLM-Ascend 当前主线源码确实保留了一个同步 PyTorch causal-conv fallback，并说明其中按请求 `.item()` 会同步 NPU；但主线代码不能替代 `v0.30.0rc2-a3` 固定镜像的运行日志。当前只能严谨写成：xLLM 比该 vLLM 镜像快 15.50%，且算子融合/回退是合理调查方向；在补齐镜像源码 SHA 或服务日志前，不能把 15.50% 定量归因于该 fallback。

## 可重复代码门禁

运行：

```bash
python3 audit_hotpaths.py
```

脚本用 `git show <提交>:<文件>` 检查十个关键路径，解析 xLLM 实测日志，并重新计算三框架吞吐差异。xLLM/SGLang 使用被测固定提交；vLLM 当前主线检查只用于说明机制，不能替代被测镜像源码。输出见 [`audit-results.json`](audit-results.json)。它只证明“源码中存在、实测配置启用”，不替代 profiler 或端到端 A/B。

## 下一轮最小实验

1. **GDN SuperOp A/B**：同一 xLLM 版本、同一卡、2048/2048、并发 1/4/16，关闭与开启 MegaGDN；记录每 token kernel 数和 GDN device time。
2. **三框架 profiler**：固定 batch 16 的稳定 Decode 区间，各截取 200–500 token；报告 Host gap、kernel launch 数、causal-conv、GDN recurrence、Full Attention、采样和 D2H。
3. **服务层剥离**：若框架提供 offline engine，使用相同预分词 token 直接压 engine，区分 HTTP/tokenizer 与 model runner 差异。
4. **QPS/SLO 曲线**：2048/2048 扫 offered QPS，以 TPOT P99≤50 ms 的 request goodput 判胜，避免单个饱和点外推。

## 理论依据

- [xLLM Technical Report](https://arxiv.org/abs/2510.14686)：将优势分为 engine 侧自适应图、调度重叠、算子/通信优化和 service 侧全局调度；本轮只验证了 engine 中的窄场景。
- [SGLang paper](https://arxiv.org/abs/2312.07104)：核心是 RadixAttention、结构化程序运行时与前缀复用；本轮关闭了 Radix Cache，因此没有覆盖它最强的多轮/共享前缀场景。
- [vLLM PagedAttention paper](https://arxiv.org/abs/2309.06180)：核心收益来自降低 KV 碎片和复制、提高可合批请求数；本轮单卡并发 16、KV 容量充足，未命中 PagedAttention 的最大收益区间。

## 当前判断

xLLM 在 Qwen3.5-4B、2K/2K、单卡16并发上的优势已经成立；Schedule Overlap 对 xLLM 自身贡献 3.46%，但不能与跨框架差距直接相减。相对 SGLang 剩余约 3%–4% 的首要嫌疑是更大的 MegaGDN 融合边界和较短的执行链路，代码证据强但仍需 profiler。相对 vLLM 的 15.50% 差距还缺固定镜像的服务日志或源码 SHA，暂不做过度归因。
