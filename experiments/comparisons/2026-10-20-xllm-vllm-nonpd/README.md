# xLLM 非 PD 基线与 vLLM-Ascend 对照准备（2026-10-20）

## 结论与更正

用户提供的 vLLM 数据（TTFT 10.624 秒、TPOT 16.222 毫秒、输出 426.77 token/s）实际启用了 PD 分离，不能与 xLLM 四实例非 PD 结果计算倍率，也不能用于判断非 PD 下哪个框架更快。此前基于该截图计算的 2.19×、8.91×、5.18× 差距全部撤回。

当前唯一完成相同目标负载实测的是 xLLM：GLM-5.3-Flash W8A8、4×TP8、81920 输入、600 输出、48 请求、并发 16，TTFT 均值 23.268 秒、TPOT 均值 144.600 毫秒、输出吞吐 82.3239 token/s，48/48 请求成功。

截至 2026-10-20，未找到公开的“GLM-5.3-Flash W8A8、Ascend 910C、32 NPU、非 PD、81920/600、并发 16”完整结果。因此当前没有可信的 xLLM/vLLM 非 PD 性能排名，必须同机复测。

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

vLLM 当前候选优势来自可验证的实现组合，而不是尚未测量的最终性能：

1. MTP3 比 xLLM 当前 MTP1 有更高的每步提交 token 上限；真实收益仍取决于接受率。
2. `FULL_DECODE_ONLY` 可减少逐 token 的 Host/Kernel launch 开销；xLLM 当前有效结果关闭图模式。
3. vLLM-Ascend 已为 GLM 的 KDA、KeyPool、NoPE sparse MLA 和混合缓存提供专用实现，适配路线见 [#15665](https://github.com/vllm-project/vllm-ascend/issues/15665)。
4. vLLM 统一 token budget 调度 chunked prefill、prefix cache 和 Decode；xLLM 当前 MTP 会关闭 P/D 混批并选择 Prefill-first，见 [xLLM scheduler_policy.cpp](https://github.com/xLLM-AI/xllm/blob/ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb/xllm/core/scheduler/scheduler_policy.cpp#L1061-L1105)。

这些代码特征只形成“vLLM 值得实测”的假设。没有同机结果前，不写成“vLLM 一定比 xLLM 快”。

## 待执行的公平对照

仓库已准备 [vLLM 非 PD 部署方案](../../vllm/2026-10-20-nonpd-plan/README.md)。第一组严格控制为 32 个逻辑设备、同一权重、同一 AISBench 数据、81920/600、48 请求、并发16。优先测试 4节点×TP8（DP4），若 vLLM A3 路径要求官方 TP16 布局，则改为2节点×TP16（DP2）并在结果中明确拓扑差异。

必须保存镜像 digest、vLLM/vLLM-Ascend/CANN 版本、权重哈希、设备映射、启动日志、MTP 接受率、prefix hit、原始请求结果和 NPU 利用率，之后才能给出框架结论。

## 证据

- [xLLM 四实例非 PD 完整结果](../../xllm/2026-10-20-nonpd/README.md)
- [xLLM 原始性能摘要](../../xllm/2026-10-20-nonpd/performance-summary.json)
- [xLLM 原生缓存与 MTP 计数](../../xllm/2026-10-20-nonpd/native-runtime-evidence.json)
- [vLLM-Ascend GLM-5.3-Flash 教程](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/GLM5.3-Flash.md)
- [vLLM-Ascend GLM-5.3-Flash 适配路线](https://github.com/vllm-project/vllm-ascend/issues/15665)
