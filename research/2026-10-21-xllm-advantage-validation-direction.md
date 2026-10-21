# xLLM 优势验证方向校准（2026-10-21）

## 结论

此前没有测出 xLLM 的整体优势，主要原因不是优化开关遗漏，而是测试场景没有命中其公开证据中收益最大的区域：Qwen3.5-4B、单卡、80K 输入、固定 16 并发会让超长 Prefill、混合注意力状态和显存容量主导结果；xLLM 论文的主结果使用 Qwen3 系列、2048/2048、按 TPOT SLO 扫请求率，并重点展示多卡扩展。论文对比的 vLLM-Ascend 还是 v0.10.rc1，不能直接外推到当前 v0.30.0rc2。

当前正确方向分成两条：

1. **先验证 xLLM 当前代码中最具体的模型优化**：Qwen3.5-27B、TP2/TP4、GDN Decode SuperOp 与外置 MTP draft，分别做开关消融和 xLLM/vLLM-Ascend 对照。
2. **再复现论文所处的负载区间**：Qwen3-1.7B/4B、2048/2048、1/2/4 卡，扫描 offered QPS，在相同 TPOT SLO 下比较 goodput 和扩展效率。

80K/600 保留为长上下文兼容性和容量压力测试，不再作为寻找 xLLM 普遍优势的主基准。

## 论文、仓库和 Issue 证据

| 候选能力 | 公开证据 | 当前判断 | 对应实验 |
| --- | --- | --- | --- |
| Adaptive Graph | 技术报告中 Qwen3-1.7B 吞吐 2385.58→3038.88 token/s（+27.4%），Qwen3-4B 1539.93→1671.39（+8.5%），均为 2048/2048 | 小模型、短迭代更容易受 Host/Launch 开销影响；双方现在都有图能力，优势必须实测 | Qwen3-1.7B/4B，图开/关，QPS 扫描 |
| Schedule Overlap | 报告中 DS-Distill-Qwen-1.5B 为 8709.90→10223.15 token/s（+17.4%），7B 只有约 +0.6%；本项目 4B、80K 仅 +0.48% | 收益随模型和负载变化，不能当通用优势 | 1.5B/1.7B，1000/1000 与 2048/2048 消融 |
| Qwen3.5 GDN 融合 | 官方 workflow 的 Qwen3.5-27B TP2 案例中，BS1–4 kernel 加速 1.445×–1.793×；端到端 TPOT 在并发 1/4 改善 6.24%/16.01% | 当前最具体、最可归因的 xLLM Engine 优势候选 | 2048/64，并发 1/4，SuperOp 开/关 |
| Qwen3.5 MTP | 官方 workflow 的 TP4、20K/1K、并发 1：MTP0 为 46.39 token/s、18.67ms TPOT；MTP3 为 66.82 token/s、12.05ms，MTP4/5 反而下降 | MTP3 是明确候选；MTP 本身并非 xLLM 独有，要与 vLLM 同深度对照 | 20K/1K，并发 1，MTP 0/1/2/3 深度扫描 |
| 多卡扩展 | 报告称 910C 上最高 2.2× vLLM-Ascend，Qwen3-32B 8 卡业务场景为 3.1×；基线版本较旧 | 只能作为待复现假设，重点看当前版本 TP1/2/4 扩展效率 | Qwen3-32B，1200/40 与 2048/2048 |
| 动态 PD / Online-Offline | 报告的 Azure Code 动态调度为 Minimal Load 的 1.67×；多模态 EPD goodput 9.5/7.2/5.1 req/s | 属于 xLLM-Service 系统级候选，不应与单实例 Engine 吞吐混为一谈 | 多节点、突发 trace、SLO goodput，后置进行 |
| 多流通信计算重叠 | 论文报告单层 exposed communication 9.3→2.5ms | 当前主分支 `master.cpp` 对多机 `enable_multi_stream_parallel` 直接报错“refactoring now” | 当前排除，待正式恢复后再测 |
| Zero-Evict | 设计目标是减少抢占/重算 | 本项目并发 32 的差异不足 0.2%，且没有记录到真实 eviction/preemption，实验未触发机制 | 只有出现 vLLM 抢占/重算和 xLLM 驱逐计数后才重测 |

公开 Issue 也不支持“xLLM 单卡天然更快”的结论：[#832](https://github.com/xLLM-AI/xllm/issues/832) 报告 Qwen3-14B 单卡约 420 token/s，而 vLLM-Ascend 约 579 token/s，维护者建议升级版本但未给出完整复现结果；[#2135](https://github.com/xLLM-AI/xllm/issues/2135) 记录 Qwen3.5 DP2/TP8/EP16 图模式在空 DP shard 下崩溃。因此先验证有代码和模型证据的窄优势，再决定是否扩大。

## 第一优先级：Qwen3.5-27B Engine 优势

### 资源和模型

- 同一台或同一组 910C，优先 TP2；有连续 4 卡时补 TP4。
- 同一份 Qwen3.5-27B 权重、tokenizer、dtype/量化、最大长度。
- xLLM 与 vLLM-Ascend 分别调优，但任何一侧开启 MTP 时，另一侧必须使用相同 draft 权重和 speculative depth。
- 启动前保存 `npu-smi info`；存在他人进程、HBM 占用或 K8s 设备时跳过，不清理、不抢占。

### A：GDN SuperOp 消融

- 输入/输出：2048/64。
- 并发：1、4。
- 固定 Graph 和 Schedule Overlap 开启。
- xLLM：默认 SuperOp 对比 `XLLM_DISABLE_QWEN35_GDN_DECODE_SUPER_OP=1`。
- 至少 5 个 warmup 请求和 3 轮正式重复；保存逐请求 TTFT、TPOT、E2E、吞吐和服务日志。
- 只有输出一致、日志中 SuperOp 调用数符合层数×Decode token、fallback 为 0，性能才有效。

### B：外置 MTP 深度扫描

- 输入/输出：20000/1000。
- 并发：1；若显存允许再补 4/16。
- 深度：0、1、2、3；深度 3 通过正确性后再决定是否测 4/5。
- 必须先导出 draft 目录，确认存在 `mtp_layer_parameters.safetensors` 且 `model_type=qwen3_5_mtp`。
- MTP 是否生效以服务端 `/vars` 的 `speculative_num_accepted_tokens_total` 和 `speculative_num_draft_tokens_total` 增量为准，不使用客户端 streaming chunk 推断。
- 先做固定 prompt、temperature=0 的 MTP0/MTP3 输出一致性；失败即停止性能测试。

### 成功标准

- 正确性门禁全部通过。
- 三轮中位数至少提升 5%，且没有以 TTFT、错误率或显存溢出换取单一 TPOT 改善。
- 跨框架结论以 SLA pass → request goodput → output throughput → TTFT → TPOT 的顺序判断。
- 若只在 xLLM 内部消融成立，应写成“xLLM 该优化有效”，不能写成“xLLM 优于 vLLM”。

## 第二优先级：论文区间复现

- 模型：Qwen3-1.7B、Qwen3-4B；再根据显存补 Qwen3-32B。
- 工作负载：2048 输入/2048 输出；ShareGPT 或固定随机 token 数据，同时报告实际 token 长度。
- 设备：TP1/2/4；同卡、同 NUMA、同 CANN 驱动和同客户端。
- 方法：逐档提升 offered QPS，不使用单个固定并发点；找出 TPOT P99≤50ms 时的最高 goodput，并绘制 TTFT/TPOT/吞吐帕累托曲线。
- 消融：Graph 开/关、Schedule Overlap 开/关；SGLang 和 vLLM 使用各自官方推荐图/异步配置。
- 扩展效率：`TPn 吞吐 / (n × TP1 吞吐)`；配合 profiler 记录 Host gap、图重放和 HCCL exposed time。

## 暂不投入机时的方向

- 多流并行：当前主分支显式不可用。
- Qwen3.5 DP 图低并发：公开 Issue 尚有空 shard 崩溃问题。
- 2P2D/3P1D：除非目标变为动态 PD 或异构 TP 的系统级收益，不再用静态 PD 拓扑替代 Engine 公平对照。
- Prefix Cache：此前结果混入准入策略差异。下一次必须同时报告请求 goodput、TTFT SLO 达标率、cache hit、队列时间，不能只看 TPOT。
- 未合入 PR：可作为单独候选分支，不进入框架正式版本基线。

## 测试执行顺序

1. 只读检查空闲机器、现有权重和 draft 文件，建立资源清单。
2. 在空闲卡上跑 Qwen3.5-27B TP2 的 SuperOp A/B；若没有 27B 权重则先跑 Qwen3-1.7B 图 A/B。
3. 导出/核验 draft 后跑 MTP0→1→2→3，先正确性、后性能。
4. 用相同卡和权重部署 vLLM-Ascend，按同一矩阵测试；然后补 SGLang。
5. 完成 QPS/SLO 扫描和 TP1/2/4 扩展后，才判断 xLLM 的优势是否成立。

## 来源

- [xLLM Technical Report](https://arxiv.org/abs/2510.14686)
- [xLLM 官方仓库](https://github.com/xLLM-AI/xllm)
- [xLLM workflow：MTP benchmark lessons](https://github.com/xLLM-AI/xllm-workflow/blob/main/skills/xllm-npu-benchmark/references/mtp-benchmark-lessons.md)
- [xLLM workflow：Qwen3.5 GDN Decode case](https://github.com/xLLM-AI/xllm-workflow/blob/main/skills/tilelang-pto-kernel-workflow/references/qwen35-gdn-case.md)
- [Issue #832：Qwen3-14B xLLM/vLLM 性能疑问](https://github.com/xLLM-AI/xllm/issues/832)
- [Issue #2135：Qwen3.5 DP 图空 shard 崩溃](https://github.com/xLLM-AI/xllm/issues/2135)
- [Issue #2034：Qwen3.5 Async MTP](https://github.com/xLLM-AI/xllm/issues/2034)
- [Issue #2060：Qwen3.5 异构 TP PD](https://github.com/xLLM-AI/xllm/issues/2060)

本次源码检查基于 xLLM `53f22bb603556dfb933f3f14a993cfbd20ce2e26`（2026-10-21）。
