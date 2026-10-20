# 历史资料与实验索引

以 GLM 系列模型研究推理框架，当前实机对象改为 GLM-5.3-Flash W8A8，先完成 xLLM 验证，再扩展 SGLang、vLLM-Ascend，对比调度、算子、KV Cache 和通信带来的差异，逐步形成自己的框架。早期 GLM-5.3 W8A8C8 调研材料保留其原适用条件。

当前阶段：GLM-5.3-Flash W8A8 的 xLLM 四实例非 PD 完成 81920/600、并发16前缀负载48/48请求；TTFT均值23.268秒、TPOT均值144.6毫秒、集群输出82.32 token/s，是3P1D吞吐的3.72倍。MTP/缓存/重叠开启，图与KPool候选因验证失败关闭；尚非精度验收通过。K8s未改。

## 日报

| 日期 | 工作详情 | 阶段主要目标 | 阶段次要目标 |
| --- | --- | --- | --- |
| 2026/10/20 | 完成四实例非PD长输入复测：48/48成功，输出82.32 token/s；确认3P1D偏TTFT、非PD偏Decode吞吐，0.85显存并发OOM，0.80稳定。[结果及原始材料](experiments/xllm/2026-10-20-nonpd/README.md)。 | 找出xLLM当前最佳有效配置 | 对比PD与非PD的SLO取舍 |
| 2026/10/19 | 完成xLLM 3P1D长输入前缀压测：48/48成功，TTFT 8.945秒、TPOT 637.1毫秒、输出22.10 token/s；定位PD压缩池缓存限制与OOM，图关闭、精度待验证。[结果及原始材料](xllm-3p1d-prefix-perf-2026-10-19.md)。 | 对比框架性能与实现瓶颈 | |
| 2026/10/15 | 整理 xLLM 非 PD 材料；完成三框架调度、KV Cache、通信源码对比及论文分析。[资料](implementation-comparison.md)；实机待验证。 | GLM-5.3 C8 的 xLLM 部署验证与框架对比准备 | |

## 资料与代码

- [GLM-5.3-Flash 三框架研究路线](research/2026-10-20-framework-comparison-roadmap.md)：汇总最新官方适配状态与十四篇论文，规划共同基线、机制消融、在线 goodput、夜间机时和自研模块。

- [xLLM 非PD基线与vLLM对照准备](experiments/comparisons/2026-10-20-xllm-vllm-nonpd/README.md)：更正此前误用的vLLM PD截图；公开资料没有同条件非PD数据，保留源码假设和公平复测条件。

- [vLLM 非PD部署方案](experiments/vllm/2026-10-20-nonpd-plan/README.md)：v0.30.0RC1镜像选择、32设备DP/TP拓扑、时间估计和参数化启动脚本。

- [四实例非PD当前最佳性能](experiments/xllm/2026-10-20-nonpd/README.md)：同一80K/600负载的3P1D对照、显存边界、原生缓存/MTP计数和完整原始结果。

- [3P1D长输入前缀性能](xllm-3p1d-prefix-perf-2026-10-19.md)：用户脚本语义、完整48请求结果、失败配置与KPool源码限制。

- [2P2D性能调优与指标口径](xllm-performance-tuning-2026-10-19.md)：基线、失败图模式、Decode合批与重复测试。

- [xLLM优势候选与跨模型实验](xllm-strengths-and-experiments-2026-10-19.md)：历史评测边界、对手已有能力、源码限制与自研模块问题。
- [合批实验及正确性回退](xllm-batch-canary-2026-10-19.md)：失败反例、对照修正与当前2P2D配置。

- [2P2D与四台非PD对照](xllm-topology-ab-2026-10-19.md)：历史吞吐/延迟及原始证据；默认选择已被后续正确性复核撤回。

- [Decode ECC 定位与恢复](xllm-decode-ecc-recovery-2026-10-19.md)：全部rank日志、硬件计数、设备隔离和顺序验收。

- [Flash 2P2D 部署与验证](xllm-flash-2p2d-2026-10-19.md)：说明旧 C8 限制、VLM/LLM 差异和实际测试范围。

- [三框架实现对比](implementation-comparison.md)：调度、KV Cache、昇腾通信、设计取舍与控制变量实验。
- [固定提交源码索引](implementation-source-index.json)：31 个函数/模块入口及其支持的结论。

- [性能对比方法](benchmark-methodology.md)：Qwen PD 文章分析、指标口径、显存预算、帕累托与 SLO 实验。
- [论文与源码阅读路线](literature-map.md)：八篇文献的机制、评测思路和迁移边界。
- [实验记录模板](benchmark-run-template.json)：待测字段留空，保留配置、资源和原始证据。

- [2026-10-15 调研记录](research-2026-10-15.md)：源码入口、参考工程、版本差异和后续验证问题。
- [历史 C8 四台非 PD 候选方案](non-pd-plan.md)：并行拓扑、镜像依赖和验收顺序。
- [历史 C8 公共参数](common.json)与[命令生成器](render_commands.py)：旧模型候选方案，仅生成命令，与当前 Flash 实测配置分开。
- [历史候选版本](versions.json)与[调研源码快照](sources.json)：保留当时检索记录，当前 Flash 运行配置见实测报告。

## 后续推进

1. 基于当前稳定实例扩展负载及SLO测试，分析batch、图和通信瓶颈。
2. 按后续需求补充精度与长稳验证，逐项调优批处理、图与MTP。
3. 在相同模型、硬件和负载条件下引入 SGLang、vLLM-Ascend；分别记录统一基线和各自调优结果。
4. 用性能剖析与控制变量实验定位差异，再确定自研模块。

日报只记录当天实际新增进展，附成果链接；未运行的方案标为待验证。本次仅比较xLLM拓扑，尚无跨框架快慢的实测结论。
