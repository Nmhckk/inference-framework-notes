# GLM-5.3 推理框架调研与实践

以 GLM 系列模型研究推理框架，当前实机对象改为 GLM-5.3-Flash W8A8，先完成 xLLM 验证，再扩展 SGLang、vLLM-Ascend，对比调度、算子、KV Cache 和通信带来的差异，逐步形成自己的框架。早期 GLM-5.3 W8A8C8 调研材料保留其原适用条件。

当前阶段：GLM-5.3-Flash W8A8的xLLM 3P1D完成81920/600、并发16前缀负载48/48请求；TTFT均值8.945秒、TPOT均值637.1毫秒、集群输出22.10 token/s。MTP/缓存/重叠开启，图因OOM与异常输出关闭；尚非精度验收通过，生成性能仍需优化。K8s未改。

## 日报

| 日期 | 工作详情 | 阶段主要目标 | 阶段次要目标 |
| --- | --- | --- | --- |
| 2026/10/19 | 完成xLLM 3P1D长输入前缀压测：48/48成功，TTFT 8.945秒、TPOT 637.1毫秒、输出22.10 token/s；定位PD压缩池缓存限制与OOM，图关闭、精度待验证。[结果及原始材料](xllm-3p1d-prefix-perf-2026-10-19.md)。 | 对比框架性能与实现瓶颈 | |
| 2026/10/15 | 整理 xLLM 非 PD 材料；完成三框架调度、KV Cache、通信源码对比及论文分析。[资料](implementation-comparison.md)；实机待验证。 | GLM-5.3 C8 的 xLLM 部署验证与框架对比准备 | |

## 资料与代码

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
