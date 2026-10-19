# GLM-5.3 推理框架调研与实践

以 GLM 系列模型研究推理框架，当前实机对象改为 GLM-5.3-Flash W8A8，先完成 xLLM 验证，再扩展 SGLang、vLLM-Ascend，对比调度、算子、KV Cache 和通信带来的差异，逐步形成自己的框架。早期 GLM-5.3 W8A8C8 调研材料保留其原适用条件。

当前阶段：完成Flash W8A8的2P2D硬件隔离恢复及同四台非PD对照。当前保留四台非PD：并发4时1K/8K固定128输出吞吐分别30.03/20.92 token/s，为当前2P2D的2.55/2.08倍。仅为batch1小规模基线，未运行全量精度，未改K8s。

## 日报

| 日期 | 工作详情 | 阶段主要目标 | 阶段次要目标 |
| --- | --- | --- | --- |
| 2026/10/19 | 隔离Decode故障NPU；完成2P2D/非PD同资源对照，按实测保留四台非PD，结果及代码已归档。[记录](xllm-topology-ab-2026-10-19.md)。 | xLLM稳定部署与性能基线 | |
| 2026/10/15 | 整理 xLLM 非 PD 材料；完成三框架调度、KV Cache、通信源码对比及论文分析。[资料](implementation-comparison.md)；实机待验证。 | GLM-5.3 C8 的 xLLM 部署验证与框架对比准备 | |

## 资料与代码

- [2P2D与四台非PD对照](xllm-topology-ab-2026-10-19.md)：当前选择、吞吐/延迟、原始证据及可恢复配置。

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
