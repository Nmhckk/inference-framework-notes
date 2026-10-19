# GLM-5.3 推理框架调研与实践

以 GLM 系列模型研究推理框架，当前实机对象改为 GLM-5.3-Flash W8A8，先完成 xLLM 验证，再扩展 SGLang、vLLM-Ascend，对比调度、算子、KV Cache 和通信带来的差异，逐步形成自己的框架。早期 GLM-5.3 W8A8C8 调研材料保留其原适用条件。

当前阶段：Flash W8A8 的 xLLM 文本 2P2D 容器部署完成，功能检查 12/12 通过；GSM8K 抽样 63/64、状态检索 16/16；144 条固定长度压测完成；自然停止补测触发 Decode NPU 异常，稳定性与完整精度参考对照待补。

## 日报

| 日期 | 工作详情 | 阶段主要目标 | 阶段次要目标 |
| --- | --- | --- | --- |
| 2026/10/19 | 完成 Flash xLLM 2P2D 抽样精度回归（63/64）及吞吐/延迟基线，归档 Decode 异常与待修复问题；[记录](xllm-flash-2p2d-2026-10-19.md)。 | xLLM 精度验收与性能基线 | |
| 2026/10/15 | 整理 xLLM 非 PD 材料；完成三框架调度、KV Cache、通信源码对比及论文分析。[资料](implementation-comparison.md)；实机待验证。 | GLM-5.3 C8 的 xLLM 部署验证与框架对比准备 | |

## 资料与代码

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

1. 定位自然停止补测的 Decode NPU 异常和长等待，完成稳定性回归。
2. 补充完整精度与同题参考对照，再逐项调优批处理、图与 MTP。
3. 在相同模型、硬件和负载条件下引入 SGLang、vLLM-Ascend；分别记录统一基线和各自调优结果。
4. 用性能剖析与控制变量实验定位差异，再确定自研模块。

日报只记录当天实际新增进展，附成果链接；未运行的方案标为待验证。当前没有任何框架快慢的实测结论。
