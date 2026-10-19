# 推理框架论文与源码阅读路线

更新：2026-10-15。先从问题出发阅读设计、实验与消融，再回到当前源码，最后形成可验证假设。下表是首轮文献梳理，重点阅读机制与评测章节；尚未复现论文实验，也不把历史加速倍数迁移到 GLM-5.3 C8。

## 当前优先阅读

| 论文 | 核心问题与阅读重点 | 对本项目的实验启发 | 迁移边界 |
| --- | --- | --- | --- |
| [xLLM Technical Report](https://arxiv.org/html/2510.14686v1) | 服务与引擎解耦，执行流水、图模式、xTensor、MTP；第 5 节评测 | 对照当前源码，逐项验证图、流水和 MTP 的时延与显存收益 | 报告覆盖的模型/版本不等于当前 GLM C8 路径；论文有 PD 不代表该路径支持 PD |
| [PagedAttention / vLLM](https://arxiv.org/html/2309.06180v1) | 分页 KV 与共享，内存碎片对 batch 的限制；第 6 节 | 记录有效 KV 容量、占用与抢占/重算，解释吞吐拐点 | 论文的 E2E/输出长度归一化时延包含首 Token 成本，不等于当前 TPOT |
| [SGLang](https://arxiv.org/html/2312.07104v2) | RadixAttention、缓存感知调度与多调用程序；第 6.3 节消融 | 无共享前缀/高共享前缀分开测；区分缓存结构、调度与应用并行贡献 | 当时基线的能力与当前版本不同，不能据旧论文断言今天 vLLM 没有前缀缓存 |
| [Sarathi-Serve](https://arxiv.org/html/2403.02310v1) | Chunked Prefill 与持续 Decode，吞吐和尾延迟权衡；第 5 节 | 扫描 token budget/chunk size，观察长 Prefill 对 Decode 停顿的影响 | 单看最大吞吐会遗漏尾延迟代价；CUDA 研究实现不直接用于昇腾 |
| [DistServe](https://arxiv.org/html/2401.09670v3) | P/D 干扰、资源划分、通信与 goodput；第 6 节 | 扫描到达率与 SLO，拆分排队、计算和 KV 传输 | 文中 per-GPU goodput 是满足达标率目标的最大到达率/卡数；与工具逐轮达标请求数/秒区分 |
| [TaiChi](https://arxiv.org/html/2508.01989v1) | PD 合并与分离在不同 TTFT/TPOT 约束下的权衡；第 4 节 | 用多组时延约束比较方案，研究混合部署是否有适用区间 | 该版本评测为 A100 上的 Qwen2.5，不能直接作为 910C 最优拓扑证据 |

阅读顺序建议：xLLM → PagedAttention → SGLang → Sarathi-Serve → DistServe → TaiChi。先弄清当前系统，再理解缓存、调度、分离架构之间的联系。

## 服务层与通信补充

| 文献 | 借鉴点 | 本项目落点 |
| --- | --- | --- |
| [Mooncake](https://arxiv.org/html/2407.00079v1) | KV 中心的服务组织、分布式缓存和过载处理 | 比较命中率与传输成本，记录拒绝和超时，避免只统计成功请求 |
| [P/D-Serve](https://arxiv.org/html/2408.08147v1) | Ascend/MindSpore 上的 P/D 组织、请求转发及 D2D 传输优化 | 补充昇腾互连、负载变化与集群服务视角；论文规模与软件栈不等于四台 xLLM |

这些论文提出的机制值得研究，具体收益依赖模型结构、负载分布、硬件互连和实现版本。新设计至少要同时说明适用场景、额外开销和失败边界。

## 可配套阅读的工程

| 工程 | 入口 | 阅读用途 |
| --- | --- | --- |
| xLLM | [当前源码地图](research-2026-10-15.md) | 将报告机制映射到 C8、调度和执行代码 |
| Mini-SGLang | [固定版本结构说明](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/docs/structures.md)、[功能说明](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/docs/features.md) | 用较小实现理解 API、Tokenizer、Scheduler、Detokenizer 与执行流程；作为学习参考，不作为已适配 GLM/昇腾的部署候选 |
| Sarathi-Serve | [官方仓库](https://github.com/microsoft/sarathi-serve)中的 `osdi-experiments` | 研究实验配置怎样对应论文图；项目自述为研究原型 |
| DistServe | [官方仓库](https://github.com/LLMServe/DistServe) | 研究 PD 配置与性能分解；兼容性和维护状态需在复现前核实 |
| vLLM 压测 | [固定提交的 CLI 指南](https://github.com/vllm-project/vllm/blob/fc8e08a2a607968226376c53a30cfa6337578f79/docs/benchmarking/cli.md) | 核对统计口径、缓存污染和在线压测流程 |
| AIPerf | [v0.12.0 指标定义](https://docs.nvidia.com/aiperf/v0.12.0/reference/ai-perf-metrics-reference) | 对照文章中的 Total TPS、用户生成速度和 ITL；实际选用时再锁定版本 |

## 每篇文献留下什么

每次阅读只推进一个明确问题，保存以下内容：论文版本与对应代码、负载及硬件、机制、对比基线、收益成立的条件、额外成本、下一步实验。不要只积累论文标题或最大加速倍数。

近期可验证假设：

- 重复前缀较多时，缓存策略可能主要降低 Prefill 工作量；用实际命中率和 TTFT 解释收益。
- Prefill 分块可能减轻 Decode 停顿，但 chunk 太小可能增加执行开销；需要扫描参数而非简单开关比较。
- 图模式和 MTP 的收益可能被显存预留与缓存容量抵消；观察不同并发区间的前沿变化。
- PD 的收益可能受 KV 传输和资源划分影响；非 PD 是有比较价值的基线，不只是启动失败后的替代方案。

上述都是实验假设。评测方法和验收口径见 [性能对比方法](benchmark-methodology.md)。

## 2026-10-19 补充：把优势拆成可证伪的机制

- [vAttention](https://arxiv.org/abs/2405.04437)及[作者工程](https://github.com/microsoft/vattention)：用虚拟内存维持连续地址，同时按需分配物理内存。与xLLM xTensor对照时，研究页映射、分配、算子兼容与设备适配成本；虚拟连续KV不是xLLM独有思想，不把CUDA原型当作Ascend现成能力。
- Sarathi-Serve用于理解合批/chunk预算与Decode停顿；DistServe用于比较PD隔离后的SLO达标吞吐；SGLang论文用于共享前缀/分支程序。三者分别对应不同瓶颈，不把论文排行榜拼成通用框架排名。
- xLLM技术报告的评测基线为旧vLLM-Ascend与MindIE，未提供本项目所需的当前SGLang对照；新研究索引见[优势候选与实验设计](xllm-strengths-and-experiments-2026-10-19.md)。
