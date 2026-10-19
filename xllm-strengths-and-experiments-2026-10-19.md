# xLLM的优势候选、比较边界与优化实验

2026-10-19。项目目标是找出不同框架在什么负载、硬件和约束下有优势，并把可复现机制用于自己的框架设计。部署GLM只是第一条验证路径。本页将作者评测、源码机制与本项目实测分开；没有把功能存在视为已经跑赢SGLang/vLLM。

## 直接结论

xLLM值得重点研究的是国产加速器上的执行优化，以及服务层与执行引擎协同，尤其动态P/D角色、生成式推荐和多后端工程集成。Graph、prefix cache、PD、EPLB、异步调度都不是它独有的能力；优势要靠具体实现和完整服务的达标吞吐证明。

目前本项目只实测了xLLM自己的拓扑差异，尚未部署SGLang/vLLM作为同条件对手，不能宣布跨框架赢家。**后续检索复核发现非PD基线也有失败，先前按吞吐选择非PD的推荐已撤回；性能原始数据仍保留，不作为质量验收通过。**[已完成拓扑实验](xllm-topology-ab-2026-10-19.md)。

## 公开评测是否能证明领先

[xLLM技术报告](https://arxiv.org/html/2510.14686v1#S5)发表于2025年10月，比较对象为vLLM-Ascend v0.10.rc1、MindIE v2.1.rc1，硬件为910B/910C。作者报告Qwen3 0.6B–32B、2048输入/输出、TPOT约束50ms的实验中，910C的xLLM相对对应vLLM-Ascend最高2.2倍。该节没有SGLang基线。

这是作者自测、旧版本和特定SLO/形状的证据，不是2026年当前版本的结论；不能外推到CUDA vLLM，也不能外推到所有GLM或其他模型。报告通过调整请求率匹配SLO，区别于本项目的固定并发闭环测试；未经原始配置、量化、算子和资源口径复核，不合并两类TPS。后续复现需拿到实际启动配置、原始请求与精度记录。

## 机制比较：优势在哪里、又不能说明什么

| 方向 | xLLM已找到的机制 | 对比对象已有能力与边界 | 可验证的优势假设 |
| --- | --- | --- | --- |
| 国产硬件与Host执行 | C++服务/调度、brpc与原生算子；模型可走C++或Python；国产多后端适配 | SGLang含Rust gateway，vLLM也有异步调度和原生算子；不能说C++必然快。当前Flash仍是Python模型前向 | 在Host受限的小模型/小步长下，调度与提交空泡可能更少；需时间线剖析 |
| MoE计算通信重叠 | 多流微批、硬件算子和专家负载均衡路径 | vLLM具备DBO/EPLB，SGLang具备EP及重叠；跨硬件不能只比开关名 | 在特定Ascend算子与拓扑下减少暴露在关键路径上的通信，而不是“只有xLLM能重叠” |
| 动态P/D与SLO路由 | service中有TTFT/TPOT预测、P/D选择及角色索引切换代码 | SGLang Gateway已有PD与分别的P/D路由策略；vLLM也支持分离部署并可接外部服务层 | 在输入/输出比和到达率变化时，比固定P:D资源划分更少空闲、更多请求达标 |
| 在线/离线协同 | 官方服务文档提出利用空闲资源跑离线请求、在线优先 | 需要与带调度/编排层的完整对手比较，不能拿全套xLLM-service对比裸engine | 波峰在线达标不下降、波谷离线完成量提高。当前固定提交未完成整条策略的运行验证，标为文档能力 |
| xTensor/虚拟内存KV | 逻辑连续地址、物理页分配和映射，缓存块管理入口已找到 | 与分页缓存是内存布局/算子接口的不同取舍；SGLang/vLLM也有分页、分层缓存等 | 普通Attention模型在碎片与容量压力下保持兼容算子并减少分配开销；不能直接用于当前Flash |
| 生成式推荐REC | `backend=rec`、固定步数调度、多步执行、beam与有效item过滤的专用路径 | SGLang v0.5.19已加入beam search；一般greedy、JSON约束生成不等同于同beam搜索；应比较固定步整图/过滤/系统集成成本而非宣称只有xLLM支持beam | 固定少量decode轮次、大beam、有效SID约束下减少Host往返与候选管理成本，是值得优先做的场景差异 |
| 全局缓存与路由 | 本地/Host/远端缓存与实例选择协同 | SGLang HiCache已有多级存储和预取，Gateway有cache-aware策略；不能说主流框架只有轮询 | 在同命中率、同资源和同层级存储下提高跨实例复用后的达标吞吐，且不让冷请求饥饿 |

上表比较框架族的机制；CUDA文档中的功能不自动代表vLLM-Ascend或SGLang-NPU已支持，后续必须锁定实际可运行的硬件后端与版本。

SGLang [v0.5.19发布说明](https://github.com/sgl-project/sglang/releases/tag/v0.5.19)明确加入beam search，并列出与推测解码、PD、DP attention、HiCache等组合的限制。旧issue曾说不支持，不能沿用旧结论。REC优势应比较端到端专用执行路径，而非beam功能有无。

证据与反例：

- [xLLM服务层](https://docs.xllm-ai.com/en/features/xllm_service_overview/)、[多流范围](https://docs.xllm-ai.com/en/features/multi_streams/)、[推荐设计](https://docs.xllm-ai.com/en/design/generative_recommendation_design/)。REC的固定调度适合固定轮次和同步候选比较，但新请求可能等待更久，不能套到开放式聊天。
- [SGLang EP/TBO/EPLB](https://docs.sglang.io/docs/advanced_features/expert_parallelism)、[SGLang Gateway](https://docs.sglang.io/docs/advanced_features/sgl_model_gateway)、[HiCache](https://docs.sglang.io/docs/advanced_features/hicache_design)：缓存感知与分层缓存已经存在。
- [vLLM EP/DBO/EPLB](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)、[Graph](https://docs.vllm.ai/en/latest/design/cuda_graphs/)、[DP负载均衡](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/)：已有同类优化，内部DP路由文档描述使用running/waiting队列，且允许接外部负载均衡。比较时须记录实际服务组件。

## 固定提交中核查到的入口与限制

本轮xLLM源码固定 `ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb`，service固定 `b232b7094f17fbab68f010290554b012ee384a0b`；公开文档为2026-10-19访问的动态页面，不能假定全部与运行提交一致。

- [调度重叠](https://github.com/xLLM-AI/xllm/blob/ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb/xllm/core/scheduler/continuous_scheduler.cpp#L526)：逐步执行和overlap分支明确分开，后续应测CPU准备时间、设备空泡与状态同步。
- [服务预测与切换](https://github.com/xLLM-AI/xllm-service/blob/b232b7094f17fbab68f010290554b012ee384a0b/xllm_service/scheduler/managers/instance_mgr.cpp#L940)：用预测TPOT选D、TTFT选P，存在双向角色索引切换；源码存在并不证明Flash已具备任意角色动态切换的正确性与收益。
- [虚拟内存KV限制](https://github.com/xLLM-AI/xllm/blob/ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb/xllm/core/kv_cache/storage/kv_cache.cpp#L347)：明确拒绝index cache、conv cache、SSM cache和linear attention。因此该机制作为其他普通Attention模型的研究对象，不能直接给当前Flash打开。
- [KDA多序列布局](https://github.com/xLLM-AI/xllm/blob/ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb/xllm/python/attention/kda_linear_attention.py#L348)：对packed多序列decode与cu_seqlens有专门处理，并对多序列prefill采取逐序列路径。说明batch>1有实现依据，但仍需要实机数值与状态隔离检查。
- [Flash官方部署范围](https://github.com/xLLM-AI/xllm/blob/ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb/docs/src/content/docs/en/cookbook/autoregressive_models/glm/glm_5_3_flash.md)：DP/EP/prefix cache/PD/MTP仍标开发验收中。不能把通用开关全开作为优化方案；当前普通KV的文本PD已经单独验证，不代表其余组合通过。

## 当前部署的优化顺序

1. 单机batch1→2：保留另外三台，用同一个直连接口，1K输入/64输出、并发2、4请求×2轮；先检查4题自然停止和8条1K/8K独立键值检索，任何数值/状态/传输失败即不推广。
2. 合批通过后，再以相同batch只改变调度overlap；混合短长输入，观察长Prefill插入时Decode间隔与排队。不能只看总TPS。
3. 单独验证ACL Graph：核对Flash所选算子和捕获状态，记录图预留、KV剩余、实际命中率与warmup时间。不要把Qwen图收益直接套用到Flash。
4. MTP最后单独评估，记录接受率、额外状态槽、真实输出token和答案一致性；同模型不同推测设置属于算法/配置收益，单列。

单机测试需临时移出service路由。启用service routing的实例直接HTTP调用曾返回200但空body，基线预热被严格判失败，尚未计入任何性能结果；统一路由服务未异常。后续baseline与候选均关闭单机service routing，通过相同直连端点比较，避免入口差异。

## 跨模型、跨框架实验清单

| 实验 | 建议对象与负载 | 固定变量 | 解释差异所需证据 |
| --- | --- | --- | --- |
| Host/Graph开销 | Qwen3-8B或32B dense，短输出、低并发及并发扫描 | 同精度、权重、Attention后端；统一条件和各自调优分开 | CPU调度/序列化时间、kernel间空泡、图命中、SLO达标吞吐 |
| MoE通信 | DeepSeek系列等支持的MoE，专家负载均衡/偏斜两组 | 同EP/TP拓扑、冗余专家预算、token路由分布 | dispatch/GEMM/combine、最慢rank、暴露通信、EPLB调整成本 |
| 共享前缀 | 已支持prefix cache的Qwen，重复比例0/50/90%，长公共前缀 | 同tokenizer/块大小/缓存容量、冷暖各一组 | cached token、TTFT、缓存抖动、低命中请求尾延迟 |
| 动态P/D | 同一已支持模型，输入/输出比切换及突发到达 | 完整P+D资源与传输网络，校准预测器；固定拓扑对照动态策略 | 队列、角色切换次数/耗时、KV搬运、TTFT/TPOT SLO达标率 |
| KV内存管理 | 普通Attention模型，长短序列混合到达与缓存压力 | 同KV精度/可用HBM；单列page/索引开销 | 有效容量、分配延迟、碎片/映射成本、重算和OOM |
| REC专用执行 | 等价推荐模型，固定3/少量步、beam扫描、有效item约束 | 同beam/top-k、候选质量与约束，不能对比greedy | Host往返、候选排序/过滤、E2E延迟、有效候选质量 |

这些是候选实验，不代表权重已存在、其他框架已部署或功能已经跑通。NPU比较使用实际可用的Ascend实现与匹配依赖；GPU对比另外建实验组，不混入910C数字。动态到达测试同时保存端到端延迟和SLO达标吞吐；固定并发小样本不声称饱和峰值或稳定P99。

## 论文如何用于解释机制

[vAttention](https://arxiv.org/abs/2405.04437)已经研究虚拟连续KV与按需物理分配，因此xTensor值得比较的是国产后端实现、映射成本及算子适配，而非宣布内存思想独占。Sarathi-Serve帮助设计chunk/合批与Decode停顿对照；DistServe帮助约束P/D资源与达标吞吐；SGLang帮助区分缓存复用和程序级并行。阅读重点与迁移边界见[文献路线](literature-map.md)。

## 为自研框架保留的设计问题

优先沉淀统一请求/状态生命周期、可替换调度策略、设备/后端接口、统一trace和可复现负载，再根据实测决定是否实现虚拟连续KV、动态P/D或固定步REC执行。每个模块都应有一个能证伪其收益的对照实验，并记录收益、代价、适用范围与维护成本。此时不急于fork一整个框架并把所有开关迁入。

本轮单机batch2未通过检索门槛，batch1直连控制及原路由非PD复核也出现失败，尚不能归因batch2；候选未进入性能阶段，不推广；证据及恢复对照见[合批实验](xllm-batch-canary-2026-10-19.md)。这也说明框架通用能力不能代替具体模型路径验收。
