# GLM-5.3-Flash 推理框架研究路线（2026-10-20）

目标不是简单排出 xLLM、vLLM 和 SGLang 的总名次，而是找出各框架在哪种负载、SLO 和硬件拓扑下占优，并把可复用机制沉淀为自己的框架设计。本文件汇总当前官方实现、论文和已有实测，区分已验证事实、源码推断和待测假设。

## 当前判断

1. **下一套实机框架优先 vLLM-Ascend。** 官方 GLM-5.3-Flash 文档已给出 Atlas 800I A2/A3 的共置式非 PD 和 PD 配置，覆盖 W8A8、EP、MTP 与 Decode 图；首轮可以直接复用官方配置，再与 xLLM 的 32 设备非 PD 结果对齐。
2. **SGLang 放在 vLLM 之后。** SGLang 的通用调度、RadixAttention、执行重叠和 HiCache 很值得研究，主线也已经出现 GLM-5.3-Flash 的 KPool、混合 GDN 状态和 PD 代码；但截至本次核查，官方 Ascend 支持表只列到 GLM-5.2，NPU 路线图仍把 GLM-5.x 支持和优化列为推进项。代码存在不等于当前镜像已经完成模型、量化、图和算子联合验证。
3. **xLLM 当前有实测基线，但优势尚不能泛化。** 四实例非 PD 在 80K/600、并发16下达到 82.32 output tok/s，说明多 Decode 实例比当前 3P1D 更适合该负载；图模式、KPool 私有复用和 PD 路径的正确性或性能尚未形成有效优势。
4. **GLM-5.3-Flash 的核心难点已经从普通 KV 分页变成混合状态生命周期。** MLA KV、NoPE KeyPool/indexer、KDA recurrent/conv state、MTP 草稿状态需要共同完成分配、复用、回收、图捕获与 PD 传输。仅比较“PagedAttention 和 RadixAttention”不足以解释这个模型。

## 三框架当前证据矩阵

| 层面 | xLLM | vLLM-Ascend | SGLang | 当前研究结论 |
| --- | --- | --- | --- | --- |
| 调度 | C++ 连续调度，PrefillFirst、DecodeFirst、Unified 等显式策略；当前 GLM Flash+MTP 实际路径仍偏 Prefill-first | V1 用统一 token budget 表示 Prefill、Decode、缓存命中和推测 token；官方示例固定 `max-num-batched-tokens=8192` | 缓存感知排序、`PrefillAdder` 准入、Radix 锁引用、event-loop overlap 和撤回机制 | 比较 token budget、chunk 与请求优先级，不能只看策略名称 |
| 普通前缀缓存 | 块哈希 PrefixCache；当前实测前缀命中约90% | BlockPool/KVCacheManager，模型专用路径要求 Hybrid KV manager | Radix 树将前缀复用与调度结合，可扩展到 HiCache | 用 0/50/93/100% 重复率和冷/热两组实验拆开计算节省与排序收益 |
| GLM 混合状态 | 当前部署能运行 KDA/KeyPool，但图与状态索引补丁未通过完整正确性门槛 | 官方路线把 MLA、KeyPool、tail、KDA state 纳入联合缓存配置；关闭 Hybrid KV manager 会直接拒绝启动 | 主线有 hybrid GDN/KPool 内存池、压缩 tail 与 PD 元数据；Ascend 模型支持仍需确认 | 这是本模型最值得做成自研统一接口的模块 |
| 图执行 | 当前有效结果为 eager；实验图路径出现短检索回归 | 官方 A2 使用 target `FULL_DECODE_ONLY`，MTP draft eager；A3 已支持 draft graph | 通用 CUDA Graph/NPUGraph 能力丰富，但 GLM 混合状态长期上下文存在公开回归案例 | 图性能必须绑定输出一致性、捕获尺寸与额外 HBM 预算一起评价 |
| 推测解码 | 当前实测 MTP=1，接受率约86.82%（含预热） | 官方 A2 非 PD MTP3；A3 示例可到 MTP5，模型专用 draft graph 已支持 | 支持 MTP/EAGLE/DFlash 等多种模式，GLM 路线也在推进 DFlash2 | 记录 committed tokens/step、接受长度分布、验证成本和额外缓存，而非只记录开关 |
| MoE/通信 | HCCL、Mega-MoE 包装、Mooncake；需要 trace 证明实际选中路径 | EP、MC2/All2AllV、模型专用量化和融合算子；官方配置已启用 EP | 多种 MoE backend、通信重叠和 expert placement；NPU 能力依赖 sgl-kernel-npu/通信后端 | 保存每专家 token、最大 rank 时间、dispatch/GEMM/combine，定位尾部 rank |
| PD | xLLM 有通用 Mooncake，但 GLM 私有状态和 KPool 组合仍受限；现有 3P1D Decode 瓶颈明显 | GLM 混合状态 PD 已有合入项和官方 1P1D 配置，但 layerwise offload、128K 冷热一致性仍有开放问题 | PD connector、HiCache、动态角色切换较完整；GLM+Ascend 联合支持尚未形成官方稳定配方 | PD 是否更好取决于 SLO、P/D 比例和传输，不作为默认目标 |
| 工程成熟度 | 已在当前机器跑通，日志和恢复行为已有直接证据 | GLM-5.3-Flash Ascend 官方支持最明确，适合作为下一框架 | 通用能力强、变化快，适合第三阶段研究和吸收设计 | 先完成可比结果，再投入高风险适配 |

### vLLM-Ascend 的模型专用实现

- [GLM-5.3-Flash 官方教程](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/GLM5.3-Flash.md)基于 v0.30.0RC1，明确支持 A2/A3。A2 共置式配置为 DP2×TP8，启用 EP、W8A8、MTP3 和 `FULL_DECODE_ONLY`。
- [适配路线 #15665](https://github.com/vllm-project/vllm-ascend/issues/15665)把 KeyPoolIndexer 与 LightningIndexer 区分开，并记录 Hybrid cache、NoPE、PD、量化和性能优化的合入状态。PD 基础能力已合入，但 MemCache/Mooncake layerwise 跟进和 128K 冷热差异仍未完全关闭。
- 固定源码 `1026eedfd2a13662d54f78e1c45566989a7157af` 中，[cache_config.py](https://github.com/vllm-project/vllm-ascend/blob/1026eedfd2a13662d54f78e1c45566989a7157af/vllm_ascend/models/glm5next/cache_config.py)联合描述 MLA/indexer/tail 布局；[indexer_kpool.py](https://github.com/vllm-project/vllm-ascend/blob/1026eedfd2a13662d54f78e1c45566989a7157af/vllm_ascend/attention/indexer_kpool.py)维护 KPool 元数据、MTP 持久 slot mapping 和 NoPE 约束。
- 这形成一个待验证假设：vLLM 在 GLM-5.3-Flash 上可能优于当前 xLLM 的部分，不是通用 PagedAttention，而是模型状态被纳入统一 scheduler/cache/graph 生命周期，以及更深的 MTP 和 Decode 图组合。

### SGLang 的价值与当前边界

- [SGLang 论文](https://arxiv.org/abs/2312.07104)的核心是 RadixAttention、缓存感知调度和结构化程序执行。它适合重复系统提示、工具说明、多轮和树状 Agent 请求。
- 当前主线的 [GLM-5.3-Flash cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx)面向 NVIDIA/AMD，低延迟和高吞吐配方对 MTP 的选择不同：低延迟使用推测解码，高并发吞吐可能关闭推测解码以避免验证开销。
- [Ascend 支持表](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/ascend-npus/reference/support_models.mdx)当前列出 GLM-5/5.1/5.2，尚未列出 GLM-5.3-Flash；[NPU 2026 H2 路线图 #25598](https://github.com/sgl-project/sglang/issues/25598)仍将 GLM-5.x、DSpark/DFlash、HiCache 和 Mooncake NPU 支持列为目标。
- [GLM-5.3-Flash 问题跟踪 #37524](https://github.com/sgl-project/sglang/issues/37524)说明即使 CUDA 路径也曾出现 MTP、图重放、FP8 KV、HiCache 恢复和工具调用正确性问题。它提醒本项目必须把长上下文状态恢复和真实 Agent 输出加入门槛。
- SGLang 的通用优势仍值得吸收：Radix 树与调度联动、执行重叠、分层缓存、丰富的请求结构；但在 910C 上应先做“能否正确启动”的资格检查，再进入性能比较。

### xLLM 应继续验证的优势

- [xLLM 技术报告](https://arxiv.org/abs/2510.14686)提出服务/引擎解耦、xTensor 虚拟连续内存、图执行、MTP 与通信优化。源码有 C++ 调度和 Mooncake 集成，工程路径较短，适合研究低 Host 开销和服务恢复。
- 当前直接证据仅支持：四个非 PD TP8 实例在既定长输入前缀负载上，比 xLLM 自身 3P1D 有更高 Decode 容量；尚不支持“xLLM 比 vLLM/SGLang 更快”。
- 后续 profiler 若显示 Host gap、调度或 kernel launch 占比低，C++ 控制面不能成为主要优势解释；若占比高，再进一步拆出 C++ 服务层、图和异步执行各自贡献。

## 论文阅读矩阵

| 主题 | 文献 | 可迁移机制 | 对本项目的验证问题 |
| --- | --- | --- | --- |
| KV 分页 | [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) | 分页、引用计数、共享前缀，降低碎片和重复 | 相同 HBM 下可接纳多少请求；GLM 私有状态是否成为新容量上限 |
| 前缀树调度 | [SGLang](https://arxiv.org/abs/2312.07104) | RadixAttention、缓存感知排序 | 高命中请求的收益是否以低命中请求 p99 为代价 |
| 长 Prefill 调度 | [Sarathi-Serve](https://arxiv.org/abs/2403.02310) | chunked prefill、decode-maximal batch | chunk 从2K到16K时 TTFT、TPOT、吞吐和最大 chunk gap 如何变化 |
| PD 资源规划 | [DistServe](https://arxiv.org/abs/2401.09670) | P/D 分离、阶段并行和 goodput | 2P2D、3P1D、非 PD 在同总卡数和相同 SLO 下谁能承载更高到达率 |
| PD 硬件分工 | [Splitwise](https://arxiv.org/abs/2311.18677) | 根据 Prefill/Decode 特性独立选资源 | 同构 910C 上还有多少隔离收益，KV 传输是否抵消收益 |
| KV 中心服务 | [Mooncake](https://arxiv.org/abs/2407.00079) | 分布式 KV 池、预测式拒绝、长上下文服务 | 缓存命中、传输、拒绝和超时是否完整进入吞吐分母 |
| 昇腾 PD | [P/D-Serve](https://arxiv.org/abs/2408.08147) | Ascend 集群上的 KV 传输和调度 | HCCL/SDMA/RDMA 路径、跳数和块生命周期如何影响尾延迟 |
| 分布式前缀路由 | [Preble](https://arxiv.org/abs/2407.00023) | 在缓存复用与负载均衡间联合决策 | 四实例路由是否应优先命中节点；热点前缀会不会制造负载倾斜 |
| 非前缀复用 | [CacheBlend](https://arxiv.org/abs/2405.16444) | 选择性重算并融合多个缓存片段 | 面向 RAG/Agent 的片段复用能否扩展出独立能力，精度如何验收 |
| 虚拟连续 KV | [vAttention](https://arxiv.org/abs/2405.04437) | 虚拟地址连续、物理页按需分配 | 与 xTensor 比较页映射成本、算子约束、图兼容和 NPU 适配成本 |
| 多头推测 | [Medusa](https://arxiv.org/abs/2401.10774) | 多草稿头、树验证 | 接受率和验证成本在低并发、高并发下的交叉点 |
| 特征级推测 | [EAGLE](https://arxiv.org/abs/2401.15077) | 特征预测、lossless speculative sampling | 与模型原生 MTP 比较额外权重、接受长度、显存和维护成本 |
| SLA 调度 | [Past-Future Scheduler](https://arxiv.org/abs/2507.10150) | 根据历史输出分布预测未来峰值内存 | 能否减少过度保守排队和激进准入导致的 OOM/抢占 |

论文中的历史倍数只用于理解机制。不同模型、GPU/NPU、量化、并发和指标定义下的数字不进入三框架排名。

## 实验路线

### 阶段 0：资格门槛

每个框架先完成以下检查，任何一项失败都不进入性能榜：

- 固定镜像 digest、代码提交、CANN/torch-npu、权重与 tokenizer 哈希。
- 关闭 MTP 和图，完成短请求、8K/128、80K/600、共享前缀和冷前缀请求。
- 对固定提示保存首32个 token、结束原因、输出长度及参考输出；检查重复、乱码、工具调用和状态复用污染。
- 再分别打开 MTP、图、缓存恢复，每次重复正确性门槛。不能用“HTTP 200”替代正确输出。

### 阶段 1：严格共同基线

使用相同 W8A8 权重、32个逻辑设备、chat template、采样参数和 AISBench 数据。先关闭前缀复用、MTP、图和 PD，保留框架运行所必需的模型专用缓存。测试：

- 1K/128、8K/1K、80K/600；并发1/4/8/16/32。
- 保存 TTFT、TPOT、ITL/chunk gap、Output TPS、Total TPS、失败率、峰值 HBM 和实际 batch。
- xLLM 与 vLLM 优先做 4节点×TP8；若官方 vLLM A3 路径必须 TP16，则单列 2节点×TP16，不伪装成同拓扑。

### 阶段 2：逐机制消融

| 实验 | 扫描项 | 必须保存 | 主要回答的问题 |
| --- | --- | --- | --- |
| 前缀缓存 | repeat 0/50/93/100%，冷启动/预热 | 实际命中 token、Prefill 时间、路由目标、p99 | Radix/块哈希/路由谁真正少算了多少 |
| Prefill chunk | 2K/4K/8K/16K | 每步 P/D token、最大输出停顿、队列 | 长 Prefill 怎样影响 Decode |
| MTP | off/1/3/5（按框架支持） | 接受长度直方图、committed token/step、验证时间、HBM | 低延迟与高吞吐下何时值得开启 |
| 图 | eager/Decode图/完整图 | capture尺寸、预留与实占HBM、回放命中、输出一致性 | launch收益是否被显存和动态形状抵消 |
| MoE并行 | 可行的 TP/DP/EP 组合 | 每专家token、最大rank、dispatch/GEMM/combine | 通信、专家不均还是算子成为瓶颈 |
| PD | 非PD、2P2D、3P1D | P排队/计算、传输、D排队/计算、全部设备 | 在何种SLO和输入输出比下值得分离 |

### 阶段 3：各框架最佳配置

给每个框架相同的调优时间和配置试验数量，在共同精度门槛、总设备数与负载下启用其原生优势。共同基线与最佳配置分表。比较帕累托前沿，而不是只报单个最大 TPS：

- 横轴为用户生成速度或 TPOT，纵轴为 Output TPS。
- 点必须附 TTFT、失败率、设备数和完整配置。
- PD 的吞吐除以全部 P+D 设备；缓存预热成本和未完成请求进入统计。

### 阶段 4：在线工作负载与稳定性

- 用固定并发之外的 Poisson 到达、突发流量和混合 ISL/OSL，扫描到达率直到队列持续增长。
- 定义至少两组 SLO，例如交互优先和吞吐优先，报告满足约束的 goodput。
- 连续运行2小时，注入单实例退出和路由重连；记录恢复时间、失败请求与缓存失效。
- 加入真实工具/Agent 长系统提示，验证前缀复用、reasoning parser、tool parser 和长上下文状态正确性。

### 阶段 5：原因归因

对每个显著差距至少给出一种直接证据：

- profiler 中算子、collective、Host gap 和空闲时间；
- scheduler 日志中的排队、batch、chunk、抢占和实际 token budget；
- 缓存层的命中、分配失败、回收、offload/transfer 字节；
- MTP 接受率及草稿/验证开销；
- MoE 每 rank 与每专家负载。

只有“控制变量变化 → 内部指标变化 → 端到端指标变化”三者闭环，才归因为框架机制。

## 夜间服务器计划

白天完成镜像、启动命令、数据集、结果模板和源码阅读，夜间只执行必须占机的步骤。

| 夜次 | 任务 | 退出条件 | 预期产物 |
| --- | --- | --- | --- |
| N1 | vLLM-Ascend 环境核查、镜像导入、非 PD 启动 | 短请求与8K请求正确，版本和设备清单完整 | 部署记录、启动日志、资格门槛结果 |
| N2 | 32设备共同基线和80K/600复测 | 至少两次有效运行，无OOM和错误输出 | 原始结果、基线表、与xLLM同条件对照 |
| N3 | MTP、图、chunk、缓存消融 | 每个候选通过短/长正确性门槛 | 各开关收益与失败边界 |
| N4 | vLLM最佳配置、并发/到达率曲线、profiler | 形成帕累托候选并解释主要瓶颈 | 性能曲线、trace、精简日报 |
| N5 | SGLang Ascend 资格检查 | 明确官方镜像/分支是否支持当前W8A8；不能则形成阻塞证据 | 兼容矩阵、缺失补丁清单 |
| N6+ | SGLang基线与优化，或转向其他已适配模型验证通用机制 | 与前两框架使用相同统计口径 | 三框架机制对照 |

服务器时间不足时优先级为：正确性门槛 > vLLM共同基线 > vLLM原生优化 > profiler > SGLang适配。没有有效共同基线前，不花整夜搜索极端峰值。

## 自研框架候选模块

当前先形成接口方向，不急着实现：

1. **统一状态描述器**：把 token KV、线性注意力 recurrent/conv state、稀疏 indexer/KeyPool、MTP state 作为不同 state group，声明分页方式、生命周期、图安全性和传输方式。
2. **状态感知准入与调度**：用真实峰值状态预算决定准入，支持 Prefill/Decode token budget、缓存收益、SLO 与抢占成本联合决策。
3. **缓存与计算联合路由**：借鉴 RadixAttention/Preble，在命中收益和实例负载之间选择，而不是纯轮询。
4. **拓扑控制器**：根据 ISL/OSL、到达率和 SLO，在非 PD、2P2D、3P1D 或动态角色之间选择；把 KV 传输成本纳入模型。
5. **异步执行与图安全边界**：统一描述哪些状态可捕获、哪些 shape 需要重编译、哪些操作必须 eager，避免模型适配靠散落补丁维持。
6. **可归因观测**：请求级时间线贯穿 router、scheduler、cache、transfer、model runner 和 collective，为每个端到端差距保留证据。

最终方向要由跨框架实验决定。若差距主要来自模型算子，自研重点应放在后端和状态布局；若差距来自队列、缓存路由与恢复，则优先做服务控制面。
