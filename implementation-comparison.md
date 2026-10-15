# xLLM、SGLang 与 vLLM 的调度、KV Cache 和通信实现对比

日期：2026-10-15。面向四台昇腾 910C、GLM-5.3-W8A8C8 项目。当前完成源码与论文分析，尚无实机性能结论。

三家均有连续调度、前缀复用及多种通信路径。差异主要在请求排序与准入、缓存回收方式、执行重叠，以及模型实际选中的硬件后端。**不能用“RadixAttention 比 PagedAttention 快”解释整个框架差距：前者还涉及复用索引与调度，分页则解决存储和访问问题，两者并非互斥。**

本文固定 xLLM `cbc03ff`、SGLang `9ea5472`、vLLM `fc8e08a`、vLLM-Ascend `23d1d90`；完整提交、函数与行号见 [源码索引](implementation-source-index.json)。这些是独立阅读快照，尤其 vLLM 与 vLLM-Ascend 尚未验证能作为一套运行版本。既有 [部署候选版本](versions.json) 保持不变。

## 先明确三种“混合”和三类通信

非 PD 是一个实例承担 Prefill 和 Decode；混合批是一个执行批次同时包含两种 token；计算通信重叠是执行时间线上的并行。三者需要分别记录，不能由其中一个开关推断另外两个。

| 通信类别 | 传什么、何时传 | 应观测什么 |
| --- | --- | --- |
| TP / DP 相关集合通信 | 张量切分后的部分结果或 token，通常每层发生；DP 有时还需步数及 batch 协调 | collective 次数、消息量、暴露在关键路径上的等待 |
| MoE EP 分发与合并 | token 按路由发到专家，专家输出再回到原位置 | 每专家 token 分布、dispatch/GEMM/combine 时间及最大 rank 尾部 |
| PD / 分层缓存传输 | 跨实例或 HBM/Host 移动 KV、元数据、量化附加信息 | 注册、排队、打包、传输、重排、完成通知与块生命周期 |

Mooncake KV 传输不能代替所有 TP/EP 通信；使用 HCCL 也不意味着这些路径的实现或性能相同。[xLLM 通信][X6]、[SGLang NPU][S7]、[vLLM 平台分发][V6]、[Ascend MoE][A2] 提供了不同层次的实际入口。

## 调度：每一步让哪些请求做多少工作

### xLLM

`ContinuousSchedulerBase` 使用策略对象。工厂先根据配置生成 batch mode，再选择 `PrefillFirstPolicy`、`DecodeFirstPolicy` 或 `UnifiedPolicy`。[策略选择][X1]

- **PrefillFirst**：先处理未完成的 Prefill chunk，再接新 Prefill；普通路径在没有选中 Prefill 时调度 Decode。打开 Chunked Prefill 本身不保证 Decode 能加入同一批。[实际顺序][X3]
- **DecodeFirst**：先接纳 Decode，再分配剩余 token/序列/时延预算给 Prefill；新请求准入还参考每个 DP 组的占用，代码中有 1.1 倍已用块的预留系数。[预算与准入][X2]
- **Unified**：混批且采用 `multi_slo_and_prio` 时，P/D 进入统一队列；配置启用时用时延预测控制 chunk，内存不足会抢占并回队。这是可选路径，不是所有非 PD 实例的默认行为。[选择条件][X1]、[抢占][X10]

**对项目的影响：** 当前候选方案关闭 `enable_mix_batch`，所以不能称为已启用 Decode 优先混批。新阅读快照还会在 CP>1、有效 KV 分片>1 或 MTP>0 时关闭混批。后续打开优化必须核对最终生效配置，不能把互斥开关的组合当成一个公平测试点。[条件代码][X1]

**取舍分析：** 显式策略与预算便于做 TTFT/TPOT 定向实验；代价是策略、模型限制和预测误差相互影响。Prefill 优先可能让长请求插入时的 Decode 间隔变大；Decode 优先也可能增加新请求排队。这些是控制流推导出的待测风险，不是当前集群已经出现的问题。

### SGLang

链路为等待队列 → `SchedulePolicy.calc_priority` → `PrefillAdder` → batch 执行。排序可选 FCFS、LPM、DFS_WEIGHT、HRRN 等；缓存感知策略先计算前缀命中，影响谁先进入。`PrefillAdder` 在准入过程中固定匹配节点，核算未计算输入、生成空间、页面取整和请求槽位等预算，不能把“命中缓存”直接等同于“可立即运行”。[排序][S2]、[调用链][S3]、[准入][S4]

`event_loop_overlap` 可以在当前批执行之后、其结果处理之前推进上一批结果，减少 CPU 准备与设备执行的串行空档；连续 Prefill、依赖和同步边界仍可能关闭本批 overlap。显存不足时 `retract_decode` 缩减运行批，并根据请求类型及备份条件选择撤回或终止，不是无代价地释放 KV。[循环][S1]、[撤回][S6]

**取舍分析：** 重复长前缀、多轮或树状请求值得研究缓存感知排序；小 batch、短步长值得研究调度重叠。树查询、排序与引用维护有 CPU 成本；优先处理高命中请求也可能伤害低命中请求的排队时延。需按命中率分组报告 p99，而不是只看整体均值。

### vLLM V1

`Scheduler.schedule` 用“已计算 token 数”和“本轮需要追上的 token 数”统一表示工作，覆盖 Prefill、Decode、前缀命中与推测 token。实现先遍历 running，再处理 waiting，受 token budget、活动请求数和 KV 分配约束；running 中也可能有未完成 Prefill，不能将其简化为“严格 Decode 优先”。[调度入口][V1]

块分配失败时按策略选抢占对象；`_preempt_request` 释放块、清零计算进度并重新入队。恢复后可能利用仍存在的前缀块，因此重置进度不等于所有历史 token 必然全部重算。另有 `AsyncScheduler` 通过输出占位维护尚未返回的结果，并处理抢占后的过期输出。[抢占][V2]、[异步调度][V3]、[缓存查询][V4]

**取舍分析：** 统一 token 进度便于把多种执行方式纳入同一预算模型；普通调度顺序不等于按缓存命中收益排序。高压下的抢占、重复计算和异步状态维护会增加尾延迟或实现复杂度。与 SGLang 比较时必须记录异步开关，不能说“只有 SGLang 能重叠 CPU 调度”。

## KV Cache：复用索引、分配器和算子布局要分开

| 框架与路径 | 读到的实现 | 潜在收益与代价 |
| --- | --- | --- |
| xLLM 普通 PrefixCache | token 块的链式哈希；逐块命中，遇 miss 停止；LRU 回收跳过在用块。[源码][X4] | 可以复用共享前缀；收益受块边界和实际命中影响，缓存保留也占用容量 |
| SGLang 普通 RadixCache | token 序列加 `extra_key` 命名空间；匹配会按页对齐并可能分裂节点；锁引用保护活跃路径，从可淘汰叶节点逐步回收。[源码][S5] | 树结构便于复用分叉前缀并参与排序；不是任意 token 精度的无成本复用，页大小与树维护都需测量 |
| vLLM V1 | `KVCacheManager` 查询前缀块；`BlockPool` 管理块引用计数和空闲队列。命中块增加引用，归还后保留哈希直到被复用淘汰。[查询][V4]、[分配与回收][V5] | 分配/复用边界清晰；缓存命中、页面取整及 KV group 协调影响有效容量；完整命中仍需考虑末 token logits 计算 |

这里分析的是指定类，不能概括所有混合注意力、HiCache、SWA 或模型专用分配器。实际启用哪个类，要在部署日志中确认。

[xLLM 报告](https://arxiv.org/html/2510.14686v1) 的 xTensor 用虚拟连续地址映射离散物理页，目的是兼顾算子访问与按需分配。当前工厂确有虚拟页和 Host offload 分支，但由开关选择；不能只引用论文就认定本项目 C8 默认走 xTensor，也不能把论文对分页访问的取舍直接当作今天所有后端的性能排名。[工厂][X5]

[PagedAttention 论文](https://arxiv.org/html/2309.06180v1) 解释了分页及共享为什么能降低 KV 内存浪费；[SGLang 论文](https://arxiv.org/html/2312.07104v2) 进一步讨论 RadixAttention 与缓存感知调度。可迁移到本项目的是实验问题：相同容量下多接纳了多少请求、少算了多少 Prefill、额外花了多少 CPU 时间。旧论文的基线版本与今天不同，加速倍数不能照搬。

**C8 特别注意：** 缓存的逻辑块管理不等于 INT8 存储格式兼容。GLM 的主 KV、RoPE、量化 scale 和 Indexer 缓存需要分别核对。当前 xLLM worker 仍拒绝这条 MLA INT8 路径的 PD 模式；有通用传输引擎并不能消除这一限制。[启动检查][X9]。既有权重能否完整加载仍需真实配置、分片与算子执行证据，当前资料不能替代验收。

## 通信：要追到实际后端和完成事件

### xLLM 的昇腾路径

Python HCCL 包装将 collective 放到调用方当前 NPU stream，并通过 `xllm_ops` 调用。接口要求连续、非空缓冲区，图捕获还约束 HCCL 扩展模式及对象生命周期。`mega_moe` 包装融合 EP dispatch、专家计算和 combine，并区分 BF16 与带 scale 的 W8A8 输入。是否在 GLM 当前分支实际选中，要继续核对模型分发与运行 trace。[HCCL][X6]、[融合包装][X7]

独立的 Mooncake 引擎负责注册内存、检查 byte region、批量提交 KV 传输；这是另外一条生命周期，不能靠 collective 完成就推断 KV 已可读。[传输][X8]

**取舍分析：** 当前 stream 和融合接口有利于减少启动与中间步骤；相应代价是对算子包、ABI、布局和同步协议的依赖更强。C++ 控制面也不自动保证端到端更快：只有 CPU 调度处于关键路径时，减少控制开销才可能明显改善服务指标。

### SGLang 的昇腾路径

`NpuCommunicator.all_reduce` 调用设备进程组；另有 `quant_all_reduce`，执行动态 INT8 量化、数据与 scale 的 all-gather、还原和求和。后者有潜在带宽收益，也增加量化/还原开销，且是通信精度变化，不能与 W8A8 权重量化混为一谈。[代码][S7]

`ascend_fuseep` 路径调用 `fused_deep_moe`，加载后还有专用权重与 scale 布局处理。它复用了 `DeepEPBuffer` 抽象，但不能据类名认定使用 NVIDIA 的原生 DeepEP 二进制。[代码][S8]

Ascend PD 传输类虽然继承 Mooncake 包装，实际导入 `memfabric_hybrid.TransferEngine`，默认 SDMA，另有 device RDMA 初始化处理。依赖存在、协议选择正确、KV 格式一致，是三项不同检查。[代码][S9]

**取舍分析：** 多种通信路径提供调优空间；量化 collective 和融合 EP 需要各自精度与后端兼容验证。不能把普通 all-reduce、FuseEP 与 KV 传输的收益归因到同一个开关。

### vLLM 与 vLLM-Ascend

vLLM 核心按平台选择 communicator；GPU 侧的多种 all-to-all manager 只是候选后端集合。Ascend 的 `NPUCommunicator` 适配公共接口，MoE 实际还在插件的 `TokenDispatcherWithMC2 / AllGather / All2AllV` 中分发；MC2 调用 `npu_moe_distribute_dispatch_v2` 和 `combine_v2`。读 CUDA manager 的 DeepEP/NVLink 分支不足以解释 910C。[核心接口][V6]、[候选 manager][V7]、[NPU 适配][A1]、[实际 dispatcher][A2]、[MoE 编排][A3]

KV connector 又分 scheduler 端与 worker 端：前者决定命中、分配后状态和传输元数据，后者注册、发起和报告完成。Ascend 自己的 Mooncake 门面将默认别名指向 pull 实现，push 类的声明还不等于实现完成。这是该 connector 的边界，不代表插件所有 PD 路径都缺 push。[核心 connector][V8]、[Ascend connector][A4]

**取舍分析：** 平台与 connector 接口有利于复用核心调度；代价是核心版本、插件、传输库与算子之间的接口组合必须锁定。NPU 的占位 all2all manager 还明确不宣称支持该故障容错接口，不能把 GPU 后端能力整体移植为昇腾能力。[适配代码][A1]

## 结合论文形成可验证判断

| 问题 | 论文给出的机制 | 当前代码对应 | 本项目应怎样判断 |
| --- | --- | --- | --- |
| 长 Prefill 是否打断 Decode | [Sarathi-Serve](https://arxiv.org/html/2403.02310v1) 将 Prefill 分块并限制每步预算，研究吞吐与 token 间隔的权衡 | xLLM DecodeFirst、SGLang PrefillAdder、vLLM token budget | 固定负载扫 chunk/budget；同时看 TTFT、TPOT、p99 token 间隔，不能只看吞吐 |
| 前缀复用能否解释差异 | [SGLang](https://arxiv.org/html/2312.07104v2) 的缓存与调度消融；[PagedAttention](https://arxiv.org/html/2309.06180v1) 的内存管理 | Radix 树与块哈希均有当前实现 | 先随机无共享前缀，再引入固定共享比例；按真正少算的 Prefill tokens 与排队变化解释 |
| PD 是否值得 | [DistServe](https://arxiv.org/html/2401.09670v3) 联合考虑 P/D 干扰、阶段资源配置和带宽 | 各框架的 connector、传输引擎与阶段调度 | 在格式支持后再测；把 P 排队、P 计算、传输、D 排队、D 计算分开，全部 P/D 设备计入资源 |
| 虚拟内存与融合是否有效 | [xLLM 报告](https://arxiv.org/html/2510.14686v1) 的 xTensor 与执行优化 | 配置选择的虚拟页路径和 NPU 算子入口 | 确认 GLM 实际走到的分支；用启动显存账本和 trace，不能只由框架语言推断 |

这些机制彼此作用。例如高命中率会减少 Prefill 计算，使 CPU 排序或 EP 通信更显著；增加并发可以提高吞吐，也可能把缓存推到抢占区。分析时要找瓶颈随配置变化的位置，而不是给框架固定排一个名次。

## 无服务器阶段产物与下一轮实验

本轮产物为实现分析、31 个固定提交源码入口，以及下面的实验设计；没有生成任何性能数字。服务器到位后的执行顺序如下，均先通过同一精度门槛。

| 顺序 | 控制变量 | 需要保存的证据 | 能回答什么 |
| --- | --- | --- | --- |
| 1 | 相同权重、模板、采样、输出预算；先非 PD，关闭额外推测解码 | 完整版本、实际 backend/调度器、总卡数、精度与启动显存 | 是否具备可比较基础；不能把量化或输出差异当框架收益 |
| 2 | 长度与到达序列不变，仅切换前缀复用；冷/热状态分开 | 命中 token 数、实际 Prefill token 数、CPU 调度时间、分组 TTFT | 缓存结构与排序分别贡献多少 |
| 3 | 共享比例固定，扫描 chunk 与 token budget；长短请求混合 | 每批 P/D token 数、排队、p99 token 间隔、SLO 达标率 | Prefill 干扰是否被消除，代价是否转移给 TTFT |
| 4 | 仅切换受支持的调度 overlap/async，图与 MTP 保持固定 | CPU/设备时间线、同步点、批次间设备空档 | 是否由主机调度限制；异步收益是否真实落到关键路径 |
| 5 | 逐渐提高并发直到出现回收/抢占 | 活跃/可回收/保留块，抢占次数、重算量、失败率 | 吞吐拐点来自 KV 容量还是计算饱和 |
| 6 | 固定资源总量，选择可容纳模型的 TP/DP/EP 组合；再单测融合通信 | collective/dispatch/combine 消息量与时长、专家负载偏斜、关键路径暴露时间 | 通信量、拓扑还是不均衡决定差距 |
| 7 | 模型 C8 与传输格式均验证后，才增加 PD 对照 | 阶段队列、实际传输字节及时间、总资源下的 goodput/Pareto | 分离能否抵消传输与资源切分成本 |

报告两个层次：共同可用功能下的基线，以及各自经过调优的最好结果。chunk、page 等参数应在各自支持范围扫描，不能强迫不同实现使用无效的同一数值。通信总时长与计算重叠时，不应直接相加作为端到端耗时；优先看落在关键路径上、未被遮蔽的等待。

自研框架的第一步可先建立统一的“准入决策—缓存状态—通信等待”观测接口，再根据上述消融确定要改调度、分配器还是执行后端。当前证据尚不支持复制某一家整体架构，或承诺 SGLang 必然快于 vLLM。

指标定义、8K/1K 工作负载和帕累托方法延续 [性能对比方法](benchmark-methodology.md)。

[X1]: https://github.com/xLLM-AI/xllm/blob/cbc03ff72c826a9b443d055f29f08ec8f2ecfe50/xllm/core/scheduler/scheduler_policy.cpp#L1061
[X2]: https://github.com/xLLM-AI/xllm/blob/cbc03ff72c826a9b443d055f29f08ec8f2ecfe50/xllm/core/scheduler/decode_first_policy.cpp#L30
[X3]: https://github.com/xLLM-AI/xllm/blob/cbc03ff72c826a9b443d055f29f08ec8f2ecfe50/xllm/core/scheduler/prefill_first_policy.cpp#L125
[X4]: https://github.com/xLLM-AI/xllm/blob/cbc03ff72c826a9b443d055f29f08ec8f2ecfe50/xllm/core/kv_cache/prefix/prefix_cache.cpp#L28
[X5]: https://github.com/xLLM-AI/xllm/blob/cbc03ff72c826a9b443d055f29f08ec8f2ecfe50/xllm/core/framework/block/kv_cache_manager_factory.cpp#L134
[X6]: https://github.com/xLLM-AI/xllm/blob/cbc03ff72c826a9b443d055f29f08ec8f2ecfe50/xllm/python/distributed/npu/hccl.py#L30
[X7]: https://github.com/xLLM-AI/xllm/blob/cbc03ff72c826a9b443d055f29f08ec8f2ecfe50/xllm/python/kernels_npu/moe.py#L928
[X8]: https://github.com/xLLM-AI/xllm/blob/cbc03ff72c826a9b443d055f29f08ec8f2ecfe50/xllm/core/framework/transfer/mooncake_transfer_engine.cpp#L280
[X9]: https://github.com/xLLM-AI/xllm/blob/cbc03ff72c826a9b443d055f29f08ec8f2ecfe50/xllm/core/runtime/worker_impl.cpp#L521
[X10]: https://github.com/xLLM-AI/xllm/blob/cbc03ff72c826a9b443d055f29f08ec8f2ecfe50/xllm/core/scheduler/unified_policy.cpp#L306
[S1]: https://github.com/sgl-project/sglang/blob/9ea5472833296e97724f12df716936576d34ddbb/python/sglang/srt/managers/scheduler.py#L1975
[S2]: https://github.com/sgl-project/sglang/blob/9ea5472833296e97724f12df716936576d34ddbb/python/sglang/srt/managers/schedule_policy.py#L168
[S3]: https://github.com/sgl-project/sglang/blob/9ea5472833296e97724f12df716936576d34ddbb/python/sglang/srt/managers/scheduler.py#L4028
[S4]: https://github.com/sgl-project/sglang/blob/9ea5472833296e97724f12df716936576d34ddbb/python/sglang/srt/managers/schedule_policy.py#L1347
[S5]: https://github.com/sgl-project/sglang/blob/9ea5472833296e97724f12df716936576d34ddbb/python/sglang/srt/mem_cache/radix_cache.py#L364
[S6]: https://github.com/sgl-project/sglang/blob/9ea5472833296e97724f12df716936576d34ddbb/python/sglang/srt/managers/schedule_batch.py#L3336
[S7]: https://github.com/sgl-project/sglang/blob/9ea5472833296e97724f12df716936576d34ddbb/python/sglang/srt/distributed/device_communicators/npu_communicator.py#L23
[S8]: https://github.com/sgl-project/sglang/blob/9ea5472833296e97724f12df716936576d34ddbb/python/sglang/srt/hardware_backend/npu/moe/fuseep.py#L41
[S9]: https://github.com/sgl-project/sglang/blob/9ea5472833296e97724f12df716936576d34ddbb/python/sglang/srt/disaggregation/ascend/transfer_engine.py#L26
[V1]: https://github.com/vllm-project/vllm/blob/fc8e08a2a607968226376c53a30cfa6337578f79/vllm/v1/core/sched/scheduler.py#L591
[V2]: https://github.com/vllm-project/vllm/blob/fc8e08a2a607968226376c53a30cfa6337578f79/vllm/v1/core/sched/scheduler.py#L1571
[V3]: https://github.com/vllm-project/vllm/blob/fc8e08a2a607968226376c53a30cfa6337578f79/vllm/v1/core/sched/async_scheduler.py#L14
[V4]: https://github.com/vllm-project/vllm/blob/fc8e08a2a607968226376c53a30cfa6337578f79/vllm/v1/core/kv_cache_manager.py#L264
[V5]: https://github.com/vllm-project/vllm/blob/fc8e08a2a607968226376c53a30cfa6337578f79/vllm/v1/core/block_pool.py#L666
[V6]: https://github.com/vllm-project/vllm/blob/fc8e08a2a607968226376c53a30cfa6337578f79/vllm/distributed/parallel_state.py#L548
[V7]: https://github.com/vllm-project/vllm/blob/fc8e08a2a607968226376c53a30cfa6337578f79/vllm/distributed/device_communicators/all2all.py#L67
[V8]: https://github.com/vllm-project/vllm/blob/fc8e08a2a607968226376c53a30cfa6337578f79/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py#L768
[A1]: https://github.com/vllm-project/vllm-ascend/blob/23d1d9055f2161b3c15613c7fc309f7d5d2ef629/vllm_ascend/distributed/device_communicators/npu_communicator.py#L23
[A2]: https://github.com/vllm-project/vllm-ascend/blob/23d1d9055f2161b3c15613c7fc309f7d5d2ef629/vllm_ascend/ops/fused_moe/token_dispatcher.py#L109
[A3]: https://github.com/vllm-project/vllm-ascend/blob/23d1d9055f2161b3c15613c7fc309f7d5d2ef629/vllm_ascend/ops/fused_moe/moe_comm_method.py#L68
[A4]: https://github.com/vllm-project/vllm-ascend/blob/23d1d9055f2161b3c15613c7fc309f7d5d2ef629/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/connector.py#L1
