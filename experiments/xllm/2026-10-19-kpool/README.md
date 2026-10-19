# xLLM Decode KPool 与图模式实验

目标：在原四台机器的 GLM-5.3-Flash W8A8 3P1D 上，改善 81920/600、48 请求、并发 16 的 Decode 性能。有效整机基线仍为输出 22.10 token/s、TTFT 8.945 秒、TPOT 637.1 毫秒。候选已完成整机复测，未取得收益，关闭复用候选并恢复基线配置。

## 已验证的局部实验

| 实验 | 条件与结果 | 结论 |
| --- | --- | --- |
| 全历史 Gather 向量化 | B8、82534：3.509 → 7.726 ms，输出逐位相同 | 长输入更慢，不采用 |
| Triton 融合池化 | 长输入存在超出阈值的差异；B1、81920 也未加速 | 不采用 |
| 精确历史校验后复用池 | 30 个 NPU 案例逐位相同；B8、81920 重复相同历史：10.473 → 4.146 ms | 仅局部候选，不能外推 TPS |

复用案例覆盖追加、MTP 多 token、回滚、历史改写、批次重排、APE 变化、左侧无效位置和全无效历史。追加 valid mask 与布局一致性检查后，35 个 NPU 用例全部逐位一致；长历史重复微基准为 10.503 → 4.234 ms。短历史复用更慢，因此整模型候选仅在 eager、长度至少 4096、batch 不超过 8 时启用。

## 实现边界

保留上游 PD 私有压缩池禁用条件，不直接删除 `not layer.cache_transfer_enabled`。候选保存私有历史快照，仅在原始历史与有效掩码逐位一致时复用完整池，不完整尾池走原实现。B8 长上下文会额外占用数 GiB HBM，须验证实际显存峰值。

图模式不执行此复用分支。关闭 MTP 的 Decode 图已完成 1/2/4/8 桶捕获；P/D MTP 配置已统一。重启期间发现路由 TIME_WAIT 检查误判及 Decode 失效 master 地址，已修复端口检查并重新注册；重新注册后 1K 输入、并发 4 的四条请求全部返回重复乱码（0/4），此图配置被否决，不进行正式性能比较；反例见 `graph-mtp0-failure-raw.tar.gz`。已恢复 eager＋MTP1，候选通过 1K 和 8K 各 4 条已知答案回归（8/8），随后启动相同命令的 48 请求压测。短回归不等同于长输入完整精度验收。

## 剖析范围

`profile-summary.json` 来自一次带记录形状与显存的冷启动 Decode 步。11 次 pool marker 累计 272.93 ms，history gather 147.84 ms，属于嵌套、侵入式 CPU 时间，不能当作稳态占比。诊断脚本导出 trace 后调用不支持的 `key_averages()` 导致退出，这是诊断代码问题，已移除该 hook。

## 文件

- `pool-reuse-validation.json`：增加独立 valid-mask guard 前的 NPU 微基准。
- `gather-microbenchmark.json`：被否决的向量化 Gather 实验。
- `profile-summary.json`：单步剖析摘要。
- 测试脚本与 `pool_reuse.py`：实验代码，整机复测无收益，不作为生产补丁。

最终比较必须使用相同用户脚本与负载，保存真实运行开关、失败记录、TTFT、TPOT 和集群输出 TPS。k8s 不操作。

## 复现与资源

上游 xLLM 固定提交 `ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb`。`model-integration.patch` 修改模型调用处，`pool_reuse.py` 对应模型目录中的 `glm5_next_pool_reuse.py`；开关为 `XLLM_PD_VERIFIED_POOL_REUSE=1`。本目录为实验代码，不提供未经验证的默认生产启用。

NPU 微基准使用 CANN 9、torch 2.9、torch_npu 2.9.0.post2；脚本按原实验路径 `/work/kpool-optimization` 导入同目录原始实现。完整运行参数见 `runtime-kpool-reuse-eager-20261019.json`。本轮保持 3P1D、各 TP8、D batch8、D memory0.75、MTP1、图关闭；唯一模型行为候选为 Decode 的精确池复用。测试前短回归使用独立短输入，正式前缀生成排除前一轮所选长输入 ID。

## 对框架比较的意义

xLLM 的 C++ 调度层并不意味着模型的 Python 执行路径必然更快。本实验的瓶颈候选集中在长历史整理、池化和 Python/NPU 调用开销；同一框架切换模型实现或图路径就会改变表现。后续对比 SGLang、vLLM 时，应同时记录是否重建全历史、缓存失效策略、图可用性、调度合批和通信路径，而不能只按框架名字归因。当前数据只支持 xLLM 内部实验，不能据此判定三框架胜负。

## 图模式状态索引检查

`npu_paged_attention.py` 的缓存绑定将带 DSA index 的混合模型标记为 `is_mla=True`；`decode_acl_graph.py::_fill_entry` 却据此跳过 `linear_state_indices` 更新，即使 `_has_linear_cache=True`。CPU 检查直接执行该方法，桩替代无关 NPU 元数据 kernel：输入索引 `[3,7,11,19]` 后，原版混合路径仍为 `[0,0,0,0]`；候选条件补充 `not self._has_linear_cache` 后能正确复制，纯 MLA 与非 MLA 线性路径保持原行为。

检查结果见 `graph-metadata-probe.json`，候选改动见 `graph-linear-state-proposed.patch`。这是元数据更新层面的复现，尚未应用到本轮服务，也不能替代 NPU 图执行、padding、MTP 和端到端正确性验证。

## 满负载覆盖限制

当前复用条件按 history tensor 的行数限制为 8，而非按客户端并发数限制。MTP 验证可能展开多行，因此 D batch8 不保证复用分支覆盖满负载。正式压测中，layer3 计数曾停留在 hits=284、misses=216；这是分支覆盖不足的线索，不能当作完整运行的命中率。直接放宽到 16 行会增加私有历史快照显存，需重新验证显存余量、正确性和吞吐，不能仅凭微基准默认开启。

## 完整整机复测（48/48）

使用用户给定 AISBench 命令：81920 输入、600 输出、48 请求、并发16、request_rate0.93、prefix_cache、repeat_rate0.93、prefix_test、dp2。每条实际输出600，共28800 token；模板后输入81932×45、81933×1、81934×2。正式测量耗时1436.655秒，预填充前缀预热不计入该耗时。

| 指标 | 原基线 | 精确池复用候选 |
| --- | ---: | ---: |
| TTFT均值（秒） | 8.9446 | 30.6614 |
| TTFT中位数（秒） | 4.0605 | 11.2938 |
| TPOT均值（毫秒） | 637.1 | 666.1 |
| TPOT中位数（毫秒） | 374.6 | 413.1 |
| 集群输出（token/s） | 22.1016 | 20.0466 |
| 请求成功数 | 48/48 | 48/48 |

候选输出吞吐低约9.3%，未取得端到端收益。相同脚本和参数，但两轮使用新选取的前缀数据，且未做多次随机交叉试验；不能将全部差异归因为补丁。原生前缀token命中率约89.99%，MTP接受率约82.33%（计数窗口含工具预热，不等于正式48条独立统计）。图关闭，MTP1、TP8、3P1D；候选微基准加速不能外推整机。

决定：关闭精确池复用候选，恢复eager＋MTP1基线。图索引补丁继续列为待NPU验证，不纳入本次性能结果。32个活跃设备健康检查通过，当前HBM双比特错误为0；K8s容器ID与记录一致。本次仅证明压测完成及短回归通过，不是完整长上下文精度验收。

证据：[摘要](performance-summary.json)、[延迟CSV](performance-latency.csv)、[原始完整压测归档](performance-raw.tar.gz)、[设备健康](active-device-health.json)、[K8s保留检查](k8s-preservation.json)。

## 状态索引补丁实机验证（后续实验）

补丁将跳过 recurrent state 索引更新的条件限制为纯 MLA；混合 MLA/KDA 仍复制索引，并保持 padding 清零。已保留原文件校验值和回退副本，源补丁见 `graph-state-index.patch`，测试扩展见 `graph-state-index-test.patch`。镜像缺少 pytest 入口需要的 `xllm_export` 扩展，单元测试在收集前阻塞，不能记作通过。

| 实机配置 | 图捕获 | 输出检查 | 决定 |
| --- | --- | --- | --- |
| 补丁＋MTP1，小桶上限4 | draft 1/2/4，target 2/4/8 捕获完成 | 1K检索0/4，重复乱码，全部达到600上限 | 不测正式性能 |
| 补丁＋MTP0，D batch8 | 1/2/4/8 捕获完成 | 1K 4/4、8K 4/4；80K执行4/4、检索2/4，另有漏检/思考截断 | 继续性能试验，不宣称长文本精度通过 |

MTP0下未复现连续乱码；这支持该补丁改善基础图路径，但不能证明它独自解决所有图问题。MTP1图仍不可用，后续需要分别检查draft与verify的状态更新。正式复测使用原81920/600、48请求、并发16命令；P/D MTP均关闭，D图开启，私有池复用关闭。与eager＋MTP1基线比较的是部署配置整体，不能单独归因于图加速。当前正式性能仍在运行。

失败样本与日志：`graph-state-fix-mtp1-raw.tar.gz`、`graph-state-fix-mtp1-logs.tar.gz`。MTP0验证样本：`graph-state-fix-mtp0-1024-raw.tar.gz`、`graph-state-fix-mtp0-8192-raw.tar.gz`、`graph-state-fix-mtp0-81920-raw.tar.gz`。实际配置：`runtime-graph-state-fix-mtp0.json`。
