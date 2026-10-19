# xLLM Decode KPool 与图模式实验

目标：在原四台机器的 GLM-5.3-Flash W8A8 3P1D 上，改善 81920/600、48 请求、并发 16 的 Decode 性能。有效整机基线仍为输出 22.10 token/s、TTFT 8.945 秒、TPOT 637.1 毫秒。候选尚未完成整机复测。

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
- 测试脚本与 `pool_reuse.py`：候选代码，尚未整机验收，不作为生产补丁。

最终比较必须使用相同用户脚本与负载，保存真实运行开关、失败记录、TTFT、TPOT 和集群输出 TPS。k8s 不操作。

## 复现与资源

上游 xLLM 固定提交 `ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb`。`model-integration.patch` 修改模型调用处，`pool_reuse.py` 对应模型目录中的 `glm5_next_pool_reuse.py`；开关为 `XLLM_PD_VERIFIED_POOL_REUSE=1`。本目录为实验代码，不提供未经验证的默认生产启用。

NPU 微基准使用 CANN 9、torch 2.9、torch_npu 2.9.0.post2；脚本按原实验路径 `/work/kpool-optimization` 导入同目录原始实现。完整运行参数见 `runtime-kpool-reuse-eager-20261019.json`。本轮保持 3P1D、各 TP8、D batch8、D memory0.75、MTP1、图关闭；唯一模型行为候选为 Decode 的精确池复用。测试前短回归使用独立短输入，正式前缀生成排除前一轮所选长输入 ID。

## 对框架比较的意义

xLLM 的 C++ 调度层并不意味着模型的 Python 执行路径必然更快。本实验的瓶颈候选集中在长历史整理、池化和 Python/NPU 调用开销；同一框架切换模型实现或图路径就会改变表现。后续对比 SGLang、vLLM 时，应同时记录是否重建全历史、缓存失效策略、图可用性、调度合批和通信路径，而不能只按框架名字归因。当前数据只支持 xLLM 内部实验，不能据此判定三框架胜负。
