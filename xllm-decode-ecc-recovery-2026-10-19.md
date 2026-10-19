# Decode NPU 异常定位与硬件隔离回归

2026-10-19。本轮先处理 2P2D 稳定性，再做完整精度和性能测试。当前已识别 D-B 的物理设备 7 存在不可纠正 HBM 多比特 ECC，并将该 Decode 迁移到同机空闲设备 8–15；恢复后 first-8 自然停止测试8/8成功、8/8正确，C1×8、C2×16、C4×32自然停止回归全部正常结束。用户随后缩小验收范围，完整1319题及后续正式性能测试链已取消，尚未运行；当前等待确认“混部”的具体含义，服务拓扑保持不变。

## 根因证据与故障传播

此前只看 rank 0，读到 `507035 / DDR MTE out of range`，堆栈在 `gather_index_history`。开启同步报错后，异常提前到 AllReduce 后的 `hc_post`，底层 plog 明确故障 kernel 为 `aiv_all_reduce_bfloat16_t`。但继续核查全部 rank 后，发现 rank 7 更早发生 `HBM MULTI BIT ECC / Uncorrectable ECC`，错误码 `507054`，事件 `0x80e01801`。

同步定位这一轮，rank 7 在服务端同机时间 13:35:03 报告 HBM 故障，rank 0 在 13:35:11 才报告 AIV 通信异常。较早保留的 rank 7 日志也有相同 ECC 记录。独立切换 AI_CPU 通信后，rank 7 仍报 ECC。因此，AIV 模式本身不是本次全部故障的充分解释，不能凭 rank 0 的末端堆栈给 KV 索引或通信代码定责。

```mermaid
flowchart LR
    A[设备7不可纠正HBM错误] --> B[rank7执行失败]
    B --> C[其他rank集合通信失败]
    C --> D[rank0出现507035]
    D --> E[Decode被摘除或请求不能完成]
    E --> F[SSE缺少DONE和usage]
```

只读 ECC 查询记录：HBM 双比特当前错误计数 3、累计错误计数 6、已隔离页 1、待隔离页 5；单比特当前错误计数 64,548,277。该设备 `health` 概览仍显示 OK，说明单看概览不足以排除内存故障。诊断证据见 [结构化记录](decode-ecc-evidence.json)。实际芯片修复属于运维事项，本次没有重置设备、清除计数或宣称修好硬件。

## 隔离方案

- 原 D-B 使用物理设备 0–7，故障设备 7 不再由本项目实例使用。
- 新建独立 Decode 容器，仅映射空闲的物理设备 8–15；通过 CANN 查询确认容器逻辑设备 0–7 对应物理设备 8–15。新组八个设备的健康状态为 OK、ECC 各计数均为 0。
- P-A、P-B、D-A 设备不变，仍为四台各一个 TP8 实例，总计 32 个活动逻辑设备，保持 2P2D。
- 使用原容器的本地运行环境快照，依赖版本与模型二进制保持一致；基础镜像仍是已归档的官方 20260911 镜像。新快照只在服务器本地使用，不需要重新下载模型或大镜像。
- 恢复原 `HCCL_OP_EXPANSION_MODE=AIV`，关闭同步调试；不以 AI_CPU 或同步执行绕过报错后就宣布修复。模型与算子数学代码未修改。
- 原容器保留路由、专用 etcd 和故障日志；仅停止本项目旧 Decode 进程。K8s 未操作，容器 ID 只读对照一致。

已增加启动保护：旧容器暴露设备7或未暴露健康8–15时拒绝启动D-B，避免后续重启误用坏设备。旧容器拒绝启动、新容器隔离检查均已验证。

[恢复smoke汇总](recovery-smoke-summary.json)、[原始记录](recovery-smoke-raw.tar.gz)。运行配置见 [恢复配置](decode-recovery-runtime.json)，测试前活动设备检查见 [健康摘要](active-device-health-summary.json)。

## 已执行回归与范围调整

恢复后的自然停止请求完整性：smoke 8/8、C1 8/8、C2 16/16、C4 32/32，均返回完整结束标记、usage及可见答案。对应数学正确数分别为8、8、15、31；各组题目有重叠，不能把合并分数当成独立精度样本。1K/8K/16K状态检索也已16/16通过。完整GSM8K和正式性能复测未开始；按用户最新要求取消自动续跑。

以下为原先设计的顺序，现已停止完整验收阶段：

1. 相同自然停止 first-8 GSM8K 用例，验证完整 SSE、最终 content 与 usage。
2. 自然停止 C1×8、C2×16、C4×32，加上 1K/8K/16K 状态检索 16 条。任一阶段错误即停止后续测试，不自动重试覆盖失败。
3. 门槛通过后运行完整 GSM8K main/test 1319 题，沿用 four-shot、temperature=1、top_p=0.95、seed 请求字段 42、输出上限 2048，保留全部错误和截断。完整集合不等于高精度权重或非 PD 对照已经完成。
4. 完整精度门槛通过后，重新测量 1K/128、8K/128、并发 1/2/4、三次重复的吞吐与延迟。

测试采用同一客户端单调时钟，服务器墙钟未修改。旧精度和性能数据继续保留，但其使用过故障设备；不能把修复前后的性能差异直接归因于软件优化。若后续另行要求完整验证，新结果将另存目录，不覆盖旧结果。

## 参考

- [官方同步报错说明](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/ENV/docs/en/environment_variable_reference/ASCEND_LAUNCH_BLOCKING.md)：同步模式用于定位，会影响性能。
- [CANN 设备 ID 映射](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/runtimeapi/aclcppdevg_03_2060.html)：区分容器逻辑 ID、用户设备 ID 与物理 ID。
- [固定版本 HCCL 调用](https://github.com/xLLM-AI/xllm/blob/ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb/xllm/core/kernels/npu/collectives.cpp)。
