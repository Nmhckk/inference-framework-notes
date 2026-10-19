# Flash 部署模式小规模对照

2026-10-19。用户以实际效果选择部署模式；2P2D 不是预设性能赢家。全量1319题精度任务已取消，本轮仅做必要正确性检查及小规模吞吐、延迟对照。已完成对照并选择四台非PD作为当前默认；2P2D方案保留可恢复。

## 控制变量

同四台910C、每台TP8，共32个活动逻辑设备；相同Flash W8A8权重、xLLM提交及算子、auto KV缓存类型。故障设备始终隔离，第四台仅使用物理8–15。batch=1、chunk=8192、图/MTP/prefix cache/调度重叠关闭。K8s不变。

A为两台Prefill加两台Decode；B为四个DEFAULT实例，每个独立完成Prefill和Decode。两种模式均通过相同xLLMService端口访问，B显式开启service routing，PD关闭。只改变本项目模型进程；服务及专用etcd保持运行。

输入目标1K/8K、固定输出128 token；并发1/4，每格4条测量请求，重复两轮；各格先以对应并发预热，不计入吞吐。实际输入token、request ID、SSE事件和服务端路由均归档。固定128包括推理token，不能当作最终可见答案速度。输出吞吐按总输出token除客户端测量墙钟时长，所有P/D设备均计入资源分母。

## 判断范围

比较吞吐、TTFT和端到端延迟，同时检查成功率与实际路由覆盖。不用8条样本宣称稳定P99或饱和峰值，不把本轮固定batch1结果推广到优化后的框架上限。2P2D只有两台Decode，四个合并实例有四台可做Decode；低并发短输出下PD可能无法弥补传输和资源分工代价。长输入混合请求、更多并发或不同P/D配比可能改变结果，需要独立验证。

## KV Cache 与 PD 的限制为何不矛盾

固定提交 `ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb` 的 `worker_impl.cpp` 中，报错位于 `kv_cache_dtype()=="int8"` 条件内部：禁止INT8量化KV Cache和PD同时启用，原判断仍在，没有通过删除检查绕过。当前Flash使用W8A8权重/激活，KV为auto，文本LLM+Python后端，因此不触发这条C8限制。prefix cache关闭只影响跨请求前缀复用，不代表推理时不用KV Cache。

[固定提交源码](https://github.com/xLLM-AI/xllm/blob/ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb/xllm/core/runtime/worker_impl.cpp#L520)。

## 实测结果与选择

两模式各8格、32条测量请求及20条预热请求全部成功；实际prompt token及128输出逐条匹配。两轮吞吐用总输出token除两轮测量时长合并，延迟对每格8条请求取中位数。

| 输入目标 | 并发 | 2P2D输出TPS | 非PD输出TPS | 非PD/PD | PD/非PD中位TTFT秒 | PD/非PD中位总延迟秒 |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1 | 6.14 | 7.62 | 1.24× | 0.717 / 0.731 | 20.80 / 16.79 |
| 1024 | 4 | 11.79 | 30.03 | 2.55× | 1.076 / 0.705 | 31.19 / 16.67 |
| 8192 | 1 | 4.45 | 5.16 | 1.16× | 8.173 / 8.155 | 28.83 / 24.36 |
| 8192 | 4 | 10.07 | 20.92 | 2.08× | 12.269 / 8.115 | 38.98 / 24.05 |

**选择：四台各一个非PD实例，继续通过同一路由提供服务。** 所有四组吞吐均更高，端到端中位延迟更低；1K并发1的TTFT为0.731秒，相比PD的0.717秒略高，小样本不宣称统计差异。PD测量每条匹配一个P和一个D，四实例各出现16次；非PD每条只匹配一个DEFAULT实例，四实例各8次。没有把单机结果乘四冒充集群实测。

当前batch1下，PD仅有两台专做Decode，而非PD四台均可Decode，是资源分工的直接差异。但单请求TPOT也有差异，尚无profiler证据把全部收益分摊给传输、调度或算子；不能仅用“省去KV传输”解释全部提升。其他并发、长输出、SLO和调优设置需要另测。

非PD切换后first8自然停止检查8/8正确、8/8有可见答案且自然结束。此小样本用于部署回归，不代表完整模型精度。测试结束后32个活动设备health均OK、当前双比特ECC及待隔离双比特页均0；K8s运行容器ID与原基线一致。没有重置故障设备、清计数、修改驱动、改模型/算子数学代码或运行完整1319题。

## 当前运行与复现

四台各TP8、DEFAULT角色、`enable_service_routing=true`、`enable_disagg_pd=false`默认值、`backend=llm`、Python模型实现、auto KV。第四台继续使用仅暴露健康物理8–15的恢复容器，其他三台物理0–7。统一API端口28888，模型名`glm53-flash`。具体源提交、镜像和依赖沿用[恢复配置](decode-recovery-runtime.json)。

- [汇总JSON](topology-ab-summary.json)、[CSV](topology-ab-summary.csv)：两轮数据和指标口径。
- [原始A/B记录](topology-ab-raw.tar.gz)：逐条SSE时序、usage、服务端请求ID匹配与8题答案。
- [运行与检查摘要](topology-ab-runtime.json)：实际角色/参数、设备映射、测试后健康及K8s保留情况。
- [固定长度压测](benchmark_pd.py)、[自然停止检查](benchmark_natural.py)、[汇总程序](summarize_topology_ab.py)。
- [非PD启动](launch_flash_combined.sh)、[公共启动参数](launch_flash_baseline.sh)、[PD启动](launch_flash_pd.sh)。这些脚本在已准备的专用容器内运行；不是任意镜像的通用安装教程。

本地拓扑切换控制器保留归档日志，仅停止本项目匹配角色的模型进程。当前没有自动测试链继续运行；恢复PD前先确认现有角色，使用相应停止/切换方式，不能在DEFAULT实例占用端口时直接启动PD。
