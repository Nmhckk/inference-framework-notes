# xLLM 四实例非 PD 长输入性能（2026-10-20）

当前可用配置中，四台各运行一个 `DEFAULT` 实例明显优于 3P1D。相同 AISBench 参数下，48/48 请求成功，集群输出吞吐为 **82.3239 token/s**；3P1D 基线为 22.1016 token/s。本结果是固定长度性能压力测试，不是完整精度验收，也不能直接作为与 vLLM 的公平框架排名。

## 结果

```bash
python3 aisbench_test.py --input_len 81920 --output_len 600 --data_num 48 \
  --concurrency 16 --request_rate 0.93 --dataset_type prefix_cache \
  --repeat_rate 0.93 --prefix_test --dp 2
```

脚本中的 `--dp 2` 只生成两份前缀预热请求，不是服务端数据并行度。

| 指标 | 四实例非 PD | 3P1D 基线 | 变化 |
| --- | ---: | ---: | ---: |
| TTFT 均值 | 23.268 秒 | 8.9446 秒 | 2.60×，变慢 |
| TTFT 中位数 | 11.917 秒 | 4.0605 秒 | 2.93×，变慢 |
| TPOT 均值 | 144.6 毫秒 | 637.1 毫秒 | 降低 77.3% |
| TPOT 中位数 | 134.3 毫秒 | 374.6 毫秒 | 降低 64.1% |
| 输出吞吐 | 82.3239 token/s | 22.1016 token/s | **3.72×** |
| 测量耗时 | 349.838 秒 | 1303.074 秒 | 缩短 73.2% |

48 条请求均输出 600 token，共 28,800 token。模板后的输入长度为 81932×44、81933×1、81934×2、81935×1；客户端最大并发 16、平均在途并发 15.0784。输出吞吐已用 `28800 / 349.8377645` 复核。

这组结果体现了清晰的拓扑取舍：3P1D 用三台承担 Prefill，TTFT 更低，但单个 Decode 实例成为生成瓶颈；四个非 PD 实例同时提供 Decode 容量，TPOT 和总吞吐大幅改善，同时长 Prefill 与 Decode 在实例内争用资源，TTFT 上升。这是本轮数据支持的解释，仍需 profiler 才能细分排队、算子与通信占比。

## 最终配置

- GLM-5.3-Flash W8A8；四台主机、每实例 TP8，共 32 个活跃逻辑设备。
- 四个 `DEFAULT` 非 PD 实例，经同一 xLLM router 做服务路由。
- MTP=1、prefix cache、schedule overlap、chunked prefill 全部开启。
- `max_tokens_per_chunk_for_prefill=8192`、`max_tokens_per_batch=10240`、`max_seqs_per_batch=8`、linear state slots=64。
- eager 模式；`max_memory_utilization=0.80`。
- 图模式关闭；实验图配置曾在压测后出现短检索回归，不能计为有效优化。
- KPool 私有池复用关闭；3P1D 端到端复测从 22.10 降到 20.05 token/s。

原生请求日志覆盖正式 48 条和 AISBench 的 1 条主阶段预热，共 49 条，四实例分布为 13/12/12/12。其前缀命中 token 比例为 89.99%，MTP draft 接受率为 86.82%。这两个比例包含预热，范围没有伪装成仅正式 48 条。

## 显存与稳定性结论

相同配置把 `max_memory_utilization` 设为 0.85 时，四个实例均在 80K、16 并发的动态执行阶段申请约 3.00 GiB 失败，当时每实例只剩约 2.10 GiB；40 条客户端请求失败、8 条滞留，结果作废。降到 0.80 后，给激活和临时缓冲多留约 3 GiB，48/48 完成且四台无 OOM。

因此，对这个混合 GDN/KPool 模型，增加 KV Cache 预算并不等于端到端更快；并发激活和历史池化临时缓冲也必须留出显存。当前 0.80 是已验证的稳定点，并非宣称所有负载下的全局最优。

压测前一次 OOM 会让全部实例从路由表暂时消失；实例恢复注册后，router HTTP 服务没有自动恢复，需只重启本项目的独立 router/etcd。这个现象暴露了 xLLM 服务发现与路由恢复的稳定性风险。节点 8 的系统时钟还比其他节点慢约 7 分钟，可能放大 etcd 租约波动；本轮未修改系统时间。

## 与 vLLM 参考值的边界

用户提供的 vLLM 截图为 TTFT 10.624 秒、TPOT 16.222 毫秒、输出 426.77 token/s。xLLM 当前结果仍明显较低；但截图对应的模型权重、P/D 拓扑、每实例设备数和原始数据集尚未确认，因此这里只保留方向性差距，不计算严格的框架胜负。

本轮说明 xLLM 在该负载上的主要收益来自扩大 Decode 实例数、MTP、前缀缓存和调度重叠。当前 GLM-5.3-Flash 的 PD 路径还会关闭 indexer 私有压缩池缓存，单 D 又限制生成容量；PD 更适合 TTFT 优先的配置，非 PD 更适合本次 80K/600、并发 16 的吞吐目标。

## 验证范围与材料

四台活跃设备健康检查通过，当前 HBM 双 bit 错误计数为 0；K8s 容器 ID 与基线完全一致（17/17/17/11），未操作 K8s。四台运行二进制和关键 Python 源文件哈希一致，图文件已恢复原版。

- [结构化性能摘要](performance-summary.json) · [延迟原表](performance-latency.csv) · [完整 AISBench 原始结果](performance-raw.tar.gz) · [0.85 显存 OOM 证据](memory85-oom-evidence.json)
- [原生路由、缓存与 MTP 证据](native-runtime-evidence.json) · [实际进程参数](runtime-processes.json) · [源码哈希](source-hashes.json)
- [设备健康](active-npu-health.json) · [K8s 保持检查](k8s-preservation.json) · [运行清单](manifest.json)
- [运行脚本](run_combined_aisbench.py) · [重启脚本](restart_combined_optimized.py) · [摘要校验](summarize_combined_aisbench.py) · [证据采集](collect_combined_evidence.py)

后续若继续冲峰值，优先做相同数据的冷启动复测和 profiler 分解，再评估可验证输出的非 PD 图路径；现有 PD 图补丁、KPool 复用和 0.85 显存配置均不能作为默认开启项。
