# vLLM-Ascend GLM-5.3-Flash 非 PD 实测（2026-10-20）

在一台空闲的 910C 服务器上完成 GLM-5.3-Flash W8A8 非 PD 混部实测。服务采用 DP2×TP8，共使用 16 个逻辑设备；81920/600、48 请求、并发16负载下，48/48 成功，输出吞吐 **185.34 token/s**，TTFT 均值 **17.58 秒**，TPOT 均值 **49.55 毫秒**。

这是性能与短输出功能验证，不是完整精度验收。测试只使用确认空闲的主机和独立容器，未停止他人进程，未修改 K8s。

## 结果

| 指标 | 结果 |
| --- | ---: |
| 成功请求 | 48/48 |
| 实际输入长度 | 81932–81934 token |
| 每请求输出 | 600 token |
| 总输出 | 28,800 token |
| 测试耗时 | 155.392 秒 |
| 输出吞吐 | **185.3375 token/s** |
| TTFT 均值 / P50 / P90 | **17.577 / 1.944 / 49.541 秒** |
| TPOT 均值 / P50 / P90 | **49.547 / 34.991 / 101.112 毫秒** |

数据与前一轮 xLLM 使用同一组 48 条主请求和 2 条前缀预热请求，文件 SHA256 也一致。负载参数为 `input_len=81920`、`output_len=600`、`data_num=48`、`concurrency=16`、`request_rate=0.93`、`repeat_rate=0.93`。客户端按两个 DP 端口轮询，各承载24条正式请求。

本次使用自编流式客户端复现上述语义，没有调用 AISBench 包。TTFT 取首个内容或 reasoning chunk，TPOT 为 `(请求总耗时-TTFT)/(completion_tokens-1)`，输出吞吐为 `28800/墙钟时间`。因此它适合做当前阶段的接近同条件比较，后续仍需统一客户端复测一次。

## 拓扑和优化

- 镜像：`quay.io/ascend/vllm-ascend:v0.30.0rc1-a3`，digest `sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83`。
- 单机 16 个逻辑设备；两个外部 DP rank，每个 rank TP8，并开启 expert parallel。
- W8A8 权重、INT8 KV Cache、prefix cache、chunked prefill。
- 5-token MTP；运行时载入 `Glm5NextMTPModel`。
- `FULL_DECODE_ONLY` 图模式，四组 Decode 图完成捕获。
- FlashComm1、DSA CP、prefill/fused MC2、shared-expert DP。
- `max_model_len=98304`、`max_num_seqs=16`、`max_num_batched_tokens=4096`、显存利用率0.92。

两路 rank 各得到约 27.5 GiB KV Cache，可缓存 3,284,081 token；98,304 长度下理论并发为33.41。压测后的累计计数显示，前缀缓存查询 4,085,208 token、命中 3,649,536 token，命中率 **89.34%**；MTP 产生 40,605 个 draft token、接受 20,906 个，draft token 接受率 **51.49%**。这些计数包含启动后的预热和诊断请求，未伪装成只覆盖正式48条。

## 与 xLLM 当前结果

| 指标 | vLLM 非 PD | xLLM 非 PD | vLLM 变化 |
| --- | ---: | ---: | ---: |
| 活跃逻辑设备 | 16 | 32 | 一半 |
| TTFT 均值 | 17.577 秒 | 23.268 秒 | 降低 24.5% |
| TPOT 均值 | 49.547 毫秒 | 144.600 毫秒 | 降低 65.7% |
| 输出吞吐 | 185.338 token/s | 82.324 token/s | **2.25×** |
| 每活跃逻辑设备吞吐 | 11.584 token/s | 2.573 token/s | **4.50×** |

这组数据强烈支持 vLLM 当前组合更适合该长输入、高重复前缀、持续 Decode 的负载。原因与运行证据一致：vLLM 的 Decode 图已实际捕获，5-token MTP 每步具有更高的提交上限，INT8 KV 给并发留出空间，DP2×TP8 同时保留两路 Decode 容量；xLLM 当前有效结果仍是 eager、MTP1，图补丁曾发生正确性回归。

它还不是最终框架排名。两者来自不同服务器，vLLM 使用16个逻辑设备和自编客户端，xLLM 使用32个逻辑设备和 AISBench；框架版本、主机拓扑及客户端实现没有完全锁齐。当前可报告“本轮实测 vLLM 明显领先”，不能把全部差距都归因于单一框架机制。下一次有同一批空闲服务器时，应统一客户端、主机和设备数，依次关闭图、MTP、INT8 KV 与前缀缓存做消融。

## 排障结论和剩余空间

- `recompute_scheduler_enable` 只允许 PD Decode，非 PD 配置会被校验器拒绝，本轮没有强行启用。
- fused MC2 与 shared-expert multistream 冲突；保留 fused MC2，关闭 multistream。
- 当前镜像缺少 `vllm._deepselect_C`，没有启用 DeepSelect。
- 当前镜像缺少 `causal_conv1d_update_npu`，因而使用带同步警告的 PyTorch fallback；这是下一轮最明确的镜像/算子优化点。
- 容器未特权运行，`/proc/irq` 只读，IRQ affinity 调优未生效。
- 正式压测后 16 个逻辑设备健康均为 OK，服务日志未匹配到 ERROR、OOM 或 ECC。

## 材料

- [结构化摘要](performance-summary.json) · [48条逐请求结果](performance-details.jsonl) · [运行时证据](runtime-evidence.json)
- [参数化 rank 启动脚本](run_nonpd_rank.sh) · [80K流式压测客户端](benchmark_80k.py)
- [xLLM 同负载结果](../../xllm/2026-10-20-nonpd/README.md)
