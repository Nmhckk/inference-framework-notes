# xLLM 与 vLLM 同机非 PD 对照（2026-10-20）

本轮把 xLLM 部署到 vLLM 实测所用的同一台 910C-64G 超节点，使用同一份 GLM-5.3-Flash W8A8 权重、同一组 80K 数据和相同客户端负载。xLLM 以两个独立 TP8 `DEFAULT` 实例使用全部 16 个逻辑设备，对齐 vLLM 的 DP2×TP8 资源规模。48/48 请求成功，服务在测试后仍健康。

## 同机结果

| 指标 | xLLM TP8×2 | vLLM DP2×TP8 | 差距 |
| --- | ---: | ---: | ---: |
| 输出吞吐 | 46.90 token/s | 185.34 token/s | vLLM 3.95× |
| TTFT 均值 | 52.09 秒 | 17.58 秒 | xLLM 2.96× |
| TTFT P50 | 35.64 秒 | 1.94 秒 | xLLM 18.34× |
| TTFT P90 | 148.51 秒 | 49.54 秒 | xLLM 3.00× |
| TPOT 均值 | 236.26 毫秒 | 49.55 毫秒 | xLLM 4.77× |
| 测量耗时 | 614.02 秒 | 155.39 秒 | xLLM 3.95× |

两边均为 81920 目标输入、实际 81932–81934 prompt token、600 输出、48 请求、并发 16、0.93 request/s、93% 重复前缀。数据文件哈希完全一致。测量客户端也完全相同，仅端口从 vLLM 的 `8000,8001` 改为 xLLM 的 `19600,19700`。

## xLLM 配置与运行证据

- 官方开发镜像 `quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260911`，镜像 ID `sha256:ce452e...12d6`；xLLM 源码基线 `ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb`。
- 单机两个 TP8 实例，分别绑定逻辑设备 `0–7` 与 `8–15`；每实例 `max_seqs_per_batch=8`。
- W8A8 权重、prefix cache、chunked prefill、schedule overlap、KPool 与 MTP=1 开启；`max_tokens_per_batch=10240`、prefill chunk=8192、linear state slots=64、显存比例 0.80。
- eager 模式。此前图模式在长压测后出现输出回归，因此本轮未把不稳定图路径计入有效性能。
- 每条正式请求原生日志均记录 `73,728` 个 prefix-cache token；48 条 MTP 共接受 13,450 / 提议 15,321 个 draft token，接受率 87.79%。
- 两个副本分别完成 24 条请求，吞吐分配均衡；副本 0/1 的平均 TPOT 为 250.49/222.02 毫秒。

[客户端汇总](performance-summary-client.json) · [逐请求数据](performance-details.jsonl) · [启动与压测脚本](evidence/) · [原生缓存/MTP请求日志](evidence/replica0-requests.log)

## 为什么当前 vLLM 明显更快

这次已经消除了主机、权重、设备数、数据和客户端差异，差距主要来自当前可稳定启用的执行路径：

1. vLLM 使用 INT8 KV Cache，xLLM 日志显示 `kv_cache_dtype: auto`，当前实际按 BF16 容量规划。vLLM 可保留更多 KV 块并降低长上下文访存压力。
2. vLLM 启用了 `FULL_DECODE_ONLY` 图；xLLM 只能使用已验证正确的 eager 路径。Decode 每步都要重复提交大量小算子，图捕获对 TPOT 的影响直接。
3. vLLM 使用 5-token MTP，xLLM 当前稳定配置为 MTP=1。xLLM 的单步 draft 接受率虽高，但单次可跳过的主模型步数更少。
4. vLLM 同时启用 FlashComm1、DSA CP、prefill/fused MC2 与 shared-expert DP；xLLM 当前两个 TP8 副本没有等价的同机 DP/EP 融合路径。
5. xLLM 的 Prefix Cache 确实命中约 90% 前缀，但每个新请求仍需处理约 8K 未命中 token；并发排队使 TTFT P50/P90 明显上升。缓存生效并不足以抵消 Decode 和通信路径的差距。

因此，本轮结论是“当前 GLM-5.3-Flash 稳定配置下 vLLM 更快”，不把它扩大成所有模型和负载的框架结论。xLLM 的优势仍集中在多机 PD 编排、原生 KPool/GDN 状态管理和灵活实例角色；这些能力需要在 TTFT 优先、多节点和异构 P/D 比例场景单独验证。

## 环境修复记录

新宿主的 Docker seccomp 默认规则会让该开发镜像在 `pthread_create` 时报 `Operation not permitted`，容器使用 `seccomp=unconfined` 后恢复。基础镜像不包含源码编译产生的 `custom_xllm_math` 与 `xlite` 运行库，本轮从同一构建产物中恢复并以只读目录挂载。镜像的 42 个 OCI 内容层在导入前逐一校验，最终镜像 ID与固定值一致。

本轮只停止了本项目的 vLLM 容器；未操作 K8s 和其他用户容器。xLLM 两个端点目前保持运行，`/v1/models` 均返回 200。
