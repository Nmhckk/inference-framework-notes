# GLM 5.3 Flash 的 xLLM 2P2D 部署与验证

2026-10-19。当前已完成 GLM-5.3-Flash W8A8 的四实例 2P2D 容器部署，12 项功能检查全部通过。正式精度回归正在进行，性能压测尚未启动。以下区分旧模型限制、当前运行路径与实际验证范围。

## 为什么此前不支持 PD，现在可以部署

| 对象与配置 | 已确认结论 | 原因 |
| --- | --- | --- |
| 早期 GLM-5.3 W8A8C8，`kv_cache_dtype=int8` | 当时检查的公开代码拒绝 C8 与 PD 组合；此限制没有删除 | packed C8 缓存布局及传输不能仅靠 PD 开关完成适配 |
| 当前 GLM-5.3-Flash W8A8，`backend=vlm` | 本次实机建链失败 | `VLMEngine` 调用不带传输的缓存分配；报 `KVCacheTransfer not initialized` |
| 当前 Flash W8A8，`backend=llm`、`model_impl=python`、`kv_cache_dtype=auto` | 已实测跑通文本 2P2D 功能链路 | `LLMEngine` 在 P/D 角色使用带传输的缓存分配；Python 模型可由文本工厂加载 |

模型从 GLM-5.3 C8 改成 Flash W8A8、缓存精度从 C8 改成随模型 dtype 的 auto、执行后端从 VLM 改为文本 LLM，是三个独立条件。当前结果不能作为“原 GLM-5.3 W8A8C8 已支持 PD”的证据，也不能推广为 Flash 视觉功能通过验证。此前把“不支持”泛化到所有 xLLM PD 路径的表述过宽，应按模型、缓存精度及后端限定。

固定源码为 `ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb`：

- [量化 KV 与 PD 的拒绝条件](https://github.com/xLLM-AI/xllm/blob/ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb/xllm/core/runtime/worker_impl.cpp#L548)。当前仍保留。
- [VLM 缓存分配](https://github.com/xLLM-AI/xllm/blob/ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb/xllm/core/distributed_runtime/vlm_engine.cpp#L285)。
- [LLM 按角色选择带传输分配](https://github.com/xLLM-AI/xllm/blob/ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb/xllm/core/distributed_runtime/llm_engine.cpp#L486)。
- [Python 文本模型工厂](https://github.com/xLLM-AI/xllm/blob/ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb/xllm/models/model_registry.cpp#L626)。

本次部署的上述三个缓存实现文件与离线固定源码的 SHA256 一致。构建补丁仅补充 TileLang AOT 主进程和子进程显式导入 `torch_npu`，没有删除 C8 PD 拒绝条件或修改模型 kernel 数学逻辑。

## 已运行配置

- 2 个 Prefill、2 个 Decode，每实例 TP8，使用每台设备编号 0–7，共 32 个活动逻辑设备。四台机器各有 16 个逻辑设备，首轮未使用全部设备。
- 文本 LLM 后端，Python 模型实现，PUSH 缓存传输；`reasoning_parser=glm5`。
- Prefill chunk 8192，实例 `max_seqs_per_batch=1`，线性状态槽上限 16。
- 调度重叠、前缀缓存、图模式、MTP 关闭。这是功能基线，尚未调优到峰值性能。
- 镜像：`quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260911`；manifest digest：`sha256:4fec4b05d8e698c830662f085445b77392d3b0e1d74f78be133f36628832100a`。
- xLLM-Service：`b232b7094f17fbab68f010290554b012ee384a0b`；xllm_ops：`b94b87318c2826738c65d6e6f242aa9c1881a3ab`。
- 全部部署及测试在专用 Docker 容器内进行，K8s 未操作。

## 功能证据与精度范围

固定 2P2D 拓扑完成 12/12 功能检查，覆盖算术、固定文本、多轮记忆、长提示、SSE 流式和客户端并发 4。逐请求 ID 对应恰好一个 P 和一个 D，四台各处理 6 个请求。[功能证据](pd-functional-evidence.json)。

12 项功能测试不等于模型精度验收。此前非 PD 对照中保留了长提示推理重复、最终回答错误或输出预算耗尽的记录，尚未确认原因。

GSM8K 回归使用官方 main/test 的固定随机抽样 64 题，抽样种子 `20261019`；four-shot 固定为 train 前四题。采用权重附带的采样默认值 temperature=1.0、top_p=0.95，输出上限 2048，客户端并发 4。只提取最终 content 中 `####` 或 boxed 后的数值；reasoning 中的答案不计分。错误、截断和缺失回答均计入结果，不自动重试覆盖失败。

这轮采用透明的标准库 HTTP 测试脚本，并非 EvalScope 输出。预先设定的初轮回归门槛为正确率至少 95%、无传输错误、无输出截断；这只用于决定是否进入首轮性能测量，不等于完整 1319 题或全面能力精度验收。另做 1K、8K、16K 长上下文已知答案检索与并发状态复用，要求全部通过。

## 性能测试口径

精度两项门槛通过后，再进行 1K/128、8K/128 输入输出负载、并发 1/2/4、每档三次重复的初轮扫描。各单元独立预热，按服务返回的 usage 计算真实输入和输出 token，记录请求延迟、TTFT、TPOT、首个可见答案时间及流式 chunk 间隔。固定输出压测开启 ignore_eos，不用 max_tokens 代替实际计数。

计时统一使用一台客户端的单调时钟；服务器墙钟有偏差，未修改主机时间。吞吐报告整套 2P2D 结果，并同时列出按四台整机、全部 32 个活动逻辑设备归一化的值，不只除以 Decode 设备。少量请求不用于宣称稳定 P99 或最优吞吐。

当前状态：精度运行中；吞吐和延迟结果待精度门槛通过后补充。
