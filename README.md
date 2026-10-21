# 推理框架调研与实践

以相同基础模型比较 xLLM、SGLang、vLLM 的调度、KV Cache、算子和通信实现，逐步确定自研框架方向。已从 GLM-5.3-Flash 扩展到 xLLM 官方重点适配的 **Qwen3.5** 模型族；早期 GLM-5.3 C8 资料保留原适用范围。

## 当前结果

已完成 xLLM 与 vLLM-Ascend 的同机非 PD 实测：同一台 910C-64G 超节点、同一份权重、16个逻辑设备、相同数据和客户端下，xLLM TP8×2 为 **TTFT 52.09秒、TPOT 236.26毫秒、输出46.90 token/s**；vLLM DP2×TP8 为 **17.58秒、49.55毫秒、185.34 token/s**，两边均48/48成功。当前已验证性能配置下 vLLM 吞吐为3.95倍；主要配置差异是 INT8 KV、Decode图、5-token MTP和通信融合。xLLM 压测后约12分钟 API 无响应并需要重启，稳定性门禁未通过；结论不扩大到所有模型与负载。

新增 Qwen3.5-4B BF16 单卡同机对照：80K输入、600输出、16并发下，xLLM为 **65.52 token/s、TTFT P90 65.61秒、TPOT P50 220.05毫秒**；vLLM-Ascend为 **69.12 token/s、69.62秒、208.88毫秒**，均48/48成功。vLLM吞吐领先5.5%，xLLM TTFT P90领先5.8%，说明换到xLLM重点模型后差距明显收窄，但尚未整体反超。

- [vLLM 当前非 PD 最佳实测](experiments/vllm/2026-10-20-nonpd-run/README.md)：DP2×TP8、INT8 KV、5-token MTP、Decode图和前缀缓存的配置、指标与运行证据。
- [Qwen3.5-4B 单卡同机对照](experiments/comparison/2026-10-21-qwen35-xllm-vllm/README.md)：同卡公平配置、逐请求数据、服务日志和源码归因。
- [xLLM 与 vLLM 同机对照](experiments/xllm/2026-10-20-samehost-tp8x2/README.md)：同主机、同权重、同设备数、同数据和同客户端的结果与归因。
- [xLLM 图、MTP 与 INT8 边界验证](experiments/xllm/2026-10-20-optimization-sweep/README.md)：图输出反例、KV/Indexer INT8 代码限制、补丁和待复测项。
- [xLLM 当前非 PD 结果](experiments/xllm/2026-10-20-nonpd/README.md)
- [xLLM 与 vLLM 首轮对照](experiments/comparisons/2026-10-20-xllm-vllm-nonpd/README.md)：区分实测差异、代码机制和未统一的控制变量。
- [vLLM 非 PD 部署方案](experiments/vllm/2026-10-20-nonpd-plan/README.md)：固定镜像、32设备拓扑、优化参数、时间估计和参数化启动脚本。
- [3P1D 基线与原始证据](xllm-3p1d-prefix-perf-2026-10-19.md)
- [KPool 与图模式候选实验](experiments/xllm/2026-10-19-kpool/README.md)：局部结果与整机结果分开，失败方案保留。
- [历史报告与资料目录](CATALOG.md)

## 阅读入口

| 主题 | 入口 |
| --- | --- |
| 后续研究路线 | [GLM-5.3-Flash 三框架研究规划](research/2026-10-20-framework-comparison-roadmap.md) |
| 三框架实现与优势分析 | [源码对比](implementation-comparison.md) · [xLLM 优势候选](xllm-strengths-and-experiments-2026-10-19.md) |
| 公平性能比较 | [测量方法](benchmark-methodology.md) · [实验模板](benchmark-run-template.json) |
| 论文 | [文献路线](literature-map.md) |
| 部署与故障 | [Flash 部署](xllm-flash-2p2d-2026-10-19.md) · [ECC 恢复](xllm-decode-ecc-recovery-2026-10-19.md) |
| 历史 C8 方案 | [非 PD 准备](non-pd-plan.md) |

## 精简日报

| 日期 | 工作详情 | 阶段主要目标 | 阶段次要目标 |
| --- | --- | --- | --- |
| 2026/10/21 | 选择xLLM官方重点支持的Qwen3.5-4B，完成xLLM与vLLM-Ascend单卡同机80K/600、16并发对照：xLLM 65.52 token/s、TTFT P90 65.61秒，vLLM 69.12 token/s、69.62秒，均48/48成功；结合Zero-Evict、Schedule Overlap和vLLM抢占路径完成源码归因。[结果与证据](experiments/comparison/2026-10-21-qwen35-xllm-vllm/README.md) | 扩展模型并识别xLLM优势边界 | Qwen3.5-27B W8A8＋MTP、KV压力测试 |
| 2026/10/20 | 完成xLLM与vLLM同机非PD对照：xLLM为TTFT 52.09秒、TPOT 236.26毫秒、46.90 token/s，vLLM为17.58秒、49.55毫秒、185.34 token/s，均48/48成功；继续完成图/MTP/INT8消融，确认主KV INT8不支持当前NPU Flash路径、KPool拒绝Indexer INT8，Graph＋MTP1为0/2、Graph＋MTP0已完成4/10正确，均不得计性能。外部训练占卡后停止测试，未动K8s及他人任务。[同机结果](experiments/xllm/2026-10-20-samehost-tp8x2/README.md) · [优化边界](experiments/xllm/2026-10-20-optimization-sweep/README.md) | 完成三框架可归因对比 | 空机复测MTP2与draft TP1 |
| 2026/10/19 | 完成 3P1D 长输入基线，输出 22.10 token/s；继续定位 KPool 和图模式问题，复用候选20.05 token/s无改善；图补丁测得34.90 token/s，但压测后回归0/4，已否决并回退，原配置8K仍2/4，正确性继续排查。[结果](xllm-3p1d-prefix-perf-2026-10-19.md) | 对比框架性能与实现瓶颈 | |
| 2026/10/16 | 推进 Flash 容器部署准备，核查离线依赖与 PD 兼容性，准备多机验证。 | xLLM 部署与兼容性验证 | |
| 2026/10/15 | 完成三框架调度、KV Cache、通信源码与文献整理。[资料](implementation-comparison.md) | 框架对比准备 | |

## 归档约定

新增实验统一放在 `experiments/<框架>/<日期与主题>/`，同目录保存说明、配置、摘要和原始证据；首页只展示已验证结果。既有根目录文件保持旧链接可用，由资料目录集中导航。成功、失败和待验证状态分别标明，微基准收益不替代端到端性能。每个阶段及时提交 Git；不上传登录凭据、私钥、模型权重或镜像。
