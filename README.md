# 推理框架调研与实践

以相同基础模型比较 xLLM、SGLang、vLLM 的调度、KV Cache、算子和通信实现，逐步确定自研框架方向。已从 GLM-5.3-Flash 扩展到 xLLM 官方重点适配的 **Qwen3.5** 模型族；早期 GLM-5.3 C8 资料保留原适用范围。

## 当前结果

已完成 xLLM 与 vLLM-Ascend 的同机非 PD 实测：同一台 910C-64G 超节点、同一份权重、16个逻辑设备、相同数据和客户端下，xLLM TP8×2 为 **TTFT 52.09秒、TPOT 236.26毫秒、输出46.90 token/s**；vLLM DP2×TP8 为 **17.58秒、49.55毫秒、185.34 token/s**，两边均48/48成功。当前已验证性能配置下 vLLM 吞吐为3.95倍；主要配置差异是 INT8 KV、Decode图、5-token MTP和通信融合。xLLM 压测后约12分钟 API 无响应并需要重启，稳定性门禁未通过；结论不扩大到所有模型与负载。

新增 Qwen3.5-4B BF16 单卡同机对照：80K输入、600输出、16并发下，xLLM为 **65.52 token/s、TTFT P90 65.61秒、TPOT P50 220.05毫秒**；vLLM-Ascend为 **69.12 token/s、69.62秒、208.88毫秒**，均48/48成功。vLLM吞吐领先5.5%，xLLM TTFT P90领先5.8%，说明换到xLLM重点模型后差距明显收窄，但尚未整体反超。

继续完成 xLLM 优势消融：Schedule Overlap 使吞吐提升 **0.48%**、TPOT P50 降低 **1.33%**；Zero-Evict 在32并发高 KV 压力下各指标差异低于0.2%，未验证出收益。93%重复前缀负载下，xLLM Prefix Cache 的 TPOT P50 为11.43毫秒，但 TTFT P50为111.01秒、总吞吐80.29 token/s；vLLM为57.33毫秒、1.04秒和261.21 token/s。xLLM确有更低的单用户Decode间隔，代价是Prefill串行和整体goodput下降。

论文、官方 workflow、主分支代码和公开 Issue 复核后，已校准优势验证方向：此前 80K/600 单卡测试没有命中论文的主要优势区间。下一步优先验证 **Qwen3.5-27B 的 GDN Decode SuperOp＋MTP3**，再按论文条件测试 **Qwen3-1.7B/4B、2048/2048、TPOT SLO 下的 QPS 与 TP1/2/4 扩展**。当前主分支的多机多流并行会显式报错，已排除；静态 PD 不再代替 Engine 公平对照。

按校准后的 2K/2K Decode 负载完成三轮同卡复测：Qwen3.5-4B BF16、2033–2034 输入、2048 输出、16 并发下，xLLM 最佳配置为 **1099.10 token/s、TTFT P90 1.313秒、TPOT均值14.118毫秒**；vLLM-Ascend rc2 为 **951.61 token/s、1.530秒、16.304毫秒**。xLLM 吞吐领先 **15.50%**，三轮标准差仅0.91 token/s。Schedule Overlap 消融确认贡献 **3.46%** 吞吐和 **3.85%** TPOT；Graph 在该并发下没有稳定增益。服务日志确认 Qwen3.5 的 fused `split_qkv_rmsnorm_mrope` NPU kernel 在8个Full-Attention层生效。

继续补齐 SGLang 同卡三轮基线：Decode Full Graph＋Overlap 为 **1057.02 token/s、TPOT 14.371毫秒**，xLLM 吞吐仍高 **3.98%**；关闭 SGLang Overlap 后降至 **937.22 token/s**，消融确认 Overlap 贡献 **12.78%** 吞吐。Full-Attention 的 QKV/Norm/RoPE 融合两边都有；进一步固定提交验证确认，GDN 主体的宽边界 MegaGDN 是 xLLM 在被测 NPU 版本中的独有实现：每个满批 Decode step 的框架级设备算子边界为24，对 SGLang 至少72。结构独占性已经成立，具体性能贡献仍待 profiler。SGLang 在910C上还需显式设置BF16 state，否则默认FP32 state首请求报错。

- [vLLM 当前非 PD 最佳实测](experiments/vllm/2026-10-20-nonpd-run/README.md)：DP2×TP8、INT8 KV、5-token MTP、Decode图和前缀缓存的配置、指标与运行证据。
- [Qwen3.5-4B 单卡同机对照](experiments/comparison/2026-10-21-qwen35-xllm-vllm/README.md)：同卡公平配置、逐请求数据、服务日志和源码归因。
- [xLLM 优势消融与 Prefix Cache 对照](experiments/comparison/2026-10-21-qwen35-xllm-advantage-validation/README.md)：Schedule Overlap、Zero-Evict、状态槽显存与两框架前缀缓存实测。
- [xLLM 优势验证方向校准](research/2026-10-21-xllm-advantage-validation-direction.md)：论文、仓库、Issue 与代码证据，Qwen3.5 GDN/MTP 和 Qwen3 SLO 测试路线；[可执行测试矩阵](research/2026-10-21-xllm-advantage-test-matrix.csv)。
- [Qwen3.5-4B 论文区间同卡复测](experiments/comparison/2026-10-21-qwen35-paper-slo/README.md)：三轮 xLLM/vLLM 对照、Schedule Overlap 与 Graph 消融、逐请求数据和源码归因。
- [Qwen3.5-4B 三框架同卡对照](experiments/comparison/2026-10-21-qwen35-three-framework/README.md)：补齐SGLang三轮结果与Overlap消融，校正三框架融合、图和调度归因。
- [Qwen3.5 三框架代码级性能归因](experiments/comparison/2026-10-21-qwen35-code-attribution/README.md)：固定提交核查调度、MegaGDN、状态缓存和融合调用路径；区分已证实因果、强候选与缺失证据，并附可执行代码门禁。
- [xLLM MegaGDN 独占优势验证](experiments/comparison/2026-10-21-megagdn-validation/README.md)：9项自动检查确认固定版本中xLLM将GDN主体的至少三个设备算子边界收进一个SuperOp；同卡三轮性能方向一致，并明确尚缺直接A/B的因果边界。
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
| 2026/10/21 | 完成Qwen3.5-4B三框架同卡复测：2K/2K、16并发下xLLM 1099.10、SGLang 1057.02、vLLM 951.61 token/s；xLLM分别领先3.98%/15.50%。继续完成MegaGDN专项验证，9/9代码门禁通过：xLLM每个满批Decode step为24个GDN主体设备边界，SGLang至少72个；结构独占性成立，性能贡献待Profiler定量。[MegaGDN验证](experiments/comparison/2026-10-21-megagdn-validation/README.md) · [三框架结果](experiments/comparison/2026-10-21-qwen35-three-framework/README.md) | 用可证伪实验确认xLLM优势 | 补匹配Profiler和数值等价A/B |
| 2026/10/20 | 完成xLLM与vLLM同机非PD对照：xLLM为TTFT 52.09秒、TPOT 236.26毫秒、46.90 token/s，vLLM为17.58秒、49.55毫秒、185.34 token/s，均48/48成功；继续完成图/MTP/INT8消融，确认主KV INT8不支持当前NPU Flash路径、KPool拒绝Indexer INT8，Graph＋MTP1为0/2、Graph＋MTP0已完成4/10正确，均不得计性能。外部训练占卡后停止测试，未动K8s及他人任务。[同机结果](experiments/xllm/2026-10-20-samehost-tp8x2/README.md) · [优化边界](experiments/xllm/2026-10-20-optimization-sweep/README.md) | 完成三框架可归因对比 | 空机复测MTP2与draft TP1 |
| 2026/10/19 | 完成 3P1D 长输入基线，输出 22.10 token/s；继续定位 KPool 和图模式问题，复用候选20.05 token/s无改善；图补丁测得34.90 token/s，但压测后回归0/4，已否决并回退，原配置8K仍2/4，正确性继续排查。[结果](xllm-3p1d-prefix-perf-2026-10-19.md) | 对比框架性能与实现瓶颈 | |
| 2026/10/16 | 推进 Flash 容器部署准备，核查离线依赖与 PD 兼容性，准备多机验证。 | xLLM 部署与兼容性验证 | |
| 2026/10/15 | 完成三框架调度、KV Cache、通信源码与文献整理。[资料](implementation-comparison.md) | 框架对比准备 | |

## 归档约定

新增实验统一放在 `experiments/<框架>/<日期与主题>/`，同目录保存说明、配置、摘要和原始证据；首页只展示已验证结果。既有根目录文件保持旧链接可用，由资料目录集中导航。成功、失败和待验证状态分别标明，微基准收益不替代端到端性能。每个阶段及时提交 Git；不上传登录凭据、私钥、模型权重或镜像。
