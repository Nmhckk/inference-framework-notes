# 推理框架调研与实践

以相同基础模型比较 xLLM、SGLang、vLLM 的调度、KV Cache、算子和通信实现，逐步确定自研框架方向。当前实机模型是 **GLM-5.3-Flash W8A8**；早期 GLM-5.3 C8 资料保留原适用范围。

## 当前结果

已完成 vLLM-Ascend 非 PD 首轮实测：单机 DP2×TP8、16个逻辑设备，81920/600、48请求、并发16下，**TTFT 17.577秒，TPOT 49.547毫秒，输出185.34 token/s**，48/48成功。相同数据和负载下，xLLM四实例非PD为82.32 token/s；vLLM用一半活跃逻辑设备实现2.25倍总吞吐、低65.7% TPOT。客户端和服务器尚未完全锁齐，当前作为强预备结论，不是最终框架排名。

- [vLLM 当前非 PD 最佳实测](experiments/vllm/2026-10-20-nonpd-run/README.md)：DP2×TP8、INT8 KV、5-token MTP、Decode图和前缀缓存的配置、指标与运行证据。
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
| 2026/10/20 | 完成 vLLM-Ascend 非PD实测：单机16逻辑设备、48/48成功，TTFT 17.577秒、TPOT 49.547毫秒、输出185.34 token/s；相同负载下，用xLLM一半的活跃设备实现2.25倍总吞吐。已保存配置、逐请求数据、MTP接受率和前缀命中证据；未动K8s及他人任务。[结果](experiments/vllm/2026-10-20-nonpd-run/README.md) · [对比](experiments/comparisons/2026-10-20-xllm-vllm-nonpd/README.md) | 完成三框架可归因对比 | 统一主机与客户端后做图/MTP/KV消融 |
| 2026/10/19 | 完成 3P1D 长输入基线，输出 22.10 token/s；继续定位 KPool 和图模式问题，复用候选20.05 token/s无改善；图补丁测得34.90 token/s，但压测后回归0/4，已否决并回退，原配置8K仍2/4，正确性继续排查。[结果](xllm-3p1d-prefix-perf-2026-10-19.md) | 对比框架性能与实现瓶颈 | |
| 2026/10/16 | 推进 Flash 容器部署准备，核查离线依赖与 PD 兼容性，准备多机验证。 | xLLM 部署与兼容性验证 | |
| 2026/10/15 | 完成三框架调度、KV Cache、通信源码与文献整理。[资料](implementation-comparison.md) | 框架对比准备 | |

## 归档约定

新增实验统一放在 `experiments/<框架>/<日期与主题>/`，同目录保存说明、配置、摘要和原始证据；首页只展示已验证结果。既有根目录文件保持旧链接可用，由资料目录集中导航。成功、失败和待验证状态分别标明，微基准收益不替代端到端性能。每个阶段及时提交 Git；不上传登录凭据、私钥、模型权重或镜像。
