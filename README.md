# 推理框架调研与实践

以相同基础模型比较 xLLM、SGLang、vLLM 的调度、KV Cache、算子和通信实现，逐步确定自研框架方向。当前实机模型是 **GLM-5.3-Flash W8A8**；早期 GLM-5.3 C8 资料保留原适用范围。

## 当前结果

原四台机器，3P1D，81920 输入 / 600 输出，48 请求，并发 16：**TTFT 8.945 秒，TPOT 637.1 毫秒，集群输出 22.10 token/s**。48/48 请求完成，属于性能记录，尚非完整精度验收。

- [正式基线与原始证据](xllm-3p1d-prefix-perf-2026-10-19.md)
- [KPool 与图模式候选实验](experiments/xllm/2026-10-19-kpool/README.md)：局部结果与整机结果分开，失败方案保留。
- [历史报告与资料目录](CATALOG.md)

## 阅读入口

| 主题 | 入口 |
| --- | --- |
| 三框架实现与优势分析 | [源码对比](implementation-comparison.md) · [xLLM 优势候选](xllm-strengths-and-experiments-2026-10-19.md) |
| 公平性能比较 | [测量方法](benchmark-methodology.md) · [实验模板](benchmark-run-template.json) |
| 论文 | [文献路线](literature-map.md) |
| 部署与故障 | [Flash 部署](xllm-flash-2p2d-2026-10-19.md) · [ECC 恢复](xllm-decode-ecc-recovery-2026-10-19.md) |
| 历史 C8 方案 | [非 PD 准备](non-pd-plan.md) |

## 精简日报

| 日期 | 工作详情 | 阶段主要目标 | 阶段次要目标 |
| --- | --- | --- | --- |
| 2026/10/19 | 完成 3P1D 长输入基线，输出 22.10 token/s；继续定位 KPool 和图模式问题，复用候选复测20.05 token/s，未改善，保留22.10基线。[结果](xllm-3p1d-prefix-perf-2026-10-19.md) | 对比框架性能与实现瓶颈 | |
| 2026/10/16 | 推进 Flash 容器部署准备，核查离线依赖与 PD 兼容性，准备多机验证。 | xLLM 部署与兼容性验证 | |
| 2026/10/15 | 完成三框架调度、KV Cache、通信源码与文献整理。[资料](implementation-comparison.md) | 框架对比准备 | |

## 归档约定

新增实验统一放在 `experiments/<框架>/<日期与主题>/`，同目录保存说明、配置、摘要和原始证据；首页只展示已验证结果。既有根目录文件保持旧链接可用，由资料目录集中导航。成功、失败和待验证状态分别标明，微基准收益不替代端到端性能。每个阶段及时提交 Git；不上传登录凭据、私钥、模型权重或镜像。
