# xLLM GLM-5.3-Flash 优化边界验证

本轮在单机 910C-64G 超节点上逐项验证 ACL Graph、MTP、KV/Indexer INT8 和任务流水线。判断顺序固定为：启动成功、短长上下文状态正确、80K/600 性能。没有通过正确性门禁的组合不记录性能收益。

## 结论

| 候选项 | 结果 | 依据 |
| --- | --- | --- |
| W8A8 权重 | 已使用 | 当前权重就是 W8A8，属于权重量化。 |
| 主 KV Cache INT8 | GLM-5.3-Flash/NPU 不支持 | 固定源码只允许 MLU，或 NPU 上 GLM-5.2 SFA packed C8；`glm5_next` 会硬失败。 |
| Indexer Cache INT8 | 当前模型组合不支持 | NPU 通用层允许，但 GLM-5.3-Flash 开启 KPool；容量计算硬检查要求 KPool 的压缩 index cache 为 BF16。 |
| ACL Graph＋MTP1 | 输出错误 | 1K 状态复用前两条均失败，首条生成到长度上限并出现重复乱码。 |
| ACL Graph＋MTP0 | 输出错误 | 已完成10条中仅4条命中；错误会串入其他上下文值或占位值。 |
| Unified MTP Graph | 代码不接收 `glm5_next` | 固定源码要求 target `model_type=glm_moe_dsa`，属于 GLM-5.2 路径。 |
| MTP task pipeline | 代码不接收 `glm5_next` draft | 固定源码要求 draft `model_type=glm_moe_dsa_mtp`。 |
| Graph double buffer＋MTP | 互斥 | xLLM 启动代码在 speculative token 大于0时自动关闭 double buffer。 |
| MTP draft body TP1、MTP2 adaptive | 待空机复测 | 23:17:41 起外部训练占满16张卡，随后 OOM 不能归因给 xLLM 参数。 |

当前可用的最佳配置仍是 **eager＋MTP1＋prefix cache＋chunked prefill＋schedule overlap**。同机 TP8×2 的既有有效结果为 TTFT 52.09秒、TPOT 236.26毫秒、输出46.90 token/s，48/48成功；本轮没有得到可替换它的新配置。

## 图模式补丁与反证

候选补丁只修正 `decode_acl_graph.py` 的 linear-state 索引选择：有线性状态缓存时，不再因为 attention backend 同时带 MLA 标志而丢弃 `linear_state_indices`。补丁见 [graph-linear-state-index.patch](patches/graph-linear-state-index.patch)。

补丁后图捕获和服务启动都成功，但输出门禁失败：

- Graph＋MTP1：已完成2条，正确0条；第一条达到1024输出上限并出现重复乱码。
- Graph＋MTP0：已完成10条，正确4条；1K顺序为2/4，已完成的1K并发为1/2，8K并发为1/4。

这说明该补丁必要但不充分；剩余问题更接近 KPool/DSA/GDN 混合状态在图回放时的地址或索引更新，而不是单纯 MTP 验证错误。完整逐请求结果见 [results-summary.json](results-summary.json)，原始日志和响应在 [`raw/`](raw/)。

## INT8 的准确含义

“xLLM 能不能 INT8”需要拆成三件事：

1. **模型权重 INT8/W8A8：可以，当前已启用。**
2. **主 KV Cache INT8：本版本的 NPU `glm5_next` 不可以。**
3. **DSA Indexer Cache INT8：NPU 基础能力存在，但当前 KPool 布局要求 BF16，不能与 GLM-5.3-Flash 直接组合。**

因此不能通过把 `--kv_cache_dtype=int8` 或 `--indexer_cache_dtype=int8` 强行加入命令获得有效加速；前者属于未实现路径，后者会在启动容量检查阶段被拒绝。

## 代码边界

验证基于 xLLM `ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb`：

- [`worker_impl.cpp`](https://github.com/xLLM-AI/xllm/blob/ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb/xllm/core/runtime/worker_impl.cpp#L524-L548)：NPU 主 KV INT8 仅保留 GLM-5.2 SFA packed C8 例外，并禁止与 PD 同开。
- [`kv_cache_estimation.cpp`](https://github.com/xLLM-AI/xllm/blob/ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb/xllm/core/kv_cache/layout/kv_cache_estimation.cpp#L95-L109)：KPool 与 Indexer Cache INT8 的硬冲突。
- [`unified_mtp_worker_impl.cpp`](https://github.com/xLLM-AI/xllm/blob/ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb/xllm/core/runtime/unified_mtp_worker_impl.cpp#L223-L270)：Unified MTP Graph 的模型和拓扑约束。
- [`mtp_worker_impl.cpp`](https://github.com/xLLM-AI/xllm/blob/ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb/xllm/core/runtime/mtp_worker_impl.cpp#L841-L852)：任务流水线要求 GLM DSA MTP draft。
- [`xllm.cpp`](https://github.com/xLLM-AI/xllm/blob/ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb/xllm/xllm.cpp#L325-L332)：MTP 自动关闭 graph double buffer。
- [官方 GLM-5.3-Flash 指南](https://docs.xllm-ai.com/en/cookbook/autoregressive_models/glm/glm_5_3_flash/)：当前公开配方是单机非 PD、Graph＋MTP1；本机同提交实测说明该配方在当前 W8A8/CANN/910C 组合仍需正确性修复。

## 后续顺序

1. 空机恢复后先复测 eager MTP1，作为环境基线。
2. 再测 MTP2/3 的接受率、TPOT和吞吐，并打开 adaptive speculative；每档先过1K/8K状态门禁。
3. draft body TP1 单独测试显存分布和权重共享，不能与外部占卡混测。
4. 图模式需要继续追踪 KPool pool cache、index cache、GDN linear state 三类 slot mapping；修复前保持关闭。
5. 如果 GLM-5.3-Flash 图路径仍无解，切换到 xLLM 已成熟优化的 Qwen 系列做框架能力基准，同时保留 GLM 作为模型兼容性基准。

本轮只停止了本项目容器；发现外部训练后立即停止使用服务器，没有修改 K8s或他人进程。
