# 四台 910C 的 xLLM 非 PD 部署候选方案

状态：待服务器验证。沿用 2026-10-14 准备的固定版本，2026-10-15 整理归档。以下参数为首轮定位基线，不是已测最优配置。

## 拓扑与参数

一个跨四台机器的服务，每台 16 个逻辑 NPU，共 64 个 worker。Attention DP4、TP16、CP1；MoE EP64、TP1。四个 DP 组共享跨机 EP 通信，不是四套独立模型服务。

| 项目 | 首轮候选值 |
| --- | --- |
| 模型实现与实例角色 | `model_impl=python`、`instance_role=DEFAULT` |
| 非 PD | `enable_disagg_pd=false` |
| 主 KV 与 Indexer | `kv_cache_dtype=int8`、`indexer_cache_dtype=int8` |
| 序列/Token 预算 | `max_seqs_per_batch=8`、`max_tokens_per_batch=4096` |
| Cache block 与内存比例 | `block_size=128`、`max_memory_utilization=0.85` |
| 首轮关闭 | MTP、图模式、task pipeline、schedule overlap、prefix cache、混合批 |
| 首轮开启 | Chunked Prefill |

非 PD 表示同一实例完成 Prefill 和 Decode；是否将二者放入同一批次是另一个开关。此版本 `nnodes=64` 表示 worker 总数；LLM Attention TP 由并行拓扑推导，不能把其他框架的 TP 参数直接复制过来。[并行拓扑源码](https://github.com/xLLM-AI/xllm/blob/ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb/xllm/core/framework/parallel_state/parallel_topology.cpp)

## 镜像与依赖

| 候选开发镜像标签 | 已核查安装层中的组件 | 定位 |
| --- | --- | --- |
| `xllm-dev-a3-arm-cann9-20260911` | CANN 9.0.0 路径、Python 3.11.15、torch 2.9.0+cpu、torch_npu 2.9.0.post2+gitbe0dd17 | 默认候选 |
| `xllm-dev-cann9.1.0-a3-openeuler24.03-py3.12` | CANN 9.1.0 路径、Python 3.12.13、torch 2.10.0+cpu、torch_npu 2.10.0.post4+git5dd8ef3 | 备用候选 |

镜像来自 `quay.io/jd_xllm/xllm-ai`，完整 digest 和子模块提交见 [versions.json](versions.json)。上述信息来自此前镜像元数据与安装层静态核查；并非最终容器运行环境的 `pip freeze`，也未执行 NPU 算子。

依赖核查重点：`npu_dynamic_block_quant`、`npu_kv_quant_sparse_flash_attention`、`xllm_ops.quant_lightning_indexer` 及 metadata 接口。按固定提交构建 xLLM、torch_npu_ops、xllm_ops，确认 CANN 自定义算子环境在实际启动 shell 中生效。不要用同名但不同构建的 wheel 随意替换，不通过删除 PD 限制或修改 dtype 绕过失败。

[官方 GLM 环境说明](https://github.com/xLLM-AI/xllm/blob/ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb/docs/src/content/docs/zh/cookbook/autoregressive_models/glm/glm_5.md) · [固定版本算子接口](https://github.com/xLLM-AI/xllm/blob/ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb/xllm/python/kernels_npu/sparse_attention.py)

## 命令生成材料

`common.json` 与 `render_commands.py` 放在同一目录。下面使用文档保留地址，仅用于本地生成文本，不是实际服务器地址：

```bash
python3 render_commands.py --hosts 192.0.2.10 192.0.2.11 192.0.2.12 192.0.2.13
```

生成器仅使用 Python 标准库，不连接网络、不探测 NPU、不执行生成的命令。正式部署前，需替换分配后的四个地址，准备真实 HCCL ranktable、检查每机可见的 16 个逻辑 NPU，并核实服务、rendezvous 和 HCCL 端口。服务端口 9600–9615 与 master 29680 仅为候选值。

## 待验证清单

1. 环境：镜像摘要、驱动/CANN、torch_npu 与算子 ABI，四机设备和通信一致。
2. 权重：所有 shard 完整，配置、Tokenizer、量化描述、scale/offset 与 rotation 匹配；不能仅凭目录名判断兼容。
3. 算子：主 KV 和 Indexer 的 INT8 路径分别执行，核对布局、shape 和数值结果。
4. 启动：rank 0–63 完整，通信初始化成功，权重加载无缺键，短请求无 NaN 或异常输出。
5. 精度：固定评测集、采样配置、模板和输出预算，通过后再做负载与长上下文测试。
6. 性能：记录 TTFT、TPOT/ITL、吞吐、满足时延目标的有效吞吐、HBM 和稳定性；逐项打开优化并复测。

验证记录：此前已检查生成器参数及 rank 规则；本次在本地用文档保留地址再次验证能生成四条命令。实机部署、算子执行、精度和性能均未验证。
