# 固定源码边界摘录

源码提交：`ad9e405650c1bbb0af53d2c8e821dbdcbe5e79eb`。

## 主 KV INT8

`worker_impl.cpp` 先判断 `kv_cache_dtype == "int8"`。非 MLU 后端只有 `enable_mla_packed_c8(..., model_type)` 返回真才继续，注释明确该例外是 GLM-5.2 SFA C8；其余 NPU 模型直接 `LOG(FATAL)`。同一段还明确禁止 KV INT8 与 PD 同时启用。

## KPool 与 Indexer INT8

`kv_cache_estimation.cpp::index_slot_size` 在 `model_args.index_kpool_compress()` 为真时执行：

```cpp
CHECK(!enable_indexer_cache_quantization)
    << "KPool requires BF16 compressed index cache.";
```

因此 CLI 支持 NPU Indexer INT8 不代表 GLM-5.3-Flash KPool 路径支持。

## 图、MTP 与任务流水线

- `UnifiedMtpWorkerImpl::supports_unified_configuration()` 要求 target `model_type == "glm_moe_dsa"`，当前 Flash 是 `glm5_next`。
- `MTPWorkerImpl::create_task_pipeline()` 要求 draft `model_type == "glm_moe_dsa_mtp"`。
- `xllm.cpp` 在 speculative token 大于0时自动关闭 graph double buffer。

这些都是启动或能力检查，不应靠删除检查强行开启；要支持 `glm5_next`，必须补齐相应状态布局、图输入更新和正确性测试。
