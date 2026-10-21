# xLLM MegaGDN 优势验证报告

> 日期：2026-10-21<br>
> 对象：xLLM、SGLang、vLLM-Ascend<br>
> 模型：Qwen3.5-4B BF16<br>
> 硬件：单台 Ascend 910C-64G，物理设备 0<br>
> 结论适用范围：本文固定的代码提交、模型、硬件和 2K/2K Decode 负载

## 1. 可以直接引用的结论

在本文固定版本中，**xLLM 的 Qwen3.5 NPU Decode 有一项经过源码验证、SGLang 同版本没有的结构优势：`MegaGdnDecode` 把每个 GDN 层的 causal convolution、Q/K L2Norm、recurrent state 更新、RMSNorm/Z gate，以及 Conv/SSM 两类状态写回，合并进一次设备调用。**

Qwen3.5-4B 共 32 层，其中 24 层为 GDN、8 层为 Full Attention。并发 16 时，xLLM 每个满批 Decode step 的 GDN 主体只有 **24 个框架可见设备算子边界**；SGLang NPU 固定版本至少有 **72 个**。因此 xLLM 的边界数量至少减少 **48 个，即 66.7%**。

同一台 910C、同一权重、同一输入和客户端下，三轮端到端结果为：

| 框架 | 吞吐均值 | 吞吐标准差 | TPOT均值 | 成功请求 |
| --- | ---: | ---: | ---: | ---: |
| xLLM，Graph关闭、Overlap开启 | **1099.10 tok/s** | 0.91 | **14.118 ms** | 144/144 |
| SGLang，Decode Full Graph、Overlap开启 | 1057.02 tok/s | 14.49 | 14.371 ms | 144/144 |
| vLLM-Ascend rc2，Decode Graph、Async Scheduling | 951.61 tok/s | 1.54 | 16.304 ms | 144/144 |

xLLM 相对 SGLang 的吞吐高 **3.98%**，TPOT 低 **1.76%**；xLLM 三轮最低吞吐仍比 SGLang 三轮最高吞吐高 **3.08%**。代码结构和性能方向一致，支持 MegaGDN 是首要解释候选。

严谨边界也必须同时保留：xLLM 没有数值等价的 `MegaGDN off` 路径，本轮也没有两框架匹配的 NPU profiler timeline。因此本文证明的是：

| 待验证问题 | 结果 | 证据 |
| --- | --- | --- |
| xLLM 是否存在更宽的 GDN 融合边界 | **通过** | 固定提交源码、固定 xllm_ops gitlink、设备 kernel |
| 被测 SGLang NPU 是否仍为拆分路径 | **通过** | 固定提交源码 |
| 同卡端到端 xLLM 是否更快 | **通过** | 两边各三轮，288/288 请求成功 |
| 3.98% 是否全部来自 MegaGDN | **尚未证明** | 缺少同框架开关 A/B 或匹配 profiler |

所以对外最准确的表述是：

> 在 Qwen3.5-4B、Ascend 910C、2K输入/2K输出、并发16的被测版本中，xLLM 独有的宽边界 MegaGDN SuperOp 将每层 Decode 的 GDN 主体合并为一次设备调用，使框架可见的 GDN 设备算子边界较 SGLang 至少减少66.7%；同卡三轮吞吐高3.98%。结构优势和性能优势均已验证，MegaGDN 的独立性能贡献仍需 profiler 定量。

## 2. 验证范围与固定版本

为防止“当前主分支已经变化”造成结论漂移，所有源码判断都固定到本次实测对应版本。

| 项目 | 固定版本 | 本文用途 |
| --- | --- | --- |
| xLLM | [`53f22bb603556dfb933f3f14a993cfbd20ce2e26`](https://github.com/xLLM-AI/xllm/tree/53f22bb603556dfb933f3f14a993cfbd20ce2e26) | Qwen3.5 模型层与 NPU Decode 调用路径 |
| xllm_ops | [`b94b87318c2826738c65d6e6f242aa9c1881a3ab`](https://github.com/xLLM-AI/xllm-ops/tree/b94b87318c2826738c65d6e6f242aa9c1881a3ab) | `MegaGdnDecode` 算子合同与设备 kernel |
| SGLang | [`e9b22a0818292eedc0630b2b72cdfcb4ae81bf85`](https://github.com/sgl-project/sglang/tree/e9b22a0818292eedc0630b2b72cdfcb4ae81bf85) | Ascend GDN backend 与 Qwen3.5 模型层 |
| vLLM-Ascend | `v0.30.0rc2-a3` 镜像 | 端到端性能参照；归档缺少非空服务日志，不参与“独占算子”判定 |

xLLM 提交中的 `third_party/xllm_ops` gitlink 正好指向上述 `b94b873`。验证脚本会检查这个绑定，避免拿新算子源码解释旧运行结果。

## 3. 为什么选择 Qwen3.5 GDN Decode

Qwen3.5 是 Full Attention 与 Gated Delta Network 混合模型。实测日志显示：

```text
总层数：32
Full Attention：3, 7, 11, 15, 19, 23, 27, 31，共8层
GDN：32 - 8 = 24层
```

Decode 每一步只产生一个新 token。此时单个算子的计算规模较小，重复的设备调用、同步点、临时张量和状态读写更容易成为可见成本。24 个 GDN 层占全部层的 75%，所以 GDN 热路径的结构差异会在每个输出 token 上重复 24 次。

这个优势预计在以下条件下最明显：GDN 层占比高、输出较长、Decode 占总时间比例高、中小 batch（特别是 batch 不超过 MegaGDN 的 32 上限），以及单个小算子较短的场景。在长 Prefill、大 batch 被拆成多个 chunk、Full Attention 占比高或对手的图重放已经充分隐藏 Host 开销时，收益可能减小。

## 4. xLLM 代码路径：一个调用覆盖整个 GDN 主体

### 4.1 Python 模型层

xLLM 的固定源码位于 [`gated_delta_net.py#L155-L189`](https://github.com/xLLM-AI/xllm/blob/53f22bb603556dfb933f3f14a993cfbd20ce2e26/xllm/python/layers/npu/qwen3_5/gated_delta_net.py#L155-L189)。下面保留原参数并逐行标注作用：

```python
def _decode(self, hidden, conv_state, ssm_state,
            read_state_indices, write_state_indices):
    batch_size = hidden.shape[0]

    # 两个 GEMM 完成输入投影，不计入本文比较的“GDN 主体边界”。
    mixed_qkvz = self.in_proj_qkvz(hidden)
    mixed_ba = self.in_proj_ba(hidden)

    # split/view 主要改变张量视图；随后把全部输入交给一个 SuperOp。
    mixed_qkv, z = mixed_qkvz.split((self.conv_dim, self.value_dim), dim=-1)
    b, a = mixed_ba.split((self.num_v_heads, self.num_v_heads), dim=-1)
    z = z.view(batch_size, self.num_v_heads, self.value_head_dim)

    # MegaGDN 最大 batch 为32。本轮 batch=16，因此每层、每步只调用一次。
    for start in range(0, batch_size, 32):
        output = kernels.mega_gdn_decode(
            mixed_qkv[start:end].contiguous(),  # 未卷积的 Q/K/V
            z[start:end].contiguous(),          # 输出门控
            b[start:end].contiguous(),          # beta gate
            a[start:end].contiguous(),          # decay gate
            self.conv1d_weight,                 # 宽度4 depthwise conv
            conv_state,                         # 三个历史输入的状态缓存
            self.A_log,
            self.dt_bias,
            ssm_state,                          # recurrent FP32 state
            read_state_indices[start:end],      # 本步读取的 request slot
            write_state_indices[start:end],     # 本步更新的 request slot
            self.norm_weight,                   # 最终 RMSNorm 权重
            True,                               # FLA state layout
        )
```

这里没有 `enable_mega_gdn` 分支，也没有 `causal_conv1d → recurrence → norm` 的 NPU 回退链。满足构造约束后，Decode 必然进入一次 `kernels.mega_gdn_decode()`；不满足约束时构造阶段报错，而不是静默换成拆分路径。

### 4.2 算子合同与数学

xllm_ops 的 [`MegaGdnDecode 设计文档`](https://github.com/xLLM-AI/xllm-ops/blob/b94b87318c2826738c65d6e6f242aa9c1881a3ab/docs/zh/mega_gdn_decode.md#L22-L34)明确写明一次设备调用融合五部分：宽度4 causal convolution、Q/K L2Norm、gate/decay/recurrent update、RMSNorm/Z gate、Conv/SSM state 更新。

核心数学为：

```text
conv = SiLU(W0*s0 + W1*s1 + W2*s2 + W3*qkv)

q_hat = q / sqrt(sum(q*q) + 1e-6) / sqrt(128)
k_hat = k / sqrt(sum(k*k) + 1e-6)

g       = -exp(a_log) * softplus(a + dt_bias, threshold=20)
decay   = exp(g)
beta    = sigmoid(b)
H       = decay * H_read
pred    = H^T * k_hat
delta   = (v - pred) * beta
H_write = H + k_hat * delta^T
o       = H_write^T * q_hat

out = RMSNorm(BF16_RINT(o), norm_weight, eps=1e-6) * SiLU(z)
```

合同还固定了精度与状态语义：Conv 输出在 BF16 舍入，Q/K norm、decay、SSM state 与 RMSNorm 使用 FP32，最终输出再回 BF16；Conv state 和 SSM state 按独立 read/write slot 更新。这也是不能随意拼出一个“关闭融合”版本做 A/B 的原因：只要舍入点、state layout 或 slot 所有权变化，比较就不再只包含一个变量。

### 4.3 设备 kernel

固定 kernel [`mega_gdn_decode_pto_kernel.h`](https://github.com/xLLM-AI/xllm-ops/blob/b94b87318c2826738c65d6e6f242aa9c1881a3ab/xllm_ops/mega_gdn_decode/op_kernel/mega_gdn_decode_pto_kernel.h)提供直接证据：

```cpp
// Phase 1: causal convolution and convolution-state update.
...
NormalizeQk128<true>(...);   // Q normalization
NormalizeQk128<false>(...);  // K normalization
...
// Phase 4: recurrent GDN update, RMSNorm, gate, and final output.
```

`Run()` 同时接收 `read_state_indices_handle` 和 `write_state_indices_handle`，在设备侧完成 Conv/SSM 状态的读取与写回。阶段之间仍可能访问 GM，本文没有声称“完全没有 HBM 流量”；能够确定的是这些阶段没有回到 Python 再发射下一段框架算子，也不需要框架管理三段之间的独立生命周期。

## 5. SGLang 代码路径：NPU 至少三个边界

### 5.1 Ascend GDN backend

固定源码 [`ascend_gdn_backend.py#L114-L168`](https://github.com/sgl-project/sglang/blob/e9b22a0818292eedc0630b2b72cdfcb4ae81bf85/python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py#L114-L168)：

```python
# 边界1：单独调用 NPU causal convolution，同时更新 conv state。
mixed_qkv = torch.ops.npu.causal_conv1d(
    mixed_qkv,
    self._get_conv_weights_t(layer),
    conv_states=conv_states,
    cache_indices=cache_indices,
    run_mode=1,
)

# split/view 不计作独立设备计算边界。
query, key, value = torch.split(mixed_qkv, [...], dim=-1)
query = query.view(...)
key = key.view(...)
value = value.view(...)

# 边界2：GDN recurrence kernel dispatcher。
core_attn_out = self.kernel_dispatcher.decode(
    q=query, k=key, v=value,
    a=a, b=b,
    A_log=layer.A_log,
    dt_bias=layer.dt_bias,
    ssm_states=ssm_states,
    cache_indices=cache_indices,
)
return core_attn_out
```

### 5.2 返回模型层后的 Norm/Z gate

backend 返回后，[`qwen3_5.py#L1060-L1112`](https://github.com/sgl-project/sglang/blob/e9b22a0818292eedc0630b2b72cdfcb4ae81bf85/python/sglang/srt/models/qwen3_5.py#L1060-L1112)继续执行：

```python
attn_result = self.attn(...)
core_attn_out = attn_result

# 边界3：RMSNorm 与 Z gate；之后才进入共有的 out_proj GEMM。
core_attn_out = self.norm(core_attn_out, z)
output, _ = self.out_proj(core_attn_out)
```

SGLang 也存在 fused projection/conv 实现，但固定提交的门禁是：

```python
_gdn_decode_fused_proj_conv = (
    _is_cuda and envs.SGLANG_ENABLE_GDN_DECODE_FUSED_PROJ_CONV.get()
)
```

因此该路径只在 CUDA 生效，Ascend NPU 不会进入。本文结论不是“SGLang 完全没有融合算子”，而是：**在被测 NPU 固定版本中，它没有 xLLM 这种覆盖整个 GDN 主体的宽边界 SuperOp。**

## 6. 两条调用链的直接对照

排除两边共有的输入投影与输出投影后：

```text
xLLM / 每个 GDN 层 / 每个 Decode step

in_proj_qkvz + in_proj_ba
        │
        ▼
MegaGdnDecode
  ├─ causal-conv + Conv state
  ├─ Q/K L2Norm
  ├─ gate/decay/recurrent update + SSM state
  └─ RMSNorm + Z gate
        │
        ▼
out_proj

框架可见 GDN 主体边界：1
```

```text
SGLang NPU / 每个 GDN 层 / 每个 Decode step

input projections
        │
        ▼
npu.causal_conv1d                 边界1
        │
        ▼
kernel_dispatcher.decode          边界2
        │
        ▼
self.norm(core_attn_out, z)       边界3
        │
        ▼
out_proj

框架可见 GDN 主体边界下界：3
```

“边界”不等同于 profiler 中最终看到的 kernel 数。一个边界内部可能包含多个 device kernel；SGLang 的 Full Decode Graph 会减少 Host 重放成本，但图重放不会自动把三个数学阶段编译成一个 kernel。准确 kernel 数和各段 device time 必须由 profiler 给出。

## 7. 24 对 72 的推导

### 7.1 每个满批 Decode step

```text
L_total = 32
L_full_attention = 8
L_gdn = 32 - 8 = 24

xLLM = 24层 × 1边界/层 = 24
SGLang >= 24层 × 3边界/层 = 72
减少 = 72 - 24 = 48
减少比例 = 48 / 72 = 66.666...% ≈ 66.7%
```

本轮并发为16，低于 MegaGDN 最大 batch 32，因此 xLLM 不会因 chunking 把一次满批拆成两次调用。若 `B > 32`，xLLM 每层边界近似变为 `ceil(B/32)`，上述比例不能直接沿用。

### 7.2 48请求、2048输出的工作量模型

```text
满批等价 Decode step = 2048 × (48 / 16) = 6144

xLLM 边界工作量 = 6144 × 24 = 147,456
SGLang 边界工作量下界 = 6144 × 72 = 442,368
下界差值 = 442,368 - 147,456 = 294,912
```

服务实际使用 continuous batching，并不一定形成三个严格分隔的波次。因此这是便于比较的**满批等价工作量模型**，不是 profiler 计数。每步24对至少72的结构下界不依赖三波假设。

## 8. 性能实验与原始数据

### 8.1 控制变量

| 项目 | 设置 |
| --- | --- |
| 主机/NPU | 同一台 Ascend 910C-64G，物理设备0 |
| 模型 | 同一 Qwen3.5-4B BF16 权重 |
| 输入 | 48条固定输入，实际 token 2033–2034 |
| 输出 | 每请求2048 token |
| 并发/到达 | 16；`request_rate=0` |
| Prefix Cache/MTP/PD | 全部关闭 |
| Schedule Overlap | 两边开启 |
| xLLM Graph | 关闭；该配置是 xLLM 实测最佳稳定配置 |
| SGLang Graph | Full Decode Graph 开启 |

xLLM 关闭 Graph、SGLang 开启 Graph 代表两框架各自在该负载下的有效配置。结果仍由 xLLM 领先，说明领先不能归因于 xLLM 图模式。

### 8.2 三轮数据

每轮输出 token 总数固定为 `48 × 2048 = 98,304`。

| 框架 | 轮次 | Wall time | 输出吞吐 | TTFT P90 | TPOT均值 | 成功 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| xLLM | r1 | 89.441 s | 1099.089 tok/s | 1.305 s | 14.109 ms | 48/48 |
| xLLM | r2 | 89.514 s | 1098.194 tok/s | 1.292 s | 14.128 ms | 48/48 |
| xLLM | r3 | 89.366 s | 1100.018 tok/s | 1.343 s | 14.119 ms | 48/48 |
| SGLang | r1 | 94.497 s | 1040.285 tok/s | 3.837 s | 14.373 ms | 48/48 |
| SGLang | r2 | 92.267 s | 1065.431 tok/s | 1.659 s | 14.368 ms | 48/48 |
| SGLang | r3 | 92.275 s | 1065.333 tok/s | 1.648 s | 14.373 ms | 48/48 |

SGLang r1 的 TTFT 较高，但 TPOT 与 r2/r3 几乎相同。为避免挑选有利轮次，主结论使用全部三轮算术均值，同时报告最保守的区间比较。

### 8.3 数据如何计算

```text
throughput = total_output_tokens / wall_time

xLLM r1 = 98,304 / 89.44131064 = 1099.089 tok/s
SGLang r1 = 98,304 / 94.49722309 = 1040.285 tok/s

xLLM mean = (1099.089 + 1098.194 + 1100.018) / 3
          = 1099.100 tok/s
SGLang mean = (1040.285 + 1065.431 + 1065.333) / 3
            = 1057.016 tok/s

吞吐领先 = (1099.100 / 1057.016 - 1) × 100% = 3.981%

TPOT绝对差 = 14.371483 - 14.118390 = 0.253092 ms/token
TPOT相对差 = (14.118390 / 14.371483 - 1) × 100% = -1.761%

xLLM最低值 = 1098.194
SGLang最高值 = 1065.431
最保守区间差 = (1098.194 / 1065.431 - 1) × 100% = 3.075%
```

xLLM 三轮区间与 SGLang 三轮区间没有重叠。标准差使用样本标准差，xLLM 为0.912 tok/s，SGLang为14.490 tok/s。

## 9. 为什么更宽的融合边界可能更快

下面是源码能够支持的机制解释，具体占比仍需 profiler：

1. **减少框架级设备调用。** 每个 GDN 层由至少三个边界缩成一个；24层每步少至少48个边界，降低 Host 下发、队列管理和边界同步的机会。
2. **缩短中间张量生命周期。** SGLang 的 causal-conv 结果要交给 recurrence，结果再返回模型层做 norm。xLLM 让算子内部统一管理这些阶段，减少框架层临时张量和索引准备。
3. **统一状态槽语义。** MegaGDN 同时获得 Conv state、SSM state 以及 read/write slot，可以在一个设备合同中完成两类状态更新。
4. **保留设备侧优化空间。** kernel 源码显示 Conv、Q/K normalize、recurrence 与 norm/gate 处于同一设备程序，可使用内部 barrier、预取和 buffer 复用。本文不把这种机会夸大为“完全消除 HBM 访问”。
5. **收益按输出 token 重复。** 2K输出时重复2048步；模型有24个GDN层，微小的单层差异可以积累成端到端差异。

SGLang 的 Decode Graph 可以显著减少 Host 重放开销，所以 xLLM 的优势并非简单等于“少48次 Python 调用”。本轮 xLLM 在关闭 Graph 时仍领先，说明宽边界可能同时受益于设备执行链、状态处理和中间张量管理；各项占比需要 timeline 区分。

## 10. 已排除或不能用于解释本轮差距的因素

### Full-Attention 融合不是 xLLM 独占

xLLM 有 `split_qkv_rmsnorm_mrope`；SGLang NPU 固定版本也有 `split_qkvgate_gemma_rmsnorm_rope`。两边都融合 Full-Attention 的 QKV 拆分、Q/K Norm 与 RoPE。

### Graph 不是 xLLM 本轮领先来源

xLLM 最佳结果关闭 Graph；SGLang 开启 Full Decode Graph。xLLM 自身 Graph A/B 没有稳定收益，因此不能用 xLLM Graph 解释3.98%。

### MTP、Prefix Cache、PD 与量化均未开启

三框架本轮均未启用这些机制，它们没有参与本文差距。

### 混合状态缓存不是独占能力

xLLM 为 Full-Attention KV 和 GDN 固定状态使用不同缓存；SGLang `MambaPool` 与 vLLM hybrid Mamba cache 也采用相似思路。这是混合模型的正确基础设计，不能单独解释胜负。

### vLLM 只作为端到端参照

vLLM-Ascend rc2 三轮均值为951.61 tok/s，xLLM 高15.50%。但归档的 vLLM 服务日志为空，无法证明该镜像最终走了哪条 GDN kernel 或 fallback。本文不使用当前主线源码替代旧镜像证据，因此不声称 MegaGDN 单独造成对 vLLM 的15.50%差距。

## 11. 为什么没有伪造 MegaGDN 开关 A/B

被测 xLLM 没有 `enable_mega_gdn` 开关，也没有数值等价的 NPU split reference path。直接把 SGLang 小算子链移入 xLLM 会同时改变：Conv 与 recurrent 的舍入点、SSM state layout、缓存 slot 所有权、临时张量布局、backend 与 Graph 捕获范围。这样的实验一次改变多个变量，无法把差值归因于“是否融合”。

所以本轮使用三段可复核证据链：固定源码证明结构差异、同卡三轮证明性能方向、自动脚本检查两者没有错配；独立因果占比留给 profiler 或未来的数值等价 reference kernel。

## 12. 自动验证与复现

本目录提供 [`validate_megagdn.py`](validate_megagdn.py)，执行9项检查：固定 gitlink、xLLM 单一调用点、融合合同、设备 kernel、SGLang causal-conv/recurrence 拆分、SGLang后置Norm、32层模型结构、xLLM运行标志，以及三轮性能区间。

```bash
python3 validate_megagdn.py \
  --xllm-repo /path/to/xllm \
  --sglang-repo /path/to/sglang
```

脚本直接读取仓库内压缩运行证据，无需预先解压，并生成 [`validation-results.json`](validation-results.json)。本次输出：

```text
validation checks: 9/9 PASS
operator boundaries per full-batch decode step: xLLM=24, SGLang>=72
measured throughput: xLLM=1099.100, SGLang=1057.016, delta=3.981%
```

自动检查证明源码边界、运行配置和结果方向，不冒充 NPU profiler。

## 13. 下一次空机如何完成因果闭环

优先使用 xLLM 自带在线 profiler，不修改模型数学：

```bash
--enable_online_profile=true \
--profile_backend=torch \
--profile_dir=/tmp/xllm-megagdn-profile \
--enable_task_pipeline=false

curl -X POST http://127.0.0.1:9977/start_profile
# 只采集 batch16 下200–500个稳定Decode step
curl -X POST http://127.0.0.1:9977/stop_profile
```

SGLang 使用同一输入、batch16和相同稳定区间采集 timeline。必须报告：每步实际 GDN kernel 数、24层累计 causal-conv/recurrence/Norm device time、kernel 间 idle 与 Host gap、H2D/D2H 与状态读写时间、profiler 扰动，以及首轮和稳态的分开统计。

当前端到端 TPOT 差值约为 **0.253 ms/token**。只有当 xLLM 的 GDN device time/launch gap 优势能覆盖同量级差值，且其他阶段没有反向差异时，才可以把3.98%主要归因于 MegaGDN。

## 14. 证据入口

- [本专项机器可读结果](validation-results.json)
- [Qwen3.5 三框架同卡结果](../2026-10-21-qwen35-three-framework/README.md)
- [Qwen3.5 论文区间实测与原始证据包](../2026-10-21-qwen35-paper-slo/README.md)
- [三框架代码级性能归因](../2026-10-21-qwen35-code-attribution/README.md)
- [xLLM 技术报告](https://arxiv.org/abs/2510.14686)

本文最终结论分为三层：**结构独占性已验证，端到端领先已验证，独立因果贡献待 profiler。** 这三个层次不能互相替代。
