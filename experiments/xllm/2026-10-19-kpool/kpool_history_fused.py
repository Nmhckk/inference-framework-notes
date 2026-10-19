# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Stateless KPool history compression without full-size fp32 temporaries.

No persistent cache is introduced: transferred/recycled/MTP history is read
on every invocation. Invalid and incomplete pools keep the reference domain.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _pool_history_kernel(
    packed, valid, first, ape, output, indices, pool_valid,
    length: tl.constexpr, pools: tl.constexpr,
    stride_b: tl.constexpr, stride_t: tl.constexpr, stride_d: tl.constexpr,
    valid_b: tl.constexpr, valid_t: tl.constexpr,
    ape_r: tl.constexpr, ape_d: tl.constexpr,
    DIM: tl.constexpr, RATE: tl.constexpr, TILE: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    pool = tl.program_id(1) * TILE + tl.arange(0, TILE)
    member = tl.arange(0, RATE)
    dim = tl.arange(0, DIM)
    begin = tl.load(first + batch)
    positions = begin + pool[:, None] * RATE + member[None, :]
    safe = tl.minimum(positions, length - 1)
    mask = (pool[:, None] < pools) & (positions < length)
    live = tl.load(valid + batch * valid_b + safe * valid_t, mask=mask, other=0).to(tl.int1) & mask
    offsets = batch * stride_b + safe[:, :, None] * stride_t + dim[None, None, :] * stride_d
    keys = tl.load(packed + offsets, mask=mask[:, :, None], other=0).to(tl.float32)
    gates = tl.load(packed + offsets + DIM * stride_d, mask=mask[:, :, None], other=0).to(tl.float32)
    bias = tl.load(ape + member[:, None] * ape_r + dim[None, :] * ape_d).to(tl.float32)
    logits = tl.where(live[:, :, None], gates + bias[None, :, :], float('-inf'))
    maximum = tl.max(logits, 1)
    exponent = tl.where(live[:, :, None], tl.exp(logits - maximum[:, None, :]), 0.0)
    denominator = tl.sum(exponent, 1)
    weights = (exponent / tl.maximum(denominator[:, None, :], 1.0e-30)).to(tl.bfloat16).to(tl.float32)
    product = (weights * keys).to(tl.bfloat16).to(tl.float32)
    result = tl.sum(product, 1)
    tl.store(output + (batch * pools + pool[:, None]) * DIM + dim[None, :], result, mask=pool[:, None] < pools)
    tl.store(indices + (batch * pools + pool[:, None]) * RATE + member[None, :], tl.where(live, positions, -1), mask=pool[:, None] < pools)
    complete = tl.sum(live.to(tl.int32), 1) == RATE
    tl.store(pool_valid + batch * pools + pool, complete, mask=pool < pools)


def pooled_states_fused(
    packed_states: torch.Tensor,
    key_valid: torch.Tensor,
    ape: torch.Tensor,
    head_dim: int,
    rate: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compress current dense history, retaining invalid candidates and tails."""
    batch, length = packed_states.shape[:2]
    if packed_states.dtype != torch.bfloat16 or head_dim != 128 or rate != 4 or length == 0:
        raise ValueError("Fused history pooling requires nonempty BF16, dim128/rate4")
    first = torch.where(
        key_valid.any(-1), key_valid.to(torch.int32).argmax(-1),
        torch.full((batch,), length, dtype=torch.long, device=packed_states.device),
    )
    pools = triton.cdiv(length, rate)
    output = torch.empty((batch, pools, head_dim), dtype=packed_states.dtype, device=packed_states.device)
    indices = torch.empty((batch, pools, rate), dtype=torch.long, device=packed_states.device)
    valid = torch.empty((batch, pools), dtype=torch.bool, device=packed_states.device)
    _pool_history_kernel[(batch, triton.cdiv(pools, 4))](
        packed_states, key_valid, first, ape, output, indices, valid,
        length, pools, *packed_states.stride(), *key_valid.stride(), *ape.stride(),
        DIM=head_dim, RATE=rate, TILE=4,
    )
    return output, indices, valid
