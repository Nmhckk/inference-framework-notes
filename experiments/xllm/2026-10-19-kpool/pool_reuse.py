# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Eager-only pool reuse guarded by exact current-history comparisons.

No request identity is inferred from physical block IDs. Reordered, recycled,
transferred or rewritten history invalidates reuse through exact equality.
Private snapshots own storage; incomplete pools are recomputed from scratch.
"""
from __future__ import annotations
from collections.abc import Callable
import torch

PoolData = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class VerifiedHistoryPoolReuse:
    """Keep an immutable copy of the last history for one indexer instance."""

    def __init__(self, reference: Callable[..., PoolData]) -> None:
        self._reference = reference
        self._history: torch.Tensor | None = None
        self._ape: torch.Tensor | None = None
        self._valid: torch.Tensor | None = None
        self._result: PoolData | None = None
        self._length = 0
        self._layout: tuple[int, int] | None = None
        self.hits = 0
        self.misses = 0

    def __call__(self, packed: torch.Tensor, valid: torch.Tensor,
                 ape: torch.Tensor, head_dim: int, rate: int) -> PoolData:
        batch, length, width = packed.shape
        reuse = 0
        compatible = (
            self._history is not None and self._result is not None and self._valid is not None
            and self._history.shape[0] == batch and self._history.shape[2] == width
            and self._history.device == packed.device and self._history.dtype == packed.dtype
            and self._ape is not None and torch.equal(self._ape, ape)
            and self._layout == (head_dim, rate)
        )
        if compatible and length and bool(valid[:, 0].all().item()):
            # A nonzero first valid key shifts every pool boundary. Require
            # both snapshots to start at position zero, retaining generic
            # reference semantics for left padding and all-invalid histories.
            if bool(self._valid[:, 0].all().item()):
                reuse = min(length, self._length) // rate * rate
                if (not torch.equal(packed[:, :reuse], self._history[:, :reuse])
                        or not torch.equal(valid[:, :reuse], self._valid[:, :reuse])):
                    reuse = 0
        if reuse:
            prefix_pools = reuse // rate
            prefix = tuple(t[:, :prefix_pools] for t in self._result)
            if reuse < length:
                # With a valid first key, pool alignment remains absolute even
                # if the tail starts with invalid positions. A generic suffix
                # call could shift alignment, so use the reference on the whole
                # input whenever the suffix starts invalid.
                if not bool(valid[:, reuse].all().item()):
                    reuse = 0
                else:
                    tail = self._reference(packed[:, reuse:], valid[:, reuse:], ape, head_dim, rate)
                    tail = (tail[0], torch.where(tail[1] >= 0, tail[1] + reuse, -1), tail[2])
                    result = tuple(torch.cat((a, b), dim=1) for a, b in zip(prefix, tail))
            else:
                result = prefix
        if not reuse:
            result = self._reference(packed, valid, ape, head_dim, rate)
            self.misses += 1
        else:
            self.hits += 1
        capacity = (length + 127) // 128 * 128
        if (self._history is None or self._history.shape[0] != batch
                or self._history.shape[1] < length or self._history.shape[2] != width
                or self._history.device != packed.device or self._history.dtype != packed.dtype):
            self._history = torch.empty((batch, capacity, width), dtype=packed.dtype, device=packed.device)
            self._valid = torch.empty((batch, capacity), dtype=valid.dtype, device=valid.device)
            reuse = 0
        self._valid[:, reuse:length].copy_(valid[:, reuse:length])
        self._history[:, reuse:length].copy_(packed[:, reuse:length])
        self._ape = ape.clone()
        self._length = length
        self._layout = (head_dim, rate)
        self._result = result
        return result
