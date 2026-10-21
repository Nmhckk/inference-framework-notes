#!/usr/bin/env python3
"""Print the tables and deltas used by this experiment report."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parent / "results"
FILES = {
    "xllm_base": ROOT / "baseline-xllm/summary.json",
    "vllm_base": ROOT / "baseline-vllm/summary.json",
    "overlap_off": ROOT / "overlap-off-80k-c16/summary.json",
    "high_default": ROOT / "high-off-80k-c32/summary.json",
    "high_zero": ROOT / "high-zero-80k-c32/summary.json",
    "xllm_prefix": ROOT / "xllm-prefix-on-80k-c16/summary.json",
    "xllm_prefix_64": ROOT / "xllm-prefix-64-80k-c16/summary.json",
    "vllm_prefix": ROOT / "vllm-prefix-on-80k-c16/summary.json",
}
METRICS = (
    "output_throughput_tps",
    "ttft_p50_s",
    "ttft_p90_s",
    "tpot_p50_ms",
    "tpot_p90_ms",
)


def pct(new: float, old: float) -> float:
    return (new / old - 1.0) * 100.0


data = {name: json.loads(path.read_text()) for name, path in FILES.items()}
for name, row in data.items():
    print(name, *(f"{metric}={row[metric]:.4f}" for metric in METRICS))

print("\ndeltas (%)")
for label, new, old in (
    ("overlap on/off", "xllm_base", "overlap_off"),
    ("zero/default", "high_zero", "high_default"),
    ("xllm prefix/off", "xllm_prefix", "xllm_base"),
    ("xllm 64/auto", "xllm_prefix_64", "xllm_prefix"),
    ("vllm prefix/off", "vllm_prefix", "vllm_base"),
    ("xllm/vllm prefix", "xllm_prefix", "vllm_prefix"),
):
    print(label, *(f"{metric}={pct(data[new][metric], data[old][metric]):+.2f}%" for metric in METRICS))
