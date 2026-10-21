#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "2026-10-21-qwen35-paper-slo" / "export" / "comparison-summary.json"


def mean(values):
    return statistics.mean(values)


def stdev(values):
    return statistics.stdev(values)


def pct(new, old):
    return (new / old - 1) * 100


def load_sglang(prefix: str):
    runs = []
    for index in (1, 2, 3):
        path = HERE / "evidence" / "results" / f"{prefix}-r{index}" / "summary.json"
        runs.append(json.loads(path.read_text()))
    fields = [
        "output_throughput_tps",
        "ttft_mean_s",
        "ttft_p50_s",
        "ttft_p90_s",
        "tpot_mean_ms",
        "tpot_p50_ms",
        "tpot_p90_ms",
    ]
    return {
        "runs": runs,
        "mean": {field: mean([run[field] for run in runs]) for field in fields},
        "stdev": {field: stdev([run[field] for run in runs]) for field in fields},
    }


prior = json.loads(PRIOR.read_text())
xllm = prior["groups"]["xllm_overlap_graph_off"]
vllm = prior["groups"]["vllm_rc2_graph_async"]
sglang = load_sglang("sglang-bf16-c16")
sglang_no_overlap = load_sglang("sglang-bf16-overlap-off-c16")
sglang_steady_tps = mean(
    [run["output_throughput_tps"] for run in sglang["runs"][1:]]
)

result = {
    "schema_version": 1,
    "workload": prior["workload"],
    "groups": {
        "xllm_graph_off_overlap_on": xllm,
        "sglang_graph_full_overlap_on": sglang,
        "sglang_graph_full_overlap_off": sglang_no_overlap,
        "vllm_rc2_full_decode_graph_async": vllm,
    },
    "deltas": {
        "xllm_vs_sglang_all_three_runs": {
            "throughput_pct": pct(
                xllm["mean"]["throughput_tps"],
                sglang["mean"]["output_throughput_tps"],
            ),
            "tpot_mean_pct": pct(
                xllm["mean"]["tpot_mean_ms"],
                sglang["mean"]["tpot_mean_ms"],
            ),
        },
        "xllm_vs_sglang_steady_r2_r3": {
            "throughput_pct": pct(xllm["mean"]["throughput_tps"], sglang_steady_tps)
        },
        "sglang_overlap_on_vs_off": {
            "throughput_pct": pct(
                sglang["mean"]["output_throughput_tps"],
                sglang_no_overlap["mean"]["output_throughput_tps"],
            ),
            "tpot_mean_pct": pct(
                sglang["mean"]["tpot_mean_ms"],
                sglang_no_overlap["mean"]["tpot_mean_ms"],
            ),
        },
        "sglang_vs_vllm": {
            "throughput_pct": pct(
                sglang["mean"]["output_throughput_tps"],
                vllm["mean"]["throughput_tps"],
            ),
            "tpot_mean_pct": pct(
                sglang["mean"]["tpot_mean_ms"],
                vllm["mean"]["tpot_mean_ms"],
            ),
        },
        "xllm_vs_vllm": prior["deltas"]["xllm_vs_vllm"],
    },
}

(HERE / "comparison-summary.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n"
)

rows = [
    (
        "xLLM graph=off overlap=on",
        xllm["mean"]["throughput_tps"],
        xllm["stdev"]["throughput_tps"],
        xllm["mean"]["ttft_p90_s"],
        xllm["mean"]["tpot_mean_ms"],
    ),
    (
        "SGLang graph=full overlap=on",
        sglang["mean"]["output_throughput_tps"],
        sglang["stdev"]["output_throughput_tps"],
        sglang["mean"]["ttft_p90_s"],
        sglang["mean"]["tpot_mean_ms"],
    ),
    (
        "SGLang graph=full overlap=off",
        sglang_no_overlap["mean"]["output_throughput_tps"],
        sglang_no_overlap["stdev"]["output_throughput_tps"],
        sglang_no_overlap["mean"]["ttft_p90_s"],
        sglang_no_overlap["mean"]["tpot_mean_ms"],
    ),
    (
        "vLLM rc2 full-decode-graph async",
        vllm["mean"]["throughput_tps"],
        vllm["stdev"]["throughput_tps"],
        vllm["mean"]["ttft_p90_s"],
        vllm["mean"]["tpot_mean_ms"],
    ),
]
with (HERE / "summary.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        [
            "configuration",
            "repetitions",
            "throughput_mean_tok_s",
            "throughput_stdev",
            "ttft_p90_mean_s",
            "tpot_mean_ms",
            "successful_requests",
        ]
    )
    for name, throughput, throughput_sd, ttft_p90, tpot in rows:
        writer.writerow(
            [name, 3, f"{throughput:.3f}", f"{throughput_sd:.3f}", f"{ttft_p90:.3f}", f"{tpot:.3f}", 144]
        )
