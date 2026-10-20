"""Validate and summarize an exact 48-request non-PD AISBench run."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--memory", type=float, required=True)
p.add_argument("--mtp", type=int, default=1)
a = p.parse_args()
root = Path(__file__).resolve().parent / "results" / a.run

candidates = []
for path in root.glob("outputs/*/performances/vllm-api-stream-chat/gsm8k.json"):
    summary = json.loads(path.read_text())
    if summary.get("Total Requests", {}).get("total") == 48:
        candidates.append((path, summary))
assert len(candidates) == 1, "No unique completed 48-request report"
path, summary = candidates[0]
assert summary["Failed Requests"]["total"] == 0, summary
assert summary["Success Requests"]["total"] == 48, summary
assert summary["Total Generated Tokens"]["total"] == 48 * 600, summary

latency = {
    row["Performance Parameters"]: row
    for row in csv.DictReader(path.with_suffix(".csv").open())
    if row.get("Performance Parameters")
}
rows = [
    json.loads(line)
    for line in path.with_name("gsm8k_details.jsonl").read_text().splitlines()
    if line.strip()
]
assert len(rows) == 48
assert all(row["success"] and row["output_tokens"] == 600 for row in rows)


def numeric(value):
    return float(re.match(r"[-+0-9.eE]+", str(value)).group())


duration = numeric(summary["Benchmark Duration"]["total"]) / 1000
output_tps = numeric(summary["Output Token Throughput"]["total"])
assert abs(output_tps - 28800 / duration) < 0.001, (output_tps, duration)

result = {
    "run": a.run,
    "scope": "AISBench performance pressure data only; no full model accuracy claim",
    "requests": 48,
    "successes": 48,
    "failed": 0,
    "output_tokens": 28800,
    "actual_input_lengths": dict(
        sorted(
            (str(length), sum(row["input_tokens"] == length for row in rows))
            for length in {row["input_tokens"] for row in rows}
        )
    ),
    "actual_output_lengths": {"600": 48},
    "ttft_mean_seconds": numeric(latency["TTFT"]["Average"]) / 1000,
    "ttft_median_seconds": numeric(latency["TTFT"]["Median"]) / 1000,
    "ttft_p90_seconds": numeric(latency["TTFT"]["P90"]) / 1000,
    "tpot_mean_ms": numeric(latency["TPOT"]["Average"]),
    "tpot_median_ms": numeric(latency["TPOT"]["Median"]),
    "tpot_p90_ms": numeric(latency["TPOT"]["P90"]),
    "cluster_output_tps": output_tps,
    "benchmark_duration_seconds": duration,
    "max_client_concurrency": summary["Max Concurrency"]["total"],
    "observed_average_concurrency": summary["Concurrency"]["total"],
    "resources": {
        "hosts": 4,
        "active_logical_devices": 32,
        "topology": "4xDEFAULT (non-PD)",
        "TP_per_instance": 8,
        "max_seqs_per_instance": 8,
        "memory_utilization": a.memory,
        "graph": False,
        "MTP_tokens": a.mtp,
        "prefix_cache": True,
        "schedule_overlap": True,
        "chunked_prefill": True,
        "prefill_chunk_tokens": 8192,
    },
    "source_report": str(path.relative_to(root)),
    "dataset_sha256": {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in (root / "dataset").glob("*.jsonl")
    },
    "AISBench_aggregates": summary,
    "AISBench_latency_rows": latency,
}
(root / "validated-summary.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2)
)
print(
    json.dumps(
        {key: value for key, value in result.items() if not key.startswith("AISBench_")},
        ensure_ascii=False,
        indent=2,
    )
)
