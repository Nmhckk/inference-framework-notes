#!/usr/bin/env python3
"""Streaming client used for the 80K/600 run; configure paths through env vars."""

import concurrent.futures
import json
import math
import os
from pathlib import Path
import statistics
import threading
import time

import requests

ROOT = Path(os.environ.get("RESULT_DIR", "."))
DATASET = Path(os.environ["DATASET"])
PREFIX = Path(os.environ["PREFIX_DATASET"])
PORTS = [int(value) for value in os.environ.get("PORTS", "8000,8001").split(",")]
MODEL = os.environ.get("MODEL_NAME", "glm53-flash")
REQUEST_RATE = float(os.environ.get("REQUEST_RATE", "0.93"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "16"))
OUTPUT_LEN = int(os.environ.get("OUTPUT_LEN", "600"))
DETAILS = ROOT / "performance-details.jsonl"
SUMMARY = ROOT / "performance-summary-client.json"
write_lock = threading.Lock()


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    if low == high:
        return values[low]
    return values[low] * (high - pos) + values[high] * (pos - low)


def run_one(index, prompt, port, max_tokens, phase):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first = usage = error = status = None
    chunks = 0
    try:
        with requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=(30, 3600),
        ) as response:
            status = response.status_code
            response.raise_for_status()
            for raw in response.iter_lines():
                if not raw.startswith(b"data: ") or raw == b"data: [DONE]":
                    continue
                item = json.loads(raw[6:])
                if item.get("usage"):
                    usage = item["usage"]
                for choice in item.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                        chunks += 1
                        if first is None:
                            first = time.perf_counter()
    except Exception as exc:  # Preserve per-request failures in the raw result.
        error = f"{type(exc).__name__}: {exc}"
    ended = time.perf_counter()
    completion_tokens = (usage or {}).get("completion_tokens")
    ttft = None if first is None else first - started
    tpot = None
    if ttft is not None and completion_tokens and completion_tokens > 1:
        tpot = (ended - started - ttft) / (completion_tokens - 1)
    record = {
        "phase": phase,
        "index": index,
        "port": port,
        "http_status": status,
        "success": error is None and status == 200 and completion_tokens == max_tokens,
        "error": error,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "ttft_s": ttft,
        "tpot_ms": None if tpot is None else tpot * 1000,
        "latency_s": ended - started,
        "stream_chunks": chunks,
    }
    if phase == "measure":
        with write_lock:
            with DETAILS.open("a") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    prompts = [json.loads(line)["question"] for line in DATASET.open()]
    prefix_prompts = [json.loads(line)["question"] for line in PREFIX.open()]
    DETAILS.write_text("")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(PORTS)) as pool:
        warmup = [
            future.result()
            for future in [
                pool.submit(run_one, i, prefix_prompts[i], port, 1, "prefix_warmup")
                for i, port in enumerate(PORTS)
            ]
        ]

    wall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = []
        for i, prompt in enumerate(prompts):
            delay = wall_start + i / REQUEST_RATE - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            futures.append(pool.submit(run_one, i, prompt, PORTS[i % len(PORTS)], OUTPUT_LEN, "measure"))
        records = [future.result() for future in futures]
    wall_seconds = time.perf_counter() - wall_start
    successful = [record for record in records if record["success"]]
    ttft = [record["ttft_s"] for record in successful]
    tpot = [record["tpot_ms"] for record in successful]
    total_output = sum(record["completion_tokens"] for record in successful)
    summary = {
        "request_count": len(records),
        "success_count": len(successful),
        "concurrency": CONCURRENCY,
        "request_rate": REQUEST_RATE,
        "prefix_warmup": warmup,
        "wall_seconds": wall_seconds,
        "total_output_tokens": total_output,
        "output_throughput_tokens_per_second": total_output / wall_seconds,
        "ttft_mean_seconds": statistics.mean(ttft),
        "ttft_p50_seconds": percentile(ttft, 0.5),
        "ttft_p90_seconds": percentile(ttft, 0.9),
        "tpot_mean_milliseconds": statistics.mean(tpot),
        "tpot_p50_milliseconds": percentile(tpot, 0.5),
        "tpot_p90_milliseconds": percentile(tpot, 0.9),
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
