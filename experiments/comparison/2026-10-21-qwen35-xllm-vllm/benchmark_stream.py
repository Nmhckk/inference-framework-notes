#!/usr/bin/env python3
"""Framework-neutral OpenAI streaming benchmark with auditable per-request rows."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import statistics
import threading
import time

import requests


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def load_prompts(path: Path, limit: int) -> list[str]:
    prompts = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            prompt = row.get("question") or row.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError("dataset row must contain a string question or prompt")
            prompts.append(prompt)
            if len(prompts) == limit:
                break
    if len(prompts) != limit:
        raise ValueError(f"requested {limit} prompts, found {len(prompts)}")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--request-rate", type=float, default=0.93)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    prompts = load_prompts(args.dataset, args.requests)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "details.jsonl"
    detail_path.write_text("")
    write_lock = threading.Lock()

    def run_one(index: int, prompt: str, max_tokens: int, phase: str) -> dict:
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        started = time.perf_counter()
        first = None
        usage = None
        chunks = 0
        pieces = []
        status = None
        error = None
        try:
            with requests.post(
                args.endpoint.rstrip("/") + "/v1/chat/completions",
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
                        text = (
                            delta.get("content")
                            or delta.get("reasoning")
                            or delta.get("reasoning_content")
                            or ""
                        )
                        if text:
                            pieces.append(text)
                            chunks += 1
                            if first is None:
                                first = time.perf_counter()
        except Exception as exc:  # evidence is recorded instead of hidden
            error = f"{type(exc).__name__}: {exc}"
        ended = time.perf_counter()
        completion_tokens = (usage or {}).get("completion_tokens")
        prompt_tokens = (usage or {}).get("prompt_tokens")
        ttft = None if first is None else first - started
        tpot_ms = None
        if ttft is not None and completion_tokens and completion_tokens > 1:
            tpot_ms = (ended - started - ttft) * 1000 / (completion_tokens - 1)
        record = {
            "phase": phase,
            "index": index,
            "status": status,
            "success": error is None and status == 200 and completion_tokens == max_tokens,
            "error": error,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "ttft_s": ttft,
            "tpot_ms": tpot_ms,
            "latency_s": ended - started,
            "stream_chunks": chunks,
            "output_sha256": hashlib.sha256("".join(pieces).encode()).hexdigest(),
            "output_preview": "".join(pieces)[:160],
        }
        if phase == "measure":
            with write_lock:
                with detail_path.open("a") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(json.dumps(record, ensure_ascii=False), flush=True)
        return record

    for index in range(args.warmup):
        warm = run_one(index, prompts[index], min(32, args.max_tokens), "warmup")
        if warm["error"] or warm["status"] != 200:
            raise RuntimeError(f"warmup failed: {warm}")

    wall_start = time.perf_counter()
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for index, prompt in enumerate(prompts):
            if args.request_rate > 0:
                target = wall_start + index / args.request_rate
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            futures.append(pool.submit(run_one, index, prompt, args.max_tokens, "measure"))
        rows = [future.result() for future in futures]
    wall_s = time.perf_counter() - wall_start

    ok = [row for row in rows if row["success"]]
    ttft = [row["ttft_s"] for row in ok if row["ttft_s"] is not None]
    tpot = [row["tpot_ms"] for row in ok if row["tpot_ms"] is not None]
    output_tokens = sum(row["completion_tokens"] for row in ok)
    prompt_tokens = [row["prompt_tokens"] for row in ok if row["prompt_tokens"] is not None]
    summary = {
        "framework": args.framework,
        "model": args.model,
        "endpoint": args.endpoint,
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "requests": args.requests,
        "successes": len(ok),
        "concurrency": args.concurrency,
        "request_rate": args.request_rate,
        "max_tokens": args.max_tokens,
        "wall_s": wall_s,
        "output_tokens": output_tokens,
        "output_throughput_tps": output_tokens / wall_s,
        "ttft_mean_s": statistics.mean(ttft) if ttft else None,
        "ttft_p50_s": percentile(ttft, 0.50),
        "ttft_p90_s": percentile(ttft, 0.90),
        "tpot_mean_ms": statistics.mean(tpot) if tpot else None,
        "tpot_p50_ms": percentile(tpot, 0.50),
        "tpot_p90_ms": percentile(tpot, 0.90),
        "prompt_tokens_min": min(prompt_tokens, default=None),
        "prompt_tokens_max": max(prompt_tokens, default=None),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print("SUMMARY", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
