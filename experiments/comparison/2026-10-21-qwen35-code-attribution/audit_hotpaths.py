#!/usr/bin/env python3
"""Audit pinned source paths and packaged runtime evidence for Qwen3.5.

This is a code-path gate, not a performance benchmark.  It verifies that the
mechanisms discussed in README.md exist at the pinned source revisions and
that the packaged logs prove which options ran in the measured experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any


XLLM_COMMIT = "53f22bb603556dfb933f3f14a993cfbd20ce2e26"
SGLANG_COMMIT = "e9b22a0818292eedc0630b2b72cdfcb4ae81bf85"


def _git_show(repo: Path, revision: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _line_hits(text: str, pattern: str) -> list[int]:
    regex = re.compile(pattern)
    return [index for index, line in enumerate(text.splitlines(), start=1) if regex.search(line)]


def _check_source(
    results: list[dict[str, Any]],
    *,
    framework: str,
    scope: str,
    repo: Path,
    revision: str,
    path: str,
    pattern: str,
    mechanism: str,
) -> None:
    source = _git_show(repo, revision, path)
    lines = _line_hits(source, pattern)
    results.append(
        {
            "framework": framework,
            "scope": scope,
            "mechanism": mechanism,
            "status": "PASS" if lines else "FAIL",
            "revision": revision,
            "path": path,
            "lines": lines,
            "pattern": pattern,
        }
    )


def _read_metrics(summary_csv: Path) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            metrics[row["configuration"]] = {
                "throughput_tok_s": float(row["throughput_mean_tok_s"]),
                "tpot_ms": float(row["tpot_mean_ms"]),
                "ttft_p90_s": float(row["ttft_p90_mean_s"]),
            }
    return metrics


def _pct(new: float, old: float) -> float:
    return (new / old - 1.0) * 100.0


def main() -> None:
    script_path = Path(__file__).resolve()
    workspace = script_path.parents[5]
    publication = script_path.parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--xllm-repo", type=Path, default=workspace / "research/xllm-glm53")
    parser.add_argument(
        "--sglang-repo",
        type=Path,
        default=workspace / "research/implementation-2026-10-15/sglang",
    )
    parser.add_argument(
        "--vllm-ascend-repo",
        type=Path,
        default=workspace / "research/vllm-ascend-compat",
    )
    parser.add_argument(
        "--vllm-repo",
        type=Path,
        default=workspace / "research/implementation-2026-10-15/vllm",
    )
    parser.add_argument("--output", type=Path, default=script_path.with_name("audit-results.json"))
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    source_checks = [
        ("xLLM", "tested-commit", args.xllm_repo, XLLM_COMMIT,
         "xllm/python/layers/npu/qwen3_5/attention.py",
         r"kernels\.split_qkv_rmsnorm_mrope\(",
         "QKV split + Q/K RMSNorm + mRoPE fused call"),
        ("xLLM", "tested-commit", args.xllm_repo, XLLM_COMMIT,
         "xllm/python/layers/npu/qwen3_5/gated_delta_net.py",
         r"kernels\.mega_gdn_decode\(",
         "single-token MegaGDN fused decode call"),
        ("xLLM", "tested-commit", args.xllm_repo, XLLM_COMMIT,
         "xllm/core/scheduler/continuous_scheduler.cpp",
         r"step_with_schedule_overlap|result_callback_\(last_batch_\)",
         "one-step producer/consumer schedule overlap"),
        ("xLLM", "tested-commit", args.xllm_repo, XLLM_COMMIT,
         "xllm/core/runtime/worker_impl.cpp",
         r"replace_token|synchronize_stream=\*/false",
         "device-side previous-token handoff"),
        ("xLLM", "tested-commit", args.xllm_repo, XLLM_COMMIT,
         "xllm/core/kv_cache/block/linear_state_block_manager.cpp",
         r"LinearStateBlockManager::allocate_for_sequence",
         "separate fixed-slot recurrent-state allocator"),
        ("SGLang", "tested-commit", args.sglang_repo, SGLANG_COMMIT,
         "python/sglang/srt/models/qwen3_5.py",
         r"split_qkvgate_gemma_rmsnorm_rope\(",
         "QKV/gate + Q/K RMSNorm + RoPE fused call"),
        ("SGLang", "tested-commit", args.sglang_repo, SGLANG_COMMIT,
         "python/sglang/srt/managers/scheduler.py",
         r"def event_loop_overlap|result_queue\.append|pop_and_process",
         "event-loop overlap with previous-result queue"),
        ("SGLang", "tested-commit", args.sglang_repo, SGLANG_COMMIT,
         "python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py",
         r"torch\.ops\.npu\.causal_conv1d|kernel_dispatcher\.decode",
         "separate causal-conv and GDN-dispatch stages"),
    ]
    for check in source_checks:
        _check_source(
            results,
            framework=check[0],
            scope=check[1],
            repo=check[2],
            revision=check[3],
            path=check[4],
            pattern=check[5],
            mechanism=check[6],
        )

    vllm_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.vllm_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _check_source(
        results,
        framework="vLLM",
        scope="current-main-not-tested-image",
        repo=args.vllm_repo,
        revision=vllm_revision,
        path="vllm/v1/core/sched/async_scheduler.py",
        pattern=r"num_output_placeholders|spec_token_ids = self\._spec_token_placeholders",
        mechanism="async scheduler advances requests with output placeholders",
    )

    vllm_ascend_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.vllm_ascend_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _check_source(
        results,
        framework="vLLM-Ascend",
        scope="current-main-not-tested-image",
        repo=args.vllm_ascend_repo,
        revision=vllm_ascend_revision,
        path="vllm_ascend/patch/worker/patch_triton.py",
        pattern=r"falling back to the.*PyTorch|syncs per request",
        mechanism="current main documents synchronous causal-conv fallback risk",
    )

    paper_dir = publication / "experiments/comparison/2026-10-21-qwen35-paper-slo/export"
    xllm_log = (paper_dir / "logs/exact-clean-server.log").read_text(encoding="utf-8")
    fused_layers = len(re.findall(r"using fused split_qkv_rmsnorm_mrope kernel", xllm_log))
    cache_match = re.search(
        r"blocks: (\d+).*linear_blocks: (\d+), state_slots: (\d+).*gdn_state_bytes: ([0-9.]+ MB)",
        xllm_log,
    )
    runtime = {
        "xllm": {
            "schedule_overlap_enabled": "enable_schedule_overlap: 1" in xllm_log,
            "graph_disabled": "enable_graph: 0" in xllm_log,
            "mtp_disabled": "num_speculative_tokens: 0" in xllm_log,
            "fused_full_attention_layers": fused_layers,
            "cache": (
                {
                    "paged_kv_blocks": int(cache_match.group(1)),
                    "linear_state_blocks": int(cache_match.group(2)),
                    "state_slots": int(cache_match.group(3)),
                    "gdn_state_bytes": cache_match.group(4),
                }
                if cache_match
                else None
            ),
        },
        "vllm": {
            "launch_async_scheduling": "--async-scheduling" in (paper_dir / "launch_vllm.sh").read_text(),
            "launch_full_decode_graph": "FULL_DECODE_ONLY" in (paper_dir / "launch_vllm.sh").read_text(),
            "packaged_server_log_bytes": (paper_dir / "logs/vllm-server.log").stat().st_size,
            "attribution_log_status": "MISSING_EVIDENCE"
            if (paper_dir / "logs/vllm-server.log").stat().st_size == 0
            else "PRESENT",
        },
    }

    three_way = publication / "experiments/comparison/2026-10-21-qwen35-three-framework"
    sglang_log = (three_way / "evidence/logs/server-bf16.log").read_text(encoding="utf-8")
    runtime["sglang"] = {
        "overlap_enabled": "'disable_overlap_schedule': False" in sglang_log,
        "radix_cache_disabled": "'disable_radix_cache': True" in sglang_log,
        "mamba_ssm_bfloat16": "SGLANG_MAMBA_SSM_DTYPE = bfloat16" in sglang_log,
        "decode_graph_captured": "Capture target decode NPU graph end" in sglang_log,
        "decode_graph_replayed": "npu graph: True" in sglang_log,
    }
    metrics = _read_metrics(three_way / "summary.csv")
    xllm = metrics["xLLM graph=off overlap=on"]
    sglang_on = metrics["SGLang graph=full overlap=on"]
    sglang_off = metrics["SGLang graph=full overlap=off"]
    vllm = metrics["vLLM rc2 full-decode-graph async"]
    deltas = {
        "xllm_vs_sglang_throughput_pct": _pct(xllm["throughput_tok_s"], sglang_on["throughput_tok_s"]),
        "xllm_vs_vllm_throughput_pct": _pct(xllm["throughput_tok_s"], vllm["throughput_tok_s"]),
        "sglang_overlap_throughput_pct": _pct(
            sglang_on["throughput_tok_s"], sglang_off["throughput_tok_s"]
        ),
    }

    payload = {
        "schema_version": 1,
        "purpose": "source and runtime code-path gate; not a performance benchmark",
        "source_checks": results,
        "runtime_evidence": runtime,
        "measured_metrics": metrics,
        "derived_deltas": deltas,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failed = [item for item in results if item["status"] != "PASS"]
    print(f"source checks: {len(results) - len(failed)}/{len(results)} PASS")
    print(f"xLLM fused full-attention layers in runtime log: {fused_layers}")
    print(f"vLLM packaged attribution log: {runtime['vllm']['attribution_log_status']}")
    print(f"wrote {args.output}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
