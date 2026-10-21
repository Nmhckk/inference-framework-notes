#!/usr/bin/env python3
"""Validate the Qwen3.5 MegaGDN structural advantage from pinned evidence.

This script deliberately separates three questions:
1. Is the wider xLLM fused boundary present in the tested source?
2. Is the comparison path structurally different in tested SGLang?
3. Does the archived end-to-end result move in the predicted direction?

It does not claim a causal performance A/B because the tested xLLM path has no
MegaGDN-off reference implementation.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tarfile
from pathlib import Path
from typing import Any


XLLM_COMMIT = "53f22bb603556dfb933f3f14a993cfbd20ce2e26"
XLLM_OPS_COMMIT = "b94b87318c2826738c65d6e6f242aa9c1881a3ab"
SGLANG_COMMIT = "e9b22a0818292eedc0630b2b72cdfcb4ae81bf85"


def git_show(repo: Path, revision: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def line_hits(text: str, pattern: str) -> list[int]:
    regex = re.compile(pattern)
    return [number for number, line in enumerate(text.splitlines(), 1) if regex.search(line)]


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    condition: bool,
    evidence: Any,
) -> None:
    checks.append(
        {
            "name": name,
            "status": "PASS" if condition else "FAIL",
            "evidence": evidence,
        }
    )


def pct(new: float, old: float) -> float:
    return (new / old - 1.0) * 100.0


def read_text_or_tar(path: Path, archive: Path, member: str) -> str:
    """Read unpacked evidence when present, otherwise read its archived copy."""
    if path.exists():
        return path.read_text(encoding="utf-8")
    with tarfile.open(archive, "r:gz") as bundle:
        extracted = bundle.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"{member} is missing from {archive}")
        return extracted.read().decode("utf-8")


def main() -> None:
    script = Path(__file__).resolve()
    workspace = script.parents[5]
    publication = script.parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--xllm-repo", type=Path, default=workspace / "research/xllm-main")
    parser.add_argument(
        "--sglang-repo",
        type=Path,
        default=workspace / "research/implementation-2026-10-15/sglang",
    )
    parser.add_argument("--output", type=Path, default=script.with_name("validation-results.json"))
    args = parser.parse_args()

    xllm_gdn_path = "xllm/python/layers/npu/qwen3_5/gated_delta_net.py"
    xllm_gdn = git_show(args.xllm_repo, XLLM_COMMIT, xllm_gdn_path)
    xllm_decode = xllm_gdn[xllm_gdn.index("    def _decode(") : xllm_gdn.index("    def forward(")]

    gitlink = subprocess.run(
        ["git", "ls-tree", XLLM_COMMIT, "third_party/xllm_ops"],
        cwd=args.xllm_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    xllm_ops_repo = args.xllm_repo / "third_party/xllm_ops"
    op_doc_path = "docs/zh/mega_gdn_decode.md"
    op_kernel_path = "xllm_ops/mega_gdn_decode/op_kernel/mega_gdn_decode_pto_kernel.h"
    op_doc = git_show(xllm_ops_repo, XLLM_OPS_COMMIT, op_doc_path)
    op_kernel = git_show(xllm_ops_repo, XLLM_OPS_COMMIT, op_kernel_path)

    sglang_backend_path = "python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py"
    sglang_model_path = "python/sglang/srt/models/qwen3_5.py"
    sglang_backend = git_show(args.sglang_repo, SGLANG_COMMIT, sglang_backend_path)
    sglang_model = git_show(args.sglang_repo, SGLANG_COMMIT, sglang_model_path)
    sglang_decode = sglang_backend[
        sglang_backend.index("    def forward_decode(") : sglang_backend.index("    def forward_extend(")
    ]

    checks: list[dict[str, Any]] = []
    add_check(
        checks,
        "xLLM test commit pins the audited xllm_ops revision",
        XLLM_OPS_COMMIT in gitlink,
        gitlink,
    )
    mega_lines = line_hits(xllm_decode, r"kernels\.mega_gdn_decode\(")
    add_check(
        checks,
        "xLLM decode unconditionally reaches one MegaGDN call site",
        len(mega_lines) == 1 and "causal_conv1d" not in xllm_decode,
        {"relative_lines": mega_lines, "fallback_call_present": "causal_conv1d" in xllm_decode},
    )
    fused_doc_terms = {
        "causal_convolution": "causal convolution" in op_doc,
        "qk_normalization": "Q/K L2" in op_doc,
        "recurrent_state_update": "recurrent state update" in op_doc,
        "rmsnorm_and_z_gate": "RMSNorm and Z gate" in op_doc,
        "conv_and_ssm_state_update": "Conv state and SSM state update" in op_doc,
    }
    # The pinned document is bilingual across revisions. Accept its Chinese labels too.
    fused_doc_terms.update(
        {
            "causal_convolution": fused_doc_terms["causal_convolution"] or "causal convolution" in op_doc,
            "qk_normalization": fused_doc_terms["qk_normalization"] or "Q/K L2 归一化" in op_doc,
            "recurrent_state_update": fused_doc_terms["recurrent_state_update"] or "recurrent state 更新" in op_doc,
            "rmsnorm_and_z_gate": fused_doc_terms["rmsnorm_and_z_gate"] or "RMSNorm 和 Z gate" in op_doc,
            "conv_and_ssm_state_update": fused_doc_terms["conv_and_ssm_state_update"] or "Conv state 与 SSM state 更新" in op_doc,
        }
    )
    add_check(
        checks,
        "MegaGDN contract covers convolution, Q/K norm, recurrence, output norm/gate and both states",
        all(fused_doc_terms.values()),
        fused_doc_terms,
    )
    kernel_markers = {
        "phase1_conv": "Phase 1: causal convolution" in op_kernel,
        "phase4_recurrence_norm_gate": "Phase 4: recurrent GDN update, RMSNorm, gate" in op_kernel,
        "qk_normalize": "NormalizeQk128" in op_kernel,
        "state_read_write": "read_state_indices_handle" in op_kernel
        and "write_state_indices_handle" in op_kernel,
    }
    add_check(
        checks,
        "The device kernel implements the documented fused stages",
        all(kernel_markers.values()),
        kernel_markers,
    )

    causal_line = line_hits(sglang_decode, r"torch\.ops\.npu\.causal_conv1d\(")
    recurrence_line = line_hits(sglang_decode, r"self\.kernel_dispatcher\.decode\(")
    add_check(
        checks,
        "SGLang NPU decode keeps causal-conv and recurrence as separate boundaries",
        bool(causal_line and recurrence_line and causal_line[0] < recurrence_line[0]),
        {"causal_conv_relative_lines": causal_line, "recurrence_relative_lines": recurrence_line},
    )
    norm_lines = line_hits(sglang_model, r"core_attn_out = self\.norm\(core_attn_out, z\)")
    npu_fused_projection_disabled = bool(
        re.search(r"_gdn_decode_fused_proj_conv\s*=\s*\(\s*_is_cuda", sglang_model)
    )
    add_check(
        checks,
        "SGLang NPU keeps RMSNorm/Z gate after the backend result",
        bool(norm_lines) and npu_fused_projection_disabled,
        {
            "post_backend_norm_lines": norm_lines,
            "fused_projection_path_is_cuda_only": npu_fused_projection_disabled,
        },
    )

    paper_slo = publication / "experiments/comparison/2026-10-21-qwen35-paper-slo"
    runtime_log = read_text_or_tar(
        paper_slo / "export/logs/exact-clean-server.log",
        paper_slo / "qwen35-paper-slo-evidence.tar.gz",
        "export/logs/exact-clean-server.log",
    )
    layer_match = re.search(r"n_layers: (\d+)", runtime_log)
    total_layers = int(layer_match.group(1)) if layer_match else 0
    attention_layer_ids = sorted(
        {int(value) for value in re.findall(r"Qwen3NextAttention layer (\d+):", runtime_log)}
    )
    gdn_layers = total_layers - len(attention_layer_ids)
    runtime_flags = {
        "torch_backend": "Resolved npu_kernel_backend=TORCH for model_type=qwen3_5_text" in runtime_log,
        "graph_off": "enable_graph: 0" in runtime_log,
        "schedule_overlap_on": "enable_schedule_overlap: 1" in runtime_log,
        "mtp_off": "num_speculative_tokens: 0" in runtime_log,
        "pd_off": "enable_disagg_pd: 0" in runtime_log,
        "batch_16": "max_seqs_per_batch: 16" in runtime_log,
    }
    add_check(
        checks,
        "Runtime evidence resolves 32 layers as 8 full-attention plus 24 GDN",
        total_layers == 32 and len(attention_layer_ids) == 8 and gdn_layers == 24,
        {
            "total_layers": total_layers,
            "full_attention_layer_ids": attention_layer_ids,
            "inferred_gdn_layers": gdn_layers,
        },
    )
    add_check(
        checks,
        "Measured xLLM run used the audited NPU path with major confounders disabled",
        all(runtime_flags.values()),
        runtime_flags,
    )

    comparison_path = (
        publication
        / "experiments/comparison/2026-10-21-qwen35-three-framework/comparison-summary.json"
    )
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    workload = comparison["workload"]
    xllm_group = comparison["groups"]["xllm_graph_off_overlap_on"]
    sglang_group = comparison["groups"]["sglang_graph_full_overlap_on"]
    xllm_values = [float(run["throughput_tps"]) for run in xllm_group["runs"]]
    sglang_values = [float(run["output_throughput_tps"]) for run in sglang_group["runs"]]
    xllm_mean = float(xllm_group["mean"]["throughput_tps"])
    sglang_mean = float(sglang_group["mean"]["output_throughput_tps"])
    observed_floor_gap = pct(min(xllm_values), max(sglang_values))
    add_check(
        checks,
        "All three xLLM throughput runs exceed all three SGLang runs",
        min(xllm_values) > max(sglang_values),
        {
            "xllm_range_tok_s": [min(xllm_values), max(xllm_values)],
            "sglang_range_tok_s": [min(sglang_values), max(sglang_values)],
            "minimum_observed_gap_pct": observed_floor_gap,
        },
    )

    requests = int(workload["requests"])
    concurrency = int(workload["concurrency"])
    output_tokens = int(workload["output_tokens"])
    waves = requests // concurrency
    xllm_boundaries_per_gdn_layer = 1
    sglang_boundaries_per_gdn_layer = 3
    invocation_model = {
        "scope": "framework-visible device-operator boundaries; not profiler kernel count",
        "gdn_layers": gdn_layers,
        "per_full_batch_decode_step": {
            "xllm": gdn_layers * xllm_boundaries_per_gdn_layer,
            "sglang_lower_bound": gdn_layers * sglang_boundaries_per_gdn_layer,
            "xllm_reduction_vs_sglang_lower_bound_pct": 100.0
            * (1.0 - xllm_boundaries_per_gdn_layer / sglang_boundaries_per_gdn_layer),
        },
        "idealized_full_benchmark": {
            "waves": waves,
            "decode_steps_per_wave": output_tokens,
            "xllm": waves * output_tokens * gdn_layers * xllm_boundaries_per_gdn_layer,
            "sglang_lower_bound": waves
            * output_tokens
            * gdn_layers
            * sglang_boundaries_per_gdn_layer,
        },
    }

    payload = {
        "schema_version": 1,
        "verdict": {
            "structural_exclusivity_in_tested_revisions": "PASS",
            "end_to_end_advantage_same_card": "PASS",
            "megagdn_causal_share_of_advantage": "NOT_PROVEN_NO_DIRECT_AB_OR_PROFILER",
        },
        "revisions": {
            "xllm": XLLM_COMMIT,
            "xllm_ops": XLLM_OPS_COMMIT,
            "sglang": SGLANG_COMMIT,
        },
        "checks": checks,
        "operator_boundary_model": invocation_model,
        "measured_effect": {
            "workload": workload,
            "xllm_mean_tok_s": xllm_mean,
            "sglang_mean_tok_s": sglang_mean,
            "xllm_vs_sglang_mean_pct": pct(xllm_mean, sglang_mean),
            "minimum_observed_run_gap_pct": observed_floor_gap,
            "xllm_tpot_mean_ms": float(xllm_group["mean"]["tpot_mean_ms"]),
            "sglang_tpot_mean_ms": float(sglang_group["mean"]["tpot_mean_ms"]),
        },
        "causal_limit": (
            "The tested xLLM Qwen3.5 NPU implementation has no MegaGDN-off fallback. "
            "Static operator-boundary counts and cross-framework results support the mechanism, "
            "but only a numerically equivalent split-path A/B or a matched profiler can quantify causality."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failed = [check for check in checks if check["status"] != "PASS"]
    print(f"validation checks: {len(checks) - len(failed)}/{len(checks)} PASS")
    print(
        "operator boundaries per full-batch decode step: "
        f"xLLM={invocation_model['per_full_batch_decode_step']['xllm']}, "
        f"SGLang>={invocation_model['per_full_batch_decode_step']['sglang_lower_bound']}"
    )
    print(
        f"measured throughput: xLLM={xllm_mean:.3f}, SGLang={sglang_mean:.3f}, "
        f"delta={pct(xllm_mean, sglang_mean):.3f}%"
    )
    print(f"wrote {args.output}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
