"""Collect read-only native evidence after a four-engine non-PD run."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


run = sys.argv[1]
here = Path(__file__).resolve().parent
result_root = here / "results" / run
assert result_root.is_dir()
containers = {
    8: "xllm-flash-runtime-20261019",
    9: "xllm-flash-runtime-20261019",
    10: "xllm-flash-runtime-20261019",
    11: "xllm-flash-decode-healthy-20261019",
}


def remote(node, command):
    return subprocess.run(
        ["python3", str(here / "remote_run.py"), str(node), command],
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    ).stdout


counter_names = [
    "speculative_num_accepted_tokens_total",
    "speculative_num_committed_tokens_total",
    "speculative_num_draft_tokens_total",
    "speculative_num_drafts_total",
    "prefix_cache_hit_tokens_total",
    "prefix_cache_prompt_tokens_total",
    "prefix_cache_token_hit_rate_perc",
]
evidence = {
    "scope": "Native counters and completed request logs since the cold engine start; includes AISBench warmups",
    "nodes": {},
}
for node, container in containers.items():
    lines = remote(
        node,
        f"docker exec {container} grep 'max_tokens: 600' /work/logs/default/node_0.log || true",
    ).splitlines()
    parsed = []
    for line in lines:
        row = {}
        for key in [
            "prompt_tokens",
            "num_prefix_cache_tokens",
            "generated_tokens",
            "speculative_accepted_tokens",
            "speculative_proposed_tokens",
        ]:
            match = re.search(rf"{key}: ([0-9]+)", line)
            if match:
                row[key] = int(match.group(1))
        for key in ["ttft", "avg tpot", "speculative_token_acceptance_rate"]:
            match = re.search(rf"{key}: ([0-9.]+)", line)
            if match:
                row[key.replace(" ", "_")] = float(match.group(1))
        parsed.append(row)

    counters = {}
    for pattern in ["speculative*", "prefix*"]:
        text = remote(11, f"curl -s http://172.16.12.{node}:19600/vars/{pattern}")
        for line in text.splitlines():
            if " : " not in line:
                continue
            key, value = line.split(" : ", 1)
            if key in counter_names:
                counters[key] = float(value)
    evidence["nodes"][str(node)] = {
        "completed_600_token_requests_including_warmup": len(parsed),
        "requests": parsed,
        "native_counters": counters,
    }

all_requests = [
    row for node in evidence["nodes"].values() for row in node["requests"]
]
evidence["aggregate"] = {
    "logged_600_token_requests_including_one_main_warmup": len(all_requests),
    "route_counts_including_warmup": {
        node: row["completed_600_token_requests_including_warmup"]
        for node, row in evidence["nodes"].items()
    },
    "logged_generated_tokens": sum(row.get("generated_tokens", 0) for row in all_requests),
    "logged_prefix_cache_tokens": sum(
        row.get("num_prefix_cache_tokens", 0) for row in all_requests
    ),
    "logged_prompt_tokens": sum(row.get("prompt_tokens", 0) for row in all_requests),
    "logged_speculative_accepted_tokens": sum(
        row.get("speculative_accepted_tokens", 0) for row in all_requests
    ),
    "logged_speculative_proposed_tokens": sum(
        row.get("speculative_proposed_tokens", 0) for row in all_requests
    ),
}
aggregate = evidence["aggregate"]
aggregate["logged_prefix_cache_hit_rate"] = (
    aggregate["logged_prefix_cache_tokens"] / aggregate["logged_prompt_tokens"]
)
aggregate["logged_speculative_acceptance_rate"] = (
    aggregate["logged_speculative_accepted_tokens"]
    / aggregate["logged_speculative_proposed_tokens"]
)
(result_root / "native-runtime-evidence.json").write_text(
    json.dumps(evidence, indent=2) + "\n"
)

for source, target in [
    ("k8s-preservation-final-20261019.json", "k8s-preservation.json"),
    ("active-npu-health-recovery-20261019.json", "active-npu-health.json"),
    ("runtime-3p1d-no-graph-20261019.json", "runtime-processes.json"),
]:
    shutil.copy2(here / source, result_root / target)
print(json.dumps(evidence["aggregate"], indent=2))
