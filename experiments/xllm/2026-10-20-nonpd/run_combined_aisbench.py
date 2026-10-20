"""Run the exact 80K/600 AISBench workload against four DEFAULT engines."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--profile", required=True)
p.add_argument("--require-gate", action="store_true")
p.add_argument("--memory", type=float, default=0.80)
a = p.parse_args()
logs = Path("/work/logs")
run = "perf-combined-aisbench-" + a.profile + "-20261020"
root = logs / run
assert not root.exists(), "Refuse to overwrite prior benchmark"
root.mkdir()
(root / "dataset").mkdir()
state_path = logs / ("diag-" + run + "-status.json")
state = {"phase": "waiting_for_readiness", "run": run}


def save() -> None:
    state_path.write_text(json.dumps(state, indent=2))


save()
try:
    if a.require_gate:
        gate = logs / ("diag-combined-" + a.profile + "-20261020-status.json")
        assert gate.exists() and json.loads(gate.read_text()).get("phase") == "passed", "required gate did not pass"
    for _ in range(120):
        try:
            for node in range(8, 12):
                assert json.load(urllib.request.urlopen(f"http://172.16.12.{node}:19600/v1/models", timeout=2))["data"]
            assert json.load(urllib.request.urlopen("http://172.16.12.11:28888/v1/models", timeout=2))["data"]
            break
        except (AssertionError, OSError, ValueError):
            time.sleep(3)
    else:
        raise RuntimeError("engines/router not ready")
    args = [
        sys.executable,
        "aisbench_test.py",
        "--input_len", "81920",
        "--output_len", "600",
        "--data_num", "48",
        "--concurrency", "16",
        "--request_rate", "0.93",
        "--dataset_type", "prefix_cache",
        "--repeat_rate", "0.93",
        "--prefix_test",
        "--dp", "2",
    ]
    work = Path("/work/bench-combined-aisbench-" + a.profile + "-20261020")
    shutil.copytree("/work/bench-3p1d-aisbench-20261019", work)
    config = work / "config.py"
    config.write_text(config.read_text().replace("/work/logs/perf-3p1d-aisbench-20261019", str(root)))
    manifest = {
        "command": args,
        "profile": a.profile,
        "target_input_tokens": 81920,
        "target_output_tokens": 600,
        "requests": 48,
        "concurrency_cap": 16,
        "request_rate": 0.93,
        "configured_prefix_ratio": 0.93,
        "warmup_prefix_copies": 2,
        "seed": 1,
        "active_logical_devices": 32,
        "hosts": 4,
        "topology": "4xDEFAULT (non-PD)",
        "settings": {
            "graph": False,
            "MTP_tokens": 1,
            "prefix_cache": True,
            "schedule_overlap": True,
            "max_seqs_per_batch": 8,
            "memory_utilization": a.memory,
        },
        "quality_scope": "pre/post deterministic retrieval gates; no full model accuracy claim",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    env = dict(
        os.environ,
        PYTHONPATH="/work/benchmark-tools/aisbench-3.1.0-owned",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
    )
    state["phase"] = "running"
    save()
    with (root / "wrapper.log").open("w") as stream:
        result = subprocess.run(args, cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT)
    state.update(phase="client_finished", exit_code=result.returncode)
    save()
    raise SystemExit(result.returncode)
except Exception as error:
    state.update(phase="failed", error=repr(error))
    save()
    raise
