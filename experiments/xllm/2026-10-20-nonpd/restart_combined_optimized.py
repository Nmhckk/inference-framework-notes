"""Cold-restart the four owned DEFAULT engines with a selected MTP depth."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shlex
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("--mtp", type=int, choices=[0, 1], required=True)
p.add_argument("--memory", type=float, choices=[0.70, 0.75, 0.80, 0.85], default=0.75)
a = p.parse_args()
root = Path(__file__).resolve().parent
remote_root = "/data/wio/xllm-glm53-flash"
containers = {
    8: "xllm-flash-runtime-20261019",
    9: "xllm-flash-runtime-20261019",
    10: "xllm-flash-runtime-20261019",
    11: "xllm-flash-decode-healthy-20261019",
}


def run(node: int, command: str, timeout: int = 180) -> str:
    result = subprocess.run(
        ["python3", str(root / "remote_run.py"), str(node), command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    print(node, result.returncode, result.stdout[-1200:], result.stderr[-300:], flush=True)
    result.check_returncode()
    return result.stdout


client_check = r'''from pathlib import Path
active=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit(): continue
 try: parts=[part.decode(errors='replace') for part in (p/'cmdline').read_bytes().split(b'\0') if part]
 except OSError: continue
 scripts=[part for part in parts[1:] if part.startswith('/work/')]
 if any(script.endswith(('aisbench_test.py','evaluate_state_reuse.py')) or '/run_3p1d_' in script or '/run_combined_' in script for script in scripts): active.append((p.name,parts[:4]))
assert not active, 'Owned test clients are active: '+repr(active)
'''
run(11, "docker exec xllm-flash-build-20261016 python3 -c " + shlex.quote(client_check))

archive = f"before-combined-mtp{a.mtp}-" + str(time.time_ns())


def stop(node: int) -> None:
    container = containers[node]
    run(
        node,
        f"mkdir -p {remote_root}/logs/{archive}; "
        f"cp -a {remote_root}/logs/default {remote_root}/logs/{archive}/ 2>/dev/null || true; "
        f"docker exec {container} python3 /work/stop_flash_instance.py default",
    )


with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(stop, containers))

environment = " ".join(
    [
        "XLLM_ENABLE_GRAPH=false",
        "XLLM_MAX_SEQS=8",
        "XLLM_LINEAR_SLOTS=64",
        "XLLM_SCHEDULE_OVERLAP=true",
        "XLLM_PREFIX_CACHE=true",
        f"XLLM_SPECULATIVE_TOKENS={a.mtp}",
        "XLLM_PD_VERIFIED_POOL_REUSE=0",
        f"XLLM_MEMORY_UTILIZATION={a.memory}",
        "XLLM_NPU_MEMORY_FRACTION=0.95",
        "XLLM_MAX_BATCH_TOKENS=10240",
        "XLLM_PREFILL_CHUNK=8192",
    ]
)


def launch(node: int) -> None:
    container = containers[node]
    run(
        node,
        f"docker exec -d {container} bash -c "
        + shlex.quote(
            f"export {environment}; bash /work/launch_flash_combined.sh {node} "
            f"> /work/logs/combined-optimized-mtp{a.mtp}-launch.log 2>&1"
        ),
    )


with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(launch, containers))
print(f"RESTARTED_FOUR_DEFAULT_ENGINES_MTP{a.mtp}", flush=True)
