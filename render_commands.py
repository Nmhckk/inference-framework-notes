#!/usr/bin/env python3
"""Print four xLLM launch commands. Never connect, spawn, or write files."""

import argparse
import ipaddress
import json
from pathlib import Path
import shlex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", nargs=4, required=True,
                        help="Assigned IPv4 addresses in machine-rank order")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_name("common.json"))
    parser.add_argument("--port", type=int, default=9600)
    parser.add_argument("--master-port", type=int, default=29680)
    parser.add_argument("--rank-table", default="/etc/xllm/ranktable-64.json")
    args = parser.parse_args()
    try:
        hosts = [str(ipaddress.IPv4Address(host)) for host in args.hosts]
        config = json.loads(args.config.read_text())
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if len(set(hosts)) != 4:
        parser.error("Four distinct hosts are required")
    if not (1024 <= args.port <= 65520 and 1024 <= args.master_port <= 65000):
        parser.error("Invalid service/master port range")
    if args.port <= args.master_port <= args.port + 15:
        parser.error("Master port overlaps local service ports")
    required = {"nnodes": 64, "dp_size": 4, "ep_size": 64, "cp_size": 1,
                "kv_split_size": 1, "layerwise_split_size": 1,
                "enable_disagg_pd": False, "model_impl": "python",
                "kv_cache_dtype": "int8", "indexer_cache_dtype": "int8"}
    for key, value in required.items():
        if config.get(key) != value:
            parser.error(f"This plan requires {key}={value!r}")
    forbidden = {"node_rank", "host", "port", "master_node_addr", "instance_role", "tp_size"}
    if forbidden.intersection(config):
        parser.error("Rank, endpoint, role, and LLM TP are managed by this plan")
    if not Path(args.rank_table).is_absolute():
        parser.error("Rank-table path must be absolute inside the container")

    print("# GENERATED TEXT ONLY. No machine has been contacted.")
    print("# Run only after assignment, release of all 64 NPUs, and preflight.")
    print("# Each container must expose exactly 16 logical NPUs in matching order.")
    print("# Reserve HCCL and rendezvous port ranges separately; see the plan.")
    for machine_rank, host in enumerate(hosts):
        print(f"\n# machine_rank={machine_rank}, host={host}, "
              f"global_ranks={machine_rank * 16}..{machine_rank * 16 + 15}")
        command = ["xllm", "serve", f"--machine-rank={machine_rank}",
                   f"--port={args.port}", f"--host={host}",
                   f"--master_node_addr={hosts[0]}:{args.master_port}",
                   "--instance_role=DEFAULT", f"--rank_tablefile={args.rank_table}",
                   f"--log-dir=/work/logs/nonpd-machine-{machine_rank}"]
        for key, value in config.items():
            encoded = str(value).lower() if isinstance(value, bool) else str(value)
            command.append(f"--{key}={encoded}")
        print(shlex.join(command))


if __name__ == "__main__":
    main()
