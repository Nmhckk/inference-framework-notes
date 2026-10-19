#!/usr/bin/env bash
set -euo pipefail
test -f /.dockerenv
case "${1:-}" in
 8|9) export INSTANCE_ROLE=PREFILL ;;
 10|11) export INSTANCE_ROLE=DECODE ;;
 *) echo 'Usage inside the assigned container: launch_flash_pd.sh <8|9|10|11>' >&2; exit 1 ;;
esac
export HOST_IP="172.16.12.$1"
export XLLM_BACKEND=llm
# The underlying launcher checks host/role, devices, artifacts and free ports.
exec bash /work/launch_flash_baseline.sh
