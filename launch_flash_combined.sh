#!/usr/bin/env bash
set -euo pipefail
test -f /.dockerenv
case "${1:-}" in 8|9|10|11) ;; *) exit 1 ;; esac
if [[ "$1" == 11 ]]; then
 for device_id in {8..15}; do test -c "/dev/davinci${device_id}"; done
 test ! -e /dev/davinci7
fi
export INSTANCE_ROLE=DEFAULT HOST_IP="172.16.12.$1" XLLM_BACKEND=llm
export XLLM_SERVICE_ROUTING=true
exec bash /work/launch_flash_baseline.sh
