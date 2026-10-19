#!/usr/bin/env bash
set -euo pipefail
test -f /.dockerenv
case "${1:-}" in
 8|9) export INSTANCE_ROLE=PREFILL ;;
 10|11) export INSTANCE_ROLE=DECODE ;;
 *) echo 'Usage inside the assigned container: launch_flash_pd.sh <8|9|10|11>' >&2; exit 1 ;;
esac
# Node11 physical device7 has uncorrectable HBM ECC. Its Decode must use
# the isolated recovery container exposing only physical devices8 through15.
if [[ "$1" == 11 ]]; then
 for device_id in {8..15}; do
  test -c "/dev/davinci${device_id}" || { echo 'Missing healthy recovery device; refusing Decode launch' >&2; exit 1; }
 done
 test ! -e /dev/davinci7 || { echo 'Faulty device7 is exposed; refusing Decode launch' >&2; exit 1; }
fi
export HOST_IP="172.16.12.$1"
export XLLM_BACKEND=llm
# The underlying launcher checks host/role, devices, artifacts and free ports.
exec bash /work/launch_flash_baseline.sh
