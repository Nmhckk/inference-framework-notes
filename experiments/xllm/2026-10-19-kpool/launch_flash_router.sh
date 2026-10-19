#!/usr/bin/env bash
set -euo pipefail
test -f /.dockerenv
source /work/ascend_env.sh
ROOT=/work
ETCD_ROOT=$ROOT/toolchains/etcd-v3.5.21-linux-arm64
ROUTER=$ROOT/sources/xllm-service/build/xllm_service/xllm_master_serving
test -x "$ROUTER"
mkdir -p "$ROOT/toolchains" "$ROOT/logs/router" "$ROOT/etcd-data"
if [ ! -x "$ETCD_ROOT/etcd" ]; then
 tar -xzf "$ROOT/offline-deps/etcd-v3.5.21-linux-arm64.tar.gz" -C "$ROOT/toolchains"
fi
python3 - <<'PY'
import socket
for port in (12389,12390,28888,28889):
 with socket.socket() as s:
  s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  s.bind(('0.0.0.0',port))
PY
pids=()
trap 'for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; wait || true' EXIT INT TERM
"$ETCD_ROOT/etcd" --name=xllm-flash-20261016 --data-dir="$ROOT/etcd-data" \
 --listen-client-urls=http://172.16.12.11:12389 \
 --advertise-client-urls=http://172.16.12.11:12389 \
 --listen-peer-urls=http://127.0.0.1:12390 \
 --initial-advertise-peer-urls=http://127.0.0.1:12390 \
 --initial-cluster=xllm-flash-20261016=http://127.0.0.1:12390 \
 --initial-cluster-token=xllm-flash-20261016 \
 > "$ROOT/logs/router/etcd.log" 2>&1 &
pids+=("$!")
python3 - <<'PY'
import json,time,urllib.request
for attempt in range(30):
 try:
  with urllib.request.urlopen('http://172.16.12.11:12389/health',timeout=2) as r:
   if json.load(r).get('health') in (True,'true'):break
 except Exception:pass
 time.sleep(1)
else:raise SystemExit('Dedicated etcd failed health check')
PY
ENABLE_DECODE_RESPONSE_TO_SERVICE=true "$ROUTER" \
 --etcd_addr=172.16.12.11:12389 --etcd_namespace=flash-20261016 \
 --http_server_port=28888 --rpc_server_port=28889 \
 --reasoning_parser=glm5 \
 --tokenizer_path=/models/glm53-flash \
 > "$ROOT/logs/router/router.log" 2>&1 &
pids+=("$!")
printf '%s\n' "${pids[@]}" > "$ROOT/logs/router/pids"
wait -n
