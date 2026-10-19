#!/usr/bin/env bash
set -euo pipefail
# Run INSIDE the dedicated Flash container, after build/operator verification.
source /work/ascend_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.90
export OMP_NUM_THREADS=12
export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=300
export HCCL_OP_EXPANSION_MODE="${XLLM_HCCL_OP_EXPANSION_MODE:-AIV}"
export HCCL_IF_BASE_PORT=47440
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=/work/sources/xllm:${PYTHONPATH:-}
export TRITON_BINARY_PATH=/work/sources/xllm/build/lib.linux-aarch64-cpython-311/xllm/triton_npu/binary
test -f "$TRITON_BINARY_PATH/l2norm_fwd_kernel2_loop.npubin"
TORCH_NPU_ROOT="$(python3 -c 'import importlib.metadata as m;print(m.distribution("torch_npu").locate_file("torch_npu"))')"
export LD_LIBRARY_PATH="$TORCH_NPU_ROOT/lib:${LD_LIBRARY_PATH:-}"
XLLM_PATH=/work/sources/xllm/build/xllm/core/server/xllm
MODEL_PATH=/models/glm53-flash
ROLE=${INSTANCE_ROLE:-DEFAULT}
HOST_IP=${HOST_IP:-127.0.0.1}
pd_args=()
case "$ROLE" in
 DEFAULT)
  if [[ "${XLLM_SERVICE_ROUTING:-false}" == true ]]; then
   pd_args=(--enable_service_routing=true --instance_role=DEFAULT
    --etcd_addr=172.16.12.11:12389 --etcd_namespace=flash-20261016)
  fi
  ;;
 PREFILL|DECODE)
  case "$HOST_IP:$ROLE" in
   172.16.12.8:PREFILL|172.16.12.9:PREFILL|172.16.12.10:DECODE|172.16.12.11:DECODE) ;;
   *) echo 'Host/role does not match the four-host plan' >&2; exit 1 ;;
  esac
  pd_args=(--enable_disagg_pd=true --instance_role="$ROLE"
   --kv_cache_transfer_mode=PUSH --etcd_addr=172.16.12.11:12389
   --etcd_namespace=flash-20261016 --disagg_pd_port=43432)
  ;;
 *) echo 'Unknown INSTANCE_ROLE' >&2; exit 1 ;;
esac
LOG_DIR=/work/logs/${ROLE,,}
mkdir -p "$LOG_DIR"
test -x "$XLLM_PATH"
python3 - "$ROLE" <<'PY'
import socket,sys
ports=list(range(19600,19608))+[29680]
if sys.argv[1]!='DEFAULT':ports+=[43432]+list(range(26000,26008))
for port in ports:
 with socket.socket() as s:
  s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
  try:s.bind(('0.0.0.0',port))
  except OSError as error:raise RuntimeError(f'Port {port} is unavailable: {error}') from error
PY
# MTP and prefix caching are disabled for the first numerical/loading baseline.
pids=()
trap 'for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; wait || true' EXIT INT TERM
for ((i=0;i<8;i++)); do
 "$XLLM_PATH" --model="$MODEL_PATH" --model_id=glm53-flash \
  --host="$HOST_IP" --port=$((19600+i)) --master_node_addr="$HOST_IP:29680" \
  --nnodes=8 --node_rank="$i" --communication_backend=hccl \
  --max_memory_utilization=0.85 --enable_chunked_prefill=true \
  --enable_schedule_overlap=false --enable_prefix_cache=false \
  --max_tokens_per_chunk_for_prefill=8192 --enable_mix_batch=false \
  --enable_shm=false --enable_graph=false --model_impl=python --backend="${XLLM_BACKEND:-vlm}" \
  --reasoning_parser=glm5 \
  --max_seqs_per_batch="${XLLM_MAX_SEQS:-1}" --max_linear_state_cache_slots=16 --max_body_size=268435456 \
  --transfer_listen_port=$((26000+i)) "${pd_args[@]}" \
  > "$LOG_DIR/node_${i}.log" 2>&1 &
 pids+=("$!")
done
printf '%s\n' "${pids[@]}" > "$LOG_DIR/pids"
wait -n
