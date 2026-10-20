#!/usr/bin/env bash
set -euo pipefail
test -f /.dockerenv

replica="${1:?replica 0 or 1 is required}"
case "$replica" in
  0)
    devices=0,1,2,3,4,5,6,7
    http_base=19600
    transfer_base=26000
    master_port=29680
    hccl_base=47440
    ;;
  1)
    devices=8,9,10,11,12,13,14,15
    http_base=19700
    transfer_base=26100
    master_port=29780
    hccl_base=47540
    ;;
  *) echo "replica must be 0 or 1" >&2; exit 2 ;;
esac

source /work/ascend_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.90
export OMP_NUM_THREADS=12
export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=300
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_IF_BASE_PORT="$hccl_base"
export ASCEND_RT_VISIBLE_DEVICES="$devices"
export PYTHONPATH=/work/sources/xllm:${PYTHONPATH:-}
export TRITON_BINARY_PATH=/work/sources/xllm/build/lib.linux-aarch64-cpython-311/xllm/triton_npu/binary

torch_npu_root="$(python3 -c 'import importlib.metadata as m; print(m.distribution("torch_npu").locate_file("torch_npu"))')"
export LD_LIBRARY_PATH="/usr/local/python3.11.15/lib/python3.11/site-packages/xlite/lib:/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_xllm_math/op_api/lib:$torch_npu_root/lib:${LD_LIBRARY_PATH:-}"

xllm=/work/sources/xllm/build/xllm/core/server/xllm
model=/models/glm53-flash
host_ip=172.16.10.171
log_dir=/work/logs/replica${replica}
mkdir -p "$log_dir"
test -x "$xllm"
test -f "$TRITON_BINARY_PATH/l2norm_fwd_kernel2_loop.npubin"

pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait || true
}
trap cleanup EXIT INT TERM

for ((rank=0; rank<8; rank++)); do
  "$xllm" \
    --model="$model" --model_id=glm53-flash \
    --host="$host_ip" --port=$((http_base+rank)) \
    --master_node_addr="$host_ip:$master_port" \
    --nnodes=8 --node_rank="$rank" --communication_backend=hccl \
    --max_memory_utilization=0.80 \
    --enable_chunked_prefill=true --max_tokens_per_batch=10240 \
    --enable_schedule_overlap=true --enable_prefix_cache=true \
    --max_tokens_per_chunk_for_prefill=8192 --enable_mix_batch=false \
    --enable_shm=false --enable_graph=false \
    --model_impl=python --backend=llm --reasoning_parser=glm5 \
    --max_seqs_per_batch=8 --max_linear_state_cache_slots=64 \
    --max_body_size=268435456 \
    --transfer_listen_port=$((transfer_base+rank)) \
    --speculative_algorithm=MTP --draft_model="$model" --num_speculative_tokens=1 \
    >"$log_dir/node_${rank}.log" 2>&1 &
  pids+=("$!")
done
printf '%s\n' "${pids[@]}" >"$log_dir/pids"
wait -n
