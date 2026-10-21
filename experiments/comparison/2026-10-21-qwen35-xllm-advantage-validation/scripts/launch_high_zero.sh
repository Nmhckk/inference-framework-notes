#!/usr/bin/env bash
set -eo pipefail

export ASCEND_RT_VISIBLE_DEVICES=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.90
export OMP_NUM_THREADS=24
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_IF_BASE_PORT=54433

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1

# The image keeps custom operator registration in interactive shell startup.
# Export it explicitly because the benchmark launches through non-interactive
# `docker exec`.
for vendor in custom_xllm_math custom_transformer glm_next_transformer kpool_transformer; do
  vendor_root="/usr/local/Ascend/cann-9.0.0/opp/vendors/${vendor}"
  if [[ -d "${vendor_root}" ]]; then
    export ASCEND_CUSTOM_OPP_PATH="${vendor_root}:${ASCEND_CUSTOM_OPP_PATH:-}"
    export LD_LIBRARY_PATH="${vendor_root}/op_api/lib:${LD_LIBRARY_PATH:-}"
  fi
done
fla_root=/usr/local/python3.11.15/lib/python3.11/site-packages/fla_npu/opp/vendors/fla_npu_transformer
if [[ -d "${fla_root}" ]]; then
  export ASCEND_CUSTOM_OPP_PATH="${fla_root}:${ASCEND_CUSTOM_OPP_PATH:-}"
  export LD_LIBRARY_PATH="${fla_root}/op_api/lib:${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH=/work/xllm:${PYTHONPATH:-}
export TRITON_BINARY_PATH=/work/xllm/build/lib.linux-aarch64-cpython-311/xllm/triton_npu/binary
torch_npu_root="$(python3 -c 'import importlib.metadata as m; print(m.distribution("torch_npu").locate_file("torch_npu"))')"
export LD_LIBRARY_PATH="/work/xllm/build/lib.linux-aarch64-cpython-311/xllm:/usr/local/python3.11.15/lib/python3.11/site-packages/xlite/lib:${torch_npu_root}/lib:${LD_LIBRARY_PATH:-}"

exec /work/xllm/build/xllm/core/server/xllm \
  --model=/models/Qwen3.5-4B \
  --model_id=qwen35-4b \
  --host=0.0.0.0 \
  --port=18080 \
  --master_node_addr=127.0.0.1:19748 \
  --nnodes=1 \
  --node_rank=0 \
  --block_size=128 \
  --max_memory_utilization=0.90 \
  --max_tokens_per_batch=8192 \
  --max_tokens_per_chunk_for_prefill=8192 \
  --max_seqs_per_batch=32 \
  --max_concurrent_requests=64 \
  --max_body_size=268435456 \
  --enable_prefix_cache=false \
  --enable_chunked_prefill=true \
  --enable_schedule_overlap=true \
  --enable_graph=true \
  --enable_shm=false \
  --backend=llm \
  --task=generate \
  --use_zero_evict=true \
  --max_decode_token_per_sequence=600 \
  --communication_backend=hccl
