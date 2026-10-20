#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH inside the container}"
: "${LOCAL_IP:?set LOCAL_IP for this host}"
: "${NIC_NAME:?set NIC_NAME for this host}"
: "${MASTER_IP:?set MASTER_IP to DP rank 0 host}"
: "${DP_RANK:?set DP_RANK to 0..DP_SIZE-1}"

DP_SIZE="${DP_SIZE:-4}"
TP_SIZE="${TP_SIZE:-8}"
HTTP_PORT="${HTTP_PORT:-8000}"
RPC_PORT="${RPC_PORT:-12321}"
VISIBLE_DEVICES="${VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export VLLM_USE_V2_MODEL_RUNNER=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=1024
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=1200
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export HCCL_IF_IP="$LOCAL_IP"
export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"

args=(
  vllm serve "$MODEL_PATH"
  --host 0.0.0.0
  --port "$HTTP_PORT"
  --max-model-len 133120
  --data-parallel-size "$DP_SIZE"
  --data-parallel-size-local 1
  --data-parallel-start-rank "$DP_RANK"
  --data-parallel-address "$MASTER_IP"
  --data-parallel-rpc-port "$RPC_PORT"
  --tensor-parallel-size "$TP_SIZE"
  --enable-expert-parallel
  --seed 1024
  --served-model-name glm53-flash
  --safetensors-load-strategy prefetch
  --max-num-seqs 32
  --max-num-batched-tokens 8192
  --trust-remote-code
  --quantization ascend
  --gpu-memory-utilization 0.85
  --enable-prefix-caching
  --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp","enforce_eager":true}'
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,16,32,64,96,128]}'
)

if [[ "$DP_RANK" != "0" ]]; then
  args+=(--headless)
else
  args+=(--api-server-count 1)
fi

exec "${args[@]}"
