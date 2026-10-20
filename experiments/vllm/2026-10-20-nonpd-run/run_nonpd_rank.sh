#!/usr/bin/env bash
set -euo pipefail

: "${DEVICES:?comma-separated logical device IDs are required}"
: "${PORT:?HTTP port is required}"
: "${DP_RANK:?data-parallel rank is required}"
: "${HOST_IP:?host IP is required}"

MODEL_PATH="${MODEL_PATH:-/models/glm53-flash}"
MODEL_NAME="${MODEL_NAME:-glm53-flash}"
BUSINESS_IFACE="${BUSINESS_IFACE:-business}"
DP_RPC_PORT="${DP_RPC_PORT:-12335}"

export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'
export HCCL_IF_IP="$HOST_IP"
export VLLM_HOST_IP="$HOST_IP"
export GLOO_SOCKET_IFNAME="$BUSINESS_IFACE"
export TP_SOCKET_IFNAME="$BUSINESS_IFACE"
export HCCL_SOCKET_IFNAME="$BUSINESS_IFACE"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V2_MODEL_RUNNER=0
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=1024
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=1200
export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"
export PYTHONUNBUFFERED=1

exec vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --data-parallel-size 2 \
  --data-parallel-rank "$DP_RANK" \
  --data-parallel-address "$HOST_IP" \
  --data-parallel-rpc-port "$DP_RPC_PORT" \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name "$MODEL_NAME" \
  --safetensors-load-strategy prefetch \
  --max-num-seqs 16 \
  --max-model-len 98304 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --quantization ascend \
  --gpu-memory-utilization 0.92 \
  --kv-cache-dtype int8 \
  --reasoning-parser glm45 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{"num_speculative_tokens":5,"method":"deepseek_mtp","enforce_eager":false}' \
  --additional-config '{"enable_flashcomm1":true,"enable_dsa_cp":true,"enable_prefill_mc2":true,"enable_fused_mc2":1,"enable_shared_expert_dp":true,"multistream_overlap_shared_expert":false}'
