#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/models/Qwen3.5-4B}
PORT=${PORT:-18080}

exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name qwen35-4b \
  --host 0.0.0.0 \
  --port "$PORT" \
  --device npu \
  --tp-size 1 \
  --mem-fraction-static 0.90 \
  --max-running-requests 16 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 8192 \
  --disable-radix-cache \
  --attention-backend ascend \
  --mamba-ssm-dtype bfloat16 \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 16 \
  "$@"
