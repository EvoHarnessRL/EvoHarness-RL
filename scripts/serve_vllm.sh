#!/usr/bin/env bash
# Serve a policy (or consolidator) checkpoint behind an OpenAI-compatible API.
#   MODEL=/path/to/checkpoint bash scripts/serve_vllm.sh
set -euo pipefail

MODEL="${MODEL:?set MODEL to a HF model id or local checkpoint}"
NAME="${NAME:-$MODEL}"
PORT="${PORT:-8000}"
TP="${TP:-1}"

exec vllm serve "$MODEL" \
  --served-model-name "$NAME" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.85}" \
  --trust-remote-code
