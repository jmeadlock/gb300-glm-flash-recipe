#!/bin/bash
# launch-big-v1.sh — V1 recipe for full GLM-5.3 (744B, NVFP4) on ONE DGX Station GB300.
# Locked 2026-09-05 after the overnight campaign (see research/big-glm-v1-campaign-2026-09-05.md).
#
#   single-stream 33.8 tok/s (n=3) | 4-stream 57.7 agg | greedy-identical to v5 (20/20 @ temp 0)
#   cold prefill 10.6-12.6k tok/s at 12k-47k tokens | tool calls + reasoning separation intact
#   agent loop (real tool execution) correct | KV 8 GiB = 92,672 tokens = 1.41x a 65k request
#
# Changes vs v5: --cpu-offload-gb 200 -> 188 (KV right-sized to 2x65k frees 12 GiB of HBM for experts, +3.7%),
# --max-num-seqs 8 -> 4, --kv-cache-memory pinned. Kernel stays FLASHINFER_TRTLLM: cutlass (-10%) and cutedsl (+0%)
# both change greedy output and were rejected.
#
# REQUIRES: CDMM enabled (see README "Enable CDMM before you offload anything"), image vllm-glm53-uva:v0.28.0-2cf0a691,
# and recipes/sitecustomize-bigv1.py bind-mounted (autotune cache pin: relaunch in ~18 min instead of ~50).
#
# Usage: launch-big-v1.sh [name] [extra vllm args...]
set -e
RUN="${1:-v1}"; shift || true
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${MODEL_DIR:-/home/exx/models/GLM-5.3-NVFP4-big}"
CACHE_DIR="${CACHE_DIR:-$HOME/vllm-cache}"
API_KEY_FILE="${API_KEY_FILE:-$HOME/.glm_api_key}"
AT_KEY="${AT_KEY:-bigv1-shared}"     # autotune cache key; keep constant across relaunches of the same kernel family
TRACE_ENV=()
if [ -n "${ROUTE_TRACE_DIR:-}" ]; then mkdir -p "$ROUTE_TRACE_DIR"; TRACE_ENV=(-e ROUTE_TRACE_DIR=/trace -v "$ROUTE_TRACE_DIR:/trace"); fi

docker run -d --name "glm53-big-$RUN" --gpus all --shm-size 32g --network host \
  -v "$MODEL_DIR:/model:ro" \
  -v "$CACHE_DIR:/root/.cache/vllm" \
  -v "$HERE/sitecustomize-bigv1.py:/usr/lib/python3.12/sitecustomize.py:ro" \
  "${TRACE_ENV[@]}" \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e "VLLM_AUTOTUNE_CACHE_KEY=$AT_KEY" \
  -e VLLM_API_KEY="$(cat "$API_KEY_FILE")" \
  vllm-glm53-uva:v0.28.0-2cf0a691 \
  /model --host 0.0.0.0 --port 30001 \
  --served-model-name glm-5.3-big \
  --trust-remote-code \
  --quantization modelopt \
  --load-format safetensors \
  --offload-backend uva \
  --cpu-offload-gb 188 \
  --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype bfloat16 \
  --kv-cache-memory 8589934592 \
  --max-model-len 65536 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  "$@"
echo "launched glm53-big-$RUN  (autotune key $AT_KEY; ready in ~18 min warm-cache / ~50 min first time)"
