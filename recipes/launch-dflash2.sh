#!/bin/bash
# Launch GLM-5.3-Flash NVFP4 + DFlash2 speculative decoding (interactive-optimized).
# Best for C1-C8. For batch workloads (C16+) use launch-ar.sh instead.
set -e
MODEL_DIR=${MODEL_DIR:-/home/exx/models/GLM-5.3-Flash-NVFP4/aa28e1f54130286c95fee10d0705c74ce8743734/original}
DRAFT_DIR=${DRAFT_DIR:-/home/exx/models/GLM-5.3-Flash-DFlash2}
API_KEY=${API_KEY:?set API_KEY}

docker rm -f glm53-flash 2>/dev/null || true
docker run -d --name glm53-flash --restart unless-stopped \
  --gpus all --shm-size 32g --network host \
  -v "$MODEL_DIR":/model:ro \
  -v "$DRAFT_DIR":/draft:ro \
  glm53-nvfp4-sglang:gb300-v2 \
  bash -c "cd /sgl-workspace/sglang && python3 -m sglang.launch_server \
    --model-path /model --host 0.0.0.0 --port 30000 \
    --quantization modelopt_fp4 --trust-remote-code \
    --api-key $API_KEY --served-model-name glm-5.3-flash \
    --cuda-graph-max-bs 32 --max-running-requests 32 \
    --max-prefill-tokens 8192 --chunked-prefill-size 8192 \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path /draft \
    --speculative-draft-attention-backend fa4"
echo "Launched. Wait for /health then RUN warmup.sh — do not skip it."
