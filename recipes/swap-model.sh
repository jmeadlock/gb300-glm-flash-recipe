#!/bin/bash
# Quick-swap between same-architecture NVFP4 checkpoints (e.g. stock <-> derisked).
# Any glm5_next NVFP4 snapshot is a drop-in for this recipe: same image, same flags,
# same draft model. Swap = container relaunch + warmup (~5 min wall).
#
# Usage: MODEL_DIR=/path/to/variant/original API_KEY=... ./swap-model.sh
set -e
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the variant snapshot dir}
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
    --tool-call-parser glm47 --reasoning-parser glm45 \
    --cuda-graph-max-bs 32 --max-running-requests 32 \
    --max-prefill-tokens 8192 --chunked-prefill-size 8192 \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path /draft \
    --speculative-draft-attention-backend fa4"
echo "Swapped to: $MODEL_DIR"
echo "Wait for /health, then RUN warmup.sh. Served name stays glm-5.3-flash — no client changes."
