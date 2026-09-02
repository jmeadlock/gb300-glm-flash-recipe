#!/bin/bash
# Build the GLM-5.3-Flash-capable SGLang image (PR #36507 branch + latest transformers),
# then freeze it so production boots need no network.
set -e
docker pull lmsysorg/sglang:latest
docker run -d --name glm53-build lmsysorg/sglang:latest sleep infinity
docker exec glm53-build bash -c '
  cd /sgl-workspace/sglang &&
  git fetch origin pull/36507/head:pr36507 --depth 80 &&
  git checkout pr36507 &&
  pip install --upgrade transformers
'
docker commit glm53-build glm53-nvfp4-sglang:gb300-v2
docker rm -f glm53-build
echo "Image frozen: glm53-nvfp4-sglang:gb300-v2"
