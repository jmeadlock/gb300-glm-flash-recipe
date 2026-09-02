#!/bin/bash
# Post-start warmup: exercises each batch shape once so users never pay
# first-hit kernel autotune (up to 30s per shape). Run after EVERY server start.
set -e
API_KEY=${API_KEY:?set API_KEY}
URL=${URL:-http://localhost:30000/v1/chat/completions}

fire() {  # fire N concurrent short requests
  local n=$1
  for i in $(seq 1 $n); do
    curl -s -o /dev/null -m 300 "$URL" \
      -H "Content-Type: application/json" -H "Authorization: Bearer $API_KEY" \
      -d "{\"model\":\"glm-5.3-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"warmup $n-$i $(date +%s%N)\"}],\"max_tokens\":64}" &
  done
  wait
  echo "warmed C$n"
}

for c in 1 4 8 16 32; do fire $c; done
echo "Warmup complete — server is production-ready."
