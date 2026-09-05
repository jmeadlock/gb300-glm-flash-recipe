#!/bin/bash
# watch_ready.sh <runname> — poll container until vLLM is serving or the container dies. Prints key log lines.
RUN="$1"; N="glm53-big-$RUN"
for i in $(seq 1 120); do
  sleep 30
  st=$(docker inspect -f '{{.State.Status}}' "$N" 2>/dev/null)
  if [ "$st" != "running" ]; then echo "DEAD status=$st"; docker logs "$N" 2>&1 | grep -iE "error|Traceback|exception" | tail -20; exit 1; fi
  if docker logs "$N" 2>&1 | grep -q "Application startup complete"; then
    echo "READY after $((i*30))s"
    docker logs "$N" 2>&1 | grep -E "AUTOTUNE_KEY|ROUTE_TRACE|NvFp4 MoE backend|Total CPU offloaded|Loading weights took|GPU KV cache size|autotune cache file|Saved .* configs|Autotuning process|fp8_e4m3|scaling factor|Initializing routed experts|Application startup" | cut -c1-220
    exit 0
  fi
  if [ $((i % 10)) -eq 0 ]; then echo "t=$((i*30))s $(docker logs "$N" 2>&1 | tail -1 | cut -c1-120)"; fi
done
echo "TIMEOUT"; exit 2
