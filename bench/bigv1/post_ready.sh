#!/bin/bash
# post_ready.sh <runname> [trace] — everything that runs against a READY container, in order:
#   1. tools smoke (big_smoke.py)   2. greedy-equivalence capture   3. bench3 (n=3 prose + code)
#   4. [trace] corpus drive (writes marks)   5. summary line into LEDGER
RUN="$1"; DO_TRACE="$2"; C=/home/milo/big-v1-campaign; R="$C/runs/$RUN"; mkdir -p "$R"
export API_KEY="$(cat /home/milo/.glm_api_key)"
cd /home/milo
echo "=== post_ready $RUN $(date -Is)" | tee -a "$R/post_ready.log"
echo "--- smoke" | tee -a "$R/post_ready.log"
python3 big_smoke.py 2>&1 | grep -E "MODELS|TOOL|REASON|PASS|FAIL|tool_calls|Error" | head -8 | tee -a "$R/post_ready.log"
echo "--- greedy capture" | tee -a "$R/post_ready.log"
python3 "$C/greedy_equiv.py" "$R/greedy.json" 2>&1 | tail -1 | tee -a "$R/post_ready.log"
if [ -f "$C/runs/r1-base/greedy.json" ] && [ "$RUN" != "r1-base" ]; then
  python3 "$C/greedy_equiv.py" --compare "$C/runs/r1-base/greedy.json" "$R/greedy.json" | tee -a "$R/post_ready.log"
fi
echo "--- bench3" | tee -a "$R/post_ready.log"
bash "$C/bench3.sh" "$RUN" 2>&1 | tee -a "$R/post_ready.log"
if [ "$DO_TRACE" = "trace" ]; then
  echo "MARK bench_and_gates $(date +%s.%3N)" > "$R/marks.txt"
  echo "--- trace corpus" | tee -a "$R/post_ready.log"
  python3 "$C/trace_corpus.py" 2>&1 | tee -a "$R/post_ready.log" | grep -E "^MARK" >> "$R/marks.txt"
  ls -la "$C/trace/$RUN/" | tee -a "$R/post_ready.log"
fi
echo "=== post_ready done $RUN $(date -Is)" | tee -a "$R/post_ready.log"
