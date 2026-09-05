#!/bin/bash
# bench3.sh <runname> — n=3 prose bench (C1/C4/C8) + 1 code bench, appended to runs/<run>/bench.txt
RUN="$1"; C=/home/milo/big-v1-campaign; O="$C/runs/$RUN/bench.txt"
export API_KEY="$(cat /home/milo/.glm_api_key)"
cd /home/milo
echo "=== bench start $(date -Is) run=$RUN" >> "$O"
for i in 1 2 3; do
  echo "--- prose rep $i" >> "$O"
  python3 bench_big.py 2>&1 | grep BENCH >> "$O"
done
echo "--- code rep 1" >> "$O"
python3 bench_big_code.py 2>&1 | grep BENCH >> "$O"
echo "=== bench end $(date -Is)" >> "$O"
date +%s > "$C/runs/$RUN/bench_end_epoch.txt"
python3 - "$O" <<'PY'
import sys, json, re, statistics as st
rows = {}
for line in open(sys.argv[1]):
    if line.startswith("BENCH"):
        d = json.loads(line[5:]); rows.setdefault(d["C"], []).append(d["agg_tok_s"])
for c in sorted(rows):
    v = rows[c][:3]  # prose reps (code rep appended last per C; show separately)
    print(f"C{c} prose n={len(v)} agg mean={st.mean(v):.1f} min={min(v):.1f} max={max(v):.1f}" + (f"  code={rows[c][3]:.1f}" if len(rows[c]) > 3 else ""))
PY
