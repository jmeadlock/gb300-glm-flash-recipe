#!/usr/bin/env python3
"""Cold prefill probe. TTFT = first streamed delta of ANY kind (thinking models emit
reasoning first — waiting for visible content overstates TTFT). Nonce at prompt START
defeats prefix caching. Run ON the serving box: prefill_probe.py API_KEY [sizes...]"""
import json, time, urllib.request, uuid, sys

URL = "http://localhost:30000/v1/chat/completions"
KEY = sys.argv[1]
sizes = [int(x) for x in (sys.argv[2:] or ["8000", "32000"])]

BASE = ("The quick brown fox jumps over the lazy dog near the riverbank while autumn "
        "leaves drift slowly downward through golden afternoon light onto the forest floor. ")

def ttft_for(target_tokens):
    reps = max(1, target_tokens // 34)
    prompt = f"[cold-{uuid.uuid4().hex}] " + BASE * reps + " Reply with just OK."
    body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time(); ttft = None; ptoks = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data:"): continue
            if b"[DONE]" in line: break
            if ttft is None:
                ttft = time.time() - t0
            try:
                d = json.loads(line[5:])
                if d.get("usage"): ptoks = d["usage"].get("prompt_tokens")
            except Exception: pass
    return ttft, ptoks

out = {}
for s in sizes:
    ttft, ptoks = ttft_for(s)
    out[f"prefill_{s}"] = {"prompt_tokens": ptoks, "ttft_s": round(ttft, 2),
                           "prefill_toks_per_s": round(ptoks / ttft) if ptoks and ttft else None}
print(json.dumps(out, indent=1))
