#!/usr/bin/env python3
"""Bench v3 — concurrency sweep against tuned server (auth-aware).
Reads API key from argv[1]. Run ON the GB300."""
import json, time, urllib.request, uuid, concurrent.futures, sys

URL = "http://localhost:30000/v1/chat/completions"
KEY = sys.argv[1]

def chat(prompt, max_tokens=256, stream=False):
    body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0}
    data = json.dumps(body).encode()
    req = urllib.request.Request(URL, data=data, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    dt = time.time() - t0
    u = out.get("usage", {})
    return dt, u.get("completion_tokens", 0), u.get("prompt_tokens", 0)

def stream_ttft(prompt, max_tokens=2):
    body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": True}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if line.startswith(b"data:") and b"[DONE]" not in line:
                return time.time() - t0
    return None

res = {}

# warm up
chat("hello", 8)

# single-stream decode
runs = []
for _ in range(3):
    dt, ct, _ = chat("Write a detailed essay about the history of computing.", 512)
    runs.append(round(ct / dt, 1))
res["decode_c1"] = {"runs": runs, "median": sorted(runs)[1]}

# concurrency sweep with per-request nonce
def one(i):
    dt, ct, _ = chat(f"[req {uuid.uuid4().hex[:8]}] Write a short story about a robot. Item {i}.", 256)
    return dt, ct

for c in (4, 8, 16, 32):
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
        outs = list(ex.map(one, range(c)))
    wall = time.time() - t0
    total_tok = sum(ct for _, ct in outs)
    res[f"c{c}"] = {"agg_toks_per_s": round(total_tok / wall, 1),
                    "per_stream": round(total_tok / wall / c, 1),
                    "wall_s": round(wall, 1)}

# TTFT under load: fire 8 background, measure streamed TTFT of 9th
with concurrent.futures.ThreadPoolExecutor(max_workers=9) as ex:
    futs = [ex.submit(one, i) for i in range(8)]
    time.sleep(1.0)
    ttft = stream_ttft(f"[{uuid.uuid4().hex[:8]}] Quick answer: what is 7*6?")
    concurrent.futures.wait(futs)
res["ttft_under_8_load_s"] = round(ttft, 2) if ttft else None

print(json.dumps(res, indent=1))
