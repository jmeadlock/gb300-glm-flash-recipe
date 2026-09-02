#!/usr/bin/env python3
"""GB300 bench v2 — fixed prefill probe (TTFT = first delta of ANY kind incl. reasoning),
real prompt sizes, and a cleaner concurrency sweep."""
import json, time, urllib.request, uuid, concurrent.futures, statistics as st

URL = "http://localhost:30000/v1/chat/completions"

def stream_req(prompt, max_tokens, temperature=0.0):
    body = {"model": "/model", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: "): continue
            payload = line[6:]
            if payload == "[DONE]": break
            d = json.loads(payload)
            if d.get("usage"): usage = d["usage"]
            ch = d.get("choices")
            if ch and ttft is None:
                delta = ch[0].get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    ttft = time.time() - t0
    return {"ttft": ttft, "total": time.time() - t0, "usage": usage}

def big_prompt(target_tokens, nonce):
    # ~1.3 tokens/word for this text; repeat until big enough
    para = ("In the long history of computation, engineers repeatedly discovered that memory "
            "bandwidth, not arithmetic, bounds real workloads. From mercury delay lines to "
            "HBM stacks, every generation rebalanced the same equation of capacity, latency, "
            "and cost, and every generation was surprised again. ")
    words_needed = int(target_tokens / 1.3)
    reps = words_needed // len(para.split()) + 1
    return nonce + " " + para * reps + "\nSummarize the above in one word."

results = {}
stream_req("Say OK.", 5)  # warmup

# Cold prefill: nonce at start, max_tokens=2, TTFT = first any-delta
for label, target in [("prefill_8k", 8000), ("prefill_32k", 32000)]:
    r = stream_req(big_prompt(target, uuid.uuid4().hex), 2)
    pt = r["usage"]["prompt_tokens"]
    results[label] = {"prompt_tokens": pt, "ttft_s": round(r["ttft"], 2),
                      "prefill_toks_per_s": round(pt / r["ttft"], 0)}

# Concurrency: fixed 256-token decode, temperature 0, distinct nonce prompts
def one(i):
    return stream_req(f"Nonce {uuid.uuid4().hex}. Write a 300 word story about a lighthouse.", 256)
for c in [8, 16, 32]:
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
        rs = list(ex.map(one, range(c)))
    wall = time.time() - t0
    toks = sum(r["usage"]["completion_tokens"] for r in rs)
    ttfts = [r["ttft"] for r in rs if r["ttft"]]
    results[f"concurrency_{c}"] = {"agg_toks_per_s": round(toks / wall, 1),
                                   "per_stream": round(toks / wall / c, 1),
                                   "median_ttft_s": round(st.median(ttfts), 2),
                                   "wall_s": round(wall, 1)}

print(json.dumps(results, indent=2))
