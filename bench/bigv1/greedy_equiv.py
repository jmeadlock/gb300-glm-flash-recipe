#!/usr/bin/env python3
"""greedy_equiv.py <outfile> — 20 fixed prompts, temp 0, effort low, max_tokens 200, non-stream.
Saves {prompt_idx: text} JSON. Compare two runs with: greedy_equiv.py --compare a.json b.json
Bit-exact match is the bar for 'quality-neutral' between kernel/backend/offload changes; small
divergence after N tokens is reported as first-divergence index so we can judge fp noise vs real change."""
import json, os, sys, urllib.request

if sys.argv[1] == "--compare":
    a, b = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
    same = 0; first_div = []
    for k in sorted(a, key=int):
        x, y = a[k], b.get(k, "")
        if x == y: same += 1; continue
        i = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        first_div.append((int(k), i, len(x), len(y)))
    print(f"GREEDY_EQUIV identical={same}/{len(a)}")
    for k, i, lx, ly in first_div: print(f"  prompt {k}: first divergence at char {i} (len {lx} vs {ly})")
    sys.exit(0)

base = os.getenv("BASE_URL", "http://127.0.0.1:30001/v1").rstrip("/")
model = os.getenv("MODEL", "glm-5.3-big")
H = {"Authorization": f"Bearer {os.environ['API_KEY']}", "Content-Type": "application/json"}
PROMPTS = [
    "Explain the difference between TCP and UDP in exactly three sentences.",
    "Write a Python function that returns the nth Fibonacci number iteratively.",
    "What is the capital of Australia and why is it not Sydney?",
    "Summarize the plot of Macbeth in five sentences.",
    "Give me a bash one-liner to find the ten largest files under the current directory.",
    "Explain what a Grace-Blackwell superchip is, briefly.",
    "Translate to French: The meeting has been moved to Thursday afternoon.",
    "List four causes of the fall of the Western Roman Empire, one line each.",
    "Write a SQL query that returns the top 5 customers by total order value.",
    "What does the Rust borrow checker prevent? Two sentences.",
    "Describe how a mixture-of-experts transformer routes tokens.",
    "Compute 17 * 23 and show the work.",
    "Write a haiku about a data center at night.",
    "What is the time complexity of binary search and why?",
    "Explain NVFP4 quantization to an engineer in four sentences.",
    "Give three arguments against premature optimization.",
    "Write a regex that matches an IPv4 address and explain each part.",
    "What is the difference between a process and a thread?",
    "Draft a two-sentence polite decline to a meeting invitation.",
    "Explain why speculative decoding is lossless when the target model verifies drafts.",
]
out = {}
for i, q in enumerate(PROMPTS):
    p = {"model": model, "temperature": 0, "max_tokens": 200, "chat_template_kwargs": {"reasoning_effort": "low"},
         "messages": [{"role": "user", "content": q}]}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(p).encode(), headers=H)
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    out[str(i)] = (m.get("reasoning_content") or "") + "\u241f" + (m.get("content") or "")
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(f"GREEDY saved {len(out)} outputs to {sys.argv[1]}")
