#!/usr/bin/env python3
"""Snapshot vLLM spec-decode counters from /metrics. Usage: spec_metrics.py [label]"""
import os, sys, urllib.request, json, re
base = os.getenv("BASE_URL", "http://127.0.0.1:30001").replace("/v1", "")
key = os.environ.get("API_KEY", "")
req = urllib.request.Request(base + "/metrics", headers={"Authorization": f"Bearer {key}"})
txt = urllib.request.urlopen(req, timeout=30).read().decode()
out = {}
for line in txt.splitlines():
    if line.startswith("vllm:spec_decode_num_") and "_per_pos" not in line:
        m = re.match(r"(vllm:spec_decode_num_\w+)\{[^}]*\}\s+([\d.e+]+)", line)
        if m:
            out[m.group(1).replace("vllm:spec_decode_num_", "")] = float(m.group(2))
label = sys.argv[1] if len(sys.argv) > 1 else "snap"
d = out.get("drafts_total") or out.get("drafts", 0)
dt = out.get("draft_tokens_total") or out.get("draft_tokens", 0)
at = out.get("accepted_tokens_total") or out.get("accepted_tokens", 0)
if d:
    out["mean_acceptance_length"] = round(1 + at / d, 3)
if dt:
    out["draft_acceptance_rate"] = round(at / dt, 3)
print("SPEC", label, json.dumps(out))
