#!/usr/bin/env python3
"""v1_gates.py — endpoint gates for the V1 recipe against a READY container.
  1. 4-turn tool-using conversation (Hermes shape): find -> count -> hash -> summarize, tools executed locally.
  2. cold-prefill ladder: 8k / 16k / 32k / 64k-token prompts, TTFT and prefill tok/s.
Env: BASE_URL, API_KEY, MODEL. Prints GATE lines + JSON summary.
"""
import os, json, time, hashlib, subprocess, requests
BASE = os.environ.get("BASE_URL", "http://127.0.0.1:30001/v1"); KEY = os.environ.get("API_KEY", "x")
MODEL = os.environ.get("MODEL", "glm-5.3-big"); HDR = {"Authorization": f"Bearer {KEY}"}
out = {}

def chat(msgs, tools=None, max_tokens=600, effort="low"):
    body = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens, "temperature": 0,
            "reasoning_effort": effort}
    if tools: body["tools"] = tools
    t = time.time(); r = requests.post(f"{BASE}/chat/completions", json=body, headers=HDR, timeout=600)
    r.raise_for_status(); j = r.json(); j["_wall"] = round(time.time() - t, 2); return j

# ---- gate 1: agent loop ----
TOOLS = [{"type": "function", "function": {"name": "terminal", "description": "Run a shell command in the sandbox, return stdout.",
          "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}]
os.makedirs("/tmp/v1gate", exist_ok=True)
for i in range(7): open(f"/tmp/v1gate/f{i}.txt", "w").write(f"line {i}\n" * (i + 1))
msgs = [{"role": "system", "content": "You are a careful agent. Use the terminal tool; one command per call; finish with a one-line answer."},
        {"role": "user", "content": "In /tmp/v1gate: list the .txt files, count total lines across them, then print the sha256 of the concatenation of all files sorted by name. Report: file count, total lines, and the first 12 hex chars of the hash."}]
turns = 0; tool_calls = 0; final = None; t0 = time.time()
while turns < 8:
    j = chat(msgs, TOOLS); m = j["choices"][0]["message"]; turns += 1
    msgs.append({k: v for k, v in m.items() if k in ("role", "content", "tool_calls") and v is not None})
    if m.get("tool_calls"):
        for tc in m["tool_calls"]:
            tool_calls += 1
            args = json.loads(tc["function"]["arguments"]); cmd = args.get("cmd", "")
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30, cwd="/tmp/v1gate")
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": (res.stdout + res.stderr)[:4000]})
        continue
    final = m.get("content") or ""; break
expect_hash = hashlib.sha256(b"".join(open(f"/tmp/v1gate/f{i}.txt", "rb").read() for i in range(7))).hexdigest()[:12]
out["agent"] = {"turns": turns, "tool_calls": tool_calls, "wall_s": round(time.time() - t0, 1),
                "final": (final or "")[:300], "expect_files": 7, "expect_lines": 28, "expect_hash12": expect_hash,
                "hash_ok": expect_hash in (final or ""), "lines_ok": "28" in (final or ""), "files_ok": "7" in (final or "")}
print("GATE agent", json.dumps(out["agent"]))

# ---- gate 2: cold prefill ladder ----
base = open("/usr/share/dict/words").read() if os.path.exists("/usr/share/dict/words") else ("lorem ipsum dolor sit amet " * 20000)
out["prefill"] = []
for target in (8000, 16000, 32000, 64000):
    text = (base * 40)[: target * 4]   # ~4 chars/token
    body = {"model": MODEL, "messages": [{"role": "user", "content": text + "\n\nReply with the single word OK."}],
            "max_tokens": 4, "temperature": 0, "reasoning_effort": "low", "stream": True}
    t = time.time(); first = None
    with requests.post(f"{BASE}/chat/completions", json=body, headers=HDR, stream=True, timeout=900) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line and line.startswith(b"data:") and b"content" in line:
                first = time.time(); break
    usage = chat([{"role": "user", "content": text[:200]}], max_tokens=1)  # cheap: just to ensure alive
    # get prompt tokens via a non-stream call with max_tokens=1
    r2 = requests.post(f"{BASE}/chat/completions", json={**body, "stream": False, "max_tokens": 1}, headers=HDR, timeout=900).json()
    ptok = r2.get("usage", {}).get("prompt_tokens")
    ttft = round((first or time.time()) - t, 2)
    row = {"target": target, "prompt_tokens": ptok, "ttft_s": ttft, "prefill_tok_s": round(ptok / ttft) if ptok and ttft else None}
    out["prefill"].append(row); print("GATE prefill", json.dumps(row))
print("V1_GATES " + json.dumps(out))
