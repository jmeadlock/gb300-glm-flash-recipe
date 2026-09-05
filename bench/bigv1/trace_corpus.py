#!/usr/bin/env python3
"""trace_corpus.py — drive the endpoint with a mixed workload so the ROUTE_TRACE hook captures
representative routing. Prints per-segment token counts. Env: API_KEY, BASE_URL, MODEL, SEG (comma list).
Segments: prose_low, prose_max, code_low, code_max, tools_low (multi-turn tool-call replay).
Each segment streams a set of prompts sequentially at C1 (interactive shape) and records wall/tokens.
"""
import json, os, time, urllib.request, random

base = os.getenv("BASE_URL", "http://127.0.0.1:30001/v1").rstrip("/")
model = os.getenv("MODEL", "glm-5.3-big")
key = os.environ["API_KEY"]
H = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
SEGS = os.getenv("SEG", "prose_low,code_low,tools_low,prose_max,code_max").split(",")
NONCE = str(int(time.time()))
random.seed(7)

PROSE = [
    "Write a detailed essay on the history of the Roman aqueducts, at least 800 words.",
    "Explain how the Federal Reserve sets interest rates and the transmission mechanism to mortgages, 700 words.",
    "Write a short story about a lighthouse keeper who discovers the light has started responding to him, 900 words.",
    "Compare the strategic situations of Athens and Sparta before the Peloponnesian War in 800 words.",
    "Explain how mRNA vaccines work to a curious high-school student, 700 words.",
    "Write a product review of a hypothetical electric standing desk, 600 words, balanced.",
]
CODE = [
    "Write a Python module implementing an LRU cache with TTL, thread-safe, with type hints, docstrings and a pytest suite.",
    "Implement Dijkstra's algorithm in Rust with a binary heap, generic over edge weight, with unit tests.",
    "Write a bash script that rotates logs in /var/log/app, compresses files older than 7 days, deletes older than 90, and is idempotent. Explain each section.",
    "Write a TypeScript React hook useDebouncedFetch(url, ms) with abort on unmount and a test using vitest.",
    "Refactor this into a clean state machine and explain: def f(s,e):\n if s=='idle' and e=='start': return 'run'\n if s=='run' and e=='pause': return 'paused'\n if s=='paused' and e=='resume': return 'run'\n if e=='stop': return 'idle'\n return s",
    "Write a SQL migration and query for a table of sensor readings partitioned by day, with a rollup view of hourly p50/p95, for Postgres 16.",
]
TOOLS = [{"type": "function", "function": {"name": "terminal", "description": "Run a shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
         {"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
         {"type": "function", "function": {"name": "write_file", "description": "Write a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}]

def chat(messages, effort, max_tokens, tools=None):
    p = {"model": model, "stream": True, "temperature": 0, "max_tokens": max_tokens,
         "stream_options": {"include_usage": True}, "chat_template_kwargs": {"reasoning_effort": effort},
         "messages": messages}
    if tools: p["tools"] = tools; p["tool_choice"] = "auto"
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(p).encode(), headers=H)
    usage = None; content = ""; calls = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                content += delta.get("content") or ""
                for tc in delta.get("tool_calls") or []:
                    i = tc.get("index", 0); c = calls.setdefault(i, {"name": "", "args": ""})
                    f = tc.get("function") or {}
                    c["name"] += f.get("name") or ""; c["args"] += f.get("arguments") or ""
    return usage or {}, content, [calls[i] for i in sorted(calls)]

def seg_simple(prompts, effort, max_tokens):
    tot = 0; t0 = time.monotonic()
    for i, q in enumerate(prompts):
        u, _, _ = chat([{"role": "user", "content": f"[{NONCE}-{i}] {q}"}], effort, max_tokens)
        tot += u.get("completion_tokens", 0)
    return tot, time.monotonic() - t0

def fake_tool_result(name, args):
    try: a = json.loads(args)
    except Exception: a = {}
    if name == "terminal":
        cmd = a.get("command", "")
        if "ls" in cmd: return "notes.md  data.csv  build.log  src/  tests/\n"
        if "wc" in cmd: return " 412 notes.md\n 9981 data.csv\n 233 build.log\n"
        if "sha256" in cmd or "shasum" in cmd: return "3b1f9c0d8a7e4f2b6c5d1e0a9f8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b  data.csv\n"
        if "grep" in cmd: return "data.csv:17:sensor_7,2026-09-04T22:11:03Z,41.7\ndata.csv:88:sensor_7,2026-09-04T23:02:41Z,42.1\n"
        return "ok\n"
    if name == "read_file":
        p = a.get("path", "")
        if p.endswith("notes.md"): return "# Notes\n- rotate logs weekly\n- sensor_7 drifts +0.4/day; recalibrate 2026-09-10\n- TODO: hash data.csv before upload\n"
        if p.endswith("build.log"): return "[ok] compile 212 files\n[warn] deprecated API in src/io.py:44\n[ok] tests 188 passed, 2 skipped\n"
        return "(empty)\n"
    if name == "write_file": return "written\n"
    return "unknown tool\n"

def seg_tools(effort, max_tokens, tasks):
    tot = 0; t0 = time.monotonic(); turns = 0
    for i, task in enumerate(tasks):
        msgs = [{"role": "system", "content": "You are an agent with terminal, read_file and write_file tools. Use tools; do not guess file contents. Finish with a one-line final answer."},
                {"role": "user", "content": f"[{NONCE}-t{i}] {task}"}]
        for hop in range(8):
            u, content, calls = chat(msgs, effort, max_tokens, tools=TOOLS)
            tot += u.get("completion_tokens", 0); turns += 1
            if not calls: break
            msgs.append({"role": "assistant", "content": content or None, "tool_calls": [
                {"id": f"call_{i}_{hop}_{k}", "type": "function", "function": {"name": c["name"], "arguments": c["args"] or "{}"}} for k, c in enumerate(calls)]})
            for k, c in enumerate(calls):
                msgs.append({"role": "tool", "tool_call_id": f"call_{i}_{hop}_{k}", "content": fake_tool_result(c["name"], c["args"])})
    return tot, time.monotonic() - t0, turns

TASKS = [
    "List the files in the working directory, read notes.md, then compute the sha256 of data.csv and report the first 12 hex chars along with the recalibration date from the notes.",
    "Count the lines in every file in the directory, read build.log, and write a file summary.txt with the line counts and the number of warnings. Then report the total lines.",
    "Find every sensor_7 line in data.csv with grep, read notes.md for the drift note, and write drift.md that states the drift rate and the two timestamps you found. Report the drift rate.",
    "Read build.log and notes.md, then write todo.md that merges the TODO from notes with any warnings from the build log. Report how many items todo.md contains.",
]

for seg in SEGS:
    if seg == "prose_low": n, s = seg_simple(PROSE, "low", 1200); print(f"SEG {seg} tokens={n} wall={s:.0f}s tok_s={n/s:.1f}", flush=True)
    elif seg == "prose_max": n, s = seg_simple(PROSE[:3], "max", 4000); print(f"SEG {seg} tokens={n} wall={s:.0f}s tok_s={n/s:.1f}", flush=True)
    elif seg == "code_low": n, s = seg_simple(CODE, "low", 1500); print(f"SEG {seg} tokens={n} wall={s:.0f}s tok_s={n/s:.1f}", flush=True)
    elif seg == "code_max": n, s = seg_simple(CODE[:3], "max", 4000); print(f"SEG {seg} tokens={n} wall={s:.0f}s tok_s={n/s:.1f}", flush=True)
    elif seg == "tools_low": n, s, t = seg_tools("low", 800, TASKS); print(f"SEG {seg} tokens={n} wall={s:.0f}s tok_s={n/s:.1f} turns={t}", flush=True)
    # marker so the analyzer can split segments by wall-clock
    print(f"MARK {seg} {time.time():.3f}", flush=True)
