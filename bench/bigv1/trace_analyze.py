#!/usr/bin/env python3
"""trace_analyze.py <trace_dir> [marks.txt]
Reads trace-<pid>.i16 (int16 [tokens, layers, topk]) + .steps (time n L K per step).
Reports, per MoE layer and overall:
  - top-k expert coverage: share of routings captured by the top 25%/50% most popular experts (static placement)
  - LRU simulation at capacities 64/96/128/160 of 256 experts per layer (temporal locality)
  - decode-only vs all-token (prefill inflates uniformity; decode is what C1 pays for)
  - cross-segment stability: overlap of top-128 sets between segments if marks.txt given
Outputs JSON summary + a compact table.
"""
import sys, os, glob, json, numpy as np
from collections import OrderedDict

d = sys.argv[1]
marks = []
if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
    for line in open(sys.argv[2]):
        if line.startswith("MARK"):
            _, seg, ts = line.split(); marks.append((seg, float(ts)))

data = []; steps = []
for f in sorted(glob.glob(os.path.join(d, "trace-*.i16"))):
    pid = f.split("trace-")[1].split(".")[0]
    st = [l.split() for l in open(f.replace(".i16", ".steps"))]
    st = [(float(a), int(b), int(c), int(e)) for a, b, c, e in st]
    L, K = st[0][2], st[0][3]
    arr = np.fromfile(f, dtype=np.int16)
    n = arr.size // (L * K)
    arr = arr[: n * L * K].reshape(n, L, K)
    data.append(arr); steps.extend(st)
X = np.concatenate(data)
ntag = sum(nt for _, nt, _, _ in steps)
X = X[:ntag]  # steps file may lag the binary by a few flushed steps
L, K = X.shape[1], X.shape[2]
# per-step token counts let us tag decode steps (n<=8 tokens) vs prefill chunks
tags = np.concatenate([np.full(nt, 0 if nt <= 8 else 1, dtype=np.int8) for _, nt, _, _ in steps])[: X.shape[0]]
tstamps = np.concatenate([np.full(nt, ts) for ts, nt, _, _ in steps])[: X.shape[0]]
moe_layers = [l for l in range(L) if (X[:, l, :] >= 0).any()]
E = int(X[X >= 0].max()) + 1
print(f"tokens={X.shape[0]} layers={L} moe_layers={len(moe_layers)} topk={K} experts={E} decode_tokens={(tags==0).sum()} prefill_tokens={(tags==1).sum()}")

def coverage(Y):
    """Y: [tokens, K] expert ids for one layer. Returns dict of static-cache hit rates."""
    Y = Y[Y >= 0]
    if Y.size == 0: return None
    cnt = np.bincount(Y, minlength=E).astype(np.float64)
    order = np.argsort(-cnt); cum = np.cumsum(cnt[order]) / cnt.sum()
    out = {}
    for frac in (0.25, 0.375, 0.5, 0.625):
        out[f"top{int(frac*100)}"] = round(float(cum[int(E * frac) - 1]), 4)
    out["gini"] = round(float(1 - 2 * np.trapz(np.sort(cnt) / cnt.sum()) / E + 1 / E), 4)
    return out, order

def lru_sim(Y, cap):
    """LRU over expert ids for one layer, per-token top-k access; returns hit rate."""
    cache = OrderedDict(); hits = 0; tot = 0
    for row in Y:
        for e in row:
            if e < 0: continue
            tot += 1
            if e in cache: cache.move_to_end(e); hits += 1
            else:
                cache[e] = 1
                if len(cache) > cap: cache.popitem(last=False)
    return hits / tot if tot else 0.0

summary = {"per_layer": {}, "overall": {}}
dec = tags == 0
agg_static = {f: [] for f in ("top25", "top37", "top50", "top62")}
agg_lru = {c: [] for c in (64, 96, 128, 160)}
tops = {}
for l in moe_layers:
    Y = X[dec, l, :]
    cov = coverage(Y)
    if cov is None: continue
    cov, order = cov
    tops[l] = order[:128]
    lr = {c: round(lru_sim(Y[:: max(1, Y.shape[0] // 4000)], c), 4) for c in agg_lru}  # subsample for speed
    summary["per_layer"][l] = {"static": cov, "lru": lr}
    for f in agg_static: agg_static[f].append(cov[f])
    for c in agg_lru: agg_lru[c].append(lr[c])
summary["overall"]["static_decode_mean"] = {f: round(float(np.mean(v)), 4) for f, v in agg_static.items()}
summary["overall"]["lru_decode_mean"] = {c: round(float(np.mean(v)), 4) for c, v in agg_lru.items()}
# offloaded layers in v5 = first ~42 MoE layers: report those separately (that is where bytes cross C2C)
off = moe_layers[:42]
summary["overall"]["static_decode_offloaded42"] = {f: round(float(np.mean([summary["per_layer"][l]["static"][f] for l in off])), 4) for f in agg_static}
summary["overall"]["lru_decode_offloaded42"] = {c: round(float(np.mean([summary["per_layer"][l]["lru"][c] for l in off])), 4) for c in agg_lru}
# all-token (incl prefill) static coverage for contrast
allcov = []
for l in moe_layers:
    c = coverage(X[:, l, :])
    if c: allcov.append(c[0]["top50"])
summary["overall"]["static_alltokens_top50_mean"] = round(float(np.mean(allcov)), 4)

# segment stability
if marks:
    bounds = []; prev = tstamps.min() - 1
    for seg, ts in marks:
        bounds.append((seg, prev, ts)); prev = ts
    segtops = {}
    for seg, a, b in bounds:
        m = dec & (tstamps > a) & (tstamps <= b)
        if m.sum() < 500: continue
        segtops[seg] = {l: coverage(X[m, l, :])[1][:128] for l in off if coverage(X[m, l, :])}
        summary["overall"].setdefault("seg_tokens", {})[seg] = int(m.sum())
    names = list(segtops)
    ov = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            o = [len(set(segtops[a][l]) & set(segtops[b][l])) / 128 for l in segtops[a] if l in segtops[b]]
            ov[f"{a}|{b}"] = round(float(np.mean(o)), 3)
    summary["overall"]["top128_overlap_between_segments"] = ov
    # hit rate on segment B when placement is chosen from segment A (the real test of a static split)
    cross = {}
    for a in names:
        for b in names:
            if a == b: continue
            m = dec & (tstamps > dict((s, (x, y)) for s, x, y in bounds)[b][0]) & (tstamps <= dict((s, (x, y)) for s, x, y in bounds)[b][1])
            hr = []
            for l in off:
                if l not in segtops[a]: continue
                Y = X[m, l, :]; Y = Y[Y >= 0]
                if Y.size: hr.append(float(np.isin(Y, segtops[a][l]).mean()))
            if hr: cross[f"place_from={a} eval_on={b}"] = round(float(np.mean(hr)), 4)
    summary["overall"]["static128_cross_segment_hit"] = cross

print("SUMMARY " + json.dumps(summary["overall"]))
print("\nlayer  top25  top50  top62 | lru64 lru128 lru160   (decode tokens)")
for l in moe_layers:
    s = summary["per_layer"].get(l)
    if not s: continue
    st, lr = s["static"], s["lru"]
    flag = "*" if l in off else " "
    print(f"{l:3d}{flag}  {st['top25']:.3f}  {st['top50']:.3f}  {st['top62']:.3f} | {lr[64]:.3f}  {lr[128]:.3f}  {lr[160]:.3f}")
json.dump(summary, open(os.path.join(d, "summary.json"), "w"), indent=1)
