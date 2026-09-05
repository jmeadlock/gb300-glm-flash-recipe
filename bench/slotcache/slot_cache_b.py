#!/usr/bin/env python3
"""slot_cache_b.py — Step B: device-side, fixed-shape, CUDA-graph-capturable slot cache for one GLM-5.3 MoE layer.

Everything from step A moved onto the GPU:
  - expert->slot map, slot->expert map, per-slot last-used clock: device tensors
  - miss detection, LRU victim pick (topk over last-used), map updates: fixed-shape torch ops with a sink index
  - row copies host->slot: a Triton kernel that reads a mask and only moves bytes for misses
  - ONE trtllm_fp4_block_scale_routed_moe launch over the S-slot tensors
The whole per-layer step is captured in a CUDA graph and replayed. Correctness vs an eager all-HBM reference on
every step. ALSO re-measures yesterday's eager numbers under graph capture (all-HBM, all-UVA, hot window, cold
window with 0 and 8 hits, two-window+combine) because launch overhead was suspected to dominate those.
Env: SLOTS (128), M (1), STEPS (300), LOCALITY (0.75).
"""
import os, time, json, torch, triton, triton.language as tl
from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe, trtllm_fp4_block_scale_moe
from flashinfer import SfLayout, nvfp4_quantize
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
from vllm.model_executor.layers.fused_moe.utils import trtllm_moe_pack_topk_ids_weights

torch.manual_seed(0)
dev = "cuda"
E, K, HID, INTER = 256, 8, 6144, 2048
S = int(os.environ.get("SLOTS", "128")); M = int(os.environ.get("M", "1"))
STEPS = int(os.environ.get("STEPS", "300")); LOCALITY = float(os.environ.get("LOCALITY", "0.75"))
assert M == 1, "step B handles decode M=1; M>1 needs per-step expert dedupe (step D)"
out = {"S": S, "M": M, "steps": STEPS, "locality": LOCALITY}
BIG = 1 << 60

# ---------------- weights: full set in HBM (reference) + pinned host (backing) + S slots (HBM) ----------------
def nvfp4_weights(n, k):
    w = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=dev)
    s = (torch.rand(E, n, k // 16, device=dev) * 0.5 + 0.5).to(torch.float8_e4m3fn)
    return w, s
w1, w1s = nvfp4_weights(2 * INTER, HID); w2, w2s = nvfp4_weights(HID, INTER)
g1_scale = torch.rand(E, device=dev) + 0.5; g1_alpha = torch.rand(E, device=dev) + 0.5; g2_alpha = torch.rand(E, device=dev) + 0.5
def to_uva(t):
    c = t.cpu().pin_memory(); return get_accelerator_view_from_cpu_tensor(c), c
w1_h, _a = to_uva(w1); w1s_h, _b = to_uva(w1s); w2_h, _c = to_uva(w2); w2s_h, _d = to_uva(w2s)
slot = lambda t: torch.empty((S,) + tuple(t.shape[1:]), dtype=t.dtype, device=dev)
w1_slot, w1s_slot, w2_slot, w2s_slot = slot(w1), slot(w1s), slot(w2), slot(w2s)
g1_scale_slot = torch.zeros(S, device=dev); g1_alpha_slot = torch.zeros(S, device=dev); g2_alpha_slot = torch.zeros(S, device=dev)
# per-expert scalars stacked so the copy kernel can move them as one [E,3] fp32 row
scal = torch.stack([g1_scale, g1_alpha, g2_alpha], 1).contiguous()          # [E,3]
scal_slot = torch.zeros(S, 3, device=dev)
row_bytes = sum(t[0].numel() * t[0].element_size() for t in (w1, w1s, w2, w2s))
out["row_MB"] = round(row_bytes / 1e6, 1)

# ---------------- Triton masked row copy (int64 elements) ----------------
@triton.jit
def masked_row_copy(src_ptr, dst_ptr, src_idx_ptr, dst_idx_ptr, mask_ptr, row_elems, BLOCK: tl.constexpr):
    k = tl.program_id(0); blk = tl.program_id(1)
    m = tl.load(mask_ptr + k)
    s = tl.load(src_idx_ptr + k).to(tl.int64); d = tl.load(dst_idx_ptr + k).to(tl.int64)
    off = blk * BLOCK + tl.arange(0, BLOCK)
    valid = (off < row_elems) & (m != 0)
    v = tl.load(src_ptr + s * row_elems + off, mask=valid, other=0)
    tl.store(dst_ptr + d * row_elems + off, v, mask=valid)

def as_rows64(t):  # [E or S, ...] -> [rows, elems] int64 view
    n = t.shape[0]; return t.view(torch.int64).reshape(n, -1) if t.element_size() == 1 else t.view(torch.int64).reshape(n, -1)
pairs = [(as_rows64(w1_h), as_rows64(w1_slot)), (as_rows64(w1s_h), as_rows64(w1s_slot)),
         (as_rows64(w2_h), as_rows64(w2_slot)), (as_rows64(w2s_h), as_rows64(w2s_slot))]
scal64_h = scal.view(torch.int64).reshape(E, -1) if False else None  # scalars copied via gather below (tiny)
BLOCK = 2048
def copy_misses(src_e, dst_s, miss_u8):
    for src, dst in pairs:
        n = src.shape[1]
        masked_row_copy[(K, triton.cdiv(n, BLOCK))](src, dst, src_e, dst_s, miss_u8, n, BLOCK=BLOCK)
    # scalars: [K,3] gather then masked scatter into slots (hits scatter into sink row S)
    vals = scal[src_e.long()]                                    # src_e is E (sink) for hits -> scal_pad row
    scal_slot_pad.index_put_((dst_s.long(),), vals)              # dst_s = S (sink) for hits

scal = torch.cat([scal, torch.zeros(1, 3, device=dev)], 0)       # row E = sink
scal_slot_pad = torch.zeros(S + 1, 3, device=dev)                # row S = sink
scal_slot = scal_slot_pad[:S]

# ---------------- cache state (device) ----------------
e2s = torch.full((E + 1,), -1, dtype=torch.int32, device=dev)   # index E = sink
s2e = torch.full((S + 1,), E, dtype=torch.int32, device=dev)    # index S = sink; empty slots point at sink expert E
last_used = torch.full((S,), -1, dtype=torch.int64, device=dev)
step_t = torch.zeros((), dtype=torch.int64, device=dev)
protect = torch.zeros(S + 1, dtype=torch.bool, device=dev)
# static I/O buffers for the graph
ids_buf = torch.zeros(K, dtype=torch.int32, device=dev)          # M=1 -> K ids
wts_buf = torch.zeros(1, K, dtype=torch.bfloat16, device=dev)
x_fp4_buf = torch.zeros(1, HID // 2, dtype=torch.uint8, device=dev)
x_sf_buf = torch.zeros(1, HID // 16, dtype=torch.float8_e4m3fn, device=dev)
out_buf = torch.zeros(1, HID, dtype=torch.bfloat16, device=dev)
stat_miss = torch.zeros((), dtype=torch.int64, device=dev)

def launch(w1_, w1s_, w2_, w2s_, g1s, g1a, g2a, n_exp, ids, wts, x_fp4, x_sf, output=None):
    packed = trtllm_moe_pack_topk_ids_weights(ids.view(1, K), wts)
    return trtllm_fp4_block_scale_routed_moe(
        topk_ids=packed, routing_bias=None, hidden_states=x_fp4, hidden_states_scale=x_sf,
        gemm1_weights=w1_, gemm1_weights_scale=w1s_, gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
        gemm2_weights=w2_, gemm2_weights_scale=w2s_, gemm2_bias=None,
        output1_scale_scalar=g1s, output1_scale_gate_scalar=g1a, output2_scale_scalar=g2a,
        num_experts=n_exp, top_k=K, n_group=0, topk_group=0, intermediate_size=INTER,
        local_expert_offset=0, local_num_experts=n_exp, routed_scaling_factor=None,
        routing_method_type=1, do_finalize=True, tune_max_num_tokens=1, output=output)

def cache_step():
    """One layer's decode step, fixed shapes only. Reads ids_buf/wts_buf/x_*; writes out_buf."""
    sl = e2s[ids_buf.long()]                                   # [K] slot or -1
    miss = sl < 0
    hit_slots = torch.where(miss, torch.full_like(sl, S), sl)
    protect.zero_(); protect.index_fill_(0, hit_slots.long(), True)
    score = last_used.masked_fill(protect[:S], BIG)
    victims = torch.topk(score, K, largest=False).indices.to(torch.int32)   # K LRU slots, none hit this step
    rank = (torch.cumsum(miss.to(torch.int32), 0) - 1).clamp_(min=0)
    dst = torch.where(miss, victims[rank.long()], torch.full_like(sl, S))   # sink S for hits
    old_e = s2e[dst.long()]                                     # experts being evicted (sink E for hits/empty)
    e2s.index_put_((old_e.long(),), torch.full_like(old_e, -1))
    src_e = torch.where(miss, ids_buf, torch.full_like(ids_buf, E))
    e2s.index_put_((src_e.long(),), dst)
    s2e.index_put_((dst.long(),), src_e)
    copy_misses(src_e, dst, miss.to(torch.uint8))
    sl_final = torch.where(miss, dst, sl)
    last_used.index_put_((sl_final.long(),), step_t.expand(K))
    step_t.add_(1); stat_miss.add_(miss.sum())
    launch(w1_slot, w1s_slot, w2_slot, w2s_slot, scal_slot[:, 0].contiguous(), scal_slot[:, 1].contiguous(), scal_slot[:, 2].contiguous(),
           S, sl_final, wts_buf, x_fp4_buf, x_sf_buf, output=out_buf)

# ---------------- routing stream with temporal locality ----------------
recent = list(range(K))
def route():
    row = set()
    g = torch.Generator().manual_seed(int(torch.randint(0, 1 << 30, (1,)).item()))
    while len(row) < K:
        if recent and torch.rand(1, generator=g).item() < LOCALITY: e = recent[int(torch.randint(0, len(recent), (1,), generator=g))]
        else: e = int(torch.randint(0, E, (1,), generator=g))
        row.add(e)
    ids = sorted(row); recent[:] = (recent + ids)[-64:]
    return torch.tensor(ids, dtype=torch.int32), torch.softmax(torch.randn(1, K), -1).to(torch.bfloat16)
def make_act():
    x = torch.randn(1, HID, device=dev, dtype=torch.bfloat16)
    q = nvfp4_quantize(x, torch.tensor([1.0], device=dev), sfLayout=SfLayout.layout_linear, per_token_activation=False)
    return q[0].view(torch.uint8).reshape(1, HID // 2), q[1].view(torch.float8_e4m3fn).reshape(1, HID // 16)

# ---------------- warm-up + capture ----------------
xf, xs = make_act(); x_fp4_buf.copy_(xf); x_sf_buf.copy_(xs)
for _ in range(3):
    i, w = route(); ids_buf.copy_(i); wts_buf.copy_(w); cache_step()
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    cache_step()
torch.cuda.synchronize()
# reset state after capture so the measured run starts cold
e2s.fill_(-1); s2e.fill_(E); last_used.fill_(-1); step_t.zero_(); stat_miss.zero_(); recent[:] = list(range(K))

# ---------------- run: correctness every step + timing ----------------
maxdiff = 0.0; ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True); t_ms = 0.0
for st in range(STEPS):
    i, w = route(); ids_buf.copy_(i); wts_buf.copy_(w)
    ev0.record(); g.replay(); ev1.record(); torch.cuda.synchronize(); t_ms += ev0.elapsed_time(ev1)
    ref = launch(w1, w1s, w2, w2s, g1_scale, g1_alpha, g2_alpha, E, ids_buf, wts_buf, x_fp4_buf, x_sf_buf)
    ref = ref[0] if isinstance(ref, (list, tuple)) else ref
    d = float((out_buf.float() - ref.float()).abs().max()); maxdiff = max(maxdiff, d)
    if st == 0: out["maxabs_ref"] = float(ref.float().abs().max())
    # invariant check every 50 steps: every resident expert maps back to its slot
    if st % 50 == 49:
        s2e_c = s2e[:S].tolist(); e2s_c = e2s[:E].tolist()
        for s_, e_ in enumerate(s2e_c):
            if e_ != E: assert e2s_c[e_] == s_, f"map inconsistent at step {st}: slot {s_} expert {e_} e2s={e2s_c[e_]}"
misses = int(stat_miss); hits = STEPS * K - misses
out.update({"maxabs_diff_cache_vs_ref": maxdiff, "hits": hits, "misses": misses, "hit_rate": round(hits / (STEPS * K), 3),
            "ms_per_step_graph_incl_misses": round(t_ms / STEPS, 4)})
# steady state (no misses): replay with the same ids repeatedly
for _ in range(5): g.replay()
torch.cuda.synchronize(); ev0.record()
for _ in range(100): g.replay()
ev1.record(); torch.cuda.synchronize(); out["ms_per_step_graph_all_hits"] = round(ev0.elapsed_time(ev1) / 100, 4)

# ---------------- re-measure yesterday's eager numbers under graph capture ----------------
def graph_time(fn, iters=100):
    for _ in range(3): fn()
    torch.cuda.synchronize(); gg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gg): fn()
    torch.cuda.synchronize(); gg.replay(); torch.cuda.synchronize()
    ev0.record()
    for _ in range(iters): gg.replay()
    ev1.record(); torch.cuda.synchronize(); return round(ev0.elapsed_time(ev1) / iters, 4)
out["G_ms_single_all_hbm"] = graph_time(lambda: launch(w1, w1s, w2, w2s, g1_scale, g1_alpha, g2_alpha, E, ids_buf, wts_buf, x_fp4_buf, x_sf_buf))
out["G_ms_single_all_uva_TODAY"] = graph_time(lambda: launch(w1_h, w1s_h, w2_h, w2s_h, g1_scale, g1_alpha, g2_alpha, E, ids_buf, wts_buf, x_fp4_buf, x_sf_buf))
# window launches via the routing-inside kernel (needs logits): hot window [0,H) HBM, cold window [H,E) UVA
H = S
logits_buf = torch.zeros(1, E, device=dev, dtype=torch.bfloat16)
def wlaunch(w1_, w1s_, w2_, w2s_, off, n_local, fin=False):
    return trtllm_fp4_block_scale_moe(routing_logits=logits_buf, routing_bias=None, hidden_states=x_fp4_buf, hidden_states_scale=x_sf_buf,
        gemm1_weights=w1_, gemm1_weights_scale=w1s_, gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
        gemm2_weights=w2_, gemm2_weights_scale=w2s_, gemm2_bias=None, output1_scale_scalar=g1_scale, output1_scale_gate_scalar=g1_alpha,
        output2_scale_scalar=g2_alpha, num_experts=E, top_k=K, n_group=1, topk_group=1, intermediate_size=INTER,
        local_expert_offset=off, local_num_experts=n_local, routed_scaling_factor=1.0, routing_method_type=1, do_finalize=fin, tune_max_num_tokens=1)
def set_logits(all_hot):
    l = torch.randn(1, E, device=dev, dtype=torch.bfloat16)
    if all_hot: l[:, :H] += 8.0
    else: l[:, H:] += 8.0
    logits_buf.copy_(l)
set_logits(all_hot=True)
out["G_ms_cold_window_uva_0_hits"] = graph_time(lambda: wlaunch(w1_h, w1s_h, w2_h, w2s_h, H, E - H))
out["G_ms_hot_window_hbm_8_hits"] = graph_time(lambda: wlaunch(w1, w1s, w2, w2s, 0, H))
set_logits(all_hot=False)
out["G_ms_cold_window_uva_8_hits"] = graph_time(lambda: wlaunch(w1_h, w1s_h, w2_h, w2s_h, H, E - H))
out["G_ms_hot_window_hbm_0_hits"] = graph_time(lambda: wlaunch(w1, w1s, w2, w2s, 0, H))
# two-window + combine (the design killed yesterday) under graph, at ~75% hot
l = torch.randn(1, E, device=dev, dtype=torch.bfloat16); l[:, :H] += 1.2; logits_buf.copy_(l)
out["two_window_hot_share"] = round(float((torch.topk(logits_buf.float(), K, -1).indices < H).float().mean()), 3)
def two_window():
    a = wlaunch(w1, w1s, w2, w2s, 0, H); b = wlaunch(w1_h, w1s_h, w2_h, w2s_h, H, E - H)
    acc = torch.zeros(1, HID, device=dev, dtype=torch.float32)
    for res in (a, b):
        g2, ew, e2p = res; idx = e2p.view(-1).long(); wt = ew.view(-1).float(); valid = idx >= 0
        rows = torch.where(valid, idx, torch.zeros_like(idx)); tok = torch.zeros(K, dtype=torch.long, device=dev)
        acc.index_add_(0, tok, g2[rows].float() * (wt * valid.float())[:, None])
    return acc
out["G_ms_two_window_plus_combine_75hot"] = graph_time(two_window)
print("SLOT_CACHE_B_RESULT " + json.dumps(out))
