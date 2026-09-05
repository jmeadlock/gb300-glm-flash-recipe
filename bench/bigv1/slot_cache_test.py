#!/usr/bin/env python3
"""slot_cache_test.py — Step A of the hot-expert cache build (design v2 §5).

One MoE layer, GLM-5.3 shape (E=256, K=8, hid=6144, inter=2048, NVFP4). All 256 experts live in pinned host
memory (today's UVA path). S HBM slots hold resident experts. Per step:
  1. router -> topk_ids (global expert ids)
  2. misses = experts with no slot; evict LRU victims; cudaMemcpyAsync rows host->slot on a side stream
  3. remap topk_ids -> slot ids
  4. ONE trtllm_fp4_block_scale_routed_moe launch over the S-slot tensors
Compared against the reference: ONE launch over the full 256-expert tensors in HBM (same routing).
Gates: max-abs diff == 0 across many steps with churn; per-step time vs today's all-UVA single launch.
Routing stream: synthetic token stream with tunable temporal locality so the cache actually churns.
"""
import os, time, json, torch
from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
from flashinfer import SfLayout, nvfp4_quantize
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
from vllm.model_executor.layers.fused_moe.utils import trtllm_moe_pack_topk_ids_weights

torch.manual_seed(0)
dev = "cuda"
E, K, HID, INTER = 256, 8, 6144, 2048
S = int(os.environ.get("SLOTS", "128"))
M = int(os.environ.get("M", "1"))
STEPS = int(os.environ.get("STEPS", "400"))
LOCALITY = float(os.environ.get("LOCALITY", "0.75"))  # prob a routing repeats a recently-used expert
PREFETCH = int(os.environ.get("PREFETCH", "0"))        # 1 = issue copies for step t+1 while step t computes
out = {"S": S, "M": M, "steps": STEPS, "locality": LOCALITY, "prefetch": PREFETCH}

# ---------- weights ----------
def nvfp4_weights(n, k):
    w = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=dev)
    s = (torch.rand(E, n, k // 16, device=dev) * 0.5 + 0.5).to(torch.float8_e4m3fn)
    return w, s
w1, w1s = nvfp4_weights(2 * INTER, HID); w2, w2s = nvfp4_weights(HID, INTER)
# per-expert scalars (slot-indexed in the cache)
g1_scale = torch.rand(E, device=dev) + 0.5; g1_alpha = torch.rand(E, device=dev) + 0.5; g2_alpha = torch.rand(E, device=dev) + 0.5

def to_uva(t):
    c = t.cpu().pin_memory(); return get_accelerator_view_from_cpu_tensor(c), c
w1_h, _a = to_uva(w1); w1s_h, _b = to_uva(w1s); w2_h, _c = to_uva(w2); w2s_h, _d = to_uva(w2s)
row_bytes = (w1[0].numel() + w1s[0].numel() + w2[0].numel() + w2s[0].numel())
out["row_MB"] = round(row_bytes / 1e6, 1)

# ---------- slot tensors (HBM) ----------
w1_slot = torch.empty((S,) + tuple(w1.shape[1:]), dtype=w1.dtype, device=dev)
w1s_slot = torch.empty((S,) + tuple(w1s.shape[1:]), dtype=w1s.dtype, device=dev)
w2_slot = torch.empty((S,) + tuple(w2.shape[1:]), dtype=w2.dtype, device=dev)
w2s_slot = torch.empty((S,) + tuple(w2s.shape[1:]), dtype=w2s.dtype, device=dev)
g1_scale_slot = torch.empty(S, device=dev); g1_alpha_slot = torch.empty(S, device=dev); g2_alpha_slot = torch.empty(S, device=dev)

# ---------- cache state (host-side python for now; step D moves it to device) ----------
e2s = torch.full((E,), -1, dtype=torch.int32, device=dev)   # expert -> slot
s2e = [-1] * S                                                 # slot -> expert (python)
lru = []                                                       # slot ids, LRU at front
copy_stream = torch.cuda.Stream()
stats = {"misses": 0, "hits": 0, "copies": 0}

def fetch(expert, slot):
    # copy row + scales + scalars from pinned host into the slot, on the copy stream
    with torch.cuda.stream(copy_stream):
        w1_slot[slot].copy_(w1_h[expert], non_blocking=True)
        w1s_slot[slot].copy_(w1s_h[expert], non_blocking=True)
        w2_slot[slot].copy_(w2_h[expert], non_blocking=True)
        w2s_slot[slot].copy_(w2s_h[expert], non_blocking=True)
        g1_scale_slot[slot] = g1_scale[expert]; g1_alpha_slot[slot] = g1_alpha[expert]; g2_alpha_slot[slot] = g2_alpha[expert]
    stats["copies"] += 1

def ensure_resident(expert_ids):
    """expert_ids: python list of unique global ids needed this step."""
    e2s_cpu = e2s.tolist()
    needed = set(expert_ids)
    for e in expert_ids:
        if e2s_cpu[e] >= 0:
            stats["hits"] += 1; s = e2s_cpu[e]
            lru.remove(s); lru.append(s)
            continue
        stats["misses"] += 1
        # victim: LRU slot not needed this step; or a free slot
        if len(lru) < S:
            free = [i for i in range(S) if s2e[i] < 0][0]; slot = free
        else:
            for cand in lru:
                if s2e[cand] not in needed: slot = cand; break
            lru.remove(slot); old = s2e[slot]; e2s_cpu[old] = -1
        s2e[slot] = e; e2s_cpu[e] = slot; lru.append(slot)
        fetch(e, slot)
    e2s.copy_(torch.tensor(e2s_cpu, dtype=torch.int32), non_blocking=True)

# ---------- activations + routing stream ----------
def make_act():
    x = torch.randn(M, HID, device=dev, dtype=torch.bfloat16)
    q = nvfp4_quantize(x, torch.tensor([1.0], device=dev, dtype=torch.float32), sfLayout=SfLayout.layout_linear, per_token_activation=False)
    return q[0].view(torch.uint8).reshape(M, HID // 2), q[1].view(torch.float8_e4m3fn).reshape(M, HID // 16)

recent = list(range(K))  # experts used recently, for locality
def route():
    ids = []
    for _ in range(M):
        row = set()
        while len(row) < K:
            if recent and torch.rand(1).item() < LOCALITY:
                e = recent[int(torch.randint(0, len(recent), (1,)).item())]
            else:
                e = int(torch.randint(0, E, (1,)).item())
            row.add(e)
        ids.append(sorted(row))
    flat = [e for r in ids for e in r]
    recent[:] = (recent + flat)[-64:]
    topk_ids = torch.tensor(ids, dtype=torch.int32, device=dev)
    topk_w = torch.softmax(torch.randn(M, K, device=dev), dim=-1).to(torch.bfloat16)
    return topk_ids, topk_w

def launch(w1_, w1s_, w2_, w2s_, g1s, g1a, g2a, n_exp, x_fp4, x_sf, ids, wts):
    packed = trtllm_moe_pack_topk_ids_weights(ids, wts)
    return trtllm_fp4_block_scale_routed_moe(
        topk_ids=packed, routing_bias=None, hidden_states=x_fp4, hidden_states_scale=x_sf,
        gemm1_weights=w1_, gemm1_weights_scale=w1s_, gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
        gemm2_weights=w2_, gemm2_weights_scale=w2s_, gemm2_bias=None,
        output1_scale_scalar=g1s, output1_scale_gate_scalar=g1a, output2_scale_scalar=g2a,
        num_experts=n_exp, top_k=K, n_group=0, topk_group=0, intermediate_size=INTER,
        local_expert_offset=0, local_num_experts=n_exp, routed_scaling_factor=None,
        routing_method_type=1, do_finalize=True, tune_max_num_tokens=M)

# ---------- correctness over STEPS with churn ----------
maxdiff = 0.0; t_cache = 0.0; t_ref_hbm = 0.0; t_today = 0.0
x_fp4, x_sf = make_act()
for step in range(STEPS):
    ids, wts = route()
    uniq = sorted(set(ids.view(-1).tolist()))
    # cache path
    torch.cuda.synchronize(); t0 = time.perf_counter()
    ensure_resident(uniq)
    torch.cuda.current_stream().wait_stream(copy_stream)
    slot_ids = e2s[ids.long()]
    assert int(slot_ids.min()) >= 0
    y_cache = launch(w1_slot, w1s_slot, w2_slot, w2s_slot, g1_scale_slot, g1_alpha_slot, g2_alpha_slot, S, x_fp4, x_sf, slot_ids.contiguous(), wts)
    torch.cuda.synchronize(); t_cache += time.perf_counter() - t0
    # reference: full tensors in HBM
    t0 = time.perf_counter()
    y_ref = launch(w1, w1s, w2, w2s, g1_scale, g1_alpha, g2_alpha, E, x_fp4, x_sf, ids, wts)
    torch.cuda.synchronize(); t_ref_hbm += time.perf_counter() - t0
    # today: full tensors via UVA
    t0 = time.perf_counter()
    y_today = launch(w1_h, w1s_h, w2_h, w2s_h, g1_scale, g1_alpha, g2_alpha, E, x_fp4, x_sf, ids, wts)
    torch.cuda.synchronize(); t_today += time.perf_counter() - t0
    y_cache = y_cache[0] if isinstance(y_cache, (list, tuple)) else y_cache
    y_ref = y_ref[0] if isinstance(y_ref, (list, tuple)) else y_ref
    d = float((y_cache.float() - y_ref.float()).abs().max()); maxdiff = max(maxdiff, d)
    if step == 0: out["maxabs_ref"] = float(y_ref.float().abs().max())

out.update({"maxabs_diff_cache_vs_ref": maxdiff, "hits": stats["hits"], "misses": stats["misses"],
            "hit_rate": round(stats["hits"] / max(1, stats["hits"] + stats["misses"]), 3),
            "ms_per_step_cache_incl_copies": round(t_cache / STEPS * 1e3, 3),
            "ms_per_step_ref_all_hbm": round(t_ref_hbm / STEPS * 1e3, 3),
            "ms_per_step_today_all_uva": round(t_today / STEPS * 1e3, 3)})
# steady-state kernel-only time of the cache path (no misses): re-run last step
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(50):
    launch(w1_slot, w1s_slot, w2_slot, w2s_slot, g1_scale_slot, g1_alpha_slot, g2_alpha_slot, S, x_fp4, x_sf, slot_ids.contiguous(), wts)
torch.cuda.synchronize(); out["ms_kernel_only_S_slots"] = round((time.perf_counter() - t0) / 50 * 1e3, 3)
out["python_overhead_note"] = "ensure_resident is host-side python; step D moves map+LRU to device"
print("SLOT_CACHE_RESULT " + json.dumps(out))
