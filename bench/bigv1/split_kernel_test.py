#!/usr/bin/env python3
"""split_kernel_test.py — Step 1 go/no-go for the hot/cold expert cache.

Question: can trtllm_fp4_block_scale_moe compute a 256-expert MoE layer as TWO window launches
(experts [0,H) from an HBM tensor, experts [H,256) from a pinned-host UVA tensor) with
do_finalize=False + our own fp32 combine, and (a) match the single-launch result, (b) not read
the out-of-window expert rows, (c) be faster when the hot window actually has most of the hits?

Synthetic GLM-5.3-shaped layer: E=256, K=8, hidden=6144, inter=2048, NVFP4 weights + e4m3 block
scales, random. Routing: top-8 from random logits, optionally skewed toward [0,H).
No model needed. Uses the vLLM-in-image FlashInfer build.
"""
import os, sys, time, json, torch
import flashinfer
from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

torch.manual_seed(0)
dev = "cuda"
E, K, HID, INTER = 256, 8, 6144, 2048
H = int(os.environ.get("HOT", "128"))
M = int(os.environ.get("M", "1"))               # tokens per step (decode = 1)
SKEW = float(os.environ.get("SKEW", "0.75"))     # fraction of routings that land in [0,H)
ITERS = int(os.environ.get("ITERS", "50"))
out = {"H": H, "M": M, "skew": SKEW}

# ---- weights (NVFP4 packed: 2 fp4 per byte -> uint8 [E, N, K/2]; scales e4m3 [E, N, K/16]) ----
def nvfp4_weights(n, k):
    w = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=dev)
    s = (torch.rand(E, n, k // 16, device=dev) * 0.5 + 0.5).to(torch.float8_e4m3fn)
    return w, s
w1, w1s = nvfp4_weights(2 * INTER, HID)   # gemm1: [E, 2*inter, hid/2]
w2, w2s = nvfp4_weights(HID, INTER)       # gemm2: [E, hid, inter/2]
out["w1_bytes_per_expert"] = w1[0].numel() + w1s[0].numel()
out["w2_bytes_per_expert"] = w2[0].numel() + w2s[0].numel()

# ---- host (pinned UVA) copies of the cold window, exactly like vLLM's offloader does it ----
def to_uva(t):
    c = t.cpu().pin_memory()
    return get_accelerator_view_from_cpu_tensor(c), c
w1_uva, _k1 = to_uva(w1); w1s_uva, _k2 = to_uva(w1s)
w2_uva, _k3 = to_uva(w2); w2s_uva, _k4 = to_uva(w2s)

# ---- activations: pre-quantized nvfp4 hidden states + scales (kernel takes fp4 input) ----
x_bf16 = torch.randn(M, HID, device=dev, dtype=torch.bfloat16)
from flashinfer import SfLayout, nvfp4_quantize
_q = nvfp4_quantize(x_bf16, torch.tensor([1.0], device=dev, dtype=torch.float32),
                    sfLayout=SfLayout.layout_linear, per_token_activation=False)
x_fp4, x_sf = _q[0], _q[1]
x_fp4 = x_fp4.view(torch.uint8).reshape(M, HID // 2)
x_sf = x_sf.view(torch.float8_e4m3fn).reshape(M, HID // 16)
out["act_shapes"] = [tuple(x_fp4.shape), tuple(x_sf.shape)]

# ---- routing logits, skewed toward the hot window ----
logits = torch.randn(M, E, device=dev, dtype=torch.bfloat16)
if SKEW > 0:
    # binary-search the boost so measured hot share ~= SKEW
    lo, hi = 0.0, 6.0
    base = logits.clone()
    for _ in range(30):
        mid = (lo + hi) / 2; l2 = base.clone(); l2[:, :H] += mid
        share = float((torch.topk(l2.float(), K, dim=-1).indices < H).float().mean())
        if share < SKEW: lo = mid
        else: hi = mid
    logits = base; logits[:, :H] += (lo + hi) / 2
    # sanity: measured hit share
    topk = torch.topk(logits.float(), K, dim=-1).indices
    out["measured_hot_share"] = round(float((topk < H).float().mean()), 3)

g1_alpha = torch.ones(E, device=dev, dtype=torch.float32)
g1_scale = torch.ones(E, device=dev, dtype=torch.float32)
g2_alpha = torch.ones(E, device=dev, dtype=torch.float32)

def launch(w1_, w1s_, w2_, w2s_, off, n_local, finalize):
    return trtllm_fp4_block_scale_moe(
        routing_logits=logits, routing_bias=None,
        hidden_states=x_fp4, hidden_states_scale=x_sf,
        gemm1_weights=w1_, gemm1_weights_scale=w1s_, gemm1_bias=None,
        gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
        gemm2_weights=w2_, gemm2_weights_scale=w2s_, gemm2_bias=None,
        output1_scale_scalar=g1_scale, output1_scale_gate_scalar=g1_alpha, output2_scale_scalar=g2_alpha,
        num_experts=E, top_k=K, n_group=1, topk_group=1, intermediate_size=INTER,
        local_expert_offset=off, local_num_experts=n_local, routed_scaling_factor=1.0,
        routing_method_type=1, do_finalize=finalize, tune_max_num_tokens=M,
    )

# ---- A: reference single launch, everything from HBM ----
ref = launch(w1, w1s, w2, w2s, 0, E, True)
ref = ref[0] if isinstance(ref, (list, tuple)) else ref
torch.cuda.synchronize()

# ---- B: two windows, both from HBM, manual combine (numerics of the split itself) ----
def split_combine(w1h, w1sh, w2h, w2sh, w1c, w1sc, w2c, w2sc):
    a = launch(w1h, w1sh, w2h, w2sh, 0, H, False)
    b = launch(w1c, w1sc, w2c, w2sc, H, E - H, False)
    # each returns [gemm2_out (M*K_local?, hid), expert_weights, expanded_idx_to_permuted_idx]
    def fin(res, acc):
        g2, ew, e2p = res
        idx = e2p.view(-1).long(); wts = ew.view(-1).float()
        valid = idx >= 0
        rows = torch.where(valid, idx, torch.zeros_like(idx))
        tok = torch.arange(M, device=dev).repeat_interleave(K)
        contrib = g2[rows].float() * (wts * valid.float())[:, None]
        acc.index_add_(0, tok, contrib)
        return acc
    acc = torch.zeros(M, HID, device=dev, dtype=torch.float32)
    fin(a, acc); fin(b, acc)
    return acc.to(torch.bfloat16), a, b

split_hbm, a_res, b_res = split_combine(w1, w1s, w2, w2s, w1, w1s, w2, w2s)
torch.cuda.synchronize()
out["split_return_shapes"] = [tuple(t.shape) for t in a_res]
out["maxabs_ref"] = float(ref.float().abs().max())
out["maxabs_diff_split_hbm"] = float((split_hbm.float() - ref.float()).abs().max())
out["rel_diff_split_hbm"] = out["maxabs_diff_split_hbm"] / max(out["maxabs_ref"], 1e-6)

# ---- C: hot window from HBM, cold window from pinned UVA (the real design) ----
split_mixed, _, _ = split_combine(w1, w1s, w2, w2s, w1_uva, w1s_uva, w2_uva, w2s_uva)
torch.cuda.synchronize()
out["maxabs_diff_split_mixed_vs_split_hbm"] = float((split_mixed.float() - split_hbm.float()).abs().max())

# ---- D: timing. today's path = single launch, ALL weights from UVA. design = hot HBM + cold UVA ----
def timeit(fn):
    for _ in range(5): fn()
    torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(ITERS): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / ITERS * 1e3
out["ms_single_all_hbm"] = round(timeit(lambda: launch(w1, w1s, w2, w2s, 0, E, True)), 3)
out["ms_single_all_uva_TODAY"] = round(timeit(lambda: launch(w1_uva, w1s_uva, w2_uva, w2s_uva, 0, E, True)), 3)
out["ms_split_hot_hbm_cold_uva_DESIGN"] = round(timeit(lambda: split_combine(w1, w1s, w2, w2s, w1_uva, w1s_uva, w2_uva, w2s_uva)), 3)
out["ms_split_both_hbm"] = round(timeit(lambda: split_combine(w1, w1s, w2, w2s, w1, w1s, w2, w2s)), 3)
# does the cold launch read rows it doesn't own?  time the cold window alone with hot-skewed routing:
# if it only touches its ~25% of hits it should be ~4x faster than the all-UVA single launch
out["ms_cold_window_only_uva"] = round(timeit(lambda: launch(w1_uva, w1s_uva, w2_uva, w2s_uva, H, E - H, False)), 3)
out["ms_hot_window_only_hbm"] = round(timeit(lambda: launch(w1, w1s, w2, w2s, 0, H, False)), 3)
_a = launch(w1, w1s, w2, w2s, 0, H, False); _b = launch(w1_uva, w1s_uva, w2_uva, w2s_uva, H, E - H, False)
def _comb():
    acc = torch.zeros(M, HID, device=dev, dtype=torch.float32)
    for res in (_a, _b):
        g2, ew, e2p = res; idx = e2p.view(-1).long(); wts = ew.view(-1).float(); valid = idx >= 0
        rows = torch.where(valid, idx, torch.zeros_like(idx)); tok = torch.arange(M, device=dev).repeat_interleave(K)
        acc.index_add_(0, tok, g2[rows].float() * (wts * valid.float())[:, None])
    return acc
out["ms_combine_only"] = round(timeit(_comb), 3)
out["ms_kernels_only_design"] = round(out["ms_hot_window_only_hbm"] + out["ms_cold_window_only_uva"], 3)
out["speedup_design_vs_today"] = round(out["ms_single_all_uva_TODAY"] / out["ms_split_hot_hbm_cold_uva_DESIGN"], 2)
print("SPLIT_KERNEL_RESULT " + json.dumps(out))
