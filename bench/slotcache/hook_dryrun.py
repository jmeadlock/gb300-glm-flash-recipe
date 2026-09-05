#!/usr/bin/env python3
"""hook_dryrun.py — exercise slot_cache_hook against the REAL vLLM TrtLlmNvFp4ExpertsModular class with synthetic
GLM-shaped weights, no model load. Verifies: (1) the hook installs, (2) _invoke_kernel on a cached layer matches
the un-hooked path bit-for-bit at M=1 over churn, (3) bypass for M>threshold works, (4) graph capture works.
"""
import os, sys, time, json, torch
os.environ.setdefault("SLOT_CACHE", "128")
sys.path.insert(0, "/w")
import slot_cache_hook as H
H.install()
# importing these AFTER install() so the meta-path finder patches them
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import TrtLlmNvFp4ExpertsModular, TrtLlmNvFp4ExpertsBase
from vllm.model_executor.layers.fused_moe.oracle import nvfp4 as oracle
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig, FusedMoEConfig, FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.utils import trtllm_moe_pack_topk_ids_weights
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
import flashinfer
from flashinfer import SfLayout, nvfp4_quantize

out = {}
out["forced_modular"] = oracle.backend_to_kernel_cls(oracle.NvFp4MoeBackend.FLASHINFER_TRTLLM) == [TrtLlmNvFp4ExpertsModular]
dev = "cuda"; E, K, HID, INTER = 256, 8, 6144, 2048; S = int(os.environ["SLOT_CACHE"])
torch.manual_seed(0)

# ---- synthetic kernel-format weights, UVA-offloaded like the real offloader does ----
def nv(n, k):
    w = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=dev)
    s = (torch.rand(E, n, k // 16, device=dev) * 0.5 + 0.5).to(torch.float8_e4m3fn)
    return w, s
w13, w13s = nv(2 * INTER, HID); w2, w2s = nv(HID, INTER)
def uva(t):
    c = t.cpu().pin_memory(); v = get_accelerator_view_from_cpu_tensor(c); return v, c
w13_u, k1 = uva(w13); w2_u, k2 = uva(w2)
g1_alphas = torch.rand(E, device=dev) + 0.5; g2_alphas = torch.rand(E, device=dev) + 0.5; a2_gscale = torch.rand(E, device=dev) + 0.5

# ---- build a minimal Modular experts object without the full vLLM layer machinery ----
class FakeLayer(torch.nn.Module):
    pass
layer = FakeLayer(); layer.layer_name = "fake.moe"
layer.w13_weight = torch.nn.Parameter(w13_u, requires_grad=False); layer.w13_weight._vllm_is_uva_offloaded = True
layer.w2_weight = torch.nn.Parameter(w2_u, requires_grad=False); layer.w2_weight._vllm_is_uva_offloaded = True
layer.w13_weight_scale_2 = torch.nn.Parameter(torch.ones(E, device=dev), requires_grad=False)
layer.w2_weight_scale_2 = torch.nn.Parameter(torch.ones(E, device=dev), requires_grad=False)
layer.w13_input_scale = torch.nn.Parameter(torch.ones(E, device=dev), requires_grad=False)
layer.w2_input_scale = torch.nn.Parameter(torch.ones(E, device=dev), requires_grad=False)
layer._slot_cache_pending = True

# Construct the experts object by bypassing __init__ (its real ctor needs FusedMoEConfig plumbing); set the fields
# _invoke_kernel and process_weights_after_loading touch.
ex = TrtLlmNvFp4ExpertsModular.__new__(TrtLlmNvFp4ExpertsModular)
layer.w13_weight_scale = torch.nn.Parameter(w13s, requires_grad=False); layer.w2_weight_scale = torch.nn.Parameter(w2s, requires_grad=False)
class QC:  # duck-typed quant_config; scales resolve to the layer Parameters' current data (as vLLM's property does)
    g1_alphas = g1_alphas; g2_alphas = g2_alphas; a2_gscale = a2_gscale
    @property
    def w1_scale(self): return layer.w13_weight_scale.data
    @property
    def w2_scale(self): return layer.w2_weight_scale.data
class MC:
    is_act_and_mul = True; max_num_tokens = 8192; dp_size = 1; use_deferred_moe_finalize = False
ex.quant_config = QC(); ex.moe_config = MC(); ex.per_token_activation = False
ex.topk = K; ex.intermediate_size_per_partition = INTER; ex.hidden_dim = HID
ex.local_num_experts = E; ex.ep_rank = 0
ex.gemm1_alpha = None; ex.gemm1_beta = None; ex.gemm1_clamp_limit = None; ex.is_situ = False
ex.g1_scale_c = g1_alphas * a2_gscale
class MoeCfgStub: pass
ex._get_chunk_size = lambda: 8192

# pwal builds the cache (patched Base.process_weights_after_loading)
try:
    TrtLlmNvFp4ExpertsBase.process_weights_after_loading(ex, layer)
except Exception as e:
    out["pwal_error"] = repr(e)[:200]
out["cache_registered"] = layer.w13_weight.data.data_ptr() in H._registry
lc = H._registry.get(layer.w13_weight.data.data_ptr())
if lc is not None:
    out["slot_gb"] = round(sum(t.numel() * t.element_size() for t in lc.slots.values()) / 1e9, 3)

# ---- reference: un-hooked kernel over the full UVA tensors ----
x = torch.randn(1, HID, device=dev, dtype=torch.bfloat16)
q = nvfp4_quantize(x, torch.tensor([1.0], device=dev), sfLayout=SfLayout.layout_linear, per_token_activation=False)
xq = q[0].view(torch.uint8).reshape(1, HID // 2); xs = q[1].view(torch.float8_e4m3fn).reshape(1, HID // 16).view(torch.uint8)
from vllm.model_executor.layers.fused_moe.config import MoEActivation
act = MoEActivation.SILU if hasattr(MoEActivation, "SILU") else list(MoEActivation)[0]
def ref_call(ids, wts):
    o = torch.empty(1, HID, device=dev, dtype=torch.bfloat16)
    # call the ORIGINAL body: temporarily unregister
    saved = H._registry.pop(layer.w13_weight.data.data_ptr())
    try:
        TrtLlmNvFp4ExpertsModular._invoke_kernel(ex, o, xq, layer.w13_weight.data, layer.w2_weight.data, wts, ids, act, E, xs)
    finally:
        H._registry[layer.w13_weight.data.data_ptr()] = saved
    return o
def hook_call(ids, wts):
    o = torch.empty(1, HID, device=dev, dtype=torch.bfloat16)
    TrtLlmNvFp4ExpertsModular._invoke_kernel(ex, o, xq, layer.w13_weight.data, layer.w2_weight.data, wts, ids, act, E, xs)
    return o

recent = list(range(K)); LOC = 0.75
def route():
    row = set()
    while len(row) < K:
        e = recent[int(torch.randint(0, len(recent), (1,)))] if torch.rand(1).item() < LOC else int(torch.randint(0, E, (1,)))
        row.add(e)
    ids = sorted(row); recent[:] = (recent + ids)[-64:]
    return torch.tensor([ids], dtype=torch.int32, device=dev), torch.softmax(torch.randn(1, K, device=dev), -1).to(torch.bfloat16)

maxd = 0.0; STEPS = 200
try:
    for st in range(STEPS):
        ids, wts = route()
        a = hook_call(ids, wts); b = ref_call(ids, wts); torch.cuda.synchronize()
        maxd = max(maxd, float((a.float() - b.float()).abs().max()))
    out["maxabs_diff_hook_vs_ref"] = maxd; out["steps"] = STEPS
    lc = H._registry.get(layer.w13_weight.data.data_ptr()); out["cache_registered"] = lc is not None
    out["scale_is_uva_view"] = bool(lc is not None and len(lc._keep_host) == 2)
    out["misses"] = int(lc.misses); out["hit_rate"] = round(1 - int(lc.misses) / (STEPS * K), 3)
except Exception as e:
    import traceback; out["forward_error"] = traceback.format_exc()[-1500:]

# ---- bypass check: M=32 tokens must take the original path (no cache mutation) ----
try:
    m0 = int(lc.misses)
    ids32 = torch.randint(0, E, (32, K), dtype=torch.int32, device=dev); w32 = torch.softmax(torch.randn(32, K, device=dev), -1).to(torch.bfloat16)
    x32 = torch.randn(32, HID, device=dev, dtype=torch.bfloat16)
    q32 = nvfp4_quantize(x32, torch.tensor([1.0], device=dev), sfLayout=SfLayout.layout_linear, per_token_activation=False)
    o32 = torch.empty(32, HID, device=dev, dtype=torch.bfloat16)
    TrtLlmNvFp4ExpertsModular._invoke_kernel(ex, o32, q32[0].view(torch.uint8).reshape(32, HID // 2), layer.w13_weight.data, layer.w2_weight.data, w32, ids32, act, E, q32[1].view(torch.uint8).reshape(32, -1))
    torch.cuda.synchronize(); out["bypass_ok"] = (int(lc.misses) == m0)
except Exception as e:
    out["bypass_error"] = repr(e)[:300]

# ---- graph capture of the hooked M=1 path ----
try:
    ids_b, wts_b = route(); o = torch.empty(1, HID, device=dev, dtype=torch.bfloat16)
    def step():
        TrtLlmNvFp4ExpertsModular._invoke_kernel(ex, o, xq, layer.w13_weight.data, layer.w2_weight.data, wts_b, ids_b, act, E, xs)
    for _ in range(3): step()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): step()
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); ev0.record()
    for _ in range(100): g.replay()
    ev1.record(); torch.cuda.synchronize(); out["graph_ms_all_hits"] = round(ev0.elapsed_time(ev1) / 100, 4)
    # churn under graph replay: change ids_b in place between replays, compare to ref
    md = 0.0
    for st in range(100):
        i2, w2_ = route(); ids_b.copy_(i2); wts_b.copy_(w2_); g.replay(); torch.cuda.synchronize()
        r = ref_call(ids_b, wts_b); md = max(md, float((o.float() - r.float()).abs().max()))
    out["graph_churn_maxabs_diff"] = md
except Exception as e:
    import traceback; out["graph_error"] = traceback.format_exc()[-800:]
print("HOOK_DRYRUN " + json.dumps(out))
