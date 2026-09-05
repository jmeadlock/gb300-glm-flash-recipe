# slot_cache_hook.py — hot-expert slot cache for vLLM 0.28 TRT-LLM NVFP4 MoE under UVA offload.
# Loaded by sitecustomize.py when SLOT_CACHE=<slots> is set. Design v2 (research/hot-expert-cache-design-v2.md).
#
# Per offloaded MoE layer: S HBM slot rows for w13/w2 (+ block scales + per-expert scalars), the pinned-host UVA
# tensors as backing, expert->slot / slot->expert maps + LRU clock on device, a fused Triton bookkeeping kernel,
# a masked Triton row copy for misses, then ONE trtllm_fp4_block_scale_routed_moe launch over the slot tensors.
# Fixed shapes, no host sync -> CUDA-graph capturable. Bypass to plain UVA when M*K > S (prefill).
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

S_SLOTS = int(os.environ.get("SLOT_CACHE", "0"))
BYPASS_ABOVE = int(os.environ.get("SLOT_CACHE_BYPASS_TOKENS", "16"))   # M > this -> bypass (prefill)
_LOG = lambda m: sys.stderr.write(f"SLOT_CACHE {m}\n")
BIG = 1 << 62

_registry = {}      # id(w13_param.data_ptr) -> LayerCache
_stats = {"steps": 0}


def _install_triton():
    import triton, triton.language as tl

    @triton.jit
    def fused_bookkeeping(ids_ptr, e2s_ptr, s2e_ptr, last_ptr, step_ptr,
                          src_out_ptr, dst_out_ptr, mask_out_ptr, slot_out_ptr, miss_count_ptr,
                          N: tl.constexpr, S: tl.constexpr, E: tl.constexpr, SB: tl.constexpr):
        step = tl.load(step_ptr)
        soff = tl.arange(0, SB); smask = soff < S
        last = tl.load(last_ptr + soff, mask=smask, other=(1 << 62))
        for k in tl.static_range(N):
            e = tl.load(ids_ptr + k); sl = tl.load(e2s_ptr + e)
            last = tl.where((soff == sl) & (sl >= 0), (1 << 62), last)
        nmiss = 0
        for k in tl.static_range(N):
            e = tl.load(ids_ptr + k); sl = tl.load(e2s_ptr + e)
            is_miss = sl < 0
            vmin = tl.min(last, 0)
            victim = tl.min(tl.where(last == vmin, soff, SB), 0)
            dst = tl.where(is_miss, victim, S)
            old_e = tl.load(s2e_ptr + dst)
            if is_miss:
                tl.store(e2s_ptr + old_e, -1)
                tl.store(e2s_ptr + e, dst)
                tl.store(s2e_ptr + dst, e)
                last = tl.where(soff == dst, (1 << 62), last)
                nmiss += 1
            final = tl.where(is_miss, dst, sl)
            tl.store(src_out_ptr + k, tl.where(is_miss, e, E))
            tl.store(dst_out_ptr + k, dst)
            tl.store(mask_out_ptr + k, is_miss.to(tl.int8))
            tl.store(slot_out_ptr + k, final)
            tl.store(last_ptr + final, step, mask=final < S)
        tl.store(step_ptr, step + 1)
        tl.atomic_add(miss_count_ptr, nmiss)

    @triton.jit
    def masked_row_copy(src_ptr, dst_ptr, src_idx_ptr, dst_idx_ptr, mask_ptr, row_elems, BLOCK: tl.constexpr):
        k = tl.program_id(0); blk = tl.program_id(1)
        m = tl.load(mask_ptr + k)
        s = tl.load(src_idx_ptr + k).to(tl.int64); d = tl.load(dst_idx_ptr + k).to(tl.int64)
        off = blk * BLOCK + tl.arange(0, BLOCK)
        valid = (off < row_elems) & (m != 0)
        v = tl.load(src_ptr + s * row_elems + off, mask=valid, other=0)
        tl.store(dst_ptr + d * row_elems + off, v, mask=valid)

    return triton, fused_bookkeeping, masked_row_copy


class LayerCache:
    def __init__(self, name, w13, w2, w13_scale, w2_scale, scalars, S):
        # w13/w2: UVA device views of pinned host [E, ...]; scales: [E, ...] on device; scalars: list of [E] fp32
        self.name = name; self.S = S; self.E = w13.shape[0]
        dev = w13_scale.device if w13_scale.is_cuda else torch.device("cuda")
        self.dev = dev
        self.host = {"w13": w13, "w2": w2}
        self.slots = {"w13": torch.empty((S,) + tuple(w13.shape[1:]), dtype=w13.dtype, device=dev),
                      "w2": torch.empty((S,) + tuple(w2.shape[1:]), dtype=w2.dtype, device=dev)}
        # scales + scalars are small and resident: keep [E] versions and gather into [S] slot views on each swap
        # Resident [E] block scales -> pinned host (UVA view), same trick as the weights. Frees ~0.6 GB/layer of
        # HBM; the bypass (prefill) path reads them over C2C, which is cheap (scales are 1/16 of weight bytes).
        # Requires the scale objects to be the layer's Parameters so the rebind is visible to quant_config.
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
        self.res, self.res_slots, self._keep_host = {}, {}, []
        for k, v in (("w13_scale", w13_scale), ("w2_scale", w2_scale)):
            self.res_slots[k] = torch.empty((S,) + tuple(v.shape[1:]), dtype=v.dtype, device=dev)
            if isinstance(v, torch.nn.Parameter) and v.is_cuda and os.environ.get("SLOT_CACHE_SCALES_TO_HOST", "1") == "1":
                try:
                    import exact_pin as _ep; host = _ep.exact_pinned_like(v.detach().to("cpu"))
                except Exception:
                    host = v.detach().to("cpu").pin_memory()
                self._keep_host.append(host)
                v.data = get_accelerator_view_from_cpu_tensor(host)
                self.res[k] = v.data
            else:
                self.res[k] = v.data if isinstance(v, torch.nn.Parameter) else v
        self.scalars = [s for s in scalars]                      # list of [E] tensors (may be None)
        self.scalar_slots = [torch.zeros(S, dtype=s.dtype, device=dev) if s is not None else None for s in self.scalars]
        self.e2s = torch.full((self.E + 1,), -1, dtype=torch.int32, device=dev)
        self.s2e = torch.full((S + 1,), self.E, dtype=torch.int32, device=dev)
        self.last = torch.full((S,), -1, dtype=torch.int64, device=dev)
        self.step = torch.zeros((), dtype=torch.int64, device=dev)
        self.misses = torch.zeros((), dtype=torch.int64, device=dev)
        self.SB = 1
        while self.SB < S: self.SB *= 2
        self.rows = {k: (self._rows64(self.host[k]), self._rows64(self.slots[k])) for k in ("w13", "w2")}
        self.rows_res = {k: (self._rows64(self.res[k]), self._rows64(self.res_slots[k])) for k in self.res}
        # pad scalars with a sink row so hits scatter harmlessly
        self.scal_pad = [torch.cat([s, torch.zeros(1, dtype=s.dtype, device=dev)]) if s is not None else None for s in self.scalars]
        self.scal_slot_pad = [torch.zeros(S + 1, dtype=s.dtype, device=dev) if s is not None else None for s in self.scalars]
        self.scalar_slots = [p[:S] if p is not None else None for p in self.scal_slot_pad]
        self.bufs = {}

    @staticmethod
    def _rows64(t):
        n = t.shape[0]
        flat = t.contiguous().view(torch.uint8) if t.element_size() == 1 else t.contiguous().view(torch.uint8)
        return flat.view(torch.int64).reshape(n, -1)

    def bufs_for(self, N):
        if N not in self.bufs:
            d = self.dev
            self.bufs[N] = dict(src=torch.zeros(N, dtype=torch.int32, device=d), dst=torch.zeros(N, dtype=torch.int32, device=d),
                                mask=torch.zeros(N, dtype=torch.int8, device=d), slot=torch.zeros(N, dtype=torch.int32, device=d))
        return self.bufs[N]


_triton = None
def _ensure_triton():
    global _triton
    if _triton is None:
        _triton = _install_triton()
    return _triton


def _cache_forward(lc, topk_ids, N):
    """topk_ids: int32 [N] flat global expert ids. Returns int32 [N] slot ids; performs misses."""
    triton, fused, mcopy = _ensure_triton()
    b = lc.bufs_for(N)
    fused[(1,)](topk_ids, lc.e2s, lc.s2e, lc.last, lc.step, b["src"], b["dst"], b["mask"], b["slot"], lc.misses,
                N=N, S=lc.S, E=lc.E, SB=lc.SB)
    BLOCK = 2048
    for k in ("w13", "w2"):
        src, dst = lc.rows[k]; n = src.shape[1]
        mcopy[(N, triton.cdiv(n, BLOCK))](src, dst, b["src"], b["dst"], b["mask"], n, BLOCK=BLOCK)
    for k in lc.rows_res:
        src, dst = lc.rows_res[k]; n = src.shape[1]
        mcopy[(N, triton.cdiv(n, BLOCK))](src, dst, b["src"], b["dst"], b["mask"], n, BLOCK=BLOCK)
    for sp, ssp in zip(lc.scal_pad, lc.scal_slot_pad):
        if sp is not None:
            ssp.index_put_((b["dst"].long(),), sp[b["src"].long()])
    return b["slot"]


def install():
    if S_SLOTS <= 0:
        return
    import importlib.abc, importlib.util

    # ---- 1) force the Modular experts class (routing outside the kernel) ----
    def _patch_oracle(module):
        orig = module.backend_to_kernel_cls
        def backend_to_kernel_cls(backend):
            cls = orig(backend)
            if backend == module.NvFp4MoeBackend.FLASHINFER_TRTLLM:
                from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import TrtLlmNvFp4ExpertsModular
                _LOG("forcing TrtLlmNvFp4ExpertsModular")
                return [TrtLlmNvFp4ExpertsModular]
            return cls
        module.backend_to_kernel_cls = backend_to_kernel_cls

    # ---- 2) offloader: after UVA offload of a module, register its MoE weights with a slot cache ----
    def _patch_uva(module):
        orig = module.UVAOffloader._maybe_offload_to_cpu if hasattr(module, "UVAOffloader") else None
        cls = getattr(module, "UVAOffloader", None) or next(c for c in vars(module).values() if isinstance(c, type) and hasattr(c, "_maybe_offload_to_cpu"))
        orig = cls._maybe_offload_to_cpu
        def _maybe_offload_to_cpu(self, mod):
            r = orig(self, mod)
            # find RoutedExperts submodules whose w13/w2 are now UVA views
            for name, sub in mod.named_modules():
                w13 = getattr(sub, "w13_weight", None); w2 = getattr(sub, "w2_weight", None)
                if w13 is None or w2 is None: continue
                if getattr(w13, "_vllm_is_uva_offloaded", False) and getattr(w2, "_vllm_is_uva_offloaded", False):
                    sub._slot_cache_pending = True
            return r
        cls._maybe_offload_to_cpu = _maybe_offload_to_cpu
        _LOG("uva offloader hook installed")

    # ---- 3) experts: after process_weights_after_loading (weights are in kernel format), build the cache;
    #         in _invoke_kernel, redirect cached layers through the slot path ----
    def _patch_experts(module):
        Mod = module.TrtLlmNvFp4ExpertsModular
        Base = module.TrtLlmNvFp4ExpertsBase
        orig_pwal = Base.process_weights_after_loading
        def process_weights_after_loading(self, layer):
            orig_pwal(self, layer)
            if getattr(layer, "_slot_cache_pending", False):
                # Weights are transient HBM copies inside device_loading_context here; the UVA re-offload happens
                # after we return. Defer the cache build to the first _invoke_kernel, where w1/w2 are final.
                qc = self.quant_config
                w1s = getattr(layer, "w13_weight_scale", qc.w1_scale); w2s = getattr(layer, "w2_weight_scale", qc.w2_scale)
                self._slot_cache_deferred = getattr(self, "_slot_cache_deferred", {})
                self._slot_cache_deferred[id(layer)] = (getattr(layer, "layer_name", "?"), w1s, w2s,
                                                       [self.g1_scale_c, qc.g1_alphas, qc.g2_alphas])
                layer._slot_cache_pending = False
                layer._slot_cache_layer = True
                _LOG(f"deferred cache for {getattr(layer, 'layer_name', '?')}")
        Base.process_weights_after_loading = process_weights_after_loading

        orig_invoke = Mod._invoke_kernel
        def _invoke_kernel(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts, a1q_scale):
            lc = _registry.get(w1.data_ptr())
            if lc is None and getattr(self, "_slot_cache_deferred", None):
                # first call for a deferred layer: w1/w2 are now the final UVA views. Match by scale identity.
                for key, (name, w1s, w2s, scalars) in list(self._slot_cache_deferred.items()):
                    if w1s.data.data_ptr() == self.quant_config.w1_scale.data_ptr():
                        if not w1.is_cuda or w1.device.type != "cuda":
                            break
                        lc = LayerCache(name, w1, w2, w1s, w2s, scalars, S_SLOTS)
                        _registry[w1.data_ptr()] = lc
                        del self._slot_cache_deferred[key]
                        _LOG(f"cache built for {name}: S={S_SLOTS} E={lc.E} slot bytes={sum(t.numel()*t.element_size() for t in lc.slots.values())/1e9:.2f} GB")
                        break
            M = hidden_states.shape[0]
            if lc is None or M > BYPASS_ABOVE or M * topk_ids.shape[1] > lc.S:
                return orig_invoke(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts, a1q_scale)
            N = M * topk_ids.shape[1]
            flat = topk_ids.reshape(-1).to(torch.int32)
            slot_ids = _cache_forward(lc, flat, N).view(topk_ids.shape)
            # swap quant_config views to the slot tensors for this call
            return _invoke_slot(self, orig_invoke, lc, output, hidden_states, topk_weights, slot_ids, activation, a1q_scale)
        Mod._invoke_kernel = _invoke_kernel
        _LOG("experts hook installed")

    def _invoke_slot(self, orig_invoke, lc, output, hidden_states, topk_weights, slot_ids, activation, a1q_scale):
        # Re-implement _invoke_kernel body against slot tensors (mirrors trtllm_nvfp4_moe.py:344-410).
        import flashinfer
        from vllm.model_executor.layers.fused_moe.utils import trtllm_moe_pack_topk_ids_weights
        block_scale, per_token_scale = a1q_scale, None
        packed = trtllm_moe_pack_topk_ids_weights(slot_ids, topk_weights)
        g1c, g1a, g2a = lc.scalar_slots
        flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe(
            topk_ids=packed, routing_bias=None, hidden_states=hidden_states,
            hidden_states_scale=block_scale.view(torch.float8_e4m3fn).reshape(*hidden_states.shape[:-1], -1),
            gemm1_weights=lc.slots["w13"], gemm1_weights_scale=lc.res_slots["w13_scale"].view(torch.float8_e4m3fn), gemm1_bias=None,
            gemm1_alpha=self.gemm1_alpha, gemm1_beta=self.gemm1_beta, gemm1_clamp_limit=self.gemm1_clamp_limit,
            gemm2_weights=lc.slots["w2"], gemm2_weights_scale=lc.res_slots["w2_scale"].view(torch.float8_e4m3fn), gemm2_bias=None,
            output1_scale_scalar=g1c, output1_scale_gate_scalar=g1a, output2_scale_scalar=g2a,
            num_experts=lc.S, top_k=self.topk, n_group=0, topk_group=0, intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=0, local_num_experts=lc.S, routed_scaling_factor=None, routing_method_type=1,
            do_finalize=True, activation_type=_act_int(activation), per_token_scale=per_token_scale, output=output,
            tune_max_num_tokens=lc.S)

    def _act_int(activation):
        from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import activation_to_flashinfer_int
        return activation_to_flashinfer_int(activation)

    targets = {
        "vllm.model_executor.layers.fused_moe.oracle.nvfp4": _patch_oracle,
        "vllm.model_executor.offloader.uva": _patch_uva,
        "vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe": _patch_experts,
    }

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in targets: return None
            fn = targets.pop(name)
            if not targets:
                try: sys.meta_path.remove(self)
                except ValueError: pass
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None: return None
            loader = spec.loader; orig_exec = loader.exec_module
            def exec_module(module, _orig=orig_exec, _fn=fn):
                _orig(module); _fn(module)
            loader.exec_module = exec_module
            return spec
    sys.meta_path.insert(0, _Finder())
    _LOG(f"installed: S={S_SLOTS} bypass_above_tokens={BYPASS_ABOVE}")
