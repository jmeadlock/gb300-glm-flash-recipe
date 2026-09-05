# Hot-expert cache for full GLM-5.3 on one GB300 — design v2 (2026-09-05, 06:40 PDT)

Status: design settled by measurement; ready to build. Supersedes the "VMM page mixing" (dead: 91 GB/s host reads)
and "two kernel launches per layer" (dead: launch cost is fixed, see §2) ideas from earlier tonight.

## 1. What the measurements say

| fact | value | source |
|---|---|---|
| v5/V1 decode, single stream | 32.6 / 33.8 tok/s | r1, r2 n=3 |
| host-read ceiling over C2C (pinned cudaHostAlloc) | ~344–360 GB/s | c2c_bench, hostmap_probe |
| VMM host pages / WC / managed | 91 / 91 / migrates to HBM | vmm_probe, hostmap_probe |
| 128 resident experts per layer capture | 73% static (same domain), 75% LRU, 55% static cross-domain | r1 trace, 79k tokens |
| per-layer MoE launch, M=1, all weights HBM | 0.19 ms | split_kernel_test |
| per-layer MoE launch, M=1, all weights pinned UVA (today) | 0.50 ms | split_kernel_test |
| a *window* launch (128 of 256 experts) | 0.25 ms **regardless of hits (0 or 8)** | split_kernel_test skew 0/0.75/1.0 |
| 19 MB expert row H2D copy at CE speed | ~55 µs | c2c_bench (358 GB/s) |

Sanity: 0.50 ms × 75 MoE layers = 38 ms/token ≈ measured 30 ms budget → the synthetic layer is faithful.

## 2. Why two launches cannot work
The TRT-LLM NVFP4 kernel's cost at M=1 is fixed per launch (~0.25 ms for a 128-expert window, ~0.19 for 256 in HBM)
and does not fall when fewer experts are hit. Two window launches = 0.5 ms before any combine, equal to today's
single UVA launch. Any design that adds a launch per layer loses at decode. **One launch per layer, period.**

## 3. The design: slot cache + id remap, one launch
Per offloaded MoE layer *l*:

- **HBM slot tensors** `w13_slots[l]: [S, 2*inter, hid/2]`, `w2_slots[l]: [S, hid, inter/2]` + matching block-scale
  slots. S = number of resident slots (≈128 at V1's budget; tune per layer later). Allocated once; addresses never
  change → CUDA-graph safe.
- **Pinned host backing** `w13_host[l], w2_host[l]` for all 256 experts: exactly today's UVA tensors. Never freed.
- **Map** `expert→slot: int32[256]` (−1 = not resident) and `slot→expert: int32[S]`, both on GPU, plus an LRU clock.
- **Forward (one launch):** run the router as the Modular path does (outside the kernel, `topk_ids`,
  `topk_weights`), then:
  1. `miss = topk_ids where expert→slot == −1` (≤ 8 per token per layer; ~2 expected at 75% hit).
  2. For each miss: pick a victim slot (LRU), `cudaMemcpyAsync` the 19 MB row from `w*_host[l][e]` into the
     slot on the **copy engine stream**, update both maps. Bounded: ≤ 8 copies × 55 µs = 0.44 ms worst case,
     ~0.11 ms expected — and it overlaps with the previous layer's kernel if we prefetch one layer ahead (§5).
  3. Remap: `slot_ids = expert→slot[topk_ids]` (all ≥ 0 now).
  4. **Single launch** of `trtllm_fp4_block_scale_routed_moe(topk_ids=slot_ids, gemm1_weights=w13_slots[l], …,
     num_experts=S, local_expert_offset=0, local_num_experts=S, do_finalize=True)`. Per-expert scales/alphas
     (`g1_alphas`, `g2_alphas`, `w1_scale`, `w2_scale`) are slot-indexed too → they are swapped with the row.
- **Numerics:** same kernel, same bytes, same routing weights; only the *index* differs. Expected greedy-identical
  to V1. The gate will say.

Why this beats today even before tuning: today every expert read is host (0.50 ms/layer). With S=128 and 75% hit,
6 of 8 reads are HBM; the 2 misses are CE copies (~0.11 ms, overlappable) and then HBM reads. Layer time → ~0.19 ms
kernel + non-overlapped copy. Rough token budget: 75 × ~0.25 = ~19 ms → **~50 tok/s** without prefetch,
**~70–80** with one-layer-ahead prefetch hiding the copies. Honest range: 50–80 tok/s C1 vs 33.8.

## 4. Where it plugs into vLLM 0.28 (image `vllm-glm53-uva:v0.28.0-2cf0a691`)
- Force the **Modular** TRT-LLM class (`TrtLlmNvFp4ExpertsModular`) instead of Monolithic for offloaded layers:
  router outside the kernel gives us `topk_ids` to remap. (`oracle/nvfp4.py:backend_to_kernel_cls` prefers
  Monolithic; hook it.)
- Replace the UVA offloader's per-parameter `p.data = get_accelerator_view_from_cpu_tensor(cpu)` for
  `w13_weight`/`w2_weight` (+ their scales) with: keep the host view **and** allocate the HBM slot tensor; the
  module's forward sees the slot tensor.
- Wrap `TrtLlmNvFp4ExpertsModular.apply` (`trtllm_nvfp4_moe.py:412`) with the miss-fetch + remap before
  `_invoke_kernel`. Memory budget: S × 21 MB × 42 layers (the fully-offloaded ones) ≈ 113 GB at S=128 — that is
  more than the ~55 GB those layers get today, so **S must be chosen against the real budget**: at V1's 81 GB
  resident budget for those layers, S ≈ 92/layer (hit ≈ 63% by the trace). Better: give the *partially* resident
  layers (v5 keeps 33 layers fully in HBM) the same treatment and free their cold half → S≈128 everywhere fits.
  This is the per-layer tuning knob; the trace analyzer already reports per-layer hit curves.
- All of the above is a `sitecustomize.py` import-hook patch like tonight's two fixes; no image rebuild.

## 5. Build order (each step has a gate; stop if a gate fails)
| step | what | gate |
|---|---|---|
| A (no model) | slot cache class + remap on the synthetic layer: one launch over S slots, miss-fetch via CE stream; compare vs 256-expert reference | max-abs diff == 0 vs single-launch; layer time ≈ 0.19 ms + copies |
| B (no model) | one-layer-ahead prefetch: fetch layer l+1's misses on the CE stream while layer l computes | measured overlap; layer time → ~0.2 ms at 25% miss |
| C (model, night) | static S=128 in the real model, no eviction (profile-seeded from r1 trace) | greedy 20/20 vs V1; C1 ≥ 45 |
| D (model, night) | LRU eviction + prefetch live | greedy 20/20; C1 ≥ 60; no stalls at C4/C8 |
| E (model, night) | per-layer S tuning; agent-loop, prefill ladder, 4-turn gates → **V2** | full endpoint bar |

Steps A+B are pure GPU work and can run any time the box has ~10 GB HBM free — i.e. alongside V1 serving. Tonight.

## 6. Closed tonight, with evidence
- VMM host-page mixing in one VA: 91 GB/s (4× slower than pinned). `vmm_probe`, `hostmap_probe`.
- `cudaMallocManaged` + preferred-location host as a cold tier: CDMM migrates to HBM on first touch (1021 GB/s = HBM).
- Two kernel launches per layer (hot window + cold window + combine): launch cost is fixed per window, 0.25 ms
  each at M=1 regardless of hits; combine adds 0.35 ms. Net 0.5× today. `split_kernel_test` ×3.
- Static placement tables: 55% cross-domain. `trace_analyze` on r1.
- Alternative NVFP4 MoE kernels (cutlass −10%, cutedsl ±0), both non-greedy-equivalent. r3, r4.
