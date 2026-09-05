# Big GLM-5.3 V1 campaign — one GB300, 2026-09-04/05

Overnight execution of [single-station-optimization-plan.md](single-station-optimization-plan.md). Every number below
was measured on DGX Station GB300 (driver 595.84, CUDA 13.2, CDMM on), image `vllm-glm53-uva:v0.28.0-2cf0a691`.
Raw ledger and every script: [`bench/bigv1/`](../bench/bigv1/). Blog: https://al-engr.com/gb300-glm-53-testing.html

## Result: V1 recipe = [`recipes/launch-big-v1.sh`](../recipes/launch-big-v1.sh)

| | v5 (n=1, Sep 2) | v5 re-measured (r1, n=3) | **V1 (r2 config, n=3)** |
|---|---|---|---|
| single stream, prose | 33.3 | 32.6 | **33.8** |
| 4 streams aggregate | 55.4 | 55.5 | **57.7** |
| 8 streams aggregate | 72.4 | 70.6 | 57.3 (capped: `--max-num-seqs 4`) |
| code prompt C1 | 33.2 | 32.4 | 33.7 |
| experts offloaded | 200 GiB | 200 GiB | **188 GiB** |
| KV | 18 GiB / 211k tok | same | 8 GiB / 92,672 tok (1.41× a 65k request) |
| greedy vs r1 (20 prompts, temp 0) | — | — | **20/20 identical** |

Contract: warm, streaming, temperature 0, `reasoning_effort=low`, 512 completion tokens, n=3 means.

V1 gates (all pass): tool-call smoke (parsed, reasoning separated, no markup leak); 20/20 greedy-identical to v5;
agent loop with real tool execution (find/count/hash, 7 files / 28 lines / sha `6bac6345dfb0`, 2 turns, 3.8 s wall);
cold prefill ladder 5.9k tok → 1.11 s, 11.9k → 1.02 s, 23.7k → 2.24 s, 47.4k → 3.77 s (10.6–12.6k tok/s).

V1 is a +3.7% single-stream step. The campaign's value is what it measured, not the 1.2 tok/s.

## What was closed, with the number that closed it

| candidate | measured | verdict |
|---|---|---|
| `--moe-backend flashinfer_cutlass` (r3) | C1 30.4 (−10%), greedy 2/20 | closed: slower *and* changes output |
| `--moe-backend flashinfer_cutedsl` (r4) | C1 33.4 (±0), greedy 1/20 | closed: no gain, changes output |
| KV right-sizing "+10–14%" (plan estimate) | +3.7% | it frees 12 GiB, not 28; linear in offloaded bytes |
| copy-engine gather vs zero-copy | 358 vs 361 GB/s | no difference; CE path not worth building |
| VMM host pages as a cold tier (one mixed VA) | 91 GB/s reads (pinned: 344) | closed |
| `cudaMallocManaged` + prefer-host as cold tier | 1021 GB/s = migrated to HBM under CDMM | closed |
| two kernel launches per layer (hot window + cold window + combine) | 0.25 ms per window launch **regardless of hits**; 0.50 + 0.35 combine vs 0.50 today | closed: launch cost is fixed at M=1 |
| static expert placement from a profile | 55% hit cross-domain (uniform 50%) | closed: hot set is domain-specific |

## What was measured that changes the plan

**Link.** Host-read ceiling over C2C on this box: 358 GB/s copy engine, 361 zero-copy, 324–357 at expert-slab
granularity (8 random 18 MB rows, 1–8 streams). v5's implied 222 GB/s at C1 has ~1.6× headroom; the physics ceiling
without residency is ~47 tok/s (6.7 GB/token ÷ 360 GB/s + ~2.5 ms resident work).

**Routing trace** (79,119 tokens: bench, prose/code at low+max effort, replayed tool session; decode tokens, first 42
MoE layers — the offloaded ones; uniform = 25/50/62% at 64/128/160 resident):

| resident/layer | static, same domain | LRU | static, cross-domain |
|---|---|---|---|
| 64 | 50.0% | 53.5% | — |
| 96 | 62.7% | 65.6% | — |
| 128 | **73.4%** | **75.4%** | 54–59% |
| 160 | 82.6% | 83.7% | — |

~40% of routings hit the top 32 experts in every MoE layer. Top-128 hot-set overlap prose↔code is 59%. LRU matches
same-domain static and beats every cross-domain static split → the cache must be adaptive.

**Kernel behaviour** (synthetic GLM-shaped layer, M=1): all-HBM single launch 0.19 ms; all-pinned-UVA (today)
0.50 ms (×75 layers = 38 ms/token, consistent with the measured budget); a 128-expert window launch 0.25 ms whether
it hits 0 or 8 experts.

**Slot cache proof (step A of the build, no model):** 128 HBM slots + pinned backing for 256 + `expert→slot` remap
before ONE `trtllm_fp4_block_scale_routed_moe` launch. 300 steps with LRU churn at 85% hit: **max-abs diff 0.0 vs
the full-HBM reference**; steady-state kernel time 0.233 ms/layer. Miss handling is still host-side Python (18 ms/step
— the next step moves the map and LRU to the device and prefetches one layer ahead).

## Two fixes in the image, no rebuild (bind-mounted `sitecustomize.py`)

1. **Autotune cache pin.** vLLM keys the FlashInfer autotune cache on a hash of the whole engine config, so any flag
   change (even `--cpu-offload-gb`) re-pays a ~29-minute autotune. Overriding `flashinfer_autotune_cache_hash` with a
   fixed key made it 1 second (21–84 configs loaded); relaunch ≈ 18 min instead of ≈ 50. Kernel-family changes
   correctly add to the same cache.
2. **`--enable-return-routed-experts` on MLA models.** `get_routed_experts_attn_gid` does
   `isinstance(spec, FullAttentionSpec)`; at TP1 vLLM wraps GLM's `MLAAttentionSpec` in `UniformTypeKVCacheSpecs`
   and the check fails with "Routed-experts capture requires a full-attention KV cache group". Unwrapping the aggregate
   fixes it. Upstream bug in vLLM 0.28.

## Next: the adaptive hot-expert cache (design v2)

Per offloaded layer: S HBM slots, pinned-host backing for all 256 experts (today's UVA tensors), `expert→slot` map,
router-side remap of `topk_ids`, **one** kernel launch. Misses are 21 MB row copies on the copy-engine stream
(~60 µs each, ~2 expected per layer at 75% hit), prefetched one layer ahead. Same kernel, same bytes — the greedy
gate decides. Honest range: 50 tok/s C1 without prefetch, 70–80 with, vs 33.8. Build order: device-side map + LRU →
one-layer-ahead prefetch → static S=128 in the real model (greedy 20/20, C1 ≥ 45) → LRU live (C1 ≥ 60) → per-layer S
tuning + endpoint gates → V2.

Hard rules unchanged: no weights below 4-bit, no KV quantization, greedy-equivalent or it does not ship.
