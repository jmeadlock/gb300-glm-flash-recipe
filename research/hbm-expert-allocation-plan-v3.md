# HBM Expert Allocation — build plan v3 (2026-09-05, reconciled with Astra review)

Supersedes the execution order in `hot-expert-cache-design-v2.md`. Design (slot cache, one TRT-LLM launch,
expert→slot remap) is unchanged. What changes: **measure ownership and the graph critical path before touching
slot counts or prefetch.** Astra's review (`glm53-astra-next-steps.md`, `glm53-astra-research.md`) drove this.

## Reconciliation with results as of 11:05 CDT

| Astra item | Status | Evidence |
|---|---|---|
| `_rows64(t.contiguous())` may clone UVA banks into HBM | **Cleared.** Same storage, 0.0 MB HBM delta on real UVA views, both pinning paths | `ROWS64_AUDIT` in LEDGER |
| GB/GiB accounting was sloppy | **Corrected.** torch reports GiB: capacity 249.81 GiB = 268.2 GB; sc2 allocated 240.90 GiB = 258.7 GB. Residual "other" (non-expert weights + first-forward workspace) = **39.2 GB**, not 21.4 | this doc §Budget |
| Four OOMs = budget not understood | **Diagnosed, three distinct causes:** (1) cache built inside `device_loading_context` → transient HBM copies stranded; (2) vLLM keeps full [E] block scales resident = 45.3 GB; (3) `torch.pin_memory()` pow2 rounding: 3.22 GB → 4 GiB, 33% host waste → 491 GB shmem wall | LEDGER sc1–sc5 lines, `PIN_TEST` |
| Copies are Triton SM kernels, not copy-engine DMA | **Acknowledged.** Measured 55.4 µs/miss = 383 GB/s effective (at the link ceiling, so the SM path is not leaving bandwidth on the table — but it occupies SMs and is on the critical path) | `SLOT_CACHE_B_RESULT G_ms_copy_only_*` |
| Exact one-layer-ahead prefetch is impossible without schedule change | **Agreed; retired as a core claim.** Previous-token ids are a *prediction*. Deferred until demand-fill is correct and measured | — |
| Scale bytes are 12.5% of weights, not 1/16 | Correct (1/16 of *elements*, but e4m3 vs nibbles). Per expert: 18.874 MB weights + 2.359 MB scales = 21.234 MB | — |
| Graph-replay baseline across V1-mono / modular-no-cache / slot all-hit / slot churn | **Partially done** on synthetic layer (0.041 HBM / 0.467 UVA / 0.068 all-hit / 0.146 @85%). **Not done on the real model** | B2 ledger |

## Budget (corrected, per real tensor shapes)

Per expert-row: w13 12.58 MB + w2 6.29 MB = 18.874 MB weights; scales 2.359 MB; total 21.234 MB.

HBM (S=112, all 75 MoE layers cached, scales on host):
slots 158.5 GB + slot-scales 19.8 + KV 8.6 + other 39.2 = **226.1 GB of 268.2 GB → 42 GB headroom** for graph pools, MLA workspace, allocator reserve.

Host (exact pin): 407.7 GB experts+scales of 494 GB Grace. pow2-rounded would be 543.6 GB → this is why sc3/4/5 died.

## Execution order (gates, each must pass before the next)

1. **sc6-s112 up** — first real-model slot-cache run (loading now). Read back: cache count 75, peak HBM from `torch.cuda.max_memory_allocated` + nvidia-smi, host RSS.
2. **Ownership proof** — per-layer ledger from the live process: slot tensors, scale slots, maps, per-N buffers; assert no [E] scale bank remains on device; log allocator reserved vs allocated. Script: `hbm_ledger.py` (runs via a debug endpoint or `docker exec` py-spy-free snapshot of `torch.cuda.memory_stats`).
3. **Correctness gate** — tools smoke → **greedy 20/20 vs V1** (bf16 KV, temp 0). Fail = stop and read; do not tune.
4. **Graph critical path, real model** — per-token decode time decomposition: attach the routed-expert trace hook (already in sitecustomize), count misses/layer/token from `lc.misses`, compute measured ms/token vs predicted `75×(0.068 + miss×0.0554)+other`. Separate all-hit vs miss-heavy tokens by the trace.
5. **Bench** — n=3 C1/C4/C8 prose + code, cold prefill ladder, agent-loop gate. Report vs V1 33.8. Milestones 45 / 60; anything higher needs receipts.
6. **Per-layer S from traces** — replay `trace/r1-base` (and a fresh trace from sc6) through the LRU sim per layer; allocate slots to maximize saved transfer bytes per HBM byte. Layers with flat routing get fewer slots or full residency; skewed layers get more. Deploy as `SLOT_CACHE_PER_LAYER=<json>`.
7. **Hit-path overhead** — compact missed rows before launching copy grids (today: N×ceil(row/2048) programs regardless of misses); fuse scalar `index_put_`. Only if step 4 shows it matters.
8. **Prediction (optional)** — previous-token ids as prefetch hints on a side stream; gate on useful-byte precision and demand-stall saved, not overlap.
9. Spec decode K=1/2 — only after all of the above, with byte + acceptance receipts.

## Production and quality rules (unchanged)
- `glm53-big-v1-keep` is the rollback; relaunch with `recipes/launch-big-v1.sh` (~18 min).
- ≥4-bit weights, bf16 KV, greedy 20/20 vs V1 is the judge. No FP8 KV, no MTP, no sub-4-bit.
- Prefill bypass stays (M > 16 tokens → plain UVA kernel). Cached slots never remove access to absent experts.

## Side finding to publish
V1 today wastes ~63 GB of Grace memory to `pin_memory()` pow2 rounding (189 GiB requested → ~252 GiB pinned). Harmless at V1's budget; fatal at the slot cache's. `exact_pin.py` fixes both.
