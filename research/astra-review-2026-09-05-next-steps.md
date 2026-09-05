# GLM-5.3 on one GB300: Astra research review

Scope: advice only. No Station commands, changes or benchmarks executed in this review.

## Sources
- Blog: https://al-engr.com/gb300-glm-53-testing.html
- Design: https://github.com/jmeadlock/gb300-glm-flash-recipe/blob/main/research/hot-expert-cache-design-v2.md
- Campaign: https://github.com/jmeadlock/gb300-glm-flash-recipe/blob/main/research/big-glm-v1-campaign-2026-09-05.md
- Session: @session:milo/20260905_092550_7c096b; especially messages 149650, 149747, 149776–149808.
- Actual local hook inspected: /tmp/glm53doc/slot_cache_hook.py (273 lines at inspection).
- PyTorch contiguous semantics: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html
- FreeToken primary reference: https://github.com/FlashML-org/FreeToken

## Verdict
Continue the slot-cache route, but prioritize memory ownership and graph-replayed critical-path cost over prefetch or more flag sweeps. The blog trails the implementation.

Session 149650 already retracts the eager fixed-launch-floor claim: graph all-HBM 0.041 ms, UVA 0.467 ms, empty window 0.012 ms; initial slot path 0.205 ms at synthetic 85% hits, then real-class dryrun all-hits 0.0749/0.0761 ms and exact churn. These are session-reported measurements, not rerun here. Two-window design is not disproven by eager timing; numerical finalize correctness is still a separate gate. Do not restart that route until incumbent slot integration yields a measured verdict.

## Immediate audit
1. Reconcile raw bytes and peak allocations across load/repack/cache initialization/prefill/graph capture. Existing arithmetic treated 249.81 GiB as 249.8 GB and 240.9 GiB as 240.9 GB: actual decimal values are 268.231 and 258.664 GB. Therefore inferred residual/other memory is not reliable. Do not claim this explains every OOM.
2. Deduplicate storage by pointer/allocation: original UVA weights, transformed/repacked tensors, scale aliases, slot tensors, pinned allocator retention, temporary pageable plus pinned copies, non-PyTorch memory and graph pools. shm configured capacity is not committed usage; do not double-count shared/file/allocator categories.
3. `_rows64` calls `.contiguous()`: assert real post-load contiguity/storage identity before assuming it is a view. Noncontiguous UVA views could materialize copies. Hypothesis, not established OOM cause.
4. Current scale-to-host dryrun should exercise real Parameter objects and actual transformed layouts. Minimal fake-layer success is not a whole-loader lifecycle proof.

## Performance next
- Graph replay baseline: monolithic V1, modular/no-cache, slot/all-hits, trace-driven slot/churn. Same layout, slot count and batch. Account for full forward, not only routed GEMM.
- Current `_cache_forward` has bookkeeping + four SM-copy launches + scalar gather/scatter + routed MoE. Not a copy-engine-prefetch implementation. Optimize hit-path overhead, compact missed experts before launching full row grids, and fuse tiny scalar work if measurements support it.
- Trace-driven all-layer slot budget; do not transfer first-42-layer hit-rate curves to all 71/75 cached layers without measuring. Optimize saved transfer time per HBM byte; allow fully resident layers and smaller caches elsewhere.
- Current-token next-layer routing depends on previous-layer output. Previous-token IDs provide prediction, not oracle prefetch. Establish synchronous exact demand-fill first. Gate prediction by useful-byte precision, demand-miss stall saved and wasted traffic, not overlap alone.
- Keep prefill bypass; cached slots must never remove access to absent experts. Protect active slots, duplicate routed IDs across requests, cold start, domain switch, prefill/decode transitions, graph replay and batch-shape transitions.

## Arithmetic sanity (Python-calculated)
For documented H=6144,I=2048,E=256,K=8:
- Packed weight row 18.874368 MB; block scales 2.359296 MB; combined 21.233664 MB. Scale bytes are 12.5% of packed weight bytes, not 1/16.
- 128 slots x 75 layers including scales = 203.8431744 GB, before other weights/KV/workspaces.
- Ideal 21.233664 MB transfer at 360 GB/s = 58.9824 us.
- Illustrative 71 cached layers, 75% hit, top8: 3.015180288 GB demand-fill/token, 8.3755008 ms ideal transfer time. At 70%: 3.6182163456 GB, 10.05060096 ms. These omit cache overhead and compute and are not speed forecasts.

## Acceptance sequence
Memory-ownership proof -> whole-model exact demand-fill -> measured graph critical path -> trace-driven per-layer allocation -> optional prediction -> tiny speculative K=1/2 only if byte and acceptance measurements justify it.
Initial success is a stable quality-neutral win over 33.8 tok/s. Use 45–60 as milestones, not promises; 70–80 or 100+ require real model receipts.
