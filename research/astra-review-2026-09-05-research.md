# GLM-5.3 NVFP4 single-GB300 slot-cache research notes

Scope: primary-source/read-only critique for a single DGX Station GB300, speed-first but quality-neutral: >=4-bit weights and unquantized/bf16 KV. No benchmarks or machine mutations were performed.

## Prioritized conclusions

1. **Continue the vLLM/TRT-LLM NVFP4 slot-cache path, but treat the current artifact as a source-audit target, not a recommendation-ready serving recipe.** The public post and design doc correctly identify the winning shape: keep the TRT-LLM NVFP4 MoE kernel, keep fixed HBM slot tensors, remap expert ids to slot ids, and avoid a second MoE launch per layer.[1][2] The latest `/tmp/glm53doc/slot_cache_hook.py` materially advances this: misses/LRU bookkeeping are now device-side Triton, synthetic graph churn is exact per `hook_dryrun.py`, and graph all-hit dry-run numbers were reported by the user as ~0.075 ms.[3][4] But the full-model session has already seen four OOM launches and is only reported running at `sc5-s112 OFFLOAD320`; this makes HBM budget/source audit the next decision gate, not more design prose.[3]

2. **Top risk: `_rows64(t.contiguous())` may clone non-contiguous UVA-backed full expert tensors into HBM.** In the hook, `_rows64` always calls `t.contiguous().view(torch.uint8)` before flattening.[3] If vLLM's final UVA weight view is already contiguous, this is harmless; if it is a non-contiguous accelerator view over pinned host storage, `contiguous()` can allocate a CUDA tensor and copy the full offloaded bank into HBM, which would explain OOM behavior. This is an audit risk, not proven from source alone. The next advice should be: assert/log `is_contiguous`, `device`, `storage/data_ptr`, and allocation deltas at cache-build time before declaring any slot size viable.

3. **The copy path is not yet the design doc's copy-engine path.** The v2 design says misses should be `cudaMemcpyAsync` on a copy-engine stream and overlapped one layer ahead.[2] The current hook copies rows with Triton `masked_row_copy` kernels for `w13`, `w2`, scale banks, plus scalar `index_put_` updates.[3] That is GPU-kernel copy from UVA to HBM, not a CUDA runtime H2D copy-engine submission. NVIDIA documents that memcpy APIs can be asynchronous depending on arguments, and pinned-memory async transfers are the right class for overlap, but also warns API calls may block for undocumented reasons.[10] FreeToken's source independently carries a very relevant warning: `cudaMemcpyBatchAsync` can silently degrade to a synchronous host-blocking copy when a batch mixes large entries with small registered-host entries, so FreeToken deliberately fuses/excludes small banks from the per-run batch.[6] Recommendation change: do not assume 55-60 µs CE overlap from the current hook; either switch to an explicit fused async copy plan or measure the Triton UVA-copy kernels as first-class kernels.

4. **Exact one-layer-ahead routing lookahead is not available without changing the execution schedule.** The router for layer `l+1` depends on hidden states after layer `l`; exact `topk_ids` for `l+1` therefore are not known before layer `l` runs. The design's "prefetch one layer ahead" can only be exact after computing the next layer's router, which reduces or eliminates overlap with the preceding layer's compute.[2] FreeToken avoids overclaiming this by making cache miss handling part of its offload backend and by separately handling prefill double-buffering/overlap.[5][6] Recommendation change: retire "exact one-layer-ahead prefetch" as a core claim unless the schedule is split into router-early / copy / MoE phases and measured; use LRU residency to reduce misses first.

5. **CUDA graph safety is plausible only if everything uses fixed addresses, fixed shapes, and device-memory state updates.** NVIDIA's CUDA Graph constraints say graph topology is static; memory addresses and kernel parameters are captured; all referenced addresses must remain valid for the graph lifetime; CPU code is not captured; and data-dependent branching cannot change topology without special conditional nodes.[9] The hook's fixed slot tensors, fixed-size buffers keyed by `N`, and device-side maps align with those constraints.[3] The current code still needs full-model confirmation for `index_put_`, dynamic buffer creation in `bufs_for(N)`, fallback/bypass paths, and any stream interactions during vLLM graph capture.[3][4]

6. **HBM budget should be computed from actual source tensors and launch args, not from `S=128` intuition.** The hook launch script offloads routed expert weights only and sets `--kv-cache-dtype bfloat16` plus `--kv-cache-memory 8589934592`.[12] The hook allocates per-layer slot tensors for both `w13` and `w2`, optionally moves scale tensors to pinned host, allocates scale slots, scalar slot pads, maps, and per-`N` buffers.[3] At GLM-shaped dimensions, weights alone are roughly 18.9 MB/expert/layer before scale/scalar overhead; `S=112` over 75 MoE layers is already about 158 GiB of slot weights, and `S=128` is about 181 GiB before runtime/graphs/KV/non-expert weights. Four OOM launches make `S=128 everywhere` too aggressive until a real layer count, offload match list, scale residency, allocator reserve, and graph-pool budget are read back.

7. **Launch-vs-kernel lesson still stands, but the metric changed.** The public post measured 0.25 ms per 128-expert MoE window launch regardless of hits and rejected hot/cold two-MoE-launch designs.[1] The current hook preserves one TRT-LLM MoE launch but adds bookkeeping/copy/scalar work before it.[3] CUDA Graphs may remove host launch overhead, but the device time of those kernels and any copy contention still counts. Therefore judge the implementation by full per-layer graph replay time at realistic miss rates, not by "one MoE launch" wording alone.

## Primary-source findings that materially affect recommendation

- **FreeToken supports the concept, not a drop-in GB300/GLM-5.3 route.** Its README claims global LRU expert caching, graph-compatible execution, and elastic memory management for edge MoE serving.[5] Its supported-models doc says the offload backend keeps experts in host RAM with an LRU GPU expert-slot cache; supports GLM-5.2 NVFP4; and has `offload`, `cpu`, and `hybrid` modes.[7] Its source is directly relevant because it implements device-side cache state, graph-safe stats, fused multi-bank copy descriptors, per-layer pinned-residency checks, and explicit small-bank handling.[6] But a FreeToken issue says current Ubuntu packages/install docs are x86-64 and ARM64/DGX Spark support was only requested, so FreeToken is a reference architecture, not the immediate GB300 production path.[8]

- **thecodacus/llama.cpp and ggml PRs strengthen the caution against synchronous miss paths and static placement.** thecodacus has explicit MoE expert-cache commits, including "keep hot routed experts resident in VRAM" and graph wiring.[11] ggml PR #24524 reports a CUDA-side adaptive cache for CPU-resident experts, but its key design keeps `MUL_MAT_ID` on CPU and dispatches GPU cached hits while CPU threads compute misses; it also documents prior approaches that moved the whole node to GPU and made cache misses synchronous, causing regressions, and it notes per-layer sync points can lose even with 97-99% hit rate.[13] This materially argues against a naive all-GPU miss-critical path unless GB300 measurements prove the miss kernels are hidden.

- **NVIDIA TensorRT-LLM docs support the TRT-LLM NVFP4 backend choice and argue against native MTP as the next lever.** NVIDIA's GLM-5 guide applies to GLM-5.2/5.3, says NVFP4 on B200/GB200 uses `moe_config.backend: TRTLLM`, and states MTP is not currently supported with the NVFP4 checkpoint.[14] That aligns with keeping the same TRT-LLM NVFP4 kernel and postponing MTP/speculative work until residency is stable.

- **catid's numbers remain an external physics reference, not a single-Station target.** catid's primary benchmark uses full GLM-5.3 NVFP4 + DFlash2 on 2x DGX Station GB300 and reports TP2+EP2 C1 165.5 tok/s code and 107.4 tok/s prose; PP2+DFlash2 reports C1 154.9 code and 106.3 prose, with much higher aggregate rates at large concurrency.[15] Those results do not contradict the single-Station slot cache; they show what all-HBM/multi-station plus speculation can do, while the current task is one Station with offload.

## Recommended next decision gates

1. **Source audit before another slot-size recommendation:** prove `_rows64` does not clone UVA banks into HBM; if it does, replace with a stride-aware view/copy kernel or a FreeToken-style pointer/feature-byte descriptor.
2. **Budget gate:** generate a per-layer, per-bank HBM ledger from actual tensors after `process_weights_after_loading`: non-expert resident weights, slot weights, scale slots, scalar slots, maps/buffers, graph pools, KV, and allocator reserve.
3. **Copy-path gate:** separate all-hit, 1-miss, 2-miss, and 8-miss graph replay time; classify whether copies are SM kernels or copy-engine DMA; do not reuse the 55-60 µs CE model for Triton copies.
4. **Graph gate:** capture and replay the real vLLM decode graph with changing in-place `topk_ids`/weights and no CPU-side allocations/sync inside capture; compare greedy outputs 20/20 vs V1 with bf16 KV.
5. **Policy gate:** keep >=4-bit weights and bf16/unquantized KV; do not reopen FP8 KV, IQ2/sub-4-bit, pruned quality-changing checkpoints, or native MTP until the slot cache is correct and budget-stable.

## Sources

[1] https://al-engr.com/gb300-glm-53-testing.html
[2] https://raw.githubusercontent.com/jmeadlock/gb300-glm-flash-recipe/main/research/hot-expert-cache-design-v2.md
[3] file:///tmp/glm53doc/slot_cache_hook.py
[4] file:///tmp/glm53doc/hook_dryrun.py
[5] https://github.com/FlashML-org/FreeToken
[6] https://raw.githubusercontent.com/FlashML-org/FreeToken/main/python/freetoken/moe/offload_cache.py
[7] https://raw.githubusercontent.com/FlashML-org/FreeToken/main/docs/models.md
[8] https://github.com/FlashML-org/FreeToken/issues/22
[9] https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/constraints.html
[10] https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html
[11] https://github.com/thecodacus/llama.cpp/commits/fable5/moe-cache-readme
[12] file:///tmp/glm53doc/launch-slotcache.sh
[13] https://github.com/ggml-org/llama.cpp/pull/24524
[14] https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md
[15] https://github.com/catid/dgx_station_benchmarks/blob/main/glm-5.3/README.md
