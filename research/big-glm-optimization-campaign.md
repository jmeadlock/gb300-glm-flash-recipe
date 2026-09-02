# Full GLM-5.3 NVFP4 on one GB300 — optimization campaign (2026-09-02)

Target: `incoai/GLM-5.3-NVFP4@54e52520` (433 GB) on vLLM 0.28.0, selective UVA offload of routed-expert weights to Grace memory. Constraint: **quality-neutral only** — bf16 KV cache, no KV quant, no lower-bit weights.

Bench: `bench_big.py` — streaming, temperature 0, `reasoning_effort=low`, 512 completion tokens, warm C1/C4/C8, TTFT excluded from decode rate. Numbers are output tok/s. Prose = essay prompt; code = Python module prompt.

## Scoreboard

| Variant | Offload (GiB) | KV dtype | KV tokens | Spec decode | C1 | C4 agg | C8 agg |
|---|---:|---|---:|---|---:|---:|---:|
| v1 | 270 | fp8 (scale 1.0 — uncalibrated) | 1,678k | — | 23.7–25.1 | 42.8 | 65.9 |
| v3 | 245 | bf16 | 588k | — | 30.0 | 48.4 | 63.7 |
| v4 | 245 | bf16 | 416k | DFlash2 K7 | 13.75 | 23.9 | 35.0 |
| **v5 (retained)** | **200** | bf16 | 211k | — | **33.3** | **55.4** | **72.4** |
| v5 code prompt | 200 | bf16 | 211k | — | 33.2 | 54.9 | 67.7 |

v5 vs v3: **+11% C1, +14% C4, +14% C8**. v5 vs v1 (the first working config, same morning): +33–40% C1 *and* a better KV cache.

v5 prefill (cold, prefix-cache-defeated): 6.6k tok → 1.24 s; 26k → 4.83 s; 49k → 8.85 s (~5.4–5.6k tok/s). TTFT under C8 load 0.66 s. Tool-call turn 2.1 s.

Retained launch: `/home/milo/launch-big-v5.sh` on the box (v3 flags with `--cpu-offload-gb 200 --gpu-memory-utilization 0.95`). KV 18.2 GiB = 211k tokens = 3.2× a 65k request at `--max-num-seqs 8`.

## What we learned

**Bytes over C2C is the only first-order lever.** Every GiB of routed experts moved from Grace into HBM buys decode speed: 270→245→200 GiB offload went 24→30→33 tok/s at C1. The floor is set by the KV cache you need: at 65k context and max-seqs 8, ~200 GiB offload leaves 18 GiB / 211k tokens, which is the practical limit for an interactive endpoint. Going lower (185–190) would leave <10 GiB KV for maybe +4% — not worth the capacity.

**Speculative decoding loses on an offloaded MoE — measured, not theorized.** DFlash2 (`incoai/GLM-5.3-DFlash2`, block 8, K7) cut C1 in half: 30.0 → 13.75. From `/metrics` over 4,507 verify steps: draft acceptance rate 0.098, mean acceptance length 1.68. On a fully-HBM 2×GB300 rig the same drafter accepts 2.4–2.9 proposals/verify on code and only 1.25–1.33 on prose (catid's sweep), so our prose acceptance is normal — the *cost* side is what changes: each verify step routes 8 positions independently and can touch up to 8× the offloaded expert bytes for ~1.7 useful tokens. When decode is bandwidth-bound on expert fetch, that's a loss regardless of drafter quality. MTP has the same physics with worse acceptance; skipped.

**Autotune cache is keyed by config.** FlashInfer re-tunes (~30 min) whenever MoE batch shapes change (spec decode on/off, offload split). The mounted `/home/milo/vllm-cache` accumulates every key, so *restarts* of a known config skip it, but each *new* config pays once. Full cold start for a new config ≈ 17 min weights + 2 min KV/profile + 29 min autotune ≈ 50 min.

**Async scheduling** stays on with dflash (it's inside vLLM's `EagleModelTypes`), so v4 was not confounded by a scheduler change.

## Levers evaluated and closed

| Lever | Status |
|---|---|
| Lower offload budget | **Done — v5 retained** |
| DFlash2 / MTP | Retired (measured loss) |
| Hot-expert GPU cache (vLLM PR #37190, LFRU) | Right idea, wrong backend: unmerged, restricted to Triton/CPU MoE paths, not the FLASHINFER_TRTLLM NVFP4 kernel we need. Watch. |
| `--offload-backend prefetch` | Moves whole layers (256 experts to use 8). Wrong for sparse decode. |
| KV fp8 / lower-bit weights | Out of scope by rule |
| vLLM 0.29rc1 | No offload-path changes in the notes; not worth a rebuild |
| SGLang OffloaderV2 | 1.41 tok/s C1 (measured earlier); dead end |

Concurrency remains the cheap throughput multiplier (C8 ≈ 2.2× C1 on the same bytes), and `reasoning_effort=low` per request is the real-world wall-clock win for agent turns.

Helper: `spec_metrics.py` snapshots vLLM's `/metrics` spec-decode counters into acceptance rate / mean acceptance length — use it before/after any bench on a speculative config.
