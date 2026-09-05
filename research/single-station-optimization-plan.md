# Full GLM-5.3 on one GB300 — optimization campaign plan (research run 2026-09-04)

Target: al-engr.com/gb300-glm-53-testing.html (v5: 33.3 tok/s C1 / 55.4 C4 / 72.4 C8, 200 GiB experts in Grace).
Rule: quality-neutral only (no sub-4-bit, no KV quant, no pruning without a blind gate). Spec decode is allowed (lossless).

## 0. The one-paragraph verdict

"Keep the experts in HBM" is physically impossible on one Station at ≥4-bit: the 700B routed-expert params need ≤2.3 bits/param to fit in ~200 GB, and every checkpoint that fits (EXL3-TR3 3.0–3.4 bpw at 316–355 GB, AQLM hybrid 310 GB, GGUF IQ2/IQ1 217–260 GB) is either sub-4-bit, rank-sliced for TP4, or both. What IS possible is keeping the *hot* experts in HBM. The measured data already says C1 is not bandwidth-bound: C8 moves ~1.6–1.9× more host bytes per second than C1 (72.4 agg vs 33.3), so the C1 wall is memory-level parallelism of a GEMV reading over C2C, not the C2C link. Two levers follow: (a) get closer to the link ceiling at C1 (kernel/gather path, ~1.4× max), and (b) cut bytes crossing C2C by popularity-aware placement (the only path to 2×+). Lever (b) is gated on one cheap measurement we have not taken: the actual expert-routing histogram of GLM-5.3 on our workload. vLLM 0.28 already ships the instrument (`--enable-return-routed-experts`, TRTLLM NVFP4 path supports capture). That measurement is experiment #1.

## 1. Evidence base (all checked 2026-09-04)

### Hardware ceiling (spec + microbenchmarks)
- DGX Station GB300: Grace LPDDR5X 496 GB at **396 GB/s** (NVIDIA product page + DGX Station dev guide — NOT the 546 GB/s of the datacenter Grace); NVLink-C2C 900 GB/s bidirectional (450/dir); HBM3e ~252–269 GB at 7.1–8 TB/s. → Host-read ceiling = min(396, 450) × efficiency.
- GH200 proxies for GPU reads of host memory: 420 GB/s (93% of C2C, arXiv 2408.11556, kernel direct access); nvbandwidth ~350–360 GB/s (Stas Bekman ml-engineering; ACM 3723851.3723853 at 78%). On the Station the LPDDR5X (396) binds before C2C → expect **~330–370 GB/s achievable**.
- CDMM (our current mode): pinned host memory stays GPU-readable over C2C; only migration hints are disabled (NVIDIA blog 2025-10-14). UVA zero-copy is unaffected.

### What v5 actually does (read from vLLM 0.28.0 source in the frozen image)
- `UVAOffloader` moves whole *parameters* (`routed_experts.w13_weight` = one [256,…] tensor per layer) to pinned host and hands the kernel a CUDA view. No staging gather, no per-expert granularity, no prefetch. Budget fills in module order → v5 log: "Total CPU offloaded parameters: 201.0" ≈ the first ~42 of 75 MoE layers fully offloaded, last ~33 fully resident.
- MoE kernel = `TrtLlmNvFp4ExpertsMonolithic` (router fused into kernel). It **supports routing-replay capture** (`supports_routing_replay_capture() → True`) — `--enable-return-routed-experts` gives per-token `[layer, topk]` expert ids (RoutedExpertsCapturer, adapted from SGLang). Incompatible only with PP>1/DCP>1 (we are TP1).
- KV: 18.19 GiB = 210,880 tok at max-num-seqs 8 / 65k. Log says `--kv-cache-memory=19135879988` reproduces v5 exactly.

### Bandwidth arithmetic (per-expert 3×6144×2048 = 37.7M params ≈ 20.1 MB NVFP4; 8 active × ~42 offloaded layers)
| C | steps/s | uniform-unique experts/layer | GB/step | implied host read GB/s |
|---|---|---|---|---|
| 1 | 33.3 | 7.9 | 6.7 | ~222 |
| 4 | 13.9 | 30.1 | 25.4 | ~352 |
| 8 | 9.05 | 56.7 | 47.8 | ~433 (> LPDDR5X peak → routing is NOT uniform at C8, i.e. skew/locality exists) |

Readings: C1 sits at ~56–62% of the LPDDR5X peak; C4/C8 push the link to/over its ceiling. So (i) up to ~1.4× is available at C1 from a more latency-tolerant access pattern (ceiling ≈ 6.7 GB / 350 GB/s + ~2.5 ms resident ≈ 46 tok/s), and (ii) the C8 overshoot is the first hard hint that GLM-5.3 routing has exploitable concentration.

C1 scenario table (same 179 GiB resident budget, hit% = share of currently-offloaded traffic made resident by popularity placement):
| hit% | @226 GB/s | @300 | @400 |
|---|---|---|---|
| 0 | 33 | 44 | 59 |
| 35 | 51 | 68 | 90 |
| 50 | 66 | 88 | 116 |
| 65 | 94 | 124 | 164 |

Spec-decode physics (unique-expert model, 1 seq): K=1 → 1.94× bytes for ~1.85 tok (wash); K=2 → 1.25× bytes/token; K=7 measured 4.2× (the v4 loss). Spec decode only turns positive after most expert reads are HBM-resident.

### vLLM ecosystem (vLLM v0.28.0 is still the latest release, 2026-08-26 — we are current)
- **PR #37190** (`--moe-expert-cache-size`, LFRU): still OPEN, head `c66868c8`, mergeable=dirty, last activity 2026-08-28. Since our Sept-2 look: ported to main's `RoutedExperts` refactor by yasinyaman (GB10 validated), `expert_map` contract (global→slot, −1 absent), piecewise CUDA graphs (no more `--enforce-eager`). Still **BF16 + FP8 per-tensor only** (Triton fused_moe path), **not** FLASHINFER_TRTLLM / ModelOpt NVFP4. Maintainer (mgoin) concerns were graphs + eviction correctness; both addressed, still unmerged; a public "caching is meaningless for decode" argument (guqiong96) vs papers/field reports (thecodacus llama.cpp fork +21–78%).
- **RFC #38256** comments (Aug 3–30) — the most useful field data:
  - yasinyaman: on unified memory a `cudaHostRegister`-ed pool row is kernel-readable → zero-copy beat the H2D fill path **4.3× at C8** (OLMoE, GB10). Confirms UVA zero-copy is the right base on Grace-class hardware.
  - MorrisZJ (Nokia **WiSP**, vLLM plugin, arXiv 2606.21868): speculative expert prefetch is **net-negative at bs=1** (competes with mandatory reads on the same link); routing signal is worth more as a *sizing/residency* signal than a latency lever; joint expert-cache/KV sizing gives 1.07–1.19×.
  - ysys143: FreeToken's `lru_ensure` Triton kernel does lookup/evict/admit on-device, CUDA-graph capturable (avoids the ~6× host-side prepare floor).
- `#41447` (--moe-gpu-prefetch) stale/duplicate; `#37824` RIY runtime expert pruning (rebased July, unmerged — pruning, so gated by rule); `#36796` GH200 UMA offload crash (NUMA mode, not CDMM — we already avoided it).

### Other engines
- **FreeToken** (FlashML-org): issue #22 — builds and serves from source on GB10 aarch64 with `TVM_FFI_CUDA_ARCH_LIST` pin, zero patches (two independent confirmations Aug 25–26; jlacroix82/freetoken-gb10-guide). Supported models include **GLM-5.2 NVFP4** (same `glm_moe_dsa` family), GLM-5.3-Flash, DeepSeek-V4-Flash; full GLM-5.3 not listed (unverified whether the loader accepts it). Its `offload` backend = GPU LRU expert slot-cache with on-device admission; on GB10 (no separate HBM) it is "actively harmful" — on GB300 (separate HBM + C2C) it is exactly the right shape. CPU-expert path is scalar on aarch64 (irrelevant: we want GPU-over-C2C, not CPU compute). Paper number: 14.9 tok/s full GLM-5.2 NVFP4 on ONE RTX PRO 6000 96 GB over PCIe; our box has 2.6× the cache capacity and ~8× the miss bandwidth.
- **llama.cpp**: full GLM-5.3 GGUF arch is `glm-dsa` (unsloth UD-IQ4_XS 365 GB, UD-Q4_K_XL 467 GB; AesSedai IQ4_XS 367 GB w/ imatrix; emwesoft NVFP4-MTP-GGUF 473 GB). thecodacus `perf` fork = VRAM-resident hot-expert cache, **bit-identical** hot/cold merge, profile-driven (`llama-moe-trace`), stacks with MTP: Qwen3.6-35B 42→74 tok/s on a 3060; wired for `qwen35moe`/`deepseek2`/`laguna` only. Upstream GLM-5.3-*Flash* PRs open (#27752/#27773/#27917); no full-GLM-5.3 cache wiring. Q4_K_XL is *bigger* than NVFP4 (no fit gain); IQ4_XS is 4.25 bpw imatrix — nominally ≥4-bit but a different quality tier than calibrated NVFP4 → needs a KLD/blind gate before it counts as neutral.
- **SGLang** 0.5.18 (Aug 22): HiCache L2/DCP/Kimi-K3 — nothing new for selective expert offload; OffloaderV2 remains whole-module (1.4 tok/s measured). **TensorRT-LLM**: weight streaming excludes plugin weights, NVFP4 MoE requires the plugin → no. **KTransformers**: x86 AMX, no Grace path; CPU compute of the cold half on 72 Neoverse cores shares the same 396 GB/s LPDDR5X the GPU reads — cannot beat GPU-over-C2C on the same bytes (Astra concurs).

### Checkpoints (HF, sizes = sum of weight files)
| repo | format | GB | fits 225 GB HBM? | note |
|---|---|---|---|---|
| incoai / RadixArk / Inferact / local-inference-lab GLM-5.3-NVFP4 | ModelOpt NVFP4 | 465 (433 GiB) | no (~240 to offload) | ours; catid's recipes now pin local-inference-lab@cca10d1 (metadata-audit only) |
| nota-ai/GLM-5.3-Nota-NVFP4-Global-Pruned-17.75 | compressed-tensors NVFP4, 17.75% experts pruned | 392 | no | pruned → quality gate required |
| davidsyoung/GLM-5.3-EXL3-TR3 3.0/3.25/3.42 bpw | EXL3 TR3 (TP4 rank-sliced) | 316/339/355 | no | sub-4-bit + format lineage → out |
| jarrelscy NVFP4-AQLM-hybrid | mixed AQLM | 310 | no | sub-4-bit → out |
| unsloth UD-IQ4_XS / AesSedai IQ4_XS | GGUF 4.25 bpw imatrix | 365/367 | no | llama.cpp only; gate needed |
| unsloth UD-IQ2_M … UD-IQ1_S | GGUF | 239 … 217 | borderline | IQ2/IQ1 → out by rule |
| incoai/GLM-5.3-DFlash2 @425aa61 | 2B drafter, block 8 | 4.9 | — | v4 showed it loses on offloaded MoE |
Base MTP head (`num_nextn_predict_layers: 1`) survives in the NVFP4 checkpoints (layers.78.*).

### Community reference points
- catid 2× GB300 (all-HBM): SGLang TP2+EP2 DFlash2 K7 C1 **code 165.5 / prose 107.4**; vLLM PP2 K4 C1 code 154.9 / prose 106.3; 64K cold prefill 8–26k tok/s. Our v5 prose C1 33.3 = **31% of his two-Station prose figure** (the blog's "21%" compares prose against his code number — correct it).
- TheAhmadOsman (Feb 2026): GLM-5.2 NVFP4 on a Station at 256k ctx ≈ 32 tok/s (independent single-box offload datapoint ≈ ours).
- alecqfong: DSF on one Station 300 C1 / 1875 agg — different model class, not comparable.

### Astra consult (gpt-6-astra via openai-codex, verified usage file; raw: work/glm53-big-optimization-astra/out.txt sha 6dc61ccf…)
Verdict: "pursue measured placement, not engine tourism." Ranking: (1) right-size KV to real occupancy, (2) trace-driven static placement — but NOT config-only (fused tensors need slice-level support), (3) dynamic cache (higher risk; simulate static vs dynamic on identical traces first), (4) small-lookahead speculation only after profiling verification, (5) CPU expert compute / engine migration lowest. Byte fraction ≠ access fraction; the 229 GB/s is inferred not measured; single best experiment = replay real routed-expert accesses through the real UVA kernel while sweeping outstanding work — plateau = roof, rising-with-parallelism = latency-bound at C1. Do NOT: treat n=1 as convergence; treat reasoning_effort=low as quality-neutral; equate Q4_K with NVFP4; extrapolate DeepSeek popularity to GLM; retry whole-module offload; port #37190 blindly. My disposition: ACCEPT all; the C8-overshoot observation above is our answer to its question 2 without a new microbenchmark, but nvbandwidth still gets run.

## 2. The campaign

Constraints: big GLM cannot co-reside with production DSF on :30003 → every experiment lives in the approved 22:00→08:00 window; ~50 min per relaunch (17 min load + ~30 min FlashInfer autotune per new config shape; same-shape restarts ~3 min via the mounted cache); 4–5 experiments per night realistic. Tonight's window is already owned by the DSF SPS-requal campaign (`dsfv-spsrq-p0` running, hard-stop 06:00/06:15 PDT) → first big-GLM night is 2026-09-05 22:00 at the earliest. Bench = `/home/milo/bench_big.py` (warm C1/C4/C8, 512 tok, prose) + a code prompt + `spec_metrics.py` where relevant. Every candidate: A→B→A bracket if the delta is <15%; tools smoke + reasoning separation + greedy-equivalence vs v5 at temp 0 on a 20-prompt fixture before it can be called a win; Hermes nonce agent loop before it can be called an endpoint.

### Phase 0 — measure before touching anything (night 1, ~2 h)
- **E0.1 nvbandwidth** on the Station (build from github.com/NVIDIA/nvbandwidth, 5 min): `host_to_device_memcpy_ce`, `host_to_device_memcpy_sm`, `device_to_host_*`. Gives THIS box's achievable GPU-reads-host number. Decides whether the C1 headroom is 1.3× or 1.6×.
- **E0.2 routing trace**: relaunch v5 + `--enable-return-routed-experts` (verify the flag does not change the autotune key — it should not touch MoE shapes) and hook `RoutedExpertsCapturer.capture` with a ~20-line sitecustomize monkeypatch that accumulates per-layer expert histograms + a per-layer LRU-hit simulator (capacities 64/96/128/160 of 256) and dumps JSON every 500 steps. Corpus: ~100k generated tokens across prose, code, and a replayed Hermes tool session (the workload that matters), with `reasoning_effort` low AND max (thinking tokens route differently). Deliverables: (a) top-50% expert share per layer, (b) static-placement hit rate at the v5 byte budget, (c) LRU hit rate at the same budget, (d) per-domain stability (prose vs code vs tool profile overlap). This single night answers whether Phase 2 is worth 2 weeks. Bench numbers from this run also serve as the n=3 baseline (v5 currently has n=1 per cell).
- **E0.3 prefix-cache check** (same relaunch): nonce-at-start 30k prompt → turn-2 append TTFT; confirm hits with the DSA 64-token block size. Agent wall-time lever; no decode effect.

### Phase 1 — zero-engineering relaunches (night 1–2, one each)
- **E1 KV right-sizing** (Astra #1): `--max-num-seqs 2 --kv-cache-memory ~5 GiB` (2×65k) or `--max-model-len 32768 --max-num-seqs 4`; drop `--cpu-offload-gb` 200 → ~172. Expected +10–14% C1 (linear in offloaded bytes: 30.0→33.3 came from 245→200). Cost: C8 fan-out. Ship as a second launcher ("interactive profile") rather than replacing v5.
- **E2 MoE kernel backend A/B under UVA**: `--moe-runner-backend flashinfer_cutlass` / cutedsl / triton-emulation NVFP4 paths (`fused_moe/experts/` has trtllm, flashinfer_cutlass, cutedsl, nvfp4_emulation). Different kernels = different memory-access patterns over zero-copy; the trtllm cubin is opaque and we cannot raise its outstanding-load count, but another kernel might already be more latency-tolerant. Expect anything from −30% to +30%; one relaunch each; stop at the first that beats v5 by >10% on both prose and code. (SGLang's cutlass path died on the 2048-intermediate padding; vLLM's may not.)
- **E3 gather-to-HBM instead of zero-copy** (small patch, 3–5 days, only if nvbandwidth `memcpy_ce` ≫ our measured 222 GB/s): copy-engine `cudaMemcpyAsync` of the 8 routed experts (8 contiguous 20 MB slabs) into an HBM staging buffer, then run the kernel from HBM with a remapped `expert_map`. Requires the *modular* TRTLLM NVFP4 path (router outside the kernel; exists for the EP/EPLB case) so topk_ids are known before the expert GEMM. Ceiling ≈ 0.44 ms/layer × 42 = 18.7 ms + resident ≈ 46 tok/s at 360 GB/s. Note yasinyaman measured the opposite on GB10 — GB10 has no HBM; on GB300 the CE path is a legitimate hypothesis, not a repeat.

### Phase 2 — popularity-aware placement (gated on E0.2; 1–2 weeks engineering)
- **If static skew is strong** (top-50% experts ≥ ~70% of routings on our corpus, stable across domains): implement hot/cold split per offloaded layer in vLLM — resident `[H,…]` slice + pinned `[E−H,…]` slice, two kernel launches with complementary `expert_map`s, fp32 combine before the final cast. Quality: bit-exact if the partial sums are combined before bf16 rounding (llama.cpp fork does this); yasinyaman's expert-split rounds per group → document exactly which we ship. Placement chosen offline from the E0.2 profile (merged prose+code+tool, per llama.cpp's "one merged profile within 1% of specialists" finding). Payoff per the table: 50% hit → ~66 tok/s C1; 65% → ~94.
- **If skew is weak but temporal locality is strong** (LRU sim ≫ static at the same capacity): dynamic cache. Options in order: FreeToken (below, Lane B1), #37190 lineage port to trtllm NVFP4 (weeks; the `expert_map` contract now matches what trtllm's modular path already consumes), WiSP plugin (check NVFP4 support first).
- **If neither** (uniform routing): stop — physics says ~46 tok/s is the single-Station ceiling at ≥4-bit, and the honest post is "second Station or accept 33–46."

### Phase 3 — speculative decode, re-opened only after Phase 2
Native MTP K=1–2 (no drafter weights, head already in the checkpoint) with `spec_metrics.py` acceptance; win condition per Astra: draft+verify time < emitted tokens × baseline token time, measured on unique-expert bytes per emitted token. Do not re-run DFlash2 K7.

### Lane B — other engines (parallel, does not block A)
- **B1 FreeToken on the Station** (1–2 days): source build for sm_103 aarch64 following the GB10 recipe (`TVM_FFI_CUDA_ARCH_LIST="10.3"`), `--moe-backend offload` with the NVFP4 checkpoint; first check whether the loader accepts full GLM-5.3 (GLM-5.2 is supported and same arch). Gates: tools/reasoning parse via its OpenAI-compatible API, greedy equivalence vs v5, then bench. This is the one "different engine" with a real chance because its slot cache + on-device LRU is exactly the missing mechanism and it already speaks NVFP4.
- **B2 llama.cpp + thecodacus cache fork** (1–2 weeks, lower priority): needs `glm-dsa` wiring in the fork, a Blackwell/Grace CUDA build, and — because Q4_K_XL is bigger than NVFP4 — either the emwesoft NVFP4-GGUF (unverified that llama.cpp CUDA executes NVFP4 blocks) or an IQ4_XS quality gate (KLD vs bf16 + blind agent tasks). Its bit-identical hot/cold merge and profile tooling are the reference design for Phase 2 regardless.
- **B3** SGLang/TRT-LLM/KTransformers: closed for now; re-check SGLang monthly for a selective expert offloader.

### Blog corrections to make regardless (no new runs needed)
1. "~21% of catid's two-Station PP2 figure" → prose C1 33.3 vs his prose 106–107 = ~31%; code-vs-code would need our code number (33.2) vs his 155–165 = ~21%. State both.
2. Add the physics paragraph: why "all experts in HBM" cannot happen at ≥4-bit on one Station (2.3 bits/param), and that the real lever is hot-expert residency, pending the routing profile.
3. Add the C1-vs-C8 bandwidth table (C1 ~56–62% of LPDDR5X peak) as the evidence the C1 wall is latency, not link bandwidth.
4. Cold-prefill table is present; keep.

## 3. What NOT to do (merged Astra + mine)
- No IQ2/IQ1/AQLM/EXL3-2bpw "fits in HBM" experiments — rule violation, and the TR3 artifacts are TP4 formats anyway.
- No DFlash2 K7 / MTP K5 re-runs on the offloaded model.
- No SGLang OffloaderV2 layer-group sweeps; no whole-layer `prefetch` backend.
- No CPU-compute-of-experts port (KTransformers/scalar FreeToken CPU path) on Grace.
- No claiming a kernel-backend win from a single C1 cell; bracket it and judge at C8 too.
- Do not treat `reasoning_effort=low` bench numbers as a quality-neutral serving default; it is the bench convention, not a recommendation.
- Do not run any of this against :30003 outside the approved window.
