# Alternative paths for full GLM-5.3 753B on one DGX Station GB300

Date researched: 2026-09-02

Scope: full `zai-org/GLM-5.3` / `glm_moe_dsa`, not GLM-5.3-Flash. Baseline provided by the local experiment: `incoai/GLM-5.3-NVFP4` / ModelOpt NVFP4, 433 GiB, expert offload through patched SGLang OffloaderV2, correct protocol, **C1 1.41 tok/s**. Production comparison point: GLM-5.3-Flash NVFP4 + DFlash2 is **218-234 tok/s C1** on the same Station.

## Executive answer

There is no single-Station drop-in that turns full GLM-5.3 753B into a Flash-like local model today. The credible paths are, in order:

1. **Use a full-model DFlash2 or MTP path only after fixing the memory/offload bottleneck.** A real full GLM-5.3 DFlash2 drafter exists (`incoai/GLM-5.3-DFlash2`) and publishes TP4 GB300 numbers, but speculative decoding accelerates target forwards; it cannot erase a 433 GiB offloaded target that spends most decode time moving or computing experts from Grace memory.[5] On all-GPU TP4, DFlash2 gives roughly **2.16-3.39× C1** and **1.64-2.84× C32 aggregate** versus autoregressive depending on task.[5]
2. **Prototype EXL3/TR3 ≥3-bit full GLM-5.3 if the runtime target is allowed to move from SGLang stock to a forked EXL3 stack.** Full `glm_moe_dsa` EXL3 artifacts exist at 3.0, 3.25, and 3.42 bpw, and the 3.25/3.42 cards report low KLD against a BF16 teacher with FP8 KV.[10][11] The catch: these are explicitly not loadable by vanilla ExLlamaV3; they require an exllamav3-b12x / SparkInfer-lineage stack and mixed-K patches.[10]
3. **Use NVFP4-preserving expert pruning if quality loss is acceptable and two GPUs or a future single-GPU pruning target is acceptable.** Nota's 17.75% globally expert-pruned NVFP4 checkpoint is real and keeps 15,792 of 19,200 routed experts with per-layer variable counts, but it is advertised for **2×B300**, not one GB300, and requires a patched vLLM model file.[12]
4. **Investigate active-expert transfer / hot-expert caching as an implementation project, not an existing GLM-5.3 solution.** ik_llama.cpp has an actual `--offload-only-active-experts` idea, but its author says it affects prompt processing only, not token generation, and can disappear when a batch activates most experts.[17] This is the right mechanism to borrow, but not a usable full GLM-5.3 serving path yet.
5. **Hardware scale-out is the only high-confidence performance path.** SGLang and vLLM recipes target TP4/TP8+ for GLM-5.3 NVFP4/FP8, and catid's public full-GLM-5.3 measurements use two DGX Stations across 400 GbE RoCE / GPUDirect RDMA, not one offloaded Station.[3][4][20]

James explicitly does not want IQ2-class recommendations. GGUF IQ1/IQ2 artifacts exist, but they are excluded from recommended paths; the only GGUF path worth considering here is **≥3-bit** and only if llama.cpp/ik_llama.cpp proves full `glm_moe_dsa` correctness and useful GPU execution on GB300.

## What exists for `glm_moe_dsa` as of 2026-09-02

| Area | Real artifacts/support found | Single GB300 feasibility | C1 latency impact | Aggregate throughput impact |
|---|---|---:|---:|---:|
| Official FP8/BF16 | `zai-org/GLM-5.3` ships FP8; BF16 repo exists; official card lists SGLang, vLLM, TokenSpeed, Transformers, KTransformers, Unsloth.[1] | FP8 is too large for one B300 HBM without offload; BF16 is out. | Poor if offloaded; good only all-GPU. | Good only multi-GPU. |
| ModelOpt NVFP4 | `incoai/GLM-5.3-NVFP4` is 433 GiB, routed-expert linears W4A4 NVFP4, FP8 KV, served directly by SGLang and vLLM on Blackwell.[2] SGLang documents NVFP4 as experimental and ~0.45 TB.[3] | Proven to load locally only with expert offload patch; speed bad. | Current local C1 1.41 tok/s; DFlash could multiply forwards but not fix offload stalls. | With offload, likely still low; all-GPU TP4/TP8 is the intended path. |
| Full DFlash2 | `incoai/GLM-5.3-DFlash2` exists for full GLM-5.3; 2B BF16 draft, block size 8 / 7 proposals, SGLang + vLLM support.[5] | Draft fits, target is the blocker. | On all-GPU TP4: 2.16-3.39× C1 over AR depending task.[5] On current offload path: unknown, likely bounded by expert movement. | On all-GPU TP4: 1.64-2.84× at C32.[5] |
| Native MTP | SGLang GLM-5.3 docs say the checkpoint has one nextn/MTP layer and supports EAGLE-style speculative decoding with IndexShare for GLM DSA.[3] vLLM recipe enables MTP.[4] | Same target memory blocker. | All-GPU TP4: 1.83-2.63× C1 in Inco's comparison, less than DFlash2. | All-GPU TP4: 1.48-2.35× C8/C32 depending task.[5] |
| EXL3/TR3 ≥3-bit | Full `davidsyoung/GLM-5.3-EXL3-TR3-3.0bpw` and 3.25bpw exist for `glm_moe_dsa`; 3.25 quantizes routed experts as 192 K3 + 64 K4 and keeps dense/attention/shared/norm/embed/head BF16.[10][11] | Plausible if a forked stack runs on one GB300; not stock. | Could be the largest C1 jump if it fits HBM and avoids Grace expert offload; unmeasured on one GB300. | Good if batch/MTP/DCP support works; fork risk high. |
| GGUF ≥3-bit | Unsloth publishes full GLM-5.3 GGUF sizes: UD-IQ3_XXS 282 GB, UD-Q3_K_XL 343 GB, UD-IQ4_XS 365 GB, Q4_K_XL 467 GB, etc.[8] AtomicChat documents imatrix GGUF conversion and requires a llama.cpp build with GLM-5.3 support.[9] | 3-bit GGUF still exceeds visible HBM before KV unless partially offloaded; llama.cpp full-DSA maturity is less proven than SGLang/vLLM. | Likely better than current SGLang offload only if more fixed tensors stay GPU-resident and expert movement is smarter. | llama.cpp batching may help prefill; less likely to beat SGLang for throughput. |
| AWQ/GPTQ | No GLM-5.3 full AWQ/GPTQ checkpoint found in HF search; search for GLM-5.3 AWQ/GPTQ returned no direct results. Related GLM-5 AWQ/GPTQ tooling is not proof for GLM-5.3. | Not currently a path. | None until artifact+kernel exists. | None. |
| Expert pruning | Nota 17.75% global-pruned NVFP4 exists; it keeps 208-256 experts/layer and needs patched vLLM.[12] | Still advertised for 2×B300, not one. Could become single-GB300 if pruning is pushed further, with quality risk. | If it fit all-HBM, very large; otherwise still offload-bound. | Good if all-GPU; pruning reduces model work and memory. |
| Hot expert cache / active expert transfer | Existing local SGLang OffloaderV2 offloads whole modules. ik_llama.cpp implements active-expert transfer for MoE prompt processing only.[17] | Implementation project. | Decode improvement needs token-generation active transfer or Grace-side MoE kernels; not present in cited path. | Prefill/batch improvement plausible when active experts are sparse; gains vanish if batches activate many experts.[17] |
| Prompt/session radix cache | SGLang's GLM-5.3 docs discuss cache hygiene for agent clients, and SGLang exposes session-aware radix cache in recent releases per local recipe context.[3] | High for repeated agent sessions; does not fix decode. | Big TTFT/prefill win on repeated prefixes; no per-token speedup. | Improves aggregate service efficiency when requests share prefixes/sessions. |
| PLE / layerwise prefill | KTransformers GLM family docs describe heterogeneous CPU-GPU expert inference and layerwise/prefill-oriented mechanisms for GLM-5.x-family serving.[22] SGLang GLM-5.3 docs include PD disaggregation and LayerSplit for prefill workers.[3] | Useful for long-prefill/multi-worker layouts, not single-stream decode on one GPU. | May worsen short C1 decode because of extra coordination. | Can improve long-context prefill capacity/TTFT in distributed layouts. |
| Direct coherent-memory execution | GB300 has hardware-coherent CPU/GPU memory via NVLink-C2C, and CDMM lets the NVIDIA driver manage GPU memory separately while the GPU can still access system memory over NVLink-C2C.[18][19] | Hardware supports it; current LLM kernels/loaders mostly do not execute arbitrary expert GEMMs directly from Grace LPDDR at HBM-like speed. | Potentially better than PCIe offload, but current measured SGLang path remains 1.41 tok/s. | May help staging/offload; needs kernel-level work. |
| Multi-node / second Station | vLLM/SGlang docs prescribe multi-GPU; catid's full GLM-5.3 recipe uses two DGX Stations with 400 GbE RoCE NCCL/GPUDirect RDMA.[4][20] | Not one Station, but highest-confidence path. | Strongest path to usable C1. | Strongest path to high aggregate throughput. |

## Ranking by feasibility

### 1. Multi-GPU / multi-node full GLM-5.3 NVFP4 or FP8

Feasibility: **high if hardware is added; low if constrained to one Station.**

SGLang documents GLM-5.3 FP8/BF16/NVFP4 deployment around multi-GPU layouts and says BF16 needs 8×B300 or multi-node setups, while vLLM's recipe starts with 8×H200/H20/B200-class nodes and offers NVFP4 Blackwell flags for expert parallelism.[3][4] catid's public full-GLM-5.3 recipes report a distributed endpoint across two DGX Stations, one GB300 per host, using dual 400 GbE ConnectX-8 RoCE rails for NCCL over GPUDirect RDMA.[20]

Expected impact: this is the only path that attacks the root problem — the full target no longer needs Grace expert offload. C1 should move from offload-bound single-digit tok/s into the published all-GPU class; aggregate throughput should scale much more cleanly with batching than the current one-GPU offload path.

### 2. Full GLM-5.3 DFlash2 / MTP after all-GPU or near-all-GPU residency

Feasibility: **high for artifact availability; medium for single-Station usefulness.**

The full DFlash2 drafter exists and is distinct from the Flash drafter: `incoai/GLM-5.3-DFlash2` targets `zai-org/GLM-5.3`, is a 2B BF16 draft model, and has SGLang and vLLM serving examples.[5] The underlying DFlash/Spec V2 work is a real SGLang path, using block diffusion, KV injection, and overlap scheduling to reduce target-forward bubbles.[6] Its published evaluation on four GB300 GPUs shows DFlash2 beating both autoregressive and native MTP across C1, C8, and C32; at C1 it reports 244-383 output tok/s versus ~113 tok/s AR depending task.[5]

Expected impact: on an all-GPU target, DFlash2 is the biggest known C1 latency lever: about 2-3.4×. On the current one-Station offload run, the speedup may be much smaller because speculative verification still invokes the target and the target is waiting on offloaded experts. It is worth testing only after the loader avoids whole-module Grace bottlenecks or after moving to multi-GPU.

### 3. EXL3/TR3 full-model ≥3-bit route

Feasibility: **medium as a research prototype; low as a stock SGLang replacement.**

EXL3 full GLM-5.3 artifacts are real. The 3.25bpw card says it is a trellis quantization of `zai-org/GLM-5.3`, with routed experts in layers 3-78 plus MTP quantized to EXL3, dense MLP/attention/norms/embeddings/head kept BF16, and the serving stack requiring exllamav3-b12x / SparkInfer-lineage patches.[10] The public vLLM-EXL3 plugin confirms the out-of-tree design: `--quantization exl3`, packed routed experts through `exllamav3_ext`, fork lineage required, and GLM-5.3 remaining fork-only until upstream architecture support lands.[14] ExLlamaV3 itself advertises EXL3 quantization, tensor/expert parallel inference, dynamic batching, speculative decoding, and OpenAI-compatible serving through TabbyAPI, but the repo support list is not enough by itself for this full GLM-5.3 mixed-K artifact.[13]

Expected impact: if a 3.0-3.25bpw full GLM-5.3 artifact can be loaded mostly/all in the 269 GB visible HBM with FP8 KV and correct GLM parsing, it could improve C1 by avoiding Grace expert traffic entirely. That is likely a larger latency win than DFlash on top of an offloaded ModelOpt checkpoint. Throughput depends on whether the fork has working batching, MTP/DCP, sparse DSA, and Blackwell kernels; do not assume SGLang-grade scheduling.

### 4. NVFP4-preserving expert pruning / sparsification

Feasibility: **medium for two GPUs; low for one without more pruning.**

Nota's global-pruned NVFP4 GLM-5.3 exists, prunes 17.75% of experts globally rather than uniformly, keeps 15,792 of 19,200 routed experts, stores per-layer expert counts in config, and requires a patched vLLM `deepseek_v2.py` because the architecture now has variable experts per layer.[12] The model card reports Terminal-Bench 2.1 and DeepSWE retention around 94-95% versus BF16 within large single-run uncertainty, but it advertises the serving target as 2×B300.[12]

Expected impact: pruning is attractive because it reduces both memory and routed expert compute while staying close to the ModelOpt/NVFP4 path. The current 17.75% cut is not enough for one GB300 HBM, but a more aggressive one-GPU pruning variant could be more realistic than inventing a new offloader. C1 and aggregate would both improve if it becomes all-HBM; if it still offloads, benefit is proportional to fewer cold experts moved/computed.

### 5. GGUF ≥3-bit / llama.cpp / ik_llama.cpp active-expert path

Feasibility: **medium for artifacts; low-to-medium for performance.**

Unsloth publishes full GLM-5.3 GGUF tiers including UD-IQ3_XXS 282 GB, UD-Q3_K_XL 343 GB, UD-IQ4_XS 365 GB, and UD-Q4_K_XL 467 GB.[8]
AtomicChat describes an imatrix conversion from original FP8 weights and notes the model needs a llama.cpp build with GLM-5.3 support.[9]
llama.cpp's public GLM-5-Next PR is specifically for GLM-5.3-Flash, not full GLM-5.3, and its comments still mention correctness flags and performance measurements on Flash UD-IQ1_S.[15]
A separate llama.cpp NVFP4 PR adds a GGML NVFP4 type and ModelOpt HF-to-GGUF repacking with scalar CPU and ARM NEON paths, but that is quant-format support, not proof of full GLM-5.3 `glm_moe_dsa` serving.[16]

Do not recommend IQ2. IQ1/IQ2 may be useful for experiments or emergency low-memory demos, but James asked to avoid IQ2-class paths; practical research should start at 3-bit and expect quality/runtime verification.

The most useful idea here is not generic GGUF: it is active expert transfer. ik_llama.cpp's PR says that for hybrid CPU/GPU MoE inference, when experts are stored in RAM only activated experts are copied to GPU, but the author explicitly limits that behavior to prompt processing and says token generation does not copy RAM experts to GPU.[17]

Expected impact: GGUF ≥3-bit might reduce storage enough to experiment with one-Station HBM+LPDDR placement, but it probably does not beat an SGLang/vLLM all-GPU path. Active-expert transfer could improve prefill and batched prompt processing; C1 decode will remain poor unless token-generation active expert movement or Grace-optimized CPU MoE kernels exist.

### 6. SGLang/vLLM/TensorRT-LLM kernel upgrades without changing residency

Feasibility: **high to test; low to solve alone.**

SGLang's GLM-5.3 page lists DSA backend choices, ModelOpt FP4/NVFP4, FP8 KV on Blackwell, native MTP, DFlash2, DP-Attention + DeepEP, PD disaggregation, context parallelism, and LayerSplit.[3] The GLM-5.2 optimization report shows real kernel/runtime improvements: Spec V2, IndexShare MTP, DSA sync removal, TopK-V2, indexer prologue fusion, and GEMM improvements, with >500 TPS on 8×B300 for GLM-5.2 NVFP4 bs=1.[7] TensorRT-LLM also has a GLM-5 Blackwell guide and shows GLM-5 FP8 + MTP performance on 8×B200.[21]

Expected impact: these are essential for all-GPU or multi-node serving, but not enough when the model is spilling 160-270 GiB of experts into Grace LPDDR. Expect modest single-Station C1 gains unless a kernel specifically avoids whole offloaded module movement.

### 7. Prompt cache / session radix cache

Feasibility: **high for agent workloads; not a decode-speed fix.**

GLM-5.3's long context and agent-style operation make prefix reuse important. The SGLang GLM-5.3 docs call out client-side prompt stability issues for Claude Code-style traffic, including how per-request prompt divergence defeats prefix cache reuse.[3]

Expected impact: C1 steady-state decode remains 1.41 tok/s on the current offloaded target, but TTFT for repeated long agent sessions can drop dramatically because the server avoids re-prefilling stable system/history prefixes. Aggregate throughput improves when many requests share cached prefixes; it does not make cold full-model decode usable.

### 8. PLE / layerwise prefill / PD disaggregation / context parallelism

Feasibility: **medium for distributed prefill; low for one-Station interactive decode.**

SGLang documents context parallelism for GLM-5.3 prefill and says it can reduce TTFT under long context, but also introduces all-gather overhead that can increase decode latency for unified deployment or short prefill.[3] The same page mentions PD disaggregation and LayerSplit for prefill workers; KTransformers' GLM-5.2 tutorial demonstrates CPU-GPU heterogeneous inference by offloading experts to CPU while selected experts stay on GPU.[22]

Expected impact: this is a throughput and long-context TTFT lever, not a rescue for single-token decode on a one-GPU offloaded full model. It is most relevant in a two-Station or larger deployment where one worker specializes in prefill and another in decode.

### 9. Direct coherent-memory execution / CDMM / Grace LPDDR as a weight tier

Feasibility: **hardware high, software low.**

NVIDIA's CDMM blog says GB300 is hardware-coherent and CDMM makes the driver manage GPU memory instead of exposing it as OS NUMA memory, while preserving GPU access to system memory over NVLink-C2C.[18] NVIDIA's NVLink-C2C page describes high-bandwidth, coherent transfers between CPU and GPU class devices.[19]

Expected impact: this is the architectural reason a single GB300 can even attempt a 433 GiB target, but coherent addressing is not the same as HBM-resident tensor-core GEMM. To make this fast, kernels need to either stream active expert tiles efficiently from LPDDR over C2C or keep hot experts in HBM and hide cold expert movement. The current SGLang offload result is the warning sign: capacity works; latency does not.

## Recommended next experiments, without modifying runtime systems

1. **Artifact audit first:** stage no runtime changes; verify exact HF revisions and file sizes for `incoai/GLM-5.3-DFlash2`, `davidsyoung/GLM-5.3-EXL3-TR3-3.0bpw`, `davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw`, and `nota-ai/GLM-5.3-Nota-NVFP4-Global-Pruned-17.75`.
2. **EXL3 feasibility probe:** in a scratch/container-only environment, check whether the needed exllamav3-b12x/SparkInfer lineage has aarch64/SM120 wheels or can compile on GB300; do not start by downloading 600 GB.
3. **DFlash2 on current offloaded full target:** only after confirming the DFlash2 adapter works with the patched SGLang image and `glm_moe_dsa`; measure accept length and wall-clock C1. Stop if verification cycles still take tens of seconds.
4. **Expert-cache design probe:** measure router expert frequency on real James/Hermes prompts from the existing offload run. If hot experts dominate, a small HBM resident hot set plus Grace cold set is worth implementing; if top-8 spreads broadly, hot caching will not save C1.
5. **Do not spend time on IQ2, stock AWQ/GPTQ, or generic offload flags.** IQ2 is excluded by preference and likely quality risk; AWQ/GPTQ artifacts were not found for full GLM-5.3; generic SGLang V1 offload already moved attention tensors and failed locally.
6. **If full GLM-5.3 matters operationally, plan a second GB300 or rent 4×/8× Blackwell.** Multi-GPU is the high-confidence path; one-Station full GLM-5.3 remains a research systems project.

## Bottom line

The nearest realistic single-Station improvement is **not** another SGLang flag; it is either (a) a ≥3-bit EXL3 full-GLM artifact that fits enough of the model in HBM under a forked runtime, or (b) a real active/hot-expert cache that keeps frequently routed experts resident and streams only selected cold experts from Grace memory. The most reliable production answer is still hardware: run full GLM-5.3 on multiple Blackwell GPUs and use DFlash2/MTP, cache discipline, and SGLang/vLLM/TensorRT-LLM kernels there.

## Sources

[1] https://huggingface.co/zai-org/GLM-5.3
[2] https://huggingface.co/incoai/GLM-5.3-NVFP4
[3] https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3
[4] https://recipes.vllm.ai/zai-org/GLM-5.3
[5] https://huggingface.co/incoai/GLM-5.3-DFlash2
[6] https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2
[7] https://www.lmsys.org/blog/2026-07-13-glm52-optimization
[8] https://huggingface.co/unsloth/GLM-5.3-GGUF
[9] https://huggingface.co/AtomicChat/GLM-5.3-GGUF
[10] https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw
[11] https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.0bpw
[12] https://huggingface.co/nota-ai/GLM-5.3-Nota-NVFP4-Global-Pruned-17.75
[13] https://github.com/turboderp-org/exllamav3
[14] https://github.com/vcruz305/vllm-exl3
[15] https://github.com/ggml-org/llama.cpp/pull/27754
[16] https://github.com/ggml-org/llama.cpp/pull/19769
[17] https://github.com/ikawrakow/ik_llama.cpp/pull/698
[18] https://developer.nvidia.com/blog/understanding-memory-management-on-hardware-coherent-platforms
[19] https://www.nvidia.com/en-us/data-center/nvlink-c2c
[20] https://github.com/catid/dgx_station_benchmarks/blob/main/glm-5.3/recipes
[21] https://nvidia.github.io/TensorRT-LLM/deployment-guide/deployment-guide-for-glm-5-on-trtllm.html
[22] https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/GLM-5.2-Tutorial.md
