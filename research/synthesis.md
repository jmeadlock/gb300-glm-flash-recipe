# Full GLM-5.3 NVFP4 on one DGX Station GB300: engine decision

**Research date:** 2026-09-02  
**Target:** full `glm_moe_dsa` GLM-5.3, 753B parent, 256 routed experts / top-8, ModelOpt NVFP4 checkpoint (~433 GB), one DGX Station GB300 with ~269 GB CUDA-visible HBM plus Grace LPDDR5X over NVLink-C2C.

## Executive decision

**Use vLLM selective UVA expert offload as the next experiment.** Target only the fused routed-expert payloads (`experts.w13_weight` and `experts.w2_weight`) for CPU/Grace placement, and retain attention, DSA/indexer, router, shared experts, embeddings, scales, KV cache, and runtime buffers in HBM.

Do **not** spend another session tuning SGLang OffloaderV2's layer-prefetch split. That path is architecturally mismatched to sparse decode because it transfers complete offloaded expert modules. The measured 1.41 tok/s is a successful proof of fit and protocol, not a promising performance base.

Treat vLLM as a **bounded experiment**, not a predicted success. Current upstream source contains the necessary GLM architecture registration, ModelOpt NVFP4 path, exact-segment selective parameter matching, and zero-copy UVA implementation, but there is no reproducible public result for this exact GLM-5.3 checkpoint on one Station.

## Why vLLM changes the traffic model

vLLM's current `UVAOffloader` places selected parameters in CPU memory and exposes accelerator views through CUDA Unified Virtual Addressing. The GPU directly reads CPU-resident memory instead of copying the whole module before each forward. Its selective matching is segment-exact: `experts.w2_weight` matches `mlp.experts.w2_weight` but not `w2_weight_scale`. This lets the compact scales and all non-routed components remain resident in HBM. [vLLM UVA source](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/offloader/uva.py) [24] [vLLM offload config](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/config/offload.py) [25]

This matters because a fused top-k MoE kernel indexes only the routed experts. For decode, the requested memory traffic is therefore tied much more closely to top-8 expert payloads than to the complete 256-expert bank. This is the opposite of the current SGLang prefetch strategy, whose unit of movement is the offloaded fused-MoE module/layer. [SGLang OffloaderV2 PR](https://github.com/sgl-project/sglang/pull/8034) [32]

vLLM's merged selective-offload PR reported **15.5 → 31.6 tok/s** on a single GB300 for a Kimi-K2 NVFP4 configuration when offloading only `experts.w13_weight` and `experts.w2_weight`. This is not a GLM-5.3 benchmark—the PR used a different MoE and dummy weights—but it proves that selective UVA expert access can outperform indiscriminate offload dramatically on this hardware class. [vLLM PR #34535](https://github.com/vllm-project/vllm/pull/34535) [23]

Two X reports describe full GLM-5.2 NVFP4 on one GB300 at roughly **32–33 tok/s**, with attention in HBM and experts in Grace memory. They are close architectural analogues because GLM-5.3 retains the GLM-5.2 base architecture, but they are **medium-confidence anecdotes**, not reproducible recipes: one author described custom kernel patches that were not in stock upstream. [Ahmad Osman report](https://x.com/TheAhmadOsman/status/2078247891370442867) [34] [digitalix report](https://x.com/digitalix/status/2089888514485739536) [35]

Accordingly, a realistic experimental target is **prove >10 tok/s C1 while preserving protocol and quality**. The 32–33 tok/s reports are an upside reference, not a promise.

## Engine ranking

| Rank | Engine/path | Fit for one GB300 | Decision |
|---:|---|---|---|
| 1 | **vLLM selective UVA expert offload** | Best combination of native GLM/NVFP4 support, Grace-compatible zero-copy, and exact expert-only placement | **Build and test next** |
| 2 | **FreeToken hot-expert cache** | Best algorithmic design; 14.9 tok/s full GLM-5.2 NVFP4 on one RTX PRO 6000, global LRU expert cache, hybrid CPU/GPU execution | **Watch or port; currently blocked by x86_64-only support** |
| 3 | **Current patched SGLang OffloaderV2** | Proven to load full GLM-5.3 and preserve tool protocol, but whole-module prefetch produced 1.41 tok/s and CUDA graphs failed | **Keep as proof/fallback; stop tuning as primary path** |
| 4 | **TensorRT-LLM** | Excellent official GLM-5 NVFP4 kernels when weights are distributed across multiple Blackwell GPUs | **Use after adding a second GB300; not for this one-GPU offload case** |
| 5 | **llama.cpp expert-cache fork** | Correct active-expert idea and Grace ARM-friendly foundation, but GLM-5.3/NVFP4 fast-kernel support and OpenAI reasoning/tool protocol are not yet a clean path | **Secondary research branch** |
| 6 | **KTransformers/SGLang-KT** | Good CPU/GPU expert scheduling concept, but documented fast CPU kernels require AVX-512/x86, not Grace ARM | **No** |
| 7 | **EXL3/TR3 3.25–3.42 bpw** | Better quality than IQ2, but current full-model artifacts are specialized TP4 rank-sliced formats requiring a patched Sparkinfer/vLLM lineage; they still exceed one Station's HBM | **Not the first single-GB300 path** |
| 8 | **PowerInfer/FlexGen/DeepSpeed generic offload** | No current GLM DSA + ModelOpt NVFP4 + Blackwell active-expert path | **No** |

### FreeToken: best design, wrong host architecture today

FreeToken implements the architecture we actually want: full-layer double-buffered prefill streaming, a global LRU expert cache for decode, and a bandwidth-adaptive policy that divides misses between cache fill/GPU execution and CPU execution. Its paper reports **14.9 tok/s** on full NVIDIA GLM-5.2 NVFP4 using one 96 GB RTX PRO 6000, versus 7.3 tok/s for llama.cpp. [FreeToken paper](https://arxiv.org/html/2608.16157v1) [29]

The blocker is operational, not conceptual: the official installation target is Linux x86_64 and ARM64/Grace support remains an open feature request. [FreeToken ARM64 issue](https://github.com/FlashML-org/FreeToken/issues/22) [30] A Grace port is attractive medium-term because our 269 GB HBM could hold a much larger expert cache than the paper's 96 GB GPU, but no performance estimate should be claimed until ARM kernels and dependencies run.

### TensorRT-LLM: production winner only when the model is all-GPU

TensorRT-LLM officially supports the GLM-5 DSA architecture and NVFP4 through its TRTLLM MoE backend, but its published recipe requires **8× B200**. [NVIDIA GLM-5 deployment guide](https://nvidia.github.io/TensorRT-LLM/deployment-guide/deployment-guide-for-glm-5-on-trtllm.html) [21]

Its generic TensorRT weight-streaming feature is not a solution here: the documentation says weight streaming supports only **non-plugin weights** and requires GEMM plugins to be disabled, while GLM-5 NVFP4 explicitly requires the TRTLLM MoE backend/plugin. [TensorRT-LLM weight streaming](https://nvidia.github.io/TensorRT-LLM/advanced/weight-streaming.html) [31]

NVIDIA's active-expert redistribution and wide expert-parallel work is technically relevant, but is designed for multi-GPU GB200/GB300 systems with expert parallelism and redistribution across GPUs. It is not a documented one-GPU Grace expert-cache mode.

### Why not continue optimizing SGLang

The current service proves:

- the 433 GB checkpoint fits with CDMM;
- ModelOpt NVFP4 and the TRTLLM FlashInfer MoE path execute;
- tool calls and separated reasoning work;
- the current patched OffloaderV2 placement is stable with CUDA graphs disabled.

But it also proves the transfer unit is wrong for sparse decode. Reducing KV and moving another group of expert layers into HBM can only reduce the number of streamed layers discretely; it cannot convert whole-module transfers into top-k expert reads. Prefetch depth may improve overlap, and C4/C8 may improve aggregate throughput, but neither fixes C1 inter-token latency.

## Ranked experiment plan

### MUST: vLLM selective UVA proof

Build a separate pinned vLLM image from a revision containing:

- GLM `GlmMoeDsaForCausalLM` adaptation ([PR #34124](https://github.com/vllm-project/vllm/pull/34124) [28]);
- selective CPU weight offloading ([PR #34535](https://github.com/vllm-project/vllm/pull/34535) [23]);
- the current V2 model-runner offloader wiring and GB300 pin/UVA fixes.

Use explicit offload configuration equivalent to:

```text
offload_backend=uva
cpu_offload_params={experts.w13_weight, experts.w2_weight}
cpu_offload_gb=<large enough for the routed expert bank>
```

Before a full launch, run a **weight-name dry inspection** to verify that only the two packed expert payloads match. Scale tensors must remain in HBM unless a kernel proves otherwise. On Grace systems, test the current documented pin-memory workaround rather than assuming discrete-GPU behavior; vLLM source notes that pinned memory can consume GPU-visible memory on GH200-class unified-memory systems.

Acceptance gates:

1. exact 96-file checkpoint loads without conversion;
2. `/v1/models` and plain chat pass;
3. GLM reasoning and tool-call parsers pass with no markup leakage;
4. fixed 64-token C1 decode exceeds **10 tok/s** before further tuning;
5. C1/C2/C4/C8 and 8K prefill are measured;
6. FP8-KV quality warning is resolved or explicitly characterized;
7. HBM and host-memory totals prove no double allocation.

If the exact NVFP4 expert layout fails under UVA, capture the failing kernel/tensor before trying vLLM's prefetch backend. Prefetch is a fallback, not the goal, because it can regress to whole-module traffic.

### SHOULD: locate or reconstruct the 32 tok/s GLM patch

The two single-GB300 GLM-5.2 reports are too close and too large an uplift to ignore. Contact the authors for the pinned vLLM revision, Dockerfile, kernel changes, and launch command. If unavailable, diff current vLLM's GLM/FusedMoE NVFP4 path against the merged selective-UVA Kimi example.

### SHOULD: benchmark current SGLang C1/C4/C8 before stopping it

This requires no relaunch and reveals whether whole-module transfer amortizes across a batch. It does not make the model more interactive, but it determines whether the current endpoint is useful for asynchronous batch work.

### SHOULD: native MTP after an active-expert target works

GLM's native MTP/DFlash2 can multiply accepted tokens per expensive target pass, but speculative decoding only becomes high leverage once target execution is no longer dominated by loading the wrong bytes. Enable and tune MTP **after** vLLM UVA correctness and baseline throughput. The current SGLang offload path might see a limited uplift, but a 2–3× multiplier on 1.41 tok/s is still not the desired endpoint.

### LATER: port FreeToken to Grace ARM64

This is the cleanest research-development project if vLLM UVA cannot run the exact NVFP4 kernels. The likely work includes ARM64-compatible wheels for native dependencies, replacing x86 CPU kernels, validating CUDA/Triton kernels on SM100, and adapting memory registration to coherent Grace memory. Preserve FreeToken's expert cache and q-star policy rather than reducing it to another layer streamer.

### LATER: vLLM incremental expert cache

vLLM has an active RFC for a GPU expert cache and async pipeline. [vLLM expert-cache RFC](https://github.com/vllm-project/vllm/issues/38256) [26] The current implementation work has limited precision support and does not yet establish GLM-5.3 ModelOpt NVFP4 compatibility. [vLLM expert-cache PR](https://github.com/vllm-project/vllm/pull/37190) [27] Track it; do not base the immediate trial on it.

### LATER: REAP/pruned checkpoint, only with quality gates

A community `Global-Pruned-17.75` NVFP4 checkpoint could reduce a 433 GB artifact to roughly **356.14 GB** if the percentage translated directly to stored expert bytes. It would still not fit HBM, but would shrink host traffic/cache pressure. [Pruned GLM-5.3 checkpoint](https://github.com/nota-ai/GLM-5.3-Nota-NVFP4-Global-Pruned-17.75) [37] Use only after tool/reasoning/coding evaluations; expert pruning is a model change, not an engine optimization.

### NEVER for this goal

- IQ2 or similarly aggressive quantization;
- generic `--cpu-offload-gb` without exact expert parameter targeting;
- TensorRT weight streaming with the NVFP4 MoE plugin path;
- more SGLang layer-group/prefetch sweeps expecting an order-of-magnitude C1 gain;
- comparing GB10 clusters or GB300 NVL72 marketing numbers to a single Station without labeling the topology difference.

## What NVIDIA forums and X actually establish

NVIDIA forum discussion correctly describes the Station's memory as coherent but tiered: enough aggregate capacity for very large checkpoints, not 775 GB of HBM-speed memory. Users repeatedly report very low GPU utilization when generic CPU offload dominates. [NVIDIA forum memory discussion](https://forums.developer.nvidia.com/t/memory-clarification-issue-on-dgx-station-1t-llm-model-possible/368919) [38]

The strongest reproducible all-HBM comparison is Catid's work: two GB300 Stations running full GLM-5.3 NVFP4 with SGLang TP2+EP2 and DFlash2 reached **165.5 tok/s C1 code**. [Catid GLM-5.3 benchmarks](https://github.com/catid/dgx_station_benchmarks/blob/main/glm-5.3/README.md) [33] That is the production answer if another Station becomes available. It does not validate single-Station offload.

The strongest single-Station offload evidence is the pair of ~32–33 tok/s GLM-5.2 X reports, but their reproducibility is weaker. They support the hypothesis that top-k UVA/custom expert kernels can outperform layer streaming by about an order of magnitude; they do not prove stock vLLM will do so on GLM-5.3.

## Bottom line

The problem is not that the GB300 lacks aggregate memory. It is that our current engine moves weights at the **module** granularity while GLM computes at the **expert** granularity.

**Next engine:** vLLM selective UVA.  
**Target:** >10 tok/s C1 with exact NVFP4 and valid tool/reasoning protocol.  
**Upside reference:** ~32–33 tok/s, not guaranteed.  
**Best future single-GPU architecture:** FreeToken-style hot-expert cache after ARM64 support.  
**Best production solution:** a second GB300 and all-HBM TP2/EP2, where SGLang or TensorRT-LLM makes sense.

## Sources

[21] https://nvidia.github.io/TensorRT-LLM/deployment-guide/deployment-guide-for-glm-5-on-trtllm.html — TensorRT-LLM GLM-5 guide
[23] https://github.com/vllm-project/vllm/pull/34535
[24] https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/offloader/uva.py
[25] https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/config/offload.py
[26] https://github.com/vllm-project/vllm/issues/38256
[27] https://github.com/vllm-project/vllm/pull/37190
[28] https://github.com/vllm-project/vllm/pull/34124
[29] https://arxiv.org/html/2608.16157v1
[30] https://github.com/FlashML-org/FreeToken/issues/22
[31] https://nvidia.github.io/TensorRT-LLM/advanced/weight-streaming.html
[32] https://github.com/sgl-project/sglang/pull/8034
[33] https://github.com/catid/dgx_station_benchmarks/blob/main/glm-5.3/README.md
[34] https://x.com/TheAhmadOsman/status/2078247891370442867
[35] https://x.com/digitalix/status/2089888514485739536
[37] https://github.com/nota-ai/GLM-5.3-Nota-NVFP4-Global-Pruned-17.75
[38] https://forums.developer.nvidia.com/t/memory-clarification-issue-on-dgx-station-1t-llm-model-possible/368919
