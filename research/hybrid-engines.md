# Hybrid MoE inference engines for full GLM-5.3 on one GB300

## Scope and baseline

Target: full `zai-org/GLM-5.3`, not the smaller GLM-5.3-Flash variant. The official model card reports 753B parameters and lists SGLang, vLLM, TokenSpeed, Transformers, KTransformers, and Unsloth as local serving paths.[1] SGLang's GLM-5.3 cookbook describes the full model as `glm_moe_dsa`, 78 layers, 256 routed experts, top-8 active per token, 1,048,576-token context, and an MTP layer.[24] The SGLang page also says the BF16 checkpoint is about 1.5 TB, FP8 is the recommended deployment, and experimental NVFP4 quantizes only routed-expert linears/activations while leaving attention, shared experts, dense layers, MTP, embeddings, and lm head unquantized, yielding about a 0.45 TB weight footprint for a 4-GPU GB300 node.[24]

Given the local constraint of one Grace-Blackwell GB300 with about 269 GB CUDA HBM, Grace LPDDR, and CDMM, a 433-465 GB ModelOpt/RadixArk/Inferact-style NVFP4 full GLM-5.3 checkpoint cannot be loaded as a normal all-GPU model.[2][24] The current measured SGLang path copies whole offloaded FusedMoE modules and only reaches about 1.41 tok/s with 2/3 layers offloaded; a viable single-GB300 path therefore needs one of: (a) expert weights resident in Grace memory with only active experts transferred to GPU, (b) cold experts computed on Grace CPU fast enough to hide behind GPU work, or (c) a more aggressive routed-expert-only quantization format that fits HBM.

## Executive result

No surveyed engine is a drop-in, production-ready solution for full GLM-5.3 753B NVFP4 on one GB300 today. The closest research directions are:

1. **KTransformers/SGLang-KT conceptually matches the need** because it is designed for heterogeneous MoE with CPU-resident experts and SGLang API serving, but the published GLM-5.3 support is for GLM-5.3-Flash FP8 and its documented CPU expert kernel is AVX-512 FP8, not Grace ARM.[26] The full GLM-5.2 tutorial is closer architecturally and uses `--kt-num-gpu-experts`, dynamic expert update, `--kt-method FP8/BF16`, and SGLang tool/reasoning parsers, but I found no source confirming full GLM-5.3 NVFP4 on Grace ARM.[6]
2. **ik_llama.cpp has the most explicit active-expert transfer primitive**: its `--offload-only-active-experts` path copies only activated experts from RAM to GPU for MoE prompt processing, and docs expose tensor overrides to keep routed experts in RAM while fixed/shared parts stay in VRAM.[12][13] The blocker is model/format maturity for full GLM-5.3 DSA/NVFP4 GGUF and performance: the GLM-5.3 work surfaced in llama.cpp is for GLM-5.3-Flash, and the active-expert PR notes gains can disappear when a large batch activates most experts.
3. **EXL3/vLLM-fork can fit by changing the weight format**, not by active-expert offload. ExLlamaV3 lists GLM-5.2 and GLM-5.3-Flash architecture support and supports tensor/expert parallelism plus an OpenAI-compatible TabbyAPI path.[14] The vLLM-EXL3 plugin claims serving-proven GLM-5.3-Flash on vLLM fork lineages and packed routed experts that are never dequantized, but GLM-5.3 support remains fork-only until the architecture exists upstream.[15] This is promising for a quantized/forked route, not for running the existing 433 GB ModelOpt NVFP4 full checkpoint.
4. **Aphrodite/Sonar supports `GlmMoeDsaForCausalLM` and NVFP4 generally**, but I found no evidence of active-expert offload or single-GB300 full GLM-5.3 operation. Its release notes add GLM-4.5 and NVFP4, and the model registry includes `GlmMoeDsaForCausalLM`; it is still fundamentally a vLLM-style all-GPU serving stack unless using generic Transformers fallback.[17][18]
5. **PowerInfer, FlexGen, and classic DeepSpeed are not practical as-is** for this model. PowerInfer's public implementation says it only supports ReLU/ReGLU/Squared-ReLU models, which excludes GLM-5.3's SwiGLU-style MoE path.[20] FlexGen targets dense OPT-style offloading and throughput batching, not modern DSA/MoE expert routing.[21] DeepSpeed-MoE supports expert/tensor parallel inference, but not GLM-5.3 DSA/NVFP4 or active-expert HBM streaming for a single Grace-Blackwell node.[22]

## Comparison matrix

| Engine | Full `glm_moe_dsa` / GLM-5.3 support | NVFP4 support | Offload behavior | Grace ARM / GB300 fit | API/tool calling | Realistic performance on one GB300 |
|---|---|---|---|---|---|---|
| KTransformers standalone / SGLang-KT | GLM-5.2 full tutorial; GLM-5.3-Flash native tutorial; no full GLM-5.3 NVFP4 proof found.[1][6][26] | FP8/BF16 documented; GLM-5.3-Flash reads official FP8 directly. | Heterogeneous experts, GPU-expert counts, dynamic updates, and placement strategies. | AVX-512 FP8 CPU expert kernel is documented; Grace ARM likely needs new kernel or transfer-only mode. | SGLang OpenAI endpoint with `glm47`/`glm45` parsers. | Best conceptual fit, but unproven for full GLM-5.3 NVFP4 on one GB300. |
| SGLang-KTransformers integration | SGLang/KT hybrid MoE integration is tracked and described.[8][9] | Not proven for full GLM-5.3 NVFP4; SGLang native supports GLM-5.3 NVFP4 experimentally.[24] | Places GPU experts and runs/offloads the rest via KT CPU kernels. | Same KT ARM caveat; SGLang's documented NVFP4 target is TP4 4-GPU GB300, not one GPU. | Yes: SGLang OpenAI/Anthropic APIs and GLM parsers. | Prototype only if KT can load full GLM-5.3 NVFP4 or converted FP8 on Grace. |
| llama.cpp mainline | PR adds GLM-5.3-Flash (`GLM-5-Next`), not full GLM-5.3.[10] | PR adds `GGML_TYPE_NVFP4`, ModelOpt HF-to-GGUF repacking, CPU and ARM NEON support.[11] | Mainline has GPU layer/tensor offload and lazy expert mmap work, but not the cited ik active-expert policy. | ARM NEON NVFP4 exists, so Grace build is plausible; full GLM-5.3 graph maturity remains the blocker. | llama-server has OpenAI-like chat/completions; GLM tool parser not established. | No measured full-753B single-GB300 report found. |
| ik_llama.cpp | Docs mention GLM-5 guidance and MoE tensor overrides, but full GLM-5.3 DSA correctness is not proven.[13] | MXFP4 is documented; NVFP4 likely needs llama.cpp merge/conversion work.[11] | `--offload-only-active-experts` copies only activated experts from RAM to GPU for prompt processing.[12] | Attractive for CDMM, but Grace ARM CPU MoE performance is unknown. | Has llama.cpp-family server modes; no GLM `glm47` parser evidence. | Most relevant offload idea; decode remains likely CPU-limited. |
| ExLlamaV3 / EXL3 | README lists GLM-5.2 (`GlmMoeDsaForCausalLM`) and GLM-5.3-Flash (`Glm5NextForConditionalGeneration`), not full GLM-5.3 753B explicitly.[14] | Not NVFP4; requires EXL3 conversion/quantization. | Quantized packed routed experts on GPU; not active-expert offload. Supports tensor/expert parallelism. | Prebuilt wheels shown are Linux x86_64 CUDA; Grace ARM likely needs source build and extension work if no aarch64 wheel.[14] | TabbyAPI provides OpenAI-compatible server.[14] | Good if a full GLM-5.3 EXL3 pack can be produced under ~269 GB HBM; no full-753B single-GB300 measured report found. |
| vLLM-EXL3 / Sparkinfer forks | `Glm5Next` GLM-5.3-Flash is serving-proven; full GLM-5.3 remains fork-only until architecture exists upstream.[15] | Not NVFP4 for routed experts; EXL3 can compose with non-routed FP8/BF16.[16] | Packed EXL3 experts run through `exllamav3_ext`, never dequantized; no CPU expert offload. | Blackwell CUDA focus; no one-GB300 full model report found. | vLLM-compatible OpenAI endpoint in published GLM-5.2 EXL3 runtime. | GLM-5.2 EXL3 TP4 reports 48.5 tok/s C1; full GLM-5.3 NVFP4 two-GB300 PP2 reports 154.9 code tok/s C1.[16][25] |
| Aphrodite / Sonar | Model registry includes `GlmMoeDsaForCausalLM`, and release adds GLM-4.5; no GLM-5.3 cookbook or measured full model report found.[17][18] | Release notes add NVFP4 Blackwell datatype support.[17] | No active-expert offload found; supports expert load balancing/EP, but assumes enough accelerator memory. | CUDA Blackwell plausible; single GB300 insufficient for 0.45 TB all-GPU checkpoint. | vLLM-style OpenAI API; tool-parser plugin system exists but fetched docs URL was 404, so GLM-5.3 `glm47` parser support is unverified. | Not useful for one-GPU active offload without implementing a hybrid expert backend. |
| PowerInfer | Does not support GLM-5.3; public FAQ says only ReLU/ReGLU/Squared ReLU activation models are supported.[20] | No NVFP4 / GLM-DSA support found. | Hot/cold sparse neuron CPU-GPU split, not GLM MoE routed experts. | Originally consumer GPU/x86; not a GB300/Grace path without major port. | Not an OpenAI production serving path for GLM-5.3. | Not viable. |
| DeepSpeed-MoE / ZeRO-Inference | DeepSpeed-MoE supports tensor slicing for non-experts and expert parallelism/slicing for experts, but not GLM-5.3 DSA/NVFP4 in sources.[22] | No ModelOpt NVFP4 GLM-5.3 support found. | Parallelizes experts across devices; ZeRO-style offload is parameter offload, not per-token active expert streaming. | Historically x86 GPU clusters; GB300 ARM possible only through PyTorch/CUDA stack, but kernels/model support missing. | No GLM-specific OpenAI/tool parser serving path by default. | Not a practical path versus SGLang/vLLM/TRT-LLM. |
| FlexGen | Dense OPT-oriented offloading engine; repo describes high-throughput generation via compression/offloading/scheduling.[21][23] | No NVFP4 / GLM DSA support found. | Offloads weights/KV/cache across GPU/CPU/disk for large batches, not MoE active experts. | Not a modern Blackwell GLM engine. | Not a GLM tool-call serving stack. | Not viable for this target. |

## Engine notes

### KTransformers and SGLang-KT

KTransformers is the only surveyed project whose public direction matches the desired architecture: use SGLang for serving while replacing MoE execution with heterogeneous CPU/GPU expert kernels.[8][9] The GLM-5.2 tutorial launches `sglang.launch_server` with `--kt-weight-path`, `--kt-cpuinfer`, `--kt-num-gpu-experts`, `--kt-enable-dynamic-expert-update`, `--kt-expert-placement-strategy`, `--tool-call-parser glm47`, and `--reasoning-parser glm45` for both FP8 and BF16 modes.[6] The expert scheduling tutorial is relevant because GLM-5.3 has 256 experts and top-8 routing; keeping hot experts on HBM and cold experts in Grace memory is the right shape of solution.

The problem is target mismatch. The native GLM-5.3 tutorial I found is GLM-5.3-Flash, a 321B hybrid linear/sparse-attention model with 34 linear-attention layers and 11 DSA layers, not full GLM-5.3 753B.[26] It says KT reads official GLM-5.3-Flash FP8 directly, no conversion, occupies about 306 GiB, and supports AVX-512 FP8 CPU expert kernels, SM89/SM120 GPUs, heterogeneous CPU-GPU expert inference, Layerwise Prefill, multimodality, and tool calling.[26] On a Grace ARM CPU, AVX-512 is absent; unless KT has an undocumented ARM CPU expert path or can use CDMM as a fast staging area for GPU execution of active experts, this is not directly deployable on a single GB300.[26]

### llama.cpp and ik_llama.cpp

llama.cpp has active work around both GLM and NVFP4, but the cited GLM-5.3 PR is for GLM-5.3-Flash (`GLM-5-Next`) and notes correctness caveats such as needing `NVIDIA_TF32_OVERRIDE=0` and `-fa off` at the time of the PR.[10] Its NVFP4 PR is highly relevant because it adds a GGML NVFP4 type, detects ModelOpt NVFP4 models in HF-to-GGUF conversion, repacks them, and includes CPU plus ARM NEON backend pieces.[11] That makes llama.cpp/ggml one of the few paths with an ARM-friendly NVFP4 story.

The most relevant active-expert mechanism is in ik_llama.cpp, not mainline llama.cpp. PR #698 says that for hybrid CPU/GPU MoE inference, when experts are stored in RAM only activated experts are copied to the GPU; it only affects prompt processing, because experts stored in RAM are never copied to GPU for token generation.[12] ik docs also expose `--offload-only-active-experts`, `--cpu-moe`, `--n-cpu-moe`, and tensor override patterns for GLM-style deployments, with fixed/shared pieces kept in VRAM and routed experts in RAM.[13] For this GB300 use case, that means ik can reduce prefill bandwidth but may still be decode-limited by Grace CPU expert execution unless token-generation active transfer is extended.[12]

### EXL3 / ExLlamaV3 and vLLM-EXL3 forks

EXL3 is best viewed as an alternate compression path, not an offload path. ExLlamaV3's README lists GLM-5.2 (`GlmMoeDsaForCausalLM`) and GLM-5.3-Flash (`Glm5NextForConditionalGeneration`) support, flexible tensor/expert parallel inference, and an OpenAI-compatible TabbyAPI server.[14] The vLLM-EXL3 plugin is more serving-oriented: it registers `--quantization exl3`, runs routed MoE experts packed through `exllamav3_ext` kernels without dequantizing to dense at load, and says `Glm5Next` GLM-5.3-Flash is serving-proven on fork lineages with `RoutedExperts`.[15]

For full GLM-5.3, this would require producing or finding a full-model EXL3 derivative whose packed routed experts plus unquantized components fit under one GB300's HBM, and likely porting/forking model loader support if upstream vLLM lacks the full architecture.[15] Published EXL3 performance is encouraging but not directly comparable: the GLM-5.2 EXL3 TP4 Blackwell workstation card reports 48.5 tok/s C1 and 266.8 tok/s aggregate at C8 on four RTX PRO 6000 Blackwell GPUs, while catid's full GLM-5.3 NVFP4 report gives two-GB300 numbers, not one-GB300 offload numbers.[16][25]

### Aphrodite / Sonar

Aphrodite/Sonar tracks vLLM-style model and quantization support. Its v0.9.0 release lists NVFP4 as a new Blackwell datatype and GLM-4.5 among new models.[17] Its registry includes `GlmMoeDsaForCausalLM`, which is the architecture family needed for GLM-5.x DSA.[18] However, I found no source showing active-expert offload, Grace-memory expert staging, or full GLM-5.3 NVFP4 on a single GB300. It may be useful as a fast all-GPU baseline on larger systems, but it does not solve the 433-465 GB weights versus 269 GB HBM problem.

### PowerInfer, DeepSpeed, and FlexGen

PowerInfer is architecturally interesting because it exploits activation sparsity and CPU-GPU split execution, but the public project FAQ says it only supports models with ReLU/ReGLU/Squared ReLU activations.[20] GLM-5.3's routed expert MLP path is not in that supported class, and PowerInfer does not handle GLM DSA or ModelOpt NVFP4.

DeepSpeed-MoE is a cluster parallelism engine rather than a modern single-node active-expert streamer. Its tutorial states that it uses data parallelism and tensor slicing for non-expert parameters plus expert parallelism and expert slicing for expert parameters.[22] That maps to multi-GPU/multi-node MoE inference, not one GB300 with host-memory expert paging. FlexGen is even farther away: the paper/repo describe high-throughput generation for large dense models by offloading/compressing tensors across GPU, CPU, and disk, primarily for OPT-style models.[21][23]

## Practical recommendation for the GB300 recipe

1. **Do not spend time on PowerInfer, FlexGen, or stock DeepSpeed** for full GLM-5.3; they would be research ports before they answer the immediate question.
2. **Prototype KTransformers/SGLang-KT only after checking ARM support in code.** The necessary experiment is not a full model run; first verify whether KT's CPU expert kernels compile/run on aarch64 Grace or whether the code can stage active experts from LPDDR/CDMM to SM120 GPU kernels without x86 AVX-512.[26]
3. **Prototype ik_llama.cpp active-expert transfer if a full GLM-5.3 GGUF/NVFP4 conversion loads.** Its `--offload-only-active-experts` is the cleanest existing implementation of active expert H2D transfer, but the current behavior helps prompt processing more than decode, and GLM-5.3 DSA correctness/performance must be proven.[10][11][12]
4. **Keep EXL3 as the fallback compression route.** If a full GLM-5.3 EXL3 routed-expert pack can bring weights under 269 GB while preserving enough quality, it avoids Grace CPU expert latency. That is a different artifact than the existing 433 GB ModelOpt NVFP4 checkpoint.[14][15]
5. **Use published SGLang/vLLM/TRT-LLM numbers as upper bounds, not expectations.** SGLang documents full GLM-5.3 NVFP4 as a 4-GPU GB300 experimental target with benchmark data pending, and catid's measured full GLM-5.3 NVFP4 numbers use 2x DGX Station GB300, not one.[24][25] NVIDIA and vLLM recipes target 8xB200/B300-class all-GPU configurations for high performance.

## Bottom line

For the current single-GB300 system, the most plausible path to improve over whole-module SGLang offload is to borrow **ik_llama.cpp's active-expert transfer policy** or **KTransformers' SGLang-integrated expert scheduler**, but neither is proven for full GLM-5.3 NVFP4 on Grace ARM. A real implementation plan should start with small-scope loader/kernel probes: full GLM-5.3 config load, one MoE layer routed-expert placement, active-expert transfer timing over CDMM/LPDDR to SM120, and token-generation behavior. If those probes fail, the only credible near-term path is a new quantized artifact such as EXL3/other routed-expert-only compression that fits in HBM.

## Additional context from surveyed sources

NVIDIA's TensorRT-LLM guide confirms GLM-5 uses MLA with DSA, reuses a DeepSeek-V3.2-style code path, and requires explicit MoE backend choices for FP8 versus NVFP4 on Blackwell.[3]

SGLang's GLM-5.2 NVFP4 optimization report is useful as a performance ceiling because it reports more than 500 TPS on 8xB300, IndexShare/MTP optimizations, and a recipe using `--quantization modelopt_fp4` plus GLM parsers.[4]

The KTransformers repository itself advertises GLM-5.2 day-zero support and GLM-5.3-Flash support, which frames KT as active but not proof of full GLM-5.3 NVFP4 on Grace.[5]

KTransformers' expert-scheduling tutorial is the relevant design reference for GPU-resident hot experts, CPU-resident cold experts, dynamic updates, and placement strategies.[7]

The GLM-4.5/SGLang blog establishes the GLM parser lineage: GLM-4.5 used `glm45` for tool and reasoning parsers, while GLM-5.3's current docs require the newer `glm47` tool parser and `glm45` reasoning parser.[19][24]

## Sources

[1] https://huggingface.co/zai-org/GLM-5.3
[2] https://recipes.vllm.ai/zai-org/GLM-5.3
[3] https://nvidia.github.io/TensorRT-LLM/deployment-guide/deployment-guide-for-glm-5-on-trtllm.html
[4] https://www.lmsys.org/blog/2026-07-13-glm52-optimization
[5] https://github.com/kvcache-ai/ktransformers
[6] https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/GLM-5.2-Tutorial.md
[7] https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/experts-sched-Tutorial.md
[8] https://www.lmsys.org/blog/2025-10-22-KTransformers
[9] https://github.com/sgl-project/sglang/issues/11425
[10] https://github.com/ggml-org/llama.cpp/pull/27754
[11] https://github.com/ggml-org/llama.cpp/pull/19769
[12] https://github.com/ikawrakow/ik_llama.cpp/pull/698
[13] https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/parameters.md
[14] https://github.com/turboderp-org/exllamav3/blob/master/README.md
[15] https://github.com/vcruz305/vllm-exl3
[16] https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw
[17] https://github.com/dphnAI/sonar/releases/tag/v0.9.0
[18] https://sonar.dphn.ai/reference/models
[19] https://lmsys.org/blog/2025-07-31-glm4-5
[20] https://arxiv.org/abs/2312.12456
[21] https://arxiv.org/abs/2303.06865
[22] https://github.com/deepspeedai/DeepSpeed/blob/master/docs/_tutorials/mixture-of-experts-inference.md
[23] https://github.com/FMInference/FlexGen
[24] https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3
[25] https://github.com/catid/dgx_station_benchmarks/blob/main/glm-5.3/README.md
[26] https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md
