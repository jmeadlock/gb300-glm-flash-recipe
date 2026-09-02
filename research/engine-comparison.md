# Engine comparison: full GLM-5.3 753B ModelOpt NVFP4 on one DGX Station GB300

Date: 2026-09-02  
Target: full GLM-5.3 / `glm_moe_dsa` checkpoint, not GLM-5.3-Flash.  
Local constraint: one DGX Station GB300, one CUDA-visible B300/GB300 GPU with ~269 GB decimal HBM plus Grace LPDDR over NVLink-C2C/CDMM.

## Executive conclusion

| Engine | Verdict for **the existing 433 GB ModelOpt NVFP4 checkpoint on one GB300** | Why |
|---|---|---|
| **SGLang** | **Only proven path today, but experimental/slow.** Keep as the baseline; improve only by reducing offload traffic or moving to active-expert streaming. | Current SGLang `main` has native `GlmMoeDsaForCausalLM`/DSA code, ModelOpt FP4/NVFP4 MoE kernels, GLM parsers, and OffloaderV2. Local patched run serves correct tool calls but needs CUDA graphs disabled and reaches only **1.41 wall tok/s**. |
| **vLLM** | **Most plausible next experiment, but not yet proven locally.** It has the needed upstream ingredients after early-2026 merges: GLM adaptation, ModelOpt FP4/NVFP4, and name-selective/prefetch offload. | vLLM recipes target multi-GPU all-HBM serving. The new offloader can target `w13_weight`/`w2_weight` and prefetch groups, which matches the single-GB300 memory problem better than generic `--cpu-offload-gb`; likely needs GLM-specific offload parameter inclusion for scale tensors and a real GB300 smoke. |
| **TensorRT-LLM** | **Best all-GPU Blackwell engine, not a current single-GPU Grace-offload answer.** | Official docs support GLM-5/5.2/5.3, CUDA graphs, MTP, disaggregation, and NVFP4 with `moe_config.backend: TRTLLM`, but docs/examples assume enough GPU HBM and do not expose a SGLang/vLLM-style single-GPU CPU expert prefetch/offload recipe. DWDP is multi-rank/disaggregated, not a drop-in one-Station host expert pager. |

Bottom line: **do not replace the SGLang baseline with TensorRT-LLM for this exact one-GB300/offloaded target.** Try vLLM only as a controlled experiment because it has recently merged selective + prefetch offload primitives. A practical speedup likely requires finer-than-layer transfer: keep attention/dense/shared pieces in HBM and move only routed-expert payloads needed for the current layer/token, or fit a more aggressive non-ModelOpt quantized artifact entirely in HBM.

## Model facts that drive the decision

Primary model/architecture facts:

- Z.ai's GLM-5.3 card says GLM-5.3 is the same base model as GLM-5.2 with post-training gains, lists local serving via SGLang/vLLM/Transformers/etc., and reports model size as **753B parameters**.[zai-glm53]
- Hugging Face Transformers' `GlmMoeDsaConfig` at `huggingface/transformers@e15d467466e5cef29b16340887918dfd40c94dbe` defines `model_type = "glm_moe_dsa"`, `num_hidden_layers = 78`, `n_routed_experts = 256`, `num_experts_per_tok = 8`, `n_shared_experts = 1`, `moe_intermediate_size = 2048`, `hidden_size = 6144`, `max_position_embeddings = 202752`, and default first dense layers / sparse DSA layers.[hf-config]
- SGLang's GLM-5.3 cookbook describes full GLM-5.3 as **MoE + DSA + 256 routed experts top-8 + MTP**, 1,048,576-token target context, FP8 recommended deployment, BF16 around 1.5 TB, and RadixArk/ModelOpt NVFP4 as **experimental**, quantizing only routed-expert linear weights/activations while attention, shared experts, dense layers, MTP, embeddings, and LM head remain unquantized; it says that cuts weights to ~0.45 TB for a **4-GPU GB300 node**.[sgl-cookbook]
- NVIDIA's GLM-5-NVFP4 ModelOpt card is not the exact GLM-5.3 checkpoint, but it is the closest first-party ModelOpt-card evidence for the format: ModelOpt **v0.42.0**, 744B total / 40B active, Blackwell-only, supported runtime engines **vLLM and SGLang**, test hardware **B300**, and post-training quantization limited to linear operators inside transformer-block MoE.[nvidia-glm5-nvfp4]

Local measured checkpoint facts to preserve in future comparisons:

- Existing full checkpoint is **433 GB**, `glm_moe_dsa`, 78 layers, 256 routed experts, 8 active/token.
- Patched SGLang latest loaded with OffloaderV2 `group=3,num=2,prefetch=1`, `modelopt_fp4`, `flashinfer_trtllm`, and `--disable-cuda-graph`.
- Load took ~1747 s / 29 min; HBM weight memory **163.82 GB**; FP8 KV **47.69 GB / 1,073,536 tokens**; spare **37.31 GB**.
- Protocol smoke passed: `/v1/models`, parsed GLM tool call with `glm47`, reasoning separated with `glm45`, no markup leak.
- C1 wall probe: **64 output tokens / 45.42 s = 1.41 tok/s**.

## SGLang

### Architecture and quantization support

SGLang is the only path already verified locally for this exact checkpoint class.

Current source context used for this report: `sgl-project/sglang@862a909a085f68f2e87a14e5bb7f8c930612c955` (`HEAD` observed 2026-09-02). The GLM-5.3 support line is recent and still moving:

- PR **#36507** (“GLM-5.3-Flash support”) opened 2026-08-26. Its first visible commit `0b9c38484e78a43bca6419416f170aafb70180bb` says it was squashed from a Z.ai GLM-5-next branch onto `main@27c36368...`, adding a hybrid MoE architecture: MLA attention, DSA sparse attention with KPool indexer, KDA linear attention, mHC, native NEXTN draft layer, multimodal serving, FP8 weights/BF16 KV, verified on **4x GB300 TP4/EP4** and **8x H100 TP8/EP8**.[sgl-pr36507]
- The same PR includes follow-up commits around ModelOpt FP4 / shared expert handling; one visible commit `fe236ea6...` fixes ModelOpt FP4 shared-expert swiglu-fusion for GLM-5-family `swiglu_limit=10.0` cases.[sgl-pr36507]
- SGLang docs for GLM-5.3 say SGLang auto-selects DSA attention backends (`flashmla_sparse` prefill, `fa3` decode, `sgl-kernel` indexer top-k) and FP8 KV on Blackwell, routing DSA through the TensorRT-LLM backend.[sgl-cookbook]

For NVFP4 MoE:

- SGLang PR **#15022** (“fixed trtllm nvfp4 backend for moe”), merged as `ef908aeb401dcd80d0252415bb5d2d63d17bf5e8`, states that the `flashinfer_trtllm` backend for NVFP4 quantized models had bugs and fixes ModelOpt FP4 config compatibility, per-16-block NVFP4 weight-scale layout, 2D hidden-state scales to TRT-LLM, routing bias dtype handling, and routing method type handling.[sgl-pr15022]
- In current `modelopt_quant.py`, `ModelOptNvFp4Config.get_name()` returns `modelopt_fp4`; the implementation creates `w13_weight`, `w2_weight`, `w13_weight_scale`, `w2_weight_scale`, `w13_weight_scale_2`, `w2_weight_scale_2`, `w13_input_scale`, and `w2_input_scale` for NVFP4 MoE, then converts those into NVFP4 MoE kernel format and constructs the backend kernel.[sgl-modelopt]
- The same source has mixed/layer-wise ModelOpt handling for `NVFP4`, `W4A16_NVFP4`, `FP8`, and `MXFP8`, relevant because GLM NVFP4 checkpoints usually leave non-routed-expert components in BF16/FP8.[sgl-modelopt]

### CPU/expert offload granularity

SGLang has two materially different offload systems:

1. `--cpu-offload-gb` / **OffloaderV1** is generic parameter offload. Current `offloader.py` offloads parameters in module order until the byte budget is reached; it can offload part of a module and wraps the whole module forward.[sgl-offloader] Locally this offloaded MLA attention weights and failed at first forward (`w_kc` on CPU). For `glm_moe_dsa`, V1 is not appropriate.
2. **OffloaderV2** is layer-group offload. Current `offloader.py` creates V2 when `offload_group_size > 0`, explicitly rejects simultaneous `cpu_offload_gb`, groups layers, and offloads the last `offload_num_in_group` layers in each group. It prefetches `offload_prefetch_step` groups ahead on a separate `offload` CUDA stream.[sgl-offloader]

The GLM/DeepSeek model code passes a precise `submodule_accessor` for `layer.mlp.experts` and a whitelist creator for routed-expert tensors: `w13_weight`, `w2_weight`, and NVFP4 scale aliases if present.[sgl-deepseek]

Local patch requirement:

- With `flashinfer_trtllm`, the offloader whitelist and actual registered parameters disagree for this checkpoint. SGLang current code around GLM/DeepSeek still whitelists `w13_blockscale_swizzled` / `w2_blockscale_swizzled` when those attributes exist.[sgl-deepseek]
- The ModelOpt FP4 MoE implementation registers canonical parameters such as `w13_weight_scale` / `w2_weight_scale` and uses those through conversion.[sgl-modelopt]
- Local one-line patch changed the V2 whitelist from the swizzled aliases to `w13_weight_scale` / `w2_weight_scale` for this path. This is narrow and source-grounded; it should become an upstream issue/PR if we keep this route.

Offload modes in current SGLang source are `cpu`, `shm_cpu`, `sharded_gpu`, plus `meta` internally; `shm_cpu` and `sharded_gpu` assert `tp_size == 1` and use a naive-distributed rendezvous.[sgl-offloader]

### CUDA graph compatibility

SGLang's high-performance GLM path is designed to benefit from CUDA graphs. The GLM-5.2 optimization blog explicitly cites making DSA draft-extend CUDA-graphable, removing syncs, and enabling graph-friendly speculative decode as part of a >500 TPS 8xB300 result.[sgl-glm52-blog]

However, **SGLang OffloaderV2 + CPU expert offload for this full 433 GB checkpoint is not graph-compatible in the local run**:

- Current offloader source avoids starting onload while a CUDA stream is capturing (`torch.cuda.is_current_stream_capturing()` branch in `_ModuleOffloader.start_onload`) and hooks module forward by swapping parameter/buffer dicts through `wait_and_get_device_tensors()`.[sgl-offloader]
- The patched local run loaded fully, then asserted inside `Offloader.wait_and_get_device_tensors()` during prefill graph capture. Relaunching with `--disable-cuda-graph` served successfully.

Recommendation: for the full offloaded model, treat `--disable-cuda-graph` (or explicit disabled decode/prefill graph backends on newer SGLang) as required until an upstream offload+graph fix is proven.

### Likely performance and maturity

- All-HBM, multi-GPU GLM NVFP4 in SGLang is mature enough for public recipes: SGLang docs publish GLM-5.3 recipes, and the GLM-5.2 SGLang blog reports >500 TPS on 8xB300 for GLM-5.2 NVFP4 agentic workloads, using `--quantization modelopt_fp4`, FP8 KV, GLM parsers, EAGLE, and CUDA graph decode flags.[sgl-cookbook][sgl-glm52-blog]
- Single-GB300 **CPU expert offload** is not mature: the local path required a source patch and graph disable and produced only 1.41 tok/s.
- Catid's full GLM-5.3 public result is not comparable to one-GB300 offload: it uses **2x DGX Station GB300** with full-size `incoai/GLM-5.3-NVFP4@54e5252` and DFlash2, reporting SGLang TP2+EP2 C1 code 165.5 tok/s and vLLM PP2 DFlash2 C1 code 154.9 tok/s.[catid-glm53]

SGLang maturity score for this target: **B for all-GPU/multi-GPU GLM; C-/D+ for one-GPU Grace-offloaded full GLM.** It works, but the working path is a research hack rather than a production engine mode.

## vLLM

### Architecture and quantization support

Current source context used for this report: `vllm-project/vllm@76ba32160a501e5e8aadb5e1820c51c765714220` (`HEAD` observed 2026-09-02). Relevant landed changes:

- vLLM commit **`978a37c`** (“[Model] GLM adaptation (#34124)”) adds `GlmMoeDsaForCausalLM` to benchmarks/tests/model registry, maps it to the `deepseek_v2` implementation, marks `glm_moe_dsa` as a DeepSeek-MLA architecture, and adds `class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM): pass`.[vllm-glm-commit]
- Current `deepseek_v2.py` still implements GLM as that subclass of DeepSeekV2, so support depends on the DeepSeek-V2/V3/MLA code path rather than a fully separate GLM implementation.[vllm-deepseek]
- vLLM's official recipe says **vLLM 0.28.0 or newer** plus `transformers>=5.15.0` for GLM-5.3; FP8 recipes target 8xH200/H20/B200, and the Blackwell NVFP4 recipe uses `Inferact/GLM-5.3-NVFP4`, TP8, `--enable-expert-parallel`, `--kv-cache-dtype fp8_e4m3`, `glm47` tool parser, and `glm45` reasoning parser.[vllm-recipe]

For ModelOpt/NVFP4:

- vLLM PR **#20101** adds ModelOpt Qwen3 NVFP4 support and documents using `LLM(..., quantization="modelopt_fp4")`; it was merged into main as `8d765e2...` according to the PR page.[vllm-pr20101]
- Current `modelopt.py` recognizes `NVFP4` and `W4A16_NVFP4` as `modelopt_fp4`, supports KV-cache scale loading, creates NVFP4 MoE weights/scales, converts them to kernel format, and contains a guard that routes checkpoints declaring `NVFP4` but without input activations through `W4A16_NVFP4` because otherwise the W4A4 path can fold uninitialized input scales and zero experts.[vllm-modelopt]
- The vLLM recipe says Inferact's GLM-5.3-NVFP4 quantizes only MoE expert linears while shared experts, attention, embeddings, and early dense layers stay BF16.[vllm-recipe]

### CPU/expert offload granularity

vLLM has both older UVA offload and newer prefetch offload.

Current `vllm/config/offload.py` defines:

- **UVA offload**: `cpu_offload_gb` plus optional `cpu_offload_params`. If no params are named, vLLM offloads non-selectively until the GB limit; if params are named, unmatched parameters are not offloaded. Docs note this uses CPU-pinned memory / zero-copy access and requires a fast CPU-GPU interconnect.[vllm-offload-config]
- **Prefetch offload**: `offload_group_size`, `offload_num_in_group`, `offload_prefetch_step`, and `offload_params`. It groups every N layers, offloads the last M layers of each group, and uses async H2D prefetch to hide transfer latency. If `offload_params` is empty, it offloads all parameters of each offloaded layer.[vllm-offload-config]
- `arg_utils.py` exposes these as `--cpu-offload-gb`, `--cpu-offload-params`, `--offload-group-size`, `--offload-num-in-group`, `--offload-prefetch-step`, and `--offload-params`.[vllm-arg-utils]

Recent PR evidence:

- vLLM PR **#34535** (“Selective CPU Weight Offloading”), merged as `b37b679770aade27f33d20c93bf467c6a7fba65d`, adds `--cpu-offload-params` so only named parameter segments are offloaded. The PR's test plan explicitly targets **Kimi K2 NVFP4 on one GB300**, offloading only `w13_weight w2_weight`; its benchmark improves single-user throughput from ~15.5 tok/s to ~31.6 tok/s.[vllm-pr34535]
- vLLM PR **#29941** (“offloader v2: Hide weight onloading latency via prefetching”) states it adapts SGLang's GB200 offload technique, supports torch.compile and CUDA graph in vLLM, and gives a GB200 DeepSeek FP4 recipe using `--offload-group-size 2 --offload-num-in-group 1 --offload-prefetch-step 1 --offload-params w13_weight w2_weight`.[vllm-pr29941]

This is better aligned than SGLang V1 and comparable to SGLang V2 conceptually. The caveat for this GLM checkpoint: SGLang's local failure shows that **NVFP4 scale tensors matter**. A vLLM experiment should not assume `w13_weight w2_weight` alone is enough; it should inspect vLLM's registered GLM MoE parameter names and include scale tensors if prefetch offload requires them resident with weights.

### CUDA graph compatibility

vLLM's prefetch offload PR explicitly claims the adapted offloader supports torch.compile and CUDA graph within vLLM.[vllm-pr29941] That is a major difference from the local SGLang result where graph capture failed. Still, this is PR-level evidence on DeepSeek FP4/GB200, not a verified GLM-5.3 full checkpoint run on a single Station.

Recommendation: first vLLM trial should keep CUDA graphs at default for a minimal smoke, then repeat with graph disabled/capture-size reduced if the offloader fails during capture. Do not assume SGLang's graph failure transfers directly; do not assume vLLM's DeepSeek graph claim proves GLM.

### Likely performance and maturity

Most likely performance classes:

- **All-GPU/multi-GPU**: mature enough to test. vLLM official GLM-5.3 recipe covers FP8 and NVFP4 on Blackwell, but its published requirements and commands assume TP8/all-HBM class hardware.[vllm-recipe]
- **One-GB300 with prefetch expert offload**: promising but unproven. The Kimi K2 NVFP4 one-GB300 PR result (~31 tok/s after selective offload) shows the mechanism can be useful on similar hardware, but Kimi K2's architecture/working set is not GLM-5.3 DSA and the PR command used `--load-format dummy` for the test launch shown.[vllm-pr34535]
- **Full GLM-5.3 public data**: Catid's vLLM full GLM-5.3 result is **2x Station PP2 + DFlash2**, not one-Station offload, with C1 code 154.9 tok/s and C64 1,093.8 aggregate.[catid-glm53]

Required patch/experiment checklist before considering vLLM viable:

1. Use `vllm>=0.28.0` / current main with GLM adaptation and ModelOpt FP4 support.[vllm-recipe][vllm-glm-commit]
2. Launch with `--quantization modelopt_fp4` if the checkpoint does not auto-detect; keep `--tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice`.[vllm-recipe]
3. Use prefetch offload, not generic UVA first: `--offload-group-size 3 --offload-num-in-group 2 --offload-prefetch-step 1 --offload-params ...` mirroring the working SGLang memory split.
4. Include exact parameter names after inspecting vLLM's live `named_parameters()` for the GLM MoE modules: likely `w13_weight`, `w2_weight`, and possibly `w13_weight_scale`, `w2_weight_scale`, `w13_weight_scale_2`, `w2_weight_scale_2`, `w13_input_scale`, `w2_input_scale` depending on offloader implementation.
5. Start with conservative `--max-model-len` / `--max-num-seqs` to prove load + forward before a 1M KV budget.
6. Verify protocol, not just throughput: model list, `tools` request parses into `tool_calls`, `reasoning_content` separated, no `<tool_call>` or `<think>` leak.

vLLM maturity score for this target: **B for official multi-GPU GLM; C for single-GB300 offload.** It may overtake SGLang after a successful smoke because its offloader claims CUDA-graph compatibility and has selective parameter targeting, but there is no evidence yet that it loads this exact 433 GB GLM checkpoint on one Station.

## TensorRT-LLM

### Architecture and quantization support

Current source/doc context used for this report: `NVIDIA/TensorRT-LLM@acc5633df96585ecd04e57cd0f16cf5488b2d9b9` (`HEAD` observed 2026-09-02).

Official TensorRT-LLM docs are strong for GLM-5 family support:

- The GLM-5 deployment guide says it covers **GLM-5, GLM-5.2, and GLM-5.3**, and that GLM-5.3 is a weight update over GLM-5.2 with the same architecture and code path.[trt-glm-guide]
- It says GLM-5 uses **MLA with DeepSeek Sparse Attention (DSA)**, shares the same architecture as DeepSeek V3.2 with minor changes, and is served through `GlmMoeDsaForCausalLM`, reusing DeepSeek-V3.2 components.[trt-glm-guide]
- TensorRT-LLM's supported-models matrix lists `GlmMoeDsaForCausalLM` for GLM-5, GLM-5.2, GLM-5.3, with features: overlap scheduler yes, CUDA graph yes, disaggregated serving yes, chunked prefill yes, MTP yes, DFlash no, TLLM C++ sampler untested.[trt-supported]
- The GLM guide's Blackwell MoE matrix says B200/GB200 FP8 uses `DEEPGEMM`, while B200/GB200 NVFP4 uses `TRTLLM`; default MoE backend is `CUTLASS` and must be set explicitly for GLM.[trt-glm-guide]
- The NVFP4 config section says to point to the NVFP4 checkpoint and set `moe_config.backend: TRTLLM`; **MTP is not currently supported with the NVFP4 checkpoint**.[trt-glm-guide]

This is the cleanest official support story for all-GPU Blackwell GLM inference. It is not, by itself, evidence of one-GPU host expert offload.

### CPU/expert offload granularity

I found no official TensorRT-LLM GLM guide recipe equivalent to SGLang/vLLM `--offload-group-size` or `--cpu-offload-params` for one GPU plus Grace LPDDR expert storage.

What exists is different:

- TensorRT-LLM supports disaggregated serving for GLM (separate context/prefill and generation/decode workers) in the official guide.[trt-glm-guide]
- PR **#12136** adds DWDP / distributed weight-data-parallelism support for MoE inference. The PR summary describes cross-rank weight prefetching / ping-pong buffering / IPC handles for disaggregated MoE inference, i.e. a multi-rank distributed mechanism, not a documented single-rank CPU expert offload knob.[trt-pr12136]
- PR **#8880** integrates CuteDSL NVFP4 grouped GEMM for the CuteDSL MoE backend and supports B200/GB200 NVFP4, but it is a kernel/backend integration, not a host-offload recipe.[trt-pr8880]
- TensorRT-LLM examples include KV-cache host offload, but KV offload is not the same as weight/expert offload for a 433 GB checkpoint.[trt-kv-offload]

Therefore, for the exact one-GB300 target, TensorRT-LLM would likely require either:

1. an undocumented internal/experimental weight streaming mode,
2. adapting DWDP-like prefetch to a single-rank Grace memory source, or
3. a smaller quantized artifact that fits entirely in HBM.

None of those are a current documented drop-in route.

### CUDA graph compatibility

TensorRT-LLM is the strongest CUDA-graph story for GLM in the all-GPU case:

- The GLM guide lists CUDA Graph among tested GLM-5 features.[trt-glm-guide]
- The supported-model matrix says `GlmMoeDsaForCausalLM` has CUDA Graph support.[trt-supported]
- The guide exposes `cuda_graph_config` with options such as `enable_padding` and `max_batch_size`.[trt-glm-guide]

But this should not be extended to host expert offload. There is no documented single-GPU expert prefetch/offload path whose graph compatibility can be judged. If we implemented one, it would need its own capture-safety audit like the SGLang failure showed.

### Likely performance and maturity

- **All-GPU Blackwell GLM:** high maturity. TensorRT-LLM official docs are first-party NVIDIA docs for GLM-5/5.2/5.3 and list tested features.[trt-glm-guide][trt-supported]
- **NVFP4 GLM:** supported in docs with `TRTLLM` MoE backend, but MTP unavailable for NVFP4 checkpoints; this hurts single-user latency compared with SGLang/vLLM DFlash/MTP paths.[trt-glm-guide]
- **One-GB300 full 433 GB checkpoint:** no credible current path without new work. Since the weight footprint exceeds HBM, pure TensorRT-LLM serving should fail to fit unless it can use an undocumented offload path or a different quantization.

TensorRT-LLM maturity score for this target: **A-/B+ for all-GPU Blackwell GLM; D for one-GB300 host-expert offload.** It is probably the wrong immediate investment unless NVIDIA exposes/lands a single-rank Grace-memory expert prefetch mode.

## Comparative details

| Dimension | SGLang | vLLM | TensorRT-LLM |
|---|---|---|---|
| Exact architecture | Native/recent GLM path plus DeepSeek/DSA machinery; GLM-5.3 PR still active/moving.[sgl-pr36507] | `GlmMoeDsaForCausalLM` mapped to `deepseek_v2`; commit `978a37c` adds registry/MLA/spec support.[vllm-glm-commit] | Officially documents `GlmMoeDsaForCausalLM` for GLM-5/5.2/5.3; reuses DeepSeek-V3.2 components.[trt-glm-guide][trt-supported] |
| ModelOpt NVFP4 | `--quantization modelopt_fp4`; current code creates/loads NVFP4 MoE weights/scales and converts kernel format.[sgl-modelopt] | `modelopt_fp4`; current code recognizes `NVFP4`/`W4A16_NVFP4`, creates/loads NVFP4 MoE weights/scales.[vllm-modelopt] | Docs say NVFP4 GLM on Blackwell requires `moe_config.backend: TRTLLM`; default CUTLASS is wrong.[trt-glm-guide] |
| Tool/reasoning parsers | SGLang GLM docs require `glm47` tool parser and `glm45` reasoning parser; local smoke confirms.[sgl-cookbook] | vLLM recipe uses `--tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice`.[vllm-recipe] | GLM model support documented; separate parser support is improving, but the cited GLM guide is mostly engine config. Verify tool parsing before agent use. |
| Offload granularity | V1 generic byte-budget is unsafe for DSA/MLA; V2 groups layers and targets `layer.mlp.experts` tensors, prefetching ahead.[sgl-offloader][sgl-deepseek] | UVA: byte-budget plus optional parameter name filter. Prefetch: group layers + named offload params + async H2D; PRs explicitly target MoE FP4 on GB200/GB300.[vllm-offload-config][vllm-pr34535][vllm-pr29941] | No documented single-GPU CPU expert offload knob. DWDP is distributed/disaggregated MoE weight prefetch, not a simple one-Station mode.[trt-pr12136] |
| CUDA graphs with offload | Fails locally for this checkpoint; disable graphs required. All-GPU GLM optimization is graph-heavy.[sgl-glm52-blog] | PR #29941 claims prefetch offload supports torch.compile and CUDA graph, but not proven for GLM-5.3 one-GB300.[vllm-pr29941] | Official GLM docs list CUDA Graph support, but no host-weight-offload graph story.[trt-supported] |
| Known required patches | Local patch: offload whitelist must use canonical `w13_weight_scale` / `w2_weight_scale`, not swizzled aliases, for `flashinfer_trtllm`; disable CUDA graph. | Unknown until tested. Likely need exact `--offload-params` list including scales, and maybe ModelOpt GLM scale/weight mapping fixes if SGLang bug has an analog. | Would require new single-rank Grace/expert offload work or an all-HBM artifact. |
| Performance expectation on one GB300 | Proven **1.41 tok/s** wall C1 with whole-module expert prefetch; production Flash remains much faster. | Unknown; plausible improvement over SGLang if graph-compatible prefetch and better parameter targeting work. Kimi K2 NVFP4 one-GB300 PR showed ~31 tok/s with selective offload, but not GLM.[vllm-pr34535] | Should be fast if weights fit HBM; they do not. No one-GB300 offload number found. |
| Maturity for this exact target | Proven but hacky. | Promising but unproven. | Not currently viable for this memory shape. |

## Recommended next actions

1. **Keep SGLang patched baseline as the reference artifact**, but label it experimental and slow. It is the only engine that has actually served the checkpoint on one GB300.
2. **Run a read-only vLLM feasibility prep before touching runtime:** inspect current vLLM GLM MoE `named_parameters()` with a tiny/dummy config or offline model init if possible; determine exact `--offload-params` needed for `w13/w2` plus scales. Then plan one controlled load smoke on the GB300 later.
3. **Do not spend near-term time on TensorRT-LLM for single-Station offload** unless a new NVIDIA branch/doc exposes single-rank Grace expert offload. TensorRT-LLM is the better all-GPU/multi-GPU production engine, not the current memory-oversubscription solution.
4. **If performance matters more than preserving this exact checkpoint format**, investigate an all-HBM artifact (EXL3/other lower-bpw routed-expert quant) or an active-expert streamer. Whole-layer/group expert prefetch over Grace LPDDR is the likely reason for 1.41 tok/s.
5. **For any future engine smoke, verify agent protocol first:** GLM `tools` request must yield structured `tool_calls`, `reasoning_content` must separate, and no raw `<tool_call>` / `<think>` markup may leak.

## Source index

[catid-glm53]: https://github.com/catid/dgx_station_benchmarks/blob/main/glm-5.3/README.md  
[hf-config]: https://github.com/huggingface/transformers/blob/e15d467466e5cef29b16340887918dfd40c94dbe/src/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py  
[nvidia-glm5-nvfp4]: https://huggingface.co/nvidia/GLM-5-NVFP4  
[sgl-cookbook]: https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3  
[sgl-deepseek]: https://github.com/sgl-project/sglang/blob/862a909a085f68f2e87a14e5bb7f8c930612c955/python/sglang/srt/models/deepseek_v2.py  
[sgl-glm52-blog]: https://www.lmsys.org/blog/2026-07-13-glm52-optimization  
[sgl-modelopt]: https://github.com/sgl-project/sglang/blob/862a909a085f68f2e87a14e5bb7f8c930612c955/python/sglang/srt/layers/quantization/modelopt_quant.py  
[sgl-offloader]: https://github.com/sgl-project/sglang/blob/862a909a085f68f2e87a14e5bb7f8c930612c955/python/sglang/srt/utils/offloader.py  
[sgl-pr15022]: https://github.com/sgl-project/sglang/pull/15022  
[sgl-pr36507]: https://github.com/sgl-project/sglang/pull/36507  
[trt-glm-guide]: https://github.com/NVIDIA/TensorRT-LLM/blob/acc5633df96585ecd04e57cd0f16cf5488b2d9b9/docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md  
[trt-kv-offload]: https://github.com/NVIDIA/TensorRT-LLM/blob/acc5633df96585ecd04e57cd0f16cf5488b2d9b9/examples/llm-api/llm_kv_cache_offloading.py  
[trt-pr12136]: https://github.com/NVIDIA/TensorRT-LLM/pull/12136  
[trt-pr8880]: https://github.com/NVIDIA/TensorRT-LLM/pull/8880  
[trt-supported]: https://github.com/NVIDIA/TensorRT-LLM/blob/acc5633df96585ecd04e57cd0f16cf5488b2d9b9/docs/source/models/supported-models.md  
[vllm-arg-utils]: https://github.com/vllm-project/vllm/blob/76ba32160a501e5e8aadb5e1820c51c765714220/vllm/engine/arg_utils.py  
[vllm-deepseek]: https://github.com/vllm-project/vllm/blob/76ba32160a501e5e8aadb5e1820c51c765714220/vllm/model_executor/models/deepseek_v2.py  
[vllm-glm-commit]: https://github.com/vllm-project/vllm/commit/978a37c  
[vllm-modelopt]: https://github.com/vllm-project/vllm/blob/76ba32160a501e5e8aadb5e1820c51c765714220/vllm/model_executor/layers/quantization/modelopt.py  
[vllm-offload-config]: https://github.com/vllm-project/vllm/blob/76ba32160a501e5e8aadb5e1820c51c765714220/vllm/config/offload.py  
[vllm-pr20101]: https://github.com/vllm-project/vllm/pull/20101  
[vllm-pr29941]: https://github.com/vllm-project/vllm/pull/29941  
[vllm-pr34535]: https://github.com/vllm-project/vllm/pull/34535  
[vllm-recipe]: https://recipes.vllm.ai/zai-org/GLM-5.3  
[zai-glm53]: https://huggingface.co/zai-org/GLM-5.3
