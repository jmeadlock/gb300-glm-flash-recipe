# NVIDIA forums + X field evidence for single DGX Station GB300 / Grace-Blackwell MoE inference

Scope: measured or explicitly claimed inference results found on NVIDIA Developer Forums, GitHub benchmark artifacts linked from those discussions/X, and X posts surfaced by X search. Focus terms were DGX Station GB300 / Grace-Blackwell, 500B-1T MoE-class models, GLM-5.x, Kimi, DeepSeek, Nemotron, CPU/Grace expert offload, PLE, coherent/unified memory, NVLink-C2C, SGLang, vLLM, TensorRT-LLM, and KTransformers. The target comparison point is our local baseline: **single Station GB300; 433 GB full GLM-5.3 NVFP4; SGLang whole-layer expert offload; C1 1.41 tok/s**.

## Bottom line

1. **The strongest measured single-GB300 corpus is Catid's `dgx_station_benchmarks` repository, not the NVIDIA forum threads themselves.** It publishes direct measurements on DGX Station GB300 systems, states these are **GB300 systems, not DGX Spark/GB10**, and records hardware as 1x NVIDIA GB300, 256,703 MiB HBM, 1,300 W power limit, NVIDIA Grace 72-core CPU, 744 GiB system memory, driver 595.84.[catid-repo][catid-guide]
2. **No independently detailed public single-Station result found exactly matching our baseline of full GLM-5.3 NVFP4 with SGLang whole-layer expert offload at ~433 GB.** Public single-station GLM-5.3-Flash results are for the 320B/18B-active Flash model, fitting much more favorably in HBM; full GLM-5.3 results in Catid are on **2x DGX Station GB300**, not single station.[catid-glm53][catid-glm53flash]
3. **The closest public single-station expert-offload evidence is GLM-5.2-NVFP4 on DGX Station/GB300, reported on X by Ahmad Osman and Alex Ziskind/StorageReview.** Those posts claim ~32-33 tok/s decode at 256k / WikiText-style context and describe attention in HBM with experts spilled/offloaded to LPDDR5X/Grace memory.[ahmad-glm52][digitalix-glm52][storagereview-glm52]
4. **NVIDIA Developer Forums contain useful Station threads, but the Station-specific early posts are mostly estimates, rep call notes, or links out.** A forum user reported NVIDIA-rep estimates for Kimi 2.5 1.1T at 40-50 tok/s total and Nemotron Ultra 550B at ~35 tok/s C1, but this was not a reproduced benchmark log.[nvforum-station-bench]
5. **Most very-large Kimi/Nemotron forum measurements are GB10 Spark clusters or B200/GB200/GB300 NVL72 data-center runs, not single Station GB300.** They are useful for algorithm/runtime ideas, but not apples-to-apples against a single Station.[nvforum-kimi-gb10][nvforum-kimi-b200][nvforum-nemotron-gb10]
6. **I found no measured KTransformers-on-single-GB300 Station result for the target class.** It appears in search terms/ecosystem discussions, but the measured GB300 evidence found was SGLang/vLLM/DSpark/DFlash2/TensorRT-LLM/FlashInfer-oriented.

## Platform distinctions used in this report

| Label | What it means here | Include as comparable? |
|---|---|---|
| **Single DGX Station GB300 / GB300 workstation** | One desk-side Grace-Blackwell Station-class system with one server-class GB300, ~250.7 GiB runtime HBM plus Grace LPDDR coherent memory. Catid reports 744 GiB system memory and 256,703 MiB HBM.[catid-guide] | **Yes**, if single serving engine on one station. |
| **2x DGX Station GB300** | Two Station systems over 400GbE/RoCE / GPUDirect Data Direct. Catid reports near-line-rate 392.1 Gb/s GPUDirect and 389.8 Gb/s NCCL bus bandwidth on one 400GbE rail.[catid-net] | Not directly comparable to single Station; useful for topology ideas. |
| **DGX Spark / GB10** | Consumer/desktop Grace-Blackwell Spark nodes, 128 GB unified memory each, SM121; most community cluster recipes use 2-16 nodes. | **Not comparable** except as MoE/offload/runtime ideas. |
| **GB200/B200 multi-GPU / GB300 NVL72** | Data-center 8x/16x B200 nodes or rack-scale NVL72. | Marketing/datacenter comparison only. |

## Evidence table: single Station / GB300 workstation

| Source | Hardware | Model | Quant / size | Engine / runtime | Context / workload | Concurrency | Measured tok/s | Fit/offload notes | Confidence |
|---|---|---|---|---|---|---:|---:|---|---|
| Catid `DeepSeek-V4-Flash-0731` README | 1x DGX Station, 1x server-class NVIDIA GB300 | DeepSeek-V4-Flash-0731, 304B total / 13B active | Native mixed FP4 experts + FP8 dense | SGLang v0.5.16; AR and DSpark | Fixed decode: 8,192 in / 1,024 out, temp 0, FP8 hybrid KV | C1-C128 | **DSpark C1 345.8 output tok/s**; AR C1 151.2; DSpark C128 6,511.1 aggregate raw; prefill 20,453 tok/s at 8K, 34,446 at 128K | No valid DFlash number; DFlash rejected because public draft/checkpoint mismatch. High concurrency has repetition-quality warning. | High: machine-readable repo data + methodology.[catid-deepseek] |
| Catid `GLM-5.3-Flash` README | 1x DGX Station GB300 | GLM-5.3-Flash, 320B total / 18B active, 45 layers, 288 experts, top-8 | `LibertAIDAI/GLM-5.3-Flash-NVFP4` exact revision; official FP8 also measured on 2x | SGLang; NVFP4 TP1/AR and TP1/DFlash2 | Exact 8,192-token prompt and 1,024-token output; cold prefill 8K/64K/128K | C1-C64 | **DFlash2 C1 187.1 tok/s**; AR C1 126.3; AR C64 1,005.1 aggregate; DFlash2 C64 512.7; AR 64K prefill 27,782 prompt tok/s | This is **Flash 320B**, not full GLM-5.3; likely not comparable to 433 GB whole-layer-offload baseline. | High.[catid-glm53flash] |
| Catid `Qwen3.8-Flash-Next` README | 1x DGX Station GB300 | Qwen3.8-Flash-Next, Flash-Next MoE | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89` | SGLang TP1/AR; TP1/MTP3 + ReplaySSM | Decode C1/C16/C64; 64K cold prefill | C1, C16, C64 | **MTP3+ReplaySSM C1 354.6 tok/s**; AR C1 202.1; AR C64 4,090.4; 64K prefill 38,653 prompt tok/s | Not 500B+; useful for PLE/ReplaySSM/speculation ideas. | High.[catid-qwenflash] |
| Catid `Ornith-1.5-397B` README | 1x DGX Station GB300; 2x Station comparison also included | Ornith-1.5-397B MoE | Official ModelOpt NVFP4 W4A4, 221.65 GiB local checkpoint | vLLM 0.27.1; 1x TP1; FP8 E4M3 KV | Sustained decode, exact 8,192 in / 1,024 out; 30s window; prefill 8K/64K/128K | C1-C16 on 1x | **C1 129.8 tok/s**; C16 952.3 aggregate; 1x 128K prefill 26,607 prompt tok/s | Fits one GB300; **CPU weight offload not used**. 2x PP2 C128 reaches 3,799.6 aggregate, but not single Station. | High.[catid-ornith] |
| Catid `MiniMax M3` README | 1x DGX Station GB300 | MiniMax M3 multimodal understanding | NVIDIA `MiniMax-M3-NVFP4`, 250.14 GB payload / 232.96 GiB; ModelOpt mixed NVFP4 | vLLM 0.27.1, FlashInfer/TensorRT-LLM attention + CUTLASS MSA decode | Exact 8K in / 1K out sustained; cold 8K/64K/128K prefill | C1-C16 on 1x; C32+ capacity-limited | **C1 152.6 tok/s**; C16 1,575.4 aggregate; 128K prefill 25,285 prompt tok/s | CPU offload disabled; language-model-only for text benchmark. | High.[catid-minimax] |
| X: Ahmad Osman | DGX Station; attached image reportedly shows NVIDIA GB300 with 256,703 MiB HBM | GLM 5.2 NVFP4 | NVFP4; not REAP | Custom/offload stack described in image as exact NVFP4 expert payload pinned in Grace RAM / CUDA UVA, SM103 CUTLASS FP4 MoE | 256k context; FP8 KV cache | Single request only | **3,000 tok/s prefill; 32 tok/s decode** | Explicitly expert/offload flavored: Grace RAM/CUDA UVA; not enough public method detail to compare strictly. | Medium: X post, screenshot/image evidence via X search, no extracted raw log.[ahmad-glm52] |
| X: Alex Ziskind / digitalix | DGX Station GB300 | GLM-5.2-NVFP4 | 433 GiB quantized weights | Not fully specified in extracted X search result | “Same WikiText-2 evaluation context” as his other box tests | C1 and C16 | **32.9 tok/s C1; 165 tok/s at 16 concurrent; 315 ms TTFT** | Attention in HBM; experts spilled/offloaded to 496 GB LPDDR5X. | Medium: X post surfaced by X search; no public raw artifact in this search.[digitalix-glm52] |
| X: StorageReview | MSI XpertStation WS300 GB300 workstation | GLM-5.2 | 433 GB model | Not specified in X post; workstation review linked | Not a token benchmark in found post | N/A | No tok/s; reports **6,134 TFLOPS NVFP4 measured** | 218 GB in HBM and 216 GB experts offloaded to Grace; 748 GB coherent memory. | Medium for memory split; low for inference throughput because no tok/s in found post.[storagereview-glm52] |
| X: Alec Fong | Single GB300 Station | DeepSeekV4 Flash | Not specified | Not specified | 128k context TTFT claim | Not specified | **300 tok/s single stream; 1,875 tok/s peak aggregate; TTFT 5s for 128k** | Strong claim but no raw log found in this pass. | Medium-low: X anecdote with numbers, insufficient method.[alec-deepseek] |
| X: MrCatid earlier DS4F post | GB300 DGX Station | DS4F / DeepSeek V4 Flash | Not specified in X search output | DFlash / llm-inference-bench follow-up | Follow-up says C64 context; details superseded by Catid repo DeepSeek page | Single and batch/high C | Initial **261 tok/s single, 1,479 tok/s batch 16**; follow-up **305.7 tok/s single, 4,023.8 tok/s sustained C64** | Superseded/clarified by repo's DeepSeek-V4-Flash-0731 DSpark data; treat repo as authoritative. | Medium for post; high if using repo instead.[mrcatid-ds4f][catid-deepseek] |

## Evidence table: 2x Station / non-single-Station but still GB300

| Source | Hardware | Model | Quant / engine | Context / concurrency | Tok/s | Why it matters / caveat |
|---|---|---|---|---|---:|---|
| Catid `GLM-5.3` README | **2x DGX Station GB300** | Full GLM-5.3 | Published `incoai/GLM-5.3-NVFP4` 464.8 GB; DFlash2; SGLang TP2+EP2 and patched vLLM PP2 | 8,192 in / 1,024 out; C1-C64; cold prefill 8K/64K/128K | SGLang TP2+EP2: **165.5 C1 code**, 107.4 C1 prose, 570.0 C32 aggregate; PP2 DFlash2: 154.9 C1, 742.0 C16, 1,093.8 C64; 64K prefill 25,854 tok/s | Highly relevant as a full-GLM-5.3 recipe, but **not single Station** and no whole-layer CPU/Grace offload.[catid-glm53] |
| Catid `GLM-5.2` README | **2x DGX Station GB300**, 400GbE RoCE | GLM-5.2, 753B total / 40B active | NVIDIA `GLM-5.2-NVFP4`, vLLM 0.27.1, TP2 + expert parallelism, FlashInfer CuTeDSL, FP8 KV | 8K in / 1K out, 30s sustained decode; prefill 8K/64K/128K | **68.0 C1**, 1,261.0 C64, 2,012.4 C128 aggregate; 128K prefill 7,244 tok/s | Repo explicitly says **no CPU offload** and 1x does not fit; useful lower bound for native no-offload 2x behavior.[catid-glm52] |
| Catid networking | **2x DGX Station GB300**, ConnectX-8 | Network path | GPUDirect Data Direct RDMA / NCCL | 400GbE rail | 392.1 Gb/s one-way GPUDirect; 389.8 Gb/s NCCL bus bandwidth | Important for 2x Station PP/TP designs; not single-station inference.[catid-net] |
| X: SGLang project | 4x GB300 and 8x B300 | GLM-5.2 NVFP4 | SGLang v0.5.15; launch commands, `--quantization modelopt_fp4`, FP8 KV, EAGLE | Batch size 1 claim for 4x GB300 | **~450 tok/s on 4x GB300**, 500+ tok/s/user on 8x B300 | Project/team announcement, not single Station; useful for SGLang tuning flags.[sglang-glm52] |

## Evidence table: DGX Spark / GB10 — useful ideas, not Station evidence

| Source | Hardware | Model | Quant / engine | Context / concurrency | Tok/s | Relevance / caveat |
|---|---|---|---|---|---:|---|
| NVIDIA forum: GLM-5.2 unpruned @ 200K | **4x DGX Spark GB10**, 128 GB unified each, sm_121, RoCE | GLM-5.2 unpruned, all 256 experts, QuantTrio Int4-Int8Mix | vLLM native multi-node TP=4; fp8_ds_mla KV; MTP k=4; FLASHMLA_SPARSE; DSA | 200,064-token KV pool; 256-token probes, warm engine | **27.0 tok/s single**, 30.7 C2, **52.5 aggregate C4** | Good GB10 recipe; not Station. Highlights version pin drift, drop-caches during weight load, MTP-overhang patch, and IB verification.[nvforum-glm52-gb10] |
| NVIDIA forum: Nemotron-3-Ultra 550B | **4x DGX Spark GB10**, 200GbE RoCE | NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 | SGLang 0.5.12, TP=4/EP=4, ModelOpt mixed FP4 expert FFN + FP8/BF16 exceptions, FP8 KV | 262k/524k context; n1/n4/n8 | **10.2 C1**, 29.3 C4, **43.4 n8 peak** (~5.3 tok/s/request) | Useful for MoE runner selection: `flashinfer_cutlass` only viable; `triton` crashes. Not Station; 512K context essentially free due NoPE/Mamba.[nvforum-nemotron-gb10] |
| NVIDIA forum: Kimi K2.6 / Qwen 397B | **8x DGX Spark GB10** | Kimi-K2.6; Qwen 3.5-397B FP8 | vLLM TP8, Ray/no-Ray tests | Max seqs 4, batched 8192 | Kimi K2.6 **12-13 tok/s**; Qwen 397B **31 tok/s on 4 nodes**, **35 tok/s on 8 nodes** | Shows scaling bottlenecks on GB10/RoCE; not Station.[nvforum-kimi26-gb10] |
| NVIDIA forum: Kimi K3 full | **16x DGX Spark GB10** with MikroTik CRS804 and 400-to-4x100Gb breakouts | moonshotai/Kimi-K3 | DSpark / Inferact/Kimi-K3-DSpark | llama-benchy pp2048/tg1500 at d4000/d16000 | **21.71-25.39 tok/s avg decode**, 37-38 tok/s peak; prefill 654.9-758.8 tok/s | Large Kimi works only across many GB10 nodes here; not single Station.[nvforum-kimi-gb10] |
| NVIDIA forum: GLM-4.7-FP8 | **4x DGX Spark GB10** | GLM-4.7-FP8, 355B / 32B active | SGLang + EAGLE; pre-tuned MoE kernel configs for GB10; 200Gbps RoCE | 202,752 context | **20-27 tok/s** | Older GLM recipe; useful caution that GB10 needs `lmsysorg/sglang:spark` / sm_121 kernels; not Station.[nvforum-glm47-gb10] |

## Evidence table: data-center / marketing / not comparable

| Source | Hardware | Model | Stack | Numbers | Classification |
|---|---|---|---|---:|---|
| NVIDIA forum announcement: Qwen3.8-Flash-Next | **GB300 NVL72**, 72 Blackwell Ultra GPUs, 130 TB/s NVLink | Qwen3.8-Flash-Next, 176B, 6B active, native 262k context extendable to 1M | FP8 TensorRT-LLM; SGLang/vLLM/TRT-LLM support links | **>16,000 tokens/sec per GPU**, **>200 tokens/sec per user** | Vendor/announcement, rack-scale, not Station. Useful only as marketing ceiling and stack support evidence.[nvforum-qwen-nvl72] |
| NVIDIA forum: Kimi K3 on two 8xB200 nodes | **16x B200 GPUs**, two nodes | Kimi K3 | vLLM bench serve; speculative decoding | Output throughput 312.63 prompt-heavy, 378.14 decode-heavy, 329.64 balanced; total throughput up to 2,841 tok/s | Data-center B200, not Station. Note balanced section command appears to reference Qwen3.6-27B-NVFP4 despite text saying Kimi K3, so inspect carefully before reuse.[nvforum-kimi-b200] |
| NVIDIA forum Station benchmark thread | DGX Station GB300, but no local run by poster | Kimi 2.5 1.1T; Nemotron Ultra 550B; GLM 5.2 | Numbers gathered from NVIDIA rep / conference link | Kimi 2.5 ~40-50 tok/s total; Nemotron Ultra ~35 tok/s C1; GLM 5.2 “60 tps for 4bit 504B param model” via external writeup | Useful anecdote, **unsupported as benchmark evidence** without raw commands/logs.[nvforum-station-bench] |

## Notes on offload / coherent memory / PLE ideas

- Catid's measured Station guide argues to **keep CPU offload out of GPU-native comparisons**: if weights exceed runtime-visible HBM, record a one-node no-fit result; CPU/disk offload measures a different system dominated by host memory and interconnect traffic.[catid-guide]
- That said, the X/offload reports closest to our baseline use exactly the path we care about: **attention or hot layers in HBM, expert payloads in Grace LPDDR / CUDA UVA**, with FP8 KV and SM103 FP4 MoE kernels.[ahmad-glm52][digitalix-glm52]
- The NVIDIA forum memory-clarification thread claims HBM3e and LPDDR5X memory are both directly addressable by B300, C2C-connected at 900 GB/s, with roughly 252 GB fast HBM3e and 496 GB slower DDR5/LPDDR usable for GPU access via mechanisms such as vLLM `--cpu-offload-gb`; treat this as knowledgeable forum guidance, not a measured benchmark.[nvforum-memory]
- PLE-like / prediction-layer evidence appears most clearly in Qwen3.8-Flash-Next discussions (N-gram embeddings / PLE table; Spark forum titles mention HashK-PLE and disk streaming), but I did not find a **single Station GB300 + 500B-1T MoE + PLE** measurement. The single Station Qwen3.8-Flash-Next result is much smaller than the target class but shows ReplaySSM/MTP can produce 354.6 C1 and multi-thousand aggregate decode when the architecture supports it.[catid-qwenflash]
- For multi-node but relevant networking, Catid's dual-Station Data Direct results show that one 400GbE rail can deliver ~392 Gb/s GPU RDMA and ~389.8 Gb/s NCCL bus bandwidth, so 2x Station PP/TP experiments are not doomed by a host-memory path if Data Direct is actually active.[catid-net]

## Comparison to our baseline

Our local baseline — **single Station GB300, 433 GB full GLM-5.3 NVFP4, SGLang whole-layer expert offload, C1 1.41 tok/s** — is much slower than the public GLM-5.2 offload anecdotes (~32 tok/s) and far slower than Catid's in-HBM / 2x-station results. The likely reasons to investigate, based on the field evidence, are:

1. **Model identity and fit path:** Public high GLM-5.3-Flash numbers are 320B/18B-active Flash, not full 433 GB GLM-5.3; full GLM-5.3 public results use 2x GB300 and DFlash2/PP2 or TP2+EP2, not single-station whole-layer offload.[catid-glm53][catid-glm53flash]
2. **Offload granularity:** Ahmad/digitalix describe **expert payload** offload or attention-in-HBM/expert-in-LPDDR, not whole-layer expert offload. Whole-layer movement may be dramatically worse than expert-cache/UVA/pinned payload designs.[ahmad-glm52][digitalix-glm52]
3. **Kernel path:** Catid repeatedly warns that quantization labels do not identify executed kernels and that startup logs must confirm intended attention/GEMM/MoE/quantization kernels. For GB300, generic fallback paths can perform very differently.[catid-guide]
4. **Speculative/draft path:** Most high decode numbers use DSpark, DFlash2, MTP, EAGLE, or ReplaySSM. Catid's DeepSeek C1 jumps from 151.2 AR to 345.8 DSpark; GLM-5.3-Flash C1 jumps from 126.3 AR to 187.1 DFlash2.[catid-deepseek][catid-glm53flash]
5. **KV / context policy:** Public results label FP8 KV versus BF16 KV and separate fixed decode from prefill/natural-output audits. Our comparison should keep context, KV dtype, exact input/output, prefix caching, and concurrency aligned.[catid-guide]
6. **Quality gates:** Catid flags DeepSeek high-concurrency raw throughput because repetition audits failed in 8-12% of C64/C128 outputs; speed alone is not sufficient.[catid-deepseek]

## Unsupported or marketing claims flagged

- **Kimi 2.5 1.1T ~40-50 tok/s on DGX Station**: reported second-hand from an NVIDIA rep in a forum thread; no commands, raw logs, or model revision.[nvforum-station-bench]
- **Nemotron Ultra 550B ~35 tok/s C1 on DGX Station**: same second-hand forum note; public detailed Nemotron 550B benchmark found was 4x DGX Spark at ~10 tok/s C1 / 43 tok/s n8, not Station.[nvforum-station-bench][nvforum-nemotron-gb10]
- **Qwen3.8-Flash-Next >16,000 tokens/sec/GPU on GB300 NVL72**: NVIDIA announcement / marketing-style rack-scale figure, not a field single-Station result.[nvforum-qwen-nvl72]
- **StorageReview 6,134 TFLOPS NVFP4 for GLM-5.2 offload**: useful hardware/offload claim, but not a token-throughput result in the surfaced X post.[storagereview-glm52]
- **KTransformers**: no measured single-Station result found; do not cite as supported for this target yet.

## Source links

[catid-repo]: https://github.com/catid/dgx_station_benchmarks
[catid-guide]: https://raw.githubusercontent.com/catid/dgx_station_benchmarks/main/dgx-station-guide/README.md
[catid-net]: https://raw.githubusercontent.com/catid/dgx_station_benchmarks/main/gb300-networking/README.md
[catid-deepseek]: https://raw.githubusercontent.com/catid/dgx_station_benchmarks/main/deepseek-v4-flash-0731/README.md
[catid-glm53flash]: https://raw.githubusercontent.com/catid/dgx_station_benchmarks/main/glm-5.3-flash/README.md
[catid-glm53]: https://raw.githubusercontent.com/catid/dgx_station_benchmarks/main/glm-5.3/README.md
[catid-glm52]: https://raw.githubusercontent.com/catid/dgx_station_benchmarks/main/glm-5.2/README.md
[catid-ornith]: https://raw.githubusercontent.com/catid/dgx_station_benchmarks/main/ornith-1.5-397b/README.md
[catid-qwenflash]: https://raw.githubusercontent.com/catid/dgx_station_benchmarks/main/qwen3.8-flash-next/README.md
[catid-minimax]: https://raw.githubusercontent.com/catid/dgx_station_benchmarks/main/minimax-m3/README.md
[nvforum-station-bench]: https://forums.developer.nvidia.com/t/looking-for-dgx-station-gb300-ultra-llm-inference-benchmarks/372047
[nvforum-memory]: https://forums.developer.nvidia.com/t/memory-clarification-issue-on-dgx-station-1t-llm-model-possible/368919
[nvforum-qwen-nvl72]: https://forums.developer.nvidia.com/t/qwen3-8-flash-next-176b-now-available/381413
[nvforum-glm52-gb10]: https://forums.developer.nvidia.com/t/glm-5-2-unpruned-200k-context-on-4x-dgx-spark-27-tok-s-single-52-5-tok-s-c4/377879
[nvforum-nemotron-gb10]: https://forums.developer.nvidia.com/t/nemotron-3-ultra-550b-a55b-nvfp4-on-4x-dgx-spark-via-sglang-tp-4-ep-4-roce-it-works-42-43-tok-s-n8-peak/372680
[nvforum-kimi26-gb10]: https://forums.developer.nvidia.com/t/kimi-2-6-and-qwen-3-5-397b-fp8-on-8xgb10-cluster/369446
[nvforum-kimi-gb10]: https://forums.developer.nvidia.com/t/full-kimi-k3-running-on-16x-gb10-cluster/379174
[nvforum-glm47-gb10]: https://forums.developer.nvidia.com/t/running-glm-4-7-fp8-355b-moe-on-4x-dgx-spark-with-sglang-eagle-speculative-decoding/359256
[nvforum-kimi-b200]: https://forums.developer.nvidia.com/t/ruuning-kimi-k3-across-two-nvidia-8xb200-nodes-using-vllm/378623
[mrcatid-glm53]: https://x.com/MrCatid/status/2093552797828424166
[mrcatid-glm53flash]: https://x.com/MrCatid/status/2093196362640773227
[mrcatid-ds4f]: https://x.com/MrCatid/status/2089820209116803158
[sglang-glm52]: https://x.com/sgl_project/status/2075721488456654861
[ahmad-glm52]: https://x.com/TheAhmadOsman/status/2078247891370442867
[digitalix-glm52]: https://x.com/digitalix/status/2089888514485739536
[storagereview-glm52]: https://x.com/storagereview/status/2091979502335164613
[alec-deepseek]: https://x.com/alecqfong/status/2085484123855155295
