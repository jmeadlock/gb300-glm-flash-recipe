# vLLM selective-UVA result — GLM-5.3 NVFP4 on one GB300

Date: 2026-09-02

## Result

The full `incoai/GLM-5.3-NVFP4` checkpoint is serving successfully on one DGX Station GB300 with vLLM selective CUDA UVA offload.

- Image: `vllm-glm53-uva:v0.28.0-2cf0a691`
- Model: `glm-5.3-big`
- Endpoint: `:30001`
- Container: `glm53-big-vllm`
- State after verification: running, not OOM-killed
- UVA backend: `UVAOffloader`
- Selective parameters: `routed_experts.w13_weight`, `routed_experts.w2_weight`
- CPU/Grace offload budget: 270 GiB
- Maximum context: 65,536 tokens
- Maximum sequences: 8
- Maximum batched tokens: 8,192
- Quantization: ModelOpt NVFP4

## Startup and memory

- 87 checkpoint shards loaded.
- Weight loading: 1,017.05 seconds.
- GPU model-weight allocation reported by vLLM: 145.17 GiB.
- CUDA graph capture completed successfully.
- Available KV cache: 74.57 GiB.
- KV capacity: 1,678,592 tokens.
- Maximum reported concurrency at 65,536 tokens/request: 25.61x.
- Steady idle GB300 allocation after verification: approximately 231,240 MiB of 256,703 MiB.
- Host memory after verification: approximately 83 GiB available; approximately 360 GiB shared.

The first startup spent roughly 30 minutes in FlashInfer autotuning after model load. Its 198 MiB vLLM compile/autotune cache is preserved at `/home/milo/vllm-cache` and the launcher now mounts that path at `/root/.cache/vllm` for future starts.

## Correctness and protocol

- `/v1/models` returned `glm-5.3-big`.
- `get_weather` tool call parsed correctly.
- Tool response finished with `tool_calls`, exposed 204 reasoning characters separately, and leaked no `<tool_call>` or `<think>` markup.
- vLLM exposes separated reasoning in the OpenAI-compatible message field named `reasoning` (not SGLang's `reasoning_content`).
- A 512-token reasoning probe completed normally, kept 1,038 reasoning characters separate from final content, and correctly answered that 9.9 is larger than 9.11.

## Throughput

All values below are wall-clock completion-token throughput from the same local benchmark script and 64 completion tokens per request.

| Concurrency | vLLM aggregate tok/s | Mean per-request tok/s | SGLang aggregate baseline | vLLM speedup |
|---:|---:|---:|---:|---:|
| 1 | 23.65 | 23.66 | 1.41 | 16.77x |
| 4 | 42.80 | 10.74 | 5.62 | 7.62x |
| 8 | 65.92 | 8.26 | 11.22 | 5.88x |

A separate C1 smoke probe measured 24.98 tok/s over 64 tokens.

## Warnings and limits

- vLLM warns that ModelOpt NVFP4 support is experimental.
- The checkpoint does not provide explicit FP8 KV scaling factors. vLLM uses a scale of 1.0 and warns that accuracy can drop if that scale is inappropriate.
- The model's `generation_config.json` overrides vLLM defaults with temperature 1.0 and top-p 0.95 unless a request supplies its own sampling parameters. Verification requests explicitly used temperature 0.
- Benchmark throughput includes all completion tokens, including hidden reasoning tokens when present.
- The benchmark proves strong single-Station operation, not equivalence to an all-HBM multi-GPU deployment.

## Operational files

- Launcher: `/home/milo/launch-vllm-uva.sh`
- Persistent cache: `/home/milo/vllm-cache`
- Smoke test: `/home/milo/big_smoke.py`
- Concurrency benchmark: `/home/milo/big_concurrency_bench.py`
- SGLang rollback baseline remains intact as stopped container `glm53-big` using image `glm53-big:v2`.

## Shutdown and resume state

At the user's request, all running GB300 Docker containers were stopped cleanly and the host was powered off on 2026-09-02. Final verification showed both ICMP and SSH unreachable. The working `glm53-big-vllm` container, its image, model checkpoint, launcher, benchmark scripts, and persistent cache remain on disk.

To resume after power is restored:

1. Verify CDMM before loading the model: `/proc/driver/nvidia/params` must show `CoherentGPUMemoryMode: "driver"`.
2. Confirm `glm53-flash` and `glm53-big` remain stopped.
3. Start the proven container: `docker start glm53-big-vllm`.
4. Follow `docker logs -f glm53-big-vllm` until `Application startup complete` appears. Weight loading still takes roughly 17 minutes; the preserved cache should avoid repeating completed compile/autotune work where cache keys remain valid.
5. Run `/home/milo/big_smoke.py`, then `/home/milo/big_concurrency_bench.py`, supplying the existing API key through the environment rather than a command-line flag.
6. If the saved container cannot start, remove only `glm53-big-vllm` and recreate it with `/home/milo/launch-vllm-uva.sh`. Do not remove the model checkpoint, SGLang rollback container, or `/home/milo/vllm-cache`.

## Decision

Keep the current 270 GiB selective-UVA configuration as the verified winner. It exceeds the predefined strong threshold of 20 tok/s C1 and reaches 65.92 aggregate tok/s at concurrency 8. Do not spend another 17–50 minute reload on speculative MTP or placement tuning until a workload demonstrates a concrete need. The next defensible A/B, if needed, is a smaller UVA budget such as 250 GiB to retain more routed-expert payload in HBM while allowing vLLM to reduce KV allocation automatically.
