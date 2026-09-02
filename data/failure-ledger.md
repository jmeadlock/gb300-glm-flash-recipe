# Failure ledger — every wall between "hardware works" and "first token"

Recorded verbatim so you can grep your error into ours. Same order we hit them.

| # | Attempt | Failure |
|---|---|---|
| 0 | `docker pull lmsysorg/sglang:blackwell` | tag does not exist (guessed) |
| 1 | pinned v0.5.16 digest + `CUDA_VISIBLE_DEVICES=1` | container saw 0 GPUs — DGX OS docker config already hides the display GPU; drop the env var |
| 2 | mounted archive wrapper dir | `Unrecognized model in /model. Should have a 'model_type' key in its config.json.` — mount the `original/` snapshot inside the wrapper |
| 3 | pinned v0.5.16 + correct mount | `model type 'glm5_next' but Transformers does not recognize this architecture` — image transformers too old |
| 3b | `:latest` (transformers 5.12.1) | same as #3 |
| 4 | `:latest` + `pip install -U transformers` | `ValueError: 'qwen3_asr' is already used by a Transformers config` — SGLang v0.5.x hard-registers qwen3_asr; new transformers ships it natively |
| 5a | `:latest` + `transformers==5.16.0` (checkpoint's declared version) | same collision — the declared version is what the quantizer ran (a dev build), not a release that knows the arch |
| 5b | pinned image + `transformers==5.16.0` | same collision from the pinned image too |
| 6 | `:latest` + sed `exist_ok=True` patch + transformers 5.16.0 | past the collision, but 5.16.0 genuinely lacks `glm5_next` |
| 7 | + latest transformers + patch | arch recognized! …no native model in release SGLang: `Unsupported TP style 'mla_kv_a_proj'` from the generic Transformers backend |
| 8 | SGLang **main** from source | same TP failure — GLM-5.3-Flash not merged to main (as of 2026-09-01) |
| 9 | **PR #36507 branch** + patch | `SyntaxError: keyword argument repeated: exist_ok` — the branch already fixed registrations; drop the sed patch |
| 10 | PR #36507 branch, clean | **loads, serves, 141.8 tok/s** |

## Bonus: full GLM-5.3 (704 GB FP8) does not fit one Station

Three attempts with `--cpu-offload-gb` (500/620/560) all OOM'd — GPU-side, then
host-side twice. dmesg showed the scheduler peaking at **1.37 TB total-vm with 741 GB
in shm**: SGLang's offload loader stages the full weight set through host shared
memory, so peak host demand ≈ model size + buffers > the machine's 744 GiB total.
Not tunable around with stock SGLang. Verdict: dual-Station or streaming-loader
territory. NVFP4 Flash *is* the single-Station model.

## Also worth knowing

- Benchmark TTFT on a thinking model must count `reasoning_content` deltas, or
  "TTFT" silently includes thinking time. Our first prefill probe was garbage for
  this reason.
- The v1 "concurrency cliff" (15 s TTFT at C8+) was first-hit kernel autotune per
  batch shape, not a server defect. Warm up after every start (see README).
