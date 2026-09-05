#!/usr/bin/env python3
"""GB300 host-memory read microbench (Phase 0 / E0.1).
Measures what the GPU actually gets when reading Grace LPDDR5X over NVLink-C2C:
  1. cudaMemcpy H2D (copy engine), pinned
  2. SM zero-copy read of the UVA view (what vLLM UVA offload does), whole buffer
  3. SM zero-copy read at expert granularity: 8 random 20 MB slabs out of a 8 GiB pool
     (one decode step of one MoE layer), single stream and 2/4/8 concurrent streams
Prints GB/s (decimal). No model, no server. Runs in the frozen vllm image.
"""
import time, torch, json, random
from vllm.model_executor.offloader.uva import get_accelerator_view_from_cpu_tensor  # noqa

GiB = 1024**3
POOL = 8 * GiB
EXPERT = 3 * 6144 * 2048 // 2  # NVFP4 nibbles: 18.9 MB per expert (w13+w2)
res = {}

def gbps(nbytes, s): return round(nbytes / s / 1e9, 1)

x = torch.empty(POOL, dtype=torch.uint8, pin_memory=True)
x.view(torch.int32).random_()
torch.cuda.synchronize()

# 1. copy engine H2D
d = torch.empty(POOL, dtype=torch.uint8, device="cuda")
for _ in range(2): d.copy_(x, non_blocking=True)
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(3): d.copy_(x, non_blocking=True)
torch.cuda.synchronize(); res["memcpy_h2d_ce"] = gbps(3 * POOL, time.perf_counter() - t)
del d; torch.cuda.empty_cache()

# 2. SM zero-copy, whole buffer, reduction kernel
v = get_accelerator_view_from_cpu_tensor(x)
vi = v.view(torch.int32)
for _ in range(2): vi.sum()
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(3): vi.sum()
torch.cuda.synchronize(); res["sm_zerocopy_sum_whole"] = gbps(3 * POOL, time.perf_counter() - t)

# 2b. SM zero-copy, whole buffer, copy kernel (read host, write HBM) - closer to a GEMV's read side
out = torch.empty(POOL // 4, dtype=torch.int32, device="cuda")
for _ in range(2): out.copy_(vi)
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(3): out.copy_(vi)
torch.cuda.synchronize(); res["sm_zerocopy_copy_whole"] = gbps(3 * POOL, time.perf_counter() - t)
del out; torch.cuda.empty_cache()

# 3. expert-granularity: 8 random slabs of EXPERT bytes, gather via index_select on [n_slabs, EXPERT] view
n_slabs = POOL // EXPERT
slabs = vi.view(-1)[: n_slabs * (EXPERT // 4)].view(n_slabs, EXPERT // 4)
def step(stream, k=8):
    idx = torch.tensor(random.sample(range(n_slabs), k), device="cuda")
    with torch.cuda.stream(stream):
        return torch.index_select(slabs, 0, idx)
for nstreams in (1, 2, 4, 8):
    streams = [torch.cuda.Stream() for _ in range(nstreams)]
    for _ in range(3):
        for s in streams: step(s)
    torch.cuda.synchronize()
    iters = 20; t = time.perf_counter()
    for _ in range(iters):
        for s in streams: step(s)
    torch.cuda.synchronize(); dt = time.perf_counter() - t
    res[f"sm_zerocopy_8x{EXPERT//10**6}MB_streams{nstreams}"] = gbps(iters * nstreams * 8 * EXPERT, dt)
    res[f"ms_per_layerstep_streams{nstreams}"] = round(dt / iters / nstreams * 1000, 2)

# 3b. same gather but from HBM-resident copy (what a resident layer costs)
hb = slabs.clone()
def step_h(k=8):
    idx = torch.tensor(random.sample(range(n_slabs), k), device="cuda")
    return torch.index_select(hb, 0, idx)
for _ in range(3): step_h()
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(50): step_h()
torch.cuda.synchronize(); dt = time.perf_counter() - t
res["hbm_8x_gather_ms"] = round(dt / 50 * 1000, 3); res["hbm_8x_gather_GBps"] = gbps(50 * 8 * EXPERT, dt)

print("C2C_BENCH " + json.dumps(res))
