#!/usr/bin/env python3
"""vmm_probe.py — can ONE contiguous device VA range be backed by a MIX of HBM and Grace (host NUMA) pages
under CDMM on this driver? If yes, hot/cold expert placement needs no kernel or model changes.
Layout mimics one w13 tensor: E rows x 12 MiB (6 x 2 MiB pages). Hot rows -> device physical, cold -> host NUMA 0.
Checks: (1) allocations succeed, (2) kernel reads see the right bytes on both kinds of row (correctness),
(3) per-row read bandwidth hot vs cold vs a plain UVA pinned tensor (the v5 path).
"""
import time, json, ctypes, numpy as np, torch
from cuda.bindings import driver as cu
import cupy

def chk(res):
    err, *vals = res if isinstance(res, tuple) else (res,)
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error {err}")
    return vals[0] if len(vals) == 1 else vals

torch.cuda.init(); dev = 0
chk(cu.cuInit(0))
ROW = 12 * 2**20; E = 128; HOT = 64
out = {}

# --- properties
prop_dev = cu.CUmemAllocationProp(); prop_dev.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
prop_dev.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE; prop_dev.location.id = dev
prop_host = cu.CUmemAllocationProp(); prop_host.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
prop_host.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA; prop_host.location.id = 0
gran_dev = chk(cu.cuMemGetAllocationGranularity(prop_dev, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM))
try:
    gran_host = chk(cu.cuMemGetAllocationGranularity(prop_host, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM))
except Exception as e:
    print("PROBE_RESULT " + json.dumps({"host_numa_supported": False, "err": repr(e)})); raise SystemExit
out["granularity_dev"] = gran_dev; out["granularity_host"] = gran_host
assert ROW % gran_dev == 0 and ROW % gran_host == 0, (gran_dev, gran_host)

# --- reserve VA, map rows
total = ROW * E
va = chk(cu.cuMemAddressReserve(total, max(gran_dev, gran_host), 0, 0))
handles = []
acc = cu.CUmemAccessDesc(); acc.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE; acc.location.id = dev
acc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
t0 = time.perf_counter()
for r in range(E):
    prop = prop_dev if r < HOT else prop_host
    h = chk(cu.cuMemCreate(ROW, prop, 0)); handles.append(h)
    chk(cu.cuMemMap(int(va) + r * ROW, ROW, 0, h, 0))
chk(cu.cuMemSetAccess(va, total, [acc], 1))
out["map_ms_for_128_rows"] = round((time.perf_counter() - t0) * 1000, 1)

# --- wrap as torch tensor via cupy unowned memory
mem = cupy.cuda.UnownedMemory(int(va), total, owner=None)
arr = cupy.ndarray((E, ROW // 4), dtype=cupy.int32, memptr=cupy.cuda.MemoryPointer(mem, 0))
t = torch.as_tensor(arr, device="cuda")
assert t.data_ptr() == int(va)

# --- correctness: write per-row pattern from a kernel, read back via both a kernel and a D2H copy
rows_pat = torch.arange(E, device="cuda", dtype=torch.int32).view(E, 1)
t.copy_(rows_pat.expand(E, ROW // 4))
torch.cuda.synchronize()
sums = t.to(torch.int64).sum(dim=1)
exp = torch.arange(E, device="cuda", dtype=torch.int64) * (ROW // 4)
out["kernel_read_correct"] = bool(torch.equal(sums, exp))
h2 = t[HOT + 3].cpu()
out["cold_row_d2h_correct"] = bool((h2 == HOT + 3).all().item())

# --- bandwidth: gather 8 random rows (one MoE step's active experts) from hot-only, cold-only, mixed
def bw(idx_pool, iters=30):
    idx = torch.tensor(np.random.choice(idx_pool, 8, replace=False), device="cuda")
    for _ in range(3): torch.index_select(t, 0, idx)
    torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(iters):
        idx = torch.tensor(np.random.choice(idx_pool, 8, replace=False), device="cuda")
        torch.index_select(t, 0, idx)
    torch.cuda.synchronize(); dt = time.perf_counter() - s
    return round(iters * 8 * ROW / dt / 1e9, 1)
out["gather8_hot_rows_GBps"] = bw(list(range(HOT)))
out["gather8_cold_rows_GBps"] = bw(list(range(HOT, E)))
out["gather8_mixed_GBps"] = bw(list(range(E)))
# reference: v5-style pinned UVA tensor of the same shape
pinned = torch.empty((E, ROW // 4), dtype=torch.int32, pin_memory=True)
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
uva = get_accelerator_view_from_cpu_tensor(pinned)
def bw_uva(iters=30):
    idx = torch.randint(0, E, (8,), device="cuda")
    for _ in range(3): torch.index_select(uva, 0, idx)
    torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(iters):
        idx = torch.randint(0, E, (8,), device="cuda"); torch.index_select(uva, 0, idx)
    torch.cuda.synchronize(); return round(iters * 8 * ROW / (time.perf_counter() - s) / 1e9, 1)
out["gather8_uva_pinned_GBps"] = bw_uva()
# hbm reference
hb = torch.empty((E, ROW // 4), dtype=torch.int32, device="cuda")
def bw_hbm(iters=30):
    idx = torch.randint(0, E, (8,), device="cuda")
    for _ in range(3): torch.index_select(hb, 0, idx)
    torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(iters):
        idx = torch.randint(0, E, (8,), device="cuda"); torch.index_select(hb, 0, idx)
    torch.cuda.synchronize(); return round(iters * 8 * ROW / (time.perf_counter() - s) / 1e9, 1)
out["gather8_hbm_GBps"] = bw_hbm()

# --- remap test: flip one cold row to hot in place (what a placement update would do)
out["host_numa_supported"] = True
print("PROBE_PARTIAL " + json.dumps(out), flush=True)
r = HOT + 5
try:
    saved = t[r].clone(); torch.cuda.synchronize()
    chk(cu.cuMemUnmap(int(va) + r * ROW, ROW)); chk(cu.cuMemRelease(handles[r]))
    h = chk(cu.cuMemCreate(ROW, prop_dev, 0)); handles[r] = h
    chk(cu.cuMemMap(int(va) + r * ROW, ROW, 0, h, 0)); chk(cu.cuMemSetAccess(int(va) + r * ROW, ROW, [acc], 1))
    t[r].copy_(saved); torch.cuda.synchronize()
    out["remap_row_correct"] = bool((t[r] == r).all().item())
except Exception as e:
    out["remap_row_correct"] = False; out["remap_err"] = repr(e)[:160]
print("PROBE_RESULT " + json.dumps(out))
