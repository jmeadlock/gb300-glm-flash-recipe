#!/usr/bin/env python3
"""hostmap_probe.py — which host-memory mapping path gets the fast C2C read on this box under CDMM?
Same gather benchmark (8 random 12 MiB rows out of 64) for each mapping method. Prints GB/s per method.
Methods:
  A torch pin_memory (cudaHostAlloc default)                 -> reference (v5 UVA path)
  B cudaHostAlloc(cudaHostAllocMapped|WriteCombined)
  C malloc + cudaHostRegister(Mapped)                          (cudaHostRegister on ordinary pages)
  D cuMemCreate HOST_NUMA (VMM)                                (yesterday's 85 GB/s result)
  E cuMemCreate HOST (non-NUMA) (VMM)
  F cudaMallocManaged + cudaMemAdvise(PreferredLocation=CPU, AccessedBy=GPU)
  G cuMemCreate HOST_NUMA + cuMemSetAccess for BOTH host and device locations
"""
import time, json, ctypes, numpy as np, torch
from cuda.bindings import driver as cu, runtime as rt
import cupy

def chk(res):
    err, *vals = res if isinstance(res, tuple) else (res,)
    ok = (cu.CUresult.CUDA_SUCCESS, rt.cudaError_t.cudaSuccess)
    if err not in ok: raise RuntimeError(f"CUDA error {err}")
    return vals[0] if len(vals) == 1 else vals

torch.cuda.init(); chk(cu.cuInit(0)); dev = 0
ROW = 12 * 2**20; E = 64; TOTAL = ROW * E
out = {}

def as_torch(ptr):
    mem = cupy.cuda.UnownedMemory(int(ptr), TOTAL, owner=None)
    arr = cupy.ndarray((E, ROW // 4), dtype=cupy.int32, memptr=cupy.cuda.MemoryPointer(mem, 0))
    return torch.as_tensor(arr, device="cuda")

def bench(t, iters=30):
    idx = torch.randint(0, E, (8,), device="cuda")
    for _ in range(3): torch.index_select(t, 0, idx)
    torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(iters):
        idx = torch.randint(0, E, (8,), device="cuda"); torch.index_select(t, 0, idx)
    torch.cuda.synchronize(); return round(iters * 8 * ROW / (time.perf_counter() - s) / 1e9, 1)

def run(name, fn):
    try:
        t = fn()
        t.fill_(1); torch.cuda.synchronize()
        out[name] = bench(t)
    except Exception as e:
        out[name] = f"ERR {repr(e)[:90]}"
    print(name, out[name], flush=True)

# A
def A():
    p = torch.empty((E, ROW // 4), dtype=torch.int32, pin_memory=True)
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    return get_accelerator_view_from_cpu_tensor(p)
run("A_torch_pinned_uva", A)

# B
def B():
    flags = rt.cudaHostAllocMapped | rt.cudaHostAllocWriteCombined
    hp = chk(rt.cudaHostAlloc(TOTAL, flags))
    dp = chk(rt.cudaHostGetDevicePointer(hp, 0))
    return as_torch(dp)
run("B_cudaHostAlloc_mapped_wc", B)

# C
def C():
    buf = np.empty(TOTAL, dtype=np.uint8)
    hp = buf.ctypes.data
    chk(rt.cudaHostRegister(hp, TOTAL, rt.cudaHostRegisterMapped))
    dp = chk(rt.cudaHostGetDevicePointer(hp, 0))
    C.keep = buf
    return as_torch(dp)
run("C_malloc_hostRegister", C)

def vmm(loc_type, loc_id, both_access=False):
    prop = cu.CUmemAllocationProp(); prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = loc_type; prop.location.id = loc_id
    gran = chk(cu.cuMemGetAllocationGranularity(prop, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM))
    va = chk(cu.cuMemAddressReserve(TOTAL, gran, 0, 0))
    h = chk(cu.cuMemCreate(TOTAL, prop, 0))
    chk(cu.cuMemMap(va, TOTAL, 0, h, 0))
    accs = []
    a = cu.CUmemAccessDesc(); a.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE; a.location.id = dev
    a.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE; accs.append(a)
    if both_access:
        b = cu.CUmemAccessDesc(); b.location.type = loc_type; b.location.id = loc_id
        b.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE; accs.append(b)
    chk(cu.cuMemSetAccess(va, TOTAL, accs, len(accs)))
    return as_torch(va)

run("D_vmm_HOST_NUMA", lambda: vmm(cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA, 0))
run("E_vmm_HOST", lambda: vmm(cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST, 0))
run("G_vmm_HOST_NUMA_dualaccess", lambda: vmm(cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA, 0, both_access=True))

# F managed
def F():
    p = chk(rt.cudaMallocManaged(TOTAL, rt.cudaMemAttachGlobal))
    loc = rt.cudaMemLocation(); loc.type = rt.cudaMemLocationType.cudaMemLocationTypeHost; loc.id = 0
    try:
        chk(rt.cudaMemAdvise_v2(p, TOTAL, rt.cudaMemoryAdvise.cudaMemAdviseSetPreferredLocation, loc))
        dl = rt.cudaMemLocation(); dl.type = rt.cudaMemLocationType.cudaMemLocationTypeDevice; dl.id = dev
        chk(rt.cudaMemAdvise_v2(p, TOTAL, rt.cudaMemoryAdvise.cudaMemAdviseSetAccessedBy, dl))
    except Exception as e:
        out["F_advise_note"] = repr(e)[:80]
    # populate from host so pages are resident on host first
    ctypes.memset(int(p), 1, TOTAL)
    return as_torch(p)
run("F_managed_preferHost", F)

print("HOSTMAP_RESULT " + json.dumps(out))
