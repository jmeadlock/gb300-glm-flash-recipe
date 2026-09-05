import torch, os, subprocess
def shmem():
    for l in open("/proc/meminfo"):
        if l.startswith("Shmem:"): return int(l.split()[1]) // 1024
def rss():
    return int(open(f"/proc/{os.getpid()}/status").read().split("RssShmem:")[1].split()[0]) // 1024
base = rss()
w13 = torch.empty(256, 2 * 2048, 6144 // 2, dtype=torch.uint8).pin_memory()      # 3.22 GB
a = rss() - base
w2 = torch.empty(256, 6144, 2048 // 2, dtype=torch.uint8).pin_memory()            # 1.61 GB
b = rss() - base - a
print(f"PIN_TEST w13 {w13.numel()/1e9:.2f} GB -> pinned rss +{a/1024:.2f} GiB ; w2 {w2.numel()/1e9:.2f} GB -> +{b/1024:.2f} GiB")
# alternative: cudaHostAlloc directly (no caching allocator rounding) via cuda-python
try:
    from cuda.bindings import runtime as rt
    base2 = rss()
    err, ptr = rt.cudaHostAlloc(w13.numel(), rt.cudaHostAllocMapped)
    c = rss() - base2
    print(f"PIN_TEST cudaHostAlloc {w13.numel()/1e9:.2f} GB -> +{c/1024:.2f} GiB")
except Exception as e:
    print("cudaHostAlloc probe skipped:", repr(e)[:120])
# with the allocator's rounding disabled?
print("PYTORCH_PINNED", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), os.environ.get("PYTORCH_PINNED_ALLOC_CONF"))
