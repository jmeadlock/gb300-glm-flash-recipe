#!/usr/bin/env python3
"""Audit probe: does _rows64(t.contiguous()) clone a UVA view? Run against the real hook + real UVA views.
Also: exact HBM delta of building one LayerCache at S=112 with real tensor shapes."""
import os, sys, json, torch
sys.path.insert(0, "/w"); os.environ.setdefault("SLOT_CACHE", "112")
import slot_cache_hook as H
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
import exact_pin
E, H_, I = 256, 6144, 2048
out = {}
def mem(): torch.cuda.synchronize(); return torch.cuda.memory_allocated()
# real-shaped UVA views, both pinning paths
w13c = torch.randint(0, 255, (E, 2 * I, H_ // 2), dtype=torch.uint8)
w2c = torch.randint(0, 255, (E, H_, I // 2), dtype=torch.uint8)
for tag, pinner in (("torch_pin", lambda t: t.pin_memory()), ("exact_pin", exact_pin.exact_pinned_like)):
    p13 = pinner(w13c); p2 = pinner(w2c)
    v13 = get_accelerator_view_from_cpu_tensor(p13); v2 = get_accelerator_view_from_cpu_tensor(p2)
    m0 = mem()
    r13 = H.LayerCache._rows64(v13); r2 = H.LayerCache._rows64(v2)
    m1 = mem()
    out[tag] = dict(view_contig=v13.is_contiguous(), view_dev=str(v13.device), rows_same_storage=(r13.data_ptr() == v13.data_ptr()),
                    rows_dev=str(r13.device), hbm_delta_MB=round((m1 - m0) / 1e6, 1))
    # full LayerCache build cost at S=112 with real-shaped scales
    s13 = torch.nn.Parameter((torch.rand(E, 2 * I, H_ // 16, device="cuda") * 0.5 + 0.5).to(torch.float8_e4m3fn), requires_grad=False)
    s2 = torch.nn.Parameter((torch.rand(E, H_, I // 16, device="cuda") * 0.5 + 0.5).to(torch.float8_e4m3fn), requires_grad=False)
    sc = [torch.rand(E, device="cuda") for _ in range(3)]
    m2 = mem()
    lc = H.LayerCache("probe", v13, v2, s13, s2, sc, 112)
    m3 = mem()
    out[tag]["layercache_hbm_delta_GB"] = round((m3 - m2) / 1e9, 3)
    out[tag]["expected_slots_GB"] = round(sum(t.numel() * t.element_size() for t in lc.slots.values()) / 1e9 + sum(t.numel() * t.element_size() for t in lc.res_slots.values()) / 1e9, 3)
    out[tag]["scales_moved_to_host"] = not s13.data.is_cuda or len(lc._keep_host) == 2
    out[tag]["scale_param_dev_after"] = str(s13.data.device)
    del lc, r13, r2, v13, v2, p13, p2, s13, s2; torch.cuda.empty_cache()
print("ROWS64_AUDIT " + json.dumps(out))
