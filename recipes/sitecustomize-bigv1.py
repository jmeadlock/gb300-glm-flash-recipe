# sitecustomize.py — bind-mounted over /usr/lib/python3.12/sitecustomize.py in the vLLM container.
# Keeps Ubuntu's apport hook, then installs a lazy import hook that patches
# RoutedExpertsManager.store_batch to append every step's routing tensor to disk.
# Activated only when ROUTE_TRACE_DIR is set. Zero effect on the compute graph.
try:
    import apport_python_hook
except ImportError:
    pass
else:
    apport_python_hook.install()

import os, sys
_OUT = os.environ.get("ROUTE_TRACE_DIR")
_TARGET = "vllm.model_executor.layers.fused_moe.routed_experts_capturer"
_AT_KEY = os.environ.get("VLLM_AUTOTUNE_CACHE_KEY")  # pin the FlashInfer autotune cache dir by name
_AT_TARGET = "vllm.model_executor.warmup.flashinfer_autotune_cache"

if _AT_KEY:
    import importlib.abc, importlib.util

    def _patch_at(module):
        module.flashinfer_autotune_cache_hash = lambda runner: _AT_KEY
        sys.stderr.write(f"AUTOTUNE_KEY hook installed: {_AT_KEY}\n")

    class _ATFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _AT_TARGET:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader
            orig_exec = loader.exec_module

            def exec_module(module, _orig=orig_exec):
                _orig(module)
                _patch_at(module)

            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _ATFinder())

if _OUT:
    import importlib.abc, importlib.util

    def _patch(module):
        import numpy as np, time
        # --- fix 1: get_routed_experts_attn_gid rejects UniformTypeKVCacheSpecs wrappers even when
        # every inner spec is a FullAttentionSpec (MLA models on TP1 land here). Unwrap.
        from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs
        def _gid(kv_cache_config):
            for gid, group in enumerate(kv_cache_config.kv_cache_groups):
                spec = group.kv_cache_spec
                if isinstance(spec, FullAttentionSpec):
                    return gid
                if isinstance(spec, UniformTypeKVCacheSpecs) and all(
                    isinstance(s, FullAttentionSpec) for s in spec.kv_cache_specs.values()
                ):
                    return gid
            raise ValueError("Routed-experts capture requires a full-attention KV cache group.")
        module.get_routed_experts_attn_gid = _gid
        sys.stderr.write("ROUTE_TRACE attn_gid unwrap patch installed\n")
        # --- trace hook
        cls = module.RoutedExpertsManager
        orig = cls.store_batch
        os.makedirs(_OUT, exist_ok=True)
        pid = os.getpid()
        fb = open(os.path.join(_OUT, f"trace-{pid}.i16"), "ab")
        fs = open(os.path.join(_OUT, f"trace-{pid}.slots.i64"), "ab")
        fm = open(os.path.join(_OUT, f"trace-{pid}.steps"), "a")
        st = {"n": 0}

        def store_batch(self, data, slot_mapping):
            try:
                d = np.ascontiguousarray(data, dtype=np.int16)
                s = np.ascontiguousarray(slot_mapping, dtype=np.int64)
                fb.write(d.tobytes()); fs.write(s.tobytes())
                fm.write(f"{time.time():.3f} {d.shape[0]} {d.shape[1]} {d.shape[2]}\n")
                st["n"] += d.shape[0]
                if st["n"] % 2000 < d.shape[0]:
                    fb.flush(); fs.flush(); fm.flush()
            except Exception as e:  # never break serving
                sys.stderr.write(f"ROUTE_TRACE error: {e!r}\n")
            return orig(self, data, slot_mapping)

        cls.store_batch = store_batch
        sys.stderr.write(f"ROUTE_TRACE hook installed pid={pid} dir={_OUT}\n")

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _TARGET:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader
            orig_exec = loader.exec_module

            def exec_module(module, _orig=orig_exec):
                _orig(module)
                _patch(module)

            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
