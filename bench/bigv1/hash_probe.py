#!/usr/bin/env python3
"""Compute the FlashInfer autotune cache hash for a given `vllm serve` argv WITHOUT loading weights.
Usage: python3 hash_probe.py <vllm serve args...>
Prints HASH <sha256>. Used to pre-seed autotune_configs.json for configs whose MoE kernel shapes
are unchanged (offload-GB / instrumentation-only changes)."""
import sys, hashlib
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.compilation.caching import aot_compile_hash_factors

parser = make_arg_parser(FlexibleArgumentParser())
args = parser.parse_args(sys.argv[1:])
if getattr(args, "model_tag", None):
    args.model = args.model_tag
ea = AsyncEngineArgs.from_cli_args(args)
cfg = ea.create_engine_config()
factors = aot_compile_hash_factors(cfg)
print("HASH", hashlib.sha256(str(factors).encode()).hexdigest())
