"""Build checked TP1 FP8 backports while preserving the existing cache layout.

Lazy GDN remains an explicit experimental opt-in with separate qualification.
"""

import argparse
import json
import shutil
from pathlib import Path

from stock_gdn_lazy_patches import overlay
from tp1_lazy_backports import digest, evidence

from qwen_r9700_lab.diagnostic_contract import authenticate, private_json, seal, write_private


def prepare(args):
    if args.enable_lazy_gdn:
        if args.lazy is None or args.vllm is None:
            raise ValueError("--enable-lazy-gdn requires --lazy and --vllm")
    elif args.lazy is not None or args.vllm is not None:
        raise ValueError("--lazy and --vllm require explicit --enable-lazy-gdn")
    parent = private_json(args.performance)
    authenticate(parent)
    if parent.get("lazy_gdn"):
        raise ValueError("start from the original GEMM bundle with the existing cache layout")
    for name, expected in parent["sources"].items():
        if digest(args.runtime_sources / name) != expected:
            raise ValueError(f"parent runtime source changed: {name}")
    fp8 = {
        "build": str(args.build),
        "qualification": str(args.fp8),
        "qualification_sha256": digest(args.fp8),
        "silu_enabled": False,
    }
    evidence(fp8, kind="fp8")
    lazy = None
    if args.enable_lazy_gdn:
        lazy = {
            "qualification": str(args.lazy),
            "qualification_sha256": digest(args.lazy),
            "state_abi": "stock-fp32-lazy-v1",
            "production_disk_restore": "UNQUALIFIED",
        }
        evidence(lazy, kind="lazy")
    # The compiled SiLU reference is faster on this pinned stack. Keep that
    # producer; residual norm+quant and tuple-aware consumers are still enabled.
    args.output.mkdir(mode=0o700)
    runtime = args.output / "runtime"
    shutil.copytree(args.runtime_sources, runtime)
    names = (
        "optimized_d7_performance.py",
        "optimized_d7_worker.py",
        "optimized_d7_startup.py",
        "stock_fp8_epilogue.py",
        "stock_fp8_epilogue.hip",
        "stock_fp8_stream.py",
        "build_stock_fp8_epilogue.py",
        "probe_stock_fp8_epilogue.py",
        "tp1_lazy_backports.py",
    )
    if lazy is not None:
        names += (
            "stock_gdn_lazy_kernel.py",
            "stock_gdn_lazy_runtime.py",
            "stock_gdn_lazy_patches.py",
            "stock_gdn_lazy_patches.json",
            "probe_stock_gdn_lazy.py",
        )
    for name in names:
        shutil.copy2(Path(__file__).with_name(name), runtime / name)
        parent["sources"][name] = digest(runtime / name)
    python_paths = [str(runtime)]
    if lazy is not None:
        python_root = args.output / "python"
        python_root.mkdir()
        lazy["overlay"] = overlay(args.vllm, python_root / "vllm")
        python_paths.append(str(python_root))
    parent["backport_parent"] = parent.pop("sha256")
    parent.update(tp1_fp8=fp8)
    if lazy is not None:
        parent["lazy_gdn"] = lazy
    write_private(args.output / "performance.json", seal(parent))
    launch = {
        "pythonpath_prepend": python_paths,
        "environment": {
            "QWEN_STOCK_GDN_LAZY": "1" if lazy is not None else "0",
            "RADIANCE_FP8_STREAM": "0",
            # Select the native-quant tuple-aware linear/GDN consumers before
            # their import and weight loading. Approximate traced quant stays off.
            "RADIANCE_MXFP4_HOIST_QUANT": "1",
            "RADIANCE_MXFP4_TRACED_QUANT": "0",
            "RADIANCE_MXFP4_PUREQUANT": "0",
            "RADIANCE_GDN_LAZY": "0",
            "TORCHINDUCTOR_EMULATE_PRECISION_CASTS": "1",
            "QWEN_OPTIMIZED_PERFORMANCE": str(args.output / "performance.json"),
        },
        "cache_abi": "stock-fp32-lazy-v1" if lazy is not None else "unchanged",
        "scope": (
            "Stage-qualified single-sequence TP1 D7 prototype; no production disk offload"
            if lazy is not None
            else "TP1 FP8 and GEMM backports; existing nine-slot GDN state layout"
        ),
    }
    (args.output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    print(str(args.output / "performance.json"), flush=True)


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("performance", "runtime-sources", "build", "fp8", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--enable-lazy-gdn", action="store_true", help="Opt into experimental lazy GDN"
    )
    parser.add_argument("--lazy", type=Path, help="Lazy GDN evidence; requires --enable-lazy-gdn")
    parser.add_argument("--vllm", type=Path, help="Pinned vLLM source; requires --enable-lazy-gdn")
    return parser


if __name__ == "__main__":
    prepare(argument_parser().parse_args())
