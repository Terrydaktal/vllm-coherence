"""Check the additional prefill row domain against independently split operators.

Run with exclusive GPU access. Uses synthetic inputs and the model's norm
weights only; no conversations. Full-model, lifecycle and snapshot tests remain
separate qualification requirements.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def run(args):
    import benchmark_prefill_attention as attention
    import benchmark_prefill_conv as conv
    import benchmark_prefill_projection as projection
    import benchmark_prefill_scan as scan
    import probe_stock_gdn_norm_quant as norm

    args.output.mkdir(mode=0o700, exist_ok=True)
    names = {
        "attention": "attention.json",
        "projection": "projection.json",
        "recurrent-state": "scan.json",
        "convolution-state": "conv.json",
        "gdn-normalization": "gdn-norm.json",
    }
    if any((args.output / names[stage]).exists() for stage in args.stages):
        raise ValueError("refusing to overwrite an existing operator qualification")
    widths = [2049, 2560, 3295, 3296, 3297, 4095, 4096]

    def progress(stage):
        (args.output / "progress.json").write_text(json.dumps({"stage": stage}) + "\n")
        print(stage, flush=True)

    if "attention" in args.stages:
        progress("attention")
        attention.run(
            SimpleNamespace(
                baseline=args.baseline_attention,
                candidates=[args.attention],
                output=args.output / "attention.json",
                full=True,
                extra_shapes=[
                    (width, context) for width in widths for context in (60000, 253792)
                ],
                kv_formats=["float8_e4m3fn", "bfloat16"],
                contexts=[60000, 200000],
                rows=1648,
                repeats=3,
            )
        )
    if "projection" in args.stages:
        progress("projection")
        projection.run(
            SimpleNamespace(
                baseline=args.baseline_projection,
                candidates=[args.projection],
                output=args.output / "projection.json",
                full=True,
                extra_rows=widths,
                timing_rows=[1648],
                repeats=3,
            )
        )
    if "recurrent-state" in args.stages:
        progress("recurrent-state")
        scan.run(
            {
                "rows": [9, 65, 672, 976, 1648, 2048, *widths],
                "patterns": ["random", "gates", "tiny", "large"],
                "tiles": [32],
                "prepared": True,
                "prepared_tiles": [4],
                "repeats": 1,
                "output": str(args.output / "scan.json"),
            }
        )
    if "convolution-state" in args.stages:
        from stock_gdn_runtime import CONV_SOURCE

        if hashlib.sha256(args.convolution.read_bytes()).hexdigest() != CONV_SOURCE:
            raise ValueError("stock convolution source binding differs")
        spec = importlib.util.spec_from_file_location(
            "qwen_stock_runtime_convolution", args.convolution
        )
        native = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = native
        spec.loader.exec_module(native)
        progress("convolution-state")
        conv.run(
            {
                "rows": [9, 65, 672, 976, 1648, 2048, *widths],
                "repeats": 1,
                "output": str(args.output / "conv.json"),
            }
        )
    if "gdn-normalization" in args.stages:
        progress("gdn-normalization")
        norm.run(
            SimpleNamespace(
                model=args.model,
                rows=1000,
                row_invariant=True,
                output=args.output / "gdn-norm.json",
            )
        )
    progress("SELECTED_STAGES_PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "baseline-attention",
        "attention",
        "baseline-projection",
        "projection",
        "model",
        "output",
        "convolution",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=[
            "attention",
            "projection",
            "recurrent-state",
            "convolution-state",
            "gdn-normalization",
        ],
        default=[
            "attention",
            "projection",
            "recurrent-state",
            "convolution-state",
            "gdn-normalization",
        ],
    )
    run(parser.parse_args())
