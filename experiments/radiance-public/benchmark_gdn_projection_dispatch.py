"""Time the original M1/M8, leaked prefill, and runtime-qualified projection."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from benchmark_gdn_prefill_direct import get_weights
from benchmark_prefill_bank_norm import measure
from gdn_prefill_direct import (
    _project_for_runtime_rows,
    aligned_tiled_projection,
    tiled_norm_quant,
)
from mxfp4_fold_precision import make_row_ref
from prefill_gemm_alignment import AlignedPrefillGemm
from stock_gdn_norm_quant import fused


def measure_case(rows, weight, scale, ref, projection, gemm, repeats):
    inputs = (rows, 48, 128)
    args_x = {"dtype": torch.bfloat16, "device": "cuda"}
    x, z = (torch.randn(inputs, **args_x) for _ in range(2))
    norm_weight = torch.randn(128, **args_x)

    def baseline(x, z):
        q, s = fused(x, z, norm_weight, 1e-6)
        # Mirror prefill_alignment_runtime: GDN outputs switch away from the
        # native split decode kernel above M8, and use tiled inputs from M64.
        if rows > 8:
            return projection(q, s, weight, scale, ref, tiled=rows >= 64)
        out = torch.empty((rows, 5120), **args_x)
        gemm.launch(
            q.data_ptr(),
            weight.data_ptr(),
            scale.data_ptr(),
            ref.data_ptr(),
            s.data_ptr(),
            out.data_ptr(),
            rows,
            5120,
            6144,
            torch.cuda.current_stream().cuda_stream,
        )
        return out

    def leaked():
        q, s = tiled_norm_quant(x, z, norm_weight, 1e-6)
        return aligned_tiled_projection(projection, q, s, weight, scale, ref, True)

    calls = {"projection_direct": 0, "direct_rows": {}, "fallback_rows": {}}
    prior = {weight.data_ptr(): baseline}

    def repaired():
        return _project_for_runtime_rows(x, z, weight, prior, calls, leaked)

    a, b, c = baseline(x, z), leaked(), repaired()
    item = {
        "rows": rows,
        "baseline_repaired_different_elements": int(
            torch.count_nonzero(a.view(torch.int16) != c.view(torch.int16))
        ),
        "baseline_leaked_different_elements": int(
            torch.count_nonzero(a.view(torch.int16) != b.view(torch.int16))
        ),
        "output_elements": a.numel(),
        "route": "fallback" if calls["fallback_rows"] else "direct",
        "leaked_vs_repaired": measure(leaked, repaired, repeats, graph=rows <= 64),
    }
    return item


def run(args):
    torch.manual_seed(20261126)
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location("bench.radiance_mxfp4_fp8", args.gemm)
    gemm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemm)
    scratch = torch.empty(4 * 64 * 36864, dtype=torch.float32, device="cuda")
    indexing = torch.zeros(36864 // 128 + 8, dtype=torch.int32, device="cuda")
    gemm.set_decode_scratch(
        scratch.data_ptr(), scratch.numel() * 4, indexing.data_ptr()
    )
    weight, scale, _ = get_weights(args.model)
    ref = make_row_ref(scale)
    projection = AlignedPrefillGemm(args.projection)
    results = {
        "gemm_sha256": hashlib.sha256(args.gemm.read_bytes()).hexdigest(),
        "aligned_sha256": projection.manifest["sha256"],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [],
    }
    for rows in args.rows:
        item = measure_case(rows, weight, scale, ref, projection, gemm, args.repeats)
        results["cases"].append(item)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(
            rows,
            item["route"],
            item["baseline_leaked_different_elements"],
            item["leaked_vs_repaired"]["control_ms"],
            item["leaked_vs_repaired"]["candidate_ms"],
            flush=True,
        )
        if item["baseline_repaired_different_elements"]:
            raise ValueError(
                f"runtime-dispatched projection changed baseline at {rows} rows"
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("gemm", "model", "projection", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--rows", type=int, nargs="+", default=[1, 8, 16, 63, 64, 128, 1648])
    p.add_argument("--repeats", type=int, default=15)
    run(p.parse_args())
