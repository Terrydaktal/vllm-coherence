"""Paired GPU operator timings for the two independent GDN prefill experiments."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from gdn_prefill_direct import _prepared_scan_into, aligned_tiled_projection, tiled_norm_quant
from prefill_activation_tiles import pack
from prefill_gemm_alignment import AlignedPrefillGemm
from prepared_prefill_scan import run
from stock_gdn_norm_quant import fused


def exact(x, y):
    return torch.equal(x.view(torch.uint8), y.view(torch.uint8))


def timing(control, candidate, repeats):
    functions = (control, candidate)
    for fn in functions:
        fn()
    torch.cuda.synchronize()
    samples = [[], []]
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for index in range(repeats):
        for arm in (index % 2, 1 - index % 2):
            start.record()
            functions[arm]()
            end.record()
            end.synchronize()
            samples[arm].append(start.elapsed_time(end))
    return {
        "control_ms": statistics.median(samples[0]),
        "candidate_ms": statistics.median(samples[1]),
        "control_samples": samples[0],
        "candidate_samples": samples[1],
    }


def scan_case(rows, repeats):
    kwargs = {"device": "cuda", "dtype": torch.bfloat16}
    initial = torch.randn((48, 128, 128), device="cuda") * 0.01
    params = (
        initial,
        torch.randn(rows, 10240, **kwargs) * 0.1,
        torch.randn(rows, 48, **kwargs),
        torch.randn(rows, 48, **kwargs),
        torch.rand(48, device="cuda"),
        torch.randn(48, **kwargs),
    )
    guarded = torch.full((rows + 2, 48, 128), 42, **kwargs)
    output = guarded[1:-1]
    run_into = _prepared_scan_into()

    def control():
        values, final = run(*params)
        output.copy_(values)
        return final

    def candidate():
        _, final = run_into(*params, output=output)
        return final

    reference_state = control()
    reference_output = output.clone()
    candidate_state = candidate()
    assert exact(reference_state, candidate_state), "GDN final state changed"
    assert exact(reference_output, output), "GDN scan output changed"
    assert bool((guarded[[0, -1]] == 42).all()), "GDN wrote outside destination"
    return timing(control, candidate, repeats)


def projection_case(rows, repeats, projection, weights):
    kwargs = {"device": "cuda", "dtype": torch.bfloat16}
    params = (torch.randn(rows, 48, 128, **kwargs), torch.randn(rows, 48, 128, **kwargs),
              torch.randn(128, **kwargs), 1.0e-6)

    def pack_control():
        row_major, scale = fused(*params)
        return pack(row_major), scale

    def pack_candidate():
        return tiled_norm_quant(*params)

    expected_bytes, expected_scales = pack_control()
    observed_bytes, observed_scales = pack_candidate()
    assert exact(expected_bytes, observed_bytes), "GDN tiled activation bytes changed"
    assert exact(expected_scales, observed_scales), "GDN FP8 scales changed"
    if projection is None:
        return timing(pack_control, pack_candidate, repeats)

    def control():
        row_major, scale = fused(*params)
        return projection(row_major, scale, *weights, tiled=True, wperm=True)

    def candidate():
        tiled, scale = pack_candidate()
        return aligned_tiled_projection(projection, tiled, scale, *weights, True)

    assert exact(control(), candidate()), "GDN output projection values changed"
    return timing(control, candidate, repeats)


def get_weights(model):
    from safetensors import safe_open
    from radiance_mxfp4 import make_row_ref, permute_w

    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.language_model.layers.0.linear_attn.out_proj"

    def load(suffix):
        key = f"{prefix}.{suffix}"
        with safe_open(model / index[key], framework="pt", device="cpu") as f:
            return f.get_tensor(key).cuda()

    weight = permute_w(load("weight"), 5120, 6144)
    scale = load("weight_scale").T.contiguous()
    return weight, scale, make_row_ref(scale)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("scan", "projection", "both"), default="both")
    parser.add_argument("--rows", type=int, nargs="+", default=(1648, 3296))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--projection", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.projection) != bool(args.model):
        parser.error("--projection and --model must be provided together")
    if any(not 128 <= rows <= 4096 for rows in args.rows) or args.repeats < 1:
        parser.error("require 128–4096 rows and positive repeats")

    torch.manual_seed(20261126)
    projection = AlignedPrefillGemm(args.projection) if args.projection else None
    weights = get_weights(args.model) if args.model else None
    results = {
        "device": torch.cuda.get_device_name(),
        "projection_includes_gemm": projection is not None,
        "cases": [],
    }
    for rows in args.rows:
        for mode in ("scan", "projection") if args.mode == "both" else (args.mode,):
            outcome = scan_case(rows, args.repeats) if mode == "scan" else projection_case(
                rows, args.repeats, projection, weights
            )
            case = {"mode": mode, "rows": rows, "correctness": "byte_exact", **outcome}
            results["cases"].append(case)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(case), flush=True)


if __name__ == "__main__":
    main()
