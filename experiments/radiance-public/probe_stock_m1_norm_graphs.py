"""Check captured stock-M1 normalization with changing inputs and weights."""

import argparse
import hashlib
import json
import os
from itertools import product
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal, write_private


def qualify(args):
    import torch
    from probe_stock_gdn_sequence import compare_tensors
    from stock_m1_norm import StockM1Norm
    from vllm.ir.ops import fused_add_rms_norm, rms_norm

    candidate = StockM1Norm(args.build)
    originals = [rms_norm.impls["native"].impl_fn, fused_add_rms_norm.impls["native"].impl_fn]
    torch.manual_seed(414370)
    checks = []
    captures = 0
    for count, groups in product((1, 2, 7, 8, 9, 65), (1, 4, 24)):
        width = 5120 if groups == 1 else 256
        shape = (count, width) if groups == 1 else (count, groups, width)
        x = torch.randn(shape, device="cuda").bfloat16()
        residual = torch.randn_like(x)
        weight = (torch.randn(width, device="cuda") * 0.1).bfloat16()
        for with_residual in (False, True) if groups == 1 else (False,):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    candidate(x, residual if with_residual else None, weight, 1e-6)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                actual = candidate(x, residual if with_residual else None, weight, 1e-6)
            captures += 1
            for amplitude in (0.001, 1.0, 10000.0):
                x.copy_((torch.randn(shape, device="cuda") * amplitude).bfloat16())
                residual.copy_((torch.randn(shape, device="cuda") * amplitude).bfloat16())
                weight.copy_((torch.randn(width, device="cuda") * 0.1).bfloat16())
                saved_x, saved_residual, saved_weight = x.clone(), residual.clone(), weight.clone()
                graph.replay()
                if not all(
                    torch.equal(before.view(torch.uint8), after.view(torch.uint8))
                    for before, after in (
                        (saved_x, x),
                        (saved_residual, residual),
                        (saved_weight, weight),
                    )
                ):
                    raise DiagnosticError("captured normalization changed an input")
                expected, carries = [], []
                widened = weight.float() + 1
                for i in range(count):
                    if with_residual:
                        y, carry = originals[1](x[i : i + 1], residual[i : i + 1], widened, 1e-6)
                        carries.append(carry)
                    else:
                        y = originals[0](x[i : i + 1], widened, 1e-6)
                    expected.append(y)
                label = (
                    f"rows-{count}-groups-{groups}-residual-{with_residual}-amplitude-{amplitude}"
                )
                output = actual[0] if with_residual else actual
                checks.append(compare_tensors(label, output, torch.cat(expected), args.output))
                if with_residual:
                    checks.append(
                        compare_tensors(
                            label + "-carry", actual[1], torch.cat(carries), args.output
                        )
                    )
            del graph
    return {"checks": checks, "captures": captures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu or not os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"):
        raise DiagnosticError("normalization graph probe requires admission and shared GPU lease")
    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease

    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    result, error = {}, None
    try:
        with gpu_lease(args.output / "gpu-lease"):
            result = qualify(args)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    passed = (
        error is None
        and result.get("captures") == 24
        and len(result.get("checks", [])) == 90
        and all(row["equal"] for row in result["checks"])
    )
    report = seal(
        {
            "status": "TESTED" if passed else "FAILED_OR_DISCREPANT",
            "error": error,
            **result,
            "formal_equivalence": "UNPROVED",
            "scope": (
                "24 captured graphs, three input amplitudes; changed inputs, residual and weights."
            ),
            "build": json.loads((args.build / "build.json").read_text())["sha256"],
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    write_private(args.output / "probe-result.json", report)
    print(json.dumps({k: report[k] for k in ("status", "error", "sha256")}), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
