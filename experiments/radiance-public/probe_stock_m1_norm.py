"""Independent native M1-vs-batch norm diagnostic and candidate qualification."""

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
    original = [rms_norm.impls["native"].impl_fn, fused_add_rms_norm.impls["native"].impl_fn]
    torch.manual_seed(14371)
    checks, baseline, moments_checks = [], [], []
    for count, magnitude, groups in product((1, 2, 7, 8, 9, 65), (0.001, 1.0, 10000.0), (1, 4, 24)):
        width = 5120 if groups == 1 else 256
        shape = (count, width) if groups == 1 else (count, groups, width)
        x = (torch.randn(shape, device="cuda") * magnitude).bfloat16()
        residual = (torch.randn_like(x.float()) * magnitude).bfloat16()
        weight = (torch.randn(width, device="cuda") * 0.1).bfloat16()
        widened = weight.float() + 1
        for with_residual in (False, True) if groups == 1 else (False,):
            serial, serial_residual = [], []
            for row in range(count):
                if with_residual:
                    result, carry = original[1](
                        x[row : row + 1], residual[row : row + 1], widened, 1e-6
                    )
                    serial_residual.append(carry)
                else:
                    result = original[0](x[row : row + 1], widened, 1e-6)
                serial.append(result)
            serial = torch.cat(serial)
            expected_residual = torch.cat(serial_residual) if with_residual else None
            batched = (
                original[1](x, residual, widened, 1e-6)[0]
                if with_residual
                else original[0](x, widened, 1e-6)
            )
            moments = torch.empty((count * groups, 2), device=x.device, dtype=torch.float32)
            actual = candidate(
                x, residual if with_residual else None, weight, 1e-6, moments=moments
            )
            if with_residual:
                output, carry = actual
            else:
                output = actual
            label = (
                f"rows-{count}-groups-{groups}-amplitude-{magnitude}-residual-{int(with_residual)}"
            )
            baseline.append(compare_tensors("stock-" + label, batched, serial, args.output))
            checks.append(compare_tensors("candidate-" + label, output, serial, args.output))
            variances, inverses = [], []
            for row in range(count):
                z = x[row : row + 1].float()
                if with_residual:
                    z = z + residual[row : row + 1].float()
                var = z.pow(2).mean(-1)
                variances.append(var.reshape(-1))
                inverses.append(torch.rsqrt(var + 1e-6).reshape(-1))
            for col, (name, expected) in enumerate(
                (("variance", variances), ("inverse", inverses))
            ):
                moments_checks.append(
                    compare_tensors(
                        name + "-" + label,
                        moments[:, col].contiguous(),
                        torch.cat(expected),
                        args.output,
                    )
                )
            if with_residual:
                checks.append(
                    compare_tensors("carry-" + label, carry, expected_residual, args.output)
                )
    return checks, baseline, moments_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu or not os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"):
        raise DiagnosticError("stock normalization probe requires admission and shared GPU lease")
    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease

    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    checks, baseline, moments_checks, error = [], [], [], None
    try:
        with gpu_lease(args.output / "gpu-lease"):
            checks, baseline, moments_checks = qualify(args)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    status = (
        "TESTED"
        if error is None
        and len(checks) == 90
        and len(moments_checks) == 144
        and all(x["equal"] for x in [*checks, *moments_checks])
        else "FAILED"
    )
    result = seal(
        {
            "status": status,
            "error": error,
            "checks": checks,
            "stock_batch_comparisons": baseline,
            "moment_comparisons": moments_checks,
            "scope": (
                "Six batch sizes, three amplitudes, hidden norm with/without residual, "
                "Q24/K4 head norm, against independent stock M1 calls."
            ),
            "formal_equivalence": "UNPROVED",
            "build": json.loads((args.build / "build.json").read_text())["sha256"],
            "probe_source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    write_private(args.output / "probe-result.json", result)
    print(json.dumps({k: result[k] for k in ("status", "error", "sha256")}), flush=True)
    return 0 if status == "TESTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
