"""Qualify the GDN M1 row layout across decode widths, strides and graph replay."""

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
    from stock_m1_gdn_norm import StockM1GdnNorm

    candidate = StockM1GdnNorm()
    torch.manual_seed(619104)
    checks, baseline, graphs = [], [], 0

    def reference(x, z, weight):
        return torch.cat(
            [
                candidate.native.layer_norm_fwd(
                    x[i : i + 48],
                    weight,
                    None,
                    1e-6,
                    z=z[i : i + 48],
                    norm_before_gate=True,
                    is_rms_norm=True,
                    activation="silu",
                )[0]
                for i in range(0, x.shape[0], 48)
            ]
        )

    for count, stride, weight_type in product(range(1, 9), (1, 2), (torch.bfloat16, torch.float32)):
        x = torch.empty((count * 48 * stride, 128), device="cuda", dtype=torch.bfloat16)[::stride]
        z = torch.empty_like(x)
        weight = torch.empty(128, device="cuda", dtype=weight_type)
        graph = None
        for amplitude in (0.001, 1.0, 10000.0):
            x.copy_((torch.randn(x.shape, device="cuda") * amplitude).bfloat16())
            z.copy_((torch.randn_like(z.float()) * 4).bfloat16())
            weight.copy_(torch.randn_like(weight.float()))
            saved = (x.clone(), z.clone(), weight.clone())
            expected = reference(x, z, weight)
            actual = candidate(x, z, weight, 1e-6)
            label = f"width-{count}-stride-{stride}-weight-{weight_type}-amplitude-{amplitude}"
            checks.append(compare_tensors(label, actual, expected, args.output))
            original_batch = candidate.native.layer_norm_fwd(
                x,
                weight,
                None,
                1e-6,
                z=z,
                norm_before_gate=True,
                is_rms_norm=True,
                activation="silu",
            )[0]
            baseline.append(
                compare_tensors("original-" + label, original_batch, expected, args.output)
            )
            if weight_type == torch.bfloat16:
                if graph is None:
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        candidate(x, z, weight, 1e-6)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        replayed = candidate(x, z, weight, 1e-6)
                    graphs += 1
                graph.replay()
                checks.append(compare_tensors("graph-" + label, replayed, expected, args.output))
            if not all(
                torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
                for a, b in zip(saved, (x, z, weight), strict=True)
            ):
                raise DiagnosticError("GDN norm changed a read-only input")
        del graph
    return checks, baseline, graphs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu or not os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"):
        raise DiagnosticError("GDN norm qualification requires admission and shared GPU lease")
    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease

    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    with gpu_lease(args.output / "gpu-lease"):
        checks, baseline, graphs = qualify(args)
    passed = len(checks) == 144 and graphs == 16 and all(r["equal"] for r in checks)
    report = seal(
        {
            "status": "TESTED" if passed else "FAILED",
            "checks": checks,
            "graphs": graphs,
            "stock_batch_comparisons": baseline,
            "public_counterexamples": sum(not row["equal"] for row in baseline),
            "scope": (
                "Widths 1 through 8, contiguous/strided rows, BF16/FP32 weights, three amplitudes; "
                "16 graphs with changed inputs and weights."
            ),
            "formal_equivalence": "UNPROVED",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    write_private(args.output / "probe-result.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "checks": len(checks),
                "graphs": graphs,
                "sha256": report["sha256"],
                "public_counterexamples": report["public_counterexamples"],
            }
        ),
        flush=True,
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
