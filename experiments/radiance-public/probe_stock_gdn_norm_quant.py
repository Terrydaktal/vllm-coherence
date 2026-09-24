"""Exact GDN norm/quant bytes and scales versus the unchanged corrected path."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def run(args):
    import torch
    from safetensors import safe_open
    from stock_gdn_norm_quant import fused
    from stock_m1_gdn_norm import StockM1GdnNorm
    from vllm import _custom_ops as ops

    torch.set_num_threads(4)
    torch.manual_seed(91848128)
    oracle = StockM1GdnNorm()
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    names = sorted(k for k in index if k.endswith(".linear_attn.norm.weight"))
    if len(names) != 48:
        raise ValueError("checkpoint GDN inventory changed")
    weights = []
    for name in names:
        with safe_open(args.model / index[name], framework="pt", device="cpu") as f:
            weights.append(f.get_tensor(name).to("cuda"))
    report = {
        "status": "RUNNING",
        "input_rows_per_site": args.rows,
        "sites": 48,
        "checks": [],
        "timings": [],
        "private_chat_read": False,
        "row_invariant": args.row_invariant,
        "reference": "Pinned corrected FLA GDN norm, then unmodified native FP8 quantizer",
        "source_sha256": hashlib.sha256(
            Path(__file__).with_name("stock_gdn_norm_quant.py").read_bytes()
        ).hexdigest(),
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def reference(x, z, w):
        xx, zz = x.reshape(-1, 128), z.reshape(-1, 128)
        if args.row_invariant and x.shape[0] > 8:
            # The M1 oracle wrapper admits at most eight token rows. Preserve
            # its arithmetic by partitioning, not by bypassing that admission.
            yy = torch.cat(
                [
                    oracle(xx[start : start + 384], zz[start : start + 384], w, 1e-6)
                    for start in range(0, len(xx), 384)
                ],
                dim=0,
            )
        elif x.shape[0] <= 8:
            yy = oracle(xx, zz, w, 1e-6)
        else:
            yy = oracle.native.layer_norm_fwd(
                xx,
                w,
                None,
                1e-6,
                z=zz,
                norm_before_gate=True,
                is_rms_norm=True,
                activation="silu",
            )[0]
        return ops.scaled_fp8_quant(
            yy.reshape(x.shape[0], 6144), scale=None, use_per_token_if_dynamic=True
        )

    def equal(a, b):
        return torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )

    x = torch.randn(args.rows, 48, 128, device="cuda", dtype=torch.bfloat16)
    backing = torch.randn(args.rows, 64, 128, device="cuda", dtype=torch.bfloat16)
    z = backing[:, :48]
    z.mul_(4)
    x[0].zero_()
    x[1] *= 1e-20
    x[2] *= 10000
    z[3].fill_(-64)
    z[4].fill_(64)
    for site, w in enumerate(weights):
        for width in (1, 8):
            good = 0
            for start in range(0, args.rows, width):
                xx, zz = x[start : start + width], z[start : start + width]
                a, b = reference(xx, zz, w), fused(xx, zz, w, 1e-6)
                if not all(equal(i, j) for i, j in zip(a, b, strict=True)):
                    report["failure"] = {
                        "site": site,
                        "width": width,
                        "row": start,
                        "unequal_bytes": [
                            int(
                                (
                                    i.contiguous().view(torch.uint8)
                                    != j.contiguous().view(torch.uint8)
                                ).sum()
                            )
                            for i, j in zip(a, b, strict=True)
                        ],
                    }
                    report["status"] = "FAILED"
                    save()
                    raise RuntimeError(
                        "GDN norm/quant mismatch; numerical aggregate saved"
                    )
                good += xx.shape[0]
            report["checks"].append(
                {"site": site, "width": width, "matching_rows": good}
            )
        save()
    for width in (9, 320, 1000, 1648, 2048):
        xx = x.repeat((3, 1, 1))[:width].contiguous()
        zz = z.repeat((3, 1, 1))[:width].contiguous()
        a, b = reference(xx, zz, weights[0]), fused(xx, zz, weights[0], 1e-6)
        ok = all(equal(i, j) for i, j in zip(a, b, strict=True))
        report["checks"].append({"prefill_width": width, "equal": ok})
        save()
        if not ok:
            report["status"] = "FAILED"
            save()
            raise RuntimeError(
                "prefill norm/quant differs from the selected arithmetic contract"
            )
    mutated = b[0].clone()
    mutated.view(torch.uint8).reshape(-1)[0] ^= 1
    report["negative_control_detected"] = not equal(b[0], mutated)
    for width in (1, 8, 1648):
        xx = x.repeat((2, 1, 1))[:width].contiguous()
        zz = z.repeat((2, 1, 1))[:width].contiguous()
        w = weights[0]
        expected = reference(xx, zz, w)
        for name, fn in (
            ("reference", lambda xx=xx, zz=zz, w=w: reference(xx, zz, w)),
            ("fused", lambda xx=xx, zz=zz, w=w: fused(xx, zz, w, 1e-6)),
        ):
            fn()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = fn()
            for t in output:
                t.view(torch.uint8).fill_(42)
            graph.replay()
            if not all(equal(i, j) for i, j in zip(expected, output, strict=True)):
                raise RuntimeError("graph replay mismatch")
            # One replay contains many operators; Python launch gaps must not
            # be charged to a kernel that takes only a few microseconds.
            repetitions = 32 if width > 8 else 100
            packed = torch.cuda.CUDAGraph()
            with torch.cuda.graph(packed):
                for _ in range(repetitions):
                    fn()
            for _ in range(3):
                packed.replay()
            samples = []
            for _ in range(9):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                packed.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / repetitions)
            report["timings"].append(
                {"width": width, "path": name, "median_ms": statistics.median(samples)}
            )
    report["status"] = "SAMPLE_CHECKED"
    save()
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "sites",
                    "input_rows_per_site",
                    "negative_control_detected",
                    "timings",
                )
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("/models/Qwen3.8-27B-Uncensored-MXFP4-awq")
    )
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument(
        "--row-invariant",
        action="store_true",
        help="Use the declared M1 arithmetic for prefill rows too",
    )
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
