"""Compare candidate prefill attention with the qualified arithmetic, not tokens.

Synthetic queries and paged KV only. Every output element must be byte exact;
timing uses the same inputs, randomized candidate order and GPU events. No model
weights, chat messages, or production snapshots are read or modified.
"""

import argparse
import gc
import json
import random
import statistics
from pathlib import Path

from prefill_attention_alignment import AlignedPrefillAttention

from qwen_r9700_lab.diagnostic_contract import seal, write_private


def run(args):
    import torch

    baseline = AlignedPrefillAttention(args.baseline)
    variants = [("baseline", baseline)] + [
        (p.name, AlignedPrefillAttention(p)) for p in args.candidates
    ]
    assert len({name for name, _ in variants}) == len(variants)
    torch.manual_seed(902728)
    order = random.Random(902728)
    report = {
        "schema": "urn:coherence:prefill-attention-speed:v1",
        "status": "RUNNING",
        "builds": {name: op.manifest["sha256"] for name, op in variants},
        "contract": "Byte equality to qualified TILE16 prefill, plus independent query workgroups on boundary cases",
        "cases": [],
        "timings": [],
    }
    scratch_bytes = max(op.manifest["scratch_bytes"] for _, op in variants)
    scratch = torch.empty(scratch_bytes, device="cuda", dtype=torch.uint8)
    shapes = [
        (17, 17),
        (65, 1030),
        (129, 2033),
        (65, 60000),
        (65, 200000),
        (65, 253792),
    ]
    if args.full:
        shapes += [
            (9, 9),
            (31, 79),
            (32, 1025),
            (33, 1030),
            (63, 4097),
            (64, 4096),
            (127, 61111),
            (128, 61112),
            (255, 65537),
            (256, 65536),
            (257, 65537),
            (1003, 60000),
            (1648, 200000),
            (2048, 253792),
        ]
    shapes += [tuple(shape) for shape in getattr(args, "extra_shapes", [])]

    def reference(q, kv, table, lengths, output, ks, vs):
        # The qualified baseline admits at most 2048 query rows. Larger
        # candidates must equal its chronological, independently split chunks.
        for first in range(0, len(q), 2048):
            last = min(first + 2048, len(q))
            baseline(
                q[first:last],
                kv,
                table,
                lengths - len(q) + last,
                scratch,
                output[first:last],
                ks=ks,
                vs=vs,
            )

    def inputs(rows, context, dtype, salt):
        blocks = (context + 15) // 16
        # Generate in small chunks: the near-limit BF16 case should not retain
        # a full-size FP32 temporary in addition to its KV allocation.
        physical_blocks = min(blocks, getattr(args, "physical_pages", None) or blocks)
        kv = torch.empty((physical_blocks, 4, 16, 512), device="cuda", dtype=dtype)
        for first in range(0, physical_blocks, 512):
            tail = min(first + 512, physical_blocks)
            kv[first:tail] = torch.randn((tail - first, 4, 16, 512), device="cuda") * (
                0.125 if salt % 2 else 1.0
            )
        q = (
            torch.randn((rows, 24, 256), device="cuda") * (1.0 if salt % 2 else 8.0)
        ).bfloat16()
        table = (
            torch.randperm(blocks, device="cuda", dtype=torch.int64)
            .to(torch.int32)
            .view(1, -1)
        )
        table.remainder_(physical_blocks)
        lengths = torch.tensor([context], device="cuda", dtype=torch.int32)
        ks = torch.tensor([0.125, 1.0, 3.0, 16.0], device="cuda")
        vs = torch.tensor([0.25, 1.0, 2.0, 0.5], device="cuda")
        return q, kv, table, lengths, ks, vs

    dtypes = [getattr(torch, name) for name in args.kv_formats]
    for dtype in dtypes:
        for salt, (rows, context) in enumerate(shapes):
            q, kv, table, lengths, ks, vs = inputs(rows, context, dtype, salt)
            expected = torch.empty_like(q)
            reference(q, kv, table, lengths, expected, ks, vs)
            if rows <= 65:
                # Separately scheduled one-query workgroups catch split-boundary
                # or group-lifetime mistakes shared by batched implementations.
                single = torch.empty_like(q)
                for row in range(rows):
                    lens = torch.tensor(
                        [context - rows + row + 1], device="cuda", dtype=torch.int32
                    )
                    baseline(
                        q[row : row + 1],
                        kv,
                        table,
                        lens,
                        scratch,
                        single[row : row + 1],
                        ks=ks,
                        vs=vs,
                    )
                assert torch.equal(
                    single.view(torch.int16), expected.view(torch.int16)
                ), "qualified reference batch disagrees with serial query workgroups"
                del single, lens
            for name, op in variants[1:]:
                actual = torch.full_like(q, float("nan"))
                op(
                    q,
                    kv,
                    table,
                    lengths,
                    scratch,
                    actual,
                    ks=ks,
                    vs=vs,
                    context_limit=context,
                )
                different = int(
                    torch.count_nonzero(
                        actual.view(torch.int16) != expected.view(torch.int16)
                    ).item()
                )
                finite = bool(torch.isfinite(actual).all().item())
                case = {
                    "candidate": name,
                    "rows": rows,
                    "context": context,
                    "dtype": str(dtype),
                    "different": different,
                    "elements": actual.numel(),
                    "finite": finite,
                    "serial_checked": rows <= 65,
                    "physical_pages": kv.shape[0],
                }
                report["cases"].append(case)
                print(json.dumps(case), flush=True)
                if different or not finite:
                    report["status"] = "MISMATCH"
                    write_private(args.output, seal(report))
                    raise RuntimeError("candidate changes prefill arithmetic")
                del actual
            del q, kv, table, lengths, ks, vs, expected
            gc.collect()
            torch.cuda.empty_cache()

    for context in args.contexts:
        rows = min(args.rows, context)
        q, kv, table, lengths, ks, vs = inputs(rows, context, torch.float8_e4m3fn, 1)
        outputs = {name: torch.empty_like(q) for name, _ in variants}
        for _ in range(3):
            for name, op in variants:
                op(
                    q,
                    kv,
                    table,
                    lengths,
                    scratch,
                    outputs[name],
                    ks=ks,
                    vs=vs,
                    context_limit=context,
                )
        torch.cuda.synchronize()
        samples = {name: [] for name, _ in variants}
        for _ in range(args.repeats):
            chosen = variants.copy()
            order.shuffle(chosen)
            for name, op in chosen:
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                begin.record()
                op(
                    q,
                    kv,
                    table,
                    lengths,
                    scratch,
                    outputs[name],
                    ks=ks,
                    vs=vs,
                    context_limit=context,
                )
                end.record()
                end.synchronize()
                samples[name].append(begin.elapsed_time(end))
        for name, _ in variants[1:]:
            assert torch.equal(
                outputs[name].view(torch.int16), outputs["baseline"].view(torch.int16)
            )
        record = {
            "context": context,
            "rows": rows,
            "samples_ms": samples,
            "mean_ms": {
                name: statistics.mean(values) for name, values in samples.items()
            },
            "median_ms": {
                name: statistics.median(values) for name, values in samples.items()
            },
        }
        report["timings"].append(record)
        print(json.dumps(record), flush=True)
        del q, kv, table, lengths, ks, vs, outputs
        gc.collect()
    report["status"] = "SAMPLE_CHECKED"
    write_private(args.output, seal(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[2651, 60000, 200000]
    )
    parser.add_argument("--rows", type=int, default=1003)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--kv-formats",
        nargs="+",
        choices=("float8_e4m3fn", "bfloat16"),
        default=["float8_e4m3fn", "bfloat16"],
    )
    run(parser.parse_args())
