"""Compare prefill projection launch geometry without changing split-K arithmetic.

Synthetic activations, weights and folded scales; no conversation data. Report
every output bit, both physical weight layouts, and randomized paired timings.
"""

import argparse
import gc
import json
import random
import statistics
from pathlib import Path

from prefill_gemm_alignment import AlignedPrefillGemm

from qwen_r9700_lab.diagnostic_contract import seal, write_private


def run(args):
    import torch

    baseline = AlignedPrefillGemm(args.baseline)
    variants = [("baseline", baseline)] + [
        (p.name, AlignedPrefillGemm(p)) for p in args.candidates
    ]
    if len({name for name, _ in variants}) != len(variants):
        raise ValueError("candidate names must be unique")
    torch.manual_seed(902729)
    order = random.Random(902729)
    report = {
        "schema": "urn:coherence:prefill-projection-speed:v1",
        "status": "RUNNING",
        "builds": {name: op.manifest["sha256"] for name, op in variants},
        "contract": "Byte equality to qualified four-split FP32 ordered-merge output projection",
        "cases": [],
        "timings": [],
    }
    rows = [9, 65, 257, 1003]
    if args.full:
        rows += [31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1023, 1024, 1648, 2048]
    rows += list(getattr(args, "extra_rows", []))
    for k in (6144, 17408):
        n = 5120
        weight = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
        ref = torch.randint(124, 132, (n,), device="cuda", dtype=torch.uint8)
        # The qualified folded path admits deltas [-6,8]. Exercise each of them
        # rather than using uniform unit scales or underflow-only inputs.
        ws = (
            (
                ref.to(torch.int16).unsqueeze(0)
                + torch.randint(-6, 9, (k // 32, n), device="cuda", dtype=torch.int16)
            )
            .to(torch.uint8)
            .contiguous()
        )
        for m in rows:
            q = (torch.randn((m, k), device="cuda") * 8).to(torch.float8_e4m3fn)
            scale = torch.rand(m, device="cuda") * 0.1 + 0.0001
            for tiled in (False, True) if m in (9, 65) else (True,):
                for wp in (False, True):
                    expected = torch.cat(
                        [
                            baseline(
                                q[first : first + 2048],
                                scale[first : first + 2048],
                                weight,
                                ws,
                                ref,
                                tiled=tiled,
                                wperm=wp,
                            )
                            for first in range(0, m, 2048)
                        ]
                    )
                    for name, op in variants[1:]:
                        actual = op(q, scale, weight, ws, ref, tiled=tiled, wperm=wp)
                        different = int(
                            torch.count_nonzero(
                                actual.view(torch.int16) != expected.view(torch.int16)
                            ).item()
                        )
                        case = {
                            "candidate": name,
                            "rows": m,
                            "k": k,
                            "n": n,
                            "tiled": tiled,
                            "wperm": wp,
                            "different": different,
                            "elements": actual.numel(),
                            "finite": bool(torch.isfinite(actual).all().item()),
                        }
                        report["cases"].append(case)
                        print(json.dumps(case), flush=True)
                        if different or not case["finite"]:
                            report["status"] = "MISMATCH"
                            write_private(args.output, seal(report))
                            raise RuntimeError("projection arithmetic changed")
                        del actual
                    del expected
            if m in args.timing_rows:
                samples = {name: [] for name, _ in variants}
                for _ in range(3):
                    for _, op in variants:
                        op(q, scale, weight, ws, ref, tiled=True, wperm=True)
                torch.cuda.synchronize()
                for _ in range(args.repeats):
                    chosen = variants.copy()
                    order.shuffle(chosen)
                    for name, op in chosen:
                        begin = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        begin.record()
                        actual = op(q, scale, weight, ws, ref, tiled=True, wperm=True)
                        end.record()
                        end.synchronize()
                        samples[name].append(begin.elapsed_time(end))
                        del actual
                report["timings"].append(
                    {
                        "rows": m,
                        "k": k,
                        "samples_ms": samples,
                        "mean_ms": {
                            name: statistics.mean(v) for name, v in samples.items()
                        },
                        "median_ms": {
                            name: statistics.median(v) for name, v in samples.items()
                        },
                    }
                )
            del q, scale
        del weight, ws, ref
        gc.collect()
    report["status"] = "SAMPLE_CHECKED"
    write_private(args.output, seal(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing-rows", type=int, nargs="+", default=[1003, 1648])
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--full", action="store_true")
    run(parser.parse_args())
