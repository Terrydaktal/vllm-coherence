"""Compare optimized attention to independent FP64 dense attention on synthetic data."""

import argparse
import json
import math
from pathlib import Path

from attention_precision import digest
from attention_precision_runtime import PrecisionAttention
from probe_m1_attention_precision import R4D_SHA256, backend_stopped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backend_stopped("http://127.0.0.1:8080")
    import r4d
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(192419)
    if digest(r4d.__file__) != R4D_SHA256:
        raise ValueError("oracle negative control changed")
    fixed = PrecisionAttention(args.build)
    checks = []
    for dtype in (torch.float8_e4m3fn, torch.bfloat16):
        for length in (32, 64, 127, 256, 511, 512, 1023, 1024, 2047, 2048):
            width, blocks = 32, math.ceil(length / 16)
            query = (torch.randn(width, 24, 256) * 0.5).bfloat16().cuda()
            kv = torch.randn(blocks, 4, 16, 512).to(dtype).cuda()
            table = torch.arange(blocks, dtype=torch.int32, device="cuda")[None]
            lengths = torch.tensor([length], dtype=torch.int32, device="cuda")
            scratch = torch.empty(32 * 24 * 32 * 1032, dtype=torch.uint8, device="cuda")
            q = query.cpu().double().transpose(0, 1)
            flat = (
                kv.cpu()
                .float()
                .permute(1, 0, 2, 3)
                .reshape(4, -1, 512)[:, :length]
                .double()
            )
            k = flat[:, :, :256].repeat_interleave(6, dim=0)
            v = flat[:, :, 256:].repeat_interleave(6, dim=0)
            score = (q @ k.transpose(-1, -2)) / 16
            mask = (
                torch.arange(length)[None, :]
                > torch.arange(length - width, length)[:, None]
            )
            score.masked_fill_(mask[None], -torch.inf)
            expected = (score.softmax(-1) @ v).transpose(0, 1).contiguous()
            phase_checks = {}
            for phase in ("decode", "prefill"):
                outputs = {}
                for name, engine in (("old", r4d), ("fixed", fixed)):
                    output = torch.empty_like(query)
                    dtype_name = "bf16" if dtype == torch.bfloat16 else "fp8"
                    fn = getattr(engine, f"attn_{phase}_h256_gqa6_{dtype_name}kv")
                    for row in range(width) if phase == "decode" else (None,):
                        if row is None:
                            qi, out, lens, rows = query, output, lengths, width
                        else:
                            qi, out, rows = (
                                query[row : row + 1],
                                output[row : row + 1],
                                1,
                            )
                            lens = torch.tensor(
                                [length - width + row + 1],
                                dtype=torch.int32,
                                device="cuda",
                            )
                        fn(
                            qi.data_ptr(),
                            kv.data_ptr(),
                            table.data_ptr(),
                            lens.data_ptr(),
                            out.data_ptr(),
                            0,
                            0,
                            scratch.data_ptr(),
                            1,
                            rows,
                            24,
                            4,
                            256,
                            16,
                            blocks,
                            kv.stride(0),
                            kv.stride(1),
                            1 / 16,
                            0,
                            length,
                            torch.cuda.current_stream().cuda_stream,
                        )
                    values = output.cpu().double()
                    error = values - expected
                    outputs[name] = {
                        "rms": float(error.square().mean().sqrt()),
                        "max_abs": float(error.abs().max()),
                        "bf16_equal_elements": int(
                            (values == expected.bfloat16().double()).sum()
                        ),
                        "finite": bool(torch.isfinite(values).all()),
                    }
                # Independent tolerance chosen for BF16 output at these input scales.
                # Also require improvement against the old path on the same inputs.
                ok = (
                    outputs["fixed"]["finite"]
                    and outputs["fixed"]["max_abs"] <= 0.016
                    and outputs["fixed"]["rms"] <= outputs["old"]["rms"] * 1.01
                )
                phase_checks[phase] = {"outputs": outputs, "passed": ok}
            record = {
                "dtype": str(dtype),
                "context": length,
                "rows": width,
                "elements": expected.numel(),
                "phases": phase_checks,
            }
            checks.append(record)
            print(
                json.dumps(
                    {
                        "dtype": str(dtype),
                        "context": length,
                        "passed": all(c["passed"] for c in phase_checks.values()),
                        "rms": {
                            p: {n: v["rms"] for n, v in c["outputs"].items()}
                            for p, c in phase_checks.items()
                        },
                    }
                ),
                flush=True,
            )
    report = {
        "status": "SAMPLE_CHECKED"
        if all(v["passed"] for c in checks for v in c["phases"].values())
        else "MISMATCH",
        "checks": checks,
        "build_sha256": digest(args.build / "build.json"),
        "probe_sha256": digest(__file__),
        "reference": "independent dense CPU FP64, exact widened stored BF16/FP8 inputs; tolerance-based, not bit-exact arbitrary-input proof",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "SAMPLE_CHECKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
