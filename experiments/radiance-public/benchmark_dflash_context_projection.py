"""Measure partition invariance with the installed DFlash weights and synthetic captured states."""

import json
import statistics
from pathlib import Path

import torch
from dflash_context_projection import project
from torch.nn import functional


def run(worker, input_path, output_path):
    file, dest = Path(input_path).resolve(), Path(output_path).resolve()
    if (
        not file.is_relative_to("/prefill-diagnosis/refresh")
        or not dest.is_relative_to("/bench")
        or dest.exists()
    ):
        raise ValueError("expected fresh private synthetic fixture report")
    src = torch.load(file, map_location="cpu", weights_only=True)
    if src.shape != (2049, 5120) or src.dtype != torch.bfloat16:
        raise ValueError("unexpected synthetic source row layout")
    model = worker.model_runner.speculator.model.model
    weight, bias = model._fused_kv_weight, model._fused_kv_bias
    x = src.repeat(2, 1)[:3296].to(weight.device).contiguous()

    def original(act):
        return functional.linear(act, weight, bias)

    def chunked(act, n):
        blocks = []
        for i in range(0, len(act), n):
            piece = act[i : i + n]
            if len(piece) < n:
                padded = torch.empty((n, act.shape[1]), dtype=act.dtype, device=act.device)
                padded[: len(piece)].copy_(piece)
                padded[len(piece) :].zero_()
                blocks.append(original(padded)[: len(piece)])
            else:
                blocks.append(original(piece))
        return torch.cat(blocks) if len(blocks) > 1 else blocks[0]

    variants = {
        "release": original,
        "32x128x64": lambda a: project(a, weight, bias, bm=32),
        "64x128x64": lambda a: project(a, weight, bias, bm=64),
        "32x64x64": lambda a: project(a, weight, bias, bm=32, bn=64),
        "chunk128": lambda a: chunked(a, 128),
    }
    groups = [(2049, 1648, 401), (2528, 1648, 880), (3296, 1648, 1648)]
    widths = (1, 8, 401, 880, 1648, 2049, 2528, 3296)
    report = {
        "weight_dtype": str(weight.dtype),
        "weight_shape": list(weight.shape),
        "bias": bias is not None,
        "variants": {},
    }
    for label, fn in variants.items():
        item = report["variants"][label] = {"equivalence": {}, "time_us": {}}
        for full, left, right in groups:
            assert left + right == full
            monolithic = fn(x[:full]).cpu()
            second = fn(x[:full]).cpu()
            split = torch.cat((fn(x[:left]).cpu(), fn(x[left:full]).cpu()))
            m = monolithic.view(torch.int16) != split.view(torch.int16)
            item["equivalence"][str(full)] = {
                "differing": int(m.sum()),
                "differing_rows": int(m.any(-1).sum()),
                "elements": m.numel(),
                "repeat_exact": torch.equal(monolithic.view(torch.int16), second.view(torch.int16)),
            }
            del split, second, m
        for m in widths:
            a = x[:m]
            fn(a)
            samples = []
            for _ in range(5):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(3):
                    fn(a)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000 / 3)
            item["time_us"][str(m)] = round(statistics.median(samples), 3)
        dest.write_text(json.dumps(report, indent=2) + "\n")
    return report


def compare_release(worker, input_path, output_path):
    """Compare selected partition-invariant arithmetic with the installed BF16 GEMM."""
    source = Path(input_path).resolve()
    dest = Path(output_path).resolve()
    if (
        not source.is_relative_to("/prefill-diagnosis/refresh")
        or not dest.is_relative_to("/bench")
        or dest.exists()
    ):
        raise ValueError("expected private fixture and fresh report")
    model = worker.model_runner.speculator.model.model
    weight, bias = model._fused_kv_weight, model._fused_kv_bias
    a = torch.load(source, map_location="cpu", weights_only=True)
    if a.shape != (2049, 5120) or a.dtype != torch.bfloat16:
        raise ValueError("expected authenticated 2049-row synthetic capture")
    x = a.repeat(2, 1)[:3296].to(weight.device).contiguous()
    norm = {}

    def fixed(rows):
        return project(rows, weight, bias, bm=64, bn=128, bk=64)

    for width in (1, 8, 401, 880, 1648, 2049, 2528, 3296):
        act = x[:width]
        old, new = functional.linear(act, weight, bias), fixed(act)
        old_i, new_i = old.view(torch.int16), new.view(torch.int16)
        different = (old_i != new_i).sum().item()
        minreg = maxreg = diff2 = 0.0
        maxulp = 0
        for start in range(0, width, 128):
            orig = old[start : start + 128]
            changed = new[start : start + 128]
            delta = changed.float() - orig.float()
            diff2 += delta.square().sum().item()
            minreg, maxreg = min(minreg, delta.min().item()), max(maxreg, delta.max().item())
            lhs, rhs = orig.view(torch.int16).int(), changed.view(torch.int16).int()
            lhs = torch.where(lhs < 0, -32768 - lhs, lhs)
            rhs = torch.where(rhs < 0, -32768 - rhs, rhs)
            maxulp = max(maxulp, (lhs - rhs).abs().max().item())
        single = fixed(act[-1:])
        suffix_rows = min(8, width)
        suffix = fixed(act[-suffix_rows:])
        norm[str(width)] = {
            "total_elements": old.numel(),
            "changed_elements": different,
            "finite": bool(torch.isfinite(old).all() and torch.isfinite(new).all()),
            "max_abs": max(-minreg, maxreg),
            "rms": (diff2 / old.numel()) ** 0.5,
            "max_bfloat16_ulp": maxulp,
            "last_m1_row_exact": torch.equal(new[-1:].view(torch.int16), single.view(torch.int16)),
            "last_m8_rows_exact": torch.equal(
                new[-suffix_rows:].view(torch.int16), suffix.view(torch.int16)
            ),
        }
    dest.write_text(json.dumps(norm, indent=2) + "\n")
    return norm
