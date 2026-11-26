"""Compare admitted row shapes to M1, using the exact 60K divergent embedding."""

import json
import statistics
from functools import partial
from pathlib import Path


def run(worker, build, fixture, output):
    import torch
    from wide_input_norm import operators

    op, narrow, wide = operators(build)
    path = Path(fixture).resolve()
    out = Path(output).resolve()
    if (
        not path.is_relative_to("/prefill-diagnosis/refresh")
        or not out.is_relative_to("/prefill-diagnosis/refresh")
        or out.exists()
    ):
        raise ValueError("expected private synthetic capture and new report")
    tensor = torch.load(path, weights_only=True, map_location="cpu")["1.before.args.0"]
    if tensor.shape != (3, 5120) or tensor.dtype != torch.bfloat16:
        raise ValueError("expected the three exact captured input-embedding rows")
    model = getattr(
        worker.model_runner.model, "language_model", worker.model_runner.model
    )
    norm = model.model.layers[0].input_layernorm
    w = norm.weight
    eps = norm.variance_epsilon
    generator = torch.Generator(device="cpu").manual_seed(220417)
    x0 = torch.cat(
        (tensor, torch.randn((20, 5120), generator=generator).to(tensor.dtype))
    ).cuda()
    r0 = torch.randn(x0.shape, generator=generator).to(tensor.dtype).cuda()

    def mismatches(lhs, rhs):
        return [
            int(
                (
                    x.contiguous().view(torch.uint8) != y.contiguous().view(torch.uint8)
                ).sum()
            )
            if x is not None and y is not None
            else 0
            if x is y
            else -1
            for x, y in zip(lhs, rhs, strict=True)
        ]

    original = op._init_fn
    report = {
        "old_binary": narrow.manifest["binary_sha256"],
        "wide_binary": wide.manifest["binary_sha256"],
        "cases": [],
    }

    def measured(fn):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        v = fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000, v

    for residual in (False, True):
        refs = [
            narrow.norm(x0[i : i + 1], r0[i : i + 1] if residual else None, w, eps)
            for i in range(len(x0))
        ]
        once = [
            torch.cat(xs) if xs[0] is not None else None
            for xs in zip(*refs, strict=True)
        ]
        for width in (8, 880, 1648, 2048, 2049, 2528, 3296, 4096):
            indices = torch.arange(width, device=x0.device) % len(x0)
            x = x0[indices]
            r = r0[indices] if residual else None
            reference = [
                v.view(torch.uint8)[indices].view(v.dtype)
                if v is not None and v.element_size() == 1
                else v[indices]
                if v is not None
                else None
                for v in once
            ]
            old = partial(original, x, r, w, eps, "0/input_layernorm")
            new = partial(wide.norm, x, r, w, eps)
            old()
            new()
            times = [[], []]
            compared = []
            for j in range(9):
                for k in (1, 0) if j % 2 else (0, 1):
                    us, ret = measured((old, new)[k])
                    times[k].append(us)
                    if j == 0:
                        compared.append(mismatches(ret, reference))
            case = {
                "rows": width,
                "residual": residual,
                "old_differing_bytes": compared[0],
                "new_differing_bytes": compared[1],
                "old_us": statistics.median(times[0]),
                "new_us": statistics.median(times[1]),
                "repeats": len(times[0]),
            }
            report["cases"].append(case)
            out.write_text(json.dumps(report, indent=2) + "\n")
            if any(compared[1]):
                raise ValueError(f"wide M1 arithmetic mismatch at width {width}")
    return report
