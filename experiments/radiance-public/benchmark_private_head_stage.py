"""Native head-only timing and recall on consecutive private Pi hidden states."""

from __future__ import annotations

import hashlib
import json
import random
import statistics
import time
import types
from pathlib import Path

from private_head_probe import apply_variant


def summarize(values):
    ordered = sorted(values)
    return {
        "samples": len(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "min_ms": min(values),
        "max_ms": max(values),
    }


def run(spec, root: Path, reports: Path):
    import radiance_drafthead as dh
    import torch
    from safetensors import safe_open
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

    torch.set_num_threads(2)
    model = Path(spec["native_config"]["model"])
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shard = model / index["weight_map"]["lm_head.weight"]
    with safe_open(str(shard), framework="pt", device="cpu") as source:
        weight = source.get_tensor("lm_head.weight").to("cuda")
    marker = json.loads(next(root.glob("hook-*.json")).read_text())
    if (
        marker["head_dtype"] not in ("None", "torch.bfloat16")
        or marker["quant_method"] != "UnquantizedEmbeddingMethod"
    ):
        raise ValueError("unreviewed target reference dispatch; stage timing suppressed")
    head = types.SimpleNamespace(weight=weight, quant_method=UnquantizedEmbeddingMethod())
    state = types.SimpleNamespace(head_dtype=None, _radiance_topk_only=True)
    exact = types.MethodType(LogitsProcessor._apply_head, state)
    packing = dh._quantize_head_now(state, head)
    state._radiance_fast_head = state._apply_head
    state._radiance_exact_head = exact
    state._radiance_topk_only = True
    modes = ("full", "block80", "global128", "global256")
    totals = {
        mode: {
            "rows": 0,
            "argmax_retained": 0,
            "any_maximum_retained": 0,
            "top20_complete": 0,
            "retained_logit_mismatches": 0,
            "argmax_equal": 0,
            "max_retained_logit_difference": 0.0,
        }
        for mode in modes
    }
    timings = {mode: [] for mode in modes}
    wall_timings = {mode: [] for mode in modes}
    shape_timings = {}
    files = sorted(root.glob("head-*.pt"))
    if not files:
        raise ValueError("no consecutive head captures")
    measurement_points = min(256, len(files))
    measured_indices = (
        {round(i * (len(files) - 1) / (measurement_points - 1)) for i in range(measurement_points)}
        if len(files) > 1
        else {0}
    )
    randomizer = random.Random(731)
    shapes = {}
    sources = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), Path(dh.__file__)]
    }
    start = time.monotonic()
    for number, file in enumerate(files):
        saved = torch.load(file, map_location="cpu", weights_only=True)
        hidden = saved["hidden"].to("cuda")
        rows = len(hidden)
        shapes[str(rows)] = shapes.get(str(rows), 0) + 1
        actual_reference = exact(head, hidden, None)
        if "reference_sha256" in saved:
            copied = actual_reference.detach().cpu().contiguous()
            actual_sha = hashlib.sha256(copied.view(torch.uint8).numpy().tobytes()).hexdigest()
            if (
                actual_sha != saved["reference_sha256"]
                or list(copied.shape) != saved["reference_shape"]
                or str(copied.dtype) != saved["reference_dtype"]
            ):
                raise ValueError("standalone full head differs from captured reference digest")
            reference = actual_reference
        else:
            reference = saved["reference"].to("cuda")
            if not torch.equal(actual_reference, reference):
                raise ValueError("standalone full head differs from captured in-model reference")
        threshold = reference.topk(20, dim=-1).values[:, -1:]
        maxima = reference.max(-1, keepdim=True).values
        winner = reference.argmax(-1, keepdim=True)
        for mode in modes:
            result = apply_variant(state, head, hidden, None, mode)
            finite = torch.isfinite(result)
            delta = torch.where(finite, (result.float() - reference.float()).abs(), 0)
            counter = totals[mode]
            counter["rows"] += rows
            counter["argmax_retained"] += int(finite.gather(1, winner).sum())
            counter["any_maximum_retained"] += int((finite & (reference == maxima)).any(-1).sum())
            counter["top20_complete"] += int((finite | (reference < threshold)).all(-1).sum())
            counter["argmax_equal"] += int((result.argmax(-1) == winner[:, 0]).sum())
            counter["retained_logit_mismatches"] += int((delta != 0).sum())
            counter["max_retained_logit_difference"] = max(
                counter["max_retained_logit_difference"], float(delta.max())
            )
        if number in measured_indices:
            # Warm every implementation/shape before measuring. Alternate order.
            for mode in modes:
                for _ in range(3):
                    apply_variant(state, head, hidden, None, mode)
            torch.cuda.synchronize()
            for _ in range(5):
                order = list(modes)
                randomizer.shuffle(order)
                for mode in order:
                    before = torch.cuda.Event(enable_timing=True)
                    after = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    wall_start = time.perf_counter()
                    before.record()
                    output = apply_variant(state, head, hidden, None, mode)
                    after.record()
                    after.synchronize()
                    wall_ms = (time.perf_counter() - wall_start) * 1000
                    elapsed_ms = before.elapsed_time(after)
                    timings[mode].append(elapsed_ms)
                    wall_timings[mode].append(wall_ms)
                    shape_timings.setdefault(str(rows), {}).setdefault(mode, []).append(elapsed_ms)
                    del output
        if number % 256 == 0 or number == len(files) - 1:
            print(
                json.dumps(
                    {
                        "phase": "head_stage",
                        "captures": number + 1,
                        "total_captures": len(files),
                        "elapsed_seconds": time.monotonic() - start,
                    }
                ),
                flush=True,
            )
    result = {
        "status": "NATIVE_HEAD_REPLAY_MEASURED",
        "gpu_used": True,
        "head_shape": list(weight.shape),
        "packing": packing,
        "capture_calls": len(files),
        "capture_rows": sum(int(rows) * count for rows, count in shapes.items()),
        "row_shape_histogram": shapes,
        "reference_matches_in_model": True,
        "selection_scope": "all consecutive saved calls; no failure filtering",
        "recall": totals,
        "gpu_event_times": {k: summarize(v) for k, v in timings.items()},
        "synchronized_wall_times": {k: summarize(v) for k, v in wall_timings.items()},
        "shape_times": {
            shape: {k: summarize(v) for k, v in modes.items()}
            for shape, modes in shape_timings.items()
        },
        "source_sha256": sources,
        "total_seconds": time.monotonic() - start,
        "limitations": [
            "Global variants are approximate, not certified.",
            "Event timing includes head launch gaps; wall timing includes synchronization.",
            "Observed recall is specific to this private Pi request and continuation.",
            "The CPU interval certificate has no qualified native fast implementation.",
        ],
    }
    (reports / "head-stage.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "phase": "head_stage_complete",
                "rows": result["capture_rows"],
                "timings": result["gpu_event_times"],
                "recall": totals,
            }
        ),
        flush=True,
    )
    return result
