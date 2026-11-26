"""Quick paired GPU benchmark of padding work in the installed Global-512 head."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

import head_live_rows
import radiance_verifyhead as target


def benchmark(model):
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    keys = [key for key in index if key.split(".")[-2:] == ["lm_head", "weight"]]
    if len(keys) != 1:
        raise ValueError("model requires one explicit target head")
    key = keys[0]
    with safe_open(model / index[key], framework="pt") as tensors:
        weight = tensors.get_tensor(key).cuda()
    head = SimpleNamespace(tp_size=1, weight=weight)
    state = SimpleNamespace()
    target._dh._quantize_head_now(state, head)
    original = target._apply_head_global
    candidate = head_live_rows.live_rows_variant(target)
    phases = {"control": original, "live_rows": candidate}
    generator = torch.Generator(device="cuda").manual_seed(20261126)
    report = {
        "device": torch.cuda.get_device_name(),
        "model": model.name,
        "shape": list(weight.shape),
        "top_k": target.GLOBAL_TOPK,
        "scope": "isolated GPU head with real weights; includes dispatch between kernels",
        "source_sha256": {
            Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (target, target._dh, head_live_rows)
        },
        "results": {},
    }
    for rows in (1, 4, 8, 16):
        inputs = [
            torch.randn((rows, weight.shape[1]), dtype=weight.dtype, device="cuda", generator=generator)
            for _ in range(8)
        ]
        # The zero input deliberately ties all coarse scores at the cutoff.
        inputs.append(torch.zeros_like(inputs[0]))
        differences = 0
        for hidden in inputs:
            control = original(state, head, hidden)
            experiment = candidate(state, head, hidden)
            differences += int(torch.count_nonzero(
                control.view(torch.int16) != experiment.view(torch.int16)
            ).item())
        samples = {name: [] for name in phases}
        for trial in range(12):
            order = tuple(phases) if trial % 2 == 0 else tuple(reversed(phases))
            for name in order:
                function = phases[name]
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for hidden in inputs[:-1]:
                    function(state, head, hidden)
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) / 8)
        report["results"][str(rows)] = {
            "checked_rows": len(inputs) * rows,
            "different_logits": differences,
            "samples_ms": samples,
            "median_ms": {key: statistics.median(values) for key, values in samples.items()},
        }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with torch.inference_mode():
        result = benchmark(args.model)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if any(row["different_logits"] for row in result["results"].values()):
        raise SystemExit("live-row head changed target logits")
