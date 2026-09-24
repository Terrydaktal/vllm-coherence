"""Opt-in, head-only comparison inside an isolated copy of the serving release.

The target continues using its original configured result. The observer compares
256/512/full on identical hidden rows; only numeric aggregates leave the worker.
No production loader imports this module. Timing excludes comparison/reporting.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path


def distribution(logits, *, top_k=40, top_p=0.95, temperature=1.0):
    """Diagnostic distribution; backend-specific boundary tie order is unproved."""
    import torch

    if not 0 < top_k <= logits.shape[-1] or not 0 < top_p <= 1 or temperature <= 0:
        raise ValueError("invalid diagnostic sampling parameters")
    values, ids = (logits.float() / temperature).topk(top_k, dim=-1)
    probabilities = values.softmax(-1)
    probabilities *= (probabilities.cumsum(-1) - probabilities) < top_p
    probabilities /= probabilities.sum(-1, keepdim=True)
    return torch.zeros_like(logits, dtype=torch.float32).scatter(-1, ids, probabilities)


def compare(reference, candidate):
    import torch

    if (
        reference.shape != candidate.shape
        or reference.ndim != 2
        or reference.shape[1] < 40
    ):
        raise ValueError(
            "comparison needs matching vocabulary rows with at least 40 tokens"
        )
    if (
        not torch.isfinite(reference).all()
        or torch.isnan(candidate).any()
        or torch.isposinf(candidate).any()
    ):
        raise ValueError("invalid logits")
    finite = torch.isfinite(candidate)
    if not (finite.sum(-1) >= 40).all():
        raise ValueError("candidate has insufficient sampling support")
    winner = reference.argmax(-1, keepdim=True)
    difference = torch.where(finite, (candidate.float() - reference.float()).abs(), 0)
    stats = {
        "rows": len(reference),
        "argmax_retained": int(finite.gather(1, winner).sum()),
        "argmax_equal": int((candidate.argmax(-1) == winner[:, 0]).sum()),
        "retained_values": int(finite.sum()),
        "retained_value_mismatches": int((difference != 0).sum()),
        "max_retained_logit_difference": float(difference.max()),
    }
    for k in (10, 20, 40):
        ref_values, ref_ids = reference.topk(k, dim=-1)
        cand_ids = candidate.topk(k, dim=-1).indices
        stats[f"top{k}_complete_including_ties"] = int(
            (finite | (reference < ref_values[:, -1:])).all(-1).sum()
        )
        stats[f"top{k}_same_set"] = int(
            (ref_ids.sort(-1).values == cand_ids.sort(-1).values).all(-1).sum()
        )
        stats[f"top{k}_same_order"] = int((ref_ids == cand_ids).all(-1).sum())
    ref_probs, cand_probs = distribution(reference), distribution(candidate)
    tv = (ref_probs - cand_probs).abs().sum(-1) / 2
    stats["diagnostic_tv_sum"] = float(tv.sum())
    stats["diagnostic_tv_max"] = float(tv.max())
    stats["rows_with_diagnostic_probability_difference"] = int((tv > 0).sum())
    stats["excluded_reference_probability_mass_sum"] = float(
        ref_probs.masked_fill(finite, 0).sum()
    )
    return stats


def accumulate(total, row):
    for key, value in row.items():
        total[key] = (
            max(total.get(key, 0), value)
            if key.startswith("max_") or key.endswith("_max")
            else total.get(key, 0) + value
        )


def timing_summary(values):
    if not values:
        return None
    ordered = sorted(values)
    return {
        "samples": len(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "samples_ms": values,
    }


def observe(module, original, root):
    report = {
        "schema": "urn:coherence:head-candidate-depth:v1",
        "status": "running",
        "modes": {name: {} for name in ("global256", "global512", "full")},
        "by_workload": {},
        "calls": 0,
        "shape_counts": {},
        "negative_control_detected": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "verify_head_sha256": hashlib.sha256(
            Path(module.__file__).read_bytes()
        ).hexdigest(),
        "privacy": "Numeric aggregates only; no hidden tensors, text, token IDs or logits saved.",
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 40},
        "limitations": [
            "Head-only agreement against the installed corrected full head, not model accuracy.",
            "Top-k set/order use torch.topk tie ordering; retention includes every boundary tie.",
            "Probability comparison is diagnostic, not a certified replay of the native sampler.",
            "Generation has observer overhead and is not a throughput benchmark.",
            "Head timings include native dispatch gaps, excluding comparison and reporting work.",
        ],
    }
    samples = {name: [] for name in report["modes"]}
    randomizer = random.Random(24512)
    seen_control = None
    control = {}
    previous_check = 0.0

    def write():
        report["timings_m8"] = {
            name: timing_summary(values) for name, values in samples.items()
        }
        report["updated_at"] = time.time()
        temporary = root / "result.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(root / "result.json")

    @functools.wraps(original)
    def wrapped(state, head, hidden, bias=None):
        nonlocal control, seen_control, previous_check
        import torch

        result = original(state, head, hidden, bias)
        if torch.cuda.is_current_stream_capturing():
            return result
        now = time.monotonic()
        if now - previous_check > 0.2:
            previous_check = now
            marker = root / "control.json"
            if marker.exists():
                timestamp = marker.stat().st_mtime_ns
                if timestamp != seen_control:
                    control = json.loads(marker.read_text())
                    seen_control = timestamp
        if not control.get("armed") or report["calls"] >= control.get(
            "max_calls", 1536
        ):
            return result
        if (
            module.GLOBAL_TOPK not in (256, 512)
            or bias is not None
            or hidden.shape[0] not in (1, 8)
        ):
            raise ValueError(
                "benchmark requires a production Global-256/512 M1/M8 target path"
            )
        observed_mode = f"global{module.GLOBAL_TOPK}"
        if report.setdefault("observed_target_mode", observed_mode) != observed_mode:
            raise ValueError("serving target-head depth changed during the comparison")
        label = control["label"]
        if label not in ("coding", "reasoning", "prose"):
            raise ValueError("unsupported public workload label")

        def global_call(depth):
            previous = module.GLOBAL_TOPK
            try:
                module.GLOBAL_TOPK = depth
                return module._apply_head_global(state, head, hidden, None)
            finally:
                module.GLOBAL_TOPK = previous

        functions = {
            "global256": lambda: global_call(256),
            "global512": lambda: global_call(512),
            "full": lambda: state._radiance_exact_head(head, hidden, None),
        }
        reference = functions["full"]()
        if not report["negative_control_detected"]:
            corrupted = reference.clone()
            corrupted.scatter_(1, reference.argmax(-1, keepdim=True), -float("inf"))
            fault = compare(reference, corrupted)
            if fault["argmax_retained"] != 0 or fault["diagnostic_tv_sum"] <= 0:
                raise ValueError(
                    "comparison failed to detect the removed winning token"
                )
            report["negative_control_detected"] = True
            report["head_shape"] = list(head.weight.shape)
            report["full_head_callable"] = getattr(
                state._radiance_exact_head,
                "__qualname__",
                str(type(state._radiance_exact_head)),
            )
            report["draft_head_sha256"] = hashlib.sha256(
                Path(module._dh.__file__).read_bytes()
            ).hexdigest()
            report["hardware"] = torch.cuda.get_device_name()
            report["torch_version"] = torch.__version__
        workload = report["by_workload"].setdefault(
            label, {name: {} for name in functions}
        )
        for name, function in functions.items():
            candidate = reference if name == "full" else function()
            if name == observed_mode and not torch.equal(candidate, result):
                raise ValueError(
                    "probe does not reproduce the actual returned target logits"
                )
            stats = compare(reference, candidate)
            accumulate(report["modes"][name], stats)
            accumulate(workload[name], stats)
        report["calls"] += 1
        shape = str(hidden.shape[0])
        report["shape_counts"][shape] = report["shape_counts"].get(shape, 0) + 1
        if hidden.shape[0] == 8 and report["calls"] % 32 == 0:
            # Warm every method, then alternate them on this identical hidden input.
            for function in functions.values():
                for _ in range(3):
                    function()
            torch.cuda.synchronize()
            for _ in range(5):
                order = list(functions)
                randomizer.shuffle(order)
                for name in order:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    timed = functions[name]()
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end))
                    del timed
        write()
        return result

    return wrapped


def install():
    """Import hook used only by the explicitly isolated benchmark container."""
    root = Path(os.environ["QWEN_HEAD_DEPTH_BENCH_ROOT"])
    if (
        not str(root).startswith("/dev/shm/qwen-head-depth-")
        or root.stat().st_mode & 0o077
    ):
        raise ValueError("benchmark directory must be private tmpfs")

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "radiance_verifyhead":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            original = spec.loader

            class Loader(importlib.abc.Loader):
                def create_module(self, specification):
                    return original.create_module(specification)

                def exec_module(self, module):
                    original.exec_module(module)
                    module._apply_head_gated = observe(
                        module, module._apply_head_gated, root
                    )

            spec.loader = Loader()
            return spec

    sys.meta_path.insert(0, Finder())


def main():
    """Run natural coding and reasoning requests against the isolated server."""
    import argparse
    import urllib.request

    from benchmark_pi_coding_json_compaction import (
        CODING_PROMPT,
        MODEL,
        THINKING_PROMPT,
        _read_sse,
        _render_user_turn,
        _turn_suffix,
    )
    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--abi", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    raw = args.fixture.read_bytes()
    sequence = json.loads(raw)["prefix"]
    if len(sequence) != 60000 or any(type(token) is not int for token in sequence):
        raise ValueError("expected the retained private 60K Pi prefix")
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    identity = {
        "id": hashlib.sha256(str(args.root).encode()).hexdigest(),
        "generation": hashlib.sha256(
            (str(args.root) + ":initial").encode()
        ).hexdigest(),
        "title": "isolated head candidate-depth benchmark",
        "cwd": "/qualification/head-candidate-depth",
        "session_file": "",
    }
    metadata = {
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "prefix_tokens": 60000,
        "stages": [],
        "snapshot_abi": args.abi,
        "privacy": "Private fixture consumed programmatically; no text or token IDs saved.",
    }
    for number, (label, prompt, thinking) in enumerate(
        [("coding", CODING_PROMPT, False), ("reasoning", THINKING_PROMPT, True)]
    ):
        rendered = _render_user_turn(
            opener, args.base_url, prompt, thinking=thinking, timeout=60
        )
        sequence.extend(_turn_suffix(tokenizer, sequence, rendered, first=number == 0))
        control = {"armed": True, "label": label, "max_calls": (number + 1) * 768}
        (args.root / "control.tmp").write_text(json.dumps(control))
        (args.root / "control.tmp").replace(args.root / "control.json")
        body = {
            "model": MODEL,
            "prompt": sequence,
            "max_tokens": 16384,
            "ignore_eos": False,
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "seed": 24512,
            "stream": True,
            "stream_options": {"include_usage": True},
            "return_token_ids": True,
            "add_special_tokens": False,
            "stop": ["<|im_end|>"],
            "cache_salt": f"qwen-chat-cache-v1:{identity['id']}:{identity['generation']}",
            "kv_transfer_params": {
                "qwen_chat": identity,
                "qwen_snapshot_abi": args.abi,
            },
        }
        request = urllib.request.Request(
            args.base_url + "/v1/completions",
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        start = time.monotonic()
        print(
            json.dumps(
                {"phase": "request", "label": label, "input_tokens": len(sequence)}
            ),
            flush=True,
        )
        with opener.open(request, timeout=1800) as response:
            completion = _read_sse(
                response, sequence, tokenizer=tokenizer, started=start
            )
        metadata["stages"].append(
            {
                "label": label,
                "input_tokens": len(sequence),
                "output_tokens": len(completion["token_ids"]),
                "finish_reason": completion["finish_reason"],
                "elapsed_seconds_including_observation": time.monotonic() - start,
            }
        )
        sequence.extend(completion["token_ids"])
        (args.root / "requests.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata["stages"][-1]), flush=True)
    (args.root / "control.json").write_text(json.dumps({"armed": False}))
    result = json.loads((args.root / "result.json").read_text())
    if result["calls"] < 320 or not result["negative_control_detected"]:
        raise ValueError("insufficient checked rows or missing negative control")
    result["status"] = "MEASURED"
    result["requests"] = metadata
    (args.root / "completed.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "phase": "complete",
                "calls": result["calls"],
                "rows": result["modes"]["full"]["rows"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
