"""Compare the stock and candidates-only DFlash2 INT2 heads using the same graph inputs."""

import argparse
import hashlib
import importlib.util
import json
import os
import random
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import drafter_operator_candidates as candidates


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def time_graphs(functions, repeats, trials):
    import torch

    graphs = {}
    # Each graph holds the same number of launches, all operands are kept alive.
    for name, function in functions.items():
        function()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = function()
        graphs[name] = graph, output
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = {name: [] for name in graphs}
    rng = random.Random(1247)
    for _ in range(trials + 1):
        order = list(graphs)
        rng.shuffle(order)
        for name in order:
            graph = graphs[name][0]
            start.record()
            for _ in range(repeats):
                graph.replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / repeats)
    return {name: values[1:] for name, values in samples.items()}


def case_report(functions, inputs, repeats, trials):
    import torch

    differences = {name: 0 for name in functions}
    for args in inputs:
        expect = functions["stock"](*args)
        for name, func in functions.items():
            if name != "stock":
                actual = func(*args)
                if actual.shape != expect.shape or actual.dtype != expect.dtype:
                    raise ValueError(f"{name} changed output shape or dtype")
                differences[name] += (actual.view(torch.int16) != expect.view(torch.int16)).sum().item()
    bound = {name: (lambda f=func: f(*inputs[0])) for name, func in functions.items()}
    samples = time_graphs(bound, repeats, trials)
    return {
        "checked_elements_per_arm": expect.numel() * len(inputs),
        "different_output_elements": differences,
        "samples_ms": samples,
        "median_ms": {name: statistics.median(values) for name, values in samples.items()},
    }


def head_cases(model, outdir, repeats, trials, generator):
    import torch
    import radiance_drafthead as stock
    from safetensors import safe_open

    output = outdir / "radiance_drafthead_candidate.py"
    candidates.write_draft_head(stock.__file__, output)
    spec = importlib.util.spec_from_file_location("radiance_drafthead_candidate", output)
    candidate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = candidate
    spec.loader.exec_module(candidate)
    if stock.RERANK != 80:
        raise ValueError("expected deployed RADIANCE_DRAFT_RERANK=80")
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    key, = [name for name in index if name.split(".")[-2:] == ["lm_head", "weight"]]
    with safe_open(model / index[key], framework="pt") as weights:
        w = weights.get_tensor(key).cuda()
    head = SimpleNamespace(tp_size=1, weight=w)
    state = SimpleNamespace(_radiance_topk_only=True, head_dtype=None)
    stock._quantize_head_now(state, head)
    cases = {}
    for mode, rows in (("dflash2", 7), ("dflash2", 8), ("dflash2", 16), ("argmax", 7), ("coarse_only", 7)):
        state._radiance_topk_only = mode != "argmax"
        active_head = head if mode != "coarse_only" else SimpleNamespace(weight=None)
        cases_key = f"{mode}_M{rows}"
        xs = [torch.randn((rows, w.shape[1]), device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(5)]
        xs.extend([xs[0][:1].expand_as(xs[0]), torch.zeros_like(xs[0])])
        funcs = {
            "stock": lambda x: stock._apply_head_int2(state, active_head, x, None),
            "live_no_dense": lambda x: candidate._apply_head_int2(state, active_head, x, None),
        }
        cases[cases_key] = {"shape": [rows, *w.shape], **case_report(funcs, [(x,) for x in xs], repeats, trials)}
        print(cases_key, cases[cases_key]["median_ms"], flush=True)
    return cases, {"draft_head": digest(stock.__file__), "candidate_head": digest(output)}


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("head",), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--trials", type=int, default=11)
    args = parser.parse_args()
    if min(args.repeats, args.trials) < 1:
        parser.error("positive trial counts required")
    generator = torch.Generator(device="cuda").manual_seed(20261126)
    with torch.inference_mode():
        cases, sources = head_cases(args.model, args.output.parent, args.repeats, args.trials, generator)
    sources.update({"benchmark": digest(__file__), "variants": digest(candidates.__file__)})
    report = {
        "device": torch.cuda.get_device_name(), "model": args.model.name, "kind": args.case,
        "scope": "isolated operator with real weights and synthetic inputs, CUDA graph replays",
        "seed": generator.initial_seed(), "torch": torch.__version__, "hip": torch.version.hip,
        "repeats": args.repeats, "trials": args.trials, "source_sha256": sources,
        "draft_rerank": os.environ.get("RADIANCE_DRAFT_RERANK"), "cases": cases,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if any(any(case["different_output_elements"].values()) for case in cases.values()):
        raise SystemExit("one or more candidates differ from baseline")


if __name__ == "__main__":
    main()
