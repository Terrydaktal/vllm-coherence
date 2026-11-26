"""Paired GPU operator timings for direct GDN prefill scan output."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from gdn_prefill_direct import _prepared_scan_into
from prepared_prefill_scan import run


def exact(x, y):
    return torch.equal(x.view(torch.uint8), y.view(torch.uint8))


def timing(control, candidate, repeats):
    functions = (control, candidate)
    for fn in functions:
        fn()
    torch.cuda.synchronize()
    samples = [[], []]
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for index in range(repeats):
        for arm in (index % 2, 1 - index % 2):
            start.record()
            functions[arm]()
            end.record()
            end.synchronize()
            samples[arm].append(start.elapsed_time(end))
    return {
        "control_ms": statistics.median(samples[0]),
        "candidate_ms": statistics.median(samples[1]),
        "control_samples": samples[0],
        "candidate_samples": samples[1],
    }


def scan_case(rows, repeats):
    kwargs = {"device": "cuda", "dtype": torch.bfloat16}
    initial = torch.randn((48, 128, 128), device="cuda") * 0.01
    params = (
        initial,
        torch.randn(rows, 10240, **kwargs) * 0.1,
        torch.randn(rows, 48, **kwargs),
        torch.randn(rows, 48, **kwargs),
        torch.rand(48, device="cuda"),
        torch.randn(48, **kwargs),
    )
    guarded = torch.full((rows + 2, 48, 128), 42, **kwargs)
    output = guarded[1:-1]
    run_into = _prepared_scan_into()

    def control():
        values, final = run(*params)
        output.copy_(values)
        return final

    def candidate():
        _, final = run_into(*params, output=output)
        return final

    reference_state = control()
    reference_output = output.clone()
    candidate_state = candidate()
    assert exact(reference_state, candidate_state), "GDN final state changed"
    assert exact(reference_output, output), "GDN scan output changed"
    assert bool((guarded[[0, -1]] == 42).all()), "GDN wrote outside destination"
    return timing(control, candidate, repeats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=(1648, 3296))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(not 128 <= rows <= 4096 for rows in args.rows) or args.repeats < 1:
        parser.error("require 128–4096 rows and positive repeats")

    torch.manual_seed(20261126)
    results = {
        "device": torch.cuda.get_device_name(),
        "cases": [],
    }
    for rows in args.rows:
        case = {"mode": "scan", "rows": rows, "correctness": "byte_exact", **scan_case(rows, args.repeats)}
        results["cases"].append(case)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(case), flush=True)


if __name__ == "__main__":
    main()
