"""Test spatial GDN scan tiles without changing chronological arithmetic."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def run(args):
    import stock_gdn_scan_kernel as module
    import torch
    from optimized_prefill_scan import PrefillScan
    from stock_gdn_scan import StockScan

    torch.manual_seed(9181000)
    torch.set_num_threads(4)
    scan = StockScan(torch, "cuda:0")
    original = module.stock_gdn_scan_kernel
    state = torch.randn(48, 128, 128, device="cuda") * 0.02
    qkv = torch.randn(2048, 10240, device="cuda", dtype=torch.bfloat16) * 0.5
    a = (torch.randn(2048, 48, device="cuda") - 3).bfloat16()
    b = torch.randn_like(a)
    a_log = torch.randn(48, device="cuda") * 0.1
    bias = torch.randn(48, device="cuda", dtype=torch.bfloat16) * 0.1
    a[321].fill_(-64)
    a[999].fill_(64)
    b[127].fill_(-32)
    results = []
    report = {"status": "RUNNING", "checks": results}

    class Tile:
        def __init__(self, size):
            self.size = size

        def __getitem__(self, grid):
            def launch(*pargs, **kwargs):
                kwargs["BV"] = self.size
                return original[(128 // self.size, 48)](*pargs, **kwargs)

            return launch

    for count in (1, 8, 64, 320, 1000, 1648, 2048):
        module.stock_gdn_scan_kernel = original
        ref = scan.run(state, qkv[:count], a[:count], b[:count], a_log, bias)
        for tile in (8, 16, 32):
            module.stock_gdn_scan_kernel = Tile(tile)
            actual = scan.run(state, qkv[:count], a[:count], b[:count], a_log, bias)
            equal = torch.equal(
                actual.outputs.view(torch.uint8), ref.outputs.view(torch.uint8)
            ) and torch.equal(
                actual.final_state.view(torch.uint8), ref.final_state.view(torch.uint8)
            )
            samples = []
            for _ in range(9):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                scan.run(state, qkv[:count], a[:count], b[:count], a_log, bias)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            results.append(
                {
                    "tokens": count,
                    "tile": tile,
                    "exact": equal,
                    "median_ms": statistics.median(samples),
                }
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        module.stock_gdn_scan_kernel = original
        candidate = PrefillScan(torch, "cuda:0")
        actual = candidate.run(state, qkv[:count], a[:count], b[:count], a_log, bias)
        results.append(
            {
                "tokens": count,
                "tile": "selected",
                "exact": torch.equal(
                    actual.outputs.view(torch.uint8), ref.outputs.view(torch.uint8)
                )
                and torch.equal(
                    actual.final_state.view(torch.uint8), ref.final_state.view(torch.uint8)
                ),
            }
        )
    module.stock_gdn_scan_kernel = original
    mutated = actual.final_state.clone()
    mutated.view(torch.uint8).flatten()[0] ^= 1
    report["negative_control_detected"] = not torch.equal(
        mutated.view(torch.uint8), actual.final_state.view(torch.uint8)
    )
    report["sources"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("probe_prefill_tiles.py", "optimized_prefill_scan.py")
    }
    report["status"] = "SAMPLE_CHECKED" if all(c["exact"] for c in results) else "FAILED"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "SAMPLE_CHECKED":
        raise RuntimeError("spatial scan differs from reference")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
