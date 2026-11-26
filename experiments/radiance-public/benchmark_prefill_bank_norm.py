"""Quick matched R9700 operator tests with synthetic activations and public projection weights."""

import argparse
import json
import statistics
from pathlib import Path
from types import MethodType

import torch
from gdn_prefill_direct import _prepared_scan_into
from prefill_bank_norm import hoisted_recurrent, indexed_scan_into


def exact(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def measure(control, candidate, repeats, graph=False):
    ops = [control, candidate]
    for op in ops:
        op()
    torch.cuda.synchronize()
    if graph:
        graphs = []
        for op in ops:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                op()
            graphs.append(g)
        ops = [g.replay for g in graphs]
    batch = 16 if graph else 1
    samples = [[], []]
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for i in range(repeats):
        for j in (i % 2, 1 - i % 2):
            start.record()
            for _ in range(batch):
                ops[j]()
            end.record()
            end.synchronize()
            samples[j].append(start.elapsed_time(end) / batch)
    return {
        "control_ms": statistics.median(samples[0]),
        "candidate_ms": statistics.median(samples[1]),
        "samples_ms": samples,
        "graph_replays_per_sample": batch if graph else 0,
    }


def scan(rows, repeats):
    original = torch.randn(6, 48 * 128 * 128 + 128, device="cuda") * 0.02
    backing = [original.clone(), original.clone()]
    banks = [slab[:, :-128].view(6, 48, 128, 128) for slab in backing]
    outputs = [
        torch.full((rows + 2, 48, 128), 19, device="cuda", dtype=torch.bfloat16)
        for _ in banks
    ]
    tensors = [
        torch.randn(rows, width, dtype=torch.bfloat16, device="cuda") * 0.1
        for width in (10240, 48, 48)
    ]
    params = [
        *tensors,
        torch.randn(48, device="cuda"),
        torch.randn(48, device="cuda", dtype=torch.bfloat16),
    ]
    slot = torch.zeros(1, device="cuda", dtype=torch.int32)
    resumed = torch.zeros(1, device="cuda", dtype=torch.bool)
    into = _prepared_scan_into()

    def control():
        selected = banks[0].index_select(0, slot.long())
        initial = torch.where(
            resumed[:, None, None, None], selected, torch.zeros_like(selected)
        )[0].contiguous()
        _, final = into(initial, *params, output=outputs[0][1:-1])
        banks[0].index_copy_(0, slot.long(), final.unsqueeze(0))

    def candidate():
        indexed_scan_into(banks[1], slot, resumed, *params, outputs[1][1:-1])

    for i, has_previous in ((0, False), (5, True), (2, False), (2, True)):
        slot.fill_(i)
        resumed.fill_(has_previous)
        outputs[1][1:-1].fill_(13)
        control()
        candidate()
        if not exact(backing[0], backing[1]) or not exact(outputs[0], outputs[1]):
            raise ValueError("indexed prefill changed a bank, output, or canary")
    return {
        "stage": "indexed_state",
        "rows": rows,
        "correctness": "exact full banks, padding, outputs and canaries across four states",
        **measure(control, candidate, repeats),
    }


def bias_case(qualification, repeats):
    directory = qualification / "diagnostics/packed-gdn-build-001"
    import sys

    sys.path.insert(0, str(directory))
    from packed_recurrent import StockIndexedAdapter

    adapter = StockIndexedAdapter(torch)
    wide = torch.randn(48, device="cuda").bfloat16().float()
    fused = MethodType(hoisted_recurrent(adapter, [wide]), adapter)
    packed = torch.randn(8, 10240, dtype=torch.bfloat16, device="cuda")
    q, k, v = [t.view(8, -1, 128) for t in packed.split((2048, 2048, 6144), dim=1)]
    gates = [torch.randn(8, 48, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    log = torch.randn(48, device="cuda")
    state = torch.randn(10, 48, 128, 128, device="cuda") * 0.05
    banks = [state.clone(), state.clone()]
    outs = [torch.empty_like(v, memory_format=torch.contiguous_format) for _ in banks]
    cu = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
    indices = torch.tensor([[3, 6, 1, 4, 2, 8, 7, 5]], device="cuda", dtype=torch.int32)
    accepted = torch.ones(1, device="cuda", dtype=torch.int32)

    def call(which, val):
        (adapter if which == 0 else fused)(
            q,
            k,
            v,
            *gates,
            log,
            val,
            banks[which],
            outs[which],
            cu,
            indices,
            accepted,
            1,
            48,
            16,
            128**-0.5,
        )

    for i in range(1, 9):
        accepted.fill_(i)
        val = wide.clone() if i == 7 else wide
        call(0, val)
        call(1, val)
        if not exact(banks[0], banks[1]) or not exact(outs[0], outs[1]):
            raise ValueError("hoisted BF16 bias changed outputs or state")
    return {
        "stage": "packed_decode_bias",
        "rows": 8,
        "correctness": "all eight acceptances, full banks and outputs; uncached-bias fallback",
        **measure(lambda: call(0, wide), lambda: call(1, wide), repeats, graph=True),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for opt in ("qualification", "build", "output"):
        p.add_argument("--" + opt, type=Path, required=True)
    p.add_argument("--bank-rows", type=int, nargs="+", default=(129, 1648, 3296))
    p.add_argument("--repeats", type=int, default=9)
    p.add_argument(
        "--modes",
        nargs="+",
        choices=("bias", "bank"),
        default=("bias", "bank"),
    )
    args = p.parse_args()
    torch.manual_seed(20261126)
    torch.set_num_threads(4)
    result = {
        "device": torch.cuda.get_device_name(),
        "cases": [],
        "build": json.loads((args.build / "prefill-build.json").read_text()),
    }

    def record(case):
        result["cases"].append(case)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in case.items()
                    if k in ("stage", "rows", "layer", "control_ms", "candidate_ms")
                }
            ),
            flush=True,
        )

    if "bias" in args.modes:
        record(bias_case(args.qualification, args.repeats))
    if "bank" in args.modes:
        for rows in args.bank_rows:
            record(scan(rows, args.repeats))


if __name__ == "__main__":
    main()
