"""Compare cold and resumed GDN convolution slot initialization with the current prefill adapter."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from benchmark_prefill_bank_norm import exact, measure
from prefill_conv_slot import initialize_conv_slot


def case(history, transposed, repeats):
    slots, dim, padding = 5, 10240, 67
    # Canary every slot and each gap. Distinct negative zero payloads are checked as raw bytes.
    template = torch.randn(
        (slots, dim * history + padding), device="cuda", dtype=torch.bfloat16
    )
    template[:, ::71] = -0.0
    backing = [template.clone(), template.clone()]
    states = [
        block[:, :-padding].view(slots, history, dim).transpose(-1, -2)
        if transposed
        else block[:, :-padding].view(slots, dim, history)
        for block in backing
    ]
    selected = torch.tensor([0], dtype=torch.int32, device="cuda")
    resumed = torch.tensor([False], device="cuda")

    def stock():
        old = states[0].index_select(0, selected.long())
        states[0].index_copy_(
            0,
            selected.long(),
            torch.where(resumed[:, None, None], old, torch.zeros_like(old)),
        )

    def candidate():
        if not initialize_conv_slot(states[1], selected, resumed):
            raise ValueError("indexed convolution optimization was bypassed")

    for slot in (0, 4, 2):
        for is_warm in (True, False):
            selected.fill_(slot)
            resumed.fill_(is_warm)
            for slab in backing:
                slab.copy_(template)
            stock()
            candidate()
            if not exact(backing[0], backing[1]):
                raise ValueError(
                    "selected convolution history, neighboring slot or padding differed"
                )
    results = []
    for is_warm in (True, False):
        resumed.fill_(is_warm)
        results.append(
            {
                "warm": is_warm,
                "history": history,
                "transposed": transposed,
                "tested_slots": [0, 4, 2],
                "canary_bytes_exact": True,
                **measure(stock, candidate, repeats),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=41)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(29233)
    result = {
        "source_sha256": hashlib.sha256(
            Path(__file__).with_name("prefill_conv_slot.py").read_bytes()
        ).hexdigest(),
        "device": torch.cuda.get_device_name(),
        "cases": [],
    }
    for history in (3, 10):
        for transposed in (False, True):
            result["cases"].extend(case(history, transposed, args.repeats))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["cases"]:
        print(
            {
                k: row[k]
                for k in ("history", "transposed", "warm", "control_ms", "candidate_ms")
            }
        )


if __name__ == "__main__":
    main()
