"""Build standalone operator candidates from the exact qualified sources, without a GPU."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"qualified source changed at {old!r}")
    return text.replace(old, new)


def bank_kernel(source):
    node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "prepared_scan"
    )
    kernel = "\n".join(
        source.splitlines()[node.decorator_list[0].lineno - 1 : node.end_lineno]
    )
    kernel = replace(kernel, "def prepared_scan(", "def indexed_scan(")
    kernel = replace(
        kernel,
        "    states,\n",
        "    slots,\n    resumed,\n    stride_bank: tl.constexpr,\n    num_slots: tl.constexpr,\n",
    )
    kernel = replace(
        kernel,
        "state = tl.load(initial + offsets).to(tl.float32)",
        """slot = tl.load(slots).to(tl.int64)
    valid = (slot >= 0) & (slot < num_slots)
    tl.device_assert(valid, "GDN prefill index outside state bank")
    if not valid:
        return
    start = tl.load(resumed) != 0
    address = initial + slot * stride_bank + offsets
    state = tl.load(address, mask=start, other=0.0).to(tl.float32)""",
    )
    kernel = replace(
        kernel, "tl.store(states + offsets, state)", "tl.store(address, state)"
    )
    return "from vllm.triton_utils import tl, triton\n\n\n" + kernel + "\n"


def build(source, output):
    output.mkdir(exist_ok=True, parents=True)
    scan = source / "prepared_prefill_scan.py"
    indexed = output / "prefill_indexed_kernel.py"
    indexed.write_text(bank_kernel(scan.read_text()))
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (scan, indexed)
    }
    (output / "prefill-build.json").write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)
