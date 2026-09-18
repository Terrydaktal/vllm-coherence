"""CPU-only summaries of compiled D7 timing and private forced-token evidence."""

import argparse
import hashlib
from pathlib import Path

from d7_stage_attribution import attribute_trace

from qwen_r9700_lab.conformance_topk import aggregate, compare_rows, require
from qwen_r9700_lab.diagnostic_contract import authenticate, private_json, seal, write_private


def compare(left, right):
    a, b = private_json(left), private_json(right)
    for value in (a, b):
        authenticate(value)
    require(a["continuation"] == b["continuation"], "forced input fixture differs")
    require(len(a["rows"]) == len(b["rows"]) > 0, "incomplete replay")
    pairs = []
    first = None
    for x, y in zip(a["rows"], b["rows"], strict=True):
        require(x["absolute_position"] == y["absolute_position"], "position alignment differs")
        row = compare_rows(x["logits"], y["logits"])
        pairs.append(row)
        if first is None and not row["full_logits_exact"]:
            first = x["position"]
    return seal(
        {
            "reference_sha256": a["sha256"],
            "candidate_sha256": b["sha256"],
            "fixture": a["continuation"],
            "prefill": compare_rows(a["prefill"], b["prefill"]),
            "decode": aggregate(pairs),
            "first_different_position": first,
            "scope": "fixed-token full-vocabulary digests and top-k, not latent-state proof",
        }
    )


def profile(path):
    raw = path.read_bytes()
    trace = private_json(path)["traceEvents"]
    attributed = attribute_trace(trace)
    # Graph launch correlation may not be exposed by every Kineto/ROCm version.
    # Preserve unmapped kernels rather than inventing their stage ownership.
    kernels = {}
    for event in trace:
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        record = kernels.setdefault(event["name"], {"count": 0, "microseconds": 0})
        record["count"] += 1
        record["microseconds"] += event["dur"]
    return seal(
        {
            "trace_sha256": hashlib.sha256(raw).hexdigest(),
            "profile_steps": 8,
            "attribution": attributed,
            "kernels": kernels,
            "scope": "GPU kernel durations; separate from unprofiled end-to-end timing",
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compare", "profile"))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (
        compare(args.reference, args.candidate)
        if args.command == "compare"
        else profile(args.trace)
    )
    write_private(args.output, result)


if __name__ == "__main__":
    main()
