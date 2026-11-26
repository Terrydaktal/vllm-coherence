"""Compare aligned, post-operator target tensors in a bounded cold/decoded synthetic capture."""

import argparse
import hashlib
import json
from difflib import SequenceMatcher
from pathlib import Path

import torch

from qwen_r9700_lab.diagnostic_contract import authenticate, seal


def load_file(root, position):
    capture = json.loads((root.parent / "capture.json").read_bytes())
    first_batch = root.parent / capture["batches"][0]["file"]
    observed = torch.load(first_batch, map_location="cpu", weights_only=True, mmap=True)
    if min(observed["positions"]) != position:
        raise ValueError("the first recorded tensor row must match --position")
    matches = []
    for path in root.glob("group-*.json"):
        meta = json.loads(path.read_bytes())
        authenticate(meta)
        if position in meta["positions"]:
            matches.append(meta)
    if len(matches) != 1:
        raise ValueError(
            f"expected a single group containing the requested position in {root}"
        )
    meta = matches[0]
    path = root / meta["tensor_file"]
    with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != meta["tensor_sha256"]:
            raise ValueError("stage tensor capture changed")
    return meta, torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def compare(root, position):
    def select(raw, event, phase):
        result = {}
        for item in event[phase]:
            key = item["key"]
            name = key.split(".", 2)[2]
            tensor = raw[key]
            if tensor.shape[0] != 1:
                raise ValueError("a one-position --stage-window 1 is required")
            result[name] = (tensor, item["dtype"])
        return result

    def compare_phase(left, right):
        deltas = {}
        for key in left.keys() & right.keys():
            a, ta = left[key]
            b, tb = right[key]
            if a.shape != b.shape or ta != tb or a.dtype != b.dtype:
                deltas[key] = {"type_or_shape_changed": True}
                continue
            if torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
                continue
            held = (
                a
                if a.itemsize == 1
                else a.view(
                    {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.itemsize]
                )
            )
            reference = b.view(held.dtype)
            deltas[key] = {
                "dtype": ta,
                "elements": a.numel(),
                "different": int((held != reference).count_nonzero()),
            }
        return {
            "matched": len(left.keys() & right.keys()),
            "only_prefill": sorted(left.keys() - right.keys()),
            "only_decode": sorted(right.keys() - left.keys()),
            "differences": deltas,
        }

    (ma, a), (mb, b) = [
        load_file(root / arm / "stages", position) for arm in ("prefill", "decode")
    ]
    ea, eb = ma["events"], mb["events"]
    names_a = [e["operation"] for e in ea]
    names_b = [e["operation"] for e in eb]
    before = after = 0
    differences = []
    gaps = []
    for tag, aa, az, ba, bz in SequenceMatcher(
        None, names_a, names_b, autojunk=False
    ).get_opcodes():
        if tag != "equal":
            gaps.append(
                {
                    "prefill_index": aa,
                    "decode_index": ba,
                    "prefill_operations": names_a[aa:az],
                    "decode_operations": names_b[ba:bz],
                }
            )
            continue
        for x, y in zip(ea[aa:az], eb[ba:bz], strict=True):
            before += 1
            outputs = compare_phase(select(a, x, "after"), select(b, y, "after"))
            after += outputs["matched"]
            if (
                outputs["differences"]
                or outputs["only_prefill"]
                or outputs["only_decode"]
            ):
                event = {
                    "prefill_index": x["index"],
                    "decode_index": y["index"],
                    "operation": x["operation"],
                    "outputs": outputs,
                }
                if not differences:
                    event["inputs"] = compare_phase(
                        select(a, x, "before"), select(b, y, "before")
                    )
                differences.append(event)
    return seal(
        {
            "position": position,
            "prefill": ma["sha256"],
            "decode": mb["sha256"],
            "total_prefill": len(ea),
            "total_decode": len(eb),
            "aligned_calls": before,
            "compared_output_tensors": after,
            "unaligned_runs": gaps,
            "different_events": differences,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--position", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.root, args.position)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "first": result["different_events"][:1],
                "unaligned_runs": len(result["unaligned_runs"]),
                "aligned_calls": result["aligned_calls"],
            }
        )
    )
