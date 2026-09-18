"""Compare private activation traces; publish only hashes, counts and error statistics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    authenticate,
    private_json,
    seal,
    write_private,
)


def key(row):
    return tuple(row[k] for k in ("position", "module", "phase", "path"))


def checked_bytes(root, row):
    name = row["file"]
    if Path(name).name != name:
        raise DiagnosticError("invalid private trace member")
    raw = (root / name).read_bytes()
    if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise DiagnosticError("private trace payload changed")
    return raw


def difference(left, right, dtype):
    a, b = np.frombuffer(left, dtype=np.uint8), np.frombuffer(right, dtype=np.uint8)
    different = np.flatnonzero(a != b)
    result = {"different_bytes": len(different), "first_differing_byte": int(different[0])}
    types = {"torch.float32": "<f4", "torch.float64": "<f8", "torch.float16": "<f2"}
    if dtype == "torch.bfloat16":
        x, y = [
            (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
            for raw in (left, right)
        ]
    elif dtype in types:
        x, y = [np.frombuffer(raw, dtype=types[dtype]) for raw in (left, right)]
    else:
        return result
    result["different_elements"] = int(np.count_nonzero(x != y))
    finite = np.isfinite(x) & np.isfinite(y)
    result["nonfinite_elements"] = int(np.count_nonzero(~finite))
    if np.any(finite):
        delta = x[finite].astype(np.float64) - y[finite].astype(np.float64)
        result["max_abs"] = float(np.max(np.abs(delta)))
        result["rmse"] = float(np.sqrt(np.mean(delta * delta)))
    return result


def compare(left_root, right_root):
    docs = [private_json(p / "manifest.json") for p in (left_root, right_root)]
    for doc in docs:
        authenticate(doc)
        if (
            doc["schema"] != "urn:qwen:private-decoder-row-trace:v1"
            or not doc["positions"]
            or not doc["records"]
        ):
            raise DiagnosticError("unsupported private row trace")
    if docs[0]["positions"] != docs[1]["positions"] or docs[0]["inventory"] != docs[1]["inventory"]:
        raise DiagnosticError("private traces cover different domains")
    tables = [{key(row): row for row in doc["records"]} for doc in docs]
    if any(len(t) != len(d["records"]) for t, d in zip(tables, docs, strict=True)):
        raise DiagnosticError("private trace repeats a boundary")
    if tables[0].keys() != tables[1].keys():
        raise DiagnosticError("private traces cover different tensor boundaries")
    by_position, differences = {}, []
    for row in docs[0]["records"]:
        peer = tables[1][key(row)]
        if (row["dtype"], row["shape"], row["bytes"]) != (
            peer["dtype"],
            peer["shape"],
            peer["bytes"],
        ):
            raise DiagnosticError("private trace tensor layout changed")
        a, b = checked_bytes(left_root, row), checked_bytes(right_root, peer)
        position = row["position"]
        counts = by_position.setdefault(position, {"compared": 0, "different": 0, "first": None})
        counts["compared"] += 1
        if a != b:
            item = {k: row[k] for k in ("position", "module", "phase", "path", "dtype", "shape")}
            item.update(difference(a, b, row["dtype"]))
            differences.append(item)
            counts["different"] += 1
            if counts["first"] is None:
                counts["first"] = item
    return {
        "trace_sha256": [d["sha256"] for d in docs],
        "compared": len(tables[0]),
        "different": len(differences),
        "positions": [{"position": p, **by_position[p]} for p in sorted(by_position)],
        "first_differences": differences[:40],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--baseline-index", type=int, required=True)
    args = parser.parse_args()
    roots = [args.corpus / args.revision / arm / "000" for arm in ("m1", "m8")]
    reproduced = {}
    for arm, root in zip(("m1", "m8"), roots, strict=True):
        actual = private_json(root / "rows.json")
        expected = private_json(args.baseline / arm / f"{args.baseline_index:03d}/rows.json")
        for doc in (actual, expected):
            authenticate(doc)
        if not 0 < len(actual["rows"]) <= len(expected["rows"]):
            raise DiagnosticError("invalid minimized replay length")
        reproduced[arm] = {
            "prefill": actual["prefill"]["logits_sha256"] == expected["prefill"]["logits_sha256"],
            "decode": sum(
                a["absolute_position"] == b["absolute_position"]
                and a["logits"]["logits_sha256"] == b["logits"]["logits_sha256"]
                for a, b in zip(actual["rows"], expected["rows"], strict=False)
            ),
            "positions": len(actual["rows"]),
        }
    result = seal(
        {
            "status": "MEASURED",
            "reproduced": reproduced,
            **compare(*(p / "trace" for p in roots)),
            "scope": "Private activation byte comparison; no text or token IDs in report.",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    write_private(args.output, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "reproduced", "compared", "different", "positions", "sha256")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
