"""CPU-only comparison of authenticated private eager/compiled tensor captures."""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

from qwen_r9700_lab.conformance_execution_modes import admit_pair
from qwen_r9700_lab.conformance_mode_boundaries import admit_bridge, compare_group, summarize
from qwen_r9700_lab.conformance_precision_intervention import admit_precision_intervention
from qwen_r9700_lab.conformance_silu_intervention import admit_silu_intervention
from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    authenticate,
    private_json,
    write_private,
)


def load(root, item):
    import torch

    name = item["file"]
    if Path(name).name != name:
        raise DiagnosticError("capture filename escapes its evidence directory")
    path = root / name
    metadata = private_json(path.with_suffix(".json"))
    authenticate(metadata)
    if metadata["sha256"] != item["sha256"] or metadata["positions"] != item["positions"]:
        raise DiagnosticError("capture metadata does not match its manifest")
    with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != metadata["tensor_sha256"]:
            raise DiagnosticError("captured tensor file changed")
    tensors = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    # NumPy cannot represent BF16. Use its exact two-byte storage, retaining
    # shape and element size; no arithmetic comparison or tolerance is applied.
    values = {
        k: (v.view(torch.int16) if v.dtype == torch.bfloat16 else v).numpy()
        for k, v in tensors.items()
    }
    return metadata, values


def run(left, right, phase, runs, bridges, *, admit=admit_pair):
    sides = []
    for root in runs:
        paths = {
            "measurement": root / "measurement.json",
            "config": root / "fixed-bf16/requested-config.json",
            "runtime": root / "fixed-bf16/actual-runtime.json",
            "pass": root / "fixed-bf16/pass-00.json",
        }
        sides.append({key: private_json(path) for key, path in paths.items()})
    admission = admit(*sides)
    if admission["captures"] != [True, True]:
        raise DiagnosticError("both runs must contain boundary captures")
    filename = "prefill-manifest.json" if phase == "prefill" else "manifest.json"
    a, b = [private_json(root / filename) for root in (left, right)]
    for root, record, side, bridge in zip((left, right), (a, b), sides, bridges, strict=True):
        authenticate(record)
        decode = private_json(root / "manifest.json")
        admit_bridge(bridge, side["pass"], decode, record if phase == "prefill" else None)
    groups = []
    if len(a["batches"]) != len(b["batches"]):
        raise DiagnosticError("execution modes have different capture group counts")
    for ia, ib in zip(a["batches"], b["batches"], strict=True):
        ma, ta = load(left, ia)
        mb, tb = load(right, ib)
        groups.append(compare_group(ma, mb, ta, tb))
    expected = [p for batch in a["batches"] for p in batch["positions"]]
    if phase == "decode" and len(expected) != 320:
        raise DiagnosticError("decode comparison requires all 320 positions")
    if phase == "prefill" and (expected != a["positions"] or expected != b["positions"]):
        raise DiagnosticError("sampled prefill positions differ")
    return summarize(
        groups,
        expected_positions=expected,
        sources={
            "left": a["sha256"],
            "right": b["sha256"],
            "phase": phase,
            "mode_admission": admission["sha256"],
            "bridges": [value["sha256"] for value in bridges],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("left", "right", "output", "left-run", "right-run", "left-bridge", "right-bridge"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    intervention = parser.add_mutually_exclusive_group()
    intervention.add_argument("--silu-intervention", action="store_true")
    intervention.add_argument("--precision-casts-intervention", action="store_true")
    parser.add_argument("--base-driver", type=Path)
    parser.add_argument("--experiment-driver", type=Path)
    args = parser.parse_args()
    admission = admit_pair
    if args.silu_intervention:
        if args.base_driver is None or args.experiment_driver is None:
            raise DiagnosticError("SiLU intervention requires both reviewed driver sources")
        admission = partial(
            admit_silu_intervention,
            base_driver=hashlib.sha256(args.base_driver.read_bytes()).hexdigest(),
            experiment_driver=hashlib.sha256(args.experiment_driver.read_bytes()).hexdigest(),
        )
    elif args.base_driver is not None or args.experiment_driver is not None:
        raise DiagnosticError(
            "driver normalization is only allowed for the declared SiLU intervention"
        )
    elif args.precision_casts_intervention:
        admission = admit_precision_intervention
    result = run(
        args.left,
        args.right,
        args.phase,
        [args.left_run, args.right_run],
        [private_json(args.left_bridge), private_json(args.right_bridge)],
        admit=admission,
    )
    write_private(args.output, result)
    print(
        json.dumps(
            {
                "positions": result["positions"],
                "first": result["first_observed_different_boundary"],
                "sha256": result["sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
