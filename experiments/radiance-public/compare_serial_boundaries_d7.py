"""Compare saved compiled M1/M8 target activations by logical position; CPU only."""

import argparse
import json
from pathlib import Path

from compare_execution_modes_d7 import load as load_run
from compare_mode_boundaries_d7 import load

from qwen_r9700_lab.conformance_mode_boundaries import admit_bridge, compare_group, summarize
from qwen_r9700_lab.conformance_serial_boundaries import admit_serial_pair, pack_serial_groups
from qwen_r9700_lab.conformance_topk import require
from qwen_r9700_lab.diagnostic_contract import authenticate, private_json, write_private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "serial",
        "batched",
        "serial-run",
        "batched-run",
        "serial-bridge",
        "batched-bridge",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sides = [load_run(r) for r in (args.serial_run, args.batched_run)]
    admission = admit_serial_pair(*sides)
    roots = [args.serial, args.batched]
    manifests = [private_json(r / "manifest.json") for r in roots]
    bridges = [private_json(r) for r in (args.serial_bridge, args.batched_bridge)]
    for side, manifest, bridge in zip(sides, manifests, bridges, strict=True):
        authenticate(manifest)
        admit_bridge(bridge, side["pass"], manifest)
    sm, bm = manifests
    require(
        len(sm["batches"]) == 320 and len(bm["batches"]) == 40,
        "not a complete 320-position capture",
    )
    groups = []
    derived = []
    for index, batched in enumerate(bm["batches"]):
        serial = [load(args.serial, item) for item in sm["batches"][index * 8 : index * 8 + 8]]
        packed, values = pack_serial_groups(serial)
        actual, tensors = load(args.batched, batched)
        groups.append(compare_group(packed, actual, values, tensors))
        derived.append({k: packed[k] for k in ("sha256", "source_captures", "skipped_layouts")})
    expected = [p for batch in bm["batches"] for p in batch["positions"]]
    require(len(expected) == 320, "logical decode positions incomplete")
    report = summarize(
        groups,
        expected_positions=expected,
        sources={
            "phase": "decode",
            "admission": admission,
            "serial_manifest": sm["sha256"],
            "batched_manifest": bm["sha256"],
            "derived_views": derived,
            "bridges": [b["sha256"] for b in bridges],
        },
    )
    write_private(args.output, report)
    print(
        json.dumps(
            {
                "positions": report["positions"],
                "first": report["first_observed_different_boundary"],
                "boundaries": len(report["boundaries"]),
                "sha256": report["sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
