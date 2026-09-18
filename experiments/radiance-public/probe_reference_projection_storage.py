#!/usr/bin/env python3
"""Measure verified copy/reflink projection of archived native evidence, CPU only."""

import argparse
import time
from pathlib import Path

from qwen_r9700_lab.conformance_boundaries import compare_boundaries
from qwen_r9700_lab.conformance_native_reference import project_serial_reference
from qwen_r9700_lab.conformance_state import compare_frames
from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    authenticate,
    private_json,
    seal,
    write_private,
)


def probe(source_plan, requested_plan, source, independent, output):
    source_plan, requested_plan = private_json(source_plan), private_json(requested_plan)
    schedule = private_json(independent / "schedule.json")
    authenticate(schedule)
    if schedule["plan"] != requested_plan["sha256"]:
        raise DiagnosticError("independent capture is not from the requested plan")
    output.mkdir(mode=0o700)
    rows = []
    for mode in ("copy", "reflink"):
        destination = output / mode
        started = time.monotonic()
        projection = project_serial_reference(
            source_plan, requested_plan, source, destination, reflink=mode == "reflink"
        )
        elapsed = time.monotonic() - started
        comparisons = [
            compare_frames(independent / entry["name"], destination / entry["name"])
            for entry in schedule["frames"]
        ]
        write_private(output / (mode + "-state-comparison.json"), {"comparisons": comparisons})
        boundary = compare_boundaries(
            independent / "boundaries", destination / "boundaries", output / (mode + "-boundaries")
        )
        if not comparisons or not all(c["equal"] for c in comparisons) or not boundary["equal"]:
            raise DiagnosticError("projected native capture differs from independent capture")
        blobs = list(destination.rglob("*.bin"))
        for blob in blobs:
            original = source / blob.relative_to(destination)
            if blob.stat().st_ino == original.stat().st_ino or blob.stat().st_nlink != 1:
                raise DiagnosticError("projection shares a mutable inode")
        rows.append(
            {
                "method": mode,
                "projection_seconds": elapsed,
                "projection": projection["sha256"],
                "matching_states": len(comparisons),
                "observations": projection["observations"],
                "boundary_comparison": boundary["sha256"],
                "tensor_files": len(blobs),
                "logical_bytes": sum(p.stat().st_size for p in blobs),
            }
        )
    result = seal(
        {
            "schema": "urn:qwen:serial-reference-storage-pilot:v1",
            "source_plan": source_plan["sha256"],
            "requested_plan": requested_plan["sha256"],
            "independent_schedule": schedule["sha256"],
            "rows": rows,
            "gpu_used": False,
            "scope": "Archived native byte comparisons only; no native arithmetic qualification",
        }
    )
    write_private(output / "result.json", result)
    return result


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-plan", "requested-plan", "source", "independent", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(probe(**vars(args)), sort_keys=True))
