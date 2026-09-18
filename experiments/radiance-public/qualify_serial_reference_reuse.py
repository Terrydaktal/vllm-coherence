#!/usr/bin/env python3
"""Native admission for shared M1 captures; never claims D7 matrix-case completion."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from qwen_r9700_lab.conformance_boundaries import compare_boundaries
from qwen_r9700_lab.conformance_campaign import build_campaign, verify_runtime_artifacts
from qwen_r9700_lab.conformance_gpu_lease import gpu_lease
from qwen_r9700_lab.conformance_queue import replace_private
from qwen_r9700_lab.conformance_reference_store import SerialReferenceStore
from qwen_r9700_lab.conformance_runtime import worker_environment
from qwen_r9700_lab.conformance_scenarios import (
    aligned_frames,
    native,
    plan_for,
    require,
    serial_reference_for_d7,
    tokens,
)
from qwen_r9700_lab.diagnostic_contract import authenticate, private_json, seal, write_private


def run(prior, root):
    authenticate(prior)
    root.mkdir(mode=0o700)
    campaign = build_campaign({**prior["spec"], "reuse_serial_reference": True})
    write_private(root / "campaign.json", campaign)
    spec = campaign["spec"]
    verify_runtime_artifacts(spec, root)
    selected = [
        (index, case)
        for index, case in enumerate(campaign["cases"])
        if case["family"] == "forced_d7"
        and case["context"] == 63
        and case["seed"] == 17
        and case["axes"]["accepted"] in {0, 7}
    ]
    require(len(selected) == 2, "native reuse admission requires both declared prefix domains")
    selected.sort(key=lambda item: item[1]["axes"]["accepted"])
    started = time.monotonic()

    def phase(name, **detail):
        replace_private(
            root,
            "progress.json",
            seal({"phase": name, "elapsed_seconds": time.monotonic() - started, **detail}),
        )

    rows = []
    phase("waiting_for_gpu_lease")
    with gpu_lease(root / "gpu-lease"):
        for ordinal, (index, case) in enumerate(selected):
            case_root = root / f"case-{index:05d}"
            case_root.mkdir(mode=0o700)
            write_private(case_root / "input.json", {"campaign": campaign, "case": case})
            previous_environment = dict(os.environ)
            try:
                os.environ.update(worker_environment(spec, case_root))
                width = case["axes"]["accepted"]
                requested = plan_for(spec, case, tokens(spec, width + 4, case["seed"] + 1701))
                phase("populate_or_project_serial_baseline", case=case["id"])
                before = time.monotonic()
                shared = serial_reference_for_d7(spec, case, requested, case_root)
                shared_seconds = time.monotonic() - before
                reuse = private_json(case_root / "serial/reuse.json")
                authenticate(reuse)
                require(reuse["hit"] is bool(ordinal), "unexpected native reference hit/miss")
                phase("fresh_native_serial_execution", case=case["id"])
                before = time.monotonic()
                fresh = native(spec, requested, case_root / "fresh-native", speculation=False)
                fresh_seconds = time.monotonic() - before
                phase("compare_native_states_and_boundaries", case=case["id"])
                comparison = case_root / "fresh-comparison"
                comparison.mkdir(mode=0o700)
                states = aligned_frames(shared, fresh, comparison, require_equal=False)
                boundary = compare_boundaries(
                    shared / "boundaries", fresh / "boundaries", comparison / "boundaries"
                )
                require(
                    all(s["equal"] for s in states) and boundary["equal"],
                    "shared baseline differs from fresh native serial execution; evidence retained",
                )
                rows.append(
                    {
                        "case_domain": case["id"],
                        "reference_hit": reuse["hit"],
                        "reference_seconds": shared_seconds,
                        "fresh_native_seconds": fresh_seconds,
                        "matching_states": len(states),
                        "boundary_report": boundary["sha256"],
                        "observations": boundary["observations"],
                        "projection": reuse["projection"],
                    }
                )
                write_private(case_root / "reference-admission.json", seal(rows[-1]))
            finally:
                os.environ.clear()
                os.environ.update(previous_environment)
    phase("retire_disposable_reference_store")
    with SerialReferenceStore(root / "serial-reference-store") as store:
        retirement = store.retire()
    result = seal(
        {
            "schema": "urn:qwen:serial-reference-native-admission:v1",
            "prior_campaign": prior["sha256"],
            "campaign": campaign["sha256"],
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rows": rows,
            "retirement": retirement["sha256"],
            "gpu_used": True,
            "matrix_cases_completed": 0,
            "scope": "Native M1 prefix/full-domain reuse only; D7 candidates were not executed",
            "universal_equivalence": "UNPROVED",
        }
    )
    write_private(root / "result.json", result)
    phase("complete", result=result["sha256"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    require(args.allow_gpu, "native reference qualification requires explicit GPU authorization")
    print(json.dumps(run(private_json(args.campaign), args.output), sort_keys=True))
