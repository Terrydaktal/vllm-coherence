"""Capture a separately identified CPU replay for reference optimization checks.

This does not change a campaign or claim qualification from capture completion.
The resulting frames must still be compared with the preserved serial reference.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

from mxfp4_lookup_reference import LookupQuantizedQwenReference, MXFP4Lookup
from parallel_ordered_reference import ParallelOrderedLinear

from qwen_r9700_lab import conformance_reference as reference
from qwen_r9700_lab.conformance_artifacts import reference_runtime_identity
from qwen_r9700_lab.conformance_boundaries import (
    BoundaryRecorder,
    compare_boundaries,
    detailed_stages,
)
from qwen_r9700_lab.conformance_model import Checkpoint, QuantizedQwenReference, state_names
from qwen_r9700_lab.conformance_replay import (
    OUTPUT_COMPONENTS,
    CampaignWriter,
    observation_domain,
    reference_code_identity,
    scheduled_inputs,
    validate_plan,
    write_model_frame,
)
from qwen_r9700_lab.conformance_state import compare_frames, read_frame
from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    authenticate,
    private_json,
    seal,
    write_private,
)


def compare(reference_root, candidate_root, output):
    """Compare saved bytes while admitting only the recorded execution change."""
    roots = (reference_root, candidate_root)
    plans = [private_json(p / "plan.json") for p in roots]
    schedules = [private_json(p / "schedule.json") for p in roots]
    execution = private_json(candidate_root / "parallel-execution.json")
    capture_report = private_json(candidate_root / "parallel-capture.json")
    for document in (*plans, *schedules, execution, capture_report):
        authenticate(document)
    if (
        {k: v for k, v in plans[0].items() if k not in {"execution", "sha256"}}
        != {k: v for k, v in plans[1].items() if k not in {"execution", "sha256"}}
        or plans[1]["execution"] != execution["sha256"]
        or execution["parent_plan"] != plans[0]["sha256"]
        or execution["parent_execution"] != plans[0]["execution"]
        or capture_report["execution"] != execution["sha256"]
        or capture_report["parent_plan"] != plans[0]["sha256"]
        or capture_report["schedule"] != schedules[1]["sha256"]
        or capture_report["status"] != "CAPTURED"
        or capture_report["fallbacks"] != 0
    ):
        raise DiagnosticError("different reference inputs or unbound parallel execution")
    if "unpacking" in execution:
        authenticate(execution["unpacking"])
        unpacking = capture_report.get("unpacking", {})
        if (
            unpacking.get("execution") != execution["unpacking"]["sha256"]
            or type(unpacking.get("calls")) is not int
            or unpacking["calls"] <= 0
            or type(unpacking.get("fallbacks")) is not int
            or not 0 <= unpacking["fallbacks"] <= unpacking["calls"]
        ):
            raise DiagnosticError("unbound or unexercised lookup decoder")
    expected = list(scheduled_inputs(plans[0]))
    for root, plan, schedule in zip(roots, plans, schedules, strict=True):
        if (
            schedule.get("schema") != "urn:qwen:conformance-schedule:v1"
            or schedule["plan"] != plan["sha256"]
            or schedule["contract"] != plan["contract"]
            or len(schedule["frames"]) != len(expected)
        ):
            raise DiagnosticError("parallel comparison schedule is incomplete")
        for want, entry in zip(expected, schedule["frames"], strict=True):
            if {k: v for k, v in entry.items() if k != "sha256"} != want:
                raise DiagnosticError("parallel comparison changed the consumed/pending schedule")
            frame = read_frame(root / entry["name"])
            if (
                frame["sha256"] != entry["sha256"]
                or frame["execution"] != plan["execution"]
                or frame["adapter"] != plan["adapter"]
                or frame["coverage"] != schedule["coverage"]
                or any(frame[k] != v for k, v in want.items() if k != "name")
            ):
                raise DiagnosticError("parallel comparison frame identity or schedule changed")
    output.mkdir(mode=0o700)
    frames = []
    for entry in expected:
        report = compare_frames(reference_root / entry["name"], candidate_root / entry["name"])
        write_private(output / (entry["name"] + ".json"), report)
        frames.append(report)
    boundaries = {
        name: compare_boundaries(reference_root / name, candidate_root / name, output / name)
        for name in ("boundaries", "semantic")
    }
    equal = all(r["equal"] for r in [*frames, *boundaries.values()])
    report = seal(
        {
            "schema": "urn:qwen:parallel-reference-comparison:v1",
            "status": "TESTED" if equal else "FAILED",
            "equal": equal,
            "reference_plan": plans[0]["sha256"],
            "candidate_execution": execution["sha256"],
            "capture": capture_report["sha256"],
            "state_frames": len(frames),
            "state_comparisons": [r["sha256"] for r in frames],
            "boundaries": boundaries,
            "scope": "Byte comparison of every saved state/logit and recorded layer boundary.",
            "formal_equivalence": "UNPROVED",
        }
    )
    write_private(output / "report.json", report)
    return report


def capture(plan, root, *, workers=4, unpack_lookup=False):
    validate_plan(plan)
    if "reference_linear" not in plan:
        raise DiagnosticError("qualification requires the pinned serial C implementation")
    runtime = reference_runtime_identity()
    if "reference_runtime" in plan and plan["reference_runtime"] != runtime["sha256"]:
        raise DiagnosticError("qualification CPU runtime differs from the preserved reference")
    started = time.monotonic()
    unpacker = MXFP4Lookup(reference) if unpack_lookup else None
    with ParallelOrderedLinear(reference, plan["reference_linear"], workers=workers) as operation:
        execution = seal(
            {
                "schema": "urn:qwen:parallel-reference-execution:v1",
                "parent_plan": plan["sha256"],
                "parent_execution": plan["execution"],
                "linear": operation.execution_binding,
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "core_sources": reference_code_identity(),
                "runtime": runtime,
                "gpu_used": False,
                "qualification": "CAPTURE_PENDING_COMPARISON",
                **({"unpacking": unpacker.execution_binding} if unpacker is not None else {}),
            }
        )
        candidate_plan = seal(
            {**{k: v for k, v in plan.items() if k != "sha256"}, "execution": execution["sha256"]}
        )
        checkpoint = Checkpoint(Path(plan["checkpoint"]), plan["checkpoint_files"])
        model = None
        try:
            model_type = (
                LookupQuantizedQwenReference if unpacker is not None else QuantizedQwenReference
            )
            model = model_type(
                checkpoint,
                kv_scales=plan["kv_scales"],
                contract=plan["contract"],
                execution=execution["sha256"],
                adapter=plan["adapter"],
                reference_profile=plan.get("reference_profile", "radiance-fp8"),
                **({"unpacker": unpacker} if unpacker is not None else {}),
            )
            if "reference_semantics" in plan and (
                model.config != plan["reference_semantics"]["weights"]["config"]
            ):
                raise DiagnosticError("checkpoint configuration differs from the reference")
            model._linear = operation
            campaign = CampaignWriter(
                root,
                candidate_plan,
                coverage=state_names(model.config) + OUTPUT_COMPONENTS,
                backend="experimental-parallel-ordered-c-numpy-reference",
            )
            write_private(root / "parallel-execution.json", execution)
            positions, inputs = observation_domain(candidate_plan)
            common = {
                "contract": model.contract,
                "execution": model.execution,
                "adapter": model.adapter,
                "positions": positions,
                "layers": model.config["num_hidden_layers"],
                "input_digests": inputs,
            }
            boundaries = BoundaryRecorder(root / "boundaries", **common)
            semantic = BoundaryRecorder(
                root / "semantic", **common, layer_stages=detailed_stages(model.config)
            )

            def observe(position, layer, stage, value):
                boundaries.record(position, layer, stage, value)
                semantic.record(position, layer, stage, value)

            model.capture = observe
            for token in plan["prefix"]:
                logits = model.step(token)
                print(f"CPU prefix {len(model.tokens)}/{len(plan['prefix'])}", flush=True)
            for index, expected in enumerate(campaign.expected):
                if index:
                    begin = len(model.tokens) - len(plan["prefix"])
                    end = expected["consumed"] - len(plan["prefix"])
                    for token in plan["forced_tokens"][begin:end]:
                        logits = model.step(token)
                path = root / expected["name"]
                write_model_frame(
                    model, path, phase=expected["phase"], pending=expected["pending"], logits=logits
                )
                campaign.record(path)
            boundaries.finish()
            semantic.finish()
            schedule = campaign.finish()
            report = seal(
                {
                    "schema": "urn:qwen:parallel-reference-capture:v1",
                    "execution": execution["sha256"],
                    "parent_plan": plan["sha256"],
                    "schedule": schedule["sha256"],
                    "elapsed_seconds": time.monotonic() - started,
                    "finished_ns": time.time_ns(),
                    "calls": operation.calls,
                    "parallel_calls": operation.parallel_calls,
                    "fallbacks": operation.fallbacks,
                    "gpu_used": False,
                    "status": "CAPTURED",
                    "qualification": "UNPROVED_PENDING_SERIAL_COMPARISON",
                    "published_to_session": False,
                    **(
                        {
                            "unpacking": {
                                "execution": unpacker.execution_binding["sha256"],
                                "calls": unpacker.calls,
                                "fallbacks": unpacker.fallbacks,
                            }
                        }
                        if unpacker is not None
                        else {}
                    ),
                }
            )
            write_private(root / "parallel-capture.json", report)
            return report
        finally:
            if model is not None:
                model.close()
            else:
                checkpoint.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--unpack-lookup", action="store_true")
    args = parser.parse_args()
    result = capture(
        private_json(args.plan), args.output, workers=args.workers, unpack_lookup=args.unpack_lookup
    )
    print(f"CAPTURED {result['sha256']}; serial byte comparison still required", flush=True)


if __name__ == "__main__":
    main()
