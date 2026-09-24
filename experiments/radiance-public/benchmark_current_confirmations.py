"""Repeat finite M1/M8 and eager/compiled checks on an authenticated release.

This is forced-token correctness replay, never a serving-speed benchmark.
Bootstrap the release in a disposable container. The only head override is the
full BF16 comparison head; the production Global-512 setting is not changed.
Private token IDs and ranked logits remain in the explicitly supplied directory.
"""

import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path


ARMS = ("compiled-m1", "compiled-m8", "eager-m1", "eager-m8", "stages")
OPERATOR_INVENTORY = {
    "normalization": 161, "residual_norm": 129, "residual_carry": 129,
    "fused_norm_quant": 129, "fused_norm_quant_m1": 129, "dynamic_fp8": 129,
    "gdn_gated_norm": 48, "gdn_fused_quant_m1": 48,
    "mlp_silu_gate": 4, "attention_sigmoid_gate": 4, "mlp_fused_quant_m1": 320,
    "rope": 36, "rope_dispatch": 36, "rope_tail": 36,
    "embedding": 4, "kv_write": 4, "kv_fp8_storage": 1,
    "mxfp4_projection": 496, "mxfp4_coefficient": 496,
    "gdn_recurrence": 48, "gdn_output": 48, "gdn_slot_isolation": 48,
    "gdn_long_decay": 5, "gdn_convolution": 48, "convolution_history": 48,
    "convolution_slot_isolation": 48,
    "attention_periodic_long_context": 88, "attention_memory_isolation": 88,
    "head_rerank": 3, "head_coefficient": 3, "head_full_m1": 3,
}
STAGE_INVENTORY = {
    "Embedding + first input normalization + FP8 production",
    "Layer input residual/normalization + FP8 production",
    "Post-attention/GDN residual/normalization + FP8 production",
    "GDN input projection", "GDN convolution", "GDN recurrence and gates",
    "GDN output gated normalization + FP8 production", "GDN output projection",
    "Attention input projection", "Attention Q/K normalization",
    "Attention Q/K normalization, RoPE and layout", "Attention KV write",
    "Attention decode and split-KV merge", "Attention output gating",
    "Attention output activation FP8 quantization", "Attention output projection",
    "MLP gate/up projection", "MLP SiLU and gating", "MLP down input FP8 quantization",
    "MLP down projection", "Final normalization/layout", "Full BF16 comparison head",
}


def current_config(base_config, spec, lane, **kwargs):
    config = base_config(spec, lane, **kwargs)
    if kwargs["execution_mode"] == "compiled-no-graphs":
        config["worker_cls"] = "native_d7_tape_worker.NativeTapeWorker"
    if kwargs["execution_mode"] == "eager":
        # The eager control must include the already-qualified RNE RoPE repair.
        # ExecutionModeWorker alone is the historical, unaligned eager path.
        config["worker_cls"] = "rotary_mode_d7_worker.RotaryRneWorker"
    return config


def bootstrap(args):
    old = sys.argv
    sys.argv = [str(args.patches / "bootstrap_radiance_release.py"), "--check-only"]
    sys.path.insert(0, str(args.patches))
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        sys.argv = old
    # Start a fresh interpreter so the authenticated release's sitecustomize and
    # import paths are applied exactly as on a normal server launch.
    os.environ["RADIANCE_VERIFY_HEAD"] = "0"
    os.environ.pop("RADIANCE_VERIFY_HEAD_GLOBAL_TOPK", None)
    os.environ["QWEN_OPTIMIZED_STARTUP_RECEIPT"] = str(args.output / "before-compile.json")
    os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), "worker", *old[2:]])


def worker(args):
    from types import SimpleNamespace

    import benchmark_optimized_d7 as benchmark
    from qwen_r9700_lab.diagnostic_contract import authenticate, private_json, seal, write_private

    fixture = private_json(args.fixture)
    authenticate(fixture)
    if len(fixture["prefix"]) != 60000 or len(fixture["output"]) != 321:
        raise ValueError("confirmations require 60,000 input tokens and 320 prediction rows")
    if os.environ.get("RADIANCE_VERIFY_HEAD") != "0":
        raise ValueError("full BF16 comparison head required")
    profile = json.loads((args.patches / "runtime-radiance-1.0.16.json").read_text())
    manifest_path = args.patches / "optimized-release.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != profile["optimized_d7"]["manifest_sha256"]:
        raise ValueError("release manifest identity changed")
    spec = private_json(args.spec)
    package = Path("/opt/vllm/lib/python3.12/site-packages")
    names = sorted(set(spec["binding"]["files"]) | set(profile["source_preimages"]))
    binding = seal({
        "schema": "urn:qwen:radiance-native-binding:v1",
        "files": {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in names},
        "release_manifest_sha256": profile["optimized_d7"]["manifest_sha256"],
        "scope": "Authenticated installed source identity, not a correctness certificate",
    })
    spec = {"binding": binding, "native_config": spec["native_config"]}
    actual_spec = args.output / "installed-spec.json"
    write_private(actual_spec, spec)
    write_private(args.output / "execution-binding.json", {
        "arm": args.arm,
        "fixture_sha256": fixture["sha256"],
        "release_manifest_sha256": profile["optimized_d7"]["manifest_sha256"],
        "source_binding_sha256": binding["sha256"],
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "head": "full-bf16",
        "production_head": "global512",
        "snapshot_reuse": False,
        "serving_timing_measurement": False,
    })
    lane = "fixed-bf16"
    (args.output / lane).mkdir(mode=0o700)
    (args.private / lane).mkdir(mode=0o700)
    original_config = benchmark.make_config
    benchmark.make_config = lambda spec, lane, **kw: current_config(original_config, spec, lane, **kw)
    benchmark.worker(SimpleNamespace(
        spec=actual_spec, fixture=args.fixture, output=args.output, private=args.private,
        lane=lane, m1=args.arm.endswith("m1"), isolated_capture=False,
        execution_mode=("compiled-no-graphs" if args.arm == "stages" else
                        "eager" if args.arm.startswith("eager") else "compiled"),
        correctness=True, repeats=0, profile=False, with_correctness=False,
        tokens=321, profile_warmup_rounds=8, profile_rounds=8, profile_chunk_rounds=8,
    ))


def compare_pair(left, right, *, left_width, right_width):
    from qwen_r9700_lab.conformance_topk import aggregate, compare_rows, require
    from qwen_r9700_lab.diagnostic_contract import authenticate

    for report, width in ((left, left_width), (right, right_width)):
        authenticate(report)
        require(report["schema"] == "urn:qwen:d7-equivalence-private-rows:v1", "wrong row schema")
        require(len(report["rows"]) == 320, "incomplete 320-row confirmation")
        for i, row in enumerate(report["rows"]):
            require(row["position"] == i and row["absolute_position"] == 60000 + i,
                    "prediction position mismatch")
            require(row["target_rows"] == width, "wrong actual target width")
    require(left["continuation"] == right["continuation"], "different forced inputs")
    compared = [compare_rows(a["logits"], b["logits"])
                for a, b in zip(left["rows"], right["rows"], strict=True)]
    return {
        "decode": aggregate(compared),
        "prefill": compare_rows(left["prefill"], right["prefill"]),
        "first_different_prediction": next((i for i, row in enumerate(compared)
                                             if not row["full_logits_exact"]), None),
    }


def summarize(root, output):
    from qwen_r9700_lab.conformance_topk import require
    from qwen_r9700_lab.diagnostic_contract import authenticate, private_json, seal

    reports, bindings, executions = {}, {}, {}
    for arm in ARMS[:4]:
        reports[arm] = private_json(root / "private" / arm / "fixed-bf16/pass-00/correctness/rows.json")
        bindings[arm] = private_json(root / arm / "execution-binding.json")
        metadata = private_json(root / arm / "fixed-bf16/actual-runtime.json")
        sample = private_json(root / arm / "fixed-bf16/pass-00.json")
        authenticate(metadata)
        authenticate(sample)
        compiled = arm.startswith("compiled")
        require(metadata["enforce_eager"] is not compiled, "execution mode changed")
        require(metadata["compilation_mode"] == (3 if compiled else 0), "wrong compiler mode")
        require(metadata["graph_mode"] == ("PIECEWISE" if compiled else "NONE"), "wrong graph mode")
        require(metadata["runtime"]["flags"]["RADIANCE_VERIFY_HEAD"] == "0", "wrong comparison head")
        count = sample["observation"]["observation"]["counts"].get("target_graph_replays", 0)
        require((count > 0) is compiled, "actual graph execution differs")
        if not compiled:
            require(sample["observation"]["rotary_intervention"]["calls"] > 0,
                    "eager control omitted its existing RNE repair")
        executions[arm] = {
            "metadata_sha256": metadata["sha256"],
            "private_rows_sha256": reports[arm]["sha256"],
            "sample_sha256": sample["sha256"],
            "binding": bindings[arm],
            "compilation_mode": metadata["compilation_mode"],
            "graph_mode": metadata["graph_mode"],
            "target_graph_replays": count,
            "eager_rotary_repair": metadata.get("rotary_intervention"),
        }
    for key in ("release_manifest_sha256", "source_binding_sha256", "fixture_sha256"):
        require(len({b[key] for b in bindings.values()}) == 1, "different execution binding: " + key)
    pairs = [("compiled-m1", "compiled-m8"), ("eager-m1", "compiled-m1"),
             ("eager-m8", "compiled-m8"), ("eager-m1", "compiled-m8")]
    comparisons = {
        a + "_vs_" + b: compare_pair(reports[a], reports[b], left_width=int(a[-1]), right_width=int(b[-1]))
        for a, b in pairs
    }
    passed = all(c["decode"]["full_logits_exact"] == 320 and c["prefill"]["full_logits_exact"]
                 for c in comparisons.values())
    report = seal({
        "schema": "coherence-current-320-confirmations-v1",
        "status": "SAMPLE_CHECKED" if passed else "DISAGREEMENT_OBSERVED",
        "prefix_tokens": 60000, "decode_tokens_per_arm": 320, "prefill_predictions_per_arm": 1,
        "fixture_sha256": bindings["compiled-m8"]["fixture_sha256"],
        "optimized_manifest_sha256": bindings["compiled-m8"]["release_manifest_sha256"],
        "scope": "Current corrected release: common forced Pi tokens, fresh prefill in every arm; full BF16 comparison head; no snapshot reuse. Approximate Global-512 support has separate evidence.",
        "limits": "Finite agreement, not arbitrary-input proof, independent model certification, natural completion quality, all rejection histories or snapshot/concurrency qualification. This 320-token slice was recovered from the retained Pi corpus and differs from the historical standalone 320 fixture.",
        "executions": executions, "comparisons": comparisons,
        "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "private_text_decoded_or_published": False,
    })
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "comparisons": len(comparisons), "tokens_per_arm": 320}))


def summarize_operators(source, output):
    """Keep the operator audit's exact scope and tolerances in a small public record."""
    from collections import defaultdict

    from qwen_r9700_lab.diagnostic_contract import seal

    audit = json.loads(source.read_text())
    if audit["rows_per_site"] != 320 or not audit["checks"]:
        raise ValueError("incomplete 320-row operator audit")
    checks = defaultdict(list)
    for check in audit["checks"]:
        if type(check.get("passed")) is not bool:
            raise ValueError("operator check lacks a boolean result")
        checks[check["stage"]].append(check)
    if {k: len(v) for k, v in checks.items()} != OPERATOR_INVENTORY:
        raise ValueError("incomplete operator inventory")
    groups = {}
    for stage, rows in checks.items():
        groups[stage] = {
            "checks": len(rows),
            "passed": sum(r["passed"] for r in rows),
            "sites": len({r["site"] for r in rows}),
            "elements": sum(r.get("elements", 0) for r in rows),
            "max_absolute_error": max((r.get("max_abs", 0) for r in rows), default=0),
            "max_relative_l2_error": max((r.get("relative_l2", 0) for r in rows), default=0),
            "bf16_differences_from_fp64_oracle": sum(r.get("bf16_mismatches", 0) for r in rows),
            "criteria": [json.loads(x) for x in sorted({json.dumps(r["criterion"], sort_keys=True)
                        for r in rows if "criterion" in r})],
        }
    failed = sum(not c["passed"] for c in audit["checks"])
    if failed != audit["failed_checks"]:
        raise ValueError("operator failure count is inconsistent")
    if (audit["status"] == "SAMPLE_CHECKED") != (failed == 0 and not audit["errors"]):
        raise ValueError("operator status conceals failures")
    report = seal({
        "schema": "coherence-current-operator-confirmations-v1",
        "status": audit["status"],
        "optimized_manifest_sha256": audit["release_sha256"],
        "audit_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "probe_sha256": audit["probe_sha256"],
        "oracle_sha256": audit["oracle_sha256"],
        "sources": audit["sources"],
        "rows_per_site": audit["rows_per_site"],
        "checks": len(audit["checks"]), "failed_checks": failed, "errors": audit["errors"],
        "groups": groups, "elapsed_seconds": audit["elapsed_seconds"],
        "reference": audit["reference"],
        "limits": audit["limits"] + [
            "FP64 checks allow declared numerical error; BF16 differences from the oracle are retained, not relabelled exact equality.",
            "Projection checks sample output channels; attention uses structured pages. Broader prior checks have separate evidence.",
        ],
        "private_chat_read": audit["private_chat_read"],
    })
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checks": report["checks"], "failed_checks": failed}))


def summarize_stages(root, arm, output):
    from analyze_native_d7_stages import aggregate_groups
    from qwen_r9700_lab.diagnostic_contract import authenticate, private_json, seal

    def checked(path):
        value = private_json(path)
        authenticate(value)
        return value

    metadata = checked(root / arm / "fixed-bf16/actual-runtime.json")
    if metadata["enforce_eager"] or metadata["compilation_mode"] != 3 or metadata["graph_mode"] != "NONE":
        raise ValueError("stage replay must use the declared compiled diagnostic mode")
    if metadata["runtime"]["flags"]["RADIANCE_VERIFY_HEAD"] != "0":
        raise ValueError("wrong stage comparison head")
    private = root / "private" / arm / "fixed-bf16/pass-00"
    control = checked(root / "private/compiled-m8/fixed-bf16/pass-00/correctness/rows.json")
    actual = checked(private / "correctness/rows.json")
    bridge = compare_pair(control, actual, left_width=8, right_width=8)
    if bridge["decode"]["full_logits_exact"] != 320 or not bridge["prefill"]["full_logits_exact"]:
        raise ValueError("diagnostic replay does not reproduce the compiled graph control")
    binding = private_json(root / arm / "execution-binding.json")
    control_binding = private_json(root / "compiled-m8/execution-binding.json")
    for key in ("release_manifest_sha256", "source_binding_sha256", "fixture_sha256"):
        if binding[key] != control_binding[key]:
            raise ValueError("stage execution binding changed: " + key)
    paths = sorted(private.glob("native-tape-group-*.json"))
    if len(paths) != 40:
        raise ValueError("incomplete 320-position stage capture")
    groups = [checked(path) for path in paths]
    for i, group in enumerate(groups):
        if group["first_absolute_position"] != 60000 + 8 * i:
            raise ValueError("missing or duplicate stage positions")
        if not group.get("injected_early_projection_fault_detected"):
            raise ValueError("early corruption did not reach the checked vocabulary output")
        if group["diagnostic_sources"] != groups[0]["diagnostic_sources"]:
            raise ValueError("diagnostic source identities changed")
    aggregate = aggregate_groups(groups)
    if set(aggregate["inventory"]) != STAGE_INVENTORY:
        raise ValueError("incomplete current stage inventory")
    for entry in aggregate["inventory"].values():
        if set(entry["columns"]) != {"fixed", "final_modes"}:
            raise ValueError("missing current M1/M8 or eager/compiled stage comparison")
    passed = all(values["full_logits_exact"] == values["positions"] == 320
                 for stage in aggregate["stages"].values() for values in stage.values())
    report = seal({
        "schema": "coherence-current-stage-confirmations-v1",
        "status": "SAMPLE_CHECKED" if passed else "DISAGREEMENT_OBSERVED",
        "prefix_tokens": 60000, "decode_positions": 320, "groups": 40,
        "optimized_manifest_sha256": binding["release_manifest_sha256"],
        "fixture_sha256": binding["fixture_sha256"],
        "binding": binding, "metadata_sha256": metadata["sha256"],
        "group_sha256": [g["sha256"] for g in groups],
        "diagnostic_sources": groups[0]["diagnostic_sources"],
        "compiled_graph_bridge": bridge,
        "comparisons": {"fixed": "current M1 row execution versus current M8",
                        "final_modes": "current eager versus compiled M8"},
        "stages": aggregate["stages"], "inventory": aggregate["inventory"],
        "per_instance": aggregate["per_instance"], "local_checks": aggregate["local_checks"],
        "negative_controls_passed_groups": 40, "cache_restored_groups": 40,
        "scope": "Each stage receives the same captured correct inputs, one layer instance at a time. Unequal local output/state requires native suffix replay; equality can reuse the validated suffix. A position passes only when every layer instance passes. Compiled diagnostic execution disables graphs and is bridged to the production graph control; no timing claim.",
        "limits": "Finite equivalence between actual implementations, not an independent mathematical model. Shared custom operators remain shared. Fused normalization/FP8 is checked jointly; attention decode and merge are checked jointly. Drafter, full session lifecycle, every rejection history and approximate Global-512 completeness have separate evidence.",
        "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "stages": len(aggregate["stages"]), "positions": 320}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bootstrap", "worker", "report", "operator-report", "stage-report"))
    parser.add_argument("--arm", choices=ARMS)
    for name in ("fixture", "spec", "output", "private"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--operator-audit", type=Path)
    parser.add_argument("--stage-arm", default="stages")
    parser.add_argument("--patches", type=Path, default=Path("/patches"))
    args = parser.parse_args()
    os.umask(0o077)
    if args.command == "stage-report":
        if args.root is None or args.output is None:
            parser.error("stage-report requires --root and --output")
        summarize_stages(args.root, args.stage_arm, args.output)
        return
    if args.command == "operator-report":
        if args.operator_audit is None or args.output is None:
            parser.error("operator-report requires --operator-audit and --output")
        summarize_operators(args.operator_audit, args.output)
        return
    if args.command == "report":
        if args.root is None or args.output is None:
            parser.error("report requires --root and --output")
        summarize(args.root, args.output)
        return
    if any(getattr(args, k) is None for k in ("arm", "fixture", "spec", "output", "private")):
        parser.error("execution requires --arm, --fixture, --spec, --output and --private")
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.private.mkdir(parents=True, exist_ok=True, mode=0o700)
    {"bootstrap": bootstrap, "worker": worker}[args.command](args)


if __name__ == "__main__":
    main()
