"""Seal aggregate prefill-alignment evidence without publishing token/tensor data."""

import argparse
import json
from pathlib import Path

from prefill_alignment_runtime import digest, validate
from prepare_prefill_alignment_release import RUNTIME

from qwen_r9700_lab.diagnostic_contract import authenticate, seal


def collect(args):
    source = Path(__file__).resolve().parent

    def read(path):
        return json.loads(path.read_text())

    report = {
        "schema": "urn:coherence:prefill-alignment:v1",
        "status": "SAMPLE_CHECKED",
        "sources": {name: digest(source / name) for name in RUNTIME},
        "diagnostic_sources": {
            name: digest(source / name)
            for name in (
                "prefill_divergence_probe.py",
                "diagnose_prefill_decode.py",
                "qualify_response_end.py",
                Path(__file__).name,
            )
        },
        "builds": {},
        "attention_cases": read(args.attention_cases)["cases"],
        "projection_cases": read(args.projection_cases),
        "model_comparisons": [read(p) for p in args.comparisons],
        "baseline_comparisons": [read(p) for p in args.baselines],
        "tool_continuations": [],
        "evidence_files": {},
        "scope": (
            "Forced identical synthetic token history; compiled pinned single-R9700 "
            "TP1 backend; same full BF16 M8 head on both hidden captures. Final native "
            "builds installed through the production adapter in the isolated worker. "
            "Release startup and lifecycle are checked separately. Sample evidence, "
            "not an arbitrary-input proof, independent full-model reference, "
            "intelligence-loss percentage or loop-rate measurement."
        ),
    }
    entry = {"sources": report["sources"]}
    maximum = getattr(args, "max_prefill_rows", 2048)
    if maximum != 2048:
        report["max_prefill_rows"] = entry["max_prefill_rows"] = maximum
    for name, path in (
        ("prepared_scan", args.scan_evidence),
        ("input_tiles", args.input_evidence),
        ("dynamic_conv", args.conv_evidence),
    ):
        if path:
            report[name] = read(path)
            entry[name] = True
    if entry.get("input_tiles"):
        entry["input_binary_sha256"] = report["input_tiles"]["binary_sha256"]
    if entry.get("dynamic_conv"):
        entry["conv_native_sha256"] = report["dynamic_conv"]["native_source_sha256"]
    for name, root in (("attention", args.attention), ("projection", args.projection)):
        build = read(root / "build.json")
        authenticate(build)
        report["builds"][name] = build["sha256"]
        entry[name] = {"build": str(root), "build_sha256": build["sha256"]}
    if read(args.attention_cases)["build"] != report["builds"]["attention"]:
        raise ValueError("attention samples cover a different build")
    if args.installation_parent:
        parent = read(args.installation_parent)
        authenticate(parent)
        changed = [
            name
            for name in RUNTIME
            if parent["sources"][name] != report["sources"][name]
        ]
        if changed != ["optimized_d7_performance.py"] or any(
            parent[key] != report[key]
            for key in (
                "builds",
                "attention_cases",
                "projection_cases",
                "model_comparisons",
            )
        ):
            raise ValueError(
                "installation-only evidence bridge cannot change numerical artifacts"
            )
        report["installation_revision"] = {
            "parent_report_sha256": digest(args.installation_parent),
            "changed_sources": changed,
            "change": "Give the prefill wrapper a separate HookSet around the existing decode attention wrapper; native arithmetic and production prefill adapter unchanged.",
            "scope": "Reuses numerical samples of unchanged kernels/adapter; packaged startup and continuation checks are separate deployment evidence.",
        }
    for path in args.tool_evidence:
        evidence = read(path)
        if evidence["status"] != "PASS_FOR_DECLARED_SCOPE" or not all(
            c["same_tokens"] for c in evidence["cases"]
        ):
            raise ValueError("tool-continuation comparison failed")
        report["tool_continuations"].append(evidence)
    for path in [
        args.attention_cases,
        args.projection_cases,
        *args.comparisons,
        *args.baselines,
        *args.tool_evidence,
        *([args.scan_evidence] if args.scan_evidence else []),
        *([args.input_evidence] if args.input_evidence else []),
        *([args.conv_evidence] if args.conv_evidence else []),
    ]:
        # Run name and digest identify the preserved capture, without exporting
        # filesystem roots, prompt IDs or activation payloads.
        name = "/".join(path.parts[-3:])
        if name in report["evidence_files"]:
            raise ValueError("duplicate evidence name")
        report["evidence_files"][name] = digest(path)
    output = args.output
    if output.exists():
        raise ValueError("refusing to replace an existing qualification")
    pending = output.with_suffix(output.suffix + ".pending")
    with pending.open("x") as stream:
        stream.write(json.dumps(seal(report), indent=2) + "\n")
    pending.chmod(0o600)
    entry.update(qualification=str(pending), qualification_sha256=digest(pending))
    try:
        validate(entry)
        pending.rename(output)
    finally:
        pending.unlink(missing_ok=True)
    return {"status": report["status"], "sha256": digest(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "attention",
        "projection",
        "attention-cases",
        "projection-cases",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("comparisons", "baselines", "tool-evidence"):
        parser.add_argument("--" + name, type=Path, nargs="+", required=True)
    parser.add_argument("--installation-parent", type=Path)
    parser.add_argument("--scan-evidence", type=Path)
    parser.add_argument("--input-evidence", type=Path)
    parser.add_argument("--conv-evidence", type=Path)
    parser.add_argument(
        "--max-prefill-rows", type=int, choices=(2048, 4096), default=2048
    )
    print(json.dumps(collect(parser.parse_args())))
