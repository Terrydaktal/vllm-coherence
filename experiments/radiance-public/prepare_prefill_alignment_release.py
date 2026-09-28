"""Freeze qualified prefill kernels and bind a new snapshot arithmetic identity."""

import argparse
import json
import shutil
from pathlib import Path

from prefill_alignment_runtime import digest, validate

from qwen_r9700_lab.diagnostic_contract import authenticate, seal

RUNTIME = (
    "prefill_attention_alignment.py",
    "prefill_gemm_alignment.py",
    "prefill_alignment_runtime.py",
    "prepared_prefill_scan.py",
    "prefill_dynamic_conv.py",
    "prefill_activation_tiles.py",
    "optimized_stock_norm.py",
    "optimized_d7_performance.py",
    "optimized_d7_worker.py",
    "stock_gdn_norm_quant.py",
)


def normalization_admission_revision(parent_runtime, source):
    """Reject every inherited normalization change except the exact row guards."""
    records = {}
    for name, anchor in (
        ("optimized_stock_norm.py", "(prefill_aligned and 8 < x.shape[0] <= 2048)"),
        ("stock_gdn_norm_quant.py", "or not 1 <= x.shape[0] <= 2048"),
    ):
        old_path, new_path = parent_runtime / name, source / name
        before, after = old_path.read_text(), new_path.read_text()
        if (
            before.count(anchor) != 1
            or before.replace(anchor, anchor.replace("2048", "4096")) != after
        ):
            raise ValueError(
                "wider prefill may only extend the qualified normalization row guard"
            )
        records[name] = {
            "parent_sha256": digest(old_path),
            "current_sha256": digest(new_path),
        }
    return records


def prepare(
    parent,
    attention,
    projection,
    qualification,
    profile,
    output,
    *,
    preserve_snapshot_contract=False,
    activation_tiles_qualification=None,
    gdn_qualification=None,
):
    source = Path(__file__).resolve().parent
    manifest = json.loads((parent / "optimized-release.json").read_text())
    profile = json.loads(profile.read_text())
    if profile["optimized_d7"]["manifest_sha256"] != digest(
        parent / "optimized-release.json"
    ):
        raise ValueError("parent payload and profile differ")
    entry = {
        "qualification": str(qualification),
        "qualification_sha256": digest(qualification),
        "sources": {name: digest(source / name) for name in RUNTIME},
    }
    evidence = json.loads(qualification.read_text())
    widened = evidence.get("max_prefill_rows", 2048) == 4096
    perf_path = Path(manifest["performance"]).relative_to("/qualification")
    parent_perf = json.loads((parent / perf_path).read_text())
    refreshed_gdn = None
    if widened:
        # Only these admission guards may change in the inherited norm code.
        # Their arithmetic bodies and all decode dispatch remain byte-identical.
        # Fresh operator evidence covers the newly admitted rows; a full-model
        # replay alone is insufficient to authorize a wider domain.
        entry["normalization_admission_revision"] = normalization_admission_revision(
            parent / perf_path.parent / "runtime", source
        )
        if gdn_qualification is None:
            raise ValueError("wider prefill needs fresh GDN norm/quant evidence")
        from pi_prefill_admission import gdn_evidence

        inherited = parent_perf["gdn_norm_quant"]
        refreshed_gdn = {
            **inherited,
            "qualification": str(gdn_qualification),
            "qualification_sha256": digest(gdn_qualification),
            "sources": {name: digest(source / name) for name in inherited["sources"]},
        }
        norm_report = gdn_evidence(refreshed_gdn)
        widths = {
            c["prefill_width"]: c["equal"]
            for c in norm_report["checks"]
            if "prefill_width" in c
        }
        if any(
            widths.get(width) is not True
            for width in (2049, 2560, 3295, 3296, 3297, 4095, 4096)
        ):
            raise ValueError(
                "wider prefill GDN norm/quant boundary checks are incomplete"
            )
        entry["max_prefill_rows"] = 4096
    preserved = None
    if preserve_snapshot_contract:
        old = manifest.get("prefill_alignment", {})
        contract = profile["optimized_d7"]["arithmetic"].get("prefill_alignment")
        if (
            not old.get("contract")
            or contract != old["contract"]
            or any(
                old.get("sources", {}).get(name) != entry["sources"][name]
                and not (widened and name == "optimized_stock_norm.py")
                for name in (
                    "optimized_stock_norm.py",
                    "optimized_d7_performance.py",
                    "optimized_d7_worker.py",
                )
            )
            or not evidence.get("tool_continuations")
            or any(
                run.get("status") != "PASS_FOR_DECLARED_SCOPE"
                or not run.get("cases")
                or not all(case.get("same_tokens") is True for case in run["cases"])
                for run in evidence["tool_continuations"]
            )
        ):
            raise ValueError(
                "snapshot compatibility needs the unchanged corrected decode contract and exact continuation evidence"
            )
        preserved = dict(contract)
        entry["arithmetic_reference"] = {
            "parent_manifest_sha256": digest(parent / "optimized-release.json"),
            "qualification_sha256": entry["qualification_sha256"],
            "scope": "Same declared corrected arithmetic, qualified by exact operator/state and full-model samples. Reference hashes name the arithmetic baseline; current implementation hashes remain in builds, sources and the release manifest. Not a universal proof.",
        }
    entry.update(
        {
            name: bool(evidence.get(name))
            for name in ("prepared_scan", "input_tiles", "dynamic_conv")
        }
    )
    if entry["input_tiles"]:
        entry["input_binary_sha256"] = evidence["input_tiles"]["binary_sha256"]
    if entry["dynamic_conv"]:
        entry["conv_native_sha256"] = evidence["dynamic_conv"]["native_source_sha256"]
    for name, root in (("attention", attention), ("projection", projection)):
        data = json.loads((root / "build.json").read_text())
        authenticate(data)
        entry[name] = {"build": str(root), "build_sha256": data["sha256"]}
    validate(entry)
    inherited_tiles = json.loads((parent / perf_path).read_text()).get(
        "activation_tiles"
    )
    refreshed_tiles = None
    if inherited_tiles:
        changed = any(
            digest(source / name) != expected
            for name, expected in inherited_tiles["sources"].items()
        )
        if changed and activation_tiles_qualification is None:
            raise ValueError(
                "changed packing source needs fresh activation-layout qualification"
            )
    if activation_tiles_qualification is not None:
        from prefill_tiles_admission import evidence as check_activation_tiles

        if inherited_tiles is None:
            raise ValueError("parent has no activation-layout qualification to refresh")
        refreshed_tiles = {
            **inherited_tiles,
            "qualification": str(activation_tiles_qualification),
            "qualification_sha256": digest(activation_tiles_qualification),
            "sources": {
                name: digest(source / name) for name in inherited_tiles["sources"]
            },
        }
        check_activation_tiles(refreshed_tiles)
    for name, expected in manifest["files"].items():
        path = Path(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or (parent / path).is_symlink()
            or digest(parent / path) != expected
        ):
            raise ValueError("parent artifact changed or unsafe")
    output.mkdir(mode=0o700)
    files = {}

    def copy(origin, relative):
        target = output / relative
        if origin.is_symlink() or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe release artifact")
        target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        shutil.copy2(origin, target)
        target.chmod(0o600)
        files[str(relative)] = digest(target)

    def write(relative, value):
        target = output / relative
        target.write_text(json.dumps(value, indent=2) + "\n")
        target.chmod(0o600)
        files[str(relative)] = digest(target)

    for name in manifest["files"]:
        copy(parent / name, Path(name))
    base = Path("diagnostics/prefill-alignment-v1")
    for name, root in (("attention", attention), ("projection", projection)):
        data = json.loads((root / "build.json").read_text())
        for filename, expected in data["files"].items():
            if Path(filename).name != filename or digest(root / filename) != expected:
                raise ValueError("native prefill artifact changed")
            copy(root / filename, base / name / filename)
        copy(root / "build.json", base / name / "build.json")
        entry[name]["build"] = "/qualification/" + str(base / name)
    copy(qualification, base / "qualification.json")
    entry["qualification"] = "/qualification/" + str(base / "qualification.json")
    entry["contract"] = {
        "attention": "same per-query TILE16, M1 split boundaries and FP32 merge order as decode",
        "output_projections": "four contiguous 128-coefficient-aligned K ranges; ordered FP32 merge then BF16",
        "final_normalization": "qualified M1 residual normalization for prefill and decode",
        "attention_binary_sha256": json.loads((attention / "build.json").read_text())[
            "files"
        ]["candidate.so"],
        "projection_binary_sha256": json.loads((projection / "build.json").read_text())[
            "files"
        ]["candidate.so"],
        "snapshot_compatibility": "previous prefill states must be rebuilt; token histories remain valid",
    }
    if preserved is not None:
        # This is the identity of the unchanged numerical reference, not a
        # claim that the new implementation has the reference binary's hash.
        entry["contract"] = preserved
    repair_path = Path(manifest["repair"]).relative_to("/qualification")
    runtime = perf_path.parent / "runtime"
    for name in list(files):
        if Path(name).name in RUNTIME:
            copy(source / Path(name).name, Path(name))
    for name in RUNTIME:
        copy(source / name, runtime / name)
    if refreshed_gdn is not None:
        for name in refreshed_gdn["sources"]:
            for relative in list(files):
                if Path(relative).name == name:
                    copy(source / name, Path(relative))
            copy(source / name, runtime / name)
    repair = json.loads((output / repair_path).read_text())
    repair["prefill_alignment_parent"] = repair.pop("sha256")
    for name in repair["sources"]:
        if name in RUNTIME:
            repair["sources"][name] = digest(source / name)
    repair = seal(repair)
    write(repair_path, repair)
    perf = json.loads((output / perf_path).read_text())
    if refreshed_gdn is not None:
        relative = base / "gdn-wide-qualification.json"
        copy(gdn_qualification, relative)
        refreshed_gdn["qualification"] = "/qualification/" + str(relative)
        perf["gdn_norm_quant"] = refreshed_gdn
        perf["sources"].update(refreshed_gdn["sources"])
    if refreshed_tiles is not None:
        relative = base / "activation-tiles.json"
        copy(activation_tiles_qualification, relative)
        refreshed_tiles["qualification"] = "/qualification/" + str(relative)
        perf["activation_tiles"] = refreshed_tiles
    perf["prefill_alignment_parent"] = perf.pop("sha256")
    perf["reference_repair"] = repair["sha256"]
    perf["sources"].update({name: digest(source / name) for name in RUNTIME})
    perf["prefill_alignment"] = entry
    perf["full_model_status"] = "SAMPLE_CHECKED_PREFILL_DECODE_ALIGNMENT"
    perf = seal(perf)
    write(perf_path, perf)
    manifest.update(
        parent_manifest_sha256=digest(parent / "optimized-release.json"),
        host_root=str(output),
        files=dict(sorted(files.items())),
        repair_sha256=files[str(repair_path)],
        performance_sha256=files[str(perf_path)],
        prefill_alignment=entry,
    )
    (output / "optimized-release.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    optimized = profile["optimized_d7"]
    optimized.update(
        manifest_sha256=digest(output / "optimized-release.json"),
        host_root=str(output),
        prefill_alignment=entry,
    )
    if preserved is None:
        optimized["arithmetic"].update(
            prefill_alignment=entry["contract"],
            repair_sha256=repair["sha256"],
            performance_sha256=perf["sha256"],
        )
    optimized["arithmetic_evidence_scope"] = (
        "Pinned single-R9700 prefill/decode forced-token samples and independent operator checks; not a universal proof or loop-rate measurement."
    )
    (output / "runtime-radiance-1.0.16.json").write_text(
        json.dumps(profile, indent=2) + "\n"
    )
    return {
        "manifest": str(output / "optimized-release.json"),
        "status": "QUALIFIED_NOT_DEPLOYED",
        "files": len(files),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "parent",
        "attention",
        "projection",
        "qualification",
        "profile",
        "output",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument(
        "--gdn-qualification",
        type=Path,
        help="Fresh row-invariant GDN norm/quant checks when extending prefill to 4096 rows",
    )
    parser.add_argument(
        "--activation-tiles-qualification",
        type=Path,
        help="Fresh native layout/GEMM/graph evidence when the inherited packing source changes",
    )
    parser.add_argument(
        "--preserve-snapshot-contract",
        action="store_true",
        help="Keep the corrected numerical reference identity for a qualified implementation-only optimization",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.parent,
                args.attention,
                args.projection,
                args.qualification,
                args.profile,
                args.output,
                preserve_snapshot_contract=args.preserve_snapshot_contract,
                activation_tiles_qualification=args.activation_tiles_qualification,
                gdn_qualification=args.gdn_qualification,
            )
        )
    )
