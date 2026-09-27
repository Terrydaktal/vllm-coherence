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
    "optimized_stock_norm.py",
    "optimized_d7_performance.py",
    "optimized_d7_worker.py",
)


def prepare(parent, attention, projection, qualification, profile, output):
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
    for name, root in (("attention", attention), ("projection", projection)):
        data = json.loads((root / "build.json").read_text())
        authenticate(data)
        entry[name] = {"build": str(root), "build_sha256": data["sha256"]}
    validate(entry)
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
    perf_path = Path(manifest["performance"]).relative_to("/qualification")
    repair_path = Path(manifest["repair"]).relative_to("/qualification")
    runtime = perf_path.parent / "runtime"
    for name in list(files):
        if Path(name).name in RUNTIME:
            copy(source / Path(name).name, Path(name))
    for name in RUNTIME:
        copy(source / name, runtime / name)
    repair = json.loads((output / repair_path).read_text())
    repair["prefill_alignment_parent"] = repair.pop("sha256")
    for name in repair["sources"]:
        if name in RUNTIME:
            repair["sources"][name] = digest(source / name)
    repair = seal(repair)
    write(repair_path, repair)
    perf = json.loads((output / perf_path).read_text())
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
            )
        )
    )
