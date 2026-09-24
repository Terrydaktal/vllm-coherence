"""Freeze the repaired attention and admission checks into a new numerical release."""

import argparse
import json
import shutil
from pathlib import Path

from attention_precision import digest
from attention_precision_release import validate
from prepare_optimized_pi_release import reseal

RUNTIME = (
    "attention_precision.py",
    "attention_precision_release.py",
    "attention_precision_runtime.py",
    "build_stock_m1_attention_shared.py",
    "m1_followup_guards.py",
    "stock_m1_attention.py",
    "stock_m1_attention_shared.py",
    "probe_stock_m1_attention_shared.py",
    "probe_attention_precision_repair.py",
    "probe_attention_precision_oracle.py",
    "probe_m1_attention_precision.py",
)


def prepare(parent, build, witnesses, alignment, oracle, profile, output):
    source = Path(__file__).resolve().parent
    manifest = json.loads((parent / "optimized-release.json").read_text())
    profile = json.loads(profile.read_text())
    if profile["optimized_d7"]["manifest_sha256"] != digest(
        parent / "optimized-release.json"
    ):
        raise ValueError("parent payload and profile differ")
    entry = {
        "build": str(build),
        "build_sha256": digest(build / "build.json"),
        "sources": {name: digest(source / name) for name in RUNTIME},
    }
    for name, path in (
        ("witnesses", witnesses),
        ("alignment", alignment),
        ("oracle", oracle),
    ):
        entry[name] = str(path)
        entry[name + "_sha256"] = digest(path)
    metadata, reports = validate(entry)
    for name, expected in manifest["files"].items():
        p = Path(name)
        if (
            p.is_absolute()
            or ".." in p.parts
            or (parent / p).is_symlink()
            or digest(parent / p) != expected
        ):
            raise ValueError("parent artifact changed or unsafe")
    output.mkdir(mode=0o700)
    files = {}

    def copy(origin, relative):
        target = output / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if origin.is_symlink():
            raise ValueError("indirect release input")
        shutil.copy2(origin, target)
        target.chmod(0o600)
        files[str(relative)] = digest(target)

    def write(relative, value):
        target = output / relative
        target.write_text(json.dumps(value, indent=2) + "\n")
        target.chmod(0o600)
        files[str(relative)] = digest(target)

    def rel(path):
        return Path(path).relative_to("/qualification")

    for name in manifest["files"]:
        copy(parent / name, Path(name))
    base = Path("diagnostics/attention-precision-v1")
    for name in ("build.json", *metadata["files"]):
        copy(build / name, base / "build" / name)
    for name, path in (
        ("witnesses", witnesses),
        ("alignment", alignment),
        ("oracle", oracle),
    ):
        destination = base / name / "result.json"
        copy(path, destination)
        entry[name] = "/qualification/" + str(destination)
    entry["build"] = "/qualification/" + str(base / "build")
    entry["contract"] = {
        "query_key": "exact BF16 operands, FP32 dot, post-dot scale",
        "probability_value": "FP32 softmax, high+residual 16-bit probability terms; exact storage widening",
        "partials": "FP32 unnormalised split output and denominator; one final BF16 rounding",
        "cache_format": "OCP E4M3/uint8 or BF16; FNUZ rejected",
        "packed_gdn": "existing packed inner state layout; non-overlapping outer slots",
        "native_sha256": metadata["files"]["native.so"],
        "shared_sha256": metadata["binary_sha256"],
    }
    perf_path, repair_path = rel(manifest["performance"]), rel(manifest["repair"])
    runtime = perf_path.parent / "runtime"
    for name in list(files):
        if Path(name).name in RUNTIME:
            copy(source / Path(name).name, Path(name))
    for name in RUNTIME:
        copy(source / name, runtime / name)
    repair = json.loads((output / repair_path).read_text())
    repair["attention_precision_parent"] = repair.pop("sha256")
    for name in repair["sources"]:
        if name in RUNTIME:
            repair["sources"][name] = digest(source / name)
    repair["attention_precision"] = entry
    reseal(repair)
    write(repair_path, repair)
    perf = json.loads((output / perf_path).read_text())
    perf["attention_precision_parent"] = perf.pop("sha256")
    perf["reference_repair"] = repair["sha256"]
    perf["stages"]["attention"] = {
        "build": entry["build"],
        "build_sha256": metadata["sha256"],
        "qualification": str(Path(entry["alignment"]).parent),
        "qualification_sha256": reports["alignment"]["sha256"],
    }
    for name in RUNTIME:
        perf["sources"][name] = digest(output / runtime / name)
    perf["attention_precision"] = entry
    perf["full_model_status"] = "NOT_RUN_FOR_NEW_ATTENTION_ARITHMETIC"
    reseal(perf)
    write(perf_path, perf)
    manifest.update(
        parent_manifest_sha256=digest(parent / "optimized-release.json"),
        host_root=str(output),
        files=dict(sorted(files.items())),
        repair_sha256=files[str(repair_path)],
        performance_sha256=files[str(perf_path)],
        attention_precision=entry,
    )
    (output / "optimized-release.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    optimized = profile["optimized_d7"]
    optimized.update(
        manifest_sha256=digest(output / "optimized-release.json"),
        attention_precision=entry,
    )
    optimized["arithmetic"].update(
        attention_precision=entry["contract"],
        repair_sha256=repair["sha256"],
        performance_sha256=perf["sha256"],
    )
    optimized["arithmetic_evidence_scope"] = (
        "Independent attention operator tests and graph M1/M8 samples; previous full-model 10K result is historical."
    )
    (output / "runtime-radiance-1.0.16.json").write_text(
        json.dumps(profile, indent=2) + "\n"
    )
    return {
        "manifest": str(output / "optimized-release.json"),
        "files": len(files),
        "status": "OPERATOR_QUALIFIED_NOT_DEPLOYED",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "parent",
        "build",
        "witnesses",
        "alignment",
        "oracle",
        "profile",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    print(
        json.dumps(
            prepare(
                a.parent,
                a.build,
                a.witnesses,
                a.alignment,
                a.oracle,
                a.profile,
                a.output,
            )
        )
    )
