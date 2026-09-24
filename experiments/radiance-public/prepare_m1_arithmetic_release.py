"""Freeze tested M1 repairs into a separate payload; never activate a server.

Only manifest-listed parent artifacts are copied. The output profile carries a
new numerical identity, so snapshot binding regeneration cannot reuse old KV.
Prior full-model results remain historical, not evidence for the new arithmetic.
"""

import argparse
import json
import shutil
from pathlib import Path

from m1_arithmetic_release import validate
from mxfp4_fold_precision import digest
from prepare_optimized_pi_release import reseal


def prepare(parent, build, qualification, activation, output, profile):
    source = Path(__file__).resolve().parent
    original_manifest = parent / "optimized-release.json"
    manifest = json.loads(original_manifest.read_text())
    profile = json.loads(profile.read_text())
    if profile["optimized_d7"]["manifest_sha256"] != digest(original_manifest):
        raise ValueError("parent payload and profile differ")
    entry = {
        "build": str(build),
        "build_sha256": digest(build / "build.json"),
        "qualification": str(qualification),
        "qualification_sha256": digest(qualification),
    }
    metadata, report = validate(entry)
    candidate = metadata["variants"]["candidate"]
    from prefill_tiles_admission import evidence

    activation_entry = {
        "qualification": str(activation),
        "qualification_sha256": digest(activation),
        "binary_sha256": candidate["binary_sha256"],
        "wrapper_sha256": candidate["python_sha256"],
        "sources": {
            name: digest(source / name)
            for name in (
                "prefill_activation_tiles.py",
                "probe_prefill_activation_tiles.py",
            )
        },
    }
    evidence(activation_entry)
    for name, expected in manifest["files"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or (parent / path).is_symlink():
            raise ValueError("unsafe parent artifact")
        if digest(parent / path) != expected:
            raise ValueError("parent artifact changed")
    output.mkdir(mode=0o700)
    files = {}

    def copy(origin, relative):
        target = output / relative
        target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        if origin.is_symlink():
            raise ValueError("indirect M1 artifact")
        shutil.copy2(origin, target)
        files[str(relative)] = digest(target)

    def write(relative, data):
        target = output / relative
        target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        target.write_text(json.dumps(data, indent=2) + "\n")
        files[str(relative)] = digest(target)

    def relative(path):
        return Path(path).relative_to("/qualification")

    base = Path("diagnostics/m1-arithmetic-20260924")
    for name in manifest["files"]:
        copy(parent / name, Path(name))
    copy(build / "build.json", base / "build/build.json")
    for variant in metadata["variants"]:
        for name in (
            "radiance_mxfp4.py",
            "radiance_mxfp4_fp8.hip",
            "radiance_mxfp4_fp8.so",
        ):
            copy(build / variant / name, base / "build" / variant / name)
    copy(qualification, base / "operators.json")
    copy(activation, base / "activation.json")
    entry.update(
        build="/qualification/" + str(base / "build"),
        qualification="/qualification/" + str(base / "operators.json"),
    )
    entry["contract"] = {
        "gdn_softplus": "FP32 log1p(exp(x)); same M1/M8/prefill recurrence",
        "mxfp4_fold": "exact E4M3 exponent differences -6..8; wider spans per-block",
        "sources": report["source_sha256"],
        "gemm_binary": candidate["binary_sha256"],
    }
    runtime = relative(manifest["performance"]).parent / "runtime"
    replace = (
        "stock_gdn_scan_kernel.py",
        "stock_gdn_lazy_kernel.py",
        "mxfp4_dispatch.py",
        "prefill_tiles_admission.py",
        "prefill_activation_tiles.py",
        "probe_prefill_activation_tiles.py",
    )
    # Replace every importable copy, preventing a later PYTHONPATH entry from
    # reintroducing the old scan. Existing numerical operators remain untouched.
    for name in list(files):
        if Path(name).name in replace:
            copy(source / Path(name).name, Path(name))
    for name in (
        *replace,
        "m1_arithmetic_release.py",
        "mxfp4_fold_precision.py",
        "patch_gdn_stable_softplus.py",
        "probe_m1_arithmetic_repairs.py",
        "probe_eager_m1_independent.py",
    ):
        copy(source / name, runtime / name)
    repair_path, perf_path = (
        relative(manifest["repair"]),
        relative(manifest["performance"]),
    )
    repair = json.loads((output / repair_path).read_text())
    performance = json.loads((output / perf_path).read_text())
    repair["arithmetic_parent"] = repair.pop("sha256")
    repair["sources"]["stock_gdn_scan_kernel.py"] = report["source_sha256"][
        "stock_gdn_scan_kernel.py"
    ]
    repair["m1_arithmetic"] = entry
    repair["scope"] = (
        "Stable GDN gates; operator samples recorded separately from historical full-model evidence."
    )
    if "prerequisite_model_check" in repair:
        repair["historical_prerequisite_model_check"] = repair.pop(
            "prerequisite_model_check"
        )
    reseal(repair)
    write(repair_path, repair)
    performance["arithmetic_parent"] = performance.pop("sha256")
    performance["reference_repair"] = repair["sha256"]
    for name in tuple(performance["sources"]):
        performance["sources"][name] = digest(output / runtime / name)
    for name in (
        "m1_arithmetic_release.py",
        "mxfp4_fold_precision.py",
        "patch_gdn_stable_softplus.py",
    ):
        performance["sources"][name] = digest(output / runtime / name)
    performance["gemm_dispatch"]["m1_arithmetic"] = entry
    performance["gemm_scope"] = (
        "Existing width dispatch plus independently checked M1 coefficient repair."
    )
    activation_entry["qualification"] = "/qualification/" + str(
        base / "activation.json"
    )
    performance["activation_tiles"] = activation_entry
    scan_sources = {
        name: digest(source / name)
        for name in (
            "optimized_prefill_scan.py",
            "stock_gdn_scan_kernel.py",
            "probe_m1_arithmetic_repairs.py",
        )
    }
    scan_report = {
        "status": "SAMPLE_CHECKED",
        "sources": scan_sources,
        "negative_control_detected": report["checks"]["gdn"][
            "negative_control_detected"
        ],
        "checks": report["checks"]["gdn"]["prefill_checks"],
        "operator_report_sha256": entry["qualification_sha256"],
        "scope": "Public parameters and synthetic activations; prefill versus serial corrected M1.",
    }
    write(base / "scan.json", scan_report)
    performance["prefill_scan"] = {
        "qualification": "/qualification/" + str(base / "scan.json"),
        "qualification_sha256": files[str(base / "scan.json")],
        "sources": scan_sources,
    }
    if "full_model_qualification" in performance:
        performance["historical_full_model_qualification"] = performance.pop(
            "full_model_qualification"
        )
    performance["full_model_status"] = "NOT_RUN_FOR_NEW_ARITHMETIC"
    reseal(performance)
    write(perf_path, performance)
    if "performance_qualification" in manifest:
        manifest["historical_performance_qualification"] = manifest.pop(
            "performance_qualification"
        )
    manifest.update(
        parent_manifest_sha256=digest(original_manifest),
        host_root=str(output),
        files=dict(sorted(files.items())),
        m1_arithmetic=entry,
        repair_sha256=files[str(repair_path)],
        performance_sha256=files[str(perf_path)],
    )
    # Do not admit the uncorrected libr4d GEMM with shifted references.
    manifest["environment"]["RADIANCE_MXFP4_R4D_DECODE_MAX_M"] = "0"
    path = output / "optimized-release.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    optimized = profile["optimized_d7"]
    optimized.update(manifest_sha256=digest(path), m1_arithmetic=entry)
    optimized["arithmetic"].update(
        m1_arithmetic=entry["contract"],
        repair_sha256=repair["sha256"],
        performance_sha256=performance["sha256"],
    )
    optimized["arithmetic_evidence_scope"] = (
        "Independent M1 operator checks; historical full-model results do not qualify the new arithmetic."
    )
    if "performance_qualification" in optimized:
        optimized["historical_performance_qualification"] = optimized.pop(
            "performance_qualification"
        )
    (output / "runtime-radiance-1.0.16.json").write_text(
        json.dumps(profile, indent=2) + "\n"
    )
    return {
        "manifest": str(path),
        "sha256": digest(path),
        "status": "OPERATOR_QUALIFIED_CANDIDATE_NOT_DEPLOYED",
        "files": len(files),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "build", "qualification", "activation", "output", "profile"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.parent,
                args.build,
                args.qualification,
                args.activation,
                args.output,
                args.profile,
            )
        )
    )
