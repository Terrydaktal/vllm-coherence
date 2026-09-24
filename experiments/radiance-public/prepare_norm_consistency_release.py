"""Freeze the row-invariant normalization repair with its native evidence."""

import argparse
import json
import shutil
from pathlib import Path

from pi_prefill_admission import gdn_evidence
from prepare_optimized_pi_release import digest, reseal
from tp1_lazy_backports import evidence

RUNTIME = (
    "stock_fp8_epilogue.hip",
    "stock_fp8_epilogue.py",
    "stock_gdn_norm_quant.py",
    "probe_stock_fp8_epilogue.py",
    "probe_stock_gdn_norm_quant.py",
    "tp1_lazy_backports.py",
    "pi_prefill_admission.py",
)


def prepare(parent, build, fp8, gdn, audit, contract, profile, output):
    source = Path(__file__).resolve().parent
    manifest = json.loads((parent / "optimized-release.json").read_text())
    profile = json.loads(profile.read_text())
    if profile["optimized_d7"]["manifest_sha256"] != digest(
        parent / "optimized-release.json"
    ):
        raise ValueError("parent profile and payload differ")
    fp8_entry = {
        "build": str(build),
        "qualification": str(fp8),
        "qualification_sha256": digest(fp8),
        "prefill_enabled": True,
        "silu_enabled": False,
        "row_invariant": True,
    }
    gdn_entry = {
        "qualification": str(gdn),
        "qualification_sha256": digest(gdn),
        "row_invariant": True,
        "sources": {
            name: digest(source / name)
            for name in ("stock_gdn_norm_quant.py", "probe_stock_gdn_norm_quant.py")
        },
    }
    evidence(fp8_entry, kind="fp8")
    gdn_evidence(gdn_entry)
    checked = json.loads(audit.read_text())
    metadata = json.loads((build / "build.json").read_text())
    if (
        checked.get("status") != "SAMPLE_CHECKED"
        or checked.get("failed_checks") != 0
        or checked.get("arithmetic_contract_sha256") != digest(contract)
        or checked["sources"]["candidate_norm_quant"]["sha256"]
        != metadata["binary_sha256"]
        or checked["sources"]["candidate_gdn_norm_quant"]["sha256"]
        != digest(source / "stock_gdn_norm_quant.py")
    ):
        raise ValueError("independent contract audit does not cover the candidate")
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
        if origin.is_symlink() or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe release dependency")
        target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        shutil.copy2(origin, target)
        target.chmod(0o600)
        files[str(relative)] = digest(target)

    def write(relative, value):
        target = output / relative
        target.write_text(json.dumps(value, indent=2) + "\n")
        target.chmod(0o600)
        files[str(relative)] = digest(target)

    def native(relative):
        return "/qualification/" + str(relative)

    for name in manifest["files"]:
        copy(parent / name, Path(name))
    base = Path("diagnostics/norm-consistency-v2")
    for name in ("build.json", "candidate.so", "stock_fp8_epilogue.hip"):
        copy(build / name, base / "build" / name)
    for label, path in (
        ("fp8", fp8),
        ("gdn", gdn),
        ("audit", audit),
        ("contract", contract),
    ):
        copy(path, base / (label + ".json"))
    fp8_entry.update(
        build=native(base / "build"), qualification=native(base / "fp8.json")
    )
    gdn_entry["qualification"] = native(base / "gdn.json")
    perf_path = Path(manifest["performance"]).relative_to("/qualification")
    repair_path = Path(manifest["repair"]).relative_to("/qualification")
    runtime = perf_path.parent / "runtime"
    for name in list(files):
        if Path(name).name in RUNTIME:
            copy(source / Path(name).name, Path(name))
    for name in RUNTIME:
        copy(source / name, runtime / name)
    entry = {
        "qualification": native(base / "audit.json"),
        "qualification_sha256": digest(audit),
        "contract": {
            "profile": native(base / "contract.json"),
            "profile_sha256": digest(contract),
            "hidden_norm": "M1 512-lane reduction for every admitted row count",
            "gdn_norm": "M1 four-adjacent-components per lane; final BF16 boundary",
            "norm_binary_sha256": metadata["binary_sha256"],
            "gdn_source_sha256": digest(source / "stock_gdn_norm_quant.py"),
        },
    }
    repair = json.loads((output / repair_path).read_text())
    repair["normalization_parent"] = repair.pop("sha256")
    for name in repair["sources"]:
        if name in RUNTIME:
            repair["sources"][name] = digest(source / name)
    repair["normalization_consistency"] = entry
    write(repair_path, reseal(repair))
    perf = json.loads((output / perf_path).read_text())
    perf["normalization_parent"] = perf.pop("sha256")
    perf["reference_repair"] = repair["sha256"]
    perf.update(
        tp1_fp8=fp8_entry, gdn_norm_quant=gdn_entry, normalization_consistency=entry
    )
    for name in RUNTIME:
        perf["sources"][name] = digest(source / name)
    perf["full_model_status"] = "NOT_RUN_AFTER_NORMALIZATION_REPAIR"
    write(perf_path, reseal(perf))
    manifest.update(
        parent_manifest_sha256=digest(parent / "optimized-release.json"),
        host_root=str(output),
        files=dict(sorted(files.items())),
        repair_sha256=files[str(repair_path)],
        performance_sha256=files[str(perf_path)],
        normalization_consistency=entry,
    )
    (output / "optimized-release.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    optimized = profile["optimized_d7"]
    optimized.update(
        manifest_sha256=digest(output / "optimized-release.json"),
        host_root=str(output),
        normalization_consistency=entry,
    )
    optimized["arithmetic"].update(
        normalization_consistency=entry["contract"],
        repair_sha256=repair["sha256"],
        performance_sha256=perf["sha256"],
    )
    optimized["arithmetic_evidence_scope"] = (
        "Native M1/prefill norm consistency and independent operator checks; older full-model alignment is historical."
    )
    (output / "runtime-radiance-1.0.16.json").write_text(
        json.dumps(profile, indent=2) + "\n"
    )
    return {"manifest": str(output / "optimized-release.json"), "files": len(files)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "parent",
        "build",
        "fp8",
        "gdn",
        "audit",
        "contract",
        "profile",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(prepare(**vars(parser.parse_args()))))
