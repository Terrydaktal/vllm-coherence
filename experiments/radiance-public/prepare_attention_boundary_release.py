"""Freeze a sampled shared-attention repair without copying private token data."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import authenticate, seal


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_evidence(build, qualification, comparison, parent_digest):
    for document in (build, qualification, comparison):
        authenticate(document)
    if (
        build.get("status") != "BUILT_UNTESTED"
        or build.get("kernel_abi") != "qwen-stock-m1-shared-attention-v1"
        or qualification.get("status") != "SAMPLE_CHECKED"
        or qualification.get("build") != build["sha256"]
        or not qualification.get("graph_checks")
        or not qualification.get("negative_control_detected")
    ):
        raise ValueError(
            "attention build has no matching successful operator qualification"
        )
    checks = qualification.get("checks", [])
    covered = {(c["dtype"], c["prefix"]) for c in checks}
    required = {
        (dtype, base + offset)
        for dtype in ("torch.float8_e4m3fn", "torch.bfloat16", "torch.uint8")
        for base in (0, 1024, 60000, 200000)
        for offset in range(16)
    }
    required.update(
        (dtype, boundary + offset)
        for dtype in ("torch.float8_e4m3fn", "torch.bfloat16", "torch.uint8")
        for boundary in (512, 59904, 200192)
        for offset in range(-7, 1)
    )
    if not required <= covered or any(
        c.get("rows") != 8
        or c.get("candidate_mismatches") != 0
        or c.get("baseline_mismatches") != 0
        for c in checks
    ):
        raise ValueError("attention qualification is incomplete or differs from M1")
    cases = comparison.get("contexts", {})
    if (
        comparison.get("status") != "SAMPLE_CHECKED"
        or comparison.get("candidate_build") != build["sha256"]
        or comparison.get("parent_manifest_sha256") != parent_digest
        or set(cases) != {"0K", "60K", "200K"}
        or any(
            c.get("output_exact") is not True
            or c.get("generated_tokens", 0) < 1000
            or c.get("round_capture_complete") is not True
            for c in cases.values()
        )
    ):
        raise ValueError(
            "release requires matching natural Pi outputs and complete round captures"
        )


def prepare(parent, build_dir, qualification_dir, comparison_path, output):
    parent_path = parent / "optimized-release.json"
    manifest = json.loads(parent_path.read_text())
    build = json.loads((build_dir / "build.json").read_text())
    qualification = json.loads((qualification_dir / "result.json").read_text())
    comparison = json.loads(comparison_path.read_text())
    validate_evidence(build, qualification, comparison, digest(parent_path))
    if (
        digest(build_dir / "candidate.so") != build["binary_sha256"]
        or digest(build_dir / "r4d_attn_decode_h256_gqa6.hip") != build["source_sha256"]
    ):
        raise ValueError("qualified attention source or binary changed")
    for name, expected in manifest["files"].items():
        relative = Path(name)
        artifact = parent / relative
        if relative.is_absolute() or ".." in relative.parts or artifact.is_symlink():
            raise ValueError("unsafe parent artifact")
        if digest(artifact) != expected:
            raise ValueError("parent release changed")
    output.mkdir(mode=0o700)
    files = {}

    def copy(origin, relative):
        if relative.is_absolute() or ".." in relative.parts or origin.is_symlink():
            raise ValueError("unsafe attention dependency")
        target = output / relative
        target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        shutil.copy2(origin, target)
        files[str(relative)] = digest(target)

    for name in manifest["files"]:
        copy(parent / name, Path(name))
    base = Path("diagnostics/attention-page-boundary-v1")
    for name in (
        "build.json",
        "candidate.so",
        "r4d_attn_decode_h256_gqa6.hip",
        "r4d.h",
        "r4d_common.h",
        "r4d_dt16.h",
    ):
        copy(build_dir / name, base / "build" / name)
    copy(qualification_dir / "result.json", base / "operator" / "result.json")
    copy(comparison_path, base / "pi-comparison.json")
    perf_path = Path(manifest["performance"]).relative_to("/qualification")
    performance = json.loads((output / perf_path).read_text())
    authenticate(performance)
    performance["performance_parent"] = performance.pop("sha256")
    performance["stages"]["attention"] = {
        "build": "/qualification/" + str(base / "build"),
        "build_sha256": build["sha256"],
        "qualification": "/qualification/" + str(base / "operator"),
        "qualification_sha256": qualification["sha256"],
    }
    evidence = {
        "path": "/qualification/" + str(base / "pi-comparison.json"),
        "sha256": comparison["sha256"],
        "scope": "Exact sampled M1 attention outputs and natural Pi output agreement at "
        "0K/60K/200K; no full-vocabulary or universal equivalence claim.",
    }
    performance["attention_boundary_qualification"] = evidence
    (output / perf_path).write_text(json.dumps(seal(performance), indent=2) + "\n")
    files[str(perf_path)] = digest(output / perf_path)
    manifest.update(
        parent_manifest_sha256=digest(parent_path),
        host_root=str(output),
        files=dict(sorted(files.items())),
        performance_sha256=files[str(perf_path)],
        attention_boundary_qualification=evidence,
    )
    target = output / "optimized-release.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"manifest": str(target), "sha256": digest(target)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "build", "qualification", "comparison", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.parent,
                args.build,
                args.qualification,
                args.comparison,
                args.output,
            )
        )
    )
