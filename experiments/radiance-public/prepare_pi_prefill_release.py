"""Freeze a qualified prefill extension without copying private replay data."""

import argparse
import json
import shutil
from pathlib import Path

from pi_prefill_admission import digest
from prepare_pi_prefill_bundle import SOURCES

from qwen_r9700_lab.diagnostic_contract import authenticate, seal


def prepare(parent, job, bundle, output, comparison):
    manifest = json.loads((parent / "optimized-release.json").read_text())
    checked = json.loads(comparison.read_text())
    authenticate(checked)
    if (
        checked["decode"]["positions"] < 320
        or checked["decode"]["full_logits_exact"] != checked["decode"]["positions"]
        or checked["prefill"]["full_logits_exact"] is not True
    ):
        raise ValueError("release requires complete exact prefill and decode comparison")
    for name, expected in manifest["files"].items():
        p = Path(name)
        if p.is_absolute() or ".." in p.parts or (parent / p).is_symlink():
            raise ValueError("unsafe parent release artifact")
        if digest(parent / name) != expected:
            raise ValueError("parent release changed")
    job_relative = Path("preflight") / job.name
    output.mkdir(mode=0o700)
    files = dict(manifest["files"])

    def copy(origin, relative):
        if relative.is_absolute() or ".." in relative.parts or origin.is_symlink():
            raise ValueError("unsafe release dependency")
        target = output / relative
        target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        shutil.copy2(origin, target)
        files[str(relative)] = digest(target)

    for name in files.copy():
        copy(parent / name, Path(name))
    perf_relative = Path(manifest["performance"]).relative_to("/qualification")
    performance = json.loads((output / perf_relative).read_text())
    authenticate(performance)
    candidate = json.loads((bundle / "performance.json").read_text())
    authenticate(candidate)
    if candidate.get("activation_tiles") and (
        checked.get("candidate_performance_sha256") != candidate["sha256"]
        or checked.get("activation_tiles", {}).get("tiled", 0) <= 0
    ):
        raise ValueError("full-model evidence does not cover this tiled prefill implementation")
    runtime = perf_relative.parent / "runtime"
    for name in SOURCES:
        copy(bundle / "runtime" / name, runtime / name)
        performance["sources"][name] = files[str(runtime / name)]
    for key in ("tp1_fp8", "gdn_norm_quant", "prefill_scan", "activation_tiles"):
        if key not in candidate:
            continue
        entry = candidate[key]
        performance[key] = entry
        paths = [Path(entry["qualification"])]
        if key == "tp1_fp8":
            paths.extend(
                Path(entry["build"]) / n
                for n in ("build.json", "candidate.so", "stock_fp8_epilogue.hip")
            )
        for path in paths:
            relative = path.relative_to("/qualification")
            source = job / relative.relative_to(job_relative)
            copy(source, relative)
    evidence_path = job_relative / "evidence" / comparison.name
    copy(comparison, evidence_path)
    performance["performance_parent"] = performance.pop("sha256")
    performance["full_model_qualification"] = {
        "path": "/qualification/" + str(evidence_path),
        "sha256": checked["sha256"],
        "scope": "320 forced decode rows and prefill; full-vocabulary hashes; no universal proof",
    }
    (output / perf_relative).write_text(json.dumps(seal(performance), indent=2) + "\n")
    files[str(perf_relative)] = digest(output / perf_relative)
    manifest.update(
        host_root=str(output),
        files=dict(sorted(files.items())),
        performance_sha256=files[str(perf_relative)],
    )
    manifest["performance_qualification"] = performance["full_model_qualification"]
    target = output / "optimized-release.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(target), "sha256": digest(target)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("parent", "job", "bundle", "output", "comparison"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.parent, args.job, args.bundle, args.output, args.comparison)
