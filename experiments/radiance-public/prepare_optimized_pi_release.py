"""Freeze the tested existing-layout D7 bundle for production, without chat data."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reseal(document):
    document.pop("sha256", None)
    canonical = json.dumps(document, allow_nan=False, sort_keys=True, separators=(",", ":"))
    document["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return document


def prepare(source, output, attention_source=None, target_head="full-bf16"):
    source, output = source.resolve(), output.absolute()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    files = {}

    def copy(relative):
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("release dependency escapes its root")
        origin, destination = source / relative, output / relative
        if not origin.is_file() or origin.is_symlink():
            raise ValueError(f"release dependency is missing or indirect: {relative}")
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copy2(origin, destination)
        files[str(relative)] = digest(destination)

    def relative(path):
        return Path(path).relative_to("/qualification")

    def source_tree(root):
        base = source / root
        paths = list(base.glob("*.py"))
        package = base / "qwen_r9700_lab"
        if package.is_dir():
            paths.extend(package.rglob("*.py"))
        for path in sorted(paths):
            if "__pycache__" not in path.parts:
                copy(path.relative_to(source))

    def build(root):
        root = relative(root)
        # Build directories contain source, binaries and manifests; never captures.
        for path in sorted((source / root).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".hip", ".h", ".so", ".json"}:
                copy(path.relative_to(source))

    bundle = Path("preflight/tp1-lazy-backport-v1/bundle-existing-v1")
    repair_path = Path("preflight/d7-stock-repair-bundle-003.json")
    repair = json.loads((source / repair_path).read_text())
    performance = json.loads((source / bundle / "performance.json").read_text())
    launch = json.loads((source / bundle / "launch.json").read_text())
    if performance.get("lazy_gdn") or launch["environment"].get("QWEN_STOCK_GDN_LAZY") != "0":
        raise ValueError("production release must keep the existing state layout")
    trees = [
        str(bundle / "runtime"),
        "preflight/d7-rotary-intervention-v1/src",
        "preflight/d7-rotary-intervention-v1/experiments/radiance-public",
        "preflight/stock-gdn-model-v22/runtime",
        "preflight/stock-gdn-model-v22/experiments/radiance-public",
    ]
    for root in trees:
        source_tree(root)
    # The FP8 source is authenticated by the runtime evidence gate.
    copy(bundle / "runtime/stock_fp8_epilogue.hip")
    for name in ("performance.json", "launch.json"):
        copy(bundle / name)
    copy(repair_path)
    copy(relative(repair["convolution"]))
    build(repair["norm_build"])
    for stage in performance["stages"].values():
        build(stage["build"])
        copy(relative(stage["qualification"]) / "result.json")
    gemm = performance["gemm_dispatch"]
    build(gemm["build"])
    for evidence in gemm["qualifications"].values():
        copy(relative(evidence["path"]))
    fp8 = performance["tp1_fp8"]
    build(fp8["build"])
    copy(relative(fp8["qualification"]))
    admission_changes = []
    if attention_source is not None:
        target = bundle / "runtime/stock_m1_attention.py"
        old_hash = files[str(target)]
        shutil.copy2(attention_source, output / target)
        files[str(target)] = digest(output / target)
        change = {
            "file": str(target),
            "old_sha256": old_hash,
            "new_sha256": files[str(target)],
            "scope": (
                "Admit graph-padded buffers and slice their live query/output prefix; "
                "unchanged native attention arithmetic and existing state layout. "
                "Earlier numerical evidence covers the parent implementation."
            ),
        }
        admission_changes.append(change)
        repair["admission_parent"] = repair["sha256"]
        repair["admission_changes"] = [change]
        repair["sources"]["stock_m1_attention.py"] = files[str(target)]
        reseal(repair)
        performance["admission_parent"] = performance["sha256"]
        performance["reference_repair"] = repair["sha256"]
        performance["admission_changes"] = [change]
        reseal(performance)
        for path, document in (
            (repair_path, repair),
            (bundle / "performance.json", performance),
        ):
            (output / path).write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
            files[str(path)] = digest(output / path)
    if target_head not in ("full-bf16", "global256"):
        raise ValueError("unsupported target head")
    head_env = {"RADIANCE_VERIFY_HEAD": "0"}
    if target_head == "global256":
        head_env.update(RADIANCE_VERIFY_HEAD="1", RADIANCE_VERIFY_HEAD_GLOBAL_TOPK="256")
    report = {
        "schema": "urn:qwen:optimized-pi-release:v1",
        "host_root": str(output),
        "container_root": "/qualification",
        "state_layout": "existing-nine-slot",
        "target_head": target_head,
        "repair": "/qualification/" + str(repair_path),
        "repair_sha256": files[str(repair_path)],
        "performance": "/qualification/" + str(bundle / "performance.json"),
        "performance_sha256": files[str(bundle / "performance.json")],
        "pythonpath": ["/qualification/" + root for root in trees],
        "environment": {
            **launch["environment"],
            "QWEN_OPTIMIZED_REPAIR": "/qualification/" + str(repair_path),
            **head_env,
        },
        "admission_changes": admission_changes,
        "files": dict(sorted(files.items())),
    }
    manifest = output / "optimized-release.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest), "sha256": digest(manifest), "files": len(files)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention-source", type=Path)
    parser.add_argument("--target-head", choices=("full-bf16", "global256"), default="full-bf16")
    args = parser.parse_args()
    prepare(args.source, args.output, args.attention_source, args.target_head)
