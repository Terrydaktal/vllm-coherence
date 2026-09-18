"""Create a separate corrected-runtime bundle using the checked GEMM backport."""

import argparse
import shutil
from pathlib import Path

from build_mxfp4_dispatch import digest
from mxfp4_dispatch import validate

from qwen_r9700_lab.conformance_topk import require
from qwen_r9700_lab.diagnostic_contract import authenticate, private_json, seal, write_private


def prepare(args):
    manifest = private_json(args.performance)
    authenticate(manifest)
    for name, expected in manifest["sources"].items():
        require(digest(args.runtime_sources / name) == expected, "parent runtime source changed")
    entry = {
        "build": str(args.build),
        "build_sha256": digest(args.build / "build.json"),
        "qualifications": {},
    }
    for name, path in (("automatic", args.automatic), ("split4", args.split4)):
        entry["qualifications"][name] = {"path": str(path), "sha256": digest(path)}
    validate(entry)
    args.output.mkdir(mode=0o700)
    runtime = args.output / "runtime"
    runtime.mkdir()
    for file in args.runtime_sources.glob("*.py"):
        shutil.copy2(file, runtime / file.name)
    for name in (
        "optimized_d7_performance.py",
        "mxfp4_dispatch.py",
        "build_mxfp4_dispatch.py",
        "probe_mxfp4_dispatch.py",
    ):
        shutil.copy2(Path(__file__).with_name(name), runtime / name)
        manifest["sources"][name] = digest(runtime / name)
    manifest["gemm_dispatch"] = entry
    manifest["performance_parent"] = manifest.pop("sha256")
    manifest["gemm_scope"] = "Sample-checked GEMM dispatch only; earlier repair stages unchanged"
    manifest = seal(manifest)
    write_private(args.output / "performance.json", manifest)
    print(str(args.output / "performance.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("performance", "runtime-sources", "build", "automatic", "split4", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    prepare(parser.parse_args())
