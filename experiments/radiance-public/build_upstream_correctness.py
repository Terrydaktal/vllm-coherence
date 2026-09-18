#!/usr/bin/env python3
"""Build an isolated native candidate after applying the pinned source bundle.

Run in a CPU-only instance of the pinned build image. The source tree must be a
disposable copy, not a live runtime. No built artifact is installed into serving.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from patch_upstream_correctness import BUNDLE, install, sha256

RELEASE = Path(__file__).resolve().parent


def build(
    source: Path, output: Path, component: str, *, jobs: int = 4, cpu_tests: bool = False
) -> dict:
    source = source.resolve(strict=True)
    output = output.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("build output must be outside the pinned source tree")
    if not 1 <= jobs <= 16:
        raise ValueError("jobs must be between 1 and 16")
    output.mkdir(parents=True, exist_ok=False)
    source_receipt = install(source, component)
    (output / "source-receipt.json").write_text(json.dumps(source_receipt, indent=2) + "\n")
    env = dict(os.environ)
    env.update(CMAKE_BUILD_PARALLEL_LEVEL=str(jobs), MAX_JOBS=str(jobs))
    commands: list[tuple[list[str], Path]] = []
    if component == "xgrammar" and cpu_tests:
        build_dir = output / "cmake"
        build_dir.mkdir()
        # The pinned project's config.cmake overrides command-line -D defaults.
        (build_dir / "config.cmake").write_text(
            "set(CMAKE_BUILD_TYPE Release)\n"
            "set(XGRAMMAR_BUILD_PYTHON_BINDINGS OFF)\n"
            "set(XGRAMMAR_BUILD_CXX_TESTS ON)\n"
            "set(XGRAMMAR_ENABLE_CPPTRACE OFF)\n"
            "set(XGRAMMAR_ENABLE_INTERNAL_CHECK ON)\n"
        )
        test = source / "tests/cpp/test_qwen_backports.cc"
        if test.exists() or test.is_symlink():
            raise ValueError("candidate regression target already exists")
        shutil.copyfile(BUNDLE / "tests/xgrammar_regressions.cc", test)
        commands = [
            (["cmake", "-S", str(source), "-B", str(build_dir), "-G", "Ninja"], source),
            (["cmake", "--build", str(build_dir), "--parallel", str(jobs)], source),
            (
                ["ctest", "--test-dir", str(build_dir), "--output-on-failure", "-j", str(jobs)],
                source,
            ),
        ]
    elif component in {"triton", "xgrammar"}:
        project = (
            source / "python"
            if component == "triton" and (source / "python/pyproject.toml").is_file()
            else source
        )
        if not (project / "pyproject.toml").is_file():
            raise ValueError("pinned native Python build project is missing")
        if component == "triton":
            env["TRITON_HOME"] = str(output / "triton-cache")
            if not (source / ".git").exists():
                # Preserve the base pin's package identity for archive builds;
                # the changed binary is identified separately by its SHA-256.
                env["TRITON_WHEEL_VERSION_SUFFIX"] = "+gitf0b55c07"
        # uv uses the pinned project's build requirements; serving packages are
        # untouched. Keep downloaded compiler dependencies isolated in output.
        commands = [
            (
                [
                    "uv",
                    "build",
                    "--wheel",
                    "--python",
                    sys.executable,
                    "--build-constraint",
                    str(BUNDLE / "build-constraints.txt"),
                    "--out-dir",
                    str(output / "dist"),
                    str(project),
                ],
                source,
            )
        ]
    elif component == "rocr":
        if cpu_tests:
            raise ValueError("ROCr recipe already runs its CPU backoff regression")
        runtime = source / "projects/rocr-runtime"
        # This recipe verifies both existing poll-backoff source hashes before
        # compiling. The registration repair must never remove the idle-CPU fix.
        commands = [
            (
                [
                    "bash",
                    str(RELEASE / "rocr-poll-backoff/build.sh"),
                    str(runtime),
                    str(output / "cmake"),
                ],
                source,
            )
        ]
    else:
        raise ValueError("unknown native component")
    receipt = {
        "schema": "urn:qwen:native-backport-build:v1",
        "component": component,
        "manifest_sha256": source_receipt["manifest_sha256"],
        "source_receipt_sha256": sha256((output / "source-receipt.json").read_bytes()),
        "commands": [cmd for cmd, _ in commands],
        "python": sys.version,
        "python_executable": sys.executable,
        "gpu_executed": False,
        "gpu_qualification": "NOT_RUN",
        "status": "BUILDING",
    }
    receipt_path = output / "build-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        with (output / "build.log").open("wb") as log:
            for cmd, cwd in commands:
                subprocess.run(
                    cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
                )
        artifacts = {}
        for path in output.rglob("*"):
            if (
                path.is_file()
                and not path.is_symlink()
                and (path.suffix in {".whl", ".a"} or ".so" in path.name)
            ):
                with path.open("rb") as stream:
                    artifacts[str(path.relative_to(output))] = hashlib.file_digest(
                        stream, "sha256"
                    ).hexdigest()
        if not artifacts:
            raise ValueError("native build produced no artifact")
        receipt.update(status="CPU_TESTED" if cpu_tests else "BUILT", artifacts=artifacts)
    except BaseException as error:
        receipt.update(status="FAILED", error_type=type(error).__name__)
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("triton", "rocr", "xgrammar"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cpu-tests", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.source, args.output, args.component, jobs=args.jobs, cpu_tests=args.cpu_tests
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
