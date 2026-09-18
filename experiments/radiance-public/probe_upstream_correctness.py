#!/usr/bin/env python3
"""Run focused native backport regressions under the campaign's GPU lease.

No GPU package is imported by this controller. The isolated child runs synthetic
compiler/convolution probes; skipped tests never count as qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import sysconfig
import xml.etree.ElementTree as ET
from pathlib import Path

from patch_upstream_correctness import BUNDLE

from qwen_r9700_lab.conformance_gpu_lease import gpu_lease
from qwen_r9700_lab.conformance_transport import OwnedProcess

PROBES = {
    "triton_guarded_loop": ("triton_guarded_loop.py", 2),
    "short_prefill_convolution": ("short_prefill_convolution.py", 2),
}


def junit_status(path: Path, expected: int) -> dict:
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    errors = sum(bool(list(c.iter("error"))) for c in cases)
    failures = sum(bool(list(c.iter("failure"))) for c in cases)
    skipped = sum(bool(list(c.iter("skipped"))) for c in cases)
    return {
        "status": "TESTED"
        if len(cases) == expected and not (errors or failures or skipped)
        else "FAILED",
        "expected": expected,
        "observed": len(cases),
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
    }


def verify_probe_binding(binding: dict, selected: str) -> None:
    files = binding.get("files")
    package = Path(sysconfig.get_path("purelib"))
    required = {str(package / "triton/_C/libtriton.so")}
    if selected == "short_prefill_convolution":
        required.add(str(package / "vllm/model_executor/layers/mamba/ops/causal_conv1d.py"))
    if not isinstance(files, dict) or not required <= files.keys():
        raise ValueError("binding must include the compiler and sources used by this probe")
    for name, expected in files.items():
        path = Path(name)
        if not path.is_absolute():
            raise ValueError("runtime binding paths must be absolute")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"runtime binding differs: {path.name}")


def run(root: Path, binding_path: Path, selected: str, timeout: float) -> dict:
    if not os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"):
        raise ValueError("use the same explicit GPU lock as the running campaign")
    binding = json.loads(binding_path.read_text())
    verify_probe_binding(binding, selected)
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    filename, count = PROBES[selected]
    probe = BUNDLE / "tests" / filename
    report = {
        "schema": "urn:qwen:upstream-native-probe:v1",
        "probe": selected,
        "binding_sha256": hashlib.sha256(binding_path.read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((BUNDLE / "manifest.json").read_bytes()).hexdigest(),
        "proof": "UNPROVED",
        "gpu_executed": False,
        "status": "NOT_RUN",
    }
    try:
        with gpu_lease(root / "lease"):
            env = dict(os.environ)
            env["QWEN_UPSTREAM_GPU_TESTS"] = "1"
            # No production cache or real conversations enter these probes.
            env["TRITON_CACHE_DIR"] = str(root / "triton-cache")
            child = OwnedProcess(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(probe),
                    "-q",
                    "--override-ini=addopts=",
                    f"--junitxml={root / 'tests.xml'}",
                ],
                root / "process",
                env=env,
                timeout=timeout,
            )
            report["gpu_executed"] = None
            code = child.wait()
            report.update(junit_status(root / "tests.xml", count))
            report["exit_code"] = code
            report["gpu_executed"] = True if report["status"] == "TESTED" else None
            if code:
                report["status"] = "FAILED"
    except Exception as error:
        report.update(
            status="TIMEOUT" if isinstance(error, TimeoutError) else "ERROR",
            error_type=type(error).__name__,
        )
    finally:
        (root / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", choices=PROBES, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if not args.gpu:
        parser.error("GPU execution was not explicitly enabled")
    report = run(args.output.resolve(), args.binding.resolve(), args.probe, args.timeout)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "TESTED" else 2)


if __name__ == "__main__":
    main()
