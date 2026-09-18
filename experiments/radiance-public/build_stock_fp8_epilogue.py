"""Build the corrected TP1 FP8 epilogues without touching the GPU."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(output):
    output.mkdir(mode=0o700)
    source = Path(__file__).with_name("stock_fp8_epilogue.hip")
    local = output / source.name
    local.write_bytes(source.read_bytes())
    cmd = [
        "/opt/rocm/bin/hipcc",
        "-O3",
        "-std=c++17",
        "--offload-arch=gfx1201",
        "-shared",
        "-fPIC",
        "-Wall",
        "-Wextra",
        "-ffp-contract=off",
        "-mcumode",
        "-cuid=stock_fp8_" + digest(local),
        str(local),
        "-o",
        str(output / "candidate.so"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    (output / "compiler.log").write_text(result.stdout + result.stderr)
    report = {
        "status": "BUILT_UNTESTED" if result.returncode == 0 else "BUILD_FAILED",
        "source_sha256": digest(source),
        "command": cmd,
        "gpu_used": False,
        "binary_sha256": digest(output / "candidate.so") if result.returncode == 0 else None,
        "abi": "qwen-stock-fp8-epilogue-v1",
    }
    (output / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(build(parser.parse_args().output))
