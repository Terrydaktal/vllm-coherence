"""Backport GGZ14's TP1 decode-width dispatch, with an unchanged-source control.

The input is Radiance 1.0.16's pinned source, not GGZ14's newer kernel. No GPU
is used by this builder. Installation is deliberately separate from building.
"""

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

HIP_SHA256 = "8d334af24f198d8481fd5e3df3c01eb881c1ec67042913b774c1c289fc8e50cd"
PYTHON_SHA256 = "7fd10b2d5b6a6c3853583eb71ca10dfcaf6784694c7941bd1e28968addcb88f2"
ORIGINAL_BINARY_SHA256 = "8c895e15556970efeb325b3930807dbb41d9149058e29c9d9cbe497ecd7dcd51"
UPSTREAM_COMMIT = "3f542b7cbfce3fa0d01dc55665af4c77a7093ce8"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patched_sources(hip, python):
    for text, expected in ((hip, HIP_SHA256), (python, PYTHON_SHA256)):
        if hashlib.sha256(text.encode()).hexdigest() != expected:
            raise ValueError("pinned Radiance 1.0.16 source changed")
    hip = hip.replace(
        "#define DEC_MAX_N 32768",
        "// TP1 Qwen gate_up has N=34816; retain the existing decode arithmetic.\n"
        "#define DEC_MAX_N 36864",
    ).replace("N(<=32768) x 4 B = 32 MiB", "N(<=36864) x 4 B = 36 MiB")
    # The automatic policy uses one split at these widths. A forced wider split
    # changes FP32 accumulation order relative to the old folded fallback.
    # Keep that existing fallback instead of broadening the numerical contract.
    hip = hip.replace(
        "N <= DEC_MAX_N &&\n        (dks == 1 || have_scratch)",
        "N <= DEC_MAX_N &&\n        (N <= 32768 || dks == 1) &&\n"
        "        (dks == 1 || have_scratch)",
    )
    python = python.replace(
        "                    _decode_scratch[0] = torch.empty(\n",
        "                    # Match the widened native dispatch before graph capture.\n"
        "                    from vllm.distributed import get_tensor_model_parallel_world_size\n"
        "                    _dec_max_n = (36864 if "
        "get_tensor_model_parallel_world_size() == 1 else 32768)\n"
        "                    _decode_scratch[0] = torch.empty(\n",
    ).replace("4 * max(64, DECODE_MAX_M) * 32768", "4 * max(64, DECODE_MAX_M) * _dec_max_n")
    python = python.replace("32768 // 128 + 8", "_dec_max_n // 128 + 8")
    python = python.replace(
        "# from the env so the default (64) allocates exactly what it always has;\n"
        "                    # a 16-concurrent serve sets "
        "RADIANCE_MXFP4_DECODE_MAX_M=128 and pays the\n"
        "                    # extra 32 MiB only then.",
        "# from the decode band: TP1 uses 36 MiB at 64 rows; TP>1 retains 32 MiB.\n"
        "                    # A 128-row decode band doubles the corresponding allocation.",
    )
    return hip, python


def build(source, output):
    hip = (source / "radiance_mxfp4_fp8.hip").read_text()
    python = (source / "radiance_mxfp4.py").read_text()
    patched_hip, patched_python = patched_sources(hip, python)
    output.mkdir(mode=0o700)
    includes = shlex.split(
        subprocess.check_output([sys.executable, "-m", "pybind11", "--includes"], text=True)
    )
    compiler = "/opt/rocm/bin/hipcc"
    report = {
        "status": "BUILT_UNTESTED",
        "upstream_commit": UPSTREAM_COMMIT,
        "original_binary_sha256": ORIGINAL_BINARY_SHA256,
        "variants": {},
        "compiler": subprocess.check_output([compiler, "--version"], text=True),
    }
    for name, native, wrapper in (
        ("control", hip, python),
        ("candidate", patched_hip, patched_python),
    ):
        folder = output / name
        folder.mkdir()
        local = folder / "radiance_mxfp4_fp8.hip"
        local.write_text(native)
        (folder / "radiance_mxfp4.py").write_text(wrapper)
        binary = folder / "radiance_mxfp4_fp8.so"
        command = [
            compiler,
            "-O3",
            "-std=c++17",
            "-fPIC",
            "-shared",
            "--offload-arch=gfx1201",
            "-Wno-unused-result",
            *includes,
            str(local),
            "-o",
            str(binary),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        (folder / "compiler.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
        report["variants"][name] = {
            "command": command,
            "source_sha256": digest(local),
            "binary_sha256": digest(binary),
            "python_sha256": digest(folder / "radiance_mxfp4.py"),
        }
        print(json.dumps({"built": name, **report["variants"][name]}), flush=True)
    (output / "build.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build(args.source, args.output)
