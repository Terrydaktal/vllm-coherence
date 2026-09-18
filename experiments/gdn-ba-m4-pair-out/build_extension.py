#!/usr/bin/env python3
"""Build the stopped-GPU GDN B/A M4-pair out-parameter experiment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    build = args.build_directory.resolve()
    build.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1201")

    from torch.utils.cpp_extension import load

    module = load(
        name="qwen_gdn_ba_m4_pair_out_gfx1201_v2",
        sources=[
            str(root / "gdn_ba_m4_pair_ext.cpp"),
            str(root / "gdn_ba_m4_pair_kernel.cu"),
        ],
        build_directory=str(build),
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        with_cuda=True,
        verbose=True,
    )
    print(Path(module.__file__).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
