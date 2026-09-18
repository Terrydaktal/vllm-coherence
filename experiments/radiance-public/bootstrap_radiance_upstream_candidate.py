#!/usr/bin/env python3
"""Fresh-container candidate entrypoint; never patches a live serving process.

Native build bindings and a separate snapshot namespace are mandatory. This
entrypoint is deliberately separate from the currently qualified launcher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import bootstrap_radiance_release as release
from patch_upstream_correctness import BUNDLE, candidate_data_abi, install, sha256

NATIVE_PATHS = {
    "triton": "/opt/vllm/lib/python3.12/site-packages/triton/_C/libtriton.so",
    "rocr": "/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1.21.0",
    "xgrammar": "/opt/vllm/lib/python3.12/site-packages/xgrammar/libxgrammar_bindings.so",
}


def install_before_streaming(package: Path, run_path):
    """Compose before the first release patch, after its pristine validation."""
    expected_script = Path(__file__).with_name("patch_streaming_snapshot.py").resolve()

    def composed(path, *args, **kwargs):
        if Path(path).resolve() != expected_script:
            raise ValueError("unexpected release bootstrap source-patch step")
        receipt = install(package, "python")
        result = run_path(path, *args, **kwargs)
        (package / "qwen_upstream_backports_receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        return result

    return composed


def verify_native_bindings(path: Path) -> dict:
    binding = json.loads(path.read_text())
    if binding.get("manifest_sha256") != sha256((BUNDLE / "manifest.json").read_bytes()):
        raise ValueError("native artifacts were built for a different backport bundle")
    components = binding.get("components", {})
    if set(components) != {"triton", "rocr", "xgrammar"}:
        raise ValueError("candidate requires all three rebuilt native components")
    for name, artifacts in components.items():
        if set(artifacts) != {NATIVE_PATHS[name]}:
            raise ValueError(f"native binding does not name the installed runtime library: {name}")
        for filename, expected in artifacts.items():
            artifact = Path(filename)
            if not artifact.is_absolute() or ".so" not in artifact.name:
                raise ValueError("native bindings must name installed shared libraries")
            with artifact.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError(f"installed native artifact differs: {name}")
    return components


def isolate_compile_caches(env: dict, identity: str) -> None:
    for key, fallback in {
        "TRITON_CACHE_DIR": "/tmp/qwen-triton",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/qwen-inductor",
        "VLLM_CACHE_ROOT": "/tmp/qwen-vllm",
    }.items():
        base = Path(env.get(key, fallback)).expanduser()
        if not base.is_absolute():
            raise ValueError(f"candidate compile cache must have an absolute path: {key}")
        env[key] = str(base / ("upstream-" + identity))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-bindings", type=Path, required=True)
    args, model_args = parser.parse_known_args()
    components = verify_native_bindings(args.native_bindings)
    parent = json.loads(Path(__file__).with_name("snapshot-abi-chat-cache-v1.json").read_text())[
        "storage"
    ]["data_abi"]
    expected = candidate_data_abi(parent, native_bindings=components)
    if os.environ.get("QWEN_RADIANCE_CACHE_ABI") != expected:
        raise ValueError(f"candidate requires its own snapshot data ABI: {expected}")
    isolate_compile_caches(os.environ, expected)
    # Offloading scheduler source belongs to both patch sets. Interpose before
    # the streaming patch, not at the later chat-install call; the latter would
    # already have modified a source preimage required by this bundle.
    release.runpy = SimpleNamespace(
        run_path=install_before_streaming(
            Path("/opt/vllm/lib/python3.12/site-packages"), release.runpy.run_path
        )
    )
    sys.argv = [sys.argv[0], *model_args]
    release.main()


if __name__ == "__main__":
    main()
