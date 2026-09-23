#!/usr/bin/env python3
"""Apply verified correctness repairs and chat integration to pinned Radiance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import runpy
import shutil
import sys
from pathlib import Path

from patch_chat_snapshot import install
from patch_dflash_sampling_rng import install as install_dflash_sampling_rng
from patch_draft_head_initialization import install as install_draft_head_initialization
from patch_gdn_extreme_decay import (
    LIBRARY_SHA256 as GDN_LIBRARY_SHA256,
)
from patch_gdn_extreme_decay import (
    build as build_gdn_correction,
)
from patch_gdn_extreme_decay import (
    install as install_gdn_correction,
)
from patch_verify_head_memory import install as install_verify_head_memory


def main():
    source = Path(__file__).resolve().parent
    profile = json.loads((source / "runtime-radiance-1.0.16.json").read_text())
    if profile.get("optimized_d7"):
        from optimized_pi_release import configure

        configure(profile, source)
    package = Path("/opt/vllm/lib/python3.12/site-packages")
    for name, expected in profile["package_versions"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Radiance package differs from the release profile: {name}")
    for name, expected in profile["source_preimages"].items():
        if hashlib.sha256((package / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Radiance source differs from the release profile: {name}")
    for name, expected in profile["kernel_hashes"].items():
        if hashlib.sha256((package / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Radiance kernel differs from the release profile: {name}")
    # Preserve the published image's vLLM, DFlash, parser and kernel fixes.
    # The local overlay adds chat snapshot lifecycle, scheduling and telemetry.
    scheduler = package / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
    os.environ["QWEN_SNAPSHOT_SCHEDULER"] = str(scheduler)
    runpy.run_path(str(source / "patch_streaming_snapshot.py"))
    os.environ.pop("QWEN_SNAPSHOT_SCHEDULER")
    install(package, source / "radiance_cache.py", source / "radiance_chat_tier.py")
    if profile.get("optimized_d7", {}).get("target_head", {}).get("mode") == "global256":
        shutil.copyfile(
            source / "radiance_verifyhead_global.py", package / "radiance_verifyhead.py"
        )
    install_verify_head_memory(package)
    # DFlash shares its real head after load_weights; allocated placeholder data
    # must never be mistaken for loaded weights or packed into the INT2 head.
    install_draft_head_initialization(package)
    # A rejected draft must not share its Gumbel draw with the target's
    # replacement. Backport the independently qualified upstream stream salt.
    install_dflash_sampling_rng(package)
    # The native scan can clamp valid recurrent contributions to zero for large
    # decay spans. Recompute affected heads with the verified bounded recurrence.
    library = build_gdn_correction(
        source / "gdn_extreme_decay_reference.hip", package / "qwen_gdn_extreme_decay.so"
    )
    install_gdn_correction(package, library, GDN_LIBRARY_SHA256)
    shutil.copyfile(
        source / "radiance_request_guard.py", package / "qwen_radiance_request_guard.py"
    )
    print("Radiance release and local chat integration verified", flush=True)
    if sys.argv[1:] == ["--check-only"]:
        return
    os.chdir("/")
    os.execv("/opt/radiance_entrypoint.sh", ["/opt/radiance_entrypoint.sh", *sys.argv[1:]])


if __name__ == "__main__":
    main()
