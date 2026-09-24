#!/usr/bin/env python3
"""Export only manifest-listed serving code, receipts and kernels; never captures."""

import argparse
import json
import tarfile
from pathlib import Path

from coherence_cli import ROOT, digest, verify_payload

PATCHES = [
    "bootstrap_radiance_release.py",
    "optimized_pi_release.py",
    "runtime-radiance-1.0.16.json",
    "patch_streaming_snapshot.py",
    "patch_chat_snapshot.py",
    "radiance_chat_tier.py",
    "radiance_fair_scheduler.py",
    "radiance_request_guard.py",
    "patch_dflash_sampling_rng.py",
    "patch_draft_head_initialization.py",
    "patch_verify_head_memory.py",
    "radiance_verifyhead_global.py",
    "patch_gdn_extreme_decay.py",
    "gdn_extreme_decay_reference.hip",
    "qwen-fixed-v22.3.jinja",
]


def package(source, output):
    manifest = verify_payload(source)
    support = "support/libhsa-runtime64.so.1.21.0"
    backoff = json.loads(
        (
            ROOT / "experiments/radiance-public/rocr-poll-backoff/runtime.json"
        ).read_text()
    )
    if digest(source / support) != backoff["library_sha256"]:
        raise ValueError("ROCr backoff library mismatch")
    paths = sorted([*manifest["files"], "optimized-release.json", support])
    with tarfile.open(output, "w:xz", preset=6) as archive:
        for name in paths:
            path = source / name
            if not path.is_file() or path.is_symlink():
                raise ValueError("only regular files may be published")
            record = tarfile.TarInfo(name)
            record.size = path.stat().st_size
            record.mode = 0o600
            record.mtime = 0
            with path.open("rb") as data:
                archive.addfile(record, data)
    patch_paths = ["experiments/radiance-public/" + name for name in PATCHES]
    patch_paths += [
        "src/qwen_r9700_lab/" + name
        for name in ("radiance_cache.py", "radiance_memory.py")
    ]
    profile = json.loads(
        (ROOT / "experiments/radiance-public/runtime-radiance-1.0.16.json").read_text()
    )
    return {
        "schema": "urn:coherence:release:v1",
        "version": "0.1.0",
        "platform": "linux-amd64-gfx1201",
        "url": "https://github.com/Terrydaktal/vllm-coherence/releases/download/v0.1.0/coherence-runtime-0.1.0-linux-amd64-gfx1201.tar.xz",
        "sha256": digest(output),
        "bytes": output.stat().st_size,
        "manifest_sha256": digest(source / "optimized-release.json"),
        "image": "docker.io/magiccodingman/vllm-radiance@" + profile["image_digest"],
        "upstream_commit": profile["source_commit"],
        "patch_files": patch_paths,
        "integration_sha256": {name: digest(ROOT / name) for name in patch_paths},
        "support_files": {support: backoff["library_sha256"]},
        "scope": "Frozen serving payload; existing nine-slot state layout; full-head numerical controls and separate approximate global-512 head. Repackaging does not create new GPU qualification.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = package(args.source, args.output)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"archive_bytes": result["bytes"], "sha256": result["sha256"]}))
