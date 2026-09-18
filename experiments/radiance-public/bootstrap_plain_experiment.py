"""Start the pinned image without local snapshot or scheduling integration.

Diagnostic control only: retain the reviewed sampler correction and the
verification-head memory repair, while bypassing local cache/scheduler patches.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/patches")
from patch_verify_head_memory import install as install_memory_fix
from patch_dflash_sampling_rng import install as install_sampling_fix


def main():
    package = Path("/opt/vllm/lib/python3.12/site-packages")
    profile = json.loads(Path("/patches/runtime-radiance-1.0.16.json").read_text())
    for name, version in profile["package_versions"].items():
        assert importlib.metadata.version(name) == version, name
    for group in ("source_preimages", "kernel_hashes"):
        for name, expected in profile[group].items():
            assert hashlib.sha256((package / name).read_bytes()).hexdigest() == expected, name
    install_memory_fix(package)
    manifest = json.loads(Path("/benchmark/manifest.json").read_text())
    if manifest.get("r4d_dispatch_audit"):
        source = Path("/benchmark/r4d_dispatch_audit.py")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest == manifest["r4d_dispatch_audit_sha256"]
        assert os.environ["QWEN_R4D_AUDIT_SHA256"] == digest
        assert os.environ["QWEN_R4D_AUDIT_MODE"] == manifest["r4d_dispatch_audit"]
        shutil.copyfile(source, package / "qwen_r4d_dispatch_audit.py")
        (package / "00_qwen_r4d_dispatch_audit.pth").write_text(
            "import qwen_r4d_dispatch_audit; qwen_r4d_dispatch_audit.install_from_environment()\n")
    if manifest.get("capture_decoder_layers"):
        from capture_radiance_layers import install as install_layer_capture

        source = Path("/benchmark/capture_radiance_layers.py")
        assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["layer_capture_sha256"]
        print(json.dumps({"decoder_layer_capture": install_layer_capture(
            package, source, Path("/benchmark/legacy_layer_diagnostic.py"))}), flush=True)
    if manifest.get("gdn_operator_layers"):
        source = Path("/benchmark/capture_gdn_operator.py")
        assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["gdn_operator_capture_sha256"]
        shutil.copyfile(source, package / "qwen_gdn_operator_capture.py")
    if manifest.get("reference_linear"):
        if manifest.get("reference_linear_fast"):
            from patch_reference_linear_fast_experiment import install as install_reference
        else:
            from patch_reference_linear_experiment import install as install_reference
        print(json.dumps({"diagnostic_linear_reference": install_reference(package)}), flush=True)
    if manifest.get("gdn_extreme_decay_repair"):
        from patch_gdn_extreme_decay import install as install_gdn_correction

        print(json.dumps({"experiment_gdn_correction": install_gdn_correction(
            package, Path("/benchmark/gdn_extreme_decay_reference.so"),
            manifest["gdn_repair_sha256"])}), flush=True)
    print(json.dumps({"sampling_fix": install_sampling_fix(package),
                      "local_snapshot_and_scheduler_integration": False}), flush=True)
    os.chdir("/")
    os.execv("/opt/radiance_entrypoint.sh", ["/opt/radiance_entrypoint.sh", *sys.argv[1:]])


if __name__ == "__main__":
    main()
