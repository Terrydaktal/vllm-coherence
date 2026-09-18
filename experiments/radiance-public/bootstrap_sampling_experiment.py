"""Apply the candidate sampling correction after production preimage checks."""

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

from patch_dflash_sampling_rng import install

original_exec = os.execv


def checked_exec(path, argv):
    if path != "/opt/radiance_entrypoint.sh":
        raise ValueError("unexpected experiment bootstrap handoff")
    result = install(Path("/opt/vllm/lib/python3.12/site-packages"))
    print(json.dumps({"experiment_sampling_patch": result}), flush=True)
    manifest = json.loads(Path("/benchmark/manifest.json").read_text())
    if manifest.get("audit_verify_head"):
        from audit_verify_head import install as install_head_audit
        from audit_verify_head import verify_helpers

        source = Path("/benchmark/audit_verify_head.py")
        if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["verify_head_audit_sha256"]:
            raise ValueError("target-head audit source binding differs")
        verify_helpers(Path("/benchmark"), manifest)
        print(
            json.dumps(
                {
                    "verify_head_audit": install_head_audit(
                        Path("/opt/vllm/lib/python3.12/site-packages"), source
                    )
                }
            ),
            flush=True,
        )
    if manifest.get("gdn_extreme_decay_repair"):
        assert not manifest.get("gdn_span_audit"), (
            "keep the correction replay free of intrusive probes"
        )
        from patch_gdn_extreme_decay import install as install_gdn_correction

        print(
            json.dumps(
                {
                    "experiment_gdn_correction": install_gdn_correction(
                        Path("/opt/vllm/lib/python3.12/site-packages"),
                        Path("/benchmark/gdn_extreme_decay_reference.so"),
                        manifest["gdn_repair_sha256"],
                    )
                }
            ),
            flush=True,
        )
    if manifest.get("gdn_span_audit"):
        from patch_gdn_span_audit import install as install_audit

        print(
            json.dumps(
                {
                    "experiment_gdn_audit": install_audit(
                        Path("/opt/vllm/lib/python3.12/site-packages")
                    )
                }
            ),
            flush=True,
        )
    if manifest.get("bf16_gdn_gate_reference"):
        from patch_gdn_gate_precision_experiment import install as install_gate_reference

        print(
            json.dumps(
                {
                    "experiment_gdn_gate_reference": install_gate_reference(
                        Path("/opt/vllm/lib/python3.12/site-packages")
                    )
                }
            ),
            flush=True,
        )
    original_exec(path, argv)


os.execv = checked_exec
sys.path.insert(0, "/patches")
runpy.run_path("/patches/bootstrap_radiance_release.py", run_name="__main__")
