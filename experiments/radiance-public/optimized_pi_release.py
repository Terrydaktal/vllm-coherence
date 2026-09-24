"""Authenticate the frozen optimized serving payload before starting vLLM."""

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path


def compatible_output_head_contract(previous, current):
    """Reuse KV only when the sole changes select logits after the backbone.

    The processed token prefix is already part of each snapshot's identity.
    Output-head selection does not change KV, GDN or convolution for that prefix.
    Keep the old canonical contract/hash; runtime metadata records the new head.
    """
    if not previous:
        return current

    def normalized(contract):
        contract = deepcopy(contract)
        env = contract.get("kernel_environment", {})
        for name in (
            "RADIANCE_VERIFY_HEAD",
            "RADIANCE_VERIFY_HEAD_GLOBAL_TOPK",
            "RADIANCE_VERIFY_HEAD_MAX_M",
        ):
            env.pop(name, None)
        contract.get("serving", {}).pop("target_verify_head", None)
        return contract

    return previous if normalized(previous) == normalized(current) else current


def configure(profile, patches, *, root=Path("/qualification"), environ=None):
    if environ is None:
        environ = os.environ
    entry = profile["optimized_d7"]
    path = Path(patches) / "optimized-release.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["manifest_sha256"]:
        raise ValueError("optimized production manifest changed")
    manifest = json.loads(path.read_text())
    if (
        manifest["schema"] != "urn:qwen:optimized-pi-release:v1"
        or manifest["state_layout"] != "existing-nine-slot"
        or manifest["container_root"] != "/qualification"
        or not manifest["files"]
    ):
        raise ValueError("unsupported optimized serving payload")
    for name, expected in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("optimized payload path escapes its root")
        artifact = root / relative
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError(f"optimized payload missing: {name}")
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
            raise ValueError(f"optimized payload changed: {name}")
    env = manifest["environment"]
    if manifest.get("m1_arithmetic") and (
        entry.get("m1_arithmetic") != manifest["m1_arithmetic"]
        or entry.get("arithmetic", {}).get("m1_arithmetic")
        != manifest["m1_arithmetic"]["contract"]
    ):
        raise ValueError("M1 repair requires its own snapshot arithmetic identity")
    required = {
        "QWEN_STOCK_GDN_LAZY": "0",
        "RADIANCE_GDN_LAZY": "0",
        "TORCHINDUCTOR_EMULATE_PRECISION_CASTS": "1",
    }
    head = manifest.get("target_head", "full-bf16")
    if head in ("global256", "global512"):
        if entry.get("target_head", {}).get("mode") != head:
            raise ValueError("optimized target head differs from release profile")
        required.update(
            RADIANCE_VERIFY_HEAD="1",
            RADIANCE_VERIFY_HEAD_GLOBAL_TOPK=head.removeprefix("global"),
        )
    elif head == "full-bf16":
        required["RADIANCE_VERIFY_HEAD"] = "0"
    else:
        raise ValueError("unsupported optimized target head")
    if any(env.get(key) != value for key, value in required.items()):
        raise ValueError("optimized arithmetic/state-layout contract changed")
    if manifest.get("attention_precision"):
        if (
            entry.get("attention_precision") != manifest["attention_precision"]
            or entry.get("arithmetic", {}).get("attention_precision")
            != manifest["attention_precision"]["contract"]
        ):
            raise ValueError(
                "attention precision requires its own snapshot arithmetic identity"
            )
    if manifest.get("normalization_consistency") and (
        entry.get("normalization_consistency") != manifest["normalization_consistency"]
        or entry.get("arithmetic", {}).get("normalization_consistency")
        != manifest["normalization_consistency"]["contract"]
    ):
        raise ValueError(
            "normalization repair requires its own snapshot arithmetic identity"
        )
    environ.update(env)
    if manifest.get("attention_precision"):
        environ["QWEN_ATTENTION_PRECISION_BUILD"] = manifest["attention_precision"][
            "build"
        ]
    # The frozen preflight trees deliberately retain their historical modules,
    # but some of those names also exist in the pinned Radiance image.  Putting
    # the image's authenticated site-packages directory first prevents a stale
    # preflight copy from shadowing the current runtime integration.  In
    # particular, the old v1 ``qwen_radiance_fair_scheduler`` emitted only the
    # legacy phase schema and silently removed round/acceptance telemetry even
    # though the v2 module was present in the image.
    runtime_pythonpath = [
        "/opt/vllm/lib/python3.12/site-packages",
        *manifest["pythonpath"],
    ]
    environ["PYTHONPATH"] = os.pathsep.join(runtime_pythonpath)
    return manifest
