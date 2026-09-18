"""Capture one real GDN prefill invocation for independent private replay."""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
from contextlib import contextmanager
from pathlib import Path

SOURCE_SHA256 = "e05531ec81eac7e401a579a5edeb14cf7c00220df366c8a30ca65c7b5e81a331"
INPUTS = ("q", "k", "v", "A", "g", "beta", "initial_state", "cu_seqlens")


@contextmanager
def capture_prefill(native, helpers, request, layer, root, private):
    """Temporarily bind only the selected layer's single eager prefill call."""
    if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("GDN operator source differs from the qualified capture interface")
    original = native.fused_prefill
    signature = inspect.signature(original)
    calls = 0

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        nonlocal calls
        import torch

        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        if calls or not values["output_final_state"]:
            raise ValueError("operator capture needs exactly one state-producing invocation")
        captured = {}
        for name in INPUTS:
            value = values[name]
            if not isinstance(value, torch.Tensor):
                raise ValueError("GDN input is not a tensor")
            cpu = value.detach().to("cpu", copy=True).contiguous()
            captured[name] = {
                "tensor": cpu,
                "sha256": helpers._tensor_sha256(cpu),
                "layout": helpers._tensor_layout(value),
            }
        result = original(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("GDN native capture did not execute the supported path")
        for name, value in zip(("output", "final_state"), result, strict=True):
            cpu = value.detach().to("cpu", copy=True).contiguous()
            captured[name] = {
                "tensor": cpu,
                "sha256": helpers._tensor_sha256(cpu),
                "layout": helpers._tensor_layout(value),
            }
        capsule_id = request["capture_id"] + f"-gdn-layer{layer}"
        capsule_root = Path(private) / "layer-capsules"
        capsule_root.mkdir(mode=0o700, exist_ok=True)
        if capsule_root.is_symlink() or capsule_root.stat().st_mode & 0o077:
            raise ValueError("GDN operator capsule directory is not private")
        payload = {
            "schema": "qwen-gdn-operator-capsule-v1",
            "request": request,
            "layer": layer,
            "source_sha256": SOURCE_SHA256,
            "scale": float(values["scale"]),
            "chunk": int(native.CHUNK),
            "tensors": captured,
        }
        identity = helpers._write_tensor_capsule(capsule_root / (capsule_id + ".pt"), payload)
        report = {
            "schema": "qwen-gdn-operator-capture-v1",
            "complete": True,
            "capsule_id": capsule_id,
            "capsule_sha256": identity,
            "input_sha256": request["input_sha256"],
            "layer": layer,
            "source_sha256": SOURCE_SHA256,
            "shapes": {name: list(row["tensor"].shape) for name, row in captured.items()},
            "nonfinite_values": sum(
                int((~torch.isfinite(row["tensor"])).sum()) for row in captured.values()
            ),
        }
        (Path(root) / f"gdn-operator-layer{layer}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        calls += 1
        return result

    native.fused_prefill = wrapped
    try:
        yield
        if calls != 1:
            raise ValueError("selected layer did not reach the GDN capture boundary")
    finally:
        native.fused_prefill = original
