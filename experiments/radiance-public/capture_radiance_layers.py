"""Adapt the original M8 tensor-capture helpers to the pinned Radiance model.

An explicitly armed, one-token diagnostic captures selected decoder rows in
protected RAM. No chat text or token IDs are recorded in public reports. GPU
copies synchronize only during that diagnostic; its times are not benchmarks.
The original W4A16 fixed-slot and Quest hooks are deliberately not installed.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import os
import re
import shutil
import threading
import time
from contextlib import nullcontext
from pathlib import Path

LEGACY_SHA256 = "96c6cafb9fdef9f99b256b5977fd9068f99f8d60d0b1ffd0a5478eeed9a05694"
MODEL_PREIMAGE = "5aeac6c81bfa7c680e517106aecb7741a0a4e90fa52de77f0be1eadebc7516fa"
MODEL_SOURCE = "vllm/model_executor/models/qwen3_next.py"
APPENDIX = '''

# Isolated, explicitly armed decoder capture; original forward math is retained.
from qwen_radiance_layer_capture import patch as _qwen_patch_layer_capture
from pathlib import Path as _QwenCapturePath
_qwen_patch_layer_capture(Qwen3NextModel, Qwen3NextDecoderLayer,
    _QwenCapturePath('/benchmark'), _QwenCapturePath('/benchmark/legacy_layer_diagnostic.py'))
'''


def install(package, source, legacy_source):
    package, source, legacy_source = map(Path, (package, source, legacy_source))
    if hashlib.sha256(legacy_source.read_bytes()).hexdigest() != LEGACY_SHA256:
        raise ValueError("legacy capture helper identity changed")
    target = package / MODEL_SOURCE
    original = target.read_text()
    if original.endswith(APPENDIX):
        original = original[:-len(APPENDIX)]
    if hashlib.sha256(original.encode()).hexdigest() != MODEL_PREIMAGE:
        raise ValueError("decoder source is outside the qualified capture interface")
    updated = original + APPENDIX
    compile(updated, str(target), "exec")
    shutil.copyfile(source, package / "qwen_radiance_layer_capture.py")
    target.write_text(updated)
    return {"model_source_sha256": hashlib.sha256(updated.encode()).hexdigest(),
            "legacy_helpers_sha256": LEGACY_SHA256,
            "capture_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}


def load_legacy_helpers(source):
    source = Path(source)
    if hashlib.sha256(source.read_bytes()).hexdigest() != LEGACY_SHA256:
        raise ValueError("legacy capture helpers do not match their recorded source")
    # Import the unchanged module solely for its generic helpers. Its install()
    # is never called. Its old import-time selector still requires a valid value.
    name = "QWEN_M8_LAYER_DIAGNOSTIC_POSITIONS"
    previous = os.environ.get(name)
    os.environ[name] = "1"
    try:
        spec = importlib.util.spec_from_file_location("qwen_m8_capture_helpers", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
    return module


def validate_request(value, private_root):
    private_root = Path(private_root)
    if (private_root.is_symlink() or not private_root.is_dir()
            or private_root.stat().st_mode & 0o077):
        raise ValueError("decoder captures require a private RAM directory")
    if value.get("schema") != "qwen-radiance-layer-capture-request-v1":
        raise ValueError("unknown decoder capture request")
    if not re.fullmatch(r"[a-z0-9_-]{1,100}", value.get("capture_id", "")):
        raise ValueError("invalid decoder capture identity")
    positions = value.get("positions")
    if (not isinstance(positions, list) or not 1 <= len(positions) <= 8
            or any(type(p) is not int or not 0 <= p < 253792 for p in positions)
            or positions != sorted(set(positions))):
        raise ValueError("invalid decoder capture positions")
    if type(value.get("expected_layers")) is not int or not 1 <= value["expected_layers"] <= 64:
        raise ValueError("invalid decoder layer count")
    if not re.fullmatch(r"[0-9a-f]{64}", value.get("input_sha256", "")):
        raise ValueError("decoder capture requires an authenticated prompt hash")
    operator_layers = value.get("gdn_operator_layers", [])
    if (not isinstance(operator_layers, list)
            or any(type(layer) is not int or not 0 <= layer < value["expected_layers"]
                   for layer in operator_layers)
            or operator_layers != sorted(set(operator_layers))):
        raise ValueError("invalid selected GDN operator layers")
    return value


class Capture:
    def __init__(self, request, selected, layers, helpers):
        self.request, self.selected, self.helpers = request, selected, helpers
        self.layer_ids = {id(layer): index for index, layer in enumerate(layers)}
        if len(self.layer_ids) != request["expected_layers"]:
            raise ValueError("loaded decoder layer count differs from capture request")
        self.records = {}
        self.aliases = {}
        self.started = time.monotonic()

    def tensor(self, layer, stage, value):
        import torch

        if value is None:
            return
        if not isinstance(value, torch.Tensor):
            raise ValueError("unsupported decoder capture value")
        for row, position in self.selected:
            view = value[row:row + 1]
            layout = self.helpers._tensor_layout(view)
            cpu = view.detach().to(device="cpu", copy=True).contiguous()
            key = f"{layer}:{stage}:{position}"
            if key in self.records:
                raise ValueError("decoder capture stage ran twice")
            self.records[key] = {"tensor": cpu, "layout": layout,
                                 "sha256": self.helpers._tensor_sha256(cpu),
                                 "nonfinite": int((~torch.isfinite(cpu)).sum().item())}

    def output(self, layer, stage, value):
        if isinstance(value, tuple):
            for index, tensor in enumerate(value):
                self.tensor(layer, f"{stage}.{index}", tensor)
        else:
            self.tensor(layer, stage, value)

    def alias(self, layer, stage, left, right):
        if left is not None and right is not None:
            for row, position in self.selected:
                self.aliases[f"{layer}:{stage}:{position}"] = self.helpers._storage_relation(
                    left[row:row + 1], right[row:row + 1])

    def save(self, root, private_root):
        expected = {(layer, position) for layer in range(self.request["expected_layers"])
                    for _, position in self.selected}
        observed = {(int(key.split(":")[0]), int(key.rsplit(":", 1)[1]))
                    for key in self.records if ":output.0:" in key}
        if expected != observed:
            raise ValueError("decoder capture did not observe every selected layer")
        capsule_root = Path(private_root) / "layer-capsules"
        capsule_root.mkdir(mode=0o700, exist_ok=True)
        if capsule_root.is_symlink() or capsule_root.stat().st_mode & 0o077:
            raise ValueError("decoder capsule directory is not private")
        capsule = capsule_root / (self.request["capture_id"] + ".pt")
        digest = self.helpers._write_tensor_capsule(capsule, {
            "schema": "qwen-radiance-layer-capsule-v1", "request": self.request,
            "records": self.records, "aliases": self.aliases,
            "legacy_helpers_sha256": LEGACY_SHA256,
        })
        report = {"schema": "qwen-radiance-layer-capture-v1", "complete": True,
                  "capture_id": self.request["capture_id"], "capsule_sha256": digest,
                  "input_sha256": self.request["input_sha256"],
                  "legacy_helpers_sha256": LEGACY_SHA256,
                  "positions": [position for _, position in self.selected],
                  "layers": self.request["expected_layers"], "tensors": len(self.records),
                  "nonfinite_values": sum(row["nonfinite"] for row in self.records.values()),
                  "synchronizing_diagnostic_seconds": time.monotonic() - self.started}
        temporary = Path(root) / "layer-capture-result.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(Path(root) / "layer-capture-result.json")


def patch(model_cls, layer_cls, root, legacy_source, private_root=Path("/private-fixtures")):
    if getattr(model_cls.forward, "_qwen_radiance_layer_capture", False):
        return
    helpers = load_legacy_helpers(legacy_source)
    model_forward, layer_forward = model_cls.forward, layer_cls.forward
    active = threading.local()
    request = None
    completed = False

    @functools.wraps(model_forward)
    def model_wrapped(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        nonlocal request, completed
        capture = None
        if not completed:
            marker = Path(root) / "layer-capture-request.json"
            if request is None and marker.exists():
                request = validate_request(json.loads(marker.read_text()), private_root)
            if request is not None:
                rows = helpers._position_rows(positions, int(positions.shape[-1]))
                selected = [(row, axes[0]) for row, axes in enumerate(rows)
                            if axes[0] in request["positions"]]
                if selected:
                    if [position for _, position in selected] != request["positions"]:
                        raise ValueError("selected capture positions span different model calls")
                    capture = Capture(request, selected, self.layers, helpers)
        active.capture = capture
        try:
            result = model_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds)
            if capture is not None:
                capture.output(-1, "model_output", result)
                capture.save(root, private_root)
                completed = True
            return result
        finally:
            active.capture = None

    @functools.wraps(layer_forward)
    def layer_wrapped(self, hidden_states, residual, positions=None, **kwargs):
        capture = getattr(active, "capture", None)
        if capture is None:
            return layer_forward(self, hidden_states, residual, positions, **kwargs)
        index = capture.layer_ids[id(self)]
        capture.tensor(index, "input_hidden", hidden_states)
        capture.tensor(index, "input_residual", residual)
        capture.alias(index, "input_hidden_residual", hidden_states, residual)
        handles = []
        for name in ("input_layernorm", "linear_attn", "self_attn",
                     "post_attention_layernorm", "mlp"):
            child = getattr(self, name, None)
            if child is not None:
                def record(module, args, output, stage=name):
                    capture.output(index, stage, output)
                handles.append(child.register_forward_hook(record))
        try:
            operator = nullcontext()
            if index in request.get("gdn_operator_layers", []):
                import radiance_gdn
                from qwen_gdn_operator_capture import capture_prefill

                operator = capture_prefill(radiance_gdn, helpers, request, index, root, private_root)
            with operator:
                result = layer_forward(self, hidden_states, residual, positions, **kwargs)
            capture.output(index, "output", result)
            capture.alias(index, "output_hidden_residual", *result)
            return result
        finally:
            for handle in handles:
                handle.remove()

    model_wrapped._qwen_radiance_layer_capture = True
    model_cls.forward, layer_cls.forward = model_wrapped, layer_wrapped
