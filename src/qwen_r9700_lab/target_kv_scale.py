"""Default-off calibration/application hook for target static FP8 KV scales.

The target checkpoint does not carry KV-cache scales, so pinned vLLM otherwise
uses one scalar value of 1.0 for every full-attention layer.  This qualification
hook records only aggregate absolute maxima from the exact 16 target
full-attention layers, or applies a private, provenance-bound scale document
before the first cache write.  It never records prompts, tokens, hidden states,
or KV tensors.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import stat
import threading
from pathlib import Path
from typing import Any

MODE_ENV = "QWEN_TARGET_KV_SCALE_MODE"
OUTPUT_ENV = "QWEN_TARGET_KV_SCALE_OUTPUT"
FILE_ENV = "QWEN_TARGET_KV_SCALE_FILE"
MULTIPLIER_ENV = "QWEN_TARGET_KV_SCALE_MULTIPLIER"
ARM_ENV = "QWEN_TARGET_KV_SCALE_CAPTURE_ARM_FILE"

EXPECTED_LAYER_INDICES = tuple(range(3, 64, 4))
EXPECTED_LAYER_NAMES = frozenset(
    f"language_model.model.layers.{index}.self_attn.attn" for index in EXPECTED_LAYER_INDICES
)
EXPECTED_LAYERS = len(EXPECTED_LAYER_NAMES)
# With the qualified 4,096 scheduler budget, the public 60,298-token request
# performs 15 target prefill writes and the stock BF16 reference performs 125
# verifier rounds for its 512-token completion.  The hook is armed only after
# startup, so exactly 140 samples cover the complete reference trajectory.
SAMPLES_PER_LAYER = 140
FP8_RUNTIME_MAX = 448.0

RUNTIME_ROOT = Path(
    "/home/lewis/.local/share/qwen-r9700/runtimes/vllm-rocm/.venv/lib/python3.12/site-packages"
)
TARGET_ROOT = Path("/home/lewis/models/Qwen3.8-27B-int4-AutoRound")
PINNED_FILES = {
    "rocm_attn": (
        RUNTIME_ROOT / "vllm/v1/attention/backends/rocm_attn.py",
        "c27f7131b66d3c284d8f1fa826cfc5058b94823b1fb708d6d9345a35ae1ed36b",
    ),
    "kv_cache_quant": (
        RUNTIME_ROOT / "vllm/model_executor/layers/quantization/kv_cache.py",
        "cd09817e065d5b0074382fc04a2374ca0ba1c879b3cd125b62f4738e1631ea8a",
    ),
    "cache_writer": (
        RUNTIME_ROOT / "vllm/v1/attention/ops/triton_reshape_and_cache_flash.py",
        "6cac51475b8c656992a21b2d150acd3a16a95a7ab7d49aab151a3ef13c24b80d",
    ),
    "target_config": (
        TARGET_ROOT / "config.json",
        "15173d7a487c88112a02804701ab2cc8f8dd4631a3ef67c8d9bb2c66d30debce",
    ),
    "target_index": (
        TARGET_ROOT / "model.safetensors.index.json",
        "adf387dee183d109e95cdc4d4988fcd966de326861ff4d66876692170f1e03ad",
    ),
    "target_tokenizer": (
        TARGET_ROOT / "tokenizer.json",
        "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    ),
}

_LOCK = threading.Lock()
_SAMPLES: dict[str, dict[str, Any]] = {}
_APPLIED: set[str] = set()
_SCALE_DOCUMENT: dict[str, Any] | None = None
_OUTPUT: Path | None = None
_CAPTURE_ARM: Path | None = None
_RUNTIME_HASHES: dict[str, str] = {}
_STARTED = False
_TARGET_MODULE = "vllm.v1.attention.backends.rocm_attn"


class TargetKVScaleError(RuntimeError):
    """Qualification contract failure."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_runtime() -> dict[str, str]:
    observed: dict[str, str] = {}
    for label, (path, expected) in PINNED_FILES.items():
        if not path.is_file() or path.is_symlink():
            raise TargetKVScaleError(f"pinned {label} is not a regular file: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise TargetKVScaleError(
                f"pinned {label} drift: expected {expected}, observed {actual}"
            )
        observed[label] = actual
    return observed


def _is_target_impl(impl: Any, *, capture: bool) -> bool:
    cache_dtype = str(impl.kv_cache_dtype)
    return (
        tuple(impl.sliding_window) == (-1, -1)
        and int(impl.num_heads) == 24
        and int(impl.num_kv_heads) == 4
        and int(impl.head_size) == 256
        and cache_dtype == ("auto" if capture else "fp8_e4m3")
        and str(impl.fp8_dtype) == "torch.float8_e4m3fn"
    )


def _layer_name(layer: Any) -> str:
    name = getattr(layer, "layer_name", None)
    if not isinstance(name, str) or not name:
        raise TargetKVScaleError("eligible target attention layer has no stable name")
    if name not in EXPECTED_LAYER_NAMES:
        raise TargetKVScaleError(f"unexpected eligible target attention layer: {name}")
    return name


def _recommended_scale(maximum_abs: float) -> float:
    if not 0.0 < maximum_abs < float("inf"):
        raise TargetKVScaleError(f"invalid captured absolute maximum: {maximum_abs}")
    return maximum_abs / FP8_RUNTIME_MAX


def _contract() -> dict[str, Any]:
    return {
        "layers": EXPECTED_LAYERS,
        "layer_names": sorted(EXPECTED_LAYER_NAMES),
        "samples_per_layer": SAMPLES_PER_LAYER,
        "qualified_prompt_tokens": 60298,
        "qualified_completion_tokens": 512,
        "fp8_runtime_dtype": "torch.float8_e4m3fn",
        "fp8_max": FP8_RUNTIME_MAX,
        "sliding_window": [-1, -1],
        "num_heads": 24,
        "num_kv_heads": 4,
        "head_size": 256,
        "capture_target_kv_cache_dtype": "auto",
        "apply_target_kv_cache_dtype": "fp8_e4m3",
    }


def _capture_document() -> dict[str, Any] | None:
    if set(_SAMPLES) != EXPECTED_LAYER_NAMES:
        return None
    if any(
        len(values[side]) < SAMPLES_PER_LAYER for values in _SAMPLES.values() for side in ("k", "v")
    ):
        return None
    layers: dict[str, Any] = {}
    for name in sorted(_SAMPLES):
        k_samples = _SAMPLES[name]["k"][:SAMPLES_PER_LAYER]
        v_samples = _SAMPLES[name]["v"][:SAMPLES_PER_LAYER]
        k_max = max(k_samples)
        v_max = max(v_samples)
        layers[name] = {
            "samples": SAMPLES_PER_LAYER,
            "k_measurement": _SAMPLES[name]["k_measurement"],
            "k_absmax_samples": k_samples,
            "v_absmax_samples": v_samples,
            "k_absmax": k_max,
            "v_absmax": v_max,
            "k_static_scale": _recommended_scale(k_max),
            "v_static_scale": _recommended_scale(v_max),
        }
    return {
        "schema": "qwen-r9700-target-kv-scale-v1",
        "contract": _contract(),
        "runtime_sha256": dict(sorted(_RUNTIME_HASHES.items())),
        "layers": layers,
    }


def _write_create_only(path: Path, document: dict[str, Any]) -> None:
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _capture_armed() -> bool:
    assert _CAPTURE_ARM is not None
    if not _CAPTURE_ARM.exists():
        return False
    if not _CAPTURE_ARM.is_file() or _CAPTURE_ARM.is_symlink():
        raise TargetKVScaleError("capture arm path must be a regular file")
    metadata = _CAPTURE_ARM.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
        raise TargetKVScaleError("capture arm file must be owned by this user with mode 0600")
    if metadata.st_size != 0:
        raise TargetKVScaleError("capture arm file must be empty")
    return True


def _record_maxima(
    layer: Any,
    *,
    k_max: float,
    v_max: float,
    k_measurement: str,
) -> None:
    name = _layer_name(layer)
    with _LOCK:
        state = _SAMPLES.setdefault(
            name,
            {"k": [], "v": [], "k_measurement": k_measurement},
        )
        if state["k_measurement"] != k_measurement:
            raise TargetKVScaleError(f"mixed K measurement paths observed for {name}")
        if len(state["k"]) >= SAMPLES_PER_LAYER:
            return
        state["k"].append(k_max)
        state["v"].append(v_max)
        document = _capture_document()
        if document is None or _OUTPUT is None:
            return
        _write_create_only(_OUTPUT, document)
    print(
        f"[qwen-target-kv-scale] calibration ready layers={EXPECTED_LAYERS} "
        f"samples={SAMPLES_PER_LAYER} output={_OUTPUT}",
        flush=True,
    )


def _record_unfused_capture(layer: Any, key: Any, value: Any) -> None:
    _record_maxima(
        layer,
        k_max=float(key.detach().abs().amax().item()),
        v_max=float(value.detach().abs().amax().item()),
        k_measurement="post_rope_absmax",
    )


def _record_fused_capture(layer: Any, key: Any, value: Any, *, is_neox: bool) -> None:
    key_float = key.detach().float()
    if key_float.shape[-1] != 256:
        raise TargetKVScaleError(f"unexpected fused key width: {key_float.shape[-1]}")
    if is_neox:
        first, second = key_float[..., :128], key_float[..., 128:]
    else:
        pairs = key_float.reshape(*key_float.shape[:-1], 128, 2)
        first, second = pairs[..., 0], pairs[..., 1]
    k_bound = float((first.square() + second.square()).sqrt().amax().item())
    _record_maxima(
        layer,
        k_max=k_bound,
        v_max=float(value.detach().abs().amax().item()),
        k_measurement="pre_rope_pair_l2_bound",
    )


def _load_scale_document(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise TargetKVScaleError(f"scale document is not a regular file: {path}")
    metadata = path.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise TargetKVScaleError("scale document must be owned by this user and private")
    if metadata.st_size > 1024 * 1024:
        raise TargetKVScaleError("scale document exceeds the 1 MiB bound")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "qwen-r9700-target-kv-scale-v1":
        raise TargetKVScaleError("unsupported scale document schema")
    if document.get("contract") != _contract():
        raise TargetKVScaleError("scale document contract does not match")
    if document.get("runtime_sha256") != dict(sorted(_RUNTIME_HASHES.items())):
        raise TargetKVScaleError("scale document runtime identity does not match")
    layers = document.get("layers")
    if not isinstance(layers, dict) or set(layers) != EXPECTED_LAYER_NAMES:
        raise TargetKVScaleError("scale document layer map is incomplete or unexpected")
    for name, values in layers.items():
        if not isinstance(values, dict):
            raise TargetKVScaleError(f"invalid scale layer entry: {name}")
        if values.get("samples") != SAMPLES_PER_LAYER:
            raise TargetKVScaleError(f"scale sample count is invalid for {name}")
        if values.get("k_measurement") not in {
            "post_rope_absmax",
            "pre_rope_pair_l2_bound",
        }:
            raise TargetKVScaleError(f"scale K measurement is invalid for {name}")
        for side in ("k", "v"):
            samples = values.get(f"{side}_absmax_samples")
            if not isinstance(samples, list) or len(samples) != SAMPLES_PER_LAYER:
                raise TargetKVScaleError(f"scale samples are incomplete for {name}:{side}")
            observed = [float(sample) for sample in samples]
            maximum = float(values[f"{side}_absmax"])
            if maximum != max(observed):
                raise TargetKVScaleError(f"scale maximum does not match samples for {name}:{side}")
            if float(values[f"{side}_static_scale"]) != _recommended_scale(maximum):
                raise TargetKVScaleError(f"static scale does not match maximum for {name}:{side}")
    return document


def _apply_scale(layer: Any) -> None:
    assert _SCALE_DOCUMENT is not None
    name = _layer_name(layer)
    with _LOCK:
        if name in _APPLIED:
            return
    values = _SCALE_DOCUMENT["layers"].get(name)
    if not isinstance(values, dict):
        raise TargetKVScaleError(f"eligible target layer absent from scale file: {name}")
    multiplier = float(os.environ.get(MULTIPLIER_ENV, "1.0"))
    if multiplier not in {0.75, 1.0, 1.25}:
        raise TargetKVScaleError(f"{MULTIPLIER_ENV} must be 0.75, 1.0, or 1.25")
    k_scale = float(values["k_static_scale"]) * multiplier
    v_scale = float(values["v_static_scale"]) * multiplier
    layer._k_scale.fill_(k_scale)
    layer._v_scale.fill_(v_scale)
    layer._k_scale_float = k_scale
    layer._v_scale_float = v_scale
    layer._k_scale_cpu.fill_(k_scale)
    layer._v_scale_cpu.fill_(v_scale)
    with _LOCK:
        first = name not in _APPLIED
        _APPLIED.add(name)
    if first:
        print(
            f"[qwen-target-kv-scale] applied layer={name} "
            f"k={k_scale:.9g} v={v_scale:.9g} multiplier={multiplier:g}",
            flush=True,
        )


def _patch_module(module: Any) -> None:
    rocm_attention_impl = getattr(module, "RocmAttentionImpl", None)
    if rocm_attention_impl is None:
        raise TargetKVScaleError("ROCm attention module has no RocmAttentionImpl")
    original_kv = rocm_attention_impl.do_kv_cache_update
    original_fused = rocm_attention_impl.do_rope_and_kv_cache_update
    if getattr(original_kv, "_qwen_target_kv_scale", False) and getattr(
        original_fused, "_qwen_target_kv_scale", False
    ):
        return

    @functools.wraps(original_kv)
    def wrapped(self, layer, key, value, kv_cache, slot_mapping):
        mode = os.environ[MODE_ENV]
        if mode == "capture":
            if _is_target_impl(self, capture=True) and _capture_armed():
                _record_unfused_capture(layer, key, value)
        elif _is_target_impl(self, capture=False):
            _apply_scale(layer)
        return original_kv(self, layer, key, value, kv_cache, slot_mapping)

    wrapped._qwen_target_kv_scale = True  # type: ignore[attr-defined]
    rocm_attention_impl.do_kv_cache_update = wrapped

    @functools.wraps(original_fused)
    def wrapped_fused(
        self,
        layer,
        query,
        key,
        value,
        positions,
        cos_sin_cache,
        is_neox,
        kv_cache,
        layer_slot_mapping,
    ):
        mode = os.environ[MODE_ENV]
        if mode == "capture":
            if _is_target_impl(self, capture=True) and _capture_armed():
                _record_fused_capture(layer, key, value, is_neox=bool(is_neox))
        elif _is_target_impl(self, capture=False):
            _apply_scale(layer)
        return original_fused(
            self,
            layer,
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            kv_cache,
            layer_slot_mapping,
        )

    wrapped_fused._qwen_target_kv_scale = True  # type: ignore[attr-defined]
    rocm_attention_impl.do_rope_and_kv_cache_update = wrapped_fused


class _ScaleLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader) -> None:
        self._wrapped = wrapped

    def create_module(self, spec):
        create_module = getattr(self._wrapped, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module) -> None:
        self._wrapped.exec_module(module)
        _patch_module(module)
        print("[qwen-target-kv-scale] hooks ready", flush=True)


class _ScaleFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET_MODULE:
            return None
        try:
            import sys

            sys.meta_path.remove(self)
        except ValueError:
            pass
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise TargetKVScaleError(f"cannot resolve pinned module {fullname}")
        spec.loader = _ScaleLoader(spec.loader)
        return spec


def _install_import_hook() -> str:
    import sys

    loaded = sys.modules.get(_TARGET_MODULE)
    if loaded is not None:
        _patch_module(loaded)
        return "patched"
    if not any(isinstance(finder, _ScaleFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _ScaleFinder())
    return "armed"


def start_target_kv_scale() -> bool:
    """Start the explicit target calibration or application hook."""

    global _CAPTURE_ARM, _STARTED, _OUTPUT, _RUNTIME_HASHES, _SCALE_DOCUMENT
    mode = os.environ.get(MODE_ENV, "off")
    if mode == "off":
        return False
    if mode not in {"capture", "apply"}:
        raise TargetKVScaleError(f"{MODE_ENV} must be off, capture, or apply")
    if _STARTED:
        return True
    _RUNTIME_HASHES = _verify_runtime()
    if mode == "capture":
        output = Path(os.environ.get(OUTPUT_ENV, ""))
        if not output.is_absolute() or output.exists() or output.is_symlink():
            raise TargetKVScaleError(f"{OUTPUT_ENV} must be an absent absolute path")
        if not output.parent.is_dir() or output.parent.is_symlink():
            raise TargetKVScaleError("scale output parent must be a real directory")
        metadata = output.parent.stat()
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise TargetKVScaleError("scale output parent must be owned by this user and private")
        _OUTPUT = output
        arm = Path(os.environ.get(ARM_ENV, ""))
        if not arm.is_absolute() or arm.exists() or arm.is_symlink():
            raise TargetKVScaleError(f"{ARM_ENV} must be an absent absolute path")
        if arm == output or not arm.parent.is_dir() or arm.parent.is_symlink():
            raise TargetKVScaleError("capture arm parent must be a real directory")
        arm_metadata = arm.parent.stat()
        if arm_metadata.st_uid != os.getuid() or stat.S_IMODE(arm_metadata.st_mode) & 0o077:
            raise TargetKVScaleError("capture arm parent must be owned by this user and private")
        _CAPTURE_ARM = arm
    else:
        _SCALE_DOCUMENT = _load_scale_document(Path(os.environ.get(FILE_ENV, "")))
    _STARTED = True
    hook_state = _install_import_hook()
    print(f"[qwen-target-kv-scale] {hook_state} mode={mode}", flush=True)
    return True


__all__ = ["TargetKVScaleError", "start_target_kv_scale"]
