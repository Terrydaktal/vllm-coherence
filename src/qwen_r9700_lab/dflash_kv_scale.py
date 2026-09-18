"""Default-off calibration/application hook for DFlash static FP8 KV scales.

The trained draft checkpoint carries no KV scales, so pinned vLLM falls back to
one scalar value of 1.0 for every K and V cache.  This qualification hook records
only aggregate absolute maxima from the five draft attention layers, or applies
the resulting per-layer scalar scales before the first cache write.  It never
records prompts, token IDs, hidden states, or KV values.
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

MODE_ENV = "QWEN_DFLASH_KV_SCALE_MODE"
OUTPUT_ENV = "QWEN_DFLASH_KV_SCALE_OUTPUT"
FILE_ENV = "QWEN_DFLASH_KV_SCALE_FILE"
MULTIPLIER_ENV = "QWEN_DFLASH_KV_SCALE_MULTIPLIER"

EXPECTED_LAYERS = 5
SAMPLES_PER_LAYER = 16
# The pinned gfx1201 runtime reports torch.float8_e4m3fn for its actual cache
# storage, whose finite representable maximum is 448.  Bind the calibration to
# that observed runtime dtype; FNUZ builds have a different scale convention
# and are rejected by the geometry guard.
FP8_RUNTIME_MAX = 448.0

PINNED_FILES = {
    "rocm_attn": (
        Path(
            "/home/lewis/.local/share/qwen-r9700/runtimes/vllm-rocm/.venv/lib/"
            "python3.12/site-packages/vllm/v1/attention/backends/rocm_attn.py"
        ),
        "c27f7131b66d3c284d8f1fa826cfc5058b94823b1fb708d6d9345a35ae1ed36b",
    ),
    "kv_cache_quant": (
        Path(
            "/home/lewis/.local/share/qwen-r9700/runtimes/vllm-rocm/.venv/lib/"
            "python3.12/site-packages/vllm/model_executor/layers/quantization/kv_cache.py"
        ),
        "cd09817e065d5b0074382fc04a2374ca0ba1c879b3cd125b62f4738e1631ea8a",
    ),
    "cache_writer": (
        Path(
            "/home/lewis/.local/share/qwen-r9700/runtimes/vllm-rocm/.venv/lib/"
            "python3.12/site-packages/vllm/v1/attention/ops/"
            "triton_reshape_and_cache_flash.py"
        ),
        "6cac51475b8c656992a21b2d150acd3a16a95a7ab7d49aab151a3ef13c24b80d",
    ),
    "dflash_model": (
        Path(
            "/home/lewis/.local/share/qwen-r9700/runtimes/vllm-rocm/.venv/lib/"
            "python3.12/site-packages/vllm/model_executor/models/qwen3_dflash2.py"
        ),
        "0f8be7dc032134a25a6bb8341dce5f0391e1ab26a816ab85e003249911f2a1ef",
    ),
    "draft_config": (
        Path("/home/lewis/models/Qwen3.8-27B-DFlash2-W4A16-vllm/config.json"),
        "7cc608bfc20ad2d40ad2b2ceb26f320f783b4b506e511e09f1bbdea69d73426b",
    ),
}

_LOCK = threading.Lock()
_SAMPLES: dict[str, dict[str, Any]] = {}
_APPLIED: set[str] = set()
_SCALE_DOCUMENT: dict[str, Any] | None = None
_OUTPUT: Path | None = None
_RUNTIME_HASHES: dict[str, str] = {}
_STARTED = False
_TARGET_MODULE = "vllm.v1.attention.backends.rocm_attn"


class DFlashKVScaleError(RuntimeError):
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
            raise DFlashKVScaleError(f"pinned {label} is not a regular file: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise DFlashKVScaleError(
                f"pinned {label} drift: expected {expected}, observed {actual}"
            )
        observed[label] = actual
    return observed


def _is_dflash_impl(impl: Any) -> bool:
    return (
        tuple(impl.sliding_window) == (2047, 0)
        and int(impl.num_heads) == 32
        and int(impl.num_kv_heads) == 8
        and int(impl.head_size) == 128
        and str(impl.kv_cache_dtype) == "fp8_e4m3"
        and str(impl.fp8_dtype) == "torch.float8_e4m3fn"
    )


def _layer_name(layer: Any) -> str:
    name = getattr(layer, "layer_name", None)
    if not isinstance(name, str) or not name:
        raise DFlashKVScaleError("eligible DFlash attention layer has no stable name")
    return name


def _recommended_scale(maximum_abs: float) -> float:
    if not 0.0 < maximum_abs < float("inf"):
        raise DFlashKVScaleError(f"invalid captured absolute maximum: {maximum_abs}")
    return maximum_abs / FP8_RUNTIME_MAX


def _capture_document() -> dict[str, Any] | None:
    if len(_SAMPLES) != EXPECTED_LAYERS:
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
        "schema": "qwen-r9700-dflash-kv-scale-v2",
        "contract": {
            "layers": EXPECTED_LAYERS,
            "samples_per_layer": SAMPLES_PER_LAYER,
            "fp8_runtime_dtype": "torch.float8_e4m3fn",
            "fp8_max": FP8_RUNTIME_MAX,
            "sliding_window": [2047, 0],
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_size": 128,
            "kv_cache_dtype": "fp8_e4m3",
        },
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


def _record_maxima(
    layer: Any,
    *,
    k_max: float,
    v_max: float,
    k_measurement: str,
) -> None:
    global _OUTPUT
    name = _layer_name(layer)
    with _LOCK:
        if name not in _SAMPLES and len(_SAMPLES) >= EXPECTED_LAYERS:
            raise DFlashKVScaleError(f"more than {EXPECTED_LAYERS} eligible draft layers observed")
        state = _SAMPLES.setdefault(
            name,
            {"k": [], "v": [], "k_measurement": k_measurement},
        )
        if state["k_measurement"] != k_measurement:
            raise DFlashKVScaleError(f"mixed K measurement paths observed for {name}")
        if len(state["k"]) >= SAMPLES_PER_LAYER:
            return
    with _LOCK:
        state = _SAMPLES[name]
        if len(state["k"]) >= SAMPLES_PER_LAYER:
            return
        state["k"].append(k_max)
        state["v"].append(v_max)
        document = _capture_document()
        if document is None or _OUTPUT is None:
            return
        _write_create_only(_OUTPUT, document)
    print(
        f"[qwen-dflash-kv-scale] calibration ready layers={EXPECTED_LAYERS} "
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
    if key_float.shape[-1] != 128:
        raise DFlashKVScaleError(f"unexpected fused key width: {key_float.shape[-1]}")
    if is_neox:
        first, second = key_float[..., :64], key_float[..., 64:]
    else:
        pairs = key_float.reshape(*key_float.shape[:-1], 64, 2)
        first, second = pairs[..., 0], pairs[..., 1]
    # RoPE is an orthogonal 2-D rotation.  The pair norm is therefore an exact
    # position-independent upper bound on every post-RoPE component.
    k_bound = float((first.square() + second.square()).sqrt().amax().item())
    _record_maxima(
        layer,
        k_max=k_bound,
        v_max=float(value.detach().abs().amax().item()),
        k_measurement="pre_rope_pair_l2_bound",
    )


def _load_scale_document(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise DFlashKVScaleError(f"scale document is not a regular file: {path}")
    metadata = path.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise DFlashKVScaleError("scale document must be owned by this user and private")
    if metadata.st_size > 1024 * 1024:
        raise DFlashKVScaleError("scale document exceeds the 1 MiB bound")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "qwen-r9700-dflash-kv-scale-v2":
        raise DFlashKVScaleError("unsupported scale document schema")
    contract = document.get("contract")
    expected_contract = {
        "layers": EXPECTED_LAYERS,
        "samples_per_layer": SAMPLES_PER_LAYER,
        "fp8_runtime_dtype": "torch.float8_e4m3fn",
        "fp8_max": FP8_RUNTIME_MAX,
        "sliding_window": [2047, 0],
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_size": 128,
        "kv_cache_dtype": "fp8_e4m3",
    }
    if contract != expected_contract:
        raise DFlashKVScaleError("scale document contract does not match")
    if document.get("runtime_sha256") != dict(sorted(_RUNTIME_HASHES.items())):
        raise DFlashKVScaleError("scale document runtime identity does not match")
    layers = document.get("layers")
    if not isinstance(layers, dict) or len(layers) != EXPECTED_LAYERS:
        raise DFlashKVScaleError("scale document layer map is incomplete")
    for name, values in layers.items():
        if not isinstance(name, str) or not isinstance(values, dict):
            raise DFlashKVScaleError("invalid scale layer entry")
        if values.get("samples") != SAMPLES_PER_LAYER:
            raise DFlashKVScaleError(f"scale sample count is invalid for {name}")
        if values.get("k_measurement") not in {
            "post_rope_absmax",
            "pre_rope_pair_l2_bound",
        }:
            raise DFlashKVScaleError(f"scale K measurement is invalid for {name}")
        for side in ("k", "v"):
            samples = values.get(f"{side}_absmax_samples")
            if not isinstance(samples, list) or len(samples) != SAMPLES_PER_LAYER:
                raise DFlashKVScaleError(f"scale samples are incomplete for {name}:{side}")
            observed = [float(sample) for sample in samples]
            maximum = float(values[f"{side}_absmax"])
            if maximum != max(observed):
                raise DFlashKVScaleError(f"scale maximum does not match samples for {name}:{side}")
            expected_scale = _recommended_scale(maximum)
            if float(values[f"{side}_static_scale"]) != expected_scale:
                raise DFlashKVScaleError(f"static scale does not match maximum for {name}:{side}")
    return document


def _apply_scale(layer: Any) -> None:
    assert _SCALE_DOCUMENT is not None
    name = _layer_name(layer)
    with _LOCK:
        if name in _APPLIED:
            return
    values = _SCALE_DOCUMENT["layers"].get(name)
    if not isinstance(values, dict):
        raise DFlashKVScaleError(f"eligible draft layer absent from scale file: {name}")
    multiplier = float(os.environ.get(MULTIPLIER_ENV, "1.0"))
    if not 0.5 <= multiplier <= 2.0:
        raise DFlashKVScaleError(f"{MULTIPLIER_ENV} must be in [0.5, 2.0]")
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
            f"[qwen-dflash-kv-scale] applied layer={name} "
            f"k={k_scale:.9g} v={v_scale:.9g} multiplier={multiplier:g}",
            flush=True,
        )


def _patch_module(module: Any) -> None:
    rocm_attention_impl = getattr(module, "RocmAttentionImpl", None)
    if rocm_attention_impl is None:
        raise DFlashKVScaleError("ROCm attention module has no RocmAttentionImpl")
    original_kv = rocm_attention_impl.do_kv_cache_update
    original_fused = rocm_attention_impl.do_rope_and_kv_cache_update
    if getattr(original_kv, "_qwen_dflash_kv_scale", False) and getattr(
        original_fused, "_qwen_dflash_kv_scale", False
    ):
        return

    @functools.wraps(original_kv)
    def wrapped(self, layer, key, value, kv_cache, slot_mapping):
        if _is_dflash_impl(self):
            mode = os.environ[MODE_ENV]
            if mode == "capture":
                _record_unfused_capture(layer, key, value)
            else:
                _apply_scale(layer)
        return original_kv(self, layer, key, value, kv_cache, slot_mapping)

    wrapped._qwen_dflash_kv_scale = True  # type: ignore[attr-defined]
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
        if _is_dflash_impl(self):
            mode = os.environ[MODE_ENV]
            if mode == "capture":
                _record_fused_capture(layer, key, value, is_neox=bool(is_neox))
            else:
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

    wrapped_fused._qwen_dflash_kv_scale = True  # type: ignore[attr-defined]
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
        print("[qwen-dflash-kv-scale] hooks ready", flush=True)


class _ScaleFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET_MODULE:
            return None
        # Remove this one-shot finder before delegating so a failed import cannot
        # recurse back through it.  Any patch failure propagates through the
        # target import and therefore fails the EngineCore startup closed.
        try:
            import sys

            sys.meta_path.remove(self)
        except ValueError:
            pass
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise DFlashKVScaleError(f"cannot resolve pinned module {fullname}")
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


def start_dflash_kv_scale() -> bool:
    """Start the explicit calibration or application hook."""

    global _STARTED, _OUTPUT, _RUNTIME_HASHES, _SCALE_DOCUMENT
    mode = os.environ.get(MODE_ENV, "off")
    if mode == "off":
        return False
    if mode not in {"capture", "apply"}:
        raise DFlashKVScaleError(f"{MODE_ENV} must be off, capture, or apply")
    if _STARTED:
        return True
    _RUNTIME_HASHES = _verify_runtime()
    if mode == "capture":
        output = Path(os.environ.get(OUTPUT_ENV, ""))
        if not output.is_absolute() or output.exists() or output.is_symlink():
            raise DFlashKVScaleError(f"{OUTPUT_ENV} must be an absent absolute path")
        if not output.parent.is_dir() or output.parent.is_symlink():
            raise DFlashKVScaleError("scale output parent must be a real directory")
        parent_metadata = output.parent.stat()
        if parent_metadata.st_uid != os.getuid() or stat.S_IMODE(parent_metadata.st_mode) & 0o077:
            raise DFlashKVScaleError("scale output parent must be owned by this user and private")
        _OUTPUT = output
    else:
        scale_path = Path(os.environ.get(FILE_ENV, ""))
        _SCALE_DOCUMENT = _load_scale_document(scale_path)
    _STARTED = True
    hook_state = _install_import_hook()
    print(f"[qwen-dflash-kv-scale] {hook_state} mode={mode}", flush=True)
    return True


__all__ = ["DFlashKVScaleError", "start_dflash_kv_scale"]
