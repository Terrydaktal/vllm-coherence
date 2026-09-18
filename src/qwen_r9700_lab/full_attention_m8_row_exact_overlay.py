"""Render an authenticated full-attention M8 serial-row projection overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "urn:qwen-r9700:full-attention-m8-row-exact-overlay:v2"
LM_HEAD_ONLY_SCHEMA = "urn:qwen-r9700:lm-head-m8-row-exact-overlay:v1"
TARGET_MODULE = "vllm.model_executor.models.qwen3_next"
QWEN3_NEXT_RUNTIME_SHA256 = "b1c0d96b5a14be788f5b9c4bf46d38926230297d42cb07452da453be02ae8bbe"
LM_HEAD_TARGET_MODULE = "vllm.model_executor.models.qwen3_5"
QWEN3_5_RUNTIME_SHA256 = "f9d3218305e2ca92919f55fa617c85f925bc9def5ca7f2ae18e9d968fb56c62e"
ENABLE_ENV = "QWEN_FULL_ATTENTION_M8_ROW_EXACT"
REQUIRED_ENV = "QWEN_FULL_ATTENTION_M8_ROW_EXACT_REQUIRED"
SITE_SHA_ENV = "QWEN_FULL_ATTENTION_M8_ROW_EXACT_SITE_SHA256"
RUNTIME_SHA_ENV = "QWEN_FULL_ATTENTION_M8_ROW_EXACT_RUNTIME_SHA256"
LM_HEAD_RUNTIME_SHA_ENV = "QWEN_LM_HEAD_M8_ROW_EXACT_RUNTIME_SHA256"
LM_HEAD_ONLY_ENABLE_ENV = "QWEN_LM_HEAD_M8_ROW_EXACT"
LM_HEAD_ONLY_REQUIRED_ENV = "QWEN_LM_HEAD_M8_ROW_EXACT_REQUIRED"
LM_HEAD_ONLY_SITE_SHA_ENV = "QWEN_LM_HEAD_M8_ROW_EXACT_SITE_SHA256"

_PRODUCTION_M8_ENVIRONMENT = {
    "QWEN_CODING_TURBO_QUEST96": "1",
    "QWEN_DFLASH_GREEDY_LOOP_ESCAPE": "0",
    "QWEN_DFLASH_GREEDY_M8_VERIFIER": "1",
    "QWEN_DFLASH_WHOLE_MODEL_ACCEPTED_REPLAY": "1",
    "QWEN_FIXED_SLOT_TARGET_ONLY_ORACLE": "0",
    "QWEN_LM_HEAD_PREFIX_SERIAL_M8": "1",
    "QWEN_LM_HEAD_SERIAL_M8": "1",
    "QWEN_QUEST_M8_DUALPHASE": "1",
    "QWEN_QUEST_M8_ROW_LOCAL": "1",
}
_OVERLAY_ENVIRONMENT = (
    ENABLE_ENV,
    REQUIRED_ENV,
    SITE_SHA_ENV,
    RUNTIME_SHA_ENV,
    LM_HEAD_RUNTIME_SHA_ENV,
    LM_HEAD_ONLY_ENABLE_ENV,
    LM_HEAD_ONLY_REQUIRED_ENV,
    LM_HEAD_ONLY_SITE_SHA_ENV,
)


class RowExactOverlayError(RuntimeError):
    """The source command, chain, or destination violated the overlay contract."""


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable_private_file(path: Path, label: str) -> bytes:
    path = path.expanduser().absolute()
    try:
        before = path.lstat()
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise RowExactOverlayError(f"cannot read {label}: {error}") from error

    def identity(value: os.stat_result) -> tuple[int, int, int, int]:
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns

    if (
        identity(before) != identity(after)
        or not stat.S_ISREG(before.st_mode)
        or path.is_symlink()
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) & 0o077
    ):
        raise RowExactOverlayError(f"{label} must be one stable owned private regular file")
    return payload


def _split_command(payload: bytes) -> tuple[list[str], list[str]]:
    try:
        tokens = shlex.split(payload.decode())
    except (UnicodeDecodeError, ValueError) as error:
        raise RowExactOverlayError(f"base command is invalid: {error}") from error
    if tokens[:3] != ["exec", "/usr/bin/env", "-i"]:
        raise RowExactOverlayError("base command must use exec /usr/bin/env -i")
    environment: list[str] = []
    index = 3
    while index < len(tokens) and "=" in tokens[index]:
        environment.append(tokens[index])
        index += 1
    if not environment or index == len(tokens):
        raise RowExactOverlayError("base command lacks environment or server argv")
    names = [token.split("=", 1)[0] for token in environment]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise RowExactOverlayError("base command contains an invalid environment")
    return environment, tokens[index:]


def _environment_map(environment: Sequence[str]) -> dict[str, str]:
    return dict(token.split("=", 1) for token in environment)


def _option_json(argv: Sequence[str], name: str) -> Mapping[str, Any]:
    indices = [index for index, token in enumerate(argv) if token == name]
    if len(indices) != 1 or indices[0] + 1 >= len(argv):
        raise RowExactOverlayError(f"base command must contain one exact {name}")
    try:
        value = json.loads(argv[indices[0] + 1])
    except json.JSONDecodeError as error:
        raise RowExactOverlayError(f"base command {name} is invalid JSON") from error
    if not isinstance(value, dict):
        raise RowExactOverlayError(f"base command {name} must be a JSON object")
    return value


def _validate_production_m8(environment: Mapping[str, str], argv: Sequence[str]) -> None:
    observed = {name: environment.get(name) for name in _PRODUCTION_M8_ENVIRONMENT}
    if observed != _PRODUCTION_M8_ENVIRONMENT:
        raise RowExactOverlayError(
            "base command is not the production speculative-M8, Quest96, escape-off arm"
        )
    speculative = _option_json(argv, "--speculative-config")
    if speculative.get("method") != "dflash" or speculative.get("num_speculative_tokens") != 7:
        raise RowExactOverlayError("production M8 requires exact DFlash7 geometry")
    max_num_seqs = [
        argv[index + 1] for index, token in enumerate(argv[:-1]) if token == "--max-num-seqs"
    ]
    if max_num_seqs != ["1"]:
        raise RowExactOverlayError("production M8 requires --max-num-seqs 1")


def _find_chained_site(pythonpath: str) -> Path:
    entries = pythonpath.split(":")
    if not entries or any(not entry or not Path(entry).is_absolute() for entry in entries):
        raise RowExactOverlayError("base command lacks an absolute PYTHONPATH chain")
    for entry in entries:
        root = Path(entry)
        module = root / "sitecustomize.py"
        package = root / "sitecustomize"
        if package.exists() or package.is_symlink():
            raise RowExactOverlayError("package-form sitecustomize chains are unsupported")
        if module.exists() or module.is_symlink():
            return module
    raise RowExactOverlayError("base command PYTHONPATH has no chained sitecustomize")


def _authenticate_chained_site(
    environment: Mapping[str, str], chained_site: Path, chained_sha256: str
) -> list[str]:
    direct_claims = sorted(
        name
        for name, value in environment.items()
        if name.endswith("SITE_SHA256") and value == chained_sha256
    )
    if direct_claims:
        return direct_claims

    manifest_value = environment.get("QWEN_CODING_TURBO_ASSURANCE_MANIFEST", "")
    if manifest_value:
        manifest_path = Path(manifest_value)
        manifest_payload = _stable_private_file(manifest_path, "assurance artifact manifest")
        try:
            manifest = json.loads(manifest_payload)
        except json.JSONDecodeError as error:
            raise RowExactOverlayError("assurance artifact manifest is invalid JSON") from error
        relative = "assurance/capture_site/sitecustomize.py"
        entries = manifest.get("files") if isinstance(manifest, dict) else None
        claims = [
            entry
            for entry in entries or ()
            if isinstance(entry, dict) and entry.get("path") == relative
        ]
        expected_site = manifest_path.parent / "files" / relative
        if (
            not isinstance(entries, list)
            or len(claims) != 1
            or claims[0].get("sha256") != chained_sha256
            or expected_site.absolute() != chained_site.absolute()
        ):
            raise RowExactOverlayError(
                "assurance artifact manifest does not authenticate the active chained site"
            )
        return ["QWEN_CODING_TURBO_ASSURANCE_MANIFEST@" + _digest(manifest_payload)]
    raise RowExactOverlayError(
        "base command does not authenticate its active chained sitecustomize"
    )


def _site_source(
    destination: Path,
    *,
    chained_site: Path,
    chained_site_sha256: str,
    chained_pythonpath: str,
    prior_site_sha256: str | None,
    lm_head_only: bool,
) -> bytes:
    outer_pythonpath = f"{destination}:{chained_pythonpath}"
    return f'''"""Authenticated full-attention M8 serial-row projection bootstrap."""
import functools
import hashlib
import os
import runpy
import stat
import sys
from pathlib import Path

_ROOT = Path({str(destination)!r})
_SELF = _ROOT / "sitecustomize.py"
_CHAIN = Path({str(chained_site)!r})
_CHAIN_SHA256 = {chained_site_sha256!r}
_CHAIN_PYTHONPATH = {chained_pythonpath!r}
_OUTER_PYTHONPATH = {outer_pythonpath!r}
_TARGET = {TARGET_MODULE!r}
_RUNTIME_SHA256 = {QWEN3_NEXT_RUNTIME_SHA256!r}
_LM_HEAD_TARGET = {LM_HEAD_TARGET_MODULE!r}
_LM_HEAD_RUNTIME_SHA256 = {QWEN3_5_RUNTIME_SHA256!r}
_METHOD_MARKER = "_qwen_full_attention_m8_row_exact_site_sha256"
_LM_HEAD_METHOD_MARKER = "_qwen_lm_head_partial_m8_row_exact_site_sha256"
_PRIOR_SITE_SHA256 = {prior_site_sha256!r}
_ENABLE_ATTENTION = {not lm_head_only!r}
_ENABLE_ENV = {LM_HEAD_ONLY_ENABLE_ENV if lm_head_only else ENABLE_ENV!r}
_REQUIRED_ENV = {LM_HEAD_ONLY_REQUIRED_ENV if lm_head_only else REQUIRED_ENV!r}
_SITE_SHA_ENV = {LM_HEAD_ONLY_SITE_SHA_ENV if lm_head_only else SITE_SHA_ENV!r}

def _stable_digest(path, label, private=False):
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise RuntimeError(f"{{label}} identity is unsafe")
    if private and (before.st_uid != os.getuid() or stat.S_IMODE(before.st_mode) & 0o077):
        raise RuntimeError(f"{{label}} is not owned and private")
    payload = path.read_bytes()
    after = path.lstat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise RuntimeError(f"{{label}} changed during authentication")
    return hashlib.sha256(payload).hexdigest()

if (
	os.environ.get(_ENABLE_ENV) != "1"
	or os.environ.get(_REQUIRED_ENV) != "1"
):
    raise RuntimeError("M8 row-exact identity is absent")
if _ENABLE_ATTENTION and os.environ.get({RUNTIME_SHA_ENV!r}) != _RUNTIME_SHA256:
    raise RuntimeError("full-attention M8 row-exact runtime identity differs")
if os.environ.get({LM_HEAD_RUNTIME_SHA_ENV!r}) != _LM_HEAD_RUNTIME_SHA256:
    raise RuntimeError("LM-head M8 row-exact runtime identity differs")
if os.environ.get("QWEN_DFLASH_GREEDY_LOOP_ESCAPE") != "0":
    raise RuntimeError("full-attention M8 row-exact requires loop escape disabled")
if os.environ.get("QWEN_DFLASH_GREEDY_M8_VERIFIER") != "1":
    raise RuntimeError("full-attention M8 row-exact requires the production M8 verifier")
if os.environ.get("QWEN_FIXED_SLOT_TARGET_ONLY_ORACLE") != "0":
    raise RuntimeError("full-attention M8 row-exact forbids target-only execution")
if os.environ.get("PYTHONPATH") != _OUTER_PYTHONPATH:
    raise RuntimeError("full-attention M8 row-exact PYTHONPATH differs")
_SITE_SHA256 = _stable_digest(_SELF, "row-exact site", private=True)
if _SITE_SHA256 != os.environ.get(_SITE_SHA_ENV):
    raise RuntimeError("M8 row-exact site SHA256 mismatch")
if _stable_digest(_CHAIN, "chained site", private=True) != _CHAIN_SHA256:
    raise RuntimeError("full-attention M8 row-exact chained-site SHA256 mismatch")
if (_ENABLE_ATTENTION and _TARGET in sys.modules) or _LM_HEAD_TARGET in sys.modules:
    raise RuntimeError("Qwen model code was imported before row-exact overlay installation")

os.environ["PYTHONPATH"] = _CHAIN_PYTHONPATH
if _PRIOR_SITE_SHA256 is not None:
    if os.environ.get(_SITE_SHA_ENV) != _SITE_SHA256:
        raise RuntimeError("outer row-exact site declaration changed before chaining")
    os.environ[_SITE_SHA_ENV] = _PRIOR_SITE_SHA256
try:
    runpy.run_path(str(_CHAIN), run_name="_qwen_full_attention_m8_row_exact_chain")
    if os.environ.get("PYTHONPATH") != _CHAIN_PYTHONPATH:
        raise RuntimeError("chained site changed PYTHONPATH")
finally:
    if _PRIOR_SITE_SHA256 is not None:
        os.environ[_SITE_SHA_ENV] = _SITE_SHA256
    os.environ["PYTHONPATH"] = _OUTER_PYTHONPATH
if (_ENABLE_ATTENTION and _TARGET in sys.modules) or _LM_HEAD_TARGET in sys.modules:
    raise RuntimeError("chained site imported Qwen model code before row-exact hook registration")

import combined_runtime_patch as retained  # noqa: E402

finder_type = getattr(retained, "_Finder", None)
patches = getattr(finder_type, "_PATCHES", None)
if not isinstance(finder_type, type) or type(patches) is not dict:
    raise RuntimeError("combined-runtime finder contract changed")
active_finders = [value for value in sys.meta_path if isinstance(value, finder_type)]
if len(active_finders) != 1:
    raise RuntimeError("combined-runtime finder is not installed exactly once")
previous_patch = patches.get(_TARGET)
if previous_patch is not None and not callable(previous_patch):
    raise RuntimeError("existing qwen3_next patch is not callable")
previous_lm_head_patch = patches.get(_LM_HEAD_TARGET)
if previous_lm_head_patch is not None and not callable(previous_lm_head_patch):
    raise RuntimeError("existing qwen3_5 patch is not callable")
previous_marker = (
    getattr(previous_patch, _METHOD_MARKER, None) if previous_patch is not None else None
)
previous_lm_head_marker = (
    getattr(previous_lm_head_patch, _LM_HEAD_METHOD_MARKER, None)
    if previous_lm_head_patch is not None
    else None
)
if previous_marker is not None and previous_marker not in (
    _SITE_SHA256, _PRIOR_SITE_SHA256
):
    raise RuntimeError("a different row-exact patch is already registered")
if previous_lm_head_marker is not None and previous_lm_head_marker not in (
    _SITE_SHA256, _PRIOR_SITE_SHA256
):
    raise RuntimeError("a different LM-head row-exact patch is already registered")

def _shape(tensor, label):
    try:
        return tuple(int(value) for value in tensor.shape)
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError(label + " has an invalid tensor shape") from error

def _patch_qwen3_next(module):
    source_value = getattr(module, "__file__", "")
    source = Path(source_value)
    if not source.is_absolute() or _stable_digest(source, "qwen3_next runtime") != _RUNTIME_SHA256:
        raise RuntimeError("qwen3_next runtime SHA256 mismatch")
    if getattr(module, "__name__", None) != _TARGET:
        raise RuntimeError("qwen3_next module identity changed")
    cls = getattr(module, "Qwen3NextAttention", None)
    current = getattr(cls, "_project_qkv_gate", None)
    current_marker = getattr(current, _METHOD_MARKER, None)
    if current_marker is not None:
        if current_marker != _SITE_SHA256:
            raise RuntimeError("qwen3_next already carries a different row-exact wrapper")
        return
    if previous_patch is not None:
        previous_patch(module)
    cls = getattr(module, "Qwen3NextAttention", None)
    original = getattr(cls, "_project_qkv_gate", None)
    if not isinstance(cls, type) or not callable(original):
        raise RuntimeError("qwen3_next projection ABI changed")
    inherited_marker = getattr(original, _METHOD_MARKER, None)
    if inherited_marker is not None:
        if inherited_marker != _SITE_SHA256:
            raise RuntimeError("chained qwen3_next patch has a conflicting row-exact marker")
        return
    torch_module = getattr(module, "torch", None)
    tensor_type = getattr(torch_module, "Tensor", None)
    concatenate = getattr(torch_module, "cat", None)
    if not isinstance(tensor_type, type) or not callable(concatenate):
        raise RuntimeError("qwen3_next torch tensor ABI changed")

    @functools.wraps(original)
    def row_exact(self, qkv, positions):
        if not isinstance(qkv, tensor_type) or not isinstance(positions, tensor_type):
            return original(self, qkv, positions)
        try:
            qkv_shape = tuple(int(value) for value in qkv.shape)
            position_shape = tuple(int(value) for value in positions.shape)
        except (AttributeError, TypeError, ValueError):
            return original(self, qkv, positions)
        rows = qkv_shape[0] if len(qkv_shape) == 2 else 0
        one_dimensional_positions = position_shape == (rows,)
        mrope_positions = position_shape == (3, rows)
        if not 2 <= rows <= 8 or not (
            one_dimensional_positions or mrope_positions
        ):
            return original(self, qkv, positions)

        components = [[], [], [], []]
        component_kinds = [None, None, None, None]
        component_shapes = [None, None, None, None]
        component_dtypes = [None, None, None, None]
        component_devices = [None, None, None, None]
        for row in range(rows):
            row_positions = (
                positions[row : row + 1]
                if one_dimensional_positions
                else positions[:, row : row + 1]
            )
            result = original(self, qkv[row : row + 1], row_positions)
            if type(result) is not tuple or len(result) != 4:
                raise RuntimeError("row-exact projection must return one exact four-tuple")
            for component_index, value in enumerate(result):
                if value is None:
                    if component_index != 3:
                        raise RuntimeError("row-exact Q/K/V components may not be None")
                    if component_kinds[component_index] not in (None, "none"):
                        raise RuntimeError("row-exact gate mixed None and tensor results")
                    component_kinds[component_index] = "none"
                    continue
                if not isinstance(value, tensor_type):
                    raise RuntimeError("row-exact projection returned a non-tensor component")
                if component_kinds[component_index] == "none":
                    raise RuntimeError("row-exact gate mixed None and tensor results")
                component_kinds[component_index] = "tensor"
                shape = _shape(value, "row-exact projection component")
                if not shape or shape[0] != 1:
                    raise RuntimeError("row-exact projection component is not one row")
                dtype = getattr(value, "dtype", None)
                device = getattr(value, "device", None)
                if dtype is None or device is None:
                    raise RuntimeError("row-exact projection component lacks dtype/device identity")
                tail = shape[1:]
                if component_shapes[component_index] is None:
                    component_shapes[component_index] = tail
                    component_dtypes[component_index] = dtype
                    component_devices[component_index] = device
                elif (
                    component_shapes[component_index] != tail
                    or component_dtypes[component_index] != dtype
                    or component_devices[component_index] != device
                ):
                    raise RuntimeError(
                        "row-exact projection component identity changed across rows"
                    )
                components[component_index].append(value)

        outputs = []
        for component_index, values in enumerate(components):
            if component_kinds[component_index] == "none":
                if component_index != 3 or values:
                    raise RuntimeError("row-exact None component contract changed")
                outputs.append(None)
                continue
            if component_kinds[component_index] != "tensor" or len(values) != rows:
                raise RuntimeError("row-exact projection component coverage is incomplete")
            output = concatenate(tuple(values), dim=0)
            if not isinstance(output, tensor_type):
                raise RuntimeError("row-exact concatenation returned a non-tensor")
            expected_shape = (rows, *component_shapes[component_index])
            if (
                _shape(output, "row-exact concatenated component") != expected_shape
                or getattr(output, "dtype", None) != component_dtypes[component_index]
                or getattr(output, "device", None) != component_devices[component_index]
            ):
                raise RuntimeError("row-exact concatenated component identity changed")
            outputs.append(output)
        return outputs[0], outputs[1], outputs[2], outputs[3]

    setattr(row_exact, _METHOD_MARKER, _SITE_SHA256)
    cls._project_qkv_gate = row_exact
    print("[qwen-full-attention-m8-row-exact] M2..M8 serial row projection armed", flush=True)

if _ENABLE_ATTENTION and previous_marker is None:
    setattr(_patch_qwen3_next, _METHOD_MARKER, _SITE_SHA256)
    patches[_TARGET] = _patch_qwen3_next
elif _ENABLE_ATTENTION and patches.get(_TARGET) is not previous_patch:
    raise RuntimeError("row-exact finder patch identity changed during installation")

def _patch_qwen3_5(module):
    source_value = getattr(module, "__file__", "")
    source = Path(source_value)
    if (
        not source.is_absolute()
        or _stable_digest(source, "qwen3_5 runtime") != _LM_HEAD_RUNTIME_SHA256
    ):
        raise RuntimeError("qwen3_5 runtime SHA256 mismatch")
    if getattr(module, "__name__", None) != _LM_HEAD_TARGET:
        raise RuntimeError("qwen3_5 module identity changed")
    cls = getattr(module, "Qwen3_5ForCausalLMBase", None)
    current = getattr(cls, "compute_logits", None)
    current_marker = getattr(current, _LM_HEAD_METHOD_MARKER, None)
    if current_marker is not None:
        if current_marker != _SITE_SHA256:
            raise RuntimeError("qwen3_5 already carries a different LM-head row-exact wrapper")
        return
    if previous_lm_head_patch is not None:
        previous_lm_head_patch(module)
    cls = getattr(module, "Qwen3_5ForCausalLMBase", None)
    original = getattr(cls, "compute_logits", None)
    if not isinstance(cls, type) or not callable(original):
        raise RuntimeError("qwen3_5 LM-head ABI changed")
    inherited_marker = getattr(original, _LM_HEAD_METHOD_MARKER, None)
    if inherited_marker is not None:
        if inherited_marker != _SITE_SHA256:
            raise RuntimeError("chained qwen3_5 patch has a conflicting LM-head marker")
        return
    torch_module = getattr(module, "torch", None)
    tensor_type = getattr(torch_module, "Tensor", None)
    concatenate = getattr(torch_module, "cat", None)
    if not isinstance(tensor_type, type) or not callable(concatenate):
        raise RuntimeError("qwen3_5 torch tensor ABI changed")

    @functools.wraps(original)
    def row_exact_logits(self, hidden_states):
        if not isinstance(hidden_states, tensor_type):
            return original(self, hidden_states)
        try:
            shape = tuple(int(value) for value in hidden_states.shape)
        except (AttributeError, TypeError, ValueError):
            return original(self, hidden_states)
        rows = shape[0] if len(shape) == 2 else 0
        if not 2 <= rows <= 8:
            return original(self, hidden_states)
        outputs = []
        tail = None
        dtype = None
        device = None
        for row in range(rows):
            result = original(self, hidden_states[row : row + 1])
            if not isinstance(result, tensor_type):
                raise RuntimeError("row-exact LM head returned a non-tensor result")
            result_shape = _shape(result, "row-exact LM-head result")
            if not result_shape or result_shape[0] != 1:
                raise RuntimeError("row-exact LM-head result is not one row")
            current_tail = result_shape[1:]
            current_dtype = getattr(result, "dtype", None)
            current_device = getattr(result, "device", None)
            if current_dtype is None or current_device is None:
                raise RuntimeError("row-exact LM-head result lacks dtype/device identity")
            if tail is None:
                tail = current_tail
                dtype = current_dtype
                device = current_device
            elif tail != current_tail or dtype != current_dtype or device != current_device:
                raise RuntimeError("row-exact LM-head result identity changed across rows")
            outputs.append(result)
        output = concatenate(tuple(outputs), dim=0)
        if (
            not isinstance(output, tensor_type)
            or _shape(output, "row-exact LM-head output") != (rows, *tail)
            or getattr(output, "dtype", None) != dtype
            or getattr(output, "device", None) != device
        ):
            raise RuntimeError("row-exact LM-head concatenation changed identity")
        return output

    setattr(row_exact_logits, _LM_HEAD_METHOD_MARKER, _SITE_SHA256)
    cls.compute_logits = row_exact_logits
    print("[qwen-lm-head-m8-row-exact] M2..M8 serial row projection armed", flush=True)

if previous_lm_head_marker is None:
    setattr(_patch_qwen3_5, _LM_HEAD_METHOD_MARKER, _SITE_SHA256)
    patches[_LM_HEAD_TARGET] = _patch_qwen3_5
elif patches.get(_LM_HEAD_TARGET) is not previous_lm_head_patch:
    raise RuntimeError("LM-head row-exact finder patch identity changed during installation")
'''.encode()


def _write_exclusive(path: Path, payload: bytes, mode: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def render(args: argparse.Namespace) -> Mapping[str, Any]:
    destination = args.destination.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise RowExactOverlayError("destination is create-only")
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_base_sha256):
        raise RowExactOverlayError("expected base command SHA256 is malformed")
    expected_runtime = getattr(args, "expected_qwen3_next_sha256", QWEN3_NEXT_RUNTIME_SHA256)
    if expected_runtime != QWEN3_NEXT_RUNTIME_SHA256:
        raise RowExactOverlayError("qwen3_next runtime SHA256 differs from the pinned source")

    base_payload = _stable_private_file(args.base_command, "base command")
    base_sha256 = _digest(base_payload)
    if base_sha256 != args.expected_base_sha256:
        raise RowExactOverlayError("base command SHA256 mismatch")
    environment, argv = _split_command(base_payload)
    env = _environment_map(environment)
    lm_head_only = bool(getattr(args, "lm_head_only", False))
    present_overlay = {name for name in _OVERLAY_ENVIRONMENT if name in env}
    prior_site_sha256: str | None = None
    if present_overlay:
        if lm_head_only:
            raise RowExactOverlayError("base command already has a row-exact overlay")
        expected_prior = {ENABLE_ENV, REQUIRED_ENV, SITE_SHA_ENV, RUNTIME_SHA_ENV}
        if (
            present_overlay != expected_prior
            or env.get(ENABLE_ENV) != "1"
            or env.get(REQUIRED_ENV) != "1"
            or env.get(RUNTIME_SHA_ENV) != QWEN3_NEXT_RUNTIME_SHA256
            or not re.fullmatch(r"[0-9a-f]{64}", env.get(SITE_SHA_ENV, ""))
        ):
            raise RowExactOverlayError("base command has an incompatible row-exact overlay")
        prior_site_sha256 = env[SITE_SHA_ENV]
    _validate_production_m8(env, argv)

    chained_pythonpath = env.get("PYTHONPATH", "")
    chained_site = _find_chained_site(chained_pythonpath)
    chained_payload = _stable_private_file(chained_site, "chained sitecustomize")
    chained_sha256 = _digest(chained_payload)
    chained_sha_claims = _authenticate_chained_site(env, chained_site, chained_sha256)

    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent_metadata = destination.parent.lstat()
    if (
        not stat.S_ISDIR(parent_metadata.st_mode)
        or destination.parent.is_symlink()
        or parent_metadata.st_uid != os.getuid()
        or stat.S_IMODE(parent_metadata.st_mode) & 0o077
    ):
        raise RowExactOverlayError("destination parent must be one owned private directory")
    staging = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise RowExactOverlayError("staging destination already exists")
    staging.mkdir(mode=0o700)
    try:
        site_payload = _site_source(
            destination,
            chained_site=chained_site,
            chained_site_sha256=chained_sha256,
            chained_pythonpath=chained_pythonpath,
            prior_site_sha256=prior_site_sha256,
            lm_head_only=lm_head_only,
        )
        site_sha256 = _digest(site_payload)
        additions = (
            {
                LM_HEAD_ONLY_ENABLE_ENV: "1",
                LM_HEAD_ONLY_REQUIRED_ENV: "1",
                LM_HEAD_RUNTIME_SHA_ENV: QWEN3_5_RUNTIME_SHA256,
                LM_HEAD_ONLY_SITE_SHA_ENV: site_sha256,
            }
            if lm_head_only
            else {
                ENABLE_ENV: "1",
                REQUIRED_ENV: "1",
                RUNTIME_SHA_ENV: QWEN3_NEXT_RUNTIME_SHA256,
                LM_HEAD_RUNTIME_SHA_ENV: QWEN3_5_RUNTIME_SHA256,
                SITE_SHA_ENV: site_sha256,
            }
        )
        rewritten_environment = list(environment)
        for index, token in enumerate(rewritten_environment):
            if token.startswith("PYTHONPATH="):
                rewritten_environment[index] = f"PYTHONPATH={destination}:{chained_pythonpath}"
            else:
                name = token.split("=", 1)[0]
                if name in additions:
                    rewritten_environment[index] = f"{name}={additions[name]}"
        else:  # pragma: no cover - guarded by _find_chained_site
            if not any(token.startswith("PYTHONPATH=") for token in rewritten_environment):
                raise RowExactOverlayError("base command lacks PYTHONPATH")
        rewritten_environment.extend(
            f"{name}={value}" for name, value in additions.items() if name not in env
        )
        command_payload = (
            shlex.join(["exec", "/usr/bin/env", "-i", *rewritten_environment, *argv]) + "\n"
        ).encode()

        manifest = {
            "schema": LM_HEAD_ONLY_SCHEMA if lm_head_only else SCHEMA,
            "classification": (
                "non_promotable_lm_head_partial_width_repair_candidate"
                if lm_head_only
                else "non_promotable_full_attention_m8_diagnostic"
            ),
            "promotable": False,
            "base_command": {
                "path": str(args.base_command.expanduser().absolute()),
                "sha256": base_sha256,
            },
            "chained_site": {
                "path": str(chained_site),
                "sha256": chained_sha256,
                "authenticated_by_environment": chained_sha_claims,
            },
            "command_sha256": _digest(command_payload),
            "environment_delta": {
                "PYTHONPATH": f"{destination}:{chained_pythonpath}",
                **additions,
            },
            "patch_contract": {
                "retained_prior_row_exact_site_sha256": prior_site_sha256,
                "supported_verification_widths": list(range(2, 9)),
                "full_attention": (
                    None
                    if lm_head_only
                    else {
                        "module": TARGET_MODULE,
                        "method": "Qwen3NextAttention._project_qkv_gate",
                        "position_shapes": ["[width]", "[3,width]"],
                        "other_shapes_call_original_once": True,
                    }
                ),
                "lm_head": {
                    "module": LM_HEAD_TARGET_MODULE,
                    "method": "Qwen3_5ForCausalLMBase.compute_logits",
                    "other_shapes_call_original_once": True,
                },
                "serial_row_order": list(range(8)),
            },
            "qwen3_next_runtime_sha256": QWEN3_NEXT_RUNTIME_SHA256,
            "qwen3_5_runtime_sha256": QWEN3_5_RUNTIME_SHA256,
            "site_sha256": site_sha256,
        }
        manifest_payload = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
        _write_exclusive(staging / "sitecustomize.py", site_payload, 0o600)
        _write_exclusive(staging / "command.sh", command_payload, 0o700)
        _write_exclusive(staging / "manifest.json", manifest_payload, 0o600)
        if destination.exists() or destination.is_symlink():
            raise RowExactOverlayError("destination appeared during create-only rendering")
        staging.rename(destination)
        return {**manifest, "manifest_sha256": _digest(manifest_payload)}
    except BaseException:
        if staging.exists() and staging.parent == destination.parent:
            shutil.rmtree(staging)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen-full-attention-m8-row-exact-overlay")
    parser.add_argument("--base-command", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument(
        "--expected-qwen3-next-sha256",
        default=QWEN3_NEXT_RUNTIME_SHA256,
        help="must equal the pinned qwen3_next runtime source SHA256",
    )
    parser.add_argument(
        "--lm-head-only",
        action="store_true",
        help="retain the exact-K attention path and add only serial M2..M8 LM-head rows",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = render(_parser().parse_args(argv))
    except (OSError, RowExactOverlayError) as error:
        print(f"qwen-full-attention-m8-row-exact-overlay: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
