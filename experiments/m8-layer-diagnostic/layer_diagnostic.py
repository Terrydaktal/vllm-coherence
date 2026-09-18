"""Qualification-only M1/M8 decoder-boundary capture.

The capture is deliberately inactive unless its complete authenticated environment
contract is present.  It records hashes only: no prompt text, token IDs, or tensor
payloads leave the worker.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.abc
import importlib.util
import json
import os
import stat
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

_MODULE = "vllm.model_executor.models.qwen3_next"
_ENABLED = "QWEN_M8_LAYER_DIAGNOSTIC"
_REQUIRED = "QWEN_M8_LAYER_DIAGNOSTIC_REQUIRED"
_SELF_SHA = "QWEN_M8_LAYER_DIAGNOSTIC_MODULE_SHA256"
_OUTPUT_ROOT = "QWEN_M8_LAYER_DIAGNOSTIC_OUTPUT_ROOT"
_POSITIONS = "QWEN_M8_LAYER_DIAGNOSTIC_POSITIONS"
_TENSOR_CAPSULE = "QWEN_M8_LAYER_DIAGNOSTIC_TENSOR_CAPSULE"
_QUEST_PAGE_COMPARE = "QWEN_M8_LAYER_DIAGNOSTIC_QUEST_PAGE_COMPARE"
_QUEST_MODULE = "quest_vllm_attention"
_MAX_POSITION_COUNT = 64
_LOCK = threading.Lock()
_STREAM: Any | None = None
_STREAM_PATH: Path | None = None
_QUEST_STREAM: Any | None = None
_QUEST_STREAM_PATH: Path | None = None
_QUEST_ATTENTION_REFERENCE: dict[str, Any] | None = None
_PASS_INDEX = 0
_PENDING: list[dict[str, Any]] = []
_CAPSULE_WRITTEN = False


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positions() -> frozenset[int]:
    raw = os.environ.get(_POSITIONS, "")
    try:
        values = tuple(int(value) for value in raw.split(","))
    except ValueError as error:
        raise RuntimeError("M8 layer diagnostic positions are not integers") from error
    if (
        not values
        or len(values) > _MAX_POSITION_COUNT
        or tuple(sorted(set(values))) != values
        or any(value < 1 for value in values)
    ):
        raise RuntimeError("M8 layer diagnostic positions are invalid")
    return frozenset(values)


_SELECTED_POSITIONS = _positions()


def _position_rows(positions: Any, rows: int) -> list[list[int]]:
    raw = positions.detach().cpu().tolist()
    if positions.ndim == 1:
        if len(raw) != rows:
            raise RuntimeError("M8 layer diagnostic position count differs from rows")
        return [[int(value)] for value in raw]
    if positions.ndim == 2 and int(positions.shape[-1]) == rows:
        return [
            [int(raw[axis][row]) for axis in range(int(positions.shape[0]))] for row in range(rows)
        ]
    raise RuntimeError(f"unsupported decoder position shape {tuple(positions.shape)}")


def _tensor_sha256(value: Any) -> str:
    import torch

    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _tensor_layout(value: Any) -> dict[str, Any]:
    """Capture the physical view contract hidden by a logical tensor hash."""

    shape = [int(item) for item in value.shape]
    stride = [int(item) for item in value.stride()]
    if any(item < 0 for item in stride):
        raise RuntimeError("M8 layer diagnostic does not support negative tensor strides")
    element_size = int(value.element_size())
    storage_offset = int(value.storage_offset())
    maximum_element = storage_offset
    for extent, step in zip(shape, stride, strict=True):
        if extent:
            maximum_element += (extent - 1) * step
    storage = value.untyped_storage()
    return {
        "contiguous": bool(value.is_contiguous()),
        "data_ptr": int(value.data_ptr()),
        "device": str(value.device),
        "dtype": str(value.dtype),
        "element_size": element_size,
        "shape": shape,
        "storage_byte_interval": [
            storage_offset * element_size,
            (maximum_element + 1) * element_size,
        ],
        "storage_data_ptr": int(storage.data_ptr()),
        "storage_nbytes": int(storage.nbytes()),
        "storage_offset": storage_offset,
        "stride": stride,
    }


def _row_layouts(value: Any, selected: list[tuple[int, int]]) -> dict[int, dict[str, Any]]:
    return {row: _tensor_layout(value[row : row + 1]) for row, _position in selected}


def _storage_relation(left: Any, right: Any) -> dict[str, Any]:
    left_layout = _tensor_layout(left)
    right_layout = _tensor_layout(right)
    same_storage = left_layout["storage_data_ptr"] == right_layout["storage_data_ptr"]
    left_interval = left_layout["storage_byte_interval"]
    right_interval = right_layout["storage_byte_interval"]
    overlaps = bool(
        same_storage
        and max(left_interval[0], right_interval[0]) < min(left_interval[1], right_interval[1])
    )
    return {"overlaps": overlaps, "same_storage": same_storage}


def _capsule_spec() -> tuple[int, int, int] | None:
    raw = os.environ.get(_TENSOR_CAPSULE, "")
    if not raw:
        return None
    try:
        values = tuple(int(item) for item in raw.split(":"))
    except ValueError as error:
        raise RuntimeError("M8 tensor capsule must be PASS:LAYER:POSITION") from error
    if (
        len(values) != 3
        or values[0] < 0
        or not 0 <= values[1] < 64
        or values[2] not in _SELECTED_POSITIONS
    ):
        raise RuntimeError("M8 tensor capsule selector is outside the diagnostic contract")
    return values


_CAPSULE_SPEC = _capsule_spec()


def _write_tensor_capsule(path: Path, payload: dict[str, Any]) -> str:
    """Durably preserve private exact tensors for one selected counterexample."""

    import torch

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return _digest(path)


def _selected_rows(position_rows: list[list[int]]) -> list[tuple[int, int]]:
    selected: list[tuple[int, int]] = []
    for row, axes in enumerate(position_rows):
        matches = sorted(set(axes) & _SELECTED_POSITIONS)
        if len(matches) > 1:
            raise RuntimeError("one decoder row matched multiple diagnostic positions")
        if matches:
            selected.append((row, matches[0]))
    return selected


def _detail_hook(
    detail: dict[str, dict[int, str]],
    label: str,
    selected: list[tuple[int, int]],
):
    def hook(_module: Any, _arguments: Any, result: Any) -> None:
        primary = result[0] if isinstance(result, tuple) else result
        if not hasattr(primary, "shape"):
            raise RuntimeError(f"diagnostic module {label} returned no tensor")
        detail[f"{label}_sha256"] = {
            row: _tensor_sha256(primary[row : row + 1]) for row, _position in selected
        }
        if isinstance(result, tuple):
            if len(result) != 2:
                raise RuntimeError(f"diagnostic module {label} returned an invalid tuple")
            if result[1] is not None:
                if not hasattr(result[1], "shape"):
                    raise RuntimeError(
                        f"diagnostic module {label} returned a non-tensor secondary value"
                    )
                detail[f"{label}_secondary_sha256"] = {
                    row: _tensor_sha256(result[1][row : row + 1]) for row, _position in selected
                }

    return hook


def _detail_input_hook(
    detail: dict[str, dict[int, str]],
    label: str,
    selected: list[tuple[int, int]],
):
    def hook(_module: Any, arguments: Any) -> None:
        if not isinstance(arguments, tuple) or not arguments:
            raise RuntimeError(f"diagnostic module {label} received no positional tensor")
        primary = arguments[0]
        if not hasattr(primary, "shape"):
            raise RuntimeError(f"diagnostic module {label} received no tensor")
        detail[f"{label}_sha256"] = {
            row: _tensor_sha256(primary[row : row + 1]) for row, _position in selected
        }

    return hook


def _state_source_detail(
    detail: dict[str, dict[int, str]],
    label: str,
    selected: list[tuple[int, int]],
    attention: Any,
    state_indices: Any,
) -> None:
    """Capture the exact canonical GDN source selected by one decode request."""

    gdn_module = sys.modules.get("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")
    if gdn_module is None:
        raise RuntimeError("GDN diagnostic module is unavailable")
    flat_indices = state_indices.reshape(-1)
    if flat_indices.numel() < 1:
        raise RuntimeError("GDN diagnostic state-index vector is empty")
    source_index = int(flat_indices[:1].detach().cpu().item())
    conv_state = (
        attention.kv_cache[0]
        if gdn_module.is_conv_state_dim_first()
        else attention.kv_cache[0].transpose(-1, -2)
    )
    ssm_state = attention.kv_cache[1]
    if source_index < 0 or source_index >= conv_state.size(0) or source_index >= ssm_state.size(0):
        raise RuntimeError(f"GDN diagnostic source index is out of range: {source_index}")
    detail[f"{label}_state_index"] = {row: str(source_index) for row, _position in selected}
    detail[f"{label}_conv_state_sha256"] = {
        row: _tensor_sha256(conv_state[source_index : source_index + 1])
        for row, _position in selected
    }
    detail[f"{label}_ssm_state_sha256"] = {
        row: _tensor_sha256(ssm_state[source_index : source_index + 1])
        for row, _position in selected
    }


def _private_root() -> Path:
    root = Path(os.environ.get(_OUTPUT_ROOT, ""))
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise RuntimeError("M8 layer diagnostic output root is unsafe")
    status = root.stat()
    if status.st_uid != os.getuid() or stat.S_IMODE(status.st_mode) != 0o700:
        raise RuntimeError("M8 layer diagnostic output root must be owned mode 0700")
    return root


def _stream() -> Any:
    global _STREAM, _STREAM_PATH
    if _STREAM is not None:
        return _STREAM
    root = _private_root()
    path = root / f"layers-{os.getpid()}.jsonl"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    _STREAM = os.fdopen(descriptor, "wb")
    _STREAM_PATH = path
    header = {
        "schema": "qwen-r9700.m1-m8-layer-diagnostic-header.v1",
        "pid": os.getpid(),
        "positions": sorted(_SELECTED_POSITIONS),
        "module_sha256": os.environ[_SELF_SHA],
    }
    _STREAM.write(json.dumps(header, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    _STREAM.flush()
    os.fsync(_STREAM.fileno())
    return _STREAM


def _publish_pass(records: list[dict[str, Any]]) -> None:
    stream = _stream()
    for record in records:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    stream.flush()
    os.fsync(stream.fileno())


def _quest_stream() -> Any:
    global _QUEST_STREAM, _QUEST_STREAM_PATH
    if _QUEST_STREAM is not None:
        return _QUEST_STREAM
    root = _private_root()
    path = root / f"quest-page-compare-{os.getpid()}.jsonl"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    _QUEST_STREAM = os.fdopen(descriptor, "wb")
    _QUEST_STREAM_PATH = path
    header = {
        "schema": "qwen-r9700.quest-page-compare-header.v1",
        "pid": os.getpid(),
        "positions": sorted(_SELECTED_POSITIONS),
        "module_sha256": os.environ[_SELF_SHA],
    }
    _QUEST_STREAM.write(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    _QUEST_STREAM.flush()
    os.fsync(_QUEST_STREAM.fileno())
    return _QUEST_STREAM


def _publish_quest_pages(records: list[dict[str, Any]]) -> None:
    with _LOCK:
        stream = _quest_stream()
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
            stream.write(b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _patch(module: ModuleType) -> None:
    model_cls = module.Qwen3NextModel
    model_previous = model_cls.forward
    layer_cls = module.Qwen3NextDecoderLayer
    layer_previous = layer_cls.forward
    if getattr(layer_previous, "_qwen_m8_layer_diagnostic", False):
        return

    @functools.wraps(model_previous)
    def model_wrapped(
        self: Any,
        input_ids: Any,
        positions: Any,
        intermediate_tensors: Any = None,
        inputs_embeds: Any = None,
    ):
        rows = int(positions.shape[-1])
        if 1 <= rows <= 8:
            position_rows = _position_rows(positions, rows)
            selected = _selected_rows(position_rows)
            if selected:
                batched_embeddings = (
                    inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
                )
                with _LOCK:
                    for row, position in selected:
                        token_id = (
                            None
                            if input_ids is None
                            else int(input_ids[row : row + 1].detach().cpu().item())
                        )
                        serial_embedding = (
                            batched_embeddings[row : row + 1]
                            if input_ids is None
                            else self.embed_input_ids(input_ids[row : row + 1])
                        )
                        _PENDING.append(
                            {
                                "schema": "qwen-r9700.m1-m8-model-input.v1",
                                "pass_index": _PASS_INDEX,
                                "rows": rows,
                                "row": row,
                                "position": position,
                                "token_id": token_id,
                                "inputs_embeds": inputs_embeds is not None,
                                "batched_embedding_sha256": _tensor_sha256(
                                    batched_embeddings[row : row + 1]
                                ),
                                "serial_embedding_sha256": _tensor_sha256(serial_embedding),
                            }
                        )
        return model_previous(
            self,
            input_ids,
            positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    @functools.wraps(layer_previous)
    def wrapped(self: Any, hidden_states: Any, residual: Any, positions: Any = None, **kwargs: Any):
        global _PASS_INDEX, _PENDING
        if positions is None or not 1 <= int(hidden_states.shape[0]) <= 8:
            return layer_previous(
                self,
                hidden_states=hidden_states,
                residual=residual,
                positions=positions,
                **kwargs,
            )
        rows = int(hidden_states.shape[0])
        position_rows = _position_rows(positions, rows)
        selected = _selected_rows(position_rows)
        if not selected:
            return layer_previous(
                self,
                hidden_states=hidden_states,
                residual=residual,
                positions=positions,
                **kwargs,
            )
        input_hashes = {
            row: {
                "input_hidden_sha256": _tensor_sha256(hidden_states[row : row + 1]),
                "input_residual_sha256": (
                    None if residual is None else _tensor_sha256(residual[row : row + 1])
                ),
            }
            for row, _position in selected
        }
        detail: dict[str, dict[int, Any]] = {}
        attention_type = None
        original_project_ba = None
        original_spec_conv = None
        original_spec_recurrence = None
        original_spec_fused = None
        original_spec_postconv = None
        original_spec_trusted_replay = None
        original_non_spec = None
        gdn_module = None
        original_conv_update = None
        original_recurrent_update = None
        is_gdn_layer = self.layer_type == "linear_attention"
        is_full_attention_layer = self.layer_type == "full_attention"
        full_attention_type = None
        original_full_project_qkv_gate = None
        if is_gdn_layer:
            # The production M8 retained-micro patch intentionally bypasses the
            # in_proj_ba module's forward method, so a normal module hook cannot
            # observe its result.  Temporarily intercept the exact _project_ba
            # invocation made by this layer.  Hash before returning because the
            # result can alias a persistent GEMM output workspace.
            attention = self.linear_attn
            attention_type = type(attention)
            original_project_ba = attention_type._project_ba

            @functools.wraps(original_project_ba)
            def capture_project_ba(instance: Any, states: Any) -> Any:
                result = original_project_ba(instance, states)
                if instance is attention:
                    detail["attention_ba_actual_sha256"] = {
                        row: _tensor_sha256(result[row : row + 1]) for row, _position in selected
                    }
                return result

            attention_type._project_ba = capture_project_ba
            original_spec_conv = attention_type._forward_core_decode_spec_fixed_slot_conv_m1
            original_spec_recurrence = attention_type._forward_core_decode_spec_fixed_slot_packed_m1
            original_spec_fused = attention_type._forward_core_decode_spec_fused_norm
            original_spec_postconv = attention_type._forward_core_decode_spec_post_conv_fused_norm
            original_spec_trusted_replay = attention_type._forward_core_decode_spec_trusted_replay
            original_non_spec = attention_type._forward_core_decode_non_spec

            @functools.wraps(original_spec_conv)
            def capture_spec_conv(instance: Any, **arguments: Any) -> Any:
                if instance is attention:
                    _state_source_detail(
                        detail,
                        "gdn_spec_conv_source",
                        selected,
                        attention,
                        arguments["state_indices"],
                    )
                    detail["gdn_preconv_mixed_qkv_sha256"] = {
                        row: _tensor_sha256(arguments["mixed_qkv"][row : row + 1])
                        for row, _position in selected
                    }
                result = original_spec_conv(instance, **arguments)
                if instance is attention:
                    detail["gdn_postconv_mixed_qkv_sha256"] = {
                        row: _tensor_sha256(result[row : row + 1]) for row, _position in selected
                    }
                return result

            @functools.wraps(original_spec_recurrence)
            def capture_spec_recurrence(instance: Any, **arguments: Any) -> Any:
                if instance is attention:
                    _state_source_detail(
                        detail,
                        "gdn_spec_recurrence_source",
                        selected,
                        attention,
                        arguments["state_indices"],
                    )
                    for name in ("mixed_qkv", "a", "b", "output_gate"):
                        value = arguments[name]
                        detail[f"gdn_spec_{name}_sha256"] = {
                            row: _tensor_sha256(value[row : row + 1]) for row, _position in selected
                        }
                result = original_spec_recurrence(instance, **arguments)
                if instance is attention:
                    value = arguments["core_attn_out"]
                    detail["gdn_spec_normalized_output_sha256"] = {
                        row: _tensor_sha256(value[row : row + 1]) for row, _position in selected
                    }
                return result

            @functools.wraps(original_spec_fused)
            def capture_spec_fused(instance: Any, *arguments: Any, **keyword_arguments: Any) -> Any:
                if arguments:
                    raise RuntimeError("speculative GDN diagnostic requires keyword arguments")
                if instance is attention:
                    metadata = keyword_arguments["attn_metadata"]
                    state_indices = metadata.spec_state_indices_tensor
                    if state_indices is None:
                        raise RuntimeError("speculative GDN diagnostic lacks state indices")
                    _state_source_detail(
                        detail,
                        "gdn_spec_entry_source",
                        selected,
                        attention,
                        state_indices,
                    )
                    for name in ("mixed_qkv", "a", "b", "output_gate"):
                        value = keyword_arguments[name]
                        detail[f"gdn_spec_entry_{name}_sha256"] = {
                            row: _tensor_sha256(value[row : row + 1]) for row, _position in selected
                        }
                result = original_spec_fused(instance, **keyword_arguments)
                if instance is attention:
                    value = keyword_arguments["core_attn_out"]
                    detail["gdn_spec_exit_normalized_output_sha256"] = {
                        row: _tensor_sha256(value[row : row + 1]) for row, _position in selected
                    }
                    metadata = keyword_arguments["attn_metadata"]
                    _state_source_detail(
                        detail,
                        "gdn_spec_exit_source",
                        selected,
                        attention,
                        metadata.spec_state_indices_tensor,
                    )
                return result

            @functools.wraps(original_spec_trusted_replay)
            def capture_spec_trusted_replay(
                instance: Any, *arguments: Any, **keyword_arguments: Any
            ) -> Any:
                if arguments:
                    raise RuntimeError("trusted GDN diagnostic requires keyword arguments")
                if instance is attention:
                    _state_source_detail(
                        detail,
                        "gdn_trusted_entry_source",
                        selected,
                        attention,
                        keyword_arguments["state_indices"],
                    )
                    for name in ("mixed_qkv", "a", "b", "output_gate"):
                        value = keyword_arguments[name]
                        detail[f"gdn_trusted_entry_{name}_sha256"] = {
                            row: _tensor_sha256(value[row : row + 1]) for row, _position in selected
                        }
                result = original_spec_trusted_replay(instance, **keyword_arguments)
                if instance is attention:
                    value = keyword_arguments["core_attn_out"]
                    detail["gdn_trusted_exit_normalized_output_sha256"] = {
                        row: _tensor_sha256(value[row : row + 1]) for row, _position in selected
                    }
                return result

            @functools.wraps(original_spec_postconv)
            def capture_spec_postconv(
                instance: Any, *arguments: Any, **keyword_arguments: Any
            ) -> Any:
                if arguments:
                    raise RuntimeError("post-convolution GDN diagnostic requires keyword arguments")
                if instance is attention:
                    metadata = keyword_arguments["attn_metadata"]
                    state_indices = metadata.spec_state_indices_tensor
                    if state_indices is None:
                        raise RuntimeError("post-convolution GDN diagnostic lacks state indices")
                    _state_source_detail(
                        detail,
                        "gdn_postconv_entry_source",
                        selected,
                        attention,
                        state_indices,
                    )
                    for name in ("mixed_qkv", "a", "b", "output_gate"):
                        value = keyword_arguments[name]
                        detail[f"gdn_postconv_entry_{name}_sha256"] = {
                            row: _tensor_sha256(value[row : row + 1]) for row, _position in selected
                        }
                    for name in (
                        "_recoverssm_pending_accepted",
                        "_recoverssm_record_state_slot",
                        "_recoverssm_trusted_safe_accepted",
                    ):
                        value = getattr(attention, name, None)
                        if value is not None:
                            encoded = json.dumps(value.detach().cpu().reshape(-1).tolist())
                            detail[f"gdn_postconv_entry{name}"] = {
                                row: encoded for row, _position in selected
                            }
                result = original_spec_postconv(instance, **keyword_arguments)
                if instance is attention:
                    value = keyword_arguments["core_attn_out"]
                    detail["gdn_postconv_exit_normalized_output_sha256"] = {
                        row: _tensor_sha256(value[row : row + 1]) for row, _position in selected
                    }
                    metadata = keyword_arguments["attn_metadata"]
                    _state_source_detail(
                        detail,
                        "gdn_postconv_exit_source",
                        selected,
                        attention,
                        metadata.spec_state_indices_tensor,
                    )
                    for name in (
                        "_recoverssm_pending_accepted",
                        "_recoverssm_record_state_slot",
                        "_recoverssm_trusted_safe_accepted",
                    ):
                        state_value = getattr(attention, name, None)
                        if state_value is not None:
                            encoded = json.dumps(state_value.detach().cpu().reshape(-1).tolist())
                            detail[f"gdn_postconv_exit{name}"] = {
                                row: encoded for row, _position in selected
                            }
                return result

            @functools.wraps(original_non_spec)
            def capture_non_spec(instance: Any, **arguments: Any) -> Any:
                if instance is attention:
                    metadata = arguments["attn_metadata"]
                    state_indices = metadata.non_spec_state_indices_tensor
                    if state_indices is None:
                        raise RuntimeError("non-spec GDN diagnostic lacks state indices")
                    _state_source_detail(
                        detail,
                        "gdn_non_spec_source",
                        selected,
                        attention,
                        state_indices,
                    )
                    for name in ("mixed_qkv", "a", "b"):
                        value = arguments[name]
                        detail[f"gdn_non_spec_{name}_sha256"] = {
                            row: _tensor_sha256(value[row : row + 1]) for row, _position in selected
                        }
                result = original_non_spec(instance, **arguments)
                if instance is attention:
                    value = arguments["core_attn_out"]
                    detail["gdn_non_spec_recurrent_output_sha256"] = {
                        row: _tensor_sha256(value[row : row + 1]) for row, _position in selected
                    }
                return result

            attention_type._forward_core_decode_spec_fixed_slot_conv_m1 = capture_spec_conv
            attention_type._forward_core_decode_spec_fixed_slot_packed_m1 = capture_spec_recurrence
            attention_type._forward_core_decode_spec_fused_norm = capture_spec_fused
            attention_type._forward_core_decode_spec_post_conv_fused_norm = capture_spec_postconv
            attention_type._forward_core_decode_spec_trusted_replay = capture_spec_trusted_replay
            attention_type._forward_core_decode_non_spec = capture_non_spec

            gdn_module = sys.modules["vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"]
            original_conv_update = gdn_module.causal_conv1d_update
            original_recurrent_update = gdn_module.fused_recurrent_gated_delta_rule_packed_decode

            @functools.wraps(original_conv_update)
            def capture_conv_update(*arguments: Any, **keyword_arguments: Any) -> Any:
                result = original_conv_update(*arguments, **keyword_arguments)
                value = result if hasattr(result, "shape") else keyword_arguments.get("out")
                if value is not None and hasattr(value, "shape") and value.size(0) == rows:
                    detail["gdn_reference_postconv_mixed_qkv_sha256"] = {
                        row: _tensor_sha256(value[row : row + 1]) for row, _position in selected
                    }
                return result

            @functools.wraps(original_recurrent_update)
            def capture_recurrent_update(*arguments: Any, **keyword_arguments: Any) -> Any:
                result = original_recurrent_update(*arguments, **keyword_arguments)
                value = keyword_arguments.get("out")
                if value is not None and hasattr(value, "shape") and value.size(0) == rows:
                    detail["gdn_reference_recurrent_output_sha256"] = {
                        row: _tensor_sha256(value[row : row + 1]) for row, _position in selected
                    }
                return result

            gdn_module.causal_conv1d_update = capture_conv_update
            gdn_module.fused_recurrent_gated_delta_rule_packed_decode = capture_recurrent_update
        elif is_full_attention_layer:
            full_attention = self.self_attn
            full_attention_type = type(full_attention)
            original_full_project_qkv_gate = full_attention_type._project_qkv_gate

            @functools.wraps(original_full_project_qkv_gate)
            def capture_full_project_qkv_gate(instance: Any, qkv: Any, full_positions: Any) -> Any:
                global _CAPSULE_WRITTEN

                import torch

                q_weight = instance.q_norm.effective_weight()
                k_weight = instance.k_norm.effective_weight()
                q_gate_raw, k_raw, v_raw = qkv.split(
                    [instance.q_size * 2, instance.kv_size, instance.kv_size], dim=-1
                )
                position_axis = full_positions[0] if full_positions.ndim == 2 else full_positions
                if instance is full_attention:
                    qkv_before_sha256 = _tensor_sha256(qkv)
                    for name, value in (
                        ("qkv", qkv),
                        ("q_gate_raw", q_gate_raw),
                        ("k_raw", k_raw),
                        ("v_raw", v_raw),
                    ):
                        detail[f"full_attention_{name}_sha256"] = {
                            row: _tensor_sha256(value[row : row + 1]) for row, _position in selected
                        }
                        detail[f"full_attention_{name}_layout"] = _row_layouts(value, selected)
                    detail["full_attention_raw_storage_relations"] = {
                        row: {
                            "q_gate_to_k": _storage_relation(q_gate_raw, k_raw),
                            "q_gate_to_v": _storage_relation(q_gate_raw, v_raw),
                            "k_to_v": _storage_relation(k_raw, v_raw),
                        }
                        for row, _position in selected
                    }
                    detail["full_attention_cuda_stream"] = {
                        row: str(int(torch.cuda.current_stream(qkv.device).cuda_stream))
                        for row, _position in selected
                    }
                    detail["full_attention_q_weight_sha256"] = {
                        row: _tensor_sha256(q_weight) for row, _position in selected
                    }
                    detail["full_attention_k_weight_sha256"] = {
                        row: _tensor_sha256(k_weight) for row, _position in selected
                    }
                    detail["full_attention_rope_position"] = {
                        row: str(int(position_axis[row].detach().cpu().item()))
                        for row, _position in selected
                    }
                    detail["full_attention_cos_sin_sha256"] = {
                        row: _tensor_sha256(
                            instance.rotary_emb.cos_sin_cache[
                                int(position_axis[row].detach().cpu().item()) : int(
                                    position_axis[row].detach().cpu().item()
                                )
                                + 1
                            ]
                        )
                        for row, _position in selected
                    }
                result = original_full_project_qkv_gate(instance, qkv, full_positions)
                if instance is full_attention:
                    if not isinstance(result, tuple) or len(result) != 4:
                        raise RuntimeError(
                            "full-attention QKV/gate projection returned an invalid result"
                        )
                    for name, value in zip(("q", "k", "v", "gate"), result, strict=True):
                        if value is None:
                            detail[f"full_attention_projected_{name}_sha256"] = {
                                row: "none" for row, _position in selected
                            }
                        elif hasattr(value, "shape"):
                            detail[f"full_attention_projected_{name}_sha256"] = {
                                row: _tensor_sha256(value[row : row + 1])
                                for row, _position in selected
                            }
                        else:
                            raise RuntimeError(f"full-attention projected {name} is not a tensor")
                    detail["full_attention_qkv_unchanged_sha256"] = {
                        row: _tensor_sha256(qkv) for row, _position in selected
                    }
                    detail["full_attention_qkv_unchanged"] = {
                        row: _tensor_sha256(qkv) == qkv_before_sha256 for row, _position in selected
                    }
                    detail["full_attention_output_storage_relations"] = {
                        row: {
                            f"input_qkv_to_{name}": _storage_relation(qkv, value)
                            for name, value in zip(("q", "k", "v", "gate"), result, strict=True)
                            if value is not None
                        }
                        | {
                            f"{left_name}_to_{right_name}": _storage_relation(
                                result[left_index], result[right_index]
                            )
                            for left_index, left_name in enumerate(("q", "k", "v", "gate"))
                            for right_index, right_name in enumerate(("q", "k", "v", "gate"))
                            if left_index < right_index
                            and result[left_index] is not None
                            and result[right_index] is not None
                        }
                        for row, _position in selected
                    }
                    for result_index, name in enumerate(("q", "k", "v", "gate")):
                        value = result[result_index]
                        if value is not None:
                            detail[f"full_attention_projected_{name}_layout"] = _row_layouts(
                                value, selected
                            )

                    capsule_selected = (
                        _CAPSULE_SPEC is not None
                        and _CAPSULE_SPEC[0] == _PASS_INDEX
                        and _CAPSULE_SPEC[1] == int(self.layer_idx)
                        and any(position == _CAPSULE_SPEC[2] for _row, position in selected)
                    )
                    serial_rows = (
                        range(int(qkv.shape[0]))
                        if capsule_selected
                        else (row for row, _position in selected)
                    )
                    serial_results: dict[int, Any] = {}
                    for row in serial_rows:
                        serial_positions = (
                            full_positions[row : row + 1]
                            if full_positions.ndim == 1
                            else full_positions[:, row : row + 1]
                        )
                        serial_results[row] = original_full_project_qkv_gate(
                            instance, qkv[row : row + 1], serial_positions
                        )
                    for result_index, name in enumerate(("q", "k", "v", "gate")):
                        values: dict[int, str] = {}
                        for row, _position in selected:
                            value = serial_results[row][result_index]
                            values[row] = "none" if value is None else _tensor_sha256(value)
                        detail[f"full_attention_serial_{name}_sha256"] = values
                    if capsule_selected:
                        if _CAPSULE_WRITTEN:
                            raise RuntimeError("M8 tensor capsule selector matched more than once")
                        capsule_path = _private_root() / (
                            f"qk-pass-{_PASS_INDEX}-layer-{int(self.layer_idx)}-"
                            f"position-{_CAPSULE_SPEC[2]}.pt"
                        )
                        selected_position_rows = position_axis.to(torch.int64)
                        capsule = {
                            "schema": "urn:qwen-r9700:m8-qk-counterexample-capsule:v1",
                            "module_sha256": os.environ[_SELF_SHA],
                            "pass_index": _PASS_INDEX,
                            "layer_index": int(self.layer_idx),
                            "selected_position": _CAPSULE_SPEC[2],
                            "cuda_stream": int(torch.cuda.current_stream(qkv.device).cuda_stream),
                            "qkv": qkv.detach().contiguous().cpu(),
                            "positions": full_positions.detach().contiguous().cpu(),
                            "q_weight": q_weight.detach().contiguous().cpu(),
                            "k_weight": k_weight.detach().contiguous().cpu(),
                            "cos_sin_rows": instance.rotary_emb.cos_sin_cache.index_select(
                                0, selected_position_rows
                            )
                            .detach()
                            .contiguous()
                            .cpu(),
                            "actual": tuple(
                                None if value is None else value.detach().contiguous().cpu()
                                for value in result
                            ),
                            "serial": tuple(
                                tuple(
                                    None if value is None else value.detach().contiguous().cpu()
                                    for value in serial_results[row]
                                )
                                for row in range(int(qkv.shape[0]))
                            ),
                            "layouts": {
                                "qkv": _tensor_layout(qkv),
                                "q_gate_raw": _tensor_layout(q_gate_raw),
                                "k_raw": _tensor_layout(k_raw),
                                "v_raw": _tensor_layout(v_raw),
                                "actual": [
                                    None if value is None else _tensor_layout(value)
                                    for value in result
                                ],
                            },
                        }
                        capsule_sha256 = _write_tensor_capsule(capsule_path, capsule)
                        detail["full_attention_tensor_capsule"] = {
                            row: {
                                "path": str(capsule_path),
                                "sha256": capsule_sha256,
                            }
                            for row, _position in selected
                        }
                        _CAPSULE_WRITTEN = True
                return result

            full_attention_type._project_qkv_gate = capture_full_project_qkv_gate

        modules = (
            (
                (self.input_layernorm, "input_norm"),
                (self.linear_attn.in_proj_qkvz, "attention_qkvz_projection"),
                (self.linear_attn.in_proj_ba, "attention_ba_projection"),
                (self.linear_attn, "linear_attention_output"),
                (self.linear_attn.norm, "linear_attention_norm"),
                (self.linear_attn.out_proj, "linear_attention_output_projection"),
                (self.post_attention_layernorm, "post_attention_norm"),
                (self.mlp.gate_up_proj, "mlp_gate_up_projection"),
                (self.mlp.act_fn, "mlp_activation"),
                (self.mlp.down_proj, "mlp_down_projection"),
                (self.mlp, "mlp_output"),
            )
            if is_gdn_layer
            else (
                (
                    (self.input_layernorm, "input_norm"),
                    (self.self_attn.qkv_proj, "full_attention_qkv_projection"),
                    (self.self_attn.attn, "full_attention_core_output"),
                    (self.self_attn.o_proj, "full_attention_output_projection"),
                    (self.self_attn, "full_attention_output"),
                    (self.post_attention_layernorm, "post_attention_norm"),
                    (self.mlp.gate_up_proj, "mlp_gate_up_projection"),
                    (self.mlp.act_fn, "mlp_activation"),
                    (self.mlp.down_proj, "mlp_down_projection"),
                    (self.mlp, "mlp_output"),
                )
                if is_full_attention_layer
                else ()
            )
        )
        handles = [
            module.register_forward_hook(_detail_hook(detail, label, selected))
            for module, label in modules
        ]
        if is_gdn_layer:
            handles.append(
                self.linear_attn.out_proj.register_forward_pre_hook(
                    _detail_input_hook(
                        detail,
                        "linear_attention_output_projection_input",
                        selected,
                    )
                )
            )
        elif is_full_attention_layer:
            handles.append(
                self.self_attn.o_proj.register_forward_pre_hook(
                    _detail_input_hook(
                        detail,
                        "full_attention_output_projection_input",
                        selected,
                    )
                )
            )
        try:
            result = layer_previous(
                self,
                hidden_states=hidden_states,
                residual=residual,
                positions=positions,
                **kwargs,
            )
        finally:
            if attention_type is not None and original_project_ba is not None:
                attention_type._project_ba = original_project_ba
            if attention_type is not None and original_spec_conv is not None:
                attention_type._forward_core_decode_spec_fixed_slot_conv_m1 = original_spec_conv
            if attention_type is not None and original_spec_recurrence is not None:
                attention_type._forward_core_decode_spec_fixed_slot_packed_m1 = (
                    original_spec_recurrence
                )
            if attention_type is not None and original_spec_fused is not None:
                attention_type._forward_core_decode_spec_fused_norm = original_spec_fused
            if attention_type is not None and original_spec_postconv is not None:
                attention_type._forward_core_decode_spec_post_conv_fused_norm = (
                    original_spec_postconv
                )
            if attention_type is not None and original_spec_trusted_replay is not None:
                attention_type._forward_core_decode_spec_trusted_replay = (
                    original_spec_trusted_replay
                )
            if attention_type is not None and original_non_spec is not None:
                attention_type._forward_core_decode_non_spec = original_non_spec
            if gdn_module is not None and original_conv_update is not None:
                gdn_module.causal_conv1d_update = original_conv_update
            if gdn_module is not None and original_recurrent_update is not None:
                gdn_module.fused_recurrent_gated_delta_rule_packed_decode = (
                    original_recurrent_update
                )
            if full_attention_type is not None and original_full_project_qkv_gate is not None:
                full_attention_type._project_qkv_gate = original_full_project_qkv_gate
            for handle in handles:
                handle.remove()
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("decoder layer did not return hidden/residual state")
        output_hidden, output_residual = result
        layer_index = int(self.layer_idx)
        with _LOCK:
            for row, position in selected:
                _PENDING.append(
                    {
                        "schema": "qwen-r9700.m1-m8-layer-boundary.v1",
                        "pass_index": _PASS_INDEX,
                        "layer_index": layer_index,
                        "layer_type": str(self.layer_type),
                        "rows": rows,
                        "row": row,
                        "position": position,
                        **input_hashes[row],
                        "output_hidden_sha256": _tensor_sha256(output_hidden[row : row + 1]),
                        "output_residual_sha256": _tensor_sha256(output_residual[row : row + 1]),
                        "detail": {label: values[row] for label, values in detail.items()},
                    }
                )
            if layer_index == 63:
                expected = 65 * len(selected)
                if len(_PENDING) != expected:
                    raise RuntimeError(
                        f"decoder diagnostic pass has {len(_PENDING)} records, expected {expected}"
                    )
                _publish_pass(_PENDING)
                _PENDING = []
                _PASS_INDEX += 1
        return result

    wrapped._qwen_m8_layer_diagnostic = True  # type: ignore[attr-defined]
    model_wrapped._qwen_m8_layer_diagnostic = True  # type: ignore[attr-defined]
    model_cls.forward = model_wrapped
    layer_cls.forward = wrapped
    print("[qwen-m8-layer-diagnostic] decoder boundary capture armed", flush=True)


def _patch_quest(module: ModuleType) -> None:
    """Compare production M8 Quest selection and attention with serial-Q1 controls."""

    row_local = getattr(module, "_compiled_select_row_local_m8_logical_pages", None)
    scalar = getattr(module, "_compiled_select_logical_pages", None)
    cached = getattr(module, "_cached_gemm_select_logical_pages", None)
    attention = getattr(module, "_try_compiled_selected_attention", None)
    if not all(callable(value) for value in (row_local, scalar, cached, attention)):
        raise RuntimeError("Quest comparator cannot resolve selector and attention functions")
    if getattr(row_local, "_qwen_m8_quest_page_compare", False) or getattr(
        cached, "_qwen_m8_quest_page_compare", False
    ):
        return

    def record_comparison(
        *,
        result: Any,
        kwargs: dict[str, Any],
        scalar_rows: list[Any],
        selector_path: str,
    ) -> None:
        sequence_length = int(kwargs.get("sequence_length", -1))
        if sequence_length not in _SELECTED_POSITIONS:
            return
        if not isinstance(result, tuple) or len(result) != 5:
            raise RuntimeError("Quest M8 selector returned an invalid comparison result")

        selected_rows, selected_counts, _union_pages, _union_masks, _union_counts = result
        query = kwargs.get("query")
        if tuple(getattr(query, "shape", ())) != (8, 24, 256):
            raise RuntimeError("Quest page comparator requires query shape [8,24,256]")
        if tuple(getattr(selected_rows, "shape", ())) != (8, 96):
            raise RuntimeError("Quest page comparator requires selected shape [8,96]")

        # Preserve the production outputs before the independent scalar calls.
        # These copies are observation-only and are never returned to attention.
        production_rows = selected_rows.detach().clone()
        production_counts = selected_counts.detach().clone()
        counts = production_counts.to(device="cpu").tolist()
        production_cpu = production_rows.to(device="cpu")
        scalar_cpu = [pages.to(device="cpu") for pages in scalar_rows]
        layer_identity = kwargs.get("layer_identity")
        layer_name = (
            str(layer_identity[0])
            if isinstance(layer_identity, tuple) and layer_identity
            else None
        )
        records: list[dict[str, Any]] = []
        for row in range(8):
            count = int(counts[row])
            if not 0 < count <= int(production_cpu.shape[1]):
                raise RuntimeError("Quest page comparator observed an invalid selected count")
            production = production_cpu[row, :count].contiguous()
            reference = scalar_cpu[row].contiguous()
            if int(reference.numel()) != count:
                raise RuntimeError("Quest scalar selector returned a different page count")
            unequal = (production != reference).nonzero().reshape(-1).tolist()
            first = int(unequal[0]) if unequal else None
            records.append(
                {
                    "schema": "qwen-r9700.quest-page-compare.v1",
                    "sequence_length": sequence_length,
                    "layer_name": layer_name,
                    "selector_path": selector_path,
                    "row": row,
                    "count": count,
                    "equal": not unequal,
                    "mismatch_count": len(unequal),
                    "first_mismatch_index": first,
                    "production_first_mismatch_page": (
                        int(production[first].item()) if first is not None else None
                    ),
                    "scalar_first_mismatch_page": (
                        int(reference[first].item()) if first is not None else None
                    ),
                    "production_sha256": _tensor_sha256(production),
                    "scalar_sha256": _tensor_sha256(reference),
                }
            )
        _publish_quest_pages(records)

    @functools.wraps(row_local)
    def compare_row_local(**kwargs: Any) -> Any:
        result = row_local(**kwargs)
        sequence_length = int(kwargs.get("sequence_length", -1))
        if sequence_length not in _SELECTED_POSITIONS:
            return result
        query = kwargs["query"]
        scalar_kwargs = {
            name: kwargs[name]
            for name in (
                "key_cache",
                "logical_physical_blocks",
                "block_size",
                "sequence_length",
                "budget_pages",
                "recent_pages",
                "key_scale",
                "key_scale_float",
            )
        }
        scalar_rows = []
        for row in range(8):
            _physical, pages = scalar(query=query[row : row + 1], **scalar_kwargs)
            scalar_rows.append(pages.detach().clone())
        record_comparison(
            result=result,
            kwargs=kwargs,
            scalar_rows=scalar_rows,
            selector_path="compiled-row-local",
        )
        return result

    @functools.wraps(cached)
    def compare_cached(**kwargs: Any) -> Any:
        global _QUEST_ATTENTION_REFERENCE

        result = cached(**kwargs)
        sequence_length = int(kwargs.get("sequence_length", -1))
        if not kwargs.get("row_local", False) or sequence_length not in _SELECTED_POSITIONS:
            return result
        query = kwargs["query"]
        scalar_kwargs = {name: value for name, value in kwargs.items() if name != "query"}
        scalar_kwargs["row_local"] = False
        scalar_rows = []
        for row in range(8):
            _physical, pages = cached(query=query[row : row + 1], **scalar_kwargs)
            scalar_rows.append(pages.detach().clone())
        record_comparison(
            result=result,
            kwargs=kwargs,
            scalar_rows=scalar_rows,
            selector_path="cached-gemm",
        )

        # Reproduce the exact selector geometry used by the serial-Q1 path for
        # row zero: one query replicated to the fixed M8 arithmetic ABI.  This
        # is an independent, uncached-centroid reference; unlike the scalar
        # calls above it can expose a coherent error in the cached centroids.
        repeated_query = query[0:1].expand(8, -1, -1).contiguous()
        reference_kwargs = {
            name: kwargs[name]
            for name in (
                "key_cache",
                "logical_physical_blocks",
                "block_size",
                "sequence_length",
                "budget_pages",
                "recent_pages",
                "key_scale",
                "key_scale_float",
                "layer_identity",
            )
        }
        reference = row_local(query=repeated_query, **reference_kwargs)
        if not isinstance(reference, tuple) or len(reference) != 5:
            raise RuntimeError("Quest serial-Q1 selector returned an invalid result")
        production_count = int(result[1][0].detach().cpu().item())
        reference_count = int(reference[1][0].detach().cpu().item())
        production_pages = result[0][0, :production_count].detach().clone()
        reference_pages = reference[0][0, :reference_count].detach().clone()
        if production_count != reference_count:
            raise RuntimeError("Quest cached and serial-Q1 selectors returned different counts")
        unequal = (
            production_pages.to(device="cpu") != reference_pages.to(device="cpu")
        ).nonzero().reshape(-1).tolist()
        _publish_quest_pages(
            [
                {
                    "schema": "qwen-r9700.quest-authoritative-page-compare.v1",
                    "sequence_length": sequence_length,
                    "layer_name": str(kwargs["layer_identity"][0]),
                    "row": 0,
                    "count": production_count,
                    "equal": not unequal,
                    "mismatch_count": len(unequal),
                    "first_mismatch_index": int(unequal[0]) if unequal else None,
                    "production_sha256": _tensor_sha256(production_pages),
                    "serial_q1_sha256": _tensor_sha256(reference_pages),
                }
            ]
        )
        if _QUEST_ATTENTION_REFERENCE is not None:
            raise RuntimeError("Quest attention reference was not consumed by the previous layer")
        _QUEST_ATTENTION_REFERENCE = {
            "layer_name": str(kwargs["layer_identity"][0]),
            "sequence_length": sequence_length,
            "selected_pages": reference[2].detach().clone(),
            "selected_row_masks": reference[3].detach().clone(),
            "selected_count_device": reference[4].detach().clone(),
        }
        return result

    @functools.wraps(attention)
    def compare_attention(**kwargs: Any) -> Any:
        global _QUEST_ATTENTION_REFERENCE

        result = attention(**kwargs)
        sequence_length = int(kwargs.get("sequence_length", -1))
        query = kwargs.get("query")
        if sequence_length not in _SELECTED_POSITIONS or tuple(getattr(query, "shape", ())) != (
            8,
            24,
            256,
        ):
            return result
        reference = _QUEST_ATTENTION_REFERENCE
        if reference is None or int(reference["sequence_length"]) != sequence_length:
            raise RuntimeError("Quest attention comparator lacks its serial-Q1 selection")
        current_key = kwargs.get("current_key")
        current_value = kwargs.get("current_value")
        if tuple(getattr(current_key, "shape", ())) != (8, 4, 256) or tuple(
            getattr(current_value, "shape", ())
        ) != (8, 4, 256):
            raise RuntimeError("Quest attention comparator requires current K/V [8,4,256]")

        serial_output = kwargs["output"].new_empty((8, 24, 256))
        serial_kwargs = dict(kwargs)
        serial_kwargs.update(
            {
                "query": query[0:1].expand(8, -1, -1).contiguous(),
                "output": serial_output,
                "current_key": current_key[0:1].expand(8, -1, -1).contiguous(),
                "current_value": current_value[0:1].expand(8, -1, -1).contiguous(),
                "selected_physical": reference["selected_pages"],
                "selected_pages": reference["selected_pages"],
                "selected_row_masks": reference["selected_row_masks"],
                "selected_count_device": reference["selected_count_device"],
            }
        )
        if not attention(**serial_kwargs):
            raise RuntimeError("Quest serial-Q1 attention control rejected its input")
        production = kwargs["output"][0:1].detach().clone()
        serial = serial_output[0:1].detach().clone()
        mismatches = int((production != serial).count_nonzero().detach().cpu().item())
        maximum_error = float((production.float() - serial.float()).abs().amax().cpu().item())
        _publish_quest_pages(
            [
                {
                    "schema": "qwen-r9700.quest-attention-compare.v1",
                    "sequence_length": sequence_length,
                    "layer_name": reference["layer_name"],
                    "row": 0,
                    "equal": mismatches == 0,
                    "mismatch_count": mismatches,
                    "max_abs_error": maximum_error,
                    "production_sha256": _tensor_sha256(production),
                    "serial_q1_sha256": _tensor_sha256(serial),
                }
            ]
        )
        _QUEST_ATTENTION_REFERENCE = None
        return result

    compare_row_local._qwen_m8_quest_page_compare = True  # type: ignore[attr-defined]
    compare_cached._qwen_m8_quest_page_compare = True  # type: ignore[attr-defined]
    compare_attention._qwen_m8_quest_page_compare = True  # type: ignore[attr-defined]
    module._compiled_select_row_local_m8_logical_pages = compare_row_local
    module._cached_gemm_select_logical_pages = compare_cached
    module._try_compiled_selected_attention = compare_attention
    print("[qwen-m8-layer-diagnostic] Quest M8/serial-Q1 comparator armed", flush=True)


class _Loader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader) -> None:
        self._wrapped = wrapped

    def create_module(self, spec):
        create = getattr(self._wrapped, "create_module", None)
        return None if create is None else create(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._wrapped.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: ModuleType | None = None):
        if fullname != _MODULE:
            return None
        try:
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot resolve {fullname}")
        spec.loader = _Loader(spec.loader)
        return spec


class _QuestLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader) -> None:
        self._wrapped = wrapped

    def create_module(self, spec):
        create = getattr(self._wrapped, "create_module", None)
        return None if create is None else create(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._wrapped.exec_module(module)
        _patch_quest(module)


class _QuestFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: ModuleType | None = None):
        if fullname != _QUEST_MODULE:
            return None
        try:
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot resolve {fullname}")
        spec.loader = _QuestLoader(spec.loader)
        return spec


def install() -> None:
    if os.environ.get(_ENABLED, "0") != "1":
        return
    if os.environ.get(_REQUIRED, "0") != "1":
        raise RuntimeError("M8 layer diagnostic must be fail-closed")
    path = Path(__file__).resolve()
    expected = os.environ.get(_SELF_SHA, "")
    if len(expected) != 64 or _digest(path) != expected:
        raise RuntimeError("M8 layer diagnostic module SHA256 mismatch")
    _private_root()
    quest_compare = os.environ.get(_QUEST_PAGE_COMPARE, "0")
    if quest_compare not in {"0", "1"}:
        raise RuntimeError("Quest page comparator flag must be 0 or 1")
    if _MODULE in sys.modules:
        _patch(sys.modules[_MODULE])
    else:
        sys.meta_path.insert(0, _Finder())
    if quest_compare == "1":
        if _QUEST_MODULE in sys.modules:
            _patch_quest(sys.modules[_QUEST_MODULE])
        else:
            sys.meta_path.insert(0, _QuestFinder())


install()
