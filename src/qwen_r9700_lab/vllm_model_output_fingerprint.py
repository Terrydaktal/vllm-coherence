"""Deterministic, fail-closed fingerprints for vLLM ``ModelRunnerOutput`` values.

The scheduler uses this digest to bind the exact worker output selected by a
serial-validated transaction to the later ``update_from_output`` call.  It does
not use ``repr`` or pickle: both can omit state or execute ambient code.  Every
supported value is encoded with an explicit type tag, length framing, class
identity, container topology, tensor metadata, and exact element bytes.
Unsupported or concurrently changing values are rejected.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import struct
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path


class ModelOutputFingerprintError(RuntimeError):
    """A model output cannot be bound without ambiguity."""


def _frame(tag: bytes, payload: bytes) -> bytes:
    return tag + struct.pack(">Q", len(payload)) + payload


def _class_identity(value: object) -> bytes:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}".encode()


class _Encoder:
    def __init__(self) -> None:
        self._seen: dict[int, int] = {}

    def _reference(self, value: object) -> bytes | None:
        identity = id(value)
        previous = self._seen.get(identity)
        if previous is not None:
            return _frame(b"R", struct.pack(">Q", previous))
        self._seen[identity] = len(self._seen)
        return None

    def encode(self, value: object) -> bytes:
        if value is None:
            return b"N"
        if type(value) is bool:
            return b"B1" if value else b"B0"
        if type(value) is int:
            return _frame(b"I", str(value).encode("ascii"))
        if type(value) is float:
            return _frame(b"F", struct.pack(">d", value))
        if type(value) is str:
            return _frame(b"S", value.encode("utf-8"))
        if type(value) is bytes:
            return _frame(b"Y", value)
        if type(value) is bytearray:
            reference = self._reference(value)
            if reference is not None:
                return reference
            first = bytes(value)
            if first != bytes(value):
                raise ModelOutputFingerprintError("bytearray changed while fingerprinting")
            return _frame(b"A", first)
        if type(value) is memoryview:
            reference = self._reference(value)
            if reference is not None:
                return reference
            first = value.tobytes()
            if first != value.tobytes():
                raise ModelOutputFingerprintError("memoryview changed while fingerprinting")
            return _frame(b"M", first)
        if isinstance(value, enum.Enum):
            payload = _frame(b"C", _class_identity(value)) + self.encode(value.value)
            return _frame(b"E", payload)
        if isinstance(value, Path):
            return _frame(b"P", str(value).encode("utf-8"))

        numpy_value = self._encode_numpy(value)
        if numpy_value is not None:
            return numpy_value
        torch_value = self._encode_torch(value)
        if torch_value is not None:
            return torch_value

        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            reference = self._reference(value)
            if reference is not None:
                return reference
            payload = [_frame(b"C", _class_identity(value))]
            for field in dataclasses.fields(value):
                payload.append(_frame(b"K", field.name.encode("utf-8")))
                payload.append(self.encode(getattr(value, field.name)))
            return _frame(b"D", b"".join(payload))

        if isinstance(value, tuple) and hasattr(value, "_fields"):
            reference = self._reference(value)
            if reference is not None:
                return reference
            fields = tuple(value._fields)
            if len(fields) != len(value) or any(not isinstance(name, str) for name in fields):
                raise ModelOutputFingerprintError("named tuple field contract is invalid")
            payload = [_frame(b"C", _class_identity(value))]
            for name, item in zip(fields, value, strict=True):
                payload.append(_frame(b"K", name.encode("utf-8")))
                payload.append(self.encode(item))
            return _frame(b"Q", b"".join(payload))

        if type(value) in (list, tuple):
            reference = self._reference(value)
            if reference is not None:
                return reference
            tag = b"L" if type(value) is list else b"T"
            payload = struct.pack(">Q", len(value)) + b"".join(
                self.encode(item) for item in value
            )
            return _frame(tag, payload)

        if type(value) in (set, frozenset):
            reference = self._reference(value)
            if reference is not None:
                return reference
            tag = b"Z" if type(value) is set else b"X"
            encoded = sorted(_Encoder().encode(item) for item in value)
            if any(left == right for left, right in pairwise(encoded)):
                raise ModelOutputFingerprintError("set items have ambiguous encodings")
            return _frame(tag, struct.pack(">Q", len(encoded)) + b"".join(encoded))

        if isinstance(value, Mapping):
            reference = self._reference(value)
            if reference is not None:
                return reference
            keyed: list[tuple[bytes, object, object]] = []
            for key, item in value.items():
                key_payload = _Encoder().encode(key)
                keyed.append((key_payload, key, item))
            keyed.sort(key=lambda row: row[0])
            if any(left[0] == right[0] for left, right in pairwise(keyed)):
                raise ModelOutputFingerprintError("mapping keys have ambiguous encodings")
            payload = [
                _frame(b"C", _class_identity(value)),
                struct.pack(">Q", len(keyed)),
            ]
            for _key_payload, key, item in keyed:
                payload.append(self.encode(key))
                payload.append(self.encode(item))
            return _frame(b"G", b"".join(payload))

        raise ModelOutputFingerprintError(
            f"unsupported model output value: {type(value).__module__}.{type(value).__qualname__}"
        )

    def _encode_numpy(self, value: object) -> bytes | None:
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - production vLLM always carries NumPy.
            return None
        if isinstance(value, np.generic):
            value = np.asarray(value)
        if not isinstance(value, np.ndarray):
            return None
        reference = self._reference(value)
        if reference is not None:
            return reference
        if value.dtype.hasobject:
            raise ModelOutputFingerprintError("object-dtype NumPy arrays are unsupported")
        metadata_before = (value.dtype.str, value.dtype.descr, value.shape, value.strides)
        first = value.tobytes(order="C")
        second = value.tobytes(order="C")
        metadata_after = (value.dtype.str, value.dtype.descr, value.shape, value.strides)
        if metadata_before != metadata_after or first != second:
            raise ModelOutputFingerprintError("NumPy array changed while fingerprinting")
        metadata = repr(metadata_before).encode("utf-8")
        return _frame(b"U", _frame(b"H", metadata) + _frame(b"V", first))

    def _encode_torch(self, value: object) -> bytes | None:
        try:
            import torch
        except ImportError:  # pragma: no cover - unit environments may omit Torch.
            return None
        if not isinstance(value, torch.Tensor):
            return None
        reference = self._reference(value)
        if reference is not None:
            return reference
        if value.layout != torch.strided or value.device.type == "meta":
            raise ModelOutputFingerprintError("non-strided or meta Torch tensors are unsupported")
        metadata_before = (
            str(value.dtype),
            tuple(value.shape),
            tuple(value.stride()),
            int(value.storage_offset()),
            str(value.device),
            bool(value.requires_grad),
            int(value._version),
        )
        try:
            contiguous = value.detach().contiguous().cpu()
            raw = contiguous.view(torch.uint8).numpy().tobytes(order="C")
        except Exception as error:
            raise ModelOutputFingerprintError("Torch tensor bytes could not be captured") from error
        metadata_after = (
            str(value.dtype),
            tuple(value.shape),
            tuple(value.stride()),
            int(value.storage_offset()),
            str(value.device),
            bool(value.requires_grad),
            int(value._version),
        )
        if metadata_before != metadata_after:
            raise ModelOutputFingerprintError("Torch tensor changed while fingerprinting")
        metadata = repr(metadata_before).encode("utf-8")
        return _frame(b"O", _frame(b"H", metadata) + _frame(b"V", raw))


def model_output_sha256(value: object) -> str:
    """Return the exact domain-separated SHA-256 for one supported output graph."""

    payload = _Encoder().encode(value)
    return hashlib.sha256(b"qwen-vllm-model-output-v1\0" + payload).hexdigest()


def request_token_ids(value: object, request_id: str) -> tuple[int, ...]:
    """Extract one request's exact sampled IDs from a ``ModelRunnerOutput``."""

    if not isinstance(request_id, str) or not request_id:
        raise ModelOutputFingerprintError("request ID is invalid")
    try:
        req_ids = value.req_ids
        req_id_to_index = value.req_id_to_index
        sampled_token_ids = value.sampled_token_ids
    except AttributeError as error:
        raise ModelOutputFingerprintError(
            "value is not a ModelRunnerOutput-shaped object"
        ) from error
    if (
        type(req_ids) is not list
        or any(not isinstance(item, str) or not item for item in req_ids)
        or len(req_ids) != len(set(req_ids))
        or type(req_id_to_index) is not dict
        or set(req_id_to_index) != set(req_ids)
        or any(req_id_to_index[item] != index for index, item in enumerate(req_ids))
        or type(sampled_token_ids) is not list
        or len(sampled_token_ids) != len(req_ids)
    ):
        raise ModelOutputFingerprintError("ModelRunnerOutput request index is inconsistent")
    if len(req_ids) != 1:
        raise ModelOutputFingerprintError(
            "ModelRunnerOutput must contain exactly one transaction request"
        )
    if request_id not in req_id_to_index:
        raise ModelOutputFingerprintError("ModelRunnerOutput does not contain the request")
    row = sampled_token_ids[req_id_to_index[request_id]]
    if type(row) is not list or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in row
    ):
        raise ModelOutputFingerprintError("ModelRunnerOutput sampled token IDs are invalid")
    return tuple(row)
