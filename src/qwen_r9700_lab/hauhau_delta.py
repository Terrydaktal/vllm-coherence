"""Restart-safe helpers for reconstructing a BF16 model from paired GGUF deltas.

The public Hauhau Aggressive release is available as GGUF rather than as the
pre-quantization checkpoint.  This module provides the deliberately small,
auditable primitives used by ``scripts/qwen-reconstruct-hauhau-delta``:

* stable file authentication;
* zero-copy safetensors metadata and BF16 payload access;
* deterministic Hugging Face -> GGUF tensor-name mapping for Qwen3.8;
* bounded-memory randomized SVD for separating coherent edits from quantizer
  noise; and
* atomic JSON/JSONL transaction records.

It does not select a reconstruction policy by itself.  Analysis records the
evidence and a separately sealed plan decides which compact delta factors may
be applied to the pristine vanilla BF16 source.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import struct
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, BinaryIO, ClassVar

import numpy as np
from gguf import MODEL_ARCH, GGMLQuantizationType, GGUFReader, get_tensor_name_map
from gguf.quants import dequantize

SCHEMA_INVENTORY = "urn:qwen-r9700:hauhau-delta-inventory:v1"
SCHEMA_ANALYSIS_RECORD = "urn:qwen-r9700:hauhau-delta-analysis-record:v1"
SCHEMA_PLAN = "urn:qwen-r9700:hauhau-delta-plan:v1"
SCHEMA_RESULT = "urn:qwen-r9700:hauhau-delta-result:v1"

OFFICIAL_REPOSITORY = "Qwen/Qwen3.8-27B"
OFFICIAL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
HAUHAU_REPOSITORY = "HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF"
HAUHAU_REVISION = "993a5971fda8f30dd1b7eb2654792ba4415c7460"
HAUHAU_FILENAME = "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"
HAUHAU_BYTES = 31_457_990_784
HAUHAU_SHA256 = "4e7735df4d1e2ec721f2551f531b815702a2f89123238c564797eda4b0304bc2"
VANILLA_Q8_REPOSITORY = "ggml-org/Qwen3.8-27B-GGUF"
VANILLA_Q8_REVISION = "0669b98607d47046c7c2b3f801011d54a08cfccf"
VANILLA_Q8_FILENAME = "Qwen3.8-27B-Q8_0.gguf"
VANILLA_Q8_BYTES = 28_595_763_552
VANILLA_Q8_SHA256 = "f5c702d8820d36fb55985bb238fc83ee3a313e920f4b752a437c3a6a9e14e4c8"

_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
_MTP_LAYER_RE = re.compile(r"^mtp\.layers\.(\d+)\.")
_DT_BIAS_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.linear_attn\.dt_bias$")


class ReconstructionError(RuntimeError):
    """Raised when an authenticated reconstruction invariant is violated."""


def canonical_json(value: Any) -> bytes:
    """Return a stable UTF-8 JSON encoding with a terminating newline."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def sha256_bytes(payload: bytes | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def _regular_owned_file(path: Path) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as exc:
        raise ReconstructionError(f"cannot stat required file {path}: {exc}") from exc
    if not stat.S_ISREG(value.st_mode):
        raise ReconstructionError(f"required path is not a regular file: {path}")
    if value.st_uid != os.getuid():
        raise ReconstructionError(f"required file is not owned by the current user: {path}")
    if value.st_nlink != 1:
        raise ReconstructionError(f"required file must have exactly one hard link: {path}")
    return value


def stable_sha256(path: Path, *, expected_bytes: int | None = None) -> tuple[str, int]:
    """Hash a regular file while rejecting replacement or mutation during the read."""

    before = _regular_owned_file(path)
    if expected_bytes is not None and before.st_size != expected_bytes:
        raise ReconstructionError(
            f"file size differs for {path}: got {before.st_size}, expected {expected_bytes}"
        )
    digest = hashlib.sha256()
    try:
        with path.open("rb", buffering=0) as handle:
            while block := handle.read(8 * 1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise ReconstructionError(f"cannot hash required file {path}: {exc}") from exc
    after = path.lstat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_uid,
        before.st_mode,
        before.st_nlink,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_uid,
        after.st_mode,
        after.st_nlink,
    )
    if identity_before != identity_after:
        raise ReconstructionError(f"file changed while it was hashed: {path}")
    return digest.hexdigest(), before.st_size


def load_json(path: Path) -> dict[str, Any]:
    before = _regular_owned_file(path)
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReconstructionError(f"invalid JSON object {path}: {exc}") from exc
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ReconstructionError(f"JSON file changed while it was read: {path}")
    if not isinstance(value, dict):
        raise ReconstructionError(f"JSON document is not an object: {path}")
    return value


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    """Create or replace one transaction record atomically and durably."""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        fsync_directory(path.parent)
    except BaseException:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise


def append_record(path: Path, record: Mapping[str, Any]) -> None:
    """Durably append one self-authenticating canonical JSONL record."""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    body = dict(record)
    body_without_digest = canonical_json(body)
    body["record_sha256"] = sha256_bytes(body_without_digest)
    payload = canonical_json(body)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "ab", buffering=0, closefd=True) as handle:
            handle.write(payload)
            os.fsync(handle.fileno())
    finally:
        # fdopen closes the descriptor on the normal path.
        with suppress(OSError):
            os.close(descriptor)
    fsync_directory(path.parent)


def read_records(path: Path) -> list[dict[str, Any]]:
    """Strictly validate every record in an append-only analysis journal."""

    before = _regular_owned_file(path)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ReconstructionError(f"cannot read journal {path}: {exc}") from exc
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ReconstructionError(f"journal changed while it was read: {path}")
    if payload and not payload.endswith(b"\n"):
        raise ReconstructionError(f"journal ends with a partial record: {path}")
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReconstructionError(
                f"invalid journal record {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict) or not isinstance(value.get("record_sha256"), str):
            raise ReconstructionError(f"invalid journal record shape {path}:{line_number}")
        observed = value.pop("record_sha256")
        expected = sha256_bytes(canonical_json(value))
        if observed != expected:
            raise ReconstructionError(f"journal digest differs at {path}:{line_number}")
        value["record_sha256"] = observed
        result.append(value)
    return result


@dataclass(frozen=True)
class SafeTensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def n_elements(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))


class SafeTensorFile:
    """Minimal raw safetensors reader with zero-copy payload views."""

    _WIDTHS: ClassVar[dict[str, int]] = {
        "BF16": 2,
        "F16": 2,
        "F32": 4,
        "I32": 4,
        "I64": 8,
    }

    def __init__(self, path: Path):
        self.path = path
        self.identity = _regular_owned_file(path)
        with path.open("rb", buffering=0) as handle:
            header_bytes = handle.read(8)
            if len(header_bytes) != 8:
                raise ReconstructionError(f"truncated safetensors header: {path}")
            (header_size,) = struct.unpack("<Q", header_bytes)
            if header_size <= 1 or header_size > 256 * 1024 * 1024:
                raise ReconstructionError(f"implausible safetensors header size: {path}")
            encoded = handle.read(header_size)
        if len(encoded) != header_size:
            raise ReconstructionError(f"truncated safetensors JSON header: {path}")
        try:
            header = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReconstructionError(f"invalid safetensors JSON header {path}: {exc}") from exc
        if not isinstance(header, dict):
            raise ReconstructionError(f"safetensors header is not an object: {path}")
        self.data_offset = 8 + header_size
        tensors: dict[str, SafeTensorInfo] = {}
        intervals: list[tuple[int, int, str]] = []
        for name, raw in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(name, str) or not isinstance(raw, dict):
                raise ReconstructionError(f"invalid tensor entry in {path}")
            dtype = raw.get("dtype")
            shape = raw.get("shape")
            offsets = raw.get("data_offsets")
            if dtype not in self._WIDTHS:
                raise ReconstructionError(f"unsupported safetensors dtype {dtype!r} in {path}")
            if (
                not isinstance(shape, list)
                or not all(isinstance(value, int) and value >= 0 for value in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(value, int) and value >= 0 for value in offsets)
            ):
                raise ReconstructionError(f"invalid tensor metadata for {name} in {path}")
            start, end = offsets
            shape_tuple = tuple(shape)
            n_elements = int(np.prod(shape_tuple, dtype=np.int64))
            if end < start or end - start != n_elements * self._WIDTHS[dtype]:
                raise ReconstructionError(f"tensor byte extent differs for {name} in {path}")
            absolute_start = self.data_offset + start
            absolute_end = self.data_offset + end
            if absolute_end > self.identity.st_size:
                raise ReconstructionError(f"tensor exceeds safetensors file: {name} in {path}")
            tensors[name] = SafeTensorInfo(
                name=name,
                dtype=dtype,
                shape=shape_tuple,
                start=absolute_start,
                end=absolute_end,
            )
            intervals.append((absolute_start, absolute_end, name))
        intervals.sort()
        for previous, current in pairwise(intervals):
            if previous[1] > current[0]:
                raise ReconstructionError(
                    f"overlapping safetensors payloads: {previous[2]} and {current[2]}"
                )
        self.tensors = tensors

    def raw(self, name: str, *, mode: str = "r") -> np.memmap:
        info = self.tensors[name]
        return np.memmap(
            self.path,
            mode=mode,
            dtype=np.uint8,
            offset=info.start,
            shape=(info.end - info.start,),
        )

    def float32(self, name: str) -> np.ndarray:
        info = self.tensors[name]
        if info.dtype == "BF16":
            raw = np.memmap(
                self.path,
                mode="r",
                dtype="<u2",
                offset=info.start,
                shape=(info.n_elements,),
            )
            return bf16_to_float32(raw).reshape(info.shape)
        dtype = {"F16": "<f2", "F32": "<f4", "I32": "<i4", "I64": "<i8"}[info.dtype]
        raw = np.memmap(
            self.path,
            mode="r",
            dtype=dtype,
            offset=info.start,
            shape=(info.n_elements,),
        )
        return np.asarray(raw, dtype=np.float32).reshape(info.shape)

    def verify_stable(self) -> None:
        after = self.path.lstat()
        before_key = (
            self.identity.st_dev,
            self.identity.st_ino,
            self.identity.st_size,
            self.identity.st_mtime_ns,
        )
        after_key = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if before_key != after_key:
            raise ReconstructionError(f"safetensors file changed during use: {self.path}")


def bf16_to_float32(value: np.ndarray) -> np.ndarray:
    words = np.asarray(value, dtype="<u2")
    expanded = np.asarray(words, dtype=np.uint32) << np.uint32(16)
    return expanded.view(np.float32)


def float32_to_bf16(value: np.ndarray) -> np.ndarray:
    """Round float32 to BF16 using round-to-nearest, ties-to-even."""

    floats = np.asarray(value, dtype=np.float32)
    words = floats.view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((words >> np.uint32(16)) & np.uint32(1))
    rounded = words + rounding
    # Preserve NaNs as NaNs even if the rounded payload would become infinity.
    exponent = words & np.uint32(0x7F800000)
    mantissa = words & np.uint32(0x007FFFFF)
    nan_mask = (exponent == np.uint32(0x7F800000)) & (mantissa != 0)
    result = (rounded >> np.uint32(16)).astype("<u2")
    if np.any(nan_mask):
        result[nan_mask] |= np.uint16(0x0040)
    return result


def hf_to_gguf_name(name: str, *, n_layers: int = 64) -> str | None:
    """Map one official Qwen3.8 safetensors name to its GGUF tensor name."""

    mapping = get_tensor_name_map(MODEL_ARCH.QWEN35, n_layers)
    match = _DT_BIAS_RE.match(name)
    if match:
        return f"blk.{match.group(1)}.ssm_dt.bias"
    if name.startswith("model.visual."):
        return None
    if name.startswith("mtp."):
        # Embedded Qwen3.8 NextN tensors use the same stable ``mtp.`` names in
        # Hauhau's release.  Inventory verifies this against the actual file.
        return name
    candidate = name.replace("model.language_model.", "model.", 1)
    return mapping.get_name(candidate, try_suffixes=(".weight", ".bias"))


@dataclass(frozen=True)
class GgufTensor:
    name: str
    qtype: str
    shape: tuple[int, ...]
    n_elements: int
    n_bytes: int
    data_sha256: str


class GgufModel:
    """Authenticated read-only facade around ``GGUFReader``."""

    def __init__(self, path: Path):
        self.path = path
        self.identity = _regular_owned_file(path)
        try:
            self.reader = GGUFReader(path, "r")
        except (OSError, ValueError, KeyError) as exc:
            raise ReconstructionError(f"cannot parse GGUF file {path}: {exc}") from exc
        self.by_name = {tensor.name: tensor for tensor in self.reader.tensors}
        if len(self.by_name) != len(self.reader.tensors):
            raise ReconstructionError(f"GGUF has duplicate tensor names: {path}")

    def tensor(self, name: str) -> Any:
        try:
            return self.by_name[name]
        except KeyError as exc:
            raise ReconstructionError(f"GGUF tensor is missing: {name} in {self.path}") from exc

    def descriptor(self, name: str) -> GgufTensor:
        tensor = self.tensor(name)
        return GgufTensor(
            name=name,
            qtype=tensor.tensor_type.name,
            shape=tuple(int(value) for value in tensor.shape),
            n_elements=int(tensor.n_elements),
            n_bytes=int(tensor.n_bytes),
            data_sha256=sha256_bytes(memoryview(tensor.data)),
        )

    def float32(self, name: str, expected_shape: tuple[int, ...]) -> np.ndarray:
        tensor = self.tensor(name)
        try:
            value = dequantize(tensor.data, GGMLQuantizationType(tensor.tensor_type))
        except (ValueError, TypeError, NotImplementedError) as exc:
            raise ReconstructionError(
                f"cannot dequantize {name} ({tensor.tensor_type.name}) from {self.path}: {exc}"
            ) from exc
        value = np.asarray(value, dtype=np.float32)
        if value.shape == expected_shape:
            return value
        if value.ndim == 2 and value.T.shape == expected_shape:
            return np.ascontiguousarray(value.T)
        if value.size == int(np.prod(expected_shape, dtype=np.int64)):
            return np.ascontiguousarray(value.reshape(expected_shape))
        raise ReconstructionError(
            f"GGUF tensor shape differs for {name}: got {value.shape}, expected {expected_shape}"
        )

    def verify_stable(self) -> None:
        after = self.path.lstat()
        before_key = (
            self.identity.st_dev,
            self.identity.st_ino,
            self.identity.st_size,
            self.identity.st_mtime_ns,
        )
        after_key = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if before_key != after_key:
            raise ReconstructionError(f"GGUF file changed during use: {self.path}")


def randomized_svd(
    matrix: np.ndarray,
    *,
    rank: int = 8,
    oversample: int = 4,
    power_iterations: int = 2,
    seed: int = 0x484155,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute deterministic leading singular factors with bounded memory."""

    value = np.asarray(matrix, dtype=np.float32)
    if value.ndim != 2:
        raise ReconstructionError("randomized SVD requires a matrix")
    maximum = min(value.shape)
    if maximum == 0:
        raise ReconstructionError("randomized SVD cannot process an empty matrix")
    width = min(maximum, rank + oversample)
    selected_rank = min(rank, width)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((value.shape[1], width), dtype=np.float32)
    q, _ = np.linalg.qr(value @ omega, mode="reduced")
    for _ in range(power_iterations):
        z, _ = np.linalg.qr(value.T @ q, mode="reduced")
        q, _ = np.linalg.qr(value @ z, mode="reduced")
    reduced = q.T @ value
    small_u, singular, vh = np.linalg.svd(reduced, full_matrices=False)
    u = q @ small_u
    return (
        np.asarray(u[:, :selected_rank], dtype=np.float32),
        np.asarray(singular[:selected_rank], dtype=np.float32),
        np.asarray(vh[:selected_rank], dtype=np.float32),
    )


def factor_evidence(
    matrix: np.ndarray, singular: np.ndarray, *, quant_error_rms: float | None = None
) -> dict[str, Any]:
    """Return JSON-safe evidence for choosing a coherent low-rank delta."""

    value = np.asarray(matrix, dtype=np.float32)
    squared_norm, max_abs = array_energy(value)
    squared_singular = np.square(np.asarray(singular, dtype=np.float64))
    cumulative = np.cumsum(squared_singular)
    rms = float(np.sqrt(squared_norm / value.size)) if value.size else 0.0
    result: dict[str, Any] = {
        "elements": int(value.size),
        "frobenius_norm": float(np.sqrt(squared_norm)),
        "rms": rms,
        "max_abs": max_abs,
        "singular_values": [float(item) for item in singular],
        "explained_energy": [
            float(item / squared_norm) if squared_norm else 0.0 for item in cumulative
        ],
    }
    if quant_error_rms is not None:
        result["rms_over_vanilla_q8_error"] = (
            rms / quant_error_rms if quant_error_rms > 0.0 else None
        )
    return result


def array_energy(
    value: np.ndarray, *, chunk_elements: int = 4 * 1024 * 1024
) -> tuple[float, float]:
    """Compute squared L2 norm and max magnitude without a full float64 temporary."""

    flat = np.asarray(value, dtype=np.float32).reshape(-1)
    squared_norm = 0.0
    max_abs = 0.0
    for start in range(0, flat.size, chunk_elements):
        chunk = np.asarray(flat[start : start + chunk_elements], dtype=np.float64)
        squared_norm += float(np.dot(chunk, chunk))
        if chunk.size:
            max_abs = max(max_abs, float(np.max(np.abs(chunk))))
    return squared_norm, max_abs


def iter_index_tensors(
    source_root: Path,
) -> Iterator[tuple[str, Path, SafeTensorInfo]]:
    index_path = source_root / "model.safetensors.index.json"
    index = load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ReconstructionError(f"invalid weight_map in {index_path}")
    shard_cache: dict[Path, SafeTensorFile] = {}
    for name in sorted(weight_map):
        relative = weight_map[name]
        if not isinstance(name, str) or not isinstance(relative, str):
            raise ReconstructionError(f"invalid weight_map entry in {index_path}")
        shard_path = source_root / relative
        shard = shard_cache.setdefault(shard_path, SafeTensorFile(shard_path))
        if name not in shard.tensors:
            raise ReconstructionError(f"index points to missing tensor {name} in {shard_path}")
        yield name, shard_path, shard.tensors[name]


def tensor_scope(name: str) -> str:
    if name.startswith("model.visual."):
        return "vision"
    if name.startswith("mtp."):
        return "mtp"
    if _LAYER_RE.match(name) or name in {
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
    }:
        return "language"
    return "unknown"


def describe_source_file(path: Path, *, repository: str, revision: str) -> dict[str, Any]:
    digest, size = stable_sha256(path)
    return {
        "path": str(path),
        "repository": repository,
        "revision": revision,
        "bytes": size,
        "sha256": digest,
    }


def copy_exact(source: BinaryIO, destination: BinaryIO, *, length: int) -> None:
    remaining = length
    while remaining:
        payload = source.read(min(8 * 1024 * 1024, remaining))
        if not payload:
            raise ReconstructionError("source ended during exact copy")
        destination.write(payload)
        remaining -= len(payload)
