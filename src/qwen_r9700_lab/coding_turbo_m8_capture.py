# QWEN_ASSURANCE_ONLY_BEGIN: coding-turbo-m8-runtime-capture
"""Capture and pair real Quest96 standalone/M8 row-invariance trials.

This module is installed only in the generated assurance artifact.  The Quest
bridge records hashes of the exact production tensors for one fixed logical
row, and the rejection sampler later supplies that row's authoritative target
top-1.  Raw standalone and M8 records are immutable and paired offline; no
tensor values, token text, prompt text, or decoded output are persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RAW_SCHEMA = "urn:qwen-r9700:coding-turbo-m8-runtime-trial:v1"
ARTIFACT_SCHEMA = "urn:qwen-r9700:coding-turbo-artifact:v1"
CAPTURE_OUTPUT = "assurance/qwen_r9700_lab/coding_turbo_m8_capture.py"
QUALIFIED_CONTEXTS = {60_298, 249_957}
QUALIFIED_ATTENTION_LAYERS = 16
FIXED_ROWS = tuple(range(8))
TRIAL_KINDS = {"row_permutation", "sibling_randomization"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

ENABLE_ENV = "QWEN_CODING_TURBO_M8_CAPTURE"
OUTPUT_ENV = "QWEN_CODING_TURBO_M8_CAPTURE_OUTPUT"
MANIFEST_ENV = "QWEN_CODING_TURBO_ASSURANCE_MANIFEST"
CONTEXT_ENV = "QWEN_CODING_TURBO_M8_CONTEXT_TOKENS"
TARGET_POSITION_ENV = "QWEN_CODING_TURBO_M8_TARGET_POSITION"
FIXED_ROW_ENV = "QWEN_CODING_TURBO_M8_FIXED_ROW"
PHYSICAL_ROW_ENV = "QWEN_CODING_TURBO_M8_PHYSICAL_ROW"
TRIAL_KIND_ENV = "QWEN_CODING_TURBO_M8_TRIAL_KIND"
TRIAL_INDEX_ENV = "QWEN_CODING_TURBO_M8_TRIAL_INDEX"
PERTURBATION_ENV = "QWEN_CODING_TURBO_M8_PERTURBATION_SHA256"
MODE_ENV = "QWEN_CODING_TURBO_M8_MODE"

RAW_KEYS = {
    "artifact_manifest_sha256",
    "attention_output_sha256",
    "cache_mapping_sha256",
    "context_tokens",
    "fixed_row",
    "fixed_row_input_sha256",
    "layer_count",
    "layer_names_sha256",
    "mode",
    "perturbation_sha256",
    "physical_row",
    "raw_sha256",
    "schema",
    "selected_pages_sha256",
    "semantic_source_sha256",
    "target_position",
    "target_top1",
    "trial_index",
    "trial_kind",
}


class M8CaptureError(RuntimeError):
    """The live M8 trial is incomplete, unauthenticated, or divergent."""


@dataclass(frozen=True)
class CaptureConfig:
    output: Path
    artifact_manifest_sha256: str
    semantic_source_sha256: str
    context_tokens: int
    target_position: int
    fixed_row: int
    physical_row: int
    trial_kind: str
    trial_index: int
    perturbation_sha256: str
    mode: str


_LOCK = threading.Lock()
_CONFIG: CaptureConfig | None = None
_LAYER_RECORDS: dict[str, dict[str, str]] = {}
_PUBLISHED = False


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _document_digest(value: dict[str, Any], digest_key: str) -> str:
    unsigned = dict(value)
    unsigned.pop(digest_key, None)
    return _sha256_bytes(_canonical_json(unsigned))


def _owned_regular(path: Path, label: str, *, owner_only: bool = False) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise M8CaptureError(f"{label} does not exist: {path}") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise M8CaptureError(f"{label} must be a regular non-symlink file: {path}")
    if metadata.st_uid != os.getuid():
        raise M8CaptureError(f"{label} has the wrong owner: {path}")
    forbidden = 0o077 if owner_only else 0o022
    if stat.S_IMODE(metadata.st_mode) & forbidden:
        raise M8CaptureError(f"{label} has unsafe permissions: {path}")
    return metadata


def _owned_private_parent(path: Path) -> None:
    parent = path.parent
    try:
        metadata = parent.lstat()
    except FileNotFoundError as error:
        raise M8CaptureError(f"capture output parent does not exist: {parent}") from error
    if not stat.S_ISDIR(metadata.st_mode) or parent.is_symlink():
        raise M8CaptureError("capture output parent must be a real directory")
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise M8CaptureError("capture output parent must be owner-controlled mode 0700")


def _integer(raw: str, label: str, *, minimum: int, maximum: int) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise M8CaptureError(f"{label} must be an integer") from error
    if not minimum <= value <= maximum:
        raise M8CaptureError(f"{label} must be in [{minimum}, {maximum}]")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise M8CaptureError(f"{label} must be a lowercase SHA-256")
    return value


def _tensor_sha256(value: Any) -> str:
    contiguous = value.detach().contiguous()
    descriptor = {
        "device_type": contiguous.device.type,
        "dtype": str(contiguous.dtype),
        "shape": [int(size) for size in contiguous.shape],
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    digest.update(contiguous.view(__import__("torch").uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _aggregate_sha256(records: Sequence[dict[str, str]], key: str) -> str:
    return _sha256_bytes(
        _canonical_json(
            [{"layer_name": record["layer_name"], key: record[key]} for record in records]
        )
    )


def _load_config() -> CaptureConfig | None:
    global _CONFIG
    if os.environ.get(ENABLE_ENV, "0") != "1":
        return None
    if _CONFIG is not None:
        return _CONFIG

    manifest_raw = os.environ.get(MANIFEST_ENV, "")
    output_raw = os.environ.get(OUTPUT_ENV, "")
    if not manifest_raw or not output_raw:
        raise M8CaptureError("M8 capture manifest/output environment is incomplete")
    manifest_path = Path(manifest_raw)
    output = Path(output_raw)
    _owned_regular(manifest_path, "assurance artifact manifest", owner_only=True)
    _owned_private_parent(output)
    if output.exists() or output.is_symlink():
        raise M8CaptureError(f"refusing to replace existing M8 capture: {output}")
    manifest_payload = manifest_path.read_bytes()
    if len(manifest_payload) > 16 * 1024 * 1024:
        raise M8CaptureError("assurance artifact manifest exceeds 16 MiB")
    try:
        manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise M8CaptureError(f"invalid assurance artifact manifest: {error}") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != ARTIFACT_SCHEMA
        or manifest.get("artifact_kind") != "assurance"
    ):
        raise M8CaptureError("M8 capture requires an assurance artifact manifest")
    semantic_source = _sha256(manifest.get("semantic_source_sha256"), "semantic source")
    live_source = Path(__file__)
    live_sha256 = _sha256_bytes(live_source.read_bytes())
    entries = manifest.get("files")
    matches = (
        [
            entry
            for entry in entries
            if isinstance(entry, dict) and entry.get("path") == CAPTURE_OUTPUT
        ]
        if isinstance(entries, list)
        else []
    )
    if len(matches) != 1 or matches[0].get("sha256") != live_sha256:
        raise M8CaptureError("assurance manifest does not authenticate the live M8 capture module")

    context = _integer(
        os.environ.get(CONTEXT_ENV, ""),
        "context tokens",
        minimum=1,
        maximum=1_000_000,
    )
    if context not in QUALIFIED_CONTEXTS:
        raise M8CaptureError("M8 capture context is not qualified")
    target_position = _integer(
        os.environ.get(TARGET_POSITION_ENV, ""),
        "target position",
        minimum=context,
        maximum=context + 1_000_000,
    )
    fixed_row = _integer(os.environ.get(FIXED_ROW_ENV, ""), "fixed row", minimum=0, maximum=7)
    physical_row = _integer(
        os.environ.get(PHYSICAL_ROW_ENV, str(fixed_row)),
        "physical row",
        minimum=0,
        maximum=7,
    )
    trial_kind = os.environ.get(TRIAL_KIND_ENV, "")
    if trial_kind not in TRIAL_KINDS:
        raise M8CaptureError("M8 capture trial kind is invalid")
    trial_index = _integer(
        os.environ.get(TRIAL_INDEX_ENV, ""), "trial index", minimum=0, maximum=2**31 - 1
    )
    perturbation = _sha256(os.environ.get(PERTURBATION_ENV, ""), "perturbation")
    mode = os.environ.get(MODE_ENV, "")
    if mode not in {"m8", "standalone"}:
        raise M8CaptureError("M8 capture mode must be standalone or m8")
    _CONFIG = CaptureConfig(
        output=output,
        artifact_manifest_sha256=_sha256_bytes(manifest_payload),
        semantic_source_sha256=semantic_source,
        context_tokens=context,
        target_position=target_position,
        fixed_row=fixed_row,
        physical_row=physical_row,
        trial_kind=trial_kind,
        trial_index=trial_index,
        perturbation_sha256=perturbation,
        mode=mode,
    )
    return _CONFIG


def capture_enabled() -> bool:
    """Return whether the explicit assurance-only M8 capture is requested."""

    return os.environ.get(ENABLE_ENV, "0") == "1"


def record_quest_attention(
    *,
    layer_name: str | None,
    query: Any,
    current_key: Any,
    current_value: Any,
    selected_rows: Any,
    attention_output: Any,
    logical_physical_blocks: Any,
    committed_prefix_length: int,
    logical_row: int | None = None,
) -> None:
    """Record one production full-attention layer for the configured fixed row."""

    config = _load_config()
    if config is None:
        return
    if not layer_name or not isinstance(layer_name, str):
        raise M8CaptureError("Quest layer name is absent from M8 capture")
    if logical_row is not None and logical_row != config.fixed_row:
        return
    if committed_prefix_length != config.target_position - config.fixed_row:
        return
    runtime_row = config.physical_row if config.mode == "m8" else 0
    if not 0 <= runtime_row < int(query.shape[0]):
        raise M8CaptureError("configured physical row is absent from the Quest query")
    if not 0 <= runtime_row < int(attention_output.shape[0]):
        raise M8CaptureError("configured physical row is absent from the Quest output")
    if int(selected_rows.ndim) != 2 or runtime_row >= int(selected_rows.shape[0]):
        raise M8CaptureError("configured physical row is absent from Quest page selection")
    if runtime_row >= int(current_key.shape[0]) or runtime_row >= int(current_value.shape[0]):
        raise M8CaptureError("configured physical row is absent from current K/V")
    query_row = query[runtime_row : runtime_row + 1]
    current_key_row = current_key[runtime_row : runtime_row + 1]
    current_value_row = current_value[runtime_row : runtime_row + 1]
    selected_pages = selected_rows[runtime_row]
    attention_output_row = attention_output[runtime_row : runtime_row + 1]
    if int(query_row.shape[0]) != 1 or int(attention_output_row.shape[0]) != 1:
        raise M8CaptureError("M8 capture rows must be sliced to exactly one row")
    if int(current_key_row.shape[0]) != 1 or int(current_value_row.shape[0]) != 1:
        raise M8CaptureError("M8 capture current K/V must contain exactly one row")
    required_blocks = (committed_prefix_length + 1_647) // 1_648
    if required_blocks <= 0 or int(logical_physical_blocks.numel()) < required_blocks:
        raise M8CaptureError("M8 capture cache mapping is incomplete")
    input_sha256 = _sha256_bytes(
        _canonical_json(
            {
                "current_key_sha256": _tensor_sha256(current_key_row),
                "current_value_sha256": _tensor_sha256(current_value_row),
                "query_sha256": _tensor_sha256(query_row),
            }
        )
    )
    mapping_sha256 = _sha256_bytes(
        _canonical_json(
            {
                "committed_prefix_length": committed_prefix_length,
                "physical_blocks_sha256": _tensor_sha256(logical_physical_blocks[:required_blocks]),
            }
        )
    )
    record = {
        "attention_output_sha256": _tensor_sha256(attention_output_row),
        "cache_mapping_sha256": mapping_sha256,
        "fixed_row_input_sha256": input_sha256,
        "layer_name": layer_name,
        "selected_pages_sha256": _tensor_sha256(selected_pages),
    }
    with _LOCK:
        if _PUBLISHED:
            raise M8CaptureError("M8 capture observed attention after immutable publication")
        previous = _LAYER_RECORDS.get(layer_name)
        if previous is not None and previous != record:
            raise M8CaptureError(f"M8 capture layer {layer_name} was observed with different bytes")
        _LAYER_RECORDS[layer_name] = record
        if len(_LAYER_RECORDS) > QUALIFIED_ATTENTION_LAYERS:
            raise M8CaptureError("M8 capture observed more than 16 full-attention layers")


def record_target_round(*, positions: Sequence[int], target_top1: Sequence[int]) -> None:
    """Publish a complete raw trial after the authoritative sampler is reached."""

    global _PUBLISHED
    config = _load_config()
    if config is None:
        return
    matches = [
        index for index, position in enumerate(positions) if position == config.target_position
    ]
    if not matches:
        return
    if len(matches) != 1:
        raise M8CaptureError("configured target position is not unique in the sampler round")
    physical_row = matches[0]
    if physical_row != config.physical_row:
        raise M8CaptureError(
            f"target position mapped to physical row {physical_row}, expected {config.physical_row}"
        )
    if len(target_top1) != len(positions):
        raise M8CaptureError("sampler target-top1 and position lengths differ")
    with _LOCK:
        if _PUBLISHED:
            raise M8CaptureError("M8 capture attempted to publish twice")
        if len(_LAYER_RECORDS) != QUALIFIED_ATTENTION_LAYERS:
            raise M8CaptureError(
                "M8 capture reached target logits without exactly 16 attention-layer records"
            )
        ordered = [_LAYER_RECORDS[name] for name in sorted(_LAYER_RECORDS)]
        raw = {
            "artifact_manifest_sha256": config.artifact_manifest_sha256,
            "attention_output_sha256": _aggregate_sha256(ordered, "attention_output_sha256"),
            "cache_mapping_sha256": _aggregate_sha256(ordered, "cache_mapping_sha256"),
            "context_tokens": config.context_tokens,
            "fixed_row": config.fixed_row,
            "fixed_row_input_sha256": _aggregate_sha256(ordered, "fixed_row_input_sha256"),
            "layer_count": len(ordered),
            "layer_names_sha256": _sha256_bytes(
                _canonical_json([record["layer_name"] for record in ordered])
            ),
            "mode": config.mode,
            "perturbation_sha256": config.perturbation_sha256,
            "physical_row": config.physical_row,
            "schema": RAW_SCHEMA,
            "selected_pages_sha256": _aggregate_sha256(ordered, "selected_pages_sha256"),
            "semantic_source_sha256": config.semantic_source_sha256,
            "target_position": config.target_position,
            "target_top1": int(target_top1[physical_row]),
            "trial_index": config.trial_index,
            "trial_kind": config.trial_kind,
        }
        raw["raw_sha256"] = _document_digest(raw, "raw_sha256")
        payload = _canonical_json(raw)
        descriptor = os.open(
            config.output,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        directory_descriptor = os.open(
            config.output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        _PUBLISHED = True


def validate_raw_trial(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != RAW_KEYS:
        raise M8CaptureError("raw M8 trial has unknown or missing fields")
    raw = dict(value)
    if raw["schema"] != RAW_SCHEMA:
        raise M8CaptureError("raw M8 trial schema mismatch")
    for key in (
        "artifact_manifest_sha256",
        "attention_output_sha256",
        "cache_mapping_sha256",
        "fixed_row_input_sha256",
        "layer_names_sha256",
        "perturbation_sha256",
        "raw_sha256",
        "selected_pages_sha256",
        "semantic_source_sha256",
    ):
        raw[key] = _sha256(raw[key], key)
    if raw["raw_sha256"] != _document_digest(raw, "raw_sha256"):
        raise M8CaptureError("raw M8 trial self-hash mismatch")
    for key, minimum, maximum in (
        ("context_tokens", 1, 1_000_000),
        ("fixed_row", 0, 7),
        ("physical_row", 0, 7),
        ("target_position", 1, 2_000_000),
        ("target_top1", 0, 2**31 - 1),
        ("trial_index", 0, 2**31 - 1),
    ):
        if (
            isinstance(raw[key], bool)
            or not isinstance(raw[key], int)
            or not minimum <= raw[key] <= maximum
        ):
            raise M8CaptureError(f"raw M8 trial {key} is invalid")
    if raw["context_tokens"] not in QUALIFIED_CONTEXTS:
        raise M8CaptureError("raw M8 trial context is not qualified")
    if raw["fixed_row"] not in FIXED_ROWS or raw["trial_kind"] not in TRIAL_KINDS:
        raise M8CaptureError("raw M8 trial identity is invalid")
    if raw["mode"] not in {"standalone", "m8"}:
        raise M8CaptureError("raw M8 trial mode is invalid")
    if raw["layer_count"] != QUALIFIED_ATTENTION_LAYERS:
        raise M8CaptureError("raw M8 trial does not contain 16 full-attention layers")
    return raw


def load_raw_trial(path: Path) -> dict[str, Any]:
    _owned_regular(path, "raw M8 trial", owner_only=True)
    payload = path.read_bytes()
    if len(payload) > 1024 * 1024:
        raise M8CaptureError("raw M8 trial exceeds 1 MiB")
    try:
        return validate_raw_trial(json.loads(payload))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise M8CaptureError(f"invalid raw M8 trial JSON: {error}") from error


def _pair_identity(raw: dict[str, Any]) -> tuple[object, ...]:
    return (
        raw["artifact_manifest_sha256"],
        raw["semantic_source_sha256"],
        raw["context_tokens"],
        raw["fixed_row"],
        raw["target_position"],
        raw["trial_kind"],
        raw["trial_index"],
        raw["perturbation_sha256"],
    )


def pair_raw_trials(standalone: object, m8: object) -> dict[str, Any]:
    left = validate_raw_trial(standalone)
    right = validate_raw_trial(m8)
    if left["mode"] != "standalone" or right["mode"] != "m8":
        raise M8CaptureError("raw M8 pair must be standalone then m8")
    if _pair_identity(left) != _pair_identity(right):
        raise M8CaptureError("raw standalone/M8 trial identities differ")
    for key in (
        "fixed_row_input_sha256",
        "selected_pages_sha256",
        "attention_output_sha256",
        "cache_mapping_sha256",
        "layer_names_sha256",
        "target_top1",
    ):
        if left[key] != right[key]:
            raise M8CaptureError(f"raw standalone/M8 trial diverged at {key}")
    return {
        "fixed_row": left["fixed_row"],
        "fixed_row_input_sha256": left["fixed_row_input_sha256"],
        "m8_attention_output_sha256": right["attention_output_sha256"],
        "m8_cache_mapping_sha256": right["cache_mapping_sha256"],
        "m8_capture_sha256": right["raw_sha256"],
        "m8_selected_pages_sha256": right["selected_pages_sha256"],
        "m8_target_top1": right["target_top1"],
        "perturbation_sha256": left["perturbation_sha256"],
        "standalone_attention_output_sha256": left["attention_output_sha256"],
        "standalone_cache_mapping_sha256": left["cache_mapping_sha256"],
        "standalone_capture_sha256": left["raw_sha256"],
        "standalone_selected_pages_sha256": left["selected_pages_sha256"],
        "standalone_target_top1": left["target_top1"],
        "trial_index": left["trial_index"],
        "trial_kind": left["trial_kind"],
    }


def seal_raw_directories(*, standalone_dir: Path, m8_dir: Path, output: Path) -> dict[str, Any]:
    from qwen_r9700_lab.coding_turbo_m8_evidence import seal_evidence

    _owned_private_parent(output)
    if output.exists() or output.is_symlink():
        raise M8CaptureError(f"refusing to replace existing sealed M8 evidence: {output}")
    for directory, label in ((standalone_dir, "standalone"), (m8_dir, "m8")):
        metadata = directory.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or directory.is_symlink():
            raise M8CaptureError(f"{label} raw-trial root must be a real directory")
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
            raise M8CaptureError(f"{label} raw-trial root must be owner-controlled mode 0700")

    def indexed(directory: Path, mode: str) -> dict[tuple[object, ...], dict[str, Any]]:
        result: dict[tuple[object, ...], dict[str, Any]] = {}
        paths = sorted(directory.glob("*.json"))
        if not paths:
            raise M8CaptureError(f"{mode} raw-trial root contains no JSON files")
        for path in paths:
            raw = load_raw_trial(path)
            if raw["mode"] != mode:
                raise M8CaptureError(f"{path} has mode {raw['mode']}, expected {mode}")
            identity = _pair_identity(raw)
            if identity in result:
                raise M8CaptureError(f"duplicate raw M8 trial identity in {mode}: {identity}")
            result[identity] = raw
        return result

    standalone = indexed(standalone_dir, "standalone")
    m8 = indexed(m8_dir, "m8")
    if set(standalone) != set(m8):
        raise M8CaptureError("standalone and M8 raw-trial identity sets differ")
    pairs = [pair_raw_trials(standalone[identity], m8[identity]) for identity in sorted(standalone)]
    first = standalone[next(iter(sorted(standalone)))]
    contexts = {record["context_tokens"] for record in standalone.values()}
    artifacts = {record["artifact_manifest_sha256"] for record in standalone.values()}
    semantic_sources = {record["semantic_source_sha256"] for record in standalone.values()}
    if len(contexts) != 1 or len(artifacts) != 1 or len(semantic_sources) != 1:
        raise M8CaptureError("raw M8 evidence spans multiple contexts or artifact identities")
    evidence = seal_evidence(
        {
            "artifact_manifest_sha256": first["artifact_manifest_sha256"],
            "context_tokens": first["context_tokens"],
            "fixed_rows": list(FIXED_ROWS),
            "schema": "urn:qwen-r9700:coding-turbo-m8-row-invariance:v2",
            "semantic_source_sha256": first["semantic_source_sha256"],
            "trials": pairs,
        }
    )
    payload = _canonical_json(evidence)
    descriptor = os.open(
        output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seal paired production Quest96 M8 trials.")
    parser.add_argument("--standalone-dir", type=Path, required=True)
    parser.add_argument("--m8-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = seal_raw_directories(
        standalone_dir=args.standalone_dir,
        m8_dir=args.m8_dir,
        output=args.output,
    )
    print(json.dumps({"evidence_sha256": result["evidence_sha256"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
# QWEN_ASSURANCE_ONLY_END: coding-turbo-m8-runtime-capture
