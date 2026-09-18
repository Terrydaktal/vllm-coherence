"""Authenticate a complete fresh-versus-restored fixed-slot state witness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qwen_r9700_lab import assurance_instrumentation as instrumentation
from qwen_r9700_lab import assurance_provider_producer as provider

CAPTURE_SCHEMA = "urn:qwen-r9700:full-snapshot-restore-capture:v1"
PRODUCER_KIND = "snapshot_controller"
INSTRUMENTATION_SITE = "snapshot.restore.boundary"
EXPECTED_SCOPE = (
    "snapshot_restore",
    None,
    None,
    None,
    instrumentation.UNIT_PHASES["snapshot_restore"],
    None,
)
TARGET_LAYERS = 16
DRAFT_LAYERS = 5
GDN_LAYERS = 48
STATE_KEYS = {
    "logical_target_kv_sha256",
    "physical_target_kv_sha256",
    "logical_target_kv_sha256_by_layer",
    "physical_target_kv_sha256_by_layer",
    "logical_draft_kv_sha256",
    "physical_draft_kv_sha256",
    "logical_draft_kv_sha256_by_layer",
    "physical_draft_kv_sha256_by_layer",
    "gdn_state_sha256_by_layer",
    "convolution_state_sha256_by_layer",
    "positions_sha256",
    "rope_indices_sha256",
    "quant_scales_sha256",
    "cache_mapping_sha256",
    "allocator_generation",
    "allocation_topology_sha256",
    "ownership_sha256",
    "pins_sha256",
    "refcounts_sha256",
    "commit_epoch",
    "transaction_state_sha256",
    "device_error_word",
}
CAPTURE_KEYS = {
    "schema",
    "capture_sha256",
    "semantic_source_sha256",
    "arm",
    "runtime_configuration_sha256",
    "snapshot_manifest_path",
    "snapshot_manifest_sha256",
    "prompt_token_file",
    "prompt_token_file_sha256",
    "prompt_token_sha256",
    "prompt_tokens",
    "fresh",
    "restored",
    "restored_without_cold_fill",
}


class SnapshotProducerError(RuntimeError):
    """Snapshot evidence is incomplete, unauthenticated, or not exact."""


def _digest(value: object, label: str) -> str:
    try:
        return provider._require_digest(value, label)
    except provider.ProviderProducerError as error:
        raise SnapshotProducerError(str(error)) from error


def _exact_mapping(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise SnapshotProducerError(f"{label} keys differ: {observed}")
    return dict(value)


def _integer(value: object, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise SnapshotProducerError(f"{label} must be an integer in [0, {maximum}]")
    return value


def _layer_digests(value: object, label: str, count: int) -> list[str]:
    if not isinstance(value, list) or len(value) != count:
        raise SnapshotProducerError(f"{label} must contain exactly {count} layer digests")
    return [_digest(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _normalize_state(value: object, label: str) -> dict[str, Any]:
    state = _exact_mapping(value, STATE_KEYS, label)
    for key, count in (
        ("logical_target_kv_sha256_by_layer", TARGET_LAYERS),
        ("physical_target_kv_sha256_by_layer", TARGET_LAYERS),
        ("logical_draft_kv_sha256_by_layer", DRAFT_LAYERS),
        ("physical_draft_kv_sha256_by_layer", DRAFT_LAYERS),
        ("gdn_state_sha256_by_layer", GDN_LAYERS),
        ("convolution_state_sha256_by_layer", GDN_LAYERS),
    ):
        state[key] = _layer_digests(state[key], f"{label}.{key}", count)
    for key in STATE_KEYS - {
        "logical_target_kv_sha256_by_layer",
        "physical_target_kv_sha256_by_layer",
        "logical_draft_kv_sha256_by_layer",
        "physical_draft_kv_sha256_by_layer",
        "gdn_state_sha256_by_layer",
        "convolution_state_sha256_by_layer",
        "allocator_generation",
        "commit_epoch",
        "device_error_word",
    }:
        state[key] = _digest(state[key], f"{label}.{key}")
    state["allocator_generation"] = _integer(
        state["allocator_generation"], f"{label}.allocator_generation", 2**63 - 1
    )
    state["commit_epoch"] = _integer(
        state["commit_epoch"], f"{label}.commit_epoch", 2**63 - 1
    )
    state["device_error_word"] = _integer(
        state["device_error_word"], f"{label}.device_error_word", 2**32 - 1
    )
    if state["device_error_word"] != 0:
        raise SnapshotProducerError(f"{label} reports a device error")
    for aggregate, layers in (
        ("logical_target_kv_sha256", "logical_target_kv_sha256_by_layer"),
        ("physical_target_kv_sha256", "physical_target_kv_sha256_by_layer"),
        ("logical_draft_kv_sha256", "logical_draft_kv_sha256_by_layer"),
        ("physical_draft_kv_sha256", "physical_draft_kv_sha256_by_layer"),
    ):
        if state[aggregate] != provider._sha(state[layers]):
            raise SnapshotProducerError(f"{label}.{aggregate} differs from layer digests")
    return state


def _capture_hash(capture: Mapping[str, Any]) -> str:
    unsigned = dict(capture)
    unsigned.pop("capture_sha256", None)
    return provider._sha(unsigned)


def build_evidence(
    capture: object,
    *,
    header: dict[str, Any],
    expected_capture_sha256: str,
) -> dict[str, Any]:
    """Authenticate one exact fresh/restore capture and return fragment evidence."""

    value = _exact_mapping(capture, CAPTURE_KEYS, "snapshot capture")
    if value["schema"] != CAPTURE_SCHEMA:
        raise SnapshotProducerError("snapshot capture schema differs")
    if value["capture_sha256"] != _digest(
        expected_capture_sha256, "snapshot capture"
    ) or value["capture_sha256"] != _capture_hash(value):
        raise SnapshotProducerError("snapshot capture self-hash differs")
    for key in (
        "semantic_source_sha256",
        "runtime_configuration_sha256",
        "snapshot_manifest_sha256",
        "prompt_token_file_sha256",
        "prompt_token_sha256",
    ):
        value[key] = _digest(value[key], f"snapshot capture.{key}")
    if (
        value["semantic_source_sha256"] != header["semantic_source_sha256"]
        or value["arm"] != header["arm"]
        or value["prompt_token_sha256"] != header["prompt_sha256"]
    ):
        raise SnapshotProducerError("snapshot capture identity differs from campaign")
    prompt_tokens = _integer(value["prompt_tokens"], "snapshot prompt_tokens", 1_000_000)
    if prompt_tokens != header["prompt_tokens"]:
        raise SnapshotProducerError("snapshot prompt length differs from campaign")

    if not isinstance(value["prompt_token_file"], str):
        raise SnapshotProducerError("snapshot prompt token file path is invalid")
    prompt_path = Path(value["prompt_token_file"])
    tokens = provider._read_token_vector(prompt_path, value["prompt_token_file_sha256"])
    if (
        len(tokens) != prompt_tokens
        or provider._token_digest(tokens) != value["prompt_token_sha256"]
    ):
        raise SnapshotProducerError("snapshot prompt token vector differs")
    if not isinstance(value["snapshot_manifest_path"], str):
        raise SnapshotProducerError("snapshot manifest path is invalid")
    manifest_path = Path(value["snapshot_manifest_path"])
    provider._read_stable_owned_file(
        manifest_path,
        "snapshot manifest",
        expected_sha256=value["snapshot_manifest_sha256"],
        maximum=16 * 1024 * 1024,
    )
    fresh = _normalize_state(value["fresh"], "fresh state")
    restored = _normalize_state(value["restored"], "restored state")
    if value["restored_without_cold_fill"] is not True:
        raise SnapshotProducerError("snapshot restore used a cold fill")
    if fresh != restored:
        differing = next(key for key in sorted(STATE_KEYS) if fresh[key] != restored[key])
        raise SnapshotProducerError(f"restored state differs from fresh state at {differing}")
    return {
        "snapshot_manifest_sha256": value["snapshot_manifest_sha256"],
        "prompt_token_sha256": value["prompt_token_sha256"],
        **restored,
        "capture_sha256": value["capture_sha256"],
        "runtime_configuration_sha256": value["runtime_configuration_sha256"],
        "restored_without_cold_fill": True,
    }


def _fragment_context() -> tuple[dict[str, Any], str]:
    if os.environ.get(instrumentation.FULL_ASSURANCE_ENABLE_ENV) != "1":
        raise SnapshotProducerError("snapshot fragment was not explicitly enabled")
    missing = [
        name for name in instrumentation.FULL_ASSURANCE_ENVIRONMENT if not os.environ.get(name)
    ]
    if missing:
        raise SnapshotProducerError(f"snapshot fragment environment is incomplete: {missing}")
    if os.environ[instrumentation.FULL_ASSURANCE_PRODUCER_KIND_ENV] != PRODUCER_KIND:
        raise SnapshotProducerError("snapshot producer kind differs")
    source_path = Path(__file__).resolve(strict=True)
    source = provider._read_stable_owned_file(source_path, "snapshot producer source")
    source_sha256 = hashlib.sha256(source).hexdigest()
    if source_sha256 != os.environ[instrumentation.FULL_ASSURANCE_PRODUCER_SHA_ENV]:
        raise SnapshotProducerError("snapshot producer source binding differs")
    header = instrumentation.normalize_header(
        provider._load_json(
            Path(os.environ[instrumentation.FULL_ASSURANCE_HEADER_ENV]),
            "snapshot campaign header",
        )
    )
    scopes = provider._load_json(
        Path(os.environ[instrumentation.FULL_ASSURANCE_SCOPES_ENV]),
        "snapshot runtime scopes",
    )
    if scopes != instrumentation.runtime_scopes_document(header, [EXPECTED_SCOPE]):
        raise SnapshotProducerError("snapshot runtime scope identity differs")
    return header, source_sha256


def produce(capture_path: Path, capture_file_sha256: str) -> dict[str, Any]:
    header, source_sha256 = _fragment_context()
    capture = provider._load_json(
        capture_path,
        "snapshot capture",
        expected_sha256=_digest(capture_file_sha256, "snapshot capture file"),
    )
    evidence = build_evidence(
        capture,
        header=header,
        expected_capture_sha256=capture.get("capture_sha256") if isinstance(capture, dict) else "",
    )
    writer = instrumentation.FullAssuranceFragmentWriter(
        Path(os.environ[instrumentation.FULL_ASSURANCE_ROOT_ENV]),
        header,
        fragment_id=os.environ[instrumentation.FULL_ASSURANCE_FRAGMENT_ID_ENV],
        scopes=(EXPECTED_SCOPE,),
        producer_kind=PRODUCER_KIND,
        producer_sha256=source_sha256,
        sync_interval=1,
    )
    try:
        writer.append(
            unit="snapshot_restore",
            phase=instrumentation.UNIT_PHASES["snapshot_restore"],
            position=None,
            layer_index=None,
            row=None,
            evidence=evidence,
        )
        return writer.finalize(
            device_synchronize_count=0,
            installed_instrumentation_sites=(INSTRUMENTATION_SITE,),
        )
    except BaseException:
        writer.abort()
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qwen-assurance-snapshot-producer",
        description="Verify a complete fresh-versus-fixed-slot-restore state capture.",
    )
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--capture-file-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        result = produce(args.capture, args.capture_file_sha256)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        instrumentation.InstrumentationError,
        provider.ProviderProducerError,
        SnapshotProducerError,
    ) as error:
        print(f"qwen-assurance-snapshot-producer: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
