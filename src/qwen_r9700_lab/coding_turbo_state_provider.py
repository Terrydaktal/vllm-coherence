# QWEN_ASSURANCE_ONLY_BEGIN: authoritative-state-provider
"""Explicit post-commit state evidence for the Quest96 assurance artifact.

The capture hook must never discover mutable runtime state by walking arbitrary
runner attributes.  The pinned runtime instead supplies one exporter whose
component names and implementation hashes are authenticated by a producer
receipt.  This module validates and aggregates that export after accepted-path
commit.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

PRODUCER_RECEIPT_SCHEMA = "urn:qwen-r9700:coding-turbo-state-producer-receipt:v1"
CONSUMER_TRANSITION_SCHEMA = "urn:qwen-r9700:coding-turbo-state-consumer-transition-receipt:v1"
STATE_EXPORT_SCHEMA = "urn:qwen-r9700:coding-turbo-authoritative-state-export:v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

QUALIFIED_MAMBA_LAYER_NAMES = tuple(
    f"language_model.model.layers.{layer}.linear_attn"
    for quartet in range(16)
    for layer in range(quartet * 4, quartet * 4 + 3)
)
QUALIFIED_GDN_STATE_NAMES = tuple(
    f"{name}.canonical_ssm_state" for name in QUALIFIED_MAMBA_LAYER_NAMES
)
QUALIFIED_CONVOLUTION_STATE_NAMES = tuple(
    f"{name}.convolution_history" for name in QUALIFIED_MAMBA_LAYER_NAMES
)

PRODUCER_RECEIPT_KEYS = {
    "accepted_path_commit_sha256",
    "cache_mapping_exporter_sha256",
    "convolution_layer_names",
    "draft_kv_exporter_sha256",
    "fixed_slot_state_exporter_sha256",
    "gdn_layer_names",
    "receipt_sha256",
    "runtime_artifact_manifest_sha256",
    "schema",
    "semantic_source_sha256",
    "snapshot_payload_receipt_sha256",
    "snapshot_format_sha256",
    "target_kv_exporter_sha256",
}
CONSUMER_TRANSITION_KEYS = {
    "accepted_path_patch_sha256",
    "consumer_accepted_path_commit_sha256",
    "consumer_capability_sha256",
    "consumer_runtime_artifact_manifest_sha256",
    "consumer_semantic_source_sha256",
    "consumer_snapshot_format_sha256",
    "consumer_state_exporter_sha256",
    "producer_accepted_path_commit_sha256",
    "producer_receipt_sha256",
    "producer_runtime_artifact_manifest_sha256",
    "producer_semantic_source_sha256",
    "producer_snapshot_format_sha256",
    "receipt_sha256",
    "schema",
    "snapshot_format_patch_sha256",
    "snapshot_payload_receipt_sha256",
    "snapshot_selection_sha256",
}
STATE_EXPORT_KEYS = {
    "cache_mapping",
    "canonical_gdn_layers",
    "convolution_layers",
    "device_error_words",
    "draft_kv_pages",
    "draft_kv_scales",
    "rollback_generation",
    "schema",
    "target_kv_pages",
    "target_kv_scales",
}
COMPONENT_EVIDENCE_KEYS = {"name", "nonfinite_count", "tensor_sha256"}


class StateProviderError(RuntimeError):
    """The runtime cannot prove one authoritative post-commit state."""


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _document_digest(value: Mapping[str, Any], digest_key: str) -> str:
    unsigned = dict(value)
    unsigned.pop(digest_key, None)
    return _sha256(_canonical_json(unsigned))


def _require_exact_keys(value: object, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StateProviderError(f"{label} must be an object")
    missing = sorted(keys - set(value))
    extra = sorted(set(value) - keys)
    if missing or extra:
        raise StateProviderError(f"{label} key mismatch: missing={missing} extra={extra}")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise StateProviderError(f"{label} must be a lowercase SHA-256")
    return value


def _require_integer(value: object, label: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise StateProviderError(f"{label} must be an integer in [0, {maximum}]")
    return value


def _layer_names(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != 48:
        raise StateProviderError(f"{label} must contain exactly 48 names")
    if any(not isinstance(name, str) or not name for name in value):
        raise StateProviderError(f"{label} contains an invalid name")
    names = tuple(value)
    if len(set(names)) != 48:
        raise StateProviderError(f"{label} must contain 48 unique names")
    return names


@dataclass(frozen=True, slots=True)
class ProducerReceipt:
    """Authenticated code/semantic identity for the live state exporter."""

    document: dict[str, Any]
    digest: str
    gdn_layer_names: tuple[str, ...]
    convolution_layer_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConsumerTransitionReceipt:
    """Authenticated transition from immutable snapshot producer to live consumer."""

    document: dict[str, Any]
    digest: str


def seal_producer_receipt(value: object) -> dict[str, Any]:
    """Validate and self-hash an unsealed producer receipt document."""

    source_keys = PRODUCER_RECEIPT_KEYS - {"receipt_sha256"}
    source = dict(_require_exact_keys(value, source_keys, "producer receipt"))
    source["receipt_sha256"] = _document_digest(source, "receipt_sha256")
    validate_producer_receipt(source)
    return source


def validate_producer_receipt(value: object) -> ProducerReceipt:
    receipt = dict(_require_exact_keys(value, PRODUCER_RECEIPT_KEYS, "producer receipt"))
    if receipt["schema"] != PRODUCER_RECEIPT_SCHEMA:
        raise StateProviderError("producer receipt schema mismatch")
    for key in PRODUCER_RECEIPT_KEYS - {
        "convolution_layer_names",
        "gdn_layer_names",
        "schema",
    }:
        _require_sha256(receipt[key], f"producer receipt.{key}")
    if receipt["receipt_sha256"] != _document_digest(receipt, "receipt_sha256"):
        raise StateProviderError("producer receipt self-hash mismatch")
    return ProducerReceipt(
        document=receipt,
        digest=receipt["receipt_sha256"],
        gdn_layer_names=_layer_names(receipt["gdn_layer_names"], "gdn_layer_names"),
        convolution_layer_names=_layer_names(
            receipt["convolution_layer_names"], "convolution_layer_names"
        ),
    )


def seal_consumer_transition_receipt(value: object) -> dict[str, Any]:
    """Validate and self-hash an unsealed producer-to-consumer transition."""

    source_keys = CONSUMER_TRANSITION_KEYS - {"receipt_sha256"}
    source = dict(_require_exact_keys(value, source_keys, "consumer transition receipt"))
    source["receipt_sha256"] = _document_digest(source, "receipt_sha256")
    validate_consumer_transition_receipt(source)
    return source


def validate_consumer_transition_receipt(value: object) -> ConsumerTransitionReceipt:
    receipt = dict(
        _require_exact_keys(value, CONSUMER_TRANSITION_KEYS, "consumer transition receipt")
    )
    if receipt["schema"] != CONSUMER_TRANSITION_SCHEMA:
        raise StateProviderError("consumer transition receipt schema mismatch")
    for key in CONSUMER_TRANSITION_KEYS - {"schema"}:
        _require_sha256(receipt[key], f"consumer transition receipt.{key}")
    if receipt["receipt_sha256"] != _document_digest(receipt, "receipt_sha256"):
        raise StateProviderError("consumer transition receipt self-hash mismatch")
    if (
        receipt["producer_accepted_path_commit_sha256"]
        == receipt["consumer_accepted_path_commit_sha256"]
    ):
        raise StateProviderError("consumer transition does not change accepted-path semantics")
    if receipt["producer_snapshot_format_sha256"] == receipt["consumer_snapshot_format_sha256"]:
        raise StateProviderError("consumer transition does not change snapshot-format semantics")
    return ConsumerTransitionReceipt(document=receipt, digest=receipt["receipt_sha256"])


def _component(value: object, label: str) -> dict[str, Any]:
    record = dict(_require_exact_keys(value, COMPONENT_EVIDENCE_KEYS, label))
    name = record["name"]
    if not isinstance(name, str) or not name:
        raise StateProviderError(f"{label}.name is invalid")
    record["tensor_sha256"] = _require_sha256(record["tensor_sha256"], f"{label}.tensor_sha256")
    record["nonfinite_count"] = _require_integer(
        record["nonfinite_count"], f"{label}.nonfinite_count", maximum=2**63 - 1
    )
    return record


def _components(
    value: object,
    label: str,
    *,
    expected_names: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise StateProviderError(f"{label} must be a non-empty array")
    records = [_component(item, f"{label}[{index}]") for index, item in enumerate(value)]
    names = tuple(record["name"] for record in records)
    if len(set(names)) != len(names):
        raise StateProviderError(f"{label} contains duplicate component names")
    if expected_names is not None and names != expected_names:
        raise StateProviderError(f"{label} names/order differ from the producer receipt")
    return records


def _aggregate(records: list[dict[str, Any]]) -> str:
    return _sha256(_canonical_json(records))


Exporter = Callable[..., Mapping[str, Any]]


class AuthoritativeStateProvider:
    """Validate a pinned runtime exporter and return capture-hook state fields."""

    def __init__(self, receipt: ProducerReceipt, exporter: Exporter) -> None:
        if not callable(exporter):
            raise StateProviderError("authoritative exporter is not callable")
        self._receipt = receipt
        self._exporter = exporter

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        export = dict(
            _require_exact_keys(
                self._exporter(**kwargs), STATE_EXPORT_KEYS, "authoritative state export"
            )
        )
        if export["schema"] != STATE_EXPORT_SCHEMA:
            raise StateProviderError("authoritative state export schema mismatch")

        target_kv = _components(export["target_kv_pages"], "target_kv_pages")
        target_scales = _components(export["target_kv_scales"], "target_kv_scales")
        draft_kv = _components(export["draft_kv_pages"], "draft_kv_pages")
        draft_scales = _components(export["draft_kv_scales"], "draft_kv_scales")
        gdn = _components(
            export["canonical_gdn_layers"],
            "canonical_gdn_layers",
            expected_names=self._receipt.gdn_layer_names,
        )
        convolution = _components(
            export["convolution_layers"],
            "convolution_layers",
            expected_names=self._receipt.convolution_layer_names,
        )
        cache_mapping = _component(export["cache_mapping"], "cache_mapping")

        words = export["device_error_words"]
        if not isinstance(words, list) or not words:
            raise StateProviderError("device_error_words must be a non-empty array")
        device_error_word = 0
        for index, word in enumerate(words):
            device_error_word |= _require_integer(
                word, f"device_error_words[{index}]", maximum=2**32 - 1
            )
        rollback_generation = _require_integer(
            export["rollback_generation"], "rollback_generation", maximum=2**31 - 1
        )
        groups = [target_kv, target_scales, draft_kv, draft_scales, gdn, convolution]
        nonfinite_count = cache_mapping["nonfinite_count"] + sum(
            record["nonfinite_count"] for group in groups for record in group
        )
        if device_error_word != 0 or nonfinite_count != 0:
            raise StateProviderError(
                "authoritative state export reports a device error or nonfinite value"
            )

        return {
            "cache_mapping_sha256": cache_mapping["tensor_sha256"],
            "canonical_gdn_layer_sha256": [record["tensor_sha256"] for record in gdn],
            "convolution_layer_sha256": [record["tensor_sha256"] for record in convolution],
            "device_error_word": device_error_word,
            "draft_kv_scales_sha256": _aggregate(draft_scales),
            "draft_kv_sha256": _aggregate(draft_kv),
            "nonfinite_count": nonfinite_count,
            "payload_producer_receipt_sha256": self._receipt.document[
                "snapshot_payload_receipt_sha256"
            ],
            "rollback_generation": rollback_generation,
            "target_kv_scales_sha256": _aggregate(target_scales),
            "target_kv_sha256": _aggregate(target_kv),
        }


def install_authoritative_state_provider(
    runner: object, receipt: ProducerReceipt, exporter: Exporter
) -> AuthoritativeStateProvider:
    """Install the sole explicit post-commit provider on one model runner."""

    if hasattr(runner, "qwen_capture_authoritative_state"):
        raise StateProviderError("runner already has an authoritative state provider")
    provider = AuthoritativeStateProvider(receipt, exporter)
    runner.qwen_capture_authoritative_state = provider
    return provider


# QWEN_ASSURANCE_ONLY_END: authoritative-state-provider
