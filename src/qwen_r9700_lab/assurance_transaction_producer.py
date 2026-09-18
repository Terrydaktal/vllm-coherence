"""Authenticate complete provisional, commit, and fault-atomicity evidence."""

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
from qwen_r9700_lab import live_commit_validation as live_validation

CAPTURE_SCHEMA = "urn:qwen-r9700:full-transaction-capture:v1"
FAULT_CAPTURE_SCHEMA = "urn:qwen-r9700:full-transaction-fault-capture:v1"
PRODUCER_KIND = "transaction_controller"
CAPTURE_KEYS = {
    "schema",
    "capture_sha256",
    "semantic_source_sha256",
    "arm",
    "runtime_configuration_sha256",
    "events",
}
ENTRY_KEYS = {"unit", "phase", "evidence"}
PROVISIONAL_PAIRS = (
    ("canonical_before_sha256", "canonical_after_provisional_sha256"),
    ("allocation_topology_before_sha256", "allocation_topology_after_sha256"),
    ("ownership_before_sha256", "ownership_after_sha256"),
    ("pins_before_sha256", "pins_after_sha256"),
    ("refcounts_before_sha256", "refcounts_after_sha256"),
    ("target_kv_before_sha256_by_layer", "target_kv_after_provisional_sha256_by_layer"),
    ("draft_kv_before_sha256_by_layer", "draft_kv_after_provisional_sha256_by_layer"),
    ("gdn_state_before_sha256_by_layer", "gdn_state_after_provisional_sha256_by_layer"),
    (
        "convolution_state_before_sha256_by_layer",
        "convolution_state_after_provisional_sha256_by_layer",
    ),
)
COMMIT_PAIRS = (
    ("serial_reference_sha256", "published_sha256"),
    ("serial_allocation_topology_sha256", "published_allocation_topology_sha256"),
    ("serial_ownership_sha256", "published_ownership_sha256"),
    ("serial_pins_sha256", "published_pins_sha256"),
    ("serial_refcounts_sha256", "published_refcounts_sha256"),
    ("serial_target_kv_sha256_by_layer", "published_target_kv_sha256_by_layer"),
    ("serial_draft_kv_sha256_by_layer", "published_draft_kv_sha256_by_layer"),
    ("serial_gdn_state_sha256_by_layer", "published_gdn_state_sha256_by_layer"),
    (
        "serial_convolution_state_sha256_by_layer",
        "published_convolution_state_sha256_by_layer",
    ),
)
FAULT_PAIRS = (
    ("canonical_before_sha256", "canonical_after_sha256"),
    ("allocation_topology_before_sha256", "allocation_topology_after_sha256"),
    ("ownership_before_sha256", "ownership_after_sha256"),
    ("pins_before_sha256", "pins_after_sha256"),
    ("refcounts_before_sha256", "refcounts_after_sha256"),
    ("target_kv_before_sha256_by_layer", "target_kv_after_sha256_by_layer"),
    ("draft_kv_before_sha256_by_layer", "draft_kv_after_sha256_by_layer"),
    ("gdn_state_before_sha256_by_layer", "gdn_state_after_sha256_by_layer"),
    ("convolution_state_before_sha256_by_layer", "convolution_state_after_sha256_by_layer"),
)


class TransactionProducerError(RuntimeError):
    """Transaction evidence is incomplete, unauthenticated, or non-atomic."""


def _capture_hash(capture: Mapping[str, Any]) -> str:
    unsigned = dict(capture)
    unsigned.pop("capture_sha256", None)
    return provider._sha(unsigned)


def _exact_mapping(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise TransactionProducerError(f"{label} keys differ: {observed}")
    return dict(value)


def _validate_semantics(entry: dict[str, Any], index: int) -> None:
    unit = entry["unit"]
    evidence = entry["evidence"]
    if evidence["device_error_word"] != 0:
        raise TransactionProducerError(f"transaction event {index} reports a device error")
    if unit == "provisional_isolation":
        for before, after in PROVISIONAL_PAIRS:
            if evidence[before] != evidence[after]:
                raise TransactionProducerError(
                    f"provisional event {index} mutated canonical state at {after}"
                )
    elif unit == "atomic_commit":
        if evidence["provisional_sha256"] != evidence["serial_reference_sha256"]:
            raise TransactionProducerError(
                f"commit event {index} provisional state differs from serial"
            )
        for expected, observed in COMMIT_PAIRS:
            if evidence[expected] != evidence[observed]:
                raise TransactionProducerError(
                    f"commit event {index} published state differs at {observed}"
                )
        count = evidence["commit_count"]
        expected_epoch = evidence["commit_epoch_before"] + (1 if count else 0)
        if evidence["commit_epoch_after"] != expected_epoch:
            raise TransactionProducerError(f"commit event {index} advanced the wrong epoch")
        pointer_changed = (
            evidence["root_pointer_before_sha256"]
            != evidence["root_pointer_after_sha256"]
        )
        if pointer_changed != bool(count):
            raise TransactionProducerError(
                f"commit event {index} has a non-atomic root-pointer transition"
            )
        if count == 0 and evidence["published_sha256"] != evidence["canonical_before_sha256"]:
            raise TransactionProducerError(
                f"zero-commit event {index} changed canonical published state"
            )
    elif unit == "fault_injection":
        if evidence["injected"] is not True or evidence["failure_observed"] is not True:
            raise TransactionProducerError(f"fault event {index} did not observe its injection")
        for before, after in FAULT_PAIRS:
            if evidence[before] != evidence[after]:
                raise TransactionProducerError(
                    f"fault event {index} changed canonical state at {after}"
                )
    else:  # pragma: no cover - guarded by exact scope comparison
        raise TransactionProducerError(f"transaction event {index} has an unknown unit")


def _state_sha256(state: object) -> str:
    return live_validation.state_sha256(state)


def _live_success_events(ledger: object, capability: object) -> list[dict[str, Any]]:
    """Derive isolation/commit witnesses from authenticated live receipts.

    This deliberately accepts no caller-authored success event.  The live ledger
    already binds the complete canonical/candidate/serial state, prepublication
    probes, publication intent, cleanup and exact commit count.  Converting that
    evidence here keeps the assurance campaign and production invariant on one
    evidence chain.
    """

    try:
        verified_capability = live_validation.verify_capability(capability)
        verified_ledger = live_validation.verify_ledger(ledger)
    except live_validation.LiveCommitValidationError as error:
        raise TransactionProducerError(f"live commit evidence is invalid: {error}") from error
    if verified_ledger["capability_sha256"] != verified_capability["capability_sha256"]:
        raise TransactionProducerError("live ledger is bound to a different capability")
    receipts = verified_ledger["receipts"]
    observed_counts = [receipt["commit_count"] for receipt in receipts]
    if observed_counts != list(live_validation.COMMIT_COUNTS):
        raise TransactionProducerError(
            "live ledger must contain exactly one ordered receipt for commit counts 0 through 8"
        )

    events: list[dict[str, Any]] = []
    for receipt in receipts:
        count = receipt["commit_count"]
        if (
            receipt["mode"] != "validated_candidate"
            or receipt["comparison_equal"] is not True
            or receipt["candidate_state"] is None
            or receipt["candidate_enabled_after"] is not True
            or receipt["publication_source"]
            != ("unchanged" if count == 0 else "candidate")
        ):
            raise TransactionProducerError(
                f"live receipt for commit count {count} is not an equal validated candidate"
            )
        before = receipt["canonical_before"]
        candidate = receipt["candidate_state"]
        serial = receipt["serial_state"]
        published = receipt["canonical_after"]
        before_sha256 = _state_sha256(before)
        candidate_sha256 = _state_sha256(candidate)
        serial_sha256 = _state_sha256(serial)
        published_sha256 = _state_sha256(published)
        if receipt["canonical_probe_after_candidate_sha256"] != before_sha256 or receipt[
            "canonical_probe_after_serial_sha256"
        ] != before_sha256:
            raise TransactionProducerError(
                f"live receipt for commit count {count} does not prove provisional isolation"
            )

        events.append(
            {
                "unit": "provisional_isolation",
                "phase": instrumentation.UNIT_PHASES["provisional_isolation"],
                "evidence": {
                    "commit_count": count,
                    "canonical_before_sha256": before_sha256,
                    "canonical_after_provisional_sha256": before_sha256,
                    "provisional_state_sha256": candidate_sha256,
                    "allocation_topology_before_sha256": before[
                        "allocation_topology_sha256"
                    ],
                    "allocation_topology_after_sha256": before[
                        "allocation_topology_sha256"
                    ],
                    "ownership_before_sha256": before["ownership_sha256"],
                    "ownership_after_sha256": before["ownership_sha256"],
                    "pins_before_sha256": before["pins_sha256"],
                    "pins_after_sha256": before["pins_sha256"],
                    "refcounts_before_sha256": before["refcounts_sha256"],
                    "refcounts_after_sha256": before["refcounts_sha256"],
                    "target_kv_before_sha256_by_layer": before[
                        "physical_target_kv_sha256_by_layer"
                    ],
                    "target_kv_after_provisional_sha256_by_layer": before[
                        "physical_target_kv_sha256_by_layer"
                    ],
                    "draft_kv_before_sha256_by_layer": before[
                        "physical_draft_kv_sha256_by_layer"
                    ],
                    "draft_kv_after_provisional_sha256_by_layer": before[
                        "physical_draft_kv_sha256_by_layer"
                    ],
                    "gdn_state_before_sha256_by_layer": before[
                        "gdn_state_sha256_by_layer"
                    ],
                    "gdn_state_after_provisional_sha256_by_layer": before[
                        "gdn_state_sha256_by_layer"
                    ],
                    "convolution_state_before_sha256_by_layer": before[
                        "convolution_state_sha256_by_layer"
                    ],
                    "convolution_state_after_provisional_sha256_by_layer": before[
                        "convolution_state_sha256_by_layer"
                    ],
                    "device_error_word": before["device_error_word"],
                },
            }
        )
        events.append(
            {
                "unit": "atomic_commit",
                "phase": instrumentation.UNIT_PHASES["atomic_commit"],
                "evidence": {
                    "commit_count": count,
                    "canonical_before_sha256": before_sha256,
                    "provisional_sha256": candidate_sha256,
                    "serial_reference_sha256": serial_sha256,
                    "published_sha256": published_sha256,
                    "root_pointer_before_sha256": before["canonical_root_sha256"],
                    "root_pointer_after_sha256": published["canonical_root_sha256"],
                    "commit_epoch_before": before["commit_epoch"],
                    "commit_epoch_after": published["commit_epoch"],
                    "journal_sha256": receipt["publication_intent_sha256"],
                    "allocator_generation": published["allocator_generation"],
                    "serial_allocation_topology_sha256": serial[
                        "allocation_topology_sha256"
                    ],
                    "published_allocation_topology_sha256": published[
                        "allocation_topology_sha256"
                    ],
                    "serial_ownership_sha256": serial["ownership_sha256"],
                    "published_ownership_sha256": published["ownership_sha256"],
                    "serial_pins_sha256": serial["pins_sha256"],
                    "published_pins_sha256": published["pins_sha256"],
                    "serial_refcounts_sha256": serial["refcounts_sha256"],
                    "published_refcounts_sha256": published["refcounts_sha256"],
                    "serial_target_kv_sha256_by_layer": serial[
                        "physical_target_kv_sha256_by_layer"
                    ],
                    "published_target_kv_sha256_by_layer": published[
                        "physical_target_kv_sha256_by_layer"
                    ],
                    "serial_draft_kv_sha256_by_layer": serial[
                        "physical_draft_kv_sha256_by_layer"
                    ],
                    "published_draft_kv_sha256_by_layer": published[
                        "physical_draft_kv_sha256_by_layer"
                    ],
                    "serial_gdn_state_sha256_by_layer": serial[
                        "gdn_state_sha256_by_layer"
                    ],
                    "published_gdn_state_sha256_by_layer": published[
                        "gdn_state_sha256_by_layer"
                    ],
                    "serial_convolution_state_sha256_by_layer": serial[
                        "convolution_state_sha256_by_layer"
                    ],
                    "published_convolution_state_sha256_by_layer": published[
                        "convolution_state_sha256_by_layer"
                    ],
                    "device_error_word": published["device_error_word"],
                },
            }
        )
    return events


def build_entries_from_live_ledger(
    fault_capture: object,
    *,
    live_ledger: object,
    capability: object,
    header: dict[str, Any],
) -> list[dict[str, Any]]:
    """Combine receipt-derived success evidence with injected-fault evidence."""

    value = _exact_mapping(fault_capture, CAPTURE_KEYS, "transaction fault capture")
    if value["schema"] != FAULT_CAPTURE_SCHEMA:
        raise TransactionProducerError("transaction fault capture schema differs")
    if value["capture_sha256"] != _capture_hash(value):
        raise TransactionProducerError("transaction fault capture self-hash differs")
    if (
        value["semantic_source_sha256"] != header["semantic_source_sha256"]
        or value["arm"] != header["arm"]
    ):
        raise TransactionProducerError("transaction fault capture identity differs from campaign")
    if not isinstance(value["events"], list) or any(
        not isinstance(event, Mapping) or event.get("unit") != "fault_injection"
        for event in value["events"]
    ):
        raise TransactionProducerError(
            "transaction fault capture may contain only fault-injection events"
        )
    combined = {
        "schema": CAPTURE_SCHEMA,
        "semantic_source_sha256": value["semantic_source_sha256"],
        "arm": value["arm"],
        "runtime_configuration_sha256": value["runtime_configuration_sha256"],
        "events": [
            *_live_success_events(live_ledger, capability),
            *value["events"],
        ],
    }
    combined["capture_sha256"] = _capture_hash(combined)
    return build_entries(combined, header=header)


def build_entries(capture: object, *, header: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate every transaction scope before a fragment directory is created."""

    value = _exact_mapping(capture, CAPTURE_KEYS, "transaction capture")
    if value["schema"] != CAPTURE_SCHEMA:
        raise TransactionProducerError("transaction capture schema differs")
    try:
        provider._require_digest(value["capture_sha256"], "transaction capture")
        provider._require_digest(
            value["runtime_configuration_sha256"], "transaction runtime configuration"
        )
    except provider.ProviderProducerError as error:
        raise TransactionProducerError(str(error)) from error
    if value["capture_sha256"] != _capture_hash(value):
        raise TransactionProducerError("transaction capture self-hash differs")
    if (
        value["semantic_source_sha256"] != header["semantic_source_sha256"]
        or value["arm"] != header["arm"]
    ):
        raise TransactionProducerError("transaction capture identity differs from campaign")
    raw_entries = value["events"]
    if not isinstance(raw_entries, list):
        raise TransactionProducerError("transaction capture events must be an array")
    expected_scopes = set(
        instrumentation.partition_probe_scopes_by_producer(header)[PRODUCER_KIND]
    )
    entries: list[dict[str, Any]] = []
    observed: set[tuple[Any, ...]] = set()
    for index, raw in enumerate(raw_entries):
        entry = _exact_mapping(raw, ENTRY_KEYS, f"transaction event {index}")
        if entry["unit"] not in instrumentation.PRODUCER_UNITS[PRODUCER_KIND]:
            raise TransactionProducerError(f"transaction event {index} has an invalid unit")
        evidence = entry["evidence"]
        if not isinstance(evidence, dict):
            raise TransactionProducerError(f"transaction event {index} evidence is invalid")
        scope = (
            entry["unit"],
            None,
            None,
            None,
            entry["phase"],
            evidence.get("commit_count"),
        )
        if scope not in expected_scopes or scope in observed:
            raise TransactionProducerError(
                f"transaction event {index} scope is unexpected or duplicated: {scope}"
            )
        event = instrumentation.seal_event(
            {
                "schema": instrumentation.TRACE_EVENT_SCHEMA,
                "sequence": index,
                "arm": header["arm"],
                "unit": entry["unit"],
                "phase": entry["phase"],
                "position": None,
                "layer_index": None,
                "row": None,
                "evidence": evidence,
                "previous_event_sha256": None,
            }
        )
        normalized = {
            "unit": event["unit"],
            "phase": event["phase"],
            "evidence": event["evidence"],
        }
        _validate_semantics(normalized, index)
        entries.append(normalized)
        observed.add(scope)
    if observed != expected_scopes:
        missing = sorted(expected_scopes - observed, key=repr)
        raise TransactionProducerError(
            f"transaction capture is missing {len(missing)} scopes: {missing[:3]}"
        )
    return entries


def _fragment_context() -> tuple[dict[str, Any], str, tuple[tuple[Any, ...], ...]]:
    if os.environ.get(instrumentation.FULL_ASSURANCE_ENABLE_ENV) != "1":
        raise TransactionProducerError("transaction fragment was not explicitly enabled")
    missing = [
        name for name in instrumentation.FULL_ASSURANCE_ENVIRONMENT if not os.environ.get(name)
    ]
    if missing:
        raise TransactionProducerError(
            f"transaction fragment environment is incomplete: {missing}"
        )
    if os.environ[instrumentation.FULL_ASSURANCE_PRODUCER_KIND_ENV] != PRODUCER_KIND:
        raise TransactionProducerError("transaction producer kind differs")
    source_path = Path(__file__).resolve(strict=True)
    source = provider._read_stable_owned_file(source_path, "transaction producer source")
    source_sha256 = hashlib.sha256(source).hexdigest()
    if source_sha256 != os.environ[instrumentation.FULL_ASSURANCE_PRODUCER_SHA_ENV]:
        raise TransactionProducerError("transaction producer source binding differs")
    header = instrumentation.normalize_header(
        provider._load_json(
            Path(os.environ[instrumentation.FULL_ASSURANCE_HEADER_ENV]),
            "transaction campaign header",
        )
    )
    scopes = instrumentation.partition_probe_scopes_by_producer(header)[PRODUCER_KIND]
    scope_document = provider._load_json(
        Path(os.environ[instrumentation.FULL_ASSURANCE_SCOPES_ENV]),
        "transaction runtime scopes",
    )
    if scope_document != instrumentation.runtime_scopes_document(header, scopes):
        raise TransactionProducerError("transaction runtime scope identity differs")
    return header, source_sha256, scopes


def _publish_entries(
    entries: list[dict[str, Any]],
    *,
    header: dict[str, Any],
    source_sha256: str,
    scopes: tuple[tuple[Any, ...], ...],
) -> dict[str, Any]:
    writer = instrumentation.FullAssuranceFragmentWriter(
        Path(os.environ[instrumentation.FULL_ASSURANCE_ROOT_ENV]),
        header,
        fragment_id=os.environ[instrumentation.FULL_ASSURANCE_FRAGMENT_ID_ENV],
        scopes=scopes,
        producer_kind=PRODUCER_KIND,
        producer_sha256=source_sha256,
        sync_interval=256,
    )
    try:
        for entry in entries:
            writer.append(
                unit=entry["unit"],
                phase=entry["phase"],
                position=None,
                layer_index=None,
                row=None,
                evidence=entry["evidence"],
            )
        return writer.finalize(
            device_synchronize_count=0,
            installed_instrumentation_sites=instrumentation.expected_instrumentation_sites(
                PRODUCER_KIND, scopes, arm=header["arm"]
            ),
        )
    except BaseException:
        writer.abort()
        raise


def produce(capture_path: Path, capture_file_sha256: str) -> dict[str, Any]:
    header, source_sha256, scopes = _fragment_context()
    capture = provider._load_json(
        capture_path,
        "transaction capture",
        expected_sha256=provider._require_digest(
            capture_file_sha256, "transaction capture file"
        ),
    )
    entries = build_entries(capture, header=header)
    return _publish_entries(
        entries,
        header=header,
        source_sha256=source_sha256,
        scopes=scopes,
    )


def produce_from_live_ledger(
    *,
    ledger_path: Path,
    ledger_file_sha256: str,
    capability_path: Path,
    capability_file_sha256: str,
    fault_capture_path: Path,
    fault_capture_file_sha256: str,
) -> dict[str, Any]:
    header, source_sha256, scopes = _fragment_context()
    ledger = provider._load_json(
        ledger_path,
        "live commit ledger",
        expected_sha256=provider._require_digest(
            ledger_file_sha256, "live commit ledger file"
        ),
    )
    capability = provider._load_json(
        capability_path,
        "live commit capability",
        expected_sha256=provider._require_digest(
            capability_file_sha256, "live commit capability file"
        ),
    )
    fault_capture = provider._load_json(
        fault_capture_path,
        "transaction fault capture",
        expected_sha256=provider._require_digest(
            fault_capture_file_sha256, "transaction fault capture file"
        ),
    )
    entries = build_entries_from_live_ledger(
        fault_capture,
        live_ledger=ledger,
        capability=capability,
        header=header,
    )
    return _publish_entries(
        entries,
        header=header,
        source_sha256=source_sha256,
        scopes=scopes,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qwen-assurance-transaction-producer",
        description="Verify complete private-provisional, atomic-commit and fault evidence.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--capture", type=Path)
    source.add_argument("--live-ledger", type=Path)
    parser.add_argument("--capture-file-sha256")
    parser.add_argument("--live-ledger-file-sha256")
    parser.add_argument("--capability", type=Path)
    parser.add_argument("--capability-file-sha256")
    parser.add_argument("--fault-capture", type=Path)
    parser.add_argument("--fault-capture-file-sha256")
    args = parser.parse_args(argv)
    try:
        if args.capture is not None:
            if args.capture_file_sha256 is None:
                raise TransactionProducerError(
                    "--capture requires --capture-file-sha256"
                )
            if any(
                value is not None
                for value in (
                    args.live_ledger_file_sha256,
                    args.capability,
                    args.capability_file_sha256,
                    args.fault_capture,
                    args.fault_capture_file_sha256,
                )
            ):
                raise TransactionProducerError(
                    "legacy capture mode cannot be mixed with live-ledger inputs"
                )
            result = produce(args.capture, args.capture_file_sha256)
        else:
            required = {
                "--live-ledger-file-sha256": args.live_ledger_file_sha256,
                "--capability": args.capability,
                "--capability-file-sha256": args.capability_file_sha256,
                "--fault-capture": args.fault_capture,
                "--fault-capture-file-sha256": args.fault_capture_file_sha256,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise TransactionProducerError(
                    "live-ledger mode is missing " + ", ".join(missing)
                )
            if args.capture_file_sha256 is not None:
                raise TransactionProducerError(
                    "live-ledger mode cannot use --capture-file-sha256"
                )
            result = produce_from_live_ledger(
                ledger_path=args.live_ledger,
                ledger_file_sha256=args.live_ledger_file_sha256,
                capability_path=args.capability,
                capability_file_sha256=args.capability_file_sha256,
                fault_capture_path=args.fault_capture,
                fault_capture_file_sha256=args.fault_capture_file_sha256,
            )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        instrumentation.InstrumentationError,
        provider.ProviderProducerError,
        TransactionProducerError,
    ) as error:
        print(f"qwen-assurance-transaction-producer: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
