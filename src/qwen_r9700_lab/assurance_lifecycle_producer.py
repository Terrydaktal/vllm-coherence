"""Authenticate complete state across every declared runtime lifecycle scenario."""

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
from qwen_r9700_lab import assurance_snapshot_producer as snapshot

CAPTURE_SCHEMA = "urn:qwen-r9700:full-lifecycle-capture:v1"
PRODUCER_KIND = "lifecycle_controller"
CAPTURE_KEYS = {
    "schema",
    "capture_sha256",
    "semantic_source_sha256",
    "arm",
    "runtime_configuration_sha256",
    "events",
}
ENTRY_KEYS = {
    "scenario",
    "expected_outcome",
    "observed_outcome",
    "prompt_tokens",
    "cold_fill_occurred",
    "cross_session_contamination",
    "device_error_word",
    "before",
    "after",
    "reference",
}


class LifecycleProducerError(RuntimeError):
    """Lifecycle evidence is incomplete, unauthenticated, or state-divergent."""


def _capture_hash(capture: Mapping[str, Any]) -> str:
    unsigned = dict(capture)
    unsigned.pop("capture_sha256", None)
    return provider._sha(unsigned)


def _exact_mapping(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise LifecycleProducerError(f"{label} keys differ: {observed}")
    return dict(value)


def _state_digest(state: dict[str, Any]) -> str:
    return provider._sha(state)


def _validate_entry(value: object, index: int) -> dict[str, Any]:
    entry = _exact_mapping(value, ENTRY_KEYS, f"lifecycle event {index}")
    scenario = entry["scenario"]
    if scenario not in instrumentation.LIFECYCLE_SCENARIOS:
        raise LifecycleProducerError(f"lifecycle event {index} scenario is invalid")
    for key in ("expected_outcome", "observed_outcome"):
        if not isinstance(entry[key], str) or not entry[key]:
            raise LifecycleProducerError(f"lifecycle event {index} {key} is invalid")
    if entry["observed_outcome"] != entry["expected_outcome"]:
        raise LifecycleProducerError(f"lifecycle event {index} outcome differs")
    prompt_tokens = entry["prompt_tokens"]
    if (
        isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
        or prompt_tokens not in {60_298, 249_957}
    ):
        raise LifecycleProducerError(f"lifecycle event {index} context is not qualified")
    for key in ("cold_fill_occurred", "cross_session_contamination"):
        if not isinstance(entry[key], bool):
            raise LifecycleProducerError(f"lifecycle event {index} {key} is not boolean")
    if entry["cross_session_contamination"]:
        raise LifecycleProducerError(f"lifecycle event {index} crossed session state")
    if scenario.startswith(("restored_", "restart_", "offload_", "rapid_switch_")) and entry[
        "cold_fill_occurred"
    ]:
        raise LifecycleProducerError(f"lifecycle event {index} used a cold fill")
    if entry["device_error_word"] != 0:
        raise LifecycleProducerError(f"lifecycle event {index} reports a device error")
    try:
        before = snapshot._normalize_state(entry["before"], f"lifecycle event {index} before")
        after = snapshot._normalize_state(entry["after"], f"lifecycle event {index} after")
        reference = snapshot._normalize_state(
            entry["reference"], f"lifecycle event {index} reference"
        )
    except snapshot.SnapshotProducerError as error:
        raise LifecycleProducerError(str(error)) from error
    if after != reference:
        differing = next(key for key in sorted(snapshot.STATE_KEYS) if after[key] != reference[key])
        raise LifecycleProducerError(
            f"lifecycle event {index} differs from serial reference at {differing}"
        )
    if scenario in {"missing_snapshot", "corrupt_snapshot"} and after != before:
        differing = next(key for key in sorted(snapshot.STATE_KEYS) if after[key] != before[key])
        raise LifecycleProducerError(
            f"lifecycle event {index} changed state after rejected snapshot at {differing}"
        )
    return {
        "scenario": scenario,
        "expected_outcome": entry["expected_outcome"],
        "observed_outcome": entry["observed_outcome"],
        "state_before_sha256": _state_digest(before),
        "state_after_sha256": _state_digest(after),
        "reference_state_sha256": _state_digest(reference),
        "prompt_tokens": prompt_tokens,
        "cold_fill_occurred": entry["cold_fill_occurred"],
        "cross_session_contamination": entry["cross_session_contamination"],
        "device_error_word": entry["device_error_word"],
        "complete_before": before,
        "complete_after": after,
        "complete_reference": reference,
    }


def build_entries(capture: object, *, header: dict[str, Any]) -> list[dict[str, Any]]:
    value = _exact_mapping(capture, CAPTURE_KEYS, "lifecycle capture")
    if value["schema"] != CAPTURE_SCHEMA:
        raise LifecycleProducerError("lifecycle capture schema differs")
    try:
        provider._require_digest(value["capture_sha256"], "lifecycle capture")
        provider._require_digest(
            value["runtime_configuration_sha256"], "lifecycle runtime configuration"
        )
    except provider.ProviderProducerError as error:
        raise LifecycleProducerError(str(error)) from error
    if value["capture_sha256"] != _capture_hash(value):
        raise LifecycleProducerError("lifecycle capture self-hash differs")
    if (
        value["semantic_source_sha256"] != header["semantic_source_sha256"]
        or value["arm"] != header["arm"]
    ):
        raise LifecycleProducerError("lifecycle capture identity differs from campaign")
    raw_events = value["events"]
    if not isinstance(raw_events, list):
        raise LifecycleProducerError("lifecycle events must be an array")
    entries: list[dict[str, Any]] = []
    observed: set[str] = set()
    for index, raw in enumerate(raw_events):
        evidence = _validate_entry(raw, index)
        scenario = evidence["scenario"]
        if scenario in observed:
            raise LifecycleProducerError(f"lifecycle scenario is duplicated: {scenario}")
        instrumentation.seal_event(
            {
                "schema": instrumentation.TRACE_EVENT_SCHEMA,
                "sequence": index,
                "arm": header["arm"],
                "unit": "lifecycle",
                "phase": scenario,
                "position": None,
                "layer_index": None,
                "row": None,
                "evidence": evidence,
                "previous_event_sha256": None,
            }
        )
        entries.append(evidence)
        observed.add(scenario)
    expected = set(instrumentation.LIFECYCLE_SCENARIOS)
    if observed != expected:
        raise LifecycleProducerError(
            f"lifecycle capture is missing scenarios: {sorted(expected - observed)}"
        )
    return entries


def _fragment_context() -> tuple[dict[str, Any], str, tuple[tuple[Any, ...], ...]]:
    if os.environ.get(instrumentation.FULL_ASSURANCE_ENABLE_ENV) != "1":
        raise LifecycleProducerError("lifecycle fragment was not explicitly enabled")
    missing = [
        name for name in instrumentation.FULL_ASSURANCE_ENVIRONMENT if not os.environ.get(name)
    ]
    if missing:
        raise LifecycleProducerError(f"lifecycle fragment environment is incomplete: {missing}")
    if os.environ[instrumentation.FULL_ASSURANCE_PRODUCER_KIND_ENV] != PRODUCER_KIND:
        raise LifecycleProducerError("lifecycle producer kind differs")
    source_path = Path(__file__).resolve(strict=True)
    source = provider._read_stable_owned_file(source_path, "lifecycle producer source")
    source_sha256 = hashlib.sha256(source).hexdigest()
    if source_sha256 != os.environ[instrumentation.FULL_ASSURANCE_PRODUCER_SHA_ENV]:
        raise LifecycleProducerError("lifecycle producer source binding differs")
    header = instrumentation.normalize_header(
        provider._load_json(
            Path(os.environ[instrumentation.FULL_ASSURANCE_HEADER_ENV]),
            "lifecycle campaign header",
        )
    )
    scopes = instrumentation.partition_probe_scopes_by_producer(header)[PRODUCER_KIND]
    scope_document = provider._load_json(
        Path(os.environ[instrumentation.FULL_ASSURANCE_SCOPES_ENV]),
        "lifecycle runtime scopes",
    )
    if scope_document != instrumentation.runtime_scopes_document(header, scopes):
        raise LifecycleProducerError("lifecycle runtime scope identity differs")
    return header, source_sha256, scopes


def produce(capture_path: Path, capture_file_sha256: str) -> dict[str, Any]:
    header, source_sha256, scopes = _fragment_context()
    capture = provider._load_json(
        capture_path,
        "lifecycle capture",
        expected_sha256=provider._require_digest(
            capture_file_sha256, "lifecycle capture file"
        ),
    )
    entries = build_entries(capture, header=header)
    writer = instrumentation.FullAssuranceFragmentWriter(
        Path(os.environ[instrumentation.FULL_ASSURANCE_ROOT_ENV]),
        header,
        fragment_id=os.environ[instrumentation.FULL_ASSURANCE_FRAGMENT_ID_ENV],
        scopes=scopes,
        producer_kind=PRODUCER_KIND,
        producer_sha256=source_sha256,
        sync_interval=16,
    )
    try:
        for evidence in entries:
            writer.append(
                unit="lifecycle",
                phase=evidence["scenario"],
                position=None,
                layer_index=None,
                row=None,
                evidence=evidence,
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qwen-assurance-lifecycle-producer",
        description="Verify complete state across all fixed-slot lifecycle scenarios.",
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
        LifecycleProducerError,
    ) as error:
        print(f"qwen-assurance-lifecycle-producer: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
