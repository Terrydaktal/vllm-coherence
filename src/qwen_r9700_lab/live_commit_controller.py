"""Runtime controller for independent serial validation before canonical commit.

This module is the control-plane half of the arbitrary-input guarantee.  It does
not know how to execute Qwen.  A release-bound driver must provide independent
candidate and serial private states, complete normalized state observations, and
an atomic publication primitive.  The controller orders those operations,
detects pre-publication mutation, preserves a counterexample before fallback,
quarantines a failed candidate, and emits a verifier-compatible receipt.

The driver boundary is intentionally narrow.  A future GPU adapter cannot call
the candidate implementation and then label the result "serial": its runtime
hash and implementation family are authenticated against the capability before
the first transition.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from qwen_r9700_lab import live_commit_validation as validation

COUNTEREXAMPLE_SCHEMA = "urn:qwen-r9700:live-serial-counterexample:v1"
PUBLICATION_INTENT_SCHEMA = "urn:qwen-r9700:live-serial-publication-intent:v2"
PUBLICATION_RECOVERY_SCHEMA = "urn:qwen-r9700:live-serial-publication-recovery:v1"
RECEIPT_DYNAMIC_KEYS = {
    "publication_intent_sha256",
    "canonical_after",
    "publication_atomic",
    "candidate_private_state_destroyed",
    "serial_private_state_destroyed",
    "receipt_sha256",
}
RECEIPT_TEMPLATE_KEYS = validation.RECEIPT_KEYS - RECEIPT_DYNAMIC_KEYS
PUBLICATION_INTENT_KEYS = {
    "schema",
    "capability_sha256",
    "session_id",
    "request_id",
    "round_index",
    "commit_count",
    "mode",
    "snapshot_handle_id",
    "canonical_before",
    "selected_source",
    "selected_handle_id",
    "selected_state",
    "selected_token_ids",
    "candidate_handle_id",
    "serial_handle_id",
    "candidate_failure",
    "counterexample_sha256",
    "receipt_template",
}
PUBLICATION_RECOVERY_KEYS = {
    "schema",
    "publication_intent_sha256",
    "capability_sha256",
    "session_id",
    "request_id",
    "round_index",
    "disposition",
    "canonical_state",
    "publication_atomic",
    "candidate_private_state_destroyed",
    "serial_private_state_destroyed",
    "counterexample_sha256",
}


class LiveCommitControllerError(RuntimeError):
    """A transition could not be proven safe for canonical publication."""


@dataclass(frozen=True)
class CanonicalSnapshot:
    """One immutable canonical root and its complete semantic observation."""

    handle_id: str
    state: dict[str, Any]


@dataclass(frozen=True)
class PrivateTransition:
    """A complete transition held outside canonical storage."""

    handle_id: str
    implementation_family: str
    token_ids: tuple[int, ...]
    state: dict[str, Any]
    model_output: object | None = None


@dataclass(frozen=True)
class Publication:
    """The canonical state observed after one root publication."""

    state: dict[str, Any]
    atomic: bool


@dataclass(frozen=True)
class PublicationRecovery:
    """Idempotent driver verdict for an interrupted root publication."""

    state: dict[str, Any]
    atomic: bool
    disposition: str
    candidate_private_state_destroyed: bool
    serial_private_state_destroyed: bool


@runtime_checkable
class LiveTransitionDriver(Protocol):
    """Release-specific state execution and publication operations."""

    release_runtime_sha256: str
    serial_runtime_sha256: str
    candidate_implementation_family: str
    serial_implementation_family: str

    def snapshot_canonical(self) -> CanonicalSnapshot: ...

    def probe_canonical(self) -> dict[str, Any]: ...

    def execute_candidate(
        self,
        snapshot: CanonicalSnapshot,
        request_payload: object,
        commit_count: int | None,
    ) -> PrivateTransition: ...

    def execute_serial(
        self,
        snapshot: CanonicalSnapshot,
        request_payload: object,
        commit_count: int,
    ) -> PrivateTransition: ...

    def publish_private(
        self,
        snapshot: CanonicalSnapshot,
        transition: PrivateTransition,
        publication_intent_sha256: str,
    ) -> Publication: ...

    def restore_canonical(self, snapshot: CanonicalSnapshot) -> Publication: ...

    def destroy_private(self, transition: PrivateTransition) -> None: ...

    def cleanup_failed_private(
        self,
        role: str,
        session_id: str,
        request_id: str,
        round_index: int,
    ) -> None: ...

    def recover_publication(self, intent: dict[str, Any]) -> PublicationRecovery: ...

def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _failure(stage: str, error: BaseException) -> dict[str, str]:
    return {
        "stage": stage,
        "exception_type": type(error).__name__,
        "message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
        "traceback_sha256": hashlib.sha256(traceback.format_exc().encode()).hexdigest(),
    }


def _synthetic_failure(stage: str, message: str) -> dict[str, str]:
    return {
        "stage": stage,
        "exception_type": "InvariantViolation",
        "message_sha256": hashlib.sha256(message.encode()).hexdigest(),
        "traceback_sha256": hashlib.sha256(b"").hexdigest(),
    }


def _handle_id(value: object, label: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    return validation._identifier(value, label)


def _normalize_intent(value: object) -> dict[str, Any]:
    intent = validation._exact_dict(value, PUBLICATION_INTENT_KEYS, "publication intent")
    if intent["schema"] != PUBLICATION_INTENT_SCHEMA:
        raise LiveCommitControllerError("publication intent schema differs")
    intent["capability_sha256"] = validation._sha256(
        intent["capability_sha256"], "publication intent capability"
    )
    intent["session_id"] = validation._identifier(
        intent["session_id"], "publication intent session ID"
    )
    intent["request_id"] = validation._identifier(
        intent["request_id"], "publication intent request ID"
    )
    if (
        isinstance(intent["round_index"], bool)
        or not isinstance(intent["round_index"], int)
        or intent["round_index"] < 0
    ):
        raise LiveCommitControllerError("publication intent round index is invalid")
    if intent["commit_count"] not in validation.COMMIT_COUNTS:
        raise LiveCommitControllerError("publication intent commit count is invalid")
    intent["snapshot_handle_id"] = _handle_id(
        intent["snapshot_handle_id"], "publication intent snapshot handle"
    )
    intent["selected_handle_id"] = _handle_id(
        intent["selected_handle_id"], "publication intent selected handle"
    )
    intent["candidate_handle_id"] = _handle_id(
        intent["candidate_handle_id"],
        "publication intent candidate handle",
        nullable=True,
    )
    intent["serial_handle_id"] = _handle_id(
        intent["serial_handle_id"], "publication intent serial handle"
    )
    intent["canonical_before"] = validation.normalize_state(
        intent["canonical_before"], "publication intent canonical before"
    )
    intent["selected_state"] = validation.normalize_state(
        intent["selected_state"], "publication intent selected state"
    )
    selected_tokens = intent["selected_token_ids"]
    if (
        not isinstance(selected_tokens, list)
        or len(selected_tokens) != intent["commit_count"]
        or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in selected_tokens
        )
    ):
        raise LiveCommitControllerError("publication intent selected tokens are invalid")
    if intent["counterexample_sha256"] is not None:
        intent["counterexample_sha256"] = validation._sha256(
            intent["counterexample_sha256"], "publication intent counterexample"
        )
    template = validation._exact_dict(
        intent["receipt_template"], RECEIPT_TEMPLATE_KEYS, "publication receipt template"
    )
    expected_after = (
        intent["canonical_before"]
        if intent["commit_count"] == 0
        else intent["selected_state"]
    )
    try:
        sealed = validation.seal_receipt(
            {
                **template,
                "publication_intent_sha256": "0" * 64,
                "canonical_after": expected_after,
                "publication_atomic": True,
                "candidate_private_state_destroyed": True,
                "serial_private_state_destroyed": True,
            }
        )
    except validation.LiveCommitValidationError as error:
        raise LiveCommitControllerError("publication receipt template is invalid") from error
    normalized_template = {
        key: item for key, item in sealed.items() if key in RECEIPT_TEMPLATE_KEYS
    }
    intent["receipt_template"] = normalized_template
    if (
        normalized_template["capability_sha256"] != intent["capability_sha256"]
        or normalized_template["session_id"] != intent["session_id"]
        or normalized_template["request_id"] != intent["request_id"]
        or normalized_template["round_index"] != intent["round_index"]
        or normalized_template["commit_count"] != intent["commit_count"]
        or normalized_template["mode"] != intent["mode"]
        or normalized_template["canonical_before"] != intent["canonical_before"]
        or normalized_template["publication_source"] != intent["selected_source"]
        or normalized_template["counterexample_sha256"]
        != intent["counterexample_sha256"]
    ):
        raise LiveCommitControllerError("publication intent and receipt template differ")
    if normalized_template["serial_token_ids"] != selected_tokens and intent[
        "selected_source"
    ] in {"serial", "unchanged"}:
        raise LiveCommitControllerError("publication intent selected serial tokens differ")
    if (
        intent["selected_source"] == "candidate"
        and normalized_template["candidate_token_ids"] != selected_tokens
    ):
        raise LiveCommitControllerError("publication intent selected candidate tokens differ")
    if intent["candidate_handle_id"] is None and normalized_template["candidate_state"] is not None:
        raise LiveCommitControllerError("publication intent omitted a completed candidate handle")
    return intent


class DurableEvidenceJournal:
    """Owner-only, create-only evidence publication with directory fsync."""

    def __init__(self, root: Path) -> None:
        self.root = root.absolute()
        if not self.root.exists():
            self.root.mkdir(mode=0o700, parents=True)
        status = self.root.lstat()
        if (
            not stat.S_ISDIR(status.st_mode)
            or self.root.is_symlink()
            or status.st_uid != os.getuid()
            or stat.S_IMODE(status.st_mode) & 0o077
        ):
            raise LiveCommitControllerError("evidence root must be an owner-only directory")

    def _publish(self, relative: str, payload: bytes) -> Path:
        path = self.root / relative
        if path.parent != self.root or "/" in relative or relative in {"", ".", ".."}:
            raise LiveCommitControllerError("evidence filename is unsafe")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError:
            existing = path.read_bytes()
            if existing != payload:
                raise LiveCommitControllerError(
                    f"existing immutable evidence differs: {path.name}"
                ) from None
            return path
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short evidence write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return path

    def preserve_counterexample(self, value: dict[str, Any]) -> str:
        payload = _canonical(value)
        digest = hashlib.sha256(payload).hexdigest()
        self._publish(f"counterexample-{digest}.json", payload + b"\n")
        return digest

    def preserve_receipt(self, value: dict[str, Any]) -> Path:
        round_index = value["round_index"]
        digest = value["receipt_sha256"]
        return self._publish(
            f"receipt-{round_index:020d}-{digest}.json",
            _canonical(value) + b"\n",
        )

    def preserve_intent(self, value: dict[str, Any]) -> str:
        payload = _canonical(value)
        digest = hashlib.sha256(payload).hexdigest()
        self._publish(
            f"intent-{value['round_index']:020d}-{digest}.json",
            payload + b"\n",
        )
        return digest

    def _load_intent(self, path: Path, expected_sha256: str) -> dict[str, Any]:
        status = path.lstat()
        if (
            not stat.S_ISREG(status.st_mode)
            or path.is_symlink()
            or status.st_uid != os.getuid()
            or stat.S_IMODE(status.st_mode) & 0o077
        ):
            raise LiveCommitControllerError(
                f"publication intent is unsafe: {path.name}"
            )
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise LiveCommitControllerError(
                f"publication intent is unreadable: {path.name}"
            ) from error

        if (
            not isinstance(value, dict)
            or raw != _canonical(value) + b"\n"
            or hashlib.sha256(_canonical(value)).hexdigest() != expected_sha256
        ):
            raise LiveCommitControllerError(
                f"publication intent is invalid: {path.name}"
            )
        try:
            return _normalize_intent(value)
        except (LiveCommitControllerError, validation.LiveCommitValidationError) as error:
            raise LiveCommitControllerError(
                f"publication intent is invalid: {path.name}"
            ) from error

    def load_intent_by_sha256(self, expected_sha256: str) -> dict[str, Any]:
        validation._sha256(expected_sha256, "publication intent SHA-256")
        matches = list(self.root.glob(f"intent-*-{expected_sha256}.json"))
        if len(matches) != 1:
            raise LiveCommitControllerError(
                "publication recovery does not resolve exactly one intent"
            )
        return self._load_intent(matches[0], expected_sha256)

    def preserve_failure(self, value: dict[str, Any]) -> str:
        payload = _canonical(value)
        digest = hashlib.sha256(payload).hexdigest()
        self._publish(f"fatal-{digest}.json", payload + b"\n")
        return digest

    def load_receipts(self) -> list[dict[str, Any]]:
        receipts: list[tuple[int, dict[str, Any]]] = []
        pattern = re.compile(r"receipt-([0-9]{20})-([0-9a-f]{64})[.]json")
        for path in self.root.iterdir():
            match = pattern.fullmatch(path.name)
            if match is None:
                if path.name.startswith("receipt-"):
                    raise LiveCommitControllerError(
                        f"malformed receipt artifact exists: {path.name}"
                    )
                continue
            status = path.lstat()
            if (
                not stat.S_ISREG(status.st_mode)
                or path.is_symlink()
                or status.st_uid != os.getuid()
                or stat.S_IMODE(status.st_mode) & 0o077
            ):
                raise LiveCommitControllerError(f"receipt artifact is unsafe: {path.name}")
            try:
                receipt = validation.verify_receipt(json.loads(path.read_bytes()))
            except (
                UnicodeError,
                json.JSONDecodeError,
                validation.LiveCommitValidationError,
            ) as error:
                raise LiveCommitControllerError(
                    f"receipt artifact is invalid: {path.name}"
                ) from error
            if receipt["receipt_sha256"] != match.group(2):
                raise LiveCommitControllerError(f"receipt filename hash differs: {path.name}")
            intent = self.root / (
                f"intent-{receipt['round_index']:020d}-"
                f"{receipt['publication_intent_sha256']}.json"
            )
            if not intent.is_file() or intent.is_symlink():
                raise LiveCommitControllerError(
                    f"receipt publication intent is absent: {path.name}"
                )
            intent_value = self._load_intent(
                intent, receipt["publication_intent_sha256"]
            )
            if (
                intent_value["round_index"] != receipt["round_index"]
                or intent_value["capability_sha256"]
                != receipt["capability_sha256"]
                or intent_value["session_id"] != receipt["session_id"]
                or intent_value["request_id"] != receipt["request_id"]
                or intent_value["commit_count"] != receipt["commit_count"]
                or intent_value["selected_source"] != receipt["publication_source"]
                or intent_value["selected_state"] != receipt["canonical_after"]
            ):
                raise LiveCommitControllerError(
                    f"receipt and publication intent differ: {path.name}"
                )
            if any(
                receipt[key] != intent_value["receipt_template"][key]
                for key in RECEIPT_TEMPLATE_KEYS
            ):
                raise LiveCommitControllerError(
                    f"receipt and durable receipt template differ: {path.name}"
                )
            receipts.append((int(match.group(1)), receipt))
        receipts.sort(key=lambda item: item[0])
        if any(index != receipt["round_index"] for index, receipt in receipts):
            raise LiveCommitControllerError("receipt filename round differs from its payload")
        return [receipt for _, receipt in receipts]

    def preserve_recovery(self, value: dict[str, Any]) -> tuple[str, Path]:
        payload = _canonical(value)
        digest = hashlib.sha256(payload).hexdigest()
        path = self._publish(
            f"recovery-{value['publication_intent_sha256']}-{digest}.json",
            payload + b"\n",
        )
        return digest, path

    def load_recoveries(self) -> list[dict[str, Any]]:
        pattern = re.compile(r"recovery-([0-9a-f]{64})-([0-9a-f]{64})[.]json")
        recoveries: list[dict[str, Any]] = []
        for path in self.root.iterdir():
            match = pattern.fullmatch(path.name)
            if match is None:
                if path.name.startswith("recovery-"):
                    raise LiveCommitControllerError(
                        f"malformed publication recovery exists: {path.name}"
                    )
                continue
            status = path.lstat()
            if (
                not stat.S_ISREG(status.st_mode)
                or path.is_symlink()
                or status.st_uid != os.getuid()
                or stat.S_IMODE(status.st_mode) & 0o077
            ):
                raise LiveCommitControllerError(
                    f"publication recovery is unsafe: {path.name}"
                )
            try:
                raw = path.read_bytes()
                value = json.loads(raw)
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise LiveCommitControllerError(
                    f"publication recovery is unreadable: {path.name}"
                ) from error
            if (
                not isinstance(value, dict)
                or set(value) != PUBLICATION_RECOVERY_KEYS
                or value["schema"] != PUBLICATION_RECOVERY_SCHEMA
                or raw != _canonical(value) + b"\n"
                or hashlib.sha256(_canonical(value)).hexdigest() != match.group(2)
                or value["publication_intent_sha256"] != match.group(1)
            ):
                raise LiveCommitControllerError(
                    f"publication recovery is invalid: {path.name}"
                )
            if value["disposition"] not in {"committed", "rolled_back"}:
                raise LiveCommitControllerError(
                    f"publication recovery disposition is invalid: {path.name}"
                )
            value["canonical_state"] = validation.normalize_state(
                value["canonical_state"], "publication recovery canonical state"
            )
            for field in (
                "publication_atomic",
                "candidate_private_state_destroyed",
                "serial_private_state_destroyed",
            ):
                if value[field] is not True:
                    raise LiveCommitControllerError(
                        f"publication recovery is incomplete: {path.name}"
                    )
            if value["counterexample_sha256"] is not None:
                validation._sha256(
                    value["counterexample_sha256"], "publication recovery counterexample"
                )
            recoveries.append(value)
        intent_ids = [item["publication_intent_sha256"] for item in recoveries]
        if len(intent_ids) != len(set(intent_ids)):
            raise LiveCommitControllerError("publication intent has multiple recovery records")
        return recoveries

    def unresolved_intents(self, resolved_intents: set[str]) -> list[Path]:
        pattern = re.compile(r"intent-([0-9]{20})-([0-9a-f]{64})[.]json")
        pending: list[Path] = []
        for path in self.root.iterdir():
            match = pattern.fullmatch(path.name)
            if match is None:
                if path.name.startswith("intent-"):
                    raise LiveCommitControllerError(
                        f"malformed publication intent exists: {path.name}"
                    )
                continue
            if match.group(2) not in resolved_intents:
                self._load_intent(path, match.group(2))
                pending.append(path)
        return sorted(pending)

    def preserve_ledger(self, value: dict[str, Any]) -> Path:
        return self._publish(
            f"ledger-{value['ledger_sha256']}.json",
            _canonical(value) + b"\n",
        )


class LiveCommitController:
    """Execute candidate and authoritative serial transitions before publish."""

    def __init__(
        self,
        capability: object,
        driver: LiveTransitionDriver,
        journal: DurableEvidenceJournal,
        *,
        session_id: str,
    ) -> None:
        self.capability = validation.verify_capability(capability)
        if not isinstance(driver, LiveTransitionDriver):
            raise LiveCommitControllerError("live transition driver contract is incomplete")
        self.driver = driver
        self.journal = journal
        self.session_id = validation._identifier(session_id, "controller session ID")
        self._verify_controller_identity()
        self._verify_driver_identity()
        self.receipts = self.journal.load_receipts()
        self.recoveries = self.journal.load_recoveries()
        self.candidate_quarantined = False
        self.counterexample_sha256: str | None = None
        previous: dict[str, Any] | None = None
        for expected_index, receipt in enumerate(self.receipts):
            if (
                receipt["round_index"] != expected_index
                or receipt["session_id"] != self.session_id
                or receipt["capability_sha256"] != self.capability["capability_sha256"]
            ):
                raise LiveCommitControllerError(
                    "existing receipt sequence has a different controller identity"
                )
            if previous is not None and receipt["canonical_before"] != previous:
                raise LiveCommitControllerError("existing receipt canonical chain is broken")
            if self.candidate_quarantined and receipt["mode"] != "quarantined_serial":
                raise LiveCommitControllerError(
                    "existing receipt sequence re-enabled a quarantined candidate"
                )
            if receipt["counterexample_sha256"] is not None:
                self.counterexample_sha256 = receipt["counterexample_sha256"]
            self.candidate_quarantined = (
                self.candidate_quarantined or not receipt["candidate_enabled_after"]
            )
            previous = receipt["canonical_after"]
        for recovery in self.recoveries:
            self._validate_existing_recovery(recovery)
            if recovery["disposition"] == "rolled_back":
                self.candidate_quarantined = True
                self.counterexample_sha256 = recovery["counterexample_sha256"]
        resolved_intents = {
            receipt["publication_intent_sha256"] for receipt in self.receipts
        } | {
            recovery["publication_intent_sha256"] for recovery in self.recoveries
        }
        unresolved = self.journal.unresolved_intents(resolved_intents)
        if len(unresolved) > 1:
            raise LiveCommitControllerError(
                "multiple unresolved pre-publication intents cannot be ordered safely"
            )
        if unresolved:
            intent_sha256 = unresolved[0].name.removesuffix(".json").rsplit("-", 1)[1]
            self._recover_intent(
                self.journal._load_intent(unresolved[0], intent_sha256), intent_sha256
            )

    def _validate_existing_recovery(self, recovery: dict[str, Any]) -> None:
        intent = self.journal.load_intent_by_sha256(
            recovery["publication_intent_sha256"]
        )
        if (
            recovery["capability_sha256"] != self.capability["capability_sha256"]
            or recovery["session_id"] != self.session_id
            or recovery["request_id"] != intent["request_id"]
            or recovery["round_index"] != intent["round_index"]
        ):
            raise LiveCommitControllerError(
                "publication recovery has a different controller identity"
            )
        expected_state = (
            intent["selected_state"]
            if recovery["disposition"] == "committed"
            else intent["canonical_before"]
        )
        if recovery["canonical_state"] != expected_state:
            raise LiveCommitControllerError(
                "publication recovery canonical state differs from its disposition"
            )
        matching = [
            receipt
            for receipt in self.receipts
            if receipt["publication_intent_sha256"]
            == recovery["publication_intent_sha256"]
        ]
        if recovery["disposition"] == "committed" and len(matching) != 1:
            raise LiveCommitControllerError(
                "committed publication recovery lacks exactly one durable receipt"
            )
        if recovery["disposition"] == "rolled_back" and matching:
            raise LiveCommitControllerError(
                "rolled-back publication recovery unexpectedly has a receipt"
            )

    def _recover_intent(self, intent: dict[str, Any], intent_sha256: str) -> None:
        """Resolve one crash window using the driver's idempotent root oracle."""

        if (
            intent["capability_sha256"] != self.capability["capability_sha256"]
            or intent["session_id"] != self.session_id
            or intent["round_index"] != len(self.receipts)
        ):
            raise LiveCommitControllerError(
                "unresolved publication intent has a different controller identity"
            )
        if self.receipts and intent["canonical_before"] != self.receipts[-1]["canonical_after"]:
            raise LiveCommitControllerError(
                "unresolved publication intent does not extend the durable receipt chain"
            )
        try:
            recovery = self.driver.recover_publication(intent)
        except Exception as error:
            self.journal.preserve_failure(
                {
                    "schema": "urn:qwen-r9700:live-serial-fatal:v1",
                    "capability_sha256": self.capability["capability_sha256"],
                    "session_id": self.session_id,
                    "request_id": intent["request_id"],
                    "round_index": intent["round_index"],
                    "canonical_before_sha256": validation._digest(
                        intent["canonical_before"]
                    ),
                    "failure": _failure("publication.recovery", error),
                }
            )
            raise LiveCommitControllerError(
                "interrupted publication could not be recovered safely"
            ) from error
        if (
            not isinstance(recovery, PublicationRecovery)
            or not recovery.atomic
            or recovery.disposition not in {"committed", "rolled_back"}
            or not recovery.candidate_private_state_destroyed
            or not recovery.serial_private_state_destroyed
        ):
            raise LiveCommitControllerError(
                "driver returned an incomplete publication recovery"
            )
        state = validation.normalize_state(
            recovery.state, "recovered publication canonical state"
        )
        probe = validation.normalize_state(
            self.driver.probe_canonical(), "recovered publication canonical probe"
        )
        if probe != state:
            raise LiveCommitControllerError(
                "recovered publication state differs from the canonical probe"
            )
        expected = (
            intent["selected_state"]
            if recovery.disposition == "committed"
            else intent["canonical_before"]
        )
        if state != expected:
            raise LiveCommitControllerError(
                "recovered publication state differs from the durable intent"
            )

        counterexample = intent["counterexample_sha256"]
        if recovery.disposition == "rolled_back":
            counterexample = self.journal.preserve_counterexample(
                {
                    "schema": COUNTEREXAMPLE_SCHEMA,
                    "capability_sha256": self.capability["capability_sha256"],
                    "session_id": self.session_id,
                    "request_id": intent["request_id"],
                    "round_index": intent["round_index"],
                    "canonical_before_sha256": validation._digest(
                        intent["canonical_before"]
                    ),
                    "candidate_state_sha256": (
                        None
                        if intent["receipt_template"]["candidate_state"] is None
                        else validation._digest(
                            intent["receipt_template"]["candidate_state"]
                        )
                    ),
                    "candidate_token_ids": intent["receipt_template"][
                        "candidate_token_ids"
                    ],
                    "candidate_failure": _synthetic_failure(
                        "publication.recovery_rollback",
                        "interrupted publication restored the prior canonical root",
                    ),
                    "serial_state_sha256": validation._digest(
                        intent["receipt_template"]["serial_state"]
                    ),
                    "serial_token_ids": intent["receipt_template"]["serial_token_ids"],
                    "first_difference": {
                        "field": "transaction_state_sha256",
                        "left": intent["selected_state"]["transaction_state_sha256"],
                        "right": intent["canonical_before"]["transaction_state_sha256"],
                    },
                }
            )
            self.candidate_quarantined = True
            self.counterexample_sha256 = counterexample
        else:
            receipt = validation.seal_receipt(
                {
                    **intent["receipt_template"],
                    "publication_intent_sha256": intent_sha256,
                    "canonical_after": state,
                    "publication_atomic": True,
                    "candidate_private_state_destroyed": True,
                    "serial_private_state_destroyed": True,
                }
            )
            self.journal.preserve_receipt(receipt)
            self.receipts.append(receipt)
            if not receipt["candidate_enabled_after"]:
                self.candidate_quarantined = True
                self.counterexample_sha256 = receipt["counterexample_sha256"]

        recovery_record = {
            "schema": PUBLICATION_RECOVERY_SCHEMA,
            "publication_intent_sha256": intent_sha256,
            "capability_sha256": self.capability["capability_sha256"],
            "session_id": self.session_id,
            "request_id": intent["request_id"],
            "round_index": intent["round_index"],
            "disposition": recovery.disposition,
            "canonical_state": state,
            "publication_atomic": True,
            "candidate_private_state_destroyed": True,
            "serial_private_state_destroyed": True,
            "counterexample_sha256": counterexample,
        }
        self.journal.preserve_recovery(recovery_record)
        self.recoveries.append(recovery_record)

    def _verify_controller_identity(self) -> None:
        source = Path(__file__).resolve(strict=True)
        declared = Path(self.capability["controller_runtime"]["path"])
        try:
            same_file = source.samefile(declared)
        except OSError as error:
            raise LiveCommitControllerError(
                "live commit controller runtime path cannot be authenticated"
            ) from error
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if (
            not same_file
            or actual != self.capability["controller_runtime"]["sha256"]
        ):
            raise LiveCommitControllerError(
                "live commit controller runtime identity differs from capability"
            )

    def _verify_driver_identity(self) -> None:
        expected = self.capability
        observed = {
            "release_runtime_sha256": self.driver.release_runtime_sha256,
            "serial_runtime_sha256": self.driver.serial_runtime_sha256,
            "candidate_implementation_family": self.driver.candidate_implementation_family,
            "serial_implementation_family": self.driver.serial_implementation_family,
        }
        required = {
            "release_runtime_sha256": expected["release_runtime"]["sha256"],
            "serial_runtime_sha256": expected["serial_oracle_runtime"]["sha256"],
            "candidate_implementation_family": expected["candidate_implementation_family"],
            "serial_implementation_family": expected["serial_implementation_family"],
        }
        if observed != required:
            raise LiveCommitControllerError(
                "live transition driver identity differs from capability"
            )

    def _snapshot(self) -> CanonicalSnapshot:
        snapshot = self.driver.snapshot_canonical()
        if not isinstance(snapshot, CanonicalSnapshot):
            raise LiveCommitControllerError("driver returned an invalid canonical snapshot")
        state = validation.normalize_state(snapshot.state, "canonical snapshot")
        probe = validation.normalize_state(self.driver.probe_canonical(), "canonical probe")
        if probe != state:
            raise LiveCommitControllerError("canonical snapshot and live probe differ")
        return CanonicalSnapshot(snapshot.handle_id, state)

    def _require_unchanged(self, snapshot: CanonicalSnapshot, stage: str) -> str:
        probe = validation.normalize_state(self.driver.probe_canonical(), f"{stage} probe")
        if probe != snapshot.state:
            raise LiveCommitControllerError(f"canonical state changed during {stage}")
        return validation._digest(probe)

    def _restore(self, snapshot: CanonicalSnapshot, stage: str) -> None:
        publication = self.driver.restore_canonical(snapshot)
        if not isinstance(publication, Publication) or not publication.atomic:
            raise LiveCommitControllerError(f"{stage} restoration was not atomic")
        restored = validation.normalize_state(publication.state, f"{stage} restored state")
        probe = validation.normalize_state(self.driver.probe_canonical(), f"{stage} restored probe")
        if restored != snapshot.state or probe != snapshot.state:
            raise LiveCommitControllerError(f"{stage} could not restore the canonical snapshot")

    def _transition(
        self,
        value: object,
        *,
        role: str,
        commit_count: int | None,
    ) -> PrivateTransition:
        if not isinstance(value, PrivateTransition):
            raise LiveCommitControllerError(f"{role} returned an invalid private transition")
        expected_family = (
            self.driver.candidate_implementation_family
            if role == "candidate"
            else self.driver.serial_implementation_family
        )
        if value.implementation_family != expected_family:
            raise LiveCommitControllerError(f"{role} implementation family differs")
        if (
            commit_count is not None
            and len(value.token_ids) != commit_count
        ) or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in value.token_ids
        ):
            raise LiveCommitControllerError(f"{role} token count or IDs differ")
        if commit_count is None and len(value.token_ids) not in validation.COMMIT_COUNTS:
            raise LiveCommitControllerError(
                f"{role} live-derived token count is outside zero through eight"
            )
        state = validation.normalize_state(value.state, f"{role} private state")
        return PrivateTransition(
            value.handle_id,
            value.implementation_family,
            tuple(value.token_ids),
            state,
            value.model_output,
        )

    def _counterexample(
        self,
        *,
        request_id: str,
        round_index: int,
        snapshot: CanonicalSnapshot,
        candidate: PrivateTransition | None,
        serial: PrivateTransition,
        candidate_failure: dict[str, str] | None,
        first_difference: dict[str, Any] | None,
    ) -> str:
        document = {
            "schema": COUNTEREXAMPLE_SCHEMA,
            "capability_sha256": self.capability["capability_sha256"],
            "session_id": self.session_id,
            "request_id": request_id,
            "round_index": round_index,
            "canonical_before_sha256": validation._digest(snapshot.state),
            "candidate_state_sha256": (
                None if candidate is None else validation._digest(candidate.state)
            ),
            "candidate_token_ids": None if candidate is None else list(candidate.token_ids),
            "candidate_failure": candidate_failure,
            "serial_state_sha256": validation._digest(serial.state),
            "serial_token_ids": list(serial.token_ids),
            "first_difference": first_difference,
        }
        return self.journal.preserve_counterexample(document)

    def _fatal(
        self,
        *,
        request_id: str,
        round_index: int,
        stage: str,
        error: BaseException,
        snapshot: CanonicalSnapshot,
    ) -> None:
        self.journal.preserve_failure(
            {
                "schema": "urn:qwen-r9700:live-serial-fatal:v1",
                "capability_sha256": self.capability["capability_sha256"],
                "session_id": self.session_id,
                "request_id": request_id,
                "round_index": round_index,
                "canonical_before_sha256": validation._digest(snapshot.state),
                "failure": _failure(stage, error),
            }
        )

    def run_round(
        self,
        *,
        request_id: str,
        round_index: int,
        commit_count: int | None,
        request_payload: object,
        serial_fallback_commit_count: int | None = None,
    ) -> dict[str, Any]:
        """Run one transition and return a verified, durably published receipt."""

        request_id = validation._identifier(request_id, "controller request ID")
        if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
            raise LiveCommitControllerError("round index is invalid")
        if round_index != len(self.receipts):
            raise LiveCommitControllerError(
                f"round index must be the next durable index {len(self.receipts)}"
            )
        requested_commit_count = commit_count
        if requested_commit_count is not None:
            if requested_commit_count not in validation.COMMIT_COUNTS:
                raise LiveCommitControllerError("commit count is outside 0 through 8")
            if serial_fallback_commit_count is not None:
                raise LiveCommitControllerError(
                    "fixed-width execution must not provide a serial fallback width"
                )
        elif serial_fallback_commit_count not in validation.COMMIT_COUNTS:
            raise LiveCommitControllerError(
                "live-derived execution requires a serial fallback width zero through eight"
            )
        snapshot = self._snapshot()
        if self.receipts and snapshot.state != self.receipts[-1]["canonical_after"]:
            raise LiveCommitControllerError(
                "live canonical state differs from the durable receipt chain"
            )
        candidate: PrivateTransition | None = None
        serial: PrivateTransition | None = None
        candidate_failure: dict[str, str] | None = None
        candidate_enabled_before = not self.candidate_quarantined
        probe_after_candidate = validation._digest(snapshot.state)

        if candidate_enabled_before:
            try:
                candidate = self._transition(
                    self.driver.execute_candidate(snapshot, request_payload, commit_count),
                    role="candidate",
                    commit_count=commit_count,
                )
                probe_after_candidate = self._require_unchanged(snapshot, "candidate execution")
            except Exception as error:
                candidate_failure = _failure("candidate.execute", error)
                self.driver.cleanup_failed_private(
                    "candidate", self.session_id, request_id, round_index
                )
                self._restore(snapshot, "candidate failure")
                candidate = None
                self.candidate_quarantined = True

        if requested_commit_count is None:
            commit_count = (
                len(candidate.token_ids)
                if candidate is not None
                else serial_fallback_commit_count
            )
        else:
            commit_count = requested_commit_count
        assert commit_count is not None

        try:
            serial = self._transition(
                self.driver.execute_serial(snapshot, request_payload, commit_count),
                role="serial",
                commit_count=commit_count,
            )
            probe_after_serial = self._require_unchanged(snapshot, "serial execution")
        except Exception as error:
            self.driver.cleanup_failed_private("serial", self.session_id, request_id, round_index)
            self._restore(snapshot, "serial failure")
            self._fatal(
                request_id=request_id,
                round_index=round_index,
                stage="serial.execute",
                error=error,
                snapshot=snapshot,
            )
            raise LiveCommitControllerError(
                "authoritative serial transition failed; canonical state was restored"
            ) from error

        assert serial is not None
        first_difference: dict[str, Any] | None = None
        comparison_equal = False
        if candidate is not None:
            if candidate.token_ids != serial.token_ids:
                first_difference = {
                    "field": "committed_token_ids",
                    "left": list(candidate.token_ids),
                    "right": list(serial.token_ids),
                }
            else:
                first_difference = validation.first_state_difference(
                    candidate.state, serial.state
                )
            comparison_equal = first_difference is None
            if not comparison_equal:
                self.candidate_quarantined = True

        if candidate_failure is not None or (candidate is not None and not comparison_equal):
            self.counterexample_sha256 = self._counterexample(
                request_id=request_id,
                round_index=round_index,
                snapshot=snapshot,
                candidate=candidate,
                serial=serial,
                candidate_failure=candidate_failure,
                first_difference=first_difference,
            )

        if candidate is not None:
            mode = "validated_candidate"
            if comparison_equal:
                selected = candidate
                publication_source = "unchanged" if commit_count == 0 else "candidate"
            else:
                selected = serial
                publication_source = "unchanged" if commit_count == 0 else "serial"
        else:
            selected = serial
            mode = (
                "candidate_fault_serial"
                if candidate_failure is not None
                else "quarantined_serial"
            )
            publication_source = "unchanged" if commit_count == 0 else "serial"
            if self.counterexample_sha256 is None:
                raise LiveCommitControllerError("quarantined execution lacks a counterexample")

        candidate_for_receipt = candidate if mode == "validated_candidate" else None
        comparison = {
            "candidate_state_sha256": (
                None
                if candidate_for_receipt is None
                else validation._digest(candidate_for_receipt.state)
            ),
            "candidate_token_ids": (
                None if candidate_for_receipt is None else list(candidate_for_receipt.token_ids)
            ),
            "candidate_failure": candidate_failure,
            "serial_state_sha256": validation._digest(serial.state),
            "serial_token_ids": list(serial.token_ids),
            "comparison_equal": comparison_equal if mode == "validated_candidate" else False,
            "first_difference": first_difference if mode == "validated_candidate" else None,
        }
        counterexample_for_receipt = (
            None
            if mode == "validated_candidate" and comparison_equal
            else self.counterexample_sha256
        )
        receipt_template = {
            "schema": validation.RECEIPT_SCHEMA,
            "capability_sha256": self.capability["capability_sha256"],
            "session_id": self.session_id,
            "request_id": request_id,
            "round_index": round_index,
            "commit_count": commit_count,
            "mode": mode,
            "candidate_enabled_before": candidate_enabled_before,
            "candidate_enabled_after": not self.candidate_quarantined,
            "canonical_before": snapshot.state,
            "candidate_state": (
                None if candidate_for_receipt is None else candidate_for_receipt.state
            ),
            "candidate_token_ids": (
                None
                if candidate_for_receipt is None
                else list(candidate_for_receipt.token_ids)
            ),
            "candidate_failure": candidate_failure,
            "serial_state": serial.state,
            "serial_token_ids": list(serial.token_ids),
            "canonical_probe_after_candidate_sha256": probe_after_candidate,
            "canonical_probe_after_serial_sha256": probe_after_serial,
            "comparison_equal": (
                comparison_equal if mode == "validated_candidate" else False
            ),
            "first_difference": (
                first_difference if mode == "validated_candidate" else None
            ),
            "publication_source": publication_source,
            "counterexample_sha256": counterexample_for_receipt,
            "comparison_sha256": validation._digest(comparison),
        }
        intent = {
            "schema": PUBLICATION_INTENT_SCHEMA,
            "capability_sha256": self.capability["capability_sha256"],
            "session_id": self.session_id,
            "request_id": request_id,
            "round_index": round_index,
            "commit_count": commit_count,
            "mode": mode,
            "snapshot_handle_id": snapshot.handle_id,
            "canonical_before": snapshot.state,
            "selected_source": publication_source,
            "selected_handle_id": selected.handle_id,
            "selected_state": selected.state,
            "selected_token_ids": list(selected.token_ids),
            "candidate_handle_id": None if candidate is None else candidate.handle_id,
            "serial_handle_id": serial.handle_id,
            "candidate_failure": candidate_failure,
            "counterexample_sha256": counterexample_for_receipt,
            "receipt_template": receipt_template,
        }
        intent = _normalize_intent(intent)
        publication_intent_sha256 = self.journal.preserve_intent(intent)

        if commit_count == 0:
            canonical_after = snapshot.state
            publication_atomic = True
        else:
            try:
                publication = self.driver.publish_private(
                    snapshot, selected, publication_intent_sha256
                )
            except Exception as error:
                self._restore(snapshot, "publication failure")
                cleanup_errors: list[BaseException] = []
                for transition in (candidate, serial):
                    if transition is None:
                        continue
                    try:
                        self.driver.destroy_private(transition)
                    except Exception as cleanup_error:
                        cleanup_errors.append(cleanup_error)
                self._fatal(
                    request_id=request_id,
                    round_index=round_index,
                    stage="publication",
                    error=error,
                    snapshot=snapshot,
                )
                if cleanup_errors:
                    raise LiveCommitControllerError(
                        "publication failed and private-state cleanup also failed; "
                        "canonical state was restored"
                    ) from cleanup_errors[0]
                raise LiveCommitControllerError(
                    "publication failed; canonical state was restored"
                ) from error
            if not isinstance(publication, Publication) or not publication.atomic:
                self._restore(snapshot, "non-atomic publication")
                raise LiveCommitControllerError("driver did not perform an atomic publication")
            canonical_after = validation.normalize_state(
                publication.state, "published canonical state"
            )
            published_probe = validation.normalize_state(
                self.driver.probe_canonical(), "published canonical probe"
            )
            if canonical_after != selected.state or published_probe != selected.state:
                self._restore(snapshot, "incorrect publication")
                raise LiveCommitControllerError("published state differs from selected transition")
            publication_atomic = True

        candidate_destroyed = candidate is None
        try:
            if candidate is not None:
                self.driver.destroy_private(candidate)
                candidate_destroyed = True
            self.driver.destroy_private(serial)
            serial_destroyed = True
        except Exception as error:
            self._fatal(
                request_id=request_id,
                round_index=round_index,
                stage="private.cleanup",
                error=error,
                snapshot=snapshot,
            )
            raise LiveCommitControllerError("private transition cleanup failed") from error

        receipt = validation.seal_receipt(
            {
                **receipt_template,
                "publication_intent_sha256": publication_intent_sha256,
                "canonical_after": canonical_after,
                "publication_atomic": publication_atomic,
                "candidate_private_state_destroyed": candidate_destroyed,
                "serial_private_state_destroyed": serial_destroyed,
            }
        )
        self.journal.preserve_receipt(receipt)
        self.receipts.append(receipt)
        return receipt

    def finalize_ledger(self) -> dict[str, Any]:
        """Seal the complete durable session chain after the caller ends it."""

        if not self.receipts:
            raise LiveCommitControllerError("cannot finalize an empty live-commit session")
        ledger = validation.seal_ledger(
            {
                "schema": validation.LEDGER_SCHEMA,
                "capability_sha256": self.capability["capability_sha256"],
                "session_id": self.session_id,
                "initial_state": self.receipts[0]["canonical_before"],
                "receipts": self.receipts,
                "final_state": self.receipts[-1]["canonical_after"],
                "candidate_quarantined": self.candidate_quarantined,
                "request_complete": True,
                "classification": "runtime_invariant_not_finite_qualification",
                "universal_guarantee_mechanism": (
                    "independent_serial_before_every_commit"
                ),
            }
        )
        self.journal.preserve_ledger(ledger)
        return ledger
