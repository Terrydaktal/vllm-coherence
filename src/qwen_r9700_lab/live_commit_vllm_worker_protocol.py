"""Typed wire contract for private candidate and serial vLLM worker calls.

The EngineCore process and the single TP=1 GPU worker communicate through
``Executor.collective_rpc``.  This module makes that boundary explicit: a
scheduler-prepared branch request is immutable and identity-bound, and a worker
response must echo every identity before it can become a ``PrivateTransition``.
Opaque vLLM scheduler payloads are never authenticated with ``repr`` or pickle;
their producer and consumer must provide the same deterministic SHA-256.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from qwen_r9700_lab import live_commit_validation as validation

PRIVATE_WORKER_PROTOCOL_SCHEMA = "urn:qwen-r9700:private-worker-round:v1"
BranchRole = Literal["candidate", "serial"]
_IDENTIFIER = re.compile(r"^[A-Za-z0-9._:/+-]{1,256}$")


class PrivateWorkerProtocolError(RuntimeError):
    """A private worker call is malformed, ambiguous, or identity-mismatched."""


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise PrivateWorkerProtocolError(f"{label} is invalid")
    return value


def _index(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PrivateWorkerProtocolError(f"{label} is invalid")
    return value


def _commit_count(value: object, label: str, *, nullable: bool) -> int | None:
    if nullable and value is None:
        return None
    if value not in validation.COMMIT_COUNTS:
        raise PrivateWorkerProtocolError(f"{label} is invalid")
    return value


def _sha256(value: object, label: str) -> str:
    try:
        return validation._sha256(value, label)
    except validation.LiveCommitValidationError as error:
        raise PrivateWorkerProtocolError(f"{label} is invalid") from error


def _token_ids(value: object, label: str) -> tuple[int, ...]:
    if type(value) is not tuple or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in value
    ):
        raise PrivateWorkerProtocolError(f"{label} are invalid")
    return value


@dataclass(frozen=True)
class PrivateBranchRequest:
    """One scheduler-authenticated request for a branch-private GPU forward."""

    schema: str
    role: BranchRole
    session_id: str
    request_id: str
    round_index: int
    canonical_handle_id: str
    canonical_state_sha256: str
    branch_handle_id: str
    branch_payload_sha256: str
    transaction_nonce_sha256: str
    requested_commit_count: int | None
    worker_payload: object

    def validate(self, expected_role: BranchRole) -> PrivateBranchRequest:
        if self.schema != PRIVATE_WORKER_PROTOCOL_SCHEMA:
            raise PrivateWorkerProtocolError("private worker request schema differs")
        if self.role != expected_role:
            raise PrivateWorkerProtocolError("private worker request role differs")
        _identifier(self.session_id, "private worker session ID")
        _identifier(self.request_id, "private worker request ID")
        _index(self.round_index, "private worker round index")
        _identifier(self.canonical_handle_id, "private worker canonical handle")
        _sha256(self.canonical_state_sha256, "private worker canonical state")
        _identifier(self.branch_handle_id, "private worker branch handle")
        _sha256(self.branch_payload_sha256, "private worker payload")
        _sha256(self.transaction_nonce_sha256, "private worker transaction nonce")
        _commit_count(
            self.requested_commit_count,
            "private worker requested commit count",
            nullable=True,
        )
        if self.worker_payload is None:
            raise PrivateWorkerProtocolError("private worker payload is absent")
        return self


@dataclass(frozen=True)
class PreparedVllmRound:
    """The scheduler's two disjoint worker requests for one canonical snapshot."""

    session_id: str
    request_id: str
    round_index: int
    canonical_handle_id: str
    canonical_state_sha256: str
    candidate: PrivateBranchRequest
    serial: PrivateBranchRequest

    def validate(
        self,
        *,
        snapshot_handle_id: str,
        snapshot_state_sha256: str,
    ) -> PreparedVllmRound:
        _identifier(self.session_id, "prepared session ID")
        _identifier(self.request_id, "prepared request ID")
        _index(self.round_index, "prepared round index")
        _identifier(self.canonical_handle_id, "prepared canonical handle")
        _sha256(self.canonical_state_sha256, "prepared canonical state")
        self.candidate.validate("candidate")
        self.serial.validate("serial")
        common = (
            self.session_id,
            self.request_id,
            self.round_index,
            self.canonical_handle_id,
            self.canonical_state_sha256,
        )
        for branch in (self.candidate, self.serial):
            observed = (
                branch.session_id,
                branch.request_id,
                branch.round_index,
                branch.canonical_handle_id,
                branch.canonical_state_sha256,
            )
            if observed != common:
                raise PrivateWorkerProtocolError(
                    f"prepared {branch.role} identity differs from its round"
                )
        if self.candidate.branch_handle_id == self.serial.branch_handle_id:
            raise PrivateWorkerProtocolError("private branch handles alias")
        if (
            self.candidate.transaction_nonce_sha256
            != self.serial.transaction_nonce_sha256
        ):
            raise PrivateWorkerProtocolError("private branch transaction nonces differ")
        if (
            self.canonical_handle_id != snapshot_handle_id
            or self.canonical_state_sha256 != snapshot_state_sha256
        ):
            raise PrivateWorkerProtocolError(
                "prepared round differs from the authenticated canonical snapshot"
            )
        return self


@dataclass(frozen=True)
class PrivateWorkerResult:
    """One complete private branch result returned by the GPU worker."""

    schema: str
    role: BranchRole
    session_id: str
    request_id: str
    round_index: int
    canonical_handle_id: str
    canonical_state_sha256: str
    branch_handle_id: str
    branch_payload_sha256: str
    transaction_nonce_sha256: str
    implementation_family: str
    token_ids: tuple[int, ...]
    state: dict[str, Any]
    model_output: object | None

    def validate(
        self,
        request: PrivateBranchRequest,
        *,
        implementation_family: str,
    ) -> PrivateWorkerResult:
        request.validate(request.role)
        if self.schema != PRIVATE_WORKER_PROTOCOL_SCHEMA:
            raise PrivateWorkerProtocolError("private worker result schema differs")
        if self.role != request.role:
            raise PrivateWorkerProtocolError("private worker result role differs")
        expected = (
            request.session_id,
            request.request_id,
            request.round_index,
            request.canonical_handle_id,
            request.canonical_state_sha256,
            request.branch_handle_id,
            request.branch_payload_sha256,
            request.transaction_nonce_sha256,
        )
        observed = (
            self.session_id,
            self.request_id,
            self.round_index,
            self.canonical_handle_id,
            self.canonical_state_sha256,
            self.branch_handle_id,
            self.branch_payload_sha256,
            self.transaction_nonce_sha256,
        )
        if observed != expected:
            raise PrivateWorkerProtocolError(
                "private worker result identity differs from its request"
            )
        _identifier(self.implementation_family, "private worker implementation family")
        if self.implementation_family != implementation_family:
            raise PrivateWorkerProtocolError(
                "private worker implementation family differs"
            )
        token_ids = _token_ids(self.token_ids, "private worker result token IDs")
        if (
            request.requested_commit_count is not None
            and len(token_ids) != request.requested_commit_count
        ):
            raise PrivateWorkerProtocolError(
                "private worker result width differs from the requested width"
            )
        if token_ids and self.model_output is None:
            raise PrivateWorkerProtocolError(
                "private worker result omits its model output"
            )
        try:
            validation.normalize_state(self.state, "private worker result state")
        except validation.LiveCommitValidationError as error:
            raise PrivateWorkerProtocolError(
                "private worker result state is invalid"
            ) from error
        return self
