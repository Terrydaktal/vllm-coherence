"""Non-bypassable application boundary for live serial-validated commits.

The lower-level :mod:`live_commit_controller` owns transition ordering and
durable recovery.  This module is the only interface a production scheduler is
allowed to use to obtain committed tokens.  It authenticates its own release
identity, serializes canonical publications, returns only token IDs from a
verified receipt, and refuses use after the evidence ledger is finalized.

This does not by itself integrate vLLM's physical KV/GDN allocator.  Promotion
must additionally prove that the exact release runtime is a member of the same
artifact and that it emitted a controller ledger through this gate.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen_r9700_lab import live_commit_validation as validation
from qwen_r9700_lab.live_commit_controller import (
    CanonicalSnapshot,
    DurableEvidenceJournal,
    LiveCommitController,
    LiveCommitControllerError,
    LiveTransitionDriver,
)
from qwen_r9700_lab.live_commit_vllm_driver import (
    VerifiedVllmAdvance,
    VllmLiveTransitionDriver,
)


class LiveCommitRuntimeGateError(RuntimeError):
    """The production caller attempted an unauthenticated or unsafe commit."""


@dataclass(frozen=True)
class VerifiedCanonicalCommit:
    """The only commit result that may be exposed to the serving scheduler."""

    token_ids: tuple[int, ...]
    canonical_state: dict[str, Any]
    publication_source: str
    publication_intent_sha256: str
    receipt_sha256: str
    capability_sha256: str


@dataclass(frozen=True)
class _OpenProvisionalRound:
    request_id: str
    round_index: int
    request_payload: object
    canonical_snapshot: CanonicalSnapshot
    canonical_state_sha256: str


@dataclass(frozen=True)
class _PendingVerifiedAdvance:
    request_id: str
    round_index: int
    publication_intent_sha256: str
    token_ids: tuple[int, ...]


class LiveCommitRuntimeGate:
    """Serialize and authenticate every canonical serving transition."""

    def __init__(
        self,
        capability: object,
        driver: LiveTransitionDriver,
        journal: DurableEvidenceJournal,
        *,
        session_id: str,
    ) -> None:
        verified = validation.verify_capability(capability)
        self._verify_runtime_identity(verified)
        if type(driver) is not VllmLiveTransitionDriver:
            raise LiveCommitRuntimeGateError(
                "production live commit gate requires the authenticated vLLM driver"
            )
        self._controller = LiveCommitController(
            verified,
            driver,
            journal,
            session_id=session_id,
        )
        self._lock = threading.Lock()
        self._closed = False
        self._open_round: _OpenProvisionalRound | None = None
        self._pending_advance: _PendingVerifiedAdvance | None = None

    @staticmethod
    def _verify_runtime_identity(capability: dict[str, Any]) -> None:
        source = Path(__file__).resolve(strict=True)
        declared = Path(capability["enforcement_runtime"]["path"])
        try:
            same_file = source.samefile(declared)
        except OSError as error:
            raise LiveCommitRuntimeGateError(
                "live commit enforcement runtime path cannot be authenticated"
            ) from error
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if not same_file or actual != capability["enforcement_runtime"]["sha256"]:
            raise LiveCommitRuntimeGateError(
                "live commit enforcement runtime identity differs from capability"
            )

    @property
    def capability_sha256(self) -> str:
        return self._controller.capability["capability_sha256"]

    def _abort_private_and_restore(self, opened: _OpenProvisionalRound) -> bool:
        """Abort an open round and restore its authenticated canonical root if needed."""

        abort_error: BaseException | None = None
        try:
            self._controller.driver.abort_provisional_round(
                opened.request_id, opened.round_index
            )
        except BaseException as error:
            abort_error = error

        try:
            changed = (
                validation.state_sha256(self._controller.driver.probe_canonical())
                != opened.canonical_state_sha256
            )
        except BaseException:
            changed = True

        if changed or abort_error is not None:
            try:
                restored = self._controller.driver.restore_canonical(
                    opened.canonical_snapshot
                )
                restored_sha256 = validation.state_sha256(restored.state)
                probed_sha256 = validation.state_sha256(
                    self._controller.driver.probe_canonical()
                )
            except BaseException as error:
                raise LiveCommitRuntimeGateError(
                    "provisional cleanup could not restore canonical state"
                ) from error
            if (
                not restored.atomic
                or restored_sha256 != opened.canonical_state_sha256
                or probed_sha256 != opened.canonical_state_sha256
            ):
                raise LiveCommitRuntimeGateError(
                    "provisional cleanup did not restore the authenticated canonical root"
                )
        if abort_error is not None:
            raise LiveCommitRuntimeGateError(
                "provisional execution could not be aborted cleanly"
            ) from abort_error
        return changed

    def begin_provisional_round(
        self,
        *,
        request_id: str,
        round_index: int,
        request_payload: object,
    ) -> object:
        """Open the only interval in which private candidate execution may run."""

        if not self._lock.acquire(blocking=False):
            raise LiveCommitRuntimeGateError("concurrent canonical commits are forbidden")
        opened: _OpenProvisionalRound | None = None
        try:
            if self._closed:
                raise LiveCommitRuntimeGateError("live commit gate is already finalized")
            if self._pending_advance is not None:
                raise LiveCommitRuntimeGateError(
                    "verified output must be consumed before another round"
                )
            if self._open_round is not None:
                raise LiveCommitRuntimeGateError("a provisional round is already open")
            snapshot = self._controller.driver.snapshot_canonical()
            canonical_sha256 = validation.state_sha256(snapshot.state)
            opened = _OpenProvisionalRound(
                request_id=request_id,
                round_index=round_index,
                request_payload=request_payload,
                canonical_snapshot=snapshot,
                canonical_state_sha256=canonical_sha256,
            )
            prepared_payload = self._controller.driver.begin_provisional_round(
                snapshot,
                request_payload,
                request_id,
                round_index,
            )
            opened = _OpenProvisionalRound(
                request_id=request_id,
                round_index=round_index,
                request_payload=prepared_payload,
                canonical_snapshot=snapshot,
                canonical_state_sha256=canonical_sha256,
            )
            if (
                validation.state_sha256(self._controller.driver.probe_canonical())
                != canonical_sha256
            ):
                raise LiveCommitRuntimeGateError(
                    "opening provisional execution mutated canonical state"
                )
            self._open_round = opened
            return prepared_payload
        except BaseException:
            if opened is not None:
                try:
                    self._abort_private_and_restore(opened)
                except BaseException as cleanup_error:
                    raise LiveCommitRuntimeGateError(
                        "provisional begin failed and cleanup also failed"
                    ) from cleanup_error
            raise
        finally:
            self._lock.release()

    def abort_provisional_round(self, *, request_id: str, round_index: int) -> None:
        """Cancel private execution while proving the canonical root stayed unchanged."""

        if not self._lock.acquire(blocking=False):
            raise LiveCommitRuntimeGateError("concurrent canonical commits are forbidden")
        try:
            opened = self._open_round
            if opened is None:
                raise LiveCommitRuntimeGateError("no provisional round is open")
            if opened.request_id != request_id or opened.round_index != round_index:
                raise LiveCommitRuntimeGateError("provisional abort identity differs")
            changed = self._abort_private_and_restore(opened)
            self._open_round = None
            if changed:
                raise LiveCommitRuntimeGateError(
                    "aborted provisional execution changed canonical state and was restored"
                )
        finally:
            self._lock.release()

    def commit(
        self,
        *,
        request_id: str,
        round_index: int,
        commit_count: int | None,
        request_payload: object,
        serial_fallback_commit_count: int | None = None,
    ) -> VerifiedCanonicalCommit:
        """Run both arms and expose only the receipt-authenticated publication."""

        if not self._lock.acquire(blocking=False):
            raise LiveCommitRuntimeGateError("concurrent canonical commits are forbidden")
        try:
            if self._closed:
                raise LiveCommitRuntimeGateError("live commit gate is already finalized")
            opened = self._open_round
            if opened is None:
                raise LiveCommitRuntimeGateError(
                    "canonical commit was attempted without an open provisional round"
                )
            if (
                opened.request_id != request_id
                or opened.round_index != round_index
                or opened.request_payload is not request_payload
            ):
                raise LiveCommitRuntimeGateError(
                    "canonical commit identity differs from open round"
                )
            if (
                validation.state_sha256(self._controller.driver.probe_canonical())
                != opened.canonical_state_sha256
            ):
                raise LiveCommitRuntimeGateError(
                    "candidate execution mutated canonical state before validation"
                )
            receipt = validation.verify_receipt(
                self._controller.run_round(
                    request_id=request_id,
                    round_index=round_index,
                    commit_count=commit_count,
                    request_payload=request_payload,
                    serial_fallback_commit_count=serial_fallback_commit_count,
                )
            )
            if receipt["capability_sha256"] != self.capability_sha256:
                raise LiveCommitRuntimeGateError("commit receipt capability differs")
            if receipt["round_index"] != round_index:
                raise LiveCommitRuntimeGateError("commit receipt round differs")
            if commit_count is not None and receipt["commit_count"] != commit_count:
                raise LiveCommitRuntimeGateError("commit receipt width differs")
            effective_commit_count = receipt["commit_count"]
            if receipt["publication_source"] == "candidate":
                tokens = receipt["candidate_token_ids"]
            elif receipt["publication_source"] == "serial":
                tokens = receipt["serial_token_ids"]
            elif receipt["publication_source"] == "unchanged":
                tokens = []
            else:  # Defensive even though verify_receipt is exhaustive.
                raise LiveCommitRuntimeGateError("commit receipt source is unsupported")
            if tokens is None or len(tokens) != effective_commit_count:
                raise LiveCommitRuntimeGateError("commit receipt does not expose exact tokens")
            if effective_commit_count == 0:
                changed = self._abort_private_and_restore(opened)
                # The scheduler-side provisional round has now been closed.  Clear
                # the gate marker before raising so exception cleanup cannot abort
                # the same physical roots twice.
                self._open_round = None
                if changed:
                    raise LiveCommitRuntimeGateError(
                        "zero-commit provisional execution changed canonical state"
                    )
            committed = VerifiedCanonicalCommit(
                token_ids=tuple(tokens),
                canonical_state=receipt["canonical_after"],
                publication_source=receipt["publication_source"],
                publication_intent_sha256=receipt["publication_intent_sha256"],
                receipt_sha256=receipt["receipt_sha256"],
                capability_sha256=receipt["capability_sha256"],
            )
            if effective_commit_count:
                self._pending_advance = _PendingVerifiedAdvance(
                    request_id=request_id,
                    round_index=round_index,
                    publication_intent_sha256=committed.publication_intent_sha256,
                    token_ids=committed.token_ids,
                )
            return committed
        except BaseException as error:
            opened = self._open_round
            if opened is not None:
                try:
                    self._abort_private_and_restore(opened)
                except BaseException as cleanup_error:
                    raise LiveCommitRuntimeGateError(
                        "serial-validated commit failed and provisional cleanup failed"
                    ) from cleanup_error
            if isinstance(error, LiveCommitRuntimeGateError):
                raise
            if isinstance(error, LiveCommitControllerError):
                raise LiveCommitRuntimeGateError(
                    "serial-validated canonical commit failed closed"
                ) from error
            raise
        finally:
            self._open_round = None
            self._lock.release()

    def consume_verified_advance(
        self,
        *,
        request_id: str,
        round_index: int,
        publication_intent_sha256: str,
    ) -> VerifiedVllmAdvance:
        """Release exactly one verified output to EngineCore for scheduler update."""

        if not self._lock.acquire(blocking=False):
            raise LiveCommitRuntimeGateError("concurrent canonical commits are forbidden")
        try:
            if self._closed:
                raise LiveCommitRuntimeGateError("live commit gate is already finalized")
            if self._open_round is not None:
                raise LiveCommitRuntimeGateError(
                    "cannot consume verified output while a round is open"
                )
            pending = self._pending_advance
            if pending is None:
                raise LiveCommitRuntimeGateError("no verified request advance is pending")
            if (
                request_id,
                round_index,
                publication_intent_sha256,
            ) != (
                pending.request_id,
                pending.round_index,
                pending.publication_intent_sha256,
            ):
                raise LiveCommitRuntimeGateError("verified request advance identity differs")
            try:
                advance = self._controller.driver.consume_verified_advance(
                    request_id,
                    round_index,
                    publication_intent_sha256,
                    pending.token_ids,
                )
            except BaseException as error:
                raise LiveCommitRuntimeGateError(
                    "scheduler rejected the verified request advance"
                ) from error
            if advance.token_ids != pending.token_ids:
                raise LiveCommitRuntimeGateError(
                    "scheduler released different tokens than the verified receipt"
                )
            self._pending_advance = None
            return advance
        finally:
            self._lock.release()

    def finalize(self) -> dict[str, Any]:
        """Close the gate and return its authenticated complete ledger."""

        if not self._lock.acquire(blocking=False):
            raise LiveCommitRuntimeGateError("cannot finalize during a canonical commit")
        try:
            if self._closed:
                raise LiveCommitRuntimeGateError("live commit gate is already finalized")
            if self._open_round is not None:
                raise LiveCommitRuntimeGateError(
                    "cannot finalize while a provisional round is open"
                )
            if self._pending_advance is not None:
                raise LiveCommitRuntimeGateError(
                    "cannot finalize before verified output is consumed"
                )
            ledger = validation.verify_ledger(self._controller.finalize_ledger())
            if ledger["capability_sha256"] != self.capability_sha256:
                raise LiveCommitRuntimeGateError("final ledger capability differs")
            self._closed = True
            return ledger
        except LiveCommitControllerError as error:
            raise LiveCommitRuntimeGateError("live commit ledger failed closed") from error
        finally:
            self._lock.release()
