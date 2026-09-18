"""Scheduler-owned canonical roots for serial-validated live commits.

The GPU worker is not the canonical-state authority in vLLM.  Request block
tables, allocation ownership and request advancement live in the scheduler.
This module provides the scheduler-side state machine that release source hooks
can embed. Candidate and serial workers receive disjoint writable roots; the
authority itself invokes an authenticated physical-root preparer; every result
is checked against a fresh complete physical-state capture; publication is one
assignment of an immutable authority state; and ``update_from_output`` cannot
consume tokens until publication produces a matching one-shot advance permit.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal

from qwen_r9700_lab import live_commit_validation as validation
from qwen_r9700_lab.live_commit_controller import PrivateTransition
from qwen_r9700_lab.live_commit_vllm_driver import VerifiedVllmAdvance

_IDENTIFIER = re.compile(r"^[A-Za-z0-9._:/+-]{1,256}$")
BranchRole = Literal["candidate", "serial"]


class SchedulerAuthorityError(RuntimeError):
    """A scheduler operation could expose or advance unverified state."""


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise SchedulerAuthorityError(f"{label} is invalid")
    return value


def _block_groups(value: object, label: str) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, tuple) or not value:
        raise SchedulerAuthorityError(f"{label} must contain at least one group")
    normalized: list[tuple[int, ...]] = []
    for group_index, group in enumerate(value):
        if not isinstance(group, tuple):
            raise SchedulerAuthorityError(f"{label}[{group_index}] is not immutable")
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in group):
            raise SchedulerAuthorityError(f"{label}[{group_index}] contains an invalid block")
        # vLLM pads sparse/sliding logical tables with its globally reserved
        # null block (physical ID 0).  That sentinel may repeat; every real
        # physical block must remain unique inside the group.
        real_blocks = tuple(item for item in group if item != 0)
        if len(real_blocks) != len(set(real_blocks)):
            raise SchedulerAuthorityError(
                f"{label}[{group_index}] contains duplicate real blocks"
            )
        normalized.append(group)
    return tuple(normalized)


@dataclass(frozen=True)
class PhysicalRoot:
    """One complete scheduler-visible physical state root.

    Immutable prefix blocks may be shared by multiple roots.  Every resource in
    ``writable_*`` and every recurrent/position/decoder slot is branch-private.
    """

    root_id: str
    request_id: str
    allocator_generation: int
    target_blocks: tuple[tuple[int, ...], ...]
    draft_blocks: tuple[tuple[int, ...], ...]
    writable_target_blocks: tuple[tuple[int, ...], ...]
    writable_draft_blocks: tuple[tuple[int, ...], ...]
    gdn_slots: tuple[int, ...]
    convolution_slots: tuple[int, ...]
    position_state_id: str
    decoding_state_id: str

    def __post_init__(self) -> None:
        _identifier(self.root_id, "root ID")
        _identifier(self.request_id, "root request ID")
        if (
            isinstance(self.allocator_generation, bool)
            or not isinstance(self.allocator_generation, int)
            or self.allocator_generation < 0
        ):
            raise SchedulerAuthorityError("allocator generation is invalid")
        target = _block_groups(self.target_blocks, "target blocks")
        draft = _block_groups(self.draft_blocks, "draft blocks")
        if len(target) != validation.TARGET_KV_LAYER_COUNT:
            raise SchedulerAuthorityError("target block group count differs")
        if len(draft) != validation.DRAFT_LAYER_COUNT:
            raise SchedulerAuthorityError("draft block group count differs")
        writable_target = _block_groups(
            self.writable_target_blocks, "writable target blocks"
        )
        writable_draft = _block_groups(
            self.writable_draft_blocks, "writable draft blocks"
        )
        if len(target) != len(writable_target) or len(draft) != len(writable_draft):
            raise SchedulerAuthorityError("writable block group topology differs")
        for label, blocks, writable in (
            ("target", target, writable_target),
            ("draft", draft, writable_draft),
        ):
            for index, (group, write_group) in enumerate(
                zip(blocks, writable, strict=True)
            ):
                if not set(write_group).issubset(group):
                    raise SchedulerAuthorityError(
                        f"writable {label} blocks[{index}] are outside the root"
                    )
        if len(self.gdn_slots) != validation.GDN_LAYER_COUNT:
            raise SchedulerAuthorityError("GDN slot count differs")
        if len(self.convolution_slots) != validation.GDN_LAYER_COUNT:
            raise SchedulerAuthorityError("convolution slot count differs")
        if self.gdn_slots != self.convolution_slots:
            raise SchedulerAuthorityError(
                "GDN and convolution states do not use the same physical slots"
            )
        for label, values in (
            ("GDN slot", self.gdn_slots),
            ("convolution slot", self.convolution_slots),
        ):
            if len(values) != len(set(values)):
                raise SchedulerAuthorityError(f"{label}s are not unique")
            for value in values:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise SchedulerAuthorityError(f"{label} is invalid")
        _identifier(self.position_state_id, "position state ID")
        _identifier(self.decoding_state_id, "decoding state ID")

    @property
    def all_resource_ids(self) -> frozenset[str]:
        resources = {
            *(
                f"target:{group}:{block}"
                for group, blocks in enumerate(self.target_blocks)
                for block in blocks
            ),
            *(
                f"draft:{group}:{block}"
                for group, blocks in enumerate(self.draft_blocks)
                for block in blocks
            ),
            *(f"gdn:{slot}" for slot in self.gdn_slots),
            *(f"conv:{slot}" for slot in self.convolution_slots),
            f"position:{self.position_state_id}",
            f"decoding:{self.decoding_state_id}",
        }
        return frozenset(resources)

    @property
    def writable_resource_ids(self) -> frozenset[str]:
        resources = {
            *(
                f"target:{group}:{block}"
                for group, blocks in enumerate(self.writable_target_blocks)
                for block in blocks
            ),
            *(
                f"draft:{group}:{block}"
                for group, blocks in enumerate(self.writable_draft_blocks)
                for block in blocks
            ),
            *(f"gdn:{slot}" for slot in self.gdn_slots),
            *(f"conv:{slot}" for slot in self.convolution_slots),
            f"position:{self.position_state_id}",
            f"decoding:{self.decoding_state_id}",
        }
        return frozenset(resources)


@dataclass(frozen=True)
class CanonicalBinding:
    storage: PhysicalRoot
    state: dict[str, Any]
    state_sha256: str


@dataclass(frozen=True)
class RecordedTransition:
    role: BranchRole
    transition: PrivateTransition
    state_sha256: str
    model_output_sha256: str | None


@dataclass(frozen=True)
class OpenRound:
    request_id: str
    round_index: int
    base_root_id: str
    base_state_sha256: str
    allocator_generation: int
    candidate_root: PhysicalRoot
    serial_root: PhysicalRoot
    candidate_result: RecordedTransition | None = None
    serial_result: RecordedTransition | None = None


@dataclass(frozen=True)
class VerifiedAdvance:
    request_id: str
    round_index: int
    root_id: str
    token_ids: tuple[int, ...]
    publication_intent_sha256: str
    model_output_sha256: str
    model_output: object


@dataclass(frozen=True)
class AuthorityState:
    canonical: CanonicalBinding
    spare_roots: tuple[PhysicalRoot, PhysicalRoot]
    open_round: OpenRound | None
    pending_advance: VerifiedAdvance | None


@dataclass(frozen=True)
class AuthorityPublication:
    state: dict[str, Any]
    atomic: bool
    root_id: str
    selected_role: BranchRole
    publication_intent_sha256: str


def _binding(storage: PhysicalRoot, state: object) -> CanonicalBinding:
    normalized = validation.normalize_state(state, "scheduler canonical state")
    if normalized["allocator_generation"] != storage.allocator_generation:
        raise SchedulerAuthorityError(
            "canonical state allocator generation differs from physical root"
        )
    return CanonicalBinding(storage, normalized, validation.state_sha256(normalized))


def _validate_root_set(roots: tuple[PhysicalRoot, ...]) -> None:
    if len({root.root_id for root in roots}) != len(roots):
        raise SchedulerAuthorityError("physical root IDs are not unique")
    requests = {root.request_id for root in roots}
    generations = {root.allocator_generation for root in roots}
    if len(requests) != 1 or len(generations) != 1:
        raise SchedulerAuthorityError("physical roots do not share one request/generation")
    for index, left in enumerate(roots):
        for right in roots[index + 1 :]:
            if left.writable_resource_ids & right.all_resource_ids:
                raise SchedulerAuthorityError(
                    f"writable resources from {left.root_id} alias {right.root_id}"
                )
            if right.writable_resource_ids & left.all_resource_ids:
                raise SchedulerAuthorityError(
                    f"writable resources from {right.root_id} alias {left.root_id}"
                )


class SchedulerRootAuthority:
    """One request's non-bypassable scheduler publication state machine."""

    def __init__(
        self,
        canonical_root: PhysicalRoot,
        canonical_state: object,
        candidate_spare: PhysicalRoot,
        serial_spare: PhysicalRoot,
        *,
        model_output_sha256: Callable[[object], str],
        model_output_token_ids: Callable[[object, str], tuple[int, ...]],
        prepare_private_roots: Callable[
            [PhysicalRoot, PhysicalRoot, PhysicalRoot], None
        ],
        capture_complete_state: Callable[[PhysicalRoot], object],
        prepare_atomic_publication: Callable[[str, str], None],
    ) -> None:
        roots = (canonical_root, candidate_spare, serial_spare)
        _validate_root_set(roots)
        self._state = AuthorityState(
            canonical=_binding(canonical_root, canonical_state),
            spare_roots=(candidate_spare, serial_spare),
            open_round=None,
            pending_advance=None,
        )
        self._lock = threading.Lock()
        self._model_output_sha256 = model_output_sha256
        self._model_output_token_ids = model_output_token_ids
        self._prepare_private_roots = prepare_private_roots
        self._capture_complete_state = capture_complete_state
        self._prepare_atomic_publication = prepare_atomic_publication
        if (
            not callable(prepare_private_roots)
            or not callable(capture_complete_state)
            or not callable(prepare_atomic_publication)
        ):
            raise SchedulerAuthorityError("physical-root runtime callbacks are invalid")
        if self._capture_root(canonical_root, "initial canonical") != self._state.canonical:
            raise SchedulerAuthorityError(
                "initial canonical state differs from its complete physical root"
            )

    def _capture_root(self, root: PhysicalRoot, label: str) -> CanonicalBinding:
        try:
            state = self._capture_complete_state(root)
        except BaseException as error:
            raise SchedulerAuthorityError(
                f"{label} physical root could not be captured"
            ) from error
        try:
            return _binding(root, state)
        except SchedulerAuthorityError:
            raise
        except BaseException as error:
            raise SchedulerAuthorityError(f"{label} physical root is invalid") from error

    def _require_canonical_unchanged(self, label: str) -> CanonicalBinding:
        observed = self._capture_root(self._state.canonical.storage, label)
        if observed != self._state.canonical:
            raise SchedulerAuthorityError(f"{label} physical canonical state changed")
        return observed

    def _fingerprint_model_output(self, model_output: object) -> str:
        try:
            digest = self._model_output_sha256(model_output)
        except BaseException as error:
            raise SchedulerAuthorityError(
                "worker model output could not be fingerprinted"
            ) from error
        try:
            return validation._sha256(digest, "worker model output digest")
        except validation.LiveCommitValidationError as error:
            raise SchedulerAuthorityError(
                "worker model output fingerprint is invalid"
            ) from error

    def _tokens_from_model_output(
        self, model_output: object, request_id: str
    ) -> tuple[int, ...]:
        try:
            token_ids = self._model_output_token_ids(model_output, request_id)
        except BaseException as error:
            raise SchedulerAuthorityError(
                "worker model output tokens could not be extracted"
            ) from error
        if (
            type(token_ids) is not tuple
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in token_ids
            )
        ):
            raise SchedulerAuthorityError("worker model output tokens are invalid")
        return token_ids

    @property
    def state(self) -> AuthorityState:
        return self._state

    def snapshot_canonical(self) -> CanonicalBinding:
        with self._lock:
            return self._require_canonical_unchanged("snapshot")

    def probe_canonical(self) -> dict[str, Any]:
        with self._lock:
            return self._require_canonical_unchanged("probe").state

    def begin_round(self, *, request_id: str, round_index: int) -> OpenRound:
        with self._lock:
            current = self._state
            if current.open_round is not None:
                raise SchedulerAuthorityError("a scheduler provisional round is already open")
            if current.pending_advance is not None:
                raise SchedulerAuthorityError(
                    "verified output must advance the request before another round"
                )
            if request_id != current.canonical.storage.request_id:
                raise SchedulerAuthorityError("provisional request ID differs")
            if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
                raise SchedulerAuthorityError("round index is invalid")
            candidate, serial = current.spare_roots
            _validate_root_set((current.canonical.storage, candidate, serial))
            try:
                self._prepare_private_roots(
                    current.canonical.storage, candidate, serial
                )
            except BaseException as error:
                self._require_canonical_unchanged("failed private-root preparation")
                raise SchedulerAuthorityError(
                    "private physical roots could not be prepared"
                ) from error
            canonical_after_prepare = self._require_canonical_unchanged(
                "private-root preparation"
            )
            for role, root in (("candidate", candidate), ("serial", serial)):
                prepared = self._capture_root(root, f"prepared {role}")
                if prepared.state != canonical_after_prepare.state:
                    raise SchedulerAuthorityError(
                        f"prepared {role} root differs from canonical state"
                    )
            opened = OpenRound(
                request_id=request_id,
                round_index=round_index,
                base_root_id=current.canonical.storage.root_id,
                base_state_sha256=current.canonical.state_sha256,
                allocator_generation=current.canonical.storage.allocator_generation,
                candidate_root=candidate,
                serial_root=serial,
            )
            self._state = replace(current, open_round=opened)
            return opened

    def record_private_transition(
        self, role: BranchRole, transition: PrivateTransition
    ) -> None:
        with self._lock:
            current = self._state
            opened = current.open_round
            if opened is None:
                raise SchedulerAuthorityError("private transition arrived without an open round")
            if role not in ("candidate", "serial"):
                raise SchedulerAuthorityError("private transition role is invalid")
            expected_root = (
                opened.candidate_root if role == "candidate" else opened.serial_root
            )
            if transition.handle_id != expected_root.root_id:
                raise SchedulerAuthorityError("private transition handle differs from its root")
            normalized = validation.normalize_state(
                transition.state, f"{role} scheduler transition"
            )
            physical = self._capture_root(expected_root, f"recorded {role}")
            if physical.state != normalized:
                raise SchedulerAuthorityError(
                    f"{role} transition differs from its complete physical root"
                )
            self._require_canonical_unchanged(f"{role} execution")
            if normalized["allocator_generation"] != opened.allocator_generation:
                raise SchedulerAuthorityError(
                    "allocator generation changed during a provisional round"
                )
            commit_count = len(transition.token_ids)
            if commit_count and transition.model_output is None:
                raise SchedulerAuthorityError(
                    "private transition lacks its worker model output"
                )
            model_output_sha256 = (
                self._fingerprint_model_output(transition.model_output)
                if transition.model_output is not None
                else None
            )
            if transition.model_output is not None and self._tokens_from_model_output(
                transition.model_output, opened.request_id
            ) != transition.token_ids:
                raise SchedulerAuthorityError(
                    "private transition tokens differ from its worker model output"
                )
            expected_epoch = (
                current.canonical.state["commit_epoch"]
                if commit_count == 0
                else current.canonical.state["commit_epoch"] + 1
            )
            if normalized["commit_epoch"] != expected_epoch:
                raise SchedulerAuthorityError("private transition commit epoch differs")
            if normalized["logical_length"] != (
                current.canonical.state["logical_length"] + commit_count
            ):
                raise SchedulerAuthorityError("private transition logical length differs")
            recorded = RecordedTransition(
                role,
                transition,
                validation.state_sha256(normalized),
                model_output_sha256,
            )
            if role == "candidate":
                if opened.candidate_result is not None:
                    raise SchedulerAuthorityError("candidate result was recorded twice")
                opened = replace(opened, candidate_result=recorded)
            else:
                if opened.serial_result is not None:
                    raise SchedulerAuthorityError("serial result was recorded twice")
                opened = replace(opened, serial_result=recorded)
            self._state = replace(current, open_round=opened)

    def abort_round(self, *, request_id: str, round_index: int) -> None:
        with self._lock:
            current = self._state
            opened = current.open_round
            if opened is None:
                raise SchedulerAuthorityError("no scheduler provisional round is open")
            if (request_id, round_index) != (opened.request_id, opened.round_index):
                raise SchedulerAuthorityError("provisional abort identity differs")
            if (
                current.canonical.state_sha256 != opened.base_state_sha256
                or self._capture_root(
                    current.canonical.storage, "provisional abort"
                ).state_sha256
                != opened.base_state_sha256
            ):
                raise SchedulerAuthorityError(
                    "canonical state changed during provisional execution"
                )
            self._state = replace(current, open_round=None)

    def publish(
        self,
        *,
        snapshot_root_id: str,
        transition: PrivateTransition,
        publication_intent_sha256: str,
    ) -> AuthorityPublication:
        with self._lock:
            current = self._state
            opened = current.open_round
            if opened is None:
                raise SchedulerAuthorityError("publication has no open scheduler round")
            if current.pending_advance is not None:
                raise SchedulerAuthorityError("a verified scheduler advance is already pending")
            if snapshot_root_id != opened.base_root_id:
                raise SchedulerAuthorityError("publication snapshot root differs")
            intent = validation._sha256(
                publication_intent_sha256, "scheduler publication intent"
            )
            recorded_by_role = {
                "candidate": opened.candidate_result,
                "serial": opened.serial_result,
            }
            selected_role: BranchRole
            if transition.handle_id == opened.candidate_root.root_id:
                selected_role = "candidate"
            elif transition.handle_id == opened.serial_root.root_id:
                selected_role = "serial"
            else:
                raise SchedulerAuthorityError("selected transition is outside the open roots")
            recorded = recorded_by_role[selected_role]
            if recorded is None:
                raise SchedulerAuthorityError("selected transition was not recorded by scheduler")
            if (
                recorded.transition.token_ids != transition.token_ids
                or recorded.state_sha256 != validation.state_sha256(transition.state)
                or recorded.transition.model_output is not transition.model_output
            ):
                raise SchedulerAuthorityError("selected transition differs from recorded result")
            if (
                transition.model_output is None
                or recorded.model_output_sha256 is None
                or self._fingerprint_model_output(transition.model_output)
                != recorded.model_output_sha256
            ):
                raise SchedulerAuthorityError(
                    "selected worker model output changed after recording"
                )
            if (
                self._tokens_from_model_output(transition.model_output, opened.request_id)
                != transition.token_ids
            ):
                raise SchedulerAuthorityError(
                    "selected worker model output tokens changed after recording"
                )
            if not transition.token_ids:
                raise SchedulerAuthorityError(
                    "zero-commit transitions must not swap physical roots"
                )
            selected_storage = (
                opened.candidate_root
                if selected_role == "candidate"
                else opened.serial_root
            )
            unselected_storage = (
                opened.serial_root
                if selected_role == "candidate"
                else opened.candidate_root
            )
            if (
                self._capture_root(current.canonical.storage, "publication").state_sha256
                != opened.base_state_sha256
            ):
                raise SchedulerAuthorityError(
                    "canonical physical state changed before publication"
                )
            selected_physical = self._capture_root(
                selected_storage, "selected publication"
            )
            if selected_physical.state != validation.normalize_state(
                transition.state, "selected publication transition"
            ):
                raise SchedulerAuthorityError(
                    "selected transition differs from its complete physical root"
                )
            selected_binding = _binding(selected_storage, transition.state)
            advance = VerifiedAdvance(
                request_id=opened.request_id,
                round_index=opened.round_index,
                root_id=selected_storage.root_id,
                token_ids=transition.token_ids,
                publication_intent_sha256=intent,
                model_output_sha256=recorded.model_output_sha256,
                model_output=transition.model_output,
            )
            next_state = AuthorityState(
                canonical=selected_binding,
                spare_roots=(current.canonical.storage, unselected_storage),
                open_round=None,
                pending_advance=advance,
            )
            publication = AuthorityPublication(
                state=selected_binding.state,
                atomic=True,
                root_id=selected_storage.root_id,
                selected_role=selected_role,
                publication_intent_sha256=intent,
            )
            try:
                self._prepare_atomic_publication(
                    current.canonical.storage.root_id, selected_storage.root_id
                )
            except BaseException as error:
                self._require_canonical_unchanged("failed publication preparation")
                raise SchedulerAuthorityError(
                    "physical publication could not be prepared"
                ) from error
            # All validation, selected scratch restoration, synchronization and
            # result construction precede this sole composite authority write.
            # Bound block-table proxies resolve through this same immutable
            # AuthorityState pointer. No potentially failing operation follows it.
            self._state = next_state
            return publication

    def consume_verified_advance(
        self,
        *,
        request_id: str,
        round_index: int,
        publication_intent_sha256: str,
        expected_token_ids: tuple[int, ...],
    ) -> VerifiedVllmAdvance:
        with self._lock:
            current = self._state
            advance = current.pending_advance
            if advance is None:
                raise SchedulerAuthorityError("request advance lacks a verified publication")
            if (
                request_id,
                round_index,
                publication_intent_sha256,
                expected_token_ids,
            ) != (
                advance.request_id,
                advance.round_index,
                advance.publication_intent_sha256,
                advance.token_ids,
            ):
                raise SchedulerAuthorityError("request advance identity differs")
            if advance.model_output is None:
                raise SchedulerAuthorityError("request advance lacks a private model output")
            if self._fingerprint_model_output(advance.model_output) != advance.model_output_sha256:
                raise SchedulerAuthorityError(
                    "request advance model output changed after publication"
                )
            if (
                self._tokens_from_model_output(advance.model_output, advance.request_id)
                != advance.token_ids
            ):
                raise SchedulerAuthorityError(
                    "request advance model output tokens changed after publication"
                )
            result = VerifiedVllmAdvance(
                request_id=advance.request_id,
                round_index=advance.round_index,
                publication_intent_sha256=advance.publication_intent_sha256,
                token_ids=advance.token_ids,
                model_output_sha256=advance.model_output_sha256,
                model_output=advance.model_output,
            )
            self._state = replace(current, pending_advance=None)
            return result
