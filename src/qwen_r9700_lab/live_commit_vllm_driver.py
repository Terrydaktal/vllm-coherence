"""Authenticated adapter between vLLM's EngineCore and live-commit controller.

EngineCore is the transition coordinator because it owns the real
``schedule -> execute_model -> update_from_output`` boundary.  Its Scheduler is
the sole canonical-state authority: it owns immutable request roots, private
branch allocation and atomic publication.  EngineCore's real ``model_executor``
is the candidate executor; the serial executor is distinct and independently
authenticated.  Neither worker nor EngineCore may publish canonical state.
Missing, transplanted, monkey-patched, aliased, or identity-mismatched hooks
abort startup before a canonical transition can run.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen_r9700_lab import live_commit_validation as validation
from qwen_r9700_lab.live_commit_controller import (
    CanonicalSnapshot,
    PrivateTransition,
    Publication,
    PublicationRecovery,
)

SCHEDULER_HOOKS = (
    "qwen_live_begin_provisional_round",
    "qwen_live_abort_provisional_round",
    "qwen_live_prepare_private_roots",
    "qwen_live_capture_complete_state",
    "qwen_live_snapshot_canonical",
    "qwen_live_probe_canonical",
    "qwen_live_record_private_transition",
    "qwen_live_publish_private",
    "qwen_live_restore_canonical",
    "qwen_live_destroy_private",
    "qwen_live_cleanup_failed_private",
    "qwen_live_recover_publication",
    "qwen_live_model_output_sha256",
    "qwen_live_model_output_token_ids",
    "qwen_live_consume_verified_advance",
)
COORDINATOR_HOOKS = ("qwen_live_execute_serial_validated_round",)
CANDIDATE_EXECUTOR_ATTRIBUTE = "model_executor"
CANDIDATE_EXECUTOR_HOOK = "qwen_live_execute_candidate"
SERIAL_EXECUTOR_ATTRIBUTE = "qwen_live_serial_executor"
SERIAL_EXECUTOR_HOOK = "qwen_live_execute_serial"


class VllmLiveCommitDriverError(RuntimeError):
    """The exact vLLM core cannot enforce the serial-before-commit contract."""


@dataclass(frozen=True)
class VerifiedVllmAdvance:
    """Scheduler-owned one-shot handoff to ``update_from_output``."""

    request_id: str
    round_index: int
    publication_intent_sha256: str
    token_ids: tuple[int, ...]
    model_output_sha256: str
    model_output: object


class VllmLiveTransitionDriver:
    """Adapt an authenticated EngineCore, Scheduler, and two worker executors."""

    def __init__(self, engine_core: Any, capability: object) -> None:
        self.capability = validation.verify_capability(capability)
        self.engine_core = engine_core
        self._verify_adapter_identity()
        self.scheduler = self._verify_coordinator_identity()
        self._verify_scheduler_identity()
        self._scheduler_hooks = {
            name: getattr(self.scheduler, name) for name in SCHEDULER_HOOKS
        }
        self.candidate_executor = self._verify_candidate_executor_identity()
        self.serial_executor = self._verify_serial_executor_identity()
        self._candidate_execute = getattr(
            self.candidate_executor, CANDIDATE_EXECUTOR_HOOK
        )
        self._serial_execute = getattr(self.serial_executor, SERIAL_EXECUTOR_HOOK)
        self.release_runtime_sha256 = self.capability["release_runtime"]["sha256"]
        self.serial_runtime_sha256 = self.capability["serial_oracle_runtime"]["sha256"]
        self.candidate_implementation_family = self.capability[
            "candidate_implementation_family"
        ]
        self.serial_implementation_family = self.capability[
            "serial_implementation_family"
        ]

    def _verify_adapter_identity(self) -> None:
        source = Path(__file__).resolve(strict=True)
        declared = Path(self.capability["driver_runtime"]["path"])
        try:
            same_file = source.samefile(declared)
        except OSError as error:
            raise VllmLiveCommitDriverError(
                "vLLM live-commit driver path cannot be authenticated"
            ) from error
        if (
            not same_file
            or hashlib.sha256(source.read_bytes()).hexdigest()
            != self.capability["driver_runtime"]["sha256"]
        ):
            raise VllmLiveCommitDriverError(
                "vLLM live-commit driver identity differs from capability"
            )

    def _verify_coordinator_identity(self) -> Any:
        coordinator_class = type(self.engine_core)
        source_raw = inspect.getsourcefile(coordinator_class)
        if source_raw is None:
            raise VllmLiveCommitDriverError("vLLM EngineCore has no source identity")
        source = Path(source_raw).resolve(strict=True)
        declared = Path(self.capability["coordinator_runtime"]["path"])
        try:
            same_file = source.samefile(declared)
        except OSError as error:
            raise VllmLiveCommitDriverError(
                "vLLM EngineCore path cannot be authenticated"
            ) from error
        if (
            not same_file
            or hashlib.sha256(source.read_bytes()).hexdigest()
            != self.capability["coordinator_runtime"]["sha256"]
        ):
            raise VllmLiveCommitDriverError(
                "vLLM EngineCore identity differs from capability"
            )
        expected_identity = {
            "coordinator_runtime_sha256": self.capability["coordinator_runtime"][
                "sha256"
            ],
            "scheduler_runtime_sha256": self.capability["scheduler_runtime"]["sha256"],
            "model_output_fingerprint_runtime_sha256": self.capability[
                "model_output_fingerprint_runtime"
            ]["sha256"],
            "release_runtime_sha256": self.capability["release_runtime"]["sha256"],
            "serial_runtime_sha256": self.capability["serial_oracle_runtime"]["sha256"],
            "candidate_implementation_family": self.capability[
                "candidate_implementation_family"
            ],
            "serial_implementation_family": self.capability[
                "serial_implementation_family"
            ],
        }
        observed_identity = {
            key: getattr(self.engine_core, f"qwen_live_{key}", None)
            for key in expected_identity
        }
        if observed_identity != expected_identity:
            raise VllmLiveCommitDriverError(
                "vLLM EngineCore implementation identity differs from capability"
            )
        for name in COORDINATOR_HOOKS:
            class_hook = coordinator_class.__dict__.get(name)
            bound_hook = getattr(self.engine_core, name, None)
            if (
                not inspect.isfunction(class_hook)
                or not inspect.ismethod(bound_hook)
                or bound_hook.__func__ is not class_hook
                or inspect.getsourcefile(class_hook) is None
                or not Path(inspect.getsourcefile(class_hook)).resolve(strict=True).samefile(
                    source
                )
                or class_hook.__qualname__
                != f"{coordinator_class.__qualname__}.{name}"
            ):
                raise VllmLiveCommitDriverError(
                    f"vLLM EngineCore hook is not defined by the coordinator runtime: {name}"
                )
        for name in SCHEDULER_HOOKS:
            if callable(getattr(self.engine_core, name, None)):
                raise VllmLiveCommitDriverError(
                    "vLLM EngineCore exposes a canonical-state authority hook"
                )
        scheduler = getattr(self.engine_core, "scheduler", None)
        if scheduler is None or scheduler is self.engine_core:
            raise VllmLiveCommitDriverError(
                "vLLM Scheduler is absent or aliases EngineCore"
            )
        return scheduler

    def _verify_scheduler_identity(self) -> None:
        scheduler_class = type(self.scheduler)
        source_raw = inspect.getsourcefile(scheduler_class)
        if source_raw is None:
            raise VllmLiveCommitDriverError("vLLM scheduler has no source identity")
        source = Path(source_raw).resolve(strict=True)
        declared = Path(self.capability["scheduler_runtime"]["path"])
        try:
            same_file = source.samefile(declared)
        except OSError as error:
            raise VllmLiveCommitDriverError(
                "vLLM scheduler path cannot be authenticated"
            ) from error
        if (
            not same_file
            or hashlib.sha256(source.read_bytes()).hexdigest()
            != self.capability["scheduler_runtime"]["sha256"]
        ):
            raise VllmLiveCommitDriverError(
                "vLLM scheduler identity differs from capability"
            )
        expected_identity = {
            "scheduler_runtime_sha256": self.capability["scheduler_runtime"]["sha256"],
            "model_output_fingerprint_runtime_sha256": self.capability[
                "model_output_fingerprint_runtime"
            ]["sha256"],
            "release_runtime_sha256": self.capability["release_runtime"]["sha256"],
            "serial_runtime_sha256": self.capability["serial_oracle_runtime"]["sha256"],
            "candidate_implementation_family": self.capability[
                "candidate_implementation_family"
            ],
            "serial_implementation_family": self.capability[
                "serial_implementation_family"
            ],
        }
        observed_identity = {
            key: getattr(self.scheduler, f"qwen_live_{key}", None)
            for key in expected_identity
        }
        if observed_identity != expected_identity:
            raise VllmLiveCommitDriverError(
                "vLLM scheduler/worker implementation identity differs from capability"
            )
        for name in SCHEDULER_HOOKS:
            class_hook = scheduler_class.__dict__.get(name)
            bound_hook = getattr(self.scheduler, name, None)
            if (
                not inspect.isfunction(class_hook)
                or not inspect.ismethod(bound_hook)
                or bound_hook.__func__ is not class_hook
            ):
                raise VllmLiveCommitDriverError(
                    f"vLLM scheduler hook is missing or instance-patched: {name}"
                )
            hook_source_raw = inspect.getsourcefile(class_hook)
            if hook_source_raw is None:
                raise VllmLiveCommitDriverError(
                    f"vLLM scheduler hook has no source identity: {name}"
                )
            hook_source = Path(hook_source_raw).resolve(strict=True)
            try:
                hook_in_scheduler = hook_source.samefile(source)
            except OSError as error:
                raise VllmLiveCommitDriverError(
                    f"vLLM scheduler hook source cannot be authenticated: {name}"
                ) from error
            if (
                not hook_in_scheduler
                or class_hook.__qualname__ != f"{scheduler_class.__qualname__}.{name}"
            ):
                raise VllmLiveCommitDriverError(
                    f"vLLM scheduler hook is not defined by the scheduler runtime: {name}"
                )

        for forbidden in (CANDIDATE_EXECUTOR_HOOK, SERIAL_EXECUTOR_HOOK):
            if callable(getattr(self.scheduler, forbidden, None)):
                raise VllmLiveCommitDriverError(
                    "vLLM scheduler aliases a worker execution implementation"
                )

    def _verify_candidate_executor_identity(self) -> Any:
        executor = getattr(self.engine_core, CANDIDATE_EXECUTOR_ATTRIBUTE, None)
        if (
            executor is None
            or executor is self.engine_core
            or executor is self.scheduler
        ):
            raise VllmLiveCommitDriverError(
                "vLLM EngineCore model_executor is absent or aliases a state authority"
            )
        executor_class = type(executor)
        source_raw = inspect.getsourcefile(executor_class)
        if source_raw is None:
            raise VllmLiveCommitDriverError("vLLM candidate executor has no source identity")
        source = Path(source_raw).resolve(strict=True)
        declared = Path(self.capability["release_runtime"]["path"])
        try:
            same_file = source.samefile(declared)
        except OSError as error:
            raise VllmLiveCommitDriverError(
                "vLLM candidate executor source cannot be authenticated"
            ) from error
        if (
            not same_file
            or hashlib.sha256(source.read_bytes()).hexdigest()
            != self.capability["release_runtime"]["sha256"]
        ):
            raise VllmLiveCommitDriverError(
                "vLLM candidate executor identity differs from capability"
            )
        hook = executor_class.__dict__.get(CANDIDATE_EXECUTOR_HOOK)
        bound_hook = getattr(executor, CANDIDATE_EXECUTOR_HOOK, None)
        if (
            not inspect.isfunction(hook)
            or not inspect.ismethod(bound_hook)
            or bound_hook.__func__ is not hook
            or inspect.getsourcefile(hook) is None
            or not Path(inspect.getsourcefile(hook)).resolve(strict=True).samefile(source)
            or hook.__qualname__
            != f"{executor_class.__qualname__}.{CANDIDATE_EXECUTOR_HOOK}"
        ):
            raise VllmLiveCommitDriverError(
                "vLLM candidate execution hook is not defined by the release runtime"
            )
        if getattr(executor, "qwen_live_release_runtime_sha256", None) != self.capability[
            "release_runtime"
        ]["sha256"]:
            raise VllmLiveCommitDriverError(
                "vLLM candidate executor runtime identity differs from capability"
            )
        if getattr(executor, "qwen_live_implementation_family", None) != self.capability[
            "candidate_implementation_family"
        ]:
            raise VllmLiveCommitDriverError(
                "vLLM candidate executor implementation family differs from capability"
            )
        self._reject_worker_publication_hooks(executor)
        return executor

    @staticmethod
    def _reject_worker_publication_hooks(executor: Any) -> None:
        for name in SCHEDULER_HOOKS:
            if callable(getattr(executor, name, None)):
                raise VllmLiveCommitDriverError(
                    "vLLM worker executor exposes a canonical-state authority hook"
                )

    def _verify_serial_executor_identity(self) -> Any:
        executor = getattr(self.engine_core, SERIAL_EXECUTOR_ATTRIBUTE, None)
        if (
            executor is None
            or executor is self.engine_core
            or executor is self.scheduler
        ):
            raise VllmLiveCommitDriverError(
                "vLLM serial executor is absent or aliases a state authority"
            )
        if executor is self.candidate_executor:
            raise VllmLiveCommitDriverError(
                "vLLM serial executor aliases the candidate executor"
            )
        executor_class = type(executor)
        source_raw = inspect.getsourcefile(executor_class)
        if source_raw is None:
            raise VllmLiveCommitDriverError("vLLM serial executor has no source identity")
        source = Path(source_raw).resolve(strict=True)
        declared = Path(self.capability["serial_oracle_runtime"]["path"])
        try:
            same_file = source.samefile(declared)
        except OSError as error:
            raise VllmLiveCommitDriverError(
                "vLLM serial executor source cannot be authenticated"
            ) from error
        if (
            not same_file
            or hashlib.sha256(source.read_bytes()).hexdigest()
            != self.capability["serial_oracle_runtime"]["sha256"]
        ):
            raise VllmLiveCommitDriverError(
                "vLLM serial executor identity differs from capability"
            )
        hook = executor_class.__dict__.get(SERIAL_EXECUTOR_HOOK)
        bound_hook = getattr(executor, SERIAL_EXECUTOR_HOOK, None)
        if (
            not inspect.isfunction(hook)
            or not inspect.ismethod(bound_hook)
            or bound_hook.__func__ is not hook
            or inspect.getsourcefile(hook) is None
            or not Path(inspect.getsourcefile(hook)).resolve(strict=True).samefile(source)
            or hook.__qualname__
            != f"{executor_class.__qualname__}.{SERIAL_EXECUTOR_HOOK}"
        ):
            raise VllmLiveCommitDriverError(
                "vLLM serial execution hook is not defined by the serial runtime"
            )
        if getattr(executor, "qwen_live_serial_runtime_sha256", None) != self.capability[
            "serial_oracle_runtime"
        ]["sha256"]:
            raise VllmLiveCommitDriverError(
                "vLLM serial executor runtime identity differs from capability"
            )
        if getattr(executor, "qwen_live_implementation_family", None) != self.capability[
            "serial_implementation_family"
        ]:
            raise VllmLiveCommitDriverError(
                "vLLM serial executor implementation family differs from capability"
            )
        self._reject_worker_publication_hooks(executor)
        return executor

    def snapshot_canonical(self) -> CanonicalSnapshot:
        return self._scheduler_hooks["qwen_live_snapshot_canonical"]()

    def begin_provisional_round(
        self,
        snapshot: CanonicalSnapshot,
        request_payload: object,
        request_id: str,
        round_index: int,
    ) -> object:
        prepared_payload = self._scheduler_hooks[
            "qwen_live_begin_provisional_round"
        ](
            snapshot, request_payload, request_id, round_index
        )
        if prepared_payload is None:
            raise VllmLiveCommitDriverError(
                "vLLM Scheduler did not return its private round payload"
            )
        return prepared_payload

    def abort_provisional_round(self, request_id: str, round_index: int) -> None:
        self._scheduler_hooks["qwen_live_abort_provisional_round"](
            request_id, round_index
        )

    def probe_canonical(self) -> dict[str, Any]:
        return self._scheduler_hooks["qwen_live_probe_canonical"]()

    def execute_candidate(
        self,
        snapshot: CanonicalSnapshot,
        request_payload: object,
        commit_count: int | None,
    ) -> PrivateTransition:
        transition = self._candidate_execute(
            snapshot, request_payload, commit_count
        )
        if not isinstance(transition, PrivateTransition) or (
            transition.token_ids and transition.model_output is None
        ):
            raise VllmLiveCommitDriverError(
                "vLLM candidate executor did not preserve its private model output"
            )
        self._scheduler_hooks["qwen_live_record_private_transition"](
            "candidate", transition
        )
        return transition

    def execute_serial(
        self,
        snapshot: CanonicalSnapshot,
        request_payload: object,
        commit_count: int,
    ) -> PrivateTransition:
        transition = self._serial_execute(
            snapshot, request_payload, commit_count
        )
        if not isinstance(transition, PrivateTransition) or (
            commit_count and transition.model_output is None
        ):
            raise VllmLiveCommitDriverError(
                "vLLM serial executor did not preserve its private model output"
            )
        self._scheduler_hooks["qwen_live_record_private_transition"]("serial", transition)
        return transition

    def publish_private(
        self,
        snapshot: CanonicalSnapshot,
        transition: PrivateTransition,
        publication_intent_sha256: str,
    ) -> Publication:
        validation._sha256(publication_intent_sha256, "publication intent")
        return self._scheduler_hooks["qwen_live_publish_private"](
            snapshot, transition, publication_intent_sha256
        )

    def restore_canonical(self, snapshot: CanonicalSnapshot) -> Publication:
        return self._scheduler_hooks["qwen_live_restore_canonical"](snapshot)

    def destroy_private(self, transition: PrivateTransition) -> None:
        self._scheduler_hooks["qwen_live_destroy_private"](transition)

    def cleanup_failed_private(
        self,
        role: str,
        session_id: str,
        request_id: str,
        round_index: int,
    ) -> None:
        self._scheduler_hooks["qwen_live_cleanup_failed_private"](
            role, session_id, request_id, round_index
        )

    def recover_publication(self, intent: dict[str, Any]) -> PublicationRecovery:
        return self._scheduler_hooks["qwen_live_recover_publication"](intent)

    def consume_verified_advance(
        self,
        request_id: str,
        round_index: int,
        publication_intent_sha256: str,
        expected_token_ids: tuple[int, ...],
    ) -> VerifiedVllmAdvance:
        validation._sha256(publication_intent_sha256, "publication intent")
        if (
            not expected_token_ids
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in expected_token_ids
            )
        ):
            raise VllmLiveCommitDriverError("expected request-advance tokens are invalid")
        advance = self._scheduler_hooks["qwen_live_consume_verified_advance"](
            request_id,
            round_index,
            publication_intent_sha256,
            expected_token_ids,
        )
        if type(advance) is not VerifiedVllmAdvance:
            raise VllmLiveCommitDriverError(
                "vLLM scheduler returned an unauthenticated request advance"
            )
        if (
            advance.request_id != request_id
            or advance.round_index != round_index
            or advance.publication_intent_sha256 != publication_intent_sha256
            or advance.token_ids != expected_token_ids
        ):
            raise VllmLiveCommitDriverError(
                "vLLM scheduler request advance identity differs"
            )
        if (
            not advance.token_ids
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in advance.token_ids
            )
            or advance.model_output is None
        ):
            raise VllmLiveCommitDriverError(
                "vLLM scheduler request advance payload is invalid"
            )
        observed_output_sha256 = self._scheduler_hooks[
            "qwen_live_model_output_sha256"
        ](advance.model_output)
        if (
            validation._sha256(
                advance.model_output_sha256, "verified model output digest"
            )
            != validation._sha256(
                observed_output_sha256, "observed model output digest"
            )
        ):
            raise VllmLiveCommitDriverError(
                "vLLM scheduler request advance model output changed"
            )
        observed_token_ids = self._scheduler_hooks[
            "qwen_live_model_output_token_ids"
        ](advance.model_output, request_id)
        if observed_token_ids != expected_token_ids:
            raise VllmLiveCommitDriverError(
                "vLLM scheduler request advance model output tokens differ"
            )
        return advance
