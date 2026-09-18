"""Hash-bound candidate proxy for the real vLLM executor.

The proxy remains ``EngineCore.model_executor`` so ordinary lifecycle RPCs keep
working, but candidate execution can only enter the GPU worker through the
private candidate method and an authenticated ``PreparedVllmRound``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from qwen_r9700_lab import live_commit_validation as validation
from qwen_r9700_lab.live_commit_controller import CanonicalSnapshot, PrivateTransition
from qwen_r9700_lab.live_commit_vllm_worker_protocol import (
    PreparedVllmRound,
    PrivateWorkerProtocolError,
    PrivateWorkerResult,
)

_WORKER_METHOD = "qwen_live_execute_private_candidate"


class CandidateExecutorError(RuntimeError):
    """The optimized worker cannot produce an authenticated private result."""


class CandidateExecutorProxy:
    """Delegate lifecycle methods while isolating optimized branch execution."""

    def __init__(self, base_executor: object, capability: object) -> None:
        verified = validation.verify_capability(capability)
        source = Path(__file__).resolve(strict=True)
        declared = Path(verified["release_runtime"]["path"])
        try:
            same_file = source.samefile(declared)
        except OSError as error:
            raise CandidateExecutorError(
                "candidate executor path cannot be authenticated"
            ) from error
        if (
            not same_file
            or hashlib.sha256(source.read_bytes()).hexdigest()
            != verified["release_runtime"]["sha256"]
        ):
            raise CandidateExecutorError("candidate executor identity differs")
        if isinstance(base_executor, CandidateExecutorProxy) or not callable(
            getattr(base_executor, "collective_rpc", None)
        ):
            raise CandidateExecutorError("candidate base executor is invalid")
        self._base_executor = base_executor
        self.qwen_live_release_runtime_sha256 = verified["release_runtime"]["sha256"]
        self.qwen_live_implementation_family = verified[
            "candidate_implementation_family"
        ]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("qwen_live_"):
            raise AttributeError(name)
        return getattr(self._base_executor, name)

    @staticmethod
    def _snapshot_sha256(snapshot: CanonicalSnapshot) -> str:
        try:
            return validation.state_sha256(snapshot.state)
        except validation.LiveCommitValidationError as error:
            raise CandidateExecutorError("candidate snapshot state is invalid") from error

    def qwen_live_execute_candidate(
        self,
        snapshot: CanonicalSnapshot,
        request_payload: object,
        commit_count: int | None,
    ) -> PrivateTransition:
        if type(snapshot) is not CanonicalSnapshot:
            raise CandidateExecutorError("candidate snapshot type is invalid")
        if type(request_payload) is not PreparedVllmRound:
            raise CandidateExecutorError("candidate request payload type is invalid")
        try:
            prepared = request_payload.validate(
                snapshot_handle_id=snapshot.handle_id,
                snapshot_state_sha256=self._snapshot_sha256(snapshot),
            )
        except PrivateWorkerProtocolError as error:
            raise CandidateExecutorError("candidate prepared round is invalid") from error
        branch = prepared.candidate
        if commit_count is not None and branch.requested_commit_count != commit_count:
            raise CandidateExecutorError("candidate requested width differs")
        try:
            results = self._base_executor.collective_rpc(
                _WORKER_METHOD,
                args=(branch,),
            )
        except BaseException as error:
            raise CandidateExecutorError("candidate private worker RPC failed") from error
        if type(results) is not list or len(results) != 1:
            raise CandidateExecutorError(
                "candidate private worker RPC must return exactly one TP=1 result"
            )
        result = results[0]
        if type(result) is not PrivateWorkerResult:
            raise CandidateExecutorError("candidate private worker result type is invalid")
        try:
            result.validate(
                branch,
                implementation_family=self.qwen_live_implementation_family,
            )
        except PrivateWorkerProtocolError as error:
            raise CandidateExecutorError("candidate private worker result is invalid") from error
        return PrivateTransition(
            result.branch_handle_id,
            result.implementation_family,
            result.token_ids,
            result.state,
            result.model_output,
        )
