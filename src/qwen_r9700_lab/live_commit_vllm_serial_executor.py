"""Hash-bound independent serial proxy for the real vLLM executor."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from qwen_r9700_lab import live_commit_validation as validation
from qwen_r9700_lab.live_commit_controller import CanonicalSnapshot, PrivateTransition
from qwen_r9700_lab.live_commit_vllm_worker_protocol import (
    PreparedVllmRound,
    PrivateWorkerProtocolError,
    PrivateWorkerResult,
)

_WORKER_METHOD = "qwen_live_execute_private_serial"


class SerialExecutorError(RuntimeError):
    """The serial oracle cannot produce an authenticated private result."""


class SerialExecutorProxy:
    """Invoke only the separately generated M1 worker path."""

    def __init__(self, base_executor: object, capability: object) -> None:
        verified = validation.verify_capability(capability)
        source = Path(__file__).resolve(strict=True)
        declared = Path(verified["serial_oracle_runtime"]["path"])
        try:
            same_file = source.samefile(declared)
        except OSError as error:
            raise SerialExecutorError("serial executor path cannot be authenticated") from error
        if (
            not same_file
            or hashlib.sha256(source.read_bytes()).hexdigest()
            != verified["serial_oracle_runtime"]["sha256"]
        ):
            raise SerialExecutorError("serial executor identity differs")
        if isinstance(base_executor, SerialExecutorProxy) or not callable(
            getattr(base_executor, "collective_rpc", None)
        ):
            raise SerialExecutorError("serial base executor is invalid")
        self._base_executor = base_executor
        self.qwen_live_serial_runtime_sha256 = verified["serial_oracle_runtime"]["sha256"]
        self.qwen_live_implementation_family = verified["serial_implementation_family"]

    @staticmethod
    def _snapshot_sha256(snapshot: CanonicalSnapshot) -> str:
        try:
            return validation.state_sha256(snapshot.state)
        except validation.LiveCommitValidationError as error:
            raise SerialExecutorError("serial snapshot state is invalid") from error

    def qwen_live_execute_serial(
        self,
        snapshot: CanonicalSnapshot,
        request_payload: object,
        commit_count: int,
    ) -> PrivateTransition:
        if type(snapshot) is not CanonicalSnapshot:
            raise SerialExecutorError("serial snapshot type is invalid")
        if commit_count not in validation.COMMIT_COUNTS:
            raise SerialExecutorError("serial commit width is invalid")
        if type(request_payload) is not PreparedVllmRound:
            raise SerialExecutorError("serial request payload type is invalid")
        try:
            prepared = request_payload.validate(
                snapshot_handle_id=snapshot.handle_id,
                snapshot_state_sha256=self._snapshot_sha256(snapshot),
            )
        except PrivateWorkerProtocolError as error:
            raise SerialExecutorError("serial prepared round is invalid") from error
        branch = prepared.serial
        if branch.requested_commit_count not in (None, commit_count):
            raise SerialExecutorError("serial requested width differs")
        # Live speculative acceptance determines the authoritative serial width
        # only after the candidate worker returns.  The scheduler's serial
        # payload is therefore a generic immutable plan; specialize only this
        # explicit width field while preserving its payload digest and nonce.
        if branch.requested_commit_count is None:
            branch = replace(branch, requested_commit_count=commit_count)
        try:
            results = self._base_executor.collective_rpc(
                _WORKER_METHOD,
                args=(branch,),
            )
        except BaseException as error:
            raise SerialExecutorError("serial private worker RPC failed") from error
        if type(results) is not list or len(results) != 1:
            raise SerialExecutorError(
                "serial private worker RPC must return exactly one TP=1 result"
            )
        result = results[0]
        if type(result) is not PrivateWorkerResult:
            raise SerialExecutorError("serial private worker result type is invalid")
        try:
            result.validate(
                branch,
                implementation_family=self.qwen_live_implementation_family,
            )
        except PrivateWorkerProtocolError as error:
            raise SerialExecutorError("serial private worker result is invalid") from error
        return PrivateTransition(
            result.branch_handle_id,
            result.implementation_family,
            result.token_ids,
            result.state,
            result.model_output,
        )
