"""Untimed stage replay of the banked M8 GEMM against original M1/eager ops."""

import hashlib
from pathlib import Path

from native_d7_tape_worker import NativeTapeWorker
from speed_candidate_worker import SpeedCandidateWorker

from qwen_r9700_lab.conformance_instrumentation import HookSet


class SpeedTapeWorker(NativeTapeWorker):
    def load_model(self, **kwargs):
        result = super().load_model(**kwargs)
        owner, candidate = SpeedCandidateWorker._load_qualified_target_gemm(self)
        original = owner.launch
        self._speed_original_launch = original
        self._speed_launch_owner = owner
        # The released candidate is selected for the FULL M8 graph. Its M1 and
        # eager reference arms retain the original operator.
        def launch(*args):
            return (candidate if args[6] == 8 else original)(*args)
        self._speed_stage_hooks = HookSet()
        self._speed_stage_hooks.replace(owner, "launch", launch)
        return result

    def qwen_optimized_begin(self, *args, **kwargs):
        result = super().qwen_optimized_begin(*args, **kwargs)
        matrix = self._tape_observer.matrix
        if matrix is None:
            raise ValueError("banked candidate requires current-stage replay")
        original_cases = matrix.current_cases

        def reference(function):
            def run(*a, **kw):
                hooks = HookSet()
                try:
                    hooks.replace(self._speed_launch_owner, "launch", self._speed_original_launch)
                    return function(*a, **kw)
                finally:
                    hooks.close()
            return run

        def cases(call):
            for stage, instance, versions in original_cases(call):
                if call.name.startswith("radiance.mxfp4_linear"):
                    versions = {name: reference(fn) if name in {"fix1_m1", "final_eager_m8"} else fn
                                for name, fn in versions.items()}
                yield stage, instance, versions

        matrix.current_cases = cases
        return result

    def qwen_optimized_metadata(self):
        metadata = super().qwen_optimized_metadata()
        metadata["speed_stage_candidate"] = {
            "target_gemm": self._qwen_speed_target_gemm_candidate,
            "reference_operators": "original M1 and original eager M8",
            "scope": "compiled M8 operator replay; full graph correspondence checked separately",
            "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        return metadata
