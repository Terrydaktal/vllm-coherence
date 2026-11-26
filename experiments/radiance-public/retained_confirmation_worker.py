"""Run the existing 320-position diagnostic workers with the shared candidate hooks."""

import hashlib
import json
from pathlib import Path

from optimized_d7_worker import OptimizedWorker
from rotary_mode_d7_worker import RotaryRneWorker
from speed_confirmation_worker import SpeedTapeWorker
from speed_matched_stage_worker import SpeedMatchedStageWorker


class RetainedHooks:
    def load_model(self, **kwargs):
        from extended_context_prefill import install
        import fourteen_retained
        from host_embedding import _offload_target, host_embedding_during_load

        window = self.model_config.max_model_len
        install(window)
        fourteen_retained.before_load(self)
        if self.vllm_config.speculative_config is None:
            # M1 has no drafter to share the table. Move the identical target
            # parameter with the same release helper prior to graph capture.
            result = super().load_model(**kwargs)
            _, table_bytes, released, _ = _offload_target(self.model_runner.model)
            if released < table_bytes:
                raise ValueError("M1 host mapping returned an unexpected allocation size")
        else:
            with host_embedding_during_load(self):
                result = super().load_model(**kwargs)
        fourteen_retained.after_load(self)
        self._retained_diagnostic = {
            "window": window,
            "configuration": dict(self._fourteen_prepared),
            "installed": self._fourteen_results,
            "initializer_sha256": hashlib.sha256(Path(fourteen_retained.__file__).read_bytes()).hexdigest(),
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        return result

    def qwen_optimized_metadata(self):
        result = super().qwen_optimized_metadata()
        # Runtime row histograms use integer shape keys. Freeze them in their
        # JSON representation before the outer recorder computes the digest.
        result["retained_qualification"] = json.loads(json.dumps(self._retained_diagnostic))
        return result


class CompiledM1(RetainedHooks, OptimizedWorker):
    pass


class CompiledM8(RetainedHooks, SpeedMatchedStageWorker):
    pass


class Eager(RetainedHooks, RotaryRneWorker):
    pass


class Stages(RetainedHooks, SpeedTapeWorker):
    def qwen_optimized_begin(self, *args, **kwargs):
        import native_d7_tape_worker as native
        from retained_tape_dispatch import RetainedTapeDispatch, projection_fallbacks

        result = super().qwen_optimized_begin(*args, **kwargs)
        fallbacks = projection_fallbacks(self.model_runner.model)
        self._tape_observer.hooks.replace(
            native, "TapeDispatch", lambda tape: RetainedTapeDispatch(tape, fallbacks),
        )
        return result
