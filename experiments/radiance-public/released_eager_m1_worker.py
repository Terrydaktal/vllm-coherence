"""Diagnostic M1 using only the installed release's target arithmetic and host table."""

import hashlib
from pathlib import Path

from rotary_mode_d7_worker import RotaryRneWorker


class ReleasedEagerM1(RotaryRneWorker):
    def load_model(self, **kwargs):
        import extended_context_prefill
        import host_embedding

        if self.vllm_config.speculative_config is not None:
            raise ValueError("released eager M1 requires one target row and no drafter")
        context = extended_context_prefill.install(self.model_config.max_model_len)
        result = super().load_model(**kwargs)
        _, table_bytes, released, _ = host_embedding._offload_target(self.model_runner.model)
        if released < table_bytes:
            raise ValueError("released host embedding returned an unexpected allocation size")
        self._release_control = {
            "window": self.model_config.max_model_len,
            "context": context,
            "embedding_bytes": table_bytes,
            "module_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in map(Path, (__file__, host_embedding.__file__, extended_context_prefill.__file__))
            },
        }
        return result

    def qwen_optimized_metadata(self):
        metadata = super().qwen_optimized_metadata()
        metadata["released_eager_m1"] = self._release_control
        return metadata
