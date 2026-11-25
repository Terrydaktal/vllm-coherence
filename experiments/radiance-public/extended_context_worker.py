"""Extend the authenticated FP8 lane's native rotary positions and host table."""

import json
import os

from extended_context_prefill import install as install_prefill_context
from host_embedding import host_embedding_during_load
from speed_candidate_worker import SpeedCandidateWorker

DECODE_ROWS = 8


def rotary_buffer_rows(model, minimum):
    buffers = [
        module.cos_sin_cache.shape[0]
        for module in model.modules()
        if hasattr(module, "cos_sin_cache")
    ]
    if not buffers or min(buffers) < minimum:
        raise RuntimeError(f"rotary cache cannot address {minimum} positions")
    return sorted(set(buffers))


class ExtendedContextWorker(SpeedCandidateWorker):
    def load_model(self, **kwargs):
        config = self.vllm_config
        draft = config.speculative_config
        window = config.model_config.max_model_len
        expected = os.environ.get("QWEN_EXTENDED_CONTEXT_TOKENS", "319000")
        if (
            str(window) != expected
            or draft is None
            or draft.method != "dflash"
            or draft.num_speculative_tokens != DECODE_ROWS - 1
            or draft.draft_model_config.max_model_len < window
        ):
            raise RuntimeError("extended context requires the pinned FP8 DFlash profile")
        binding = install_prefill_context(window)
        with host_embedding_during_load(self):
            super().load_model(**kwargs)

        target = self.model_runner.model
        if hasattr(target, "get_language_model"):
            target = target.get_language_model()
        drafter = self.model_runner.speculator.model
        receipt = {
            "window": window,
            "draft_window": draft.draft_model_config.max_model_len,
            "target_rotary_rows": rotary_buffer_rows(target, window + DECODE_ROWS),
            "draft_rotary_rows": rotary_buffer_rows(drafter, window + DECODE_ROWS),
            "prefill_binding": binding,
        }
        print("Extended context binding: " + json.dumps(receipt, sort_keys=True), flush=True)
