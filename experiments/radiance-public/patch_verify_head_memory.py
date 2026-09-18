"""Bound the upstream verify-head liveness check and report activation failures."""

from __future__ import annotations

from pathlib import Path

LIVE_WEIGHT = """def _qwen_has_live_weight(weight):
    # The complete BF16 head is 2.37 GiB on TP1. Checking it after KV allocation
    # must not allocate another head-sized abs() tensor on every decode step.
    # Preserve the exact all-zero test, including a nonzero final chunk.
    for start in range(0, weight.shape[0], 1024):
        if float(weight[start:start + 1024].detach().abs().max()) != 0.0:
            return True
    return False


"""

OLD_GUARD = "w is None or w.dim() != 2 or float(w.data.abs().max()) == 0.0"
NEW_GUARD = "w is None or w.dim() != 2 or not _qwen_has_live_weight(w)"
OLD_HOOK = """        try:
            import radiance_verifyhead as _radiance_vh
            _radiance_vh.before_compute_logits(self, input_batch, grammar_output)
        except Exception:
            pass
"""
NEW_HOOK = """        try:
            import radiance_verifyhead as _radiance_vh
            _radiance_vh.before_compute_logits(self, input_batch, grammar_output)
        except Exception:
            # Report once and keep the exact head usable. Repeated silent OOM
            # retries would otherwise drain the allocator on every model step.
            logger.exception("Radiance verify head failed; disabled for this process")
            if '_radiance_vh' in locals():
                _radiance_vh.ENABLED = False
                _lp = _radiance_vh._state.get('lp')
                if _lp is not None:
                    _lp._radiance_fast_ok = False
"""


def install(package: Path):
    head = package / "radiance_verifyhead.py"
    runner = package / "vllm/v1/worker/gpu/model_runner.py"
    head_text, runner_text = head.read_text(), runner.read_text()
    if LIVE_WEIGHT not in head_text:
        if head_text.count(OLD_GUARD) != 1 or head_text.count("def _arm(model):") != 1:
            raise ValueError("verify-head memory patch source changed")
        head_text = head_text.replace("def _arm(model):", LIVE_WEIGHT + "def _arm(model):")
        head_text = head_text.replace(OLD_GUARD, NEW_GUARD)
    if NEW_HOOK not in runner_text:
        if runner_text.count(OLD_HOOK) != 1:
            raise ValueError("verify-head runner hook changed")
        runner_text = runner_text.replace(OLD_HOOK, NEW_HOOK)
    compile(head_text, str(head), "exec")
    compile(runner_text, str(runner), "exec")
    head.write_text(head_text)
    runner.write_text(runner_text)
