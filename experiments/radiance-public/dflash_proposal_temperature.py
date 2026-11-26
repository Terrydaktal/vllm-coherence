"""Change only the DFlash2 proposal law before query graph capture."""

import math
from types import MethodType

import torch
import triton
import triton.language as tl
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax


@triton.jit
def _walk(
    scores,
    ids,
    positions,
    mapping,
    temperatures,
    seeds,
    tokens,
    cached,
    proposal_temperature,
    STEPS: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    FP64: tl.constexpr,
):
    row = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    slot = tl.load(mapping + row * STEPS)
    valid = (slot >= 0) & (lanes < K)
    target_t = tl.load(temperatures + slot, slot >= 0, other=0.0)
    proposal_t = tl.load(proposal_temperature)
    seed = tl.load(seeds + slot, slot >= 0, other=0)
    previous = 0
    for step in range(STEPS):
        flat = row * STEPS + step
        base = flat * K
        values = tl.load(
            scores + (base + previous) * K + lanes, valid, other=-float("inf")
        ).to(tl.float32)
        # Store exactly the FP32 logits used for the walk. The existing verifier
        # divides the cached values by the request temperature independently.
        if proposal_t > 0.0 and target_t > 0.0 and target_t != proposal_t:
            values = values * (target_t / proposal_t)
        candidates = tl.load(ids + base + lanes, valid, other=0)
        pos = tl.load(positions + flat) - 1 + (1 << 30)
        _, previous = gumbel_noised_argmax(
            values.to(tl.float64 if FP64 else tl.float32),
            candidates,
            valid,
            seed,
            pos,
            target_t,
            USE_FP64=FP64,
        )
        tl.store(cached + base + lanes, values, valid)
        tl.store(
            tokens + flat, tl.load(ids + base + previous, slot >= 0, other=0), slot >= 0
        )


def install(runner):
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

    spec = runner.speculator
    if type(spec) is not DFlash2Speculator or spec.draft_logits is None:
        raise ValueError("proposal calibration requires probabilistic DFlash2")
    setting = torch.zeros(1, device=spec.draft_tokens.device, dtype=torch.float32)

    def sample_path(self, candidate_ids, scores, num_reqs):
        _walk[(num_reqs,)](
            scores.contiguous(),
            candidate_ids.contiguous(),
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            self.draft_tokens,
            self._selector_scores,
            setting,
            self.num_speculative_steps,
            self.selector_top_k,
            triton.next_power_of_2(self.selector_top_k),
            self.use_fp64_gumbel,
            num_warps=1,
        )

    spec._sample_path = MethodType(sample_path, spec)

    def set_temperature(value):
        if not math.isfinite(value) or value < 0 or value > 100:
            raise ValueError(
                "expected zero (control) or positive proposal temperature <= 100"
            )
        setting.fill_(value)

    return set_temperature
