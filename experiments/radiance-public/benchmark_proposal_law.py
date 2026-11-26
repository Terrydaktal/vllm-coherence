"""Check realized DFlash2 Q logits and target sampling after rejection at four temperatures."""

import json
import torch
from dflash_proposal_temperature import _walk
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample


def check(n=65536):
    device = "cuda"
    ids = torch.arange(n, device=device, dtype=torch.int32)
    seeds = ids.long() + 1234567
    pos = torch.full_like(seeds, 86243)
    q = torch.tensor([0.5, 0.3, 0.2], device=device)
    target = torch.tensor([0.1, 0.5, 0.4], device=device)
    logits = q.log().expand(n, 3, 3).contiguous()
    cids = torch.arange(3, device=device).expand(n, 3).contiguous()
    tokens = torch.empty_like(seeds)
    realized = torch.empty((n, 1, 3), device=device)
    cu = torch.arange(n + 1, device=device, dtype=torch.int32) * 2
    expanded = ids.repeat_interleave(2)
    local = torch.tensor([0, 1], dtype=torch.int32, device=device).repeat(n)
    targetpos = torch.stack((pos - 1, pos), dim=1).flatten()
    report = []
    for t in (0.7, 1.0):
        temps = torch.full((n,), t, device=device)
        for temperature in (0.0, 0.55, 0.7, 0.85, 1.0):
            setting = torch.tensor([temperature], device=device)
            _walk[(n,)](
                logits,
                cids,
                pos,
                ids,
                temps,
                seeds,
                tokens,
                realized,
                setting,
                1,
                3,
                4,
                False,
                num_warps=1,
            )
            q_effective = realized[0, 0] / t
            expected_q = (q.log() / (temperature or t)).softmax(0)
            torch.testing.assert_close(q_effective.softmax(0), expected_q)
            out, count = rejection_sample(
                target.log().expand(2 * n, 3).contiguous(),
                realized,
                torch.stack((ids.long() * 0, tokens), dim=1).flatten(),
                cu,
                targetpos,
                ids,
                expanded,
                local,
                temps,
                seeds,
                1,
                None,
                use_fp64=False,
                use_block_verification=False,
            )
            q_freq = tokens.bincount(minlength=3) / n
            p_freq = out[:, 0].bincount(minlength=3) / n
            # A conservative binomial finite-sample gate (six standard deviations).
            for empirical, expected in ((q_freq, expected_q), (p_freq, target)):
                if (
                    (empirical - expected).abs()
                    > 6 * (expected * (1 - expected) / n).sqrt() + 1 / n
                ).any():
                    raise AssertionError(
                        (
                            t,
                            temperature,
                            empirical.cpu().tolist(),
                            expected.cpu().tolist(),
                        )
                    )
            report.append(
                dict(
                    t=t,
                    q=temperature,
                    n=n,
                    p_observed=p_freq.tolist(),
                    q_observed=q_freq.tolist(),
                )
            )
    return report


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
