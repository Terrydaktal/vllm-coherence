"""Native GPU distribution test; synthetic probabilities, no model or chat data."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import textwrap
import time
from pathlib import Path

PROPOSAL_STREAM_OFFSET = 1 << 30


def selector_stream_offset(source: str) -> int:
    """Recognize the two reviewed selector counter expressions, or fail closed.

    The repaired selector adds the proposal stream salt itself. Merely passing
    its ordinary positions therefore cannot exercise the shared-noise defect.
    This inspection adjusts the *probe inputs*, never the installed kernel.
    """
    tree = ast.parse(textwrap.dedent(source))
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "position" for target in node.targets)
    ]
    if len(assignments) != 1 or len(assignments[0].targets) != 1:
        raise ValueError("unreviewed selector position assignment")
    expression = ast.dump(assignments[0].value, include_attributes=False)
    for offset, spelling in (
        (0, "tl.load(sample_pos_ptr + flat) - 1"),
        (PROPOSAL_STREAM_OFFSET, "tl.load(sample_pos_ptr + flat) - 1 + (1 << 30)"),
    ):
        expected = ast.dump(ast.parse(spelling, mode="eval").body, include_attributes=False)
        if expression == expected:
            return offset
    raise ValueError("unreviewed selector position expression")


def proposal_input_offset(internal_offset: int, independent: bool) -> int:
    if internal_offset not in (0, PROPOSAL_STREAM_OFFSET):
        raise ValueError("unreviewed selector stream offset")
    desired = PROPOSAL_STREAM_OFFSET if independent else 0
    return desired - internal_offset


def run(output: Path, samples: int):
    import torch
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import _selector_walk_kernel
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

    selector_source = inspect.getsource(_selector_walk_kernel.fn)
    internal_offset = selector_stream_offset(selector_source)
    device = "cuda"
    target = torch.tensor([0.1, 0.5, 0.4], device=device)
    draft = torch.tensor([0.5, 0.3, 0.2], device=device)
    n = samples
    scores = draft.log().view(1, 1, 1, 3).expand(n, 1, 3, 3).contiguous()
    candidate = (
        torch.arange(3, device=device, dtype=torch.int64).view(1, 1, 3).expand(n, 1, 3).contiguous()
    )
    state = torch.arange(n, device=device, dtype=torch.int32)
    temperature = torch.ones(n, device=device, dtype=torch.float32)
    seeds = torch.arange(n, device=device, dtype=torch.int64) + 1234567
    logits = target.log().view(1, 3).expand(2 * n, 3).contiguous()
    expanded_state = state.repeat_interleave(2)
    local_pos = torch.tensor([0, 1], device=device, dtype=torch.int32).repeat(n)
    cu_logits = torch.arange(0, 2 * n + 1, 2, device=device, dtype=torch.int32)
    tokens = torch.empty((n, 1), device=device, dtype=torch.int64)
    realized = torch.empty((n, 1, 3), device=device, dtype=torch.float32)
    rows = []
    native_rows = []
    started = time.monotonic()
    for position in (37, 86142, 132739):
        sample_pos = torch.full((n,), position + 1, device=device, dtype=torch.int64)
        positions = torch.tensor([position, position + 1], device=device, dtype=torch.int64).repeat(
            n
        )
        native = gumbel_sample(
            logits[:n], state, temperature, seeds, sample_pos - 1, apply_temperature=True
        )
        native_observed = torch.bincount(native.long(), minlength=3).float() / n
        native_rows.append(
            {
                "position": position,
                "observed": native_observed.tolist(),
                "max_error": float((native_observed - target).abs().max()),
            }
        )
        for independent in (False, True):
            input_offset = proposal_input_offset(internal_offset, independent)
            proposal_positions = sample_pos + input_offset
            _selector_walk_kernel[(n,)](
                scores,
                candidate,
                proposal_positions,
                state,
                temperature,
                seeds,
                tokens,
                realized,
                num_steps=1,
                top_k=3,
                BLOCK_K=4,
                SAMPLE_PROBABILISTIC=True,
                USE_FP64=False,
                num_warps=1,
            )
            draft_inputs = torch.zeros((n, 2), device=device, dtype=torch.int64)
            draft_inputs[:, 1] = tokens[:, 0]
            sampled, counts = rejection_sample(
                logits,
                realized,
                draft_inputs.flatten(),
                cu_logits,
                positions,
                state,
                expanded_state,
                local_pos,
                temperature,
                seeds,
                1,
                use_fp64=False,
            )
            observed = torch.bincount(sampled[:, 0].long(), minlength=3).float() / n
            proposal = torch.bincount(tokens[:, 0], minlength=3).float() / n
            row = {
                "position": position,
                "independent_proposal_noise": independent,
                "selector_internal_offset": internal_offset,
                "proposal_input_offset": input_offset,
                "effective_proposal_stream_offset": input_offset + internal_offset,
                "samples": n,
                "target": target.tolist(),
                "draft": draft.tolist(),
                "observed": observed.tolist(),
                "proposal_observed": proposal.tolist(),
                "max_error": float((observed - target).abs().max()),
                "accepted_fraction": float((counts > 1).float().mean()),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    # Greedy decoding is unaffected by the proposal RNG stream.
    temperature.zero_()
    greedy = []
    for independent in (False, True):
        _selector_walk_kernel[(n,)](
            scores,
            candidate,
            sample_pos + proposal_input_offset(internal_offset, independent),
            state,
            temperature,
            seeds,
            tokens,
            realized,
            num_steps=1,
            top_k=3,
            BLOCK_K=4,
            SAMPLE_PROBABILISTIC=True,
            USE_FP64=False,
            num_warps=1,
        )
        draft_inputs[:, 1] = tokens[:, 0]
        sampled, _ = rejection_sample(
            logits,
            realized,
            draft_inputs.flatten(),
            cu_logits,
            positions,
            state,
            expanded_state,
            local_pos,
            temperature,
            seeds,
            1,
            use_fp64=False,
        )
        greedy.append(bool((sampled[:, 0] == 1).all()))
    package = Path("/opt/vllm/lib/python3.12/site-packages")
    sources = [
        "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py",
        "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py",
        "vllm/v1/worker/gpu/sample/gumbel.py",
    ]
    value = {
        "rows": rows,
        "native_target_rows": native_rows,
        "selector_source_sha256": hashlib.sha256(selector_source.encode()).hexdigest(),
        "greedy_exact": all(greedy),
        "seconds": time.monotonic() - started,
        "source_hashes": {
            name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in sources
        },
    }
    output.write_text(json.dumps(value, indent=2) + "\n")
    assert all(row["max_error"] < 0.004 for row in rows if row["independent_proposal_noise"])
    assert all(row["max_error"] > 0.01 for row in rows if not row["independent_proposal_noise"])
    assert all(greedy)
    assert all(row["max_error"] < 0.004 for row in native_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200000)
    args = parser.parse_args()
    run(args.output, args.samples)
