"""Offer seven deterministic proposals from a matching committed history suffix."""

from functools import wraps

MATCH_MAX = 24
ENTER = 8
STAY = 3
HOT = 3
BLOCK = 512
POSITION_BITS = 20
STEPS = 7


def lookup_source(tokens, window, minimum=ENTER):
    """CPU reference. Prefer the longest match, then the most recent source."""
    length = len(tokens)
    best = (0, 0)
    for source in range(minimum, min(length - window, length - STEPS) + 1):
        matched = 0
        while (
            matched < MATCH_MAX
            and matched < source
            and tokens[source - 1 - matched] == tokens[length - 1 - matched]
        ):
            matched += 1
        if matched >= minimum:
            best = max(best, (matched, source))
    return best


def install(runner):
    """Attach to the pinned, single-stream V2 DFlash2 path before warmup."""
    import dflash_prompt_lookup_kernels as kernels
    import torch
    import triton
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

    speculator = runner.speculator
    if (
        type(speculator) is not DFlash2Speculator
        or speculator.num_speculative_steps != STEPS
        or runner.vllm_config.parallel_config.tensor_parallel_size != 1
        or getattr(runner, "adaptive_verification", None) is not None
    ):
        raise ValueError(
            "prompt lookup requires fixed-width TP1 DFlash2 with one sequence"
        )
    config = speculator.draft_model_config.hf_config
    window = config.dflash_config.get("swa_window_size") or config.sliding_window
    if (
        type(window) is not int
        or window < STEPS
        or not config.layer_types
        or set(config.layer_types) != {"sliding_attention"}
    ):
        raise ValueError("prompt lookup requires a sliding-window DFlash2 drafter")

    history = runner.req_states.all_token_ids.gpu
    lengths = runner.req_states.total_len.gpu
    logits = speculator.draft_logits
    candidates = speculator._cached_candidate_ids
    top_k = speculator.selector_top_k
    if (
        history.ndim != 2
        or history.dtype != torch.int32
        or history.stride(-1) != 1
        or history.shape[1] >= 1 << POSITION_BITS
        or lengths.shape != (history.shape[0],)
        or lengths.dtype != torch.int32
        or logits is None
        or logits.ndim != 3
        or logits.shape[:2] != (history.shape[0], STEPS)
        or logits.dtype != torch.float32
        or logits.stride(-1) != 1
        or candidates.shape != (history.shape[0], STEPS, top_k)
        or candidates.dtype != torch.int64
        or not candidates.is_contiguous()
        or any(t.device != logits.device for t in (history, lengths, candidates))
    ):
        raise ValueError("unsupported DFlash2 history or probability cache")

    best = torch.empty(
        triton.cdiv(history.shape[1], BLOCK), dtype=torch.int32, device=logits.device
    )
    previous = torch.full((1,), -1, dtype=torch.int32, device=logits.device)
    stats = torch.zeros(3, dtype=torch.int32, device=logits.device)
    request_id = None
    original = speculator.propose
    @wraps(original)
    def propose(
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        *args,
        **kwargs,
    ):
        nonlocal request_id
        args = (
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            *args,
        )
        if kwargs.get("dummy_run") or kwargs.get("is_profile"):
            return original(*args, **kwargs)
        if input_batch.num_reqs != 1 or runner.adaptive_verification is not None:
            request_id = None
            return original(*args, **kwargs)

        bound = min(int(input_batch.seq_lens_cpu_upper_bound[0]) + 1, history.shape[1])
        same_request = input_batch.req_ids[0] == request_id
        if bound < window + STAY:
            request_id = None
            return original(*args, **kwargs)
        request_id = input_batch.req_ids[0]
        proposals = original(*args, **kwargs)
        if proposals.shape != (1, STEPS) or not proposals.is_contiguous():
            raise ValueError("unexpected DFlash2 proposal tensor")
        blocks = triton.cdiv(bound - window + 1, BLOCK)
        kernels.scan[(blocks,)](
            history,
            history.stride(0),
            lengths,
            input_batch.idx_mapping,
            num_sampled,
            best,
            window,
            STEPS,
            MATCH_MAX,
            POSITION_BITS,
            BLOCK,
        )
        kernels.apply[(1,)](
            history,
            history.stride(0),
            lengths,
            input_batch.idx_mapping,
            num_sampled,
            best,
            blocks,
            previous,
            stats,
            int(same_request),
            proposals,
            logits,
            logits.stride(0),
            logits.stride(1),
            candidates,
            top_k,
            STEPS,
            ENTER,
            STAY,
            HOT,
            POSITION_BITS,
            triton.next_power_of_2(blocks),
            triton.next_power_of_2(top_k),
            num_warps=4,
        )
        return proposals

    speculator.propose = propose
    # Content-free counters for a disposable benchmark: scanned, used, hot.
    return stats
