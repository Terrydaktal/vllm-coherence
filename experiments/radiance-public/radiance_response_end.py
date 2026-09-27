"""Exact processed-response checkpoints for the synchronous DFlash chat banks.

No kernel arithmetic changes. One immutable endpoint per bank pins attention
blocks and a canonical recurrent/convolution state. The ordinary partial-hit
copy-on-write path gives a successor private writable tails. Prefix identity
uses the existing salted hash chain plus exact token IDs in the partial block;
those IDs stay in memory and are never put in telemetry.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EndBoundary:
    tokens: int
    source_column: int
    accepted_offset: int


def end_boundary(*, start, scheduled, draft, generated, visible, block_size):
    """Resolve the processed boundary, including EOS/length stops within M8.

    `generated` is the sampler count BEFORE scheduler stop truncation. The
    last correction/bonus is pending, not processed. Alignment postprocessing
    can overwrite an earlier convolution window when it canonicalizes in place;
    that exceptional truncated boundary must not be advertised as reusable.
    """
    if not (
        start >= 0
        and scheduled > 0
        and 0 <= draft < scheduled
        and 1 <= generated <= draft + 1
        and block_size > 0
    ):
        return None
    base = start + scheduled - draft
    computed = base + generated - 1
    end = min(computed, visible)
    if end < base:
        return None
    source = (start + scheduled - 1) // block_size
    aligned = computed // block_size * block_size
    offset = end - base
    if aligned >= base and source == aligned // block_size - 1:
        if end != aligned:
            return None
        offset = 0
    return EndBoundary(end, source, offset)


def prefix_identity(request, count, hash_size):
    """Identity is exact within the existing prefix-hash collision contract."""
    if count <= 0 or count > request.num_tokens:
        return None
    full, partial = divmod(count, hash_size)
    if full > len(request.block_hashes):
        return None
    return (
        request.cache_salt,
        request.block_hashes[full - 1] if full else None,
        tuple(request.all_token_ids[full * hash_size : full * hash_size + partial]),
    )


class ResponseEndCache:
    def __init__(self, manager):
        self.manager = manager
        self.entry = None
        self.pending = []
        self.copy_pins = []
        self.captured = 0
        self.hits = 0

    def clear(self):
        if self.entry is not None:
            self.manager.block_pool.free_blocks(self.entry["pins"])
            self.entry = None

    def remember(self, request, step):
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            MambaSpec,
            SlidingWindowSpec,
        )

        manager = self.manager
        if (
            step is None
            or request.num_in_flight_tokens
            or request.mm_features
            or getattr(request, "prompt_embeds", None) is not None
            or request.lora_request is not None
            or request.num_output_tokens == 0
            or request.status.name not in {"FINISHED_STOPPED", "FINISHED_LENGTH_CAPPED"}
            or not manager.prefix_cache_lookup_enabled(request)
        ):
            return False
        boundary = end_boundary(
            **step,
            visible=request.num_tokens,
            block_size=manager.coordinator.scheduler_block_size,
        )
        if boundary is None or boundary.tokens > request.num_computed_tokens:
            return False
        count = boundary.tokens
        token_identity = prefix_identity(
            request, count, manager.block_pool.hash_block_size
        )
        if token_identity is None:
            return False
        # Acceptance can cross a physical boundary on the terminal round, after
        # allocate_slots registered the previously known committed prefix.
        manager.cache_blocks(request, count)
        managers = manager.coordinator.single_type_managers
        if any(
            not isinstance(
                m.kv_cache_spec, (FullAttentionSpec, MambaSpec, SlidingWindowSpec)
            )
            or m.block_size != manager.coordinator.scheduler_block_size
            for m in managers
        ):
            return False
        recurrent = [m for m in managers if isinstance(m.kv_cache_spec, MambaSpec)]
        if any(m.mamba_cache_mode != "align" for m in recurrent):
            return False
        # Never evict a still-valid checkpoint to attempt an allocation that
        # cannot fit. This is an acceleration, not a new admission requirement.
        pool = manager.block_pool
        if pool.get_num_free_blocks() < len(recurrent):
            return False
        columns = (
            count + manager.coordinator.scheduler_block_size - 1
        ) // manager.coordinator.scheduler_block_size
        original = [list(m.req_to_blocks.get(request.request_id, ())) for m in managers]
        for m, blocks in zip(managers, original, strict=True):
            if isinstance(m.kv_cache_spec, MambaSpec):
                last = boundary.source_column + boundary.accepted_offset
                if (
                    last >= len(blocks)
                    or blocks[last].is_null
                    or blocks[boundary.source_column].is_null
                ):
                    return False
            elif len(blocks) < columns or blocks[columns - 1].is_null:
                return False
        # The endpoint pins blocks that would otherwise be freed immediately.
        # Canonical recurrent blocks have their own ownership and never replace
        # an aligned checkpoint while a connector could still be reading it.
        fresh = pool.get_new_blocks(len(recurrent))
        allocations = iter(fresh)
        result, copies, source_pins = [], [], []
        null = pool.null_block
        for group, (m, blocks) in enumerate(zip(managers, original, strict=True)):
            if isinstance(m.kv_cache_spec, MambaSpec):
                destination = next(allocations)
                conv = blocks[boundary.source_column]
                state = blocks[boundary.source_column + boundary.accepted_offset]
                result.append([null] * (columns - 1) + [destination])
                copies.append(
                    {
                        "group": group,
                        "conv": conv.block_id,
                        "state": state.block_id,
                        "destination": destination.block_id,
                        "offset": boundary.accepted_offset,
                    }
                )
                source_pins.extend((conv, state))
            else:
                result.append(blocks[:columns])
        inherited = {
            b.block_id: b
            for blocks in result
            for b in blocks
            if not b.is_null and b not in fresh
        }
        sources = {b.block_id: b for b in source_pins if not b.is_null}
        pool.touch(tuple(inherited.values()))
        pool.touch(tuple(sources.values()))
        # A prefix-cache reset may drop the endpoint before its next worker
        # frame. Keep copy destinations alive independently until that frame
        # completes, just like its source blocks.
        pool.touch(fresh)
        self.clear()
        self.entry = {
            "tokens": count,
            "identity": token_identity,
            "blocks": tuple(result),
            "pins": [*inherited.values(), *fresh],
        }
        request.qwen_response_end = {
            "tokens": count,
            "block_ids": [blocks[-1].block_id for blocks in result],
        }
        self.pending.extend(copies)
        self.copy_pins.extend([*sources.values(), *fresh])
        self.captured += 1
        return True

    def lookup(self, request):
        request.qwen_response_end_local = 0
        request._qwen_response_end_lease = None
        entry = self.entry
        if (
            entry is None
            or entry["tokens"] >= request.num_tokens
            or request.mm_features
            or getattr(request, "prompt_embeds", None) is not None
            or request.lora_request is not None
            or not self.manager.prefix_cache_lookup_enabled(request)
            or prefix_identity(
                request, entry["tokens"], self.manager.block_pool.hash_block_size
            )
            != entry["identity"]
        ):
            return None
        self.hits += 1
        request.qwen_response_end_local = entry["tokens"]
        request._qwen_response_end_lease = entry
        return (
            self.manager.create_kv_cache_blocks(entry["blocks"]),
            entry["tokens"],
            0,
            False,
        )

    def release_after_progress(self, request):
        """Under pressure, transfer ownership to an advanced continuation.

        allocate_slots has already pinned the reused pages and any CoW source
        and destination. Once its worker step completes, the old canonical
        state can be reclaimed if capacity is exhausted. Until then keep it
        available for cancellation/retry. Near the context limit retaining it
        can cause preempt/replay to revisit the same endpoint indefinitely. This
        drops only the endpoint's extra references, never the request's or a
        pending copy's references. Durable snapshots remain untouched.
        """
        entry = getattr(request, "_qwen_response_end_lease", None)
        if (entry is None or request.num_in_flight_tokens
                or request.num_computed_tokens <= entry["tokens"]):
            return False
        request._qwen_response_end_lease = None
        # A terminal step may already have installed a newer checkpoint.
        if self.entry is not entry:
            return False
        self.clear()
        return True

    def take_copies(self):
        copies, self.pending = self.pending, []
        return copies

    def copies_complete(self):
        if self.copy_pins:
            self.manager.block_pool.free_blocks(self.copy_pins)
            self.copy_pins = []


def reclaim_snapshot_history(manager, request, protected=()):
    """Under pressure, surrender optional old GDN snapshots, never live state.

    The snapshot overlay retains three extra aligned boundaries beyond vLLM's
    numerical working set. They are useful for backup, but cannot be allowed to
    force recomputation of the response itself. Use the native processed-token
    boundary, and exclude every page an offload transfer is still reading.
    """
    from vllm.v1.kv_cache_interface import MambaSpec

    if request.num_in_flight_tokens or request.num_computed_tokens <= 0:
        return 0
    pool = manager.block_pool
    before = pool.get_num_free_blocks()
    protected = set(protected)
    for group in manager.coordinator.single_type_managers:
        if not isinstance(group.kv_cache_spec, MambaSpec) or group.mamba_cache_mode != "align":
            continue
        blocks = group.req_to_blocks.get(request.request_id, ())
        # The block containing the last committed token and every speculative
        # successor remain untouched, including at an exact physical boundary.
        stop = min(len(blocks), (request.num_computed_tokens - 1) // group.block_size)
        released = []
        for i in range(stop - 1, -1, -1):
            block = blocks[i]
            if not block.is_null and block.block_id not in protected:
                released.append(block)
                blocks[i] = pool.null_block
        pool.free_blocks(released)
    return pool.get_num_free_blocks() - before


def copy_response_end(runner, copies):
    """Canonicalize only accepted state, on the current HIP stream, once/response.

    Copy tensor slices, never recompute the state. Supports both convolution
    layouts. Prefix-cache CoW will later copy this immutable canonical block.
    """
    if not copies:
        return
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    context = runner.vllm_config.compilation_config.static_forward_context
    dim_first = is_conv_state_dim_first()
    for item in copies:
        group = runner.kv_cache_config.kv_cache_groups[item["group"]]
        for name in group.layer_names:
            conv, state = context[name].kv_cache
            src, dst = conv[item["conv"]], conv[item["destination"]]
            offset = item["offset"]
            dst.zero_()
            if dim_first:
                dst[..., : src.shape[-1] - offset].copy_(src[..., offset:])
            else:
                dst[: src.shape[0] - offset].copy_(src[offset:])
            state[item["destination"]].copy_(state[item["state"]])
