"""Exact processed-response checkpoints for the synchronous DFlash chat banks.

No kernel arithmetic changes. One immutable endpoint per bank pins attention
blocks and a canonical recurrent/convolution state. The ordinary partial-hit
copy-on-write path gives a successor private writable tails. Prefix identity
uses the existing salted hash chain plus exact token IDs in the partial block;
those IDs stay in memory and are never put in telemetry.
"""

import hashlib
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger("vllm.qwen_response_end")
DECISION_REASONS = frozenset(
    {
        "no_endpoint",
        "response_end_disabled",
        "prompt_does_not_extend_endpoint",
        "multimodal_request",
        "prompt_embeddings",
        "lora_request",
        "prefix_cache_disabled",
        "prefix_identity_unavailable",
        "cache_salt_changed",
        "prefix_hash_changed",
        "partial_prefix_changed",
        "prefix_identity_changed",
        "endpoint_schema_mismatch",
        "invalid_endpoint_metadata",
        "invalid_endpoint_token_count",
        "invalid_endpoint_block_size",
        "endpoint_not_ahead",
        "aligned_endpoint",
        "unsupported_blocks_per_chunk",
        "hash_block_size_mismatch",
        "cache_group_count_mismatch",
        "cache_group_block_size_mismatch",
        "missing_dependencies",
        "pending_dependencies",
        "no_endpoint_tier",
        "matching_endpoint",
        "normal_prefix_lookup",
        "unspecified",
    }
)
DECISION_COUNTS = frozenset(
    {
        "endpoint_tokens",
        "hash_size",
        "cached_tokens",
        "dependency_count",
        "missing_dependencies",
        "pending_dependencies",
        "tier_index",
    }
)


def record_response_end_decision(request, source, outcome, reason, **counts):
    """Log each distinct admission decision, independently of optional telemetry.

    Repeated polling of the same pending/rejected endpoint is deduplicated per
    request/source/tier. Changed reasons or counts are always recorded. No
    prefix hashes, salts, token values, paths or exception messages are emitted.
    """
    value = {
        "cache_source": source
        if source in {"gpu_endpoint", "offload_endpoint", "gpu_blocks"}
        else "unknown",
        "outcome": outcome
        if outcome in {"hit", "miss", "rejected", "loading"}
        else "unknown",
        "reason": reason if reason in DECISION_REASONS else "unspecified",
    }
    for name in ("input_tokens", "computed_tokens"):
        number = getattr(
            request,
            "num_prompt_tokens" if name == "input_tokens" else "num_computed_tokens",
            None,
        )
        if type(number) is int and 0 <= number <= 2**63 - 1:
            value[name] = number
    for name, number in counts.items():
        if name in DECISION_COUNTS and type(number) is int and 0 <= number <= 2**63 - 1:
            value[name] = number
    request_id = getattr(request, "request_id", None)
    if isinstance(request_id, str):
        value["request_id"] = hashlib.sha256(request_id.encode()).hexdigest()
    salt = getattr(request, "cache_salt", None)
    if isinstance(salt, str):
        parts = salt.rsplit(":", 2)
        if len(parts) == 3:
            for name, candidate in zip(
                ("chat_id", "generation"), parts[1:], strict=True
            ):
                if len(candidate) == 64 and all(
                    c in "0123456789abcdef" for c in candidate
                ):
                    value[name] = candidate
    previous = getattr(request, "_qwen_response_end_decisions", None)
    if previous is None:
        previous = request._qwen_response_end_decisions = {}
    key = (value["cache_source"], value.get("tier_index", 0))
    if previous.get(key) == value:
        return
    previous[key] = value
    # This remains in the ordinary backend log even when the bounded recorder
    # is disabled, contended or full. It runs at admission, never per decode row.
    logger.info(
        "Response cache decision: %s",
        json.dumps(value, sort_keys=True, separators=(",", ":")),
    )
    try:
        try:
            import qwen_radiance_cache_telemetry as telemetry
        except ModuleNotFoundError as error:
            if error.name != "qwen_radiance_cache_telemetry":
                raise
            from qwen_r9700_lab import radiance_cache_telemetry as telemetry
        telemetry.emit("response_end_lookup", **value)
    except Exception:  # noqa: BLE001, S110 -- decision already logged; telemetry cannot affect reuse.
        pass


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
        reason = None
        if entry is None:
            reason = "no_endpoint"
        elif entry["tokens"] >= request.num_tokens:
            reason = "prompt_does_not_extend_endpoint"
        elif request.mm_features:
            reason = "multimodal_request"
        elif getattr(request, "prompt_embeds", None) is not None:
            reason = "prompt_embeddings"
        elif request.lora_request is not None:
            reason = "lora_request"
        elif not self.manager.prefix_cache_lookup_enabled(request):
            reason = "prefix_cache_disabled"
        else:
            actual = prefix_identity(
                request, entry["tokens"], self.manager.block_pool.hash_block_size
            )
            if actual is None:
                reason = "prefix_identity_unavailable"
            elif actual[0] != entry["identity"][0]:
                reason = "cache_salt_changed"
            elif actual[1] != entry["identity"][1]:
                reason = "prefix_hash_changed"
            elif actual[2] != entry["identity"][2]:
                reason = "partial_prefix_changed"
        record_response_end_decision(
            request,
            "gpu_endpoint",
            "rejected" if reason else "hit",
            reason or "matching_endpoint",
            endpoint_tokens=entry["tokens"] if entry else 0,
            hash_size=self.manager.block_pool.hash_block_size,
        )
        if reason:
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
        if (
            entry is None
            or request.num_in_flight_tokens
            or request.num_computed_tokens <= entry["tokens"]
        ):
            return False
        request._qwen_response_end_lease = None
        # A terminal step may already have installed a newer checkpoint.
        if self.entry is not entry:
            return False
        self.clear()
        return True

    def release_unused(self, request, protected=()):
        """Under admission pressure, evict an endpoint this request cannot use.

        A changed/shortened prompt can miss both the exact endpoint and normal
        prefix cache. Keeping the old endpoint pinned then prevents a large
        replacement prompt from fitting, so it can never make the progress
        required by release_after_progress. Drop only our optional references,
        after lookup rejected reuse and while no copy/transfer depends on them.
        Disk snapshots and ordinary prefix-cache entries remain intact.
        """
        entry = self.entry
        if (
            entry is None
            or getattr(request, "_qwen_response_end_lease", entry) is not None
            or getattr(request, "qwen_response_end_local", 0)
            or request.num_computed_tokens
            or request.num_in_flight_tokens
            or self.pending
            or self.copy_pins
            or any(block.block_id in protected for block in entry["pins"])
        ):
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
        if (
            not isinstance(group.kv_cache_spec, MambaSpec)
            or group.mamba_cache_mode != "align"
        ):
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
