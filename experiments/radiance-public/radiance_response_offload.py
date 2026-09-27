"""Persist exact DFlash response ends through the existing tiered block store.

The physical block format is unchanged. A domain-separated endpoint hash names
the partial attention block and canonical GDN/conv block for each group. Older
readers safely miss these keys. Publication verifies the complete dependency set
before retiring the preceding durable head.
"""

import hashlib
import json

SCHEMA = "urn:coherence:response-end:v1"


def fingerprint(request, count, hash_size):
    from qwen_radiance_response_end import prefix_identity

    identity = prefix_identity(request, count, hash_size)
    if identity is None:
        return None
    salt, parent, tail = identity
    data = [
        SCHEMA,
        count,
        hash_size,
        salt,
        bytes(parent).hex() if parent else None,
        tail,
    ]
    return hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


def endpoint_key(endpoint, group_idx):
    from vllm.v1.kv_offload.base import make_offload_key

    return make_offload_key(bytes.fromhex(endpoint["prefix_sha256"]), group_idx)


def compatible_endpoint(status, endpoint):
    """Corrupt, stale or differently shaped metadata is always a cache miss."""
    if not isinstance(endpoint, dict) or endpoint.get("schema") != SCHEMA:
        return False
    count, block = endpoint.get("tokens"), endpoint.get("block_size")
    if (
        type(count) is not int
        or type(block) is not int
        or block <= 0
        or count <= status.num_locally_computed_tokens
        or count >= status.req.num_prompt_tokens
        or count % block == 0
        or status.config.blocks_per_chunk != 1
        or endpoint.get("hash_size") != status.config.tokens_per_hash
        or endpoint.get("groups") != len(status.config.kv_group_configs)
        or any(g.tokens_per_block != block for g in status.config.kv_group_configs)
    ):
        return False
    return fingerprint(
        status.req, count, status.config.tokens_per_hash
    ) == endpoint.get("prefix_sha256")


def endpoint_dependencies(status, endpoint):
    """All ordinary prefix/window blocks plus one endpoint block per group."""
    count, block = endpoint["tokens"], endpoint["block_size"]
    full = count // block
    result, changing = [], []
    for group, state in zip(
        status.config.kv_group_configs, status.group_states, strict=True
    ):
        window = group.sliding_window_size_in_chunks
        if group.requires_cow_source:
            prefix = []
        elif window is None:
            prefix = state.offload_keys[:full]
        else:
            # A partial final block can require one additional physical page.
            prefix = state.offload_keys[max(0, full - window) : full]
            changing.extend(prefix)
        result.extend(prefix)
        key = endpoint_key(endpoint, group.group_idx)
        result.append(key)
        changing.append(key)
    return result, changing


def store_response_ends(scheduler, output):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        GPULoadStoreSpec,
        TransferJob,
        TransferJobStatus,
    )

    jobs = {}
    for rid in output.finished_req_ids:
        status = scheduler._req_status.get(rid)
        capture = getattr(status.req, "qwen_response_end", None) if status else None
        if capture is None:
            continue
        count = capture["tokens"]
        groups = status.config.kv_group_configs
        block = groups[0].tokens_per_block
        if (
            count % block == 0
            or status.config.blocks_per_chunk != 1
            or any(g.tokens_per_block != block for g in groups)
        ):
            continue
        endpoint = {
            "schema": SCHEMA,
            "tokens": count,
            "block_size": block,
            "hash_size": status.config.tokens_per_hash,
            "groups": len(groups),
            "prefix_sha256": fingerprint(
                status.req, count, status.config.tokens_per_hash
            ),
        }
        if endpoint["prefix_sha256"] is None:
            continue
        keys = [endpoint_key(endpoint, g.group_idx) for g in groups]
        prepared = scheduler.manager.prepare_store(keys, status.req_context)
        if prepared is None:
            continue
        # Set classification before store completion reaches the secondary tier:
        # partial attention pages belong to the bounded RAM tail, not immutable
        # full-prefix storage written after every tool call.
        for tier in getattr(scheduler.manager, "secondary_tiers", ()):
            if hasattr(tier, "set_response_end_head"):
                tier.set_response_end_head(status, endpoint)
        if not prepared.keys_to_store:
            continue
        by_key = {key: i for i, key in enumerate(keys)}
        selected = [by_key[key] for key in prepared.keys_to_store]
        sources = [capture["block_ids"][i] for i in selected]
        sizes, indices = [0] * len(groups), [0] * len(groups)
        for i in selected:
            sizes[i], indices[i] = 1, count // block
        job_id = scheduler._generate_job_id()
        status.transfer_jobs.add(job_id)
        for bid in sources:
            scheduler._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)
        scheduler._jobs[job_id] = TransferJobStatus(
            req_id=rid,
            pending_count=status.config.num_workers,
            keys=set(prepared.keys_to_store),
            is_store=True,
            fenced_block_ids=sources,
        )
        jobs[job_id] = TransferJob(
            req_id=rid,
            src_spec=GPULoadStoreSpec(
                sources, group_sizes=sizes, block_indices=indices
            ),
            dst_spec=prepared.store_spec,
        )
    return jobs


def lookup_response_end(scheduler, status):
    """Return (handled, hit); None hit means a verified endpoint is loading."""
    from vllm.v1.kv_offload.base import LookupResult

    for tier in getattr(scheduler.manager, "secondary_tiers", ()):
        if not hasattr(tier, "response_end_head"):
            continue
        endpoint = tier.response_end_head(status.req_context)
        if not compatible_endpoint(status, endpoint):
            continue
        count = endpoint["tokens"]
        keys, _ = endpoint_dependencies(status, endpoint)
        pending, missing = False, False
        for key in keys:
            found = scheduler.manager.lookup(key, status.req_context)
            missing |= found is LookupResult.MISS
            pending |= found in (LookupResult.HIT_PENDING, LookupResult.RETRY)
        if missing:
            continue
        if pending:
            return True, None
        status.req.qwen_response_end_lookup = endpoint
        status.partial_tail_boundary = count
        return True, count - status.num_locally_computed_tokens
    return False, None
