#!/usr/bin/env python3
"""Bound hybrid snapshot lookup without changing the restore transaction.

Radiance/vLLM's asynchronous sliding-window lookup continues walking backward
while the newest filesystem keys are unresolved. For Qwen3.8 Mamba/GDN groups
that speculative walk schedules every historical recurrent-state page for
promotion, although only the final state window can be loaded or observed.

This patch makes an unresolved newest suffix a scheduling barrier. Full-
attention groups retain the upstream maximal-prefix lookup, while Mamba,
sliding-attention, and draft groups stage only the suffix required by their
declared window. Recurrent/sliding persistence is restricted to the settled
tail of a finished request: historical recurrent blocks are reused while a
long prefill advances and therefore are not durable snapshot material. Each
published tail includes the complete declared window. Withheld recurrent chunks
must also remain pending in ``next_stored_chunk_idx``; otherwise ordinary full-
attention stores incorrectly mark the recurrent history as already persisted and
the settled tail is never published. A successor lookup is held while an older
finished request still exists in the connector: that state is removed only after
its complete settled-tail publication finishes. This closes the response-to-
publication race without exposing a partial snapshot or forcing a cold fill.
Restoration itself remains the upstream
single allocation, single async load and single cache publication; there are no
intermediate state commits or scheduler hand-offs.

The edit is anchored to the exact qualified vLLM 0.27.1 source and fails closed
if that source changes.
"""

from __future__ import annotations

import os
from pathlib import Path

SCHEDULER = Path(
    os.environ.get(
        "QWEN_SNAPSHOT_SCHEDULER",
        "/opt/vllm/lib/python3.12/site-packages/vllm/"
        "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py",
    )
)


def replace_once(path: Path, old: str, new: str, marker: str) -> None:
    text = path.read_text()
    if marker in text:
        print(f"[snapshot-lookup] already applied: {path}")
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"snapshot-lookup source ABI changed: {path}: "
            f"expected one anchor, found {count}"
        )
    temporary = path.with_name(f".{path.name}.snapshot-lookup.tmp")
    temporary.write_text(text.replace(old, new))
    temporary.replace(path)
    print(f"[snapshot-lookup] applied: {path}")


replace_once(
    SCHEDULER,
    """        self.manager: OffloadingManager = spec.get_manager()
        self._connector_stats = OffloadingConnectorStats()

        full_attention_groups: list[int] = []
""",
    """        self.manager: OffloadingManager = spec.get_manager()
        self._connector_stats = OffloadingConnectorStats()

        # Historical Mamba/sliding blocks are recycled while a long request
        # advances. Persisting them asynchronously can capture a later state
        # under an earlier prefix key. Only a finished request's complete tail
        # is stable snapshot material.
        self._snapshot_settled_tail_only = bool(
            spec.extra_config.get("snapshot_settled_tail_only", False)
        )

        full_attention_groups: list[int] = []
""",
    "Only a finished request's complete tail",
)


replace_once(
    SCHEDULER,
    """        req_status = self._req_status[request.request_id]
        for group_state in req_status.group_states:
""",
    """        req_status = self._req_status[request.request_id]

        # A finished predecessor remains in _req_status until its complete
        # settled tail has been submitted and every asynchronous store job has
        # completed. Do not let a successor race that publication and observe a
        # false cache miss. The scheduler will keep stepping because the
        # connector reports pending push work, then retry this lookup once the
        # predecessor is removed.
        if self._snapshot_settled_tail_only:
            unpublished_finished_req_id = next(
                (
                    req_id
                    for req_id, status in self._req_status.items()
                    if req_id != request.request_id and status.req.is_finished()
                ),
                None,
            )
            if unpublished_finished_req_id is not None:
                logger.debug(
                    "Delaying request %s until finished request %s has "
                    "published its settled snapshot tail",
                    request.request_id,
                    unpublished_finished_req_id,
                )
                return None, False

        for group_state in req_status.group_states:
""",
    "A finished predecessor remains in _req_status",
)


replace_once(
    SCHEDULER,
    """                case LookupResult.HIT_PENDING:
                    # Block is in cache, just not readable yet — counts
                    # as hit for the consecutive streak. Don't break:
                    # keep scanning to let manager kick off async lookups.
                    defer_lookup = True
                    consecutive_hits += 1
                case LookupResult.RETRY:
                    # Block location uncertain — does not count as hit.
                    # Don't break: keep scanning to let manager kick off
                    # async lookups.
                    defer_lookup = True
                    consecutive_hits = 0
""",
    """                case LookupResult.HIT_PENDING:
                    # The newest candidate suffix is present but not readable.
                    # Do not speculate farther into historical Mamba/sliding
                    # state: none of those pages can be observed once this
                    # suffix is ready, and staging them can exhaust the CPU tier.
                    return None
                case LookupResult.RETRY:
                    # This key's filesystem lookup or promotion is unresolved.
                    # Treat it as a barrier and retry the same newest suffix on
                    # the next scheduler step before looking any farther back.
                    return None
""",
    "Treat it as a barrier and retry the same newest suffix",
)

replace_once(
    SCHEDULER,
    """        store_jobs: dict[int, TransferJob] = {}
        for req_id in chain(
""",
    """        store_jobs: dict[int, TransferJob] = {}
        settled_req_ids = set(scheduler_output.finished_req_ids or ())
        for req_id in chain(
""",
    "settled_req_ids = set(scheduler_output.finished_req_ids or ())",
)


replace_once(
    SCHEDULER,
    """    def advance_stored_idx(self, num_offloadable_tokens: int) -> None:
        # max(): at the prefill->decode transition of a chunk-aligned prompt,
        # storable_chunks drops by one (the eagle exclusion kicks in), and the
        # index must not move backwards past already-stored chunks.
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.next_stored_chunk_idx = max(
                group_state.next_stored_chunk_idx,
                self.storable_chunks(group_config, group_state, num_offloadable_tokens),
            )
""",
    """    def advance_stored_idx(
        self,
        num_offloadable_tokens: int,
        preserve_sliding: bool = False,
    ) -> None:
        # max(): at the prefill->decode transition of a chunk-aligned prompt,
        # storable_chunks drops by one (the eagle exclusion kicks in), and the
        # index must not move backwards past already-stored chunks.
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            if preserve_sliding and group_config.sliding_window_size_in_chunks is not None:
                continue
            group_state.next_stored_chunk_idx = max(
                group_state.next_stored_chunk_idx,
                self.storable_chunks(group_config, group_state, num_offloadable_tokens),
            )
""",
    "preserve_sliding: bool = False",
)


replace_once(
    SCHEDULER,
    """                    abs_chunk_idx = start_chunk_idx + key_idx
                    if not is_store_reachable_swa_chunk(
""",
    """                    abs_chunk_idx = start_chunk_idx + key_idx
                    sliding_chunks = group_config.sliding_window_size_in_chunks
                    if self._snapshot_settled_tail_only and sliding_chunks is not None:
                        retained_tail = sliding_chunks + int(
                            group_config.is_eagle_group
                            or getattr(group_config, "requires_cow_source", False)
                        )
                        if (
                            req_id not in settled_req_ids
                            or abs_chunk_idx < num_chunks - retained_tail
                        ):
                            continue
                    if not is_store_reachable_swa_chunk(
""",
    "if req_id not in settled_req_ids or abs_chunk_idx < num_chunks - retained_tail",
)


replace_once(
    SCHEDULER,
    """            if not new_offload_keys:
                req_status.advance_stored_idx(num_offloadable_tokens)
                continue
""",
    """            if not new_offload_keys:
                req_status.advance_stored_idx(
                    num_offloadable_tokens,
                    preserve_sliding=(
                        self._snapshot_settled_tail_only
                        and req_id not in settled_req_ids
                    ),
                )
                continue
""",
    "preserve_sliding=(\n                        self._snapshot_settled_tail_only",
)


replace_once(
    SCHEDULER,
    """            if not store_output.keys_to_store:
                req_status.advance_stored_idx(num_offloadable_tokens)
                continue
""",
    """            if not store_output.keys_to_store:
                req_status.advance_stored_idx(
                    num_offloadable_tokens,
                    preserve_sliding=(
                        self._snapshot_settled_tail_only
                        and req_id not in settled_req_ids
                    ),
                )
                continue
""",
    "if not store_output.keys_to_store:\n                req_status.advance_stored_idx(\n",
)


replace_once(
    SCHEDULER,
    """                group_state.next_stored_chunk_idx = max(
                    group_state.next_stored_chunk_idx, num_chunks
                )
""",
    """                if not (
                    self._snapshot_settled_tail_only
                    and is_sliding_window
                    and req_id not in settled_req_ids
                ):
                    group_state.next_stored_chunk_idx = max(
                        group_state.next_stored_chunk_idx, num_chunks
                    )
""",
    "and is_sliding_window\n                    and req_id not in settled_req_ids",
)

replace_once(
    SCHEDULER,
    """        if group_config.is_eagle_group and is_decoding:
            num_chunks = max(0, num_chunks - 1)
""",
    """        # Align target recurrent snapshots with the settled draft frontier.
        # v0.28 no longer marks target Mamba groups as EAGLE. Keeping their
        # newest two states would put both ahead of the draft lookup boundary.
        has_draft_tail = group_config.is_eagle_group or (
            getattr(group_config, "requires_cow_source", False)
            and any(group.is_eagle_group for group in self.config.kv_group_configs)
        )
        if has_draft_tail and is_decoding:
            num_chunks = max(0, num_chunks - 1)
""",
    "# Align target recurrent snapshots with the settled draft frontier.",
)

# The source-only scheduler tests set QWEN_SNAPSHOT_SCHEDULER. Runtime bootstrap
# installs the separate storage modules and their two scheduler hooks as well.
if "QWEN_SNAPSHOT_SCHEDULER" not in os.environ:
    from patch_chat_snapshot import install

    install(
        SCHEDULER.parents[6],
        Path(__file__).with_name("radiance_cache.py"),
        Path(__file__).with_name("radiance_chat_tier.py"),
    )
