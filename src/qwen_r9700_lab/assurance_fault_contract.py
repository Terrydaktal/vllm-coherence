"""Canonical failure-atomic boundary inventory for Quest96/W4A16.

This is shared by the instrumented runtime and the standalone campaign verifier.
It contains no instrumentation and is therefore safe to include unchanged in the
release artifact; keeping one inventory prevents either side accepting a campaign
that the other side did not actually require.
"""

from __future__ import annotations

LAYER_COUNT = 64
QUEST_LAYERS = tuple(range(3, LAYER_COUNT, 4))
GDN_LAYERS = tuple(layer for layer in range(LAYER_COUNT) if layer not in QUEST_LAYERS)


def _expected_projection_sites() -> tuple[tuple[int, str], ...]:
    sites: list[tuple[int, str]] = []
    for layer in range(LAYER_COUNT):
        if layer in QUEST_LAYERS:
            operations = (
                "self_attn.qkv_proj",
                "self_attn.o_proj",
                "mlp.gate_up_proj",
                "mlp.down_proj",
            )
        else:
            operations = (
                "linear_attn.in_proj_qkvz",
                "linear_attn.in_proj_ba",
                "linear_attn.out_proj",
                "mlp.gate_up_proj",
                "mlp.down_proj",
            )
        sites.extend((layer, operation) for operation in operations)
    return tuple(sites)


EXPECTED_PROJECTION_SITES = _expected_projection_sites()


def _fault_sites() -> tuple[str, ...]:
    """Enumerate every mutable or failure-bearing semantic boundary in the lane."""

    sites: list[str] = []
    sites.extend(f"target_kv.write.layer_{layer}" for layer in QUEST_LAYERS)
    sites.extend(f"draft_kv.write.layer_{layer}" for layer in range(64, 69))
    sites.extend(f"gdn.convolution.layer_{layer}" for layer in GDN_LAYERS)
    sites.extend(f"gdn.recurrence.layer_{layer}" for layer in GDN_LAYERS)
    sites.extend(f"gdn.gated_rmsnorm.layer_{layer}" for layer in GDN_LAYERS)
    sites.extend(
        f"projection.layer_{layer}.{operation}"
        for layer, operation in EXPECTED_PROJECTION_SITES
    )
    for operation in (
        "qk_norm_rope",
        "quest_scoring",
        "top96_selection",
        "union_visibility",
        "historical_attention",
        "causal_tail_attention",
        "softmax_reduction",
    ):
        sites.extend(f"attention.layer_{layer}.{operation}" for layer in QUEST_LAYERS)
    sites.extend(f"residual.layer_{layer}" for layer in range(LAYER_COUNT))
    sites.extend(f"ffn.layer_{layer}" for layer in range(LAYER_COUNT))
    sites.extend(
        (
            "m8_construction.positions_rope",
            "normalization.final_rmsnorm",
            "lm_head.logits",
            "target_verification.acceptance",
            "sampler.rng_decoding",
            "scheduler.request_admission",
            "scheduler.round_bookkeeping",
            "scheduler.stop_tool_boundary",
            "scheduler.cancellation",
            "scheduler.retry",
            "allocator.allocate",
            "allocator.ownership",
            "allocator.pins",
            "allocator.refcounts",
            "allocator.page_reuse",
            "allocator.free",
            "snapshot.restore.load",
            "snapshot.restore.validate",
            "snapshot.restore.publish",
            "snapshot.offload.allocate",
            "snapshot.offload.write",
            "snapshot.offload.publish_manifest",
            "snapshot.reload.lookup",
            "snapshot.reload.read",
            "snapshot.reload.adopt",
            "commit.prepare",
            "commit.journal",
            "commit.root_pointer_swap",
            "commit.acknowledgement",
        )
    )
    if len(sites) != len(set(sites)):
        raise RuntimeError(
            "full-assurance transactional fault-site inventory duplicates a boundary"
        )
    return tuple(sites)


FAULT_SITES = _fault_sites()
