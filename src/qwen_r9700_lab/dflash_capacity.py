"""Deterministic KV-capacity model for the co-located Qwen/DFlash lane.

The pinned vLLM 0.26 hybrid-cache planner preserves the drafter's five
``SlidingWindowSpec`` groups. Their real-held blocks plateau at the window plus
the in-flight chunk; only the target's sixteen full-attention layers scale with
context length. This distinction removes gigabytes of fictitious cache demand
from the old twenty-one-full-context-layer model.

This module reports the three material layouts separately:

* stock vLLM, whose uniform group size of five pads 16/48 layers to 20/50;
* padding-free groups with stock speculative GDN state slots; and
* padding-free groups with an explicitly verified Bole/Recover state budget.

It never treats a small drafter ``max_model_len`` as a valid capacity saving:
that setting causes vLLM to skip DFlash above the cap.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import ceil

GIB = 1 << 30


@dataclass(frozen=True)
class DFlashCapacityInputs:
    # Advertised target/drafter length. The fine-prefix production contract pins
    # 253,792 so its 249,957-token fixture retains 3,835 generation positions.
    context_tokens: int = 253_792
    occupied_prefix_tokens: int = 249_957
    block_size: int = 1_648
    target_attention_layers: int = 16
    target_gdn_layers: int = 48
    draft_attention_layers: int = 5
    draft_sliding_window_tokens: int = 2_048
    max_num_batched_tokens: int = 4_096
    # dflash_config.block_size=8 is anchor-inclusive: seven mask rows are trained.
    speculative_tokens: int = 7
    stock_group_size: int = 5
    # FP8 K and V each contain four heads of dimension 256 in the target.
    # The drafter's eight heads of dimension 128 have the same byte count.
    kv_bytes_per_token_per_layer: int = 2_048
    # Exact arena reconstructed from 685 allocator blocks, five physical
    # pages/block, and 3,375,104-byte pages in server_dflash_v7.log.
    measured_max_arena_bytes: int = 11_559_731_200
    # Exactly 2,586 physical pages: the 2,585-page trained align-v3 model
    # requirement at max_model_len=253,792 plus one BlockPool null page.  This
    # exact-minimum arena has no additional durable fine-prefix replay reserve
    # at the advertised cap; an evicted replay state falls back to a coarser hit.
    explicit_cache_bytes: int = 8_728_018_944
    pool_null_pages: int = 1
    fine_prefix_replay_reserve_pages: int = 0
    # RecoverSSM keeps compact decay/key/correction factors outside vLLM's KV
    # arena.  K is duplicated per value head to avoid cross-workgroup races.
    recoverssm_sidecar_bytes_per_gdn_layer: int = 394_756

    def __post_init__(self) -> None:
        positive = (
            self.context_tokens,
            self.occupied_prefix_tokens,
            self.block_size,
            self.target_attention_layers,
            self.target_gdn_layers,
            self.draft_attention_layers,
            self.draft_sliding_window_tokens,
            self.max_num_batched_tokens,
            self.speculative_tokens,
            self.stock_group_size,
            self.kv_bytes_per_token_per_layer,
            self.measured_max_arena_bytes,
            self.explicit_cache_bytes,
            self.pool_null_pages,
            self.recoverssm_sidecar_bytes_per_gdn_layer,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("all capacity inputs must be positive")
        if self.occupied_prefix_tokens > self.context_tokens:
            raise ValueError("occupied_prefix_tokens cannot exceed context_tokens")
        if self.fine_prefix_replay_reserve_pages < 0:
            raise ValueError("fine_prefix_replay_reserve_pages cannot be negative")


@dataclass(frozen=True)
class CacheLayout:
    name: str
    group_size: int
    attention_blocks_per_layer: int
    draft_attention_blocks_per_layer: int
    gdn_state_pages_per_layer: int
    physical_pages_required: int
    bytes_required: int

    @property
    def gib_required(self) -> float:
        return self.bytes_required / GIB


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def cache_page_bytes(inputs: DFlashCapacityInputs) -> int:
    return inputs.block_size * inputs.kv_bytes_per_token_per_layer


def layout_for_group_size(
    inputs: DFlashCapacityInputs,
    group_size: int,
    *,
    gdn_state_pages_per_layer: int | None = None,
    name: str = "custom",
) -> CacheLayout:
    """Return the page-rounded active-DFlash layout.

    ``gdn_state_pages_per_layer=None`` means stock vLLM semantics: one base
    state plus one state for each speculative token.  A smaller value is only
    valid after a Bole/Recover implementation has proved its commit/rollback
    semantics; this calculator does not enable that runtime capability.
    """

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if gdn_state_pages_per_layer is None:
        gdn_state_pages_per_layer = 1 + inputs.speculative_tokens
    if gdn_state_pages_per_layer <= 0:
        raise ValueError("gdn_state_pages_per_layer must be positive")

    attention_blocks = _ceil_div(inputs.context_tokens, inputs.block_size)
    draft_attention_blocks = (
        _ceil_div(
            inputs.draft_sliding_window_tokens - 1 + inputs.max_num_batched_tokens,
            inputs.block_size,
        )
        + 1
    )
    target_attention_groups = ceil(inputs.target_attention_layers / group_size)
    draft_attention_groups = ceil(inputs.draft_attention_layers / group_size)
    gdn_groups = ceil(inputs.target_gdn_layers / group_size)
    pooled_blocks = (
        target_attention_groups * attention_blocks
        + draft_attention_groups * draft_attention_blocks
        + gdn_groups * gdn_state_pages_per_layer
    )
    physical_pages = pooled_blocks * group_size
    required = physical_pages * cache_page_bytes(inputs)
    return CacheLayout(
        name=name,
        group_size=group_size,
        attention_blocks_per_layer=attention_blocks,
        draft_attention_blocks_per_layer=draft_attention_blocks,
        gdn_state_pages_per_layer=gdn_state_pages_per_layer,
        physical_pages_required=physical_pages,
        bytes_required=required,
    )


def _layout_dict(
    layout: CacheLayout,
    *,
    arena_bytes: int,
) -> dict[str, object]:
    return {
        **asdict(layout),
        "gib_required": layout.gib_required,
        "fits_arena": layout.bytes_required <= arena_bytes,
    }


def capacity_report(inputs: DFlashCapacityInputs) -> dict[str, object]:
    page_bytes = cache_page_bytes(inputs)
    stock = layout_for_group_size(inputs, inputs.stock_group_size, name="stock_padded")
    compact_stock = layout_for_group_size(inputs, 1, name="compact_stock_states")
    compact_bole_one = layout_for_group_size(
        inputs, 1, gdn_state_pages_per_layer=1, name="compact_bole_one_state"
    )
    compact_bole_two = layout_for_group_size(
        inputs, 1, gdn_state_pages_per_layer=2, name="compact_bole_two_states"
    )
    compact_bole_three = layout_for_group_size(
        inputs, 1, gdn_state_pages_per_layer=3, name="compact_bole_three_states"
    )
    explicit_pages = inputs.explicit_cache_bytes // page_bytes
    explicit_remainder_bytes = inputs.explicit_cache_bytes % page_bytes
    model_usable_pages = explicit_pages - inputs.pool_null_pages
    if model_usable_pages < 0:
        raise ValueError("explicit cache arena cannot hold the BlockPool null pages")
    model_usable_bytes = model_usable_pages * page_bytes
    replay_reserve_bytes = inputs.fine_prefix_replay_reserve_pages * page_bytes
    contract_model_pages = model_usable_pages - inputs.fine_prefix_replay_reserve_pages
    if contract_model_pages < 0:
        raise ValueError("explicit cache arena cannot hold the fine-prefix replay reserve")
    recoverssm_sidecar_bytes = (
        inputs.target_gdn_layers * inputs.recoverssm_sidecar_bytes_per_gdn_layer
    )
    recovered_bytes = inputs.measured_max_arena_bytes - inputs.explicit_cache_bytes

    layouts = {
        "stock_padded": _layout_dict(stock, arena_bytes=inputs.measured_max_arena_bytes),
        "compact_stock_states": _layout_dict(compact_stock, arena_bytes=model_usable_bytes),
        "compact_bole_one_state": _layout_dict(compact_bole_one, arena_bytes=model_usable_bytes),
        "compact_bole_two_states": _layout_dict(compact_bole_two, arena_bytes=model_usable_bytes),
        "compact_bole_three_states": _layout_dict(
            compact_bole_three, arena_bytes=model_usable_bytes
        ),
    }
    for layout in layouts.values():
        required_pages = int(layout["physical_pages_required"])
        gross_spare_pages = explicit_pages - required_pages
        usable_spare_pages = model_usable_pages - required_pages
        configured_replay_reserve_fits = (
            usable_spare_pages >= inputs.fine_prefix_replay_reserve_pages
        )
        layout["spare_pages"] = gross_spare_pages
        layout["spare_bytes"] = gross_spare_pages * page_bytes
        layout["usable_spare_pages"] = usable_spare_pages
        layout["usable_spare_bytes"] = usable_spare_pages * page_bytes
        layout["preserves_fine_prefix_replay_reserve"] = (
            inputs.fine_prefix_replay_reserve_pages > 0 and configured_replay_reserve_fits
        )
        layout["fits_contract"] = bool(layout["fits_arena"]) and configured_replay_reserve_fits

    full_context_tokens = 262_144
    full_context_attention_blocks = _ceil_div(full_context_tokens, inputs.block_size)
    full_context_required_pages = (
        inputs.target_attention_layers * full_context_attention_blocks
        + inputs.draft_attention_layers * compact_bole_two.draft_attention_blocks_per_layer
        + inputs.target_gdn_layers * compact_bole_two.gdn_state_pages_per_layer
    )
    full_context_required_bytes = full_context_required_pages * page_bytes
    full_context_shortfall_pages = full_context_required_pages - model_usable_pages

    return {
        "inputs": asdict(inputs),
        "page_bytes": page_bytes,
        "target_attention_layers_retained_full_context": inputs.target_attention_layers,
        "draft_attention_layers_sliding_window": inputs.draft_attention_layers,
        "draft_minimum_active_max_model_len": (
            inputs.occupied_prefix_tokens + inputs.speculative_tokens + 1
        ),
        "generation_headroom_tokens": inputs.context_tokens - inputs.occupied_prefix_tokens,
        "explicit_arena": {
            "configured_bytes": inputs.explicit_cache_bytes,
            "configured_pages": explicit_pages,
            "remainder_bytes": explicit_remainder_bytes,
            "pool_null_pages": inputs.pool_null_pages,
            "pool_null_bytes": inputs.pool_null_pages * page_bytes,
            "usable_pages": model_usable_pages,
            "usable_bytes": model_usable_bytes,
            "usable_gib": model_usable_bytes / GIB,
            "fine_prefix_replay_reserve_pages": inputs.fine_prefix_replay_reserve_pages,
            "fine_prefix_replay_reserve_bytes": replay_reserve_bytes,
            "durable_fine_prefix_replay_at_safe_max": (inputs.fine_prefix_replay_reserve_pages > 0),
            "contract_model_pages": contract_model_pages,
            "contract_model_bytes": contract_model_pages * page_bytes,
        },
        "activation_headroom": {
            "bytes_recovered_vs_measured_max_arena": recovered_bytes,
            "gib_recovered_vs_measured_max_arena": recovered_bytes / GIB,
            "recoverssm_sidecar_bytes": recoverssm_sidecar_bytes,
            "recoverssm_sidecar_gib": recoverssm_sidecar_bytes / GIB,
            "net_bytes_after_recoverssm_sidecars": recovered_bytes - recoverssm_sidecar_bytes,
            "net_gib_after_recoverssm_sidecars": (recovered_bytes - recoverssm_sidecar_bytes) / GIB,
        },
        "full_context": {
            "context_tokens": full_context_tokens,
            "target_attention_blocks_per_layer": full_context_attention_blocks,
            "physical_pages_required": full_context_required_pages,
            "bytes_required": full_context_required_bytes,
            "model_usable_arena_pages": model_usable_pages,
            "shortfall_pages": full_context_shortfall_pages,
            "shortfall_bytes": full_context_shortfall_pages * page_bytes,
            "fits_arena": full_context_shortfall_pages <= 0,
        },
        "layouts": layouts,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Model active R9700 target+DFlash FP8 KV-cache capacity."
    )
    parser.add_argument("--context-tokens", type=int, default=253_792)
    parser.add_argument("--occupied-prefix-tokens", type=int, default=249_957)
    parser.add_argument("--speculative-tokens", type=int, default=7)
    parser.add_argument("--explicit-cache-bytes", type=int, default=8_728_018_944)
    parser.add_argument(
        "--gdn-state-pages",
        type=int,
        default=2,
        help="verified Bole/Recover state pages retained per GDN layer",
    )
    parser.add_argument(
        "--require-fit",
        action="store_true",
        help="exit nonzero unless the selected layout fits the model-usable production arena",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = DFlashCapacityInputs(
        context_tokens=args.context_tokens,
        occupied_prefix_tokens=args.occupied_prefix_tokens,
        speculative_tokens=args.speculative_tokens,
        explicit_cache_bytes=args.explicit_cache_bytes,
    )
    if args.gdn_state_pages <= 0:
        raise SystemExit("--gdn-state-pages must be positive")
    layout = layout_for_group_size(
        inputs,
        1,
        gdn_state_pages_per_layer=args.gdn_state_pages,
        name="selected",
    )
    report = capacity_report(inputs)
    selected = _layout_dict(
        layout,
        arena_bytes=(
            inputs.explicit_cache_bytes // cache_page_bytes(inputs) - inputs.pool_null_pages
        )
        * cache_page_bytes(inputs),
    )
    selected_usable_spare_pages = (
        inputs.explicit_cache_bytes // cache_page_bytes(inputs)
        - inputs.pool_null_pages
        - layout.physical_pages_required
    )
    selected["usable_spare_pages"] = selected_usable_spare_pages
    configured_replay_reserve_fits = (
        selected_usable_spare_pages >= inputs.fine_prefix_replay_reserve_pages
    )
    selected["preserves_fine_prefix_replay_reserve"] = (
        inputs.fine_prefix_replay_reserve_pages > 0 and configured_replay_reserve_fits
    )
    selected["fits_contract"] = bool(selected["fits_arena"]) and configured_replay_reserve_fits
    report["selected"] = selected
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_fit and not selected["fits_contract"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
