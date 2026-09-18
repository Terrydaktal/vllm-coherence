"""Allocation-free hot-path helpers for exact C1/M8 best-first B7.

All expensive discovery and pointer-table construction happens once, after the
model and KV caches are initialized.  A serving round then consists of one
device verifier launch, one alias-safe 16-layer Quest KV validation/gather
launch, one bulk 48-layer GDN publication launch, and one 16-layer Quest KV
publication launch.  The round never reads a device scalar on the host and
never iterates over model layers on the decode hot path.

This module is deliberately default-off.  The runner overlay must authenticate
the B7 consumer receipt before constructing :class:`B7GPUWorkspace`.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

B7_NODES = 7
B7_ROWS = 8
GDN_LAYERS = 48
QUEST_LAYERS = 16
VALUE_HEADS = 48
HEAD_DIM = 128
KEY_HEADS = 16
QKV_WIDTH = 2 * KEY_HEADS * HEAD_DIM + VALUE_HEADS * HEAD_DIM
CONV_HISTORY = 10
PACKED_GDN_PAGE_BYTES = 3_375_104
PACKED_GDN_CONTENT_BYTES = 3_350_528
PACKED_GDN_PADDING_BYTES = PACKED_GDN_PAGE_BYTES - PACKED_GDN_CONTENT_BYTES
QUEST_KV_HEADS = 4
QUEST_HEAD_DIM = 256
QUEST_KV_ELEMENTS = QUEST_KV_HEADS * QUEST_HEAD_DIM
QUEST_LAYER_NAMES = tuple(
    f"language_model.model.layers.{index}.self_attn.attn" for index in range(3, 64, 4)
)
QUALIFIED_PAGE_TOKENS = 1648


class B7GPUContractError(RuntimeError):
    """A persistent B7 buffer or current-lineage model ABI differed."""


def _require_b7_position_rewrite_layout(torch: Any, positions: Any, device: Any) -> None:
    """Validate one nonoverlapping row-strided MRoPE output view."""

    if (
        tuple(positions.shape) != (3, B7_ROWS)
        or positions.dtype != torch.int64
        or positions.device != device
        or positions.layout != torch.strided
    ):
        raise B7GPUContractError("B7 logical-position rewrite requires strided CUDA int64[3,8]")
    row_stride = int(positions.stride(0))
    column_stride = int(positions.stride(1))
    storage_offset = int(positions.storage_offset())
    element_size = int(positions.element_size())
    storage_nbytes = int(positions.untyped_storage().nbytes())
    if (
        column_stride != 1
        or row_stride < B7_ROWS
        or storage_offset != 0
        or element_size <= 0
        or storage_nbytes < 0
        or storage_nbytes % element_size != 0
    ):
        raise B7GPUContractError(
            "B7 logical-position rewrite requires positive nonoverlapping row strides"
        )
    storage_elements = storage_nbytes // element_size
    last_offset = storage_offset + 2 * row_stride + (B7_ROWS - 1) * column_stride
    if last_offset >= storage_elements:
        raise B7GPUContractError("B7 logical-position rewrite exceeds its backing storage")


def _identity(tensor: Any) -> tuple[object, ...]:
    return (
        int(tensor.data_ptr()),
        int(tensor.storage_offset()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
    )


@dataclass(frozen=True)
class B7VerifierOutputs:
    """Persistent device-only branch decision consumed by commit kernels."""

    accepted_count: Any
    accepted_nodes: Any
    accepted_leaf_physical_row: Any
    accepted_leaf_state_slot: Any
    status: Any

    @classmethod
    def allocate(cls, torch: Any, device: Any) -> B7VerifierOutputs:
        return cls(
            accepted_count=torch.empty((1,), dtype=torch.int32, device=device),
            accepted_nodes=torch.empty((B7_NODES,), dtype=torch.int32, device=device),
            accepted_leaf_physical_row=torch.empty((1,), dtype=torch.int64, device=device),
            accepted_leaf_state_slot=torch.empty((1,), dtype=torch.int64, device=device),
            status=torch.empty((1,), dtype=torch.int32, device=device),
        )

    def validate(self, torch: Any, device: Any) -> None:
        requested_device = torch.device(device)
        if requested_device.type != "cuda":
            raise B7GPUContractError("B7 verifier outputs require a CUDA device")
        requested_index = (
            torch.cuda.current_device()
            if requested_device.index is None
            else requested_device.index
        )
        expected = (
            (self.accepted_count, (1,), torch.int32),
            (self.accepted_nodes, (B7_NODES,), torch.int32),
            (self.accepted_leaf_physical_row, (1,), torch.int64),
            (self.accepted_leaf_state_slot, (1,), torch.int64),
            (self.status, (1,), torch.int32),
        )
        if any(
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device.type != requested_device.type
            or tensor.device.index != requested_index
            or not tensor.is_contiguous()
            for tensor, shape, dtype in expected
        ):
            raise B7GPUContractError("B7 verifier outputs have an unsupported ABI")


@dataclass
class B7GDNPointerTables:
    layers: tuple[Any, ...]
    branch_states: Any
    conv_checkpoints: Any
    canonical_states: Any
    canonical_state_slot_stride: int
    canonical_convs: Any
    canonical_conv_strides: Any
    source_indices: Any
    state_slot_counts: Any
    conv_slot_counts: Any
    published_slots: Any
    backup_states: Any
    backup_convs: Any
    backup_conv_strides: Any
    commit_kernel: Any
    rollback_kernel: Any
    triton: Any


@dataclass
class B7KVPointerTables:
    layer_names: tuple[str, ...]
    kv_cache_views: tuple[Any, ...]
    key_views: tuple[Any, ...]
    value_views: tuple[Any, ...]
    storage_view: Any
    cache_offsets: Any
    group_indices: Any
    group_indices_cpu: tuple[int, ...]
    slot_mapping_view: Any | None
    staging_key: Any
    staging_value: Any
    slot_identity: tuple[object, ...] | None
    num_blocks: int
    page_tokens: int
    key_strides: tuple[int, ...]
    value_strides: tuple[int, ...]
    gather_kernel: Any
    publish_kernel: Any
    triton: Any


_VERIFIER_KERNEL: Any | None = None


def _build_verifier_kernel() -> Any:
    global _VERIFIER_KERNEL
    if _VERIFIER_KERNEL is not None:
        return _VERIFIER_KERNEL
    triton = importlib.import_module("triton")
    globals()["tl"] = importlib.import_module("triton.language")

    @triton.jit
    def greedy_best_first_b7_verify_kernel(
        sampled_ptr,
        num_sampled_ptr,
        accepted_count_ptr,
        accepted_nodes_ptr,
        accepted_leaf_physical_row_ptr,
        accepted_leaf_state_slot_ptr,
        status_ptr,
        commit_capacity,
        target_argmax_ptr,
        transition_tokens_ptr,
        parent_ptr,
        NUM_NODES: tl.constexpr,
        NUM_ROWS: tl.constexpr,
    ):
        tl.store(status_ptr, 1)
        nodes = tl.arange(0, 8)
        valid_node = nodes < NUM_NODES
        parents = tl.load(parent_ptr + nodes, mask=valid_node, other=-2).to(tl.int32)
        tokens = tl.load(transition_tokens_ptr + nodes + 1, mask=valid_node, other=-1).to(tl.int64)
        lengths = tl.where(valid_node, 1, 0).to(tl.int32)
        next_nodes = tl.full((8,), -1, tl.int32)
        for reverse_index in tl.static_range(0, NUM_NODES):
            node = NUM_NODES - 1 - reverse_index
            expected = tl.load(target_argmax_ptr + node + 1).to(tl.int64)
            child_matches = valid_node & (parents == node) & (tokens == expected)
            child_ranks = tl.where(child_matches, lengths * NUM_ROWS + (NUM_NODES - nodes), -1)
            best_rank = tl.max(child_ranks, axis=0)
            has_child = best_rank >= 0
            best_child = tl.where(has_child, NUM_NODES - best_rank % NUM_ROWS, -1).to(tl.int32)
            best_child_length = tl.where(has_child, best_rank // NUM_ROWS, 0).to(tl.int32)
            lengths = tl.where(nodes == node, best_child_length + 1, lengths)
            next_nodes = tl.where(nodes == node, best_child, next_nodes)

        root_expected = tl.load(target_argmax_ptr).to(tl.int64)
        root_matches = valid_node & (parents == -1) & (tokens == root_expected)
        root_ranks = tl.where(root_matches, lengths * NUM_ROWS + (NUM_NODES - nodes), -1)
        best_root_rank = tl.max(root_ranks, axis=0)
        has_root = best_root_rank >= 0
        accepted_count = tl.where(has_root, best_root_rank // NUM_ROWS, 0).to(tl.int32)
        # Always reserve one output position for the exact target bonus.  When
        # an external max-token cap leaves fewer than eight positions, this
        # shortens only the accepted prefix; the replacement bonus is the same
        # target token and canonical state advances through accepted proposals
        # only, exactly like ordinary speculative decoding.
        accepted_count = tl.minimum(accepted_count, commit_capacity - 1)
        current_node = tl.where(has_root, NUM_NODES - best_root_rank % NUM_ROWS, -1).to(tl.int32)

        tl.store(sampled_ptr + nodes, -1, mask=nodes < NUM_ROWS)
        tl.store(accepted_nodes_ptr + nodes, -1, mask=valid_node)
        last_node = tl.full((), -1, tl.int32)
        for output_index in tl.static_range(0, NUM_NODES):
            active = output_index < accepted_count
            accepted_token = tl.load(
                transition_tokens_ptr + current_node + 1,
                mask=active,
                other=-1,
            ).to(tl.int64)
            tl.store(sampled_ptr + output_index, accepted_token, mask=active)
            tl.store(accepted_nodes_ptr + output_index, current_node, mask=active)
            last_node = tl.where(active, current_node, last_node)
            current_node = tl.max(tl.where(nodes == current_node, next_nodes, -1), axis=0).to(
                tl.int32
            )

        bonus_row = tl.where(accepted_count == 0, 0, last_node + 1).to(tl.int32)
        bonus = tl.load(target_argmax_ptr + bonus_row).to(tl.int64)
        tl.store(sampled_ptr + accepted_count, bonus)
        tl.store(num_sampled_ptr, accepted_count + 1)
        tl.store(accepted_count_ptr, accepted_count)
        tl.store(accepted_leaf_physical_row_ptr, bonus_row)
        tl.store(accepted_leaf_state_slot_ptr, bonus_row + 1)

    _VERIFIER_KERNEL = greedy_best_first_b7_verify_kernel
    return _VERIFIER_KERNEL


def verify_best_first_b7_out(
    torch: Any,
    *,
    target_argmax: Any,
    transition_tokens: Any,
    parent: Any,
    sampled: Any,
    num_sampled: Any,
    outputs: B7VerifierOutputs,
    commit_capacity: int = B7_ROWS,
) -> None:
    """Run the exact v475 verifier into caller-owned commit outputs."""

    outputs.validate(torch, target_argmax.device)
    if (
        isinstance(commit_capacity, bool)
        or not isinstance(commit_capacity, int)
        or not 1 <= commit_capacity <= B7_ROWS
        or tuple(target_argmax.shape) != (B7_ROWS,)
        or target_argmax.dtype != torch.int64
        or tuple(transition_tokens.shape) != (B7_ROWS,)
        or transition_tokens.dtype not in (torch.int32, torch.int64)
        or tuple(parent.shape) != (B7_NODES,)
        or parent.dtype != torch.int32
        or tuple(sampled.shape) != (1, B7_ROWS)
        or sampled.dtype != torch.int64
        or tuple(num_sampled.shape) != (1,)
        or num_sampled.dtype != torch.int32
        or any(
            tensor.device != target_argmax.device or not tensor.is_contiguous()
            for tensor in (transition_tokens, parent, sampled, num_sampled)
        )
    ):
        raise B7GPUContractError("B7 verifier received an unsupported ABI")
    kernel = _build_verifier_kernel()
    kernel[(1,)](
        sampled,
        num_sampled,
        outputs.accepted_count,
        outputs.accepted_nodes,
        outputs.accepted_leaf_physical_row,
        outputs.accepted_leaf_state_slot,
        outputs.status,
        commit_capacity,
        target_argmax,
        transition_tokens,
        parent,
        NUM_NODES=B7_NODES,
        NUM_ROWS=B7_ROWS,
        num_warps=1,
    )


def _build_gdn_kernels() -> tuple[Any, Any, Any]:
    triton = importlib.import_module("triton")
    # Triton resolves language builtins in the JIT function's globals.
    globals()["tl"] = importlib.import_module("triton.language")

    @triton.jit
    def bulk_gdn_commit_kernel(
        branch_ptrs,
        checkpoint_ptrs,
        canonical_state_ptrs,
        canonical_conv_ptrs,
        canonical_conv_strides_ptr,
        source_index_ptrs,
        state_slot_counts_ptr,
        conv_slot_counts_ptr,
        published_slots_ptr,
        backup_state_ptrs,
        backup_conv_ptrs,
        backup_conv_strides_ptr,
        physical_row_ptr,
        state_slot_ptr,
        status_ptr,
        NUM_ROWS: tl.constexpr,
        STATE_ELEMENTS: tl.constexpr,
        STATE_SLOT_STRIDE: tl.constexpr,
        CONV_ELEMENTS: tl.constexpr,
        CONV_HISTORY_SIZE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        layer = tl.program_id(0)
        offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        physical_row = tl.load(physical_row_ptr).to(tl.int64)
        state_slot = tl.load(state_slot_ptr).to(tl.int64)
        source_index_ptr = tl.load(source_index_ptrs + layer).to(tl.pointer_type(tl.int64))
        source_index = tl.load(source_index_ptr).to(tl.int64)
        state_slot_count = tl.load(state_slot_counts_ptr + layer).to(tl.int64)
        conv_slot_count = tl.load(conv_slot_counts_ptr + layer).to(tl.int64)
        prevalidated = tl.load(status_ptr).to(tl.int32) == 1
        valid_leaf = prevalidated & (
            (physical_row >= 0)
            & (physical_row < NUM_ROWS)
            & (state_slot >= 1)
            & (state_slot <= NUM_ROWS)
            & (state_slot == physical_row + 1)
            & (source_index >= 0)
            & (source_index < state_slot_count)
            & (source_index < conv_slot_count)
        )
        status_ones = tl.full((BLOCK,), 1, tl.int32)
        status_zeros = tl.full((BLOCK,), 0, tl.int32)
        status_value = tl.where((offsets == 0) & ~valid_leaf, status_zeros, status_ones)
        status_addresses = status_ptr + tl.zeros((BLOCK,), tl.int32)
        tl.atomic_and(
            status_addresses,
            status_value,
            mask=(tl.program_id(1) == 0) & (offsets == 0),
        )
        tl.store(
            published_slots_ptr + layer,
            tl.where(valid_leaf, source_index, -1),
            mask=tl.program_id(1) == 0,
        )

        canonical_state = tl.load(canonical_state_ptrs + layer).to(tl.pointer_type(tl.float32))
        branch_state = tl.load(branch_ptrs + layer).to(tl.pointer_type(tl.float32))
        backup_state = tl.load(backup_state_ptrs + layer).to(tl.pointer_type(tl.float32))
        state_mask = (offsets < STATE_ELEMENTS) & valid_leaf
        canonical_state_offset = source_index * STATE_SLOT_STRIDE + offsets
        selected_state_offset = state_slot * STATE_ELEMENTS + offsets
        prior_state = tl.load(canonical_state + canonical_state_offset, mask=state_mask, other=0.0)
        selected_state = tl.load(branch_state + selected_state_offset, mask=state_mask, other=0.0)
        tl.store(backup_state + offsets, prior_state, mask=state_mask)
        tl.store(canonical_state + canonical_state_offset, selected_state, mask=state_mask)

        canonical_conv = tl.load(canonical_conv_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
        checkpoints = tl.load(checkpoint_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
        backup_conv = tl.load(backup_conv_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
        conv_offsets = offsets - STATE_ELEMENTS
        conv_mask = (offsets >= STATE_ELEMENTS) & (conv_offsets < CONV_ELEMENTS) & valid_leaf
        logical_q = conv_offsets // CONV_HISTORY_SIZE
        logical_h = conv_offsets % CONV_HISTORY_SIZE
        stride_base = layer * 3
        canonical_stride_slot = tl.load(canonical_conv_strides_ptr + stride_base).to(tl.int64)
        canonical_stride_q = tl.load(canonical_conv_strides_ptr + stride_base + 1).to(tl.int64)
        canonical_stride_h = tl.load(canonical_conv_strides_ptr + stride_base + 2).to(tl.int64)
        backup_stride_q = tl.load(backup_conv_strides_ptr + stride_base + 1).to(tl.int64)
        backup_stride_h = tl.load(backup_conv_strides_ptr + stride_base + 2).to(tl.int64)
        canonical_conv_offset = (
            source_index * canonical_stride_slot
            + logical_q * canonical_stride_q
            + logical_h * canonical_stride_h
        )
        selected_conv_offset = physical_row * CONV_ELEMENTS + conv_offsets
        backup_conv_offset = logical_q * backup_stride_q + logical_h * backup_stride_h
        prior_conv = tl.load(canonical_conv + canonical_conv_offset, mask=conv_mask, other=0.0)
        selected_conv = tl.load(checkpoints + selected_conv_offset, mask=conv_mask, other=0.0)
        tl.store(backup_conv + backup_conv_offset, prior_conv, mask=conv_mask)
        tl.store(canonical_conv + canonical_conv_offset, selected_conv, mask=conv_mask)

    @triton.jit
    def bulk_gdn_rollback_kernel(
        canonical_state_ptrs,
        canonical_conv_ptrs,
        canonical_conv_strides_ptr,
        backup_state_ptrs,
        backup_conv_ptrs,
        backup_conv_strides_ptr,
        published_slots_ptr,
        state_slot_counts_ptr,
        conv_slot_counts_ptr,
        STATE_ELEMENTS: tl.constexpr,
        STATE_SLOT_STRIDE: tl.constexpr,
        CONV_ELEMENTS: tl.constexpr,
        CONV_HISTORY_SIZE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        layer = tl.program_id(0)
        offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        canonical_state = tl.load(canonical_state_ptrs + layer).to(tl.pointer_type(tl.float32))
        backup_state = tl.load(backup_state_ptrs + layer).to(tl.pointer_type(tl.float32))
        published_slot = tl.load(published_slots_ptr + layer).to(tl.int64)
        state_slot_count = tl.load(state_slot_counts_ptr + layer).to(tl.int64)
        conv_slot_count = tl.load(conv_slot_counts_ptr + layer).to(tl.int64)
        valid_state_slot = (published_slot >= 0) & (published_slot < state_slot_count)
        state_mask = (offsets < STATE_ELEMENTS) & valid_state_slot
        prior_state = tl.load(backup_state + offsets, mask=state_mask, other=0.0)
        tl.store(
            canonical_state + published_slot * STATE_SLOT_STRIDE + offsets,
            prior_state,
            mask=state_mask,
        )

        canonical_conv = tl.load(canonical_conv_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
        backup_conv = tl.load(backup_conv_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
        conv_offsets = offsets - STATE_ELEMENTS
        valid_conv_slot = (published_slot >= 0) & (published_slot < conv_slot_count)
        conv_mask = (offsets >= STATE_ELEMENTS) & (conv_offsets < CONV_ELEMENTS) & valid_conv_slot
        logical_q = conv_offsets // CONV_HISTORY_SIZE
        logical_h = conv_offsets % CONV_HISTORY_SIZE
        stride_base = layer * 3
        canonical_stride_slot = tl.load(canonical_conv_strides_ptr + stride_base).to(tl.int64)
        canonical_stride_q = tl.load(canonical_conv_strides_ptr + stride_base + 1).to(tl.int64)
        canonical_stride_h = tl.load(canonical_conv_strides_ptr + stride_base + 2).to(tl.int64)
        backup_stride_q = tl.load(backup_conv_strides_ptr + stride_base + 1).to(tl.int64)
        backup_stride_h = tl.load(backup_conv_strides_ptr + stride_base + 2).to(tl.int64)
        canonical_conv_offset = (
            published_slot * canonical_stride_slot
            + logical_q * canonical_stride_q
            + logical_h * canonical_stride_h
        )
        backup_conv_offset = logical_q * backup_stride_q + logical_h * backup_stride_h
        prior_conv = tl.load(backup_conv + backup_conv_offset, mask=conv_mask, other=0.0)
        tl.store(
            canonical_conv + canonical_conv_offset,
            prior_conv,
            mask=conv_mask,
        )

    return triton, bulk_gdn_commit_kernel, bulk_gdn_rollback_kernel


def _build_kv_kernels() -> tuple[Any, Any, Any]:
    triton = importlib.import_module("triton")
    globals()["tl"] = importlib.import_module("triton.language")

    @triton.jit
    def bulk_kv_gather_kernel(
        storage_base_ptr,
        cache_offsets_ptr,
        group_indices_ptr,
        slot_mappings_ptr,
        accepted_nodes_ptr,
        accepted_count_ptr,
        staging_key_ptr,
        staging_value_ptr,
        status_ptr,
        gdn_source_index_ptrs,
        gdn_state_slot_counts_ptr,
        gdn_conv_slot_counts_ptr,
        accepted_leaf_physical_row_ptr,
        accepted_leaf_state_slot_ptr,
        NUM_NODES: tl.constexpr,
        NUM_ROWS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        STRIDE_KEY_BLOCK: tl.constexpr,
        STRIDE_KEY_HEAD: tl.constexpr,
        STRIDE_KEY_CHUNK: tl.constexpr,
        STRIDE_KEY_TOKEN: tl.constexpr,
        STRIDE_KEY_LANE: tl.constexpr,
        STRIDE_VALUE_BLOCK: tl.constexpr,
        STRIDE_VALUE_HEAD: tl.constexpr,
        STRIDE_VALUE_DIM: tl.constexpr,
        STRIDE_VALUE_TOKEN: tl.constexpr,
        PAGE_TOKENS: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        SLOT_GROUP_STRIDE: tl.constexpr,
        SLOT_TOKEN_STRIDE: tl.constexpr,
        ELEMENTS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        layer = tl.program_id(0)
        accepted_offset = tl.program_id(1)
        element = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
        accepted_count = tl.load(accepted_count_ptr).to(tl.int32)
        count_valid = (accepted_count >= 0) & (accepted_count <= NUM_NODES)
        requested = count_valid & (accepted_offset < accepted_count)
        node = tl.load(
            accepted_nodes_ptr + accepted_offset,
            mask=requested,
            other=-1,
        ).to(tl.int32)
        physical_row = node + 1
        node_valid = (node >= 0) & (node < NUM_NODES)
        group_index = tl.load(group_indices_ptr + layer).to(tl.int64)
        source_slot = tl.load(
            slot_mappings_ptr + group_index * SLOT_GROUP_STRIDE + physical_row * SLOT_TOKEN_STRIDE,
            mask=requested & node_valid,
            other=-1,
        ).to(tl.int64)
        destination_row = accepted_offset + 1
        destination_slot = tl.load(
            slot_mappings_ptr
            + group_index * SLOT_GROUP_STRIDE
            + destination_row * SLOT_TOKEN_STRIDE,
            mask=requested,
            other=-1,
        ).to(tl.int64)
        source_page = source_slot // PAGE_TOKENS
        destination_page = destination_slot // PAGE_TOKENS
        slots_valid = (
            (source_slot >= 0)
            & (source_page >= 0)
            & (source_page < NUM_BLOCKS)
            & (destination_slot >= 0)
            & (destination_page >= 0)
            & (destination_page < NUM_BLOCKS)
        )
        valid_lane = requested & node_valid & slots_valid
        invalid = (~count_valid) | (requested & (~node_valid | ~slots_valid))
        leaf_row = tl.load(accepted_leaf_physical_row_ptr).to(tl.int64)
        leaf_slot = tl.load(accepted_leaf_state_slot_ptr).to(tl.int64)
        gdn_valid = (
            (leaf_row >= 0)
            & (leaf_row < NUM_ROWS)
            & (leaf_slot >= 1)
            & (leaf_slot <= NUM_ROWS)
            & (leaf_slot == leaf_row + 1)
        )
        for gdn_offset in tl.static_range(0, 3):
            gdn_layer = layer * 3 + gdn_offset
            source_ptr = tl.load(gdn_source_index_ptrs + gdn_layer).to(tl.pointer_type(tl.int64))
            source_index = tl.load(source_ptr).to(tl.int64)
            state_slots = tl.load(gdn_state_slot_counts_ptr + gdn_layer).to(tl.int64)
            conv_slots = tl.load(gdn_conv_slot_counts_ptr + gdn_layer).to(tl.int64)
            gdn_valid &= (
                (source_index >= 0) & (source_index < state_slots) & (source_index < conv_slots)
            )
        invalid |= (accepted_offset == 0) & ~gdn_valid
        status_ones = tl.full((BLOCK,), 1, tl.int32)
        status_zeros = tl.full((BLOCK,), 0, tl.int32)
        status_value = tl.where((element == 0) & invalid, status_zeros, status_ones)
        status_addresses = status_ptr + tl.zeros((BLOCK,), tl.int32)
        tl.atomic_and(
            status_addresses,
            status_value,
            mask=(tl.program_id(2) == 0) & (element == 0),
        )
        active = valid_lane & (element < ELEMENTS)
        page = source_page
        token = source_slot % PAGE_TOKENS
        head = element // HEAD_DIM
        dimension = element % HEAD_DIM
        chunk = dimension // 16
        lane = dimension % 16
        key_offset = (
            page * STRIDE_KEY_BLOCK
            + head * STRIDE_KEY_HEAD
            + chunk * STRIDE_KEY_CHUNK
            + token * STRIDE_KEY_TOKEN
            + lane * STRIDE_KEY_LANE
        )
        value_offset = (
            page * STRIDE_VALUE_BLOCK
            + head * STRIDE_VALUE_HEAD
            + dimension * STRIDE_VALUE_DIM
            + token * STRIDE_VALUE_TOKEN
        )
        key_offset_bytes = tl.load(cache_offsets_ptr + layer * 2).to(tl.int64)
        value_offset_bytes = tl.load(cache_offsets_ptr + layer * 2 + 1).to(tl.int64)
        key_cache = storage_base_ptr + key_offset_bytes
        value_cache = storage_base_ptr + value_offset_bytes
        staging_offset = (layer * NUM_NODES + accepted_offset) * ELEMENTS + element
        key_byte = tl.load(key_cache + key_offset, mask=active, other=0)
        value_byte = tl.load(value_cache + value_offset, mask=active, other=0)
        tl.store(staging_key_ptr + staging_offset, key_byte, mask=active)
        tl.store(staging_value_ptr + staging_offset, value_byte, mask=active)

    @triton.jit
    def bulk_kv_publish_kernel(
        storage_base_ptr,
        cache_offsets_ptr,
        group_indices_ptr,
        slot_mappings_ptr,
        accepted_count_ptr,
        staging_key_ptr,
        staging_value_ptr,
        status_ptr,
        NUM_NODES: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        STRIDE_KEY_BLOCK: tl.constexpr,
        STRIDE_KEY_HEAD: tl.constexpr,
        STRIDE_KEY_CHUNK: tl.constexpr,
        STRIDE_KEY_TOKEN: tl.constexpr,
        STRIDE_KEY_LANE: tl.constexpr,
        STRIDE_VALUE_BLOCK: tl.constexpr,
        STRIDE_VALUE_HEAD: tl.constexpr,
        STRIDE_VALUE_DIM: tl.constexpr,
        STRIDE_VALUE_TOKEN: tl.constexpr,
        PAGE_TOKENS: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        SLOT_GROUP_STRIDE: tl.constexpr,
        SLOT_TOKEN_STRIDE: tl.constexpr,
        ELEMENTS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        layer = tl.program_id(0)
        accepted_offset = tl.program_id(1)
        element = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
        accepted_count = tl.load(accepted_count_ptr).to(tl.int32)
        status = tl.load(status_ptr).to(tl.int32)
        count_valid = (accepted_count >= 0) & (accepted_count <= NUM_NODES)
        requested = count_valid & (accepted_offset < accepted_count)
        destination_row = accepted_offset + 1
        group_index = tl.load(group_indices_ptr + layer).to(tl.int64)
        slot = tl.load(
            slot_mappings_ptr
            + group_index * SLOT_GROUP_STRIDE
            + destination_row * SLOT_TOKEN_STRIDE,
            mask=requested,
            other=-1,
        ).to(tl.int64)
        page = slot // PAGE_TOKENS
        token = slot % PAGE_TOKENS
        slot_valid = (slot >= 0) & (page >= 0) & (page < NUM_BLOCKS)
        active = requested & slot_valid & (status == 1) & (element < ELEMENTS)
        head = element // HEAD_DIM
        dimension = element % HEAD_DIM
        chunk = dimension // 16
        lane = dimension % 16
        key_offset = (
            page * STRIDE_KEY_BLOCK
            + head * STRIDE_KEY_HEAD
            + chunk * STRIDE_KEY_CHUNK
            + token * STRIDE_KEY_TOKEN
            + lane * STRIDE_KEY_LANE
        )
        value_offset = (
            page * STRIDE_VALUE_BLOCK
            + head * STRIDE_VALUE_HEAD
            + dimension * STRIDE_VALUE_DIM
            + token * STRIDE_VALUE_TOKEN
        )
        key_offset_bytes = tl.load(cache_offsets_ptr + layer * 2).to(tl.int64)
        value_offset_bytes = tl.load(cache_offsets_ptr + layer * 2 + 1).to(tl.int64)
        key_cache = storage_base_ptr + key_offset_bytes
        value_cache = storage_base_ptr + value_offset_bytes
        staging_offset = (layer * NUM_NODES + accepted_offset) * ELEMENTS + element
        key_byte = tl.load(staging_key_ptr + staging_offset, mask=active, other=0)
        value_byte = tl.load(staging_value_ptr + staging_offset, mask=active, other=0)
        tl.store(key_cache + key_offset, key_byte, mask=active)
        tl.store(value_cache + value_offset, value_byte, mask=active)

    return triton, bulk_kv_gather_kernel, bulk_kv_publish_kernel


def build_gdn_pointer_tables(torch: Any, model: Any) -> B7GDNPointerTables:
    """Discover and bind all 48 exact B7 GDN layers once, off the hot path."""

    def storage_span_fits(tensor: Any) -> bool:
        shape = tuple(int(value) for value in tensor.shape)
        strides = tuple(int(value) for value in tensor.stride())
        if len(shape) != len(strides) or any(value <= 0 for value in shape):
            return False
        if any(value < 0 for value in strides):
            return False
        storage = tensor.untyped_storage()
        byte_offset = int(tensor.data_ptr()) - int(storage.data_ptr())
        if byte_offset < 0 or byte_offset % int(tensor.element_size()) != 0:
            return False
        last_element = sum(
            (extent - 1) * stride for extent, stride in zip(shape, strides, strict=True)
        )
        return byte_offset + (last_element + 1) * int(tensor.element_size()) <= int(
            storage.nbytes()
        )

    layers = []
    for module in model.modules():
        ensure = getattr(module, "_ensure_recoverssm_trusted_replay_buffers", None)
        if callable(ensure) and hasattr(module, "_b7_tree_state_storage"):
            ensure()
            if module._b7_tree_state_storage is not None:
                layers.append(module)
    if len(layers) != GDN_LAYERS:
        raise B7GPUContractError(
            f"B7 bulk commit requires exactly {GDN_LAYERS} GDN layers, got {len(layers)}"
        )

    branch_states = []
    checkpoints = []
    canonical_states = []
    canonical_state_slot_stride = None
    canonical_convs = []
    backup_states = []
    backup_convs = []
    canonical_conv_strides = []
    backup_conv_strides = []
    source_indices = []
    state_slot_counts = []
    conv_slot_counts = []
    expected_device = None
    for ordinal, layer in enumerate(layers):
        branch = layer._b7_tree_state_storage
        checkpoint = layer._recoverssm_trusted_conv_row_storage
        state = layer.kv_cache[1]
        conv = layer.kv_cache[0]
        if tuple(conv.shape[1:]) == (CONV_HISTORY, QKV_WIDTH):
            conv = conv.transpose(-1, -2)
        state_backup = layer._recoverssm_trusted_transaction_backup_storage
        conv_backup = layer._recoverssm_trusted_conv_transaction_backup_storage
        source_index = layer._recoverssm_trusted_source_index_long
        canonical_state_stride = tuple(int(value) for value in state.stride())
        canonical_conv_stride = tuple(int(value) for value in conv.stride())
        backup_conv_stride = tuple(int(value) for value in conv_backup.stride())
        state_elements = VALUE_HEADS * HEAD_DIM * HEAD_DIM
        conv_elements = QKV_WIDTH * CONV_HISTORY
        dense_state_stride = (state_elements, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1)
        packed_state_stride = (PACKED_GDN_PAGE_BYTES // 4, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1)
        dense_ds_stride = (conv_elements, CONV_HISTORY, 1)
        dense_sd_stride = (conv_elements, 1, QKV_WIDTH)
        packed_ds_stride = (PACKED_GDN_PAGE_BYTES // 2, CONV_HISTORY, 1)
        packed_sd_stride = (PACKED_GDN_PAGE_BYTES // 2, 1, QKV_WIDTH)
        dense_storage = canonical_state_stride == dense_state_stride and canonical_conv_stride in (
            dense_ds_stride,
            dense_sd_stride,
        )
        packed_storage = (
            canonical_state_stride == packed_state_stride
            and canonical_conv_stride in (packed_ds_stride, packed_sd_stride)
            and state.data_ptr() == conv.data_ptr() + conv_elements * 2
            and state.untyped_storage().data_ptr() == conv.untyped_storage().data_ptr()
            and int(conv.data_ptr()) - int(conv.untyped_storage().data_ptr()) >= 0
            and int(conv.data_ptr())
            - int(conv.untyped_storage().data_ptr())
            + int(conv.shape[0]) * PACKED_GDN_PAGE_BYTES
            <= int(conv.untyped_storage().nbytes())
        )
        if canonical_state_slot_stride is None:
            canonical_state_slot_stride = canonical_state_stride[0]
        if expected_device is None:
            expected_device = state.device
        if (
            tuple(branch.shape) != (9, VALUE_HEADS, HEAD_DIM, HEAD_DIM)
            or tuple(checkpoint.shape) != (B7_ROWS, QKV_WIDTH, CONV_HISTORY)
            or tuple(state.shape[1:]) != (VALUE_HEADS, HEAD_DIM, HEAD_DIM)
            or tuple(conv.shape[1:]) != (QKV_WIDTH, CONV_HISTORY)
            or int(state.shape[0]) != int(conv.shape[0])
            or tuple(state_backup.shape) != (1, VALUE_HEADS, HEAD_DIM, HEAD_DIM)
            or tuple(conv_backup.shape) != (1, QKV_WIDTH, CONV_HISTORY)
            or tuple(source_index.shape) != (1,)
            or source_index.dtype != torch.int64
            or state.device != expected_device
            or any(
                tensor.device != expected_device
                for tensor in (
                    branch,
                    checkpoint,
                    conv,
                    state_backup,
                    conv_backup,
                    source_index,
                )
            )
            or not all(tensor.is_contiguous() for tensor in (branch, checkpoint, state_backup))
            or tuple(int(value) for value in checkpoint.stride()) != dense_ds_stride
            or not (dense_storage or packed_storage)
            or canonical_state_stride[0] != canonical_state_slot_stride
            or backup_conv_stride[1:] != canonical_conv_stride[1:]
            or not all(
                storage_span_fits(tensor)
                for tensor in (
                    branch,
                    checkpoint,
                    state,
                    conv,
                    state_backup,
                    conv_backup,
                    source_index,
                )
            )
            or state.dtype != torch.float32
            or branch.dtype != torch.float32
            or state_backup.dtype != torch.float32
            or conv.dtype != torch.bfloat16
            or checkpoint.dtype != torch.bfloat16
            or conv_backup.dtype != torch.bfloat16
        ):
            raise B7GPUContractError("B7 GDN layer storage differs from the stride-aware B7 ABI")
        layer._qwen_b7_layer_ordinal = ordinal
        branch_states.append(branch)
        checkpoints.append(checkpoint)
        canonical_states.append(state)
        canonical_convs.append(conv)
        backup_states.append(state_backup)
        backup_convs.append(conv_backup)
        canonical_conv_strides.append(canonical_conv_stride)
        backup_conv_strides.append(backup_conv_stride)
        source_indices.append(source_index)
        state_slot_counts.append(int(state.shape[0]))
        conv_slot_counts.append(int(conv.shape[0]))

    device = expected_device

    def pointer_table(tensors: list[Any]) -> Any:
        return torch.tensor(
            [tensor.data_ptr() for tensor in tensors], dtype=torch.int64, device=device
        )

    triton, commit_kernel, rollback_kernel = _build_gdn_kernels()
    return B7GDNPointerTables(
        layers=tuple(layers),
        branch_states=pointer_table(branch_states),
        conv_checkpoints=pointer_table(checkpoints),
        canonical_states=pointer_table(canonical_states),
        canonical_state_slot_stride=int(canonical_state_slot_stride),
        canonical_convs=pointer_table(canonical_convs),
        canonical_conv_strides=torch.tensor(
            canonical_conv_strides, dtype=torch.int64, device=device
        ),
        source_indices=pointer_table(source_indices),
        state_slot_counts=torch.tensor(state_slot_counts, dtype=torch.int64, device=device),
        conv_slot_counts=torch.tensor(conv_slot_counts, dtype=torch.int64, device=device),
        published_slots=torch.full((GDN_LAYERS,), -1, dtype=torch.int64, device=device),
        backup_states=pointer_table(backup_states),
        backup_convs=pointer_table(backup_convs),
        backup_conv_strides=torch.tensor(backup_conv_strides, dtype=torch.int64, device=device),
        commit_kernel=commit_kernel,
        rollback_kernel=rollback_kernel,
        triton=triton,
    )


def build_kv_pointer_tables(
    torch: Any,
    kv_caches_by_layer: dict[str, Any],
    *,
    layer_group_indices: dict[str, int],
    page_tokens: int,
) -> B7KVPointerTables:
    """Bind the 16 FP8 Quest cache views once without gathering their contents."""

    from vllm.v1.attention.ops.paged_attn import PagedAttention

    if page_tokens != QUALIFIED_PAGE_TOKENS:
        raise B7GPUContractError(
            f"B7 KV compaction requires qualified page_tokens={QUALIFIED_PAGE_TOKENS}"
        )
    if any(name not in kv_caches_by_layer for name in QUEST_LAYER_NAMES):
        missing = tuple(name for name in QUEST_LAYER_NAMES if name not in kv_caches_by_layer)
        raise B7GPUContractError(f"B7 Quest KV layers are absent: {missing}")
    candidates = []
    group_indices_cpu = []
    storage_identity = None
    storage_owner = None
    for layer_name in QUEST_LAYER_NAMES:
        group_index = layer_group_indices.get(layer_name)
        if isinstance(group_index, bool) or not isinstance(group_index, int) or group_index < 0:
            raise B7GPUContractError(f"B7 Quest cache-group index for {layer_name} is invalid")
        group_indices_cpu.append(group_index)
        kv_cache = kv_caches_by_layer[layer_name]
        if (
            getattr(kv_cache, "ndim", -1) != 5
            or tuple(kv_cache.shape[:1]) != (2,)
            or int(kv_cache.shape[2]) != page_tokens
            or int(kv_cache.shape[3]) != QUEST_KV_HEADS
            or int(kv_cache.shape[4]) != QUEST_HEAD_DIM
            or kv_cache.element_size() != 1
            or kv_cache.device.type != "cuda"
        ):
            raise B7GPUContractError(
                f"B7 Quest KV layer {layer_name} differs from the qualified FP8 ABI"
            )
        storage = kv_cache.untyped_storage()
        observed_storage = (int(storage.data_ptr()), int(storage.nbytes()), str(kv_cache.device))
        if storage_identity is None:
            storage_identity = observed_storage
            storage_owner = kv_cache
        elif storage_identity != observed_storage:
            raise B7GPUContractError("B7 Quest layers do not share one qualified backing store")
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache, QUEST_KV_HEADS, QUEST_HEAD_DIM
        )
        candidates.append((layer_name, key_cache.view(torch.uint8), value_cache.view(torch.uint8)))
    if len(candidates) != QUEST_LAYERS:
        raise B7GPUContractError("B7 Quest layer enumeration changed")
    key_strides = tuple(int(value) for value in candidates[0][1].stride())
    value_strides = tuple(int(value) for value in candidates[0][2].stride())
    if any(
        tuple(int(value) for value in key.stride()) != key_strides
        or tuple(int(value) for value in value.stride()) != value_strides
        or tuple(key.shape[1:]) != (QUEST_KV_HEADS, QUEST_HEAD_DIM // 16, page_tokens, 16)
        or tuple(value.shape[1:]) != (QUEST_KV_HEADS, QUEST_HEAD_DIM, page_tokens)
        for _, key, value in candidates
    ):
        raise B7GPUContractError("B7 Quest cache layouts are not uniform")
    num_blocks = int(candidates[0][1].shape[0])
    if num_blocks <= 0 or any(
        int(key.shape[0]) != num_blocks or int(value.shape[0]) != num_blocks
        for _, key, value in candidates
    ):
        raise B7GPUContractError("B7 Quest cache block counts are not uniform")
    device = candidates[0][1].device
    layer_names = tuple(name for name, _, _ in candidates)
    if set(layer_names) - set(layer_group_indices):
        raise B7GPUContractError("B7 Quest layer-to-cache-group mapping is incomplete")
    if storage_identity is None or storage_owner is None:
        raise B7GPUContractError("B7 Quest common backing storage is absent")
    storage_base, storage_nbytes, _ = storage_identity
    offsets = []
    for _, key, value in candidates:
        pair = (int(key.data_ptr()) - storage_base, int(value.data_ptr()) - storage_base)
        for tensor, offset in zip((key, value), pair, strict=True):
            byte_span = 1 + sum(
                (int(size) - 1) * int(stride)
                for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
            )
            if offset < 0 or offset + byte_span > storage_nbytes:
                raise B7GPUContractError("B7 Quest cache view escapes its backing storage")
        offsets.append(pair)
    storage_view = torch.empty((0,), dtype=torch.uint8, device=device).set_(
        storage_owner.untyped_storage(), 0, (storage_nbytes,), (1,)
    )
    triton, gather_kernel, publish_kernel = _build_kv_kernels()
    return B7KVPointerTables(
        layer_names=layer_names,
        kv_cache_views=tuple(kv_caches_by_layer[name] for name in layer_names),
        key_views=tuple(key for _, key, _ in candidates),
        value_views=tuple(value for _, _, value in candidates),
        storage_view=storage_view,
        cache_offsets=torch.tensor(offsets, dtype=torch.int64, device=device),
        group_indices=torch.tensor(
            group_indices_cpu,
            dtype=torch.int64,
            device=device,
        ),
        group_indices_cpu=tuple(group_indices_cpu),
        slot_mapping_view=None,
        staging_key=torch.empty(
            (QUEST_LAYERS, B7_NODES, QUEST_KV_ELEMENTS),
            dtype=torch.uint8,
            device=device,
        ),
        staging_value=torch.empty(
            (QUEST_LAYERS, B7_NODES, QUEST_KV_ELEMENTS),
            dtype=torch.uint8,
            device=device,
        ),
        slot_identity=None,
        num_blocks=num_blocks,
        page_tokens=page_tokens,
        key_strides=key_strides,
        value_strides=value_strides,
        gather_kernel=gather_kernel,
        publish_kernel=publish_kernel,
        triton=triton,
    )


def bind_kv_slot_mappings_once(
    torch: Any,
    tables: B7KVPointerTables,
    slot_mappings: Any,
) -> None:
    """Bind the original persistent 2-D int64 group slot tensor once."""

    if (
        getattr(slot_mappings, "ndim", -1) != 2
        or slot_mappings.shape[1] < B7_ROWS
        or slot_mappings.dtype != torch.int64
        or slot_mappings.device != tables.staging_key.device
        or int(slot_mappings.stride(1)) != 1
        or int(slot_mappings.stride(0)) < int(slot_mappings.shape[1])
        or int(slot_mappings.stride(0)) <= 0
        or int(tables.group_indices.numel()) != QUEST_LAYERS
        or any(
            index < 0 or index >= int(slot_mappings.shape[0]) for index in tables.group_indices_cpu
        )
    ):
        raise B7GPUContractError("B7 group slot mappings have an unsupported ABI")
    storage = slot_mappings.untyped_storage()
    last_element = (
        int(slot_mappings.storage_offset())
        + (int(slot_mappings.shape[0]) - 1) * int(slot_mappings.stride(0))
        + (int(slot_mappings.shape[1]) - 1) * int(slot_mappings.stride(1))
    )
    if last_element < 0 or (last_element + 1) * slot_mappings.element_size() > storage.nbytes():
        raise B7GPUContractError("B7 group slot mappings escape their backing storage")
    identity = _identity(slot_mappings)
    if tables.slot_identity is not None:
        if identity != tables.slot_identity:
            raise B7GPUContractError("B7 Quest slot-mapping storage was rebound")
        return
    tables.slot_mapping_view = slot_mappings
    tables.slot_identity = identity


def bulk_gdn_commit(
    tables: B7GDNPointerTables,
    outputs: B7VerifierOutputs,
) -> None:
    """Publish all 48 GDN leaves with one stream-ordered launch."""

    state_elements = VALUE_HEADS * HEAD_DIM * HEAD_DIM
    conv_elements = QKV_WIDTH * CONV_HISTORY
    block = 1024
    grid = (GDN_LAYERS, tables.triton.cdiv(state_elements + conv_elements, block))
    tables.commit_kernel[grid](
        tables.branch_states,
        tables.conv_checkpoints,
        tables.canonical_states,
        tables.canonical_convs,
        tables.canonical_conv_strides,
        tables.source_indices,
        tables.state_slot_counts,
        tables.conv_slot_counts,
        tables.published_slots,
        tables.backup_states,
        tables.backup_convs,
        tables.backup_conv_strides,
        outputs.accepted_leaf_physical_row,
        outputs.accepted_leaf_state_slot,
        outputs.status,
        NUM_ROWS=B7_ROWS,
        STATE_ELEMENTS=state_elements,
        STATE_SLOT_STRIDE=tables.canonical_state_slot_stride,
        CONV_ELEMENTS=conv_elements,
        CONV_HISTORY_SIZE=CONV_HISTORY,
        BLOCK=block,
        num_warps=4,
    )


def bulk_gdn_rollback(tables: B7GDNPointerTables) -> None:
    state_elements = VALUE_HEADS * HEAD_DIM * HEAD_DIM
    conv_elements = QKV_WIDTH * CONV_HISTORY
    block = 1024
    grid = (GDN_LAYERS, tables.triton.cdiv(state_elements + conv_elements, block))
    tables.rollback_kernel[grid](
        tables.canonical_states,
        tables.canonical_convs,
        tables.canonical_conv_strides,
        tables.backup_states,
        tables.backup_convs,
        tables.backup_conv_strides,
        tables.published_slots,
        tables.state_slot_counts,
        tables.conv_slot_counts,
        STATE_ELEMENTS=state_elements,
        STATE_SLOT_STRIDE=tables.canonical_state_slot_stride,
        CONV_ELEMENTS=conv_elements,
        CONV_HISTORY_SIZE=CONV_HISTORY,
        BLOCK=block,
        num_warps=4,
    )


def bulk_kv_gather(
    tables: B7KVPointerTables,
    gdn: B7GDNPointerTables,
    outputs: B7VerifierOutputs,
) -> None:
    """Validate every dynamic source and gather all KV bytes into disjoint staging."""

    if tables.slot_identity is None or tables.slot_mapping_view is None:
        raise B7GPUContractError("B7 Quest slot mappings were not bound")
    block = 256
    grid = (QUEST_LAYERS, B7_NODES, tables.triton.cdiv(QUEST_KV_ELEMENTS, block))
    tables.gather_kernel[grid](
        tables.storage_view,
        tables.cache_offsets,
        tables.group_indices,
        tables.slot_mapping_view,
        outputs.accepted_nodes,
        outputs.accepted_count,
        tables.staging_key,
        tables.staging_value,
        outputs.status,
        gdn.source_indices,
        gdn.state_slot_counts,
        gdn.conv_slot_counts,
        outputs.accepted_leaf_physical_row,
        outputs.accepted_leaf_state_slot,
        NUM_NODES=B7_NODES,
        NUM_ROWS=B7_ROWS,
        HEAD_DIM=QUEST_HEAD_DIM,
        STRIDE_KEY_BLOCK=tables.key_strides[0],
        STRIDE_KEY_HEAD=tables.key_strides[1],
        STRIDE_KEY_CHUNK=tables.key_strides[2],
        STRIDE_KEY_TOKEN=tables.key_strides[3],
        STRIDE_KEY_LANE=tables.key_strides[4],
        STRIDE_VALUE_BLOCK=tables.value_strides[0],
        STRIDE_VALUE_HEAD=tables.value_strides[1],
        STRIDE_VALUE_DIM=tables.value_strides[2],
        STRIDE_VALUE_TOKEN=tables.value_strides[3],
        PAGE_TOKENS=tables.page_tokens,
        NUM_BLOCKS=tables.num_blocks,
        SLOT_GROUP_STRIDE=tables.slot_mapping_view.stride(0),
        SLOT_TOKEN_STRIDE=tables.slot_mapping_view.stride(1),
        ELEMENTS=QUEST_KV_ELEMENTS,
        BLOCK=block,
        num_warps=4,
    )


def bulk_kv_publish(tables: B7KVPointerTables, outputs: B7VerifierOutputs) -> None:
    """Publish prevalidated staged KV bytes without reading device status on the host."""

    if tables.slot_identity is None or tables.slot_mapping_view is None:
        raise B7GPUContractError("B7 Quest slot mappings were not bound")
    block = 256
    grid = (QUEST_LAYERS, B7_NODES, tables.triton.cdiv(QUEST_KV_ELEMENTS, block))
    tables.publish_kernel[grid](
        tables.storage_view,
        tables.cache_offsets,
        tables.group_indices,
        tables.slot_mapping_view,
        outputs.accepted_count,
        tables.staging_key,
        tables.staging_value,
        outputs.status,
        NUM_NODES=B7_NODES,
        HEAD_DIM=QUEST_HEAD_DIM,
        STRIDE_KEY_BLOCK=tables.key_strides[0],
        STRIDE_KEY_HEAD=tables.key_strides[1],
        STRIDE_KEY_CHUNK=tables.key_strides[2],
        STRIDE_KEY_TOKEN=tables.key_strides[3],
        STRIDE_KEY_LANE=tables.key_strides[4],
        STRIDE_VALUE_BLOCK=tables.value_strides[0],
        STRIDE_VALUE_HEAD=tables.value_strides[1],
        STRIDE_VALUE_DIM=tables.value_strides[2],
        STRIDE_VALUE_TOKEN=tables.value_strides[3],
        PAGE_TOKENS=tables.page_tokens,
        NUM_BLOCKS=tables.num_blocks,
        SLOT_GROUP_STRIDE=tables.slot_mapping_view.stride(0),
        SLOT_TOKEN_STRIDE=tables.slot_mapping_view.stride(1),
        ELEMENTS=QUEST_KV_ELEMENTS,
        BLOCK=block,
        num_warps=4,
    )


@dataclass
class B7BulkCommitRuntime:
    """Prebuilt state for failure-atomic accepted-path publication."""

    torch: Any
    gdn: B7GDNPointerTables
    kv: B7KVPointerTables
    outputs: B7VerifierOutputs

    @classmethod
    def build(
        cls,
        torch: Any,
        *,
        model: Any,
        kv_caches_by_layer: dict[str, Any],
        layer_group_indices: dict[str, int],
        page_tokens: int,
        device: Any,
    ) -> B7BulkCommitRuntime:
        outputs = B7VerifierOutputs.allocate(torch, device)
        outputs.validate(torch, device)
        return cls(
            torch=torch,
            gdn=build_gdn_pointer_tables(torch, model),
            kv=build_kv_pointer_tables(
                torch,
                kv_caches_by_layer,
                layer_group_indices=layer_group_indices,
                page_tokens=page_tokens,
            ),
            outputs=outputs,
        )

    def bind_slot_mappings_once(self, torch: Any, slot_mappings: Any) -> None:
        bind_kv_slot_mappings_once(torch, self.kv, slot_mappings)

    def commit(self) -> None:
        """Prevalidate/gather, assert asynchronously, then publish GDN and KV."""

        # Gather mutates staging only.  It also validates every dynamic GDN
        # source slot and every requested KV source/destination slot.  The
        # status assertion is stream ordered and does not read a device scalar
        # on the host.  Both publication kernels independently mask all stores
        # unless that one persistent word remains one.
        bulk_kv_gather(self.kv, self.gdn, self.outputs)
        self.torch._assert_async(
            self.outputs.status,
            "best-first B7 device prevalidation rejected GDN/KV commit metadata",
        )
        bulk_gdn_commit(self.gdn, self.outputs)
        try:
            bulk_kv_publish(self.kv, self.outputs)
        except BaseException:
            bulk_gdn_rollback(self.gdn)
            raise


@dataclass
class B7GPUWorkspace:
    """One runner-owned stable-address topology, verifier, and commit workspace."""

    bulk: B7BulkCommitRuntime
    parent: Any
    ancestor_mask: Any
    logical_offsets: Any
    anchor_position: Any
    parent_cpu: Any
    ancestor_mask_cpu: Any
    logical_offsets_cpu: Any
    commit_generation: Any
    active_request_id: str | None = None

    @classmethod
    def build(
        cls,
        torch: Any,
        *,
        model: Any,
        kv_caches_by_layer: dict[str, Any],
        layer_group_indices: dict[str, int],
        device: Any,
    ) -> B7GPUWorkspace:
        bulk = B7BulkCommitRuntime.build(
            torch,
            model=model,
            kv_caches_by_layer=kv_caches_by_layer,
            layer_group_indices=layer_group_indices,
            page_tokens=QUALIFIED_PAGE_TOKENS,
            device=device,
        )
        from qwen_r9700_lab.dflash_b7_device_runtime import B7CommitGeneration

        commit_generation = B7CommitGeneration(GDN_LAYERS)
        return cls(
            bulk=bulk,
            parent=torch.empty((B7_NODES,), dtype=torch.int32, device=device),
            ancestor_mask=torch.empty((B7_NODES,), dtype=torch.uint32, device=device),
            logical_offsets=torch.empty((1, B7_ROWS), dtype=torch.int64, device=device),
            anchor_position=torch.empty((3, 1), dtype=torch.int64, device=device),
            parent_cpu=torch.empty((B7_NODES,), dtype=torch.int32, device="cpu", pin_memory=True),
            ancestor_mask_cpu=torch.empty(
                (B7_NODES,), dtype=torch.uint32, device="cpu", pin_memory=True
            ),
            logical_offsets_cpu=torch.empty(
                (1, B7_ROWS), dtype=torch.int64, device="cpu", pin_memory=True
            ),
            commit_generation=commit_generation,
        )

    def stage(
        self,
        torch: Any,
        *,
        owner: object,
        request_id: str,
        metadata: dict[str, object],
        scheduled_tokens: list[int],
        positions: Any,
        slot_mappings: Any,
        commit_capacity: int = B7_ROWS,
    ) -> None:
        """Publish one exact round after physical slot mappings already exist."""

        from qwen_r9700_lab.dflash_b7_device_runtime import publish_b7_device_round
        from qwen_r9700_lab.tree_attention_contract import validate_tree_metadata
        from qwen_r9700_lab.tree_runtime import publish_tree_metadata

        if self.active_request_id is not None:
            raise B7GPUContractError("prior B7 GPU round was not cleared")
        if (
            isinstance(commit_capacity, bool)
            or not isinstance(commit_capacity, int)
            or not 1 <= commit_capacity <= B7_ROWS
        ):
            raise B7GPUContractError("B7 commit capacity must be an integer in [1, 8]")
        normalized = validate_tree_metadata(metadata, max_nodes=B7_NODES)
        if normalized.node_count != B7_NODES:
            raise B7GPUContractError("B7 GPU round requires exactly seven nodes")
        if tuple(int(token) for token in scheduled_tokens) != normalized.tokens:
            raise B7GPUContractError("B7 scheduled tokens and tree metadata differ")
        _require_b7_position_rewrite_layout(torch, positions, self.parent.device)
        # NumPy writes the caller-owned pinned buffers without allocating a
        # temporary torch tensor.  The following copies are async on the serving
        # stream and their destination addresses never change.
        self.parent_cpu.numpy()[:] = normalized.parent
        self.ancestor_mask_cpu.numpy()[:] = normalized.ancestor_mask
        self.logical_offsets_cpu.numpy()[0, :] = (
            0,
            *(depth + 1 for depth in normalized.depth),
        )
        self.parent.copy_(self.parent_cpu, non_blocking=True)
        self.ancestor_mask.copy_(self.ancestor_mask_cpu, non_blocking=True)
        self.logical_offsets.copy_(self.logical_offsets_cpu, non_blocking=True)
        self.anchor_position.copy_(positions[:, :1])
        torch.add(self.anchor_position, self.logical_offsets, out=positions)
        self.bulk.bind_slot_mappings_once(torch, slot_mappings)
        self.commit_generation.begin()
        try:
            publish_tree_metadata({request_id: normalized.as_dict()})
            publish_b7_device_round(
                owner=owner,
                request_id=request_id,
                metadata=normalized,
                parent=self.parent,
                ancestor_mask=self.ancestor_mask,
                verifier_outputs=self.bulk.outputs,
                commit_generation=self.commit_generation,
                commit_capacity=commit_capacity,
            )
        except BaseException:
            from qwen_r9700_lab.dflash_b7_device_runtime import clear_b7_device_round
            from qwen_r9700_lab.tree_runtime import clear_tree_metadata

            clear_b7_device_round(owner=owner)
            clear_tree_metadata()
            self.commit_generation.abort()
            raise
        self.active_request_id = request_id

    def commit_and_clear(self, *, owner: object) -> None:
        """Publish the verifier-selected state/KV and end the active tree round."""

        from qwen_r9700_lab.dflash_b7_device_runtime import clear_b7_device_round
        from qwen_r9700_lab.tree_runtime import clear_tree_metadata

        if self.active_request_id is None:
            raise B7GPUContractError("B7 GPU round is absent at commit")
        self.commit_generation.require_all_ready()
        try:
            self.bulk.commit()
        except BaseException:
            self.abort(owner=owner)
            raise
        self.commit_generation.mark_committed()
        clear_b7_device_round(owner=owner)
        clear_tree_metadata()
        self.active_request_id = None

    def abort(self, *, owner: object) -> None:
        """Clear process-local metadata after a failed forward without publication."""

        from qwen_r9700_lab.dflash_b7_device_runtime import clear_b7_device_round
        from qwen_r9700_lab.tree_runtime import clear_tree_metadata

        clear_b7_device_round(owner=owner)
        clear_tree_metadata()
        committed = self.commit_generation.committed_generation
        for layer in self.bulk.gdn.layers:
            layer._qwen_b7_ready_generation = committed
            layer._recoverssm_m8_cached_commit_ready = False
        self.commit_generation.abort()
        self.active_request_id = None

    def release(self) -> None:
        """Break strong KV/model references before the runner clears its caches."""

        if self.active_request_id is not None:
            raise B7GPUContractError("cannot release an active B7 GPU round")
        self.bulk = None  # type: ignore[assignment]


__all__ = [
    "B7BulkCommitRuntime",
    "B7GPUContractError",
    "B7GPUWorkspace",
    "B7VerifierOutputs",
    "bind_kv_slot_mappings_once",
    "build_gdn_pointer_tables",
    "build_kv_pointer_tables",
    "bulk_gdn_commit",
    "bulk_gdn_rollback",
    "bulk_kv_gather",
    "bulk_kv_publish",
    "verify_best_first_b7_out",
]
