"""Fail-closed CPU contract for the DFlash2 pure best-first B7 runtime.

This module owns no GPU work.  It converts one captured C1 Top-16 score lattice
into one atomic seven-token/tree-metadata payload and defines the logical
position and accepted-path commit mappings that GPU consumers must implement.
The serving overlay remains default-off until those consumers pass parity.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dflash_trained_tree import (
    BEST_FIRST_B7_CANDIDATE_INDEX_BIAS,
    BEST_FIRST_B7_DEPTH_BONUS,
    BEST_FIRST_B7_NODE_BUDGET,
    BEST_FIRST_B7_TEMPERATURE,
    GreedyTreeVerification,
    TrainedDFlashTree,
    build_best_first_dflash_tree,
    tree_position_offsets,
)
from .tree_attention_contract import TreeAttentionMetadata, validate_tree_metadata


class B7RuntimeContractError(RuntimeError):
    """The B7 payload or a required branch consumer violated its contract."""


B7_REQUIRED_ENVIRONMENT = "QWEN_DFLASH2_BEST_FIRST_B7_REQUIRED"
B7_RECEIPT_ENVIRONMENT = "QWEN_DFLASH2_BEST_FIRST_B7_RECEIPT"
B7_RECEIPT_SHA256_ENVIRONMENT = "QWEN_DFLASH2_BEST_FIRST_B7_RECEIPT_SHA256"
B7_RECEIPT_SCHEMA = "urn:qwen-r9700:dflash2-best-first-b7-consumer-receipt:v1"


@dataclass(frozen=True)
class B7ConsumerReceipt:
    """Qualification receipt required before the default-off lane can start."""

    planner: bool
    atomic_transport: bool
    logical_positions: bool
    quest_ancestor_attention: bool
    gdn_branch_recurrence: bool
    convolution_branch_state: bool
    greedy_tree_verifier: bool
    kv_accepted_path_compaction: bool
    canonical_state_commit: bool

    @classmethod
    def from_mapping(cls, value: Any) -> B7ConsumerReceipt:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise B7RuntimeContractError("B7 consumer receipt fields differ from the contract")
        if any(type(item) is not bool for item in value.values()):
            raise B7RuntimeContractError("B7 consumer receipt fields must be booleans")
        return cls(**value)

    def missing(self) -> tuple[str, ...]:
        return tuple(name for name, enabled in vars(self).items() if not enabled)

    def require_complete(self) -> None:
        missing = self.missing()
        if missing:
            raise B7RuntimeContractError(
                "B7 consumer receipt is incomplete: " + ", ".join(missing)
            )


@dataclass(frozen=True)
class B7DraftPayload:
    """One indivisible C1 proposal payload for vLLM's tree transport."""

    draft_token_ids: tuple[tuple[int, ...], ...]
    tree_metadata: tuple[dict[str, object], ...]

    def __post_init__(self) -> None:
        if len(self.draft_token_ids) != 1 or len(self.tree_metadata) != 1:
            raise B7RuntimeContractError("best-first B7 supports exactly one request")
        validate_atomic_tree_payload(self.draft_token_ids, self.tree_metadata)

    def as_vllm_lists(self) -> tuple[list[list[int]], list[dict[str, object]]]:
        """Return fresh mutable lists for the scheduler message boundary."""

        return (
            [list(row) for row in self.draft_token_ids],
            [dict(metadata) for metadata in self.tree_metadata],
        )


class B7PayloadMailbox:
    """Generation-bound, once-only proposer-to-scheduler hand-off.

    The stock runtime obtains draft tokens and optional tree metadata through
    separate methods.  That permits a new token copy to be paired with stale
    metadata.  This mailbox admits only one complete :class:`B7DraftPayload`
    and consumes both halves together for the same generation and request ID.
    """

    def __init__(self) -> None:
        self._pending: tuple[int, str, B7DraftPayload] | None = None
        self._last_consumed_generation = -1

    def publish(self, generation: int, request_id: str, payload: B7DraftPayload) -> None:
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise B7RuntimeContractError("B7 generation must be a non-negative integer")
        if not isinstance(request_id, str) or not request_id:
            raise B7RuntimeContractError("B7 request ID must be a non-empty string")
        if generation <= self._last_consumed_generation:
            raise B7RuntimeContractError("B7 payload generation is stale")
        if self._pending is not None:
            raise B7RuntimeContractError("B7 mailbox already contains an unconsumed payload")
        self._pending = (generation, request_id, payload)

    def take(self, generation: int, request_id: str) -> B7DraftPayload:
        pending = self._pending
        if pending is None:
            raise B7RuntimeContractError("B7 payload is absent")
        pending_generation, pending_request_id, payload = pending
        if generation != pending_generation or request_id != pending_request_id:
            raise B7RuntimeContractError("B7 payload generation/request binding changed")
        self._pending = None
        self._last_consumed_generation = generation
        return payload

    def discard(self) -> None:
        """Fail closed after a cancelled/error round without publishing stale data."""

        self._pending = None


def require_b7_activation(
    environment_value: str | None, receipt: B7ConsumerReceipt | None
) -> bool:
    """Validate the default-off activation flag and all mandatory consumers."""

    value = "0" if environment_value is None else environment_value
    if value not in {"0", "1"}:
        raise B7RuntimeContractError(f"{B7_REQUIRED_ENVIRONMENT} must be 0 or 1")
    if value == "0":
        return False
    if receipt is None:
        raise B7RuntimeContractError("B7 activation requires an authenticated consumer receipt")
    receipt.require_complete()
    return True


def load_b7_consumer_receipt(path: str, expected_sha256: str) -> B7ConsumerReceipt:
    """Load a hash-bound regular receipt; request metadata is never authority."""

    candidate = Path(path)
    if not candidate.is_absolute() or not candidate.is_file() or candidate.is_symlink():
        raise B7RuntimeContractError("B7 consumer receipt must be an absolute regular file")
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise B7RuntimeContractError("B7 consumer receipt SHA-256 is invalid")
    payload = candidate.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise B7RuntimeContractError("B7 consumer receipt SHA-256 mismatch")
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise B7RuntimeContractError("B7 consumer receipt is not valid JSON") from exc
    if not isinstance(document, dict) or set(document) != {"schema", "consumer"}:
        raise B7RuntimeContractError("B7 consumer receipt document fields differ")
    if document["schema"] != B7_RECEIPT_SCHEMA:
        raise B7RuntimeContractError("B7 consumer receipt schema differs")
    receipt = B7ConsumerReceipt.from_mapping(document["consumer"])
    receipt.require_complete()
    return receipt


def require_b7_activation_from_environment(
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Fail closed unless process environment names an authenticated receipt."""

    values = os.environ if environment is None else environment
    enabled = values.get(B7_REQUIRED_ENVIRONMENT)
    if enabled in {None, "0"}:
        if values.get(B7_RECEIPT_ENVIRONMENT) or values.get(B7_RECEIPT_SHA256_ENVIRONMENT):
            raise B7RuntimeContractError("B7 receipt variables require B7 activation")
        return False
    if enabled != "1":
        return require_b7_activation(enabled, None)
    receipt_path = values.get(B7_RECEIPT_ENVIRONMENT)
    receipt_sha = values.get(B7_RECEIPT_SHA256_ENVIRONMENT)
    if not receipt_path or not receipt_sha:
        raise B7RuntimeContractError("B7 activation requires receipt path and SHA-256")
    return require_b7_activation(
        enabled,
        load_b7_consumer_receipt(receipt_path, receipt_sha),
    )


def validate_atomic_tree_payload(
    draft_token_ids: Sequence[Sequence[int]],
    tree_metadata: Sequence[dict[str, object]],
) -> tuple[TreeAttentionMetadata, ...]:
    """Reject token/metadata skew before either half reaches the scheduler."""

    if len(draft_token_ids) != len(tree_metadata) or len(draft_token_ids) != 1:
        raise B7RuntimeContractError("B7 tokens and metadata must align for one C1 request")
    normalized = tuple(
        validate_tree_metadata(metadata, max_nodes=BEST_FIRST_B7_NODE_BUDGET)
        for metadata in tree_metadata
    )
    for tokens, metadata in zip(draft_token_ids, normalized, strict=True):
        normalized_tokens = tuple(int(token) for token in tokens)
        if len(normalized_tokens) != BEST_FIRST_B7_NODE_BUDGET:
            raise B7RuntimeContractError("B7 must carry exactly seven proposal nodes")
        if normalized_tokens != metadata.tokens:
            raise B7RuntimeContractError("B7 draft tokens do not match tree metadata tokens")
    return normalized


def build_best_first_b7_payload(
    candidate_ids: Sequence[Sequence[Sequence[int]]],
    edge_scores: Sequence[Sequence[Sequence[Sequence[float]]]],
) -> tuple[B7DraftPayload, TrainedDFlashTree]:
    """Plan one C1 lattice and return its atomic scheduler payload plus tree."""

    if len(candidate_ids) != 1 or len(edge_scores) != 1:
        raise B7RuntimeContractError("best-first B7 requires C1 candidate and score batches")
    tree = build_best_first_dflash_tree(
        candidate_ids[0],
        edge_scores[0],
        temperature=BEST_FIRST_B7_TEMPERATURE,
        depth_bonus=BEST_FIRST_B7_DEPTH_BONUS,
        candidate_index_bias=BEST_FIRST_B7_CANDIDATE_INDEX_BIAS,
    )
    payload = B7DraftPayload((tree.metadata.tokens,), (tree.as_dict(),))
    return payload, tree


def branch_logical_positions(
    physical_positions: Sequence[int],
    tree: TrainedDFlashTree | TreeAttentionMetadata,
) -> tuple[int, ...]:
    """Map flat M8 cache rows to branch-correct model/RoPE positions.

    Physical positions must remain consecutive so every provisional K/V row has
    a distinct cache slot.  Only the model position view is rewritten by tree
    depth; callers must compute slot mappings before applying this result.
    """

    positions = tuple(int(value) for value in physical_positions)
    if len(positions) != BEST_FIRST_B7_NODE_BUDGET + 1:
        raise B7RuntimeContractError("B7 logical position mapping requires exactly M8")
    expected = tuple(range(positions[0], positions[0] + len(positions)))
    if positions != expected:
        raise B7RuntimeContractError("B7 physical cache positions must be consecutive")
    offsets = tree_position_offsets(tree)
    if len(offsets) != len(positions):
        raise B7RuntimeContractError("B7 tree rows do not align with physical positions")
    return tuple(positions[0] + offset for offset in offsets)


def branch_logical_mrope_positions(
    physical_positions: Sequence[Sequence[int]],
    tree: TrainedDFlashTree | TreeAttentionMetadata,
) -> tuple[tuple[int, ...], ...]:
    """Apply the same depth mapping independently to every text M-RoPE axis."""

    axes = tuple(tuple(int(value) for value in axis) for axis in physical_positions)
    if len(axes) != 3:
        raise B7RuntimeContractError("Qwen text B7 M-RoPE requires exactly three axes")
    return tuple(branch_logical_positions(axis, tree) for axis in axes)


@dataclass(frozen=True)
class B7TargetRoundPlan:
    """Fixed M8 row mapping shared by Quest, GDN, KV, and the verifier."""

    metadata: TreeAttentionMetadata
    physical_rows: tuple[int, ...]
    logical_position_offsets: tuple[int, ...]
    gdn_parent_rows: tuple[int, ...]
    gdn_state_columns: tuple[int, ...]
    quest_ancestor_masks: tuple[int, ...]
    quest_tree_row_offset: int = 1

    def __post_init__(self) -> None:
        if self.metadata.node_count != BEST_FIRST_B7_NODE_BUDGET:
            raise B7RuntimeContractError("B7 target plan requires exactly seven nodes")
        if self.physical_rows != tuple(range(8)):
            raise B7RuntimeContractError("B7 target physical rows must be anchor,node0..node6")
        if len(self.logical_position_offsets) != 8:
            raise B7RuntimeContractError("B7 target logical positions must be M8")
        if len(self.gdn_parent_rows) != 7 or len(self.gdn_state_columns) != 8:
            raise B7RuntimeContractError("B7 GDN mappings have invalid width")
        if self.quest_ancestor_masks != self.metadata.ancestor_mask:
            raise B7RuntimeContractError("B7 Quest masks differ from validated metadata")

    def quest_visible_physical_rows(self, node_index: int) -> tuple[int, ...]:
        """Return anchor plus self/ancestor target rows for one proposal node."""

        return (0, *(node + 1 for node in self.metadata.visible_nodes(node_index)))


def build_b7_target_round_plan(
    tree: TrainedDFlashTree | TreeAttentionMetadata,
) -> B7TargetRoundPlan:
    """Build the one authoritative row mapping; consumers may not improvise it."""

    metadata = tree.metadata if isinstance(tree, TrainedDFlashTree) else tree
    if metadata.node_count != BEST_FIRST_B7_NODE_BUDGET:
        raise B7RuntimeContractError("B7 target plan requires exactly seven nodes")
    parent_rows = tuple(0 if parent < 0 else parent + 1 for parent in metadata.parent)
    return B7TargetRoundPlan(
        metadata=metadata,
        physical_rows=tuple(range(BEST_FIRST_B7_NODE_BUDGET + 1)),
        logical_position_offsets=(0, *(depth + 1 for depth in metadata.depth)),
        # Column zero is the immutable canonical source.  Nodes write independent
        # scratch columns 1..7; parent rows use the same physical row convention.
        gdn_parent_rows=parent_rows,
        gdn_state_columns=tuple(range(BEST_FIRST_B7_NODE_BUDGET + 1)),
        quest_ancestor_masks=metadata.ancestor_mask,
    )


def accepted_kv_compaction_plan(
    verification: GreedyTreeVerification,
) -> tuple[tuple[int, int], ...]:
    """Return immutable-source physical-row copies for accepted target K/V."""

    return verification.kv_commit_moves()


def compact_accepted_kv_rows(
    physical_rows: Sequence[Any],
    verification: GreedyTreeVerification,
) -> tuple[Any, ...]:
    """Reference accepted-prefix compaction using a pre-copy snapshot.

    Production copies must have equivalent alias safety: an earlier destination
    must never destroy a later source row before it is read.
    """

    if len(physical_rows) != BEST_FIRST_B7_NODE_BUDGET + 1:
        raise B7RuntimeContractError("B7 K/V compaction requires exactly eight physical rows")
    source = tuple(physical_rows)
    destination = list(source)
    for source_row, destination_row in accepted_kv_compaction_plan(verification):
        destination[destination_row] = source[source_row]
    return tuple(destination[: verification.emitted_count])


def committed_gdn_state(
    physical_row_states: Sequence[Any],
    verification: GreedyTreeVerification,
) -> Any:
    """Select only the sampler-approved branch state for canonical publication."""

    if len(physical_row_states) != BEST_FIRST_B7_NODE_BUDGET + 1:
        raise B7RuntimeContractError("B7 GDN state selection requires exactly eight rows")
    return physical_row_states[verification.canonical_gdn_source_row]


__all__ = [
    "B7_RECEIPT_ENVIRONMENT",
    "B7_RECEIPT_SCHEMA",
    "B7_RECEIPT_SHA256_ENVIRONMENT",
    "B7_REQUIRED_ENVIRONMENT",
    "B7ConsumerReceipt",
    "B7DraftPayload",
    "B7PayloadMailbox",
    "B7RuntimeContractError",
    "B7TargetRoundPlan",
    "accepted_kv_compaction_plan",
    "branch_logical_mrope_positions",
    "branch_logical_positions",
    "build_b7_target_round_plan",
    "build_best_first_b7_payload",
    "committed_gdn_state",
    "compact_accepted_kv_rows",
    "load_b7_consumer_receipt",
    "require_b7_activation",
    "require_b7_activation_from_environment",
    "validate_atomic_tree_payload",
]
