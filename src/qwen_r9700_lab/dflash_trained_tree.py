"""Deterministic trained-DFlash proposal-tree planning and greedy verification.

The production DFlash checkpoint predicts seven draft positions and retains a
Top-16 candidate lattice.  The stock runtime collapses that lattice to one
linear seven-token spine.  This module provides both the original planner that
retains that complete spine and the screened B7 planner that spends the same
seven-node verification budget on the most likely prefix-closed tree.

This is deliberately a dependency-free contract.  It does not enable tree
serving by itself; the active runtime must still provide branch-aware attention,
GDN state, and accepted-path commit consumers before using its output.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass

from .tree_attention_contract import TreeAttentionMetadata, validate_tree_metadata

DFLASH_DRAFT_DEPTH = 7
DFLASH_TOP_K = 16
DEFAULT_TREE_NODE_BUDGET = 15
MAX_TREE_NODE_BUDGET = 15
BEST_FIRST_B7_NODE_BUDGET = 7
BEST_FIRST_B7_TEMPERATURE = 1.4
BEST_FIRST_B7_DEPTH_BONUS = 0.5
BEST_FIRST_B7_CANDIDATE_INDEX_BIAS = 0.25


@dataclass(frozen=True)
class TrainedDFlashTree:
    """One prefix-closed tree derived from the trained Top-16 lattice."""

    metadata: TreeAttentionMetadata
    candidate_ranks: tuple[tuple[int, ...], ...]
    primary_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.candidate_ranks) != self.metadata.node_count:
            raise ValueError("candidate rank paths must align with tree nodes")
        if not 1 <= len(self.primary_ranks) <= DFLASH_DRAFT_DEPTH:
            raise ValueError("primary rank path must contain between one and seven ranks")

    @property
    def target_query_rows(self) -> int:
        """Return root row plus one verification row per proposal node."""

        return self.metadata.node_count + 1

    def as_dict(self) -> dict[str, object]:
        """Return the existing tree-attention wire representation."""

        return self.metadata.as_dict()


@dataclass(frozen=True)
class GreedyTreeVerification:
    """Target-approved output from one branch-aware greedy verification."""

    output_tokens: tuple[int, ...]
    accepted_node_indices: tuple[int, ...]

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_node_indices)

    @property
    def emitted_count(self) -> int:
        return len(self.output_tokens)

    def kv_commit_moves(self) -> tuple[tuple[int, int], ...]:
        """Return tree-row to canonical-row copies for accepted target KV.

        Verification row zero is the already-sampled root and is already in its
        canonical slot.  Proposal node ``n`` is written to temporary row
        ``n + 1``.  Accepted proposal rows must be compacted after rejection so
        the next target step sees a linear committed cache.
        """

        return tuple(
            (node + 1, accepted_offset + 1)
            for accepted_offset, node in enumerate(self.accepted_node_indices)
            if node != accepted_offset
        )

    def accepted_replay_rows(self) -> tuple[int, ...]:
        """Return target input rows whose state becomes canonical.

        Row zero processes the already-sampled anchor.  Every accepted proposal
        node is physically evaluated in row ``node + 1``.  The final emitted
        mismatch/bonus token has no target input row until the next round, so it
        is intentionally absent from this replay list.
        """

        return (0, *(node + 1 for node in self.accepted_node_indices))

    @property
    def canonical_gdn_source_row(self) -> int:
        """Return the branch row holding the exact post-commit recurrent state."""

        return 0 if not self.accepted_node_indices else self.accepted_node_indices[-1] + 1


def _validate_candidates(candidate_ids: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    if len(candidate_ids) != DFLASH_DRAFT_DEPTH:
        raise ValueError("DFlash candidate_ids must have exactly seven depth rows")
    normalized: list[tuple[int, ...]] = []
    for row in candidate_ids:
        if len(row) != DFLASH_TOP_K:
            raise ValueError("each DFlash candidate row must contain exactly Top-16 IDs")
        values = tuple(int(token) for token in row)
        if any(token < 0 for token in values):
            raise ValueError("DFlash candidate token IDs must be non-negative")
        normalized.append(values)
    return tuple(normalized)


def _validate_scores(
    edge_scores: Sequence[Sequence[Sequence[float]]],
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    if len(edge_scores) != DFLASH_DRAFT_DEPTH:
        raise ValueError("DFlash edge_scores must have exactly seven depth planes")
    normalized: list[tuple[tuple[float, ...], ...]] = []
    for plane in edge_scores:
        if len(plane) != DFLASH_TOP_K:
            raise ValueError("each DFlash score plane must have 16 predecessor rows")
        rows: list[tuple[float, ...]] = []
        for row in plane:
            if len(row) != DFLASH_TOP_K:
                raise ValueError("each DFlash score row must contain 16 candidate scores")
            values = tuple(float(score) for score in row)
            if any(math.isnan(score) or score == math.inf for score in values):
                raise ValueError("DFlash scores must not contain NaN or positive infinity")
            if all(score == -math.inf for score in values):
                raise ValueError("DFlash score rows must contain at least one finite value")
            rows.append(values)
        normalized.append(tuple(rows))
    return tuple(normalized)


def _log_probabilities(
    scores: Sequence[float], temperature: float = 1.0
) -> tuple[float, ...]:
    scaled = tuple(score / temperature for score in scores)
    maximum = max(scaled)
    denominator = sum(math.exp(score - maximum) for score in scaled if score != -math.inf)
    log_denominator = maximum + math.log(denominator)
    return tuple(-math.inf if score == -math.inf else score - log_denominator for score in scaled)


def _argmax_first(values: Sequence[float]) -> int:
    """Match torch.argmax's first-index tie behavior."""

    return max(range(len(values)), key=lambda index: (values[index], -index))


def build_trained_dflash_tree(
    candidate_ids: Sequence[Sequence[int]],
    edge_scores: Sequence[Sequence[Sequence[float]]],
    *,
    node_budget: int = DEFAULT_TREE_NODE_BUDGET,
) -> TrainedDFlashTree:
    """Build a deterministic prefix-closed tree while retaining the D7 spine.

    ``edge_scores[depth][previous_rank][candidate_rank]`` is interpreted as a
    conditional score.  The root uses predecessor row zero, matching the pinned
    DFlash linear selector.  Alternate nodes are selected by cumulative
    normalized log probability with deterministic rank-path tie breaking.
    """

    if node_budget < DFLASH_DRAFT_DEPTH or node_budget > MAX_TREE_NODE_BUDGET:
        raise ValueError("tree node budget must be in [7, 15]")
    candidates = _validate_candidates(candidate_ids)
    scores = _validate_scores(edge_scores)

    row_log_probs = tuple(tuple(_log_probabilities(row) for row in plane) for plane in scores)

    primary_ranks: list[int] = []
    previous_rank = 0
    for depth in range(DFLASH_DRAFT_DEPTH):
        rank = _argmax_first(scores[depth][previous_rank])
        primary_ranks.append(rank)
        previous_rank = rank
    primary = tuple(primary_ranks)

    paths: list[tuple[int, ...]] = []
    parents: list[int] = []
    cumulative_scores: list[float] = []
    path_to_index: dict[tuple[int, ...], int] = {}

    cumulative = 0.0
    for depth, rank in enumerate(primary):
        path = primary[: depth + 1]
        predecessor = 0 if depth == 0 else primary[depth - 1]
        cumulative += row_log_probs[depth][predecessor][rank]
        path_to_index[path] = len(paths)
        paths.append(path)
        parents.append(depth - 1)
        cumulative_scores.append(cumulative)

    frontier: list[tuple[float, tuple[int, ...]]] = []
    queued: set[tuple[int, ...]] = set()

    def enqueue_children(parent_path: tuple[int, ...]) -> None:
        depth = len(parent_path)
        if depth >= DFLASH_DRAFT_DEPTH:
            return
        predecessor = 0 if depth == 0 else parent_path[-1]
        parent_score = 0.0 if depth == 0 else cumulative_scores[path_to_index[parent_path]]
        for rank, log_probability in enumerate(row_log_probs[depth][predecessor]):
            child_path = (*parent_path, rank)
            if child_path in path_to_index or child_path in queued:
                continue
            score = parent_score + log_probability
            heapq.heappush(frontier, (-score, child_path))
            queued.add(child_path)

    enqueue_children(())
    for depth in range(1, DFLASH_DRAFT_DEPTH):
        enqueue_children(primary[:depth])

    while len(paths) < node_budget:
        if not frontier:
            raise ValueError("DFlash score lattice did not provide enough finite tree nodes")
        negative_score, path = heapq.heappop(frontier)
        queued.remove(path)
        if path in path_to_index:
            continue
        if negative_score == math.inf:
            raise ValueError("DFlash score lattice did not provide enough finite tree nodes")
        parent_path = path[:-1]
        if parent_path and parent_path not in path_to_index:
            raise AssertionError("planner frontier violated prefix closure")
        parent_index = -1 if not parent_path else path_to_index[parent_path]
        path_to_index[path] = len(paths)
        paths.append(path)
        parents.append(parent_index)
        cumulative_scores.append(-negative_score)
        enqueue_children(path)

    tokens = tuple(candidates[len(path) - 1][path[-1]] for path in paths)
    depths = tuple(len(path) - 1 for path in paths)
    ancestor_masks: list[int] = []
    for parent in parents:
        ancestor_masks.append(0 if parent < 0 else ancestor_masks[parent] | (1 << parent))
    primary_path = tuple(path_to_index[primary[:depth]] for depth in range(1, 8))
    metadata = validate_tree_metadata(
        {
            "tokens": tokens,
            "parent": parents,
            "depth": depths,
            "ancestor_mask": ancestor_masks,
            "primary_path": primary_path,
            "score": cumulative_scores,
        },
        max_nodes=MAX_TREE_NODE_BUDGET,
    )
    return TrainedDFlashTree(metadata, tuple(paths), primary)


def build_best_first_dflash_tree(
    candidate_ids: Sequence[Sequence[int]],
    edge_scores: Sequence[Sequence[Sequence[float]]],
    *,
    node_budget: int = BEST_FIRST_B7_NODE_BUDGET,
    temperature: float = 1.0,
    depth_bonus: float = 0.0,
    candidate_index_bias: float = 0.0,
) -> TrainedDFlashTree:
    """Build the screened pure best-first prefix-closed proposal tree.

    Unlike :func:`build_trained_dflash_tree`, this policy does not reserve the
    complete linear D7 spine.  The frontier starts with all 16 depth-zero
    children using predecessor row zero.  Each selected node then contributes
    its 16 children using its last rank as the predecessor.  Nodes are ranked
    by cumulative normalized log probability plus the explicitly supplied
    rank/depth terms; Python heap tuple ordering makes the rank path the
    deterministic lexicographic tie breaker.  Defaults preserve the original
    pure-B7 policy.  Serving binds the separately screened constants at its
    production call site so preserved evidence replays cannot change silently.

    The production candidate is deliberately fixed to seven proposal nodes so
    target verification remains M8 (one anchor row plus seven proposal rows).
    A branch-aware consumer is mandatory; flattening this topological node list
    would not represent any valid autoregressive proposal path.
    """

    if node_budget != BEST_FIRST_B7_NODE_BUDGET:
        raise ValueError("best-first DFlash tree node budget must be exactly 7")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("best-first DFlash temperature must be finite and positive")
    if not math.isfinite(depth_bonus):
        raise ValueError("best-first DFlash depth bonus must be finite")
    if not math.isfinite(candidate_index_bias):
        raise ValueError("best-first DFlash candidate-index bias must be finite")
    candidates = _validate_candidates(candidate_ids)
    scores = _validate_scores(edge_scores)
    # Only the root row and the children of the at most seven selected nodes can
    # enter a B7 frontier.  Normalizing all 112 lattice rows wastes Python/CPU
    # work in every decode round.  Cache the at-most-eight rows actually read;
    # this preserves the exact float/order semantics of ``_log_probabilities``.
    row_log_probabilities: dict[tuple[int, int], tuple[float, ...]] = {}

    def log_probabilities(depth: int, predecessor: int) -> tuple[float, ...]:
        key = (depth, predecessor)
        result = row_log_probabilities.get(key)
        if result is None:
            result = _log_probabilities(
                scores[depth][predecessor],
                temperature,
            )
            row_log_probabilities[key] = result
        return result

    frontier: list[tuple[float, tuple[int, ...], float]] = []
    for rank, log_probability in enumerate(log_probabilities(0, 0)):
        cumulative_score = (
            log_probability + candidate_index_bias * math.log1p(rank)
        )
        priority = cumulative_score + depth_bonus
        heapq.heappush(frontier, (-priority, (rank,), cumulative_score))

    paths: list[tuple[int, ...]] = []
    parents: list[int] = []
    cumulative_scores: list[float] = []
    path_to_index: dict[tuple[int, ...], int] = {}

    while len(paths) < node_budget:
        if not frontier:
            raise ValueError("DFlash score lattice did not provide enough finite tree nodes")
        negative_priority, path, cumulative_score = heapq.heappop(frontier)
        if negative_priority == math.inf:
            raise ValueError("DFlash score lattice did not provide enough finite tree nodes")
        parent_path = path[:-1]
        if parent_path and parent_path not in path_to_index:
            raise AssertionError("planner frontier violated prefix closure")
        parent_index = -1 if not parent_path else path_to_index[parent_path]
        path_to_index[path] = len(paths)
        paths.append(path)
        parents.append(parent_index)
        cumulative_scores.append(cumulative_score)

        depth = len(path)
        if depth >= DFLASH_DRAFT_DEPTH:
            continue
        predecessor = path[-1]
        parent_score = cumulative_score
        for rank, log_probability in enumerate(log_probabilities(depth, predecessor)):
            child_path = (*path, rank)
            child_score = (
                parent_score
                + log_probability
                + candidate_index_bias * math.log1p(rank)
            )
            child_priority = child_score + depth_bonus * len(child_path)
            heapq.heappush(frontier, (-child_priority, child_path, child_score))

    tokens = tuple(candidates[len(path) - 1][path[-1]] for path in paths)
    depths = tuple(len(path) - 1 for path in paths)
    ancestor_masks: list[int] = []
    for parent in parents:
        ancestor_masks.append(0 if parent < 0 else ancestor_masks[parent] | (1 << parent))

    # ``primary_path`` is a compatibility-only linear view.  Tree-enabled
    # execution consumes every parent/mask entry and must never flatten this
    # tree.  Prefer the deepest selected path, then the earliest best-first
    # insertion when depths tie.
    primary_leaf = max(range(len(paths)), key=lambda index: (len(paths[index]), -index))
    primary_indices: list[int] = []
    cursor = primary_leaf
    while cursor >= 0:
        primary_indices.append(cursor)
        cursor = parents[cursor]
    primary_indices.reverse()
    primary_ranks = paths[primary_leaf]

    metadata = validate_tree_metadata(
        {
            "tokens": tokens,
            "parent": parents,
            "depth": depths,
            "ancestor_mask": ancestor_masks,
            "primary_path": primary_indices,
            "score": cumulative_scores,
        },
        max_nodes=BEST_FIRST_B7_NODE_BUDGET,
    )
    return TrainedDFlashTree(metadata, tuple(paths), primary_ranks)


def verify_greedy_tree(
    tree: TrainedDFlashTree | TreeAttentionMetadata,
    target_argmax: Sequence[int],
) -> GreedyTreeVerification:
    """Apply exact greedy target verification to a branch-aware target result.

    Row zero predicts the first proposal token.  Row ``node + 1`` predicts the
    token after that node's branch context.  If duplicate proposal IDs match at
    one branch point, the longest target-matching continuation wins; index order
    breaks equal-length ties and therefore preserves the primary D7 spine.
    """

    metadata = tree.metadata if isinstance(tree, TrainedDFlashTree) else tree
    target = tuple(int(token) for token in target_argmax)
    if len(target) != metadata.node_count + 1:
        raise ValueError("target_argmax must contain root plus one row per tree node")
    if any(token < 0 for token in target):
        raise ValueError("target argmax IDs must be non-negative")

    children: dict[int, list[int]] = {-1: []}
    for node, parent in enumerate(metadata.parent):
        children.setdefault(parent, []).append(node)
        children.setdefault(node, [])

    def longest_matching_path(parent: int) -> tuple[int, ...]:
        expected = target[0 if parent < 0 else parent + 1]
        best: tuple[int, ...] = ()
        for child in children[parent]:
            if metadata.tokens[child] != expected:
                continue
            candidate = (child, *longest_matching_path(child))
            if len(candidate) > len(best) or (len(candidate) == len(best) and candidate < best):
                best = candidate
        return best

    accepted = longest_matching_path(-1)
    bonus_row = 0 if not accepted else accepted[-1] + 1
    output = (*(metadata.tokens[node] for node in accepted), target[bonus_row])
    return GreedyTreeVerification(output, accepted)


def tree_position_offsets(tree: TrainedDFlashTree | TreeAttentionMetadata) -> tuple[int, ...]:
    """Return branch-correct rotary position offsets for target query rows.

    Cache write slots remain flattened (root, node 0, node 1, ...), while model
    position IDs follow tree depth.  Keeping these concepts separate prevents
    sibling rows from aliasing the same temporary KV slot.
    """

    metadata = tree.metadata if isinstance(tree, TrainedDFlashTree) else tree
    return (0, *(depth + 1 for depth in metadata.depth))


__all__ = [
    "BEST_FIRST_B7_CANDIDATE_INDEX_BIAS",
    "BEST_FIRST_B7_DEPTH_BONUS",
    "BEST_FIRST_B7_NODE_BUDGET",
    "BEST_FIRST_B7_TEMPERATURE",
    "DEFAULT_TREE_NODE_BUDGET",
    "DFLASH_DRAFT_DEPTH",
    "DFLASH_TOP_K",
    "MAX_TREE_NODE_BUDGET",
    "GreedyTreeVerification",
    "TrainedDFlashTree",
    "build_best_first_dflash_tree",
    "build_trained_dflash_tree",
    "tree_position_offsets",
    "verify_greedy_tree",
]
