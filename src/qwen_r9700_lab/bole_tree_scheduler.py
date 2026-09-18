"""Dependency-free proposal-tree planner for the Qwen/R9700 lane.

The vLLM 0.26 custom proposer ABI still accepts one linear draft path per
request.  This module keeps the tree construction and Bole metadata separate
from that ABI so the scheduler can be upgraded without changing the
numerical proposal policy.  It is intentionally CPU-only and deterministic;
it is a planning/control layer, not a model forward or a throughput claim.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

from .bole_reference import TreePlan, build_tree_plan

Candidate = tuple[int, float]
CandidateFn = Callable[[tuple[int, ...]], Sequence[Candidate]]


@dataclass(frozen=True)
class ProposalNode:
    """One topologically ordered proposal node."""

    token_id: int
    parent: int
    depth: int
    score: float


@dataclass(frozen=True)
class ProposalTree:
    """A bounded tree plus a linear compatibility path."""

    nodes: tuple[ProposalNode, ...]
    plan: TreePlan
    primary_path: tuple[int, ...]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def max_depth(self) -> int:
        return max((node.depth for node in self.nodes), default=0)

    @property
    def primary_tokens(self) -> tuple[int, ...]:
        return tuple(self.nodes[index].token_id for index in self.primary_path)

    def path_tokens(self, node_index: int) -> tuple[int, ...]:
        if node_index < 0 or node_index >= self.node_count:
            raise IndexError("node_index is outside the proposal tree")
        path: list[int] = []
        current = node_index
        while current >= 0:
            path.append(self.nodes[current].token_id)
            current = self.nodes[current].parent
        path.reverse()
        return tuple(path)

    def as_vllm_linear_draft(self, limit: int | None = None) -> list[int]:
        """Return the best path for the current vLLM linear ABI.

        A future tree-aware scheduler should consume ``nodes`` and ``plan``
        directly.  Until then, selecting the best path is explicit and avoids
        pretending that a list of tokens is equivalent to tree verification.
        """

        tokens = self.primary_tokens
        return list(tokens if limit is None else tokens[: max(0, limit)])


def _validate_candidate(token: int, score: float) -> Candidate:
    token = int(token)
    score = float(score)
    if token < 0:
        raise ValueError("proposal token IDs must be non-negative")
    if not math.isfinite(score):
        raise ValueError("proposal scores must be finite")
    return token, score


def build_topk_tree(
    candidate_fn: CandidateFn,
    *,
    depth: int = 8,
    branch_factor: int = 2,
    max_nodes: int = 24,
    prefix: Sequence[int] = (),
) -> ProposalTree:
    """Build a deterministic bounded beam/tree from token candidates.

    ``candidate_fn`` receives the path *after* ``prefix`` and returns scored
    next-token candidates.  At each depth we retain at most ``branch_factor``
    children per retained frontier path and globally cap the tree at
    ``max_nodes``.  Candidates are ordered by descending cumulative score and
    then token ID, making the result reproducible across backends.
    """

    if depth < 1:
        raise ValueError("depth must be at least one")
    if branch_factor < 1:
        raise ValueError("branch_factor must be positive")
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")

    nodes: list[ProposalNode] = []
    frontier: list[tuple[int, tuple[int, ...], float]] = [(-1, tuple(prefix), 0.0)]
    for level in range(depth):
        expanded: list[tuple[int, tuple[int, ...], float, int]] = []
        for parent, path, cumulative in frontier:
            candidates = [_validate_candidate(*item) for item in candidate_fn(path)]
            candidates.sort(key=lambda item: (-item[1], item[0]))
            for token, score in candidates[:branch_factor]:
                expanded.append((parent, path + (token,), cumulative + score, token))
        if not expanded:
            break
        remaining = max_nodes - len(nodes)
        if remaining <= 0:
            break
        expanded.sort(key=lambda item: (-item[2], item[3], item[0]))
        selected = expanded[:remaining]
        next_frontier: list[tuple[int, tuple[int, ...], float]] = []
        for parent, path, cumulative, token in selected:
            index = len(nodes)
            nodes.append(ProposalNode(token, parent, level, cumulative))
            next_frontier.append((index, path, cumulative))
        frontier = next_frontier

    if not nodes:
        raise ValueError("candidate_fn produced no proposal nodes")

    # Convert the root-relative parent indices into a topological Bole plan.
    parent = tuple(node.parent for node in nodes)
    plan = build_tree_plan(parent)
    best = max(range(len(nodes)), key=lambda i: (nodes[i].score, -nodes[i].token_id))
    path: list[int] = []
    current = best
    while current >= 0:
        path.append(current)
        current = nodes[current].parent
    path.reverse()
    return ProposalTree(tuple(nodes), plan, tuple(path))


def fixed_topk_candidates(
    steps: Sequence[Sequence[Candidate]],
) -> CandidateFn:
    """Adapt per-depth candidates for planner tests and deterministic controls."""

    normalized = [tuple(_validate_candidate(*item) for item in step) for step in steps]

    def candidates(path: tuple[int, ...]) -> Sequence[Candidate]:
        depth = len(path)
        return normalized[depth] if depth < len(normalized) else ()

    return candidates


__all__ = [
    "Candidate",
    "ProposalNode",
    "ProposalTree",
    "build_topk_tree",
    "fixed_topk_candidates",
]
