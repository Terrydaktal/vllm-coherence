"""Canonical Bole/Medusa tree metadata contract for attention consumers.

The vLLM compatibility ABI can carry tree metadata, but carrying metadata is
not the same thing as applying an ancestor mask.  This module keeps the
serialization and validation rules in one dependency-free place so compiled
attention consumers can adopt the same contract without importing vLLM or
torch.

The contract is deliberately integer-only at the attention boundary:

* ``tokens``/``parent``/``depth`` describe a topologically ordered tree;
* ``ancestor_mask[i]`` is a bit mask of nodes visible to node ``i``;
* ``primary_path`` identifies the compatibility path used by flattened ABIs.

Consumers must either use ``ancestor_mask`` or explicitly report that they do
not support tree verification.  A flattened token list must never be silently
reported as a tree-aware verification result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


MAX_TREE_NODES = 32


def _ints(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"tree metadata {name} must be a sequence")
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tree metadata {name} contains a non-integer") from exc


@dataclass(frozen=True)
class TreeAttentionMetadata:
    """Validated, immutable metadata for one proposal tree."""

    tokens: tuple[int, ...]
    parent: tuple[int, ...]
    depth: tuple[int, ...]
    ancestor_mask: tuple[int, ...]
    primary_path: tuple[int, ...]
    score: tuple[float, ...] = ()

    @property
    def node_count(self) -> int:
        return len(self.tokens)

    @property
    def max_depth(self) -> int:
        return max(self.depth, default=0)

    def as_dict(self) -> dict[str, object]:
        """Return the stable wire representation used by the vLLM overlay."""
        result: dict[str, object] = {
            "tokens": list(self.tokens),
            "parent": list(self.parent),
            "depth": list(self.depth),
            "ancestor_mask": list(self.ancestor_mask),
            "primary_path": list(self.primary_path),
        }
        if self.score:
            result["score"] = list(self.score)
        return result

    def visible_nodes(self, node_index: int) -> tuple[int, ...]:
        """Return the self-plus-ancestors visibility set for one node."""
        if node_index < 0 or node_index >= self.node_count:
            raise IndexError("node index outside tree metadata")
        mask = self.ancestor_mask[node_index] | (1 << node_index)
        return tuple(index for index in range(self.node_count) if mask & (1 << index))

    def backend_buffers(self) -> dict[str, tuple[int, ...]]:
        """Return fixed-width integer buffers for a compiled consumer.

        The current cap of 32 nodes means one ``uint32`` is sufficient for
        every row.  Keeping this conversion here prevents Python/vLLM callers
        from inventing different parent or mask layouts when the HIP consumer
        is enabled.
        """
        return {
            "tokens": self.tokens,
            "parent": self.parent,
            "depth": self.depth,
            "ancestor_mask": self.ancestor_mask,
            "primary_path": self.primary_path,
        }


def validate_tree_metadata(
    metadata: Mapping[str, Any], *, max_nodes: int = MAX_TREE_NODES
) -> TreeAttentionMetadata:
    """Validate and normalize one proposer metadata mapping.

    Validation is intentionally strict: malformed parent or mask data would
    turn a tree verifier into an incorrect dense/flattened verifier.  The
    function is pure and safe to use in unit tests and in the process-local
    scheduler hand-off.
    """

    if not isinstance(metadata, Mapping):
        raise ValueError("tree metadata must be a mapping")
    tokens = _ints(metadata.get("tokens"), "tokens")
    parent = _ints(metadata.get("parent"), "parent")
    depth = _ints(metadata.get("depth"), "depth")
    ancestor_mask = _ints(metadata.get("ancestor_mask"), "ancestor_mask")
    primary_path = _ints(metadata.get("primary_path"), "primary_path")
    count = len(tokens)
    if count < 1 or count > max_nodes:
        raise ValueError(f"tree node count must be in [1, {max_nodes}]")
    if not (len(parent) == len(depth) == len(ancestor_mask) == count):
        raise ValueError("tree metadata arrays must have equal node length")
    if len(primary_path) < 1 or len(primary_path) > count:
        raise ValueError("primary_path must contain at least one valid node")
    if any(token < 0 for token in tokens):
        raise ValueError("tree token IDs must be non-negative")
    for index, (parent_index, node_depth, mask) in enumerate(
        zip(parent, depth, ancestor_mask)
    ):
        if parent_index < -1 or parent_index >= index:
            raise ValueError("tree parents must be -1 or topologically earlier")
        if node_depth < 0 or (index == 0 and node_depth != 0):
            raise ValueError("invalid tree depth")
        if parent_index < 0:
            if node_depth != 0:
                raise ValueError("root nodes must have depth zero")
        elif node_depth != depth[parent_index] + 1:
            raise ValueError("tree depth does not follow parent depth")
        if mask < 0 or mask >> count:
            raise ValueError("ancestor mask contains an out-of-range bit")
        if parent_index >= 0 and not (mask & (1 << parent_index)):
            raise ValueError("ancestor mask must include the parent")
        if mask & (1 << index):
            raise ValueError("ancestor mask must not include the node itself")
    for previous, current in zip(primary_path, primary_path[1:]):
        if previous < 0 or previous >= count or current < 0 or current >= count:
            raise ValueError("primary_path contains an invalid node")
        if parent[current] != previous:
            raise ValueError("primary_path is not a contiguous parent path")
    raw_scores = metadata.get("score", ())
    score = tuple(float(item) for item in raw_scores) if raw_scores else ()
    if score and len(score) != count:
        raise ValueError("score must be empty or aligned with tree nodes")
    return TreeAttentionMetadata(tokens, parent, depth, ancestor_mask, primary_path, score)


def normalize_tree_metadata(
    metadata: Mapping[str, Any] | None, *, max_nodes: int = MAX_TREE_NODES
) -> dict[str, object] | None:
    """Validate metadata and return a plain JSON/ABI-safe mapping."""

    if metadata is None:
        return None
    return validate_tree_metadata(metadata, max_nodes=max_nodes).as_dict()


__all__ = [
    "MAX_TREE_NODES",
    "TreeAttentionMetadata",
    "normalize_tree_metadata",
    "validate_tree_metadata",
]
