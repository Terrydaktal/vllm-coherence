"""Stable-address device hand-off for one exact best-first B7 target round.

The scheduler tree remains the serialized source of truth.  The target runner
validates it once, copies its parent and ancestor arrays into caller-owned
persistent CUDA buffers, and publishes only references to those buffers here.
GDN and Quest can then consume the same addresses in every layer without a
per-layer allocation or host-to-device copy.

This module intentionally launches no GPU work and never reads a device value.
The publisher owns stream ordering; consumers fail closed when the round is
absent, stale, ambiguous, or rebound to different storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

from .tree_attention_contract import TreeAttentionMetadata, validate_tree_metadata


class B7DeviceRuntimeError(RuntimeError):
    """The stable-address B7 device hand-off violated its contract."""


class B7CommitGeneration:
    """One shared host generation checked naturally by all 48 layer forwards."""

    def __init__(self, layer_count: int = 48) -> None:
        if layer_count <= 0:
            raise B7DeviceRuntimeError("B7 commit generation needs positive layer count")
        self.layer_count = layer_count
        self.current_generation = -1
        self.committed_generation = -1
        self.ready_mask = 0
        self.active = False

    def begin(self) -> int:
        if self.active or self.current_generation != self.committed_generation:
            raise B7DeviceRuntimeError("prior B7 generation was not committed")
        self.current_generation += 1
        self.ready_mask = 0
        self.active = True
        return self.current_generation

    def require_layer_forward(self, ordinal: int, prior_generation: int) -> int:
        if not self.active or not 0 <= ordinal < self.layer_count:
            raise B7DeviceRuntimeError("B7 layer forward has no active generation")
        if prior_generation != self.committed_generation:
            raise B7DeviceRuntimeError("B7 layer observed an uncommitted prior generation")
        if self.ready_mask & (1 << ordinal):
            raise B7DeviceRuntimeError("B7 layer executed twice in one generation")
        return self.current_generation

    def mark_layer_ready(self, ordinal: int) -> None:
        self.ready_mask |= 1 << ordinal

    def require_all_ready(self) -> None:
        if not self.active or self.ready_mask != (1 << self.layer_count) - 1:
            raise B7DeviceRuntimeError("B7 bulk commit is missing one or more GDN layers")

    def mark_committed(self) -> None:
        self.require_all_ready()
        self.committed_generation = self.current_generation
        self.active = False

    def abort(self) -> None:
        """Discard an uncommitted host generation after a failed target forward."""

        self.current_generation = self.committed_generation
        self.ready_mask = 0
        self.active = False


@dataclass(frozen=True)
class B7DeviceRound:
    """One C1/M8 tree and the persistent device views containing its topology."""

    request_id: str
    generation: int
    metadata: TreeAttentionMetadata
    parent: Any
    ancestor_mask: Any
    parent_identity: tuple[object, ...]
    ancestor_identity: tuple[object, ...]
    verifier_outputs: Any | None
    commit_generation: B7CommitGeneration | None
    commit_capacity: int


_LOCK = Lock()
_ACTIVE: B7DeviceRound | None = None
_OWNER_IDENTITIES: dict[int, tuple[tuple[object, ...], tuple[object, ...]]] = {}
_OWNER_GENERATIONS: dict[int, int] = {}


def _tensor_identity(tensor: Any, *, name: str, dtype_name: str) -> tuple[object, ...]:
    """Validate one exact CUDA topology view without reading its contents."""

    try:
        shape = tuple(tensor.shape)
        dtype = str(tensor.dtype)
        device = tensor.device
        contiguous = bool(tensor.is_contiguous())
        identity = (
            int(tensor.data_ptr()),
            int(tensor.storage_offset()),
            shape,
            tuple(tensor.stride()),
            dtype,
            str(device),
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise B7DeviceRuntimeError(f"B7 {name} is not a tensor-like device view") from error
    if (
        shape != (7,)
        or dtype != dtype_name
        or getattr(device, "type", None) != "cuda"
        or not contiguous
    ):
        raise B7DeviceRuntimeError(f"B7 {name} must be contiguous CUDA {dtype_name[6:]}[7]")
    return identity


def publish_b7_device_round(
    *,
    owner: object,
    request_id: str,
    metadata: dict[str, object] | TreeAttentionMetadata,
    parent: Any,
    ancestor_mask: Any,
    verifier_outputs: Any | None = None,
    commit_generation: B7CommitGeneration | None = None,
    commit_capacity: int = 8,
) -> B7DeviceRound:
    """Publish one validated round using storage permanently owned by ``owner``."""

    if not isinstance(request_id, str) or not request_id:
        raise B7DeviceRuntimeError("B7 device round requires one non-empty request ID")
    normalized = (
        metadata
        if isinstance(metadata, TreeAttentionMetadata)
        else validate_tree_metadata(metadata, max_nodes=7)
    )
    if normalized.node_count != 7:
        raise B7DeviceRuntimeError("B7 device round requires exactly seven nodes")
    if (
        isinstance(commit_capacity, bool)
        or not isinstance(commit_capacity, int)
        or not 1 <= commit_capacity <= 8
    ):
        raise B7DeviceRuntimeError("B7 commit capacity must be an integer in [1, 8]")
    parent_identity = _tensor_identity(parent, name="parent", dtype_name="torch.int32")
    ancestor_identity = _tensor_identity(
        ancestor_mask,
        name="ancestor mask",
        dtype_name="torch.uint32",
    )
    if str(parent.device) != str(ancestor_mask.device):
        raise B7DeviceRuntimeError("B7 parent and ancestor buffers must share one device")

    owner_key = id(owner)
    identities = (parent_identity, ancestor_identity)
    with _LOCK:
        stable = _OWNER_IDENTITIES.setdefault(owner_key, identities)
        if stable != identities:
            raise B7DeviceRuntimeError("B7 owner rebound its persistent topology storage")
        generation = _OWNER_GENERATIONS.get(owner_key, -1) + 1
        _OWNER_GENERATIONS[owner_key] = generation
        round_state = B7DeviceRound(
            request_id=request_id,
            generation=generation,
            metadata=normalized,
            parent=parent,
            ancestor_mask=ancestor_mask,
            parent_identity=parent_identity,
            ancestor_identity=ancestor_identity,
            verifier_outputs=verifier_outputs,
            commit_generation=commit_generation,
            commit_capacity=commit_capacity,
        )
        global _ACTIVE
        if _ACTIVE is not None:
            raise B7DeviceRuntimeError("prior B7 device round was not cleared")
        _ACTIVE = round_state
        return round_state


def get_b7_device_round() -> B7DeviceRound | None:
    """Return the current immutable round descriptor, never a copied tensor."""

    with _LOCK:
        return _ACTIVE


def require_b7_device_round(
    metadata: dict[str, object] | TreeAttentionMetadata | None = None,
) -> B7DeviceRound:
    """Require one active round and optionally bind it to serialized metadata."""

    with _LOCK:
        active = _ACTIVE
    if active is None:
        raise B7DeviceRuntimeError("B7 device round is absent")
    if metadata is not None:
        normalized = (
            metadata
            if isinstance(metadata, TreeAttentionMetadata)
            else validate_tree_metadata(metadata, max_nodes=7)
        )
        if normalized != active.metadata:
            raise B7DeviceRuntimeError("B7 serialized and device topology differ")
    if (
        _tensor_identity(active.parent, name="parent", dtype_name="torch.int32")
        != active.parent_identity
    ):
        raise B7DeviceRuntimeError("B7 parent storage identity changed after publication")
    if (
        _tensor_identity(
            active.ancestor_mask,
            name="ancestor mask",
            dtype_name="torch.uint32",
        )
        != active.ancestor_identity
    ):
        raise B7DeviceRuntimeError("B7 ancestor storage identity changed after publication")
    return active


def require_b7_verifier_outputs() -> Any:
    """Return the runner-owned device outputs or reject an incomplete round."""

    active = require_b7_device_round()
    outputs = active.verifier_outputs
    if outputs is None:
        raise B7DeviceRuntimeError("B7 verifier output workspace is absent")
    required = (
        "accepted_count",
        "accepted_nodes",
        "accepted_leaf_physical_row",
        "accepted_leaf_state_slot",
        "status",
    )
    if any(getattr(outputs, name, None) is None for name in required):
        raise B7DeviceRuntimeError("B7 verifier output workspace is incomplete")
    return outputs


def clear_b7_device_round(*, owner: object) -> None:
    """End exactly the round published by ``owner`` before draft execution."""

    owner_key = id(owner)
    with _LOCK:
        global _ACTIVE
        if _ACTIVE is None:
            return
        stable = _OWNER_IDENTITIES.get(owner_key)
        if stable != (_ACTIVE.parent_identity, _ACTIVE.ancestor_identity):
            raise B7DeviceRuntimeError("another owner attempted to clear the B7 device round")
        _ACTIVE = None


def _reset_for_tests() -> None:
    """Clear all process state; tests only."""

    with _LOCK:
        global _ACTIVE
        _ACTIVE = None
        _OWNER_IDENTITIES.clear()
        _OWNER_GENERATIONS.clear()


__all__ = [
    "B7DeviceRound",
    "B7DeviceRuntimeError",
    "B7CommitGeneration",
    "clear_b7_device_round",
    "get_b7_device_round",
    "publish_b7_device_round",
    "require_b7_device_round",
    "require_b7_verifier_outputs",
]
