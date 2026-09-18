"""Process-local hand-off for optional tree-aware attention backends.

The stock vLLM 0.26 scheduler carries speculative tokens but not proposal-tree
metadata.  The tree ABI overlay publishes the metadata immediately before a
worker forward so an attention/GDN backend can consume the current request
mapping without changing the normal linear-token path.

This module intentionally has no torch or vLLM dependency.  It is safe to
import from compiled-kernel adapters and is a no-op until the overlay publishes
metadata.
"""

from __future__ import annotations

from threading import Lock
from typing import Any

from .tree_attention_contract import normalize_tree_metadata


_LOCK = Lock()
_ACTIVE: dict[str, dict[str, Any] | None] = {}


def publish_tree_metadata(metadata: dict[str, dict[str, Any] | None] | None) -> None:
    """Replace the metadata for the currently executing scheduler step."""
    with _LOCK:
        _ACTIVE.clear()
        if metadata:
            for req_id, value in metadata.items():
                if value is not None:
                    value = normalize_tree_metadata(value)
                _ACTIVE[str(req_id)] = value


def get_tree_metadata(req_id: str | None = None) -> Any:
    """Return all active metadata, or one request's metadata."""
    with _LOCK:
        if req_id is None:
            return dict(_ACTIVE)
        return _ACTIVE.get(req_id)


def clear_tree_metadata() -> None:
    """Clear the current-step hand-off after a forward or test."""
    with _LOCK:
        _ACTIVE.clear()


def get_single_tree_metadata() -> dict[str, object] | None:
    """Return the sole active tree, or ``None`` for an ambiguous step.

    The current C1 profile is single-request.  Refusing to guess when multiple
    requests are active prevents an attention call without a request ID from
    applying another request's ancestor mask.
    """
    with _LOCK:
        values = [value for value in _ACTIVE.values() if value is not None]
        if len(values) != 1:
            return None
        return dict(values[0])


__all__ = [
    "clear_tree_metadata",
    "get_single_tree_metadata",
    "get_tree_metadata",
    "publish_tree_metadata",
]
