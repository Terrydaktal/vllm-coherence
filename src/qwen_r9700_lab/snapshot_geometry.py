"""Canonical fixed-slot checkpoint geometry for snapshot tooling.

This module is deliberately independent of vLLM and of any serving lane.  The
controller, verifier, and runtime-shadow renderer use the same constants and
formula so a lane optimization cannot silently redefine a durable checkpoint
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

SNAPSHOT_GEOMETRY_ABI = "qwen-fixed-slot-checkpoint-geometry-v2"

TOKENS_PER_HASH = 8
PAGE_TOKENS = 1_648
MAX_PROMPT_TOKENS = 253_792

CURRENT_ROOT_PROMPT_TOKENS = 60_298
CURRENT_ROOT_HASH_ALIGNED_TOKENS = 60_296
CURRENT_ROOT_REPLAY_BOUNDARY_TOKENS = 60_288

# This is the first qualified geometry with six physical page ordinals.  Its
# final five ordinals satisfy the sliding-window retention topology (including
# the partial final page) while retaining one exact recurrent-state page per
# GDN group.
SHORT_ROOT_PROMPT_TOKENS = 8_258
SHORT_ROOT_HASH_ALIGNED_TOKENS = 8_256
SHORT_ROOT_REPLAY_BOUNDARY_TOKENS = 8_248

SLIDING_WINDOW_RETAINED_PAGES = 5
MIN_CHECKPOINT_PAGE_ORDINALS = SLIDING_WINDOW_RETAINED_PAGES + 1
CHECKPOINT_STRIDE_TOKENS = PAGE_TOKENS


class SnapshotGeometryError(ValueError):
    """The requested checkpoint cannot satisfy the fixed-slot ABI."""


@dataclass(frozen=True, slots=True)
class FixedSlotCheckpointGeometry:
    """The three token boundaries and page topology for one prompt prefix."""

    prompt_tokens: int
    hash_aligned_tokens: int
    replay_boundary_tokens: int
    tokens_per_hash: int = TOKENS_PER_HASH
    page_tokens: int = PAGE_TOKENS

    @property
    def full_attention_pages(self) -> int:
        return (self.replay_boundary_tokens + self.page_tokens - 1) // self.page_tokens

    @property
    def requires_partial_snapshot(self) -> bool:
        return self.replay_boundary_tokens % self.page_tokens != 0

    @property
    def replay_hash_units(self) -> int:
        return (self.hash_aligned_tokens - self.replay_boundary_tokens) // self.tokens_per_hash


def boundary_geometry(prompt_tokens: int) -> FixedSlotCheckpointGeometry:
    """Return the lane-independent ``P -> H -> R`` arithmetic.

    ``H`` is the greatest eight-token hash boundary at or before ``P`` and
    ``R`` is normally one hash unit earlier.  If ``H`` or that one-unit
    boundary lands on a physical page edge, ``R`` moves back one additional
    hash unit.  This gives every supported terminal ``P`` a source page that
    can be copied before it is overwritten.  Policy gates (60K production
    root versus the optional short root) are intentionally applied separately.
    """

    if (
        type(prompt_tokens) is not int
        or prompt_tokens < TOKENS_PER_HASH
        or prompt_tokens > MAX_PROMPT_TOKENS
    ):
        raise SnapshotGeometryError("prompt length is outside the snapshot ABI range")
    hash_aligned_tokens = prompt_tokens // TOKENS_PER_HASH * TOKENS_PER_HASH
    replay_hash_units = 2 if hash_aligned_tokens % PAGE_TOKENS in (0, TOKENS_PER_HASH) else 1
    replay_boundary_tokens = hash_aligned_tokens - replay_hash_units * TOKENS_PER_HASH
    if replay_boundary_tokens < 0:
        raise SnapshotGeometryError("checkpoint replay boundary is negative")
    return FixedSlotCheckpointGeometry(
        prompt_tokens=prompt_tokens,
        hash_aligned_tokens=hash_aligned_tokens,
        replay_boundary_tokens=replay_boundary_tokens,
    )


def checkpoint_geometry(
    prompt_tokens: int,
    *,
    allow_short_root: bool = False,
) -> FixedSlotCheckpointGeometry:
    """Return a policy-qualified checkpoint geometry.

    The existing 60,298-token minimum remains the default.  The 8,258-token
    root is available only through an explicit opt-in and still enforces the
    five-page sliding-window topology.
    """

    if type(allow_short_root) is not bool:
        raise SnapshotGeometryError("allow_short_root must be a boolean")
    minimum = SHORT_ROOT_PROMPT_TOKENS if allow_short_root else CURRENT_ROOT_PROMPT_TOKENS
    geometry = boundary_geometry(prompt_tokens)
    if geometry.prompt_tokens < minimum:
        raise SnapshotGeometryError("prompt length precedes the enabled checkpoint root")
    if geometry.full_attention_pages < MIN_CHECKPOINT_PAGE_ORDINALS:
        raise SnapshotGeometryError("checkpoint lacks the retained-page topology")
    return geometry


def checkpoint_prompt_tokens(
    ordinal: int,
    *,
    allow_short_root: bool = False,
) -> int:
    """Return checkpoint ``ordinal`` on the page-stride milestone lattice."""

    if type(ordinal) is not int or ordinal < 0:
        raise SnapshotGeometryError("checkpoint ordinal must be a non-negative integer")
    root = SHORT_ROOT_PROMPT_TOKENS if allow_short_root else CURRENT_ROOT_PROMPT_TOKENS
    prompt_tokens = root + ordinal * CHECKPOINT_STRIDE_TOKENS
    # Apply the same maximum and topology checks as arbitrary input.
    return checkpoint_geometry(
        prompt_tokens,
        allow_short_root=allow_short_root,
    ).prompt_tokens


def next_checkpoint_prompt_tokens(
    completed_prompt_tokens: int,
    *,
    allow_short_root: bool = False,
) -> int:
    """Return the first milestone strictly beyond a durable prefix length."""

    if type(completed_prompt_tokens) is not int or completed_prompt_tokens < 0:
        raise SnapshotGeometryError("completed prompt length must be a non-negative integer")
    root = SHORT_ROOT_PROMPT_TOKENS if allow_short_root else CURRENT_ROOT_PROMPT_TOKENS
    if completed_prompt_tokens < root:
        return root
    ordinal = (completed_prompt_tokens - root) // CHECKPOINT_STRIDE_TOKENS + 1
    return checkpoint_prompt_tokens(ordinal, allow_short_root=allow_short_root)
