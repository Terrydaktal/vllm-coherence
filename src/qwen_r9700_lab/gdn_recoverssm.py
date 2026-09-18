"""Exact linear-chain RecoverSSM oracle and capacity contract.

The production HIP kernel keeps one complete recurrent checkpoint and one
compact factor record for the previous speculative verification window.  This
module mirrors that state machine using plain Python lists so rollback and
late-commit semantics can be regression-tested without a GPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

Vector = list[float]
Matrix = list[Vector]

CHECKPOINT_BLOCK_SIZE = 8
TRAINED_SPECULATIVE_TOKENS = CHECKPOINT_BLOCK_SIZE - 1
TRAINED_SPEC_WIDTH = CHECKPOINT_BLOCK_SIZE
LEGACY_SPEC_WIDTH = TRAINED_SPEC_WIDTH + 1
CAPABILITY = (
    "qwen-linear-chain-v2/state_pages=1/spec_width=8/mamba_mode=none/"
    "max_num_seqs=1/checkpoint_block=8"
)
LEGACY_CAPABILITY = "qwen-linear-chain-v1/state_pages=1/spec_width=9/mamba_mode=none"
STATE_PAGES = 1
ALIGN_CAPABILITY = (
    "qwen-linear-chain-v3/state_pages=2/spec_width=8/mamba_mode=align/"
    "max_num_seqs=1/checkpoint_block=8"
)
LEGACY_ALIGN_CAPABILITY = (
    "qwen-linear-chain-v2/state_pages=2/spec_width=9/mamba_mode=align/max_num_seqs=1"
)
ALIGN_STATE_PAGES = 2
PRODUCTION_SPEC_WIDTH = TRAINED_SPEC_WIDTH
MAX_KERNEL_WIDTH = 16


@dataclass(frozen=True)
class Step:
    """One normalized GDN recurrence input for one value head."""

    q: tuple[float, ...]
    k: tuple[float, ...]
    v: tuple[float, ...]
    beta: float
    decay: float


@dataclass(frozen=True)
class CompactRecord:
    """Exact factors needed to reconstruct any prefix of a linear window."""

    keys: tuple[tuple[float, ...], ...]
    corrections: tuple[tuple[float, ...], ...]
    decays: tuple[float, ...]

    @property
    def width(self) -> int:
        return len(self.decays)


@dataclass(frozen=True)
class Verification:
    outputs: tuple[tuple[float, ...], ...]
    full_states: tuple[Matrix, ...]
    record: CompactRecord


@dataclass(frozen=True)
class AlignSidecar:
    """Request-local factors bound to one physical prefix checkpoint."""

    record: CompactRecord | None
    record_state_slot: int
    pending_accepted: int


@dataclass(frozen=True)
class DFlashPrefixHitPlan:
    """Exact aligned replay points shared by EAGLE attention and Mamba."""

    replay_boundary_tokens: int
    full_attention_hit_tokens: int
    retained_mamba_blocks: tuple[int, ...]
    reconciled_hit_tokens: int


def _copy_state(state: Sequence[Sequence[float]]) -> Matrix:
    if not state or not state[0]:
        raise ValueError("state must be non-empty")
    width = len(state[0])
    if any(len(row) != width for row in state):
        raise ValueError("state must be rectangular")
    return [list(row) for row in state]


def plan_dflash_align_prefix_hit(
    *,
    cached_prompt_tokens: int,
    block_size: int,
    alignment_tokens: int,
) -> DFlashPrefixHitPlan:
    """Model the exact align-v2 cache boundary retained for DFlash/EAGLE.

    Full attention matches the lookahead block and then drops one scheduler
    alignment unit. Sparse Mamba retention must therefore preserve both its
    ordinary replay-boundary state and the predecessor state at that shortened
    boundary. Block indices are zero based; block ``i`` ends at
    ``(i + 1) * block_size`` tokens.
    """

    if cached_prompt_tokens <= 0:
        raise ValueError("cached prompt must contain at least one token")
    if block_size <= 0 or alignment_tokens <= 0:
        raise ValueError("block and alignment sizes must be positive")
    if alignment_tokens % block_size:
        raise ValueError("scheduler alignment must be a multiple of block size")

    replay_boundary = (cached_prompt_tokens - 1) // alignment_tokens * alignment_tokens
    boundary_block = replay_boundary // block_size - 1
    predecessor_block = boundary_block - alignment_tokens // block_size
    retained = tuple(block for block in (boundary_block, predecessor_block) if block >= 0)
    full_attention_hit = max(0, replay_boundary - alignment_tokens)
    mamba_hit = max(
        (
            (block + 1) * block_size
            for block in retained
            if (block + 1) * block_size <= full_attention_hit
        ),
        default=0,
    )
    return DFlashPrefixHitPlan(
        replay_boundary_tokens=replay_boundary,
        full_attention_hit_tokens=full_attention_hit,
        retained_mamba_blocks=retained,
        reconciled_hit_tokens=min(full_attention_hit, mamba_hit),
    )


def _validate_step(step: Step, value_dim: int, key_dim: int) -> None:
    if len(step.q) != key_dim or len(step.k) != key_dim:
        raise ValueError("q/k dimension does not match state")
    if len(step.v) != value_dim:
        raise ValueError("v dimension does not match state")


def verify(checkpoint: Sequence[Sequence[float]], steps: Sequence[Step]) -> Verification:
    """Evaluate one candidate chain and retain exact rank-one factors."""

    if not 1 <= len(steps) <= MAX_KERNEL_WIDTH:
        raise ValueError(f"verification width must be in [1,{MAX_KERNEL_WIDTH}]")
    state = _copy_state(checkpoint)
    value_dim = len(state)
    key_dim = len(state[0])
    outputs: list[tuple[float, ...]] = []
    states: list[Matrix] = []
    keys: list[tuple[float, ...]] = []
    corrections: list[tuple[float, ...]] = []
    decays: list[float] = []
    for step in steps:
        _validate_step(step, value_dim, key_dim)
        for value in range(value_dim):
            for key in range(key_dim):
                state[value][key] *= step.decay
        correction: list[float] = []
        for value in range(value_dim):
            projection = sum(state[value][key] * step.k[key] for key in range(key_dim))
            delta = (step.v[value] - projection) * step.beta
            correction.append(delta)
            for key in range(key_dim):
                state[value][key] += step.k[key] * delta
        outputs.append(
            tuple(
                sum(state[value][key] * step.q[key] for key in range(key_dim))
                for value in range(value_dim)
            )
        )
        states.append(_copy_state(state))
        keys.append(step.k)
        corrections.append(tuple(correction))
        decays.append(step.decay)
    return Verification(
        tuple(outputs),
        tuple(states),
        CompactRecord(tuple(keys), tuple(corrections), tuple(decays)),
    )


def recover(
    checkpoint: Sequence[Sequence[float]],
    record: CompactRecord,
    accepted: int,
) -> Matrix:
    """Late-commit exactly the accepted prefix, discarding its rejected suffix."""

    if accepted < 0 or accepted > record.width:
        raise ValueError("accepted count is outside the compact record")
    state = _copy_state(checkpoint)
    value_dim = len(state)
    key_dim = len(state[0])
    for index in range(accepted):
        key_vector = record.keys[index]
        correction = record.corrections[index]
        if len(key_vector) != key_dim or len(correction) != value_dim:
            raise ValueError("compact record dimension does not match checkpoint")
        for value in range(value_dim):
            for key in range(key_dim):
                state[value][key] *= record.decays[index]
            for key in range(key_dim):
                state[value][key] += key_vector[key] * correction[value]
    return state


def migrate_align_sidecar(
    sidecar: AlignSidecar,
    *,
    source_slot: int,
    destination_slot: int,
    accepted: int,
) -> AlignSidecar:
    """Move a compact record with its base checkpoint at an align boundary.

    The recurrent page copy is always the *base* source page.  ``accepted`` is
    preserved separately and is not converted into a temporal-page offset.
    """

    if accepted < 0 or accepted > MAX_KERNEL_WIDTH:
        raise ValueError(f"accepted target-row count must be in [0,{MAX_KERNEL_WIDTH}]")
    if source_slot < 0 or destination_slot < 0:
        raise ValueError("physical state slots must be non-negative")
    if sidecar.record is None or sidecar.record_state_slot != source_slot:
        return AlignSidecar(None, -1, -1)
    if accepted > sidecar.record.width:
        raise ValueError("accepted target-row count exceeds the compact record width")
    return AlignSidecar(sidecar.record, destination_slot, accepted)


def consume_align_sidecar(
    checkpoint: Sequence[Sequence[float]],
    sidecar: AlignSidecar,
    *,
    current_slot: int,
) -> tuple[Matrix, AlignSidecar]:
    """Late-commit a matching cached prefix and atomically clear its record.

    A slot mismatch represents an unrelated prefix-cache entry and therefore
    fails closed by discarding the factors without modifying its checkpoint.
    """

    if (
        sidecar.record is None
        or sidecar.pending_accepted < 0
        or sidecar.record_state_slot != current_slot
    ):
        return _copy_state(checkpoint), AlignSidecar(None, -1, -1)
    committed = recover(checkpoint, sidecar.record, sidecar.pending_accepted)
    return committed, AlignSidecar(None, -1, -1)


def copy_align_states(
    temporal_checkpoint: Sequence[Sequence[float]],
    conv_state: Sequence[Sequence[float]],
    *,
    accepted: int,
    spec_width: int = PRODUCTION_SPEC_WIDTH,
) -> tuple[Matrix, Matrix, int]:
    """Mirror RecoverSSM align copy semantics for temporal and conv state.

    The temporal checkpoint is copied without an accepted-token column offset.
    The convolution window keeps the ordinary ``accepted - 1`` left shift and
    the request-visible accepted counter resets to the neutral value one.
    """

    if spec_width <= 0 or spec_width > MAX_KERNEL_WIDTH:
        raise ValueError(f"spec_width must be in [1,{MAX_KERNEL_WIDTH}]")
    if accepted < 0 or accepted > spec_width:
        raise ValueError(f"accepted target-row count must be in [0,{spec_width}]")
    conv = _copy_state(conv_state)
    token_bias = max(accepted - 1, 0)
    if token_bias:
        width = len(conv[0])
        for row in conv:
            shifted = row[token_bias:]
            row[:] = shifted + [0.0] * (width - len(shifted))
    return _copy_state(temporal_checkpoint), conv, 1


def compact_sidecar_bytes(
    *,
    value_heads: int = 48,
    width: int = PRODUCTION_SPEC_WIDTH,
    key_dim: int = 128,
    value_dim: int = 128,
    requests: int = 1,
) -> int:
    """FP32 correction + duplicated K + decay, plus one int32 record count."""

    if min(value_heads, width, key_dim, value_dim, requests) <= 0:
        raise ValueError("all dimensions must be positive")
    correction = requests * value_heads * width * value_dim * 4
    duplicated_key = requests * value_heads * width * key_dim * 4
    decay = requests * value_heads * width * 4
    record_count = requests * 4
    return correction + duplicated_key + decay + record_count


def speculative_conv_width(conv_kernel: int, speculative_tokens: int) -> int:
    """Conv history inside the sole page remains widened for rollback."""

    if conv_kernel <= 0 or speculative_tokens < 0:
        raise ValueError("invalid convolution geometry")
    return conv_kernel - 1 + speculative_tokens
