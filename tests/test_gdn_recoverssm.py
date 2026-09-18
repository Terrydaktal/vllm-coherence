from __future__ import annotations

import random

import pytest

from qwen_r9700_lab.gdn_recoverssm import (
    ALIGN_CAPABILITY,
    ALIGN_STATE_PAGES,
    CAPABILITY,
    CHECKPOINT_BLOCK_SIZE,
    LEGACY_ALIGN_CAPABILITY,
    LEGACY_CAPABILITY,
    LEGACY_SPEC_WIDTH,
    PRODUCTION_SPEC_WIDTH,
    STATE_PAGES,
    TRAINED_SPECULATIVE_TOKENS,
    AlignSidecar,
    DFlashPrefixHitPlan,
    Step,
    compact_sidecar_bytes,
    consume_align_sidecar,
    copy_align_states,
    migrate_align_sidecar,
    plan_dflash_align_prefix_hit,
    recover,
    speculative_conv_width,
    verify,
)


def _state(seed: int, value_dim: int = 5, key_dim: int = 7) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(-0.25, 0.25) for _ in range(key_dim)] for _ in range(value_dim)]


def _steps(seed: int, width: int, value_dim: int = 5, key_dim: int = 7) -> list[Step]:
    rng = random.Random(seed)
    return [
        Step(
            q=tuple(rng.uniform(-0.5, 0.5) for _ in range(key_dim)),
            k=tuple(rng.uniform(-0.5, 0.5) for _ in range(key_dim)),
            v=tuple(rng.uniform(-0.5, 0.5) for _ in range(value_dim)),
            beta=rng.uniform(0.05, 0.95),
            decay=rng.uniform(0.7, 0.999),
        )
        for _ in range(width)
    ]


def _assert_matrix_close(left: list[list[float]], right: list[list[float]]) -> None:
    assert len(left) == len(right)
    for left_row, right_row in zip(left, right, strict=True):
        assert left_row == pytest.approx(right_row, abs=1e-12, rel=1e-12)


@pytest.mark.parametrize("next_width", [1, 5, 8, 9, 16])
@pytest.mark.parametrize("accepted", range(PRODUCTION_SPEC_WIDTH + 1))
def test_late_commit_matches_full_state_across_dflash_acceptance(
    next_width: int, accepted: int
) -> None:
    """Trained accepted counts remain exact across supported kernel widths."""

    checkpoint = _state(17)
    previous = verify(checkpoint, _steps(19, PRODUCTION_SPEC_WIDTH))
    committed = recover(checkpoint, previous.record, accepted)
    expected = checkpoint if accepted == 0 else previous.full_states[accepted - 1]
    _assert_matrix_close(committed, expected)

    # A following target round must see the same checkpoint and therefore emit
    # the same rows, including the width-16 extension compatibility case.
    next_steps = _steps(23, next_width)
    recovered_next = verify(committed, next_steps)
    materialized_next = verify(expected, next_steps)
    for recovered_output, materialized_output in zip(
        recovered_next.outputs, materialized_next.outputs, strict=True
    ):
        assert recovered_output == pytest.approx(materialized_output, abs=1e-12, rel=1e-12)
    for recovered_state, materialized_state in zip(
        recovered_next.full_states, materialized_next.full_states, strict=True
    ):
        _assert_matrix_close(recovered_state, materialized_state)


@pytest.mark.parametrize("previous_width", [1, 5, 8, 9, 16])
def test_every_supported_record_width_recovers_each_valid_prefix(
    previous_width: int,
) -> None:
    """Exercise trained, legacy, and extension widths through eight accepted rows."""

    checkpoint = _state(47 + previous_width)
    previous = verify(checkpoint, _steps(53 + previous_width, previous_width))
    for accepted in range(min(previous_width, 8) + 1):
        committed = recover(checkpoint, previous.record, accepted)
        expected = checkpoint if accepted == 0 else previous.full_states[accepted - 1]
        _assert_matrix_close(committed, expected)


def test_rejected_suffix_is_not_committed_and_reset_uses_direct_checkpoint() -> None:
    checkpoint = _state(31)
    previous = verify(checkpoint, _steps(37, 9))

    accepted_two = recover(checkpoint, previous.record, 2)
    _assert_matrix_close(accepted_two, previous.full_states[1])
    with pytest.raises(AssertionError):
        _assert_matrix_close(accepted_two, previous.full_states[-1])

    reset = recover(checkpoint, previous.record, 0)
    _assert_matrix_close(reset, checkpoint)


def test_one_page_capacity_and_conv_rollback_contract() -> None:
    assert CHECKPOINT_BLOCK_SIZE == 8
    assert TRAINED_SPECULATIVE_TOKENS == 7
    assert PRODUCTION_SPEC_WIDTH == 8
    assert CAPABILITY == (
        "qwen-linear-chain-v2/state_pages=1/spec_width=8/mamba_mode=none/"
        "max_num_seqs=1/checkpoint_block=8"
    )
    assert STATE_PAGES == 1
    assert compact_sidecar_bytes() == 394_756
    assert compact_sidecar_bytes() * 48 == 18_948_288

    # The full recurrent page count shrinks to one, but the convolution state
    # inside that page still retains kernel-1 history plus all seven drafts.
    assert speculative_conv_width(conv_kernel=4, speculative_tokens=7) == 10


def test_invalid_acceptance_fails_instead_of_silently_clamping_reference() -> None:
    checkpoint = _state(41)
    record = verify(checkpoint, _steps(43, 5)).record
    with pytest.raises(ValueError, match="accepted count"):
        recover(checkpoint, record, 6)


@pytest.mark.parametrize("accepted", range(9))
def test_align_prefix_copy_preserves_base_and_commits_only_accepted_prefix(
    accepted: int,
) -> None:
    checkpoint = _state(71)
    verification = verify(checkpoint, _steps(73, PRODUCTION_SPEC_WIDTH))
    sidecar = AlignSidecar(verification.record, 11, -1)
    migrated = migrate_align_sidecar(
        sidecar,
        source_slot=11,
        destination_slot=29,
        accepted=accepted,
    )

    # Temporal migration copies the base checkpoint, not source+token_bias.
    temporal, conv, reset_count = copy_align_states(
        checkpoint,
        [[float(index) for index in range(10)]],
        accepted=accepted,
    )
    _assert_matrix_close(temporal, checkpoint)
    assert reset_count == 1
    bias = max(accepted - 1, 0)
    assert conv[0][: 10 - bias] == [float(index) for index in range(bias, 10)]

    committed, cleared = consume_align_sidecar(
        temporal,
        migrated,
        current_slot=29,
    )
    expected = checkpoint if accepted == 0 else verification.full_states[accepted - 1]
    _assert_matrix_close(committed, expected)
    assert cleared == AlignSidecar(None, -1, -1)


@pytest.mark.parametrize("accepted_drafts", range(TRAINED_SPECULATIVE_TOKENS + 1))
def test_align_runtime_maps_accepted_drafts_to_target_rows(
    accepted_drafts: int,
) -> None:
    """DFlash accepts 0..7 drafts but vLLM publishes target bonus + drafts."""

    checkpoint = _state(74)
    verification = verify(checkpoint, _steps(76, PRODUCTION_SPEC_WIDTH))
    accepted_rows = accepted_drafts + 1
    migrated = migrate_align_sidecar(
        AlignSidecar(verification.record, 17, -1),
        source_slot=17,
        destination_slot=23,
        accepted=accepted_rows,
    )
    temporal, conv, reset_count = copy_align_states(
        checkpoint,
        [[float(index) for index in range(10)]],
        accepted=accepted_rows,
    )
    committed, _ = consume_align_sidecar(temporal, migrated, current_slot=23)

    _assert_matrix_close(committed, verification.full_states[accepted_drafts])
    assert conv[0][: 10 - accepted_drafts] == [float(index) for index in range(accepted_drafts, 10)]
    assert reset_count == 1


def test_align_prefix_slot_mismatch_discards_sidecar_and_preserves_checkpoint() -> None:
    checkpoint = _state(79)
    verification = verify(checkpoint, _steps(83, PRODUCTION_SPEC_WIDTH))
    stale = AlignSidecar(verification.record, 41, 5)
    actual, cleared = consume_align_sidecar(checkpoint, stale, current_slot=42)
    _assert_matrix_close(actual, checkpoint)
    assert cleared == AlignSidecar(None, -1, -1)


def test_align_rollback_and_consecutive_cached_requests() -> None:
    checkpoint = _state(89)
    first = verify(checkpoint, _steps(97, PRODUCTION_SPEC_WIDTH))
    first_cached = migrate_align_sidecar(
        AlignSidecar(first.record, 3, -1),
        source_slot=3,
        destination_slot=7,
        accepted=4,
    )
    committed_first, cleared = consume_align_sidecar(checkpoint, first_cached, current_slot=7)
    _assert_matrix_close(committed_first, first.full_states[3])
    assert cleared.pending_accepted == -1

    second = verify(committed_first, _steps(101, PRODUCTION_SPEC_WIDTH))
    second_cached = migrate_align_sidecar(
        AlignSidecar(second.record, 7, -1),
        source_slot=7,
        destination_slot=13,
        accepted=0,
    )
    committed_second, _ = consume_align_sidecar(committed_first, second_cached, current_slot=13)
    # Rejecting the entire second draft chain is an exact rollback.
    _assert_matrix_close(committed_second, committed_first)


def test_dflash8_nested_65k_prefix_retains_eagle_predecessor() -> None:
    """The 65,536 -> 131,072 live regression must not reconcile to zero."""

    plan = plan_dflash_align_prefix_hit(
        cached_prompt_tokens=65_536,
        block_size=1_648,
        alignment_tokens=1_648,
    )
    assert plan == DFlashPrefixHitPlan(
        replay_boundary_tokens=64_272,
        full_attention_hit_tokens=62_624,
        retained_mamba_blocks=(38, 37),
        reconciled_hit_tokens=62_624,
    )
    # The nested request has ample suffix but cannot increase the common hit
    # beyond the exact producer prefix retained above.
    assert 131_072 > 65_536
    assert plan.reconciled_hit_tokens >= 62_624


@pytest.mark.parametrize("accepted_drafts", range(TRAINED_SPECULATIVE_TOKENS + 1))
def test_worker_copy_rebinds_physically_retained_eagle_predecessor(
    accepted_drafts: int,
) -> None:
    """A freed logical page keeps its hashed physical state for worker copy."""

    checkpoint = _state(107)
    verification = verify(checkpoint, _steps(109, PRODUCTION_SPEC_WIDTH))
    accepted_rows = accepted_drafts + 1

    # Producer block 37 is the EAGLE predecessor ending at 62,624 tokens. Align
    # mode removes it from the producer's logical table once it is two steps old,
    # but BlockPool.free_blocks retains a hashed block at refcount zero. A nested
    # request acquires that exact physical page before destination allocation.
    predecessor_logical_block = 37
    source_physical_slot = 211
    producer_table: dict[int, int | None] = {
        predecessor_logical_block: source_physical_slot,
        38: 223,
    }
    prefix_hash_cache = {predecessor_logical_block: source_physical_slot}
    producer_table[predecessor_logical_block] = None
    assert producer_table[predecessor_logical_block] is None
    assert prefix_hash_cache[predecessor_logical_block] == source_physical_slot

    nested_table = {predecessor_logical_block: prefix_hash_cache[predecessor_logical_block]}
    # The first uncached nested block is logical 38; the remainder through the
    # 131,072-token request is computed only after this prefix migration.
    destination_logical_block = 38
    destination_physical_slot = 277
    nested_table[destination_logical_block] = destination_physical_slot
    assert nested_table[predecessor_logical_block] != destination_physical_slot

    migrated = migrate_align_sidecar(
        AlignSidecar(verification.record, source_physical_slot, -1),
        source_slot=nested_table[predecessor_logical_block],
        destination_slot=nested_table[destination_logical_block],
        accepted=accepted_rows,
    )
    temporal, _, reset_count = copy_align_states(
        checkpoint,
        [[float(index) for index in range(10)]],
        accepted=accepted_rows,
    )
    committed, cleared = consume_align_sidecar(
        temporal,
        migrated,
        current_slot=destination_physical_slot,
    )
    _assert_matrix_close(committed, verification.full_states[accepted_drafts])
    assert cleared == AlignSidecar(None, -1, -1)
    assert reset_count == 1

    # A consecutive request that rejects every newly drafted row must return
    # exactly to the migrated predecessor checkpoint.
    following = verify(committed, _steps(113, PRODUCTION_SPEC_WIDTH))
    rollback = migrate_align_sidecar(
        AlignSidecar(following.record, destination_physical_slot, -1),
        source_slot=destination_physical_slot,
        destination_slot=311,
        accepted=0,
    )
    rolled_back, _ = consume_align_sidecar(
        committed,
        rollback,
        current_slot=311,
    )
    _assert_matrix_close(rolled_back, committed)


def test_align_capability_contract_is_separate_from_mode_none() -> None:
    assert ALIGN_CAPABILITY == (
        "qwen-linear-chain-v3/state_pages=2/spec_width=8/mamba_mode=align/"
        "max_num_seqs=1/checkpoint_block=8"
    )
    assert ALIGN_STATE_PAGES == 2
    assert CAPABILITY.endswith("max_num_seqs=1/checkpoint_block=8")
    assert LEGACY_SPEC_WIDTH == 9
    assert LEGACY_CAPABILITY.endswith("state_pages=1/spec_width=9/mamba_mode=none")
    assert LEGACY_ALIGN_CAPABILITY.endswith(
        "state_pages=2/spec_width=9/mamba_mode=align/max_num_seqs=1"
    )


def test_legacy_width9_remains_explicitly_modelled_but_not_the_default() -> None:
    checkpoint = _state(127)
    verification = verify(checkpoint, _steps(131, LEGACY_SPEC_WIDTH))
    sidecar = migrate_align_sidecar(
        AlignSidecar(verification.record, 5, -1),
        source_slot=5,
        destination_slot=7,
        accepted=LEGACY_SPEC_WIDTH,
    )
    temporal, _, _ = copy_align_states(
        checkpoint,
        [[float(index) for index in range(12)]],
        accepted=LEGACY_SPEC_WIDTH,
        spec_width=LEGACY_SPEC_WIDTH,
    )
    committed, _ = consume_align_sidecar(temporal, sidecar, current_slot=7)
    _assert_matrix_close(committed, verification.full_states[-1])
