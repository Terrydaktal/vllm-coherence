"""CPU-only generated-token journal checks; every token is synthetic."""

import dataclasses
import fcntl
import os
import stat
from array import array

import pytest

from qwen_r9700_lab import radiance_token_continuation as continuation

HISTORY = "1" * 64
OUTPUT = "2" * 64
RAW_PREFIX = [10, 11]
OUTPUT_IDS = [20, 21, 99]
CANONICAL = [10, 11, 80, 99]
CURRENT = CANONICAL + [12, 13]


def identity(**changes):
    values = {
        "cache_salt": "synthetic:chat:generation",
        "model": "synthetic-model",
        "tokenizer": "synthetic-tokenizer",
        "template": "synthetic-template",
        "configuration": "synthetic-cache-generation-and-abi",
    }
    return continuation.ContinuationIdentity(**(values | changes))


def record(ledger, *, who=None, lease=None, **changes):
    who = who or identity()
    lease = lease or ledger.begin(who)
    values = {
        "history_identity": HISTORY,
        "output_identity": OUTPUT,
        "message_count": 3,
        "processed_tokens": None,
        "success": True,
    }
    values.update(changes)
    return ledger.record_completed(lease, RAW_PREFIX, OUTPUT_IDS, **values)


def propose(ledger, *, who=None, **changes):
    values = {
        "canonical_previous_ids": CANONICAL,
        "history_identity": HISTORY,
        "output_identity": OUTPUT,
        "boundary_token_ids": {99},
        "history_unchanged": True,
        "output_unchanged": True,
        "configuration_unchanged": True,
    }
    values.update(changes)
    current = values.pop("current_ids", CURRENT)
    return ledger.propose(who or identity(), current, **values)


def append_suffix(ledger, *, who=None, **changes):
    who = who or identity()
    previous = ledger.get_record(who)
    values = {
        "history_identity": HISTORY,
        "output_identity": OUTPUT,
        "expected_version": previous.version if previous else None,
        "boundary_token_ids": {99},
        "lease": changes.get("lease") if "lease" in changes else ledger.begin(who),
    }
    values.update(changes)
    suffix = values.pop("suffix_ids", [12, 13])
    return ledger.append_verified_suffix(who, suffix, **values)


@pytest.mark.parametrize("durable", [False, True])
def test_verified_suffix_keeps_raw_prefix_without_canonical_reencoding(
    tmp_path, durable
):
    ledger = continuation.TokenContinuationLedger(
        tmp_path / "private" if durable else None
    )
    assert record(ledger, processed_tokens=4)
    previous = ledger.get_record(identity())
    lease = ledger.begin(identity())
    proposal = append_suffix(ledger, lease=lease)
    assert list(proposal.tokens) == RAW_PREFIX + OUTPUT_IDS + [12, 13]
    assert proposal.processed_tokens == 4
    assert proposal.version == previous.version
    assert ledger.get_record(identity()) == previous
    ledger.abort(lease)


@pytest.mark.parametrize(
    "tokens", [[], [True], [-1], [2**31], [1.5], "tokens", object()]
)
def test_verified_suffix_rejects_empty_or_unsupported_cpu_tokens(tokens):
    ledger = continuation.TokenContinuationLedger()
    assert record(ledger)
    result = append_suffix(ledger, suffix_ids=tokens)
    assert not result.applied
    assert result.reason in {"prompt_not_extended", "unsupported_tokens"}


@pytest.mark.parametrize("name", ["history_identity", "output_identity"])
def test_verified_suffix_rejects_changed_message_hash(name):
    ledger = continuation.TokenContinuationLedger()
    assert record(ledger)
    assert (
        append_suffix(ledger, **{name: "3" * 64}).reason == "history_or_output_changed"
    )


@pytest.mark.parametrize("boundary", [set(), {98}, {True}, {-1}, set(range(65)), "99"])
def test_verified_suffix_requires_stored_authenticated_boundary(boundary):
    ledger = continuation.TokenContinuationLedger()
    assert record(ledger)
    assert (
        append_suffix(ledger, boundary_token_ids=boundary).reason
        == "unsupported_boundary"
    )


@pytest.mark.parametrize("version", [None, "", "x" * 32, "0" * 32, 1])
def test_verified_suffix_rejects_wrong_expected_version(version):
    ledger = continuation.TokenContinuationLedger()
    assert record(ledger)
    assert append_suffix(ledger, expected_version=version).reason == "journal_changed"


def test_verified_suffix_rejects_missing_history_boundary_and_changed_construction():
    ledger = continuation.TokenContinuationLedger()
    assert record(ledger, message_count=None)
    assert append_suffix(ledger).reason == "history_boundary_missing"
    ledger = continuation.TokenContinuationLedger()
    assert record(ledger)
    assert (
        append_suffix(ledger, who=identity(configuration="changed")).reason
        == "identity_changed"
    )


def test_verified_suffix_rejects_cancelled_or_overlapping_lease():
    ledger = continuation.TokenContinuationLedger()
    assert record(ledger)
    first = ledger.begin(identity())
    second = ledger.begin(identity())
    assert append_suffix(ledger, lease=first).reason == "overlapping_or_stale_request"
    assert append_suffix(ledger, lease=second).reason == "overlapping_or_stale_request"
    ledger.abort(first)
    ledger.abort(second)
    assert append_suffix(ledger, lease=first).reason == "overlapping_or_stale_request"
    assert append_suffix(ledger, lease=None).reason == "overlapping_or_stale_request"


@pytest.mark.parametrize("kwargs", [{"max_tokens": 6}, {"max_bytes": 24}])
def test_verified_suffix_enforces_total_raw_history_budget(kwargs):
    ledger = continuation.TokenContinuationLedger(**kwargs)
    assert record(ledger)
    assert append_suffix(ledger).reason == "token_budget"


def test_verified_suffix_revalidates_other_writer_and_corrupted_disk(tmp_path):
    root = tmp_path / "private"
    ledger = continuation.TokenContinuationLedger(root)
    assert record(ledger)
    previous = ledger.get_record(identity())
    lease = ledger.begin(identity())
    other = continuation.TokenContinuationLedger(root)
    assert record(other)
    assert (
        append_suffix(ledger, lease=lease, expected_version=previous.version).reason
        == "journal_changed"
    )
    ledger.abort(lease)
    lease = ledger.begin(identity())
    path = root / (identity().key + ".ctok")
    path.write_bytes(b"corrupt")
    assert not append_suffix(ledger, lease=lease).applied
    assert ledger.health["read_errors"] > 0


class SyntheticTokenizer:
    """20+21 and80 decode alike, but the encoder canonically emits80."""

    def decode(self, ids):
        replacement = list(ids)
        for index in range(len(replacement) - 1):
            if replacement[index : index + 2] == [20, 21]:
                replacement[index : index + 2] = [80]
                break
        return tuple(replacement)

    def encode(self, text):
        return list(text)


def test_alternate_segmentation_preserves_original_tokens_without_text_only_reuse():
    ledger = continuation.TokenContinuationLedger()
    assert record(ledger)
    tokenizer = SyntheticTokenizer()
    old = ledger.get_record(identity())
    canonical = tokenizer.encode(tokenizer.decode(old.tokens))
    assert canonical == CANONICAL
    new_input = CANONICAL + [12, 13]
    result = propose(ledger, canonical_previous_ids=canonical, current_ids=new_input)
    assert result.applied
    assert list(result.tokens) == RAW_PREFIX + OUTPUT_IDS + [12, 13]
    assert new_input == CURRENT
    assert result.processed_tokens is None
    assert result.contract == "generated_tokens_v1"
    # This is intentionally a different numerical input from canonical text.
    assert list(result.tokens) != new_input
    assert list(old.tokens) == RAW_PREFIX + OUTPUT_IDS


def test_known_pending_last_token_does_not_become_falsely_processed():
    ledger = continuation.TokenContinuationLedger()
    assert record(ledger, processed_tokens=4)
    result = propose(ledger)
    assert list(result.tokens) == [10, 11, 20, 21, 99, 12, 13]
    assert result.processed_tokens == 4
    assert ledger.get_record(identity()).processed_tokens == 4


@pytest.mark.parametrize(
    "name",
    [
        "history_unchanged",
        "output_unchanged",
        "configuration_unchanged",
    ],
)
@pytest.mark.parametrize("value", [False, None, 1, "true"])
def test_strict_construction_checks_cannot_be_guessed(name, value):
    ledger = continuation.TokenContinuationLedger()
    assert record(ledger)
    assert propose(ledger, **{name: value}).reason == "construction_changed"


@pytest.mark.parametrize("name", ["history_identity", "output_identity"])
def test_history_and_delivered_output_identity_are_required(name):
    ledger = continuation.TokenContinuationLedger()
    assert record(ledger)
    assert propose(ledger, **{name: "3" * 64}).reason == "history_or_output_changed"


@pytest.mark.parametrize(
    "name", ["model", "tokenizer", "template", "configuration", "cache_salt"]
)
def test_changed_generation_or_numerical_construction_falls_back(name):
    ledger = continuation.TokenContinuationLedger()
    assert record(ledger)
    result = propose(ledger, who=identity(**{name: "different-synthetic-identity"}))
    assert not result.applied
    assert result.reason in {"identity_changed", "no_record"}


@pytest.mark.parametrize("durable", [False, True])
def test_changed_construction_can_bank_a_fresh_completed_request(tmp_path, durable):
    ledger = continuation.TokenContinuationLedger(
        tmp_path / "private" if durable else None
    )
    assert record(ledger)
    previous = identity()
    changed = identity(configuration="new-synthetic-abi")
    assert not propose(ledger, who=changed).applied
    lease = ledger.begin(changed)
    assert ledger.get_record(changed) is None
    assert record(ledger, who=changed, lease=lease)
    assert propose(ledger, who=changed).applied
    assert not propose(ledger, who=previous).applied


def test_whitespace_or_other_canonical_token_change_is_not_hidden():
    ledger = continuation.TokenContinuationLedger()
    assert record(ledger)
    changed = [10, 11, 81, 99, 12, 13]
    result = propose(ledger, current_ids=changed)
    assert result.reason == "canonical_prefix_changed"
    assert changed == [10, 11, 81, 99, 12, 13]


@pytest.mark.parametrize(
    "boundaries", [set(), {98}, {True}, {2**31}, range(100), set(range(65))]
)
def test_unrecognized_or_unsupported_boundary_falls_back(boundaries):
    ledger = continuation.TokenContinuationLedger()
    assert record(ledger)
    assert (
        propose(ledger, boundary_token_ids=boundaries).reason == "unsupported_boundary"
    )


def test_both_original_and_canonical_end_must_be_same_special_token():
    ledger = continuation.TokenContinuationLedger()
    assert record(ledger)
    result = propose(
        ledger,
        canonical_previous_ids=[10, 11, 80, 98],
        current_ids=[10, 11, 80, 98, 12],
        boundary_token_ids={98, 99},
    )
    assert result.reason == "unsupported_boundary"


@pytest.mark.parametrize("current", [[], CANONICAL, CANONICAL[:-1]])
def test_request_must_extend_previous_complete_canonical_prefix(current):
    ledger = continuation.TokenContinuationLedger()
    assert record(ledger)
    assert propose(ledger, current_ids=current).reason == "prompt_not_extended"


def test_unknown_message_boundary_is_not_treated_as_valid_history():
    ledger = continuation.TokenContinuationLedger()
    assert record(ledger, message_count=None)
    assert propose(ledger).reason == "history_boundary_missing"


def test_own_lease_can_read_previous_record_and_propose():
    ledger = continuation.TokenContinuationLedger()
    assert record(ledger)
    lease = ledger.begin(identity())
    assert ledger.get_record(identity()).message_count == 3
    assert propose(ledger, lease=lease).applied
    assert not propose(ledger).applied
    ledger.abort(lease)
    assert ledger.health["pending"] == 0


def test_overlap_invalidates_both_writers_and_preserves_prior_journal():
    ledger = continuation.TokenContinuationLedger()
    assert record(ledger)
    version = ledger.get_record(identity()).version
    first = ledger.begin(identity())
    second = ledger.begin(identity())
    assert not propose(ledger, lease=first).applied
    assert not propose(ledger, lease=second).applied
    assert not record(ledger, lease=first)
    assert not record(ledger, lease=second)
    assert ledger.get_record(identity()).version == version
    assert ledger.health["pending"] == 0
    assert record(ledger)


def test_cancellation_and_failure_never_publish_partial_output():
    ledger = continuation.TokenContinuationLedger()
    assert not record(ledger, success=False)
    assert ledger.get_record(identity()) is None
    assert ledger.health["pending"] == 0
    lease = ledger.begin(identity())
    ledger.abort(lease)
    assert not record(ledger, lease=lease)
    assert ledger.get_record(identity()) is None


def test_pending_and_payload_budgets_remain_bounded():
    ledger = continuation.TokenContinuationLedger(
        max_chats=2, max_pending=2, max_bytes=40
    )
    for ordinal in range(4):
        assert record(ledger, who=identity(cache_salt=f"synthetic-{ordinal}"))
    assert ledger.health["records"] == 2
    assert ledger.health["retained_bytes"] <= 40
    assert ledger.begin(identity(cache_salt="pending-one")) is not None
    assert ledger.begin(identity(cache_salt="pending-two")) is not None
    assert ledger.begin(identity(cache_salt="pending-three")) is None
    assert ledger.health["pending"] == 2


def test_expanded_native_prefix_must_fit_context_not_just_canonical_input():
    ledger = continuation.TokenContinuationLedger(max_tokens=8)
    assert record(ledger)
    result = propose(ledger, current_ids=CANONICAL + [1, 2, 3, 4])
    assert result.reason == "token_budget"
    assert not result.applied


@pytest.mark.parametrize("tokens", [[True], [-1], [2**31], [1.5], "tokens", object()])
def test_invalid_cpu_tokens_are_rejected_without_array_or_device_protocol(tokens):
    ledger = continuation.TokenContinuationLedger()
    lease = ledger.begin(identity())
    assert not ledger.record_completed(
        lease,
        tokens,
        OUTPUT_IDS,
        history_identity=HISTORY,
        output_identity=OUTPUT,
        message_count=3,
    )
    assert ledger.health["pending"] == 0


def test_tensor_like_object_is_never_materialized():
    class Forbidden:
        def __iter__(self):
            raise AssertionError("device iteration")

        def __array__(self):
            raise AssertionError("device conversion")

    ledger = continuation.TokenContinuationLedger()
    assert record(ledger)
    assert propose(ledger, current_ids=Forbidden()).reason == "unsupported_tokens"


@pytest.mark.parametrize("processed", [-1, 6, True, "4"])
def test_invalid_processed_counts_cannot_be_claimed(processed):
    ledger = continuation.TokenContinuationLedger()
    assert not record(ledger, processed_tokens=processed)


@pytest.mark.parametrize("count", [-1, 1025, True, "3"])
def test_invalid_message_counts_cannot_be_claimed(count):
    ledger = continuation.TokenContinuationLedger()
    assert not record(ledger, message_count=count)


def test_packed_tokens_are_immutable_and_repr_hides_values():
    packed = continuation._pack(array("i", [10, 20, 99]), 10)
    assert list(packed[::-1]) == [99, 20, 10]
    assert list(packed[1:]) == [20, 99]
    assert packed[-1] == 99
    with pytest.raises(IndexError):
        _ = packed[3]
    with pytest.raises(dataclasses.FrozenInstanceError):
        packed._data = b""
    assert repr(packed) == "PackedTokenIds(count=3)"
    assert "synthetic" not in repr(identity())


def test_durable_cold_restore_retains_unknown_processed_count_and_message_boundary(
    tmp_path,
):
    root = tmp_path / "private"
    first = continuation.TokenContinuationLedger(root)
    assert record(first)
    second = continuation.TokenContinuationLedger(root)
    journal = second.get_record(identity())
    assert list(journal.tokens) == RAW_PREFIX + OUTPUT_IDS
    assert journal.message_count == 3
    assert journal.processed_tokens is None
    assert propose(second).applied
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    for path in root.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert len(list(root.glob("*.ctok"))) == 1


def test_competing_instances_use_durable_version_cas(tmp_path):
    root = tmp_path / "private"
    first = continuation.TokenContinuationLedger(root)
    second = continuation.TokenContinuationLedger(root)
    a = first.begin(identity())
    b = second.begin(identity())
    assert record(first, lease=a)
    assert not record(second, lease=b)
    assert second.health["last_reason"] == "journal_changed"
    assert propose(second).applied


def test_proposal_refuses_version_changed_after_lease_begin(tmp_path):
    root = tmp_path / "private"
    first = continuation.TokenContinuationLedger(root)
    assert record(first)
    lease = first.begin(identity())
    second = continuation.TokenContinuationLedger(root)
    assert record(second)
    assert propose(first, lease=lease).reason == "journal_changed"
    first.abort(lease)


def test_durable_retention_keeps_only_latest_bounded_chats(tmp_path):
    root = tmp_path / "private"
    ledger = continuation.TokenContinuationLedger(root, max_chats=2)
    for ordinal in range(4):
        assert record(ledger, who=identity(cache_salt=f"synthetic-{ordinal}"))
    assert len(list(root.glob("*.ctok"))) == 2
    restored = continuation.TokenContinuationLedger(root, max_chats=2)
    assert restored.get_record(identity(cache_salt="synthetic-0")) is None
    assert restored.get_record(identity(cache_salt="synthetic-3")) is not None


@pytest.mark.parametrize(
    "kind", ["corrupt", "truncated", "oversized", "public", "symlink", "hardlink"]
)
def test_unsafe_or_corrupt_durable_record_never_falls_back_to_stale_memory(
    tmp_path, kind
):
    root = tmp_path / "private"
    ledger = continuation.TokenContinuationLedger(root)
    assert record(ledger)
    path = root / (identity().key + ".ctok")
    if kind == "corrupt":
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
    elif kind == "truncated":
        path.write_bytes(b"partial")
    elif kind == "oversized":
        with path.open("wb") as stream:
            stream.truncate(continuation.MAX_TOKENS * 4 + continuation.MAX_HEADER + 100)
    elif kind == "public":
        path.chmod(0o644)
    elif kind == "symlink":
        moved = tmp_path / "outside"
        path.rename(moved)
        path.symlink_to(moved)
    else:
        os.link(path, tmp_path / "outside")
    assert ledger.get_record(identity()) is None
    assert not propose(ledger).applied
    assert ledger.health["read_errors"] > 0


def test_symlinked_or_public_storage_directory_fails_closed(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    assert continuation.TokenContinuationLedger(linked).begin(identity()) is None
    assert list(outside.iterdir()) == []
    outside.chmod(0o755)
    assert continuation.TokenContinuationLedger(outside).begin(identity()) is None


def test_contended_file_lock_has_no_blocking_wait(tmp_path):
    root = tmp_path / "private"
    ledger = continuation.TokenContinuationLedger(root)
    assert record(ledger)
    fd = os.open(root / ".lock", os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert ledger.get_record(identity()) is None
        assert ledger.begin(identity()) is None
    finally:
        os.close(fd)


def test_file_fsync_failure_does_not_publish_or_leave_temporary_payload(
    tmp_path, monkeypatch
):
    root = tmp_path / "private"
    ledger = continuation.TokenContinuationLedger(root)
    assert record(ledger)
    before = (root / (identity().key + ".ctok")).read_bytes()

    def fail(_fd):
        raise OSError("synthetic fsync failure; never emitted")

    monkeypatch.setattr(continuation.os, "fsync", fail)
    assert not record(ledger)
    assert (root / (identity().key + ".ctok")).read_bytes() == before
    assert not list(root.glob(".tmp-*"))
    assert ledger.health["pending"] == 0
    assert "synthetic" not in str(ledger.health)


def test_atomic_write_retries_short_writes_and_fsyncs_file_and_directory(
    tmp_path, monkeypatch
):
    root = tmp_path / "private"
    ledger = continuation.TokenContinuationLedger(root)
    original_write, original_fsync = os.write, os.fsync
    sync_kinds = []

    def short(fd, data):
        return original_write(fd, data[:17])

    def sync(fd):
        sync_kinds.append("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        return original_fsync(fd)

    monkeypatch.setattr(continuation.os, "write", short)
    monkeypatch.setattr(continuation.os, "fsync", sync)
    assert record(ledger)
    assert sync_kinds[:2] == ["file", "directory"]
    assert propose(continuation.TokenContinuationLedger(root)).applied


def test_private_root_environment_configuration(tmp_path, monkeypatch):
    root = tmp_path / "private"
    monkeypatch.setenv("QWEN_TOKEN_CONTINUATION_ROOT", str(root))
    ledger = continuation.TokenContinuationLedger()
    assert ledger.root == root
    assert record(ledger)


def test_no_record_and_bad_bounds_are_explicit():
    ledger = continuation.TokenContinuationLedger()
    assert propose(ledger).reason == "no_record"
    for kwargs in (
        {"max_chats": 9},
        {"max_pending": 9},
        {"max_tokens": 300001},
        {"max_bytes": continuation.MAX_BYTES + 1},
    ):
        with pytest.raises(ValueError):
            continuation.TokenContinuationLedger(**kwargs)
