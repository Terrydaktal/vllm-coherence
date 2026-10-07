"""Processed-position and lifetime regressions; native continuation is separate."""

import importlib.util
import json
import logging
import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1] / "experiments/radiance-public"
spec = importlib.util.spec_from_file_location(
    "response_end_checks", ROOT / "radiance_response_end.py"
)
cache = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cache
spec.loader.exec_module(cache)

offload_spec = importlib.util.spec_from_file_location(
    "response_offload_checks", ROOT / "radiance_response_offload.py"
)
offload = importlib.util.module_from_spec(offload_spec)
offload_spec.loader.exec_module(offload)


@pytest.fixture
def endpoint_status(monkeypatch):
    monkeypatch.setitem(sys.modules, "qwen_radiance_response_end", cache)
    base = ModuleType("vllm.v1.kv_offload.base")

    class LookupResult(Enum):
        MISS = auto()
        HIT = auto()
        HIT_PENDING = auto()
        RETRY = auto()

    base.LookupResult = LookupResult
    base.make_offload_key = lambda value, group: (value, group)
    monkeypatch.setitem(sys.modules, base.__name__, base)
    groups = [
        SimpleNamespace(
            group_idx=i,
            tokens_per_block=4,
            sliding_window_size_in_chunks=window,
            requires_cow_source=cow,
        )
        for i, (window, cow) in enumerate(((None, False), (1, True), (2, False)))
    ]
    req = request(list(range(26)), [bytes([i]) * 32 for i in range(6)])
    req.num_prompt_tokens = 26
    status = SimpleNamespace(
        req=req,
        req_context=None,
        num_locally_computed_tokens=0,
        config=SimpleNamespace(
            kv_group_configs=groups, tokens_per_hash=4, blocks_per_chunk=1
        ),
        group_states=[
            SimpleNamespace(offload_keys=[(f"prefix-{j}", i) for j in range(6)])
            for i in range(3)
        ],
    )
    end = {
        "schema": offload.SCHEMA,
        "tokens": 23,
        "hash_size": 4,
        "block_size": 4,
        "groups": 3,
        "prefix_sha256": offload.fingerprint(req, 23, 4),
    }
    return status, end, LookupResult


def test_endpoint_dependency_set_retains_full_kv_and_only_current_recurrence(
    endpoint_status,
):
    status, end, _ = endpoint_status
    keys, changing = offload.endpoint_dependencies(status, end)
    assert keys == (
        status.group_states[0].offload_keys[:5]
        + [offload.endpoint_key(end, 0)]
        + [offload.endpoint_key(end, 1)]
        + status.group_states[2].offload_keys[3:5]
        + [offload.endpoint_key(end, 2)]
    )
    assert set(changing) == set(keys) - set(status.group_states[0].offload_keys[:5])
    assert not any(key in keys for key in status.group_states[1].offload_keys)


@pytest.mark.parametrize(
    "mutation",
    [
        {"tokens": True},
        {"tokens": 0},
        {"tokens": 24},
        {"tokens": 26},
        {"block_size": 0},
        {"block_size": "4"},
        {"block_size": 8},
        {"hash_size": 2},
        {"groups": 2},
        {"schema": "old"},
        {"prefix_sha256": "bad"},
    ],
)
def test_endpoint_rejects_invalid_or_incompatible_metadata(endpoint_status, mutation):
    status, end, _ = endpoint_status
    assert offload.compatible_endpoint(status, end)
    assert not offload.compatible_endpoint(status, end | mutation)


@pytest.mark.parametrize("failure", ["missing", "pending", "changed", "none"])
def test_endpoint_lookup_requires_every_dependency_and_matching_prefix(
    endpoint_status, failure
):
    status, end, result = endpoint_status
    keys, _ = offload.endpoint_dependencies(status, end)
    seen = []

    def lookup(key, context):
        seen.append(key)
        if key == keys[-1]:
            if failure == "missing":
                return result.MISS
            if failure == "pending":
                return result.HIT_PENDING
        return result.HIT

    scheduler = SimpleNamespace(
        manager=SimpleNamespace(
            secondary_tiers=[SimpleNamespace(response_end_head=lambda ctx: end)],
            lookup=lookup,
        )
    )
    if failure == "changed":
        status.req.all_token_ids[22] = 999
    answer = offload.lookup_response_end(scheduler, status)
    if failure in {"missing", "changed"}:
        assert answer == (False, None)
        assert not hasattr(status.req, "qwen_response_end_lookup")
    elif failure == "pending":
        assert answer == (True, None)
        assert not hasattr(status.req, "qwen_response_end_lookup")
    else:
        assert answer == (True, 23)
        assert status.partial_tail_boundary == 23
    assert seen == ([] if failure == "changed" else keys)


@pytest.mark.parametrize("accepted", range(8))
@pytest.mark.parametrize("start", [1640, 1641, 1647, 1648, 3290, 10000])
def test_every_d7_accept_width_and_boundary(accepted, start):
    result = cache.end_boundary(
        start=start,
        scheduled=8,
        draft=7,
        generated=accepted + 1,
        visible=start + accepted + 2,
        block_size=1648,
    )
    assert result.tokens == start + accepted + 1
    # Independent serial-state oracle: temporal slots contain successive states
    # after the anchor and each accepted proposal. Aligned postprocessing moves
    # the boundary state into slot zero only for in-place canonicalization.
    states = list(range(start + 1, start + 9))
    aligned = result.tokens // 1648 * 1648
    source = (start + 7) // 1648
    if aligned >= start + 1 and source == aligned // 1648 - 1:
        states[0] = aligned
    assert states[result.accepted_offset] == result.tokens


def test_prefill_end_and_bonus_not_counted():
    r = cache.end_boundary(
        start=0, scheduled=1703, draft=0, generated=1, visible=1704, block_size=1648
    )
    assert (r.tokens, r.source_column, r.accepted_offset) == (1703, 1, 0)


@pytest.mark.parametrize("visible", [1702, 1703, 1705, 1708])
def test_stop_within_accepted_suffix_selects_visible_state(visible):
    r = cache.end_boundary(
        start=1700, scheduled=8, draft=7, generated=8, visible=visible, block_size=1648
    )
    assert r.tokens == visible
    assert r.accepted_offset == visible - 1701


def test_truncated_in_place_aligned_window_is_not_falsely_reused():
    assert (
        cache.end_boundary(
            start=1640, scheduled=8, draft=7, generated=8, visible=1642, block_size=1648
        )
        is None
    )


@pytest.mark.parametrize(
    "changes", [{"generated": 0}, {"generated": 9}, {"draft": 8}, {"visible": 1600}]
)
def test_invalid_or_unprocessed_end_is_rejected(changes):
    args = {
        "start": 1700,
        "scheduled": 8,
        "draft": 7,
        "generated": 8,
        "visible": 1709,
        "block_size": 1648,
    }
    assert cache.end_boundary(**(args | changes)) is None


def request(ids, hashes, salt="chat-A"):
    return SimpleNamespace(
        num_tokens=len(ids), all_token_ids=ids, block_hashes=hashes, cache_salt=salt
    )


def test_exact_tail_identity_rejects_changed_token_namespace_and_chain():
    original = request([1, 2, 3, 4, 5, 6], [b"prefix"])
    expected = cache.prefix_identity(original, 6, 4)
    assert (
        cache.prefix_identity(request([1, 2, 3, 4, 5, 6, 7], [b"prefix"]), 6, 4)
        == expected
    )
    assert (
        cache.prefix_identity(request([1, 2, 3, 4, 5, 9], [b"prefix"]), 6, 4)
        != expected
    )
    assert (
        cache.prefix_identity(request([1, 2, 3, 4, 5, 6], [b"other"]), 6, 4) != expected
    )
    assert (
        cache.prefix_identity(request([1, 2, 3, 4, 5, 6], [b"prefix"], "B"), 6, 4)
        != expected
    )
    assert cache.prefix_identity(original, 7, 4) is None


def test_aligned_identity_uses_complete_prefix():
    a = request(list(range(8)), [b"first", b"second"])
    assert cache.prefix_identity(a, 8, 4) == ("chat-A", b"second", ())


def test_patch_is_idempotent_and_rejects_changed_anchors():
    spec = importlib.util.spec_from_file_location(
        "response_end_patcher", ROOT / "patch_dflash_response_cache.py"
    )
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    assert patch.replace_once("one old two", "old", "new") == "one new two"
    assert patch.replace_once("one new two", "old", "new") == "one new two"
    with pytest.raises(ValueError):
        patch.replace_once("old old", "old", "new")


@dataclass(eq=False)
class Block:
    block_id: int
    ref_cnt: int = 1
    is_null: bool = False


def fake_cache(monkeypatch):
    specs = ModuleType("vllm.v1.kv_cache_interface")
    for name in ("FullAttentionSpec", "MambaSpec", "SlidingWindowSpec"):
        setattr(specs, name, type(name, (), {}))
    monkeypatch.setitem(sys.modules, specs.__name__, specs)

    class Pool:
        hash_block_size = 8
        null_block = Block(0, 0, True)

        def __init__(self):
            self.blocks = [self.null_block]

        def get_new_blocks(self, count):
            result = [Block(len(self.blocks) + i) for i in range(count)]
            self.blocks.extend(result)
            return result

        def get_num_free_blocks(self):
            return 100

        def touch(self, blocks):
            for b in blocks:
                b.ref_cnt += 1

        def free_blocks(self, blocks):
            for b in blocks:
                assert b.ref_cnt > 0
                b.ref_cnt -= 1

    pool = Pool()
    managers = [
        SimpleNamespace(
            block_size=8,
            kv_cache_spec=getattr(specs, name)(),
            mamba_cache_mode="align",
            req_to_blocks={"old": pool.get_new_blocks(10)},
        )
        for name in ("FullAttentionSpec", "MambaSpec", "SlidingWindowSpec")
    ]
    manager = SimpleNamespace(
        block_pool=pool,
        coordinator=SimpleNamespace(
            scheduler_block_size=8, single_type_managers=managers
        ),
        prefix_cache_lookup_enabled=lambda r: True,
        cache_blocks=lambda *args: None,
        create_kv_cache_blocks=lambda blocks: blocks,
    )
    old = request(list(range(15)), [b"first"])
    old.request_id = "old"
    old.num_in_flight_tokens = 0
    old.mm_features = []
    old.lora_request = None
    old.num_output_tokens = 5
    old.num_computed_tokens = 14
    old.status = SimpleNamespace(name="FINISHED_LENGTH_CAPPED")
    step = {"start": 10, "scheduled": 8, "draft": 7, "generated": 4}
    return cache.ResponseEndCache(manager), old, step, pool, managers


def decision_records(caplog):
    return [
        json.loads(record.getMessage().removeprefix("Response cache decision: "))
        for record in caplog.records
        if record.name == "vllm.qwen_response_end"
    ]


@pytest.mark.parametrize(
    "reason",
    [
        "no_endpoint",
        "prompt_does_not_extend_endpoint",
        "multimodal_request",
        "prompt_embeddings",
        "lora_request",
        "prefix_cache_disabled",
        "prefix_identity_unavailable",
        "cache_salt_changed",
        "prefix_hash_changed",
        "partial_prefix_changed",
        "matching_endpoint",
    ],
)
def test_every_local_lookup_outcome_is_logged_without_private_data(
    monkeypatch, caplog, reason
):
    c, old, step, _, _ = fake_cache(monkeypatch)
    assert c.remember(old, step)
    newer = SimpleNamespace(**vars(old))
    newer.request_id = "PRIVATE-REQUEST-ID"
    newer.all_token_ids = [*old.all_token_ids, 99]
    newer.num_tokens += 1
    newer.num_prompt_tokens = newer.num_tokens
    if reason == "no_endpoint":
        c.clear()
    elif reason == "prompt_does_not_extend_endpoint":
        newer.num_tokens = c.entry["tokens"]
    elif reason == "multimodal_request":
        newer.mm_features = ["PRIVATE-MULTIMODAL"]
    elif reason == "prompt_embeddings":
        newer.prompt_embeds = "PRIVATE-EMBEDDINGS"
    elif reason == "lora_request":
        newer.lora_request = "PRIVATE-LORA"
    elif reason == "prefix_cache_disabled":
        c.manager.prefix_cache_lookup_enabled = lambda _: False
    elif reason == "prefix_identity_unavailable":
        newer.block_hashes = []
    elif reason == "cache_salt_changed":
        newer.cache_salt = "PRIVATE-CACHE-SALT"
    elif reason == "prefix_hash_changed":
        newer.block_hashes = [b"PRIVATE-PREFIX-HASH"]
    elif reason == "partial_prefix_changed":
        newer.all_token_ids[12] = 123456789
    with caplog.at_level(logging.INFO, logger="vllm.qwen_response_end"):
        hit = c.lookup(newer)
    assert (hit is not None) == (reason == "matching_endpoint")
    (row,) = decision_records(caplog)
    assert row["reason"] == reason
    assert row["outcome"] == ("hit" if hit else "rejected")
    assert row["cache_source"] == "gpu_endpoint"
    assert row["endpoint_tokens"] == (0 if reason == "no_endpoint" else 14)
    assert row["hash_size"] == 8
    assert len(row["request_id"]) == 64
    assert "PRIVATE-" not in caplog.text
    assert "123456789" not in caplog.text
    assert set(row) <= {
        "cache_source",
        "outcome",
        "reason",
        "input_tokens",
        "computed_tokens",
        "endpoint_tokens",
        "hash_size",
        "request_id",
    }


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (None, "no_endpoint"),
        ("PRIVATE-METADATA", "invalid_endpoint_metadata"),
        ({"schema": "PRIVATE-SCHEMA"}, "endpoint_schema_mismatch"),
        ({"tokens": True}, "invalid_endpoint_token_count"),
        ({"tokens": 0}, "endpoint_not_ahead"),
        ({"tokens": 26}, "prompt_does_not_extend_endpoint"),
        ({"tokens": 24}, "aligned_endpoint"),
        ({"block_size": 0}, "invalid_endpoint_block_size"),
        ({"hash_size": 2}, "hash_block_size_mismatch"),
        ({"groups": 2}, "cache_group_count_mismatch"),
        ({"prefix_sha256": "PRIVATE-FINGERPRINT"}, "prefix_identity_changed"),
    ],
)
def test_every_durable_metadata_rejection_is_logged_before_reading_blocks(
    endpoint_status,
    caplog,
    mutation,
    reason,
):
    status, end, _ = endpoint_status
    endpoint = (end | mutation) if isinstance(mutation, dict) else mutation
    status.req.request_id = "PRIVATE-REQUEST-ID"
    scheduler = SimpleNamespace(
        manager=SimpleNamespace(
            secondary_tiers=[SimpleNamespace(response_end_head=lambda _: endpoint)],
            lookup=lambda *_: pytest.fail(
                "rejected metadata must not issue block lookups"
            ),
        )
    )
    with caplog.at_level(logging.INFO, logger="vllm.qwen_response_end"):
        assert offload.lookup_response_end(scheduler, status) == (False, None)
    (row,) = decision_records(caplog)
    assert row["reason"] == reason
    assert row["cache_source"] == "offload_endpoint"
    assert row["outcome"] == "rejected"
    assert row["tier_index"] == 0
    assert "PRIVATE-" not in caplog.text


@pytest.mark.parametrize(
    "reason",
    [
        "unsupported_blocks_per_chunk",
        "cache_group_block_size_mismatch",
        "prefix_identity_unavailable",
        "no_endpoint_tier",
    ],
)
def test_durable_configuration_and_unavailable_prefix_rejections(
    endpoint_status, caplog, reason
):
    status, end, _ = endpoint_status
    tiers = [SimpleNamespace(response_end_head=lambda _: end)]
    if reason == "unsupported_blocks_per_chunk":
        status.config.blocks_per_chunk = 2
    elif reason == "cache_group_block_size_mismatch":
        status.config.kv_group_configs[-1].tokens_per_block = 8
    elif reason == "prefix_identity_unavailable":
        status.req.block_hashes = []
    else:
        tiers = [SimpleNamespace()]
    scheduler = SimpleNamespace(manager=SimpleNamespace(secondary_tiers=tiers))
    with caplog.at_level(logging.INFO, logger="vllm.qwen_response_end"):
        assert offload.lookup_response_end(scheduler, status) == (False, None)
    (row,) = decision_records(caplog)
    assert row["reason"] == reason


def test_dependency_polling_records_counts_and_changes_without_repeated_log_spam(
    endpoint_status, caplog
):
    status, end, result = endpoint_status
    keys, _ = offload.endpoint_dependencies(status, end)
    found = {key: result.HIT for key in keys}
    found[keys[-1]] = result.HIT_PENDING
    found[keys[0]] = result.MISS
    scheduler = SimpleNamespace(
        manager=SimpleNamespace(
            secondary_tiers=[SimpleNamespace(response_end_head=lambda _: end)],
            lookup=lambda key, _: found[key],
        )
    )
    with caplog.at_level(logging.INFO, logger="vllm.qwen_response_end"):
        for _ in range(3):
            assert offload.lookup_response_end(scheduler, status) == (False, None)
        found[keys[0]] = result.HIT
        for _ in range(3):
            assert offload.lookup_response_end(scheduler, status) == (True, None)
        found[keys[-1]] = result.HIT
        assert offload.lookup_response_end(scheduler, status) == (True, 23)
    rows = decision_records(caplog)
    assert [r["reason"] for r in rows] == [
        "missing_dependencies",
        "pending_dependencies",
        "matching_endpoint",
    ]
    assert [r["outcome"] for r in rows] == ["rejected", "loading", "hit"]
    assert [r["missing_dependencies"] for r in rows] == [1, 0, 0]
    assert [r["pending_dependencies"] for r in rows] == [1, 1, 0]
    assert all(r["dependency_count"] == len(keys) for r in rows)
    assert rows[-1]["cached_tokens"] == 23


@pytest.mark.parametrize("mode", ["disabled", "full", "failed"])
def test_rejection_is_in_backend_log_even_without_optional_recording(
    monkeypatch, tmp_path, caplog, mode
):
    from qwen_r9700_lab import radiance_cache_telemetry as telemetry

    recorder = telemetry.Recorder(
        tmp_path / "optional", capacity=1, start=False, gc_events=False
    )
    monkeypatch.setattr(
        telemetry, "_recorder", None if mode == "disabled" else recorder
    )
    if mode == "full":
        recorder.emit("already_full", 0, 0)
    elif mode == "failed":

        def fail(*_, **__):
            raise OSError("PRIVATE-WRITER-FAILURE")

        monkeypatch.setattr(telemetry, "emit", fail)
    c, old, _, _, _ = fake_cache(monkeypatch)
    with caplog.at_level(logging.INFO, logger="vllm.qwen_response_end"):
        assert c.lookup(old) is None
    (row,) = decision_records(caplog)
    assert row["reason"] == "no_endpoint"
    assert "PRIVATE-WRITER-FAILURE" not in caplog.text
    if mode == "full":
        assert recorder.dropped == 1


def test_structured_lookup_record_reaches_writer_with_whitelisted_fields(
    monkeypatch, tmp_path
):
    from qwen_r9700_lab import radiance_cache_telemetry as telemetry

    recorder = telemetry.Recorder(tmp_path / "lookup", start=False, gc_events=False)
    monkeypatch.setattr(telemetry, "_recorder", recorder)
    req = SimpleNamespace(
        request_id="PRIVATE-REQUEST-ID",
        cache_salt=f"private:{'a' * 64}:{'b' * 64}",
        num_prompt_tokens=197565,
        num_computed_tokens=0,
    )
    cache.record_response_end_decision(
        req,
        "gpu_endpoint",
        "rejected",
        "prefix_hash_changed",
        endpoint_tokens=197486,
        hash_size=16,
        raw_token_values="PRIVATE-TOKENS",
        prefix_sha256="PRIVATE-HASH",
    )
    recorder.flush()
    (row,) = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert row["stage"] == "response_end_lookup"
    assert row["reason"] == "prefix_hash_changed"
    assert row["endpoint_tokens"] == 197486
    assert row["hash_size"] == 16
    assert row["input_tokens"] == 197565
    assert row["chat_id"] == "a" * 64 and row["generation"] == "b" * 64
    assert "PRIVATE-" not in recorder.path.read_text()


def test_pins_survive_producer_free_and_are_released_once(monkeypatch):
    c, old, step, pool, groups = fake_cache(monkeypatch)
    assert c.remember(old, step)
    copies = c.take_copies()
    assert len(copies) == 1 and copies[0]["offset"] == 3
    assert copies[0]["conv"] == groups[1].req_to_blocks["old"][2].block_id
    assert copies[0]["state"] == groups[1].req_to_blocks["old"][5].block_id
    for group in groups:
        pool.free_blocks(group.req_to_blocks["old"])
    assert all(b.ref_cnt >= 1 for b in c.entry["pins"] + c.copy_pins)
    assert (
        next(
            b for b in c.entry["pins"] if b.block_id == copies[0]["destination"]
        ).ref_cnt
        == 2
    )
    c.copies_complete()
    c.copies_complete()  # duplicate completion does not double free
    assert all(b.ref_cnt == 1 for b in c.entry["pins"])
    newer = SimpleNamespace(**vars(old))
    newer.all_token_ids = [*old.all_token_ids, 99]
    newer.num_tokens += 1
    hit = c.lookup(newer)
    assert hit[1] == 14
    assert hit[0][1][0].is_null
    assert hit[0][1][-1].block_id == copies[0]["destination"]
    c.clear()
    c.clear()
    assert all(b.ref_cnt == 0 for b in pool.blocks if not b.is_null)


def test_reset_before_copy_cannot_recycle_its_destination(monkeypatch):
    c, old, step, pool, groups = fake_cache(monkeypatch)
    assert c.remember(old, step)
    copies = c.take_copies()
    for group in groups:
        pool.free_blocks(group.req_to_blocks["old"])
    c.clear()
    destination = next(b for b in pool.blocks if b.block_id == copies[0]["destination"])
    assert destination.ref_cnt == 1
    assert all(b.ref_cnt >= 1 for b in c.copy_pins)
    c.copies_complete()
    assert all(b.ref_cnt == 0 for b in pool.blocks if not b.is_null)


def test_successor_releases_obsolete_pins_only_after_progress(monkeypatch):
    c, old, step, pool, groups = fake_cache(monkeypatch)
    assert c.remember(old, step)
    entry = c.entry
    c.take_copies()
    c.copies_complete()
    for group in groups:
        pool.free_blocks(group.req_to_blocks["old"])
    newer = SimpleNamespace(**vars(old))
    newer.request_id = "new"
    newer.all_token_ids = [*old.all_token_ids, 99]
    newer.num_tokens += 1
    assert c.lookup(newer)[1] == 14
    # Admission failure or cancelled work cannot surrender the only checkpoint.
    newer.num_computed_tokens = 0
    assert not c.release_after_progress(newer)
    newer.num_computed_tokens = 14
    assert not c.release_after_progress(newer)
    # Model native allocator ownership and a pending reader independently.
    owned = list(entry["pins"])
    pool.touch(owned)
    reader = owned[-1]
    pool.touch([reader])
    newer.num_computed_tokens = 16
    newer.num_in_flight_tokens = 2
    assert not c.release_after_progress(newer)
    assert c.entry is entry
    newer.num_in_flight_tokens = 0
    assert c.release_after_progress(newer)
    assert c.entry is None
    assert not c.release_after_progress(newer)
    assert all(block.ref_cnt >= 1 for block in owned)
    pool.free_blocks(owned)
    assert reader.ref_cnt == 1
    pool.free_blocks([reader])
    assert all(block.ref_cnt == 0 for block in pool.blocks if not block.is_null)


def test_terminal_successor_does_not_release_its_new_checkpoint(monkeypatch):
    c, old, step, _, _ = fake_cache(monkeypatch)
    assert c.remember(old, step)
    newer = SimpleNamespace(**vars(old))
    newer.num_tokens += 1
    newer.all_token_ids = [*old.all_token_ids, 99]
    assert c.lookup(newer)
    assert c.remember(newer, step)
    replacement = c.entry
    newer.num_computed_tokens = 16
    assert not c.release_after_progress(newer)
    assert c.entry is replacement


@pytest.mark.parametrize(
    "change", ["different_prefix", "shorter_prompt", "same_length"]
)
def test_unused_endpoint_releases_only_its_pins_after_lookup_miss(monkeypatch, change):
    c, old, step, pool, groups = fake_cache(monkeypatch)
    assert c.remember(old, step)
    pins = list(c.entry["pins"])
    c.take_copies()
    c.copies_complete()
    for group in groups:
        pool.free_blocks(group.req_to_blocks.pop("old"))
    # A normal prefix hit or another owner may still share one of these pages.
    pool.touch([pins[0]])
    newer = SimpleNamespace(**vars(old))
    newer.request_id = "new"
    newer.num_computed_tokens = 0
    if change == "different_prefix":
        newer.block_hashes = [b"changed-prefix"]
    else:
        newer.num_tokens = 13 if change == "shorter_prompt" else 14
    assert c.lookup(newer) is None
    assert not c.release_after_progress(newer)
    assert c.release_unused(newer)
    assert c.entry is None
    assert pins[0].ref_cnt == 1
    assert all(b.ref_cnt == 0 for b in pins[1:])
    assert not c.release_unused(newer)


@pytest.mark.parametrize(
    "guard",
    [
        "no_lookup",
        "matching_lease",
        "other_lease",
        "local_hit",
        "computed",
        "in_flight",
        "queued_copy",
        "in_flight_copy",
        "transfer",
    ],
)
def test_unused_endpoint_preserves_live_ownership_and_pending_work(monkeypatch, guard):
    c, old, step, pool, _ = fake_cache(monkeypatch)
    assert c.remember(old, step)
    entry = c.entry
    c.take_copies()
    c.copies_complete()
    newer = SimpleNamespace(**vars(old))
    newer.num_computed_tokens = 0
    newer._qwen_response_end_lease = None
    protected = set()
    if guard == "no_lookup":
        del newer._qwen_response_end_lease
    elif guard in {"matching_lease", "other_lease"}:
        newer._qwen_response_end_lease = (
            entry if guard == "matching_lease" else dict(entry)
        )
    elif guard == "local_hit":
        newer.qwen_response_end_local = 14
    elif guard == "computed":
        newer.num_computed_tokens = 1
    elif guard == "in_flight":
        newer.num_in_flight_tokens = 1
    elif guard == "queued_copy":
        c.pending.append({"group": 1})
    elif guard == "in_flight_copy":
        c.copy_pins.append(entry["pins"][0])
    elif guard == "transfer":
        protected.add(entry["pins"][0].block_id)
    refs = [b.ref_cnt for b in pool.blocks]
    assert not c.release_unused(newer, protected)
    assert c.entry is entry
    assert [b.ref_cnt for b in pool.blocks] == refs


@pytest.mark.parametrize("computed, first_required", [(48, 5), (49, 6)])
def test_pressure_reclaim_preserves_current_speculative_and_transfer_state(
    monkeypatch, computed, first_required
):
    c, req, _, pool, groups = fake_cache(monkeypatch)
    monkeypatch.setattr(
        pool,
        "get_num_free_blocks",
        lambda: sum(b.ref_cnt == 0 and not b.is_null for b in pool.blocks),
    )
    req.num_computed_tokens = computed
    before = [list(group.req_to_blocks["old"]) for group in groups]
    protected = before[1][3]
    req.num_in_flight_tokens = 8
    assert cache.reclaim_snapshot_history(c.manager, req, [protected.block_id]) == 0
    req.num_in_flight_tokens = 0
    assert (
        cache.reclaim_snapshot_history(c.manager, req, [protected.block_id])
        == first_required - 1
    )
    assert groups[0].req_to_blocks["old"] == before[0]
    assert groups[2].req_to_blocks["old"] == before[2]
    after = groups[1].req_to_blocks["old"]
    assert after[first_required:] == before[1][first_required:]
    assert after[3] is protected and protected.ref_cnt == 1
    assert all(b.is_null for i, b in enumerate(after[:first_required]) if i != 3)
    # A completed transfer releases the one formerly protected old snapshot.
    assert cache.reclaim_snapshot_history(c.manager, req) == 1
    assert cache.reclaim_snapshot_history(c.manager, req) == 0


@pytest.mark.parametrize("offset", range(8))
@pytest.mark.parametrize("dim_first", [False, True])
def test_canonical_copy_preserves_bits_and_does_not_write_other_blocks(
    monkeypatch, offset, dim_first
):
    torch = pytest.importorskip("torch", reason="requires the cpu-tests environment")
    module = ModuleType("vllm.model_executor.layers.mamba.mamba_utils")
    module.is_conv_state_dim_first = lambda: dim_first
    monkeypatch.setitem(sys.modules, module.__name__, module)
    gen = torch.Generator().manual_seed(831)
    shape = (10, 3, 10) if dim_first else (10, 10, 3)
    # Arbitrary bit patterns include NaNs, infinities and signed zeros; exact
    # byte comparison is necessary, rather than approximate tensor equality.
    context, expected = {}, {}
    for layer in ("first", "second"):
        conv = torch.randint(
            -32768, 32767, shape, generator=gen, dtype=torch.int16
        ).view(torch.bfloat16)
        state = torch.randint(
            -(2**31), 2**31 - 1, (10, 3, 4), generator=gen, dtype=torch.int32
        ).view(torch.float32)
        context[layer] = SimpleNamespace(kv_cache=(conv, state))
        ec, es = conv.view(torch.int16).clone(), state.view(torch.int32).clone()
        ec[9].zero_()
        # Serial logical-history oracle, independent of tensor slicing used
        # by the implementation.
        for channel in range(3):
            for position in range(10 - offset):
                if dim_first:
                    ec[9, channel, position] = conv.view(torch.int16)[
                        1, channel, position + offset
                    ]
                else:
                    ec[9, position, channel] = conv.view(torch.int16)[
                        1, position + offset, channel
                    ]
        es[9] = state.view(torch.int32)[1 + offset]
        expected[layer] = (ec, es)
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context=context)
        ),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(layer_names=list(context))]
        ),
    )
    cache.copy_response_end(
        runner,
        [
            {
                "group": 0,
                "conv": 1,
                "state": 1 + offset,
                "destination": 9,
                "offset": offset,
            }
        ],
    )
    for layer, value in context.items():
        conv, state = value.kv_cache
        assert torch.equal(conv.view(torch.int16), expected[layer][0])
        assert torch.equal(state.view(torch.int32), expected[layer][1])


def test_cancelled_request_does_not_replace_complete_end(monkeypatch):
    c, old, step, _, _ = fake_cache(monkeypatch)
    assert c.remember(old, step)
    first = c.entry
    old.status.name = "FINISHED_ABORTED"
    assert not c.remember(old, step)
    assert c.entry is first


def test_changed_tail_or_too_short_prompt_cannot_acquire_state(monkeypatch):
    c, old, step, _, _ = fake_cache(monkeypatch)
    assert c.remember(old, step)
    old.all_token_ids[13] = 100
    assert c.lookup(old) is None
    old.all_token_ids[13] = 13
    old.num_tokens = 14
    assert c.lookup(old) is None


def test_serving_package_contains_endpoint_runtime_and_excludes_native_probe(
    monkeypatch,
):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "tools"))
    from package_runtime import PATCHES

    assert {
        "patch_dflash_response_cache.py",
        "radiance_response_end.py",
        "radiance_response_offload.py",
    } <= set(PATCHES)
    assert "response_end_probe.py" not in PATCHES
    assert "qualify_response_end_state.py" not in PATCHES
