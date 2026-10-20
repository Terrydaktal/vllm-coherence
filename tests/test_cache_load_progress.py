"""Content-free progress and worker-context contracts for disk restoration."""

from concurrent.futures import ThreadPoolExecutor
from itertools import permutations
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

from qwen_r9700_lab import radiance_cache_telemetry as telemetry

CONTEXT = {"request_id": "a" * 64, "chat_id": "b" * 64, "generation": "c" * 64, "job_id": 7}


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(telemetry, "_cache_loads", {})
    monkeypatch.setattr(telemetry, "_recorder", None)
    monkeypatch.setattr(telemetry._local, "context", {}, raising=False)
    monkeypatch.setattr(telemetry._local, "cache_load", None, raising=False)


def test_parallel_workers_keep_origin_and_publish_verified_copies_only():
    ready = Barrier(3)
    release = Event()

    def load():
        with telemetry.cache_load(total_blocks=2, total_bytes=128, workers=2) as progress:
            captured = telemetry.capture_context()

            def worker():
                assert telemetry.capture_context() == captured
                with telemetry.span("decompression"):
                    ready.wait(timeout=3)
                    assert release.wait(timeout=3)
                return bytes(64)

            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(telemetry.run_with_context, worker, captured) for _ in range(2)]
                try:
                    ready.wait(timeout=3)
                    row = telemetry.cache_load_progress(CONTEXT["request_id"])
                    assert row["active_stages"]["decompression"] == 2
                    assert row["completed_blocks"] == row["completed_bytes"] == 0
                    assert row["request_id"] == CONTEXT["request_id"]
                finally:
                    release.set()
                for future in futures:
                    data = future.result(timeout=3)
                    with telemetry.span("restore_buffer_copy"):
                        assert data == bytes(64)
                    progress.block_completed(len(data))

    telemetry.task(load, CONTEXT, 0)
    row = telemetry.cache_load_progress(CONTEXT["request_id"])
    assert row["status"] == "complete"
    assert row["completed_blocks"] == 2 and row["completed_bytes"] == 128
    assert not any(row["active_stages"].values())
    assert telemetry.capture_context() == ({}, None)
    assert telemetry.cache_load_progress("d" * 64) is None


def test_stage_unions_do_not_double_count_overlapping_workers(monkeypatch):
    now = [0]
    monkeypatch.setattr(telemetry.time, "monotonic_ns", lambda: now[0])
    progress = telemetry.CacheLoadProgress(CONTEXT, total_blocks=2, total_bytes=20, workers=2)
    progress.stage_started("decompression", 10_000_000)
    progress.stage_started("decompression", 20_000_000)
    progress.stage_finished("decompression", 10_000_000, 40_000_000)
    now[0] = 50_000_000
    row = progress.snapshot()
    assert row["stage_ms"]["decompression"] == 30
    assert row["stage_wall_ms"]["decompression"] == 40
    assert row["active_stages"]["decompression"] == 1
    progress.stage_finished("decompression", 20_000_000, 60_000_000)
    now[0] = 60_000_000
    progress.finish(True)
    row = progress.snapshot()
    assert row["stage_ms"]["decompression"] == 70
    assert row["stage_wall_ms"]["decompression"] == 50
    assert row["elapsed_ms"] == 60


def test_stage_union_preserves_boundaries_received_out_of_order(monkeypatch):
    monkeypatch.setattr(telemetry.time, "monotonic_ns", lambda: 60_000_000)
    progress = telemetry.CacheLoadProgress(CONTEXT, total_blocks=2, total_bytes=20, workers=2)
    progress.stage_started("checksum", 20_000_000)
    progress.stage_started("checksum", 10_000_000)
    progress.stage_finished("checksum", 20_000_000, 60_000_000)
    progress.stage_finished("checksum", 10_000_000, 40_000_000)
    progress.finish(True)
    row = progress.snapshot()
    assert row["stage_ms"]["checksum"] == 70
    assert row["stage_wall_ms"]["checksum"] == 50


def test_disjoint_intervals_keep_the_gap_when_callbacks_are_reordered(monkeypatch):
    monkeypatch.setattr(telemetry.time, "monotonic_ns", lambda: 40_000_000)
    progress = telemetry.CacheLoadProgress(CONTEXT, total_blocks=2, total_bytes=20, workers=2)
    progress.stage_started("disk_read", 0)
    # The first worker is preempted after measuring its end at10ms; the second
    # start callback reaches the progress lock before that delayed completion.
    progress.stage_started("disk_read", 20_000_000)
    progress.stage_finished("disk_read", 0, 10_000_000)
    progress.stage_finished("disk_read", 20_000_000, 30_000_000)
    progress.finish(True)
    row = progress.snapshot()
    assert row["stage_ms"]["disk_read"] == 20
    assert row["stage_wall_ms"]["disk_read"] == 20
    assert row["active_stages"]["disk_read"] == 0


@pytest.mark.parametrize("intervals", [
    ((0, 10), (20, 30)),
    ((0, 20), (10, 30)),
    ((5, 30), (10, 20), (10, 25)),
    ((0, 10), (10, 20), (20, 30)),
    ((0, 4), (8, 12), (20, 30)),
])
def test_actual_interval_union_is_invariant_under_callback_permutations(monkeypatch, intervals):
    monkeypatch.setattr(telemetry.time, "monotonic_ns", lambda: 100_000_000)
    events = [(index, kind) for index in range(len(intervals)) for kind in ("start", "end")]
    # Independent discrete coverage oracle: fixture endpoints are whole ms.
    expected_wall = len({unit for start, end in intervals for unit in range(start, end)})
    expected_work = sum(end - start for start, end in intervals)
    checked = 0
    for ordering in permutations(events):
        if any(ordering.index((index, "start")) > ordering.index((index, "end"))
               for index in range(len(intervals))):
            continue
        progress = telemetry.CacheLoadProgress(CONTEXT, total_blocks=len(intervals),
                                               total_bytes=len(intervals), workers=len(intervals))
        for index, kind in ordering:
            start, end = (value * 1_000_000 for value in intervals[index])
            if kind == "start":
                progress.stage_started("checksum", start)
            else:
                progress.stage_finished("checksum", start, end)
        progress.finish(True)
        row = progress.snapshot()
        assert row["stage_wall_ms"]["checksum"] == expected_wall, ordering
        assert row["stage_ms"]["checksum"] == expected_work, ordering
        assert row["active_stages"]["checksum"] == 0
        checked += 1
    assert checked == (6 if len(intervals) == 2 else 90)


def test_equal_start_times_keep_separate_workers_and_live_union(monkeypatch):
    now = [0]
    monkeypatch.setattr(telemetry.time, "monotonic_ns", lambda: now[0])
    progress = telemetry.CacheLoadProgress(CONTEXT, total_blocks=2, total_bytes=2, workers=2)
    progress.stage_started("decompression", 10_000_000)
    progress.stage_started("decompression", 10_000_000)
    progress.stage_finished("decompression", 10_000_000, 20_000_000)
    now[0] = 30_000_000
    row = progress.snapshot()
    assert row["active_stages"]["decompression"] == 1
    assert row["stage_wall_ms"]["decompression"] == 20
    progress.stage_finished("decompression", 10_000_000, 25_000_000)
    progress.finish(True)
    row = progress.snapshot()
    assert row["stage_wall_ms"]["decompression"] == 15
    assert row["stage_ms"]["decompression"] == 25
    assert row["active_stages"]["decompression"] == 0


def test_interval_storage_and_active_starts_are_bounded_by_the_job(monkeypatch):
    monkeypatch.setattr(telemetry.time, "monotonic_ns", lambda: 100_000_000)
    progress = telemetry.CacheLoadProgress(CONTEXT, total_blocks=32, total_bytes=32, workers=2)
    for index in range(32):
        progress.stage_started("disk_read", index * 2_000_000)
        progress.stage_finished("disk_read", index * 2_000_000, index * 2_000_000 + 1_000_000)
    assert len(progress.completed_intervals["disk_read"]) == 32
    assert not progress.active_starts["disk_read"]
    with pytest.raises(ValueError, match="bound"):
        progress.stage_started("disk_read", 90_000_000)
    progress.stage_started("checksum", 0)
    progress.stage_started("checksum", 1)
    with pytest.raises(ValueError, match="bound"):
        progress.stage_started("checksum", 2)
    assert sum(progress.active_starts["checksum"].values()) == 2
    assert progress.active["checksum"] == 2
    progress.stage_finished("checksum", 0, 10)
    progress.stage_finished("checksum", 1, 10)
    with pytest.raises(ValueError, match="matching start"):
        progress.stage_finished("checksum", 1, 10)
    assert progress.active["checksum"] == 0


def test_snapshot_is_throttled_but_terminal_state_is_immediate(monkeypatch):
    now = [1_000_000_000]
    monkeypatch.setattr(telemetry.time, "monotonic_ns", lambda: now[0])
    progress = telemetry.CacheLoadProgress(CONTEXT, total_blocks=1, total_bytes=64, workers=1)
    before = progress.snapshot()
    progress.block_completed(64)
    now[0] += 99_000_000
    assert progress.snapshot() is before
    now[0] += 1_000_000
    assert progress.snapshot()["completed_blocks"] == 1
    progress.finish(True)
    done = progress.snapshot()
    now[0] += 10_000_000_000
    assert progress.snapshot() is done
    assert done["elapsed_ms"] == 100


def test_error_marks_failed_and_preserves_exception_and_verified_count():
    error = ValueError("synthetic integrity failure")

    def load():
        with telemetry.cache_load(total_blocks=2, total_bytes=128, workers=1) as progress:
            progress.block_completed(64)
            with telemetry.span("checksum"):
                raise error

    with pytest.raises(ValueError) as failure:
        telemetry.task(load, CONTEXT, 0)
    assert failure.value is error
    row = telemetry.cache_load_progress(CONTEXT["request_id"])
    assert row["status"] == "failed" and row["completed_blocks"] == 1
    assert not any(row["active_stages"].values())
    assert "synthetic" not in str(row)
    assert telemetry.capture_context() == ({}, None)


def test_retained_progress_is_bounded_and_owned_by_request():
    for index in range(telemetry._MAX_CACHE_LOADS + 5):
        context = {**CONTEXT, "request_id": f"{index:064x}"}
        telemetry.task(lambda: _empty_load(), context, 0)
    assert len(telemetry._cache_loads) == telemetry._MAX_CACHE_LOADS
    assert telemetry.cache_load_progress(f"{0:064x}") is None
    assert telemetry.cache_load_progress(f"{telemetry._MAX_CACHE_LOADS + 4:064x}")["status"] == "complete"


def _empty_load():
    with telemetry.cache_load(total_blocks=0, total_bytes=0, workers=1):
        pass


def test_request_phase_and_redraw_signature_follow_load_progress(monkeypatch):
    from test_radiance_fair_scheduler import BANK_A, CHAT_A, GEN_A, load_module

    module = load_module(monkeypatch)
    request = SimpleNamespace(
        request_id="progress-request", num_prompt_tokens=100, num_computed_tokens=0,
        kv_transfer_params={"qwen_chat_id": CHAT_A, "qwen_snapshot_generation": GEN_A},
    )
    # Use the exact cache identity helper's expected shape.
    monkeypatch.setattr(module, "bank_key", lambda _: BANK_A)
    phases = module.RequestPhases()
    phases.set(request, "cache_lookup")
    before = phases.row(request)
    context = {**CONTEXT, "request_id": before["request_id"], "chat_id": CHAT_A, "generation": GEN_A}

    def load():
        with telemetry.cache_load(total_blocks=1, total_bytes=64, workers=1) as progress:
            during = phases.row(request)
            assert during["cache_load_progress"]["status"] == "loading"
            assert module.FairScheduler._phase_signature([during]) != module.FairScheduler._phase_signature([before])
            progress.block_completed(64)

    telemetry.task(load, context, 0)
    assert phases.row(request)["cache_load_progress"]["status"] == "complete"
    telemetry._cache_loads[context["request_id"]].identity["generation"] = "d" * 64
    telemetry._cache_loads[context["request_id"]].cached = None
    assert "cache_load_progress" not in phases.row(request)
