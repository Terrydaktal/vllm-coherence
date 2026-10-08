"""Content-free writer-only memory evidence for cold-prefill reclaim stalls."""

import json
from types import SimpleNamespace

import pytest

from qwen_r9700_lab import radiance_cache_telemetry as telemetry

CGROUP = "/sys/fs/cgroup/user.slice/diagnostic-fixture"


def sources():
    return {
        "/proc/self/cgroup": "0::/user.slice/diagnostic-fixture\n",
        "/proc/meminfo": "MemTotal: 1024 kB\nMemAvailable: 128 kB\nSwapTotal: 512 kB\nSwapFree: 64 kB\nSecretChat: 999 kB\n",
        "/proc/vmstat": "pswpin 21\npswpout 34\npgmajfault 55\nworkingset_refault_anon 89\nprivate_path 777\n",
        "/proc/pressure/memory": "some avg10=0.1 avg60=0.2 avg300=0.3 total=144\nfull avg10=0.0 avg60=0.0 avg300=0.0 total=13\n",
        CGROUP + "/memory.current": "987654\n",
        CGROUP
        + "/memory.stat": "anon 610\nshmem 20\nfile 210\nworkingset_refault_anon 1597\npgmajfault 233\npswpin 377\npswpout 610\nswpin_zero 987\nsecret 123\n",
        CGROUP
        + "/memory.pressure": "some avg10=0.0 total=42\nfull avg10=0.0 total=17\n",
    }


def sampler(monkeypatch, fixture=None):
    sample = telemetry.MemoryPressureSamples()
    fixture = sources() if fixture is None else fixture
    reads = []

    def read(path):
        reads.append(str(path))
        value = fixture[str(path)]
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(sample, "_read", read)
    events = []
    recorder = SimpleNamespace(
        worker_first_work_hooks=True,
        emit=lambda stage, start, end, values: events.append((stage, values)),
    )
    return sample, recorder, reads, events


def test_memory_sample_reports_absolute_counters_units_and_pressure_without_paths(
    monkeypatch,
):
    sample, recorder, reads, events = sampler(monkeypatch)
    sample.sample(recorder, 0)
    ((stage, values),) = events
    assert stage == "host_memory_pressure"
    assert values["host_mem_available_bytes"] == 131072
    assert values["host_mem_total_bytes"] == 1048576
    assert values["host_swapin_pages"] == 21 and values["host_swapout_pages"] == 34
    assert values["cgroup_memory_current_bytes"] == 987654
    assert values["cgroup_anon_refaults"] == 1597
    assert values["cgroup_swapin_zero_pages"] == 987
    assert values["host_memory_psi_some_us"] == 144
    assert values["cgroup_memory_psi_full_us"] == 17
    assert values["diagnostic_status"] == "complete"
    assert values["memory_missing_fields"] == 0
    assert sample.health()["cgroup_discovery"] == "complete"
    assert len(reads) == 7
    assert (
        telemetry.fields(values) == values
    )  # The real output sink admits every numeric field.
    encoded = json.dumps({"event": values, "health": sample.health()})
    assert "diagnostic-fixture" not in encoded and "/proc" not in encoded
    assert (
        "SecretChat" not in encoded
        and "private_path" not in encoded
        and "secret" not in encoded
    )


def test_memory_sampling_is_worker_only_one_hz_and_discovers_cgroup_once(monkeypatch):
    sample, recorder, reads, events = sampler(monkeypatch)
    recorder.worker_first_work_hooks = False
    sample.sample(recorder, 0)
    assert not reads and not events and sample.health()["status"] == "inactive"
    recorder.worker_first_work_hooks = True
    sample.sample(recorder, 0)
    sample.sample(recorder, telemetry.MEMORY_SAMPLE_NS - 1)
    assert len(events) == 1 and len(reads) == 7
    sample.sample(recorder, telemetry.MEMORY_SAMPLE_NS)
    assert len(events) == 2 and len(reads) == 13
    assert reads.count("/proc/self/cgroup") == 1


def test_serving_emit_and_flush_never_sample_memory(tmp_path, monkeypatch):
    recorder = telemetry.Recorder(tmp_path / "status", start=False, gc_events=False)
    recorder.worker_first_work_hooks = True
    reads = []
    monkeypatch.setattr(
        recorder.memory_pressure, "_read", lambda path: reads.append(path)
    )
    recorder.emit("worker_execute", 0, 1, {"scheduled_tokens": 3296})
    recorder.flush()
    assert not reads and recorder.memory_pressure.samples == 0
    health = json.loads(recorder.health_path.read_text())
    assert health["memory_pressure"]["enabled"] is True
    assert health["memory_pressure"]["status"] == "inactive"
    recorder.close()


@pytest.mark.parametrize(
    "failure", [PermissionError("private path"), FileNotFoundError("private path")]
)
def test_source_read_failures_preserve_other_metrics_and_never_zero_fill(
    monkeypatch, failure
):
    fixture = sources()
    fixture["/proc/vmstat"] = failure
    sample, recorder, _, events = sampler(monkeypatch, fixture)
    sample.sample(recorder, 0)
    values = events[0][1]
    assert "host_swapin_pages" not in values and "host_pgmajfault" not in values
    assert values["cgroup_memory_current_bytes"] == 987654
    assert values["diagnostic_status"] == "incomplete"
    assert sample.health()["read_errors"] == 1
    assert sample.health()["failed_sources"] == ["vmstat"]
    assert "host_swapin_pages" in sample.health()["missing_fields"]
    assert "private path" not in json.dumps(sample.health())


@pytest.mark.parametrize(
    "membership,status",
    [
        ("9:memory:/legacy\n", "unsupported"),
        ("0::/../../outside\n", "failed"),
        ("0::relative/path\n", "failed"),
        ("0::/has/./segment\n", "failed"),
    ],
)
def test_unsupported_or_unsafe_cgroup_is_explicit_and_never_read(
    monkeypatch, membership, status
):
    fixture = sources()
    fixture["/proc/self/cgroup"] = membership
    sample, recorder, reads, events = sampler(monkeypatch, fixture)
    sample.sample(recorder, 0)
    assert not any(path.startswith("/sys/") for path in reads)
    assert sample.health()["cgroup_discovery"] == status
    assert "cgroup" in sample.health()["missing_fields"]
    assert "cgroup_memory_current_bytes" not in events[0][1]
    assert events[0][1]["host_mem_available_bytes"] == 131072


def test_missing_and_malformed_counters_are_omitted_not_silently_accepted(monkeypatch):
    fixture = sources()
    fixture["/proc/vmstat"] = (
        "pswpin -1\npswpout 1\npswpout 2\npgmajfault 9999999999999999999999999\n"
    )
    fixture[CGROUP + "/memory.current"] = "unknown\n"
    fixture["/proc/pressure/memory"] = (
        "some total=1\nsome total=2\nsome total=3\nfull total=-1\n"
    )
    sample, recorder, _, events = sampler(monkeypatch, fixture)
    sample.sample(recorder, 0)
    values = events[0][1]
    for key in (
        "host_swapin_pages",
        "host_swapout_pages",
        "host_pgmajfault",
        "host_anon_refaults",
        "cgroup_memory_current_bytes",
        "host_memory_psi_some_us",
    ):
        assert key not in values and key in sample.health()["missing_fields"]
    assert sample.health()["parse_errors"] >= 6
    assert values["diagnostic_status"] == "incomplete"


def test_source_reads_are_bounded_and_refuse_symlinks(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"x" * (telemetry.MAX_MEMORY_SOURCE_BYTES + 1))
    with pytest.raises(ValueError, match="bound"):
        telemetry.MemoryPressureSamples._read(source)
    source.write_bytes(b"\xff")
    with pytest.raises(UnicodeError):
        telemetry.MemoryPressureSamples._read(source)
    alias = tmp_path / "alias"
    alias.symlink_to(source)
    with pytest.raises(OSError):
        telemetry.MemoryPressureSamples._read(alias)


def test_writer_loop_runs_memory_sampler_and_reports_unexpected_failure(
    tmp_path, monkeypatch
):
    recorder = telemetry.Recorder(tmp_path / "status", start=False, gc_events=False)
    calls = []
    turns = iter([False, True])
    monkeypatch.setattr(recorder.stop, "wait", lambda seconds: next(turns))

    def fail(owner, now):
        assert owner is recorder
        calls.append(now)
        raise RuntimeError("private exception path")

    monkeypatch.setattr(recorder.memory_pressure, "sample", fail)
    recorder._writer()
    assert len(calls) == 1
    health = json.loads(recorder.health_path.read_text())["memory_pressure"]
    assert health["status"] == "failed" and health["read_errors"] == 1
    assert "private exception" not in json.dumps(health)
    recorder.close()
