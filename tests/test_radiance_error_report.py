import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import qwen_r9700_lab.radiance_error_report as probe
from qwen_r9700_lab.radiance_error_report import clean, parse_journal, report_for_window


def entry(
    body,
    *,
    container="a" * 64,
    timestamp=1000000,
    source="core.py:1355",
    role="EngineCore_DP0 pid=12",
):
    message = f"\x1b[0;36m({role})\x1b[0m ERROR 09-10 19:30:00 [{source}] {body}"
    return json.dumps(
        {
            "__REALTIME_TIMESTAMP": str(timestamp * 1000),
            "CONTAINER_ID_FULL": container,
            "MESSAGE": message,
        }
    ).encode()


def traceback(
    *, container="a" * 64, timestamp=1000000, exception="RuntimeError: synthetic failure"
):
    return [
        entry(body, container=container, timestamp=timestamp + n)
        for n, body in enumerate(
            [
                "EngineCore encountered a fatal error.",
                "Traceback (most recent call last):",
                '  File "/opt/vllm/scheduler.py", line 42, in schedule',
                "    raise RuntimeError('synthetic failure')",
                exception,
            ]
        )
    ]


def test_exact_fatal_trace_survives_without_input_dumps_or_wrapper_errors():
    lines = traceback()
    lines.insert(2, entry("PRIVATE PROMPT AND TOKEN IDS", source="core.py:999"))
    lines.append(
        entry(
            "EngineDeadError: See stack trace (above)",
            source="serving.py:15",
            role="APIServer pid=11",
        )
    )
    result = parse_journal(b"\n".join(lines))
    assert len(result) == 1
    assert result[0]["summary"] == "RuntimeError: synthetic failure"
    assert 'File "/opt/vllm/scheduler.py", line 42' in result[0]["traceback"]
    assert "PRIVATE" not in json.dumps(result)
    assert "APIServer" not in json.dumps(result)
    assert "\x1b" not in result[0]["traceback"]


def test_multiple_backends_and_chained_exceptions_remain_separate():
    first = traceback(exception="ValueError: synthetic inner error")
    second = traceback(container="b" * 64, timestamp=1000010)
    first += [
        entry(line, timestamp=1000020 + n)
        for n, line in enumerate(
            [
                "",
                "The above exception was the direct cause of the following exception:",
                "",
                "Traceback (most recent call last):",
                '  File "/opt/vllm/core.py", line 3, in run',
                "AssertionError: synthetic outer error",
            ]
        )
    ]
    results = parse_journal(b"\n".join(first[:3] + second + first[3:]))
    assert len(results) == 2
    assert results[0]["exception_type"] == "AssertionError"
    assert "synthetic inner error" in results[0]["traceback"]
    assert results[1]["exception_type"] == "RuntimeError"
    assert "synthetic inner error" not in results[1]["traceback"]


def test_incomplete_and_malformed_records_do_not_become_fake_tracebacks():
    assert parse_journal(b"\n".join([*traceback()[:-1], b'{"truncated', b"not json"])) == []
    assert parse_journal(entry("Input contains EngineCore encountered a fatal error.")) == []


def test_a_multiline_journal_record_keeps_its_traceback():
    body = (
        "EngineCore failed to start.\nTraceback (most recent call last):\n"
        + '  File "/opt/vllm/core.py", line 3, in start\nException: synthetic startup failure'
    )
    result = parse_journal(entry(body))
    assert result[0]["exception_type"] == "Exception"


def test_large_traces_retain_the_start_and_terminal_cause():
    lines = traceback()[:-1]
    lines += [entry('  File "/opt/vllm/' + "x" * 1000 + '.py", line 2, in run') for _ in range(60)]
    lines.append(entry("MemoryError: synthetic allocation failure"))
    result = parse_journal(b"\n".join(lines))[0]
    assert result["truncated"]
    assert len(result["traceback"]) < 33000
    assert result["traceback"].startswith("Traceback (")
    assert result["traceback"].endswith("MemoryError: synthetic allocation failure")


def test_old_crashes_are_not_attributed_to_a_new_request():
    incidents = parse_journal(b"\n".join(traceback()))
    collected = {
        "incidents": incidents,
        "backend": {"ready": True},
        "captured_at": 2000000,
        "lookup_issue": None,
    }
    old = report_for_window(collected, 999000, 1001000)
    assert old["status"] == "found"
    new = report_for_window(collected, 1990000, 2000000)
    assert new["status"] == "unavailable" and new["incident"] is None
    latest = report_for_window(collected, 1990000, 2000000, latest=True)
    assert latest["status"] == "found" and latest["latest"]


def test_terminal_control_sequences_are_removed():
    assert clean("before\x1b]52;c;bad\x07\x1b[31mafter\x1b[0m\x00\x9b\n") == "beforeafter\n"


def test_binary_journal_message_encoding_is_supported():
    records = []
    for line in traceback():
        row = json.loads(line)
        row["MESSAGE"] = list(row["MESSAGE"].encode())
        records.append(json.dumps(row).encode())
    assert parse_journal(b"\n".join(records))[0]["exception_type"] == "RuntimeError"


def test_simultaneous_windows_share_one_bounded_probe(monkeypatch, tmp_path):
    calls = []

    def journal(since):
        calls.append(since)
        return b"\n".join(traceback()), None

    monkeypatch.setattr(probe, "bounded_journal", journal)
    monkeypatch.setattr(probe, "backend_state", lambda: {"ready": True})
    monkeypatch.setattr(probe.time, "time", lambda: 1000)
    directory = tmp_path / "shared"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: probe.collect(1000000, directory), range(2)))
    assert len(calls) == 1
    assert results[0] == results[1]
    assert {p.name for p in directory.iterdir()} == {"lock", "report.json"}


def test_container_inspection_failure_does_not_hide_trace_or_claim_it_stopped(monkeypatch):
    monkeypatch.setattr(
        probe.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=125, stdout="")
    )

    class Healthy:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(probe.urllib.request, "urlopen", lambda *a, **kw: Healthy())
    state = probe.backend_state()
    assert state["ready"] is True and state["running"] is None
