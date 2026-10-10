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
    *,
    container="a" * 64,
    timestamp=1000000,
    exception="RuntimeError: synthetic failure",
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
    assert (
        parse_journal(b"\n".join([*traceback()[:-1], b'{"truncated', b"not json"]))
        == []
    )
    assert (
        parse_journal(entry("Input contains EngineCore encountered a fatal error."))
        == []
    )


def test_a_multiline_journal_record_keeps_its_traceback():
    body = (
        "EngineCore failed to start.\nTraceback (most recent call last):\n"
        + '  File "/opt/vllm/core.py", line 3, in start\nException: synthetic startup failure'
    )
    result = parse_journal(entry(body))
    assert result[0]["exception_type"] == "Exception"


def test_large_traces_retain_the_start_and_terminal_cause():
    lines = traceback()[:-1]
    lines += [
        entry('  File "/opt/vllm/' + "x" * 1000 + '.py", line 2, in run')
        for _ in range(60)
    ]
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
    assert (
        clean("before\x1b]52;c;bad\x07\x1b[31mafter\x1b[0m\x00\x9b\n")
        == "beforeafter\n"
    )


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
    monkeypatch.setattr(probe, "host_state", lambda _: {})
    monkeypatch.setattr(probe.time, "time", lambda: 1000)
    directory = tmp_path / "shared"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: probe.collect(1000000, directory), range(2)))
    assert len(calls) == 1
    assert results[0] == results[1]
    assert {p.name for p in directory.iterdir()} == {"lock", "report.json"}


def test_container_inspection_failure_does_not_hide_trace_or_claim_it_stopped(
    monkeypatch,
):
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=125, stdout=""),
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


def test_request_error_trace_is_collected_without_an_enginecore_fatal_banner():
    lines = traceback(exception="OSError: [Errno 28] No space left on device")[1:]
    lines.append(entry("PRIVATE request dump from the same logger"))
    result = parse_journal(b"\n".join(lines))
    assert len(result) == 1
    assert result[0]["exception_type"] == "OSError"
    assert result[0]["summary"] == "OSError: [Errno 28] No space left on device"
    assert "PRIVATE" not in json.dumps(result)


def test_uvicorn_plain_asgi_trace_keeps_frames_without_vllm_logger_tags():
    bodies = [
        "ERROR: Exception in ASGI application",
        "Traceback (most recent call last):",
        '  File "/opt/vllm/server.py", line 31, in request',
        "    write_checkpoint()",
        "OSError: [Errno 28] No space left on device",
        "PRIVATE adjacent prompt dump",
    ]
    raw = b"\n".join(
        json.dumps(
            {
                "MESSAGE": f"(APIServer pid=1) {body}",
                "CONTAINER_ID_FULL": "a" * 64,
                "__REALTIME_TIMESTAMP": str(1000000 * 1000 + n),
            }
        ).encode()
        for n, body in enumerate(bodies)
    )
    result = parse_journal(raw)
    assert len(result) == 1
    assert "server.py" in result[0]["traceback"]
    assert result[0]["exception_type"] == "OSError"
    assert "PRIVATE" not in json.dumps(result)


def test_boot_failure_is_numeric_and_does_not_retain_other_kernel_text():
    messages = [
        "[Hardware Error]: System Fatal error.",
        "[Hardware Error]: CPU:9 (19:21:2) MC5_STATUS[-|UE|PCC]: 0xbea0000000000108",
        "[Hardware Error]: Execution Unit Ext. Error Code: 0",
        "PRIVATE unrelated journal message",
    ]
    raw = b"\n".join(
        json.dumps(
            {
                "__REALTIME_TIMESTAMP": str(1010000 * 1000 + n),
                "_BOOT_ID": "b" * 32,
                "MESSAGE": message,
            }
        ).encode()
        for n, message in enumerate(messages)
    )
    result = probe.parse_kernel_journal(b"\n".join(reversed(raw.splitlines())))
    assert result[0]["kind"] == "cpu_machine_check"
    assert result[0]["cpu"] == 9
    assert result[0]["fatal"] is True
    assert result[0]["unit"] == "execution"
    assert result[0]["extended_code"] == 0
    assert "PRIVATE" not in json.dumps(result)


def collected_failure(**extra):
    return {
        "incidents": [],
        "backend": {"running": False, "ready": False},
        "captured_at": 1100000,
        "lookup_issue": None,
        **extra,
    }


def test_reboot_during_request_is_explained_without_a_python_traceback():
    collected = collected_failure(
        host={
            "boot_id": "b" * 32,
            "boot_started_at": 1010000,
            "kernel_events": [
                {
                    "timestamp": 1010020,
                    "kind": "cpu_machine_check",
                    "cpu": 9,
                    "fatal": True,
                    "unit": "execution",
                    "extended_code": 0,
                }
            ],
        }
    )
    result = report_for_window(collected, 1000000, 1020000)
    assert result["diagnosis"]["kind"] == "host_restarted"
    assert "CPU 9" in result["diagnosis"]["summary"]
    assert "watchdog" in result["diagnosis"]["summary"]
    assert result["incident"] is None
    assert "restart Pi" in result["diagnosis"]["recovery"]


def test_prior_boot_failure_is_not_blamed_for_a_later_connection_error():
    collected = collected_failure(
        host={
            "boot_started_at": 900000,
            "kernel_events": [
                {
                    "timestamp": 900020,
                    "kind": "cpu_machine_check",
                    "fatal": True,
                    "cpu": 9,
                }
            ],
        }
    )
    result = report_for_window(collected, 1000000, 1020000)
    assert result["diagnosis"]["kind"] == "backend_stopped"
    assert "CPU" not in result["diagnosis"]["summary"]


def test_disk_full_and_oom_have_useful_recovery_instead_of_generic_engine_error():
    incidents = parse_journal(
        b"\n".join(traceback(exception="OSError: [Errno 28] No space left on device"))
    )
    result = report_for_window(collected_failure(incidents=incidents), 999000, 1001000)
    assert result["diagnosis"]["kind"] == "disk_full"
    assert "free space" in result["diagnosis"]["recovery"]
    result = report_for_window(
        collected_failure(backend={"oom_killed": True}), 999000, 1001000
    )
    assert result["diagnosis"]["kind"] == "host_oom"


def test_healthy_backend_is_current_status_and_not_an_explanation_of_the_past():
    result = report_for_window(
        collected_failure(backend={"running": True, "ready": True}), 999000, 1001000
    )
    assert result["diagnosis"]["kind"] == "cause_unknown"
    assert "ready" in result["diagnosis"]["summary"]


def test_manual_healthy_lookup_does_not_invent_an_interrupted_request():
    result = report_for_window(
        collected_failure(backend={"running": True, "ready": True}),
        999000,
        1001000,
        latest=True,
    )
    assert result["incident"] is None
    assert result["diagnosis"] == {
        "kind": "backend_ready",
        "summary": "Backend is ready. No recorded backend error was found.",
        "recovery": "",
    }


def test_manual_healthy_lookup_keeps_missing_history_visible():
    result = report_for_window(
        collected_failure(
            backend={"running": True, "ready": True},
            lookup_issue="journal_timeout",
        ),
        999000,
        1001000,
        latest=True,
    )
    assert result["diagnosis"]["kind"] == "history_unavailable"
    assert "could not be fully checked" in result["diagnosis"]["summary"]
    assert "No recorded backend error" not in result["diagnosis"]["summary"]
    assert "diagnostic lookup" in result["diagnosis"]["recovery"]


def test_manual_lookup_keeps_a_real_historical_failure_when_backend_is_ready():
    result = report_for_window(
        collected_failure(
            backend={"running": True, "ready": True},
            incidents=parse_journal(b"\n".join(traceback())),
        ),
        1090000,
        1100000,
        latest=True,
    )
    assert result["incident"] is not None
    assert result["diagnosis"]["kind"] == "backend_exception"
    assert "synthetic failure" in result["diagnosis"]["summary"]
