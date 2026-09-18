"""CPU qualification of campaign wiring; these tests confer no GPU qualification."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_r9700_lab.conformance_campaign import (
    FAMILIES,
    build_campaign,
    case_result,
    coverage,
    load_results,
    run_campaign,
    validate_campaign,
)
from qwen_r9700_lab.conformance_cli import main
from qwen_r9700_lab.conformance_faults import FAULTS, install_experiment, mutate_device
from qwen_r9700_lab.conformance_instrumentation import HookSet
from qwen_r9700_lab.conformance_observer import Events, read_events, wrap
from qwen_r9700_lab.conformance_runtime import MODULES, NativeServer, isolated_config, server_argv
from qwen_r9700_lab.conformance_scenarios import (
    SCENARIOS,
    same_output,
    synthetic_text,
    validate_operator_report,
)
from qwen_r9700_lab.conformance_transport import (
    Completion,
    OwnedClient,
    OwnedProcess,
    ProtocolError,
    SSEDecoder,
)
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, private_json, seal, write_private


@pytest.fixture
def spec():
    binding = seal(
        {"schema": "urn:qwen:radiance-native-binding:v1", "files": {}, "live_data_abi": "1" * 64}
    )
    return {
        "python": sys.executable,
        "python_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
        "binding": binding,
        "observer_sources": dict.fromkeys(MODULES, "2" * 64),
        "environment": {},
        "native_config": {"model": "/not-a-real-model"},
        "server_config": {
            "model": "/not-a-real-model",
            "host": "0.0.0.0",
            "port": 8012,
            "speculative_config": {"method": "dflash", "num_speculative_tokens": 7},
            "kv_transfer_config": {
                "kv_connector": "OffloadingConnector",
                "kv_connector_extra_config": {
                    "secondary_tiers": [
                        {"type": "qwen_chat_fs", "root_dir": "/never-write-production"}
                    ]
                },
            },
        },
        "checkpoint_files": {},
        "kv_scales": {},
        "reference_profile": "weight-only-bf16",
        "operator_profile": {},
        "probe_manifest": {},
        "contexts": [128],
        "seeds": [17],
        "max_context": 2048,
        "output_tokens": 32,
    }


def test_complete_declared_families_have_executable_drivers(spec):
    campaign = build_campaign(spec)
    assert set(SCENARIOS) == set(FAMILIES) == {c["family"] for c in campaign["cases"]}
    assert all(callable(SCENARIOS[c["family"]]) for c in campaign["cases"])
    cases = campaign["cases"]
    assert {c["variant"] for c in cases if c["family"] == "native_fault"} == set(FAULTS)
    assert {c["axes"]["accepted"] for c in cases if c["family"] == "forced_d7"} == set(range(8))
    assert {c["axes"]["accepted"] for c in cases if c["family"] == "rejected_suffix"} == set(
        range(7)
    )
    assert {"clean_restart", "crash_restart", "interrupted_write", "compaction", "eviction"} <= {
        c["variant"] for c in cases if c["family"] == "lifecycle"
    }
    assert {c["context"] for c in cases if c["family"] == "forced_d7"} >= {
        63,
        64,
        65,
        127,
        128,
        129,
        1647,
        1648,
        1649,
    }


@pytest.mark.parametrize("failure", [None, "construct", "connect"])
@pytest.mark.parametrize("crash", [False, True])
@pytest.mark.parametrize("capacity", [None, 12000])
def test_server_releases_its_offload_name_after_exit_or_failed_start(
    spec, tmp_path, monkeypatch, failure, crash, capacity
):
    from qwen_r9700_lab import conformance_runtime as runtime
    from qwen_r9700_lab.conformance_shm import OwnedOffloadRegion

    shm = tmp_path / "shm"
    shm.mkdir()
    other = shm / "vllm_offload_unrelated.mmap"
    other.write_bytes(b"keep")
    regions, closed = [], []

    def claim(engine_id):
        region = OwnedOffloadRegion(engine_id, directory=shm)
        regions.append(region)
        return region

    class Process:
        def __init__(self, *_args, **_kwargs):
            regions[-1].path.write_bytes(b"orphaned offload data")
            if failure == "construct":
                raise RuntimeError("injected launcher failure")

        def close(self, *, crash=False, grace_seconds=10):
            assert regions[-1].path.exists()
            assert grace_seconds > 60
            closed.append(crash)

    class Client:
        def __init__(self, *_args):
            pass

        def connect(self, *, timeout):
            if failure == "connect":
                raise RuntimeError("injected startup failure")

    spec.update(startup_timeout_seconds=2, case_timeout_seconds=3)
    extra = spec["server_config"]["kv_transfer_config"]["kv_connector_extra_config"]
    extra["cpu_bytes_to_use"] = 24000
    monkeypatch.setattr(runtime, "OwnedOffloadRegion", claim)
    monkeypatch.setattr(runtime, "OwnedProcess", Process)
    monkeypatch.setattr(runtime, "OwnedClient", Client)
    monkeypatch.setattr(runtime, "worker_environment", lambda *_: {"PYTHONPATH": ""})
    server = NativeServer(spec, tmp_path / "server", allow_gpu=True, primary_cache_bytes=capacity)
    if failure:
        with pytest.raises(RuntimeError, match="injected"):
            server.start()
    else:
        server.start()
        server.stop(crash=crash)
        server.start()
        server.stop(crash=crash)
        assert regions[0].name != regions[1].name
    assert all(not region.path.exists() for region in regions)
    assert all(region.fd is None for region in regions)
    assert other.read_bytes() == b"keep"
    receipts = list(server.root.glob("shared-memory-*.json"))
    assert len(receipts) == len(regions)
    assert all(private_json(path)["status"] == "unlinked" for path in receipts)
    settings = private_json(server.root / "server-0.json")
    configured = settings["config"]["kv_transfer_config"]["kv_connector_extra_config"]
    assert configured["cpu_bytes_to_use"] == (24000 if capacity is None else capacity)
    assert settings["variant"]["primary_cache_bytes"] == capacity
    assert extra["cpu_bytes_to_use"] == 24000
    assert len(closed) == (0 if failure == "construct" else len(regions))


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5, "10", 24000, 48000])
def test_invalid_primary_cache_override_cannot_start_a_server(spec, tmp_path, capacity):
    extra = spec["server_config"]["kv_transfer_config"]["kv_connector_extra_config"]
    extra["cpu_bytes_to_use"] = 24000
    server = NativeServer(spec, tmp_path / "server", allow_gpu=True, primary_cache_bytes=capacity)
    with pytest.raises(DiagnosticError, match="primary-cache"):
        server.start()
    assert not (server.root / "server-0.json").exists()
    assert extra["cpu_bytes_to_use"] == 24000


def test_failed_process_close_preserves_offload_mapping(spec, tmp_path):
    from qwen_r9700_lab.conformance_shm import OwnedOffloadRegion

    server = NativeServer(spec, tmp_path / "server", allow_gpu=True)
    region = OwnedOffloadRegion("conformance-" + "a" * 64, directory=tmp_path)
    region.path.write_bytes(b"still owned")
    server.offload_region = region
    server.offload_receipt = server.root / "shared-memory-0.json"

    class Process:
        def close(self, *, crash=False, grace_seconds=10):
            raise RuntimeError("process has not been stopped")

    server.process = Process()
    try:
        with pytest.raises(RuntimeError, match="not been stopped"):
            server.stop()
        assert region.path.read_bytes() == b"still owned"
        assert not server.offload_receipt.exists()
    finally:
        region.release()


def test_full_context_defaults_do_not_silently_omit_maximum(spec):
    spec.pop("contexts")
    spec["max_context"] = 253792
    spec["output_tokens"] = 256
    campaign = build_campaign(spec)
    assert campaign["spec"]["contexts"] == [8192, 32768, 60000, 128000, 200000, 253535]


def test_priority_stages_keep_all_cases_and_start_with_actual_corruption_checks(spec):
    campaign = build_campaign(spec)
    stages = [c["stage"] for c in campaign["cases"]]
    assert stages == sorted(stages, key=("pilot", "focused", "extended").index)
    assert campaign["cases"][0]["variant"] == "gdn"
    assert len([c for c in campaign["cases"] if c["stage"] == "pilot"]) == 8


def test_checkpoint_resume_skips_passed_cases_and_honours_pause(spec, tmp_path, monkeypatch):
    from qwen_r9700_lab import conformance_campaign as runner
    from qwen_r9700_lab.conformance_queue import request_pause

    campaign = build_campaign(spec)
    root = tmp_path / "queue"
    selected = [c["id"] for c in campaign["cases"][:2]]
    calls = []

    def execute(campaign, case, _root):
        calls.append(case["id"])
        if len(calls) == 1:
            request_pause(root)
        return case_result(
            campaign, case, "TESTED", started=time.monotonic(), checks=["CPU fixture"]
        )

    monkeypatch.setattr(runner, "execute_case", execute)
    report = run_campaign(campaign, root, allow_gpu=True, selected=selected)
    assert report["counts"]["TESTED"] == 1
    assert private_json(root / "checkpoint.json")["status"] == "paused"
    report = run_campaign(campaign, root, allow_gpu=True, selected=selected, resume=True)
    assert calls == selected and report["counts"]["TESTED"] == 2
    run_campaign(campaign, root, allow_gpu=True, selected=selected, resume=True)
    assert calls == selected
    assert len(list(root.glob("checkpoint-case-*.json"))) == 2


def test_resumed_failed_attempt_needs_explicit_retry_and_never_disappears(
    spec, tmp_path, monkeypatch
):
    from qwen_r9700_lab import conformance_campaign as runner

    campaign = build_campaign(spec)
    root = tmp_path / "queue"
    selected = [campaign["cases"][0]["id"]]
    calls = []

    def execute(campaign, case, _root):
        calls.append(case["id"])
        return case_result(
            campaign,
            case,
            "FAILED" if len(calls) == 1 else "TESTED",
            started=time.monotonic(),
            checks=["CPU fixture"],
        )

    monkeypatch.setattr(runner, "execute_case", execute)
    run_campaign(campaign, root, allow_gpu=True, selected=selected)
    run_campaign(campaign, root, allow_gpu=True, selected=selected, resume=True)
    assert len(calls) == 1
    assert private_json(root / "checkpoint.json")["status"] == "failed_attempt_requires_review"
    report = run_campaign(
        campaign, root, allow_gpu=True, selected=selected, resume=True, retry_failed=True
    )
    assert len(calls) == 2 and report["counts"]["FAILED"] == 1
    assert report["cases"][0]["flaky"] and not report["complete"]
    assert len(list(root.glob("case-*/result.json"))) == 2
    run_campaign(campaign, root, allow_gpu=True, selected=selected, resume=True)
    assert len(calls) == 2  # the successful retry is the latest attempt


def test_interrupted_attempt_becomes_durable_error_before_resuming(spec, tmp_path, monkeypatch):
    campaign = build_campaign(spec)
    root = tmp_path / "queue"
    (root / "case-00000").mkdir(parents=True)
    write_private(root / "campaign.json", campaign)
    write_private(
        root / "case-00000/input.json", {"campaign": campaign, "case": campaign["cases"][0]}
    )
    report = run_campaign(campaign, root, allow_gpu=True, resume=True)
    assert report["counts"]["ERROR"] == 1
    assert private_json(root / "case-00000/result.json")["error_type"] == "InterruptedAttempt"


def test_low_disk_checkpoints_before_launching_gpu_case(spec, tmp_path, monkeypatch):
    from qwen_r9700_lab import conformance_queue as queue

    monkeypatch.setattr(queue.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    campaign = build_campaign(spec)
    root = tmp_path / "queue"
    report = run_campaign(campaign, root, allow_gpu=True)
    assert report["counts"] == {"NOT_RUN": len(campaign["cases"])}
    assert private_json(root / "checkpoint.json")["status"] == "insufficient_space"
    assert not list(root.glob("case-*"))


def test_active_controller_lock_refuses_second_controller(spec, tmp_path, monkeypatch):
    from qwen_r9700_lab import conformance_campaign as runner

    campaign = build_campaign(spec)
    root = tmp_path / "queue"

    def execute(campaign, case, _root):
        with pytest.raises(DiagnosticError, match="active controller"):
            run_campaign(campaign, root, allow_gpu=True, resume=True)
        return case_result(
            campaign, case, "TESTED", started=time.monotonic(), checks=["CPU fixture"]
        )

    monkeypatch.setattr(runner, "execute_case", execute)
    run_campaign(campaign, root, allow_gpu=True, selected=[campaign["cases"][0]["id"]])


def test_time_budget_finishes_current_case_then_checkpoints_remaining_work(
    spec, tmp_path, monkeypatch
):
    from qwen_r9700_lab import conformance_campaign as runner
    from qwen_r9700_lab import conformance_queue as queue

    campaign = build_campaign(spec)
    root = tmp_path / "queue"
    clock = [0.0]
    calls = []
    monkeypatch.setattr(queue.time, "monotonic", lambda: clock[0])

    def execute(campaign, case, _root):
        calls.append(case["id"])
        clock[0] += 5
        return case_result(campaign, case, "TESTED", started=clock[0], checks=["CPU fixture"])

    monkeypatch.setattr(runner, "execute_case", execute)
    run_campaign(campaign, root, allow_gpu=True, budget_seconds=3)
    assert len(calls) == 1
    assert private_json(root / "checkpoint.json")["status"] == "time_budget_reached"
    assert private_json(root / "case-00000/result.json")["status"] == "TESTED"


def test_cleanup_failure_stops_queue_even_with_keep_going(spec, tmp_path, monkeypatch):
    from qwen_r9700_lab import conformance_campaign as runner
    from qwen_r9700_lab.conformance_gpu_lease import block_cleanup

    monkeypatch.setenv("QWEN_CONFORMANCE_GPU_LOCK", str(tmp_path / "lock"))
    campaign = build_campaign(spec)
    calls = []

    def execute(campaign, case, root):
        calls.append(case["id"])
        block_cleanup(root, process_group=17, reason="synthetic surviving worker", members=[])
        return case_result(
            campaign, case, "FAILED", started=time.monotonic(), detail="cleanup incomplete"
        )

    monkeypatch.setattr(runner, "execute_case", execute)
    root = tmp_path / "queue"
    result = run_campaign(campaign, root, allow_gpu=True, keep_going=True)
    assert len(calls) == 1
    assert result["counts"]["FAILED"] == 1
    assert private_json(root / "checkpoint.json")["status"] == "gpu_cleanup_incomplete"


def test_group_exit_check_distinguishes_live_workers_from_zombies(tmp_path, monkeypatch):
    from qwen_r9700_lab import conformance_transport as transport

    groups = {101: 50, 102: 50, 103: 50, 104: 70, 105: 50}
    monkeypatch.setattr(os, "getpgid", lambda pid: groups[pid])
    for pid, state in [(101, "D"), (102, "Z"), (103, "S"), (104, "R"), (105, "Z")]:
        path = tmp_path / str(pid)
        path.mkdir()
        fields = [state, "1", str(groups[pid]), *(["0"] * 16), "12345"]
        (path / "stat").write_text(f"{pid} (fixture ) with spaces) " + " ".join(fields))
        (path / "task").mkdir()
        if pid == 105:
            task = path / "task/106"
            task.mkdir()
            fields[0] = "D"
            (task / "stat").write_text("106 (surviving thread) " + " ".join(fields))
    assert transport.live_group_members(50, proc_root=tmp_path) == [
        {"pid": 101, "state": "D", "start_ticks": 12345},
        {"pid": 103, "state": "S", "start_ticks": 12345},
        {"pid": 105, "tid": 106, "state": "D", "start_ticks": 12345},
    ]


def test_gpu_refused_before_mutations_or_imports(spec, tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN_CONFORMANCE_GPU", "0")
    before = set(sys.modules)
    with pytest.raises(DiagnosticError, match="not authorized"):
        run_campaign(build_campaign(spec), tmp_path / "campaign")
    with pytest.raises(DiagnosticError, match="not authorized"):
        NativeServer(spec, tmp_path / "server")
    with pytest.raises(DiagnosticError, match="not armed"):
        mutate_device(None, None, "kv")
    assert not list(tmp_path.iterdir())
    assert not {"torch", "vllm", "r4d"} & (set(sys.modules) - before)


def test_empty_explicit_selection_never_turns_into_a_full_gpu_run(spec, tmp_path):
    with pytest.raises(DiagnosticError, match="empty selection"):
        run_campaign(build_campaign(spec), tmp_path / "must-not-exist", allow_gpu=True, selected=[])
    assert not list(tmp_path.iterdir())


def test_plan_and_status_cli_are_cpu_only_and_missing_cases_fail(spec, tmp_path, capsys):
    write_private(tmp_path / "spec.json", spec)
    assert (
        main(
            [
                "suite-plan",
                "--spec",
                str(tmp_path / "spec.json"),
                "--output",
                str(tmp_path / "plan.json"),
            ]
        )
        == 0
    )
    write_private(tmp_path / "campaign.json", private_json(tmp_path / "plan.json"))
    args = [
        "suite-status",
        "--campaign",
        str(tmp_path / "plan.json"),
        "--results",
        str(tmp_path),
        "--require-passing",
    ]
    assert main(args) == 1
    assert '"NOT_RUN"' in capsys.readouterr().out


def test_dropped_case_resealed_cannot_evade_inventory_gate(spec):
    campaign = build_campaign(spec)
    campaign["cases"].pop()
    with pytest.raises(DiagnosticError, match="inventory changed"):
        validate_campaign(seal({k: v for k, v in campaign.items() if k != "sha256"}))


def test_changed_source_cannot_reuse_campaign(monkeypatch, spec):
    from qwen_r9700_lab import conformance_campaign as module

    campaign = build_campaign(spec)
    monkeypatch.setattr(module, "source_identity", lambda: {"changed.py": "3" * 64})
    with pytest.raises(DiagnosticError, match="source identity"):
        validate_campaign(campaign)


def test_retry_does_not_erase_first_failure_or_missing_cases(spec):
    campaign = build_campaign(spec)
    case = campaign["cases"][0]
    fail = case_result(
        campaign, case, "FAILED", started=time.monotonic(), error_type="NumericalMismatch"
    )
    good = case_result(
        campaign, case, "TESTED", started=time.monotonic(), checks=["actual_comparison"]
    )
    report = coverage(campaign, {case["id"]: [fail, good]})
    assert not report["complete"]
    assert report["cases"][0]["status"] == "FAILED"
    assert report["cases"][0]["flaky"]
    assert report["counts"]["NOT_RUN"] == len(campaign["cases"]) - 1


@pytest.mark.parametrize("status", ["FAILED", "ERROR", "TIMEOUT", "UNSUPPORTED"])
def test_every_nonpassing_status_blocks_gate(spec, status):
    campaign = build_campaign(spec)
    result = {
        c["id"]: [case_result(campaign, c, "TESTED", started=time.monotonic(), checks=["oracle"])]
        for c in campaign["cases"]
    }
    assert coverage(campaign, result)["complete"]
    case = campaign["cases"][-1]
    result[case["id"]] = [case_result(campaign, case, status, started=time.monotonic())]
    assert not coverage(campaign, result)["complete"]


def test_empty_or_foreign_success_evidence_rejected(spec):
    campaign = build_campaign(spec)
    case = campaign["cases"][0]
    bad = case_result(campaign, case, "TESTED", started=time.monotonic())
    with pytest.raises(DiagnosticError, match="oracle evidence"):
        coverage(campaign, {case["id"]: [bad]})
    bad["campaign"] = "f" * 64
    bad = seal({k: v for k, v in bad.items() if k != "sha256"})
    with pytest.raises(DiagnosticError, match="different inputs"):
        coverage(campaign, {case["id"]: [bad]})


def test_separate_result_directories_merge_without_erasing_failed_attempts(spec, tmp_path):
    campaign = build_campaign(spec)
    case = campaign["cases"][0]
    roots = []
    for index, status in enumerate(("FAILED", "TESTED")):
        root = tmp_path / f"run-{index}"
        (root / "case-00000").mkdir(parents=True)
        write_private(root / "campaign.json", campaign)
        write_private(
            root / "case-00000/result.json",
            case_result(campaign, case, status, started=time.monotonic(), checks=["fixture"]),
        )
        roots.append(root)
    report = coverage(campaign, load_results(campaign, roots))
    assert report["counts"]["FAILED"] == 1 and not report["complete"]
    assert report["cases"][0]["flaky"]
    assert len(report["cases"][0]["attempts"]) == 2
    with pytest.raises(DiagnosticError, match="unique"):
        load_results(campaign, [roots[0], roots[0]])
    with pytest.raises(DiagnosticError, match="missing"):
        load_results(campaign, tmp_path / "absent")


def test_environment_setup_failure_still_reports_whole_incomplete_campaign(
    spec, tmp_path, monkeypatch
):
    import qwen_r9700_lab.conformance_campaign as runner

    def fail_before_launch(*_):
        raise OSError("synthetic qualification cache copy failure")

    monkeypatch.setattr(runner, "worker_environment", fail_before_launch)
    campaign = build_campaign(spec)
    report = run_campaign(campaign, tmp_path / "run", allow_gpu=True)
    assert report["counts"]["ERROR"] == 1
    assert report["counts"]["NOT_RUN"] == len(campaign["cases"]) - 1
    assert not report["complete"]
    assert (tmp_path / "run/coverage.json").is_file()
    result = private_json(tmp_path / "run/case-00000/result.json")
    assert result["error_type"] == "OSError" and result["gpu_executed"] is None


@pytest.mark.parametrize(
    "change",
    [
        {"contexts": []},
        {"contexts": [128, 128]},
        {"seeds": []},
        {"environment": {"QWEN_CONFORMANCE_GPU": "1"}},
        {"output_tokens": 1},
    ],
)
def test_invalid_domains_and_caller_hook_overrides_fail(spec, change):
    spec.update(change)
    with pytest.raises(DiagnosticError):
        build_campaign(spec)


def test_every_mutable_server_path_is_owned_and_input_config_unchanged(spec, tmp_path):
    original = copy.deepcopy(spec["server_config"])
    config = isolated_config(original, tmp_path, "a" * 64, 31111, graphs=True, speculation=True)
    assert original == spec["server_config"]
    assert config["host"] == "127.0.0.1" and config["port"] == 31111
    assert config["api_key"] == "a" * 64
    tier = config["kv_transfer_config"]["kv_connector_extra_config"]["secondary_tiers"][0]
    for key in ("root_dir", "control_directory", "tail_status_path"):
        assert Path(tier[key]).is_relative_to(tmp_path)
    assert config["additional_config"]["qwen_fair"]["status_path"] == str(tmp_path / "fair")
    assert "/never-write-production" not in json.dumps(config)
    argv = server_argv(config)
    assert "--no-async-scheduling" in argv
    assert argv.count("--middleware") == 2
    assert argv[argv.index("--shutdown-timeout") + 1] == "60"


@pytest.mark.parametrize("timeout", [0, -1, True, None, "60", 1.5])
def test_snapshot_server_refuses_abort_only_or_invalid_shutdown(spec, tmp_path, timeout):
    spec["server_config"]["shutdown_timeout"] = timeout
    with pytest.raises(DiagnosticError, match="positive shutdown timeout"):
        isolated_config(
            spec["server_config"], tmp_path, "nonce", 31111, graphs=True, speculation=True
        )


def test_explicit_longer_shutdown_budget_reaches_server(spec, tmp_path):
    spec["server_config"]["shutdown_timeout"] = 120
    config = isolated_config(
        spec["server_config"], tmp_path, "nonce", 31111, graphs=True, speculation=True
    )
    argv = server_argv(config)
    assert argv[argv.index("--shutdown-timeout") + 1] == "120"


def test_async_is_separate_from_unsupported_production_scheduler(spec, tmp_path):
    config = isolated_config(
        spec["server_config"],
        tmp_path,
        "nonce",
        31111,
        graphs=False,
        speculation=True,
        asynchronous=True,
    )
    assert config["async_scheduling"]
    assert "scheduler_cls" not in config and "kv_transfer_config" not in config
    assert "--async-scheduling" in server_argv(config)


def fixture_sse():
    events = [
        {"choices": [{"index": 0, "delta": {"reasoning_content": "Plan λ 🙂"}}]},
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {"name": "record", "arguments": '{"text":'},
                            }
                        ]
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": '"café 🙂"}'}}]
                    },
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]
    raw = ": comment\r\n\r\n" + "".join(
        "data: " + json.dumps(e, ensure_ascii=False) + "\r\n\r\n" for e in events
    )
    return (raw + "data: [DONE]\r\n\r\n").encode()


@pytest.mark.parametrize("width", [1, 2, 3, 7, 31, 4096])
def test_sse_utf8_fragmentation_preserves_exactly_one_tool(width):
    raw, decoder, completion = fixture_sse(), SSEDecoder(), Completion()
    for offset in range(0, len(raw), width):
        for event in decoder.feed(raw[offset : offset + width]):
            completion.accept(event)
    decoder.feed(b"", final=True)
    result = completion.result()
    assert result["reasoning"] == "Plan λ 🙂"
    assert result["tools"] == [
        {
            "id": "call-1",
            "name": "record",
            "arguments": '{"text":"café 🙂"}',
            "parsed_arguments": {"text": "café 🙂"},
        }
    ]


@pytest.mark.parametrize(
    "damage", ["truncate", "error", "after_done", "invalid_utf8", "duplicate_done"]
)
def test_stream_faults_never_become_a_successful_finish(damage):
    raw = fixture_sse()
    if damage == "truncate":
        raw = raw[:-9]
    elif damage == "error":
        raw = b'data: {"error":{"message":"synthetic backend failure"}}\n\n'
    elif damage == "after_done":
        raw += b'data: {"choices":[]}\n\n'
    elif damage == "duplicate_done":
        raw += b"data: [DONE]\n\n"
    else:
        raw = b"data: \xff\n\n"
    with pytest.raises(ProtocolError):
        decoder = SSEDecoder()
        decoder.feed(raw)
        decoder.feed(b"", final=True)


@pytest.mark.parametrize(
    "delta,finish",
    [
        ({"content": "About to act:"}, "tool_calls"),
        (
            {
                "tool_calls": [
                    {"index": 0, "id": "a", "function": {"name": "record", "arguments": "{"}}
                ]
            },
            "tool_calls",
        ),
        (
            {
                "tool_calls": [
                    {"index": 1, "id": "a", "function": {"name": "record", "arguments": "{}"}}
                ]
            },
            "tool_calls",
        ),
        ({"content": "truncated"}, "length"),
    ],
)
def test_finish_and_tool_shape_negative_controls(delta, finish):
    completion = Completion()
    completion.accept({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
    with pytest.raises(ProtocolError):
        completion.result()


def test_same_tokens_with_different_stop_reason_fails():
    with pytest.raises(DiagnosticError, match="stop boundary"):
        same_output(
            {"token_ids": [1], "finish_reason": "length"},
            {"token_ids": [1], "finish_reason": "stop"},
        )


def test_native_symbol_inventory_without_an_actual_call_cannot_pass():
    from qwen_r9700_lab.conformance_scenarios import require_native_dispatch

    site = "r4d.attn_decode_h256_gqa6_fp8kv"
    for calls in (
        [],
        [{"site": site, "completed": False}],
        [{"site": site, "completed": True, "exception_type": "KernelFailure"}],
    ):
        with pytest.raises(DiagnosticError, match="not observed"):
            require_native_dispatch(seal({"bindings": {site: "advertised"}, "calls": calls}), site)
    require_native_dispatch(seal({"calls": [{"site": site, "completed": True}]}), site)


def test_dispatch_catalog_reused_only_for_its_reviewed_binary(spec):
    from qwen_r9700_lab.conformance_campaign import reviewed_dispatch_binding

    unknown = reviewed_dispatch_binding(
        spec["binding"], {"kernel_hashes": {"r4d.so": "f" * 64}}, "radiance-fp8"
    )
    assert "native_entrypoints" not in unknown
    known = reviewed_dispatch_binding(
        spec["binding"],
        {
            "kernel_hashes": {
                "r4d.so": "daa7a3bf79d2a1e0a7909a6ed9ddac2f0f3ac74b878569f4eabc2ccae839aecc"
            }
        },
        "radiance-fp8",
    )
    entry = known["native_entrypoints"][0]
    assert entry["binding"]["required"] == ["attn_decode_h256_gqa6_fp8kv"]
    assert "attn_prefill_h256_gqa6_fp8kv" in entry["binding"]["exports"]
    assert entry["aliases"] == ["radiance_gdn", "radiance_r4d_attn"]


def test_synthetic_long_inputs_are_reproducible_and_nonrepeating():
    first = synthetic_text(17, 1000)
    assert first == synthetic_text(17, 1000) and first != synthetic_text(18, 1000)
    assert len(set(first.splitlines())) == 1000


def test_native_events_require_complete_chain_and_matching_execution(tmp_path):
    events = Events(tmp_path, "a" * 64)
    events.emit("native.enter", rows=8)
    events.emit("native.return", rows=8)
    assert len(read_events(tmp_path, "a" * 64)) == 2
    with pytest.raises(DiagnosticError, match="identity"):
        read_events(tmp_path, "b" * 64)
    path = next(tmp_path.glob("events-*.jsonl"))
    path.write_text(path.read_text().splitlines()[1] + "\n")
    with pytest.raises(DiagnosticError, match="ordering"):
        read_events(tmp_path, "a" * 64)


def test_native_hook_returns_original_and_records_real_error(tmp_path):
    class Subject:
        def method(self, fail=False):
            if fail:
                raise ValueError("synthetic")
            return object_marker

    object_marker = object()
    events = Events(tmp_path, "a" * 64)
    wrap(Subject, "method", events, "native")
    assert Subject().method() is object_marker
    with pytest.raises(ValueError):
        Subject().method(True)
    assert [r["event"] for r in read_events(tmp_path, "a" * 64)] == [
        "native.enter",
        "native.return",
        "native.enter",
        "native.error",
    ]


def test_unbound_native_source_never_executes_before_rejection(tmp_path, monkeypatch):
    import importlib.util

    from qwen_r9700_lab.conformance_observer import ObserverFinder

    (tmp_path / "radiance_verifyhead.py").write_text("raise AssertionError('must not execute')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    finder = ObserverFinder(
        {"observer_sources": {"radiance_verifyhead": "0" * 64}}, Events(tmp_path, "a" * 64)
    )
    found = finder.find_spec("radiance_verifyhead")
    module = importlib.util.module_from_spec(found)
    with pytest.raises(DiagnosticError, match="before import"):
        found.loader.exec_module(module)


def test_same_pid_in_distinct_runs_has_separate_event_chains(tmp_path):
    first, second = Events(tmp_path, "a" * 64), Events(tmp_path, "b" * 64)
    first.emit("first")
    second.emit("second")
    assert len(list(tmp_path.glob("events-*.jsonl"))) == 2
    assert len(read_events(tmp_path, ["a" * 64, "b" * 64])) == 2


def test_qualification_environment_clones_mutable_aiter_tree_and_sets_abi(
    spec, tmp_path, monkeypatch
):
    from qwen_r9700_lab.conformance_runtime import worker_environment

    original = tmp_path / "original-aiter"
    original.mkdir()
    (original / "source.py").write_text("immutable fixture\n")
    monkeypatch.setenv("AITER_ROOT_DIR", str(original))
    env = worker_environment(build_campaign(spec)["spec"], tmp_path / "qualification")
    copied = Path(env["AITER_ROOT_DIR"])
    assert copied != original and copied.is_relative_to(tmp_path / "qualification")
    (copied / "source.py").write_text("compiled fixture\n")
    assert (original / "source.py").read_text() == "immutable fixture\n"
    assert env["QWEN_RADIANCE_CACHE_ABI"] == spec["binding"]["live_data_abi"]


def test_cpu_native_parser_result_is_never_labelled_gpu_execution(spec):
    campaign = build_campaign(spec)
    case = next(c for c in campaign["cases"] if c["variant"] == "pi_provider_fragments")
    result = case_result(
        campaign, case, "TESTED", started=time.monotonic(), checks=["actual_provider"]
    )
    assert result["executed"] and result["gpu_executed"] is False


@pytest.mark.parametrize("omitted", [True, False])
def test_sampling_report_requires_both_positive_and_negative_observations(omitted):
    rows = [
        {"independent_proposal_noise": True, "max_error": 0.001},
        {"independent_proposal_noise": False, "max_error": 0.1},
    ]
    report = {"rows": rows, "greedy_exact": True, "native_target_rows": [{"max_error": 0.001}]}
    validate_operator_report("sampling", report)
    report["rows"] = [r for r in rows if r["independent_proposal_noise"] is not omitted]
    with pytest.raises(DiagnosticError, match="omitted"):
        validate_operator_report("sampling", report)


def test_fault_receipt_only_after_application_and_correct_step(tmp_path, monkeypatch):
    import qwen_r9700_lab.conformance_faults as faults

    path = tmp_path / "experiment.json"
    write_private(path, {"fault": "gdn", "step": 1})
    monkeypatch.setenv("QWEN_CONFORMANCE_GPU", "1")
    monkeypatch.setenv("QWEN_CONFORMANCE_NATIVE_EXPERIMENT", str(path))
    calls = []
    probe = SimpleNamespace(
        index=0,
        campaign=SimpleNamespace(expected=[0, 1], root=tmp_path),
        config={"vocab_size": 32},
        hooks=HookSet(),
        capture=lambda *a, **k: calls.append("observed"),
    )
    monkeypatch.setattr(
        faults, "mutate_device", lambda *a: calls.append("mutated") or "layer.000.gdn"
    )
    install_experiment(probe)
    probe.capture(None, {"consumed": 3}, None)
    assert not (tmp_path / "fault-applied.json").exists()
    probe.index = 1
    probe.capture(None, {"consumed": 4}, None)
    probe.capture(None, {"consumed": 4}, None)
    assert calls == ["observed", "mutated", "observed", "observed"]
    assert json.loads((tmp_path / "fault-applied.json").read_text())["component"] == "layer.000.gdn"
    probe.hooks.close()


SERVER = r"""
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
port, nonce, execution = sys.argv[1:]
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(json.dumps({"nonce":nonce,"execution":execution}).encode())
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        with open('received.json', 'w') as f: json.dump(body, f)
        self.send_response(200); self.end_headers()
        event = {"choices":[{"index":0,"text":"fixture","token_ids":[7,8],
                              "finish_reason":"length"}]}
        wire = ('data: '+json.dumps(event)+'\r\n\r\ndata: [DONE]\r\n\r\n').encode()
        for byte in wire: self.wfile.write(bytes([byte])); self.wfile.flush()
ThreadingHTTPServer(('127.0.0.1',int(port)),Handler).serve_forever()
"""


def test_real_owned_child_http_handshake_and_fragmented_response(tmp_path):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    with OwnedProcess(
        [sys.executable, "-c", SERVER, str(port), "nonce", "execution"],
        tmp_path / "process",
        env=dict(os.environ),
        timeout=15,
    ) as child:
        client = OwnedClient(child, port, "nonce", "execution")
        with pytest.raises(DiagnosticError, match="not been verified"):
            client.json("/v1/completions", {})
        client.connect(timeout=5)
        result = client.completion(
            "/v1/completions",
            {"stream": True, "prompt": [1, 2]},
            evidence=tmp_path / "stream",
            allow_length=True,
        )
        assert result["token_ids"] == [7, 8]
        assert (tmp_path / "stream.response").stat().st_mode & 0o077 == 0
        assert json.loads((tmp_path / "process/received.json").read_text())["prompt"] == [1, 2]
    assert child.process.poll() is not None
    child.close()  # idempotent: must never signal a reused PID


def test_identity_mismatch_never_sends_inference(tmp_path):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    with OwnedProcess(
        [sys.executable, "-c", SERVER, str(port), "wrong", "execution"],
        tmp_path / "process",
        env=dict(os.environ),
        timeout=15,
    ) as child:
        client = OwnedClient(child, port, "nonce", "execution")
        with pytest.raises(DiagnosticError, match="identity mismatch"):
            client.connect(timeout=5)
        assert not (tmp_path / "process/received.json").exists()


def test_child_deadline_retains_private_log_and_terminates_only_owned_process(tmp_path):
    child = OwnedProcess(
        [sys.executable, "-c", "import time; print('fixture',flush=True); time.sleep(10)"],
        tmp_path / "process",
        env=dict(os.environ),
        timeout=0.25,
    )
    with pytest.raises(TimeoutError):
        child.wait()
    assert child.process.poll() is not None
    assert (tmp_path / "process/process.log").read_text().strip() == "fixture"


@pytest.mark.parametrize("crash", [False, True])
def test_controller_death_removes_nested_owned_session_and_leaves_other_job(tmp_path, crash):
    script = r"""
import json, os, sys, time
from pathlib import Path
from qwen_r9700_lab.conformance_transport import OwnedProcess
child = OwnedProcess(
    [sys.executable, "-c", "import os,time; from pathlib import Path; "
     "Path('worker.pid').write_text(str(os.getpid())); time.sleep(60)"],
    Path('nested'), env=dict(os.environ), timeout=60)
while not Path('nested/worker.pid').exists():
    child.check(); time.sleep(0.01)
Path('ready.json').write_text(json.dumps(
    [child.process.pid, int(Path('nested/worker.pid').read_text())]))
time.sleep(60)
"""
    nested = []

    def alive(pid):
        try:
            return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
        except (FileNotFoundError, ProcessLookupError):
            return False

    with (
        OwnedProcess(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            tmp_path / "other-job",
            env=dict(os.environ),
            timeout=60,
        ) as other,
        OwnedProcess(
            [sys.executable, "-c", script],
            tmp_path / "controller",
            env=dict(os.environ),
            timeout=60,
        ) as controller,
    ):
        try:
            deadline = time.monotonic() + 5
            ready = tmp_path / "controller/ready.json"
            while not ready.exists():
                controller.check()
                assert time.monotonic() < deadline, "nested CPU fixture did not start"
                time.sleep(0.01)
            nested = json.loads(ready.read_text())
            assert all(alive(pid) for pid in nested)
            controller.close(crash=crash)
            deadline = time.monotonic() + 5
            while any(alive(pid) for pid in nested) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert not any(alive(pid) for pid in nested), "nested session escaped cleanup"
            other.check()
        finally:
            if nested and alive(nested[0]):
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(nested[0], signal.SIGKILL)


def test_owned_http_refuses_redirect_before_contacting_destination(tmp_path):
    server = SERVER.replace(
        "self.send_response(200); self.end_headers()",
        'self.send_response(307); self.send_header("Location",'
        '"http://127.0.0.1:1/must-not-connect"); self.end_headers()',
    )
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    with OwnedProcess(
        [sys.executable, "-c", server, str(port), "nonce", "execution"],
        tmp_path / "process",
        env=dict(os.environ),
        timeout=15,
    ) as child:
        client = OwnedClient(child, port, "nonce", "execution")
        with pytest.raises(ProtocolError, match="HTTP redirect"):
            client.connect(timeout=5)
        assert not client.verified
