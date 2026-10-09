"""CPU orchestration tests: real round-log capture, fake inference/worker only."""

import copy
import importlib
import json
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "experiments/radiance-public"


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    suite = importlib.import_module("benchmark_pi_suite")
    args = suite.parser().parse_args([
        "--output", str(tmp_path / "captures"), "--fixture-60k", str(tmp_path / "60k.json"),
        "--fixture-200k", str(tmp_path / "200k.json"), "--tokenizer-json", str(tmp_path / "tokenizer.json"),
        "--runtime-manifest", str(tmp_path / "runtime.json"), "--abi", "test-abi",
        "--round-log", str(tmp_path / "rounds.jsonl"), "--round-log-settle-timeout", "0",
    ])
    args.output.mkdir(mode=0o700)
    (tmp_path / "rounds-cache-jobs.jsonl").write_text(json.dumps({
        "schema": "urn:coherence:cache-job-timings:v1", "stage": "python_gc",
        "start_ns": 100, "end_ns": 200, "round": 148, "request_id": "numeric-identity",
    }) + "\n")
    (tmp_path / "rounds-cache-jobs-health.json").write_text(json.dumps({
        "schema": "urn:coherence:cache-job-timings:v1", "trace_id": "test-trace", "pid": 123,
        "dropped": 0, "write_errors": 0, "context_drops": 0, "round_sample_errors": 0,
        "worker_first_work_hooks": True, "generation_round_telemetry": True,
        "gpu_rounds": {"completed": 6, "dropped": 0, "errors": 0, "quarantined": 0},
    }))
    for path, data in ((args.fixture_60k, {"prefix": [41] * 60000}),
                       (args.fixture_200k, {"prefix": [42] * 200000}),
                       (args.tokenizer_json, {"tokenizer": "test"}),
                       (args.runtime_manifest, {"release": "test"})):
        path.write_text(json.dumps(data))
    metadata = {
        "enforce_eager": False, "compilation_mode": 3, "backend": "inductor",
        "graph_mode": "PIECEWISE", "capture_sizes": [8], "async_scheduling": False,
        "effective_capacity": {"max_model_len": 253792}, "diagnostic_sources": {"worker": "hash"},
        "startup_compatibility": {}, "repair": {"bundle": "repair-hash"},
        "performance": {"manifest": "performance-hash", "gemm_dispatch": {"binary_sha256": "gemm-hash"}},
        "runtime": {"flags": {}, "packages": {"torch": "pinned"}},
    }
    h = SimpleNamespace(suite=suite, args=args, calls=[], rpc_calls=[], analyses=[], active=None,
                        fail_stage=None, fail_arm=None, skip_round=False, short=False,
                        metadata=metadata, worker_id="worker-1", depth="512")

    class Tokenizer:
        def token_to_id(self, token):
            return 99

        def encode(self, text, add_special_tokens=False):
            return SimpleNamespace(ids=[99, 10])

    h.tokenizer = Tokenizer()
    prompts = {prompt: i for i, (_, prompt, _) in enumerate(
        (("coding", suite.chain.CODING_PROMPT, False), *suite.STAGES), start=200)}
    monkeypatch.setattr(suite.chain, "_render_user_turn",
                        lambda opener, base, prompt, **kw: [prompts[prompt], int(kw["thinking"])])

    def rpc(method, *values):
        h.rpc_calls.append((method, values))
        if method == "qwen_timing_identity":
            return {"instance": h.worker_id, "model": {"model": "synthetic"},
                    "metadata": copy.deepcopy(h.metadata), "arm_active": h.active is not None,
                    "target_head_environment": {"RADIANCE_VERIFY_HEAD": "1", "RADIANCE_VERIFY_HEAD_GLOBAL_TOPK": h.depth}}
        if method == "qwen_timing_status":
            return {"instance": h.worker_id, "arm_active": h.active is not None}
        if method == "qwen_timing_arm":
            path, mode, *_ = values
            Path(path).mkdir(mode=0o700)
            h.active = {"path": Path(path), "mode": mode}
            return {"metadata": copy.deepcopy(h.metadata)}
        if method == "qwen_timing_finish":
            arm = h.active
            h.active = None
            chunks = []
            if arm["mode"] == "profile":
                trace = arm["path"] / "chunk-000" / "profile-trace.json"
                trace.parent.mkdir()
                trace.write_text('{}')
                chunks = [{"observation": {"profile_files": [str(trace)]}}]
            result = {"metadata": copy.deepcopy(h.metadata), "chunks": chunks,
                      "rows": [{"decode_index": i, "entry_ns": i * 41000000, "scheduled_tokens": 8} for i in range(6)]}
            (arm["path"] / "worker.json").write_text(json.dumps(result))
            return result
        raise AssertionError(method)

    h.rpc = rpc

    def request(**kw):
        stage = kw["stage"]
        arm = h.active["path"].name if h.active else None
        h.calls.append({"stage": stage, "arm": arm, "identity": kw["identity"].copy(),
                        "prompt": list(kw["prompt_tokens"]), "suffix": list(kw["suffix_tokens"]),
                        "thinking": kw["thinking"]})
        if h.fail_stage == stage or (h.fail_arm and arm == h.fail_arm):
            raise RuntimeError("injected inference failure")
        ids = [900 + len(h.calls), 99]
        if stage == "coding":
            # Distinguish fast control_before from slower control_after; selection
            # must stay predetermined, not cherry-pick the fastest sample.
            duration = 70. if arm and arm.endswith("control_after") else 30.
            with args.round_log.open("a") as stream:
                for number, ms in enumerate((None, duration, 2500., 42., 44., 41.), 1):
                    if h.skip_round and number == 3:
                        continue
                    stream.write(json.dumps({"schema": suite.coding.ROUND_LOG_SCHEMA,
                        "pid": 123, "request_id": str(len(h.calls)), "round": number,
                        "chat_id": kw["identity"]["id"], "generation": kw["identity"]["generation"],
                        "observed_at_ms": int(time.time()*1000), "round_ms": ms,
                        "monotonic_ns": number * 1_000_000, "cache_trace_id": "test-trace",
                        "draft_tokens": 0 if number == 1 else 7,
                        "accepted_tokens": 0 if number == 1 else 4,
                        "acceptance_rate": 0 if number == 1 else 4/7}) + "\n")
        result = {"stage": stage, "generated_tokens": len(ids), "generation_rounds": 5,
                  "mean_generation_round_ms": 70. if arm and arm.endswith("control_after") else 30.,
                  "output_sha256": str(len(h.calls)), "minimum_output_met": not h.short,
                  "finish_reason": "stop", "prompt_tokens": len(kw["prompt_tokens"]),
                  "post_first_tokens_per_second": 80., "acceptance_rate": 4/7,
                  "cached_prompt_tokens": len(kw["prompt_tokens"]) - 2, "uncached_prompt_tokens": 2}
        if stage == "compaction":
            result["checkpoint_validation"] = {"passed": True}
        return result, ids, {"content": "PRIVATE GENERATED CONTENT"}

    monkeypatch.setattr(suite.chain, "_request", request)
    monkeypatch.setattr(suite.matched, "_request", request)
    monkeypatch.setattr(suite, "analyze_stages", lambda root, head: h.analyses.append((root, head)))
    h.execute = lambda: suite.execute(args, None, h.tokenizer, rpc)
    return h


def test_shared_captures_feed_all_tables_and_exact_60k_continuation(harness):
    h = harness
    state = h.execute()
    coding = [call for call in h.calls if call["stage"] == "coding"]
    assert len(coding) == 12  # 3 x warmup/before/profile/after, not another 4 copies.
    assert [call["stage"] for call in h.calls[8:12]] == ["prose_code", "json", "thinking", "compaction"]
    assert len(h.calls) == 16
    root = h.args.output
    contexts = h.suite.read(root / "public/pi-coding-contexts.json")
    histogram = h.suite.read(root / "public/pi-round-histogram.json")
    chain = h.suite.read(root / "public/pi-coding-json-compaction.json")
    matched = h.suite.read(root / "runs/60K/report.json")["contexts"]["60K"]["arms"]
    selected = contexts["contexts"]["60K"]
    assert selected["capture_id"] == chain["stages"][0]["capture_id"] == matched["control_after"]["capture_id"]
    assert selected["mean_generation_round_ms"] == 70.  # Not faster control_before.
    assert selected["round_capture"]["records"] == matched["control_after"]["round_capture"]["records"]
    assert histogram["measurement_mode"] == "shared_suite_control_after"
    assert histogram["coding_context_result"] == "pi-coding-contexts.json"
    assert histogram["capture_selection"] == "control_after"
    assert histogram["contexts"] == contexts["contexts"]
    assert histogram["suite_capture_id"] == contexts["suite_capture_id"] == state["id"]
    for field in ("contract_sha256", "tokenizer_sha256", "runtime_manifest_sha256",
                  "validation_failures", "status"):
        assert histogram[field] == contexts[field]
    capture = selected["round_capture"]
    assert capture["status"] == "captured"
    assert capture["record_count"] == 6
    assert capture["histogram"]["measured_round_count"] == 5
    assert sum(row["count"] for row in capture["histogram"]["bins"]) == 5
    assert capture["histogram"]["maximum_ms"] == 2500.  # Outliers remain.
    code, prose = h.calls[7], h.calls[8]
    assert prose["prompt"] == code["prompt"] + [908, 99] + prose["suffix"]
    assert prose["identity"] == code["identity"]
    assert h.analyses == [(root, "global512")]
    assert state["status"] == "complete"
    for path in (root / "continuations").glob("*.json"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    for path in (root / "public").glob("*.json"):
        text = path.read_text()
        assert "PRIVATE GENERATED CONTENT" not in text and '"token_ids":' not in text


def test_new_partial_suite_replaces_stale_histogram_without_extra_requests(harness):
    h = harness
    stale = h.args.output / "public/pi-round-histogram.json"
    h.suite.write_private(stale, {"suite_capture_id": "old-focused-capture", "status": "complete"})
    h.fail_stage = "json"
    with pytest.raises(RuntimeError, match="injected"):
        h.execute()
    partial = h.suite.read(stale)
    contexts = h.suite.read(h.args.output / "public/pi-coding-contexts.json")
    assert partial["suite_capture_id"] != "old-focused-capture"
    assert partial["suite_capture_id"] == contexts["suite_capture_id"]
    assert partial["status"] == "running"
    assert partial["contexts"] == contexts["contexts"]
    assert set(partial["contexts"]) == {"0K", "60K"}
    assert len(h.calls) == 10
    h.fail_stage = None
    h.execute()
    complete = h.suite.read(stale)
    assert complete["status"] == "complete"
    assert set(complete["contexts"]) == {"0K", "60K", "200K"}
    assert len(h.calls) == 17  # Only the failed JSON request is repeated.


def test_completed_resume_does_not_generate_again(harness):
    h = harness
    first = h.execute()
    count = len(h.calls)
    second = h.execute()
    assert len(h.calls) == count
    assert second == first


def test_resume_chain_reuses_coding_and_finished_stages(harness):
    h = harness
    h.fail_stage = "json"
    with pytest.raises(RuntimeError, match="injected"):
        h.execute()
    assert len(h.calls) == 10
    h.fail_stage = None
    h.execute()
    assert [call["stage"] for call in h.calls[10:]] == ["json", "thinking", "compaction"] + ["coding"] * 4
    assert h.calls[9]["prompt"] == h.calls[10]["prompt"]
    assert len([call for call in h.calls if call["stage"] == "coding"]) == 12


def test_interrupted_group_restarts_whole_bracket_and_preserves_evidence(harness):
    h = harness
    h.fail_arm = "60K-profile"
    with pytest.raises(RuntimeError, match="injected"):
        h.execute()
    assert h.active is None  # Profiling cleanup still ran.
    old_identity = h.calls[-1]["identity"]
    h.fail_arm = None
    before = len(h.calls)
    h.execute()
    assert [call["arm"] for call in h.calls[before:before+4]] == [None, "60K-control_before", "60K-profile", "60K-control_after"]
    assert h.calls[before]["identity"] != old_identity
    assert list((h.args.output / "abandoned").glob("60K-*/60K-profile/worker.json"))
    assert len([call for call in h.calls if call["arm"] == "0K-control_after"]) == 1


@pytest.mark.parametrize("change", ["sampler", "release", "tokenizer", "fixture", "worker", "capacity", "source", "template", "head"])
def test_changed_identity_refuses_reuse_before_inference(harness, monkeypatch, change):
    h = harness
    h.execute()
    count = len(h.calls)
    if change == "sampler":
        h.args.top_k = 20
    elif change == "release":
        h.args.runtime_manifest.write_text('{"release":"other"}')
    elif change == "tokenizer":
        h.args.tokenizer_json.write_text('{"tokenizer":"other"}')
    elif change == "fixture":
        h.args.fixture_60k.write_text(json.dumps({"prefix": [123] * 60000}))
    elif change == "worker":
        h.worker_id = "restarted-worker"
    elif change == "capacity":
        h.metadata["effective_capacity"]["max_model_len"] -= 1
    elif change == "source":
        monkeypatch.setattr(h.suite, "SOURCE_FILES", h.suite.SOURCE_FILES[:-1])
    elif change == "template":
        monkeypatch.setattr(h.suite.chain, "_render_user_turn", lambda *a, **kw: [300, 301])
    elif change == "head":
        h.depth = "256"
    with pytest.raises(ValueError, match="identity changed|head flags"):
        h.execute()
    assert len(h.calls) == count


@pytest.mark.parametrize("artifact", ["continuation", "worker", "trace"])
def test_missing_or_corrupt_evidence_is_not_silently_regenerated(harness, artifact):
    h = harness
    h.execute()
    root = h.args.output
    path = (next((root / "continuations").glob("*.json")) if artifact == "continuation" else
            root / "runs/60K/60K-control_after/worker.json" if artifact == "worker" else
            root / "runs/60K/60K-profile/chunk-000/profile-trace.json")
    path.write_text('{"corrupted":true}')
    count = len(h.calls)
    with pytest.raises(ValueError, match="artifact missing or changed"):
        h.execute()
    assert len(h.calls) == count


def test_short_output_and_missing_rounds_remain_visible(harness):
    h = harness
    h.skip_round = True
    h.short = True
    h.execute()
    report = h.suite.read(h.args.output / "public/pi-coding-contexts.json")
    assert report["status"] == "complete_with_validation_failure"
    assert len(report["validation_failures"]) == 6
    assert report["contexts"]["60K"]["round_capture"]["missing_round_numbers"] == [3]
    assert len(h.calls) == 16  # No secret retry to improve a failed result.


def test_rejects_eager_before_generation(harness):
    h = harness
    h.metadata["enforce_eager"] = True
    with pytest.raises(RuntimeError, match="compiled execution required"):
        h.execute()
    assert not h.calls


def test_cli_suite_help_and_invalid_invocation_outside_repository(tmp_path):
    script = SCRIPTS / "benchmark_pi_coding_contexts.py"
    result = subprocess.run([sys.executable, str(script), "--suite", "--help"],
                            cwd=tmp_path, text=True, capture_output=True, check=False)
    assert result.returncode == 0
    assert "--runtime-manifest" in result.stdout and "--profile-rounds" in result.stdout
    result = subprocess.run([sys.executable, str(script), "--suite"], cwd=tmp_path,
                            text=True, capture_output=True, check=False)
    assert result.returncode == 2 and "required" in result.stderr


def test_raw_diagnostics_survive_tmpfs_teardown_and_keep_causal_fields(harness):
    h = harness
    source = h.args.round_log.parent
    round_row = {"round": 148, "monotonic_ns": 700_123, "cache_trace_id": "test-trace",
                 "request_id": "private-request-hash", "computed_tokens": 60037,
                 "scheduled_shape": {"num_scheduled_tokens": 8}, "round_ms": 591.203}
    original = (json.dumps(round_row) + "\n").encode()
    h.args.round_log.write_bytes(original)
    h.args.round_log.with_name("rounds.jsonl.1").write_bytes(original.replace(b"148", b"147"))
    api_log = source / "rounds-api-123-cache-jobs.jsonl"
    api_log.write_bytes(b'{"stage":"python_gc","start_ns":50,"end_ns":60}\n')
    api_health = source / "rounds-api-123-cache-jobs-health.json"
    api_health.write_bytes(b'{"dropped":0,"write_errors":0}\n')
    (source / "session.jsonl").write_text("PRIVATE CHAT MUST NOT BE COPIED")
    archive = h.args.output / "diagnostics"
    result = h.suite.retain_diagnostics(h.args.round_log, archive)
    assert result["status"] == "retained"
    manifest = h.suite.read(archive / "manifest.json")
    assert manifest["rotation_retries"] == 0
    assert {item["file"] for item in manifest["files"]} == {
        "rounds.jsonl", "rounds.jsonl.1", "rounds-cache-jobs.jsonl", "rounds-cache-jobs-health.json",
        api_log.name, api_health.name}
    for item in manifest["files"]:
        assert (archive / item["file"]).read_bytes() == (source / item["file"]).read_bytes()
        assert h.suite.file_hash(archive / item["file"]) == item["sha256"]
        assert stat.S_IMODE((archive / item["file"]).stat().st_mode) == 0o600
    for name in [item["file"] for item in manifest["files"]] + ["session.jsonl"]:
        (source / name).unlink()  # Simulate container tmpfs loss after shutdown.
    assert (archive / "rounds.jsonl").read_bytes() == original
    assert json.loads((archive / "rounds.jsonl").read_text()) == round_row


def test_explicit_diagnostic_prefix_can_differ_from_round_writer(harness):
    h = harness
    h.args.round_log.write_text('{"round":1,"cache_trace_id":"test-trace","pid":123}\n')
    source = h.args.round_log.parent
    for suffix in ("-cache-jobs.jsonl", "-cache-jobs-health.json"):
        (source / ("rounds" + suffix)).rename(source / ("actual-recorder" + suffix))
    h.args.cache_telemetry_prefix = source / "actual-recorder"
    archive = h.args.output / "explicit-prefix"
    result = h.suite.retain_request_diagnostics(h.args, archive)
    assert result["status"] == "retained"
    manifest = h.suite.read(archive / "manifest.json")
    assert manifest["cache_telemetry_prefix"] == str(h.args.cache_telemetry_prefix)
    assert manifest["cache_telemetry_prefix_source"] == "explicit"
    assert {item["file"] for item in manifest["files"]} == {
        "rounds.jsonl", "actual-recorder-cache-jobs.jsonl",
        "actual-recorder-cache-jobs-health.json"}


def test_recorder_health_identity_resolves_independent_prefix(harness):
    h = harness
    h.args.round_log.write_text('{"round":1,"cache_trace_id":"test-trace","pid":123}\n')
    source = h.args.round_log.parent
    for suffix in ("-cache-jobs.jsonl", "-cache-jobs-health.json"):
        (source / ("rounds" + suffix)).rename(source / ("actual-recorder" + suffix))
    (source / "unrelated-cache-jobs-health.json").write_text('{"trace_id":"other","pid":321}\n')
    (source / "unrelated-cache-jobs.jsonl").write_text('{"stage":"irrelevant"}\n')
    archive = h.args.output / "resolved-prefix"
    result = h.suite.retain_diagnostics(h.args.round_log, archive)
    assert result["status"] == "retained"
    manifest = h.suite.read(archive / "manifest.json")
    assert manifest["cache_telemetry_prefix_source"] == "matching_round_trace_health"
    assert manifest["cache_telemetry_prefix"] == str(source / "actual-recorder")
    assert not (archive / "unrelated-cache-jobs.jsonl").exists()


def test_recorder_identity_outranks_stale_round_prefix_health(harness):
    h = harness
    source = h.args.round_log.parent
    h.args.round_log.write_text('{"round":1,"cache_trace_id":"test-trace","pid":123}\n')
    for suffix in ("-cache-jobs.jsonl", "-cache-jobs-health.json"):
        (source / ("actual-recorder" + suffix)).write_bytes((source / ("rounds" + suffix)).read_bytes())
    (source / "rounds-cache-jobs-health.json").write_text('{"trace_id":"old-lifetime","pid":321}\n')
    result = h.suite.retain_diagnostics(h.args.round_log, h.args.output / "stale-prefix")
    assert result["status"] == "retained"
    manifest = h.suite.read(h.args.output / "stale-prefix/manifest.json")
    assert manifest["cache_telemetry_prefix"] == str(source / "actual-recorder")
    assert manifest["cache_telemetry_prefix_source"] == "matching_round_trace_health"


@pytest.mark.parametrize("explicit", [False, True], ids=["stale-default", "wrong-explicit"])
def test_wrong_recorder_lifecycle_never_qualifies_retained_files(harness, explicit):
    h = harness
    source = h.args.round_log.parent
    h.args.round_log.write_text('{"round":1,"cache_trace_id":"current-trace","pid":456}\n')
    original = (source / "rounds-cache-jobs.jsonl").read_bytes()
    archive = h.args.output / "wrong-lifecycle"
    result = h.suite.retain_diagnostics(
        h.args.round_log, archive, source / "rounds" if explicit else None)
    assert result["status"] == "retained_with_gaps"
    assert "recorder_identity_mismatch" in {item["code"] for item in result["issues"]}
    manifest = h.suite.read(archive / "manifest.json")
    assert manifest["round_recorder_identity"] == {"trace_id": "current-trace", "pid": 456}
    assert manifest["recorder_identity_verified"] is False
    assert (archive / "rounds-cache-jobs.jsonl").read_bytes() == original
    assert manifest["cache_telemetry_prefix_source"] == ("explicit" if explicit else "round_path")


def test_missing_recorder_identity_is_explicit_without_discarding_rounds(harness):
    h = harness
    source = h.args.round_log.parent
    original = b'{"round":1,"cache_trace_id":"current-trace","pid":456}\n'
    h.args.round_log.write_bytes(original)
    (source / "rounds-cache-jobs-health.json").unlink()
    archive = h.args.output / "missing-identity"
    result = h.suite.retain_diagnostics(h.args.round_log, archive)
    assert result["status"] == "retained_with_gaps"
    assert "recorder_identity_missing" in {item["code"] for item in result["issues"]}
    assert h.suite.read(archive / "manifest.json")["recorder_identity_verified"] is False
    assert (archive / "rounds.jsonl").read_bytes() == original


def test_current_lifetime_only_is_used_to_resolve_recorder_prefix(harness):
    h = harness
    source = h.args.round_log.parent
    h.args.round_log.write_text(
        '{"round":1,"cache_trace_id":"test-trace","pid":123}\n'
        '{"round":1,"cache_trace_id":"current-trace","pid":456}\n')
    health = h.suite.read(source / "rounds-cache-jobs-health.json")
    health.update(trace_id="current-trace", pid=456)
    (source / "current-cache-jobs-health.json").write_text(json.dumps(health))
    (source / "current-cache-jobs.jsonl").write_text('{"stage":"python_gc"}\n')
    archive = h.args.output / "newest-lifetime"
    result = h.suite.retain_diagnostics(h.args.round_log, archive)
    assert result["status"] == "retained"
    manifest = h.suite.read(archive / "manifest.json")
    assert manifest["cache_telemetry_prefix"] == str(source / "current")
    assert manifest["round_recorder_identity"] == {"trace_id": "current-trace", "pid": 456}
    assert manifest["recorder_identity_verified"] is True
    assert not (archive / "rounds-cache-jobs-health.json").exists()


def test_diagnostic_rotation_retries_and_preserves_both_generations(harness, monkeypatch):
    h = harness
    old = b'{"round":1,"monotonic_ns":10}\n'
    new = b'{"round":2,"monotonic_ns":20}\n'
    h.args.round_log.write_bytes(old)
    opened = h.suite.os.open
    changed = False
    round_opens = 0

    def rotate_before_open(path, *args, **kwargs):
        nonlocal changed, round_opens
        if Path(path) == h.args.round_log:
            round_opens += 1
        # The first read resolves recorder identity. Rotate after the archive's
        # inventory, when its own round-file copy is opened.
        if Path(path) == h.args.round_log and round_opens == 2 and not changed:
            changed = True
            h.args.round_log.rename(h.args.round_log.with_name("rounds.jsonl.1"))
            h.args.round_log.write_bytes(new)
        return opened(path, *args, **kwargs)

    monkeypatch.setattr(h.suite.os, "open", rotate_before_open)
    archive = h.args.output / "rotation"
    result = h.suite.retain_diagnostics(h.args.round_log, archive)
    assert result["status"] == "retained"
    manifest = h.suite.read(archive / "manifest.json")
    assert manifest["attempts"] == 2 and manifest["rotation_retries"] == 1
    assert (archive / "rounds.jsonl.1").read_bytes() == old
    assert (archive / "rounds.jsonl").read_bytes() == new


def test_diagnostic_missing_truncated_and_recorder_loss_are_not_complete(harness):
    h = harness
    h.args.round_log.write_bytes(b'{"round":1}\n{"round":')
    (h.args.round_log.parent / "rounds-cache-jobs.jsonl").unlink()
    health = h.args.round_log.parent / "rounds-cache-jobs-health.json"
    data = h.suite.read(health)
    data.update(dropped=4, gpu_rounds={"dropped":2})
    health.write_text(json.dumps(data))
    result = h.suite.retain_diagnostics(h.args.round_log, h.args.output / "partial")
    assert result["status"] == "retained_with_gaps"
    codes = {row["code"] for row in result["issues"]}
    assert {"incomplete_record", "required_feed_missing", "recorder_reports_loss", "gpu_recorder_reports_loss"} <= codes
    assert (h.args.output / "partial/rounds.jsonl").read_bytes() == b'{"round":1}\n{"round":'


def test_diagnostic_bounds_and_symlinks_never_silently_drop_evidence(harness, monkeypatch):
    h = harness
    secret = h.args.output / "secret"
    secret.write_text("PRIVATE CHAT")
    h.args.round_log.symlink_to(secret)
    monkeypatch.setattr(h.suite, "DIAGNOSTIC_FILE_MAX_BYTES", 16)
    result = h.suite.retain_diagnostics(h.args.round_log, h.args.output / "bounded")
    assert result["status"] == "retained_with_gaps"
    assert {"non_regular_file", "byte_limit"} <= {row["code"] for row in result["issues"]}
    assert not (h.args.output / "bounded/rounds.jsonl").exists()


def test_failed_arm_and_chain_still_preserve_diagnostics(harness):
    h = harness
    h.fail_arm = "0K-profile"
    with pytest.raises(RuntimeError, match="injected"):
        h.execute()
    arm = h.args.output / "runs/0K/0K-profile/diagnostics"
    assert (arm / "rounds.jsonl").is_file() and (arm / "manifest.json").is_file()
    assert h.active is None
    h.fail_arm = None
    h.fail_stage = "json"
    with pytest.raises(RuntimeError, match="injected"):
        h.execute()
    assert list((h.args.output / "diagnostics/chain").glob("json-*/manifest.json"))


def test_missing_diagnostics_fail_qualification_without_repeating_work(harness):
    h = harness
    (h.args.round_log.parent / "rounds-cache-jobs-health.json").unlink()
    h.execute()
    report = h.suite.read(h.args.output / "public/pi-coding-contexts.json")
    assert report["status"] == "complete_with_validation_failure"
    assert all("diagnostic archive" in item for item in report["validation_failures"])
    assert len(h.calls) == 16


def test_changed_chain_diagnostic_artifact_refuses_resume(harness):
    h = harness
    state = h.execute()
    artifact = state["chain"][0]["diagnostic_artifacts"][0]
    (h.args.output / artifact["path"]).write_text("{}")
    with pytest.raises(ValueError, match="artifact missing or changed"):
        h.execute()
    assert len(h.calls) == 16


def test_diagnostic_write_failure_does_not_replace_inference_error(harness, monkeypatch):
    h = harness
    h.fail_arm = "0K-profile"

    def unavailable(*args, **kwargs):
        raise PermissionError("injected diagnostic-only write failure")

    monkeypatch.setattr(h.suite, "retain_diagnostics", unavailable)
    with pytest.raises(RuntimeError, match="injected inference failure"):
        h.execute()
    assert h.active is None
    assert list((h.args.output / "runs/0K/0K-profile/diagnostics").glob("failure.json"))


def test_continuous_rotation_exhausts_bounded_retries_explicitly(harness, monkeypatch):
    h = harness
    h.args.round_log.write_bytes(b'{"round":1}\n')
    opened = h.suite.os.open

    def rotate_every_open(path, *args, **kwargs):
        if Path(path) == h.args.round_log:
            rotated = h.args.round_log.with_name("rounds.jsonl.1")
            rotated.unlink(missing_ok=True)
            h.args.round_log.rename(rotated)
            h.args.round_log.write_bytes(b'{"round":2}\n')
        return opened(path, *args, **kwargs)

    monkeypatch.setattr(h.suite.os, "open", rotate_every_open)
    archive = h.args.output / "unstable"
    result = h.suite.retain_diagnostics(h.args.round_log, archive)
    manifest = h.suite.read(archive / "manifest.json")
    assert result["status"] == "retained_with_gaps"
    assert manifest["attempts"] == manifest["rotation_retries"] == 3
    assert "rotation_did_not_settle" in {row["code"] for row in result["issues"]}


def test_malformed_diagnostic_bytes_are_preserved_for_postmortem(harness):
    h = harness
    malformed = b'{"round":1}\nnot-json\n{"round":'
    h.args.round_log.write_bytes(malformed)
    archive = h.args.output / "malformed"
    result = h.suite.retain_diagnostics(h.args.round_log, archive)
    assert result["status"] == "retained_with_gaps"
    assert (archive / "rounds.jsonl").read_bytes() == malformed
    item = next(row for row in h.suite.read(archive / "manifest.json")["files"] if row["file"] == "rounds.jsonl")
    assert item["validated_records"] == 1 and item["parse_errors"] == 2
    assert item["incomplete_tail_bytes"] == len(b'{"round":')
