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
    chain = h.suite.read(root / "public/pi-coding-json-compaction.json")
    matched = h.suite.read(root / "runs/60K/report.json")["contexts"]["60K"]["arms"]
    selected = contexts["contexts"]["60K"]
    assert selected["capture_id"] == chain["stages"][0]["capture_id"] == matched["control_after"]["capture_id"]
    assert selected["mean_generation_round_ms"] == 70.  # Not faster control_before.
    assert selected["round_capture"]["records"] == matched["control_after"]["round_capture"]["records"]
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
