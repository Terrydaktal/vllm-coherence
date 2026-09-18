"""Independent checks for benchmark accounting and invalid measurement rejection."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE = (
    Path(__file__).resolve().parents[1] / "experiments/radiance-public/benchmark_runtime_flags.py"
)
SPEC = importlib.util.spec_from_file_location("runtime_benchmark", MODULE)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def sample(seconds, phase="generate", waiting=0):
    return {
        "monotonic": seconds,
        "phase": {"phase": phase, "request_id": "same"},
        "metrics": {
            "vllm:spec_decode_num_drafts_total": 10 * seconds,
            "vllm:spec_decode_num_draft_tokens_total": 70 * seconds,
            "vllm:spec_decode_num_accepted_tokens_total": 30 * seconds,
            "vllm:generation_tokens_total": 40 * seconds,
            "vllm:num_requests_running": 1,
            "vllm:num_requests_waiting": waiting,
            "vllm:num_preemptions_total": 0,
        },
        "thermal": {"temp2_input": 70000, "power1_average": 200000000},
    }


def test_steady_window_excludes_prefill_and_finished_time():
    value = bench.steady_summary(
        [sample(0, "prefill"), sample(10), sample(20), sample(30, "complete")]
    )
    assert value["round_ms"] == 100
    assert value["tokens_per_second"] == 40
    assert value["tokens_per_round"] == 4
    assert value["acceptance"] == pytest.approx(3 / 7)
    assert value["power_w_median"] == 200


def test_contended_or_reset_trial_is_rejected():
    with pytest.raises(ValueError, match="contention"):
        bench.steady_summary([sample(10), sample(15, waiting=1), sample(20)])
    bad = sample(20)
    bad["metrics"]["vllm:generation_tokens_total"] = 0
    with pytest.raises(ValueError, match="reset"):
        bench.steady_summary([sample(10), bad])


def test_metric_families_are_not_double_counted():
    value = bench.parse_metrics(
        'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 30\n'
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 30\n'
        "# unrelated counter\n"
    )
    assert value == {"vllm:spec_decode_num_accepted_tokens_total": 30}


def test_counter_window_excludes_prefill_and_completed_requests():
    before = sample(1, "before_request")
    before["metrics"]["vllm:num_requests_running"] = 0
    prefill = sample(10, "prefill")
    prefill["metrics"]["vllm:spec_decode_num_drafts_total"] = 10
    finished = sample(40, "complete")
    finished["metrics"]["vllm:num_requests_running"] = 0
    transformed = bench.counter_generation_window(
        [before, prefill, sample(20), sample(30), finished]
    )
    result = bench.steady_summary(transformed)
    assert result["seconds"] == 10
    assert result["tokens_per_second"] == 40
    assert result["round_ms"] == 100
    with pytest.raises(ValueError, match="pre-generation"):
        bench.counter_generation_window([sample(20), sample(30)])


def test_launch_variants_change_only_the_two_selected_runtime_flags():
    original = MODULE.with_name("launch_public_clean_snapshot_server.sh").read_text()
    root = Path("/tmp/qwen-runtime-ab-unit")
    baseline = bench.make_launcher(original, root, "A", "benchmarks/unit")
    both = bench.make_launcher(original, root, "D", "benchmarks/unit")
    assert (
        both.replace("\t-e GPU_MAX_HW_QUEUES=1 \\\n", "").replace(
            "\t-e HSA_ENABLE_MWAITX=1 \\\n", ""
        )
        == baseline
    )
    assert '--arg root_dir "/cache/benchmarks/unit/data"' in baseline
    assert "--kv-cache-memory 10000000000" in baseline
    assert "--max-model-len 253792" in baseline


def test_comparison_exposes_drift_and_rejects_different_workload(tmp_path):
    for variant, flags in bench.FLAGS.items():
        bench.write_json(
            tmp_path / f"{variant}-configuration.json", {"flags": flags, "image": "same-image"}
        )
        bench.write_json(tmp_path / f"{variant}-tool-tests.json", [{"passed": True}] * 8)
        for context in (60000, 200000):
            bench.write_json(
                tmp_path / f"{variant}-{context}-cold.json", {"phase_timings_ms": {"prefill": 1000}}
            )
            for repeat in range(1, 4):
                record = {
                    "variant": variant,
                    "context": context,
                    "trial": f"measured-{repeat}",
                    "output_tokens": 1024,
                    "prompt_sha256": str(context),
                    "output_sha256": "same-output",
                    "steady": {
                        "round_ms": (110 if variant == "A2" else 100) + repeat - 2,
                        "tokens_per_second": 40,
                    },
                    "phase_timings_ms": {"prefill": 50},
                    "usage": {"prompt_tokens_details": {"cached_tokens": context - 2}},
                    "allocator": {"allocation_retries": 0, "out_of_memory_events": 0},
                }
                bench.write_json(tmp_path / f"{variant}-{context}-measured-{repeat}.json", record)
    result = bench.comparison(tmp_path)
    assert result["identical_greedy_outputs"]
    assert result["all_tool_smoke_checks_passed"]
    assert len(result["rows"]) == 10
    assert result["rows"][0]["baseline_drift_percent"] == pytest.approx(10)
    assert result["rows"][0]["range"]["round_ms"] == [99, 101]
    changed = tmp_path / "B-60000-measured-2.json"
    record = bench.read_json(changed)
    record["output_sha256"] = "different-output"
    bench.write_json(changed, record)
    assert not bench.comparison(tmp_path)["identical_greedy_outputs"]
    record["prompt_sha256"] = "different-input"
    bench.write_json(changed, record)
    with pytest.raises(ValueError, match="prompt tokens differ"):
        bench.comparison(tmp_path)


def test_failed_kernel_startup_continues_and_restores_production(tmp_path, monkeypatch):
    root = tmp_path / "qwen-runtime-ab-failuretest"
    root.mkdir()
    original = "#!/bin/bash\n"
    (root / "production-launcher.sh").write_text(original)
    production = root / "original.sh"
    production.write_text(original)
    bench.write_json(
        root / "manifest.json",
        {
            "launcher_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "production_launcher": str(production),
            "continue_on_variant_failure": True,
        },
    )
    bench.write_json(root / "flush.json", {"pending_chats": []})
    validation = {}
    for variant in ("bad", "good"):
        (root / f"launch-{variant}.sh").write_text(original)
        validation[variant] = {
            "sha256": hashlib.sha256(original.encode()).hexdigest(),
            "shellcheck": True,
            "shfmt": True,
        }
    bench.write_json(root / "launcher-validation.json", validation)
    monkeypatch.setattr(bench, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(
        bench, "FLAGS", {"bad": {"BENCH_SETTING": "0"}, "good": {"BENCH_SETTING": "1"}}
    )
    monkeypatch.setattr(
        bench.http_server,
        "ThreadingHTTPServer",
        lambda *args: SimpleNamespace(
            serve_forever=lambda: None, shutdown=lambda: None, server_close=lambda: None
        ),
    )
    launches, workers = [], []
    state = {"running": False, "variant": None}

    class Process:
        pid = 123

        def __init__(self, argv, **kwargs):
            name = Path(argv[1]).name
            launches.append(name)
            state.update(running=True, variant=name.removeprefix("launch-").removesuffix(".sh"))

        def wait(self, **kwargs):
            return 0

    def command(argv, **kwargs):
        if argv[:2] == ["podman", "inspect"]:
            return SimpleNamespace(
                stdout=json.dumps(
                    [
                        {
                            "Config": {"Env": ["BENCH_SETTING=1"]},
                            "Image": "same",
                            "State": {"StartedAt": "now"},
                        }
                    ]
                )
            )
        if argv[:2] == ["podman", "stop"]:
            state["running"] = False
        return SimpleNamespace(stdout="", returncode=0)

    def run(argv, **kwargs):
        if argv[:3] == ["podman", "container", "exists"]:
            return SimpleNamespace(returncode=0 if argv[3] == root.name and state["running"] else 1)
        workers.append(argv[-1])
        return SimpleNamespace(returncode=0)

    def ready(*args):
        if state["variant"] == "bad":
            raise RuntimeError("synthetic kernel startup failure")

    monkeypatch.setattr(bench.subprocess, "Popen", Process)
    monkeypatch.setattr(bench.subprocess, "run", run)
    monkeypatch.setattr(bench, "command", command)
    monkeypatch.setattr(bench, "wait_ready", ready)
    assert bench.run(SimpleNamespace(root=root, port=18080)) == 0
    assert launches == ["launch-bad.sh", "launch-good.sh", "original.sh"]
    assert workers == ["good"]
    result = bench.read_json(root / "status.json")
    assert result["stage"] == "complete_with_failures" and result["production_restored"]
    assert result["variant_failures"]["bad"]["stage"] == "startup"
