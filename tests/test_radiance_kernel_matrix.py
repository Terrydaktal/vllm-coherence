"""Check the actual factor matrix and fault-detection cases before GPU experiments."""

import json
import shlex
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def matrix(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "experiments/radiance-public"
    monkeypatch.syspath_prepend(str(source))
    import benchmark_kernel_matrix

    return benchmark_kernel_matrix


def test_all_eight_combinations_and_separate_prefill_attention_override(matrix, tmp_path):
    variants = matrix.variants()
    assert len(variants) == 9  # Eight unique combinations, then the repeated baseline.
    assert len({json.dumps(flags, sort_keys=True) for flags in variants.values()}) == 8
    assert variants["K000"] == variants["K000R"]
    assert variants["K111"] == {
        "R4D_ATTN_FP8": "3",
        "RADIANCE_NORMQUANT_FUSION": "1",
        "RADIANCE_FP8_STREAM": "1",
        "RADIANCE_MXFP4_A_TILED_MIN_M": "513",
        "RADIANCE_GDN_NORM_QUANT": "1",
    }
    original = Path(matrix.__file__).with_name("launch_public_clean_snapshot_server.sh").read_text()
    generated = matrix.launcher(original, tmp_path, "K111")
    assert "-e R4D_ATTN_FP8=3 " in generated
    assert "-e R4D_ATTN_FP8=0 " not in generated
    assert str(tmp_path / "profile-K111.json") in generated
    assert "GPU_MAX_HW_QUEUES" not in generated and "HSA_ENABLE_MWAITX" not in generated
    assert '--arg root_dir "/cache/benchmarks/' in generated
    assert "--compilation-config" not in generated
    checked = subprocess.run(["bash", "-n"], input=generated, text=True, capture_output=True)
    assert checked.returncode == 0, checked.stderr
    invocation = generated[generated.index("exec podman run "):]
    arguments = shlex.split(invocation.replace("\\\n", ""))
    compilation = next(arg for arg in arguments if arg.startswith("RADIANCE_COMPILATION_CONFIG="))
    assert json.loads(compilation.split("=", 1)[1]) == {
        "cudagraph_mode": "PIECEWISE",
        "cudagraph_capture_sizes": [1, 2, 4, 8],
    }


def test_private_replay_mount_and_fixture_identity_validation(matrix, tmp_path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir()
    tokens = [12, 34, 56]
    fixture = private / "fixture-60000.json"
    value = {
        "private_replay": True,
        "nominal_context": 60000,
        "actual_context": len(tokens),
        "sha256": matrix.bench.digest(tokens),
        "tokens": tokens,
    }
    matrix.bench.write_json(fixture, value)
    monkeypatch.setenv("QWEN_BENCHMARK_PRIVATE_FIXTURES", str(private))
    monkeypatch.setattr(matrix.bench, "http", lambda *args: pytest.fail("unexpected model request"))
    assert matrix.bench.fixture("unused", tmp_path, 60000) == tokens
    assert not (tmp_path / "fixture-60000.json").exists()
    value["tokens"][0] += 1
    matrix.bench.write_json(fixture, value)
    with pytest.raises(ValueError, match="identity/count/hash"):
        matrix.bench.fixture("unused", tmp_path, 60000)
    original = Path(matrix.__file__).with_name("launch_public_clean_snapshot_server.sh").read_text()
    manifest = tmp_path / "manifest.json"
    matrix.bench.write_json(
        manifest, {"private_fixture_directory": "/dev/shm/qwen-private-replay-test"}
    )
    generated = matrix.launcher(original, tmp_path, "K000")
    assert "-v /dev/shm/qwen-private-replay-test:/private-fixtures:ro" in generated
    assert "-e QWEN_BENCHMARK_PRIVATE_FIXTURES=/private-fixtures" in generated
    matrix.bench.write_json(manifest, {"private_fixture_directory": "/home/user/private-chat"})
    with pytest.raises(ValueError, match="isolated temporary memory directory"):
        matrix.launcher(original, tmp_path, "K000")


def test_tool_checks_reject_duplicates_truncation_and_type_changes(matrix):
    expected, _ = matrix.tool_case(2)
    call = {"function": {"name": "record", "arguments": json.dumps(expected)}}
    choice = {"finish_reason": "tool_calls", "message": {"tool_calls": [call]}}
    assert matrix.check_tool(choice, expected)
    choice["message"]["tool_calls"].append(call)
    assert not matrix.check_tool(choice, expected)
    choice["message"]["tool_calls"] = [call]
    choice["finish_reason"] = "length"
    assert not matrix.check_tool(choice, expected)
    choice["finish_reason"] = "tool_calls"
    call["function"]["arguments"] = json.dumps({**expected, "enabled": 0})
    assert not matrix.check_tool(choice, expected)
    call["function"]["arguments"] = "{bad json"
    assert not matrix.check_tool(choice, expected)


def test_all_tool_fixtures_have_json_round_trippable_expected_values(matrix):
    for index in range(matrix.TOOL_CASES):
        expected, function = matrix.tool_case(index)
        assert json.loads(json.dumps(expected)) == expected
        assert set(expected) == set(function["parameters"]["required"])
    compile(matrix.FLUSH_PRODUCTION, "flush_production", "exec")


def test_cleanup_removes_only_owned_synthetic_cache(matrix, tmp_path, monkeypatch):
    root = tmp_path / "qwen-runtime-ab-cleanup"
    root.mkdir()
    cache = tmp_path / "cache"
    monkeypatch.setattr(matrix.bench, "CACHE_ROOT", cache)
    owned = cache / "benchmarks" / root.name
    owned.mkdir(parents=True)
    (owned / "synthetic.bin").write_bytes(b"synthetic")
    private = cache / "snapshots" / "private"
    private.mkdir(parents=True)
    (private / "keep.bin").write_bytes(b"keep")
    matrix.cleanup_cache(root)
    assert not owned.exists()
    assert (private / "keep.bin").read_bytes() == b"keep"
    owned.symlink_to(private, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        matrix.cleanup_cache(root)
    assert (private / "keep.bin").exists()


def test_private_comparison_keeps_failed_rows_and_rejects_mixed_requests(matrix, tmp_path):
    write = matrix.bench.write_json
    fixtures = [
        {"nominal_context": 60000, "actual_context": 56273, "sha256": "prompt-a"},
        {"nominal_context": 200000, "actual_context": 200161, "sha256": "prompt-b"},
    ]
    write(tmp_path / "manifest.json", {"private_replay_metadata": fixtures})
    with pytest.raises(ValueError, match="not completed"):
        matrix.comparison(tmp_path)
    write(
        tmp_path / "status.json", {"stage": "complete_with_failures", "production_restored": True}
    )
    rejected = {name: {"stage": "startup", "message": "dependency"} for name in ("K001", "K101")}
    write(tmp_path / "variant-failures.json", rejected)
    for variant, flags in matrix.variants().items():
        if variant in rejected:
            continue
        write(tmp_path / f"{variant}-configuration.json", {"image": "same", "flags": flags})
        write(tmp_path / f"{variant}-tool-tests.json", [{"passed": True}] * matrix.TOOL_CASES)
        checks = [
            {"actual_context": f["actual_context"], "greedy_sha256": "same"} for f in fixtures
        ]
        write(
            tmp_path / f"{variant}-correctness.json",
            {"tool_cases": matrix.TOOL_CASES, "contexts": checks},
        )
        for fixture in fixtures:
            context = fixture["actual_context"]
            write(
                tmp_path / f"{variant}-{context}-cold.json", {"phase_timings_ms": {"prefill": 1200}}
            )
            for repeat in range(1, 4):
                record = {
                    "variant": variant,
                    "context": context,
                    "trial": f"measured-{repeat}",
                    "output_tokens": 1024,
                    "prompt_sha256": fixture["sha256"],
                    "sampling": matrix.SAMPLING,
                    "output_sha256": "same",
                    "steady": {
                        "round_ms": (110 if variant == "K000R" else 100) + repeat - 2,
                        "tokens_per_second": 40,
                    },
                    "allocator": {"allocation_retries": 0, "out_of_memory_events": 0},
                }
                write(tmp_path / f"{variant}-{context}-measured-{repeat}.json", record)
    report = matrix.comparison(tmp_path)
    assert len(report["rows"]) == 16
    assert sum(r["status"] == "startup_rejected" for r in report["rows"]) == 2
    last = next(r for r in report["rows"] if r["variant"] == "K000R")
    assert last["context"] == 56273
    assert last["round_time_change_percent"] == pytest.approx(10)
    assert last["range"]["round_ms"] == [109, 111]
    write(
        tmp_path / "scope-change.json", {"omitted_variants": {"K000R": "User omitted the repeat"}}
    )
    omitted = matrix.comparison(tmp_path)
    assert (
        next(r for r in omitted["rows"] if r["variant"] == "K000R")["status"] == "omitted_by_user"
    )
    assert len(omitted["rows"]) == 15
    damaged = tmp_path / "K100-200161-measured-2.json"
    record = matrix.bench.read_json(damaged)
    record["prompt_sha256"] = "another-private-request"
    write(damaged, record)
    with pytest.raises(ValueError, match="mismatched private trial"):
        matrix.comparison(tmp_path)
