"""Verify old-stack isolation without launching a GPU process."""

from pathlib import Path

import pytest


def test_preserved_launcher_keeps_model_settings_and_isolates_cache(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[1] / "experiments/radiance-public"
    monkeypatch.syspath_prepend(str(source))
    import benchmark_old_stack as old

    original = (
        "#!/bin/bash\nset -euo pipefail\n"
        "readonly container_name=qwen38-27b-uncensored-mxfp4-public-snapshot-candidate\n"
        '--arg root_dir "/cache/snapshots/${data_abi}/data"\n'
        "exec env \\\n\tPYTHONHASHSEED=0 \\\n\tPORT=8080 \\\n"
        "\tMAXLEN=253792 KV_MEM=10000000000 RADIANCE_VERIFY_HEAD=0 \\\n"
        "\tbash ./startup-qwen3.8-27b-mxfp4.sh\n"
    )
    root = tmp_path / "qwen-runtime-ab-oldtest"
    result = old.legacy_launcher(original, root)
    assert "MAXLEN=253792 KV_MEM=10000000000 RADIANCE_VERIFY_HEAD=0" in result
    assert "AUTO_R4D=0" in result and old.OLD_R4D in result
    assert old.OLD_R4D_SHA256 in result
    assert f"/cache/benchmarks/{root.name}/data" in result
    assert "QWEN_QUALIFICATION_CONTAINER:?" in result
    assert "QWEN_QUALIFICATION_PORT:-18080" in result
    assert "bash ./startup-qwen3.8-27b-mxfp4.sh" in result
    shim = old.runtime_shim(root, "/dev/shm/qwen-private-replay-test")
    assert "QWEN_BENCHMARK_LEGACY_METRICS=1" in shim
    assert f"QWEN_RADIANCE_CACHE_ABI={old.OLD_ABI}" in shim
    assert "Refusing an unowned benchmark container" in shim
    assert ":/private-fixtures:ro" in shim
    assert len(old.OLD_ABI) == 64
    with pytest.raises(ValueError, match="private RAM"):
        old.runtime_shim(root, "/home/user/private-sessions")


def test_old_new_report_uses_equal_counter_boundaries_and_distinct_images(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[1] / "experiments/radiance-public"
    monkeypatch.syspath_prepend(str(source))
    import benchmark_old_stack as old

    fixture = {"nominal_context": 200000, "actual_context": 200161, "sha256": "same-private-input"}
    write = old.bench.write_json
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    for root, variant, scale in ((old_root, "OLD", 1), (new_root, "K000", 1.1)):
        root.mkdir()
        write(root / "manifest.json", {"private_replay_metadata": [fixture]})
        write(root / "status.json", {"stage": "complete", "production_restored": True})
        flags = old.OLD_FLAGS["OLD"] if variant == "OLD" else old.matrix.variants()["K000"]
        write(root / f"{variant}-configuration.json", {"image": variant, "flags": flags})
        write(
            root / f"{variant}-correctness.json",
            {"tool_cases": 64, "tool_passed": 63, "contexts": [{"actual_context": 200161}]},
        )
        samples = []
        for seconds in (0, 10, 20, 30):
            running = seconds in (10, 20)
            samples.append(
                {
                    "monotonic": scale * seconds,
                    "phase": {"phase": "generate" if running else "outside_generation"},
                    "metrics": {
                        "vllm:spec_decode_num_drafts_total": seconds * 10,
                        "vllm:spec_decode_num_draft_tokens_total": seconds * 70,
                        "vllm:spec_decode_num_accepted_tokens_total": seconds * 30,
                        "vllm:generation_tokens_total": seconds * 40,
                        "vllm:num_requests_running": int(running),
                        "vllm:num_requests_waiting": 0,
                        "vllm:num_preemptions_total": 0,
                    },
                    "thermal": {"temp2_input": 70000, "power1_average": 200000000},
                }
            )
        for i in range(1, 4):
            write(
                root / f"{variant}-200161-measured-{i}.json",
                {
                    "variant": variant,
                    "context": 200161,
                    "trial": f"measured-{i}",
                    "output_tokens": 1024,
                    "prompt_sha256": fixture["sha256"],
                    "sampling": old.matrix.SAMPLING,
                    "usage": {"prompt_tokens": 200161},
                    "samples": samples,
                    "output_sha256": "same-output",
                    "allocator": {"allocation_retries": 0, "out_of_memory_events": 0},
                },
            )
    report = old.comparison(old_root, new_root)
    new = next(row for row in report["rows"] if row["variant"] == "K000")
    assert new["median"]["round_ms"] == pytest.approx(110)
    assert new["round_time_change_percent_vs_old"] == pytest.approx(10)
    assert new["token_rate_change_percent_vs_old"] == pytest.approx(-100 / 11)
    p = old_root / "OLD-configuration.json"
    value = old.bench.read_json(p)
    value["image"] = "K000"
    write(p, value)
    with pytest.raises(ValueError, match="distinct pinned images"):
        old.comparison(old_root, new_root)
