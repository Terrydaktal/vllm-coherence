import importlib.util
from pathlib import Path


def analyzer():
    path = Path(__file__).parents[1] / "tools/analyze_release_timings.py"
    spec = importlib.util.spec_from_file_location("worker_stage_analyzer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_window_clips_previous_tail_and_unions_overlap():
    markers = [{"ph": "X", "cat": "user_annotation", "name": f"qwen_timing_round/{i}",
                "ts": i * 10., "dur": 9.} for i in range(4)]
    def kernel(start, duration, stream=1):
        return {"ph": "X", "cat": "kernel", "name": "synthetic_kernel", "ts": start, "dur": duration,
                "args": {"device": 0, "stream": stream}}
    tail, a, b = kernel(19., 2.), kernel(22., 4.), kernel(24., 4., 2)
    targets = [(("target_body", i * 10. + 1), []) for i in range(4)]
    result = analyzer().measure_worker_windows(
        markers + [tail, a, b], targets, [0, 1, 2],
        {id(a): "A", id(b): "B"}, {("drafter", 11.): [tail]},
    )
    row = result["rounds"][0]
    assert row["decode_index"] == 2
    assert row["stages_ms"] == {"Drafter": .001, "A": .004, "B": .004}
    assert abs(row["kernel_sum_ms"] - .009) < 1e-12
    assert abs(row["gpu_busy_ms"] - .007) < 1e-12
    assert abs(row["gpu_overlap_ms"] - .002) < 1e-12
    assert abs(row["overhead_ms"] - .003) < 1e-12
    assert row["clipped_boundary_activity_ms"] == .001
    assert len(result["omissions"]) == 2


def test_bad_target_overlap_excludes_whole_host_window_before_accumulation():
    markers = [{"ph": "X", "cat": "user_annotation", "name": f"qwen_timing_round/{i}",
                "ts": i * 10., "dur": 9.} for i in range(5)]
    def kernel(ts, duration):
        return {"ph": "X", "cat": "kernel", "name": "synthetic", "ts": ts,
                "dur": duration, "args": {"device": 0, "stream": 1}}
    tail, clean = kernel(19, 2), kernel(31, 3)
    targets = [(("target_body", i * 10. + 1), [tail] if i == 1 else []) for i in range(5)]
    result = analyzer().measure_worker_windows(markers + [tail, clean], targets,
        [2, 3], {id(clean): "Clean"}, {}, incomplete_targets={1})
    assert result["omissions"] == [{"target_index": 2, "reason": "overlapping incomplete target inventory"}]
    assert [r["decode_index"] for r in result["rounds"]] == [3]
    assert result["kernel_groups"][0]["ms_per_round"] == .003
