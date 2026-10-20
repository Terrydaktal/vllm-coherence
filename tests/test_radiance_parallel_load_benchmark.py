from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from benchmark_radiance_parallel_load import run

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "tests/benchmark_radiance_parallel_load.py"


def invoke(*arguments):
    return subprocess.run([sys.executable, str(BENCHMARK), *arguments],
                          cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                          capture_output=True, text=True, timeout=15, check=False)


def test_benchmark_cli_uses_requested_shape_and_checks_every_slot(tmp_path):
    output = tmp_path / "result.json"
    completed = invoke("--output", str(output), "--repetitions", "2", "--count", "3",
                       "--block-size", "131072")
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text())
    assert report["block_count"] == 3 and report["block_size"] == 131072
    assert report["raw_bytes"] == 3 * 131072 and report["repetitions"] == 2
    assert report["exact_destination_checks"] == 12 and report["fixture_removed"]
    assert report["gpu_measured"] is False
    assert all(len(values) == 2 for values in report["samples_ms"].values())
    assert json.loads(completed.stdout)["output"] == str(output)


@pytest.mark.parametrize("option", ["--repetitions", "--count", "--block-size"])
@pytest.mark.parametrize("value", ["0", "-1", "no"])
def test_invalid_shape_exits_before_creating_a_fixture_or_report(tmp_path, option, value):
    output = tmp_path / "not-created.json"
    completed = invoke("--output", str(output), option, value)
    assert completed.returncode == 2 and "positive integer" in completed.stderr
    assert completed.stdout == "" and not output.exists()


@pytest.mark.parametrize("kwargs", [{"repetitions": 0}, {"count": -1}, {"block_size": 0},
                                   {"count": True}])
def test_library_shape_validation_precedes_io(kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        run(**kwargs)
