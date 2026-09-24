"""Check the diagnostic's independent witnesses, controls and admission gates.

These tests reproduce unresolved source-level conditions, not repaired native
kernels. They must never be presented as GPU or model qualification.
"""

import errno
import hashlib
import importlib
import io
import json
from fractions import Fraction
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def probes(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments/radiance-public"))
    return (
        importlib.import_module("probe_m1_followup_cpu"),
        importlib.import_module("probe_m1_attention_precision"),
    )


def test_exact_storage_encoding_and_invalid_fp8_rejection(probes):
    _, attention = probes
    assert attention.fp8_bits([1.0, 1.125, 448.0, -1.0]).tolist() == [
        0x38,
        0x39,
        0x7E,
        0xB8,
    ]
    with pytest.raises(ValueError, match="not an exact E4M3"):
        attention.fp8_bits([1.01])
    for x in (1.0, 1.0078125, 2**-22, 2**-30, 2**20):
        assert float(attention.from_bf16(attention.bf16_bits(x))) == x


def test_half_rtz_has_exact_underflow_saturation_and_rounding(probes):
    cpu, _ = probes
    source = [2**-25, -(2**-25), 2**-24, 2**20, -(2**20), 1 + 17 / 4096]
    assert cpu.half_rtz(source).tolist() == [
        0.0,
        -0.0,
        2**-24,
        65504.0,
        -65504.0,
        1 + 1 / 256,
    ]
    with pytest.raises(ValueError, match="finite"):
        cpu.half_rtz([float("nan")])


def test_attention_oracle_does_not_depend_on_fp16_simulation(probes):
    cpu, attention = probes
    rows = {row["case"]: row for row in cpu.attention_witnesses()}
    for fixture in attention.fixtures():
        name, _, _, _, values, expected = fixture
        if not name.startswith("uniform_mean"):
            continue
        count = int(np.count_nonzero(values[:, 0] == 1.125))
        exact = Fraction(1) + Fraction(count, 8 * len(values))
        assert float(exact) == expected
        # Adding 1/256 is the exact BF16 midpoint above 1.0.
        expected_rounded = 1.0078125 if exact > Fraction(257, 256) else 1.0
        assert rows[name]["expected_bf16"] == expected_rounded
    assert all(rows[f"uniform_mean_{n}"]["equal"] for n in (16, 32, 512))
    assert all(not rows[f"uniform_mean_{n}"]["equal"] for n in (31, 511, 1023, 4095))
    assert rows["folded_query_underflow"]["source_arithmetic_bf16"] == 0
    assert rows["folded_query_underflow"]["expected_bf16"] > 0


def test_actual_packed_wrapper_accepts_wrong_inner_state_layout(probes):
    pytest.importorskip("torch")
    cpu, _ = probes
    root = ROOT / "benchmarks/fixtures/eager-m1-followup-20260924"
    fixture = (root / "packed_decode.py.txt").read_bytes()
    manifest = json.loads((root / "manifest.json").read_text())
    assert hashlib.sha256(fixture).hexdigest() == manifest["extracted_sha256"]
    assert manifest["source_sha256"] == cpu.PINNED["fixed_fused_recurrent.py"]
    control, witness = cpu.state_layout_witnesses(fixture.decode())
    assert (
        control["accepted_by_actual_wrapper"]
        and control["wrongly_addressed_values"] == 0
    )
    assert witness["accepted_by_actual_wrapper"]
    assert witness["wrongly_addressed_values"] == 393216
    assert not witness["native_launch_executed"]


def test_fnuz_and_ocp_are_distinct_not_interchangeable_aliases(probes):
    pytest.importorskip("torch")
    result = probes[0].format_witness()
    assert result["ocp_e4m3"][:2] == [1.0, 2.0]
    assert result["fnuz_e4m3"][:2] == [0.5, 1.0]
    assert result["ocp_e4m3"][2] == 0
    assert result["fnuz_e4m3"][2] == "NaN"


@pytest.mark.parametrize(
    "metrics",
    [
        "",
        "vllm:num_requests_running 0",
        "vllm:num_requests_running 1\nvllm:num_requests_waiting 0",
        "vllm:num_requests_running 0\nvllm:num_requests_waiting NaN",
    ],
)
def test_native_probe_refuses_busy_or_unknown_server(probes, monkeypatch, metrics):
    attention = probes[1]
    monkeypatch.setattr(
        attention.urllib.request,
        "urlopen",
        lambda *_a, **_k: io.BytesIO(metrics.encode()),
    )
    with pytest.raises(RuntimeError, match="busy or idle counters"):
        attention.idle("http://invalid")


def test_valid_idle_counters(probes, monkeypatch):
    attention = probes[1]
    metrics = b'vllm:num_requests_running{engine="0"} 0\nvllm:num_requests_waiting{engine="0"} 0'
    monkeypatch.setattr(
        attention.urllib.request, "urlopen", lambda *_a, **_k: io.BytesIO(metrics)
    )
    assert attention.idle("http://invalid") == {"running": [0.0], "waiting": [0.0]}


def test_standalone_accepts_only_connection_refusal(probes, monkeypatch):
    attention = probes[1]

    def refused(*_args, **_kwargs):
        raise URLError(ConnectionRefusedError(errno.ECONNREFUSED, "refused"))

    monkeypatch.setattr(attention.urllib.request, "urlopen", refused)
    assert attention.backend_stopped("http://invalid") == {
        "status": "connection_refused"
    }


@pytest.mark.parametrize(
    "reason", [TimeoutError("timeout"), OSError("unknown"), "HTTP 503"]
)
def test_standalone_does_not_treat_unknown_failure_as_stopped(
    probes, monkeypatch, reason
):
    attention = probes[1]

    def failed(*_args, **_kwargs):
        raise URLError(reason)

    monkeypatch.setattr(attention.urllib.request, "urlopen", failed)
    with pytest.raises(RuntimeError, match="could not establish"):
        attention.backend_stopped("http://invalid")


def test_standalone_rejects_responding_backend(probes, monkeypatch):
    attention = probes[1]
    monkeypatch.setattr(
        attention.urllib.request, "urlopen", lambda *_a, **_k: io.BytesIO(b"")
    )
    with pytest.raises(RuntimeError, match="still responds"):
        attention.backend_stopped("http://invalid")
