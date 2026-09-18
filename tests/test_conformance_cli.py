import json
import os
import subprocess
from pathlib import Path

import pytest
from conformance_fixture import tiny_plan

from qwen_r9700_lab.diagnostic_contract import write_private

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/qwen-conformance"


def test_recovery_control_call_and_certificate_cli(tmp_path):
    from test_conformance_control import SCHEMA, transcript
    from test_conformance_instrumentation import record_model
    from test_conformance_lifecycle import reference

    from qwen_r9700_lab.diagnostic_contract import digest, seal

    plan = tiny_plan(tmp_path / "checkpoint")
    write_private(tmp_path / "plan.json", plan)
    model = reference(plan)
    try:
        for token in plan["prefix"]:
            model.step(token)
        model.frame(tmp_path / "saved", phase="commit", pending=7)
        model.state[0]["gdn"].flat[0] += 1
        model.frame(tmp_path / "live", phase="step", pending=7)
    finally:
        model.close()
    result = cli(
        "recovery",
        "--plan",
        tmp_path / "plan.json",
        "--reference",
        tmp_path / "saved",
        "--live",
        tmp_path / "live",
        "--restored",
        tmp_path / "saved",
        "--output",
        tmp_path / "recovery",
    )
    assert result.returncode == 1, result.stderr
    assert json.loads(result.stdout)["classification"] == "live_differs_restored_matches_reference"
    assert "pending" not in result.stdout
    write_private(
        tmp_path / "events.json",
        seal(
            {
                "schema": SCHEMA,
                "execution": digest("test"),
                "adapter": digest("test"),
                "events": transcript(),
            }
        ),
    )
    result = cli(
        "control", "--trace", tmp_path / "events.json", "--output", tmp_path / "control.json"
    )
    assert result.returncode == 0, result.stderr
    write_private(
        tmp_path / "bounds.json",
        {"lower": ["2", "0"], "upper": ["3", "1"], "winner": 0, "bound_origin": "test fixture"},
    )
    result = cli(
        "certificate",
        "--bounds",
        tmp_path / "bounds.json",
        "--output",
        tmp_path / "certificate.json",
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["bounds_soundness"] == "ASSUMED"
    record_model(tmp_path / "calls-a")
    record_model(tmp_path / "calls-b", fault=True)
    result = cli(
        "calls",
        "--reference",
        tmp_path / "calls-a",
        "--candidate",
        tmp_path / "calls-b",
        "--output",
        tmp_path / "calls.json",
    )
    assert result.returncode == 1, result.stderr
    assert json.loads(result.stdout)["first_difference"]["site"] == "inner"


def cli(*args, program=SCRIPT):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"VIRTUAL_ENV", "PYTHONPATH", "QWEN_CONFORMANCE_PYTHON"}
    }
    env["PATH"] = "/usr/bin:/bin"
    return subprocess.run(
        [str(program), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=20,
        cwd="/tmp",
        env=env,
    )


@pytest.mark.parametrize("failure_binding", [None, "same", "wrong"])
def test_native_worker_uses_named_serializable_worker_rpc(tmp_path, monkeypatch, failure_binding):
    import importlib.machinery
    import sys
    from types import ModuleType, SimpleNamespace

    from qwen_r9700_lab import conformance_radiance
    from qwen_r9700_lab.conformance_cli import native_worker
    from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal

    plan = tiny_plan(tmp_path / "checkpoint")
    binding = seal({"fixture": "binding"})
    write_private(tmp_path / "plan.json", plan)
    write_private(tmp_path / "binding.json", binding)
    write_private(
        tmp_path / "config.json",
        {"model": plan["checkpoint"], "enforce_eager": True, "max_num_seqs": 1},
    )
    calls, settings, aborted = [], {}, []
    request_id = "qwen-isolated-conformance"

    class Engine:
        engine_core = SimpleNamespace(
            resources=SimpleNamespace(engine_dead=False), utility_results={}
        )

        def add_request(self, name, prompt, params):
            assert name == request_id
            assert prompt["prompt_token_ids"] == plan["prefix"]
            # The independent engine process may execute ahead of the client.
            # Its scheduler must stop exactly at the final forced output.
            assert params.max_tokens == len(plan["forced_tokens"])

        def has_unfinished_requests(self):
            return True

        def step(self):
            return [
                SimpleNamespace(
                    request_id=request_id,
                    outputs=[SimpleNamespace(token_ids=plan["forced_tokens"])],
                )
            ]

        def abort_request(self, names):
            aborted.extend(names)

    class LLM:
        def __init__(self, **kwargs):
            settings.update(kwargs)
            self.llm_engine = Engine()

        def collective_rpc(self, method, args=()):
            # vLLM 0.28's default transport rejects Python functions. Keep this
            # independent transport constraint even when the fake worker runs.
            if not isinstance(method, str):
                raise TypeError("worker RPC cannot serialize a Python function")
            json.dumps([method, args])
            calls.append((method, args))
            if method == "qwen_conformance_finish" and failure_binding:
                (root / "capture").mkdir()
                conformance_radiance.preserve_native_failure(
                    root / "capture",
                    plan,
                    binding if failure_binding == "same" else seal({"fixture": "other"}),
                    "finish",
                    DiagnosticError("backend stopped before the replay finished"),
                )
                raise RuntimeError("worker collective RPC failed")
            return [{"completed": True}]

    module = ModuleType("vllm")
    module.__spec__ = importlib.machinery.ModuleSpec(
        "vllm", loader=None, origin=str(tmp_path / "site-packages/vllm/__init__.py")
    )
    module.LLM = LLM
    module.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setattr(conformance_radiance, "verify_sources", lambda *args: None)
    monkeypatch.setenv("QWEN_CONFORMANCE_GPU", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("RADIANCE_VERIFY_HEAD", "0")
    monkeypatch.delenv("VLLM_ALLOW_INSECURE_SERIALIZATION", raising=False)
    root = tmp_path / "native"
    root.mkdir()
    args = tmp_path / "plan.json", tmp_path / "config.json", tmp_path / "binding.json", root
    if failure_binding:
        expected = "replay finished" if failure_binding == "same" else "another replay"
        with pytest.raises(DiagnosticError, match=expected):
            native_worker(*args)
        assert not (root / "receipt.json").exists()
    else:
        native_worker(*args)
    assert settings["worker_extension_cls"] == (
        "qwen_r9700_lab.conformance_radiance.ConformanceWorkerExtension"
    )
    assert [name for name, _ in calls] == [
        "qwen_conformance_install",
        "qwen_conformance_finish",
    ]
    assert calls[0][1] == (
        str(tmp_path / "plan.json"),
        str(root / "capture"),
        str(tmp_path / "binding.json"),
    )
    assert aborted == [request_id]
    if not failure_binding:
        assert json.loads((root / "receipt.json").read_text())["workers"] == [{"completed": True}]
    assert "VLLM_ALLOW_INSECURE_SERIALIZATION" not in os.environ


def test_pending_native_rpc_is_released_only_after_worker_death():
    from concurrent.futures import Future
    from types import SimpleNamespace

    from qwen_r9700_lab.conformance_cli import reject_dead_native_rpcs
    from qwen_r9700_lab.diagnostic_contract import DiagnosticError

    complete, cancelled, pending = Future(), Future(), Future()
    complete.set_result("original result")
    cancelled.cancel()
    client = SimpleNamespace(
        resources=SimpleNamespace(engine_dead=False),
        utility_results={1: complete, 2: cancelled, 3: pending},
    )
    with reject_dead_native_rpcs(client):
        with pytest.raises(TimeoutError):
            pending.result(timeout=0.2)
        client.resources.engine_dead = True
        with pytest.raises(DiagnosticError, match="native worker exited during RPC"):
            pending.result(timeout=2)
        assert complete.result() == "original result"
        assert cancelled.cancelled()


def test_real_cli_help_inventory_and_unarmed_gpu(tmp_path):
    result = cli("--help")
    assert result.returncode == 0
    for name in (
        "NAME",
        "SYNOPSIS",
        "DESCRIPTION",
        "OPTIONS",
        "OPERATION",
        "EXAMPLES",
        "FILES",
        "PATHS",
        "SECURITY NOTES",
        "EXIT STATUS",
        "AUTHORS",
    ):
        assert name in result.stdout
    result = cli("inventory")
    assert result.returncode == 0
    assert json.loads(result.stdout)["gpu_use_by_default"] is False
    result = cli(
        "native",
        "--plan",
        "missing",
        "--config",
        "missing",
        "--binding",
        "missing",
        "--output",
        tmp_path / "no-gpu",
    )
    assert result.returncode == 2
    assert "GPU use was not authorized" in result.stderr
    assert not (tmp_path / "no-gpu").exists()


def test_symlink_entry_uses_project_uv_environment_without_activation(tmp_path):
    link = tmp_path / "qwen-conformance"
    link.symlink_to(SCRIPT)
    result = cli("inventory", program=link)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["production_hooks_installed"] is False


def test_real_cli_reference_compare_boundary_pipeline_and_proofs(tmp_path):
    plan = tiny_plan(tmp_path / "checkpoint")
    spec = {
        k: plan[k]
        for k in ("checkpoint", "checkpoint_files", "kv_scales", "prefix", "forced_tokens")
    }
    write_private(tmp_path / "spec.json", spec)
    assert (
        cli(
            "make-plan", "--spec", tmp_path / "spec.json", "--output", tmp_path / "plan.json"
        ).returncode
        == 0
    )
    for name in ("reference", "candidate"):
        result = cli("reference", "--plan", tmp_path / "plan.json", "--output", tmp_path / name)
        assert result.returncode == 0, result.stderr
        assert "tokens" not in json.loads(result.stdout)
    result = cli(
        "compare",
        "--reference",
        tmp_path / "reference",
        "--candidate",
        tmp_path / "candidate",
        "--output",
        tmp_path / "diff",
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["equal"]
    result = cli(
        "boundaries",
        "--reference",
        tmp_path / "reference/boundaries",
        "--candidate",
        tmp_path / "candidate/boundaries",
        "--output",
        tmp_path / "boundaries",
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["equal"]
    result = cli("prove", "--output", tmp_path / "proofs")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["all_expected_results"]


def test_malformed_private_plan_does_not_echo_its_contents(tmp_path):
    secret = "private-content-sentinel-74691"
    path = tmp_path / "invalid.json"
    path.write_text(secret)
    path.chmod(0o600)
    result = cli("reference", "--plan", path, "--output", tmp_path / "out")
    assert result.returncode == 2
    assert secret not in result.stdout + result.stderr


def test_real_gate_cli_publishes_only_a_checked_revision(tmp_path):
    import numpy as np

    from qwen_r9700_lab.conformance_session import CheckedSession
    from qwen_r9700_lab.conformance_state import FrameWriter
    from qwen_r9700_lab.diagnostic_contract import digest

    coverage = ["sequence.tokens", "sequence.position", "gdn"]
    for name, corrupt in (("reference", False), ("bad", True), ("good", False)):
        writer = FrameWriter(
            tmp_path / name,
            contract=digest("contract"),
            execution=digest("public fixture"),
            adapter=digest("fixture"),
            input_digest=digest([1, 2]),
            phase="prefill",
            consumed=2,
            pending=3,
            expected=coverage,
        )
        writer.array("sequence.tokens", np.asarray([1, 2], dtype="<i4"))
        writer.array("sequence.position", np.asarray([2], dtype="<i8"))
        writer.array("gdn", np.asarray([int(corrupt)], dtype="<f4"))
        writer.finish()
    transition = {
        "contract": digest("contract"),
        "coverage": coverage,
        "reference": str(tmp_path / "reference"),
        "candidate": str(tmp_path / "bad"),
        "base_revision": 0,
        "reference_tokens": [3],
        "candidate_tokens": [3],
        "reference_stop": None,
        "candidate_stop": None,
    }
    write_private(tmp_path / "bad.json", transition)
    root = tmp_path / "authority"
    result = cli("gate", "--authority", root, "--transition", tmp_path / "bad.json", "--create")
    assert result.returncode == 1 and result.stdout == ""
    transition["candidate"] = str(tmp_path / "good")
    write_private(tmp_path / "good.json", transition)
    result = cli("gate", "--authority", root, "--transition", tmp_path / "good.json")
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["revision"] == 1 and summary["output_count"] == 1
    assert "tokens" not in summary
    session = CheckedSession(root, contract=digest("contract"), required_components=coverage)
    try:
        assert session.summary()["rejected_transitions"] == 1
        assert session.outputs_since(0) == [{"revision": 1, "tokens": (3,), "stop_reason": None}]
    finally:
        session.close()
