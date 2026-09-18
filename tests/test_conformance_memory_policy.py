"""CPU checks of process-scoped mitigation; these do not qualify the GPU driver."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import sys
from types import SimpleNamespace

import pytest

from qwen_r9700_lab import conformance_supervisor as supervisor
from qwen_r9700_lab.conformance_transport import OwnedProcess


def thp_state():
    zero = ctypes.c_ulong(0)
    return ctypes.CDLL(None).prctl(42, zero, zero, zero, zero)


def test_owned_child_and_exec_grandchild_inherit_policy_without_changing_parent(tmp_path):
    parent_before = thp_state()
    query = (
        "import ctypes,json; from pathlib import Path; "
        "z=ctypes.c_ulong(0); "
        "s=Path('/proc/self/status').read_text().splitlines(); "
        "print(json.dumps({'prctl':ctypes.CDLL(None).prctl(42,z,z,z,z),"
        "'status':[x for x in s if x.startswith('THP_enabled:')]}))"
    )
    script = (
        "import json,subprocess,sys; from pathlib import Path; "
        f"query={query!r}; "
        "child=json.loads(subprocess.check_output([sys.executable,'-c',query])); "
        "Path('observed.json').write_text(json.dumps(child))"
    )
    root = tmp_path / "worker"
    with OwnedProcess(
        [sys.executable, "-c", script], root, env=dict(os.environ), timeout=10
    ) as job:
        assert job.wait() == 0
    observed = json.loads((root / "observed.json").read_text())
    assert observed["prctl"] == 1
    assert observed["status"] == ["THP_enabled:\t0"]
    receipt = json.loads((root / "memory-policy.json").read_text())
    assert receipt["transparent_hugepages"] == "disabled"
    assert receipt["pr_get_thp_disable"] == 1
    assert receipt["host_global_policy_changed"] is False
    assert thp_state() == parent_before


@pytest.mark.parametrize("get_state", [-1, 0, 3])
def test_unconfirmed_or_except_advised_policy_is_rejected(get_state):
    libc = SimpleNamespace(prctl=lambda operation, *_: 0 if operation == 41 else get_state)
    with pytest.raises(RuntimeError, match="not confirmed disabled"):
        supervisor.disable_transparent_hugepages(libc)


def test_failed_policy_setup_stops_before_workload_launch(tmp_path, monkeypatch):
    invocation = tmp_path / "invocation.json"
    invocation.write_text(json.dumps({"argv": ["must-not-run"]}))

    def prctl(operation, *_):
        if operation == 1:  # Parent-death guardian registration succeeds.
            return 0
        ctypes.set_errno(errno.EPERM)
        return -1

    def forbidden_launch(*_, **__):
        pytest.fail("workload launched without required memory policy")

    monkeypatch.setattr(supervisor.ctypes, "CDLL", lambda *_, **__: SimpleNamespace(prctl=prctl))
    monkeypatch.setattr(supervisor.signal, "signal", lambda *_: None)
    monkeypatch.setattr(supervisor.os, "getppid", lambda: 123)
    monkeypatch.setattr(supervisor.os, "getpgrp", os.getpid)
    monkeypatch.setattr(supervisor.sys, "argv", ["guardian", "123", str(invocation)])
    monkeypatch.setattr(supervisor.subprocess, "Popen", forbidden_launch)
    with pytest.raises(OSError, match="could not disable") as raised:
        supervisor.main()
    assert raised.value.errno == errno.EPERM
    assert not (tmp_path / "memory-policy.json").exists()
