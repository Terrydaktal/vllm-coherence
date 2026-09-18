"""Real CPU children exercise delayed durable writes and forced-stop evidence."""

import os
import sys
import time

import pytest

from qwen_r9700_lab.conformance_transport import OwnedProcess
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, private_json

CHILD = r"""
import os, signal, sys, time
from pathlib import Path
def stop(*_):
    Path('stopping').write_text('yes')
    time.sleep(float(sys.argv[1]))
    with open('durable', 'wb') as out:
        out.write(b'pending tail persisted')
        out.flush()
        os.fsync(out.fileno())
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
Path('ready').write_text('yes')
while True:
    time.sleep(1)
"""


@pytest.mark.parametrize(
    "crash,delay,grace,persisted",
    [(False, 0.3, 3, True), (False, 30, 0.15, False), (True, 0, 3, False)],
)
def test_shutdown_waits_for_flush_or_records_forced_termination(
    tmp_path, crash, delay, grace, persisted
):
    child = OwnedProcess(
        [sys.executable, "-c", CHILD, str(delay)],
        tmp_path / "child",
        env=dict(os.environ),
        timeout=10,
    )
    try:
        deadline = time.monotonic() + 5
        while not (child.root / "ready").exists():
            child.check()
            assert time.monotonic() < deadline
            time.sleep(0.01)
        child.close(crash=crash, grace_seconds=grace)
        receipt = private_json(child.root / "shutdown.json")
        assert receipt["mode"] == ("crash" if crash else "graceful")
        assert receipt["grace_expired"] == (not crash and not persisted)
        assert receipt["cleanup_error"] is None
        assert (child.root / "durable").exists() == persisted
        if persisted:
            assert (child.root / "durable").read_bytes() == b"pending tail persisted"
            assert receipt["returncode"] == 0
        original = (child.root / "shutdown.json").read_bytes()
        child.close()
        assert (child.root / "shutdown.json").read_bytes() == original
    finally:
        child.close(crash=True)


@pytest.mark.parametrize("grace", [0, -1, True, None, "1", float("inf"), float("nan")])
def test_invalid_grace_does_not_signal_or_mark_the_process_closed(tmp_path, grace):
    child = OwnedProcess(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        tmp_path / "child",
        env=dict(os.environ),
        timeout=15,
    )
    try:
        with pytest.raises(DiagnosticError, match="shutdown grace"):
            child.close(grace_seconds=grace)
        assert not child.closed
        assert child.process.poll() is None
    finally:
        child.close(crash=True)
