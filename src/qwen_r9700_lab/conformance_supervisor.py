"""Linux process guardian for an owned qualification job, including nested jobs.

If its controller dies, the guardian kills its own session's process group.
Nested guardians receive the same parent-death signal when their controllers
die. No service names, global pkill, GPU API or external endpoint is involved.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import write_private


def disable_transparent_hugepages(libc):
    """Avoid the observed THP-compaction path in this owned process tree.

    Linux inherits this policy through both fork and exec. This is a scoped
    mitigation for the recorded AMD SVM stall, not a repair of the driver or
    a guarantee that no other allocation can trigger memory compaction.
    """
    # PR_SET_THP_DISABLE / PR_GET_THP_DISABLE. Pass full-width varargs.
    zero = ctypes.c_ulong(0)
    if libc.prctl(41, ctypes.c_ulong(1), zero, zero, zero) != 0:
        raise OSError(ctypes.get_errno(), "could not disable qualification huge pages")
    observed = libc.prctl(42, zero, zero, zero, zero)
    if observed != 1:
        raise RuntimeError("qualification huge-page policy was not confirmed disabled")
    return {
        "schema": "urn:qwen:qualification-memory-policy:v1",
        "transparent_hugepages": "disabled",
        "pr_get_thp_disable": observed,
        "scope": "owned_process_tree",
        "host_global_policy_changed": False,
    }


def main():
    expected_parent = int(sys.argv[1])
    invocation = json.loads(Path(sys.argv[2]).read_text())
    child = None
    stopping = False

    def parent_died(*_):
        os.killpg(os.getpgrp(), signal.SIGKILL)

    def stop(*_):
        nonlocal stopping
        stopping = True
        if child is not None:
            with contextlib.suppress(ProcessLookupError):
                child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGUSR1, parent_died)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    libc = ctypes.CDLL(None, use_errno=True)
    # PR_SET_PDEATHSIG. The second parent check closes the fork/registration race.
    if libc.prctl(1, int(signal.SIGUSR1), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "could not arm owned-process guardian")
    if os.getppid() != expected_parent or os.getpgrp() != os.getpid():
        raise RuntimeError("qualification guardian lost its owning process/session")
    policy = disable_transparent_hugepages(libc)
    write_private(Path(sys.argv[2]).parent / "memory-policy.json", policy)
    child = subprocess.Popen(invocation["argv"], stdin=subprocess.DEVNULL)
    if stopping:
        stop()
    code = child.wait()
    # Remove any orphaned worker still in OUR session before reporting the
    # command's result. A nested guardian covers workers in a nested session.
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        pid = int(entry.name)
        try:
            if os.getpgid(pid) != os.getpgrp():
                continue
            descriptor = os.pidfd_open(pid)
        except (ProcessLookupError, PermissionError):
            continue
        try:
            # Recheck after opening the pidfd; a vanished PID may have been
            # reused by an unrelated process between the first check and open.
            if os.getpgid(pid) == os.getpgrp():
                signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        except ProcessLookupError:
            pass
        finally:
            os.close(descriptor)
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    raise SystemExit(main())
