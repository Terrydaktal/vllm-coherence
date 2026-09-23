"""Exercise the actual warmup shell against container-journal and restart faults."""

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/pi-remote-qwen-radiance"
CONTAINER_ID = "a" * 64


def executable(path, source):
    path.write_text(source)
    path.chmod(0o700)


def run_warmup(tmp_path, mode):
    source = LAUNCHER.read_text()
    start = source.index('if [[ ${QWEN_PI_SKIP_WARMUP:-0} != 1 ]]; then')
    end = source.index("\nprintf 'Radiance unrestricted backend ready;", start)
    script = tmp_path / "warmup.sh"
    script.write_text(
        "set -euo pipefail\n"
        "remote_host=fake\nlocal_port=1\nreuse_existing=1\n"
        "CONTAINER=reused-backend\nbackend_log=existing\n"
        'die() { printf "%s\\n" "$*" >&2; exit 2; }\n'
        + source[start:end]
        + '\nprintf "PI_ADMITTED\\n"\n'
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    executable(
        fake_bin / "ssh",
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        '[[ $1 == -T && $2 == fake ]]\nshift 2\nexec "$@"\n',
    )
    executable(fake_bin / "sleep", "#!/usr/bin/env bash\nexit 0\n")
    executable(
        fake_bin / "podman",
        """#!/usr/bin/env python3
import datetime,json,os,pathlib,sys
p=pathlib.Path(os.environ['WARMUP_TEST_STATE'])
mode=os.environ['WARMUP_TEST_MODE']
args=sys.argv[1:]
calls=int((p/'calls').read_text()) if (p/'calls').exists() else 0
identity='a'*64
with (p/'podman-commands').open('a') as f:f.write(json.dumps(args)+'\\n')
if args[0]=='inspect':
    if args[2]=='{{.Id}}':
        assert args[3]=='reused-backend'
        print(identity)
    else:
        assert args[2]=='{{.State.Running}}' and args[3]==identity
        print('false' if mode=='restart' and calls else 'true')
elif args[0]=='logs':
    assert args[-1]==identity
    if mode=='unreadable' or (mode=='unreadable_after' and calls):sys.exit(125)
    if '--tail' in args:sys.exit(0)
    assert args[1]=='--since'
    since=datetime.datetime.fromisoformat(args[2].replace('Z','+00:00')).timestamp()
    for line in (p/'journal').read_text().splitlines():
        row=json.loads(line)
        if row['time']>=since:print(row['message'],file=sys.stderr)
else:raise AssertionError(args)
""",
    )
    executable(
        fake_bin / "curl",
        """#!/usr/bin/env python3
import json,os,pathlib,sys,time
p=pathlib.Path(os.environ['WARMUP_TEST_STATE'])
mode=os.environ['WARMUP_TEST_MODE']
calls=int((p/'calls').read_text())+1 if (p/'calls').exists() else 1
(p/'calls').write_text(str(calls))
if mode=='http_error':sys.exit(22)
if mode=='jit_always' or (mode=='jit_once' and calls==1):
    with (p/'journal').open('a') as f:
        f.write(json.dumps({'time':time.time(),'message':'JIT compilation during inference: test_kernel'})+'\\n')
""",
    )
    (tmp_path / "journal").write_text(
        json.dumps({"time": 1, "message": "JIT compilation during inference: historical"})
        + "\n"
    )
    env = dict(os.environ)
    env.update(
        PATH=str(fake_bin) + os.pathsep + env["PATH"],
        WARMUP_TEST_STATE=str(tmp_path),
        WARMUP_TEST_MODE=mode,
        QWEN_PI_SKIP_WARMUP="0",
    )
    result = subprocess.run(
        ["bash", str(script)], env=env, text=True, capture_output=True, timeout=15
    )
    calls = int((tmp_path / "calls").read_text()) if (tmp_path / "calls").exists() else 0
    return result, calls


@pytest.mark.parametrize("mode,expected_calls", [("clean", 1), ("jit_once", 2)])
def test_reuse_reads_container_journal_and_ignores_old_warnings(tmp_path, mode, expected_calls):
    result, calls = run_warmup(tmp_path, mode)
    assert result.returncode == 0, result.stderr
    assert "PI_ADMITTED" in result.stdout
    assert calls == expected_calls
    assert "historical" not in result.stderr
    commands = [json.loads(line) for line in (tmp_path / "podman-commands").read_text().splitlines()]
    assert all(cmd[-1] == CONTAINER_ID for cmd in commands if cmd[0] == "logs")


@pytest.mark.parametrize(
    "mode,expected_calls,error",
    [
        ("unreadable", 0, "could not read the remote backend log"),
        ("unreadable_after", 1, "could not inspect the remote backend log"),
        ("restart", 1, "could not inspect the remote backend log"),
        ("http_error", 1, "startup warmup failed"),
        ("jit_always", 3, "after three attempts"),
    ],
)
def test_warmup_does_not_admit_pi_when_verification_fails(tmp_path, mode, expected_calls, error):
    result, calls = run_warmup(tmp_path, mode)
    assert result.returncode != 0
    assert "PI_ADMITTED" not in result.stdout
    assert error in result.stderr
    assert calls == expected_calls
