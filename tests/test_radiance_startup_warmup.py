"""Exercise the actual warmup shell against container-journal and restart faults."""

import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/pi-remote-qwen-radiance"
CONTAINER_ID = "a" * 64


def executable(path, source):
    path.write_text(source)
    path.chmod(0o700)


def receipt_path(tmp_path):
    name = hashlib.sha256(b"reused-backend").hexdigest() + ".json"
    return tmp_path / "cache/startup-warmup" / name


def run_warmup(
    tmp_path,
    mode,
    *,
    identity=None,
    model="synthetic-model",
    abi="synthetic-abi",
    skip=False,
    payload_tokens=96,
    policy="pi-startup-warmup-v1",
):
    (tmp_path / "cache").mkdir(mode=0o700, exist_ok=True)
    source = LAUNCHER.read_text()
    source = source.replace('"max_tokens":96', f'"max_tokens":{payload_tokens}')
    source = source.replace('policy = "pi-startup-warmup-v1"', f"policy = {policy!r}")
    start = source.index("if [[ ${QWEN_PI_SKIP_WARMUP:-0} != 1 ]]; then")
    end = source.index("\nprintf 'Radiance unrestricted backend ready;", start)
    script = tmp_path / "warmup.sh"
    script.write_text(
        "set -euo pipefail\n"
        "remote_host=fake\nlocal_port=1\nreuse_existing=1\n"
        "CONTAINER=reused-backend\nbackend_log=existing\n"
        f"REMOTE_CACHE={shlex.quote(str(tmp_path / 'cache'))}\n"
        f"SNAPSHOT_ABI={shlex.quote(abi)}\nMODEL_ID={shlex.quote(model)}\n"
        'die() { printf "%s\\n" "$*" >&2; exit 2; }\n'
        + source[start:end]
        + '\nprintf "PI_ADMITTED\\n"\n'
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
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
config=json.loads((p/'identity.json').read_text())
identity=config.get('id','a'*64)
hostpid=int(os.environ['WARMUP_TEST_HOST_PID'])
workerpid=config.get('worker_pid',int(os.environ['WARMUP_TEST_PARENT_PID']))
if mode=='worker_change' and calls:workerpid=hostpid
with (p/'podman-commands').open('a') as f:f.write(json.dumps(args)+'\\n')
if args[0]=='inspect':
    if len(args)==2:
        assert args[1] in ('reused-backend',identity)
        print(json.dumps([{'Id':identity,'Image':config.get('image','synthetic-image'),
            'Path':config.get('path','/opt/vllm/bin/python'),
            'Args':config.get('args',['bootstrap.py','--model','synthetic']),
            'Config':{'Cmd':config.get('cmd',['bootstrap.py']),
                'Entrypoint':config.get('entrypoint',['/opt/vllm/bin/python']),
                'Env':config.get('env',['SYNTHETIC=1'])},
            'State':{'Running':not(mode=='restart' and calls),
                'StartedAt':config.get('started_at','2026-10-09T10:00:00.000000000Z'),
                'Pid':config.get('init_pid',hostpid)}}]))
    elif args[2]=='{{.Id}}':
        assert args[3]=='reused-backend'
        print(identity)
    else:
        assert args[2]=='{{.State.Running}}' and args[3]==identity
        print('false' if mode=='restart' and calls else 'true')
elif args[0]=='top':
    assert args[1]==identity
    assert args[2:]==['hpid','pid','comm']
    print('HPID PID COMMAND')
    print(f"{config.get('engine_pid',hostpid)} 2 VLLM::EngineCore")
    print(f'{workerpid} 3 VLLM::Worker')
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
    if not (tmp_path / "journal").exists():
        (tmp_path / "journal").write_text(
            json.dumps(
                {"time": 1, "message": "JIT compilation during inference: historical"}
            )
            + "\n"
        )
    (tmp_path / "identity.json").write_text(json.dumps(identity or {}))
    env = dict(os.environ)
    env.update(
        PATH=str(fake_bin) + os.pathsep + env["PATH"],
        WARMUP_TEST_STATE=str(tmp_path),
        WARMUP_TEST_MODE=mode,
        WARMUP_TEST_HOST_PID=str(os.getpid()),
        WARMUP_TEST_PARENT_PID=str(os.getppid()),
        QWEN_PI_SKIP_WARMUP="1" if skip else "0",
    )
    result = subprocess.run(
        ["bash", str(script)],
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    calls = (
        int((tmp_path / "calls").read_text()) if (tmp_path / "calls").exists() else 0
    )
    return result, calls


@pytest.mark.parametrize("mode,expected_calls", [("clean", 1), ("jit_once", 2)])
def test_reuse_reads_container_journal_and_ignores_old_warnings(
    tmp_path, mode, expected_calls
):
    result, calls = run_warmup(tmp_path, mode)
    assert result.returncode == 0, result.stderr
    assert "PI_ADMITTED" in result.stdout
    assert calls == expected_calls
    assert receipt_path(tmp_path).is_file()
    assert "historical" not in result.stderr
    commands = [
        json.loads(line)
        for line in (tmp_path / "podman-commands").read_text().splitlines()
    ]
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
def test_warmup_does_not_admit_pi_when_verification_fails(
    tmp_path, mode, expected_calls, error
):
    result, calls = run_warmup(tmp_path, mode)
    assert result.returncode != 0
    assert "PI_ADMITTED" not in result.stdout
    assert error in result.stderr
    assert calls == expected_calls
    assert not receipt_path(tmp_path).exists(), (
        "an unsuccessful warmup cannot be reused"
    )


def test_second_launch_reuses_clean_warmup_without_another_completion_or_jit_poll(
    tmp_path,
):
    first, calls = run_warmup(tmp_path, "clean")
    assert first.returncode == 0, first.stderr
    assert calls == 1
    marker = receipt_path(tmp_path)
    original = marker.read_bytes()
    previous_commands = (tmp_path / "podman-commands").read_text().splitlines()
    second, calls = run_warmup(tmp_path, "clean")
    assert second.returncode == 0, second.stderr
    assert "PI_ADMITTED" in second.stdout
    assert calls == 1, (
        "another Pi launch must not submit a warmup to an unchanged backend"
    )
    commands = [
        json.loads(line)
        for line in (tmp_path / "podman-commands")
        .read_text()
        .splitlines()[len(previous_commands) :]
    ]
    assert not any(command[0] == "logs" for command in commands), (
        "receipt hits also skip the JIT polling delay"
    )
    assert marker.read_bytes() == original
    assert marker.stat().st_mode & 0o777 == 0o600
    assert marker.parent.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize(
    "identity",
    [
        {"id": "b" * 64},
        {"started_at": "2026-10-09T11:00:00.000000000Z"},
        {"init_pid": os.getppid()},
        {"engine_pid": os.getppid()},
        {"worker_pid": os.getpid()},
        {"image": "another-image"},
        {"path": "/another/python"},
        {"args": ["bootstrap.py", "--different-option"]},
        {"cmd": ["another-bootstrap.py"]},
        {"entrypoint": ["/another/python"]},
        {"env": ["SYNTHETIC=2"]},
    ],
)
def test_backend_process_and_configuration_changes_invalidate_receipt(
    tmp_path, identity
):
    first, calls = run_warmup(tmp_path, "clean")
    assert first.returncode == 0, first.stderr
    original = receipt_path(tmp_path).read_bytes()
    second, calls = run_warmup(tmp_path, "clean", identity=identity)
    assert second.returncode == 0, second.stderr
    assert calls == 2
    assert receipt_path(tmp_path).read_bytes() != original
    third, calls = run_warmup(tmp_path, "clean", identity=identity)
    assert third.returncode == 0, third.stderr
    assert calls == 2, "the replacement backend/configuration is warmed only once"


@pytest.mark.parametrize(
    "changed",
    [
        {"model": "different-model"},
        {"abi": "different-abi"},
        {"payload_tokens": 95},
        {"policy": "pi-startup-warmup-v2"},
    ],
)
def test_warmup_contract_changes_invalidate_receipt(tmp_path, changed):
    first, calls = run_warmup(tmp_path, "clean")
    assert first.returncode == 0, first.stderr
    second, calls = run_warmup(tmp_path, "clean", **changed)
    assert second.returncode == 0, second.stderr
    assert calls == 2


def test_worker_replacement_during_warmup_is_not_published_or_admitted(tmp_path):
    result, calls = run_warmup(tmp_path, "worker_change")
    assert result.returncode != 0
    assert "PI_ADMITTED" not in result.stdout
    assert calls == 1
    assert not receipt_path(tmp_path).exists()


def test_explicitly_skipped_warmup_never_creates_a_success_receipt(tmp_path):
    result, calls = run_warmup(tmp_path, "clean", skip=True)
    assert result.returncode == 0, result.stderr
    assert "PI_ADMITTED" in result.stdout
    assert calls == 0
    assert not receipt_path(tmp_path).exists()
    result, calls = run_warmup(tmp_path, "clean")
    assert result.returncode == 0, result.stderr
    assert calls == 1


@pytest.mark.parametrize(
    "fault", ["corrupt", "symlink", "hardlink", "permissions", "oversized"]
)
def test_malformed_or_unsafe_receipt_cannot_skip_warmup(tmp_path, fault):
    first, calls = run_warmup(tmp_path, "clean")
    assert first.returncode == 0, first.stderr
    marker = receipt_path(tmp_path)
    if fault == "corrupt":
        marker.write_text("{not-valid-json")
    elif fault == "symlink":
        target = tmp_path / "unrelated-file"
        target.write_text("preserve this file")
        marker.unlink()
        marker.symlink_to(target)
    elif fault == "hardlink":
        os.link(marker, tmp_path / "receipt-hardlink")
    elif fault == "permissions":
        marker.chmod(0o644)
    else:
        marker.write_bytes(b"x" * 65_537)
    second, calls = run_warmup(tmp_path, "clean")
    assert second.returncode == 0, second.stderr
    assert calls == 2
    assert not marker.is_symlink()
    assert marker.stat().st_nlink == 1
    assert marker.stat().st_mode & 0o777 == 0o600
    if fault == "symlink":
        assert target.read_text() == "preserve this file"


@pytest.mark.parametrize("fault", ["permissions", "symlink"])
def test_unsafe_receipt_directory_refuses_startup(tmp_path, fault):
    directory = receipt_path(tmp_path).parent
    directory.parent.mkdir(parents=True)
    if fault == "permissions":
        directory.mkdir(mode=0o755)
    else:
        target = tmp_path / "unrelated-directory"
        target.mkdir(mode=0o700)
        directory.symlink_to(target, target_is_directory=True)
    result, calls = run_warmup(tmp_path, "clean")
    assert result.returncode != 0
    assert "PI_ADMITTED" not in result.stdout
    assert calls == 0
