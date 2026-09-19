"""Telemetry must follow the selected deployment after source reconciliation."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_r9700_lab import radiance_cache_residency as residency
from qwen_r9700_lab import radiance_error_report as errors


@pytest.mark.parametrize("container", ["vllm-coherence", "legacy-radiance-fixture"])
def test_error_probe_selects_requested_container(monkeypatch, capsys, container):
    now = int(time.time() * 1000)
    monkeypatch.setattr(errors, "CONTAINER", errors.CONTAINER)
    monkeypatch.setattr(
        sys, "argv", ["probe", str(now - 1000), str(now), "--container", container]
    )
    selected = []

    def collect(_):
        selected.append(errors.CONTAINER)
        return {
            "incidents": [],
            "backend": {},
            "captured_at": now,
            "lookup_issue": None,
        }

    monkeypatch.setattr(errors, "collect", collect)
    errors.main()
    assert selected == [container]
    assert json.loads(capsys.readouterr().out)["status"] == "unavailable"


def test_namespace_probe_transports_container_and_once(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(residency, "BACKEND_CONTAINER", residency.BACKEND_CONTAINER)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "0.5",
            str(tmp_path),
            "auto",
            "--container",
            "legacy-fixture",
            "--once",
        ],
    )
    monkeypatch.setattr(residency, "optional_json", lambda *_: None)
    monkeypatch.setattr(
        residency, "StatusChanges", lambda: SimpleNamespace(close=lambda: None)
    )
    observed = []

    class Namespace:
        def __init__(self, _):
            observed.append(residency.BACKEND_CONTAINER)

        def read(self):
            return None

    monkeypatch.setattr(residency, "LiveCacheNamespace", Namespace)
    residency.main()
    assert observed == ["legacy-fixture"]
    assert len(capsys.readouterr().out.splitlines()) == 1


@pytest.mark.parametrize("container", ["vllm-coherence", "legacy-radiance-fixture", None])
def test_error_extension_passes_container_to_remote_probe(tmp_path, container):
    root = Path(__file__).resolve().parents[1]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "args.json"
    ssh = fake_bin / "ssh"
    ssh.write_text(
        "#!/usr/bin/env python3\nimport json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['PROBE_ARGS']).write_text(json.dumps(sys.argv))\n"
        "sys.stdin.read()\n"
        "print(json.dumps({'schema':'urn:qwen-r9700:backend-error:v1','status':'unavailable'}))\n"
    )
    ssh.chmod(0o700)
    environment = {
        k: v for k, v in os.environ.items() if not k.startswith("QWEN_RADIANCE_")
    }
    environment.update(PATH=f"{fake_bin}:{os.environ['PATH']}", PROBE_ARGS=str(log))
    if container is not None:
        environment["QWEN_RADIANCE_CONTAINER"] = container
    result = subprocess.run(
        [
            "node",
            "--input-type=module",
            "-e",
            (
                "import { fetchBackendError } from './integrations/pi/qwen-radiance-errors.mjs';"
                "await fetchBackendError({since:1000,until:2000,latest:true,host:'fixture.invalid'});"
            ),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    args = json.loads(log.read_text())
    expected = container or "qwen38-27b-uncensored-mxfp4-public-snapshot-candidate"
    assert args[-3:] == ["--container", expected, "--latest"]
