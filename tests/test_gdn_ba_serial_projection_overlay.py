from __future__ import annotations

import argparse
import hashlib
import shlex
from pathlib import Path

import pytest

from qwen_r9700_lab.gdn_ba_serial_projection_overlay import (
    SerialProjectionOverlayError,
    render,
)


def _base(tmp_path: Path, *, grouped: str = "0") -> Path:
    chain = tmp_path / "chain"
    chain.mkdir(mode=0o700)
    (chain / "sitecustomize.py").write_text("# pinned chain\n")
    (chain / "sitecustomize.py").chmod(0o600)
    command = tmp_path / "base.sh"
    command.write_text(
        shlex.join(
            [
                "exec",
                "/usr/bin/env",
                "-i",
                f"PYTHONPATH={chain}",
                "QWEN_D7_RETAINED_MICROS=1",
                "QWEN_GDN_BA_BATCHED_EXACT=0",
                f"QWEN_GDN_BA_GROUPED_PREFIX_EXACT={grouped}",
                "QWEN_GDN_BA_SERIAL_ROW_EXACT=1",
                "/bin/model",
                "/weights",
            ]
        )
        + "\n"
    )
    command.chmod(0o600)
    return command


def test_render_binds_chain_and_changes_only_bootstrap(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    base = _base(tmp_path)
    destination = tmp_path / "overlay"
    manifest = render(
        argparse.Namespace(
            base_command=base,
            destination=destination,
            expected_base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
        )
    )
    assert manifest["isolated_change"].startswith("bypass retained M8 pair-out")
    command = shlex.split((destination / "command.sh").read_text())
    executable = command.index("/bin/model")
    environment = dict(token.split("=", 1) for token in command[3:executable])
    assert environment["PYTHONPATH"].startswith(f"{destination}:")
    assert environment["QWEN_GDN_BA_GROUPED_PREFIX_EXACT"] == "0"
    assert environment["QWEN_GDN_BA_SERIAL_PROJECTION_DIAGNOSTIC"] == "1"
    site = (destination / "sitecustomize.py").read_text()
    assert "retained._Finder._PATCHES[_TARGET] = _preserve_original_project_ba" in site
    assert command[executable:] == ["/bin/model", "/weights"]


def test_render_rejects_grouped_base_and_is_create_only(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    base = _base(tmp_path, grouped="1")
    args = argparse.Namespace(
        base_command=base,
        destination=tmp_path / "overlay",
        expected_base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
    )
    with pytest.raises(SerialProjectionOverlayError, match="not the serial-B/A"):
        render(args)
    args.destination.mkdir()
    with pytest.raises(SerialProjectionOverlayError, match="create-only"):
        render(args)
