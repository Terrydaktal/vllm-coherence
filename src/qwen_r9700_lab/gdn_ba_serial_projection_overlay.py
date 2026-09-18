"""Render a create-only arm that bypasses only the disproven M8 B/A pair overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import stat
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCHEMA = "urn:qwen-r9700:gdn-ba-serial-projection-overlay:v1"


class SerialProjectionOverlayError(RuntimeError):
    """The base command or create-only overlay violated its identity contract."""


def _stable_private_file(path: Path, label: str) -> bytes:
    path = path.expanduser().absolute()
    try:
        before = path.lstat()
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise SerialProjectionOverlayError(f"cannot read {label}: {error}") from error
    def identity(value: os.stat_result) -> tuple[int, int, int, int]:
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns

    if (
        identity(before) != identity(after)
        or not stat.S_ISREG(before.st_mode)
        or path.is_symlink()
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) & 0o077
    ):
        raise SerialProjectionOverlayError(f"{label} must be one stable owned private file")
    return payload


def _split_command(payload: bytes) -> tuple[list[str], list[str]]:
    try:
        tokens = shlex.split(payload.decode())
    except (UnicodeDecodeError, ValueError) as error:
        raise SerialProjectionOverlayError(f"base command is invalid: {error}") from error
    if tokens[:3] != ["exec", "/usr/bin/env", "-i"]:
        raise SerialProjectionOverlayError("base command must use exec /usr/bin/env -i")
    index = 3
    environment: list[str] = []
    while index < len(tokens) and "=" in tokens[index]:
        environment.append(tokens[index])
        index += 1
    names = [token.split("=", 1)[0] for token in environment]
    if not environment or index == len(tokens) or len(names) != len(set(names)):
        raise SerialProjectionOverlayError("base command environment/argv is invalid")
    return environment, tokens[index:]


def _site_source(
    destination: Path,
    chained_site: Path,
    chained_sha256: str,
    chained_pythonpath: str,
) -> bytes:
    outer_pythonpath = f"{destination}:{chained_pythonpath}"
    return f'''"""Authenticated single-unit serial B/A projection diagnostic."""
import hashlib
import os
import runpy
import stat
import sys
from pathlib import Path

_ROOT = Path({str(destination)!r})
_SELF = _ROOT / "sitecustomize.py"
_CHAIN = Path({str(chained_site)!r})
_CHAIN_SHA256 = {chained_sha256!r}
_CHAIN_PYTHONPATH = {chained_pythonpath!r}
_OUTER_PYTHONPATH = {outer_pythonpath!r}
_TARGET = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"

def _stable_digest(path, label):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink() or before.st_uid != os.getuid():
        raise RuntimeError(f"{{label}} identity is unsafe")
    payload = path.read_bytes()
    after = path.lstat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if identity(before) != identity(after):
        raise RuntimeError(f"{{label}} changed during authentication")
    return hashlib.sha256(payload).hexdigest()

if (
    os.environ.get("QWEN_GDN_BA_SERIAL_PROJECTION_DIAGNOSTIC") != "1"
    or os.environ.get("QWEN_GDN_BA_GROUPED_PREFIX_EXACT") != "0"
    or os.environ.get("QWEN_GDN_BA_BATCHED_EXACT") != "0"
    or os.environ.get("QWEN_GDN_BA_SERIAL_ROW_EXACT") != "1"
):
    raise RuntimeError("serial B/A projection diagnostic identity is absent")
if os.environ.get("PYTHONPATH") != _OUTER_PYTHONPATH:
    raise RuntimeError("serial B/A projection diagnostic PYTHONPATH differs")
if _stable_digest(_SELF, "serial B/A site") != os.environ.get(
    "QWEN_GDN_BA_SERIAL_PROJECTION_SITE_SHA256"
):
    raise RuntimeError("serial B/A projection site SHA256 mismatch")
if _stable_digest(_CHAIN, "chained site") != _CHAIN_SHA256:
    raise RuntimeError("serial B/A projection chained-site SHA256 mismatch")
if _TARGET in sys.modules:
    raise RuntimeError("GDN module was imported before pair-overlay isolation")
os.environ["PYTHONPATH"] = _CHAIN_PYTHONPATH
runpy.run_path(str(_CHAIN), run_name="_qwen_gdn_ba_serial_projection_chained_site")
if os.environ.get("PYTHONPATH") != _CHAIN_PYTHONPATH:
    raise RuntimeError("chained site changed PYTHONPATH")
os.environ["PYTHONPATH"] = _OUTER_PYTHONPATH
import combined_runtime_patch as retained
if retained._Finder._PATCHES.get(_TARGET) is not retained._patch_gdn:
    raise RuntimeError("retained GDN pair-overlay hook identity changed")

def _preserve_original_project_ba(module):
    print("[qwen-gdn-ba-serial-projection] retained M8 pair-out bypassed", flush=True)

retained._Finder._PATCHES[_TARGET] = _preserve_original_project_ba
'''.encode()


def render(args: argparse.Namespace) -> dict[str, Any]:
    destination = args.destination.expanduser().absolute()
    if destination.exists():
        raise SerialProjectionOverlayError("destination is create-only")
    base = _stable_private_file(args.base_command, "base command")
    base_sha256 = hashlib.sha256(base).hexdigest()
    if base_sha256 != args.expected_base_sha256:
        raise SerialProjectionOverlayError("base command SHA256 mismatch")
    environment, argv = _split_command(base)
    env = dict(token.split("=", 1) for token in environment)
    expected = {
        "QWEN_D7_RETAINED_MICROS": "1",
        "QWEN_GDN_BA_BATCHED_EXACT": "0",
        "QWEN_GDN_BA_GROUPED_PREFIX_EXACT": "0",
        "QWEN_GDN_BA_SERIAL_ROW_EXACT": "1",
    }
    if {name: env.get(name) for name in expected} != expected:
        raise SerialProjectionOverlayError("base command is not the serial-B/A retained-micro arm")
    chained_pythonpath = env.get("PYTHONPATH", "")
    if not chained_pythonpath.startswith("/"):
        raise SerialProjectionOverlayError("base command lacks an absolute PYTHONPATH")
    chained_site = Path(chained_pythonpath.split(":", 1)[0]) / "sitecustomize.py"
    chained_payload = _stable_private_file(chained_site, "chained sitecustomize")

    staging = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
    if staging.exists():
        raise SerialProjectionOverlayError("staging destination already exists")
    staging.mkdir(mode=0o700, parents=True)
    try:
        chained_sha256 = hashlib.sha256(chained_payload).hexdigest()
        site_payload = _site_source(
            destination, chained_site, chained_sha256, chained_pythonpath
        )
        site_path = staging / "sitecustomize.py"
        site_path.write_bytes(site_payload)
        site_path.chmod(0o600)
        site_sha256 = hashlib.sha256(site_payload).hexdigest()
        additions = {
            "PYTHONPATH": f"{destination}:{chained_pythonpath}",
            "QWEN_GDN_BA_SERIAL_PROJECTION_DIAGNOSTIC": "1",
            "QWEN_GDN_BA_SERIAL_PROJECTION_SITE_SHA256": site_sha256,
        }
        rewritten = [token for token in environment if token.split("=", 1)[0] not in additions]
        rewritten.extend(f"{name}={value}" for name, value in additions.items())
        command = shlex.join(["exec", "/usr/bin/env", "-i", *rewritten, *argv]) + "\n"
        command_path = staging / "command.sh"
        command_path.write_text(command)
        command_path.chmod(0o700)
        manifest = {
            "schema": SCHEMA,
            "base_command_sha256": base_sha256,
            "chained_site": str(chained_site),
            "chained_site_sha256": chained_sha256,
            "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
            "isolated_change": "bypass retained M8 pair-out; use serial-row B/A projection",
            "site_sha256": site_sha256,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifest_path.chmod(0o600)
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        staging.rename(destination)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen-gdn-ba-serial-projection-overlay")
    parser.add_argument("--base-command", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        print(json.dumps(render(_parser().parse_args(argv)), indent=2, sort_keys=True))
        return 0
    except SerialProjectionOverlayError as error:
        print(f"qwen-gdn-ba-serial-projection-overlay: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
