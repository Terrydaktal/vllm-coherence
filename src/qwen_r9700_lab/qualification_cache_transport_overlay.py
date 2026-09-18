"""Render an authenticated qualification-only fixed-slot cache-root overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import stat
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCHEMA = "urn:qwen-r9700:qualification-cache-transport-overlay:v1"
SHA256_LENGTH = 64


class QualificationCacheOverlayError(RuntimeError):
    """The overlay input or output violated its identity contract."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable_private_file(path: Path, label: str) -> bytes:
    path = path.expanduser().absolute()
    try:
        before = path.lstat()
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise QualificationCacheOverlayError(f"cannot read {label}: {error}") from error
    before_id = before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
    after_id = after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    if (
        before_id != after_id
        or not stat.S_ISREG(before.st_mode)
        or path.is_symlink()
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) & 0o077
    ):
        raise QualificationCacheOverlayError(
            f"{label} must be one stable owned private regular file"
        )
    return payload


def _split_command(payload: bytes) -> tuple[list[str], list[str]]:
    try:
        tokens = shlex.split(payload.decode())
    except (UnicodeDecodeError, ValueError) as error:
        raise QualificationCacheOverlayError(f"base command is invalid: {error}") from error
    if tokens[:3] != ["exec", "/usr/bin/env", "-i"]:
        raise QualificationCacheOverlayError("base command must use exec /usr/bin/env -i")
    index = 3
    environment: list[str] = []
    while index < len(tokens) and "=" in tokens[index]:
        environment.append(tokens[index])
        index += 1
    names = [token.split("=", 1)[0] for token in environment]
    if not environment or index == len(tokens) or len(names) != len(set(names)):
        raise QualificationCacheOverlayError("base command environment is invalid")
    return environment, tokens[index:]


def _site_source(
    *,
    destination: Path,
    chained_site: Path,
    chained_sha256: str,
    chained_pythonpath: str,
    expected_runner_sha256: str,
    expected_scheduler_sha256: str,
    expected_selection_sha256: str,
    isolated_offload_root: Path,
) -> bytes:
    outer_pythonpath = f"{destination}:{chained_pythonpath}"
    body = f'''"""Authenticated qualification-only cache-root transport."""
import hashlib
import os
import re
import runpy
import stat
import sys
from pathlib import Path

_SELF = Path({str(destination / "sitecustomize.py")!r})
_CHAIN = Path({str(chained_site)!r})
_CHAIN_SHA256 = {chained_sha256!r}
_CHAIN_PYTHONPATH = {chained_pythonpath!r}
_OUTER_PYTHONPATH = {outer_pythonpath!r}
_RUNNER = "vllm.v1.worker.gpu.model_runner"
_SCHEDULER = "vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler"
_SELECTION = (
    "vllm.distributed.kv_transfer.kv_connector.v1.offloading."
    "qwen_persistent_selection"
)
_EXPECTED = {{
    _RUNNER: {expected_runner_sha256!r},
    _SCHEDULER: {expected_scheduler_sha256!r},
    _SELECTION: {expected_selection_sha256!r},
}}
_OLD = "/home/lewis/.local/share/qwen-r9700/kv-cache/dflash-agent262-hauhau-delta-autoround-v1"
_NEW = {str(isolated_offload_root)!r}

def _digest(path):
    before = path.lstat()
    payload = path.read_bytes()
    after = path.lstat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if (not stat.S_ISREG(before.st_mode) or path.is_symlink()
            or before.st_uid != os.getuid() or identity(before) != identity(after)):
        raise RuntimeError("qualification cache overlay identity is unsafe")
    return hashlib.sha256(payload).hexdigest()

def _rewrite(payload, module_name):
    if hashlib.sha256(payload).hexdigest() != _EXPECTED[module_name]:
        raise RuntimeError(f"qualification cache {{module_name}} payload differs")
    split = re.compile(
        rb'"/home/lewis/\\.local/share/qwen-r9700/kv-cache/"\\s*'
        rb'"dflash-agent262-hauhau-delta-autoround-v1"'
    )
    replacement = repr(_NEW).encode()
    rewritten, split_count = split.subn(replacement, payload)
    quoted = re.compile(b"[\\\"']" + re.escape(_OLD.encode()) + b"[\\\"']")
    rewritten, full_count = quoted.subn(replacement, rewritten)
    expected_count = 2 if module_name.endswith("model_runner") else 1
    if split_count + full_count != expected_count or _OLD.encode() in rewritten:
        raise RuntimeError(f"qualification cache {{module_name}} replacement count differs")
    compile(rewritten, module_name, "exec")
    return rewritten

if os.environ.get("PYTHONPATH") != _OUTER_PYTHONPATH:
    raise RuntimeError("qualification cache overlay PYTHONPATH differs")
if os.environ.get("QWEN_QUALIFICATION_CACHE_TRANSPORT") != "1":
    raise RuntimeError("qualification cache overlay is not enabled")
if os.environ.get("QWEN_QUALIFICATION_OFFLOAD_ROOT") != _NEW:
    raise RuntimeError("qualification offload root differs")
if _digest(_SELF) != os.environ.get("QWEN_QUALIFICATION_CACHE_SITE_SHA256"):
    raise RuntimeError("qualification cache overlay site identity differs")
if _digest(_CHAIN) != _CHAIN_SHA256:
    raise RuntimeError("qualification cache overlay chained site differs")
if any(name in sys.modules for name in _EXPECTED):
    raise RuntimeError("qualification cache module imported before transport binding")
os.environ["PYTHONPATH"] = _CHAIN_PYTHONPATH
runpy.run_path(str(_CHAIN), run_name="_qwen_qualification_cache_chained_site")
os.environ["PYTHONPATH"] = _OUTER_PYTHONPATH
if any(name in sys.modules for name in _EXPECTED):
    raise RuntimeError("qualification cache module imported during chained bootstrap")

remaining = set(_EXPECTED)
for finder in sys.meta_path:
    payload = getattr(finder, "_payload", None)
    if isinstance(payload, bytes) and hashlib.sha256(payload).hexdigest() == _EXPECTED[_RUNNER]:
        finder._payload = _rewrite(payload, _RUNNER)
        remaining.discard(_RUNNER)
    records = getattr(finder, "_records", None)
    if not isinstance(records, dict):
        continue
    for module_name in tuple(remaining):
        record = records.get(module_name)
        if not isinstance(record, tuple) or not isinstance(record[-1], bytes):
            continue
        records[module_name] = (*record[:-1], _rewrite(record[-1], module_name))
        remaining.discard(module_name)
if remaining:
    raise RuntimeError(f"qualification cache binders are missing: {{sorted(remaining)}}")
print(f"[qwen-qualification-cache] authenticated isolated offload root={{_NEW}}", flush=True)
'''
    source = (
        '"""Authenticated qualification-only cache-root transport."""\n\n'
        "def _bootstrap():\n"
        + textwrap.indent(body, "    ")
        + "\ntry:\n"
        + "    _bootstrap()\n"
        + "except BaseException as _bootstrap_error:\n"
        + "    raise SystemExit(78) from _bootstrap_error\n"
    )
    return source.encode()


def render(args: argparse.Namespace) -> dict[str, Any]:
    destination = args.destination.expanduser().absolute()
    offload_root = args.isolated_offload_root.expanduser().absolute()
    if destination.exists():
        raise QualificationCacheOverlayError("destination is create-only")
    if not offload_root.is_absolute() or offload_root == Path("/") or ".." in offload_root.parts:
        raise QualificationCacheOverlayError("isolated offload root must be bounded and absolute")
    for value in (
        args.expected_runner_sha256,
        args.expected_scheduler_sha256,
        args.expected_selection_sha256,
    ):
        if len(value) != SHA256_LENGTH or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise QualificationCacheOverlayError("expected module SHA256 is invalid")
    base = _stable_private_file(args.base_command, "base command")
    if _sha256(base) != args.expected_base_sha256:
        raise QualificationCacheOverlayError("base command SHA256 mismatch")
    environment, argv = _split_command(base)
    env = dict(token.split("=", 1) for token in environment)
    chained_pythonpath = env.get("PYTHONPATH", "")
    if not chained_pythonpath.startswith("/"):
        raise QualificationCacheOverlayError("base command lacks an absolute PYTHONPATH")
    chained_site = Path(chained_pythonpath.split(":", 1)[0]) / "sitecustomize.py"
    chained_payload = _stable_private_file(chained_site, "chained sitecustomize")

    staging = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
    if staging.exists():
        raise QualificationCacheOverlayError("staging destination already exists")
    staging.mkdir(mode=0o700, parents=True)
    try:
        site_payload = _site_source(
            destination=destination,
            chained_site=chained_site,
            chained_sha256=_sha256(chained_payload),
            chained_pythonpath=chained_pythonpath,
            expected_runner_sha256=args.expected_runner_sha256,
            expected_scheduler_sha256=args.expected_scheduler_sha256,
            expected_selection_sha256=args.expected_selection_sha256,
            isolated_offload_root=offload_root,
        )
        site_path = staging / "sitecustomize.py"
        site_path.write_bytes(site_payload)
        site_path.chmod(0o600)
        site_sha256 = _sha256(site_payload)
        additions = {
            "PYTHONPATH": f"{destination}:{chained_pythonpath}",
            "QWEN_QUALIFICATION_CACHE_TRANSPORT": "1",
            "QWEN_QUALIFICATION_CACHE_SITE_SHA256": site_sha256,
            "QWEN_QUALIFICATION_OFFLOAD_ROOT": str(offload_root),
        }
        rewritten = [token for token in environment if token.split("=", 1)[0] not in additions]
        rewritten.extend(f"{name}={value}" for name, value in additions.items())
        command_payload = (
            shlex.join(["exec", "/usr/bin/env", "-i", *rewritten, *argv]) + "\n"
        ).encode()
        command_path = staging / "command.sh"
        command_path.write_bytes(command_payload)
        command_path.chmod(0o700)
        manifest = {
            "schema": SCHEMA,
            "base_command_sha256": args.expected_base_sha256,
            "chained_site": str(chained_site),
            "chained_site_sha256": _sha256(chained_payload),
            "command_sha256": _sha256(command_payload),
            "expected_module_sha256": {
                "runner": args.expected_runner_sha256,
                "scheduler": args.expected_scheduler_sha256,
                "selection": args.expected_selection_sha256,
            },
            "isolated_offload_root": str(offload_root),
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
    parser = argparse.ArgumentParser(prog="qwen-qualification-cache-transport-overlay")
    parser.add_argument("--base-command", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-scheduler-sha256", required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--isolated-offload-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = render(_parser().parse_args(argv))
    except (QualificationCacheOverlayError, OSError, ValueError) as error:
        print(f"qwen-qualification-cache-transport-overlay: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
