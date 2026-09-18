"""Render a create-only scheduler shadow that preserves prefix/chunk alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "urn:qwen-r9700:fixed-slot-ordinary-prefix-alignment:v1"
SCHEDULER_SHA256 = "34e62b63ef5b13a0a12841a87f73e74ffc9248b0d7bf83328482b17bb9bf1113"
MODULE = "vllm.v1.core.sched.scheduler"
ENV_NAME = "QWEN_FIXED_SLOT_ORDINARY_PREFIX_ALIGNMENT"

OLD = b'''            if replay_boundary is None:
                return num_new_tokens
            end = start + num_new_tokens
            if start < replay_boundary < end:
                return replay_boundary - start
            return num_new_tokens
'''
NEW = b'''            if replay_boundary is not None:
                end = start + num_new_tokens
                if start < replay_boundary < end:
                    return replay_boundary - start
                return num_new_tokens
            # Fixed-slot support is a server capability, not proof that this
            # ordinary request carries a fixed-slot state boundary. Fall through
            # to the normal Mamba/cache-block alignment contract below.
'''


class AlignmentError(RuntimeError):
    """The source, command, or destination violated the overlay contract."""


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def rewrite_scheduler(source: bytes) -> bytes:
    if digest(source) != SCHEDULER_SHA256:
        raise AlignmentError("scheduler source SHA256 differs")
    if source.count(OLD) != 1:
        raise AlignmentError("scheduler alignment preimage is not unique")
    rewritten = source.replace(OLD, NEW)
    if OLD in rewritten or rewritten.count(NEW) != 1:
        raise AlignmentError("scheduler alignment postimage is invalid")
    compile(rewritten, "scheduler.py", "exec")
    return rewritten


def _split_command(payload: bytes) -> tuple[list[str], list[str]]:
    try:
        tokens = shlex.split(payload.decode())
    except (UnicodeDecodeError, ValueError) as error:
        raise AlignmentError(f"base command is invalid: {error}") from error
    if tokens[:3] != ["exec", "/usr/bin/env", "-i"]:
        raise AlignmentError("base command must use exec /usr/bin/env -i")
    index = 3
    environment = []
    while index < len(tokens) and "=" in tokens[index]:
        environment.append(tokens[index])
        index += 1
    if not environment or index == len(tokens):
        raise AlignmentError("base command lacks environment or server argv")
    return environment, tokens[index:]


def _environment_map(environment: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in environment:
        name, value = token.split("=", 1)
        if not name or name in result:
            raise AlignmentError("base command has an invalid environment")
        result[name] = value
    return result


def render_site(
    destination: Path,
    *,
    module_sha256: str,
    chained_site: Path,
    chained_site_sha256: str,
    chained_pythonpath: str,
) -> bytes:
    return f'''"""Authenticated fixed-slot ordinary-prefix alignment shadow."""
import hashlib
import importlib.abc
import importlib.util
import os
import runpy
import sys
from pathlib import Path

_ROOT = Path({str(destination)!r})
_MODULE = {MODULE!r}
_SOURCE = _ROOT / "vllm/v1/core/sched/scheduler.py"
_SOURCE_SHA256 = {module_sha256!r}
_CHAIN = Path({str(chained_site)!r})
_CHAIN_SHA256 = {chained_site_sha256!r}
_CHAIN_PYTHONPATH = {chained_pythonpath!r}
_OUTER_PYTHONPATH = {f"{destination}:{chained_pythonpath}"!r}

def _stable_digest(path):
    before = path.lstat()
    payload = path.read_bytes()
    after = path.lstat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if identity(before) != identity(after) or not path.is_file() or path.is_symlink():
        raise RuntimeError("prefix-alignment file identity changed")
    return hashlib.sha256(payload).hexdigest()

if os.environ.get({ENV_NAME!r}) != "1":
    raise RuntimeError("prefix alignment is not required")
if os.environ.get("PYTHONPATH") != _OUTER_PYTHONPATH:
    raise RuntimeError("prefix-alignment PYTHONPATH mismatch")
if _stable_digest(Path(__file__).resolve()) != os.environ.get(
    "QWEN_FIXED_SLOT_ORDINARY_PREFIX_ALIGNMENT_SITE_SHA256"
):
    raise RuntimeError("prefix-alignment site SHA256 mismatch")
if _stable_digest(_SOURCE) != _SOURCE_SHA256:
    raise RuntimeError("prefix-alignment scheduler SHA256 mismatch")
if _stable_digest(_CHAIN) != _CHAIN_SHA256:
    raise RuntimeError("prefix-alignment chained site SHA256 mismatch")
os.environ["PYTHONPATH"] = _CHAIN_PYTHONPATH
runpy.run_path(str(_CHAIN), run_name="_qwen_prefix_alignment_chained_sitecustomize")
if os.environ.get("PYTHONPATH") != _CHAIN_PYTHONPATH:
    raise RuntimeError("prefix-alignment chained site changed PYTHONPATH")
os.environ["PYTHONPATH"] = _OUTER_PYTHONPATH
if _MODULE in sys.modules:
    raise RuntimeError("scheduler imported before prefix-alignment shadow installation")

class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _MODULE:
            return None
        return importlib.util.spec_from_file_location(fullname, _SOURCE)

sys.meta_path.insert(0, _Finder())
print("[qwen-fixed-slot-prefix-alignment] ordinary request alignment shadow armed", flush=True)
'''.encode()


def rewrite_command(
    payload: bytes,
    *,
    destination: Path,
    site_sha256: str,
    module_sha256: str,
) -> bytes:
    environment, argv = _split_command(payload)
    values = _environment_map(environment)
    pythonpath = values.get("PYTHONPATH", "")
    chained_sha = values.get("QWEN_LIVE62_HAUHAU_DELTA_AGGRESSIVE_SITE_SHA256", "")
    if not pythonpath.startswith("/") or len(chained_sha) != 64:
        raise AlignmentError("base command lacks the authenticated Hauhau site chain")
    additions = {
        ENV_NAME: "1",
        "QWEN_FIXED_SLOT_ORDINARY_PREFIX_ALIGNMENT_MODULE_SHA256": module_sha256,
        "QWEN_FIXED_SLOT_ORDINARY_PREFIX_ALIGNMENT_SITE_SHA256": site_sha256,
    }
    if set(additions) & values.keys():
        raise AlignmentError("base command already contains prefix-alignment identity")
    output = []
    for token in environment:
        name, value = token.split("=", 1)
        if name == "PYTHONPATH":
            output.extend(f"{key}={item}" for key, item in additions.items())
            value = f"{destination}:{value}"
        output.append(f"{name}={value}")
    if _environment_map(output).get("PYTHONPATH") != f"{destination}:{pythonpath}":
        raise AlignmentError("rendered PYTHONPATH differs")
    return (shlex.join(["exec", "/usr/bin/env", "-i", *output, *argv]) + "\n").encode()


def _write(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def render(
    *,
    scheduler_source: Path,
    base_command: Path,
    expected_base_sha256: str,
    destination: Path,
) -> Mapping[str, Any]:
    if destination.exists():
        raise AlignmentError("destination is create-only")
    source = scheduler_source.read_bytes()
    command_source = base_command.read_bytes()
    if digest(command_source) != expected_base_sha256:
        raise AlignmentError("base command SHA256 differs")
    environment, _argv = _split_command(command_source)
    values = _environment_map(environment)
    chained_root = Path(values["PYTHONPATH"].split(":", 1)[0])
    chained_site = chained_root / "sitecustomize.py"
    chained_sha = values["QWEN_LIVE62_HAUHAU_DELTA_AGGRESSIVE_SITE_SHA256"]
    module = rewrite_scheduler(source)
    site = render_site(
        destination,
        module_sha256=digest(module),
        chained_site=chained_site,
        chained_site_sha256=chained_sha,
        chained_pythonpath=values["PYTHONPATH"],
    )
    command = rewrite_command(
        command_source,
        destination=destination,
        site_sha256=digest(site),
        module_sha256=digest(module),
    )
    destination.mkdir(mode=0o700, parents=True)
    _write(destination / "vllm/v1/core/sched/scheduler.py", module, 0o600)
    _write(destination / "sitecustomize.py", site, 0o600)
    _write(destination / "command.sh", command, 0o700)
    manifest = {
        "schema": SCHEMA,
        "source_scheduler_sha256": digest(source),
        "scheduler_sha256": digest(module),
        "site_sha256": digest(site),
        "base_command_sha256": digest(command_source),
        "command_sha256": digest(command),
        "contract": {
            "server_argv_unchanged": True,
            "ordinary_requests_fall_through_to_cache_block_alignment": True,
            "explicit_fixed_slot_checkpoint_boundaries_unchanged": True,
        },
    }
    manifest_payload = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    _write(destination / "manifest.json", manifest_payload, 0o600)
    return {**manifest, "manifest_sha256": digest(manifest_payload)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-command", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--scheduler-source", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = render(
            scheduler_source=args.scheduler_source,
            base_command=args.base_command,
            expected_base_sha256=args.expected_base_sha256,
            destination=args.destination,
        )
    except (AlignmentError, OSError, KeyError) as error:
        print(f"qwen-prefix-alignment-overlay: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
