"""Render an authenticated, create-only decoder-boundary diagnostic overlay."""

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

SCHEMA = "urn:qwen-r9700:m1-m8-layer-diagnostic-overlay:v5"
MODULE_SOURCE = Path(__file__).parents[2] / "experiments/m8-layer-diagnostic/layer_diagnostic.py"
_DIAGNOSTIC_ENVIRONMENT = (
    "QWEN_M8_LAYER_DIAGNOSTIC",
    "QWEN_M8_LAYER_DIAGNOSTIC_REQUIRED",
    "QWEN_M8_LAYER_DIAGNOSTIC_MODULE_SHA256",
    "QWEN_M8_LAYER_DIAGNOSTIC_OUTPUT_ROOT",
    "QWEN_M8_LAYER_DIAGNOSTIC_POSITIONS",
    "QWEN_M8_LAYER_DIAGNOSTIC_SITE_SHA256",
    "QWEN_M8_LAYER_DIAGNOSTIC_TENSOR_CAPSULE",
    "QWEN_M8_LAYER_DIAGNOSTIC_QUEST_PAGE_COMPARE",
)


class DiagnosticOverlayError(RuntimeError):
    """The overlay input or output violated its identity contract."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_private_file(path: Path, label: str, *, allow_public_read: bool = False) -> bytes:
    path = path.expanduser().absolute()
    try:
        before = path.lstat()
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise DiagnosticOverlayError(f"cannot read {label}: {error}") from error

    def identity(value: os.stat_result) -> tuple[int, int, int, int]:
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns

    permissions = stat.S_IMODE(before.st_mode)
    permissions_invalid = permissions & 0o022 if allow_public_read else permissions & 0o077
    if (
        identity(before) != identity(after)
        or not stat.S_ISREG(before.st_mode)
        or path.is_symlink()
        or before.st_uid != os.getuid()
        or permissions_invalid
    ):
        raise DiagnosticOverlayError(f"{label} must be one stable owned private regular file")
    return payload


def _split_command(payload: bytes) -> tuple[list[str], list[str]]:
    try:
        tokens = shlex.split(payload.decode())
    except (UnicodeDecodeError, ValueError) as error:
        raise DiagnosticOverlayError(f"base command is invalid: {error}") from error
    if tokens[:3] != ["exec", "/usr/bin/env", "-i"]:
        raise DiagnosticOverlayError("base command must use exec /usr/bin/env -i")
    index = 3
    environment: list[str] = []
    while index < len(tokens) and "=" in tokens[index]:
        environment.append(tokens[index])
        index += 1
    if not environment or index == len(tokens):
        raise DiagnosticOverlayError("base command lacks environment or server argv")
    names = [token.split("=", 1)[0] for token in environment]
    if len(names) != len(set(names)):
        raise DiagnosticOverlayError("base command contains duplicate environment variables")
    return environment, tokens[index:]


def _find_chained_site(pythonpath: str) -> Path:
    """Resolve the site bootstrap Python itself will import from PYTHONPATH."""

    if not pythonpath.startswith("/"):
        raise DiagnosticOverlayError("base command lacks an absolute PYTHONPATH")
    for entry in pythonpath.split(":"):
        root = Path(entry)
        if not root.is_absolute():
            raise DiagnosticOverlayError("base PYTHONPATH contains a relative entry")
        candidate = root / "sitecustomize.py"
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    raise DiagnosticOverlayError("base PYTHONPATH contains no sitecustomize bootstrap")


def _rewrite_offload_root(
    argv: list[str], *, expected_root: Path | None, replacement_root: Path | None
) -> list[str]:
    if (expected_root is None) != (replacement_root is None):
        raise DiagnosticOverlayError(
            "expected and replacement offload roots must be supplied together"
        )
    if expected_root is None or replacement_root is None:
        return argv
    indices = [index for index, token in enumerate(argv) if token == "--kv-transfer-config"]
    if len(indices) != 1 or indices[0] + 1 >= len(argv):
        raise DiagnosticOverlayError("base command lacks one exact KV-transfer config")
    index = indices[0] + 1
    try:
        config = json.loads(argv[index])
    except json.JSONDecodeError as error:
        raise DiagnosticOverlayError("base KV-transfer config is invalid JSON") from error
    extra = config.get("kv_connector_extra_config") if isinstance(config, dict) else None
    tiers = extra.get("secondary_tiers") if isinstance(extra, dict) else None
    if (
        not isinstance(tiers, list)
        or len(tiers) != 1
        or not isinstance(tiers[0], dict)
        or tiers[0].get("type") != "fs"
        or tiers[0].get("root_dir") != str(expected_root)
    ):
        raise DiagnosticOverlayError("base KV-transfer offload root differs from expectation")
    if not replacement_root.is_absolute():
        raise DiagnosticOverlayError("replacement offload root must be absolute")
    tiers[0]["root_dir"] = str(replacement_root)
    rewritten = list(argv)
    rewritten[index] = json.dumps(config, separators=(",", ":"), sort_keys=True)
    return rewritten


def _rewrite_environment_value(
    environment: list[str], *, name: str, expected: Path, replacement: Path
) -> list[str]:
    """Replace one exact absolute path-valued environment binding."""

    if not expected.is_absolute() or not replacement.is_absolute():
        raise DiagnosticOverlayError(f"{name} roots must be absolute")
    prefix = f"{name}="
    matches = [index for index, token in enumerate(environment) if token.startswith(prefix)]
    if len(matches) != 1:
        raise DiagnosticOverlayError(f"base command lacks one exact {name} binding")
    index = matches[0]
    if environment[index] != f"{name}={expected}":
        raise DiagnosticOverlayError(f"base {name} root differs from expectation")
    rewritten = list(environment)
    rewritten[index] = f"{name}={replacement}"
    return rewritten


def _guard_directory(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as error:
        raise DiagnosticOverlayError(f"cannot inspect {label}: {error}") from error
    mode = stat.S_IMODE(info.st_mode)
    if (
        not stat.S_ISDIR(info.st_mode)
        or path.is_symlink()
        or info.st_uid != os.getuid()
        or mode & 0o022
    ):
        raise DiagnosticOverlayError(f"{label} must be one stable owned directory")
    return info


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _create_private_directory(path: Path, label: str) -> None:
    if not path.is_absolute() or path == Path(path.anchor) or len(path.parts) < 4:
        raise DiagnosticOverlayError(f"refusing unsafe {label}: {path}")
    parent = path.parent
    _guard_directory(parent, f"{label} parent")
    try:
        path.mkdir(mode=0o700)
    except FileExistsError as error:
        raise DiagnosticOverlayError(f"{label} already exists: {path}") from error
    _fsync_directory(parent)
    info = _guard_directory(path, label)
    if stat.S_IMODE(info.st_mode) != 0o700:
        raise DiagnosticOverlayError(f"{label} must be mode 0700")


def _initialize_runtime_roots(offload_root: Path, selection_root: Path) -> None:
    """Create isolated empty runtime roots required before the server starts."""

    _create_private_directory(selection_root, "diagnostic selection root")
    _create_private_directory(offload_root, "diagnostic offload root")
    metadata = offload_root / ".qwen-250k-cache-v1"
    _create_private_directory(metadata, "diagnostic metadata root")
    lifecycle = metadata / "LIFECYCLE.lock"
    try:
        fd = os.open(
            lifecycle,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
    except OSError as error:
        raise DiagnosticOverlayError(f"cannot create diagnostic lifecycle lock: {error}") from error
    try:
        os.fsync(fd)
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size != 0
        ):
            raise DiagnosticOverlayError("diagnostic lifecycle lock has an unsafe identity")
    finally:
        os.close(fd)
    _fsync_directory(metadata)


def _rewrite_cache_root(
    payload: bytes,
    *,
    expected_root: Path,
    replacement_root: Path,
    expected_occurrences: int,
    label: str,
) -> bytes:
    """Retarget exact fixed-slot root validators for an isolated diagnostic."""

    if expected_root.parent != replacement_root.parent:
        raise DiagnosticOverlayError(
            "diagnostic offload root must retain the authenticated cache parent"
        )
    expected_name = json.dumps(expected_root.name).encode()
    replacement_name = json.dumps(replacement_root.name).encode()
    expected_path = json.dumps(str(expected_root)).encode()
    replacement_path = json.dumps(str(replacement_root)).encode()
    name_count = payload.count(expected_name)
    path_count = payload.count(expected_path)
    if name_count == expected_occurrences and path_count == 0:
        old, new = expected_name, replacement_name
    elif path_count == expected_occurrences and name_count == 0:
        old, new = expected_path, replacement_path
    else:
        raise DiagnosticOverlayError(f"{label} fixed-slot root validation preimage differs")
    rewritten = payload.replace(old, new)
    if rewritten == payload or old in rewritten:
        raise DiagnosticOverlayError(f"{label} offload-root rewrite was incomplete")
    return rewritten


def _site_source(
    destination: Path,
    module_sha256: str,
    chained_site: Path,
    chained_sha256: str,
    chained_pythonpath: str,
    chained_environment: dict[str, str | None],
    runner_patch: dict[str, str | int] | None = None,
    identity_patches: tuple[dict[str, str | int], ...] = (),
) -> bytes:
    outer_pythonpath = f"{destination}:{chained_pythonpath}"
    runner_constants = ""
    runner_activation = ""
    if runner_patch is not None:
        runner_constants = f"""\n_RUNNER = _ROOT / "model_runner.py"
_RUNNER_PREIMAGE_SHA256 = {runner_patch["preimage_sha256"]!r}
_RUNNER_SHA256 = {runner_patch["sha256"]!r}
_RUNNER_BYTES = {runner_patch["bytes"]!r}
_RUNNER_MODULE = "vllm.v1.worker.gpu.model_runner"
"""
        runner_activation = """
_runner_payload = _RUNNER.read_bytes()
if (
    len(_runner_payload) != _RUNNER_BYTES
    or hashlib.sha256(_runner_payload).hexdigest() != _RUNNER_SHA256
):
    raise RuntimeError("diagnostic model runner identity differs")
_runner_matches = []
for _finder in sys.meta_path:
    if _finder.__class__.__name__ != "_RunnerFinder" or not hasattr(_finder, "_payload"):
        continue
    _globals = _finder.find_spec.__globals__
    if (
        _globals.get("_RUNNER_MODULE") == _RUNNER_MODULE
        and _globals.get("_PATCHED_RUNNER_SHA256") == _RUNNER_PREIMAGE_SHA256
    ):
        _runner_matches.append((_finder, _globals))
if len(_runner_matches) != 1:
    raise RuntimeError("diagnostic runner finder preimage differs")
_runner_finder, _runner_globals = _runner_matches[0]
_runner_globals["_PATCHED_RUNNER"] = _RUNNER
_runner_globals["_PATCHED_RUNNER_SHA256"] = _RUNNER_SHA256
_runner_globals["_PATCHED_RUNNER_BYTES"] = _RUNNER_BYTES
_runner_finder._payload = _runner_payload
for _module_name in ("lm_head_m8_direct", "lm_head_w4_topk"):
    _binder = sys.modules.get(_module_name)
    if _binder is None or _binder._RUNNER_SHA256 != _RUNNER_PREIMAGE_SHA256:
        raise RuntimeError("diagnostic LM-head runner binder preimage differs")
    _binder._RUNNER_PATH = _RUNNER
    _binder._RUNNER_SHA256 = _RUNNER_SHA256
"""
    identity_constants = f"\n_IDENTITY_PATCHES = {identity_patches!r}\n"
    identity_activation = ""
    if identity_patches:
        identity_activation = """
for _patch in _IDENTITY_PATCHES:
    _path = _ROOT / _patch["filename"]
    _payload = _path.read_bytes()
    if (
        len(_payload) != _patch["bytes"]
        or hashlib.sha256(_payload).hexdigest() != _patch["sha256"]
    ):
        raise RuntimeError(f"diagnostic {_patch['label']} identity differs")
    if _patch["module"] in sys.modules:
        raise RuntimeError(f"diagnostic {_patch['label']} module imported before retarget")
    _matches = []
    for _finder in sys.meta_path:
        _records = getattr(_finder, "_records", None)
        if _finder.__class__.__name__ != "_IdentityModuleFinder" or type(_records) is not dict:
            continue
        _record = _records.get(_patch["module"])
        if (
            type(_record) is tuple
            and len(_record) == 8
            and _record[5] == _patch["preimage_sha256"]
        ):
            _matches.append((_records, _record))
    if len(_matches) != 1:
        raise RuntimeError(f"diagnostic {_patch['label']} finder preimage differs")
    _records, _record = _matches[0]
    _records[_patch["module"]] = (
        *_record[:4],
        _path,
        _patch["sha256"],
        _patch["bytes"],
        _payload,
    )
"""
    return f'''"""Authenticated M1/M8 layer-diagnostic site bootstrap."""
import hashlib
import os
import runpy
import stat
import sys
from pathlib import Path

_ROOT = Path({str(destination)!r})
_SELF = _ROOT / "sitecustomize.py"
_MODULE = _ROOT / "layer_diagnostic.py"
_CHAIN = Path({str(chained_site)!r})
_CHAIN_SHA256 = {chained_sha256!r}
_CHAIN_PYTHONPATH = {chained_pythonpath!r}
_CHAIN_ENVIRONMENT = {chained_environment!r}
_OUTER_PYTHONPATH = {outer_pythonpath!r}
_MODULE_SHA256 = {module_sha256!r}
{runner_constants}
{identity_constants}

def _stable_digest(path, label):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink() or before.st_uid != os.getuid():
        raise RuntimeError(f"{{label}} identity is unsafe")
    payload = path.read_bytes()
    after = path.lstat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise RuntimeError(f"{{label}} changed during authentication")
    return hashlib.sha256(payload).hexdigest()

if (
    os.environ.get("QWEN_M8_LAYER_DIAGNOSTIC") != "1"
    or os.environ.get("QWEN_M8_LAYER_DIAGNOSTIC_REQUIRED") != "1"
):
    raise RuntimeError("M8 layer diagnostic identity is absent")
if os.environ.get("PYTHONPATH") != _OUTER_PYTHONPATH:
    raise RuntimeError("M8 layer diagnostic PYTHONPATH differs")
if _stable_digest(_SELF, "diagnostic site") != os.environ.get(
    "QWEN_M8_LAYER_DIAGNOSTIC_SITE_SHA256"
):
    raise RuntimeError("M8 layer diagnostic site SHA256 mismatch")
if (
    _stable_digest(_MODULE, "diagnostic module") != _MODULE_SHA256
    or os.environ.get("QWEN_M8_LAYER_DIAGNOSTIC_MODULE_SHA256") != _MODULE_SHA256
):
    raise RuntimeError("M8 layer diagnostic module SHA256 mismatch")
if _stable_digest(_CHAIN, "chained site") != _CHAIN_SHA256:
    raise RuntimeError("M8 layer diagnostic chained site SHA256 mismatch")
_OUTER_ENVIRONMENT = {{
    name: os.environ.get(name) for name in _CHAIN_ENVIRONMENT
}}
try:
    os.environ["PYTHONPATH"] = _CHAIN_PYTHONPATH
    for name, value in _CHAIN_ENVIRONMENT.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    runpy.run_path(str(_CHAIN), run_name="_qwen_m8_layer_diagnostic_chained_site")
    if os.environ.get("PYTHONPATH") != _CHAIN_PYTHONPATH:
        raise RuntimeError("chained site changed PYTHONPATH")
finally:
    os.environ["PYTHONPATH"] = _OUTER_PYTHONPATH
    for name, value in _OUTER_ENVIRONMENT.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
{runner_activation}
{identity_activation}
import layer_diagnostic  # noqa: E402,F401
'''.encode()


def render(args: argparse.Namespace) -> dict[str, Any]:
    destination = args.destination.expanduser().absolute()
    output_root = args.output_root.expanduser().absolute()
    if destination.exists():
        raise DiagnosticOverlayError("destination is create-only")
    if output_root.exists():
        raise DiagnosticOverlayError("capture output root is create-only")
    base = _stable_private_file(args.base_command, "base command")
    if hashlib.sha256(base).hexdigest() != args.expected_base_sha256:
        raise DiagnosticOverlayError("base command SHA256 mismatch")
    environment, argv = _split_command(base)
    isolation_values = (
        args.expected_existing_offload_root,
        args.offload_root,
        args.expected_existing_selection_root,
        args.selection_root,
    )
    if any(value is not None for value in isolation_values) and any(
        value is None for value in isolation_values
    ):
        raise DiagnosticOverlayError(
            "offload and selection isolation roots must all be supplied together"
        )
    if args.selection_root is not None:
        assert args.expected_existing_selection_root is not None
        environment = _rewrite_environment_value(
            environment,
            name="QWEN_FIXED_SLOT_SNAPSHOT_ROOT",
            expected=args.expected_existing_selection_root,
            replacement=args.selection_root,
        )
    argv = _rewrite_offload_root(
        argv,
        expected_root=args.expected_existing_offload_root,
        replacement_root=args.offload_root,
    )
    env = dict(token.split("=", 1) for token in environment)
    chained_pythonpath = env.get("PYTHONPATH", "")
    chained_site = _find_chained_site(chained_pythonpath)
    chained_payload = _stable_private_file(
        chained_site, "chained sitecustomize", allow_public_read=True
    )
    positions = tuple(args.positions)
    if (
        not positions
        or tuple(sorted(set(positions))) != positions
        or any(value < 1 for value in positions)
    ):
        raise DiagnosticOverlayError("positions must be ordered unique positive integers")
    if len(positions) > 64:
        raise DiagnosticOverlayError("at most 64 positions may be captured")
    capsule_values = (
        args.tensor_capsule_pass,
        args.tensor_capsule_layer,
        args.tensor_capsule_position,
    )
    if any(value is not None for value in capsule_values):
        if any(value is None for value in capsule_values):
            raise DiagnosticOverlayError(
                "tensor capsule pass, layer, and position must be supplied together"
            )
        assert all(value is not None for value in capsule_values)
        if (
            args.tensor_capsule_pass < 0
            or not 0 <= args.tensor_capsule_layer < 64
            or args.tensor_capsule_position not in positions
        ):
            raise DiagnosticOverlayError("tensor capsule selector is outside the capture")
        tensor_capsule = (
            f"{args.tensor_capsule_pass}:{args.tensor_capsule_layer}:{args.tensor_capsule_position}"
        )
    else:
        tensor_capsule = ""

    module_source = (args.module_source or MODULE_SOURCE).expanduser().absolute()
    runner_patch: dict[str, str | int] | None = None
    runtime_inputs = (
        (
            "model runner",
            "model_runner.py",
            args.model_runner_source,
            args.expected_model_runner_sha256,
            2,
            None,
        ),
        (
            "offload scheduler",
            "offload_scheduler.py",
            args.offload_scheduler_source,
            args.expected_offload_scheduler_sha256,
            1,
            "vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler",
        ),
        (
            "persistent selection",
            "qwen_persistent_selection.py",
            args.persistent_selection_source,
            args.expected_persistent_selection_sha256,
            1,
            "vllm.distributed.kv_transfer.kv_connector.v1.offloading.qwen_persistent_selection",
        ),
    )
    runtime_patches: list[tuple[str, bytes, dict[str, str | int]]] = []
    if args.offload_root is not None:
        if any(
            source is None or expected_sha is None
            for _, _, source, expected_sha, _, _ in runtime_inputs
        ):
            raise DiagnosticOverlayError(
                "isolated offload diagnostics require all runtime sources and SHA256s"
            )
        assert args.expected_existing_offload_root is not None
        for label, filename, source, expected_sha, occurrences, module in runtime_inputs:
            assert source is not None and expected_sha is not None
            payload = _stable_private_file(source, f"{label} source")
            source_sha256 = hashlib.sha256(payload).hexdigest()
            if source_sha256 != expected_sha:
                raise DiagnosticOverlayError(f"{label} source SHA256 mismatch")
            patched_payload = _rewrite_cache_root(
                payload,
                expected_root=args.expected_existing_offload_root,
                replacement_root=args.offload_root,
                expected_occurrences=occurrences,
                label=label,
            )
            patch: dict[str, str | int] = {
                "bytes": len(patched_payload),
                "filename": filename,
                "label": label,
                "preimage_sha256": source_sha256,
                "sha256": hashlib.sha256(patched_payload).hexdigest(),
            }
            if module is not None:
                patch["module"] = module
            runtime_patches.append((filename, patched_payload, patch))
        runner_patch = runtime_patches[0][2]
    elif any(
        source is not None or expected_sha is not None
        for _, _, source, expected_sha, _, _ in runtime_inputs
    ):
        raise DiagnosticOverlayError(
            "runtime patch inputs require an isolated offload-root rewrite"
        )
    staging = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
    if staging.exists():
        raise DiagnosticOverlayError("staging destination already exists")
    staging.mkdir(mode=0o700, parents=True)
    try:
        module_payload = _stable_private_file(module_source, "diagnostic module source")
        module_path = staging / "layer_diagnostic.py"
        module_path.write_bytes(module_payload)
        module_path.chmod(0o600)
        module_sha256 = hashlib.sha256(module_payload).hexdigest()
        for filename, patched_payload, _patch in runtime_patches:
            runtime_path = staging / filename
            runtime_path.write_bytes(patched_payload)
            runtime_path.chmod(0o600)
        identity_patches = tuple(patch for _, _, patch in runtime_patches[1:])
        site_payload = _site_source(
            destination,
            module_sha256,
            chained_site,
            hashlib.sha256(chained_payload).hexdigest(),
            chained_pythonpath,
            {name: env.get(name) for name in _DIAGNOSTIC_ENVIRONMENT},
            runner_patch,
            identity_patches,
        )
        site_path = staging / "sitecustomize.py"
        site_path.write_bytes(site_payload)
        site_path.chmod(0o600)
        site_sha256 = hashlib.sha256(site_payload).hexdigest()

        additions = {
            "PYTHONPATH": f"{destination}:{chained_pythonpath}",
            "QWEN_M8_LAYER_DIAGNOSTIC": "1",
            "QWEN_M8_LAYER_DIAGNOSTIC_REQUIRED": "1",
            "QWEN_M8_LAYER_DIAGNOSTIC_MODULE_SHA256": module_sha256,
            "QWEN_M8_LAYER_DIAGNOSTIC_OUTPUT_ROOT": str(output_root),
            "QWEN_M8_LAYER_DIAGNOSTIC_POSITIONS": ",".join(str(value) for value in positions),
            "QWEN_M8_LAYER_DIAGNOSTIC_SITE_SHA256": site_sha256,
            "QWEN_M8_LAYER_DIAGNOSTIC_TENSOR_CAPSULE": tensor_capsule,
            "QWEN_M8_LAYER_DIAGNOSTIC_QUEST_PAGE_COMPARE": (
                "1" if args.quest_page_compare else "0"
            ),
        }
        rewritten = [token for token in environment if token.split("=", 1)[0] not in additions]
        rewritten.extend(f"{name}={value}" for name, value in additions.items())
        command_tokens = ["exec", "/usr/bin/env", "-i", *rewritten, *argv]
        command = " ".join(shlex.quote(token) for token in command_tokens) + "\n"
        command_path = staging / "command.sh"
        command_path.write_text(command)
        command_path.chmod(0o700)
        manifest = {
            "schema": SCHEMA,
            "base_command_sha256": args.expected_base_sha256,
            "chained_site": str(chained_site),
            "chained_site_sha256": hashlib.sha256(chained_payload).hexdigest(),
            "command_sha256": _sha256(command_path),
            "module_sha256": module_sha256,
            "module_source": str(module_source),
            "runtime_patches": [patch for _, _, patch in runtime_patches],
            "offload_root": str(args.offload_root) if args.offload_root is not None else None,
            "selection_root": (
                str(args.selection_root) if args.selection_root is not None else None
            ),
            "output_root": str(output_root),
            "positions": list(positions),
            "site_sha256": site_sha256,
            "tensor_capsule": tensor_capsule or None,
            "quest_page_compare": bool(args.quest_page_compare),
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifest_path.chmod(0o600)
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if args.offload_root is not None:
            assert args.selection_root is not None
            _initialize_runtime_roots(args.offload_root, args.selection_root)
        staging.rename(destination)
        output_root.mkdir(mode=0o700, parents=True)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen-m8-layer-diagnostic-overlay")
    parser.add_argument("--base-command", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--module-source", type=Path)
    parser.add_argument("--model-runner-source", type=Path)
    parser.add_argument("--expected-model-runner-sha256")
    parser.add_argument("--offload-scheduler-source", type=Path)
    parser.add_argument("--expected-offload-scheduler-sha256")
    parser.add_argument("--persistent-selection-source", type=Path)
    parser.add_argument("--expected-persistent-selection-sha256")
    parser.add_argument(
        "--positions",
        required=True,
        type=lambda value: tuple(int(item) for item in value.split(",")),
    )
    parser.add_argument("--tensor-capsule-pass", type=int)
    parser.add_argument("--tensor-capsule-layer", type=int)
    parser.add_argument("--tensor-capsule-position", type=int)
    parser.add_argument("--quest-page-compare", action="store_true")
    parser.add_argument("--expected-existing-offload-root", type=Path)
    parser.add_argument("--offload-root", type=Path)
    parser.add_argument("--expected-existing-selection-root", type=Path)
    parser.add_argument("--selection-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = render(_parser().parse_args(argv))
    except (DiagnosticOverlayError, OSError, ValueError) as error:
        print(f"qwen-m8-layer-diagnostic-overlay: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
