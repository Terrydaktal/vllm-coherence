"""Render an authenticated Qwen3 malformed function-opener repair overlay.

Some Qwen3-family responses emit a complete XML tool call with
``<parameter=TOOL_NAME>`` where the grammar requires
``<function=TOOL_NAME>``.  The stock parser treats that first parameter tag as
opaque preamble and withholds the complete call.  This overlay adds only the
state-qualified alias transition and enables request-tool-name validation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qwen_r9700_lab.full_attention_m8_row_exact_overlay import (
    RowExactOverlayError,
    _authenticate_chained_site,
    _digest,
    _environment_map,
    _find_chained_site,
    _split_command,
    _stable_private_file,
    _write_exclusive,
)

SCHEMA = "urn:qwen-r9700:qwen3-parameter-tool-repair-overlay:v1"
TARGET_MODULE = "vllm.parser.qwen3"
QWEN3_RUNTIME_SHA256 = "8a7ee658322de7b736ea5b0f802d70dd07a124b5878b4f8ad2f99eca8e1d35fb"
PARSER_CONFIG_RUNTIME_SHA256 = (
    "f7350e0ca9124001684f1f874ee72bf6a34932d3e4b84cc84567bbccf2f3e4b9"
)
ENABLE_ENV = "QWEN3_PARAMETER_TOOL_REPAIR"
REQUIRED_ENV = "QWEN3_PARAMETER_TOOL_REPAIR_REQUIRED"
SITE_SHA_ENV = "QWEN3_PARAMETER_TOOL_REPAIR_SITE_SHA256"
RUNTIME_SHA_ENV = "QWEN3_PARAMETER_TOOL_REPAIR_RUNTIME_SHA256"
CONFIG_RUNTIME_SHA_ENV = "QWEN3_PARAMETER_TOOL_REPAIR_CONFIG_RUNTIME_SHA256"
_OVERLAY_ENVIRONMENT = (
    ENABLE_ENV,
    REQUIRED_ENV,
    SITE_SHA_ENV,
    RUNTIME_SHA_ENV,
    CONFIG_RUNTIME_SHA_ENV,
)


class Qwen3ParameterToolRepairOverlayError(RowExactOverlayError):
    """The parser-repair source, command, or destination is invalid."""


def _validate_base(environment: Mapping[str, str], argv: Sequence[str]) -> None:
    required_environment = {
        "QWEN_FULL_ATTENTION_M8_EXACT_K": "1",
        "QWEN_FULL_ATTENTION_M8_EXACT_K_REQUIRED": "1",
        "QWEN_DFLASH_CONTEXT_KV_SERIAL_EXACT": "1",
        "QWEN_DFLASH_CONTEXT_KV_SERIAL_EXACT_REQUIRED": "1",
        "VLLM_ENFORCE_STRICT_TOOL_CALLING": "1",
    }
    if any(environment.get(name) != value for name, value in required_environment.items()):
        raise Qwen3ParameterToolRepairOverlayError(
            "base command lacks the exact-K/context-KV and strict-tool identity"
        )
    for option, expected in (
        ("--tool-call-parser", "qwen3_coder"),
        ("--reasoning-parser", "qwen3"),
    ):
        values = [
            argv[index + 1]
            for index, token in enumerate(argv[:-1])
            if token == option
        ]
        if values != [expected]:
            raise Qwen3ParameterToolRepairOverlayError(
                f"base command must contain one exact {option} {expected}"
            )
    if argv.count("--enable-auto-tool-choice") != 1:
        raise Qwen3ParameterToolRepairOverlayError(
            "base command must enable automatic tool parsing exactly once"
        )


def _site_source(
    destination: Path,
    *,
    chained_site: Path,
    chained_site_sha256: str,
    chained_pythonpath: str,
) -> bytes:
    outer_pythonpath = f"{destination}:{chained_pythonpath}"
    parser_config_path = (
        "/home/lewis/.local/share/qwen-r9700/runtimes/vllm-rocm/.venv/"
        "lib/python3.12/site-packages/vllm/parser/engine/parser_engine_config.py"
    )
    return f'''"""Authenticated Qwen3 parameter-as-function repair bootstrap."""
import dataclasses
import functools
import hashlib
import os
import runpy
import stat
import sys
from pathlib import Path

_ROOT = Path({str(destination)!r})
_SELF = _ROOT / "sitecustomize.py"
_CHAIN = Path({str(chained_site)!r})
_CHAIN_SHA256 = {chained_site_sha256!r}
_CHAIN_PYTHONPATH = {chained_pythonpath!r}
_OUTER_PYTHONPATH = {outer_pythonpath!r}
_TARGET = {TARGET_MODULE!r}
_RUNTIME_SHA256 = {QWEN3_RUNTIME_SHA256!r}
_CONFIG_RUNTIME = Path({parser_config_path!r})
_CONFIG_RUNTIME_SHA256 = {PARSER_CONFIG_RUNTIME_SHA256!r}
_METHOD_MARKER = "_qwen3_parameter_tool_repair_site_sha256"

def _stable_digest(path, label, private=False):
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise RuntimeError(f"{{label}} identity is unsafe")
    if private and (before.st_uid != os.getuid() or stat.S_IMODE(before.st_mode) & 0o077):
        raise RuntimeError(f"{{label}} is not owned and private")
    payload = path.read_bytes()
    after = path.lstat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise RuntimeError(f"{{label}} changed during authentication")
    return hashlib.sha256(payload).hexdigest()

if os.environ.get({ENABLE_ENV!r}) != "1" or os.environ.get({REQUIRED_ENV!r}) != "1":
    raise RuntimeError("Qwen3 parameter-tool repair identity is absent")
if os.environ.get({RUNTIME_SHA_ENV!r}) != _RUNTIME_SHA256:
    raise RuntimeError("Qwen3 parser runtime identity differs")
if os.environ.get({CONFIG_RUNTIME_SHA_ENV!r}) != _CONFIG_RUNTIME_SHA256:
    raise RuntimeError("Qwen3 parser-config runtime identity differs")
if os.environ.get("VLLM_ENFORCE_STRICT_TOOL_CALLING") != "1":
    raise RuntimeError("Qwen3 parameter-tool repair requires strict tool calling")
if os.environ.get("PYTHONPATH") != _OUTER_PYTHONPATH:
    raise RuntimeError("Qwen3 parameter-tool repair PYTHONPATH differs")
_SITE_SHA256 = _stable_digest(_SELF, "parameter-tool repair site", private=True)
if _SITE_SHA256 != os.environ.get({SITE_SHA_ENV!r}):
    raise RuntimeError("Qwen3 parameter-tool repair site SHA256 mismatch")
if _stable_digest(_CHAIN, "chained site", private=True) != _CHAIN_SHA256:
    raise RuntimeError("Qwen3 parameter-tool repair chained-site SHA256 mismatch")
if _stable_digest(_CONFIG_RUNTIME, "parser config runtime") != _CONFIG_RUNTIME_SHA256:
    raise RuntimeError("Qwen3 parser-config source SHA256 mismatch")
if _TARGET in sys.modules:
    raise RuntimeError("Qwen3 parser was imported before repair installation")

os.environ["PYTHONPATH"] = _CHAIN_PYTHONPATH
try:
    runpy.run_path(str(_CHAIN), run_name="_qwen3_parameter_tool_repair_chain")
    if os.environ.get("PYTHONPATH") != _CHAIN_PYTHONPATH:
        raise RuntimeError("chained site changed PYTHONPATH")
finally:
    os.environ["PYTHONPATH"] = _OUTER_PYTHONPATH
if _TARGET in sys.modules:
    raise RuntimeError("chained site imported Qwen3 parser before repair registration")

import combined_runtime_patch as retained  # noqa: E402

finder_type = getattr(retained, "_Finder", None)
patches = getattr(finder_type, "_PATCHES", None)
if not isinstance(finder_type, type) or type(patches) is not dict:
    raise RuntimeError("combined-runtime finder contract changed")
active_finders = [value for value in sys.meta_path if isinstance(value, finder_type)]
if len(active_finders) != 1:
    raise RuntimeError("combined-runtime finder is not installed exactly once")
previous_patch = patches.get(_TARGET)
if previous_patch is not None and not callable(previous_patch):
    raise RuntimeError("existing Qwen3 parser patch is not callable")
previous_marker = getattr(previous_patch, _METHOD_MARKER, None) if previous_patch else None
if previous_marker is not None and previous_marker != _SITE_SHA256:
    raise RuntimeError("a different Qwen3 parameter-tool repair is registered")

def _patch_qwen3(module):
    source = Path(getattr(module, "__file__", ""))
    if (
        not source.is_absolute()
        or _stable_digest(source, "Qwen3 parser runtime") != _RUNTIME_SHA256
    ):
        raise RuntimeError("Qwen3 parser source SHA256 mismatch")
    if getattr(module, "__name__", None) != _TARGET:
        raise RuntimeError("Qwen3 parser module identity changed")
    if previous_patch is not None:
        previous_patch(module)
    original = getattr(module, "qwen3_config", None)
    if not callable(original):
        raise RuntimeError("Qwen3 parser config factory ABI changed")
    inherited_marker = getattr(original, _METHOD_MARKER, None)
    if inherited_marker is not None:
        if inherited_marker != _SITE_SHA256:
            raise RuntimeError("Qwen3 parser already has a conflicting repair")
        return
    parser_state = getattr(module, "ParserState", None)
    transition_type = getattr(module, "Transition", None)
    config_type = getattr(module, "ParserEngineConfig", None)
    if (
        parser_state is None
        or not isinstance(transition_type, type)
        or not isinstance(config_type, type)
    ):
        raise RuntimeError("Qwen3 parser declarative ABI changed")

    @functools.wraps(original)
    @functools.cache
    def repaired_config(*args, **kwargs):
        config = original(*args, **kwargs)
        if not isinstance(config, config_type):
            raise RuntimeError("Qwen3 config factory returned an unknown type")
        if config.terminals.get("PARAM_START") != "<parameter=":
            raise RuntimeError("Qwen3 parameter terminal changed")
        key = (parser_state.TOOL_PREAMBLE, "PARAM_START")
        if key in config.transitions:
            raise RuntimeError("Qwen3 parameter-tool alias is no longer unique")
        transitions = dict(config.transitions)
        transitions[key] = transition_type(parser_state.TOOL_NAME, ())
        return dataclasses.replace(
            config,
            name=config.name + "-parameter-tool-repair-v1",
            transitions=transitions,
            validate_tool_names=True,
        )

    setattr(repaired_config, _METHOD_MARKER, _SITE_SHA256)
    module.qwen3_config = repaired_config
    print(
        "[qwen3-parameter-tool-repair] state-qualified known-tool alias armed",
        flush=True,
    )

if previous_marker is None:
    setattr(_patch_qwen3, _METHOD_MARKER, _SITE_SHA256)
    patches[_TARGET] = _patch_qwen3
elif patches.get(_TARGET) is not previous_patch:
    raise RuntimeError("Qwen3 parameter-tool finder identity changed")
'''.encode()


def render(args: argparse.Namespace) -> Mapping[str, Any]:
    destination = args.destination.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise Qwen3ParameterToolRepairOverlayError("destination is create-only")
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_base_sha256):
        raise Qwen3ParameterToolRepairOverlayError(
            "expected base command SHA256 is malformed"
        )
    base_payload = _stable_private_file(args.base_command, "base command")
    base_sha256 = _digest(base_payload)
    if base_sha256 != args.expected_base_sha256:
        raise Qwen3ParameterToolRepairOverlayError("base command SHA256 mismatch")
    environment, argv = _split_command(base_payload)
    env = _environment_map(environment)
    if any(name in env for name in _OVERLAY_ENVIRONMENT):
        raise Qwen3ParameterToolRepairOverlayError(
            "base command already contains parameter-tool repair state"
        )
    _validate_base(env, argv)

    chained_pythonpath = env.get("PYTHONPATH", "")
    chained_site = _find_chained_site(chained_pythonpath)
    chained_payload = _stable_private_file(chained_site, "chained sitecustomize")
    chained_sha256 = _digest(chained_payload)
    chained_claims = _authenticate_chained_site(env, chained_site, chained_sha256)

    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent_metadata = destination.parent.lstat()
    if (
        not stat.S_ISDIR(parent_metadata.st_mode)
        or destination.parent.is_symlink()
        or parent_metadata.st_uid != os.getuid()
        or stat.S_IMODE(parent_metadata.st_mode) & 0o077
    ):
        raise Qwen3ParameterToolRepairOverlayError(
            "destination parent must be one owned private directory"
        )
    staging = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise Qwen3ParameterToolRepairOverlayError("staging destination already exists")
    staging.mkdir(mode=0o700)
    try:
        site_payload = _site_source(
            destination,
            chained_site=chained_site,
            chained_site_sha256=chained_sha256,
            chained_pythonpath=chained_pythonpath,
        )
        site_sha256 = _digest(site_payload)
        additions = {
            ENABLE_ENV: "1",
            REQUIRED_ENV: "1",
            SITE_SHA_ENV: site_sha256,
            RUNTIME_SHA_ENV: QWEN3_RUNTIME_SHA256,
            CONFIG_RUNTIME_SHA_ENV: PARSER_CONFIG_RUNTIME_SHA256,
        }
        rewritten_environment = list(environment)
        for index, token in enumerate(rewritten_environment):
            if token.startswith("PYTHONPATH="):
                rewritten_environment[index] = (
                    f"PYTHONPATH={destination}:{chained_pythonpath}"
                )
                break
        else:  # pragma: no cover - guarded by _find_chained_site
            raise Qwen3ParameterToolRepairOverlayError("base command lacks PYTHONPATH")
        rewritten_environment.extend(f"{name}={value}" for name, value in additions.items())
        command_payload = (
            shlex.join(["exec", "/usr/bin/env", "-i", *rewritten_environment, *argv])
            + "\n"
        ).encode()
        manifest = {
            "schema": SCHEMA,
            "classification": "production_safety_repair_candidate",
            "promotable": False,
            "base_command": {
                "path": str(args.base_command.expanduser().absolute()),
                "sha256": base_sha256,
            },
            "chained_site": {
                "path": str(chained_site),
                "sha256": chained_sha256,
                "authenticated_by_environment": chained_claims,
            },
            "command_sha256": _digest(command_payload),
            "environment_delta": {
                "PYTHONPATH": f"{destination}:{chained_pythonpath}",
                **additions,
            },
            "patch_contract": {
                "module": TARGET_MODULE,
                "factory": "qwen3_config",
                "state": "TOOL_PREAMBLE",
                "malformed_terminal": "<parameter=",
                "next_state": "TOOL_NAME",
                "validate_request_tool_names": True,
                "ordinary_function_transition_unchanged": True,
                "model_token_ids_unchanged": True,
            },
            "runtime_sha256": QWEN3_RUNTIME_SHA256,
            "parser_config_runtime_sha256": PARSER_CONFIG_RUNTIME_SHA256,
            "site_sha256": site_sha256,
        }
        manifest_payload = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
        _write_exclusive(staging / "sitecustomize.py", site_payload, 0o600)
        _write_exclusive(staging / "command.sh", command_payload, 0o700)
        _write_exclusive(staging / "manifest.json", manifest_payload, 0o600)
        if destination.exists() or destination.is_symlink():
            raise Qwen3ParameterToolRepairOverlayError(
                "destination appeared during create-only rendering"
            )
        staging.rename(destination)
        return {**manifest, "manifest_sha256": _digest(manifest_payload)}
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-command", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = render(args)
    except (OSError, RowExactOverlayError) as error:
        print(f"qwen3-parameter-tool-repair-overlay: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
