"""Render an authenticated serial-equivalent DFlash context-KV overlay.

DFlash normally inserts the target model's accepted hidden-state rows into its
five-layer context KV cache in one batched call.  The authoritative transition
is the same operation evaluated one committed row at a time.  This overlay
uses that exact existing M1 implementation for M2..M8 accepted transitions,
including hidden RMSNorm, W4A16 projection, K RMSNorm, RoPE, and cache writes.

The overlay is deliberately default-off and non-promotable until complete
state qualification and performance gates have passed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
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
    _validate_production_m8,
    _write_exclusive,
)

SCHEMA = "urn:qwen-r9700:dflash-context-kv-serial-overlay:v1"
ENABLE_ENV = "QWEN_DFLASH_CONTEXT_KV_SERIAL_EXACT"
REQUIRED_ENV = "QWEN_DFLASH_CONTEXT_KV_SERIAL_EXACT_REQUIRED"
SITE_SHA_ENV = "QWEN_DFLASH_CONTEXT_KV_SERIAL_EXACT_SITE_SHA256"
RUNTIME_SHA_ENV = "QWEN_DFLASH_CONTEXT_KV_SERIAL_EXACT_RUNTIME_SHA256"
TARGET_MODULE = "vllm.model_executor.models.qwen3_dflash"
QWEN3_DFLASH_RUNTIME_SHA256 = (
    "a0bbf7a19b56725bc35c954e0cdddd20cd96998be821d2ddc8e294b6e929dbf0"
)
_OVERLAY_ENVIRONMENT = (
    ENABLE_ENV,
    REQUIRED_ENV,
    SITE_SHA_ENV,
    RUNTIME_SHA_ENV,
)


class DFlashContextKVOverlayError(RowExactOverlayError):
    """The serial context-KV overlay identity or source command is invalid."""


def _site_source(
    destination: Path,
    *,
    chained_site: Path,
    chained_site_sha256: str,
    chained_pythonpath: str,
) -> bytes:
    outer_pythonpath = f"{destination}:{chained_pythonpath}"
    return f'''"""Authenticated DFlash serial context-KV bootstrap."""
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
_RUNTIME_SHA256 = {QWEN3_DFLASH_RUNTIME_SHA256!r}
_METHOD_MARKER = "_qwen_dflash_context_kv_serial_site_sha256"

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
    raise RuntimeError("DFlash serial context-KV identity is absent")
if os.environ.get({RUNTIME_SHA_ENV!r}) != _RUNTIME_SHA256:
    raise RuntimeError("DFlash serial context-KV runtime identity differs")
if os.environ.get("QWEN_DFLASH_GREEDY_LOOP_ESCAPE") != "0":
    raise RuntimeError("DFlash serial context-KV requires loop escape disabled")
if os.environ.get("QWEN_DFLASH_GREEDY_M8_VERIFIER") != "1":
    raise RuntimeError("DFlash serial context-KV requires production M8 verification")
if os.environ.get("QWEN_FIXED_SLOT_TARGET_ONLY_ORACLE") != "0":
    raise RuntimeError("DFlash serial context-KV forbids target-only execution")
if os.environ.get("QWEN_FULL_ATTENTION_M8_EXACT_K") != "1":
    raise RuntimeError("DFlash serial context-KV requires the target exact-K repair")
if os.environ.get("PYTHONPATH") != _OUTER_PYTHONPATH:
    raise RuntimeError("DFlash serial context-KV PYTHONPATH differs")
_SITE_SHA256 = _stable_digest(_SELF, "DFlash serial context-KV site", private=True)
if _SITE_SHA256 != os.environ.get({SITE_SHA_ENV!r}):
    raise RuntimeError("DFlash serial context-KV site SHA256 mismatch")
if _stable_digest(_CHAIN, "chained site", private=True) != _CHAIN_SHA256:
    raise RuntimeError("DFlash serial context-KV chained-site SHA256 mismatch")
if _TARGET in sys.modules:
    raise RuntimeError("qwen3_dflash was imported before serial context-KV installation")

os.environ["PYTHONPATH"] = _CHAIN_PYTHONPATH
try:
    runpy.run_path(str(_CHAIN), run_name="_qwen_dflash_context_kv_serial_chain")
    if os.environ.get("PYTHONPATH") != _CHAIN_PYTHONPATH:
        raise RuntimeError("chained site changed PYTHONPATH")
finally:
    os.environ["PYTHONPATH"] = _OUTER_PYTHONPATH
if _TARGET in sys.modules:
    raise RuntimeError("chained site imported qwen3_dflash before hook registration")

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
    raise RuntimeError("existing qwen3_dflash patch is not callable")
previous_marker = getattr(previous_patch, _METHOD_MARKER, None) if previous_patch else None
if previous_marker is not None and previous_marker != _SITE_SHA256:
    raise RuntimeError("a different DFlash serial context-KV patch is registered")

def _patch_qwen3_dflash(module):
    source = Path(getattr(module, "__file__", ""))
    if (
        not source.is_absolute()
        or _stable_digest(source, "qwen3_dflash runtime") != _RUNTIME_SHA256
    ):
        raise RuntimeError("qwen3_dflash runtime SHA256 mismatch")
    if getattr(module, "__name__", None) != _TARGET:
        raise RuntimeError("qwen3_dflash module identity changed")
    if previous_patch is not None:
        previous_patch(module)
    cls = getattr(module, "DFlashQwen3Model", None)
    original = getattr(cls, "precompute_and_store_context_kv", None)
    if not isinstance(cls, type) or not callable(original):
        raise RuntimeError("DFlash context-KV ABI changed")
    if getattr(original, _METHOD_MARKER, None) is not None:
        if getattr(original, _METHOD_MARKER) != _SITE_SHA256:
            raise RuntimeError("DFlash context-KV carries a conflicting wrapper")
        return

    @functools.wraps(original)
    def serial_context_kv(self, context_states, context_positions, context_slot_mapping=None):
        try:
            num_ctx = int(context_states.shape[0])
            hidden = int(context_states.shape[-1])
            position_count = int(context_positions.shape[-1])
        except (AttributeError, TypeError, ValueError):
            return original(self, context_states, context_positions, context_slot_mapping)
        if not 2 <= num_ctx <= 8:
            return original(self, context_states, context_positions, context_slot_mapping)
        if hidden != 5120 or position_count != num_ctx:
            raise RuntimeError("DFlash serial context-KV M8 input geometry changed")
        if not hasattr(self, "_num_attn_layers"):
            self._build_fused_kv_buffers()
        observed = (
            int(self._num_attn_layers), int(self._kv_size),
            int(self._head_dim), int(self._num_kv_heads),
        )
        if observed != (5, 1024, 128, 8):
            raise RuntimeError("DFlash serial context-KV model geometry changed")

        per_layer = isinstance(context_slot_mapping, (list, tuple))
        if per_layer and len(context_slot_mapping) != 5:
            raise RuntimeError("DFlash serial context-KV mapping layer count changed")
        for row in range(num_ctx):
            if context_slot_mapping is None:
                row_mapping = None
            elif per_layer:
                row_mapping = [
                    None if mapping is None else mapping[row : row + 1]
                    for mapping in context_slot_mapping
                ]
            else:
                row_mapping = context_slot_mapping[row : row + 1]
            original(
                self,
                context_states[row : row + 1],
                context_positions[..., row : row + 1],
                row_mapping,
            )

    setattr(serial_context_kv, _METHOD_MARKER, _SITE_SHA256)
    cls.precompute_and_store_context_kv = serial_context_kv
    print("[qwen-dflash-context-kv-serial] M2..M8 accepted rows use exact M1 order", flush=True)

if previous_marker is None:
    setattr(_patch_qwen3_dflash, _METHOD_MARKER, _SITE_SHA256)
    patches[_TARGET] = _patch_qwen3_dflash
elif patches.get(_TARGET) is not previous_patch:
    raise RuntimeError("DFlash serial context-KV finder identity changed")
'''.encode()


def render(args: argparse.Namespace) -> Mapping[str, Any]:
    destination = args.destination.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise DFlashContextKVOverlayError("destination is create-only")
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_base_sha256):
        raise DFlashContextKVOverlayError("expected base command SHA256 is malformed")
    base_payload = _stable_private_file(args.base_command, "base command")
    if _digest(base_payload) != args.expected_base_sha256:
        raise DFlashContextKVOverlayError("base command SHA256 mismatch")
    environment, argv = _split_command(base_payload)
    env = _environment_map(environment)
    if any(name in env for name in _OVERLAY_ENVIRONMENT):
        raise DFlashContextKVOverlayError(
            "base command already contains DFlash serial context-KV state"
        )
    _validate_production_m8(env, argv)
    if env.get("QWEN_FULL_ATTENTION_M8_EXACT_K") != "1":
        raise DFlashContextKVOverlayError("base command lacks the target exact-K repair")
    chained_pythonpath = env.get("PYTHONPATH", "")
    chained_site = _find_chained_site(chained_pythonpath)
    chained_payload = _stable_private_file(chained_site, "chained sitecustomize")
    chained_sha256 = _digest(chained_payload)
    chained_claims = _authenticate_chained_site(env, chained_site, chained_sha256)

    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent = destination.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or destination.parent.is_symlink()
        or parent.st_uid != os.getuid()
        or stat.S_IMODE(parent.st_mode) & 0o077
    ):
        raise DFlashContextKVOverlayError(
            "destination parent must be one owned private directory"
        )
    staging = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise DFlashContextKVOverlayError("staging destination already exists")
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
            RUNTIME_SHA_ENV: QWEN3_DFLASH_RUNTIME_SHA256,
            SITE_SHA_ENV: site_sha256,
        }
        rewritten = list(environment)
        for index, token in enumerate(rewritten):
            if token.startswith("PYTHONPATH="):
                rewritten[index] = f"PYTHONPATH={destination}:{chained_pythonpath}"
                break
        else:  # pragma: no cover
            raise DFlashContextKVOverlayError("base command lacks PYTHONPATH")
        rewritten.extend(f"{name}={value}" for name, value in additions.items())
        command_payload = (
            shlex.join(["exec", "/usr/bin/env", "-i", *rewritten, *argv]) + "\n"
        ).encode()
        manifest = {
            "schema": SCHEMA,
            "classification": "non_promotable_serial_context_kv_candidate",
            "promotable": False,
            "base_command_sha256": args.expected_base_sha256,
            "chained_site": {
                "path": str(chained_site),
                "sha256": chained_sha256,
                "authenticated_by_environment": chained_claims,
            },
            "command_sha256": _digest(command_payload),
            "qwen3_dflash_runtime_sha256": QWEN3_DFLASH_RUNTIME_SHA256,
            "site_sha256": site_sha256,
            "patch_contract": {
                "module": TARGET_MODULE,
                "method": "DFlashQwen3Model.precompute_and_store_context_kv",
                "scope": (
                    "M2..M8 accepted-context hidden RMSNorm, W4A16 KV projection, "
                    "K RMSNorm, RoPE, and cache writes execute as ordered M1 calls"
                ),
            },
        }
        manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        _write_exclusive(staging / "sitecustomize.py", site_payload, 0o600)
        _write_exclusive(staging / "command.sh", command_payload, 0o700)
        _write_exclusive(staging / "manifest.json", manifest_payload, 0o600)
        if destination.exists() or destination.is_symlink():
            raise DFlashContextKVOverlayError(
                "destination appeared during create-only rendering"
            )
        staging.rename(destination)
        return {**manifest, "manifest_sha256": _digest(manifest_payload)}
    except BaseException:
        if staging.exists() and staging.parent == destination.parent:
            shutil.rmtree(staging)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen-dflash-context-kv-serial-overlay")
    parser.add_argument("--base-command", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = render(_parser().parse_args(argv))
    except (OSError, ValueError, DFlashContextKVOverlayError) as error:
        print(f"qwen-dflash-context-kv-serial-overlay: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
