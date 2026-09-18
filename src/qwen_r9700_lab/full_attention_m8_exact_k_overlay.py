"""Render an authenticated production-M8 exact-K RMSNorm overlay."""

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
    QWEN3_NEXT_RUNTIME_SHA256,
    TARGET_MODULE,
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

SCHEMA = "urn:qwen-r9700:full-attention-m8-exact-k-overlay:v1"
ENABLE_ENV = "QWEN_FULL_ATTENTION_M8_EXACT_K"
REQUIRED_ENV = "QWEN_FULL_ATTENTION_M8_EXACT_K_REQUIRED"
SITE_SHA_ENV = "QWEN_FULL_ATTENTION_M8_EXACT_K_SITE_SHA256"
RUNTIME_SHA_ENV = "QWEN_FULL_ATTENTION_M8_EXACT_K_RUNTIME_SHA256"
EXTENSION_ENV = "QWEN_FULL_ATTENTION_M8_EXACT_K_EXTENSION"
EXTENSION_SHA_ENV = "QWEN_FULL_ATTENTION_M8_EXACT_K_EXTENSION_SHA256"
QUALIFICATION_SHA_ENV = "QWEN_FULL_ATTENTION_M8_EXACT_K_QUALIFICATION_SHA256"
MODULE_NAME = "qwen_exact_k_rmsnorm_v1"
_OVERLAY_ENVIRONMENT = (
    ENABLE_ENV,
    REQUIRED_ENV,
    SITE_SHA_ENV,
    RUNTIME_SHA_ENV,
    EXTENSION_ENV,
    EXTENSION_SHA_ENV,
    QUALIFICATION_SHA_ENV,
)


class ExactKOverlayError(RowExactOverlayError):
    """The exact-K overlay identity or source command is invalid."""


def _site_source(
    destination: Path,
    *,
    chained_site: Path,
    chained_site_sha256: str,
    chained_pythonpath: str,
    runtime_extension: Path,
    extension_sha256: str,
    qualification_sha256: str,
) -> bytes:
    outer_pythonpath = f"{destination}:{chained_pythonpath}"
    return f'''"""Authenticated production-M8 exact-K RMSNorm bootstrap."""
import functools
import hashlib
import importlib.util
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
_RUNTIME_SHA256 = {QWEN3_NEXT_RUNTIME_SHA256!r}
_EXTENSION = Path({str(runtime_extension)!r})
_EXTENSION_SHA256 = {extension_sha256!r}
_QUALIFICATION_SHA256 = {qualification_sha256!r}
_METHOD_MARKER = "_qwen_full_attention_m8_exact_k_site_sha256"

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

if (
    os.environ.get({ENABLE_ENV!r}) != "1"
    or os.environ.get({REQUIRED_ENV!r}) != "1"
):
    raise RuntimeError("full-attention M8 exact-K identity is absent")
if os.environ.get({RUNTIME_SHA_ENV!r}) != _RUNTIME_SHA256:
    raise RuntimeError("full-attention M8 exact-K runtime identity differs")
if os.environ.get({EXTENSION_ENV!r}) != str(_EXTENSION):
    raise RuntimeError("full-attention M8 exact-K extension path differs")
if os.environ.get({EXTENSION_SHA_ENV!r}) != _EXTENSION_SHA256:
    raise RuntimeError("full-attention M8 exact-K extension identity differs")
if os.environ.get({QUALIFICATION_SHA_ENV!r}) != _QUALIFICATION_SHA256:
    raise RuntimeError("full-attention M8 exact-K qualification identity differs")
if os.environ.get("QWEN_DFLASH_GREEDY_LOOP_ESCAPE") != "0":
    raise RuntimeError("full-attention M8 exact-K requires loop escape disabled")
if os.environ.get("QWEN_DFLASH_GREEDY_M8_VERIFIER") != "1":
    raise RuntimeError("full-attention M8 exact-K requires production M8 verification")
if os.environ.get("QWEN_FIXED_SLOT_TARGET_ONLY_ORACLE") != "0":
    raise RuntimeError("full-attention M8 exact-K forbids target-only execution")
if os.environ.get("PYTHONPATH") != _OUTER_PYTHONPATH:
    raise RuntimeError("full-attention M8 exact-K PYTHONPATH differs")
_SITE_SHA256 = _stable_digest(_SELF, "exact-K site", private=True)
if _SITE_SHA256 != os.environ.get({SITE_SHA_ENV!r}):
    raise RuntimeError("full-attention M8 exact-K site SHA256 mismatch")
if _stable_digest(_CHAIN, "chained site", private=True) != _CHAIN_SHA256:
    raise RuntimeError("full-attention M8 exact-K chained-site SHA256 mismatch")
if _stable_digest(_EXTENSION, "exact-K extension") != _EXTENSION_SHA256:
    raise RuntimeError("full-attention M8 exact-K extension SHA256 mismatch")
if _TARGET in sys.modules:
    raise RuntimeError("qwen3_next was imported before exact-K overlay installation")

os.environ["PYTHONPATH"] = _CHAIN_PYTHONPATH
try:
    runpy.run_path(str(_CHAIN), run_name="_qwen_full_attention_m8_exact_k_chain")
    if os.environ.get("PYTHONPATH") != _CHAIN_PYTHONPATH:
        raise RuntimeError("chained site changed PYTHONPATH")
finally:
    os.environ["PYTHONPATH"] = _OUTER_PYTHONPATH
if _TARGET in sys.modules:
    raise RuntimeError("chained site imported qwen3_next before exact-K hook registration")

import combined_runtime_patch as retained  # noqa: E402

spec = importlib.util.spec_from_file_location({MODULE_NAME!r}, _EXTENSION)
if spec is None or spec.loader is None:
    raise RuntimeError("could not construct exact-K extension import")
exact_k_extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exact_k_extension)

finder_type = getattr(retained, "_Finder", None)
patches = getattr(finder_type, "_PATCHES", None)
if not isinstance(finder_type, type) or type(patches) is not dict:
    raise RuntimeError("combined-runtime finder contract changed")
active_finders = [value for value in sys.meta_path if isinstance(value, finder_type)]
if len(active_finders) != 1:
    raise RuntimeError("combined-runtime finder is not installed exactly once")
previous_patch = patches.get(_TARGET)
if previous_patch is not None and not callable(previous_patch):
    raise RuntimeError("existing qwen3_next patch is not callable")
previous_marker = (
    getattr(previous_patch, _METHOD_MARKER, None) if previous_patch is not None else None
)
if previous_marker is not None and previous_marker != _SITE_SHA256:
    raise RuntimeError("a different exact-K patch is already registered")

def _patch_qwen3_next(module):
    source = Path(getattr(module, "__file__", ""))
    if not source.is_absolute() or _stable_digest(source, "qwen3_next runtime") != _RUNTIME_SHA256:
        raise RuntimeError("qwen3_next runtime SHA256 mismatch")
    if getattr(module, "__name__", None) != _TARGET:
        raise RuntimeError("qwen3_next module identity changed")
    if previous_patch is not None:
        previous_patch(module)
    cls = getattr(module, "Qwen3NextAttention", None)
    original = getattr(cls, "_project_qkv_gate", None)
    if not isinstance(cls, type) or not callable(original):
        raise RuntimeError("qwen3_next projection ABI changed")
    if getattr(original, _METHOD_MARKER, None) is not None:
        if getattr(original, _METHOD_MARKER) != _SITE_SHA256:
            raise RuntimeError("qwen3_next carries a conflicting exact-K wrapper")
        return
    torch_module = getattr(module, "torch", None)
    if torch_module is None:
        raise RuntimeError("qwen3_next torch ABI changed")

    @functools.wraps(original)
    def exact_k(self, qkv, positions):
        try:
            qkv_shape = tuple(int(value) for value in qkv.shape)
            position_shape = tuple(int(value) for value in positions.shape)
        except (AttributeError, TypeError, ValueError):
            return original(self, qkv, positions)
        if qkv_shape[0:1] != (8,) or position_shape not in ((8,), (3, 8)):
            return original(self, qkv, positions)
        expected = (24, 4, 256, 6144, 1024, 14336)
        observed = (
            int(self.num_heads), int(self.num_kv_heads), int(self.head_dim),
            int(self.q_size), int(self.kv_size), int(qkv_shape[-1]),
        )
        if observed != expected or len(qkv_shape) != 2:
            raise RuntimeError("full-attention M8 exact-K projection geometry changed")
        if self.use_fused_qk_norm_rope_gate or not self.attn_output_gate:
            raise RuntimeError("full-attention M8 exact-K requires the unfused gated path")

        q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
        orig_shape = q_gate.shape[:-1]
        q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
        q, gate = torch_module.chunk(q_gate, 2, dim=-1)
        q = q.reshape(*orig_shape, -1)
        gate = gate.reshape(*orig_shape, -1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.num_heads * self.head_dim
        )
        k_input = k.view(-1, self.num_kv_heads, self.head_dim)
        variance = exact_k_extension.qwen_variance_256(k_input)
        inverse = torch_module.rsqrt(variance + self.k_norm.variance_epsilon)
        if getattr(module, "GEMMA_RMSNORM_WEIGHT_CACHE_ENABLED", False):
            k_weight = self.k_norm.effective_weight()
        else:
            k_weight = self.k_norm.weight.float() + 1.0
        k = exact_k_extension.qwen_apply_rmsnorm_256(k_input, k_weight, inverse).view(
            -1, self.num_kv_heads * self.head_dim
        )
        rotary_positions = (
            module.qwen_text_rope_positions(positions, self._qwen_attention_prefix)
            if self._qwen_text_only
            else positions
        )
        q, k = self.rotary_emb(rotary_positions, q, k)
        return q, k, v, gate

    setattr(exact_k, _METHOD_MARKER, _SITE_SHA256)
    cls._project_qkv_gate = exact_k
    print("[qwen-full-attention-m8-exact-k] exact K RMSNorm armed", flush=True)

if previous_marker is None:
    setattr(_patch_qwen3_next, _METHOD_MARKER, _SITE_SHA256)
    patches[_TARGET] = _patch_qwen3_next
elif patches.get(_TARGET) is not previous_patch:
    raise RuntimeError("exact-K finder patch identity changed during installation")
'''.encode()


def render(args: argparse.Namespace) -> Mapping[str, Any]:
    destination = args.destination.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise ExactKOverlayError("destination is create-only")
    for value, label in (
        (args.expected_base_sha256, "base command"),
        (args.expected_extension_sha256, "extension"),
        (args.expected_qualification_sha256, "qualification"),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ExactKOverlayError(f"expected {label} SHA256 is malformed")
    base_payload = _stable_private_file(args.base_command, "base command")
    if _digest(base_payload) != args.expected_base_sha256:
        raise ExactKOverlayError("base command SHA256 mismatch")
    extension_payload = _stable_private_file(args.extension_evidence, "extension evidence")
    if _digest(extension_payload) != args.expected_extension_sha256:
        raise ExactKOverlayError("extension evidence SHA256 mismatch")
    qualification_payload = _stable_private_file(args.qualification, "qualification evidence")
    if _digest(qualification_payload) != args.expected_qualification_sha256:
        raise ExactKOverlayError("qualification evidence SHA256 mismatch")
    qualification = json.loads(qualification_payload)
    if (
        qualification.get("verdict") != "pass"
        or qualification.get("library_sha256") != args.expected_extension_sha256
    ):
        raise ExactKOverlayError("qualification does not approve this extension")

    environment, argv = _split_command(base_payload)
    env = _environment_map(environment)
    if any(name in env for name in _OVERLAY_ENVIRONMENT):
        raise ExactKOverlayError("base command already contains exact-K overlay state")
    _validate_production_m8(env, argv)
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
        raise ExactKOverlayError("destination parent must be one owned private directory")
    staging = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise ExactKOverlayError("staging destination already exists")
    staging.mkdir(mode=0o700)
    try:
        site_payload = _site_source(
            destination,
            chained_site=chained_site,
            chained_site_sha256=chained_sha256,
            chained_pythonpath=chained_pythonpath,
            runtime_extension=args.runtime_extension,
            extension_sha256=args.expected_extension_sha256,
            qualification_sha256=args.expected_qualification_sha256,
        )
        site_sha256 = _digest(site_payload)
        additions = {
            ENABLE_ENV: "1",
            REQUIRED_ENV: "1",
            RUNTIME_SHA_ENV: QWEN3_NEXT_RUNTIME_SHA256,
            SITE_SHA_ENV: site_sha256,
            EXTENSION_ENV: str(args.runtime_extension),
            EXTENSION_SHA_ENV: args.expected_extension_sha256,
            QUALIFICATION_SHA_ENV: args.expected_qualification_sha256,
        }
        rewritten = list(environment)
        for index, token in enumerate(rewritten):
            if token.startswith("PYTHONPATH="):
                rewritten[index] = f"PYTHONPATH={destination}:{chained_pythonpath}"
                break
        else:  # pragma: no cover
            raise ExactKOverlayError("base command lacks PYTHONPATH")
        rewritten.extend(f"{name}={value}" for name, value in additions.items())
        command_payload = (
            shlex.join(["exec", "/usr/bin/env", "-i", *rewritten, *argv]) + "\n"
        ).encode()
        manifest = {
            "schema": SCHEMA,
            "classification": "non_promotable_exact_k_candidate",
            "promotable": False,
            "base_command_sha256": args.expected_base_sha256,
            "chained_site": {
                "path": str(chained_site),
                "sha256": chained_sha256,
                "authenticated_by_environment": chained_claims,
            },
            "command_sha256": _digest(command_payload),
            "extension": {
                "runtime_path": str(args.runtime_extension),
                "sha256": args.expected_extension_sha256,
            },
            "qualification_sha256": args.expected_qualification_sha256,
            "qwen3_next_runtime_sha256": QWEN3_NEXT_RUNTIME_SHA256,
            "site_sha256": site_sha256,
            "patch_contract": {
                "module": TARGET_MODULE,
                "method": "Qwen3NextAttention._project_qkv_gate",
                "scope": "only M8 K RMSNorm; original Q/V/gate/RoPE retained",
            },
        }
        manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        _write_exclusive(staging / "sitecustomize.py", site_payload, 0o600)
        _write_exclusive(staging / "command.sh", command_payload, 0o700)
        _write_exclusive(staging / "manifest.json", manifest_payload, 0o600)
        if destination.exists() or destination.is_symlink():
            raise ExactKOverlayError("destination appeared during create-only rendering")
        staging.rename(destination)
        return {**manifest, "manifest_sha256": _digest(manifest_payload)}
    except BaseException:
        if staging.exists() and staging.parent == destination.parent:
            shutil.rmtree(staging)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen-full-attention-m8-exact-k-overlay")
    parser.add_argument("--base-command", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--extension-evidence", type=Path, required=True)
    parser.add_argument("--runtime-extension", type=Path, required=True)
    parser.add_argument("--expected-extension-sha256", required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--expected-qualification-sha256", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = render(_parser().parse_args(argv))
    except (OSError, ValueError, ExactKOverlayError) as error:
        print(f"qwen-full-attention-m8-exact-k-overlay: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
