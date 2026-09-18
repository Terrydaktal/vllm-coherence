"""Render a create-only authenticated full-state diagnostic overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCHEMA = "urn:qwen-r9700:full-state-diagnostic-overlay:v1"
ROOT = Path(__file__).parents[2]
CAPTURE_SOURCE = ROOT / "experiments/full-state-diagnostic/runtime_state_capture.py"
LIFECYCLE_SOURCE = ROOT / "experiments/full-state-diagnostic/kv_lifecycle_capture.py"
EXPORTER_SOURCE = ROOT / "experiments/dflash-lossless-assurance/coding_turbo_state_exporter.py"
PROVIDER_SOURCE = ROOT / "src/qwen_r9700_lab/coding_turbo_state_provider.py"
RUNTIME_ROOT = Path(
    "/home/lewis/.local/share/qwen-r9700/runtimes/vllm-rocm/.venv/lib/python3.12/site-packages/vllm"
)


class FullStateOverlayError(RuntimeError):
    """The requested overlay violated an identity or safety contract."""


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable(path: Path, label: str, *, private: bool = False) -> bytes:
    path = path.expanduser().absolute()
    before = path.lstat()
    payload = path.read_bytes()
    after = path.lstat()
    def identity(value: os.stat_result) -> tuple[int, int, int, int]:
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns
    if (
        identity(before) != identity(after)
        or not stat.S_ISREG(before.st_mode)
        or path.is_symlink()
        or before.st_uid != os.getuid()
        or (private and stat.S_IMODE(before.st_mode) & 0o077)
    ):
        raise FullStateOverlayError(f"{label} must be one stable owned regular file")
    return payload


def _split(payload: bytes) -> tuple[list[str], list[str]]:
    try:
        tokens = shlex.split(payload.decode())
    except (UnicodeDecodeError, ValueError) as error:
        raise FullStateOverlayError(f"base command is invalid: {error}") from error
    if tokens[:3] != ["exec", "/usr/bin/env", "-i"]:
        raise FullStateOverlayError("base command must begin with exec /usr/bin/env -i")
    index = 3
    environment: list[str] = []
    while index < len(tokens) and "=" in tokens[index]:
        environment.append(tokens[index])
        index += 1
    names = [token.split("=", 1)[0] for token in environment]
    if not environment or len(names) != len(set(names)) or index == len(tokens):
        raise FullStateOverlayError("base command has invalid environment/argv boundaries")
    return environment, tokens[index:]


def _model_paths(argv: list[str]) -> tuple[Path, Path]:
    try:
        serve = argv.index("serve")
        spec_index = argv.index("--speculative-config")
        spec = json.loads(argv[spec_index + 1])
    except (ValueError, IndexError, json.JSONDecodeError) as error:
        raise FullStateOverlayError("base command lacks model/DFlash identities") from error
    if (
        serve + 1 >= len(argv)
        or not isinstance(spec, dict)
        or not isinstance(spec.get("model"), str)
    ):
        raise FullStateOverlayError("base command model identities are invalid")
    target = Path(argv[serve + 1])
    draft = Path(spec["model"])
    if not target.is_absolute() or not draft.is_absolute():
        raise FullStateOverlayError("model identities must be absolute")
    return target, draft


def _site_source(
    destination: Path,
    *,
    chained_site: Path,
    chained_sha256: str,
    chained_pythonpath: str,
    capture_sha256: str,
    lifecycle_sha256: str,
    exporter_sha256: str,
    provider_sha256: str,
) -> bytes:
    outer = f"{destination}:{chained_pythonpath}"
    return f'''"""Authenticated post-proposal full-state diagnostic bootstrap."""
import hashlib
import os
import runpy
import stat
from pathlib import Path

_ROOT = Path({str(destination)!r})
_SELF = _ROOT / "sitecustomize.py"
_CAPTURE = _ROOT / "runtime_state_capture.py"
_LIFECYCLE = _ROOT / "kv_lifecycle_capture.py"
_EXPORTER = _ROOT / "runtime_state_exporter.py"
_PROVIDER = _ROOT / "qwen_r9700_lab" / "coding_turbo_state_provider.py"
_CHAIN = Path({str(chained_site)!r})
_CHAIN_SHA = {chained_sha256!r}
_CHAIN_PYTHONPATH = {chained_pythonpath!r}
_OUTER_PYTHONPATH = {outer!r}
_EXPECTED = {{
    _CAPTURE: {capture_sha256!r},
    _LIFECYCLE: {lifecycle_sha256!r},
    _EXPORTER: {exporter_sha256!r},
    _PROVIDER: {provider_sha256!r},
}}

def _digest(path):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink() or before.st_uid != os.getuid():
        raise RuntimeError("diagnostic source identity is unsafe")
    payload = path.read_bytes()
    after = path.lstat()
    before_id = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_id = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_id != after_id:
        raise RuntimeError("diagnostic source changed during authentication")
    return hashlib.sha256(payload).hexdigest()

if os.environ.get("QWEN_FULL_STATE_DIAGNOSTIC") != "1":
    raise RuntimeError("full-state diagnostic identity is absent")
if os.environ.get("PYTHONPATH") != _OUTER_PYTHONPATH:
    raise RuntimeError("full-state diagnostic PYTHONPATH differs")
if _digest(_SELF) != os.environ.get("QWEN_FULL_STATE_DIAGNOSTIC_SITE_SHA256"):
    raise RuntimeError("full-state diagnostic site SHA256 mismatch")
for _path, _sha in _EXPECTED.items():
    if _digest(_path) != _sha:
        raise RuntimeError("full-state diagnostic member SHA256 mismatch")
if _digest(_CHAIN) != _CHAIN_SHA:
    raise RuntimeError("full-state diagnostic chained site SHA256 mismatch")
os.environ["PYTHONPATH"] = _CHAIN_PYTHONPATH
try:
    runpy.run_path(str(_CHAIN), run_name="_qwen_full_state_diagnostic_chain")
    if os.environ.get("PYTHONPATH") != _CHAIN_PYTHONPATH:
        raise RuntimeError("chained site changed PYTHONPATH")
finally:
    os.environ["PYTHONPATH"] = _OUTER_PYTHONPATH
import runtime_state_capture  # noqa: E402,F401
import kv_lifecycle_capture  # noqa: E402,F401
'''.encode()


def render(args: argparse.Namespace) -> dict[str, Any]:
    destination = args.destination.expanduser().absolute()
    output_root = args.output_root.expanduser().absolute()
    if destination.exists() or destination.is_symlink() or output_root.exists():
        raise FullStateOverlayError("overlay and output destinations are create-only")
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_base_sha256):
        raise FullStateOverlayError("expected base SHA256 is malformed")
    base = _stable(args.base_command, "base command", private=True)
    if _digest(base) != args.expected_base_sha256:
        raise FullStateOverlayError("base command SHA256 mismatch")
    environment, argv = _split(base)
    env = dict(token.split("=", 1) for token in environment)
    if env.get("QWEN_DFLASH_GREEDY_LOOP_ESCAPE") != "0":
        raise FullStateOverlayError("state diagnostic requires loop escape disabled")
    target_only = env.get("QWEN_FIXED_SLOT_TARGET_ONLY_ORACLE") == "1"
    if target_only != args.target_only:
        expected = "target-only" if args.target_only else "DFlash-active"
        raise FullStateOverlayError(f"state diagnostic requires an exact {expected} base")
    pythonpath = env.get("PYTHONPATH", "")
    if not pythonpath.startswith("/"):
        raise FullStateOverlayError("base command lacks absolute PYTHONPATH")
    chained_site = Path(pythonpath.split(":", 1)[0]) / "sitecustomize.py"
    chained = _stable(chained_site, "chained sitecustomize", private=True)
    target_model, draft_model = _model_paths(argv)
    target_config = target_model / "config.json"
    draft_config = draft_model / "config.json"
    attention = RUNTIME_ROOT / "model_executor/layers/attention/attention.py"
    dflash = RUNTIME_ROOT / "model_executor/models/qwen3_dflash.py"
    bound_sources = {
        "target_config": target_config,
        "draft_config": draft_config,
        "attention": attention,
        "dflash": dflash,
    }
    bound_payloads = {name: _stable(path, name) for name, path in bound_sources.items()}
    counts = tuple(args.counts)
    if (
        not counts
        or tuple(sorted(set(counts))) != counts
        or any(not 1 <= count <= 4096 for count in counts)
    ):
        raise FullStateOverlayError("counts must be ordered unique integers in 1..4096")
    if (
        args.prompt_tokens <= 0
        or not args.request_id
        or any(ch.isspace() for ch in args.request_id)
    ):
        raise FullStateOverlayError("prompt/request identity is invalid")

    staging = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
    staging.mkdir(mode=0o700, parents=True)
    try:
        capture = _stable(CAPTURE_SOURCE, "capture source")
        lifecycle = _stable(LIFECYCLE_SOURCE, "KV lifecycle capture source")
        exporter = _stable(EXPORTER_SOURCE, "exporter source")
        provider = _stable(PROVIDER_SOURCE, "provider source")
        (staging / "runtime_state_capture.py").write_bytes(capture)
        (staging / "kv_lifecycle_capture.py").write_bytes(lifecycle)
        (staging / "runtime_state_exporter.py").write_bytes(exporter)
        package = staging / "qwen_r9700_lab"
        package.mkdir(mode=0o700)
        # This diagnostic contributes one module to the already populated
        # qwen_r9700_lab package.  A plain __init__.py here would shadow the
        # runtime's authenticated package roots and hide modules needed by the
        # DFlash speculator.  Extend the package path explicitly so the copied
        # provider and every downstream runtime module remain importable.
        namespace_init = (
            b"from pkgutil import extend_path\n"
            b"__path__ = extend_path(__path__, __name__)\n"
        )
        (package / "__init__.py").write_bytes(namespace_init)
        (package / "coding_turbo_state_provider.py").write_bytes(provider)
        for path in (
            staging / "runtime_state_capture.py",
            staging / "kv_lifecycle_capture.py",
            staging / "runtime_state_exporter.py",
            package / "__init__.py",
            package / "coding_turbo_state_provider.py",
        ):
            path.chmod(0o600)
        site = _site_source(
            destination,
            chained_site=chained_site,
            chained_sha256=_digest(chained),
            chained_pythonpath=pythonpath,
            capture_sha256=_digest(capture),
            lifecycle_sha256=_digest(lifecycle),
            exporter_sha256=_digest(exporter),
            provider_sha256=_digest(provider),
        )
        site_sha = _digest(site)
        (staging / "sitecustomize.py").write_bytes(site)
        (staging / "sitecustomize.py").chmod(0o600)
        output = output_root / "states.jsonl"
        additions = {
            "PYTHONPATH": f"{destination}:{pythonpath}",
            "QWEN_FULL_STATE_DIAGNOSTIC": "1",
            "QWEN_FULL_STATE_DIAGNOSTIC_ATTENTION_SHA256": _digest(bound_payloads["attention"]),
            "QWEN_FULL_STATE_DIAGNOSTIC_ATTENTION_SOURCE": str(attention),
            "QWEN_FULL_STATE_DIAGNOSTIC_COUNTS": ",".join(str(value) for value in counts),
            "QWEN_FULL_STATE_DIAGNOSTIC_DFLASH_SHA256": _digest(bound_payloads["dflash"]),
            "QWEN_FULL_STATE_DIAGNOSTIC_DFLASH_SOURCE": str(dflash),
            "QWEN_FULL_STATE_DIAGNOSTIC_DRAFT_CONFIG": str(draft_config),
            "QWEN_FULL_STATE_DIAGNOSTIC_DRAFT_CONFIG_SHA256": _digest(
                bound_payloads["draft_config"]
            ),
            "QWEN_FULL_STATE_DIAGNOSTIC_EXPORTER": str(destination / "runtime_state_exporter.py"),
            "QWEN_FULL_STATE_DIAGNOSTIC_EXPORTER_SHA256": _digest(exporter),
            "QWEN_FULL_STATE_DIAGNOSTIC_OUTPUT": str(output),
            "QWEN_FULL_STATE_DIAGNOSTIC_LIFECYCLE_OUTPUT_ROOT": str(output_root),
            "QWEN_FULL_STATE_DIAGNOSTIC_PROMPT_TOKENS": str(args.prompt_tokens),
            "QWEN_FULL_STATE_DIAGNOSTIC_PROVIDER": str(
                destination / "qwen_r9700_lab/coding_turbo_state_provider.py"
            ),
            "QWEN_FULL_STATE_DIAGNOSTIC_PROVIDER_SHA256": _digest(provider),
            "QWEN_FULL_STATE_DIAGNOSTIC_REQUEST_ID": args.request_id,
            "QWEN_FULL_STATE_DIAGNOSTIC_SITE_SHA256": site_sha,
            "QWEN_FULL_STATE_DIAGNOSTIC_TARGET_CONFIG": str(target_config),
            "QWEN_FULL_STATE_DIAGNOSTIC_TARGET_CONFIG_SHA256": _digest(
                bound_payloads["target_config"]
            ),
            "QWEN_FULL_STATE_DIAGNOSTIC_WIDTHS_ONLY": "1" if args.widths_only else "0",
        }
        names = set(additions)
        rewritten = [token for token in environment if token.split("=", 1)[0] not in names]
        rewritten.extend(f"{name}={value}" for name, value in additions.items())
        command = (shlex.join(["exec", "/usr/bin/env", "-i", *rewritten, *argv]) + "\n").encode()
        (staging / "command.sh").write_bytes(command)
        (staging / "command.sh").chmod(0o700)
        manifest = {
            "schema": SCHEMA,
            "classification": "non_promotable_complete_state_diagnostic",
            "promotable": False,
            "base_command_sha256": args.expected_base_sha256,
            "bound_sources": {
                name: {"path": str(path), "sha256": _digest(bound_payloads[name])}
                for name, path in bound_sources.items()
            },
            "chained_site": {"path": str(chained_site), "sha256": _digest(chained)},
            "command_sha256": _digest(command),
            "counts": list(counts),
            "members": {
                "capture": _digest(capture),
                "kv_lifecycle_capture": _digest(lifecycle),
                "exporter": _digest(exporter),
                "provider": _digest(provider),
                "site": site_sha,
            },
            "output": str(output),
            "prompt_tokens": args.prompt_tokens,
            "request_id": args.request_id,
            "target_only": target_only,
            "widths_only": args.widths_only,
        }
        manifest_payload = _canonical(manifest)
        (staging / "manifest.json").write_bytes(manifest_payload)
        (staging / "manifest.json").chmod(0o600)
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        staging.rename(destination)
        output_root.mkdir(mode=0o700, parents=True)
        return {**manifest, "manifest_sha256": _digest(manifest_payload)}
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen-full-state-diagnostic-overlay")
    parser.add_argument("--base-command", required=True, type=Path)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--prompt-tokens", required=True, type=int)
    parser.add_argument("--request-id", required=True)
    parser.add_argument(
        "--counts", required=True, type=lambda value: tuple(int(item) for item in value.split(","))
    )
    parser.add_argument("--widths-only", action="store_true")
    parser.add_argument("--target-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = render(_parser().parse_args(argv))
    except (FullStateOverlayError, OSError, ValueError) as error:
        print(f"qwen-full-state-diagnostic-overlay: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
