"""Render a create-only, authenticated vLLM bounded-loop-escape shadow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qwen_r9700_lab.repetition_escape import (
    DEFAULT_MAX_ESCAPES,
    DEFAULT_MIN_REPEATED_TOKENS,
)

SCHEMA = "urn:qwen-r9700:bounded-repetition-escape-overlay:v2"
MODEL_RUNNER_MODULE = "vllm.v1.worker.gpu.model_runner"
ASYNC_UTILS_MODULE = "vllm.v1.worker.gpu.async_utils"
ENABLE_ENV = "QWEN_BOUNDED_REPETITION_ESCAPE"
LEGACY_ENV = "QWEN_DFLASH_GREEDY_LOOP_ESCAPE"
TELEMETRY_ENV = "QWEN_BOUNDED_REPETITION_ESCAPE_TELEMETRY"
EOS_IDS_ENV = "QWEN_BOUNDED_REPETITION_ESCAPE_EOS_TOKEN_IDS"
DEFAULT_EOS_IDS = "248044,248046"

MODEL_IMPORT_ANCHOR = b"""from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler
"""
MODEL_IMPORT_REPLACEMENT = (
    MODEL_IMPORT_ANCHOR
    + b"from qwen_bounded_loop_escape_runtime import (\n"
    + b"    maybe_apply_bounded_loop_escape,\n"
    + b"    register_bounded_loop_escape_request,\n"
    + b")\n"
)
MODEL_REGISTER_ANCHOR = b"""                self.sampler.add_request(
                    req_index, prompt_len, new_req_data.sampling_params
                )
"""
MODEL_REGISTER_REPLACEMENT = (
    MODEL_REGISTER_ANCHOR
    + b"""                register_bounded_loop_escape_request(
                    sampler=self.sampler,
                    req_index=req_index,
                    sampling_params=new_req_data.sampling_params,
                )
"""
)
MODEL_SAMPLE_ANCHOR = (
    b"        return sampler_output, sampler_output.num_sampled, "
    b"sampler_output.num_rejected\n"
)
MODEL_SAMPLE_REPLACEMENT = b"""        maybe_apply_bounded_loop_escape(
            logits=logits,
            input_batch=input_batch,
            sampler=self.sampler,
            sampler_output=sampler_output,
        )
        return sampler_output, sampler_output.num_sampled, sampler_output.num_rejected
"""

ASYNC_COPY_ANCHOR = (
    b"            self.sampled_token_ids = "
    b"async_copy_to_np(sampler_output.sampled_token_ids)\n"
)
ASYNC_COPY_REPLACEMENT = ASYNC_COPY_ANCHOR + b"""            qwen_events = getattr(
                sampler_output, "qwen_loop_escape_events", None
            )
            self.qwen_loop_escape_events = (
                async_copy_to_np(qwen_events) if qwen_events is not None else None
            )
"""
ASYNC_RETURN_ANCHOR = b"""        return self.model_runner_output


class AsyncPoolingOutput"""
ASYNC_RETURN_REPLACEMENT = b"""        if self.qwen_loop_escape_events is not None:
            from qwen_bounded_loop_escape_runtime import record_loop_escape_events

            record_loop_escape_events(
                self.model_runner_output.req_ids,
                self.qwen_loop_escape_events,
            )
        return self.model_runner_output


class AsyncPoolingOutput"""


class EscapeOverlayError(RuntimeError):
    """The base sources, command, or output violated the overlay contract."""


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _replace_once(source: bytes, old: bytes, new: bytes, label: str) -> bytes:
    if source.count(old) != 1:
        raise EscapeOverlayError(f"{label} preimage is not unique")
    rewritten = source.replace(old, new)
    expected_size = len(source) - len(old) + len(new)
    if len(rewritten) != expected_size or rewritten.count(new) != 1:
        raise EscapeOverlayError(f"{label} postimage is invalid")
    return rewritten


def rewrite_model_runner(source: bytes, *, expected_sha256: str) -> bytes:
    if digest(source) != expected_sha256:
        raise EscapeOverlayError("model-runner source SHA256 differs")
    rewritten = _replace_once(
        source,
        MODEL_IMPORT_ANCHOR,
        MODEL_IMPORT_REPLACEMENT,
        "model-runner import",
    )
    rewritten = _replace_once(
        rewritten,
        MODEL_REGISTER_ANCHOR,
        MODEL_REGISTER_REPLACEMENT,
        "model-runner request stopping contract",
    )
    rewritten = _replace_once(
        rewritten,
        MODEL_SAMPLE_ANCHOR,
        MODEL_SAMPLE_REPLACEMENT,
        "model-runner sample hook",
    )
    compile(rewritten, "model_runner.py", "exec")
    return rewritten


def rewrite_async_utils(source: bytes, *, expected_sha256: str) -> bytes:
    if digest(source) != expected_sha256:
        raise EscapeOverlayError("async-utils source SHA256 differs")
    rewritten = _replace_once(
        source,
        ASYNC_COPY_ANCHOR,
        ASYNC_COPY_REPLACEMENT,
        "async event copy",
    )
    rewritten = _replace_once(
        rewritten,
        ASYNC_RETURN_ANCHOR,
        ASYNC_RETURN_REPLACEMENT,
        "async event record",
    )
    compile(rewritten, "async_utils.py", "exec")
    return rewritten


def _split_command(payload: bytes) -> tuple[list[str], list[str]]:
    try:
        tokens = shlex.split(payload.decode())
    except (UnicodeDecodeError, ValueError) as error:
        raise EscapeOverlayError(f"base command is invalid: {error}") from error
    if tokens[:3] != ["exec", "/usr/bin/env", "-i"]:
        raise EscapeOverlayError("base command must use exec /usr/bin/env -i")
    index = 3
    environment: list[str] = []
    while index < len(tokens) and "=" in tokens[index]:
        environment.append(tokens[index])
        index += 1
    if not environment or index == len(tokens):
        raise EscapeOverlayError("base command lacks environment or server argv")
    return environment, tokens[index:]


def _environment_map(environment: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in environment:
        name, value = token.split("=", 1)
        if not name or name in result:
            raise EscapeOverlayError("base command has an invalid environment")
        result[name] = value
    return result


def _stable_private_file(path: Path, label: str) -> bytes:
    before = path.lstat()
    payload = path.read_bytes()
    after = path.lstat()

    def identity(value: os.stat_result) -> tuple[int, int, int, int]:
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns

    if identity(before) != identity(after) or not path.is_file() or path.is_symlink():
        raise EscapeOverlayError(f"{label} identity changed")
    return payload


def _validate_telemetry_path(path: Path) -> None:
    if not path.is_absolute():
        raise EscapeOverlayError("telemetry path must be absolute")
    parent = path.parent
    try:
        metadata = parent.lstat()
    except OSError as error:
        raise EscapeOverlayError(f"telemetry parent is unavailable: {error}") from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or parent.is_symlink()
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise EscapeOverlayError("telemetry parent must be a private owned real directory")
    if path.exists() or path.is_symlink():
        file_metadata = path.lstat()
        if (
            not stat.S_ISREG(file_metadata.st_mode)
            or path.is_symlink()
            or file_metadata.st_uid != os.getuid()
            or stat.S_IMODE(file_metadata.st_mode) & 0o077
        ):
            raise EscapeOverlayError("telemetry target must be a private owned regular file")


def render_site(
    destination: Path,
    *,
    model_runner_sha256: str,
    async_utils_sha256: str,
    runtime_sha256: str,
    chained_site: Path,
    chained_site_sha256: str,
    chained_pythonpath: str,
) -> bytes:
    modules = {
        MODEL_RUNNER_MODULE: "vllm/v1/worker/gpu/model_runner.py",
        ASYNC_UTILS_MODULE: "vllm/v1/worker/gpu/async_utils.py",
    }
    return f'''"""Authenticated bounded repetition-escape import shadow."""
import hashlib
import importlib.abc
import importlib.util
import os
import runpy
import sys
from pathlib import Path

_ROOT = Path({str(destination)!r})
_CHAIN = Path({str(chained_site)!r})
_CHAIN_SHA256 = {chained_site_sha256!r}
_CHAIN_PYTHONPATH = {chained_pythonpath!r}
_OUTER_PYTHONPATH = {f"{destination}:{chained_pythonpath}"!r}
_FILES = {{
    "sitecustomize.py": None,
    "qwen_bounded_loop_escape_runtime.py": {runtime_sha256!r},
    "vllm/v1/worker/gpu/model_runner.py": {model_runner_sha256!r},
    "vllm/v1/worker/gpu/async_utils.py": {async_utils_sha256!r},
}}
_MODULES = {modules!r}

def _stable_digest(path):
    before = path.lstat()
    payload = path.read_bytes()
    after = path.lstat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if identity(before) != identity(after) or not path.is_file() or path.is_symlink():
        raise RuntimeError("bounded-loop-escape file identity changed")
    return hashlib.sha256(payload).hexdigest()

if os.environ.get({ENABLE_ENV!r}, "0") not in {{"0", "1"}}:
    raise RuntimeError("bounded repetition escape activation must be 0 or 1")
if os.environ.get({LEGACY_ENV!r}, "0") != "0":
    raise RuntimeError("unsafe legacy greedy-loop escape must remain disabled")
if os.environ.get("PYTHONPATH") != _OUTER_PYTHONPATH:
    raise RuntimeError("bounded-loop-escape PYTHONPATH mismatch")
if _stable_digest(Path(__file__).resolve()) != os.environ.get(
    "QWEN_BOUNDED_REPETITION_ESCAPE_SITE_SHA256"
):
    raise RuntimeError("bounded-loop-escape site SHA256 mismatch")
for relative, expected in _FILES.items():
    if expected is not None and _stable_digest(_ROOT / relative) != expected:
        raise RuntimeError("bounded-loop-escape member SHA256 mismatch: " + relative)
if _stable_digest(_CHAIN) != _CHAIN_SHA256:
    raise RuntimeError("bounded-loop-escape chained site SHA256 mismatch")
os.environ["PYTHONPATH"] = _CHAIN_PYTHONPATH
runpy.run_path(str(_CHAIN), run_name="_qwen_bounded_loop_escape_chained_sitecustomize")
if os.environ.get("PYTHONPATH") != _CHAIN_PYTHONPATH:
    raise RuntimeError("bounded-loop-escape chained site changed PYTHONPATH")
os.environ["PYTHONPATH"] = _OUTER_PYTHONPATH
for module in _MODULES:
    if module in sys.modules:
        raise RuntimeError(module + " imported before bounded-loop-escape shadow")

class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        relative = _MODULES.get(fullname)
        if relative is None:
            return None
        return importlib.util.spec_from_file_location(fullname, _ROOT / relative)

sys.meta_path.insert(0, _Finder())
print("[qwen-bounded-loop-escape] authenticated post-verification escape armed", flush=True)
'''.encode()


def rewrite_command(
    payload: bytes,
    *,
    destination: Path,
    site_sha256: str,
    telemetry: Path,
    eos_ids: str,
) -> bytes:
    environment, argv = _split_command(payload)
    values = _environment_map(environment)
    pythonpath = values.get("PYTHONPATH", "")
    if not pythonpath.startswith("/"):
        raise EscapeOverlayError("base command lacks an absolute PYTHONPATH")
    if not telemetry.is_absolute():
        raise EscapeOverlayError("telemetry path must be absolute")
    if ENABLE_ENV in values or TELEMETRY_ENV in values or EOS_IDS_ENV in values:
        raise EscapeOverlayError("base command already contains bounded-loop-escape state")
    additions = {
        ENABLE_ENV: "1",
        LEGACY_ENV: "0",
        TELEMETRY_ENV: str(telemetry),
        EOS_IDS_ENV: eos_ids,
        "QWEN_BOUNDED_REPETITION_ESCAPE_SITE_SHA256": site_sha256,
    }
    output: list[str] = []
    added = False
    for token in environment:
        name, value = token.split("=", 1)
        if name == "PYTHONPATH":
            output.extend(f"{key}={item}" for key, item in additions.items())
            value = f"{destination}:{value}"
            added = True
        elif name == LEGACY_ENV:
            continue
        output.append(f"{name}={value}")
    if not added:
        raise EscapeOverlayError("PYTHONPATH insertion failed")
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
    model_runner_source: Path,
    expected_model_runner_sha256: str,
    async_utils_source: Path,
    expected_async_utils_sha256: str,
    base_command: Path,
    expected_base_sha256: str,
    runtime_source: Path,
    destination: Path,
    telemetry: Path,
    eos_ids: str = DEFAULT_EOS_IDS,
) -> Mapping[str, Any]:
    if destination.exists():
        raise EscapeOverlayError("destination is create-only")
    _validate_telemetry_path(telemetry)
    base = _stable_private_file(base_command, "base command")
    if digest(base) != expected_base_sha256:
        raise EscapeOverlayError("base command SHA256 differs")
    environment, _argv = _split_command(base)
    values = _environment_map(environment)
    chained_pythonpath = values.get("PYTHONPATH", "")
    if not chained_pythonpath.startswith("/"):
        raise EscapeOverlayError("base command lacks an authenticated PYTHONPATH")
    chained_site = Path(chained_pythonpath.split(":", 1)[0]) / "sitecustomize.py"
    chained_site_payload = _stable_private_file(chained_site, "chained sitecustomize")

    model_source = _stable_private_file(model_runner_source, "model runner")
    async_source = _stable_private_file(async_utils_source, "async utils")
    runtime = _stable_private_file(runtime_source, "loop escape runtime")
    model_runner = rewrite_model_runner(
        model_source, expected_sha256=expected_model_runner_sha256
    )
    async_utils = rewrite_async_utils(
        async_source, expected_sha256=expected_async_utils_sha256
    )
    site = render_site(
        destination,
        model_runner_sha256=digest(model_runner),
        async_utils_sha256=digest(async_utils),
        runtime_sha256=digest(runtime),
        chained_site=chained_site,
        chained_site_sha256=digest(chained_site_payload),
        chained_pythonpath=chained_pythonpath,
    )
    command = rewrite_command(
        base,
        destination=destination,
        site_sha256=digest(site),
        telemetry=telemetry,
        eos_ids=eos_ids,
    )
    destination.mkdir(mode=0o700, parents=True)
    _write(destination / "vllm/v1/worker/gpu/model_runner.py", model_runner, 0o600)
    _write(destination / "vllm/v1/worker/gpu/async_utils.py", async_utils, 0o600)
    _write(destination / "qwen_bounded_loop_escape_runtime.py", runtime, 0o600)
    _write(destination / "sitecustomize.py", site, 0o600)
    _write(destination / "command.sh", command, 0o700)
    manifest = {
        "schema": SCHEMA,
        "base_command_sha256": digest(base),
        "command_sha256": digest(command),
        "source_model_runner_sha256": digest(model_source),
        "model_runner_sha256": digest(model_runner),
        "source_async_utils_sha256": digest(async_source),
        "async_utils_sha256": digest(async_utils),
        "runtime_sha256": digest(runtime),
        "site_sha256": digest(site),
        "telemetry": str(telemetry),
        "eos_token_ids": [int(token) for token in eos_ids.split(",")],
        "contract": {
            "legacy_seven-copy_guard_disabled": True,
            "differential_replay_requires_escape_zero": True,
            "post_target_verification_only": True,
            "normal_eos_untouched": True,
            "custom_stop_or_ignore_eos_requests_untouched": True,
            "sampling_mask_requests_untouched": True,
            "grammar_mask_preserved": True,
            "prior_output_preserved": True,
            "telemetry_failure_isolated_from_serving": True,
            "per_request_escape_budget": DEFAULT_MAX_ESCAPES,
            "minimum_repeated_tokens": DEFAULT_MIN_REPEATED_TOKENS,
        },
    }
    manifest_payload = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    _write(destination / "manifest.json", manifest_payload, 0o600)
    return {**manifest, "manifest_sha256": digest(manifest_payload)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-runner-source", type=Path, required=True)
    parser.add_argument("--expected-model-runner-sha256", required=True)
    parser.add_argument("--async-utils-source", type=Path, required=True)
    parser.add_argument("--expected-async-utils-sha256", required=True)
    parser.add_argument("--base-command", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument(
        "--runtime-source",
        type=Path,
        default=Path(__file__).with_name("vllm_repetition_escape_runtime.py"),
    )
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--eos-token-ids", default=DEFAULT_EOS_IDS)
    args = parser.parse_args(argv)
    try:
        result = render(
            model_runner_source=args.model_runner_source,
            expected_model_runner_sha256=args.expected_model_runner_sha256,
            async_utils_source=args.async_utils_source,
            expected_async_utils_sha256=args.expected_async_utils_sha256,
            base_command=args.base_command,
            expected_base_sha256=args.expected_base_sha256,
            runtime_source=args.runtime_source,
            destination=args.destination,
            telemetry=args.telemetry,
            eos_ids=args.eos_token_ids,
        )
    except (EscapeOverlayError, OSError, ValueError) as error:
        print(f"qwen-repetition-escape-overlay: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
