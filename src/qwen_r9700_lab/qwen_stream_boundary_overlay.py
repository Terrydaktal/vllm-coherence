"""Render an authenticated Qwen streaming-boundary parser repair.

vLLM's token-ID scanner can defer a recognized special token when the
detokenizer delta does not contain the token's decoded terminal.  That is
necessary for mixed chunks, where ordinary text may precede the special token,
but it is incorrect for a singleton special-token delta: the token ID itself
fixes both the boundary and its ordering.  Deferring that singleton can make
parser semantics depend on whether speculative decoding batches the same token
with adjacent text.

This overlay makes only the unambiguous case eager.  Mixed chunks, pre-existing
deferred state, and every non-special token retain the upstream implementation.
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

SCHEMA = "urn:qwen-r9700:qwen-stream-boundary-overlay:v1"
TARGET_MODULE = "vllm.parser.engine.token_id_scanner"
PARSER_MODULE = "vllm.parser.engine.parser_engine"
ENABLE_ENV = "QWEN_STREAM_BOUNDARY_REPAIR"
REQUIRED_ENV = "QWEN_STREAM_BOUNDARY_REPAIR_REQUIRED"
SITE_SHA_ENV = "QWEN_STREAM_BOUNDARY_REPAIR_SITE_SHA256"
RUNTIME_SHA_ENV = "QWEN_STREAM_BOUNDARY_REPAIR_RUNTIME_SHA256"
TRACE_DIR_ENV = "QWEN_STREAM_BOUNDARY_TRACE_DIR"
TRACE_PARSER_SHA_ENV = "QWEN_STREAM_BOUNDARY_TRACE_PARSER_SHA256"
_OVERLAY_ENVIRONMENT = (
    ENABLE_ENV,
    REQUIRED_ENV,
    SITE_SHA_ENV,
    RUNTIME_SHA_ENV,
    TRACE_DIR_ENV,
    TRACE_PARSER_SHA_ENV,
)


class QwenStreamBoundaryOverlayError(RowExactOverlayError):
    """The stream-boundary repair contract is invalid."""


def _validate_base(argv: Sequence[str]) -> None:
    required_flags = (
        "--enable-auto-tool-choice",
        "--trust-request-chat-template",
    )
    if any(argv.count(flag) != 1 for flag in required_flags):
        raise QwenStreamBoundaryOverlayError(
            "base command lacks the exact Qwen streaming parser flags"
        )

    def one_value(option: str) -> str:
        values = [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == option]
        if len(values) != 1:
            raise QwenStreamBoundaryOverlayError(f"base command must contain one exact {option}")
        return values[0]

    if one_value("--reasoning-parser") != "qwen3":
        raise QwenStreamBoundaryOverlayError("base command does not use qwen3 reasoning")
    if one_value("--tool-call-parser") != "qwen3_coder":
        raise QwenStreamBoundaryOverlayError(
            "base command does not use the qwen3_coder tool parser"
        )


def _site_source(
    destination: Path,
    *,
    chained_site: Path,
    chained_site_sha256: str,
    chained_pythonpath: str,
    expected_runtime_sha256: str,
    expected_parser_runtime_sha256: str | None,
    trace_directory: Path | None,
) -> bytes:
    outer_pythonpath = f"{destination}:{chained_pythonpath}"
    return f'''"""Authenticated Qwen singleton-boundary parser bootstrap."""
import functools
import hashlib
import json
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
_PARSER_TARGET = {PARSER_MODULE!r}
_RUNTIME_SHA256 = {expected_runtime_sha256!r}
_PARSER_RUNTIME_SHA256 = {expected_parser_runtime_sha256!r}
_TRACE_DIRECTORY = {str(trace_directory) if trace_directory is not None else None!r}
_METHOD_MARKER = "_qwen_stream_boundary_repair_site_sha256"
_TRACE_MARKER = "_qwen_stream_boundary_trace_site_sha256"

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
    raise RuntimeError("Qwen stream-boundary repair identity is absent")
if os.environ.get({RUNTIME_SHA_ENV!r}) != _RUNTIME_SHA256:
    raise RuntimeError("Qwen token scanner runtime identity differs")
if os.environ.get("PYTHONPATH") != _OUTER_PYTHONPATH:
    raise RuntimeError("Qwen stream-boundary repair PYTHONPATH differs")
_SITE_SHA256 = _stable_digest(_SELF, "stream-boundary site", private=True)
if _SITE_SHA256 != os.environ.get({SITE_SHA_ENV!r}):
    raise RuntimeError("Qwen stream-boundary site SHA256 mismatch")
if _stable_digest(_CHAIN, "chained site", private=True) != _CHAIN_SHA256:
    raise RuntimeError("Qwen stream-boundary chained-site SHA256 mismatch")
if _TARGET in sys.modules:
    raise RuntimeError("token scanner imported before stream-boundary installation")

os.environ["PYTHONPATH"] = _CHAIN_PYTHONPATH
try:
    runpy.run_path(str(_CHAIN), run_name="_qwen_stream_boundary_chain")
    if os.environ.get("PYTHONPATH") != _CHAIN_PYTHONPATH:
        raise RuntimeError("chained site changed PYTHONPATH")
finally:
    os.environ["PYTHONPATH"] = _OUTER_PYTHONPATH
if _TARGET in sys.modules:
    raise RuntimeError("chained site imported token scanner before registration")

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
    raise RuntimeError("existing token-scanner patch is not callable")
previous_marker = getattr(previous_patch, _METHOD_MARKER, None) if previous_patch else None
if previous_marker is not None and previous_marker != _SITE_SHA256:
    raise RuntimeError("a different Qwen stream-boundary repair is registered")

def _patch_scanner(module):
    source = Path(getattr(module, "__file__", ""))
    if (
        not source.is_absolute()
        or _stable_digest(source, "token scanner runtime") != _RUNTIME_SHA256
    ):
        raise RuntimeError("token scanner source SHA256 mismatch")
    if getattr(module, "__name__", None) != _TARGET:
        raise RuntimeError("token scanner module identity changed")
    if previous_patch is not None:
        previous_patch(module)
    cls = getattr(module, "TokenIDScanner", None)
    terminal_type = getattr(module, "PreLexedTerminal", None)
    text_chunk_type = getattr(module, "TextChunk", None)
    original = getattr(cls, "scan", None)
    current_marker = getattr(original, _METHOD_MARKER, None)
    if current_marker is not None:
        if current_marker != _SITE_SHA256:
            raise RuntimeError("token scanner already carries another boundary repair")
        return
    if (
        not isinstance(cls, type)
        or not isinstance(terminal_type, type)
        or not isinstance(text_chunk_type, type)
        or not callable(original)
    ):
        raise RuntimeError("token scanner ABI changed")

    @functools.wraps(original)
    def scan(self, delta_text, delta_token_ids):
        unambiguous = (
            len(delta_token_ids) == 1
            and delta_token_ids[0] in self.token_id_to_terminal
            and not self._deferred_terminals
            and not self._deferred_prefix_token_counts
            and self._deferred_trailing_token_count == 0
            and self._deferred_post_text == ""
        )
        result = original(self, delta_text, delta_token_ids)
        if not unambiguous:
            return result
        if not self._deferred_terminals:
            return result
        if (
            result
            or
            len(self._deferred_terminals) != 1
            or not isinstance(self._deferred_terminals[0], terminal_type)
            or self._deferred_terminals[0].token_id != delta_token_ids[0]
            or self._deferred_prefix_token_counts != [0]
            or self._deferred_trailing_token_count != 0
            or self._deferred_post_text != delta_text
        ):
            raise RuntimeError("singleton special-token deferral contract changed")
        terminal = self._deferred_terminals.pop()
        self._deferred_prefix_token_counts.clear()
        self._deferred_post_text = ""
        prefix = [text_chunk_type(delta_text)] if delta_text else []
        return [*prefix, terminal]

    setattr(scan, _METHOD_MARKER, _SITE_SHA256)
    cls.scan = scan
    print("[qwen-stream-boundary] singleton special-token repair armed", flush=True)

if previous_marker is None:
    setattr(_patch_scanner, _METHOD_MARKER, _SITE_SHA256)
    patches[_TARGET] = _patch_scanner
elif patches.get(_TARGET) is not previous_patch:
    raise RuntimeError("token-scanner finder identity changed")

if _TRACE_DIRECTORY is not None:
    trace_root = Path(_TRACE_DIRECTORY)
    trace_meta = trace_root.lstat()
    if (
        not trace_root.is_absolute()
        or not stat.S_ISDIR(trace_meta.st_mode)
        or trace_root.is_symlink()
        or trace_meta.st_uid != os.getuid()
        or stat.S_IMODE(trace_meta.st_mode) & 0o077
    ):
        raise RuntimeError("Qwen parser trace directory is not owned and private")
    if not _PARSER_RUNTIME_SHA256:
        raise RuntimeError("Qwen parser trace runtime identity is absent")
    if os.environ.get({TRACE_DIR_ENV!r}) != _TRACE_DIRECTORY:
        raise RuntimeError("Qwen parser trace directory differs")
    if os.environ.get({TRACE_PARSER_SHA_ENV!r}) != _PARSER_RUNTIME_SHA256:
        raise RuntimeError("Qwen parser trace runtime identity differs")
    previous_parser_patch = patches.get(_PARSER_TARGET)
    if previous_parser_patch is not None and not callable(previous_parser_patch):
        raise RuntimeError("existing parser-engine patch is not callable")
    trace_fd = None
    trace_sequence = 0

    def _trace_state(parser):
        engine = parser._engine
        scanner = engine._scanner
        return {{
            "engine_state": str(engine.state),
            "tool_index": engine.tool_index,
            "lexer_buffer": engine._lexer.buffer,
            "scanner_deferred": [
                {{"terminal": item.terminal, "token_id": item.token_id, "text": item.text}}
                for item in scanner._deferred_terminals
            ],
            "scanner_prefix_counts": list(scanner._deferred_prefix_token_counts),
            "scanner_trailing_count": scanner._deferred_trailing_token_count,
            "scanner_post_text": scanner._deferred_post_text,
            "reasoning_ended": parser._reasoning_ended,
            "tool_slots": [
                {{"name": slot.name, "name_sent": slot.name_sent, "args": slot.args}}
                for slot in parser._tool_slots
            ],
        }}

    def _trace_write(record):
        global trace_fd, trace_sequence
        if trace_fd is None:
            trace_path = trace_root / f"parser-trace-{{os.getpid()}}.jsonl"
            trace_fd = os.open(
                trace_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                0o600,
            )
        trace_sequence += 1
        record["sequence"] = trace_sequence
        payload = (json.dumps(record, sort_keys=True, ensure_ascii=False) + "\\n").encode()
        view = memoryview(payload)
        while view:
            written = os.write(trace_fd, view)
            if written <= 0:
                raise RuntimeError("Qwen parser trace write did not progress")
            view = view[written:]
        if record.get("finished"):
            os.fsync(trace_fd)

    def _patch_parser_engine(module):
        source = Path(getattr(module, "__file__", ""))
        if (
            not source.is_absolute()
            or _stable_digest(source, "parser engine runtime")
            != _PARSER_RUNTIME_SHA256
        ):
            raise RuntimeError("parser engine source SHA256 mismatch")
        cls = getattr(module, "ParserEngine", None)
        original = getattr(cls, "parse_delta", None)
        current_marker = getattr(original, _TRACE_MARKER, None)
        if current_marker is not None:
            if current_marker != _SITE_SHA256:
                raise RuntimeError("parser engine already carries another trace")
            return
        if not isinstance(cls, type) or not callable(original):
            raise RuntimeError("parser engine trace ABI changed")
        if previous_parser_patch is not None:
            previous_parser_patch(module)
            original = cls.parse_delta

        @functools.wraps(original)
        def parse_delta(
            self,
            delta_text,
            delta_token_ids,
            request,
            prompt_token_ids=None,
            *,
            finished,
        ):
            before = _trace_state(self)
            result = original(
                self,
                delta_text,
                delta_token_ids,
                request,
                prompt_token_ids,
                finished=finished,
            )
            after = _trace_state(self)
            resolved = self._engine._resolved_token_ids
            interesting = (
                finished
                or any(token_id in resolved for token_id in delta_token_ids)
                or "TOOL" in before["engine_state"]
                or "TOOL" in after["engine_state"]
                or before["scanner_deferred"]
                or after["scanner_deferred"]
            )
            if interesting:
                dumped = None
                if result is not None:
                    model_dump = getattr(result, "model_dump", None)
                    dumped = (
                        model_dump(mode="json", exclude_none=False)
                        if callable(model_dump)
                        else repr(result)
                    )
                _trace_write({{
                    "before": before,
                    "after": after,
                    "delta_text": delta_text,
                    "delta_token_ids": list(delta_token_ids),
                    "finished": finished,
                    "result": dumped,
                }})
            return result

        setattr(parse_delta, _TRACE_MARKER, _SITE_SHA256)
        cls.parse_delta = parse_delta
        print("[qwen-stream-boundary] parser state trace armed", flush=True)

    setattr(_patch_parser_engine, _TRACE_MARKER, _SITE_SHA256)
    patches[_PARSER_TARGET] = _patch_parser_engine
'''.encode()


def render(args: argparse.Namespace) -> Mapping[str, Any]:
    destination = args.destination.expanduser().absolute()
    trace_directory_arg = getattr(args, "trace_directory", None)
    parser_runtime_sha256 = getattr(args, "expected_parser_engine_sha256", None)
    trace_directory = (
        trace_directory_arg.expanduser().absolute() if trace_directory_arg is not None else None
    )
    if destination.exists() or destination.is_symlink():
        raise QwenStreamBoundaryOverlayError("destination is create-only")
    for label, value in (
        ("base command", args.expected_base_sha256),
        ("token scanner runtime", args.expected_token_scanner_sha256),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise QwenStreamBoundaryOverlayError(f"expected {label} SHA256 is malformed")
    if (trace_directory is None) != (parser_runtime_sha256 is None):
        raise QwenStreamBoundaryOverlayError(
            "parser trace directory and parser-engine SHA256 must be provided together"
        )
    if parser_runtime_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", parser_runtime_sha256
    ):
        raise QwenStreamBoundaryOverlayError("expected parser engine SHA256 is malformed")
    if trace_directory is not None:
        trace_metadata = trace_directory.lstat()
        if (
            not stat.S_ISDIR(trace_metadata.st_mode)
            or trace_directory.is_symlink()
            or trace_metadata.st_uid != os.getuid()
            or stat.S_IMODE(trace_metadata.st_mode) & 0o077
        ):
            raise QwenStreamBoundaryOverlayError(
                "parser trace directory must be one owned private directory"
            )

    base_payload = _stable_private_file(args.base_command, "base command")
    base_sha256 = _digest(base_payload)
    if base_sha256 != args.expected_base_sha256:
        raise QwenStreamBoundaryOverlayError("base command SHA256 mismatch")
    environment, argv = _split_command(base_payload)
    env = _environment_map(environment)
    if any(name in env for name in _OVERLAY_ENVIRONMENT):
        raise QwenStreamBoundaryOverlayError(
            "base command already contains the Qwen stream-boundary repair"
        )
    _validate_base(argv)

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
        raise QwenStreamBoundaryOverlayError(
            "destination parent must be one owned private directory"
        )
    staging = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise QwenStreamBoundaryOverlayError("staging destination already exists")
    staging.mkdir(mode=0o700)
    try:
        site_payload = _site_source(
            destination,
            chained_site=chained_site,
            chained_site_sha256=chained_sha256,
            chained_pythonpath=chained_pythonpath,
            expected_runtime_sha256=args.expected_token_scanner_sha256,
            expected_parser_runtime_sha256=parser_runtime_sha256,
            trace_directory=trace_directory,
        )
        site_sha256 = _digest(site_payload)
        additions = {
            ENABLE_ENV: "1",
            REQUIRED_ENV: "1",
            SITE_SHA_ENV: site_sha256,
            RUNTIME_SHA_ENV: args.expected_token_scanner_sha256,
        }
        if trace_directory is not None:
            additions[TRACE_DIR_ENV] = str(trace_directory)
            additions[TRACE_PARSER_SHA_ENV] = parser_runtime_sha256
        rewritten_environment = [
            token
            for token in environment
            if token.split("=", 1)[0] not in {"PYTHONPATH", *additions}
        ]
        rewritten_environment.append(f"PYTHONPATH={destination}:{chained_pythonpath}")
        rewritten_environment.extend(f"{name}={value}" for name, value in additions.items())
        command_payload = (
            shlex.join(["exec", "/usr/bin/env", "-i", *rewritten_environment, *argv]) + "\n"
        ).encode()
        manifest = {
            "schema": SCHEMA,
            "classification": "production_parser_correctness_repair_candidate",
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
                "method": "TokenIDScanner.scan",
                "eager_case": (
                    "singleton recognized special token whose decoded terminal is absent "
                    "from delta_text"
                ),
                "preexisting_deferred_state": "unchanged",
                "mixed_chunks": "unchanged",
                "ordinary_tokens": "unchanged",
                "semantic_goal": "chunk-partition-invariant Qwen tool boundaries",
            },
            "token_scanner_runtime_sha256": args.expected_token_scanner_sha256,
            "site_sha256": site_sha256,
            "diagnostic_parser_trace": (
                {
                    "directory": str(trace_directory),
                    "parser_engine_runtime_sha256": parser_runtime_sha256,
                }
                if trace_directory is not None
                else None
            ),
        }
        manifest_payload = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
        _write_exclusive(staging / "sitecustomize.py", site_payload, 0o600)
        _write_exclusive(staging / "command.sh", command_payload, 0o700)
        _write_exclusive(staging / "manifest.json", manifest_payload, 0o600)
        if destination.exists() or destination.is_symlink():
            raise QwenStreamBoundaryOverlayError(
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
    parser.add_argument("--expected-token-scanner-sha256", required=True)
    parser.add_argument("--expected-parser-engine-sha256")
    parser.add_argument("--trace-directory", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = render(args)
    except (OSError, RowExactOverlayError) as error:
        print(f"qwen-stream-boundary-overlay: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
