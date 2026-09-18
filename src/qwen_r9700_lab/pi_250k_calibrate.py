"""Offline chat-template calibration for a public Pi 250K fixture."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Protocol

from qwen_r9700_lab.pi_250k_fixture import (
    Pi250kFixtureError,
    _canonical_bytes,
    _file_sha256,
    _sha256_bytes,
    _sha256_text,
    effective_system_prompt,
)


class ChatTokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        preserve_thinking: bool,
        reasoning_effort: str,
    ) -> list[int] | Mapping[str, list[int]]: ...


def _load_tokenizer(model_tokenizer_dir: Path) -> ChatTokenizer:
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as error:
        raise Pi250kFixtureError(
            "transformers is required only for calibration; run with "
            "`uv run --with transformers python -m qwen_r9700_lab.pi_250k_calibrate ...`"
        ) from error
    try:
        return transformers.AutoTokenizer.from_pretrained(
            model_tokenizer_dir.as_posix(),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as error:
        raise Pi250kFixtureError(
            f"cannot load local tokenizer metadata from {model_tokenizer_dir}: {error}"
        ) from error


def _load_public_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
    try:
        manifest_path = path.resolve(strict=True)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Pi250kFixtureError(f"cannot load fixture manifest {path}: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise Pi250kFixtureError("fixture manifest must be a schema_version 1 object")
    if manifest.get("public_data_only") is not True or manifest.get("source_session") is not None:
        raise Pi250kFixtureError(
            "calibration accepts only public-data fixtures without a source session"
        )
    recorded_hash = manifest.get("manifest_payload_sha256")
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    if (
        not isinstance(recorded_hash, str)
        or _sha256_bytes(_canonical_bytes(payload)) != recorded_hash
    ):
        raise Pi250kFixtureError("fixture manifest payload hash mismatch")
    return manifest_path, manifest


def _public_file(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise Pi250kFixtureError("fixture file path must be a non-empty relative path")
    candidate = root / relative
    if candidate.is_symlink():
        raise Pi250kFixtureError(f"fixture file must not be a symbolic link: {relative}")
    try:
        path = candidate.resolve(strict=True)
    except OSError as error:
        raise Pi250kFixtureError(f"fixture file is missing: {relative}") from error
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise Pi250kFixtureError(f"fixture file is not a contained regular file: {relative}")
    return path


def _read_session(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) != 2:
            raise Pi250kFixtureError("public calibration session must contain exactly two entries")
        header, entry = (json.loads(line) for line in lines)
    except json.JSONDecodeError as error:
        raise Pi250kFixtureError(f"invalid public session JSON: {error}") from error
    if header.get("type") != "session" or header.get("version") != 3:
        raise Pi250kFixtureError("public calibration session header is not Pi v3")
    if (
        entry.get("type") != "custom_message"
        or entry.get("display") is not False
        or entry.get("customType") != "qwen-r9700-public-250k-fixture"
        or entry.get("details", {}).get("public_data_only") is not True
        or not isinstance(entry.get("content"), str)
    ):
        raise Pi250kFixtureError("session does not contain one hidden public fixture message")
    return header, entry


def calibrate_fixture(
    *,
    manifest_path: Path,
    model_tokenizer_dir: Path,
    tokenizer: ChatTokenizer | None = None,
) -> dict[str, Any]:
    """Measure the complete Qwen chat template without submitting an inference request."""

    resolved_manifest, manifest = _load_public_manifest(manifest_path)
    root = resolved_manifest.parent
    files = manifest.get("files")
    sessions = files.get("sessions") if isinstance(files, dict) else None
    if not isinstance(sessions, list) or not sessions:
        raise Pi250kFixtureError("fixture manifest has no sessions")
    first_session = sessions[0]
    session_path = _public_file(root, first_session.get("path"))
    if _file_sha256(session_path) != first_session.get("sha256"):
        raise Pi250kFixtureError("public session file hash mismatch")
    header, entry = _read_session(session_path)
    filler = entry["content"]
    if _sha256_text(filler) != manifest.get("public_filler", {}).get("sha256"):
        raise Pi250kFixtureError("public filler content hash mismatch")

    system_record = files.get("system_prompt")
    benchmark_record = files.get("benchmark_prompt")
    if not isinstance(system_record, dict) or not isinstance(benchmark_record, dict):
        raise Pi250kFixtureError("fixture prompt file records are missing")
    system_path = _public_file(root, system_record.get("path"))
    benchmark_path = _public_file(root, benchmark_record.get("path"))
    if _file_sha256(system_path) != system_record.get("sha256"):
        raise Pi250kFixtureError("base system prompt file hash mismatch")
    if _file_sha256(benchmark_path) != benchmark_record.get("sha256"):
        raise Pi250kFixtureError("benchmark prompt file hash mismatch")
    base_system = system_path.read_text(encoding="utf-8")
    benchmark = benchmark_path.read_text(encoding="utf-8")
    session_cwd = header.get("cwd")
    if not isinstance(session_cwd, str) or not session_cwd:
        raise Pi250kFixtureError("public session cwd is missing")
    effective_system = effective_system_prompt(base_system, Path(session_cwd))
    contract = manifest.get("request_contract", {})
    if _sha256_text(effective_system) != contract.get("effective_system_prompt_sha256"):
        raise Pi250kFixtureError("effective system prompt hash mismatch")
    if _sha256_text(benchmark) != contract.get("benchmark_prompt_sha256"):
        raise Pi250kFixtureError("benchmark prompt hash mismatch")

    try:
        resolved_tokenizer_dir = model_tokenizer_dir.resolve(strict=True)
    except OSError as error:
        raise Pi250kFixtureError(
            f"model tokenizer directory does not exist: {model_tokenizer_dir}"
        ) from error
    if not resolved_tokenizer_dir.is_dir():
        raise Pi250kFixtureError(
            f"model tokenizer metadata path is not a directory: {resolved_tokenizer_dir}"
        )
    exact_tokenizer = (
        tokenizer if tokenizer is not None else _load_tokenizer(resolved_tokenizer_dir)
    )
    raw_ids = exact_tokenizer.encode(filler, add_special_tokens=False)
    messages = [
        {"role": "system", "content": effective_system},
        {"role": "user", "content": filler},
        {"role": "user", "content": benchmark},
    ]
    try:
        rendered = exact_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
            preserve_thinking=True,
            reasoning_effort="xhigh",
        )
    except Exception as error:
        raise Pi250kFixtureError(f"chat-template rendering failed: {error}") from error
    rendered_ids = rendered.get("input_ids") if isinstance(rendered, Mapping) else rendered
    if not isinstance(raw_ids, list) or not isinstance(rendered_ids, list):
        raise Pi250kFixtureError("tokenizer did not return token-id lists")
    raw_tokens = len(raw_ids)
    rendered_prompt_tokens = len(rendered_ids)
    accounting = manifest.get("token_accounting", {})
    if raw_tokens != accounting.get("measured_raw_filler_tokens"):
        raise Pi250kFixtureError("chat tokenizer raw filler count differs from builder tokenizer")
    tokenizer_json = resolved_tokenizer_dir / "tokenizer.json"
    if _file_sha256(tokenizer_json) != manifest.get("tokenizer", {}).get("sha256"):
        raise Pi250kFixtureError("calibration tokenizer JSON differs from builder tokenizer")
    metadata_hashes: dict[str, str] = {}
    for filename in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        metadata_path = resolved_tokenizer_dir / filename
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise Pi250kFixtureError(
                f"required tokenizer metadata must be a regular non-symlink file: {filename}"
            )
        metadata_hashes[filename] = _file_sha256(metadata_path)
    if tokenizer is None:
        try:
            runtime_versions = {
                "transformers": importlib_metadata.version("transformers"),
                "jinja2": importlib_metadata.version("jinja2"),
            }
        except importlib_metadata.PackageNotFoundError as error:
            raise Pi250kFixtureError(
                "transformers and jinja2 distribution versions must be available for calibration"
            ) from error
    else:
        runtime_versions = {"tokenizer_provider": "injected-test-double"}
    calibrated_overhead = rendered_prompt_tokens - raw_tokens
    if calibrated_overhead < 1:
        raise Pi250kFixtureError("calibrated chat-template overhead is not positive")
    target = accounting.get("target_prompt_tokens")
    if not isinstance(target, int) or target < 1:
        raise Pi250kFixtureError("manifest target prompt tokens are invalid")
    token_id_digest = hashlib.sha256(
        ",".join(str(token_id) for token_id in rendered_ids).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "fixture_id": manifest.get("fixture_id"),
        "public_data_only": True,
        "model_tokenizer_dir": resolved_tokenizer_dir.as_posix(),
        "tokenizer_json_sha256": _file_sha256(tokenizer_json),
        "tokenizer_metadata_sha256": metadata_hashes,
        "calibration_runtime_versions": runtime_versions,
        "raw_filler_tokens": raw_tokens,
        "rendered_prompt_tokens": rendered_prompt_tokens,
        "calibrated_chat_overhead_tokens": calibrated_overhead,
        "target_prompt_tokens": target,
        "target_delta_tokens": rendered_prompt_tokens - target,
        "exact_target": rendered_prompt_tokens == target,
        "rendered_token_ids_sha256": token_id_digest,
        "message_content_sha256": {
            "system": _sha256_text(effective_system),
            "filler": _sha256_text(filler),
            "benchmark": _sha256_text(benchmark),
        },
        "inference_request_made": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline-calibrate a public Pi 250K fixture with the exact Qwen tokenizer."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--require-exact", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = calibrate_fixture(
            manifest_path=args.manifest,
            model_tokenizer_dir=args.model_tokenizer_dir,
        )
    except (OSError, Pi250kFixtureError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["exact_target"] or not args.require_exact else 3


if __name__ == "__main__":
    raise SystemExit(main())
