"""Generate assurance and release artifacts from one semantic source tree.

The release projection is not a runtime mode.  Assurance-only source regions are
removed before the release artifact is published, and declared hot-path files are
then scanned for diagnostic constructs.  Both projections are content-addressed
by one source manifest and bound by an equivalence certificate.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import symtable
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from qwen_r9700_lab.coding_turbo_state_provider import (
    CONSUMER_TRANSITION_SCHEMA,
    PRODUCER_RECEIPT_SCHEMA,
    QUALIFIED_CONVOLUTION_STATE_NAMES,
    QUALIFIED_GDN_STATE_NAMES,
    StateProviderError,
    seal_consumer_transition_receipt,
    seal_producer_receipt,
    validate_consumer_transition_receipt,
    validate_producer_receipt,
)
from qwen_r9700_lab.persistent_cache import (
    FIXED_SLOT_60K_PROFILE,
    FIXED_SLOT_PRODUCER_CONTRACT,
    FIXED_SLOT_PRODUCER_SCHEMA,
    HISTORICAL_FIXED_SLOT_PRODUCER_SCHEMA,
    CacheSafetyError,
    _validate_payload_provenance,
)

SOURCE_SCHEMA = "urn:qwen-r9700:coding-turbo-source:v1"
ARTIFACT_SCHEMA = "urn:qwen-r9700:coding-turbo-artifact:v1"
EQUIVALENCE_SCHEMA = "urn:qwen-r9700:coding-turbo-equivalence:v1"
STATE_EXPORTER_OUTPUT = "assurance/coding_turbo_state_exporter.py"
ACCEPTED_PATH_PATCH_OUTPUT = "runtime-patches/vllm-v2-external-commit-cap.patch"
SNAPSHOT_FORMAT_PATCH_OUTPUT = "runtime-patches/qwen-persistent-selection-external-commit-cap.patch"
CONSUMER_CAPABILITY_OUTPUT = "runtime-contracts/external-commit-capability.json"

BEGIN_RE = re.compile(
    rb"^[ \t]*(?:#|//)[ \t]+QWEN_ASSURANCE_ONLY_BEGIN:[ \t]*"
    rb"(?P<name>[A-Za-z][A-Za-z0-9_.-]{0,127})[ \t]*\r?\n?$"
)
END_RE = re.compile(
    rb"^[ \t]*(?:#|//)[ \t]+QWEN_ASSURANCE_ONLY_END:[ \t]*"
    rb"(?P<name>[A-Za-z][A-Za-z0-9_.-]{0,127})[ \t]*\r?\n?$"
)
MARKER_TOKEN = b"QWEN_ASSURANCE_ONLY_"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# These constructs are forbidden in a generated release hot path even when a
# template author forgot to place them inside an assurance-only region.
FORBIDDEN_RELEASE_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "debug/trace/capture environment switch",
        re.compile(rb"QWEN_[A-Z0-9_]*(?:DEBUG|TRACE|TIMING|CAPTURE|SHADOW)[A-Z0-9_]*"),
    ),
    (
        "diagnostic identifier",
        re.compile(
            rb"(?i)(?<![A-Za-z0-9_])(?:debug|diagnostic|assurance|dense_shadow|"
            rb"rowwise_q1_control|stage_timing)(?![A-Za-z0-9_])"
        ),
    ),
    ("asynchronous assertion", re.compile(rb"torch\._assert_async")),
    ("profiler range", re.compile(rb"(?:record_function|nvtx|roctxRange|hipEvent)")),
    ("hot-path wall-clock timing", re.compile(rb"time\.(?:perf_counter|monotonic_ns)")),
)

# Upstream execution APIs can contain words that overlap the diagnostic
# vocabulary without implementing diagnostics.  Keep this list exact and
# intentionally tiny: every additional entry must name a production API whose
# removal would change execution semantics.
RELEASE_IDENTIFIER_ALLOWLIST = frozenset(
    {
        "build_for_cudagraph_capture",
        # These three names select the sampler-approved recurrent and
        # convolution states.  They are authoritative accepted-path semantics,
        # not diagnostics, and removing them would mutate canonical GDN state.
        "capture_replay",
        "capture_serial",
        "capture_source",
    }
)


class ArtifactSafetyError(RuntimeError):
    """Raised when artifact identity, source structure, or release purity fails."""


@dataclass(frozen=True)
class SourceFile:
    source: str
    output: str
    hot_path: bool


@dataclass(frozen=True)
class Projection:
    assurance: bytes
    release: bytes
    region_names: tuple[str, ...]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(document: object) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


def _compact_json(document: object) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _validate_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactSafetyError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ArtifactSafetyError(f"{label} is not a normalized relative path: {value!r}")
    return path.as_posix()


def _lstat_regular_owned(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ArtifactSafetyError(f"{label} does not exist: {path}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ArtifactSafetyError(f"{label} must be a regular non-symlink file: {path}")
    if metadata.st_uid != os.getuid():
        raise ArtifactSafetyError(f"{label} has the wrong owner: {path}")
    if stat.S_IMODE(metadata.st_mode) & 0o022:
        raise ArtifactSafetyError(f"{label} must not be group/world writable: {path}")
    return metadata


def _require_owned_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ArtifactSafetyError(f"{label} does not exist: {path}") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise ArtifactSafetyError(f"{label} must be a real directory: {path}")
    if metadata.st_uid != os.getuid():
        raise ArtifactSafetyError(f"{label} has the wrong owner: {path}")
    if stat.S_IMODE(metadata.st_mode) & 0o022:
        raise ArtifactSafetyError(f"{label} must not be group/world writable: {path}")


def _load_source_manifest(path: Path) -> tuple[dict[str, Any], bytes, tuple[SourceFile, ...]]:
    _lstat_regular_owned(path, "source manifest")
    raw = path.read_bytes()
    if len(raw) > 1024 * 1024:
        raise ArtifactSafetyError("source manifest exceeds 1 MiB")
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactSafetyError(f"invalid source manifest JSON: {error}") from error
    if not isinstance(document, dict) or document.get("schema") != SOURCE_SCHEMA:
        raise ArtifactSafetyError(f"source manifest schema must be {SOURCE_SCHEMA}")
    artifact_id = document.get("artifact_id")
    semantic_contract = document.get("semantic_contract")
    if not isinstance(artifact_id, str) or not IDENTIFIER_RE.fullmatch(artifact_id):
        raise ArtifactSafetyError("artifact_id is invalid")
    if not isinstance(semantic_contract, str) or not semantic_contract.strip():
        raise ArtifactSafetyError("semantic_contract must be a non-empty string")
    raw_files = document.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ArtifactSafetyError("source manifest files must be a non-empty array")
    files: list[SourceFile] = []
    source_paths: set[str] = set()
    output_paths: set[str] = set()
    for index, entry in enumerate(raw_files):
        if not isinstance(entry, dict) or set(entry) != {"source", "output", "hot_path"}:
            raise ArtifactSafetyError(f"files[{index}] has unknown or missing fields")
        source = _validate_relative_path(entry["source"], f"files[{index}].source")
        output = _validate_relative_path(entry["output"], f"files[{index}].output")
        hot_path = entry["hot_path"]
        if not isinstance(hot_path, bool):
            raise ArtifactSafetyError(f"files[{index}].hot_path must be boolean")
        if source in source_paths or output in output_paths:
            raise ArtifactSafetyError("source and output paths must each be unique")
        source_paths.add(source)
        output_paths.add(output)
        files.append(SourceFile(source=source, output=output, hot_path=hot_path))
    normalized = _canonical_json(document)
    if raw != normalized:
        raise ArtifactSafetyError("source manifest must use canonical sorted indented JSON")
    return document, raw, tuple(files)


def _project_source(payload: bytes, relative_path: str) -> Projection:
    assurance: list[bytes] = []
    release: list[bytes] = []
    regions: list[str] = []
    active: str | None = None
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        begin = BEGIN_RE.fullmatch(line)
        end = END_RE.fullmatch(line)
        if begin:
            if active is not None:
                raise ArtifactSafetyError(
                    f"nested assurance region in {relative_path}:{line_number}"
                )
            active = begin.group("name").decode()
            if active in regions:
                raise ArtifactSafetyError(
                    f"duplicate assurance region {active!r} in {relative_path}"
                )
            regions.append(active)
            continue
        if end:
            name = end.group("name").decode()
            if active != name:
                raise ArtifactSafetyError(
                    f"unmatched assurance region end {name!r} in {relative_path}:{line_number}"
                )
            active = None
            continue
        assurance.append(line)
        if active is None:
            release.append(line)
    if active is not None:
        raise ArtifactSafetyError(f"unterminated assurance region {active!r} in {relative_path}")
    assurance_payload = b"".join(assurance)
    release_payload = b"".join(release)
    if MARKER_TOKEN in assurance_payload or MARKER_TOKEN in release_payload:
        raise ArtifactSafetyError(f"assurance marker survived projection in {relative_path}")
    return Projection(
        assurance=assurance_payload,
        release=release_payload,
        region_names=tuple(regions),
    )


def _scan_python_release(payload: bytes, relative_path: str) -> None:
    if not relative_path.endswith(".py"):
        return
    try:
        tree = ast.parse(payload, filename=relative_path)
    except (SyntaxError, UnicodeDecodeError) as error:
        raise ArtifactSafetyError(
            f"release Python is invalid in {relative_path}: {error}"
        ) from error
    forbidden_terms = ("debug", "trace", "timing", "capture", "shadow", "diagnostic", "assurance")
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            raise ArtifactSafetyError(
                f"release assertion survived in {relative_path}:{node.lineno}"
            )
        names: tuple[str, ...] = ()
        if isinstance(node, ast.Name):
            names = (node.id,)
        elif isinstance(node, ast.Attribute):
            names = (node.attr,)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = (node.name,)
        for name in names:
            if name in RELEASE_IDENTIFIER_ALLOWLIST:
                continue
            lowered = name.lower()
            if any(term in lowered for term in forbidden_terms):
                location = f"{relative_path}:{node.lineno}"
                raise ArtifactSafetyError(
                    f"release diagnostic identifier {name!r} survived in {location}"
                )

    source = payload.decode("utf-8")
    symbols = symtable.symtable(source, relative_path, "exec")
    module_definitions = {
        symbol.get_name()
        for symbol in symbols.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
    }
    injected = {
        "__builtins__",
        "__cached__",
        "__conditional_annotations__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__spec__",
    }
    permitted_globals = module_definitions | set(dir(builtins)) | injected

    def unresolved(table: symtable.SymbolTable) -> set[str]:
        missing = {
            symbol.get_name()
            for symbol in table.get_symbols()
            if symbol.is_global()
            and symbol.is_referenced()
            and symbol.get_name() not in permitted_globals
        }
        for child in table.get_children():
            missing.update(unresolved(child))
        return missing

    missing = sorted(unresolved(symbols))
    if missing:
        raise ArtifactSafetyError(
            f"release Python has unresolved global references in {relative_path}: "
            + ", ".join(missing)
        )


def _scan_release(payload: bytes, relative_path: str) -> None:
    for label, pattern in FORBIDDEN_RELEASE_PATTERNS:
        match = pattern.search(payload)
        if match:
            line = payload.count(b"\n", 0, match.start()) + 1
            raise ArtifactSafetyError(f"release {label} survived in {relative_path}:{line}")
    _scan_python_release(payload, relative_path)


def _write_file(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, mode)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _tree_digest(files: list[dict[str, Any]]) -> str:
    identity = [{"path": entry["path"], "sha256": entry["sha256"]} for entry in files]
    return _sha256_bytes(_canonical_json(identity))


def build_artifacts(source_manifest: Path, output: Path) -> dict[str, Any]:
    """Build and atomically publish paired assurance/release artifacts."""

    source_manifest = source_manifest.absolute()
    output = output.absolute()
    document, manifest_payload, files = _load_source_manifest(source_manifest)
    source_root = source_manifest.parent
    _require_owned_directory(source_root, "source root")
    _require_owned_directory(output.parent, "output parent")
    if output.exists() or output.is_symlink():
        raise ArtifactSafetyError(f"refusing to replace existing output: {output}")

    source_entries: list[dict[str, Any]] = []
    projections: list[tuple[SourceFile, Projection, int]] = []
    region_count = 0
    for spec in files:
        path = source_root / spec.source
        metadata = _lstat_regular_owned(path, f"source file {spec.source}")
        payload = path.read_bytes()
        projection = _project_source(payload, spec.source)
        if spec.hot_path:
            _scan_release(projection.release, spec.output)
        mode = 0o700 if metadata.st_mode & stat.S_IXUSR else 0o600
        source_entries.append(
            {
                "hot_path": spec.hot_path,
                "output": spec.output,
                "sha256": _sha256_bytes(payload),
                "source": spec.source,
            }
        )
        projections.append((spec, projection, mode))
        region_count += len(projection.region_names)
    if region_count == 0:
        raise ArtifactSafetyError("source tree contains no assurance-only regions")

    semantic_identity = {
        "artifact_id": document["artifact_id"],
        "files": source_entries,
        "semantic_contract": document["semantic_contract"],
        "source_manifest_sha256": _sha256_bytes(manifest_payload),
    }
    semantic_source_sha256 = _sha256_bytes(_canonical_json(semantic_identity))
    staging = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    if staging.exists() or staging.is_symlink():
        raise ArtifactSafetyError(f"staging path unexpectedly exists: {staging}")
    staging.mkdir(mode=0o700)
    try:
        artifact_manifests: dict[str, dict[str, Any]] = {}
        artifact_manifest_payloads: dict[str, bytes] = {}
        for kind in ("assurance", "release"):
            artifact_files: list[dict[str, Any]] = []
            for spec, projection, mode in projections:
                payload = projection.assurance if kind == "assurance" else projection.release
                destination = staging / kind / "files" / spec.output
                _write_file(destination, payload, mode)
                artifact_files.append(
                    {
                        "hot_path": spec.hot_path,
                        "mode": f"{mode:04o}",
                        "path": spec.output,
                        "sha256": _sha256_bytes(payload),
                    }
                )
            artifact = {
                "artifact_id": document["artifact_id"],
                "artifact_kind": kind,
                "files": artifact_files,
                "schema": ARTIFACT_SCHEMA,
                "semantic_contract": document["semantic_contract"],
                "semantic_source_sha256": semantic_source_sha256,
                "source_manifest_sha256": _sha256_bytes(manifest_payload),
                "tree_sha256": _tree_digest(artifact_files),
            }
            artifact_payload = _canonical_json(artifact)
            _write_file(staging / kind / "manifest.json", artifact_payload, 0o600)
            artifact_manifests[kind] = artifact
            artifact_manifest_payloads[kind] = artifact_payload

        regions = [
            {"file": spec.output, "names": list(projection.region_names)}
            for spec, projection, _mode in projections
            if projection.region_names
        ]
        certificate = {
            "artifact_id": document["artifact_id"],
            "assurance_manifest_sha256": _sha256_bytes(artifact_manifest_payloads["assurance"]),
            "assurance_regions": regions,
            "release_manifest_sha256": _sha256_bytes(artifact_manifest_payloads["release"]),
            "schema": EQUIVALENCE_SCHEMA,
            "semantic_contract": document["semantic_contract"],
            "semantic_source_sha256": semantic_source_sha256,
            "source_manifest_sha256": _sha256_bytes(manifest_payload),
        }
        _write_file(staging / "equivalence.json", _canonical_json(certificate), 0o600)
        _write_file(staging / "source-manifest.json", manifest_payload, 0o600)
        for directory in sorted(
            (path for path in staging.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(staging)
        staging.rename(output)
        _fsync_directory(output.parent)
    except BaseException:
        if staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return {
        "artifact_id": document["artifact_id"],
        "assurance_tree_sha256": artifact_manifests["assurance"]["tree_sha256"],
        "output": str(output),
        "release_tree_sha256": artifact_manifests["release"]["tree_sha256"],
        "semantic_source_sha256": semantic_source_sha256,
    }


def _load_canonical_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    _lstat_regular_owned(path, label)
    payload = path.read_bytes()
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactSafetyError(f"invalid {label}: {error}") from error
    if not isinstance(document, dict) or payload != _canonical_json(document):
        raise ArtifactSafetyError(f"{label} is not canonical JSON")
    return document, payload


def verify_artifacts(root: Path) -> dict[str, Any]:
    """Authenticate an existing artifact pair and repeat release-purity checks."""

    root = root.absolute()
    _require_owned_directory(root, "artifact root")
    certificate, _certificate_payload = _load_canonical_json(
        root / "equivalence.json", "equivalence certificate"
    )
    if certificate.get("schema") != EQUIVALENCE_SCHEMA:
        raise ArtifactSafetyError("equivalence certificate schema mismatch")
    source_document, source_payload, source_files = _load_source_manifest(
        root / "source-manifest.json"
    )
    if certificate.get("source_manifest_sha256") != _sha256_bytes(source_payload):
        raise ArtifactSafetyError("equivalence source manifest hash mismatch")

    manifests: dict[str, dict[str, Any]] = {}
    for kind in ("assurance", "release"):
        manifest, payload = _load_canonical_json(root / kind / "manifest.json", f"{kind} manifest")
        if manifest.get("schema") != ARTIFACT_SCHEMA or manifest.get("artifact_kind") != kind:
            raise ArtifactSafetyError(f"{kind} artifact identity mismatch")
        if manifest.get("semantic_source_sha256") != certificate.get("semantic_source_sha256"):
            raise ArtifactSafetyError(f"{kind} semantic source identity mismatch")
        if certificate.get(f"{kind}_manifest_sha256") != _sha256_bytes(payload):
            raise ArtifactSafetyError(f"{kind} manifest hash mismatch")
        entries = manifest.get("files")
        if not isinstance(entries, list):
            raise ArtifactSafetyError(f"{kind} manifest files must be an array")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ArtifactSafetyError(f"{kind} manifest file entry is invalid")
            relative = _validate_relative_path(entry.get("path"), f"{kind} file path")
            path = root / kind / "files" / relative
            metadata = _lstat_regular_owned(path, f"{kind} file {relative}")
            payload_bytes = path.read_bytes()
            if _sha256_bytes(payload_bytes) != entry.get("sha256"):
                raise ArtifactSafetyError(f"{kind} file hash mismatch: {relative}")
            if f"{stat.S_IMODE(metadata.st_mode):04o}" != entry.get("mode"):
                raise ArtifactSafetyError(f"{kind} file mode mismatch: {relative}")
            if kind == "release" and entry.get("hot_path") is True:
                _scan_release(payload_bytes, relative)
        if manifest.get("tree_sha256") != _tree_digest(entries):
            raise ArtifactSafetyError(f"{kind} tree hash mismatch")
        manifests[kind] = manifest

    if certificate.get("artifact_id") != source_document.get("artifact_id"):
        raise ArtifactSafetyError("equivalence artifact_id mismatch")
    source_outputs = {entry.output for entry in source_files}
    for kind, manifest in manifests.items():
        if {entry["path"] for entry in manifest["files"]} != source_outputs:
            raise ArtifactSafetyError(f"{kind} file set differs from semantic source")
    return {
        "artifact_id": certificate["artifact_id"],
        "assurance_tree_sha256": manifests["assurance"]["tree_sha256"],
        "release_tree_sha256": manifests["release"]["tree_sha256"],
        "semantic_source_sha256": certificate["semantic_source_sha256"],
        "verified": True,
    }


def _load_bounded_json(path: Path, label: str) -> dict[str, Any]:
    _lstat_regular_owned(path, label)
    payload = path.read_bytes()
    if len(payload) > 16 * 1024 * 1024:
        raise ArtifactSafetyError(f"{label} exceeds 16 MiB")
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactSafetyError(f"invalid {label}: {error}") from error
    if not isinstance(document, dict):
        raise ArtifactSafetyError(f"{label} must be a JSON object")
    return document


def _snapshot_payload_receipt(selection_path: Path) -> str:
    selection = _load_bounded_json(selection_path, "snapshot selection")
    bindings = selection.get("bindings")
    request = selection.get("request")
    if not isinstance(bindings, dict) or not isinstance(request, dict):
        raise ArtifactSafetyError("snapshot selection lacks producer bindings or request")
    try:
        provenance = _validate_payload_provenance(
            selection.get("payload_provenance"),
            profile=FIXED_SLOT_60K_PROFILE,
            bindings=bindings,
            request=request,
            captured_at=selection.get("captured_at"),
            source_catalog_sha256=selection.get("source_catalog_sha256"),
            label="snapshot payload provenance",
            uid=os.getuid(),
        )
    except CacheSafetyError as error:
        raise ArtifactSafetyError(f"invalid snapshot payload provenance: {error}") from error
    schema = provenance["schema"]
    if schema == FIXED_SLOT_PRODUCER_SCHEMA:
        claimed = provenance["receipt_sha256"]
    elif schema == HISTORICAL_FIXED_SLOT_PRODUCER_SCHEMA:
        claimed = provenance["attestation_sha256"]
    else:  # The shared provenance validator is fail closed; keep this local invariant explicit.
        raise ArtifactSafetyError("snapshot payload producer schema mismatch")
    if provenance["contract"] != FIXED_SLOT_PRODUCER_CONTRACT:
        raise ArtifactSafetyError("snapshot payload producer semantic contract mismatch")
    return claimed


def _manifest_file_sha256(manifest: dict[str, Any], relative_path: str) -> str:
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ArtifactSafetyError("assurance manifest has no file array")
    matches = [
        entry for entry in entries if isinstance(entry, dict) and entry.get("path") == relative_path
    ]
    if len(matches) != 1:
        raise ArtifactSafetyError(f"assurance manifest must contain exactly one {relative_path}")
    digest = matches[0].get("sha256")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise ArtifactSafetyError(f"assurance manifest has an invalid hash for {relative_path}")
    return digest


def _prove_single_file_patch(
    *,
    preimage: Path,
    postimage: Path,
    patch: Path,
    relative_path: str,
    label: str,
) -> None:
    """Apply one reviewed patch to its producer bytes and require the consumer bytes."""

    for path, path_label in (
        (preimage, f"{label} producer source"),
        (postimage, f"{label} consumer source"),
        (patch, f"{label} transition patch"),
    ):
        _lstat_regular_owned(path, path_label)
    relative = _validate_relative_path(relative_path, f"{label} relative path")
    with tempfile.TemporaryDirectory(prefix="qwen-consumer-transition-") as temporary:
        root = Path(temporary)
        staged = root / relative
        staged.parent.mkdir(mode=0o700, parents=True)
        shutil.copyfile(preimage, staged)
        result = subprocess.run(
            ["patch", "--batch", "--forward", "--fuzz=0", "--silent", "-p1", "-d", str(root)],
            input=patch.read_bytes(),
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.decode(errors="replace").strip()
            raise ArtifactSafetyError(f"{label} transition patch did not apply exactly: {detail}")
        regular_files = sorted(
            path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
        )
        if regular_files != [relative]:
            raise ArtifactSafetyError(
                f"{label} transition patch touched an unexpected file set: {regular_files}"
            )
        if staged.read_bytes() != postimage.read_bytes():
            raise ArtifactSafetyError(f"{label} transition patch does not produce the consumer")


def seal_state_producer_receipt(
    *,
    artifact_root: Path,
    snapshot_selection: Path,
    accepted_path_commit: Path,
    snapshot_format: Path,
    output: Path,
) -> dict[str, Any]:
    """Bind one assurance artifact to the immutable state payload it observes."""

    artifact_root = artifact_root.absolute()
    verified = verify_artifacts(artifact_root)
    assurance_manifest, assurance_payload = _load_canonical_json(
        artifact_root / "assurance/manifest.json", "assurance manifest"
    )
    exporter_sha256 = _manifest_file_sha256(assurance_manifest, STATE_EXPORTER_OUTPUT)
    exporter = artifact_root / "assurance/files" / STATE_EXPORTER_OUTPUT
    _lstat_regular_owned(exporter, "assurance state exporter")
    if _sha256_bytes(exporter.read_bytes()) != exporter_sha256:
        raise ArtifactSafetyError("assurance state exporter differs from its manifest")

    accepted_path_commit = accepted_path_commit.absolute()
    snapshot_format = snapshot_format.absolute()
    _lstat_regular_owned(accepted_path_commit, "accepted-path commit source")
    _lstat_regular_owned(snapshot_format, "snapshot-format source")
    snapshot_payload_receipt = _snapshot_payload_receipt(snapshot_selection.absolute())

    receipt = seal_producer_receipt(
        {
            "accepted_path_commit_sha256": _sha256_bytes(accepted_path_commit.read_bytes()),
            "cache_mapping_exporter_sha256": exporter_sha256,
            "convolution_layer_names": list(QUALIFIED_CONVOLUTION_STATE_NAMES),
            "draft_kv_exporter_sha256": exporter_sha256,
            "fixed_slot_state_exporter_sha256": exporter_sha256,
            "gdn_layer_names": list(QUALIFIED_GDN_STATE_NAMES),
            "runtime_artifact_manifest_sha256": _sha256_bytes(assurance_payload),
            "schema": PRODUCER_RECEIPT_SCHEMA,
            "semantic_source_sha256": verified["semantic_source_sha256"],
            "snapshot_payload_receipt_sha256": snapshot_payload_receipt,
            "snapshot_format_sha256": _sha256_bytes(snapshot_format.read_bytes()),
            "target_kv_exporter_sha256": exporter_sha256,
        }
    )
    output = output.absolute()
    _require_owned_directory(output.parent, "state-producer receipt parent")
    if output.exists() or output.is_symlink():
        raise ArtifactSafetyError(f"refusing to replace state-producer receipt: {output}")
    _write_file(output, _canonical_json(receipt), 0o600)
    _fsync_directory(output.parent)
    return {
        "output": str(output),
        "receipt_sha256": receipt["receipt_sha256"],
        "snapshot_payload_receipt_sha256": snapshot_payload_receipt,
    }


def seal_state_consumer_transition_receipt(
    *,
    artifact_root: Path,
    state_producer_receipt: Path,
    snapshot_selection: Path,
    producer_accepted_path_commit: Path,
    consumer_accepted_path_commit: Path,
    accepted_path_patch: Path,
    producer_snapshot_format: Path,
    consumer_snapshot_format: Path,
    snapshot_format_patch: Path,
    consumer_capability: Path,
    output: Path,
) -> dict[str, Any]:
    """Prove and seal one immutable-producer to exact-consumer source transition."""

    artifact_root = artifact_root.absolute()
    verified = verify_artifacts(artifact_root)
    assurance_manifest, assurance_payload = _load_canonical_json(
        artifact_root / "assurance/manifest.json", "assurance manifest"
    )
    producer_document = _load_bounded_json(
        state_producer_receipt.absolute(), "state-producer receipt"
    )
    try:
        producer = validate_producer_receipt(producer_document)
    except StateProviderError as error:
        raise ArtifactSafetyError(f"invalid state-producer receipt: {error}") from error

    selection = snapshot_selection.absolute()
    selection_sha256 = _sha256_bytes(selection.read_bytes())
    snapshot_payload_receipt = _snapshot_payload_receipt(selection)
    if snapshot_payload_receipt != producer.document["snapshot_payload_receipt_sha256"]:
        raise ArtifactSafetyError("snapshot selection differs from the immutable producer receipt")

    producer_accepted_path_commit = producer_accepted_path_commit.absolute()
    consumer_accepted_path_commit = consumer_accepted_path_commit.absolute()
    producer_snapshot_format = producer_snapshot_format.absolute()
    consumer_snapshot_format = consumer_snapshot_format.absolute()
    accepted_path_patch = accepted_path_patch.absolute()
    snapshot_format_patch = snapshot_format_patch.absolute()
    consumer_capability = consumer_capability.absolute()
    for path, label in (
        (producer_accepted_path_commit, "producer accepted-path source"),
        (consumer_accepted_path_commit, "consumer accepted-path source"),
        (producer_snapshot_format, "producer snapshot-format source"),
        (consumer_snapshot_format, "consumer snapshot-format source"),
        (accepted_path_patch, "accepted-path transition patch"),
        (snapshot_format_patch, "snapshot-format transition patch"),
        (consumer_capability, "consumer capability"),
    ):
        _lstat_regular_owned(path, label)

    producer_accepted_sha256 = _sha256_bytes(producer_accepted_path_commit.read_bytes())
    producer_snapshot_sha256 = _sha256_bytes(producer_snapshot_format.read_bytes())
    if producer_accepted_sha256 != producer.document["accepted_path_commit_sha256"]:
        raise ArtifactSafetyError("producer accepted-path source differs from its receipt")
    if producer_snapshot_sha256 != producer.document["snapshot_format_sha256"]:
        raise ArtifactSafetyError("producer snapshot-format source differs from its receipt")

    _prove_single_file_patch(
        preimage=producer_accepted_path_commit,
        postimage=consumer_accepted_path_commit,
        patch=accepted_path_patch,
        relative_path="vllm/v1/worker/gpu/model_runner.py",
        label="accepted-path",
    )
    _prove_single_file_patch(
        preimage=producer_snapshot_format,
        postimage=consumer_snapshot_format,
        patch=snapshot_format_patch,
        relative_path=(
            "vllm/distributed/kv_transfer/kv_connector/v1/offloading/qwen_persistent_selection.py"
        ),
        label="snapshot-format",
    )

    accepted_patch_sha256 = _sha256_bytes(accepted_path_patch.read_bytes())
    snapshot_patch_sha256 = _sha256_bytes(snapshot_format_patch.read_bytes())
    capability_sha256 = _sha256_bytes(consumer_capability.read_bytes())
    expected_artifact_members = {
        ACCEPTED_PATH_PATCH_OUTPUT: accepted_patch_sha256,
        SNAPSHOT_FORMAT_PATCH_OUTPUT: snapshot_patch_sha256,
        CONSUMER_CAPABILITY_OUTPUT: capability_sha256,
    }
    for relative_path, expected in expected_artifact_members.items():
        if _manifest_file_sha256(assurance_manifest, relative_path) != expected:
            raise ArtifactSafetyError(
                f"assurance artifact does not authenticate transition member {relative_path}"
            )

    capability_document = _load_bounded_json(consumer_capability, "consumer capability")
    runtime = capability_document.get("runtime")
    resume_binding = runtime.get("resume_binding") if isinstance(runtime, dict) else None
    greedy_transition = (
        resume_binding == "immutable-producer-to-exact-greedy-verifier-consumer-postimage"
        and runtime.get("verifier") == "default-off-exact-greedy-d7-c1-m8-target-argmax"
        and runtime.get("fallback")
        == "authenticated-general-rejection-sampler-on-fast-verifier-contract-miss"
        and runtime.get("accepted_count_source")
        == "post-sampler-num-sampled-after-external-limit-cap"
        and runtime.get("dflash_speculative_tokens") == 7
        and runtime.get("dflash_verification_rows") == 8
    )
    external_cap_transition = (
        resume_binding == "immutable-producer-to-exact-external-cap-consumer-postimage"
    )
    if (
        capability_document.get("schema") != "urn:qwen-r9700:capability:v1"
        or capability_document.get("default_off") is not True
        or capability_document.get("qualification_only") is not True
        or not isinstance(runtime, dict)
        or not (external_cap_transition or greedy_transition)
        or runtime.get("canonical_state_limit") != "prompt_len-plus-request-max_tokens"
        or runtime.get("discarded_suffix")
        != "masked-and-counted-as-rejected-before-accepted-path-replay"
    ):
        raise ArtifactSafetyError("consumer capability lacks the qualified transition contract")

    exporter_sha256 = _manifest_file_sha256(assurance_manifest, STATE_EXPORTER_OUTPUT)
    receipt = seal_consumer_transition_receipt(
        {
            "accepted_path_patch_sha256": accepted_patch_sha256,
            "consumer_accepted_path_commit_sha256": _sha256_bytes(
                consumer_accepted_path_commit.read_bytes()
            ),
            "consumer_capability_sha256": capability_sha256,
            "consumer_runtime_artifact_manifest_sha256": _sha256_bytes(assurance_payload),
            "consumer_semantic_source_sha256": verified["semantic_source_sha256"],
            "consumer_snapshot_format_sha256": _sha256_bytes(consumer_snapshot_format.read_bytes()),
            "consumer_state_exporter_sha256": exporter_sha256,
            "producer_accepted_path_commit_sha256": producer_accepted_sha256,
            "producer_receipt_sha256": producer.digest,
            "producer_runtime_artifact_manifest_sha256": producer.document[
                "runtime_artifact_manifest_sha256"
            ],
            "producer_semantic_source_sha256": producer.document["semantic_source_sha256"],
            "producer_snapshot_format_sha256": producer_snapshot_sha256,
            "schema": CONSUMER_TRANSITION_SCHEMA,
            "snapshot_format_patch_sha256": snapshot_patch_sha256,
            "snapshot_payload_receipt_sha256": snapshot_payload_receipt,
            "snapshot_selection_sha256": selection_sha256,
        }
    )
    output = output.absolute()
    _require_owned_directory(output.parent, "consumer-transition receipt parent")
    if output.exists() or output.is_symlink():
        raise ArtifactSafetyError(f"refusing to replace consumer-transition receipt: {output}")
    _write_file(output, _canonical_json(receipt), 0o600)
    _fsync_directory(output.parent)
    return {
        "consumer_runtime_artifact_manifest_sha256": receipt[
            "consumer_runtime_artifact_manifest_sha256"
        ],
        "output": str(output),
        "producer_receipt_sha256": producer.digest,
        "receipt_sha256": receipt["receipt_sha256"],
        "snapshot_payload_receipt_sha256": snapshot_payload_receipt,
    }


def rebind_assurance_consumer_transition_receipt(
    *,
    prior_artifact_root: Path,
    artifact_root: Path,
    prior_transition_receipt: Path,
    output: Path,
) -> dict[str, Any]:
    """Rebind a transition receipt after assurance-only source changes.

    Every non-assurance artifact member, the semantic contract, and the state
    exporter must remain byte-identical.  This permits diagnostic capture code
    to evolve without re-proving an unchanged runtime transition from source
    preimages that are deliberately no longer live.
    """

    prior_verified = verify_artifacts(prior_artifact_root)
    verified = verify_artifacts(artifact_root)
    prior_receipt_document, _ = _load_canonical_json(
        prior_transition_receipt, "prior consumer-transition receipt"
    )
    prior_receipt = validate_consumer_transition_receipt(prior_receipt_document)
    prior_manifest, prior_manifest_payload = _load_canonical_json(
        prior_artifact_root / "assurance" / "manifest.json",
        "prior assurance manifest",
    )
    manifest, manifest_payload = _load_canonical_json(
        artifact_root / "assurance" / "manifest.json", "assurance manifest"
    )
    prior_source, _prior_source_payload = _load_canonical_json(
        prior_artifact_root / "source-manifest.json", "prior source manifest"
    )
    source, _source_payload = _load_canonical_json(
        artifact_root / "source-manifest.json", "source manifest"
    )

    if prior_receipt.document["consumer_runtime_artifact_manifest_sha256"] != (
        _sha256_bytes(prior_manifest_payload)
    ):
        raise ArtifactSafetyError("prior transition receipt does not bind prior artifact")
    if (
        prior_receipt.document["consumer_semantic_source_sha256"]
        != prior_verified["semantic_source_sha256"]
    ):
        raise ArtifactSafetyError("prior transition receipt semantic source mismatch")
    if prior_source.get("artifact_id") != source.get("artifact_id") or prior_source.get(
        "semantic_contract"
    ) != source.get("semantic_contract"):
        raise ArtifactSafetyError("assurance rebind changed artifact identity or semantic contract")

    def runtime_members(document: dict[str, Any]) -> dict[str, tuple[str, str, bool]]:
        entries = document.get("files")
        if not isinstance(entries, list):
            raise ArtifactSafetyError("assurance manifest has no file array")
        members: dict[str, tuple[str, str, bool]] = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ArtifactSafetyError("assurance manifest contains an invalid member")
            path = entry["path"]
            if path.startswith("assurance/"):
                continue
            members[path] = (entry.get("sha256"), entry.get("mode"), entry.get("hot_path"))
        return members

    if runtime_members(prior_manifest) != runtime_members(manifest):
        raise ArtifactSafetyError("assurance rebind changed a runtime artifact member")
    prior_exporter = _manifest_file_sha256(prior_manifest, STATE_EXPORTER_OUTPUT)
    exporter = _manifest_file_sha256(manifest, STATE_EXPORTER_OUTPUT)
    if (
        prior_exporter != exporter
        or prior_receipt.document["consumer_state_exporter_sha256"] != prior_exporter
    ):
        raise ArtifactSafetyError("assurance rebind changed the state exporter")

    rebound_source = {
        key: value for key, value in prior_receipt.document.items() if key != "receipt_sha256"
    }
    rebound_source.update(
        {
            "consumer_runtime_artifact_manifest_sha256": _sha256_bytes(manifest_payload),
            "consumer_semantic_source_sha256": verified["semantic_source_sha256"],
            "consumer_state_exporter_sha256": exporter,
        }
    )
    rebound = seal_consumer_transition_receipt(rebound_source)
    output = output.absolute()
    _require_owned_directory(output.parent, "consumer-transition receipt parent")
    if output.exists() or output.is_symlink():
        raise ArtifactSafetyError(f"refusing to replace consumer-transition receipt: {output}")
    _write_file(output, _canonical_json(rebound), 0o600)
    _fsync_directory(output.parent)
    return {
        "consumer_runtime_artifact_manifest_sha256": rebound[
            "consumer_runtime_artifact_manifest_sha256"
        ],
        "output": str(output),
        "prior_receipt_sha256": prior_receipt.digest,
        "receipt_sha256": rebound["receipt_sha256"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or verify paired coding-turbo assurance/release artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="create an assurance/release artifact pair")
    build.add_argument("--source-manifest", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="authenticate an existing artifact pair")
    verify.add_argument("--root", type=Path, required=True)
    seal = subparsers.add_parser(
        "seal-state-receipt",
        help="bind an assurance artifact to one immutable fixed-slot payload producer",
    )
    seal.add_argument("--artifact-root", type=Path, required=True)
    seal.add_argument("--snapshot-selection", type=Path, required=True)
    seal.add_argument("--accepted-path-commit", type=Path, required=True)
    seal.add_argument("--snapshot-format", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    transition = subparsers.add_parser(
        "seal-consumer-transition",
        help="bind an immutable snapshot producer to one exact patched runtime consumer",
    )
    transition.add_argument("--artifact-root", type=Path, required=True)
    transition.add_argument("--state-producer-receipt", type=Path, required=True)
    transition.add_argument("--snapshot-selection", type=Path, required=True)
    transition.add_argument("--producer-accepted-path-commit", type=Path, required=True)
    transition.add_argument("--consumer-accepted-path-commit", type=Path, required=True)
    transition.add_argument("--accepted-path-patch", type=Path, required=True)
    transition.add_argument("--producer-snapshot-format", type=Path, required=True)
    transition.add_argument("--consumer-snapshot-format", type=Path, required=True)
    transition.add_argument("--snapshot-format-patch", type=Path, required=True)
    transition.add_argument("--consumer-capability", type=Path, required=True)
    transition.add_argument("--output", type=Path, required=True)
    rebind = subparsers.add_parser(
        "rebind-assurance-transition",
        help="rebind a transition after byte-identical runtime/assurance-only changes",
    )
    rebind.add_argument("--prior-artifact-root", type=Path, required=True)
    rebind.add_argument("--artifact-root", type=Path, required=True)
    rebind.add_argument("--prior-transition-receipt", type=Path, required=True)
    rebind.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            result = build_artifacts(args.source_manifest, args.output)
        elif args.command == "verify":
            result = verify_artifacts(args.root)
        elif args.command == "seal-state-receipt":
            result = seal_state_producer_receipt(
                artifact_root=args.artifact_root,
                snapshot_selection=args.snapshot_selection,
                accepted_path_commit=args.accepted_path_commit,
                snapshot_format=args.snapshot_format,
                output=args.output,
            )
        elif args.command == "seal-consumer-transition":
            result = seal_state_consumer_transition_receipt(
                artifact_root=args.artifact_root,
                state_producer_receipt=args.state_producer_receipt,
                snapshot_selection=args.snapshot_selection,
                producer_accepted_path_commit=args.producer_accepted_path_commit,
                consumer_accepted_path_commit=args.consumer_accepted_path_commit,
                accepted_path_patch=args.accepted_path_patch,
                producer_snapshot_format=args.producer_snapshot_format,
                consumer_snapshot_format=args.consumer_snapshot_format,
                snapshot_format_patch=args.snapshot_format_patch,
                consumer_capability=args.consumer_capability,
                output=args.output,
            )
        else:
            result = rebind_assurance_consumer_transition_receipt(
                prior_artifact_root=args.prior_artifact_root,
                artifact_root=args.artifact_root,
                prior_transition_receipt=args.prior_transition_receipt,
                output=args.output,
            )
    except (ArtifactSafetyError, StateProviderError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
