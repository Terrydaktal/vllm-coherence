"""Build, stage, verify, and atomically select gfx1201 kernel bundles.

The live runtime loads three independently compiled extensions.  This module
turns them into one immutable, content-addressed release so a single-kernel
experiment cannot accidentally mix unrecorded binaries.  It never starts or
stops a server and it never invokes a GPU workload; ``build`` only invokes the
existing host-side HIP extension build scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
MANIFEST_TYPE = "qwen-r9700-kernel-bundle"
DEFAULT_STORE = Path("/home/lewis/.local/share/qwen-r9700/kernel-bundles")
REQUIRED_COMPONENTS = ("native_b", "quest", "gdn")
VERSION_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62}[A-Za-z0-9])?")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class KernelWorkflowError(RuntimeError):
    """A bundle failed validation or a state-changing operation was unsafe."""


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    module: str
    build_script: str
    inputs: tuple[tuple[str, str], ...]
    extension_environment: str
    required_environment: tuple[tuple[str, str], ...]


COMPONENTS: dict[str, ComponentSpec] = {
    "native_b": ComponentSpec(
        name="native_b",
        module="rdna_w4a16_native_wmma_gfx1201_v10_native_a_fp16scale",
        build_script="experiments/bole-tree-verify/remote/build-rdna-native-w4a16",
        inputs=(
            (
                "source_cpp",
                "experiments/bole-tree-verify/remote/rdna_w4a16_native_ext.cpp",
            ),
            (
                "source_hip",
                "experiments/bole-tree-verify/remote/rdna_w4a16_native_ext.cu",
            ),
            (
                "runtime_bridge",
                "experiments/bole-tree-verify/remote/rdna_hybrid_w4a16.py",
            ),
        ),
        extension_environment="VLLM_RDNA_W4A16_NATIVE_EXTENSION",
        required_environment=(
            ("VLLM_RDNA_W4A16_NATIVE", "1"),
            ("VLLM_RDNA_W4A16_NATIVE_REQUIRED", "1"),
        ),
    ),
    "quest": ComponentSpec(
        name="quest",
        module="quest_fp8_selector_gfx1201",
        build_script=("experiments/bole-tree-verify/hip/sub1ms-quest/build-quest-fp8-selector"),
        inputs=(
            (
                "source_cpp",
                "experiments/bole-tree-verify/hip/sub1ms-quest/quest_fp8_selector_ext.cpp",
            ),
            (
                "source_hip",
                "experiments/bole-tree-verify/hip/sub1ms-quest/quest_fp8_selector_ext.cu",
            ),
            (
                "runtime_bridge",
                "experiments/bole-tree-verify/hip/sub1ms-quest/quest_vllm_attention.py",
            ),
            (
                "runtime_backend",
                "experiments/bole-tree-verify/remote/tree-abi-overlay-current/"
                "vllm/v1/attention/backends/rocm_attn.py",
            ),
        ),
        extension_environment="QWEN_QUEST_SELECTOR_EXTENSION",
        required_environment=(
            ("QWEN_QUEST_REQUIRE_COMPILED", "1"),
            ("QWEN_QUEST_REQUIRE_DIRECT", "1"),
            ("QWEN_QUEST_SELECTOR", "compiled"),
        ),
    ),
    "gdn": ComponentSpec(
        name="gdn",
        module="rdna_gdn_decode_gfx1201",
        build_script="experiments/bole-tree-verify/remote/build-rdna-gdn-decode",
        inputs=(
            (
                "source_cpp",
                "experiments/bole-tree-verify/remote/rdna_gdn_decode_ext.cpp",
            ),
            (
                "source_hip",
                "experiments/bole-tree-verify/remote/rdna_gdn_decode_ext.cu",
            ),
            (
                "runtime_bridge",
                "experiments/bole-tree-verify/remote/qwen_gdn_linear_attn.py",
            ),
        ),
        extension_environment="QWEN_GDN_RDNA_EXTENSION",
        required_environment=(
            ("QWEN_GDN_RDNA_REQUIRED", "1"),
            ("VLLM_GDN_DECODE_KERNEL", "cuda"),
        ),
    ),
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_id(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("bundle_id", None)
    return f"sha256:{hashlib.sha256(canonical_bytes(unsigned)).hexdigest()}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _require_plain_directory(path: Path, label: str, *, create: bool = False) -> Path:
    path = path.expanduser().absolute()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir() or path.is_symlink():
        raise KernelWorkflowError(f"{label} must be a real directory, not a symlink: {path}")
    return path


def _require_regular_file(path: Path, label: str) -> Path:
    path = path.expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise KernelWorkflowError(f"{label} must be a regular non-symlink file: {path}")
    return path


def _safe_relative_path(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise KernelWorkflowError(f"{label} must be a non-empty relative path")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
        raise KernelWorkflowError(f"{label} escapes the release: {value!r}")
    if str(candidate) != value:
        raise KernelWorkflowError(f"{label} is not normalized: {value!r}")
    return candidate


def _source_records(project_root: Path, spec: ComponentSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    paths = (("build_script", spec.build_script), *spec.inputs)
    for role, relative in paths:
        source = _require_regular_file(project_root / relative, f"{spec.name} {role}")
        records.append(
            {
                "role": role,
                "path": relative,
                "sha256": sha256_file(source),
                "size": source.stat().st_size,
            }
        )
    return records


def _artifact_record(path: Path, relative: str) -> dict[str, Any]:
    path = _require_regular_file(path, "kernel artifact")
    with path.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise KernelWorkflowError(f"kernel artifact is not an ELF shared object: {path}")
    return {
        "path": relative,
        "format": "elf-shared-object",
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def _copy_regular(source: Path, destination: Path) -> None:
    source = _require_regular_file(source, "source artifact")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise KernelWorkflowError(f"refusing to replace staged artifact: {destination}")
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, 8 * 1024 * 1024)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    destination.chmod(0o444)


def _validate_version(version: str) -> str:
    if VERSION_RE.fullmatch(version) is None:
        raise KernelWorkflowError(
            "version must be 1-64 ASCII letters, digits, dots, underscores, or hyphens"
        )
    return version


def _load_json_object(path: Path) -> dict[str, Any]:
    path = _require_regular_file(path, "bundle manifest")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise KernelWorkflowError(f"cannot read bundle manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise KernelWorkflowError(f"bundle manifest must contain a JSON object: {path}")
    return value


def _validate_hash_record(record: object, label: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise KernelWorkflowError(f"{label} must be an object")
    if SHA256_RE.fullmatch(str(record.get("sha256", ""))) is None:
        raise KernelWorkflowError(f"{label} has an invalid SHA-256")
    size = record.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise KernelWorkflowError(f"{label} has an invalid size")
    return record


def verify_release(
    release: Path,
    *,
    require_components: Iterable[str] = REQUIRED_COMPONENTS,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Verify manifest identity, contained artifacts, and optionally source inputs."""

    release = _require_plain_directory(release, "release")
    if project_root is not None:
        project_root = _require_plain_directory(project_root, "project root")
    manifest_path = _require_regular_file(release / "manifest.json", "bundle manifest")
    manifest = _load_json_object(manifest_path)
    expected_manifest_keys = {
        "schema_version",
        "manifest_type",
        "bundle_id",
        "version",
        "created_at",
        "target",
        "components",
    }
    if set(manifest) != expected_manifest_keys:
        raise KernelWorkflowError("kernel bundle manifest has missing or unknown top-level fields")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise KernelWorkflowError("unsupported kernel bundle schema_version")
    if manifest.get("manifest_type") != MANIFEST_TYPE:
        raise KernelWorkflowError("unexpected kernel bundle manifest_type")
    if manifest.get("target") != {"arch": "gfx1201", "runtime": "ROCm"}:
        raise KernelWorkflowError("kernel bundle target is not the pinned gfx1201 ROCm target")
    version = manifest.get("version")
    if not isinstance(version, str):
        raise KernelWorkflowError("kernel bundle version is missing")
    _validate_version(version)
    expected_id = _manifest_id(manifest)
    if manifest.get("bundle_id") != expected_id:
        raise KernelWorkflowError(
            f"bundle_id mismatch: expected {expected_id}, found {manifest.get('bundle_id')!r}"
        )

    components = manifest.get("components")
    if not isinstance(components, dict):
        raise KernelWorkflowError("kernel bundle components must be an object")
    unknown = set(components) - set(COMPONENTS)
    if unknown:
        raise KernelWorkflowError(f"unknown kernel components: {sorted(unknown)}")
    missing = set(require_components) - set(components)
    if missing:
        raise KernelWorkflowError(f"kernel bundle is missing components: {sorted(missing)}")

    release_resolved = release.resolve(strict=True)
    for name, raw_component in components.items():
        if not isinstance(raw_component, dict):
            raise KernelWorkflowError(f"component {name} must be an object")
        if set(raw_component) != {
            "module",
            "extension_environment",
            "required_environment",
            "artifact",
            "inputs",
            "build",
            "provenance",
        }:
            raise KernelWorkflowError(f"component {name} has missing or unknown fields")
        spec = COMPONENTS[name]
        if raw_component.get("module") != spec.module:
            raise KernelWorkflowError(f"component {name} has an unexpected module name")
        if raw_component.get("extension_environment") != spec.extension_environment:
            raise KernelWorkflowError(f"component {name} has an unexpected environment contract")
        if raw_component.get("required_environment") != dict(spec.required_environment):
            raise KernelWorkflowError(
                f"component {name} has unexpected required environment values"
            )
        if raw_component.get("build") != {
            "script": spec.build_script,
            "module": spec.module,
            "arch": "gfx1201",
        }:
            raise KernelWorkflowError(f"component {name} has an unexpected build contract")
        provenance = raw_component.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("kind") not in {
            "built",
            "inherited",
            "prebuilt",
        }:
            raise KernelWorkflowError(f"component {name} has invalid provenance")
        artifact = _validate_hash_record(raw_component.get("artifact"), f"{name} artifact")
        if set(artifact) != {"path", "format", "sha256", "size"}:
            raise KernelWorkflowError(f"component {name} artifact has missing or unknown fields")
        if artifact.get("format") != "elf-shared-object":
            raise KernelWorkflowError(f"component {name} has an unexpected artifact format")
        relative = _safe_relative_path(artifact.get("path"), f"{name} artifact path")
        expected_artifact_path = f"components/{name}/{spec.module}.so"
        if str(relative) != expected_artifact_path:
            raise KernelWorkflowError(f"component {name} has an unexpected artifact path")
        artifact_path = release.joinpath(*relative.parts)
        artifact_path = _require_regular_file(artifact_path, f"{name} artifact")
        if artifact_path.resolve(strict=True).parent != (release_resolved / "components" / name):
            raise KernelWorkflowError(
                f"component {name} artifact is outside its component directory"
            )
        if artifact_path.stat().st_size != artifact["size"]:
            raise KernelWorkflowError(f"component {name} artifact size mismatch")
        if sha256_file(artifact_path) != artifact["sha256"]:
            raise KernelWorkflowError(f"component {name} artifact SHA-256 mismatch")
        with artifact_path.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                raise KernelWorkflowError(f"component {name} artifact is not an ELF shared object")

        inputs = raw_component.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise KernelWorkflowError(f"component {name} inputs must be a non-empty array")
        seen_roles: set[str] = set()
        expected_inputs = {"build_script": spec.build_script, **dict(spec.inputs)}
        for index, raw_input in enumerate(inputs):
            input_record = _validate_hash_record(raw_input, f"{name} input {index}")
            if set(input_record) != {"role", "path", "sha256", "size"}:
                raise KernelWorkflowError(f"component {name} input has missing or unknown fields")
            role = input_record.get("role")
            if not isinstance(role, str) or not role or role in seen_roles:
                raise KernelWorkflowError(f"component {name} has invalid or duplicate input roles")
            seen_roles.add(role)
            source_relative = _safe_relative_path(
                input_record.get("path"), f"{name} input {index} path"
            )
            if expected_inputs.get(role) != str(source_relative):
                raise KernelWorkflowError(f"component {name} has an unexpected {role} input path")
            if project_root is not None:
                source = _require_regular_file(
                    project_root.joinpath(*source_relative.parts), f"{name} input {role}"
                )
                if source.stat().st_size != input_record["size"]:
                    raise KernelWorkflowError(f"component {name} input {role} size mismatch")
                if sha256_file(source) != input_record["sha256"]:
                    raise KernelWorkflowError(f"component {name} input {role} SHA-256 mismatch")
        if seen_roles != set(expected_inputs):
            raise KernelWorkflowError(f"component {name} input inventory is incomplete")
    return manifest


def _component_from_release(
    release: Path,
    manifest: Mapping[str, Any],
    name: str,
    staging: Path,
) -> dict[str, Any]:
    component = manifest["components"][name]
    artifact = component["artifact"]
    relative = _safe_relative_path(artifact["path"], f"{name} inherited artifact path")
    source = release.joinpath(*relative.parts)
    destination_relative = f"components/{name}/{COMPONENTS[name].module}.so"
    destination = staging / destination_relative
    _copy_regular(source, destination)
    copied = json.loads(json.dumps(component))
    copied["artifact"] = _artifact_record(destination, destination_relative)
    copied["provenance"] = {
        "kind": "inherited",
        "bundle_id": manifest["bundle_id"],
    }
    return copied


def _new_component(
    project_root: Path,
    name: str,
    artifact_path: Path,
    staging: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    spec = COMPONENTS[name]
    destination_relative = f"components/{name}/{spec.module}.so"
    destination = staging / destination_relative
    _copy_regular(artifact_path, destination)
    return {
        "module": spec.module,
        "extension_environment": spec.extension_environment,
        "required_environment": dict(spec.required_environment),
        "artifact": _artifact_record(destination, destination_relative),
        "inputs": _source_records(project_root, spec),
        "build": {
            "script": spec.build_script,
            "module": spec.module,
            "arch": "gfx1201",
        },
        "provenance": dict(provenance),
    }


def _resolve_active(store: Path, *, required: bool) -> tuple[Path, dict[str, Any]] | None:
    active = store / "active"
    if not active.exists() and not active.is_symlink():
        if required:
            raise KernelWorkflowError(
                "no active baseline exists; stage all three components for the first release"
            )
        return None
    if not active.is_symlink():
        raise KernelWorkflowError(f"active pointer is not a symlink: {active}")
    target = active.readlink()
    if target.is_absolute():
        raise KernelWorkflowError("active pointer must use a relative target")
    release = (active.parent / target).resolve(strict=True)
    releases = (store / "releases").resolve(strict=True)
    if release.parent != releases:
        raise KernelWorkflowError("active pointer escapes the releases directory")
    return release, verify_release(release)


def stage_release(
    *,
    project_root: Path,
    store: Path,
    version: str,
    artifacts: Mapping[str, Path],
    inherit_active: bool = True,
    provenance: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Create one immutable release, inheriting unmodified active components."""

    project_root = _require_plain_directory(project_root, "project root")
    store = _require_plain_directory(store, "kernel bundle store", create=True)
    releases = store / "releases"
    releases.mkdir(mode=0o755, exist_ok=True)
    releases = _require_plain_directory(releases, "kernel release store")
    version = _validate_version(version)
    if not artifacts:
        raise KernelWorkflowError("at least one component artifact is required")
    unknown = set(artifacts) - set(COMPONENTS)
    if unknown:
        raise KernelWorkflowError(f"unknown kernel components: {sorted(unknown)}")

    active_record = None
    missing = set(REQUIRED_COMPONENTS) - set(artifacts)
    if missing and inherit_active:
        active_record = _resolve_active(store, required=True)
    elif missing:
        raise KernelWorkflowError(
            f"partial release requires active inheritance; missing {sorted(missing)}"
        )

    staging = Path(tempfile.mkdtemp(prefix=".stage-", dir=releases))
    try:
        component_records: dict[str, Any] = {}
        for name in REQUIRED_COMPONENTS:
            if name in artifacts:
                component_records[name] = _new_component(
                    project_root,
                    name,
                    artifacts[name],
                    staging,
                    (provenance or {}).get(name, {"kind": "prebuilt"}),
                )
            else:
                assert active_record is not None
                component_records[name] = _component_from_release(
                    active_record[0], active_record[1], name, staging
                )

        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "manifest_type": MANIFEST_TYPE,
            "version": version,
            "created_at": _utc_now(),
            "target": {"arch": "gfx1201", "runtime": "ROCm"},
            "components": component_records,
        }
        manifest["bundle_id"] = _manifest_id(manifest)
        suffix = manifest["bundle_id"].removeprefix("sha256:")[:16]
        final = releases / f"{version}-{suffix}"
        if final.exists() or final.is_symlink():
            raise KernelWorkflowError(f"refusing to replace existing release: {final}")
        manifest_path = staging / "manifest.json"
        with manifest_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        manifest_path.chmod(0o444)
        staging.rename(final)
        staging = Path()
        verify_release(final, project_root=project_root)
        return final
    finally:
        if staging != Path() and staging.exists():
            shutil.rmtree(staging)


def environment_for_release(release: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    release = release.resolve(strict=True)
    values = {
        "QWEN_KERNEL_BUNDLE_ID": str(manifest["bundle_id"]),
        "QWEN_KERNEL_BUNDLE_MANIFEST": str(release / "manifest.json"),
    }
    for name in REQUIRED_COMPONENTS:
        component = manifest["components"][name]
        artifact = _safe_relative_path(component["artifact"]["path"], f"{name} artifact")
        values[component["extension_environment"]] = str(release.joinpath(*artifact.parts))
        values.update(
            {str(key): str(value) for key, value in component["required_environment"].items()}
        )
    return values


def _run_build(script: Path, build_dir: Path, log_path: Path) -> None:
    script = _require_regular_file(script, "component build script")
    if not os.access(script, os.X_OK):
        raise KernelWorkflowError(f"component build script is not executable: {script}")
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(script), str(build_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            # Keep stdout machine-readable: the command emits exactly one JSON
            # document there after staging. Compiler progress belongs on stderr.
            sys.stderr.write(line)
            sys.stderr.flush()
            log.write(line)
        return_code = process.wait()
        log.flush()
        os.fsync(log.fileno())
    if return_code != 0:
        raise KernelWorkflowError(f"kernel build failed with exit status {return_code}: {script}")


def build_release(
    *,
    project_root: Path,
    store: Path,
    version: str,
    components: Sequence[str],
    inherit_active: bool = True,
) -> Path:
    """Run selected host build scripts and stage their outputs as one release."""

    project_root = _require_plain_directory(project_root, "project root")
    names = tuple(dict.fromkeys(components))
    if not names:
        names = REQUIRED_COMPONENTS
    unknown = set(names) - set(COMPONENTS)
    if unknown:
        raise KernelWorkflowError(f"unknown kernel components: {sorted(unknown)}")
    if set(names) != set(REQUIRED_COMPONENTS) and not inherit_active:
        raise KernelWorkflowError(
            "--no-inherit-active requires all three components in the same build"
        )
    if set(names) != set(REQUIRED_COMPONENTS):
        # Fail before spending minutes compiling an experiment that cannot form
        # a complete release.
        _resolve_active(
            _require_plain_directory(store, "kernel bundle store", create=True), required=True
        )

    workspace = Path(tempfile.mkdtemp(prefix="qwen-kernel-build-"))
    try:
        artifacts: dict[str, Path] = {}
        provenance: dict[str, dict[str, Any]] = {}
        for name in names:
            spec = COMPONENTS[name]
            before = _source_records(project_root, spec)
            build_dir = workspace / name
            build_dir.mkdir()
            log_path = workspace / f"{name}.log"
            _run_build(project_root / spec.build_script, build_dir, log_path)
            expected = build_dir / f"{spec.module}.so"
            _require_regular_file(expected, f"{name} build output")
            after = _source_records(project_root, spec)
            if before != after:
                raise KernelWorkflowError(f"{name} inputs changed while the extension was building")
            artifacts[name] = expected
            provenance[name] = {
                "kind": "built",
                "build_log_sha256": sha256_file(log_path),
                "build_log_size": log_path.stat().st_size,
            }
        return stage_release(
            project_root=project_root,
            store=store,
            version=version,
            artifacts=artifacts,
            inherit_active=inherit_active,
            provenance=provenance,
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _pointer_release(store: Path, pointer_name: str) -> Path | None:
    pointer = store / pointer_name
    if not pointer.exists() and not pointer.is_symlink():
        return None
    if not pointer.is_symlink():
        raise KernelWorkflowError(f"{pointer_name} pointer is not a symlink: {pointer}")
    target = pointer.readlink()
    if target.is_absolute():
        raise KernelWorkflowError(f"{pointer_name} pointer must use a relative target")
    resolved = (pointer.parent / target).resolve(strict=True)
    releases = (store / "releases").resolve(strict=True)
    if resolved.parent != releases:
        raise KernelWorkflowError(f"{pointer_name} pointer escapes the releases directory")
    return resolved


def _atomic_pointer(store: Path, pointer_name: str, release: Path) -> None:
    relative = Path("releases") / release.name
    temporary = store / f".{pointer_name}-{uuid.uuid4().hex}"
    try:
        temporary.symlink_to(relative)
        temporary.replace(store / pointer_name)
        directory_fd = os.open(store, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def activate_release(store: Path, release: Path) -> dict[str, Any]:
    """Verify a complete release and atomically select it, retaining the prior pointer."""

    store = _require_plain_directory(store, "kernel bundle store")
    releases = _require_plain_directory(store / "releases", "kernel release store")
    release = _require_plain_directory(release, "release")
    if release.resolve(strict=True).parent != releases.resolve(strict=True):
        raise KernelWorkflowError("activation release must be an immediate child of store/releases")
    manifest = verify_release(release)
    current = _pointer_release(store, "active")
    if current == release.resolve(strict=True):
        return manifest
    if current is not None:
        _atomic_pointer(store, "previous", current)
    _atomic_pointer(store, "active", release)
    # Verify the pointer target after the switch.  The selected release was
    # verified before any pointer changed, so this is an invariant check rather
    # than a late validation step that could expose an untrusted binary.
    selected = _pointer_release(store, "active")
    if selected != release.resolve(strict=True):
        raise KernelWorkflowError("active pointer did not resolve to the selected release")
    return manifest


def rollback_release(store: Path) -> dict[str, Any]:
    """Atomically select the verified previous release and retain the former active one."""

    store = _require_plain_directory(store, "kernel bundle store")
    active = _pointer_release(store, "active")
    previous = _pointer_release(store, "previous")
    if active is None or previous is None:
        raise KernelWorkflowError("rollback requires both active and previous releases")
    manifest = verify_release(previous)
    verify_release(active)
    _atomic_pointer(store, "active", previous)
    _atomic_pointer(store, "previous", active)
    return manifest


def inspect_release(
    release: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    manifest = verify_release(release, project_root=project_root)
    return {
        "release": str(release.resolve(strict=True)),
        "manifest": str((release / "manifest.json").resolve(strict=True)),
        "bundle_id": manifest["bundle_id"],
        "version": manifest["version"],
        "target": manifest["target"],
        "components": {
            name: {
                "path": environment_for_release(release, manifest)[
                    COMPONENTS[name].extension_environment
                ],
                "sha256": manifest["components"][name]["artifact"]["sha256"],
                "size": manifest["components"][name]["artifact"]["size"],
                "provenance": manifest["components"][name]["provenance"],
            }
            for name in REQUIRED_COMPONENTS
        },
        "environment": environment_for_release(release, manifest),
    }


def _project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def _artifact_argument(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or name not in COMPONENTS or not raw_path:
        raise argparse.ArgumentTypeError(
            "artifact must be COMPONENT=PATH where COMPONENT is native_b, quest, or gdn"
        )
    return name, Path(raw_path)


def _release_argument(value: str, store: Path) -> Path:
    if value == "active":
        release = _pointer_release(store, "active")
        if release is None:
            raise KernelWorkflowError("no active release")
        return release
    if value == "previous":
        release = _pointer_release(store, "previous")
        if release is None:
            raise KernelWorkflowError("no previous release")
        return release
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = store / "releases" / candidate
    if candidate.name == "manifest.json" or candidate.is_file():
        candidate = candidate.parent
    if candidate.is_symlink():
        candidate = candidate.resolve(strict=True)
    return candidate


def _print_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and stage immutable gfx1201 Native-B, Quest, and GDN kernel bundles "
            "without starting inference."
        )
    )
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--project-root", type=Path, default=_project_root_from_script())
    subparsers = parser.add_subparsers(dest="operation", required=True)

    build = subparsers.add_parser("build", help="compile selected components and stage a release")
    build.add_argument("--version", required=True)
    build.add_argument("--component", action="append", choices=REQUIRED_COMPONENTS, default=[])
    build.add_argument(
        "--no-inherit-active",
        action="store_true",
        help="require all three components in this build instead of inheriting unchanged binaries",
    )

    stage = subparsers.add_parser("stage", help="stage prebuilt component artifacts")
    stage.add_argument("--version", required=True)
    stage.add_argument("--artifact", action="append", type=_artifact_argument, required=True)
    stage.add_argument("--no-inherit-active", action="store_true")

    verify = subparsers.add_parser("verify", help="verify a release and its current sources")
    verify.add_argument("release")
    verify.add_argument(
        "--artifacts-only",
        action="store_true",
        help="verify the portable bundle without requiring the source checkout to match",
    )

    activate = subparsers.add_parser("activate", help="atomically select a verified release")
    activate.add_argument("release")

    inspect = subparsers.add_parser("inspect", help="emit resolved paths, hashes, and environment")
    inspect.add_argument("release", nargs="?", default="active")
    inspect.add_argument("--format", choices=("json", "env"), default="json")
    inspect.add_argument("--verify-sources", action="store_true")

    subparsers.add_parser(
        "rollback", help="atomically reselect previous and retain the former active release"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        store = args.store.expanduser().absolute()
        project_root = args.project_root.expanduser().absolute()
        if args.operation == "build":
            release = build_release(
                project_root=project_root,
                store=store,
                version=args.version,
                components=args.component,
                inherit_active=not args.no_inherit_active,
            )
            _print_json(inspect_release(release, project_root=project_root))
        elif args.operation == "stage":
            artifacts = dict(args.artifact)
            if len(artifacts) != len(args.artifact):
                raise KernelWorkflowError("each staged component may be supplied only once")
            release = stage_release(
                project_root=project_root,
                store=store,
                version=args.version,
                artifacts=artifacts,
                inherit_active=not args.no_inherit_active,
            )
            _print_json(inspect_release(release, project_root=project_root))
        elif args.operation == "verify":
            release = _release_argument(args.release, store)
            manifest = verify_release(
                release,
                project_root=None if args.artifacts_only else project_root,
            )
            _print_json(
                {
                    "status": "verified",
                    "release": str(release.resolve(strict=True)),
                    "bundle_id": manifest["bundle_id"],
                }
            )
        elif args.operation == "activate":
            release = _release_argument(args.release, store)
            manifest = activate_release(store, release)
            _print_json(
                {
                    "status": "active",
                    "release": str(release.resolve(strict=True)),
                    "bundle_id": manifest["bundle_id"],
                }
            )
        elif args.operation == "inspect":
            release = _release_argument(args.release, store)
            payload = inspect_release(
                release,
                project_root=project_root if args.verify_sources else None,
            )
            if args.format == "env":
                for key, value in sorted(payload["environment"].items()):
                    print(f"export {key}={shlex.quote(value)}")
            else:
                _print_json(payload)
        elif args.operation == "rollback":
            manifest = rollback_release(store)
            active = _pointer_release(store, "active")
            assert active is not None
            _print_json(
                {
                    "status": "rolled-back",
                    "release": str(active),
                    "bundle_id": manifest["bundle_id"],
                }
            )
        else:  # pragma: no cover - argparse makes this unreachable.
            parser.error(f"unknown operation: {args.operation}")
    except (KernelWorkflowError, OSError, subprocess.SubprocessError) as error:
        print(f"kernel-workflow: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
