"""Execute one complete, authenticated serial/M8 assurance campaign.

The instrumentation module defines what evidence must exist.  This controller
is the fail-closed bridge from that declaration to real producer processes: it
plans every fragment, launches the sole declared producer for each fragment,
authenticates the producer's create-only output, assembles both complete traces,
and compares them before publishing a completion marker.

It deliberately cannot manufacture probe events.  A producer that exits zero
without writing its exact fragment, writes an incomplete fragment, or claims a
scope outside its producer class fails the campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qwen_r9700_lab import assurance_instrumentation as instrumentation

SPEC_SCHEMA = "urn:qwen-r9700:full-assurance-campaign-controller-spec:v2"
PLAN_SCHEMA = "urn:qwen-r9700:full-assurance-campaign-controller-plan:v2"
COMPLETE_SCHEMA = "urn:qwen-r9700:full-assurance-campaign-controller-complete:v2"
FAILURE_SCHEMA = "urn:qwen-r9700:full-assurance-campaign-controller-failure:v2"
ARMS = instrumentation.ARMS
PRODUCER_KINDS = instrumentation.PRODUCER_KINDS
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ENVIRONMENT = frozenset({"HOME", "LANG", "LC_ALL", "PATH", "TMPDIR"})
MAX_SPEC_BYTES = 4 << 20
MAX_CAPTURE_BYTES = 8 << 20
MAX_DEPENDENCY_BYTES = 8 << 30

_DESCRIPTOR_KEYS = {
    "argv",
    "cwd",
    "dependencies",
    "environment",
    "launcher_path",
    "launcher_sha256",
    "producer_path",
    "producer_sha256",
    "timeout_seconds",
}
_DEPENDENCY_KEYS = {"path", "sha256"}
_ARM_KEYS = {"header_path", "header_sha256", "producers"}
_SPEC_KEYS = {"arms", "controller_source_sha256", "schema"}


class CampaignControllerError(RuntimeError):
    """The campaign cannot establish complete authenticated evidence."""


def _canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise CampaignControllerError(f"{label} must be a lowercase SHA-256")
    return value


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_mode


def _stable_owned_file(
    path: Path, label: str, *, expected_sha256: str | None = None, maximum: int
) -> bytes:
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise CampaignControllerError(f"{label} path must be normalized and absolute")
    try:
        before = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_mode & 0o022
            or not 0 < before.st_size <= maximum
        ):
            raise CampaignControllerError(f"{label} is not a safe owned regular file")
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise CampaignControllerError(f"cannot read {label}: {error}") from error
    if _stat_identity(before) != _stat_identity(after):
        raise CampaignControllerError(f"{label} changed while being authenticated")
    observed = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and observed != expected_sha256:
        raise CampaignControllerError(
            f"{label} SHA-256 differs: expected {expected_sha256}, observed {observed}"
        )
    return payload


def _authenticate_large_owned_file(
    path: Path, label: str, *, expected_sha256: str, maximum: int
) -> None:
    """Stream-authenticate a large settled dependency without loading it into RAM."""

    if not path.is_absolute() or path != path.resolve(strict=False):
        raise CampaignControllerError(f"{label} path must be normalized and absolute")
    try:
        before = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_mode & 0o022
            or not 0 < before.st_size <= maximum
        ):
            raise CampaignControllerError(f"{label} is not a safe owned regular file")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 << 20), b""):
                digest.update(chunk)
        after = path.lstat()
    except OSError as error:
        raise CampaignControllerError(f"cannot read {label}: {error}") from error
    if _stat_identity(before) != _stat_identity(after):
        raise CampaignControllerError(f"{label} changed while being authenticated")
    observed = digest.hexdigest()
    if observed != expected_sha256:
        raise CampaignControllerError(
            f"{label} SHA-256 differs: expected {expected_sha256}, observed {observed}"
        )


def _load_json_file(path: Path, label: str, expected_sha256: str) -> object:
    payload = _stable_owned_file(
        path,
        label,
        expected_sha256=expected_sha256,
        maximum=MAX_SPEC_BYTES,
    )
    try:
        return json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignControllerError(f"{label} is not valid JSON: {error}") from error


def _write_create_only(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise CampaignControllerError(f"short write while creating {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _normalize_environment(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise CampaignControllerError(f"{label} must be an object")
    result: dict[str, str] = {}
    forbidden = instrumentation.FULL_ASSURANCE_ENVIRONMENT | {
        instrumentation.FULL_ASSURANCE_ENABLE_ENV
    }
    for name, raw in value.items():
        if (
            not isinstance(name, str)
            or not name
            or "=" in name
            or "\x00" in name
            or name in forbidden
            or not isinstance(raw, str)
            or "\x00" in raw
        ):
            raise CampaignControllerError(f"{label} contains an unsafe entry")
        result[name] = raw
    return dict(sorted(result.items()))


def _normalize_descriptor(value: object, arm: str, producer_kind: str) -> dict[str, Any]:
    label = f"{arm}.{producer_kind} producer"
    if not isinstance(value, dict) or set(value) != _DESCRIPTOR_KEYS:
        raise CampaignControllerError(f"{label} keys are invalid")
    descriptor = dict(value)
    launcher = Path(descriptor["launcher_path"])
    producer = Path(descriptor["producer_path"])
    launcher_sha256 = _require_digest(descriptor["launcher_sha256"], f"{label} launcher")
    producer_sha256 = _require_digest(descriptor["producer_sha256"], f"{label} source")
    _stable_owned_file(
        launcher,
        f"{label} launcher",
        expected_sha256=launcher_sha256,
        maximum=MAX_CAPTURE_BYTES,
    )
    _stable_owned_file(
        producer,
        f"{label} source",
        expected_sha256=producer_sha256,
        maximum=MAX_CAPTURE_BYTES,
    )
    raw_dependencies = descriptor["dependencies"]
    if not isinstance(raw_dependencies, list):
        raise CampaignControllerError(f"{label} dependencies must be a list")
    dependencies: list[dict[str, str]] = []
    dependency_paths: set[str] = set()
    for index, raw_dependency in enumerate(raw_dependencies):
        dependency_label = f"{label} dependency {index}"
        if not isinstance(raw_dependency, dict) or set(raw_dependency) != _DEPENDENCY_KEYS:
            raise CampaignControllerError(f"{dependency_label} keys are invalid")
        dependency = Path(raw_dependency["path"])
        dependency_sha256 = _require_digest(
            raw_dependency["sha256"], f"{dependency_label} source"
        )
        dependency_key = str(dependency)
        if dependency_key in dependency_paths or dependency in {launcher, producer}:
            raise CampaignControllerError(f"{dependency_label} is duplicated")
        _authenticate_large_owned_file(
            dependency,
            dependency_label,
            expected_sha256=dependency_sha256,
            maximum=MAX_DEPENDENCY_BYTES,
        )
        dependency_paths.add(dependency_key)
        dependencies.append({"path": dependency_key, "sha256": dependency_sha256})
    raw_argv = descriptor["argv"]
    if (
        not isinstance(raw_argv, list)
        or not raw_argv
        or any(not isinstance(item, str) or not item or "\x00" in item for item in raw_argv)
        or Path(raw_argv[0]) != launcher
    ):
        raise CampaignControllerError(f"{label} argv is invalid or does not execute its launcher")
    cwd = Path(descriptor["cwd"])
    try:
        cwd_metadata = cwd.lstat()
    except OSError as error:
        raise CampaignControllerError(f"cannot inspect {label} cwd: {error}") from error
    if (
        not cwd.is_absolute()
        or cwd != cwd.resolve(strict=False)
        or cwd.is_symlink()
        or not stat.S_ISDIR(cwd_metadata.st_mode)
        or cwd_metadata.st_uid != os.getuid()
        or cwd_metadata.st_mode & 0o022
    ):
        raise CampaignControllerError(f"{label} cwd is unsafe")
    timeout = descriptor["timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 86_400:
        raise CampaignControllerError(f"{label} timeout must be in 1..86400 seconds")
    descriptor["environment"] = _normalize_environment(descriptor["environment"], label)
    descriptor["dependencies"] = sorted(dependencies, key=lambda item: item["path"])
    descriptor["argv"] = list(raw_argv)
    descriptor["cwd"] = str(cwd)
    descriptor["launcher_path"] = str(launcher)
    descriptor["producer_path"] = str(producer)
    return descriptor


def _authenticate_descriptor_sources(descriptor: Mapping[str, Any], label: str) -> None:
    """Reauthenticate every executable byte immediately around producer execution."""

    _stable_owned_file(
        Path(descriptor["launcher_path"]),
        f"{label} launcher",
        expected_sha256=descriptor["launcher_sha256"],
        maximum=MAX_CAPTURE_BYTES,
    )
    _stable_owned_file(
        Path(descriptor["producer_path"]),
        f"{label} source",
        expected_sha256=descriptor["producer_sha256"],
        maximum=MAX_CAPTURE_BYTES,
    )
    for index, dependency in enumerate(descriptor["dependencies"]):
        _authenticate_large_owned_file(
            Path(dependency["path"]),
            f"{label} dependency {index}",
            expected_sha256=dependency["sha256"],
            maximum=MAX_DEPENDENCY_BYTES,
        )


def normalize_spec(value: object) -> dict[str, Any]:
    """Authenticate the complete two-arm producer routing document."""

    if not isinstance(value, dict) or set(value) != _SPEC_KEYS:
        raise CampaignControllerError("campaign-controller spec keys are invalid")
    spec = dict(value)
    if spec["schema"] != SPEC_SCHEMA:
        raise CampaignControllerError("campaign-controller spec schema differs")
    source_sha256 = _require_digest(
        spec["controller_source_sha256"], "campaign-controller source"
    )
    own_source = Path(__file__).resolve(strict=True)
    if _file_sha256(own_source) != source_sha256:
        raise CampaignControllerError("campaign-controller source binding differs")
    arms = spec["arms"]
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise CampaignControllerError("campaign-controller arms are incomplete")
    normalized_arms: dict[str, Any] = {}
    normalized_headers: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        raw_arm = arms[arm]
        if not isinstance(raw_arm, dict) or set(raw_arm) != _ARM_KEYS:
            raise CampaignControllerError(f"{arm} campaign keys are invalid")
        header_sha256 = _require_digest(raw_arm["header_sha256"], f"{arm} header")
        header_path = Path(raw_arm["header_path"])
        header = instrumentation.normalize_header(
            _load_json_file(header_path, f"{arm} header", header_sha256)
        )
        if header["arm"] != arm:
            raise CampaignControllerError(f"{arm} header names the wrong arm")
        producers = raw_arm["producers"]
        if not isinstance(producers, dict) or set(producers) != set(PRODUCER_KINDS):
            raise CampaignControllerError(f"{arm} producer inventory is incomplete")
        normalized_headers[arm] = header
        normalized_arms[arm] = {
            "header_path": str(header_path),
            "header_sha256": header_sha256,
            "producers": {
                producer_kind: _normalize_descriptor(
                    producers[producer_kind], arm, producer_kind
                )
                for producer_kind in PRODUCER_KINDS
            },
        }
    ignored = {"arm", "run_id", "position_rows"}
    for key in instrumentation.HEADER_KEYS - ignored:
        if normalized_headers[ARMS[0]][key] != normalized_headers[ARMS[1]][key]:
            raise CampaignControllerError(f"serial/M8 campaign identity differs at {key}")
    spec["arms"] = normalized_arms
    return spec


def _fragment_environment(
    descriptor: Mapping[str, Any],
    *,
    fragment_root: Path,
    header_path: Path,
    scopes_path: Path,
    fragment: Mapping[str, Any],
) -> dict[str, str]:
    environment = {
        name: os.environ[name]
        for name in SAFE_ENVIRONMENT
        if name in os.environ and name not in descriptor["environment"]
    }
    environment.update(descriptor["environment"])
    environment.update(
        {
            instrumentation.FULL_ASSURANCE_ENABLE_ENV: "1",
            instrumentation.FULL_ASSURANCE_ROOT_ENV: str(fragment_root),
            instrumentation.FULL_ASSURANCE_HEADER_ENV: str(header_path),
            instrumentation.FULL_ASSURANCE_SCOPES_ENV: str(scopes_path),
            instrumentation.FULL_ASSURANCE_FRAGMENT_ID_ENV: str(fragment["fragment_id"]),
            instrumentation.FULL_ASSURANCE_PRODUCER_KIND_ENV: str(
                fragment["producer_kind"]
            ),
            instrumentation.FULL_ASSURANCE_PRODUCER_SHA_ENV: str(
                descriptor["producer_sha256"]
            ),
        }
    )
    return environment


def _run_fragment(
    descriptor: Mapping[str, Any],
    *,
    fragment_root: Path,
    header_path: Path,
    scopes_path: Path,
    fragment: Mapping[str, Any],
    log_path: Path,
) -> None:
    label = f"fragment {fragment['fragment_id']} producer"
    _authenticate_descriptor_sources(descriptor, label)
    environment = _fragment_environment(
        descriptor,
        fragment_root=fragment_root,
        header_path=header_path,
        scopes_path=scopes_path,
        fragment=fragment,
    )
    try:
        result = subprocess.run(
            descriptor["argv"],
            cwd=descriptor["cwd"],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=descriptor["timeout_seconds"],
        )
    except subprocess.TimeoutExpired as error:
        partial = error.stdout if isinstance(error.stdout, bytes) else b""
        _write_create_only(log_path, partial[-MAX_CAPTURE_BYTES:])
        raise CampaignControllerError(
            f"fragment {fragment['fragment_id']} producer timed out"
        ) from error
    _write_create_only(log_path, result.stdout[-MAX_CAPTURE_BYTES:])
    _authenticate_descriptor_sources(descriptor, label)
    if result.returncode != 0:
        raise CampaignControllerError(
            f"fragment {fragment['fragment_id']} producer exited {result.returncode}"
        )
    header, _events, complete = instrumentation.load_complete_fragment(fragment_root)
    if (
        header["fragment_id"] != fragment["fragment_id"]
        or header["producer_kind"] != fragment["producer_kind"]
        or header["producer_sha256"] != descriptor["producer_sha256"]
        or header["scopes"] != fragment["scopes"]
        or complete["installed_instrumentation_sites_sha256"]
        != fragment["installed_sites_sha256"]
    ):
        raise CampaignControllerError(
            f"fragment {fragment['fragment_id']} output differs from its controller plan"
        )


def _prepare_root(path: Path) -> None:
    if (
        not path.is_absolute()
        or path != path.resolve(strict=False)
        or path.exists()
        or path.is_symlink()
    ):
        raise CampaignControllerError("campaign output must be a new normalized absolute path")
    try:
        parent = path.parent.lstat()
    except OSError as error:
        raise CampaignControllerError(f"cannot inspect campaign output parent: {error}") from error
    if (
        path.parent.is_symlink()
        or not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.getuid()
        or parent.st_mode & 0o077
    ):
        raise CampaignControllerError(
            "campaign output parent must be an owned private non-symlink directory"
        )
    path.mkdir(mode=0o700, parents=False)


def execute(spec_value: object, output_root: Path) -> dict[str, Any]:
    """Run all producer fragments and publish a success marker only on parity."""

    spec = normalize_spec(spec_value)
    _prepare_root(output_root)
    fragment_roots: dict[str, list[Path]] = {arm: [] for arm in ARMS}
    plan_document: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "controller_source_sha256": spec["controller_source_sha256"],
        "spec_sha256": _digest(spec),
        "arms": {},
    }
    try:
        for arm in ARMS:
            arm_root = output_root / arm
            control_root = arm_root / "control"
            fragments_root = arm_root / "fragments"
            logs_root = arm_root / "logs"
            for path in (arm_root, control_root, fragments_root, logs_root):
                path.mkdir(mode=0o700)
            header = instrumentation.normalize_header(
                _load_json_file(
                    Path(spec["arms"][arm]["header_path"]),
                    f"{arm} header",
                    spec["arms"][arm]["header_sha256"],
                )
            )
            header_path = control_root / "header.json"
            _write_create_only(header_path, _canonical(header))
            plan = instrumentation.build_fragment_plan(header)
            plan_document["arms"][arm] = plan
            for fragment in plan["fragments"]:
                fragment_id = fragment["fragment_id"]
                producer_kind = fragment["producer_kind"]
                descriptor = spec["arms"][arm]["producers"][producer_kind]
                scopes = instrumentation.runtime_scopes_document(
                    header, fragment["scopes"]
                )
                scopes_path = control_root / f"{fragment_id}.scopes.json"
                _write_create_only(scopes_path, _canonical(scopes))
                fragment_root = fragments_root / fragment_id
                log_path = logs_root / f"{fragment_id}.log"
                _run_fragment(
                    descriptor,
                    fragment_root=fragment_root,
                    header_path=header_path,
                    scopes_path=scopes_path,
                    fragment=fragment,
                    log_path=log_path,
                )
                fragment_roots[arm].append(fragment_root)
            _fsync_directory(control_root)
            _fsync_directory(fragments_root)
            _fsync_directory(logs_root)
        plan_document["plan_sha256"] = _digest(plan_document)
        _write_create_only(output_root / "plan.json", _canonical(plan_document))
        trace_roots: dict[str, Path] = {}
        assembly: dict[str, Any] = {}
        for arm in ARMS:
            trace_root = output_root / f"{arm}-trace"
            assembly[arm] = instrumentation.assemble_complete_campaign(
                trace_root, fragment_roots[arm]
            )
            trace_roots[arm] = trace_root
        serial_header, serial_events, _serial_complete = instrumentation.load_complete_trace(
            trace_roots["serial-m1"]
        )
        speculative_header, speculative_events, _speculative_complete = (
            instrumentation.load_complete_trace(trace_roots["speculative-m8"])
        )
        comparison = instrumentation.compare_complete_traces(
            serial_header,
            serial_events,
            speculative_header,
            speculative_events,
        )
        _write_create_only(output_root / "comparison.json", _canonical(comparison))
        if comparison["passed"] is not True:
            raise CampaignControllerError(
                "serial/M8 trace differs; a physical counterexample capsule is required"
            )
        qualification = instrumentation.qualify_trace_pair(
            trace_roots["serial-m1"], trace_roots["speculative-m8"]
        )
        _write_create_only(output_root / "qualification.json", _canonical(qualification))
        complete = {
            "schema": COMPLETE_SCHEMA,
            "classification": "bounded_qualification",
            "universal_equivalence_proven": False,
            "spec_sha256": _digest(spec),
            "plan_sha256": plan_document["plan_sha256"],
            "serial_assembly_sha256": _digest(assembly["serial-m1"]),
            "speculative_assembly_sha256": _digest(assembly["speculative-m8"]),
            "qualification_sha256": qualification["qualification_sha256"],
            "passed": True,
        }
        complete["complete_sha256"] = _digest(complete)
        _write_create_only(output_root / "complete.json", _canonical(complete))
        _fsync_directory(output_root)
        return complete
    except BaseException as error:
        failure = {
            "schema": FAILURE_SCHEMA,
            "spec_sha256": _digest(spec),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        failure["failure_sha256"] = _digest(failure)
        try:
            _write_create_only(output_root / "failure.json", _canonical(failure))
            _fsync_directory(output_root)
        except (OSError, CampaignControllerError):
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-assurance-campaign-controller",
        description="Run every authenticated producer in one complete serial/M8 campaign.",
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--expected-spec-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        expected = _require_digest(args.expected_spec_sha256, "spec")
        spec = _load_json_file(args.spec, "campaign-controller spec", expected)
        normalized = normalize_spec(spec)
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "schema": SPEC_SCHEMA,
                        "spec_sha256": _digest(normalized),
                        "preflight": "passed",
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        result = execute(normalized, args.output)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        instrumentation.InstrumentationError,
        CampaignControllerError,
    ) as error:
        print(f"qwen-assurance-campaign-controller: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
