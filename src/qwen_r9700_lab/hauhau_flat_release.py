"""Flatten the selected Hauhau repair chain into one authenticated bundle.

The historical live62 foundation remains an explicitly bound dependency.  All
post-foundation repairs are copied into one create-only directory, their chain
paths and PYTHONPATH contracts are retargeted to that directory, and one
manifest binds the resulting command, sites, runtime files, weights, draft,
tool parser, and compiler/runtime identities.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import re
import shlex
import stat
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCHEMA = "urn:qwen-r9700:hauhau-flat-release-candidate:v1"
EXPECTED_COMPONENTS = (
    "exact-k",
    "context-kv",
    "parser",
    "fixed-slot",
    "partial-width",
    "lm-head",
)
SITE_ENV_BY_COMPONENT = {
    "exact-k": "QWEN_FULL_ATTENTION_M8_EXACT_K_SITE_SHA256",
    "context-kv": "QWEN_DFLASH_CONTEXT_KV_SERIAL_EXACT_SITE_SHA256",
    "parser": "QWEN3_PARAMETER_TOOL_REPAIR_SITE_SHA256",
    "fixed-slot": "QWEN_FIXED_SLOT_GENERIC_PREFIX_BLOCK_SITE_SHA256",
    "partial-width": "QWEN_HAUHAU_PARTIAL_WIDTH_RUNTIME_SITE_SHA256",
    "lm-head": "QWEN_LM_HEAD_M8_ROW_EXACT_SITE_SHA256",
}
REQUIRED_ENVIRONMENT = {
    "QWEN_FULL_ATTENTION_M8_EXACT_K": "1",
    "QWEN_FULL_ATTENTION_M8_EXACT_K_REQUIRED": "1",
    "QWEN_DFLASH_CONTEXT_KV_SERIAL_EXACT": "1",
    "QWEN_DFLASH_CONTEXT_KV_SERIAL_EXACT_REQUIRED": "1",
    "QWEN3_PARAMETER_TOOL_REPAIR": "1",
    "QWEN3_PARAMETER_TOOL_REPAIR_REQUIRED": "1",
    "QWEN_FIXED_SLOT_GENERIC_PREFIX_BLOCK": "1",
    "QWEN_FIXED_SLOT_GENERIC_PREFIX_BLOCK_REQUIRED": "1",
    "QWEN_HAUHAU_PARTIAL_WIDTH_RUNTIME": "1",
    "QWEN_LM_HEAD_M8_ROW_EXACT": "1",
    "QWEN_LM_HEAD_M8_ROW_EXACT_REQUIRED": "1",
    "VLLM_ENFORCE_STRICT_TOOL_CALLING": "1",
    "QWEN_DFLASH_GREEDY_LOOP_ESCAPE": "0",
    "QWEN_FIXED_SLOT_TARGET_ONLY_ORACLE": "0",
}
GENERATED_ENVIRONMENT = {
    # A release import must never create __pycache__ entries inside the
    # authenticated artifact.  Python still compiles the source in memory.
    "PYTHONDONTWRITEBYTECODE": "1",
}
FORBIDDEN_ENVIRONMENT = {
    "QWEN_FULL_ATTENTION_M8_ROW_EXACT",
    "QWEN_FULL_ATTENTION_M8_ROW_EXACT_REQUIRED",
}


class FlatReleaseError(RuntimeError):
    """The candidate, destination, or dependency contract is invalid."""


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable_file(path: Path, label: str, *, private: bool = False) -> bytes:
    path = path.expanduser().absolute()
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise FlatReleaseError(f"{label} is not one regular non-symlink file")
    if before.st_uid != os.getuid() or stat.S_IMODE(before.st_mode) & 0o022:
        raise FlatReleaseError(f"{label} is not owned and protected from writes")
    if private and stat.S_IMODE(before.st_mode) & 0o077:
        raise FlatReleaseError(f"{label} is not private")
    payload = path.read_bytes()
    after = path.lstat()
    def identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_uid,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
        )
    if identity(before) != identity(after) or len(payload) != before.st_size:
        raise FlatReleaseError(f"{label} changed while being authenticated")
    return payload


def _write_exclusive(path: Path, payload: bytes, mode: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _literal_assignment(source: str, name: str) -> tuple[ast.AST, Any]:
    tree = ast.parse(source)
    matches: list[ast.AST] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            matches.append(node.value)
    if len(matches) != 1:
        raise FlatReleaseError(f"site has {len(matches)} assignments for {name}")
    value = matches[0]
    literal = value.args[0] if isinstance(value, ast.Call) and value.args else value
    try:
        return value, ast.literal_eval(literal)
    except (TypeError, ValueError) as error:
        raise FlatReleaseError(f"site assignment {name} is not literal") from error


def _replace_assignment(source: str, name: str, value: str) -> str:
    tree = ast.parse(source)
    matches: list[ast.Assign] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            matches.append(node)
    if len(matches) != 1:
        raise FlatReleaseError(f"site has {len(matches)} assignments for {name}")
    node = matches[0]
    if node.lineno != node.end_lineno:
        raise FlatReleaseError(f"site assignment {name} is unexpectedly multiline")
    is_path = isinstance(node.value, ast.Call) and (
        isinstance(node.value.func, ast.Name) and node.value.func.id == "Path"
    )
    replacement = f"{name} = {'Path(' + repr(value) + ')' if is_path else repr(value)}"
    lines = source.splitlines(keepends=True)
    ending = "\n" if lines[node.lineno - 1].endswith("\n") else ""
    lines[node.lineno - 1] = replacement + ending
    return "".join(lines)


def _assignment_names(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            names.add(target.id)
    return names


def _split_command(payload: bytes) -> tuple[list[tuple[str, str]], list[str]]:
    try:
        tokens = shlex.split(payload.decode())
    except (UnicodeError, ValueError) as error:
        raise FlatReleaseError("candidate command cannot be parsed") from error
    if tokens[:3] != ["exec", "/usr/bin/env", "-i"]:
        raise FlatReleaseError("candidate command is not an isolated env exec")
    environment: list[tuple[str, str]] = []
    index = 3
    while index < len(tokens) and "=" in tokens[index]:
        name, value = tokens[index].split("=", 1)
        environment.append((name, value))
        index += 1
    names = [name for name, _ in environment]
    if len(names) != len(set(names)):
        raise FlatReleaseError("candidate command has duplicate environment keys")
    argv = tokens[index:]
    if len(argv) < 4 or argv[0].endswith("/with-rocm") is False:
        raise FlatReleaseError("candidate command does not use the ROCm wrapper")
    if argv[2:4] != ["serve", next(iter(argv[3:4]), "")]:
        raise FlatReleaseError("candidate command is not a vLLM serve command")
    return environment, argv


def _site_chain(candidate_site: Path, foundation_site: Path) -> list[Path]:
    outer_to_inner: list[Path] = []
    current = candidate_site
    seen: set[Path] = set()
    while current != foundation_site:
        current = current.resolve(strict=True)
        if current in seen:
            raise FlatReleaseError("repair site chain contains a cycle")
        seen.add(current)
        source = _stable_file(current, "repair site", private=True).decode()
        outer_to_inner.append(current)
        _node, chain = _literal_assignment(source, "_CHAIN")
        current = Path(chain).resolve(strict=True)
    if len(outer_to_inner) != len(EXPECTED_COMPONENTS):
        raise FlatReleaseError(
            f"repair chain has {len(outer_to_inner)} sites, expected {len(EXPECTED_COMPONENTS)}"
        )
    return list(reversed(outer_to_inner))


def _copy_component(source_root: Path, destination: Path) -> list[dict[str, Any]]:
    destination.mkdir(mode=0o700)
    copied: list[dict[str, Any]] = []
    for source in sorted(source_root.iterdir(), key=lambda path: path.name):
        if source.name == "sitecustomize.py":
            continue
        if source.is_symlink() or not source.is_file():
            continue
        payload = _stable_file(source, f"component file {source}")
        mode = 0o700 if stat.S_IMODE(source.stat().st_mode) & 0o100 else 0o600
        target = destination / source.name
        _write_exclusive(target, payload, mode)
        copied.append(
            {
                "path": str(target.relative_to(destination.parents[1])),
                "sha256": _digest(payload),
                "bytes": len(payload),
                "mode": f"{mode:04o}",
                "source": str(source),
            }
        )
    return copied


def _tree_inventory(root: Path, label: str) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise FlatReleaseError(f"{label} root is unsafe")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise FlatReleaseError(f"{label} contains a symlink: {path}")
        if not path.is_file():
            continue
        payload = _stable_file(path, f"{label} file")
        entries.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": len(payload),
                "sha256": _digest(payload),
            }
        )
    if not entries:
        raise FlatReleaseError(f"{label} inventory is empty")
    tree_payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {"root": str(root), "files": entries, "tree_sha256": _digest(tree_payload)}


def _external_files(environment: dict[str, str], argv: Sequence[str]) -> list[dict[str, Any]]:
    candidates: set[Path] = set()
    path_pattern = re.compile(r"/home/lewis/[A-Za-z0-9_./-]+")
    for value in [*environment.values(), *argv]:
        for match in path_pattern.findall(value):
            candidate = Path(match.rstrip("',\"}"))
            if candidate.is_file() and not candidate.is_symlink():
                candidates.add(candidate)
    bindings: list[dict[str, Any]] = []
    for path in sorted(candidates):
        payload = _stable_file(path, "external runtime dependency")
        bindings.append({"path": str(path), "bytes": len(payload), "sha256": _digest(payload)})
    return bindings


def _tool_output(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise FlatReleaseError(f"cannot authenticate tool identity: {command[0]}") from error
    return result.stdout.strip()


def render(args: argparse.Namespace) -> dict[str, Any]:
    destination = args.destination.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FlatReleaseError("destination is create-only")
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_candidate_sha256):
        raise FlatReleaseError("candidate SHA256 is malformed")
    candidate = args.candidate_command.expanduser().absolute()
    candidate_payload = _stable_file(candidate, "candidate command", private=True)
    if _digest(candidate_payload) != args.expected_candidate_sha256:
        raise FlatReleaseError("candidate command SHA256 mismatch")
    foundation = args.foundation_site.expanduser().absolute().resolve(strict=True)
    foundation_payload = _stable_file(foundation, "foundation site", private=True)
    if _digest(foundation_payload) != args.expected_foundation_sha256:
        raise FlatReleaseError("foundation site SHA256 mismatch")

    environment_pairs, argv = _split_command(candidate_payload)
    environment = dict(environment_pairs)
    for name, value in REQUIRED_ENVIRONMENT.items():
        if environment.get(name) != value:
            raise FlatReleaseError(f"candidate lacks required environment identity {name}")
    present_forbidden = sorted(FORBIDDEN_ENVIRONMENT & environment.keys())
    if present_forbidden:
        raise FlatReleaseError(f"candidate retains broad row-exact repair: {present_forbidden}")
    if argv.count("--speculative-config") != 1:
        raise FlatReleaseError("candidate must contain one speculative configuration")

    sites = _site_chain(candidate.parent / "sitecustomize.py", foundation)
    pythonpath = environment.get("PYTHONPATH", "")
    _node, foundation_pythonpath = _literal_assignment(
        _stable_file(sites[0], "innermost site", private=True).decode(),
        "_CHAIN_PYTHONPATH",
    )
    if not isinstance(foundation_pythonpath, str) or not foundation_pythonpath.startswith(
        f"{foundation.parent}:"
    ):
        raise FlatReleaseError("innermost site does not bind the selected foundation")
    if not pythonpath.startswith(f"{sites[-1].parent}:"):
        raise FlatReleaseError("candidate PYTHONPATH does not start at its outer site")

    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent = destination.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or destination.parent.is_symlink()
        or parent.st_uid != os.getuid()
        or stat.S_IMODE(parent.st_mode) & 0o077
    ):
        raise FlatReleaseError("destination parent must be one owned private directory")
    staging = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FlatReleaseError("staging destination already exists")
    staging.mkdir(mode=0o700)
    try:
        component_root = staging / "components"
        component_root.mkdir(mode=0o700)
        component_dirs: list[Path] = []
        staged_component_dirs: list[Path] = []
        component_records: list[dict[str, Any]] = []
        old_site_hash_to_env = {
            environment[SITE_ENV_BY_COMPONENT[name]]: SITE_ENV_BY_COMPONENT[name]
            for name in EXPECTED_COMPONENTS
        }
        for index, (name, source_site) in enumerate(
            zip(EXPECTED_COMPONENTS, sites, strict=True), 1
        ):
            source_root = source_site.parent
            staged_root = component_root / f"{index:02d}-{name}"
            target_root = destination / "components" / f"{index:02d}-{name}"
            files = _copy_component(source_root, staged_root)
            component_dirs.append(target_root)
            staged_component_dirs.append(staged_root)
            component_records.append(
                {
                    "name": name,
                    "source_root": str(source_root),
                    "source_site_sha256": _digest(
                        _stable_file(source_site, "component source site", private=True)
                    ),
                    "files": files,
                }
            )

        foundation_parts = foundation_pythonpath.split(":")
        inner_site = foundation
        inner_site_sha = _digest(foundation_payload)
        new_site_hashes: dict[str, str] = {}
        for index, (name, source_site, target_root, staged_root) in enumerate(
            zip(
                EXPECTED_COMPONENTS,
                sites,
                component_dirs,
                staged_component_dirs,
                strict=True,
            )
        ):
            source_root = source_site.parent
            source = _stable_file(source_site, "component source site", private=True).decode()
            rewritten = source.replace(str(source_root), str(target_root))
            chain_parts = [
                str(component_dirs[position]) for position in range(index - 1, -1, -1)
            ] + foundation_parts
            outer_parts = [str(target_root), *chain_parts]
            rewritten = _replace_assignment(rewritten, "_CHAIN", str(inner_site))
            chain_sha_name = (
                "_CHAIN_SHA256"
                if "_CHAIN_SHA256" in _assignment_names(rewritten)
                else "_CHAIN_SHA"
            )
            rewritten = _replace_assignment(rewritten, chain_sha_name, inner_site_sha)
            rewritten = _replace_assignment(rewritten, "_CHAIN_PYTHONPATH", ":".join(chain_parts))
            rewritten = _replace_assignment(rewritten, "_OUTER_PYTHONPATH", ":".join(outer_parts))
            compile(rewritten, str(target_root / "sitecustomize.py"), "exec")
            site_payload = rewritten.encode()
            _write_exclusive(staged_root / "sitecustomize.py", site_payload, 0o600)
            site_sha = _digest(site_payload)
            component_records[index]["bundled_site"] = {
                "path": str((staged_root / "sitecustomize.py").relative_to(staging)),
                "sha256": site_sha,
                "bytes": len(site_payload),
                "mode": "0600",
            }
            new_site_hashes[SITE_ENV_BY_COMPONENT[name]] = site_sha
            inner_site = target_root / "sitecustomize.py"
            inner_site_sha = site_sha

        final_pythonpath = ":".join(
            [str(path) for path in reversed(component_dirs)] + foundation_parts
        )
        rewritten_environment: list[tuple[str, str]] = []
        for name, value in environment_pairs:
            if name == "PYTHONPATH":
                value = final_pythonpath
            elif name in new_site_hashes:
                value = new_site_hashes[name]
            elif value in old_site_hash_to_env:
                value = new_site_hashes[old_site_hash_to_env[value]]
            rewritten_environment.append((name, value))
        rewritten_names = {name for name, _value in rewritten_environment}
        rewritten_values = dict(rewritten_environment)
        for name, value in GENERATED_ENVIRONMENT.items():
            if name in rewritten_names:
                if rewritten_values[name] != value:
                    raise FlatReleaseError(
                        f"candidate has an incompatible generated identity {name}"
                    )
                continue
            rewritten_environment.append((name, value))
        rewritten_map = dict(rewritten_environment)
        command_payload = (
            shlex.join(
                [
                    "exec",
                    "/usr/bin/env",
                    "-i",
                    *(f"{name}={value}" for name, value in rewritten_environment),
                    *argv,
                ]
            )
            + "\n"
        ).encode()
        _write_exclusive(staging / "command.sh", command_payload, 0o700)

        model_path = Path(environment["QWEN_TARGET_MODEL_PATH"])
        speculative = json.loads(argv[argv.index("--speculative-config") + 1])
        draft_path = Path(speculative["model"])
        manifest = {
            "schema": SCHEMA,
            "classification": "non_promotable_pending_full_qualification",
            "promotable": False,
            "source_candidate": {
                "path": str(candidate),
                "sha256": _digest(candidate_payload),
            },
            "foundation": {
                "site": str(foundation),
                "site_sha256": _digest(foundation_payload),
                "pythonpath": foundation_pythonpath,
            },
            "command_sha256": _digest(command_payload),
            "environment_sha256": _digest(
                json.dumps(rewritten_environment, separators=(",", ":")).encode()
            ),
            "argv": argv,
            "components": component_records,
            "component_order": list(EXPECTED_COMPONENTS),
            "broad_attention_row_exact": False,
            "supported_verification_widths": list(range(9)),
            "model": _tree_inventory(model_path, "target model"),
            "draft": _tree_inventory(draft_path, "draft model"),
            "external_files": _external_files(rewritten_map, argv),
            "runtime": {
                "host": platform.uname()._asdict(),
                "python": sys.version,
                "rocm": _tool_output(
                    [
                        "/home/lewis/.local/share/qwen-r9700/runtimes/vllm-rocm/bin/with-rocm",
                        "/home/lewis/.local/share/qwen-r9700/runtimes/vllm-rocm/.venv/bin/python",
                        "-c",
                        (
                            "import json, torch; "
                            "print(json.dumps({'hip': torch.version.hip, "
                            "'torch': torch.__version__}, sort_keys=True))"
                        ),
                    ]
                ),
                "compiler": _tool_output(["/usr/bin/clang", "--version"]),
            },
            "qualification": {
                "status": "pending",
                "required_before_promotion": True,
                "evidence": [],
            },
        }
        manifest_payload = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
        _write_exclusive(staging / "manifest.json", manifest_payload, 0o600)
        staging_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        if destination.exists() or destination.is_symlink():
            raise FlatReleaseError("destination appeared during rendering")
        staging.rename(destination)
        parent_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return {
            **manifest,
            "manifest_sha256": _digest(manifest_payload),
            "destination": str(destination),
        }
    except BaseException:
        # Preserve a failed create-only staging tree for diagnosis.  A subsequent
        # run uses a new PID-specific path and never mutates the partial evidence.
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--candidate-command", type=Path, required=True)
    result.add_argument("--expected-candidate-sha256", required=True)
    result.add_argument("--foundation-site", type=Path, required=True)
    result.add_argument("--expected-foundation-sha256", required=True)
    result.add_argument("--destination", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = render(args)
    except (FlatReleaseError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"qwen-hauhau-flat-release: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
