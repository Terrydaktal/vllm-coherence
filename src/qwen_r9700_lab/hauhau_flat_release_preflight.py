"""Fail-closed, GPU-free preflight for one flattened Hauhau release candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

from qwen_r9700_lab.hauhau_flat_release import (
    EXPECTED_COMPONENTS,
    FORBIDDEN_ENVIRONMENT,
    GENERATED_ENVIRONMENT,
    REQUIRED_ENVIRONMENT,
    SCHEMA,
    SITE_ENV_BY_COMPONENT,
    FlatReleaseError,
    _split_command,
    _stable_file,
    _tree_inventory,
)

PREFLIGHT_SCHEMA = "urn:qwen-r9700:hauhau-flat-release-preflight:v1"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise FlatReleaseError(f"{label} SHA256 is malformed")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise FlatReleaseError(f"{label} SHA256 is malformed") from error
    return value


def _inside(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise FlatReleaseError(f"{label} path is not relative")
    unresolved = root / relative
    resolved = unresolved.resolve(strict=True)
    if resolved != unresolved.absolute() or not resolved.is_relative_to(root):
        raise FlatReleaseError(f"{label} path escapes the release")
    return resolved


def _mode(path: Path) -> str:
    return f"{stat.S_IMODE(path.lstat().st_mode):04o}"


def _verify_bound_file(
    path: Path,
    record: dict[str, Any],
    label: str,
    *,
    private: bool = False,
) -> None:
    payload = _stable_file(path, label, private=private)
    if len(payload) != record.get("bytes"):
        raise FlatReleaseError(f"{label} byte count differs")
    if _sha(payload) != _require_sha(record.get("sha256"), label):
        raise FlatReleaseError(f"{label} SHA256 differs")
    expected_mode = record.get("mode")
    if expected_mode is not None and _mode(path) != expected_mode:
        raise FlatReleaseError(f"{label} mode differs")


def _verify_components(root: Path, manifest: dict[str, Any], environment: dict[str, str]) -> None:
    components = manifest.get("components")
    if not isinstance(components, list) or len(components) != len(EXPECTED_COMPONENTS):
        raise FlatReleaseError("release component count differs")
    if manifest.get("component_order") != list(EXPECTED_COMPONENTS):
        raise FlatReleaseError("release component order differs")
    component_root = root / "components"
    expected_directories: set[str] = set()
    for index, (expected_name, record) in enumerate(
        zip(EXPECTED_COMPONENTS, components, strict=True), 1
    ):
        if not isinstance(record, dict) or record.get("name") != expected_name:
            raise FlatReleaseError(f"release component {index} identity differs")
        directory_name = f"{index:02d}-{expected_name}"
        expected_directories.add(directory_name)
        directory = component_root / directory_name
        before = directory.lstat()
        if (
            not stat.S_ISDIR(before.st_mode)
            or directory.is_symlink()
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) != 0o700
        ):
            raise FlatReleaseError(f"release component directory {expected_name} is unsafe")
        site_record = record.get("bundled_site")
        if not isinstance(site_record, dict):
            raise FlatReleaseError(f"release component {expected_name} lacks its site binding")
        site = _inside(root, site_record.get("path"), f"release component {expected_name} site")
        _verify_bound_file(
            site,
            site_record,
            f"release component {expected_name} site",
            private=True,
        )
        if environment.get(SITE_ENV_BY_COMPONENT[expected_name]) != site_record.get("sha256"):
            raise FlatReleaseError(f"release component {expected_name} environment binding differs")
        expected_files = {site.name}
        files = record.get("files")
        if not isinstance(files, list):
            raise FlatReleaseError(f"release component {expected_name} file table differs")
        for file_record in files:
            if not isinstance(file_record, dict):
                raise FlatReleaseError(f"release component {expected_name} file record differs")
            path = _inside(root, file_record.get("path"), f"release component {expected_name} file")
            if path.parent != directory:
                raise FlatReleaseError(f"release component {expected_name} file is misplaced")
            expected_files.add(path.name)
            _verify_bound_file(path, file_record, f"release component {expected_name} file")
            if path.suffix == ".py":
                compile(path.read_bytes(), str(path), "exec")
        actual_files = {path.name for path in directory.iterdir()}
        if actual_files != expected_files:
            raise FlatReleaseError(f"release component {expected_name} directory contents differ")
    actual_directories = {path.name for path in component_root.iterdir()}
    if actual_directories != expected_directories:
        raise FlatReleaseError("release component directory set differs")


def _verify_tree(record: object, label: str) -> None:
    if not isinstance(record, dict) or not isinstance(record.get("root"), str):
        raise FlatReleaseError(f"{label} binding differs")
    observed = _tree_inventory(Path(record["root"]), label)
    if observed != record:
        raise FlatReleaseError(f"{label} tree identity differs")


def _verify_external_files(records: object) -> None:
    if not isinstance(records, list):
        raise FlatReleaseError("external-file table differs")
    paths: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise FlatReleaseError(f"external file {index} record differs")
        path = record["path"]
        if path in paths:
            raise FlatReleaseError("external-file table contains duplicates")
        paths.add(path)
        _verify_bound_file(Path(path), record, f"external file {index}")


def static_preflight(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    command_path: Path,
    expected_command_sha256: str,
) -> dict[str, Any]:
    manifest_payload = _stable_file(manifest_path, "release manifest", private=True)
    if _sha(manifest_payload) != _require_sha(expected_manifest_sha256, "release manifest"):
        raise FlatReleaseError("release manifest SHA256 differs")
    try:
        manifest = json.loads(manifest_payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FlatReleaseError("release manifest is invalid JSON") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise FlatReleaseError("release manifest schema differs")
    if manifest.get("promotable") is not False or manifest.get("classification") != (
        "non_promotable_pending_full_qualification"
    ):
        raise FlatReleaseError("candidate release classification differs")
    root = manifest_path.parent.resolve(strict=True)
    if command_path.resolve(strict=True) != root / "command.sh":
        raise FlatReleaseError("release command is not colocated with its manifest")
    command_payload = _stable_file(command_path, "release command", private=True)
    expected_command = _require_sha(expected_command_sha256, "release command")
    if (
        _sha(command_payload) != expected_command
        or manifest.get("command_sha256") != expected_command
    ):
        raise FlatReleaseError("release command SHA256 differs")
    environment_pairs, argv = _split_command(command_payload)
    environment = dict(environment_pairs)
    environment_sha = _sha(json.dumps(environment_pairs, separators=(",", ":")).encode())
    if environment_sha != manifest.get("environment_sha256"):
        raise FlatReleaseError("release environment digest differs")
    if argv != manifest.get("argv"):
        raise FlatReleaseError("release argv differs")
    for name, value in REQUIRED_ENVIRONMENT.items():
        if environment.get(name) != value:
            raise FlatReleaseError(f"release environment lacks {name}")
    for name, value in GENERATED_ENVIRONMENT.items():
        if environment.get(name) != value:
            raise FlatReleaseError(f"release environment lacks generated identity {name}")
    forbidden = sorted(FORBIDDEN_ENVIRONMENT & environment.keys())
    if forbidden or manifest.get("broad_attention_row_exact") is not False:
        raise FlatReleaseError(f"release retains broad row-exact repair: {forbidden}")
    if manifest.get("supported_verification_widths") != list(range(9)):
        raise FlatReleaseError("release verification-width contract differs")
    _verify_components(root, manifest, environment)
    _verify_tree(manifest.get("model"), "target model")
    _verify_tree(manifest.get("draft"), "draft model")
    _verify_external_files(manifest.get("external_files"))
    foundation = manifest.get("foundation")
    if not isinstance(foundation, dict) or not isinstance(foundation.get("site"), str):
        raise FlatReleaseError("release foundation binding differs")
    foundation_payload = _stable_file(Path(foundation["site"]), "release foundation", private=True)
    if _sha(foundation_payload) != _require_sha(
        foundation.get("site_sha256"), "release foundation"
    ):
        raise FlatReleaseError("release foundation SHA256 differs")
    pythonpath = environment.get("PYTHONPATH", "").split(":")
    expected_prefix = [
        str(root / "components" / f"{index:02d}-{name}")
        for index, name in reversed(list(enumerate(EXPECTED_COMPONENTS, 1)))
    ]
    if pythonpath[: len(expected_prefix)] != expected_prefix:
        raise FlatReleaseError("release component import order differs")
    return {
        "schema": PREFLIGHT_SCHEMA,
        "manifest_sha256": expected_manifest_sha256,
        "command_sha256": expected_command_sha256,
        "component_order": list(EXPECTED_COMPONENTS),
        "target_tree_sha256": manifest["model"]["tree_sha256"],
        "draft_tree_sha256": manifest["draft"]["tree_sha256"],
        "static_preflight": True,
        "gpu_loaded": False,
    }


def _import_probe_code(expected_sites: dict[str, str]) -> str:
    # Importing the real CLI before model modules reproduces vLLM's finder setup.
    # The child receives only the candidate's isolated environment and never starts
    # an API server or constructs a model.
    expected = repr(
        {
            name: value
            for name, value in expected_sites.items()
            if name != "partial-width"
        }
    )
    return f"""
import importlib
import json
import sys

expected = {expected}
from vllm.entrypoints.cli.main import main
sys.argv = [\"vllm\", \"serve\", \"--help\"]
try:
    main()
except SystemExit as error:
    if error.code not in (0, None):
        raise

qwen_next = importlib.import_module(\"vllm.model_executor.models.qwen3_next\")
qwen_dflash = importlib.import_module(\"vllm.model_executor.models.qwen3_dflash\")
qwen_parser = importlib.import_module(\"vllm.parser.qwen3\")
offload = importlib.import_module(
    \"vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler\"
)
kv_manager = importlib.import_module(\"vllm.v1.core.kv_cache_manager\")
qwen35 = importlib.import_module(\"vllm.model_executor.models.qwen3_5\")
runner = importlib.import_module(\"vllm.v1.worker.gpu.model_runner\")
gdn = importlib.import_module(
    \"vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn\"
)
quest = importlib.import_module(\"quest_vllm_attention\")
rejection = importlib.import_module(
    \"vllm.v1.worker.gpu.spec_decode.rejection_sampler\"
)

checks = {{
    \"exact-k\": getattr(
        qwen_next.Qwen3NextAttention._project_qkv_gate,
        \"_qwen_full_attention_m8_exact_k_site_sha256\", None
    ),
    \"context-kv\": getattr(
        qwen_dflash.DFlashQwen3Model.precompute_and_store_context_kv,
        \"_qwen_dflash_context_kv_serial_site_sha256\", None
    ),
    \"parser\": getattr(
        qwen_parser.qwen3_config, \"_qwen3_parameter_tool_repair_site_sha256\", None
    ),
    \"fixed-slot\": getattr(
        offload.OffloadingConnectorScheduler._lookup,
        \"_qwen_fixed_slot_generic_prefix_block_site_sha256\", None
    ),
    \"fixed-slot-evict\": getattr(
        kv_manager.KVCacheManager.allocate_slots,
        \"_qwen_fixed_slot_resident_evict_site_sha256\", None
    ),
    \"lm-head\": getattr(
        qwen35.Qwen3_5ForCausalLMBase.compute_logits,
        \"_qwen_lm_head_partial_m8_row_exact_site_sha256\", None
    ),
}}
if checks[\"fixed-slot-evict\"] != expected[\"fixed-slot\"]:
    raise RuntimeError(\"fixed-slot eviction marker differs\")
checks.pop(\"fixed-slot-evict\")
if checks != expected:
    raise RuntimeError(f\"runtime patch markers differ: {{checks!r}}\")
if getattr(
    qwen_next.Qwen3NextAttention._project_qkv_gate,
    \"_qwen_full_attention_m8_row_exact_site_sha256\", None
) is not None:
    raise RuntimeError(\"broad full-attention row-exact patch is active\")
paths = {{
    \"runner\": runner.__file__,
    \"gdn_origin\": gdn.__file__,
    \"gdn_source\": getattr(
        gdn.QwenGatedDeltaNetAttention,
        \"_forward_core_decode_spec_fixed_slot_conv_m1\",
    ).__code__.co_filename,
    \"gdn_abi\": getattr(gdn, \"_QWEN_HAUHAU_PARTIAL_WIDTH_GDN_ABI\", None),
    \"quest\": quest.__file__,
    \"rejection_origin\": rejection.__file__,
    \"rejection_abi\": getattr(rejection, \"_QWEN_PARTIAL_WIDTH_REJECTION_ABI\", None),
}}
print(\"QWEN_FLAT_PREFLIGHT_RESULT=\" + json.dumps(
    {{\"markers\": checks, \"paths\": paths, \"broad_attention_patch\": False}},
    sort_keys=True,
))
"""


def import_preflight(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    command_path: Path,
    expected_command_sha256: str,
) -> dict[str, Any]:
    static = static_preflight(
        manifest_path=manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        command_path=command_path,
        expected_command_sha256=expected_command_sha256,
    )
    manifest = json.loads(manifest_path.read_bytes())
    command_payload = command_path.read_bytes()
    environment_pairs, argv = _split_command(command_payload)
    environment = dict(environment_pairs)
    expected_sites = {
        record["name"]: record["bundled_site"]["sha256"]
        for record in manifest["components"]
    }
    python = Path(argv[1]).with_name("python")
    if not python.is_file():
        raise FlatReleaseError("release Python runtime is absent")
    probe_argv = [argv[0], str(python), "-c", _import_probe_code(expected_sites)]
    try:
        completed = subprocess.run(
            probe_argv,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise FlatReleaseError("release import preflight could not run") from error
    prefix = "QWEN_FLAT_PREFLIGHT_RESULT="
    result_lines = [line for line in completed.stdout.splitlines() if line.startswith(prefix)]
    if completed.returncode != 0 or len(result_lines) != 1:
        tail = "\n".join(completed.stdout.splitlines()[-30:])
        raise FlatReleaseError(
            f"release import preflight failed with status {completed.returncode}:\n{tail}"
        )
    try:
        runtime = json.loads(result_lines[0][len(prefix) :])
    except json.JSONDecodeError as error:
        raise FlatReleaseError("release import preflight result is malformed") from error
    root = manifest_path.parent.resolve(strict=True)
    expected_paths = {
        "runner": str(root / "components" / "05-partial-width" / "model_runner.py"),
        "gdn_source": str(
            root / "components" / "05-partial-width" / "qwen_gdn_linear_attn.py"
        ),
        "quest": str(root / "components" / "05-partial-width" / "quest_vllm_attention.py"),
    }
    observed_paths = runtime.get("paths")
    if not isinstance(observed_paths, dict) or any(
        observed_paths.get(name) != value for name, value in expected_paths.items()
    ):
        raise FlatReleaseError(
            f"release partial-width runtime paths differ: {observed_paths!r}"
        )
    if observed_paths.get("gdn_abi") != (
        "b591518823bd101107ab5c6a638cc391ed8afdc8e72ad54133a434c74399c34c"
    ):
        raise FlatReleaseError("release partial-width GDN ABI differs")
    if observed_paths.get("rejection_abi") is not True:
        raise FlatReleaseError("release partial-width rejection ABI differs")
    if runtime.get("broad_attention_patch") is not False:
        raise FlatReleaseError("release import preflight enabled broad attention repair")
    return {**static, "import_preflight": True, "runtime": runtime}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--expected-manifest-sha256", required=True)
    result.add_argument("--command", type=Path, required=True)
    result.add_argument("--expected-command-sha256", required=True)
    result.add_argument("--static-only", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    function = static_preflight if args.static_only else import_preflight
    try:
        result = function(
            manifest_path=args.manifest,
            expected_manifest_sha256=args.expected_manifest_sha256,
            command_path=args.command,
            expected_command_sha256=args.expected_command_sha256,
        )
    except (FlatReleaseError, OSError, ValueError) as error:
        print(f"hauhau-flat-release-preflight: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
