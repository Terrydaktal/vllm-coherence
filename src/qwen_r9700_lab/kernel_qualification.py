"""Create and execute hash-bound qualification plans for kernel candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qwen_r9700_lab.config import ConfigurationError, find_project_root
from qwen_r9700_lab.kernel_250k_projection import (
    DEFAULT_VERIFICATION_ROWS,
    NATIVE_B_EXCLUDED_BF16_GDN_BA_COUNT,
    NATIVE_B_SCHEDULE_CONTRACT,
    NATIVE_B_W4_PROJECTION_COUNT,
    NATIVE_B_W4_SCHEDULE,
)

PLAN_TYPE = "qwen-r9700-kernel-qualification-plan"
RUN_TYPE = "qwen-r9700-kernel-qualification-stage-run"
SUMMARY_TYPE = "qwen-r9700-kernel-qualification-summary"
STAGE_ORDER = (
    "static",
    "component",
    "round_equivalence",
    "transaction",
    "lifecycle",
    "artifact_equivalence",
    "quality",
    "throughput",
)
COMPONENTS = (
    "w4a16",
    "quest",
    "gdn",
    "ffn_swiglu",
    "norm_residual",
    "lm_head",
    "kv_cache",
    "dflash",
    "runtime",
    "custom",
)
CUSTOM_GPU_COMMAND_COMPONENTS = frozenset(COMPONENTS) - {"w4a16", "quest", "gdn"}
COUNTEREXAMPLE_CORPUS_REL = Path("experiments/coding-turbo-quest96/counterexamples")
QUALIFIED_LIFECYCLES = (
    "snapshot_restore",
    "full_cache_hit",
    "partial_cache_hit",
    "cancel_retry",
    "restart",
    "offload_restore",
    "page_reuse",
    "chat_switch",
    "two_sessions",
    "two_branches",
)
BUNDLE_COMPONENT_BY_QUALIFIER = {
    "w4a16": "native_b",
    "quest": "quest",
    "gdn": "gdn",
}
DEFAULT_ROCM_PYTHON = "/home/lewis/.local/share/qwen-r9700/runtimes/vllm-rocm/.venv/bin/python"
DEFAULT_WITH_ROCM = "/home/lewis/.local/share/qwen-r9700/runtimes/vllm-rocm/bin/with-rocm"
MAX_CAPTURE_BYTES = 4 * 1024 * 1024
FAILURE_TAIL_LINES = 80
QUEST_COLD_SEED_EXACT_COMPARISONS = (
    "cold_seed_centroids",
    "cold_seed_selected",
    "cold_seed_selected_count",
    "cold_seed_split_counts",
    "cold_seed_union_masks",
    "cold_seed_union_pages",
    "historical_scores",
    "seeded_cached_historical_scores",
    "seeded_cached_selected",
    "seeded_cached_selected_count",
    "seeded_cached_split_counts",
    "seeded_cached_union_masks",
    "seeded_cached_union_pages",
    "selected",
    "selected_count",
    "split_counts",
    "union_masks",
    "union_pages",
)
PERFORMANCE_EVIDENCE_POLICY = {
    "promotion_basis": "measured_release_c1_at_both_exact_contexts",
    "required_context_tokens": [60_298, 249_957],
    "authoritative_target_attention": "quest96",
    "candidate_scope": "one_semantic_or_performance_unit_per_plan",
    "candidate_gain_definition": "source_bound_whole_system_ablation",
    "candidate_gain_default_status": "unverified_hypothesis",
    "projected_component_savings_are_promotable": False,
    "expected_tps_gains_are_promotable": False,
    "cross_candidate_gains_may_be_summed": False,
    "unqualified_attention_algorithm_substitution_is_promotable": False,
    "throughput_must_bind_semantic_source_and_release_artifact": True,
    "performance_run_may_contain_hot_path_instrumentation": False,
    "claimed_tps_gain_is_evidence": False,
    "architectural_speedup_multipliers_are_transferable": False,
    "gain_requires_same_fixture_release_ablation": True,
}
SEMANTIC_EFFECTS = (
    "exact_execution",
    "non_authoritative_proposal",
    "changes_target_definition",
)
SEMANTIC_UNITS = (
    "snapshot_restore",
    "m8_construction_positions_rope",
    "w4a16_projections",
    "quest_scoring",
    "quest_ordered_top96",
    "quest_union_row_visibility",
    "quest_historical_attention",
    "quest_causal_tail_attention",
    "quest_softmax_reduction",
    "gdn_convolution",
    "gdn_recurrence",
    "gated_rmsnorm",
    "residual_stream",
    "swiglu_ffn",
    "final_rmsnorm",
    "lm_head",
    "target_verification",
    "provisional_isolation",
    "atomic_commit",
    "draft_proposal",
    "runtime_orchestration",
)
SEMANTIC_UNIT_COMPONENTS = {
    "snapshot_restore": {"runtime", "kv_cache", "gdn"},
    "m8_construction_positions_rope": {"runtime", "dflash"},
    "w4a16_projections": {"w4a16"},
    "quest_scoring": {"quest"},
    "quest_ordered_top96": {"quest"},
    "quest_union_row_visibility": {"quest"},
    "quest_historical_attention": {"quest"},
    "quest_causal_tail_attention": {"quest"},
    "quest_softmax_reduction": {"quest"},
    "gdn_convolution": {"gdn"},
    "gdn_recurrence": {"gdn"},
    "gated_rmsnorm": {"gdn", "norm_residual"},
    "residual_stream": {"norm_residual"},
    "swiglu_ffn": {"ffn_swiglu"},
    "final_rmsnorm": {"norm_residual"},
    "lm_head": {"lm_head"},
    "target_verification": {"dflash", "runtime"},
    "provisional_isolation": {"gdn", "kv_cache", "runtime"},
    "atomic_commit": {"gdn", "kv_cache", "runtime"},
    "draft_proposal": {"dflash"},
    "runtime_orchestration": {"runtime", "custom"},
}
DEFAULT_SEMANTIC_UNIT_BY_COMPONENT = {
    "w4a16": "w4a16_projections",
    "quest": "quest_historical_attention",
    "gdn": "gdn_recurrence",
    "ffn_swiglu": "swiglu_ffn",
    "norm_residual": "residual_stream",
    "lm_head": "lm_head",
    "kv_cache": "provisional_isolation",
    "dflash": "draft_proposal",
    "runtime": "runtime_orchestration",
    "custom": "runtime_orchestration",
}


class QualificationError(RuntimeError):
    """A qualification plan, execution, or evidence gate was invalid."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise QualificationError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _write_new_json(path: Path, document: Mapping[str, Any], *, read_only: bool = True) -> None:
    """Atomically create one JSON document without replacing an existing path."""

    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".kernel-qualification-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if read_only:
            temporary.chmod(0o444)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise QualificationError(f"refusing to replace existing evidence: {path}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parse_labeled_file(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise QualificationError(f"expected LABEL=PATH, got {value!r}")
    if not label.replace("-", "_").isalnum() or any(character.isspace() for character in label):
        raise QualificationError(f"invalid candidate file label: {label!r}")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise QualificationError(f"candidate file is not a regular file: {path}")
    return label, path


def _parse_context_file(value: str) -> tuple[int, Path]:
    context_raw, separator, raw_path = value.partition("=")
    try:
        context = int(context_raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("context evidence must be CONTEXT=PATH") from error
    if not separator or context not in {60_298, 249_957} or not raw_path:
        raise argparse.ArgumentTypeError(
            "context evidence must use context 60298 or 249957 as CONTEXT=PATH"
        )
    return context, Path(raw_path)


def _context_evidence(values: Sequence[tuple[int, Path]], label: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for context, path in values:
        if context in result:
            raise QualificationError(f"{label} repeats context {context}")
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise QualificationError(f"{label} is not a regular file: {resolved}")
        result[context] = resolved
    if set(result) != {60_298, 249_957}:
        raise QualificationError(f"{label} must cover contexts 60298 and 249957")
    return result


def _parse_context_lifecycle_file(value: str) -> tuple[tuple[int, str], Path]:
    key, separator, raw_path = value.partition("=")
    context_raw, colon, lifecycle = key.partition(":")
    try:
        context = int(context_raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "lifecycle evidence must be CONTEXT:LIFECYCLE=PATH"
        ) from error
    if (
        not separator
        or not colon
        or context not in {60_298, 249_957}
        or lifecycle not in QUALIFIED_LIFECYCLES
        or not raw_path
    ):
        raise argparse.ArgumentTypeError(
            "lifecycle evidence must use a qualified CONTEXT:LIFECYCLE=PATH"
        )
    return (context, lifecycle), Path(raw_path)


def _lifecycle_evidence(
    values: Sequence[tuple[tuple[int, str], Path]],
) -> dict[tuple[int, str], Path]:
    result: dict[tuple[int, str], Path] = {}
    for key, path in values:
        if key in result:
            raise QualificationError(f"lifecycle comparison repeats context/lifecycle {key}")
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise QualificationError(f"lifecycle comparison is not a file: {resolved}")
        result[key] = resolved
    required = {
        (context, lifecycle) for context in (60_298, 249_957) for lifecycle in QUALIFIED_LIFECYCLES
    }
    if set(result) != required:
        missing = sorted(required - set(result))
        extra = sorted(set(result) - required)
        raise QualificationError(
            f"lifecycle comparisons are incomplete: missing={missing} extra={extra}"
        )
    return result


def _candidate_records(values: Sequence[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    labels: set[str] = set()
    paths: set[Path] = set()
    for value in values:
        label, path = _parse_labeled_file(value)
        if label in labels:
            raise QualificationError(f"duplicate candidate file label: {label}")
        if path in paths:
            raise QualificationError(f"duplicate candidate file path: {path}")
        labels.add(label)
        paths.add(path)
        records.append(
            {
                "label": label,
                "path": str(path),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    if not records:
        raise QualificationError("at least one --candidate-file is required")
    return records


def _file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise QualificationError(f"qualification input is not a regular file: {path}")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _counterexample_corpus_at(corpus_root: Path) -> dict[str, Any]:
    corpus_root = corpus_root.resolve()
    files = sorted(path for path in corpus_root.rglob("*.json") if path.is_file())
    if not files:
        raise QualificationError(f"counterexample corpus is empty: {corpus_root}")
    records = [
        {
            "path": str(path.relative_to(corpus_root)),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    return {
        "root": str(corpus_root),
        "files": records,
        "corpus_sha256": hashlib.sha256(_canonical_bytes(records)).hexdigest(),
    }


def _counterexample_corpus_record(root: Path) -> dict[str, Any]:
    return _counterexample_corpus_at(root / COUNTEREXAMPLE_CORPUS_REL)


def _verify_counterexample_corpus(record: object) -> None:
    if not isinstance(record, dict):
        raise QualificationError("counterexample corpus record is invalid")
    root = Path(str(record.get("root", "")))
    actual = _counterexample_corpus_at(root)
    if actual != record:
        raise QualificationError(
            "counterexample corpus changed after the plan was created; create a new plan"
        )


def _qualification_inputs(
    component: str,
    root: Path,
    stages: Sequence[Mapping[str, Any]],
    extra_paths: Sequence[Path] = (),
) -> list[dict[str, Any]]:
    remote = root / "experiments" / "bole-tree-verify" / "remote"
    quest = root / "experiments" / "bole-tree-verify" / "hip" / "sub1ms-quest"
    component_paths = {
        "w4a16": (
            remote / "build-rdna-native-w4a16",
            remote / "rdna_w4a16_native_ext.cpp",
            remote / "rdna_w4a16_native_ext.cu",
            remote / "test_native_b_fused_rows.py",
            remote / "benchmark_native_b_runtime_schedule.py",
        ),
        "quest": (
            quest / "build-quest-fp8-selector",
            quest / "quest_fp8_selector_ext.cpp",
            quest / "quest_fp8_selector_ext.cu",
            quest / "benchmark_fp8_exact_cached_selector.py",
            quest / "benchmark_fp8_logical_selector.py",
            quest / "benchmark_fp8_grouped_attention.py",
        ),
        "gdn": (
            remote / "build-rdna-gdn-decode",
            remote / "rdna_gdn_decode_ext.cpp",
            remote / "rdna_gdn_decode_ext.cu",
            remote / "test_recoverssm_gpu_parity.py",
            remote / "benchmark_recoverssm_runtime.py",
        ),
    }.get(component, ())
    paths = {path.resolve() for path in (*component_paths, *extra_paths)}
    if component in {"w4a16", "quest", "gdn"}:
        paths.update(
            {
                (root / "src" / "qwen_r9700_lab" / "kernel_250k_projection.py").resolve(),
                (root / "benchmarks" / "kernel-250k" / "r9700-dflash8-baseline-v1.json").resolve(),
            }
        )
    for stage in stages:
        for command in stage.get("commands", []):
            for item in command.get("argv", []):
                path = Path(item).expanduser()
                try:
                    if path.is_file():
                        paths.add(path.resolve())
                except OSError:
                    # Inline Python, JSON, and other opaque argv values are not
                    # filesystem qualification inputs.
                    continue
    return [_file_record(path) for path in sorted(paths)]


def _verified_bundle(
    manifest_path: Path, root: Path
) -> tuple[dict[str, Any], dict[str, Path], list[Path]]:
    """Resolve and fully verify one immutable kernel bundle for qualification."""

    from qwen_r9700_lab.kernel_workflow import KernelWorkflowError, verify_release

    requested = manifest_path.expanduser().absolute()
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise QualificationError(
            f"cannot resolve kernel bundle manifest {requested}: {error}"
        ) from error
    if resolved.name != "manifest.json" or not resolved.is_file():
        raise QualificationError(f"kernel bundle path must resolve to manifest.json: {requested}")
    try:
        document = verify_release(resolved.parent, project_root=root)
    except KernelWorkflowError as error:
        raise QualificationError(f"kernel bundle verification failed: {error}") from error

    artifacts: dict[str, Path] = {}
    bound_paths = [resolved]
    for name, component in sorted(document["components"].items()):
        artifact = (resolved.parent / component["artifact"]["path"]).resolve(strict=True)
        artifacts[name] = artifact
        bound_paths.append(artifact)
        bound_paths.extend(
            (root / item["path"]).resolve(strict=True) for item in component["inputs"]
        )
    record = {
        "requested_manifest": str(requested),
        "resolved_manifest": str(resolved),
        "bundle_id": document["bundle_id"],
        "version": document["version"],
        "artifacts": {name: str(path) for name, path in sorted(artifacts.items())},
    }
    return record, artifacts, bound_paths


def _command(
    command_id: str,
    description: str,
    argv: Sequence[str],
    root: Path,
    *,
    requires_gpu: bool,
    timeout_seconds: int,
    validator: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
    stdout_artifact: Path | None = None,
) -> dict[str, Any]:
    if not argv or any(not isinstance(item, str) or not item for item in argv):
        raise QualificationError(f"command {command_id} has an invalid argv")
    result: dict[str, Any] = {
        "id": command_id,
        "description": description,
        "argv": list(argv),
        "cwd": str(root),
        "requires_gpu": requires_gpu,
        "timeout_seconds": timeout_seconds,
        "validator": dict(validator or {"kind": "exit-zero"}),
    }
    if environment:
        result["environment"] = dict(sorted(environment.items()))
    if stdout_artifact is not None:
        result["stdout_artifact"] = str(stdout_artifact.expanduser().resolve())
    return result


def _runtime_preflight_command(
    root: Path,
    rocm_python: str,
    with_rocm: str,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    code = (
        "import json,sys,torch;"
        "available=bool(torch.cuda.is_available());"
        "props=torch.cuda.get_device_properties(0) if available else None;"
        "arch=getattr(props,'gcnArchName',None) if props is not None else None;"
        "print(json.dumps({'schema':'urn:qwen-r9700:rocm-runtime-preflight:v1',"
        "'python':sys.executable,'torch':torch.__version__,'hip':torch.version.hip,"
        "'cuda_available':available,"
        "'device_name':torch.cuda.get_device_name(0) if available else None,"
        "'architecture':arch},sort_keys=True))"
    )
    return _command(
        "rocm-runtime-preflight",
        "Fail before build or allocation unless the qualified Python, PyTorch, ROCm, and gfx1201 "
        "device are available through the standard runtime wrapper.",
        [with_rocm, rocm_python, "-c", code],
        root,
        requires_gpu=True,
        timeout_seconds=120,
        validator={"kind": "rocm-runtime", "architecture": "gfx1201"},
        environment=environment,
    )


def _static_commands(
    component: str, records: Sequence[Mapping[str, Any]], root: Path
) -> list[dict]:
    tests = {
        "w4a16": (
            "tests/test_rdna_hybrid_w4a16.py",
            "tests/test_native_b_gate_up_silu.py",
        ),
        "quest": (
            "tests/test_quest_compiled_attention.py",
            "tests/test_logical_page_contract.py",
            "tests/test_tree_attention_contract.py",
        ),
        "gdn": (
            "tests/test_rdna_gdn_decode.py",
            "tests/test_gdn_recoverssm.py",
            "tests/test_qwen_gdn_recoverssm_overlay.py",
            "tests/test_qwen_gdn_recoverssm_align_overlay.py",
        ),
        "dflash": (
            "tests/test_dflash_parity.py",
            "tests/test_dflash2_v026_overlay.py",
            "tests/test_dflash_capacity.py",
        ),
        "ffn_swiglu": ("tests/test_native_b_gate_up_silu.py",),
        "norm_residual": ("tests/test_gemma_rmsnorm_fp32_overlay.py",),
        "lm_head": (
            "tests/test_bf16_lm_head_benchmark.py",
            "tests/test_qwen_lm_head_serial_m8_overlay.py",
            "tests/test_qwen_lm_head_prefix_serial_overlay.py",
        ),
        "kv_cache": (
            "tests/test_target_kv_scale.py",
            "tests/test_dflash_kv_scale.py",
        ),
        "runtime": (
            "tests/test_runtime_stage_hooks.py",
            "tests/test_qwen_remote_deploy.py",
            "tests/test_qwen_remote_fetch.py",
        ),
        "custom": (),
    }[component]
    commands: list[dict] = []
    commands.append(
        _command(
            "semantic-counterexample-regressions",
            "Replay every preserved semantic counterexample and require rejected paths "
            "to remain off.",
            [
                "uv",
                "run",
                "--no-sync",
                "pytest",
                "-q",
                "tests/test_assurance_layer_diagnosis.py",
                "tests/test_quest_cached_gemm_selector.py",
            ],
            root,
            requires_gpu=False,
            timeout_seconds=600,
        )
    )
    if tests:
        commands.append(
            _command(
                "component-unit-tests",
                "Run the smallest existing CPU-only regression set for this component.",
                ["uv", "run", "--no-sync", "pytest", "-q", *tests],
                root,
                requires_gpu=False,
                timeout_seconds=600,
            )
        )
    python_paths = [record["path"] for record in records if str(record["path"]).endswith(".py")]
    if python_paths:
        commands.append(
            _command(
                "candidate-python-lint",
                "Lint exactly the candidate Python sources bound into this plan.",
                ["uv", "run", "--no-sync", "ruff", "check", *python_paths],
                root,
                requires_gpu=False,
                timeout_seconds=300,
            )
        )
    if not commands:
        commands.append(
            _command(
                "repository-import-smoke",
                "Import the qualification package when no component-specific static gate exists.",
                ["uv", "run", "--no-sync", "python", "-c", "import qwen_r9700_lab"],
                root,
                requires_gpu=False,
                timeout_seconds=120,
            )
        )
    return commands


def _default_component_commands(
    component: str,
    root: Path,
    evidence_root: Path,
    rocm_python: str,
    with_rocm: str,
    component_max_ms: float,
    selector_max_ms: float,
    attention_pages: int,
    bundle_artifacts: Mapping[str, Path] | None = None,
) -> list[dict]:
    remote = root / "experiments" / "bole-tree-verify" / "remote"
    quest = root / "experiments" / "bole-tree-verify" / "hip" / "sub1ms-quest"
    build_root = evidence_root / "build"
    result_root = evidence_root / "component"
    rocm_environment = {
        "PYTORCH_ROCM_ARCH": "gfx1201",
        "QWEN_ROCM_PYTHON": rocm_python,
        "QWEN_WITH_ROCM": with_rocm,
    }
    runtime_preflight = _runtime_preflight_command(root, rocm_python, with_rocm, rocm_environment)
    bundle_artifacts = bundle_artifacts or {}
    if component == "w4a16":
        split_k_enabled = os.environ.get("QWEN_RDNA_ENABLE_EXPERIMENTAL_SPLIT_K") == "1"
        split_k_disabled = os.environ.get("QWEN_RDNA_DISABLE_SPLIT_K") == "1"
        w4_environment = {
            **rocm_environment,
            "QWEN_RDNA_DISABLE_SPLIT_K": "1" if split_k_disabled else "0",
            "QWEN_RDNA_ENABLE_EXPERIMENTAL_SPLIT_K": "1" if split_k_enabled else "0",
        }
        effective_split_k = split_k_enabled and not split_k_disabled
        build_dir = build_root / "native-b"
        extension = bundle_artifacts.get(
            "native_b",
            build_dir / "rdna_w4a16_native_wmma_gfx1201_v10_native_a_fp16scale.so",
        )
        build_commands = []
        if "native_b" not in bundle_artifacts:
            build_commands.append(
                _command(
                    "w4a16-build",
                    "Build the candidate Native-B extension outside the installed runtime.",
                    [str(remote / "build-rdna-native-w4a16"), str(build_dir)],
                    root,
                    requires_gpu=False,
                    timeout_seconds=1800,
                    environment=w4_environment,
                )
            )
        return [
            runtime_preflight,
            *build_commands,
            _command(
                "w4a16-gpu-parity",
                "Compare production verification widths and projection families to the reference.",
                [
                    with_rocm,
                    rocm_python,
                    str(remote / "test_native_b_fused_rows.py"),
                    "--extension",
                    str(extension),
                    "--rows",
                    "1,2,5,6,8,9,16",
                    "--shapes",
                    "34816x5120,5120x17408,16384x5120,5120x6144,14336x5120",
                ],
                root,
                requires_gpu=True,
                timeout_seconds=1800,
                environment=w4_environment,
            ),
            _command(
                "w4a16-model-schedule-microbench",
                "Time the exact 256-projection M=9 W4 target schedule with GPU events.",
                [
                    with_rocm,
                    rocm_python,
                    str(remote / "benchmark_native_b_runtime_schedule.py"),
                    str(extension),
                    "--rows",
                    "9",
                    "--warmup",
                    "8",
                    "--repeats",
                    "21",
                ],
                root,
                requires_gpu=True,
                timeout_seconds=1800,
                validator={
                    "kind": "w4-schedule",
                    "maximum_ms": component_max_ms,
                    "experimental_split_k_enabled": effective_split_k,
                },
                environment=w4_environment,
                stdout_artifact=result_root / "native-b.json",
            ),
            _command(
                "w4a16-250k-projection",
                "Project the measured schedule into the canonical occupied-250K round budget.",
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "qwen-r9700-kernel-250k",
                    "--change-kind",
                    "native_b_implementation",
                    "--changed-stage",
                    "native_b",
                    "--component-result",
                    f"native_b={result_root / 'native-b.json'}",
                    "--qualified-stage",
                    "native_b",
                    "--emitted",
                    "5.672,7,8,9",
                    "--require-component-ready",
                ],
                root,
                requires_gpu=False,
                timeout_seconds=120,
                validator={"kind": "250k-projection"},
                stdout_artifact=result_root / "250k-projection.json",
            ),
        ]
    if component == "quest":
        artifact = bundle_artifacts.get("quest")
        build_dir = artifact.parent if artifact is not None else build_root / "quest-fp8"
        extension = artifact or build_dir / "quest_fp8_selector_gfx1201.so"
        build_commands = []
        if artifact is None:
            build_commands.append(
                _command(
                    "quest-build",
                    "Build the FP8 logical-page selector and grouped-WMMA consumer in isolation.",
                    [str(quest / "build-quest-fp8-selector"), str(build_dir)],
                    root,
                    requires_gpu=False,
                    timeout_seconds=1800,
                    environment=rocm_environment,
                )
            )
        return [
            runtime_preflight,
            *build_commands,
            _command(
                "quest-exact-cached-selector-gate",
                "Require fused cold-seeded centroids, cached scores, ordered Top96 rows, "
                "row masks, and exact 16-split compaction to equal the reference selector "
                "at both occupied contexts, including poisoned workspace reuse.",
                [
                    with_rocm,
                    rocm_python,
                    str(quest / "benchmark_fp8_exact_cached_selector.py"),
                    "--module",
                    str(extension),
                    "--output",
                    str(result_root / "quest-exact-cached-selector.json"),
                    "--contexts",
                    "60298",
                    "249957",
                    "--warmups",
                    "8",
                    "--repeats",
                    "21",
                    "--poison-repeats",
                    "100",
                ],
                root,
                requires_gpu=True,
                timeout_seconds=1800,
                validator={
                    "kind": "quest-exact-cached-selector",
                    "contexts": [60_298, 249_957],
                    "minimum_poison_repeats": 100,
                    "required_comparisons": list(QUEST_COLD_SEED_EXACT_COMPARISONS),
                    "maximum_cold_seed_overhead_ratio": 1.05,
                },
                environment=rocm_environment,
            ),
            _command(
                "quest-selector-gpu-parity-microbench",
                "Check logical-page selection exactly and time the occupied-250K selector.",
                [
                    with_rocm,
                    rocm_python,
                    str(quest / "benchmark_fp8_logical_selector.py"),
                    str(build_dir),
                    "--context",
                    "249957",
                    "--block-size",
                    "1648",
                    "--budget",
                    str(attention_pages),
                    "--recent",
                    str(min(32, attention_pages)),
                    "--warmup",
                    "8",
                    "--repeats",
                    "21",
                ],
                root,
                requires_gpu=True,
                timeout_seconds=1800,
                validator={"kind": "quest-selector", "maximum_ms": selector_max_ms},
                environment=rocm_environment,
                stdout_artifact=result_root / "quest-selector.json",
            ),
            _command(
                "quest-gpu-parity-microbench",
                "Check FP8 attention against dense PyTorch and sweep occupied-250K page budgets.",
                [
                    with_rocm,
                    rocm_python,
                    str(quest / "benchmark_fp8_grouped_attention.py"),
                    str(build_dir),
                    "--pages",
                    "128,192,256,384",
                    "--context",
                    "249957",
                    "--block-size",
                    "1648",
                    "--warmup",
                    "8",
                    "--repeats",
                    "21",
                    "--parity-pages",
                    str(attention_pages),
                    "--direct-bf16-output",
                    "--output-rows",
                    "9",
                ],
                root,
                requires_gpu=True,
                timeout_seconds=1800,
                validator={
                    "kind": "quest-fp8",
                    "pages": attention_pages,
                    "maximum_ms": component_max_ms,
                    "maximum_abs_error": 0.01,
                    "minimum_cosine": 0.9999,
                },
                environment=rocm_environment,
                stdout_artifact=result_root / "quest.json",
            ),
            _command(
                "quest-250k-projection",
                "Project measured FP8 attention into the canonical occupied-250K round budget.",
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "qwen-r9700-kernel-250k",
                    "--change-kind",
                    "quest_kernel_implementation",
                    "--changed-stage",
                    "quest",
                    "--component-result",
                    f"quest={result_root / 'quest.json'}",
                    "--qualified-stage",
                    "quest",
                    "--emitted",
                    "5.672,7,8,9",
                    "--require-component-ready",
                ],
                root,
                requires_gpu=False,
                timeout_seconds=120,
                validator={"kind": "250k-projection"},
                stdout_artifact=result_root / "250k-projection.json",
            ),
        ]
    if component == "gdn":
        artifact = bundle_artifacts.get("gdn")
        build_dir = artifact.parent if artifact is not None else build_root / "gdn"
        extension = artifact or build_dir / "rdna_gdn_decode_gfx1201.so"
        build_commands = []
        if artifact is None:
            build_commands.append(
                _command(
                    "gdn-build",
                    "Build the candidate RecoverSSM extension outside the installed runtime.",
                    [str(remote / "build-rdna-gdn-decode"), str(build_dir)],
                    root,
                    requires_gpu=False,
                    timeout_seconds=1800,
                    environment=rocm_environment,
                )
            )
        return [
            runtime_preflight,
            *build_commands,
            _command(
                "gdn-gpu-parity",
                "Compare RecoverSSM rollback/commit outputs and state to materialized MTP state.",
                [
                    with_rocm,
                    rocm_python,
                    str(remote / "test_recoverssm_gpu_parity.py"),
                    str(build_dir),
                    "--accepted",
                    "1,5,9",
                ],
                root,
                requires_gpu=True,
                timeout_seconds=1800,
                validator={"kind": "gdn-parity"},
                environment=rocm_environment,
                stdout_artifact=result_root / "gdn-parity.json",
            ),
            _command(
                "gdn-gpu-microbench",
                "Time the exact 48-layer M=9 RecoverSSM schedule with GPU events.",
                [
                    with_rocm,
                    rocm_python,
                    str(remote / "benchmark_recoverssm_runtime.py"),
                    str(extension),
                    "--warmup",
                    "8",
                    "--repeats",
                    "21",
                    "--layers",
                    "48",
                ],
                root,
                requires_gpu=True,
                timeout_seconds=1800,
                validator={"kind": "gdn-schedule", "maximum_ms": component_max_ms},
                environment=rocm_environment,
                stdout_artifact=result_root / "gdn.json",
            ),
            _command(
                "gdn-250k-projection",
                "Project measured RecoverSSM into the canonical occupied-250K round budget.",
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "qwen-r9700-kernel-250k",
                    "--change-kind",
                    "gdn_kernel_implementation",
                    "--changed-stage",
                    "gdn",
                    "--component-result",
                    f"gdn={result_root / 'gdn.json'}",
                    "--qualified-stage",
                    "gdn",
                    "--emitted",
                    "5.672,7,8,9",
                    "--require-component-ready",
                ],
                root,
                requires_gpu=False,
                timeout_seconds=120,
                validator={"kind": "250k-projection"},
                stdout_artifact=result_root / "250k-projection.json",
            ),
        ]
    return [runtime_preflight]


def _custom_command(raw: str, command_id: str, description: str, root: Path) -> dict:
    try:
        argv = shlex.split(raw)
    except ValueError as error:
        raise QualificationError(f"invalid {command_id} command: {error}") from error
    return _command(
        command_id,
        description,
        argv,
        root,
        requires_gpu=True,
        timeout_seconds=1800,
    )


def _live_commands(
    args: argparse.Namespace,
    root: Path,
    records: Sequence[Mapping[str, Any]],
    throughput_fixtures: Mapping[int, Path],
) -> dict:
    evidence_root = args.evidence_dir.expanduser().resolve()
    base = args.base_url.rstrip("/")
    metrics_url = args.metrics_url or f"{base}/metrics"
    common = [
        "--base-url",
        base,
        "--model",
        args.model,
        "--backend-id",
        args.backend_id,
        "--server-command-file",
        str(args.server_command_file.expanduser().resolve()),
        "--server-max-sequences",
        "1",
        "--exclusive-gpu",
        "--metrics-url",
        metrics_url,
    ]
    if not args.collect_gpu_telemetry:
        common.append("--no-gpu-telemetry")
    candidate_parity = evidence_root / "live" / "candidate-parity.json"
    parity_report = evidence_root / "live" / "greedy-parity.json"
    parity_argv = [
        "uv",
        "run",
        "--no-sync",
        "qwen-r9700-dflash-parity",
        "capture",
        "--backend-kind",
        "dflash",
        "--backend-id",
        args.backend_id,
        "--build-id",
        args.candidate_id,
        "--base-url",
        base,
        "--model",
        args.model,
        "--request-id",
        f"kernel-qualification-{args.candidate_id}",
        "--max-tokens",
        "160",
        "--config-file",
        f"server-command={args.server_command_file.expanduser().resolve()}",
        "--output",
        str(candidate_parity),
    ]
    for record in records:
        parity_argv.extend(["--build-file", f"{record['label']}={record['path']}"])
    return {
        "parity": [
            _command(
                "capture-candidate-greedy",
                "Capture 160 exact greedy tokens from the already-running candidate lane.",
                parity_argv,
                root,
                requires_gpu=True,
                timeout_seconds=1200,
            ),
            _command(
                "compare-greedy-token-parity",
                "Require candidate completion token IDs to equal the frozen target-only capture.",
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "qwen-r9700-dflash-parity",
                    "compare",
                    str(args.target_parity.expanduser().resolve()),
                    str(candidate_parity),
                    "--output",
                    str(parity_report),
                ],
                root,
                requires_gpu=False,
                timeout_seconds=120,
            ),
        ],
        "quality": [
            _command(
                "short-functional-quality",
                "Run complete code and sequence outputs through the static quality evaluators.",
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "qwen-r9700-c1",
                    *common,
                    "--fixture",
                    str(root / "benchmarks" / "c1" / "quality-fixture-v3.json"),
                    "--output",
                    str(evidence_root / "live" / "quality-v3.json"),
                ],
                root,
                requires_gpu=True,
                timeout_seconds=3600,
            )
        ],
        "throughput": [
            _command(
                f"true-c1-throughput-{context}",
                (
                    "Measure release-client post-first-token throughput at the exact "
                    f"occupied context {context}; projections cannot satisfy this gate."
                ),
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "qwen-r9700-c1",
                    *common,
                    "--fixture",
                    str(throughput_fixtures[context]),
                    "--output",
                    str(evidence_root / "live" / f"c1-{context}.json"),
                ],
                root,
                requires_gpu=True,
                timeout_seconds=3600,
            )
            for context in (60_298, 249_957)
        ],
    }


def _assurance_commands(
    *,
    args: argparse.Namespace,
    root: Path,
    round_certificates: Mapping[int, Path],
    state_campaigns: Mapping[int, Path],
    component_campaigns: Mapping[int, Path],
) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    semantic_source = args.semantic_source_sha256
    schemas = {
        "certificate": "urn:qwen-r9700:round-equivalence-certificate:v4",
        "state": "urn:qwen-r9700:round-equivalence-state-campaign:v6",
        "components": "urn:qwen-r9700:round-equivalence-component-campaign:v2",
    }
    for context in (60_298, 249_957):
        for kind, argv in (
            (
                "certificate",
                [
                    str(root / "scripts" / "qwen-round-equivalence"),
                    "verify",
                    "--certificate",
                    str(round_certificates[context]),
                ],
            ),
            (
                "state",
                [
                    str(root / "scripts" / "qwen-round-equivalence-campaign"),
                    "verify-state",
                    "--campaign",
                    str(state_campaigns[context]),
                ],
            ),
            (
                "components",
                [
                    str(root / "scripts" / "qwen-round-equivalence-campaign"),
                    "verify-components",
                    "--campaign",
                    str(component_campaigns[context]),
                ],
            ),
        ):
            commands.append(
                _command(
                    f"round-{kind}-{context}",
                    f"Verify the authenticated {kind} evidence at context {context}.",
                    argv,
                    root,
                    requires_gpu=False,
                    timeout_seconds=300,
                    validator={
                        "kind": "semantic-evidence",
                        "context_tokens": context,
                        "schema": schemas[kind],
                        "semantic_source_sha256": semantic_source,
                    },
                )
            )
    return commands


def _transaction_commands(
    *,
    args: argparse.Namespace,
    root: Path,
    transaction_campaigns: Mapping[int, Path],
) -> list[dict[str, Any]]:
    schema = "urn:qwen-r9700:failure-atomic-transaction-campaign:v2"
    return [
        _command(
            f"transaction-{context}",
            (
                "Verify exhaustive rollback and post-publication recovery for every "
                f"commit width and declared semantic fault point at context {context}."
            ),
            [
                str(root / "scripts" / "qwen-transaction-campaign"),
                "--campaign",
                str(transaction_campaigns[context]),
            ],
            root,
            requires_gpu=False,
            timeout_seconds=300,
            validator={
                "kind": "semantic-evidence",
                "context_tokens": context,
                "schema": schema,
                "semantic_source_sha256": args.semantic_source_sha256,
            },
        )
        for context in (60_298, 249_957)
    ]


def _comparison_commands(
    *,
    args: argparse.Namespace,
    root: Path,
    lifecycle_comparisons: Mapping[tuple[int, str], Path],
    artifact_comparisons: Mapping[int, Path],
) -> dict[str, list[dict[str, Any]]]:
    schema = "urn:qwen-r9700:coding-turbo-oracle-comparison:v1"
    lifecycle_commands: list[dict[str, Any]] = []
    for (context, lifecycle), path in sorted(lifecycle_comparisons.items()):
        comparison_kind = (
            "snapshot_equivalence" if lifecycle == "snapshot_restore" else "lifecycle_equivalence"
        )
        lifecycle_commands.append(
            _command(
                f"lifecycle-{context}-{lifecycle}",
                f"Verify exact {lifecycle} state equivalence at context {context}.",
                [
                    str(root / "scripts" / "qwen-coding-turbo-oracle"),
                    "verify",
                    "--comparison",
                    str(path),
                ],
                root,
                requires_gpu=False,
                timeout_seconds=300,
                validator={
                    "comparison_kind": comparison_kind,
                    "context_tokens": context,
                    "kind": "semantic-evidence",
                    "right_lifecycle": lifecycle,
                    "schema": schema,
                    "semantic_source_sha256": args.semantic_source_sha256,
                },
            )
        )
    artifact_commands = [
        _command(
            f"artifact-equivalence-{context}",
            f"Verify assurance/release equivalence at context {context}.",
            [
                str(root / "scripts" / "qwen-coding-turbo-oracle"),
                "verify",
                "--comparison",
                str(artifact_comparisons[context]),
            ],
            root,
            requires_gpu=False,
            timeout_seconds=300,
            validator={
                "comparison_kind": "artifact_equivalence",
                "context_tokens": context,
                "kind": "semantic-evidence",
                "schema": schema,
                "semantic_source_sha256": args.semantic_source_sha256,
            },
        )
        for context in (60_298, 249_957)
    ]
    return {"artifact_equivalence": artifact_commands, "lifecycle": lifecycle_commands}


def create_plan(args: argparse.Namespace, root: Path | None = None) -> dict[str, Any]:
    root = (root or find_project_root()).resolve()
    if args.component not in COMPONENTS:
        raise QualificationError(f"unsupported component: {args.component}")
    if not args.candidate_id.strip() or any(character.isspace() for character in args.candidate_id):
        raise QualificationError("candidate-id must be non-empty and contain no whitespace")
    if args.component_max_ms <= 0:
        raise QualificationError("component-max-ms must be positive")
    if args.selector_max_ms <= 0:
        raise QualificationError("selector-max-ms must be positive")
    if not 1 <= args.attention_pages <= 1024:
        raise QualificationError("attention-pages must be between 1 and 1024")
    if args.minimum_post_first_tps < 0 or args.minimum_e2e_tps < 0:
        raise QualificationError("TPS gates must be non-negative")
    semantic_effect = getattr(args, "semantic_effect", "exact_execution")
    if semantic_effect not in SEMANTIC_EFFECTS:
        raise QualificationError(f"unsupported semantic effect: {semantic_effect}")
    claimed_gain = getattr(args, "claimed_tps_gain", None)
    if claimed_gain is not None and claimed_gain <= 0:
        raise QualificationError("claimed-tps-gain must be positive")
    if not args.component_only and semantic_effect == "changes_target_definition":
        raise QualificationError(
            "a target-definition change cannot enter the exact Quest96 promotion lane; "
            "qualify it as a separate model/quality experiment"
        )
    semantic_unit = (
        getattr(args, "semantic_unit", None) or DEFAULT_SEMANTIC_UNIT_BY_COMPONENT[args.component]
    )
    if semantic_unit not in SEMANTIC_UNITS:
        raise QualificationError(f"unsupported semantic unit: {semantic_unit}")
    if args.component not in SEMANTIC_UNIT_COMPONENTS[semantic_unit]:
        raise QualificationError(
            f"semantic unit {semantic_unit} cannot be qualified as component {args.component}; "
            f"allowed={sorted(SEMANTIC_UNIT_COMPONENTS[semantic_unit])}"
        )
    if args.component in CUSTOM_GPU_COMMAND_COMPONENTS and not (
        args.gpu_parity_command and args.gpu_microbench_command
    ):
        raise QualificationError(
            f"{args.component} requires both --gpu-parity-command and --gpu-microbench-command"
        )
    bundle_manifest = getattr(args, "kernel_bundle_manifest", None)
    if bundle_manifest is not None and args.component not in BUNDLE_COMPONENT_BY_QUALIFIER:
        raise QualificationError(
            "--kernel-bundle-manifest is supported only for w4a16, quest, and gdn"
        )
    bundle_record: dict[str, Any] | None = None
    bundle_artifacts: dict[str, Path] = {}
    bundle_paths: list[Path] = []
    if bundle_manifest is not None:
        bundle_record, bundle_artifacts, bundle_paths = _verified_bundle(bundle_manifest, root)
    records = _candidate_records(args.candidate_file)
    evidence_root = args.evidence_dir.expanduser().resolve()
    if evidence_root.exists():
        raise QualificationError(
            f"refusing an existing evidence directory: {evidence_root}; choose a new candidate ID"
        )
    stages: list[dict[str, Any]] = [
        {
            "id": "static",
            "purpose": "Cheap CPU-only source and contract checks; never allocates the GPU.",
            "commands": _static_commands(args.component, records, root),
        }
    ]
    component_commands = _default_component_commands(
        args.component,
        root,
        evidence_root,
        args.rocm_python,
        args.with_rocm,
        args.component_max_ms,
        args.selector_max_ms,
        args.attention_pages,
        bundle_artifacts,
    )
    if args.gpu_parity_command:
        component_commands.append(
            _custom_command(
                args.gpu_parity_command,
                "custom-gpu-parity",
                "Run the explicitly supplied component parity command.",
                root,
            )
        )
    if args.gpu_microbench_command:
        component_commands.append(
            _custom_command(
                args.gpu_microbench_command,
                "custom-gpu-microbench",
                "Run the explicitly supplied component microbenchmark command.",
                root,
            )
        )
    stages.append(
        {
            "id": "component",
            "purpose": "Isolated gfx1201 build, numerical parity, and component GPU timing.",
            "commands": component_commands,
            "manual_gate": (
                "Provide --gpu-parity-command and --gpu-microbench-command for this component."
                if not component_commands
                else None
            ),
        }
    )
    if not args.component_only:
        semantic_source = getattr(args, "semantic_source_sha256", None)
        if (
            not isinstance(semantic_source, str)
            or len(semantic_source) != 64
            or any(character not in "0123456789abcdef" for character in semantic_source)
        ):
            raise QualificationError("live qualification requires --semantic-source-sha256")
        missing = [
            name
            for name, value in (
                ("--server-command-file", args.server_command_file),
                ("--target-parity", args.target_parity),
            )
            if value is None
        ]
        if missing:
            raise QualificationError(
                "live qualification requires "
                + ", ".join(missing)
                + "; use --component-only otherwise"
            )
        round_certificates = _context_evidence(
            getattr(args, "round_certificate", []), "round certificate"
        )
        state_campaigns = _context_evidence(
            getattr(args, "round_state_campaign", []), "round state campaign"
        )
        component_campaigns = _context_evidence(
            getattr(args, "round_component_campaign", []), "round component campaign"
        )
        transaction_campaigns = _context_evidence(
            getattr(args, "transaction_campaign", []), "transaction campaign"
        )
        lifecycle_comparisons = _lifecycle_evidence(getattr(args, "lifecycle_comparison", []))
        artifact_comparisons = _context_evidence(
            getattr(args, "artifact_comparison", []), "artifact comparison"
        )
        throughput_fixtures = _context_evidence(
            getattr(args, "throughput_fixture", []), "throughput fixture"
        )
        throughput_controls = _context_evidence(
            getattr(args, "throughput_control_evidence", []),
            "throughput control evidence",
        )
        server_command_file = args.server_command_file.expanduser().resolve()
        target_parity = args.target_parity.expanduser().resolve()
        if not server_command_file.is_file():
            raise QualificationError(
                f"server command evidence is not a regular file: {server_command_file}"
            )
        if not target_parity.is_file():
            raise QualificationError(
                f"target parity capture is not a regular file: {target_parity}"
            )
        live = _live_commands(args, root, records, throughput_fixtures)
        comparisons = _comparison_commands(
            args=args,
            root=root,
            lifecycle_comparisons=lifecycle_comparisons,
            artifact_comparisons=artifact_comparisons,
        )
        stages.append(
            {
                "id": "round_equivalence",
                "purpose": (
                    "Require c=0..8 target/draft state parity, sibling/permutation "
                    "invariance, and all semantic-unit boundaries at both contexts."
                ),
                "commands": _assurance_commands(
                    args=args,
                    root=root,
                    round_certificates=round_certificates,
                    state_campaigns=state_campaigns,
                    component_campaigns=component_campaigns,
                ),
            }
        )
        stages.append(
            {
                "id": "transaction",
                "purpose": (
                    "Require failure-atomic rollback and idempotent recovery at every "
                    "KV, GDN, projection, attention, residual, FFN, normalization, "
                    "LM-head, scheduler, allocator, snapshot, and publication boundary."
                ),
                "commands": _transaction_commands(
                    args=args,
                    root=root,
                    transaction_campaigns=transaction_campaigns,
                ),
            }
        )
        stages.append(
            {
                "id": "lifecycle",
                "purpose": (
                    "Require exact snapshot, cache-hit, rollback, restart, offload, "
                    "reuse, session, branch, and chat-switch state equivalence."
                ),
                "commands": comparisons["lifecycle"],
            }
        )
        stages.append(
            {
                "id": "artifact_equivalence",
                "purpose": (
                    "Require instrumentation-free release output/state to match the "
                    "assurance artifact at both contexts."
                ),
                "commands": comparisons["artifact_equivalence"],
            }
        )
        stages.extend(
            [
                {
                    "id": stage_id,
                    "purpose": {
                        "parity": "Short exact greedy target-token parity on the staged live lane.",
                        "quality": "Complete-output quality checks before interpreting TPS.",
                        "throughput": "Fixed end-to-end C1 throughput and speculation evidence.",
                    }[stage_id],
                    "commands": live[stage_id],
                }
                for stage_id in ("quality", "throughput")
            ]
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "result_type": PLAN_TYPE,
        "created_at": _utc_now(),
        "project_root": str(root),
        "candidate": {
            "id": args.candidate_id,
            "component": args.component,
            "semantic_unit": semantic_unit,
            "files": records,
        },
        "evidence_root": str(evidence_root),
        "gates": {
            "component_max_ms": args.component_max_ms,
            "selector_max_ms": args.selector_max_ms,
            "attention_pages": args.attention_pages,
            "minimum_post_first_tps": args.minimum_post_first_tps,
            "minimum_e2e_tps": args.minimum_e2e_tps,
            "require_byte_determinism": True,
        },
        "performance_evidence_policy": PERFORMANCE_EVIDENCE_POLICY,
        "optimization_claim": {
            "claimed_tps_gain": claimed_gain,
            "evidence_status": "unverified_hypothesis"
            if claimed_gain is not None
            else "not_claimed",
            "semantic_effect": semantic_effect,
            "promotable_basis": "measured_same_fixture_release_ablation_at_both_exact_contexts",
            "used_by_any_gate": False,
        },
        "promotion_order": [stage["id"] for stage in stages],
        "stages": stages,
        "qualification_inputs": _qualification_inputs(
            args.component, root, stages, extra_paths=bundle_paths
        ),
        "counterexample_corpus": _counterexample_corpus_record(root),
    }
    if not args.component_only:
        payload["throughput_controls"] = {
            str(context): _file_record(path)
            for context, path in sorted(throughput_controls.items())
        }
        payload["qualification_inputs"] = _qualification_inputs(
            args.component,
            root,
            stages,
            extra_paths=[*bundle_paths, *throughput_controls.values()],
        )
    if bundle_record is not None:
        payload["kernel_bundle"] = bundle_record
    payload["plan_id"] = f"sha256:{hashlib.sha256(_canonical_bytes(payload)).hexdigest()}"
    return payload


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise QualificationError(f"JSON document must be an object: {path}")
    return value


def load_plan(path: Path) -> dict[str, Any]:
    plan = _load_json_object(path)
    if plan.get("schema_version") != 1 or plan.get("result_type") != PLAN_TYPE:
        raise QualificationError(f"not a kernel qualification plan: {path}")
    recorded_id = plan.get("plan_id")
    unsigned = dict(plan)
    unsigned.pop("plan_id", None)
    expected = f"sha256:{hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()}"
    if recorded_id != expected:
        raise QualificationError(f"qualification plan hash mismatch: {path}")
    stages = plan.get("stages")
    if not isinstance(stages, list) or not stages:
        raise QualificationError("qualification plan has no stages")
    return plan


def verify_candidate(plan: Mapping[str, Any]) -> None:
    candidate = plan.get("candidate")
    if not isinstance(candidate, dict) or not isinstance(candidate.get("files"), list):
        raise QualificationError("qualification plan candidate record is invalid")
    qualification_inputs = plan.get("qualification_inputs", [])
    if not isinstance(qualification_inputs, list):
        raise QualificationError("qualification input records are invalid")
    for record in [*candidate["files"], *qualification_inputs]:
        if not isinstance(record, dict):
            raise QualificationError("candidate file record is invalid")
        path = Path(str(record.get("path", "")))
        if not path.is_file():
            raise QualificationError(f"candidate file disappeared: {path}")
        actual = _sha256_file(path)
        if actual != record.get("sha256"):
            raise QualificationError(
                f"candidate or qualification input changed after the plan was created: {path}; "
                "create a new plan"
            )
    _verify_counterexample_corpus(plan.get("counterexample_corpus"))


def _json_stdout(stdout: str, command_id: str) -> dict[str, Any]:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise QualificationError(f"{command_id} did not emit one JSON object: {error}") from error
    if not isinstance(value, dict):
        raise QualificationError(f"{command_id} JSON output must be an object")
    return value


def _bounded_tail(value: str, *, lines: int = FAILURE_TAIL_LINES) -> list[str]:
    return value.splitlines()[-lines:]


def _failure_diagnostic(
    *,
    command: Mapping[str, Any],
    failures: Sequence[str],
    stdout: str,
    stderr: str,
    timed_out: bool,
    launch_error: str | None = None,
    candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    combined = "\n".join((stdout, stderr, "\n".join(failures), launch_error or "")).lower()
    markers = {
        "timeout": timed_out,
        "command_launch": launch_error is not None,
        "missing_python_module": "modulenotfounderror" in combined or "no module named" in combined,
        "missing_executable_or_file": "no such file or directory" in combined
        or "cannot find the file" in combined,
        "missing_or_incompatible_abi": any(
            marker in combined
            for marker in ("undefined symbol", "missing selector abi", "lacks required abi")
        ),
        "hsa_memory_fault": any(
            marker in combined
            for marker in ("hsa_amd_memory_fault", "memory access fault", "hsa exception")
        ),
        "device_oom": any(
            marker in combined
            for marker in ("out of memory", "hip error outofmemory", "hiperroroutofmemory")
        ),
        "nonfinite": any(marker in combined for marker in ("nonfinite", "non-finite"))
        or re.search(r"\b(?:nan|inf)\b", combined) is not None,
        "engine_failure": any(
            marker in combined
            for marker in (
                "enginedeaderror",
                "enginecore failed",
                "enginecore died",
                "enginecore process exited",
                "engine core failed",
                "engine core died",
                "worker died",
            )
        ),
        "connection_failure": any(
            marker in combined
            for marker in ("connection refused", "failed to connect", "connection reset")
        ),
        "snapshot_or_cache_restore": re.search(
            r"snapshot.{0,80}(?:mismatch|failed|invalid|corrupt)|"
            r"(?:cache restore|fixed-slot|generation id).{0,80}(?:mismatch|failed|invalid)",
            combined,
        )
        is not None,
        "source_or_provenance": re.search(
            r"(?:sha-?256|hash|provenance|source).{0,80}(?:mismatch|differs|invalid|failed)",
            combined,
        )
        is not None,
        "semantic_mismatch": any(
            marker in combined
            for marker in (
                "parity failed",
                "first divergence",
                "mismatch",
                "not byte deterministic",
            )
        ),
        "performance_gate": re.search(
            r"(?:throughput|t/s|tps).{0,80}(?:exceeds|below|failed)|"
            r"(?:exceeds|below).{0,80}(?:throughput|t/s|tps)",
            combined,
        )
        is not None,
        "jit_compilation_during_inference": "jit compilation during inference" in combined,
        "runtime_fallback": "falling back to" in combined,
        "artifact_collision": any(
            marker in combined
            for marker in (
                "file exists",
                "already exists",
                "refusing to replace",
                "run is consumed",
            )
        ),
        "shell_or_wrapper_violation": any(
            marker in combined for marker in ("fish:", "unknown command: --", "qwen-remote-bash")
        ),
        "cold_fill_attempt": any(
            marker in combined
            for marker in ("cold fill", "cold-fill", "raw prompt fill", "cache miss prefill")
        ),
    }
    classifications = [name for name, present in markers.items() if present]
    if not classifications:
        classifications = ["unclassified"]
    missing_module = None
    match = re.search(r"no module named ['\"]?([^'\"\s]+)", combined)
    if match:
        missing_module = match.group(1)
    recommended_by_classification = {
        "hsa_memory_fault": "preserve the process/log and bisect the first device access boundary",
        "device_oom": "inspect live allocations and the exact VRAM/cache configuration",
        "nonfinite": "stop at the first nonfinite semantic-unit output and compare its inputs",
        "engine_failure": "inspect the EngineCore exit and the last completed lifecycle phase",
        "connection_failure": "inspect API lifecycle and server health before retrying the request",
        "snapshot_or_cache_restore": (
            "compare snapshot manifest, ownership, generations, and lengths"
        ),
        "source_or_provenance": (
            "fetch all installed runtime postimages and rebuild one source-bound plan"
        ),
        "semantic_mismatch": (
            "run the minimized counterexample and stop at the first unequal component"
        ),
        "performance_gate": "inspect measured release stage timings and emitted tokens per round",
        "jit_compilation_during_inference": (
            "extend authenticated warmup for the reported runtime shape"
        ),
        "runtime_fallback": (
            "identify and qualify the activated fallback before interpreting throughput"
        ),
        "artifact_collision": "allocate a fresh create-only qualification identity",
        "shell_or_wrapper_violation": (
            "relaunch exclusively through the standardized remote wrapper"
        ),
        "cold_fill_attempt": "stop and restore the authenticated fixed-slot snapshot instead",
    }
    semantic_units = {
        "quest_page_selection": (
            "selected page",
            "top96",
            "page visibility",
            "row mask",
            "page_row_mask",
        ),
        "quest_attention": ("attention output", "historical output", "causal tail"),
        "w4a16_projection": ("projection", "native-b", "native_b", "w4a16"),
        "gdn_state": ("gdn state", "recoverssm", "recurrent state"),
        "convolution_state": ("conv state", "convolution state", "conv carry"),
        "kv_state": ("target kv", "draft kv", "kv cache", "kv scale"),
        "position_or_rope": ("rope", "position id", "absolute position"),
        "commit_or_publication": ("commit epoch", "accepted prefix", "publication epoch"),
        "snapshot_or_cache": ("snapshot", "cache restore", "fixed-slot"),
    }
    unit_hints = [
        unit
        for unit, phrases in semantic_units.items()
        if any(phrase in combined for phrase in phrases)
    ]
    first_difference: dict[str, Any] = {}
    for field, pattern in (
        ("token_index", r"(?:first divergence|token)[^0-9]{0,24}(\d+)"),
        ("layer_index", r"layer[^0-9]{0,12}(\d+)"),
        ("row_index", r"row[^0-9]{0,12}(\d+)"),
        ("page_index", r"(?:page index|top96 index)[^0-9]{0,12}(\d+)"),
    ):
        match = re.search(pattern, combined)
        if match:
            first_difference[field] = int(match.group(1))
    artifact_paths = sorted(
        set(
            re.findall(
                r"/(?:home|data|tmp)/[^\s'\";,]+\.(?:json|jsonl|log|txt|bin|pt)",
                "\n".join((stdout, stderr)),
            )
        )
    )
    fingerprint_source = {
        "candidate": dict(candidate or {}),
        "classifications": classifications,
        "command_id": command.get("id"),
        "first_difference": first_difference or None,
        "semantic_unit_hints": unit_hints,
        "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
        "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
    }
    diagnostic = {
        "candidate": dict(candidate or {}),
        "command_id": command.get("id"),
        "phase": "preflight" if str(command.get("id", "")).endswith("preflight") else "execution",
        "classifications": classifications,
        "primary_classification": classifications[0],
        "recommended_checks": [
            recommended_by_classification[name]
            for name in classifications
            if name in recommended_by_classification
        ],
        "semantic_unit_hints": unit_hints,
        "first_difference": first_difference or None,
        "referenced_artifacts": artifact_paths,
        "failure_fingerprint_sha256": hashlib.sha256(
            _canonical_bytes(fingerprint_source)
        ).hexdigest(),
        "stdout_sha256": fingerprint_source["stdout_sha256"],
        "stderr_sha256": fingerprint_source["stderr_sha256"],
        "missing_module": missing_module,
        "failures": list(failures),
        "launch_error": launch_error,
        "stdout_tail": _bounded_tail(stdout),
        "stderr_tail": _bounded_tail(stderr),
    }
    if "semantic_mismatch" in classifications:
        diagnostic["counterexample_draft"] = {
            "candidate_id": (candidate or {}).get("id"),
            "component": (candidate or {}).get("component"),
            "semantic_unit": (candidate or {}).get("semantic_unit")
            or (unit_hints[0] if unit_hints else None),
            "first_difference": first_difference or None,
            "referenced_artifacts": artifact_paths,
            "failure_fingerprint_sha256": diagnostic["failure_fingerprint_sha256"],
            "status": "unminimized",
        }
    return diagnostic


def _validate_command(command: Mapping[str, Any], returncode: int, stdout: str) -> list[str]:
    failures: list[str] = []
    if returncode != 0:
        failures.append(f"exit status {returncode}")
        return failures
    validator = command.get("validator", {"kind": "exit-zero"})
    if not isinstance(validator, dict):
        return ["invalid validator"]
    kind = validator.get("kind")
    if kind == "exit-zero":
        return failures
    try:
        document = _json_stdout(stdout, str(command.get("id")))
        if kind == "rocm-runtime":
            if document.get("schema") != "urn:qwen-r9700:rocm-runtime-preflight:v1":
                failures.append("ROCm runtime preflight schema differs")
            if document.get("cuda_available") is not True:
                failures.append("ROCm runtime reports no GPU")
            if not document.get("torch") or not document.get("hip"):
                failures.append("ROCm runtime lacks a HIP-enabled PyTorch build")
            architecture = str(document.get("architecture") or "")
            expected_architecture = str(validator.get("architecture") or "")
            if expected_architecture not in architecture:
                failures.append(
                    f"ROCm runtime architecture {architecture!r} is not {expected_architecture!r}"
                )
        elif kind == "w4-schedule":
            if document.get("schedule_contract") != NATIVE_B_SCHEDULE_CONTRACT:
                failures.append("W4 schedule contract is not the canonical 256-call inventory")
            if document.get("rows") != DEFAULT_VERIFICATION_ROWS:
                failures.append("W4 schedule must use M=9 verification rows")
            if document.get("w4_projection_count") != NATIVE_B_W4_PROJECTION_COUNT:
                failures.append("W4 schedule must contain exactly 256 W4 projections")
            if document.get("excluded_bf16_gdn_ba_count") != NATIVE_B_EXCLUDED_BF16_GDN_BA_COUNT:
                failures.append("W4 schedule must record 48 excluded BF16 GDN b/a linears")
            expected_split_k = validator.get("experimental_split_k_enabled") is True
            if document.get("experimental_split_k_enabled") is not expected_split_k:
                failures.append("W4 schedule split-K mode does not match the hash-bound plan")
            raw_schedule = document.get("stages")
            schedule_signature = (
                tuple(
                    (item.get("name"), item.get("n"), item.get("k"), item.get("count"))
                    for item in raw_schedule
                )
                if isinstance(raw_schedule, list)
                and len(raw_schedule) == len(NATIVE_B_W4_SCHEDULE)
                and all(isinstance(item, dict) for item in raw_schedule)
                else ()
            )
            if schedule_signature != NATIVE_B_W4_SCHEDULE:
                failures.append("W4 schedule stages do not match the canonical inventory")
            measured = float(document["weighted_schedule_median_ms"])
            maximum = float(validator["maximum_ms"])
            if measured > maximum:
                failures.append(f"W4 schedule {measured:.3f} ms exceeds {maximum:.3f} ms")
        elif kind == "quest-fp8":
            parity = document["parity"]
            timing = document["timings"][str(validator["pages"])]
            if not parity.get("finite"):
                failures.append("Quest output is non-finite")
            if float(parity["max_abs"]) > float(validator["maximum_abs_error"]):
                failures.append("Quest maximum absolute error exceeds its gate")
            if float(parity["cosine"]) < float(validator["minimum_cosine"]):
                failures.append("Quest cosine is below its gate")
            measured = float(timing["median_ms"])
            maximum = float(validator["maximum_ms"])
            if measured > maximum:
                failures.append(
                    f"Quest {validator['pages']}p {measured:.3f} ms exceeds {maximum:.3f} ms"
                )
        elif kind == "quest-selector":
            parity = document["parity"]
            timing = document["timing"]
            if not all(
                parity.get(field)
                for field in ("selected_match", "selected_ascending", "recent_pages_present")
            ):
                failures.append("Quest logical selector parity failed")
            measured = float(timing["median_ms"])
            maximum = float(validator["maximum_ms"])
            if measured > maximum:
                failures.append(f"Quest selector {measured:.3f} ms exceeds {maximum:.3f} ms")
        elif kind == "quest-exact-cached-selector":
            if document.get("schema") != "urn:qwen-r9700:exact-cached-selector-gate:v1":
                failures.append("exact cached-selector evidence schema differs")
            if document.get("passed") is not True:
                failures.append("exact cached-selector gate is not passing")
            expected_contexts = list(validator.get("contexts", []))
            contract = document.get("contract")
            actual_contexts = contract.get("contexts") if isinstance(contract, dict) else None
            if actual_contexts != expected_contexts:
                failures.append("exact cached-selector contexts differ")
            context_results = document.get("contexts")
            if not isinstance(context_results, list) or len(context_results) != len(
                expected_contexts
            ):
                failures.append("exact cached-selector context results are incomplete")
            else:
                for result in context_results:
                    if not isinstance(result, dict):
                        failures.append("exact cached-selector emitted a non-object context result")
                        continue
                    context = result.get("context")
                    comparisons = result.get("comparisons")
                    poison = result.get("poison_reuse")
                    if result.get("passed") is not True:
                        failures.append(f"exact cached-selector context {context} failed")
                    required_comparisons = set(validator.get("required_comparisons", []))
                    actual_comparisons = (
                        set(comparisons) if isinstance(comparisons, dict) else set()
                    )
                    missing_comparisons = sorted(required_comparisons - actual_comparisons)
                    if missing_comparisons:
                        failures.append(
                            f"exact cached-selector context {context} lacks required comparisons: "
                            + ", ".join(missing_comparisons)
                        )
                    if (
                        not isinstance(comparisons, dict)
                        or not comparisons
                        or any(
                            not isinstance(comparison, dict)
                            or comparison.get("exact") is not True
                            or comparison.get("first_difference") is not None
                            for comparison in comparisons.values()
                        )
                    ):
                        failures.append(
                            f"exact cached-selector context {context} has a mismatched output"
                        )
                    if (
                        not isinstance(poison, dict)
                        or poison.get("exact") is not True
                        or int(poison.get("repeats", 0))
                        < int(validator.get("minimum_poison_repeats", 0))
                    ):
                        failures.append(
                            f"exact cached-selector context {context} failed workspace reuse"
                        )
                    timing = result.get("timing")
                    if not isinstance(timing, dict):
                        failures.append(
                            f"exact cached-selector context {context} lacks timing evidence"
                        )
                        continue
                    try:
                        reference_ms = float(timing["reference_median_ms_per_layer"])
                        cached_ms = float(timing["candidate_median_ms_per_layer"])
                        cold_seed_ms = float(timing["cold_seed_median_ms_per_layer"])
                    except (KeyError, TypeError, ValueError):
                        failures.append(
                            f"exact cached-selector context {context} has invalid timing evidence"
                        )
                        continue
                    if not all(
                        math.isfinite(value) and value > 0
                        for value in (reference_ms, cached_ms, cold_seed_ms)
                    ):
                        failures.append(
                            f"exact cached-selector context {context} has "
                            "nonpositive/nonfinite timing"
                        )
                    elif cached_ms >= reference_ms:
                        failures.append(
                            f"exact cached-selector context {context} cached path is not faster"
                        )
                    elif cold_seed_ms > reference_ms * float(
                        validator.get("maximum_cold_seed_overhead_ratio", 1.0)
                    ):
                        failures.append(
                            f"exact cached-selector context {context} cold seed "
                            "overhead is excessive"
                        )
        elif kind == "gdn-schedule":
            if not document.get("output_finite"):
                failures.append("GDN output is non-finite")
            measured = float(document["schedule_median_ms"])
            maximum = float(validator["maximum_ms"])
            if measured > maximum:
                failures.append(f"GDN schedule {measured:.3f} ms exceeds {maximum:.3f} ms")
        elif kind == "gdn-parity":
            results = document["results"]
            if not isinstance(results, list) or not results:
                failures.append("GDN parity emitted no cases")
            else:
                failures.extend(
                    f"GDN parity failed at accepted={result.get('accepted')}"
                    for result in results
                    if any(
                        int(result[field]) != 0
                        for field in (
                            "round_1_mismatches",
                            "round_2_mismatches",
                            "checkpoint_mismatches",
                            "pending_after_round",
                        )
                    )
                )
        elif kind == "250k-projection":
            decision = document["decision"]
            if decision.get("component_projection_ready") is not True:
                failures.append("occupied-250K component projection is not ready")
            if document.get("projection_kind") != "component_sum_not_live_occupied_measurement":
                failures.append("occupied-250K projection has an unexpected evidence label")
        elif kind == "semantic-evidence":
            if document.get("passed") is not True:
                failures.append("semantic evidence is not passing")
            if document.get("schema") != validator.get("schema"):
                failures.append("semantic evidence schema differs")
            if document.get("semantic_source_sha256") != validator.get("semantic_source_sha256"):
                failures.append("semantic evidence source differs from the candidate")
            context = document.get("context_tokens", document.get("prompt_tokens"))
            if context != validator.get("context_tokens"):
                failures.append("semantic evidence context differs")
            expected_kind = validator.get("comparison_kind")
            if expected_kind is not None and document.get("comparison_kind") != expected_kind:
                failures.append("semantic comparison kind differs")
            expected_lifecycle = validator.get("right_lifecycle")
            if (
                expected_lifecycle is not None
                and document.get("right_lifecycle") != expected_lifecycle
            ):
                failures.append("semantic comparison lifecycle differs")
        else:
            failures.append(f"unknown validator kind: {kind!r}")
    except (KeyError, TypeError, ValueError, QualificationError) as error:
        failures.append(f"invalid validator evidence: {error}")
    return failures


def _stage_by_id(plan: Mapping[str, Any], stage_id: str) -> dict[str, Any]:
    for stage in plan["stages"]:
        if isinstance(stage, dict) and stage.get("id") == stage_id:
            return stage
    raise QualificationError(f"stage is not present in this plan: {stage_id}")


def _run_evidence_path(plan: Mapping[str, Any], stage_id: str) -> Path:
    return Path(str(plan["evidence_root"])) / "stage-runs" / f"{stage_id}.json"


def _completed_stage(plan: Mapping[str, Any], stage_id: str) -> bool:
    path = _run_evidence_path(plan, stage_id)
    if not path.is_file():
        return False
    evidence = _load_json_object(path)
    return evidence.get("plan_id") == plan.get("plan_id") and evidence.get("passed") is True


def run_stage(plan: Mapping[str, Any], stage_id: str) -> dict[str, Any]:
    verify_candidate(plan)
    promotion_order = list(plan["promotion_order"])
    if stage_id not in promotion_order:
        raise QualificationError(f"stage is not present in plan: {stage_id}")
    for prerequisite in promotion_order[: promotion_order.index(stage_id)]:
        if not _completed_stage(plan, prerequisite):
            raise QualificationError(
                f"stage {stage_id} requires a passing {prerequisite} stage first"
            )
    stage = _stage_by_id(plan, stage_id)
    commands = stage.get("commands")
    if not isinstance(commands, list) or not commands:
        raise QualificationError(stage.get("manual_gate") or f"stage {stage_id} has no commands")
    evidence_root = Path(str(plan["evidence_root"]))
    (evidence_root / "live").mkdir(parents=True, exist_ok=True)
    (evidence_root / "build").mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    stage_passed = True
    for command in commands:
        if not isinstance(command, dict):
            raise QualificationError(f"stage {stage_id} has an invalid command")
        argv = command.get("argv")
        if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            raise QualificationError(f"command {command.get('id')} has invalid argv")
        environment = os.environ.copy()
        command_environment = command.get("environment", {})
        if not isinstance(command_environment, dict):
            raise QualificationError(f"command {command.get('id')} has invalid environment")
        environment.update({str(key): str(value) for key, value in command_environment.items()})
        started = time.monotonic()
        launch_error: str | None = None
        try:
            process = subprocess.run(
                argv,
                cwd=command["cwd"],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=float(command["timeout_seconds"]),
            )
            elapsed = time.monotonic() - started
            stdout = process.stdout
            stderr = process.stderr
            failures = _validate_command(command, process.returncode, stdout)
            if command.get("stdout_artifact") is not None:
                try:
                    artifact_document = _json_stdout(stdout, str(command.get("id")))
                    _write_new_json(Path(command["stdout_artifact"]), artifact_document)
                except QualificationError as error:
                    failures.append(f"cannot preserve JSON stdout artifact: {error}")
            returncode: int | None = process.returncode
            timed_out = False
        except subprocess.TimeoutExpired as error:
            elapsed = time.monotonic() - started
            stdout = (
                error.stdout.decode(errors="replace")
                if isinstance(error.stdout, bytes)
                else error.stdout
            )
            stderr = (
                error.stderr.decode(errors="replace")
                if isinstance(error.stderr, bytes)
                else error.stderr
            )
            stdout = stdout or ""
            stderr = stderr or ""
            failures = [f"timed out after {command['timeout_seconds']} seconds"]
            returncode = None
            timed_out = True
        except OSError as error:
            elapsed = time.monotonic() - started
            stdout = ""
            stderr = ""
            launch_error = f"{type(error).__name__}: {error}"
            failures = [f"command launch failed: {launch_error}"]
            returncode = None
            timed_out = False
        record = {
            "id": command["id"],
            "argv": argv,
            "cwd": command["cwd"],
            "requires_gpu": command["requires_gpu"],
            "elapsed_seconds": elapsed,
            "returncode": returncode,
            "timed_out": timed_out,
            "passed": not failures,
            "failures": failures,
            "stdout": stdout[:MAX_CAPTURE_BYTES],
            "stderr": stderr[:MAX_CAPTURE_BYTES],
            "stdout_truncated": len(stdout) > MAX_CAPTURE_BYTES,
            "stderr_truncated": len(stderr) > MAX_CAPTURE_BYTES,
        }
        if failures:
            record["diagnostic"] = _failure_diagnostic(
                command=command,
                failures=failures,
                stdout=stdout,
                stderr=stderr,
                timed_out=timed_out,
                launch_error=launch_error,
                candidate=plan.get("candidate"),
            )
        records.append(record)
        if failures:
            stage_passed = False
            break
    result = {
        "schema_version": 1,
        "result_type": RUN_TYPE,
        "completed_at": _utc_now(),
        "plan_id": plan["plan_id"],
        "candidate": plan["candidate"],
        "stage": stage_id,
        "passed": stage_passed,
        "commands": records,
    }
    if not stage_passed:
        failed_record = next(record for record in records if record["passed"] is False)
        result["failure_diagnostic"] = failed_record["diagnostic"]
    _write_new_json(_run_evidence_path(plan, stage_id), result)
    return result


def _audit_live_evidence(plan: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    root = Path(str(plan["evidence_root"])) / "live"
    failures: list[str] = []
    report: dict[str, Any] = {}
    parity_path = root / "greedy-parity.json"
    if parity_path.is_file():
        parity = _load_json_object(parity_path)
        identical = parity.get("identical_completion_token_ids") is True
        report["greedy_parity"] = {
            "identical_completion_token_ids": identical,
            "first_divergence_index": parity.get("first_divergence_index"),
        }
        if not identical:
            failures.append("greedy candidate tokens differ from target-only tokens")
    quality_path = root / "quality-v3.json"
    if quality_path.is_file():
        quality = _load_json_object(quality_path)
        passed = quality.get("quality_gate", {}).get("passed") is True
        report["quality"] = {"passed": passed}
        if not passed:
            failures.append("short functional quality gate failed")
    cases: dict[str, Any] = {}
    contexts: dict[str, Any] = {}
    post_first_values: list[float] = []
    e2e_values: list[float] = []
    for context in (60_298, 249_957):
        c1_path = root / f"c1-{context}.json"
        if not c1_path.is_file():
            if "throughput" in plan.get("promotion_order", []):
                failures.append(f"missing measured C1 evidence at context {context}")
            continue
        c1 = _load_json_object(c1_path)
        summaries = c1.get("summary_by_case")
        if not isinstance(summaries, dict) or not summaries:
            failures.append(f"C1 evidence at context {context} has no case summaries")
            continue
        controls = plan.get("throughput_controls")
        control_record = controls.get(str(context)) if isinstance(controls, dict) else None
        if not isinstance(control_record, dict):
            failures.append(f"missing hash-bound release control at context {context}")
            continue
        control_path = Path(str(control_record.get("path", "")))
        if (
            not control_path.is_file()
            or control_record.get("sha256") != _sha256_file(control_path)
            or control_record.get("size_bytes") != control_path.stat().st_size
        ):
            failures.append(f"release control evidence changed at context {context}")
            continue
        control = _load_json_object(control_path)
        control_summaries = control.get("summary_by_case")
        if not isinstance(control_summaries, dict) or set(control_summaries) != set(summaries):
            failures.append(f"release control cases differ at context {context}")
            continue
        for case_id, summary in sorted(summaries.items()):
            control_summary = control_summaries[case_id]
            if not isinstance(summary, dict) or not isinstance(control_summary, dict):
                failures.append(f"C1 ablation case {context}:{case_id} is invalid")
                continue
            failures.extend(
                f"C1 ablation contract differs at {context}:{case_id}:{field}"
                for field in (
                    "measured_repeats",
                    "output_sha256_values",
                    "prompt_tokens",
                    "completion_tokens",
                )
                if summary.get(field) != control_summary.get(field)
            )
            candidate_post = float(summary["client_post_first_token_tps_p50"])
            control_post = float(control_summary["client_post_first_token_tps_p50"])
            candidate_e2e = float(summary["client_e2e_output_tps_p50"])
            control_e2e = float(control_summary["client_e2e_output_tps_p50"])
            if candidate_post < control_post:
                failures.append(
                    f"candidate regresses post-first TPS at {context}:{case_id}: "
                    f"{candidate_post:.3f} < {control_post:.3f}"
                )
            if candidate_e2e < control_e2e:
                failures.append(
                    f"candidate regresses E2E TPS at {context}:{case_id}: "
                    f"{candidate_e2e:.3f} < {control_e2e:.3f}"
                )
            contexts.setdefault(str(context), {}).setdefault("ablation", {})[case_id] = {
                "candidate_post_first_tps_p50": candidate_post,
                "control_post_first_tps_p50": control_post,
                "post_first_tps_gain": candidate_post - control_post,
                "candidate_e2e_tps_p50": candidate_e2e,
                "control_e2e_tps_p50": control_e2e,
                "e2e_tps_gain": candidate_e2e - control_e2e,
            }
        context_post: list[float] = []
        context_e2e: list[float] = []
        for case_id, summary in sorted(summaries.items()):
            if not isinstance(summary, dict):
                failures.append(f"C1 case {context}:{case_id} summary is invalid")
                continue
            post_first = float(summary["client_post_first_token_tps_p50"])
            e2e = float(summary["client_e2e_output_tps_p50"])
            deterministic = summary.get("output_is_byte_deterministic") is True
            case_key = f"{context}:{case_id}"
            cases[case_key] = {
                "context_tokens": context,
                "post_first_tps_p50": post_first,
                "e2e_tps_p50": e2e,
                "byte_deterministic": deterministic,
                "acceptance_rate": summary.get("speculation_weighted_acceptance_rate"),
            }
            post_first_values.append(post_first)
            e2e_values.append(e2e)
            context_post.append(post_first)
            context_e2e.append(e2e)
            if plan["gates"].get("require_byte_determinism") and not deterministic:
                failures.append(f"C1 case {case_key} is not byte deterministic")
        if context_post:
            contexts.setdefault(str(context), {}).update(
                {
                    "minimum_post_first_tps_p50": min(context_post),
                    "minimum_e2e_tps_p50": min(context_e2e),
                }
            )
    if cases:
        minimum_post = min(post_first_values)
        minimum_e2e = min(e2e_values)
        report["throughput"] = {
            "contexts": contexts,
            "cases": cases,
            "minimum_post_first_tps_p50": minimum_post,
            "mean_post_first_tps_p50": sum(post_first_values) / len(post_first_values),
            "minimum_e2e_tps_p50": minimum_e2e,
            "mean_e2e_tps_p50": sum(e2e_values) / len(e2e_values),
        }
        post_gate = float(plan["gates"].get("minimum_post_first_tps", 0.0))
        e2e_gate = float(plan["gates"].get("minimum_e2e_tps", 0.0))
        if minimum_post < post_gate:
            failures.append(
                f"minimum C1 post-first TPS {minimum_post:.3f} is below {post_gate:.3f}"
            )
        if minimum_e2e < e2e_gate:
            failures.append(f"minimum C1 E2E TPS {minimum_e2e:.3f} is below {e2e_gate:.3f}")
    return report, failures


def audit_plan(plan: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    stages: dict[str, Any] = {}
    failures: list[str] = []
    for stage_id in plan["promotion_order"]:
        path = _run_evidence_path(plan, stage_id)
        if not path.is_file():
            stages[stage_id] = {"status": "not-run"}
            continue
        evidence = _load_json_object(path)
        passed = evidence.get("plan_id") == plan.get("plan_id") and evidence.get("passed") is True
        stages[stage_id] = {"status": "passed" if passed else "failed", "path": str(path)}
        if not passed:
            failures.append(f"stage {stage_id} did not pass")
    live, live_failures = _audit_live_evidence(plan)
    failures.extend(live_failures)
    projection_path = Path(str(plan["evidence_root"])) / "component" / "250k-projection.json"
    component_projection: dict[str, Any] | None = None
    if projection_path.is_file():
        projection = _load_json_object(projection_path)
        component_projection = {
            "evidence_label": projection.get("projection_kind"),
            "projected_round_ms": projection.get("round", {}).get("projected_ms"),
            "required_emitted_tokens_per_round": projection.get("round", {}).get(
                "required_emitted_tokens_per_round"
            ),
            "scenarios": projection.get("round", {}).get("scenarios"),
            "cold_fill_needed_for_this_iteration": projection.get("decision", {}).get(
                "cold_fill_needed_for_this_iteration"
            ),
            "production_250k_tps_claim_ready": projection.get("decision", {}).get(
                "production_250k_tps_claim_ready"
            ),
        }
    report = {
        "schema_version": 1,
        "result_type": SUMMARY_TYPE,
        "created_at": _utc_now(),
        "plan_id": plan["plan_id"],
        "candidate": plan["candidate"],
        "stages": stages,
        "component_250k_projection": component_projection,
        "performance_evidence_policy": plan.get(
            "performance_evidence_policy", PERFORMANCE_EVIDENCE_POLICY
        ),
        "optimization_claim": plan.get("optimization_claim"),
        "live": live,
        "passed": not failures and all(item["status"] == "passed" for item in stages.values()),
        "failures": failures,
    }
    return report, failures


def _shell_command(command: Mapping[str, Any]) -> str:
    environment = command.get("environment", {})
    prefix = ["env", *(f"{key}={value}" for key, value in sorted(environment.items()))]
    argv = [*prefix, *command["argv"]] if environment else command["argv"]
    return shlex.join(str(item) for item in argv)


def _help_description(name: str, synopsis: str, description: str) -> str:
    return f"""NAME
  {name}

SYNOPSIS
  {synopsis}

DESCRIPTION
  {description}

OPTIONS
  The accepted options are listed below."""


def _help_epilog(operation: str, examples: str) -> str:
    return f"""OPERATION
  {operation}

EXAMPLES
{examples}

FILES
  Plans are immutable JSON. Stage runs and live benchmark evidence are written below --evidence-dir.

PATHS
  Candidate and evidence paths are resolved to absolute paths when a plan is created.

SECURITY NOTES
  Commands execute without a shell. This tool never deploys, restarts, or modifies a server runtime.

EXIT STATUS
  0 on success; 1 when a qualification gate fails; 2 for invalid arguments or evidence.

AUTHORS
  Qwen R9700 inference lab contributors."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-r9700-kernel-qualify",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_help_description(
            "qwen-r9700-kernel-qualify - qualify one hash-bound kernel candidate",
            "qwen-r9700-kernel-qualify COMMAND [OPTIONS]",
            "Create, inspect, execute, and audit a fail-closed promotion plan. The staged "
            "sequence is "
            "static checks, isolated GPU parity/microbench, greedy token parity, complete-output "
            "quality, then true-C1 throughput.",
        ),
        epilog=_help_epilog(
            "Select create, show, run, or audit. No operation starts or stops a model server.",
            "  qwen-r9700-kernel-qualify create --help\n"
            "  qwen-r9700-kernel-qualify show candidate.plan.json\n"
            "  qwen-r9700-kernel-qualify run candidate.plan.json --through throughput",
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser(
        "create",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_help_description(
            "qwen-r9700-kernel-qualify create - bind a candidate to a qualification plan",
            "qwen-r9700-kernel-qualify create --candidate-id ID --component KIND "
            "--candidate-file LABEL=PATH --evidence-dir DIR --output PLAN",
            "Hash the candidate and every qualification input, then create an immutable "
            "staged plan.",
        ),
        epilog=_help_epilog(
            "Build exact argv arrays without executing any command or touching a model server.",
            "  qwen-r9700-kernel-qualify create --candidate-id q4 --component quest "
            "--candidate-file hip=kernel.cu --component-only --evidence-dir /tmp/q4 "
            "--output /tmp/q4.plan.json",
        ),
    )
    create.add_argument("--candidate-id", required=True)
    create.add_argument("--component", required=True, choices=COMPONENTS)
    create.add_argument(
        "--semantic-unit",
        choices=SEMANTIC_UNITS,
        help=(
            "single semantic boundary changed by this candidate; omitted only for the legacy "
            "component default"
        ),
    )
    create.add_argument("--candidate-file", action="append", default=[], metavar="LABEL=PATH")
    create.add_argument(
        "--semantic-effect",
        choices=SEMANTIC_EFFECTS,
        default="exact_execution",
        help=(
            "exact_execution preserves the selected target; non_authoritative_proposal may only "
            "affect speed; changes_target_definition is excluded from the exact Quest96 lane"
        ),
    )
    create.add_argument(
        "--claimed-tps-gain",
        type=float,
        help=(
            "planning hypothesis only; never satisfies a gate and remains unverified until a "
            "same-fixture release ablation at both exact contexts exists"
        ),
    )
    create.add_argument("--evidence-dir", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    create.add_argument("--component-only", action="store_true")
    create.add_argument("--gpu-parity-command")
    create.add_argument("--gpu-microbench-command")
    create.add_argument("--rocm-python", default=DEFAULT_ROCM_PYTHON)
    create.add_argument("--with-rocm", default=DEFAULT_WITH_ROCM)
    create.add_argument(
        "--kernel-bundle-manifest",
        type=Path,
        help=(
            "verified manifest.json from scripts/kernel-workflow; use its staged artifact "
            "directly instead of rebuilding it"
        ),
    )
    create.add_argument("--component-max-ms", type=float)
    create.add_argument("--selector-max-ms", type=float, default=1.0)
    create.add_argument("--attention-pages", type=int, default=128)
    create.add_argument("--base-url", default="http://127.0.0.1:8000")
    create.add_argument("--metrics-url")
    create.add_argument("--model", default="qwen3.8-27b-frozenlock")
    create.add_argument("--backend-id")
    create.add_argument("--server-command-file", type=Path)
    create.add_argument("--target-parity", type=Path)
    create.add_argument("--semantic-source-sha256")
    create.add_argument(
        "--round-certificate",
        action="append",
        type=_parse_context_file,
        default=[],
        metavar="CONTEXT=PATH",
    )
    create.add_argument(
        "--round-state-campaign",
        action="append",
        type=_parse_context_file,
        default=[],
        metavar="CONTEXT=PATH",
    )
    create.add_argument(
        "--round-component-campaign",
        action="append",
        type=_parse_context_file,
        default=[],
        metavar="CONTEXT=PATH",
    )
    create.add_argument(
        "--transaction-campaign",
        action="append",
        type=_parse_context_file,
        default=[],
        metavar="CONTEXT=PATH",
    )
    create.add_argument(
        "--lifecycle-comparison",
        action="append",
        type=_parse_context_lifecycle_file,
        default=[],
        metavar="CONTEXT:LIFECYCLE=PATH",
    )
    create.add_argument(
        "--artifact-comparison",
        action="append",
        type=_parse_context_file,
        default=[],
        metavar="CONTEXT=PATH",
    )
    create.add_argument(
        "--throughput-fixture",
        action="append",
        type=_parse_context_file,
        default=[],
        metavar="CONTEXT=PATH",
        help=(
            "exact 60298/249957 occupied-context C1 fixture; both contexts are "
            "mandatory for a live promotion plan"
        ),
    )
    create.add_argument(
        "--throughput-control-evidence",
        action="append",
        type=_parse_context_file,
        default=[],
        metavar="CONTEXT=PATH",
        help=(
            "hash-bound instrumentation-free release C1 result with the candidate disabled; "
            "both exact contexts are mandatory and must use identical cases and outputs"
        ),
    )
    create.add_argument("--collect-gpu-telemetry", action="store_true")
    create.add_argument("--minimum-post-first-tps", type=float, default=70.0)
    create.add_argument("--minimum-e2e-tps", type=float, default=0.0)

    show = subparsers.add_parser(
        "show",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_help_description(
            "qwen-r9700-kernel-qualify show - inspect an immutable plan",
            "qwen-r9700-kernel-qualify show PLAN [--stage STAGE] [--json]",
            "Verify the plan hash and display exact commands without executing them.",
        ),
        epilog=_help_epilog(
            "Render all stages, or one selected stage, as shell-quoted argv for review.",
            "  qwen-r9700-kernel-qualify show candidate.plan.json --stage component",
        ),
    )
    show.add_argument("plan", type=Path)
    show.add_argument("--stage", choices=STAGE_ORDER)
    show.add_argument("--json", action="store_true")

    run = subparsers.add_parser(
        "run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_help_description(
            "qwen-r9700-kernel-qualify run - execute one staged qualification",
            "qwen-r9700-kernel-qualify run PLAN (--stage STAGE | --through STAGE)",
            "Verify all bound hashes, enforce prerequisites, and execute argv directly without "
            "a shell.",
        ),
        epilog=_help_epilog(
            "Run one stage or every not-yet-passed stage through a selected gate, preserving "
            "evidence.",
            "  qwen-r9700-kernel-qualify run candidate.plan.json --through throughput",
        ),
    )
    run.add_argument("plan", type=Path)
    selection = run.add_mutually_exclusive_group(required=True)
    selection.add_argument("--stage", choices=STAGE_ORDER)
    selection.add_argument("--through", choices=STAGE_ORDER)

    audit = subparsers.add_parser(
        "audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_help_description(
            "qwen-r9700-kernel-qualify audit - summarize candidate promotion evidence",
            "qwen-r9700-kernel-qualify audit PLAN [--output SUMMARY]",
            "Report stage status, exact-layout 250K projection, parity, quality, and C1 TPS gates.",
        ),
        epilog=_help_epilog(
            "Read preserved evidence only; optionally require current source bytes to match "
            "the plan.",
            "  qwen-r9700-kernel-qualify audit candidate.plan.json --output summary.json",
        ),
    )
    audit.add_argument("plan", type=Path)
    audit.add_argument("--output", type=Path)
    audit.add_argument("--require-current-candidate", action="store_true")
    return parser


def _default_component_max_ms(component: str) -> float:
    return {
        "w4a16": 45.0,
        "quest": 1.5,
        "gdn": 5.0,
        "ffn_swiglu": 1.0,
        "norm_residual": 1.0,
        "lm_head": 8.0,
        "kv_cache": 1.0,
        "dflash": 20.0,
        "runtime": 1.0,
        "custom": 1.0,
    }[component]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            if args.component_max_ms is None:
                args.component_max_ms = _default_component_max_ms(args.component)
            if args.backend_id is None:
                args.backend_id = f"kernel-{args.candidate_id}"
            plan = create_plan(args)
            _write_new_json(args.output, plan)
            print(f"created {args.output}: {plan['plan_id']}")
            return 0
        plan = load_plan(args.plan)
        if args.command == "show":
            stages = [
                stage
                for stage in plan["stages"]
                if args.stage is None or stage.get("id") == args.stage
            ]
            if args.json:
                print(json.dumps(stages, indent=2, sort_keys=True))
            else:
                for stage in stages:
                    print(f"[{stage['id']}] {stage['purpose']}")
                    if stage.get("manual_gate"):
                        print(f"  BLOCKED: {stage['manual_gate']}")
                    for command in stage["commands"]:
                        print(f"  {command['id']}: {_shell_command(command)}")
            return 0
        if args.command == "run":
            if args.stage:
                result = run_stage(plan, args.stage)
                print(json.dumps(result, indent=2, sort_keys=True))
                return 0 if result["passed"] else 1
            target_index = plan["promotion_order"].index(args.through)
            for stage_id in plan["promotion_order"][: target_index + 1]:
                if _completed_stage(plan, stage_id):
                    print(f"already passed: {stage_id}")
                    continue
                result = run_stage(plan, stage_id)
                print(f"{stage_id}: {'passed' if result['passed'] else 'failed'}")
                if not result["passed"]:
                    return 1
            return 0
        if args.command == "audit":
            if args.require_current_candidate:
                verify_candidate(plan)
            report, failures = audit_plan(plan)
            if args.output:
                _write_new_json(args.output, report)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 1 if failures or not report["passed"] else 0
    except (QualificationError, ConfigurationError, OSError, ValueError) as error:
        print(f"qwen-r9700-kernel-qualify: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
