"""One-shot controller for a single fail-closed Qwen assurance request."""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "urn:qwen-r9700:assurance-one-shot:v1"
MAX_COMPLETION_TOKENS = 2049
STATE_EXPORTER_PREFLIGHT_MARKER = (
    "[qwen-dflash-assurance] initialized-runtime state exporter preflight ready"
)
EXACT_K_RUNTIME_MARKER = "[qwen-full-attention-m8-exact-k] exact K RMSNorm armed"
ARTIFACT_SCHEMA = "urn:qwen-r9700:coding-turbo-artifact:v1"
EXACT_K_CONTRACT_SEGMENT = "full-attention-m8-exact-k-rmsnorm-v1"
EXACT_K_CAPABILITY_OUTPUT = (
    "runtime-contracts/full-attention-m8-exact-k-capability.json"
)
EXACT_K_CAPABILITY_SCHEMA = (
    "urn:qwen-r9700:full-attention-m8-exact-k-capability:v1"
)
EXACT_K_CAPABILITY = "full-attention-m8-exact-k-rmsnorm-v1"
EXACT_K_SOURCE_OUTPUTS = {
    "qwen_exact_k_rmsnorm.cpp": "native/exact-k-rmsnorm/qwen_exact_k_rmsnorm.cpp",
    "qwen_exact_k_rmsnorm.hip": "native/exact-k-rmsnorm/qwen_exact_k_rmsnorm.hip",
    "build_exact_k_rmsnorm.py": (
        "runtime-tools/exact-k-rmsnorm/build_exact_k_rmsnorm.py"
    ),
    "qualify_exact_k_rmsnorm.py": (
        "runtime-tools/exact-k-rmsnorm/qualify_exact_k_rmsnorm.py"
    ),
    "full_attention_m8_exact_k_overlay.py": (
        "runtime-tools/exact-k-rmsnorm/full_attention_m8_exact_k_overlay.py"
    ),
}
EXACT_K_ENVIRONMENT = frozenset(
    {
        "QWEN_FULL_ATTENTION_M8_EXACT_K",
        "QWEN_FULL_ATTENTION_M8_EXACT_K_REQUIRED",
        "QWEN_FULL_ATTENTION_M8_EXACT_K_SITE_SHA256",
        "QWEN_FULL_ATTENTION_M8_EXACT_K_RUNTIME_SHA256",
        "QWEN_FULL_ATTENTION_M8_EXACT_K_EXTENSION",
        "QWEN_FULL_ATTENTION_M8_EXACT_K_EXTENSION_SHA256",
        "QWEN_FULL_ATTENTION_M8_EXACT_K_QUALIFICATION_SHA256",
    }
)
FIXED_SERIAL_CONV_CAPABILITY_OUTPUT = "runtime-contracts/gdn-fixed-serial-conv-m8-capability.json"
FIXED_SERIAL_CONV_CERTIFICATE_OUTPUT = "runtime-contracts/gdn-fixed-serial-conv-m8-component.json"
FIXED_SERIAL_CONV_CAPABILITY_SCHEMA = "urn:qwen-r9700:fixed-serial-conv-m8-capability:v3"
FIXED_SERIAL_CONV_CERTIFICATE_SCHEMA = "urn:qwen-r9700:fixed-serial-conv-m8-component:v3"
FIXED_SERIAL_CONV_CAPABILITY = "gdn-fixed-serial-conv-m8-v3"
FIXED_SERIAL_CONV_ENABLE_ENVIRONMENT = "QWEN_GDN_FIXED_SLOT_SERIAL_CONV_M8=1"
FIXED_SERIAL_CONV_CERTIFICATE_SOURCE = (
    "overlays/vllm-qwen-gdn-fixed-serial-conv-m8-v416/component.json"
)
QUEST_COLD_SEED_CAPABILITY_OUTPUT = (
    "runtime-contracts/quest-cached-centroid-cold-seed-capability.json"
)
QUEST_COLD_SEED_COMPONENT_OUTPUT = (
    "runtime-contracts/quest-cached-centroid-cold-seed-component.json"
)
QUEST_COLD_SEED_ATTESTATION_CAPABILITY_OUTPUT = (
    "runtime-contracts/fixed-slot-cache-attestation-capability.json"
)
QUEST_COLD_SEED_CAPABILITY_SCHEMA = "urn:qwen-r9700:quest-cached-centroid-cold-seed-capability:v1"
QUEST_COLD_SEED_COMPONENT_SCHEMA = "urn:qwen-r9700:exact-cached-selector-gate:v1"
QUEST_COLD_SEED_ATTESTATION_CAPABILITY_SCHEMA = (
    "urn:qwen-r9700:fixed-slot-cache-attestation-capability:v1"
)
QUEST_COLD_SEED_CAPABILITY = "quest-cached-centroid-cold-seed-v1"
QUEST_COLD_SEED_ATTESTATION_CAPABILITY = "qwen-fixed-slot-cache-attestation-v1"
QUEST_COLD_SEED_ENABLE_ENVIRONMENT = "QWEN_QUEST_CACHED_GEMM_COLD_SEED=1"
VLLM_RUNTIME_SITE_PACKAGES = Path(
    "/home/lewis/.local/share/qwen-r9700/runtimes/vllm-rocm/.venv/lib/python3.12/site-packages"
)
VLLM_RUNTIME_ROOT = Path("/home/lewis/.local/share/qwen-r9700/runtimes/vllm-rocm")
VLLM_WITH_ROCM = VLLM_RUNTIME_ROOT / "bin/with-rocm"
VLLM_CONSOLE_SCRIPT = VLLM_RUNTIME_ROOT / ".venv/bin/vllm"
REQUIRED_ROW_LOCAL_DUALPHASE_EXTENSION = Path(
    "/home/lewis/projects/qwen-r9700/artifacts/qualifications/"
    "20260828T-v353-static-row-pair-production/build/"
    "quest_row_local_dualphase_gfx1201.so"
)
REQUIRED_ROW_LOCAL_DUALPHASE_EXTENSION_SHA256 = (
    "c5418060080be2359ddecc46fe7fefba6290eb3a42817fb06479099959fef553"
)
QUEST_CONTINUITY_RUNTIME_BINDINGS = {
    "offload_scheduler": (
        "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py",
        "EXPECTED_DFLASH_PERSISTENT_KV_SCHEDULER_SHA256",
    ),
    "selection": (
        "vllm/distributed/kv_transfer/kv_connector/v1/offloading/qwen_persistent_selection.py",
        "EXPECTED_DFLASH_PERSISTENT_KV_SELECTION_RUNTIME_SHA256",
    ),
    "gpu_runner": (
        "vllm/v1/worker/gpu_model_runner.py",
        "EXPECTED_VLLM_TREE_GPU_RUNNER_SHA256",
    ),
    "scheduler_output": (
        "vllm/v1/core/sched/output.py",
        "EXPECTED_VLLM_TREE_SCHED_OUTPUT_SHA256",
    ),
}
QUEST_CONTINUITY_ATTESTATION_BINDINGS = {
    "offload_scheduler": "offload_scheduler_post_sha256",
    "selection": "selection_post_sha256",
    "gpu_runner": "gpu_model_runner_sha256",
    "scheduler_output": "scheduler_output_sha256",
}
DFLASH_D7_GRAPH_RUNTIME_BINDINGS = {
    "DFlash graph manager": (
        "vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py",
        "c76a2575ad7b025613514825c00299cf56f0e8c6ebdf2061aca3a2bb1af16a10",
        "308a4d9410c2f7ed94eae8161e57ddc5d6a89a25b4d8c8055ee41f908b5d4a5c",
    ),
    "DFlash2 speculator": (
        "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py",
        "bb2975ba2ee6a442c6f9748026b644bb01a26b8181da473c2a1931cae0661ca5",
        "5dc2ad893a32224ce8499b8d8177ac0ff3fc57d4eb8a9dc49f3bd2a10c825378",
    ),
}
DFLASH_B7_GRAPH_SPECULATOR_SHA256 = (
    "d6fdd37deed873ed83251618333dfcf4489150d2a7de899636cbcdbc99987032"
)
FIXED_SLOT_SELECTION_RUNTIME_RELATIVE = (
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading/qwen_persistent_selection.py"
)
TARGET_MODEL_CONFIG = Path("/home/lewis/models/Qwen3.8-27B-int4-AutoRound/config.json")
ASSURANCE_ARTIFACT_BINDINGS = {
    "capture_hook": "assurance/capture_hook.py",
    "capture_reducer": "assurance/qwen_r9700_lab/coding_turbo_capture.py",
    "capture_site": "assurance/capture_site/sitecustomize.py",
    "oracle_contract": "assurance/qwen_r9700_lab/coding_turbo_oracle.py",
    "round_equivalence_contract": "assurance/qwen_r9700_lab/round_equivalence.py",
    "state_exporter": "assurance/coding_turbo_state_exporter.py",
    "state_provider": "assurance/qwen_r9700_lab/coding_turbo_state_provider.py",
}
FULL_INSTRUMENTATION_CONTRACT_SUFFIX = "/full-assurance-instrumentation-v2"
FULL_INSTRUMENTATION_ARTIFACT_BINDINGS = {
    "full_instrumentation_contract": ("assurance/qwen_r9700_lab/assurance_instrumentation.py")
}


def has_full_instrumentation_contract(semantic_contract: object) -> bool:
    """Return whether the exact full-instrumentation contract component is present."""

    if not isinstance(semantic_contract, str):
        return False
    component = FULL_INSTRUMENTATION_CONTRACT_SUFFIX.removeprefix("/")
    return component in semantic_contract.split("/")


OFFLOAD_SHM_REGION_SIZE = 10_736_205_824
OFFLOAD_SHM_MAX_STALE = 8
_OFFLOAD_SHM_NAME = re.compile(
    r"^vllm_offload_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}\.mmap$"
)
RUNTIME_BINDINGS = frozenset(
    {
        "capture_hook",
        "capture_site",
        "launcher",
        "qwen_gdn_linear_attn",
        "qwen3_next",
        "qwen_text_rope",
        "snapshot_selection",
    }
)
RUNTIME_CAPTURE_SHA_ENVS = {
    "qwen_gdn_linear_attn": "QWEN_DFLASH_ASSURANCE_GDN_SHA256",
    "qwen3_next": "QWEN_DFLASH_ASSURANCE_QWEN3_NEXT_SHA256",
    "qwen_text_rope": "QWEN_DFLASH_ASSURANCE_TEXT_ROPE_SHA256",
}
STATE_CAPTURE_BINDINGS = frozenset(
    {
        "assurance_manifest",
        "capture_reducer",
        "oracle_contract",
        "round_equivalence_contract",
        "snapshot_payload_selection",
        "state_exporter",
        "state_consumer_transition_receipt",
        "state_producer_receipt",
        "state_provider",
    }
)
ASSURANCE_BINDINGS = RUNTIME_BINDINGS | STATE_CAPTURE_BINDINGS
FULL_INSTRUMENTATION_BINDINGS = ASSURANCE_BINDINGS | frozenset(
    FULL_INSTRUMENTATION_ARTIFACT_BINDINGS
)
RELEASE_BINDINGS = RUNTIME_BINDINGS - {"capture_hook", "capture_site"}
COMPILED_SELECTOR_PROBE_EVENT = "_compiled_select_row_local_m8_logical_pages"
CACHED_SELECTOR_PROBE_EVENT = "_cached_gemm_select_logical_pages"
CACHED_SELECTOR_COLD_SEED_PROBE_EVENT = "_cached_gemm_seed_exact_row_local_m8"
UNIQUE_DEPTH_MIXED_PROBE_EVENT = "_try_unique_depth_mixed_attention"
B7_STAGE_PROBE_EVENT = "_qwen_stage_best_first_b7"
B7_VERIFY_PROBE_EVENT = "verify_best_first_b7_out"
B7_COMMIT_PROBE_EVENT = "commit_and_clear"
B7_POSITION_STRIDE_SCHEMA = "urn:qwen-r9700:dflash-b7-position-stride-runtime:v1"
B7_POSITION_STRIDE_GATE_SHA256 = "ae71b6dd1cc5fd4223d63859259e715f89369e1e12420ee56c9580dbde86f7e6"
B7_POSITION_STRIDE_RESULT_SHA256 = (
    "9d99c812478cdf934e96158b4eb5a63c8ab36d2fa95746b2376ea83e865ba646"
)
B7_POSITION_STRIDE_RUNTIME_SHA256 = (
    "a1497d2165378221feb68f22410abf1d56da7ac80ce3c6c6c9457a07e891098a"
)
B7_POSITION_STRIDE_RESULT_SUFFIX = "20260829T-v519-mrope-position-runtime-create-only/result.json"
B7_POSITION_STRIDE_OUTPUT_NAME = "b7-mrope-position-admission.json"
B7_ACTIVE_V2_TRANSPORT_SCHEMA = "urn:qwen-r9700:dflash2-b7-active-v2-payload-transport:v1"
B7_ACTIVE_V2_TRANSPORT_GATE_SHA256 = (
    "b005b9096fa9f0786b656165ba4d41bf7613404b5b474507598ab8ff60793bbe"
)
B7_ACTIVE_V2_TRANSPORT_RESULT_SHA256 = (
    "3bd4b8df2182bc3654a3ed90be714ccf3fe5e08dcf174ae466050b356e75e70d"
)
B7_ACTIVE_V2_TRANSPORT_RUNNER_SHA256 = (
    "f9317fe9a2d0a9b927581fbe2d31983bd61746737df621f8dc8d07d15e28b36f"
)
B7_ACTIVE_V2_TRANSPORT_SPECULATOR_SHA256 = DFLASH_B7_GRAPH_SPECULATOR_SHA256
B7_ACTIVE_V2_TRANSPORT_RUNTIME_SHA256 = (
    "10a7963b067a0da50f01e6b92549bddbd217f89e998c261ca20ab80fc4d33f21"
)
B7_ACTIVE_V2_TRANSPORT_RESULT_SUFFIX = (
    "20260829T-v521-b7-active-v2-payload-transport-create-only/result.json"
)
B7_ACTIVE_V2_TRANSPORT_OUTPUT_NAME = "b7-active-v2-payload-admission.json"
QUEST_TREE_WMMA_SCHEMA = "urn:qwen-r9700:quest-b7-tree-wmma-runtime:v1"
QUEST_TREE_WMMA_GATE_SHA256 = "d6ddfdcd02afb55b1c92f3df23ffbbd0aec8f838025cb2f4082123e3370431cf"
QUEST_TREE_WMMA_RESULT_SHA256 = "4b4619db29408ad3c5312e6deda848857f6d0414fe95052ce16648f168267b9c"
QUEST_TREE_WMMA_EXTENSION_SHA256 = (
    "3264c542ae967a76827ba9b203ca0fa91b089e7db7735a05e372b5d18c9092a3"
)
QUEST_TREE_WMMA_RESULT_SUFFIX = "20260829T-v526-tree-wmma-component-create-only/result.json"
QUEST_TREE_WMMA_SOURCE_SHA256 = {
    "build_ninja": "8c52f6248198bd14163c98898ffdde5885013c6817b560c41e96947bff1ac358",
    "candidate": QUEST_TREE_WMMA_EXTENSION_SHA256,
    "control": "dd1e03b0ab0be087869f03a6d09bcf55561937676281f98d60a9ebdd20f6c14f",
    "cpp": "380823ffea4eea6e6e5e5cb76f5bfcfcf10fcd14b3e73b44042dd3b9f62375cc",
    "cpp_object": "9b33dcef6cfa55f7881fb7c7ba46988929f1f06d19b7543bd090bf3d956a2f7a",
    "cu": "2bc701121ba23f078da0a057a96d21e602dc70d67c6095512cce69175196f3b7",
    "gate": QUEST_TREE_WMMA_GATE_SHA256,
    "hip": "989f3fc6aa7da3a26379f81a10c3c58f164e3d144857ae5c279cc4da97973c87",
    "hip_object": "88eabc07a1d26a2edbaa31d587514e82d40d6fd22d8faf476d7025c0929cb54e",
}
QUEST_TREE_WMMA_SOURCE_BASENAME = {
    "build_ninja": "build.ninja",
    "candidate": "quest_fp8_selector_treefix_gfx1201.so",
    "control": "quest_fp8_selector_gfx1201.so",
    "cpp": "quest_fp8_selector_ext.cpp",
    "cpp_object": "quest_fp8_selector_ext.o",
    "cu": "quest_fp8_selector_ext.cu",
    "gate": "verify_tree_wmma_runtime.py",
    "hip": "quest_fp8_selector_ext.hip",
    "hip_object": "quest_fp8_selector_ext.cuda.o",
}
FAILURE_DIAGNOSTIC_SCHEMA = "urn:qwen-r9700:assurance-failure-diagnostic:v1"
NON_PROMOTABLE_DIAGNOSTIC_SCHEMA = "urn:qwen-r9700:layer-diagnostic-identity:v1"
NON_PROMOTABLE_DIAGNOSTIC_KEYS = {
    "artifact_kind",
    "classification",
    "context_tokens",
    "dflash_enabled",
    "lifecycle",
    "model_sha256",
    "payload_producer_receipt_sha256",
    "promotable",
    "prompt_token_ids_sha256",
    "quest_page_budget",
    "run_id",
    "runtime_artifact_manifest_sha256",
    "sampling",
    "schema",
    "semantic_source_sha256",
    "snapshot_manifest_sha256",
    "tokenizer_sha256",
}
RUNTIME_FILE_BINDINGS_SCHEMA = "urn:qwen-r9700:runtime-file-bindings:v1"
RUNTIME_FILE_BINDINGS_NAME = "runtime-file-bindings.json"
RUNTIME_FILE_BINDINGS_ENV = "QWEN_DFLASH_ASSURANCE_RUNTIME_FILE_BINDINGS"
RUNTIME_FILE_BINDINGS_SHA_ENV = "QWEN_DFLASH_ASSURANCE_RUNTIME_FILE_BINDINGS_SHA256"
RUNTIME_FILE_LABELS = frozenset(
    {
        "base_sampler",
        "dflash2_model",
        "dflash2_speculator",
        "dflash_model",
        "dflash_proposer",
        "gdn_adapter",
        "llm_base_proposer",
        "mamba_hybrid",
        "mamba_utils",
        "model_runner",
        "offload_scheduler",
        "qwen3_5",
        "qwen3_next",
        "qwen_text_rope",
        "quest_attention",
        "rejection_sampler",
        "rocm_attention",
    }
)
RUNTIME_OBSERVATIONS_SCHEMA = "urn:qwen-r9700:assurance-runtime-observations:v1"
FATAL_RUNTIME_MARKERS = frozenset(
    {
        "hsa_memory_fault",
        "engine_core_failure",
        "device_out_of_memory",
        "nonfinite_model_value",
        "snapshot_restore_failure",
        "runtime_provenance_failure",
    }
)
TREE_SCORE_CAPTURE_SCHEMA = "qwen-r9700.dflash-tree-score-lattice.v1"
TREE_RANK_CAPTURE_SCHEMA = "qwen-r9700.dflash-tree-rank-evidence.v3"
_FAILURE_MARKERS = (
    (
        "hsa_memory_fault",
        re.compile(
            r"HSA_STATUS_ERROR_MEMORY_FAULT|memory access fault|GPU fault|amdgpu.*fault",
            re.IGNORECASE,
        ),
    ),
    (
        "engine_core_failure",
        re.compile(
            r"EngineDeadError|EngineCore.*(?:failed|died|exited)|engine core.*(?:failed|died)",
            re.IGNORECASE,
        ),
    ),
    (
        "device_out_of_memory",
        re.compile(
            r"(?:HIP|CUDA|GPU|device).{0,32}out of memory|OutOfMemoryError|hipErrorOutOfMemory",
            re.IGNORECASE,
        ),
    ),
    (
        "nonfinite_model_value",
        re.compile(
            r"nonfinite|non-finite|NaN detected|Inf detected|infinite (?:logit|tensor|value)",
            re.IGNORECASE,
        ),
    ),
    (
        "snapshot_restore_failure",
        re.compile(
            r"snapshot.{0,80}(?:mismatch|failed|invalid|corrupt)|"
            r"(?:restore|cache).{0,80}(?:generation|ownership|hash).{0,40}mismatch",
            re.IGNORECASE,
        ),
    ),
    (
        "runtime_provenance_failure",
        re.compile(
            r"(?:SHA-?256|hash|provenance|authenticated).{0,80}(?:mismatch|failed|invalid)",
            re.IGNORECASE,
        ),
    ),
    (
        "jit_compilation_during_inference",
        re.compile(r"JIT compilation during inference", re.IGNORECASE),
    ),
    (
        "runtime_fallback",
        re.compile(r"falling back to", re.IGNORECASE),
    ),
    (
        "performance_configuration_warning",
        re.compile(
            r"suboptimal performance|enforce eager.{0,80}disabl(?:e|ing)|"
            r"optimizations settings.{0,80}ignored",
            re.IGNORECASE,
        ),
    ),
    (
        "upstream_documentation_error",
        re.compile(
            r"\[ERROR\].{0,100}(?:not documented|add it to the docstring)",
            re.IGNORECASE,
        ),
    ),
)
_PYTHON_EXCEPTION_LINE = re.compile(
    r"(?P<type>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Fault)):\s+"
    r"(?P<message>.+)$"
)
_WRAPPER_EXCEPTION_TYPES = frozenset(
    {
        "EngineDeadError",
        "HTTPError",
        "PublicFixtureError",
    }
)
_WRAPPER_EXCEPTION_MESSAGES = (
    re.compile(r"Engine core initialization failed", re.IGNORECASE),
    re.compile(r"API exited before health", re.IGNORECASE),
)


class AssuranceRunError(RuntimeError):
    """The one-shot run cannot satisfy its identity or lifecycle contract."""


@dataclass(frozen=True)
class FileBinding:
    path: Path
    sha256: str


@dataclass(frozen=True)
class RequestSpec:
    base_url: str
    fixture_dir: Path
    output: Path
    request_id: str
    max_tokens: int
    timeout: float
    cache_session_id: str
    cache_branch: str
    cache_manifest_sha256: str


@dataclass(frozen=True)
class RunSpec:
    artifact_mode: str
    qualification_dir: Path
    command: FileBinding
    runner: FileBinding
    bindings: dict[str, FileBinding]
    request: RequestSpec


@dataclass(frozen=True)
class OffloadReapResult:
    count: int
    bytes: int
    names: tuple[str, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_dict(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise AssuranceRunError(f"{label} keys must be exactly {sorted(keys)}")
    return value


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AssuranceRunError(f"{label} must be a lowercase SHA-256")
    return value


def _absolute_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.startswith("/"):
        raise AssuranceRunError(f"{label} must be an absolute path")
    return Path(value)


def _binding(value: object, label: str) -> FileBinding:
    row = _exact_dict(value, {"path", "sha256"}, label)
    return FileBinding(
        path=_absolute_path(row["path"], f"{label}.path"),
        sha256=_sha256(row["sha256"], f"{label}.sha256"),
    )


def _inside(path: Path, parent: Path, label: str) -> None:
    try:
        path.relative_to(parent)
    except ValueError as error:
        raise AssuranceRunError(f"{label} must be inside the qualification directory") from error


def load_spec(path: Path) -> RunSpec:
    if path.is_symlink() or not path.is_file():
        raise AssuranceRunError("spec must be a regular non-symlink file")
    try:
        root = _exact_dict(
            json.loads(path.read_text(encoding="utf-8")),
            {
                "artifact_mode",
                "bindings",
                "command",
                "qualification_dir",
                "request",
                "runner",
                "schema",
            },
            "spec",
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssuranceRunError(f"spec is not valid UTF-8 JSON: {error}") from error
    if root["schema"] != SCHEMA:
        raise AssuranceRunError("spec has the wrong schema")
    artifact_mode = root["artifact_mode"]
    if artifact_mode not in {"assurance", "release"}:
        raise AssuranceRunError("artifact_mode must be assurance or release")

    qualification_dir = _absolute_path(root["qualification_dir"], "qualification_dir")
    command = _binding(root["command"], "command")
    runner = _binding(root["runner"], "runner")
    raw_bindings = root["bindings"]
    allowed_bindings = (
        {ASSURANCE_BINDINGS, FULL_INSTRUMENTATION_BINDINGS}
        if artifact_mode == "assurance"
        else {RELEASE_BINDINGS}
    )
    if not isinstance(raw_bindings, dict) or set(raw_bindings) not in allowed_bindings:
        raise AssuranceRunError(
            "bindings must be dependency-closed and exactly one of "
            + repr([sorted(binding_set) for binding_set in allowed_bindings])
        )
    bindings = {name: _binding(value, f"bindings.{name}") for name, value in raw_bindings.items()}

    request = _exact_dict(
        root["request"],
        {
            "base_url",
            "cache_branch",
            "cache_manifest_sha256",
            "cache_session_id",
            "fixture_dir",
            "max_tokens",
            "output",
            "request_id",
            "timeout",
        },
        "request",
    )
    base_url = request["base_url"]
    if base_url != "http://127.0.0.1:8000":
        raise AssuranceRunError("request.base_url must be the pinned loopback endpoint")
    request_id = request["request_id"]
    if (
        not isinstance(request_id, str)
        or not 1 <= len(request_id) <= 128
        or any(character.isspace() for character in request_id)
    ):
        raise AssuranceRunError("request.request_id must be 1..128 non-whitespace characters")
    max_tokens = request["max_tokens"]
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or not 1 <= max_tokens <= MAX_COMPLETION_TOKENS
    ):
        raise AssuranceRunError(
            f"request.max_tokens must be an integer in 1..{MAX_COMPLETION_TOKENS}"
        )
    timeout = request["timeout"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise AssuranceRunError("request.timeout must be positive")
    for key in ("cache_session_id", "cache_branch"):
        value = request[key]
        if (
            not isinstance(value, str)
            or not 1 <= len(value) <= 128
            or any(character.isspace() for character in value)
        ):
            raise AssuranceRunError(f"request.{key} must be 1..128 non-whitespace characters")

    output = _absolute_path(request["output"], "request.output")
    _inside(command.path, qualification_dir, "command.path")
    _inside(output, qualification_dir, "request.output")
    return RunSpec(
        artifact_mode=artifact_mode,
        qualification_dir=qualification_dir,
        command=command,
        runner=runner,
        bindings=bindings,
        request=RequestSpec(
            base_url=base_url,
            fixture_dir=_absolute_path(request["fixture_dir"], "request.fixture_dir"),
            output=output,
            request_id=request_id,
            max_tokens=max_tokens,
            timeout=float(timeout),
            cache_session_id=request["cache_session_id"],
            cache_branch=request["cache_branch"],
            cache_manifest_sha256=_sha256(
                request["cache_manifest_sha256"], "request.cache_manifest_sha256"
            ),
        ),
    )


def _require_owned_regular(binding: FileBinding, label: str) -> None:
    path = binding.path
    if path.is_symlink() or not path.is_file():
        raise AssuranceRunError(f"{label} is missing or not a regular file: {path}")
    if path.stat().st_uid != os.getuid():
        raise AssuranceRunError(f"{label} has the wrong owner: {path}")
    observed = _sha256_file(path)
    if observed != binding.sha256:
        raise AssuranceRunError(
            f"{label} hash mismatch: expected {binding.sha256}, observed {observed}"
        )


def _require_qualification_dir(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise AssuranceRunError("qualification directory must be a real directory")
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise AssuranceRunError("qualification directory must be owned by the caller and mode 0700")
    return path.resolve(strict=True)


def _materialize_empty_output_directory(parent: Path, name: str, label: str) -> Path:
    """Create or authenticate one empty, caller-owned output directory.

    Empty directories are not a portable deployment artifact: archive and copy
    transports may omit them.  Materialize the directory from its authenticated
    one-shot contract before model startup, using directory-relative operations
    so a substituted symlink cannot redirect capture output.
    """

    if not name or Path(name).name != name:
        raise AssuranceRunError(f"{label} has an invalid child name")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        parent_descriptor = os.open(parent, flags)
    except OSError as error:
        raise AssuranceRunError(f"{label} parent is not a real directory: {parent}") from error
    try:
        parent_metadata = os.fstat(parent_descriptor)
        if parent_metadata.st_uid != os.getuid() or stat.S_IMODE(parent_metadata.st_mode) != 0o700:
            raise AssuranceRunError(f"{label} parent must be owned by the caller and mode 0700")
        with contextlib.suppress(FileExistsError):
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        try:
            directory_descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        except OSError as error:
            raise AssuranceRunError(f"{label} is not a real directory") from error
        try:
            metadata = os.fstat(directory_descriptor)
            if metadata.st_uid != os.getuid():
                raise AssuranceRunError(f"{label} must be owned by the caller")
            os.fchmod(directory_descriptor, 0o700)
            if stat.S_IMODE(os.fstat(directory_descriptor).st_mode) != 0o700:
                raise AssuranceRunError(f"{label} could not be restricted to mode 0700")
            if os.listdir(directory_descriptor):  # noqa: PTH208 - keep lookup fd-relative
                raise AssuranceRunError(f"{label} must be empty before the run")
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return parent / name


def _launcher_authenticates_snapshot_selection(
    bindings: dict[str, FileBinding], launcher_text: str, expected: str
) -> bool:
    """Accept a literal pin or the launcher's receipt-derived exact pin."""

    if expected in launcher_text:
        return True
    receipt_binding = bindings.get("state_consumer_transition_receipt")
    if receipt_binding is None:
        return False
    try:
        receipt = json.loads(receipt_binding.path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(receipt, dict) or receipt.get("consumer_snapshot_format_sha256") != expected:
        return False
    return all(
        marker in launcher_text
        for marker in (
            "QWEN_CODING_TURBO_STATE_CONSUMER_TRANSITION_RECEIPT",
            ".consumer_snapshot_format_sha256",
            "EXPECTED_DFLASH_PERSISTENT_KV_SELECTION_RUNTIME_SHA256=$selection_runtime_sha256",
        )
    )


def _require_cross_bindings(bindings: dict[str, FileBinding]) -> None:
    if "capture_hook" in bindings:
        capture_text = bindings["capture_hook"].path.read_text(encoding="utf-8")
        for name in ("qwen3_next", "qwen_text_rope"):
            expected = bindings[name].sha256
            if expected not in capture_text:
                raise AssuranceRunError(
                    f"capture_hook does not authenticate the selected {name} binding"
                )

    launcher_text = bindings["launcher"].path.read_text(encoding="utf-8")
    for name in sorted(RUNTIME_BINDINGS & bindings.keys()):
        if name in {"capture_hook", "capture_site", "launcher"}:
            continue
        expected = bindings[name].sha256
        authenticated = expected in launcher_text
        if name == "snapshot_selection":
            authenticated = _launcher_authenticates_snapshot_selection(
                bindings, launcher_text, expected
            )
        if not authenticated:
            raise AssuranceRunError(f"launcher does not authenticate the selected {name} binding")

    gdn_binding = bindings["qwen_gdn_linear_attn"]
    selection_text = bindings["snapshot_selection"].path.read_text(encoding="utf-8")
    if gdn_binding.sha256 not in selection_text:
        raise AssuranceRunError(
            "snapshot selection does not authorize the selected qwen_gdn_linear_attn binding"
        )


def _artifact_contract_member(spec: RunSpec, output: str, label: str) -> tuple[Path, str]:
    """Resolve and authenticate one runtime contract from the selected artifact."""

    if spec.artifact_mode == "assurance":
        manifest_path = spec.bindings["assurance_manifest"].path
    else:
        manifest_path = (
            spec.qualification_dir / "artifact-pair" / spec.artifact_mode / "manifest.json"
        )
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise AssuranceRunError(f"{label} artifact manifest is missing: {manifest_path}")
    if manifest_path.stat().st_uid != os.getuid():
        raise AssuranceRunError(f"{label} artifact manifest has the wrong owner: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssuranceRunError(f"{label} artifact manifest is invalid: {error}") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != ARTIFACT_SCHEMA
        or manifest.get("artifact_kind") != spec.artifact_mode
    ):
        raise AssuranceRunError(f"{label} artifact manifest identity is invalid")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise AssuranceRunError(f"{label} artifact manifest file table is invalid")
    matching = [
        entry for entry in entries if isinstance(entry, dict) and entry.get("path") == output
    ]
    if len(matching) != 1:
        raise AssuranceRunError(
            f"{label} artifact manifest must contain exactly one {output!r} member"
        )
    expected = _sha256(matching[0].get("sha256"), f"{label} artifact member SHA-256")
    files_root = (manifest_path.parent / "files").resolve(strict=True)
    member = (files_root / output).resolve(strict=True)
    _inside(member, files_root, label)
    binding = FileBinding(path=member, sha256=expected)
    _require_owned_regular(binding, label)
    return member, expected


def _artifact_semantic_contract(spec: RunSpec) -> str:
    """Read the exact semantic contract from the selected artifact manifest."""

    if spec.artifact_mode == "assurance":
        manifest_path = spec.bindings["assurance_manifest"].path
    else:
        manifest_path = (
            spec.qualification_dir / "artifact-pair" / spec.artifact_mode / "manifest.json"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssuranceRunError(
            f"cannot read selected artifact semantic contract: {error}"
        ) from error
    contract = manifest.get("semantic_contract") if isinstance(manifest, dict) else None
    if (
        not isinstance(contract, str)
        or not contract
        or any(not part for part in contract.split("/"))
    ):
        raise AssuranceRunError("selected artifact semantic contract is invalid")
    return contract


def _require_exact_k_preflight(
    spec: RunSpec,
    environment: dict[str, str],
) -> bool:
    """Bind the candidate M8 kernel to source, build, qualification, and runtime.

    The serial M1 arm deliberately does not load the candidate.  Every M8 arm
    carrying the exact-K semantic contract must load the authenticated extension.
    An older semantic contract may not opt into this runtime path.
    """

    contract = _artifact_semantic_contract(spec)
    bound = EXACT_K_CONTRACT_SEGMENT in contract.split("/")
    present = EXACT_K_ENVIRONMENT.intersection(environment)
    target_only = environment.get("QWEN_FIXED_SLOT_TARGET_ONLY_ORACLE", "0") == "1"
    if not bound:
        if present:
            raise AssuranceRunError(
                "exact-K runtime is enabled without the matching semantic-source contract"
            )
        return False
    if target_only:
        if present:
            raise AssuranceRunError(
                "serial M1 oracle must not load the candidate exact-K runtime"
            )
        return False
    if present != EXACT_K_ENVIRONMENT:
        missing = sorted(EXACT_K_ENVIRONMENT - present)
        raise AssuranceRunError(
            "speculative M8 exact-K runtime environment is incomplete: " + ", ".join(missing)
        )
    if (
        environment["QWEN_FULL_ATTENTION_M8_EXACT_K"] != "1"
        or environment["QWEN_FULL_ATTENTION_M8_EXACT_K_REQUIRED"] != "1"
    ):
        raise AssuranceRunError("speculative M8 exact-K runtime is not mandatory")

    capability_path, capability_sha256 = _artifact_contract_member(
        spec, EXACT_K_CAPABILITY_OUTPUT, "exact-K capability"
    )
    try:
        capability = json.loads(capability_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssuranceRunError(f"exact-K capability is invalid: {error}") from error
    if not isinstance(capability, dict):
        raise AssuranceRunError("exact-K capability must be a JSON object")
    mismatches: list[str] = []

    def check(condition: bool, label: str) -> None:
        if not condition:
            mismatches.append(label)

    check(capability.get("schema") == EXACT_K_CAPABILITY_SCHEMA, "capability.schema")
    check(capability.get("capability") == EXACT_K_CAPABILITY, "capability.identity")
    check(capability.get("universal_lossless_claim") is False, "capability.scope")
    qualification = capability.get("qualification")
    runtime = capability.get("runtime")
    sources = capability.get("source_sha256")
    qualification = qualification if isinstance(qualification, dict) else {}
    runtime = runtime if isinstance(runtime, dict) else {}
    sources = sources if isinstance(sources, dict) else {}
    check(
        qualification.get("scope") == "bounded-component-qualification-not-universal-proof",
        "qualification.scope",
    )
    check(
        runtime.get("qwen3_next_preimage_sha256") == spec.bindings["qwen3_next"].sha256,
        "runtime.qwen3_next_preimage_sha256",
    )
    for name, output in EXACT_K_SOURCE_OUTPUTS.items():
        _member, member_sha256 = _artifact_contract_member(
            spec, output, f"exact-K source {name}"
        )
        check(sources.get(name) == member_sha256, f"source_sha256.{name}")

    extension_sha256 = qualification.get("extension_sha256")
    qualification_sha256 = qualification.get("qualification_file_sha256")
    build_manifest_sha256 = qualification.get("build_manifest_sha256")
    check(
        environment["QWEN_FULL_ATTENTION_M8_EXACT_K_RUNTIME_SHA256"]
        == runtime.get("qwen3_next_preimage_sha256"),
        "environment.runtime_sha256",
    )
    check(
        environment["QWEN_FULL_ATTENTION_M8_EXACT_K_EXTENSION_SHA256"]
        == extension_sha256,
        "environment.extension_sha256",
    )
    check(
        environment["QWEN_FULL_ATTENTION_M8_EXACT_K_QUALIFICATION_SHA256"]
        == qualification_sha256,
        "environment.qualification_sha256",
    )
    check(
        re.fullmatch(
            r"[0-9a-f]{64}",
            environment["QWEN_FULL_ATTENTION_M8_EXACT_K_SITE_SHA256"],
        )
        is not None,
        "environment.site_sha256",
    )

    pythonpath = environment.get("PYTHONPATH", "")
    outer_root = Path(pythonpath.split(":", 1)[0]) if pythonpath else Path()
    site_path = outer_root / "sitecustomize.py"
    try:
        _require_owned_regular(
            FileBinding(
                site_path,
                environment["QWEN_FULL_ATTENTION_M8_EXACT_K_SITE_SHA256"],
            ),
            "exact-K runtime bootstrap",
        )
    except AssuranceRunError as error:
        mismatches.append(str(error))

    extension_path = Path(environment["QWEN_FULL_ATTENTION_M8_EXACT_K_EXTENSION"])
    if not isinstance(extension_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", extension_sha256):
        mismatches.append("qualification.extension_sha256")
    else:
        try:
            _require_owned_regular(
                FileBinding(extension_path, extension_sha256), "exact-K runtime extension"
            )
        except AssuranceRunError as error:
            mismatches.append(str(error))

    qualification_path = extension_path.parent.parent / "qualification.json"
    if not isinstance(qualification_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", qualification_sha256
    ):
        mismatches.append("qualification.qualification_file_sha256")
    else:
        try:
            _require_owned_regular(
                FileBinding(qualification_path, qualification_sha256),
                "exact-K qualification evidence",
            )
            qualification_document = json.loads(
                qualification_path.read_text(encoding="utf-8")
            )
            if not isinstance(qualification_document, dict):
                raise AssuranceRunError("exact-K qualification evidence is not an object")
            check(qualification_document.get("verdict") == "pass", "evidence.verdict")
            check(
                qualification_document.get("scope")
                == "bounded-component-qualification-not-universal-proof",
                "evidence.scope",
            )
            check(
                qualification_document.get("library_sha256") == extension_sha256,
                "evidence.library_sha256",
            )
            build_record = qualification_document.get("build_manifest")
            build_record = build_record if isinstance(build_record, dict) else {}
            check(
                build_record.get("sha256") == build_manifest_sha256,
                "evidence.build_manifest_sha256",
            )
            build_path = Path(str(build_record.get("path", "")))
            if isinstance(build_manifest_sha256, str) and re.fullmatch(
                r"[0-9a-f]{64}", build_manifest_sha256
            ):
                _require_owned_regular(
                    FileBinding(build_path, build_manifest_sha256),
                    "exact-K build manifest",
                )
            else:
                mismatches.append("qualification.build_manifest_sha256")
        except (AssuranceRunError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            mismatches.append(str(error))
    if mismatches:
        raise AssuranceRunError(
            "exact-K semantic/runtime preflight mismatches: " + ", ".join(mismatches)
        )
    del capability_sha256
    return True


def _launcher_literal_sha256(launcher_text: str, name: str) -> str | None:
    matches = re.findall(rf"^readonly {re.escape(name)}='([0-9a-f]{{64}})'$", launcher_text, re.M)
    return matches[0] if len(matches) == 1 else None


def _launcher_flag_literal_sha256(
    launcher_text: str, name: str, flag: str, *, enabled: bool
) -> str | None:
    pattern = (
        rf"^if \[\[ \$\{{{re.escape(flag)}:-0\}} == 1 \]\]; then\n"
        rf"^[ \t]+readonly {re.escape(name)}='([0-9a-f]{{64}})'\n"
        r"^else\n"
        rf"^[ \t]+readonly {re.escape(name)}='([0-9a-f]{{64}})'\n"
        r"^fi$"
    )
    matches = re.findall(pattern, launcher_text, re.M)
    if len(matches) != 1:
        return None
    return matches[0][0 if enabled else 1]


def _normalized_kernel_ast_sha256(path: Path, function_name: str) -> str:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise AssuranceRunError(f"cannot parse fixed serial M8 semantic source: {error}") from error
    functions = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    ]
    if len(functions) != 1:
        raise AssuranceRunError(
            f"fixed serial M8 semantic source must contain exactly one {function_name}"
        )
    normalized = copy.deepcopy(functions[0])
    normalized.name = "fixed_serial_conv_m8_kernel"
    if (
        normalized.body
        and isinstance(normalized.body[0], ast.Expr)
        and isinstance(normalized.body[0].value, ast.Constant)
        and isinstance(normalized.body[0].value.value, str)
    ):
        normalized.body.pop(0)
    # Python 3.13 added ``show_empty`` and changed ast.dump's default to omit
    # empty optional fields.  The component certificate is produced by the
    # model runtime's Python 3.12 interpreter, which includes those fields.
    # Request them explicitly when supported so the structural digest is
    # stable when this controller runs under a newer system Python.
    try:
        normalized_dump = ast.dump(
            normalized,
            include_attributes=False,
            show_empty=True,
        )
    except TypeError:  # Python <= 3.12 includes empty fields by default.
        normalized_dump = ast.dump(normalized, include_attributes=False)
    payload = normalized_dump.encode()
    return hashlib.sha256(payload).hexdigest()


def _semantic_conv_bias_mode(path: Path) -> str:
    """Read the actual authenticated model constructor rather than assuming config defaults."""

    try:
        module = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise AssuranceRunError(f"cannot parse GDN model constructor: {error}") from error
    classes = [
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "QwenGatedDeltaNetAttention"
    ]
    if len(classes) != 1:
        raise AssuranceRunError("GDN semantic source has no unique Qwen constructor")
    constructors = [
        node
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__"
    ]
    if len(constructors) != 1:
        raise AssuranceRunError("GDN semantic source has no unique constructor")
    bias_values: list[object] = []
    for node in ast.walk(constructors[0]):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if not any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "conv1d"
            for target in node.targets
        ):
            continue
        function = node.value.func
        function_name = (
            function.id
            if isinstance(function, ast.Name)
            else function.attr
            if isinstance(function, ast.Attribute)
            else None
        )
        if function_name != "ColumnParallelLinear":
            continue
        bias_keywords = [keyword.value for keyword in node.value.keywords if keyword.arg == "bias"]
        if (
            len(bias_keywords) == 1
            and isinstance(bias_keywords[0], ast.Constant)
            and type(bias_keywords[0].value) is bool
        ):
            bias_values.append(bias_keywords[0].value)
        else:
            bias_values.append(object())
    if bias_values == [False]:
        return "none"
    if bias_values == [True]:
        return "present"
    raise AssuranceRunError("GDN convolution bias geometry is not a unique literal boolean")


def _dflash_speculative_tokens(command_text: str) -> int:
    """Read the exact DFlash width that the authenticated command will launch."""

    try:
        arguments = shlex.split(command_text)
    except ValueError as error:
        raise AssuranceRunError(f"command quoting is invalid: {error}") from error
    positions = [
        index for index, argument in enumerate(arguments) if argument == "--speculative-config"
    ]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise AssuranceRunError(
            "fixed serial M8 requires exactly one complete --speculative-config"
        )
    try:
        config = json.loads(arguments[positions[0] + 1])
    except json.JSONDecodeError as error:
        raise AssuranceRunError(f"speculative config is invalid JSON: {error}") from error
    if not isinstance(config, dict) or config.get("method") != "dflash":
        raise AssuranceRunError("fixed serial M8 requires method=dflash")
    tokens = config.get("num_speculative_tokens")
    if type(tokens) is not int or tokens < 1:
        raise AssuranceRunError(
            "fixed serial M8 requires a positive integer num_speculative_tokens"
        )
    return tokens


def _fixed_serial_conv_contract_mismatches(
    *,
    capability: dict[str, Any],
    certificate: dict[str, Any],
    model_config: dict[str, Any],
    model_config_sha256: str,
    component_certificate_sha256: str,
    project_semantic_sha256: str | None,
    runtime_semantic_sha256: str,
    artifact_semantic_sha256: str,
    runtime_kernel_ast_sha256: str,
    runtime_bias_mode: str,
    runtime_speculative_tokens: int,
) -> list[str]:
    """Return every incompatibility so one preflight exposes the whole defect set."""

    mismatches: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            mismatches.append(message)

    target = capability.get("target_model")
    kernel = capability.get("kernel_geometry")
    production_abi = capability.get("production_abi")
    sources = capability.get("source_bindings")
    component = capability.get("component_certificate")
    check(capability.get("schema") == FIXED_SERIAL_CONV_CAPABILITY_SCHEMA, "capability.schema")
    check(capability.get("capability") == FIXED_SERIAL_CONV_CAPABILITY, "capability.identity")
    check(capability.get("default_off") is True, "capability.default_off")
    check(
        capability.get("enable_environment") == FIXED_SERIAL_CONV_ENABLE_ENVIRONMENT,
        "capability.enable_environment",
    )
    check(isinstance(target, dict), "capability.target_model")
    check(isinstance(kernel, dict), "capability.kernel_geometry")
    check(isinstance(production_abi, dict), "capability.production_abi")
    check(isinstance(sources, dict), "capability.source_bindings")
    check(isinstance(component, dict), "capability.component_certificate")
    target = target if isinstance(target, dict) else {}
    kernel = kernel if isinstance(kernel, dict) else {}
    production_abi = production_abi if isinstance(production_abi, dict) else {}
    sources = sources if isinstance(sources, dict) else {}
    component = component if isinstance(component, dict) else {}

    expected_kernel = {
        "activation": "silu",
        "bias_modes": ["none", "present"],
        "conv_width": 4,
        "dtype": "bfloat16",
        "logical_state_length": 3,
        "mixed_qkv_width": 10240,
        "physical_state_length": 10,
        "rows": 8,
        "speculative_tokens": 7,
    }
    check(kernel == expected_kernel, "capability.kernel_geometry")
    expected_certificate_abi = {
        "checkpoint_shape": [8, 10240, 10],
        "checkpoint_stride": [102400, 10, 1],
        "state_shape": [1, 10240, 10],
        "state_stride": [1687552, 1, 10240],
    }
    expected_capability_abi = {
        "checkpoints_shape": [8, 10240, 10],
        "checkpoints_stride": [102400, 10, 1],
        "component_field_names": {
            "checkpoints_shape": "checkpoint_shape",
            "checkpoints_stride": "checkpoint_stride",
        },
        "state_shape": [1, 10240, 10],
        "state_stride": [1687552, 1, 10240],
    }
    check(production_abi == expected_capability_abi, "capability.production_abi")
    check(runtime_speculative_tokens == 7, "runtime.num_speculative_tokens")
    check(
        kernel.get("rows") == runtime_speculative_tokens + 1,
        "runtime.verification_rows",
    )
    check(
        kernel.get("physical_state_length") == 4 - 1 + runtime_speculative_tokens,
        "runtime.physical_state_formula",
    )
    check(target.get("config_sha256") == model_config_sha256, "target_model.config_sha256")
    check(target.get("model_type") == "qwen3_5", "target_model.model_type")
    check(
        target.get("architecture") == "Qwen3_5ForConditionalGeneration",
        "target_model.architecture",
    )
    check(
        target.get("conv_bias_mode") == runtime_bias_mode == "none",
        "target_model.conv_bias_mode",
    )
    expected_text_geometry = {
        "activation": "silu",
        "attention_layers": 16,
        "conv_width": 4,
        "dtype": "bfloat16",
        "key_dim": 128,
        "key_heads": 16,
        "linear_attention_layers": 48,
        "value_dim": 128,
        "value_heads": 48,
    }
    check(target.get("text_geometry") == expected_text_geometry, "target_model.text_geometry")

    text_config = model_config.get("text_config")
    text_config = text_config if isinstance(text_config, dict) else {}
    layers = text_config.get("layer_types")
    layers = layers if isinstance(layers, list) else []
    observed_model = {
        "activation": text_config.get("hidden_act"),
        "attention_layers": layers.count("full_attention"),
        "conv_width": text_config.get("linear_conv_kernel_dim"),
        "dtype": text_config.get("dtype"),
        "key_dim": text_config.get("linear_key_head_dim"),
        "key_heads": text_config.get("linear_num_key_heads"),
        "linear_attention_layers": layers.count("linear_attention"),
        "value_dim": text_config.get("linear_value_head_dim"),
        "value_heads": text_config.get("linear_num_value_heads"),
    }
    check(model_config.get("model_type") == "qwen3_5", "model_config.model_type")
    check(
        model_config.get("architectures") == ["Qwen3_5ForConditionalGeneration"],
        "model_config.architectures",
    )
    check(observed_model == expected_text_geometry, "model_config.text_geometry")

    check(
        component.get("schema") == FIXED_SERIAL_CONV_CERTIFICATE_SCHEMA,
        "component_certificate.schema",
    )
    check(
        component.get("sha256") == component_certificate_sha256,
        "component_certificate.sha256",
    )
    check(
        component.get("artifact_output") == FIXED_SERIAL_CONV_CERTIFICATE_OUTPUT,
        "component_certificate.artifact_output",
    )
    check(
        component.get("path") == FIXED_SERIAL_CONV_CERTIFICATE_SOURCE,
        "component_certificate.path",
    )
    check(
        sources.get("semantic_production_source_sha256") == project_semantic_sha256,
        "source_bindings.semantic_production_source_sha256",
    )

    check(certificate.get("schema") == FIXED_SERIAL_CONV_CERTIFICATE_SCHEMA, "certificate.schema")
    check(certificate.get("passed") is True, "certificate.passed")
    qualified = certificate.get("qualified_geometry")
    expected_qualified = {
        **{key: value for key, value in expected_kernel.items() if key != "speculative_tokens"},
        "deployed_model_bias_mode": "none",
    }
    check(qualified == expected_qualified, "certificate.qualified_geometry")
    check(
        certificate.get("production_abi") == expected_certificate_abi,
        "certificate.production_abi",
    )
    certificate_sources = certificate.get("source_bindings")
    certificate_sources = certificate_sources if isinstance(certificate_sources, dict) else {}
    for field in (
        "component_source_sha256",
        "semantic_production_source_sha256",
        "causal_conv1d_reference_source_sha256",
        "component_normalized_kernel_ast_sha256",
        "semantic_normalized_kernel_ast_sha256",
    ):
        check(certificate_sources.get(field) == sources.get(field), f"certificate.{field}")
    check(
        certificate_sources.get("normalized_kernel_ast_identical") is True,
        "certificate.normalized_kernel_ast_identical",
    )
    expected_runtime_kernel_ast_sha256 = certificate_sources.get(
        "semantic_normalized_kernel_ast_sha256"
    )
    check(
        expected_runtime_kernel_ast_sha256 == runtime_kernel_ast_sha256,
        "runtime.normalized_kernel_ast_sha256"
        f"(expected={expected_runtime_kernel_ast_sha256!r}, "
        f"observed={runtime_kernel_ast_sha256!r})",
    )
    check(runtime_semantic_sha256 == artifact_semantic_sha256, "runtime.artifact_gdn_sha256")

    cases = certificate.get("cases")
    check(isinstance(cases, list), "certificate.cases")
    cases = cases if isinstance(cases, list) else []
    expected_cases = {
        (bias_mode, scenario, seed)
        for bias_mode in ("none", "present")
        for scenario in ("random", "zeros", "alternating")
        for seed in range(4)
    }
    observed_cases: set[tuple[object, object, object]] = set()
    invalid_results = False
    for case in cases:
        if not isinstance(case, dict):
            invalid_results = True
            continue
        observed_cases.add((case.get("bias_mode"), case.get("scenario"), case.get("seed")))
        if (
            case.get("output_bit_mismatches") != 0
            or case.get("final_state_bit_mismatches") != 0
            or case.get("final_state_tail_preservation_bit_mismatches") != 0
            or case.get("state_prefix_canary_bit_mismatches") != 0
            or case.get("state_suffix_canary_bit_mismatches") != 0
            or case.get("checkpoint_bit_mismatches") != [0] * 8
            or case.get("checkpoint_tail_preservation_bit_mismatches") != [0] * 8
            or case.get("accepted_prefix_bit_mismatches") != [0] * 9
        ):
            invalid_results = True
    check(
        observed_cases == expected_cases and len(cases) == len(expected_cases),
        "certificate.case_matrix",
    )
    check(not invalid_results, "certificate.bit_exact_results")
    return mismatches


def _require_fixed_serial_conv_preflight(
    spec: RunSpec,
    environment: dict[str, str],
    launcher_text: str,
    *,
    target_model_config: Path = TARGET_MODEL_CONFIG,
    target_model_config_sha256: str | None = None,
) -> None:
    enabled = environment.get("QWEN_GDN_FIXED_SLOT_SERIAL_CONV_M8", "0")
    if enabled not in {"0", "1"}:
        raise AssuranceRunError("QWEN_GDN_FIXED_SLOT_SERIAL_CONV_M8 must be 0 or 1")
    if enabled == "0":
        return

    capability_path, capability_sha256 = _artifact_contract_member(
        spec, FIXED_SERIAL_CONV_CAPABILITY_OUTPUT, "fixed serial M8 capability"
    )
    certificate_path, certificate_sha256 = _artifact_contract_member(
        spec, FIXED_SERIAL_CONV_CERTIFICATE_OUTPUT, "fixed serial M8 component certificate"
    )
    try:
        capability = json.loads(capability_path.read_text(encoding="utf-8"))
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        model_config = json.loads(target_model_config.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssuranceRunError(f"fixed serial M8 preflight input is invalid: {error}") from error
    if not all(isinstance(document, dict) for document in (capability, certificate, model_config)):
        raise AssuranceRunError("fixed serial M8 preflight documents must be JSON objects")

    launcher_capability_sha = _launcher_literal_sha256(
        launcher_text, "EXPECTED_GDN_FIXED_SERIAL_CONV_M8_CAPABILITY_SHA256"
    )
    launcher_certificate_sha = _launcher_literal_sha256(
        launcher_text, "EXPECTED_GDN_FIXED_SERIAL_CONV_M8_CERTIFICATE_SHA256"
    )
    launcher_model_sha = _launcher_literal_sha256(launcher_text, "EXPECTED_MODEL_CONFIG_SHA256")
    launcher_certificate_semantic_sha = _launcher_literal_sha256(
        launcher_text, "EXPECTED_GDN_FIXED_SERIAL_CONV_M8_CERTIFIED_SEMANTIC_SHA256"
    )
    runtime_semantic = spec.bindings["qwen_gdn_linear_attn"]
    artifact_gdn_path, artifact_gdn_sha = _artifact_contract_member(
        spec,
        "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        "fixed serial M8 artifact GDN source",
    )
    del artifact_gdn_path
    mismatches = []
    if launcher_capability_sha != capability_sha256:
        mismatches.append("launcher.capability_sha256")
    if launcher_certificate_sha != certificate_sha256:
        mismatches.append("launcher.certificate_sha256")
    observed_model_sha = _sha256_file(target_model_config)
    if target_model_config_sha256 is None:
        if launcher_model_sha != observed_model_sha:
            mismatches.append("launcher.model_config_sha256")
    elif target_model_config_sha256 != observed_model_sha:
        mismatches.append("snapshot_selection.model_config_sha256")
    try:
        runtime_bias_mode = _semantic_conv_bias_mode(runtime_semantic.path)
        runtime_kernel_sha = _normalized_kernel_ast_sha256(
            runtime_semantic.path, "_qwen_fixed_serial_conv_m8_kernel"
        )
    except AssuranceRunError as error:
        mismatches.append(str(error))
        runtime_bias_mode = "invalid"
        runtime_kernel_sha = ""
    try:
        runtime_speculative_tokens = _dflash_speculative_tokens(
            spec.command.path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, AssuranceRunError) as error:
        mismatches.append(str(error))
        runtime_speculative_tokens = -1
    mismatches.extend(
        _fixed_serial_conv_contract_mismatches(
            capability=capability,
            certificate=certificate,
            model_config=model_config,
            model_config_sha256=observed_model_sha,
            component_certificate_sha256=certificate_sha256,
            project_semantic_sha256=launcher_certificate_semantic_sha,
            runtime_semantic_sha256=runtime_semantic.sha256,
            artifact_semantic_sha256=artifact_gdn_sha,
            runtime_kernel_ast_sha256=runtime_kernel_sha,
            runtime_bias_mode=runtime_bias_mode,
            runtime_speculative_tokens=runtime_speculative_tokens,
        )
    )
    if mismatches:
        raise AssuranceRunError(
            "fixed serial M8 preflight rejected incompatible geometry/certificate; mismatches: "
            + ", ".join(dict.fromkeys(mismatches))
        )


def _require_quest_selector_extension_preflight(
    spec: RunSpec, environment: dict[str, str], launcher_text: str
) -> None:
    """Bind every cached-selector run to one live, component-qualified extension."""

    cached = environment.get("QWEN_QUEST_CACHED_GEMM_SELECTOR", "0")
    cold_seed = environment.get("QWEN_QUEST_CACHED_GEMM_COLD_SEED", "0")
    if cached not in {"0", "1"}:
        raise AssuranceRunError("QWEN_QUEST_CACHED_GEMM_SELECTOR must be 0 or 1")
    if cold_seed not in {"0", "1"}:
        raise AssuranceRunError("QWEN_QUEST_CACHED_GEMM_COLD_SEED must be 0 or 1")
    if cached == "0":
        if cold_seed == "1":
            raise AssuranceRunError("cached-GEMM cold seed requires the cached selector")
        return

    raw_path = environment.get("QWEN_QUEST_SELECTOR_EXTENSION")
    raw_sha256 = environment.get("QWEN_QUEST_SELECTOR_EXTENSION_SHA256")
    if not isinstance(raw_path, str) or not raw_path.startswith("/"):
        raise AssuranceRunError("cached selector requires an absolute Quest extension path")
    expected_sha256 = _sha256(raw_sha256, "Quest selector extension SHA-256")
    extension = FileBinding(path=Path(raw_path), sha256=expected_sha256)
    _require_owned_regular(extension, "Quest selector extension")

    raw_dualphase_path = environment.get("QWEN_QUEST_M8_ROW_LOCAL_DUALPHASE_EXTENSION")
    raw_dualphase_sha256 = environment.get("QWEN_QUEST_M8_ROW_LOCAL_DUALPHASE_EXTENSION_SHA256")
    strict_combined_row_local = (
        environment.get("QWEN_QUEST_M8_ROW_LOCAL") == "1"
        and environment.get("QWEN_DFLASH2_BEST_FIRST_B7_REQUIRED") == "1"
        and environment.get("QWEN_UNIQUE_DEPTH_MIXED_REQUIRED") == "1"
    )
    if strict_combined_row_local:
        if raw_dualphase_path != str(REQUIRED_ROW_LOCAL_DUALPHASE_EXTENSION):
            raise AssuranceRunError(
                "combined row-local M8 requires the exact qualified dual-phase extension path"
            )
        if raw_dualphase_sha256 != REQUIRED_ROW_LOCAL_DUALPHASE_EXTENSION_SHA256:
            raise AssuranceRunError(
                "combined row-local M8 requires the exact qualified dual-phase extension SHA-256"
            )
    if raw_dualphase_path is not None or raw_dualphase_sha256 is not None:
        if not isinstance(raw_dualphase_path, str) or not raw_dualphase_path.startswith("/"):
            raise AssuranceRunError(
                "row-local dual-phase attention requires an absolute extension path"
            )
        expected_dualphase_sha256 = _sha256(
            raw_dualphase_sha256,
            "row-local dual-phase extension SHA-256",
        )
        _require_owned_regular(
            FileBinding(
                path=Path(raw_dualphase_path),
                sha256=expected_dualphase_sha256,
            ),
            "row-local dual-phase extension",
        )

    mismatches: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            mismatches.append(message)

    launcher_extension_literal = (
        "EXPECTED_QUEST_TREE_WMMA_EXTENSION_SHA256"
        if environment.get("QWEN_DFLASH2_BEST_FIRST_B7_REQUIRED") == "1"
        else "EXPECTED_QUEST_SELECTOR_EXTENSION_SHA256"
    )
    check(
        _launcher_literal_sha256(launcher_text, launcher_extension_literal) == expected_sha256,
        "launcher.live_extension_sha256",
    )
    continuity_runtime_hashes: dict[str, str] = {}
    for name, (relative_path, launcher_hash_name) in QUEST_CONTINUITY_RUNTIME_BINDINGS.items():
        if name == "selection":
            expected_runtime_sha256 = spec.bindings["snapshot_selection"].sha256
            launcher_pins = set(
                re.findall(
                    rf"^[ \t]*readonly {re.escape(launcher_hash_name)}='([0-9a-f]{{64}})'$",
                    launcher_text,
                    re.M,
                )
            )
            if expected_runtime_sha256 not in launcher_pins:
                raise AssuranceRunError(
                    "launcher does not admit the authenticated Quest continuity selection runtime"
                )
        elif name == "gpu_runner":
            best_first_b7 = environment.get("QWEN_DFLASH2_BEST_FIRST_B7_REQUIRED", "0")
            if best_first_b7 not in {"0", "1"}:
                raise AssuranceRunError("QWEN_DFLASH2_BEST_FIRST_B7_REQUIRED must be 0 or 1")
            expected_runtime_sha256 = _launcher_literal_sha256(
                launcher_text, launcher_hash_name
            ) or _launcher_flag_literal_sha256(
                launcher_text,
                launcher_hash_name,
                "QWEN_DFLASH2_BEST_FIRST_B7_REQUIRED",
                enabled=best_first_b7 == "1",
            )
            if expected_runtime_sha256 is None:
                raise AssuranceRunError(
                    "launcher does not uniquely pin the Quest continuity gpu_runner runtime"
                )
        else:
            expected_runtime_sha256 = _launcher_literal_sha256(launcher_text, launcher_hash_name)
            if expected_runtime_sha256 is None:
                raise AssuranceRunError(
                    f"launcher does not uniquely pin the Quest continuity {name} runtime"
                )
        continuity_runtime_hashes[name] = expected_runtime_sha256
        _require_owned_regular(
            FileBinding(
                path=VLLM_RUNTIME_SITE_PACKAGES / relative_path,
                sha256=expected_runtime_sha256,
            ),
            f"Quest continuity {name} runtime",
        )
    if cold_seed == "0":
        if mismatches:
            raise AssuranceRunError(
                "Quest selector extension preflight rejected the control binary; mismatches: "
                + ", ".join(mismatches)
            )
        return

    capability_path, capability_sha256 = _artifact_contract_member(
        spec, QUEST_COLD_SEED_CAPABILITY_OUTPUT, "Quest cold-seed capability"
    )
    component_path, component_sha256 = _artifact_contract_member(
        spec, QUEST_COLD_SEED_COMPONENT_OUTPUT, "Quest cold-seed component"
    )
    attestation_path, attestation_sha256 = _artifact_contract_member(
        spec,
        QUEST_COLD_SEED_ATTESTATION_CAPABILITY_OUTPUT,
        "Quest cold-seed attestation capability",
    )
    attestation_source_hashes: dict[str, str] = {}
    for name, (relative_path, _launcher_hash_name) in QUEST_CONTINUITY_RUNTIME_BINDINGS.items():
        _path, source_sha256 = _artifact_contract_member(
            spec,
            relative_path,
            f"Quest cold-seed {name} reviewed runtime source",
        )
        attestation_source_hashes[name] = source_sha256
    try:
        capability = json.loads(capability_path.read_text(encoding="utf-8"))
        component = json.loads(component_path.read_text(encoding="utf-8"))
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssuranceRunError(f"Quest cold-seed preflight input is invalid: {error}") from error
    if (
        not isinstance(capability, dict)
        or not isinstance(component, dict)
        or not isinstance(attestation, dict)
    ):
        raise AssuranceRunError("Quest cold-seed preflight documents must be JSON objects")

    sources = capability.get("source_bindings")
    sources = sources if isinstance(sources, dict) else {}
    certificate = capability.get("component_certificate")
    certificate = certificate if isinstance(certificate, dict) else {}
    projection = capability.get("release_projection")
    projection = projection if isinstance(projection, dict) else {}
    attestation_sources = attestation.get("source_bindings")
    attestation_sources = attestation_sources if isinstance(attestation_sources, dict) else {}
    publication = attestation.get("publication")
    publication = publication if isinstance(publication, dict) else {}
    check(capability.get("schema") == QUEST_COLD_SEED_CAPABILITY_SCHEMA, "capability.schema")
    check(capability.get("capability") == QUEST_COLD_SEED_CAPABILITY, "capability.identity")
    check(capability.get("default_off") is True, "capability.default_off")
    check(
        capability.get("enable_environment") == QUEST_COLD_SEED_ENABLE_ENVIRONMENT,
        "capability.enable_environment",
    )
    check(certificate.get("sha256") == component_sha256, "capability.component_sha256")
    check(
        sources.get("qualified_extension_sha256") == expected_sha256,
        "capability.qualified_extension_sha256",
    )
    check(
        projection.get("external_activation_event") == CACHED_SELECTOR_COLD_SEED_PROBE_EVENT,
        "capability.activation_event",
    )
    check(
        projection.get("external_activation_required_when_enabled") is True,
        "capability.activation_required",
    )
    check(
        projection.get("production_path_present_in_both") is True,
        "capability.production_path",
    )
    check(component.get("schema") == QUEST_COLD_SEED_COMPONENT_SCHEMA, "component.schema")
    check(component.get("passed") is True, "component.passed")
    check(component.get("extension_sha256") == expected_sha256, "component.extension_sha256")
    check(
        attestation.get("schema") == QUEST_COLD_SEED_ATTESTATION_CAPABILITY_SCHEMA,
        "attestation.schema",
    )
    check(
        attestation.get("capability") == QUEST_COLD_SEED_ATTESTATION_CAPABILITY,
        "attestation.identity",
    )
    check(
        publication.get("raw_request_metadata_authority") is False,
        "attestation.raw_request_metadata_authority",
    )
    check(
        publication.get("stale_private_field_cleared_before_publication") is True,
        "attestation.stale_private_field_clearance",
    )
    for name, binding_key in QUEST_CONTINUITY_ATTESTATION_BINDINGS.items():
        expected_runtime_sha256 = continuity_runtime_hashes[name]
        check(
            attestation_source_hashes.get(name) == expected_runtime_sha256,
            f"attestation.artifact_source.{name}",
        )
        check(
            attestation_sources.get(binding_key) == expected_runtime_sha256,
            f"attestation.source_binding.{name}",
        )
    check(
        _launcher_literal_sha256(launcher_text, "EXPECTED_QUEST_CACHED_COLD_SEED_CAPABILITY_SHA256")
        == capability_sha256,
        "launcher.capability_sha256",
    )
    check(
        _launcher_literal_sha256(
            launcher_text, "EXPECTED_QUEST_CACHED_COLD_SEED_CERTIFICATE_SHA256"
        )
        == component_sha256,
        "launcher.component_sha256",
    )
    check(
        _launcher_literal_sha256(
            launcher_text,
            "EXPECTED_QUEST_CACHED_COLD_SEED_ATTESTATION_CAPABILITY_SHA256",
        )
        == attestation_sha256,
        "launcher.attestation_capability_sha256",
    )
    if mismatches:
        raise AssuranceRunError(
            "Quest cold-seed preflight rejected the live extension/ABI contract; mismatches: "
            + ", ".join(dict.fromkeys(mismatches))
        )


def _require_dflash_d7_graph_runtime_preflight(environment: dict[str, str]) -> None:
    """Require a coherent eager or graph draft runtime before consuming the run."""

    selected = environment.get("QWEN_DFLASH_D7_C1_DRAFT_GRAPH")
    if selected is None:
        return
    if selected not in {"0", "1"}:
        raise AssuranceRunError("QWEN_DFLASH_D7_C1_DRAFT_GRAPH must be 0 or 1")
    graph = selected == "1"
    best_first_b7 = environment.get("QWEN_DFLASH2_BEST_FIRST_B7_REQUIRED", "0")
    if best_first_b7 not in {"0", "1"}:
        raise AssuranceRunError("QWEN_DFLASH2_BEST_FIRST_B7_REQUIRED must be 0 or 1")
    for label, (
        relative_path,
        eager_sha256,
        graph_sha256,
    ) in DFLASH_D7_GRAPH_RUNTIME_BINDINGS.items():
        expected_sha256 = graph_sha256 if graph else eager_sha256
        if graph and best_first_b7 == "1" and label == "DFlash2 speculator":
            expected_sha256 = DFLASH_B7_GRAPH_SPECULATOR_SHA256
        _require_owned_regular(
            FileBinding(
                path=VLLM_RUNTIME_SITE_PACKAGES / relative_path,
                sha256=expected_sha256,
            ),
            f"{label} {'graph' if graph else 'eager'} runtime",
        )


def _require_b7_position_stride_result(
    binding: FileBinding,
    *,
    label: str,
    expected_gate_path: Path | None,
    expected_runtime_path: Path | None,
) -> None:
    """Deep-check one executed production-shaped B7 MRoPE admission result."""

    _require_owned_regular(binding, label)
    try:
        document = _exact_dict(
            json.loads(binding.path.read_text(encoding="utf-8")),
            {
                "anchor_mismatches",
                "gate",
                "invalid_layout_rejections",
                "padding_canary_mismatches",
                "position_mismatches",
                "production_layout",
                "rocm",
                "runtime",
                "schema",
                "status",
                "workspace_stage",
            },
            label,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssuranceRunError(f"{label} is not valid UTF-8 JSON: {error}") from error
    if document["schema"] != B7_POSITION_STRIDE_SCHEMA or document["status"] != "passed":
        raise AssuranceRunError(f"{label} schema or status differs")
    if document["production_layout"] != {
        "backing_shape": [3, 3304],
        "device": "cuda:0",
        "device_name": "AMD Radeon AI PRO R9700",
        "dtype": "torch.int64",
        "layout": "torch.strided",
        "position_shape": [3, 8],
        "position_stride": [3304, 1],
        "storage_elements": 9912,
        "storage_offset": 0,
    }:
        raise AssuranceRunError(f"{label} production position ABI differs")
    if document["workspace_stage"] != {
        "aborted_and_cleared": True,
        "slot_identity_preserved": True,
        "staged": True,
    }:
        raise AssuranceRunError(f"{label} workspace stage/abort evidence differs")
    for field in (
        "position_mismatches",
        "anchor_mismatches",
        "padding_canary_mismatches",
    ):
        if document[field] != 0:
            raise AssuranceRunError(f"{label} {field} must be zero")
    expected_rejections = {
        "nonunit_column_stride": {
            "message": "B7 logical-position rewrite requires positive nonoverlapping row strides",
            "rejected": True,
            "type": "B7GPUContractError",
        },
        "nonzero_storage_offset": {
            "message": "B7 logical-position rewrite requires positive nonoverlapping row strides",
            "rejected": True,
            "type": "B7GPUContractError",
        },
        "out_of_bounds_span": {
            "message": "B7 logical-position rewrite exceeds its backing storage",
            "rejected": True,
            "type": "B7GPUContractError",
        },
        "overlapping_rows": {
            "message": "B7 logical-position rewrite requires positive nonoverlapping row strides",
            "rejected": True,
            "type": "B7GPUContractError",
        },
    }
    if document["invalid_layout_rejections"] != expected_rejections:
        raise AssuranceRunError(f"{label} invalid-layout rejection evidence differs")
    if document["rocm"] != {"hip_version": "7.2.53211"}:
        raise AssuranceRunError(f"{label} ROCm identity differs")

    gate = _exact_dict(document["gate"], {"path", "sha256"}, f"{label}.gate")
    if gate["sha256"] != B7_POSITION_STRIDE_GATE_SHA256:
        raise AssuranceRunError(f"{label} gate SHA-256 differs")
    gate_path = _absolute_path(gate["path"], f"{label}.gate.path")
    if expected_gate_path is not None and gate_path.resolve(strict=True) != expected_gate_path:
        raise AssuranceRunError(f"{label} gate path differs")
    _require_owned_regular(
        FileBinding(gate_path, B7_POSITION_STRIDE_GATE_SHA256),
        f"{label} gate",
    )

    runtime = _exact_dict(document["runtime"], {"path", "sha256"}, f"{label}.runtime")
    if runtime["sha256"] != B7_POSITION_STRIDE_RUNTIME_SHA256:
        raise AssuranceRunError(f"{label} runtime SHA-256 differs")
    runtime_path = _absolute_path(runtime["path"], f"{label}.runtime.path")
    if (
        expected_runtime_path is not None
        and runtime_path.resolve(strict=True) != expected_runtime_path
    ):
        raise AssuranceRunError(f"{label} runtime path differs")
    _require_owned_regular(
        FileBinding(runtime_path, B7_POSITION_STRIDE_RUNTIME_SHA256),
        f"{label} runtime",
    )


def _require_b7_position_stride_admission_preflight(
    spec: RunSpec,
    environment: dict[str, str],
    launcher_text: str,
    command_text: str,
) -> None:
    """Execute and validate the exact production-shaped B7 MRoPE admission gate."""

    enabled = environment.get("QWEN_DFLASH2_BEST_FIRST_B7_REQUIRED", "0")
    if enabled not in {"0", "1"}:
        raise AssuranceRunError("QWEN_DFLASH2_BEST_FIRST_B7_REQUIRED must be 0 or 1")
    names = {
        "gate_path": "QWEN_DFLASH2_BEST_FIRST_B7_POSITION_ADMISSION",
        "gate_sha256": "QWEN_DFLASH2_BEST_FIRST_B7_POSITION_ADMISSION_SHA256",
        "result_path": "QWEN_DFLASH2_BEST_FIRST_B7_POSITION_RESULT",
        "result_sha256": "QWEN_DFLASH2_BEST_FIRST_B7_POSITION_RESULT_SHA256",
        "output_path": "QWEN_DFLASH2_BEST_FIRST_B7_POSITION_ADMISSION_OUTPUT",
    }
    present = {key: environment.get(name) for key, name in names.items()}
    if enabled == "0":
        if any(value is not None for value in present.values()):
            raise AssuranceRunError("B7 position admission variables require best-first B7")
        return
    if any(not value for value in present.values()):
        missing = [names[key] for key, value in present.items() if not value]
        raise AssuranceRunError(
            "B7 position admission variables are incomplete: " + ", ".join(missing)
        )
    if present["gate_sha256"] != B7_POSITION_STRIDE_GATE_SHA256:
        raise AssuranceRunError("B7 position admission gate SHA-256 differs")
    if present["result_sha256"] != B7_POSITION_STRIDE_RESULT_SHA256:
        raise AssuranceRunError("B7 position immutable result SHA-256 differs")
    if (
        _launcher_literal_sha256(launcher_text, "EXPECTED_B7_MROPE_POSITION_ADMISSION_SHA256")
        != B7_POSITION_STRIDE_GATE_SHA256
        or _launcher_literal_sha256(launcher_text, "EXPECTED_B7_MROPE_POSITION_RESULT_SHA256")
        != B7_POSITION_STRIDE_RESULT_SHA256
    ):
        raise AssuranceRunError("launcher does not pin the exact B7 position admission closure")

    gate_path = _absolute_path(present["gate_path"], names["gate_path"])
    result_path = _absolute_path(present["result_path"], names["result_path"])
    output_path = _absolute_path(present["output_path"], names["output_path"])
    expected_output = spec.qualification_dir.resolve(strict=True) / B7_POSITION_STRIDE_OUTPUT_NAME
    if output_path != expected_output:
        raise AssuranceRunError(
            "B7 position admission output is outside its exact create-only path"
        )
    if not result_path.as_posix().endswith(B7_POSITION_STRIDE_RESULT_SUFFIX):
        raise AssuranceRunError("B7 position immutable result path differs from v519")
    _require_owned_regular(
        FileBinding(gate_path, B7_POSITION_STRIDE_GATE_SHA256),
        "B7 position admission gate",
    )
    immutable_result = FileBinding(result_path, B7_POSITION_STRIDE_RESULT_SHA256)
    _require_b7_position_stride_result(
        immutable_result,
        label="B7 position immutable v519 result",
        expected_gate_path=None,
        expected_runtime_path=None,
    )

    pythonpath = environment.get("PYTHONPATH", "")
    roots = pythonpath.split(":") if pythonpath else []
    helper_candidates: list[Path] = []
    for raw_root in roots:
        root = Path(raw_root)
        if not root.is_absolute() or root.is_symlink() or not root.is_dir():
            raise AssuranceRunError(
                "B7 position admission PYTHONPATH roots must be real directories"
            )
        candidate = root / "qwen_r9700_lab" / "dflash_b7_gpu_runtime.py"
        if candidate.exists():
            helper_candidates.append(candidate.resolve(strict=True))
    if len(helper_candidates) != 1:
        raise AssuranceRunError("B7 position admission requires exactly one effective GPU helper")
    runtime_path = helper_candidates[0]
    _require_owned_regular(
        FileBinding(runtime_path, B7_POSITION_STRIDE_RUNTIME_SHA256),
        "B7 position admission live GPU helper",
    )
    if output_path.exists() or output_path.is_symlink():
        raise AssuranceRunError(f"B7 position admission output already exists: {output_path}")

    try:
        arguments = shlex.split(command_text)
    except ValueError as error:
        raise AssuranceRunError(
            f"B7 position admission command quoting is invalid: {error}"
        ) from error
    wrapper_positions = [
        index for index, argument in enumerate(arguments) if argument == str(VLLM_WITH_ROCM)
    ]
    if (
        len(wrapper_positions) != 1
        or wrapper_positions[0] + 1 >= len(arguments)
        or arguments[wrapper_positions[0] + 1] != str(VLLM_CONSOLE_SCRIPT)
    ):
        raise AssuranceRunError(
            "B7 position admission requires one exact with-rocm/vLLM command pair"
        )
    if VLLM_CONSOLE_SCRIPT.is_symlink() or not VLLM_CONSOLE_SCRIPT.is_file():
        raise AssuranceRunError("B7 position admission vLLM console script is missing or unsafe")
    wrapper_sha256 = _launcher_literal_sha256(launcher_text, "EXPECTED_WITH_ROCM_SHA256")
    vllm_sha256 = _launcher_literal_sha256(launcher_text, "EXPECTED_VLLM_SHA256")
    if wrapper_sha256 is None or vllm_sha256 is None:
        raise AssuranceRunError("launcher does not pin the B7 admission execution prefix")
    _require_owned_regular(
        FileBinding(VLLM_WITH_ROCM, wrapper_sha256),
        "B7 position admission ROCm wrapper",
    )
    _require_owned_regular(
        FileBinding(VLLM_CONSOLE_SCRIPT, vllm_sha256),
        "B7 position admission vLLM console script",
    )
    try:
        shebang = VLLM_CONSOLE_SCRIPT.open("rb").readline(512).decode("ascii").rstrip("\r\n")
    except (OSError, UnicodeDecodeError) as error:
        raise AssuranceRunError(
            f"cannot read B7 position admission vLLM shebang: {error}"
        ) from error
    if not shebang.startswith("#!/") or any(character.isspace() for character in shebang[2:]):
        raise AssuranceRunError("B7 position admission vLLM shebang is not one absolute Python")
    vllm_python = Path(shebang[2:])
    if vllm_python != VLLM_RUNTIME_ROOT / ".venv/bin/python" or not vllm_python.is_file():
        raise AssuranceRunError("B7 position admission vLLM Python is missing or unsafe")
    gate_environment = dict(os.environ)
    gate_environment.update(environment)
    # The position-layout gate is a model-free admission subprocess, not the
    # serving process.  Inheriting the serving probe variables would bootstrap
    # sitecustomize before run_campaign creates its private probe directory and
    # would make admission depend on an output path that intentionally does not
    # exist yet.
    for name in tuple(gate_environment):
        if name.startswith("QWEN_RELEASE_ACTIVATION_PROBE"):
            del gate_environment[name]
    gate_environment["PYTHONHASHSEED"] = "0"
    command = [
        str(VLLM_WITH_ROCM),
        str(vllm_python),
        str(gate_path),
        "--runtime-source",
        str(runtime_path),
        "--expected-runtime-sha256",
        B7_POSITION_STRIDE_RUNTIME_SHA256,
        "--expected-gate-sha256",
        B7_POSITION_STRIDE_GATE_SHA256,
        "--output",
        str(output_path),
    ]
    old_umask = os.umask(0o077)
    try:
        completed = subprocess.run(
            command,
            env=gate_environment,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
    finally:
        os.umask(old_umask)
    if completed.returncode != 0:
        evidence = (completed.stdout + completed.stderr)[-1600:]
        raise AssuranceRunError(
            f"B7 position admission gate failed with status {completed.returncode}: {evidence}"
        )
    if not output_path.is_file() or output_path.is_symlink():
        raise AssuranceRunError("B7 position admission did not create a regular result")
    if (
        output_path.stat().st_uid != os.getuid()
        or stat.S_IMODE(output_path.stat().st_mode) != 0o600
    ):
        raise AssuranceRunError("B7 position admission result must be caller-owned mode 0600")
    fresh_result = FileBinding(output_path, _sha256_file(output_path))
    _require_b7_position_stride_result(
        fresh_result,
        label="fresh B7 position admission result",
        expected_gate_path=gate_path.resolve(strict=True),
        expected_runtime_path=runtime_path,
    )


def _require_b7_active_v2_transport_result(
    binding: FileBinding,
    *,
    label: str,
    expected_sources: dict[str, Path] | None,
) -> None:
    """Deep-check one executed active-inner V2 atomic B7 payload transport result."""

    _require_owned_regular(binding, label)
    try:
        document = _exact_dict(
            json.loads(binding.path.read_text(encoding="utf-8")),
            {"device", "qualification", "schema", "sources", "transport"},
            label,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssuranceRunError(f"{label} is not valid UTF-8 JSON: {error}") from error
    if document["schema"] != B7_ACTIVE_V2_TRANSPORT_SCHEMA:
        raise AssuranceRunError(f"{label} schema differs")
    if document["device"] != {
        "hip_version": "7.2.53211",
        "name": "AMD Radeon AI PRO R9700",
        "type": "cuda:0",
    }:
        raise AssuranceRunError(f"{label} ROCm device identity differs")
    if document["qualification"] != {"passed": True}:
        raise AssuranceRunError(f"{label} qualification status differs")

    transport = _exact_dict(
        document["transport"],
        {
            "generation_count",
            "generations",
            "pinned_candidate_buffer",
            "pinned_score_buffer",
            "stale_take_rejection",
            "warmup_generation",
            "warmup_pending",
        },
        f"{label}.transport",
    )
    if transport["generation_count"] != 2:
        raise AssuranceRunError(f"{label} generation count differs")
    if (
        transport["pinned_candidate_buffer"] is not True
        or transport["pinned_score_buffer"] is not True
        or transport["warmup_generation"] != 0
        or transport["warmup_pending"] is not False
        or transport["stale_take_rejection"]
        != "best-first B7 active V2 atomic payload copy is absent"
    ):
        raise AssuranceRunError(f"{label} lifecycle or stale-take evidence differs")
    expected_tokens = (
        [1015, 1014, 1013, 1012, 1011, 1010, 1009],
        [2015, 2014, 2013, 2012, 2011, 2010, 2009],
    )
    generations = transport["generations"]
    if not isinstance(generations, list) or len(generations) != 2:
        raise AssuranceRunError(f"{label} generation evidence differs")
    for index, row_value in enumerate(generations):
        row = _exact_dict(
            row_value,
            {
                "enqueue_ns",
                "generation",
                "take_sync_and_plan_ns",
                "tokens",
                "tree_tokens_match",
            },
            f"{label}.transport.generations[{index}]",
        )
        if (
            row["generation"] != index
            or row["tokens"] != expected_tokens[index]
            or row["tree_tokens_match"] is not True
            or type(row["enqueue_ns"]) is not int
            or row["enqueue_ns"] <= 0
            or type(row["take_sync_and_plan_ns"]) is not int
            or row["take_sync_and_plan_ns"] <= 0
        ):
            raise AssuranceRunError(f"{label} generation {index} payload evidence differs")

    sources = _exact_dict(
        document["sources"],
        {"gate", "runner", "runtime", "speculator"},
        f"{label}.sources",
    )
    expected_sha256 = {
        "gate": B7_ACTIVE_V2_TRANSPORT_GATE_SHA256,
        "runner": B7_ACTIVE_V2_TRANSPORT_RUNNER_SHA256,
        "runtime": B7_ACTIVE_V2_TRANSPORT_RUNTIME_SHA256,
        "speculator": B7_ACTIVE_V2_TRANSPORT_SPECULATOR_SHA256,
    }
    if expected_sources is not None and set(expected_sources) != set(expected_sha256):
        raise AssuranceRunError(f"{label} expected source closure is incomplete")
    for name, expected_sha in expected_sha256.items():
        source = _exact_dict(sources[name], {"path", "sha256"}, f"{label}.sources.{name}")
        if source["sha256"] != expected_sha:
            raise AssuranceRunError(f"{label} {name} source SHA-256 differs")
        source_path = _absolute_path(source["path"], f"{label}.sources.{name}.path")
        if expected_sources is not None:
            expected_path = expected_sources[name].resolve(strict=True)
            if source_path.resolve(strict=True) != expected_path:
                raise AssuranceRunError(f"{label} {name} source path differs")
        _require_owned_regular(FileBinding(source_path, expected_sha), f"{label} {name} source")


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssuranceRunError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise AssuranceRunError(f"{label} must be a finite number")
    return number


def _require_quest_tree_wmma_result(
    binding: FileBinding,
    *,
    label: str,
    expected_gate_path: Path | None,
    expected_extension_path: Path | None,
) -> None:
    """Deep-check the immutable exact96/split16 Quest tree-WMMA qualification."""

    _require_owned_regular(binding, label)
    try:
        document = _exact_dict(
            json.loads(binding.path.read_text(encoding="utf-8")),
            {"device", "ordinary_q16", "qualification", "schema", "sources", "tree"},
            label,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssuranceRunError(f"{label} is not valid UTF-8 JSON: {error}") from error
    if document["schema"] != QUEST_TREE_WMMA_SCHEMA:
        raise AssuranceRunError(f"{label} schema differs")
    if document["device"] != {
        "hip_version": "7.2.53211",
        "name": "AMD Radeon AI PRO R9700",
        "type": "cuda:0",
    }:
        raise AssuranceRunError(f"{label} ROCm device identity differs")
    if document["qualification"] != {"passed": True}:
        raise AssuranceRunError(f"{label} qualification status differs")

    ordinary = _exact_dict(
        document["ordinary_q16"],
        {"candidate_median_ms", "control_median_ms", "exact_match", "selected_pages"},
        f"{label}.ordinary_q16",
    )
    candidate_q16_ms = _finite_number(
        ordinary["candidate_median_ms"], f"{label}.ordinary_q16.candidate_median_ms"
    )
    control_q16_ms = _finite_number(
        ordinary["control_median_ms"], f"{label}.ordinary_q16.control_median_ms"
    )
    if (
        ordinary["exact_match"] is not True
        or ordinary["selected_pages"] != 96
        or candidate_q16_ms <= 0
        or control_q16_ms <= 0
        or candidate_q16_ms > control_q16_ms * 1.10
    ):
        raise AssuranceRunError(f"{label} ordinary q16 compatibility differs")

    tree = _exact_dict(
        document["tree"],
        {
            "branching_mask",
            "invisible_sibling_exact",
            "parity_cases",
            "poisoned_repeat_exact",
            "production_split16",
            "rows",
            "scalar_median_ms",
            "tail_straddles",
            "tree_count",
            "tree_row_offset",
            "visible_ancestor_max_abs",
            "wmma_median_ms",
        },
        f"{label}.tree",
    )
    if (
        tree["rows"] != 8
        or tree["tree_count"] != 7
        or tree["tree_row_offset"] != 1
        or tree["branching_mask"] != [0, 0, 1, 2, 5, 10, 42]
        or tree["invisible_sibling_exact"] is not True
        or tree["poisoned_repeat_exact"] is not True
    ):
        raise AssuranceRunError(f"{label} tree geometry or masking evidence differs")
    scalar_ms = _finite_number(tree["scalar_median_ms"], f"{label}.tree.scalar_median_ms")
    wmma_ms = _finite_number(tree["wmma_median_ms"], f"{label}.tree.wmma_median_ms")
    ancestor_delta = _finite_number(
        tree["visible_ancestor_max_abs"], f"{label}.tree.visible_ancestor_max_abs"
    )
    if scalar_ms <= 0 or wmma_ms <= 0 or ancestor_delta <= 0:
        raise AssuranceRunError(f"{label} tree timing or visible-ancestor witness differs")

    parity_cases = tree["parity_cases"]
    if not isinstance(parity_cases, list) or len(parity_cases) != 4:
        raise AssuranceRunError(f"{label} parity case count differs")
    for index, topology in enumerate(("all_root", "branching", "linear", "star")):
        row = _exact_dict(
            parity_cases[index],
            {"cosine", "max_abs", "prefix", "topology"},
            f"{label}.tree.parity_cases[{index}]",
        )
        maximum = _finite_number(row["max_abs"], f"{label}.tree.parity_cases[{index}].max_abs")
        cosine = _finite_number(row["cosine"], f"{label}.tree.parity_cases[{index}].cosine")
        if (
            row["topology"] != topology
            or row["prefix"] != 60346
            or not 0 <= maximum <= 0.03
            or not 0.999 <= cosine <= 1.001
        ):
            raise AssuranceRunError(f"{label} parity case {index} differs")

    tail_rows = tree["tail_straddles"]
    if not isinstance(tail_rows, list) or len(tail_rows) != 7:
        raise AssuranceRunError(f"{label} tail-straddle evidence differs")
    for index, remainder in enumerate(range(9, 16)):
        row = _exact_dict(
            tail_rows[index],
            {"max_abs", "remainder"},
            f"{label}.tree.tail_straddles[{index}]",
        )
        maximum = _finite_number(row["max_abs"], f"{label}.tree.tail_straddles[{index}].max_abs")
        if row["remainder"] != remainder or not 0 <= maximum <= 0.03:
            raise AssuranceRunError(f"{label} tail-straddle case {remainder} differs")

    production = _exact_dict(
        tree["production_split16"],
        {
            "cosine",
            "finite",
            "max_abs",
            "poisoned_repeat_exact",
            "scalar_median_ms",
            "selected_pages",
            "speedup",
            "wmma_median_ms",
        },
        f"{label}.tree.production_split16",
    )
    production_maximum = _finite_number(
        production["max_abs"], f"{label}.tree.production_split16.max_abs"
    )
    production_cosine = _finite_number(
        production["cosine"], f"{label}.tree.production_split16.cosine"
    )
    production_scalar_ms = _finite_number(
        production["scalar_median_ms"],
        f"{label}.tree.production_split16.scalar_median_ms",
    )
    production_wmma_ms = _finite_number(
        production["wmma_median_ms"], f"{label}.tree.production_split16.wmma_median_ms"
    )
    production_speedup = _finite_number(
        production["speedup"], f"{label}.tree.production_split16.speedup"
    )
    computed_speedup = production_scalar_ms / production_wmma_ms
    if (
        production["finite"] is not True
        or production["poisoned_repeat_exact"] is not True
        or production["selected_pages"] != 96
        or not 0 <= production_maximum <= 0.03
        or not 0.999 <= production_cosine <= 1.001
        or production_wmma_ms <= 0
        or production_scalar_ms <= production_wmma_ms
        or production_speedup <= 1
        or not math.isclose(production_speedup, computed_speedup, rel_tol=1e-9, abs_tol=1e-9)
    ):
        raise AssuranceRunError(f"{label} production exact96/split16 evidence differs")

    sources = _exact_dict(
        document["sources"], set(QUEST_TREE_WMMA_SOURCE_SHA256), f"{label}.sources"
    )
    source_paths: dict[str, Path] = {}
    for name, expected_sha256 in QUEST_TREE_WMMA_SOURCE_SHA256.items():
        source = _exact_dict(sources[name], {"path", "sha256"}, f"{label}.sources.{name}")
        if source["sha256"] != expected_sha256:
            raise AssuranceRunError(f"{label} {name} source SHA-256 differs")
        path = _absolute_path(source["path"], f"{label}.sources.{name}.path")
        if path.name != QUEST_TREE_WMMA_SOURCE_BASENAME[name]:
            raise AssuranceRunError(f"{label} {name} source basename differs")
        _require_owned_regular(FileBinding(path, expected_sha256), f"{label} {name} source")
        source_paths[name] = path.resolve(strict=True)
    if expected_gate_path is not None and source_paths["gate"] != expected_gate_path:
        raise AssuranceRunError(f"{label} gate path differs")
    if expected_extension_path is not None and source_paths["candidate"] != expected_extension_path:
        raise AssuranceRunError(f"{label} candidate extension path differs")
    if len({source_paths[name].parent for name in ("cpp", "cu", "hip")}) != 1:
        raise AssuranceRunError(f"{label} source translation roots differ")
    if (
        len(
            {
                source_paths[name].parent
                for name in ("build_ninja", "candidate", "cpp_object", "hip_object")
            }
        )
        != 1
    ):
        raise AssuranceRunError(f"{label} build output roots differ")
    ninja_text = source_paths["build_ninja"].read_text(encoding="utf-8")
    for fragment in (
        f"build {source_paths['hip_object'].name}: cuda_compile {source_paths['hip']}",
        f"build {source_paths['cpp_object'].name}: compile {source_paths['cpp']}",
        "build quest_fp8_selector_treefix_gfx1201.so: link "
        "quest_fp8_selector_ext.cuda.o quest_fp8_selector_ext.o",
    ):
        if fragment not in ninja_text:
            raise AssuranceRunError(f"{label} build.ninja source/object closure differs")
    for name in ("cu", "hip"):
        source_text = source_paths[name].read_text(encoding="utf-8")
        for marker in (
            "template <bool kStageLogicalRows>",
            "if (visible_bits == 0u)",
            "quest_attention_fp8_tree_selected_scalar_reference",
        ):
            if marker not in source_text:
                raise AssuranceRunError(f"{label} {name} lacks tree-WMMA marker {marker}")


def _require_quest_tree_wmma_preflight(environment: dict[str, str], launcher_text: str) -> None:
    """Bind serving to the qualified tree-WMMA extension without rerunning its GPU gate."""

    enabled = environment.get("QWEN_DFLASH2_BEST_FIRST_B7_REQUIRED", "0")
    if enabled not in {"0", "1"}:
        raise AssuranceRunError("QWEN_DFLASH2_BEST_FIRST_B7_REQUIRED must be 0 or 1")
    names = {
        "gate_path": "QWEN_QUEST_TREE_WMMA_GATE",
        "gate_sha256": "QWEN_QUEST_TREE_WMMA_GATE_SHA256",
        "result_path": "QWEN_QUEST_TREE_WMMA_RESULT",
        "result_sha256": "QWEN_QUEST_TREE_WMMA_RESULT_SHA256",
        "extension_path": "QWEN_QUEST_TREE_WMMA_EXTENSION",
        "extension_sha256": "QWEN_QUEST_TREE_WMMA_EXTENSION_SHA256",
    }
    present = {key: environment.get(name) for key, name in names.items()}
    if enabled == "0":
        if any(value is not None for value in present.values()):
            raise AssuranceRunError("Quest tree-WMMA variables require best-first B7")
        return
    if any(not value for value in present.values()):
        missing = [names[key] for key, value in present.items() if not value]
        raise AssuranceRunError("Quest tree-WMMA variables are incomplete: " + ", ".join(missing))
    if present["gate_sha256"] != QUEST_TREE_WMMA_GATE_SHA256:
        raise AssuranceRunError("Quest tree-WMMA gate SHA-256 differs")
    if present["result_sha256"] != QUEST_TREE_WMMA_RESULT_SHA256:
        raise AssuranceRunError("Quest tree-WMMA result SHA-256 differs")
    if present["extension_sha256"] != QUEST_TREE_WMMA_EXTENSION_SHA256:
        raise AssuranceRunError("Quest tree-WMMA extension SHA-256 differs")
    launcher_expected = {
        "EXPECTED_QUEST_TREE_WMMA_GATE_SHA256": QUEST_TREE_WMMA_GATE_SHA256,
        "EXPECTED_QUEST_TREE_WMMA_RESULT_SHA256": QUEST_TREE_WMMA_RESULT_SHA256,
        "EXPECTED_QUEST_TREE_WMMA_EXTENSION_SHA256": QUEST_TREE_WMMA_EXTENSION_SHA256,
    }
    for name, expected_sha256 in launcher_expected.items():
        if _launcher_literal_sha256(launcher_text, name) != expected_sha256:
            raise AssuranceRunError(f"launcher does not pin the exact {name} closure")

    gate_path = _absolute_path(present["gate_path"], names["gate_path"])
    result_path = _absolute_path(present["result_path"], names["result_path"])
    extension_path = _absolute_path(present["extension_path"], names["extension_path"])
    if not result_path.as_posix().endswith(QUEST_TREE_WMMA_RESULT_SUFFIX):
        raise AssuranceRunError("Quest tree-WMMA immutable result path differs from v526")
    _require_owned_regular(
        FileBinding(gate_path, QUEST_TREE_WMMA_GATE_SHA256), "Quest tree-WMMA gate"
    )
    _require_owned_regular(
        FileBinding(extension_path, QUEST_TREE_WMMA_EXTENSION_SHA256),
        "Quest tree-WMMA extension",
    )
    if (
        environment.get("QWEN_QUEST_SELECTOR_EXTENSION") != str(extension_path)
        or environment.get("QWEN_QUEST_SELECTOR_EXTENSION_SHA256")
        != QUEST_TREE_WMMA_EXTENSION_SHA256
    ):
        raise AssuranceRunError(
            "live Quest selector extension is not the qualified tree-WMMA extension"
        )
    _require_quest_tree_wmma_result(
        FileBinding(result_path, QUEST_TREE_WMMA_RESULT_SHA256),
        label="Quest tree-WMMA immutable v526 result",
        expected_gate_path=gate_path.resolve(strict=True),
        expected_extension_path=extension_path.resolve(strict=True),
    )


def _require_b7_active_v2_transport_admission_preflight(
    spec: RunSpec,
    environment: dict[str, str],
    launcher_text: str,
    command_text: str,
) -> None:
    """Execute the active-inner V2 atomic payload transport gate before server launch."""

    enabled = environment.get("QWEN_DFLASH2_BEST_FIRST_B7_REQUIRED", "0")
    if enabled not in {"0", "1"}:
        raise AssuranceRunError("QWEN_DFLASH2_BEST_FIRST_B7_REQUIRED must be 0 or 1")
    names = {
        "gate_path": "QWEN_DFLASH2_BEST_FIRST_B7_ACTIVE_V2_ADMISSION",
        "gate_sha256": "QWEN_DFLASH2_BEST_FIRST_B7_ACTIVE_V2_ADMISSION_SHA256",
        "result_path": "QWEN_DFLASH2_BEST_FIRST_B7_ACTIVE_V2_RESULT",
        "result_sha256": "QWEN_DFLASH2_BEST_FIRST_B7_ACTIVE_V2_RESULT_SHA256",
        "output_path": "QWEN_DFLASH2_BEST_FIRST_B7_ACTIVE_V2_ADMISSION_OUTPUT",
    }
    present = {key: environment.get(name) for key, name in names.items()}
    if enabled == "0":
        if any(value is not None for value in present.values()):
            raise AssuranceRunError("B7 active V2 admission variables require best-first B7")
        return
    if any(not value for value in present.values()):
        missing = [names[key] for key, value in present.items() if not value]
        raise AssuranceRunError(
            "B7 active V2 admission variables are incomplete: " + ", ".join(missing)
        )
    if present["gate_sha256"] != B7_ACTIVE_V2_TRANSPORT_GATE_SHA256:
        raise AssuranceRunError("B7 active V2 admission gate SHA-256 differs")
    if present["result_sha256"] != B7_ACTIVE_V2_TRANSPORT_RESULT_SHA256:
        raise AssuranceRunError("B7 active V2 immutable result SHA-256 differs")
    if (
        _launcher_literal_sha256(launcher_text, "EXPECTED_B7_ACTIVE_V2_ADMISSION_SHA256")
        != B7_ACTIVE_V2_TRANSPORT_GATE_SHA256
        or _launcher_literal_sha256(launcher_text, "EXPECTED_B7_ACTIVE_V2_RESULT_SHA256")
        != B7_ACTIVE_V2_TRANSPORT_RESULT_SHA256
    ):
        raise AssuranceRunError("launcher does not pin the exact B7 active V2 closure")

    gate_path = _absolute_path(present["gate_path"], names["gate_path"])
    result_path = _absolute_path(present["result_path"], names["result_path"])
    output_path = _absolute_path(present["output_path"], names["output_path"])
    expected_output = (
        spec.qualification_dir.resolve(strict=True) / B7_ACTIVE_V2_TRANSPORT_OUTPUT_NAME
    )
    if output_path != expected_output:
        raise AssuranceRunError(
            "B7 active V2 admission output is outside its exact create-only path"
        )
    if not result_path.as_posix().endswith(B7_ACTIVE_V2_TRANSPORT_RESULT_SUFFIX):
        raise AssuranceRunError("B7 active V2 immutable result path differs from v521")
    _require_owned_regular(
        FileBinding(gate_path, B7_ACTIVE_V2_TRANSPORT_GATE_SHA256),
        "B7 active V2 admission gate",
    )
    _require_b7_active_v2_transport_result(
        FileBinding(result_path, B7_ACTIVE_V2_TRANSPORT_RESULT_SHA256),
        label="B7 active V2 immutable v521 result",
        expected_sources=None,
    )

    pythonpath = environment.get("PYTHONPATH", "")
    roots = pythonpath.split(":") if pythonpath else []
    runtime_candidates: list[Path] = []
    for raw_root in roots:
        root = Path(raw_root)
        if not root.is_absolute() or root.is_symlink() or not root.is_dir():
            raise AssuranceRunError(
                "B7 active V2 admission PYTHONPATH roots must be real directories"
            )
        candidate = root / "qwen_r9700_lab" / "dflash_b7_runtime.py"
        if candidate.exists():
            runtime_candidates.append(candidate.resolve(strict=True))
    if len(runtime_candidates) != 1:
        raise AssuranceRunError("B7 active V2 admission requires one effective B7 runtime")
    runtime_path = runtime_candidates[0]
    runner_path = VLLM_RUNTIME_SITE_PACKAGES / "vllm/v1/worker/gpu/model_runner.py"
    speculator_path = (
        VLLM_RUNTIME_SITE_PACKAGES / "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py"
    )
    live_sources = {
        "gate": gate_path.resolve(strict=True),
        "runner": runner_path.resolve(strict=True),
        "runtime": runtime_path,
        "speculator": speculator_path.resolve(strict=True),
    }
    for name, expected_sha in {
        "runner": B7_ACTIVE_V2_TRANSPORT_RUNNER_SHA256,
        "runtime": B7_ACTIVE_V2_TRANSPORT_RUNTIME_SHA256,
        "speculator": B7_ACTIVE_V2_TRANSPORT_SPECULATOR_SHA256,
    }.items():
        _require_owned_regular(
            FileBinding(live_sources[name], expected_sha),
            f"B7 active V2 admission live {name}",
        )
    if output_path.exists() or output_path.is_symlink():
        raise AssuranceRunError(f"B7 active V2 admission output already exists: {output_path}")

    try:
        arguments = shlex.split(command_text)
    except ValueError as error:
        raise AssuranceRunError(
            f"B7 active V2 admission command quoting is invalid: {error}"
        ) from error
    wrapper_positions = [
        index for index, argument in enumerate(arguments) if argument == str(VLLM_WITH_ROCM)
    ]
    if (
        len(wrapper_positions) != 1
        or wrapper_positions[0] + 1 >= len(arguments)
        or arguments[wrapper_positions[0] + 1] != str(VLLM_CONSOLE_SCRIPT)
    ):
        raise AssuranceRunError(
            "B7 active V2 admission requires one exact with-rocm/vLLM command pair"
        )
    wrapper_sha256 = _launcher_literal_sha256(launcher_text, "EXPECTED_WITH_ROCM_SHA256")
    vllm_sha256 = _launcher_literal_sha256(launcher_text, "EXPECTED_VLLM_SHA256")
    if wrapper_sha256 is None or vllm_sha256 is None:
        raise AssuranceRunError("launcher does not pin the B7 active V2 execution prefix")
    _require_owned_regular(
        FileBinding(VLLM_WITH_ROCM, wrapper_sha256),
        "B7 active V2 admission ROCm wrapper",
    )
    _require_owned_regular(
        FileBinding(VLLM_CONSOLE_SCRIPT, vllm_sha256),
        "B7 active V2 admission vLLM console script",
    )
    try:
        shebang = VLLM_CONSOLE_SCRIPT.open("rb").readline(512).decode("ascii").rstrip("\r\n")
    except (OSError, UnicodeDecodeError) as error:
        raise AssuranceRunError(
            f"cannot read B7 active V2 admission vLLM shebang: {error}"
        ) from error
    if not shebang.startswith("#!/") or any(character.isspace() for character in shebang[2:]):
        raise AssuranceRunError("B7 active V2 admission vLLM shebang is not one absolute Python")
    vllm_python = Path(shebang[2:])
    if vllm_python != VLLM_RUNTIME_ROOT / ".venv/bin/python" or not vllm_python.is_file():
        raise AssuranceRunError("B7 active V2 admission vLLM Python is missing or unsafe")

    gate_environment = dict(os.environ)
    gate_environment.update(environment)
    for name in tuple(gate_environment):
        if name.startswith("QWEN_RELEASE_ACTIVATION_PROBE"):
            del gate_environment[name]
    gate_environment["PYTHONHASHSEED"] = "0"
    command = [
        str(VLLM_WITH_ROCM),
        str(vllm_python),
        str(gate_path),
        "--runner",
        str(runner_path),
        "--expected-runner-sha256",
        B7_ACTIVE_V2_TRANSPORT_RUNNER_SHA256,
        "--speculator",
        str(speculator_path),
        "--expected-speculator-sha256",
        B7_ACTIVE_V2_TRANSPORT_SPECULATOR_SHA256,
        "--runtime",
        str(runtime_path),
        "--expected-runtime-sha256",
        B7_ACTIVE_V2_TRANSPORT_RUNTIME_SHA256,
        "--expected-gate-sha256",
        B7_ACTIVE_V2_TRANSPORT_GATE_SHA256,
        "--output",
        str(output_path),
    ]
    old_umask = os.umask(0o077)
    try:
        completed = subprocess.run(
            command,
            env=gate_environment,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
    finally:
        os.umask(old_umask)
    if completed.returncode != 0:
        evidence = (completed.stdout + completed.stderr)[-1600:]
        raise AssuranceRunError(
            f"B7 active V2 admission gate failed with status {completed.returncode}: {evidence}"
        )
    if not output_path.is_file() or output_path.is_symlink():
        raise AssuranceRunError("B7 active V2 admission did not create a regular result")
    metadata = output_path.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
        raise AssuranceRunError("B7 active V2 admission result must be caller-owned mode 0600")
    _require_b7_active_v2_transport_result(
        FileBinding(output_path, _sha256_file(output_path)),
        label="fresh B7 active V2 admission result",
        expected_sources=live_sources,
    )


def _fixed_slot_cache_root(command_text: str) -> Path:
    """Return the one filesystem tier the server will use for fixed-slot payloads."""

    try:
        arguments = shlex.split(command_text)
    except ValueError as error:
        raise AssuranceRunError(f"command quoting is invalid: {error}") from error
    positions = [
        index for index, argument in enumerate(arguments) if argument == "--kv-transfer-config"
    ]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise AssuranceRunError(
            "fixed-slot admission preflight requires exactly one complete --kv-transfer-config"
        )
    try:
        config = json.loads(arguments[positions[0] + 1])
    except json.JSONDecodeError as error:
        raise AssuranceRunError(
            f"fixed-slot admission preflight has invalid --kv-transfer-config JSON: {error}"
        ) from error
    if not isinstance(config, dict) or config.get("kv_connector") != "OffloadingConnector":
        raise AssuranceRunError("fixed-slot admission preflight requires the OffloadingConnector")
    extra = config.get("kv_connector_extra_config")
    if not isinstance(extra, dict):
        raise AssuranceRunError(
            "fixed-slot admission preflight requires connector extra configuration"
        )
    tiers = extra.get("secondary_tiers")
    if not isinstance(tiers, list):
        raise AssuranceRunError("fixed-slot admission preflight requires a secondary-tier list")
    fs_roots = [
        tier.get("root_dir")
        for tier in tiers
        if isinstance(tier, dict) and tier.get("type") == "fs"
    ]
    if len(fs_roots) != 1 or not isinstance(fs_roots[0], str):
        raise AssuranceRunError(
            "fixed-slot admission preflight requires exactly one filesystem tier"
        )
    cache_root = Path(fs_roots[0])
    if not cache_root.is_absolute() or cache_root.is_symlink() or not cache_root.is_dir():
        raise AssuranceRunError(
            f"fixed-slot admission cache root is not a real absolute directory: {cache_root}"
        )
    if cache_root.stat().st_uid != os.getuid():
        raise AssuranceRunError(
            f"fixed-slot admission cache root has the wrong owner: {cache_root}"
        )
    return cache_root.resolve(strict=True)


def _require_fixed_slot_resume_admission_preflight(
    spec: RunSpec, environment: dict[str, str], command_text: str
) -> None:
    """Run the live selection runtime's read-only head/source admission before model startup."""

    enabled = environment.get("QWEN_FIXED_SLOT_SNAPSHOT_EXPORT", "0")
    corrected = environment.get("QWEN_FIXED_SLOT_CORRECTED_GENERATION", "0")
    if enabled not in {"0", "1"}:
        raise AssuranceRunError("QWEN_FIXED_SLOT_SNAPSHOT_EXPORT must be 0 or 1")
    if corrected not in {"0", "1"}:
        raise AssuranceRunError("QWEN_FIXED_SLOT_CORRECTED_GENERATION must be 0 or 1")
    if enabled == "0" or corrected == "0":
        return

    cache_root = _fixed_slot_cache_root(command_text)
    selection_binding = spec.bindings["snapshot_selection"]
    live_selection = VLLM_RUNTIME_SITE_PACKAGES / FIXED_SLOT_SELECTION_RUNTIME_RELATIVE
    _require_owned_regular(
        FileBinding(path=live_selection, sha256=selection_binding.sha256),
        "live fixed-slot selection runtime",
    )

    module_name = f"qwen_fixed_slot_admission_preflight_{selection_binding.sha256}"
    module_spec = importlib.util.spec_from_file_location(module_name, live_selection)
    if module_spec is None or module_spec.loader is None:
        raise AssuranceRunError("cannot load the live fixed-slot selection runtime")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    try:
        module_spec.loader.exec_module(module)
        read_json = getattr(module, "_read_json", None)
        document_id = getattr(module, "_document_id", None)
        ref_schema = getattr(module, "REF_SCHEMA", None)
        manifest_schema = getattr(module, "MANIFEST_SCHEMA", None)
        preflight = getattr(module, "preflight_fixed_slot_root_v2_runtime_sources", None)
        if (
            not callable(read_json)
            or not callable(document_id)
            or not isinstance(ref_schema, str)
            or not isinstance(manifest_schema, str)
            or not callable(preflight)
        ):
            raise AssuranceRunError(
                "live fixed-slot selection runtime lacks read-only admission preflight"
            )
        metadata_root = cache_root / ".qwen-250k-cache-v1"
        ref_path = (
            metadata_root
            / "refs"
            / spec.request.cache_session_id
            / f"{spec.request.cache_branch}.json"
        )
        ref = read_json(ref_path, os.getuid())
        expected_ref_keys = {
            "branch",
            "generation",
            "manifest_sha256",
            "schema",
            "session_id",
            "updated_at",
        }
        if (
            not isinstance(ref, dict)
            or set(ref) != expected_ref_keys
            or ref.get("schema") != ref_schema
            or ref.get("session_id") != spec.request.cache_session_id
            or ref.get("branch") != spec.request.cache_branch
            or ref.get("manifest_sha256") != spec.request.cache_manifest_sha256
            or type(ref.get("generation")) is not int
            or ref["generation"] < 1
        ):
            raise AssuranceRunError(
                "fixed-slot admission preflight rejected the current branch ref"
            )
        manifest_path = metadata_root / "manifests" / f"{spec.request.cache_manifest_sha256}.json"
        manifest = read_json(manifest_path, os.getuid())
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != manifest_schema
            or document_id(manifest, "manifest_sha256", "manifest")
            != spec.request.cache_manifest_sha256
            or manifest.get("session")
            != {
                "branch": spec.request.cache_branch,
                "session_id": spec.request.cache_session_id,
            }
            or manifest.get("generation") != ref["generation"]
            or not isinstance(manifest.get("bindings"), dict)
        ):
            raise AssuranceRunError(
                "fixed-slot admission preflight rejected the requested manifest head"
            )
        preflight(
            manifest["bindings"],
            VLLM_RUNTIME_SITE_PACKAGES / "vllm",
        )
    except AssuranceRunError:
        raise
    except Exception as error:
        raise AssuranceRunError(
            "fixed-slot resume admission preflight rejected the requested head/runtime: "
            f"{type(error).__name__}: {error}"
        ) from error
    finally:
        sys.modules.pop(module_name, None)


def _require_assurance_artifact_bindings(bindings: dict[str, FileBinding]) -> tuple[Path, Path]:
    """Prove diagnostic imports resolve to exact files in the authenticated artifact."""

    manifest_binding = bindings["assurance_manifest"]
    try:
        manifest = json.loads(manifest_binding.path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssuranceRunError(f"cannot read assurance artifact manifest: {error}") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != ARTIFACT_SCHEMA
        or manifest.get("artifact_kind") != "assurance"
    ):
        raise AssuranceRunError("assurance artifact manifest identity is invalid")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise AssuranceRunError("assurance artifact manifest file table is invalid")
    by_path: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise AssuranceRunError("assurance artifact manifest contains a malformed entry")
        output = entry.get("path")
        digest = entry.get("sha256")
        if not isinstance(output, str) or output in by_path:
            raise AssuranceRunError("assurance artifact manifest paths are invalid or duplicated")
        by_path[output] = _sha256(digest, f"assurance manifest {output}")

    semantic_contract = manifest.get("semantic_contract")
    if semantic_contract is not None and not isinstance(semantic_contract, str):
        raise AssuranceRunError("assurance artifact semantic contract is invalid")
    expected_bindings = dict(ASSURANCE_ARTIFACT_BINDINGS)
    full_instrumentation = has_full_instrumentation_contract(semantic_contract)
    if full_instrumentation:
        expected_bindings.update(FULL_INSTRUMENTATION_ARTIFACT_BINDINGS)
    expected_binding_names = (
        FULL_INSTRUMENTATION_BINDINGS if full_instrumentation else ASSURANCE_BINDINGS
    )
    if set(bindings) != expected_binding_names:
        raise AssuranceRunError(
            "assurance bindings do not match the artifact instrumentation contract"
        )

    files_root = manifest_binding.path.parent / "files"
    for name, output in expected_bindings.items():
        binding = bindings[name]
        expected_path = (files_root / output).resolve(strict=True)
        if binding.path.resolve(strict=True) != expected_path:
            raise AssuranceRunError(
                f"{name} binding is not the authenticated artifact member: {expected_path}"
            )
        if by_path.get(output) != binding.sha256:
            raise AssuranceRunError(f"{name} binding differs from the assurance manifest")
    module_root = (files_root / "assurance").resolve(strict=True)
    capture_site_root = (module_root / "capture_site").resolve(strict=True)
    return capture_site_root, module_root


def _require_command_module_identity(
    command_text: str, capture_site_root: Path, module_root: Path
) -> None:
    """Reject duplicate/ambient Python module roots before the server can start."""

    try:
        arguments = shlex.split(command_text)
    except ValueError as error:
        raise AssuranceRunError(f"command quoting is invalid: {error}") from error
    assignments = [argument for argument in arguments if argument.startswith("PYTHONPATH=")]
    if len(assignments) != 1:
        raise AssuranceRunError("command must contain exactly one PYTHONPATH assignment")
    roots = assignments[0].removeprefix("PYTHONPATH=").split(":")
    expected = (capture_site_root, module_root)
    try:
        observed = tuple(Path(root).resolve(strict=True) for root in roots[: len(expected)])
    except (OSError, RuntimeError):
        observed = ()
    if len(roots) < len(expected) or observed != expected:
        raise AssuranceRunError(
            "command must import the authenticated bootstrap and assurance modules first"
        )


def _release_activation_probe_output(command_text: str, qualification_dir: Path) -> Path | None:
    """Validate the optional external release observer before model startup."""

    try:
        arguments = shlex.split(command_text)
    except ValueError as error:
        raise AssuranceRunError(f"command quoting is invalid: {error}") from error
    names = {
        "QWEN_RELEASE_ACTIVATION_PROBE",
        "QWEN_RELEASE_ACTIVATION_PROBE_OUTPUT",
        "QWEN_RELEASE_ACTIVATION_PROBE_SHA256",
        "QWEN_RELEASE_ACTIVATION_PROBE_SITE_SHA256",
        "QWEN_RELEASE_ACTIVATION_PROBE_GPU_TIMING",
        "QWEN_RELEASE_ACTIVATION_PROBE_UNION_STATS",
        "QWEN_RELEASE_ACTIVATION_PROBE_REQUIRED_EVENTS",
    }
    observed: dict[str, str] = {}
    for argument in arguments:
        if "=" not in argument:
            continue
        name, value = argument.split("=", 1)
        if name not in names:
            continue
        if name in observed:
            raise AssuranceRunError(f"release activation probe duplicates {name}")
        observed[name] = value
    if not observed:
        return None
    if (
        set(observed) != names
        or observed["QWEN_RELEASE_ACTIVATION_PROBE"] != "1"
        or observed["QWEN_RELEASE_ACTIVATION_PROBE_GPU_TIMING"] != "1"
        or observed["QWEN_RELEASE_ACTIVATION_PROBE_UNION_STATS"] != "1"
    ):
        raise AssuranceRunError("release activation probe environment is incomplete")

    command_environment = _command_environment(command_text)
    cached_selector = command_environment.get("QWEN_QUEST_CACHED_GEMM_SELECTOR")
    if cached_selector not in {"0", "1"}:
        raise AssuranceRunError(
            "release activation probe requires QWEN_QUEST_CACHED_GEMM_SELECTOR=0 or 1"
        )
    cached_cold_seed = command_environment.get("QWEN_QUEST_CACHED_GEMM_COLD_SEED")
    if cached_cold_seed not in {"0", "1"}:
        raise AssuranceRunError(
            "release activation probe requires QWEN_QUEST_CACHED_GEMM_COLD_SEED=0 or 1"
        )
    if cached_cold_seed == "1" and cached_selector != "1":
        raise AssuranceRunError(
            "release activation probe cold seed requires the cached-GEMM selector"
        )
    required = observed["QWEN_RELEASE_ACTIVATION_PROBE_REQUIRED_EVENTS"].split(",")
    if not required or any(not event for event in required) or len(set(required)) != len(required):
        raise AssuranceRunError("release activation probe required-event contract is invalid")
    expected_selector_event = (
        CACHED_SELECTOR_PROBE_EVENT if cached_selector == "1" else COMPILED_SELECTOR_PROBE_EVENT
    )
    inactive_selector_event = (
        COMPILED_SELECTOR_PROBE_EVENT if cached_selector == "1" else CACHED_SELECTOR_PROBE_EVENT
    )
    if expected_selector_event not in required or inactive_selector_event in required:
        raise AssuranceRunError(
            "release activation probe selector contract conflicts with runtime: "
            f"cached_gemm_selector={cached_selector}, "
            f"required_selector_event={expected_selector_event}, "
            f"inactive_selector_event={inactive_selector_event}, "
            f"configured_required_events={required}"
        )
    cold_seed_required = CACHED_SELECTOR_COLD_SEED_PROBE_EVENT in required
    if cold_seed_required != (cached_cold_seed == "1"):
        raise AssuranceRunError(
            "release activation probe cold-seed contract conflicts with runtime: "
            f"cached_gemm_cold_seed={cached_cold_seed}, "
            f"required_cold_seed_event={CACHED_SELECTOR_COLD_SEED_PROBE_EVENT}, "
            f"configured_required_events={required}"
        )
    unique_required = command_environment.get("QWEN_UNIQUE_DEPTH_MIXED_REQUIRED", "0")
    if unique_required not in {"0", "1"}:
        raise AssuranceRunError("QWEN_UNIQUE_DEPTH_MIXED_REQUIRED must be 0 or 1")
    if unique_required == "1":
        for name in (
            "QWEN_UNIQUE_DEPTH_MIXED",
            "QWEN_UNIQUE_DEPTH_MIXED_REQUIRED",
            "QWEN_UNIQUE_DEPTH_MIXED_EXPERIMENTAL",
        ):
            if command_environment.get(name) != "1":
                raise AssuranceRunError(f"required UNIQUE depth-mixed release needs {name}=1")
    if (UNIQUE_DEPTH_MIXED_PROBE_EVENT in required) != (unique_required == "1"):
        raise AssuranceRunError(
            "release activation probe UNIQUE event contract conflicts with runtime"
        )
    b7_required = command_environment.get("QWEN_DFLASH2_BEST_FIRST_B7_REQUIRED", "0")
    if b7_required not in {"0", "1"}:
        raise AssuranceRunError("QWEN_DFLASH2_BEST_FIRST_B7_REQUIRED must be 0 or 1")
    b7_events = {B7_STAGE_PROBE_EVENT, B7_VERIFY_PROBE_EVENT, B7_COMMIT_PROBE_EVENT}
    if b7_required == "1" and not b7_events.issubset(required):
        raise AssuranceRunError("release activation probe omits required best-first B7 events")
    if b7_required == "0" and b7_events.intersection(required):
        raise AssuranceRunError("release activation probe requires disabled best-first B7 events")

    expected_root = qualification_dir / "release-activation-probe"
    output_root = _materialize_empty_output_directory(
        qualification_dir,
        "probe",
        "release activation probe output directory",
    )
    expected_output = output_root / "events.jsonl"
    if Path(observed["QWEN_RELEASE_ACTIVATION_PROBE_OUTPUT"]) != expected_output:
        raise AssuranceRunError("release activation probe output is outside its run")
    assignments = [argument for argument in arguments if argument.startswith("PYTHONPATH=")]
    if len(assignments) != 1:
        raise AssuranceRunError("release activation probe requires exactly one PYTHONPATH")
    roots = assignments[0].removeprefix("PYTHONPATH=").split(":")
    if not roots or Path(roots[0]) != expected_root:
        raise AssuranceRunError("release activation probe is not the first import root")

    for directory, label in ((expected_root, "release activation probe source directory"),):
        if directory.is_symlink() or not directory.is_dir():
            raise AssuranceRunError(f"{label} is missing or not a real directory")
        metadata = directory.stat()
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
            raise AssuranceRunError(f"{label} must be owned by the caller and mode 0700")

    for filename, hash_name in (
        ("probe.py", "QWEN_RELEASE_ACTIVATION_PROBE_SHA256"),
        ("sitecustomize.py", "QWEN_RELEASE_ACTIVATION_PROBE_SITE_SHA256"),
    ):
        source = expected_root / filename
        digest = _sha256(observed[hash_name], hash_name)
        _require_owned_regular(FileBinding(path=source, sha256=digest), filename)
    return expected_output


def _require_release_activation_probe_events(path: Path, command_text: str) -> dict[str, int]:
    """Require the short diagnostic request to exercise every declared path."""

    if path.is_symlink() or not path.is_file():
        raise AssuranceRunError("release activation probe published no event stream")
    metadata = path.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
        raise AssuranceRunError(
            "release activation probe event stream has unsafe ownership or mode"
        )
    if metadata.st_size <= 0 or metadata.st_size > 1024 * 1024:
        raise AssuranceRunError("release activation probe event stream size is invalid")
    try:
        arguments = shlex.split(command_text)
        required_raw = next(
            argument.split("=", 1)[1]
            for argument in arguments
            if argument.startswith("QWEN_RELEASE_ACTIVATION_PROBE_REQUIRED_EVENTS=")
        )
    except (ValueError, StopIteration) as error:
        raise AssuranceRunError(
            "release activation probe required-event contract is absent"
        ) from error
    required = required_raw.split(",")
    if not required or any(not name for name in required) or len(set(required)) != len(required):
        raise AssuranceRunError("release activation probe required-event contract is invalid")
    observed: Counter[str] = Counter()
    records: list[dict[str, object]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if not isinstance(record, dict) or record.get("schema") != (
                "urn:qwen-r9700:quest-release-activation-probe:v1"
            ):
                raise AssuranceRunError("release activation probe event schema is invalid")
            event = record.get("event")
            if isinstance(event, str):
                observed[event] += 1
                records.append(record)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssuranceRunError(
            f"release activation probe event stream is invalid: {error}"
        ) from error
    missing = [name for name in required if name not in observed]
    if missing:
        command_environment = _command_environment(command_text)
        raise AssuranceRunError(
            "release activation probe did not exercise required paths: "
            f"missing={missing}, required={required}, "
            "cached_gemm_selector="
            f"{command_environment.get('QWEN_QUEST_CACHED_GEMM_SELECTOR')}, "
            f"observed_event_counts={dict(sorted(observed.items()))}, "
            f"event_stream={path}"
        )
    command_environment = _command_environment(command_text)
    if (
        command_environment.get("QWEN_QUEST_CACHED_GEMM_COLD_SEED") == "0"
        and observed[CACHED_SELECTOR_COLD_SEED_PROBE_EVENT]
    ):
        raise AssuranceRunError(
            "release activation probe observed disabled cached-GEMM cold seeding: "
            f"event_count={observed[CACHED_SELECTOR_COLD_SEED_PROBE_EVENT]}, "
            f"event_stream={path}"
        )
    if (
        command_environment.get("QWEN_DFLASH2_BEST_FIRST_B7_REQUIRED") == "1"
        and command_environment.get("QWEN_UNIQUE_DEPTH_MIXED_REQUIRED") == "1"
    ):
        qualified: dict[tuple[str, int], dict[str, list[int]]] = {}
        for record in records:
            request_id = record.get("request_id")
            generation = record.get("b7_generation")
            event = record.get("event")
            index = record.get("event_index")
            if (
                not isinstance(request_id, str)
                or not request_id
                or not isinstance(generation, int)
                or isinstance(generation, bool)
                or generation < 0
                or not isinstance(event, str)
                or not isinstance(index, int)
                or isinstance(index, bool)
            ):
                continue
            key = (request_id, generation)
            stages = qualified.setdefault(key, {})
            if event == UNIQUE_DEPTH_MIXED_PROBE_EVENT:
                if record.get("outcome") == "returned" and record.get("result") is True:
                    stages.setdefault("unique", []).append(index)
            elif event == "gpu_timing" and record.get("stage") == UNIQUE_DEPTH_MIXED_PROBE_EVENT:
                stages.setdefault("unique_timing", []).append(index)
            elif event == B7_STAGE_PROBE_EVENT and record.get("outcome") == "returned":
                stages.setdefault("stage", []).append(index)
            elif event == B7_VERIFY_PROBE_EVENT and record.get("outcome") == "returned":
                stages.setdefault("verify", []).append(index)
            elif event == B7_COMMIT_PROBE_EVENT and record.get("outcome") == "returned":
                stages.setdefault("commit", []).append(index)
        same_generation_order = False
        for stages in qualified.values():
            for stage in stages.get("stage", []):
                for unique in stages.get("unique", []):
                    for verify in stages.get("verify", []):
                        for commit in stages.get("commit", []):
                            if not stage < unique < verify < commit:
                                continue
                            if any(timing > unique for timing in stages.get("unique_timing", [])):
                                same_generation_order = True
                                break
                        if same_generation_order:
                            break
                    if same_generation_order:
                        break
                if same_generation_order:
                    break
            if same_generation_order:
                break
        if not same_generation_order:
            raise AssuranceRunError(
                "release activation probe lacks ordered same-generation B7+UNIQUE activation"
            )
    suppressed_records = [
        record for record in records if record.get("event") == "gpu_timing_suppressed"
    ]
    if suppressed_records:
        suppression_indices = [record.get("event_index") for record in suppressed_records]
        if any(
            not isinstance(index, int) or isinstance(index, bool) or index <= 0
            for index in suppression_indices
        ):
            raise AssuranceRunError(
                "release activation probe capture-suppression ordering is invalid"
            )
        last_suppression = max(int(index) for index in suppression_indices)
        cached_selector_enabled = command_environment.get("QWEN_QUEST_CACHED_GEMM_SELECTOR") == "1"
        expected_selector = (
            CACHED_SELECTOR_PROBE_EVENT
            if cached_selector_enabled
            else COMPILED_SELECTOR_PROBE_EVENT
        )

        def ordered_indices(event: str, **fields: object) -> list[int]:
            indices: list[int] = []
            for record in records:
                if record.get("event") != event or any(
                    record.get(name) != value for name, value in fields.items()
                ):
                    continue
                index = record.get("event_index")
                if isinstance(index, int) and not isinstance(index, bool) and index > 0:
                    indices.append(index)
            return indices

        selector_indices = [
            index
            for index in ordered_indices(
                expected_selector,
                outcome="returned",
                rows=8,
                row_local=True,
            )
            if index > last_suppression
        ]
        timing_indices = [
            index
            for index in ordered_indices("gpu_timing", stage=expected_selector)
            if index > last_suppression
            and any(selector_index < index for selector_index in selector_indices)
        ]
        if not timing_indices:
            raise AssuranceRunError(
                "release activation probe has no post-capture request-time selector GPU timing: "
                f"selector={expected_selector}, last_capture_suppressed_index={last_suppression}, "
                f"post_capture_selector_indices={selector_indices}, event_stream={path}"
            )
        if command_environment.get("QWEN_QUEST_CACHED_GEMM_COLD_SEED") == "1":
            seed_indices = [
                index
                for index in ordered_indices(
                    CACHED_SELECTOR_COLD_SEED_PROBE_EVENT,
                    outcome="returned",
                    rows=8,
                )
                if last_suppression < index < timing_indices[-1]
            ]
            if not seed_indices:
                raise AssuranceRunError(
                    "release activation probe has no post-capture request-time exact cold seed: "
                    f"last_capture_suppressed_index={last_suppression}, "
                    f"selector_gpu_timing_index={timing_indices[-1]}, event_stream={path}"
                )
    return dict(sorted(observed.items()))


def _tree_score_capture_outputs(
    command_text: str, qualification_dir: Path, spec: RunSpec
) -> tuple[Path, Path] | None:
    """Authenticate one optional, external DFlash score-lattice capture."""

    environment = _command_environment(command_text)
    names = {
        "QWEN_DFLASH_TREE_RANK_CAPTURE",
        "QWEN_DFLASH_TREE_PUBLIC_FIXTURE",
        "QWEN_DFLASH_TREE_PUBLIC_MANIFEST_SHA256",
        "QWEN_DFLASH_TREE_LAUNCHER_SHA256",
        "QWEN_DFLASH_TREE_LAUNCHER_PATH",
        "QWEN_DFLASH_TREE_PROMPT_TOKENS",
        "QWEN_DFLASH_TREE_OUTPUT_TOKENS",
        "QWEN_DFLASH_TREE_RANK_OUTPUT",
        "QWEN_DFLASH_TREE_SCORE_OUTPUT",
        "QWEN_DFLASH_TREE_CAPTURE_CONTRACT_SHA256",
        "QWEN_DFLASH_TREE_CAPTURE_HOOK_SHA256",
        "QWEN_DFLASH_TREE_CAPTURE_SITE_SHA256",
        "QWEN_DFLASH_TREE_SCORE_LATTICE_SHA256",
    }
    observed = {name: environment[name] for name in names if name in environment}
    if not observed:
        return None
    if set(observed) != names:
        raise AssuranceRunError("DFlash score-lattice capture environment is incomplete")
    if (
        observed["QWEN_DFLASH_TREE_RANK_CAPTURE"] != "1"
        or observed["QWEN_DFLASH_TREE_PUBLIC_FIXTURE"] != "1"
    ):
        raise AssuranceRunError("DFlash score-lattice capture is not explicitly enabled")
    try:
        prompt_tokens = int(observed["QWEN_DFLASH_TREE_PROMPT_TOKENS"])
        output_tokens = int(observed["QWEN_DFLASH_TREE_OUTPUT_TOKENS"])
    except ValueError as error:
        raise AssuranceRunError("DFlash score-lattice token bounds are invalid") from error
    if prompt_tokens not in {60_298, 249_957} or output_tokens != spec.request.max_tokens:
        raise AssuranceRunError("DFlash score-lattice token bounds differ from the request")
    if output_tokens < 512:
        raise AssuranceRunError("DFlash score-lattice capture requires at least 512 output tokens")

    source_root = qualification_dir / "dflash-tree-score-capture"
    output_root = _materialize_empty_output_directory(
        qualification_dir,
        "dflash-tree-score-output",
        "DFlash score-lattice output directory",
    )
    rank_output = output_root / "rank-evidence.json"
    score_output = output_root / "score-lattice.json"
    expected_paths = {
        "QWEN_DFLASH_TREE_RANK_OUTPUT": rank_output,
        "QWEN_DFLASH_TREE_SCORE_OUTPUT": score_output,
        "QWEN_DFLASH_TREE_LAUNCHER_PATH": spec.bindings["launcher"].path,
    }
    for name, expected in expected_paths.items():
        if Path(observed[name]) != expected:
            raise AssuranceRunError(f"{name} is not bound to this one-shot run")
    if observed["QWEN_DFLASH_TREE_LAUNCHER_SHA256"] != spec.bindings["launcher"].sha256:
        raise AssuranceRunError("DFlash score-lattice launcher hash differs from the run binding")

    for directory, label in (
        (source_root, "DFlash score-lattice source directory"),
        (source_root / "capture_site", "DFlash score-lattice bootstrap directory"),
    ):
        if directory.is_symlink() or not directory.is_dir():
            raise AssuranceRunError(f"{label} is missing or not a real directory")
        metadata = directory.stat()
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
            raise AssuranceRunError(f"{label} must be owned by the caller and mode 0700")

    assignments = [
        argument for argument in shlex.split(command_text) if argument.startswith("PYTHONPATH=")
    ]
    if len(assignments) != 1:
        raise AssuranceRunError("DFlash score-lattice capture requires one PYTHONPATH")
    roots = assignments[0].removeprefix("PYTHONPATH=").split(":")
    if not roots or Path(roots[0]) != source_root / "capture_site":
        raise AssuranceRunError("DFlash score-lattice bootstrap is not the first import root")

    for relative, hash_name in (
        ("capture_contract.py", "QWEN_DFLASH_TREE_CAPTURE_CONTRACT_SHA256"),
        ("capture_hook.py", "QWEN_DFLASH_TREE_CAPTURE_HOOK_SHA256"),
        ("capture_site/sitecustomize.py", "QWEN_DFLASH_TREE_CAPTURE_SITE_SHA256"),
        ("score_lattice.py", "QWEN_DFLASH_TREE_SCORE_LATTICE_SHA256"),
        ("public-manifest.json", "QWEN_DFLASH_TREE_PUBLIC_MANIFEST_SHA256"),
    ):
        digest = _sha256(observed[hash_name], hash_name)
        _require_owned_regular(
            FileBinding(path=source_root / relative, sha256=digest),
            f"DFlash score-lattice {relative}",
        )
    return rank_output, score_output


def _require_tree_score_capture_artifacts(rank_path: Path, score_path: Path) -> dict[str, Any]:
    """Validate create-only rank/score evidence after the captured engine exits."""

    documents = []
    for path, label, size_limit in (
        (rank_path, "rank evidence", 16 * 1024 * 1024),
        (score_path, "score lattice", 32 * 1024 * 1024),
    ):
        if path.is_symlink() or not path.is_file():
            raise AssuranceRunError(f"DFlash {label} was not published")
        metadata = path.stat()
        if (
            metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or not 0 < metadata.st_size <= size_limit
        ):
            raise AssuranceRunError(f"DFlash {label} has unsafe metadata")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AssuranceRunError(f"DFlash {label} is invalid JSON: {error}") from error
        if not isinstance(document, dict):
            raise AssuranceRunError(f"DFlash {label} must be a JSON object")
        documents.append(document)
    rank, score = documents
    if rank.get("schema") != TREE_RANK_CAPTURE_SCHEMA:
        raise AssuranceRunError("DFlash rank evidence schema is invalid")
    if score.get("schema") != TREE_SCORE_CAPTURE_SCHEMA:
        raise AssuranceRunError("DFlash score-lattice schema is invalid")
    score_capture = score.get("capture")
    if (
        not isinstance(score_capture, dict)
        or score_capture.get("public_rank_evidence_sha256") != _sha256_file(rank_path)
        or score_capture.get("private_sensitive") is not True
        or score_capture.get("contains_token_ids") is not False
    ):
        raise AssuranceRunError("DFlash rank/score evidence binding is invalid")
    return {
        "rank_path": str(rank_path),
        "rank_sha256": _sha256_file(rank_path),
        "score_path": str(score_path),
        "score_sha256": _sha256_file(score_path),
        "rounds": len(rank.get("rounds", [])),
    }


def _reduce_authenticated_capture(spec: RunSpec) -> Path:
    """Reconcile API tokens with canonical commit evidence using bound code only."""

    reducer_binding = spec.bindings["capture_reducer"]
    oracle_binding = spec.bindings["oracle_contract"]
    module_root = reducer_binding.path.parents[1].resolve(strict=True)
    expected_oracle = oracle_binding.path.resolve(strict=True)
    output = spec.qualification_dir / "capture" / "oracle-source.json"
    if output.exists() or output.is_symlink():
        raise AssuranceRunError("oracle source output already exists")

    reduction_program = """
import sys
from pathlib import Path

module_root, reducer_path, oracle_path, identity, rounds, states, result, output = sys.argv[1:]
sys.path.insert(0, module_root)
from qwen_r9700_lab import coding_turbo_capture as reducer
from qwen_r9700_lab import coding_turbo_oracle as oracle

if Path(reducer.__file__).resolve(strict=True) != Path(reducer_path).resolve(strict=True):
    raise SystemExit("unauthenticated capture reducer import")
if Path(oracle.__file__).resolve(strict=True) != Path(oracle_path).resolve(strict=True):
    raise SystemExit("unauthenticated oracle contract import")
reducer.reduce_capture(
    identity_path=Path(identity),
    round_stream=Path(rounds),
    state_stream=Path(states),
    result_path=Path(result),
    output=Path(output),
)
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            reduction_program,
            str(module_root),
            str(reducer_binding.path),
            str(expected_oracle),
            str(spec.qualification_dir / "oracle-identity.json"),
            str(spec.qualification_dir / "capture" / "rounds.jsonl"),
            str(spec.qualification_dir / "capture" / "states.jsonl"),
            str(spec.request.output),
            str(output),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip()[-2000:]
        raise AssuranceRunError(f"API/canonical evidence reconciliation failed: {detail}")
    if output.is_symlink() or not output.is_file() or output.stat().st_mode & 0o777 != 0o400:
        raise AssuranceRunError("capture reducer did not publish an immutable oracle source")
    return output


def _reconcile_precommit_capture(spec: RunSpec, environment: dict[str, str]) -> Path | None:
    """Validate and seal the optional post-M8/pre-commit canonical-state stream."""

    configured = environment.get("QWEN_ROUND_EQUIVALENCE_PRECOMMIT_OUTPUT")
    if configured is None:
        return None
    stream = spec.qualification_dir / "capture" / "precommit.jsonl"
    if configured != str(stream):
        raise AssuranceRunError("precommit capture is not confined to this qualification")
    _require_private_evidence(stream, "round-equivalence precommit stream", maximum_bytes=128 << 20)

    binding = spec.bindings["round_equivalence_contract"]
    module_name = f"qwen_round_equivalence_reconcile_{binding.sha256}"
    module_spec = importlib.util.spec_from_file_location(module_name, binding.path)
    if module_spec is None or module_spec.loader is None:
        raise AssuranceRunError("cannot load authenticated round-equivalence contract")
    module = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(module)
        normalize = getattr(module, "normalize_precommit_event", None)
        header_schema = getattr(module, "PRECOMMIT_HEADER_SCHEMA", None)
        if not callable(normalize) or not isinstance(header_schema, str):
            raise AssuranceRunError("round-equivalence contract lacks precommit validation")
        rows = [json.loads(line) for line in stream.read_text(encoding="utf-8").splitlines()]
        if len(rows) < 2 or not isinstance(rows[0], dict) or set(rows[0]) != {"header"}:
            raise AssuranceRunError("precommit stream lacks its header or event")
        header = rows[0]["header"]
        if (
            not isinstance(header, dict)
            or header.get("schema") != header_schema
            or header.get("request_id") != spec.request.request_id
        ):
            raise AssuranceRunError("precommit stream header identity is invalid")
        events: list[dict[str, Any]] = []
        for index, row in enumerate(rows[1:]):
            if not isinstance(row, dict) or set(row) != {"event"}:
                raise AssuranceRunError(f"precommit stream row {index + 1} is invalid")
            event = normalize(row["event"])
            if event["request_id"] != spec.request.request_id or event["round_index"] != index:
                raise AssuranceRunError("precommit stream request or round ordering differs")
            events.append(event)
    except AssuranceRunError:
        raise
    except Exception as error:
        raise AssuranceRunError(f"precommit evidence reconciliation failed: {error}") from error
    finally:
        sys.modules.pop(module_name, None)

    proposal = _round_equivalence_proposal(spec, environment)
    if proposal is not None:
        if len(events) != 1:
            raise AssuranceRunError(
                "a forced proposal qualification must contain exactly one M8 round"
            )
        if events[0]["execution_mode"] != "speculative-m8":
            raise AssuranceRunError("a forced proposal was not evaluated by speculative M8")
        if events[0]["transition_row_token_ids"][1:] != proposal["proposal_token_ids"]:
            raise AssuranceRunError(
                "captured M8 rows differ from the authenticated forced proposal"
            )

    output = spec.qualification_dir / "capture" / "precommit-source.json"
    if output.exists() or output.is_symlink():
        raise AssuranceRunError("precommit source output already exists")
    payload = {
        "event_count": len(events),
        "execution_modes": sorted({event["execution_mode"] for event in events}),
        "first_sequence_length": events[0]["sequence_length"],
        "last_round_index": events[-1]["round_index"],
        "request_id": spec.request.request_id,
        "round_equivalence_contract_sha256": binding.sha256,
        "schema": "urn:qwen-r9700:round-equivalence-precommit-source:v2",
        "stream_sha256": _sha256_file(stream),
    }
    descriptor = os.open(
        output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        os.write(descriptor, json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        os.write(descriptor, b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return output


def _round_equivalence_proposal(
    spec: RunSpec, environment: dict[str, str]
) -> dict[str, Any] | None:
    """Authenticate the optional assurance-only proposal input before model start."""

    configured = environment.get("QWEN_ROUND_EQUIVALENCE_PROPOSAL_INPUT")
    expected_sha256 = environment.get("QWEN_ROUND_EQUIVALENCE_PROPOSAL_SHA256")
    if configured is None and expected_sha256 is None:
        return None
    if configured is None or expected_sha256 is None:
        raise AssuranceRunError("round-equivalence proposal path/hash must be supplied together")
    path = spec.qualification_dir / "round-equivalence-proposal.json"
    if configured != str(path):
        raise AssuranceRunError("round-equivalence proposal is not confined to this qualification")
    _require_private_evidence(path, "round-equivalence proposal", maximum_bytes=64 << 10)
    if _sha256(expected_sha256, "round-equivalence proposal SHA256") != _sha256_file(path):
        raise AssuranceRunError("round-equivalence proposal file differs from its command hash")

    binding = spec.bindings["round_equivalence_contract"]
    module_name = f"qwen_round_equivalence_proposal_{binding.sha256}"
    module_spec = importlib.util.spec_from_file_location(module_name, binding.path)
    if module_spec is None or module_spec.loader is None:
        raise AssuranceRunError("cannot load authenticated round-equivalence contract")
    module = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(module)
        normalize = getattr(module, "normalize_forced_proposal", None)
        if not callable(normalize):
            raise AssuranceRunError("round-equivalence contract lacks proposal validation")
        proposal = normalize(json.loads(path.read_text(encoding="utf-8")))
    except AssuranceRunError:
        raise
    except Exception as error:
        raise AssuranceRunError(f"round-equivalence proposal validation failed: {error}") from error
    finally:
        sys.modules.pop(module_name, None)

    prompt_tokens_raw = environment.get("QWEN_DFLASH_ASSURANCE_PROMPT_TOKENS")
    try:
        prompt_tokens = int(prompt_tokens_raw or "")
    except ValueError as error:
        raise AssuranceRunError("assurance prompt-token binding is invalid") from error
    manifest_binding = spec.bindings["assurance_manifest"]
    try:
        manifest = json.loads(manifest_binding.path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssuranceRunError(f"assurance manifest is invalid: {error}") from error
    if not isinstance(manifest, dict):
        raise AssuranceRunError("assurance manifest must be an object")
    expected = {
        "prompt_tokens": prompt_tokens,
        "request_id": spec.request.request_id,
        "runtime_artifact_manifest_sha256": manifest_binding.sha256,
        "semantic_source_sha256": manifest.get("semantic_source_sha256"),
        "snapshot_manifest_sha256": spec.request.cache_manifest_sha256,
    }
    for field, value in expected.items():
        if proposal[field] != value:
            raise AssuranceRunError(f"round-equivalence proposal differs at {field}")
    return proposal


def _command_environment(command_text: str) -> dict[str, str]:
    """Extract the unique environment contract from an ``env`` launch command."""

    try:
        arguments = shlex.split(command_text)
    except ValueError as error:
        raise AssuranceRunError(f"command quoting is invalid: {error}") from error
    environment: dict[str, str] = {}
    for argument in arguments:
        if "=" not in argument:
            continue
        name, value = argument.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        if name in environment:
            raise AssuranceRunError(f"command duplicates environment variable {name}")
        environment[name] = value
    return environment


def _selected_integers(environment: dict[str, str], singular: str, plural: str) -> tuple[int, ...]:
    singular_value = environment.get(singular)
    plural_value = environment.get(plural)
    if bool(singular_value) == bool(plural_value):
        raise AssuranceRunError(f"layer diagnostic must set exactly one of {singular} and {plural}")
    raw = singular_value if singular_value is not None else plural_value
    assert raw is not None
    try:
        values = tuple(int(value) for value in raw.split(","))
    except ValueError as error:
        raise AssuranceRunError("layer diagnostic integer filter is invalid") from error
    if not values or len(values) != len(set(values)):
        raise AssuranceRunError("layer diagnostic integer filter is empty or duplicated")
    return values


def _require_non_promotable_diagnostic_identity(spec: RunSpec, environment: dict[str, str]) -> str:
    """Authenticate the distinct identity used by an arbitrary-context diagnosis."""

    identity_path = spec.qualification_dir / "oracle-identity.json"
    try:
        identity = _exact_dict(
            json.loads(identity_path.read_text(encoding="utf-8")),
            NON_PROMOTABLE_DIAGNOSTIC_KEYS,
            "non-promotable diagnostic identity",
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssuranceRunError(
            f"non-promotable diagnostic identity is invalid: {error}"
        ) from error
    if (
        identity["schema"] != NON_PROMOTABLE_DIAGNOSTIC_SCHEMA
        or identity["artifact_kind"] != "assurance"
        or identity["classification"] != "non_promotable_diagnostic"
        or identity["promotable"] is not False
        or identity["lifecycle"] != "snapshot_restore"
        or identity["quest_page_budget"] != 96
        or identity["sampling"] != {"temperature": 0.0, "top_k": 1, "top_p": 1.0}
    ):
        raise AssuranceRunError("non-promotable diagnostic identity contract is invalid")
    try:
        context_tokens = int(environment["QWEN_DFLASH_ASSURANCE_PROMPT_TOKENS"])
    except (KeyError, ValueError) as error:
        raise AssuranceRunError("diagnostic command prompt-token identity is invalid") from error
    if not 1 <= context_tokens <= 253_792 or identity["context_tokens"] != context_tokens:
        raise AssuranceRunError("diagnostic context identity differs from the bounded request")
    if identity["run_id"] != spec.request.request_id:
        raise AssuranceRunError("diagnostic run identity differs from the request")
    target_only = environment.get("QWEN_FIXED_SLOT_TARGET_ONLY_ORACLE", "0") == "1"
    if identity["dflash_enabled"] is not (not target_only):
        raise AssuranceRunError("diagnostic execution-arm identity differs from the command")

    manifest_binding = spec.bindings["assurance_manifest"]
    selection_binding = spec.bindings["snapshot_payload_selection"]
    receipt_binding = spec.bindings["state_producer_receipt"]
    try:
        manifest = json.loads(manifest_binding.path.read_text(encoding="utf-8"))
        selection = json.loads(selection_binding.path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_binding.path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssuranceRunError(f"diagnostic provenance input is invalid: {error}") from error
    selection_request = selection.get("request") if isinstance(selection, dict) else None
    selection_bindings = selection.get("bindings") if isinstance(selection, dict) else None
    target = (
        selection_bindings.get("target_model") if isinstance(selection_bindings, dict) else None
    )
    tokenizer = (
        selection_bindings.get("tokenizer") if isinstance(selection_bindings, dict) else None
    )
    payload_provenance = (
        selection.get("payload_provenance") if isinstance(selection, dict) else None
    )
    if not all(
        isinstance(value, dict)
        for value in (manifest, selection_request, target, tokenizer, payload_provenance, receipt)
    ):
        raise AssuranceRunError("diagnostic provenance structure is incomplete")
    expected = {
        "runtime_artifact_manifest_sha256": manifest_binding.sha256,
        "semantic_source_sha256": manifest.get("semantic_source_sha256"),
        "snapshot_manifest_sha256": spec.request.cache_manifest_sha256,
        "prompt_token_ids_sha256": selection_request.get("prompt_token_ids_sha256"),
        "model_sha256": target.get("evidence_sha256"),
        "tokenizer_sha256": tokenizer.get("evidence_sha256"),
        "payload_producer_receipt_sha256": receipt.get("snapshot_payload_receipt_sha256"),
    }
    if selection_request.get("prompt_tokens") != context_tokens:
        raise AssuranceRunError("diagnostic snapshot context differs from the request")
    if payload_provenance.get("prompt_token_ids_sha256") != expected["prompt_token_ids_sha256"]:
        raise AssuranceRunError("diagnostic prompt identity differs from payload provenance")
    for field, value in expected.items():
        if identity[field] != value:
            raise AssuranceRunError(f"diagnostic identity differs at {field}")
        _sha256(identity[field], f"diagnostic identity {field}")
    return _sha256_file(identity_path)


def _diagnostic_target_model(spec: RunSpec) -> tuple[Path, Path, str]:
    """Resolve and authenticate the target model selected by the snapshot evidence."""

    selection_path = spec.bindings["snapshot_payload_selection"].path
    try:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        target = selection["bindings"]["target_model"]
        model_root = _absolute_path(target["identity"], "diagnostic target model")
        evidence_path = _absolute_path(
            target["evidence_path"], "diagnostic target-model evidence"
        )
        evidence_sha256 = _sha256(
            target["evidence_sha256"], "diagnostic target-model evidence SHA256"
        )
    except (KeyError, TypeError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssuranceRunError(f"diagnostic target-model selection is invalid: {error}") from error
    if (
        evidence_path.is_symlink()
        or not evidence_path.is_file()
        or _sha256_file(evidence_path) != evidence_sha256
    ):
        raise AssuranceRunError("diagnostic target-model evidence differs from the selection")
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssuranceRunError(f"diagnostic target-model evidence is invalid: {error}") from error
    if not isinstance(evidence, dict) or evidence.get("identity") != str(model_root):
        raise AssuranceRunError("diagnostic target-model evidence identity differs")
    files = evidence.get("files")
    config_entries = (
        [entry for entry in files if isinstance(entry, dict) and entry.get("name") == "config.json"]
        if isinstance(files, list)
        else []
    )
    if len(config_entries) != 1:
        raise AssuranceRunError("diagnostic target-model evidence lacks one config.json")
    config_sha256 = _sha256(
        config_entries[0].get("sha256"), "diagnostic target-model config SHA256"
    )
    config_path = model_root / "config.json"
    if (
        config_path.is_symlink()
        or not config_path.is_file()
        or _sha256_file(config_path) != config_sha256
    ):
        raise AssuranceRunError("diagnostic target-model config differs from its evidence")
    return model_root, config_path, config_sha256


def _require_diagnostic_runtime_file_bindings(
    spec: RunSpec, environment: dict[str, str]
) -> dict[str, dict[str, str]]:
    """Authenticate the complete runtime table bound into the controller command."""

    expected_path = spec.qualification_dir / RUNTIME_FILE_BINDINGS_NAME
    raw_path = environment.get(RUNTIME_FILE_BINDINGS_ENV, "")
    raw_sha256 = environment.get(RUNTIME_FILE_BINDINGS_SHA_ENV, "")
    if (
        raw_path != str(expected_path)
        or _sha256(raw_sha256, RUNTIME_FILE_BINDINGS_SHA_ENV) != raw_sha256
    ):
        raise AssuranceRunError("diagnostic runtime table is not bound to this run")
    if (
        expected_path.is_symlink()
        or not expected_path.is_file()
        or _sha256_file(expected_path) != raw_sha256
    ):
        raise AssuranceRunError("diagnostic runtime table differs from its command binding")
    try:
        document = _exact_dict(
            json.loads(expected_path.read_text(encoding="utf-8")),
            {"schema", "files"},
            "diagnostic runtime table",
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssuranceRunError(f"diagnostic runtime table is invalid: {error}") from error
    if document["schema"] != RUNTIME_FILE_BINDINGS_SCHEMA:
        raise AssuranceRunError("diagnostic runtime table has the wrong schema")
    files = document["files"]
    if not isinstance(files, dict) or set(files) != RUNTIME_FILE_LABELS:
        raise AssuranceRunError("diagnostic runtime table has incomplete runtime identities")
    normalized: dict[str, dict[str, str]] = {}
    for label in sorted(RUNTIME_FILE_LABELS):
        entry = _exact_dict(files[label], {"path", "sha256"}, f"runtime file {label}")
        path = _absolute_path(entry["path"], f"runtime file {label}.path")
        digest = _sha256(entry["sha256"], f"runtime file {label}.sha256")
        normalized[label] = {"path": str(path), "sha256": digest}
    transition = json.loads(
        spec.bindings["state_consumer_transition_receipt"].path.read_text(encoding="utf-8")
    )
    expected_links = {
        "gdn_adapter": spec.bindings["qwen_gdn_linear_attn"].sha256,
        "qwen3_next": spec.bindings["qwen3_next"].sha256,
        "qwen_text_rope": spec.bindings["qwen_text_rope"].sha256,
        "model_runner": transition.get("consumer_accepted_path_commit_sha256"),
    }
    for label, digest in expected_links.items():
        if normalized[label]["sha256"] != digest:
            raise AssuranceRunError(
                f"diagnostic runtime table differs from authenticated {label}"
            )
    return normalized


def _require_private_evidence(path: Path, label: str, *, maximum_bytes: int) -> None:
    if path.is_symlink() or not path.is_file():
        raise AssuranceRunError(f"{label} is missing or not a regular file")
    metadata = path.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
        raise AssuranceRunError(f"{label} must be owned by the caller and mode 0600")
    if not 0 < metadata.st_size <= maximum_bytes:
        raise AssuranceRunError(f"{label} size is invalid")


def _reconcile_layer_diagnostic(spec: RunSpec, environment: dict[str, str]) -> Path:
    """Publish a small authenticated receipt for a targeted layer-only capture."""

    positions = _selected_integers(
        environment,
        "QWEN_DFLASH_ASSURANCE_LAYER_POSITION",
        "QWEN_DFLASH_ASSURANCE_LAYER_POSITIONS",
    )
    raw_layers = environment.get("QWEN_DFLASH_ASSURANCE_LAYER_INDICES", "")
    try:
        layers = tuple(int(value) for value in raw_layers.split(","))
    except ValueError as error:
        raise AssuranceRunError("layer diagnostic decoder-layer filter is invalid") from error
    if (
        not layers
        or len(layers) != len(set(layers))
        or any(not 0 <= layer < 64 for layer in layers)
    ):
        raise AssuranceRunError("layer diagnostic decoder-layer filter is invalid")

    capture_dir = spec.qualification_dir / "capture"
    layer_stream = capture_dir / "layers.jsonl"
    round_stream = capture_dir / "rounds.jsonl"
    _require_private_evidence(layer_stream, "layer diagnostic stream", maximum_bytes=128 << 20)
    _require_private_evidence(round_stream, "token diagnostic stream", maximum_bytes=128 << 20)

    observed_boundaries: set[tuple[int, int]] = set()
    observed_quest: set[tuple[int, int]] = set()
    header: dict[str, Any] | None = None
    try:
        with layer_stream.open(encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise AssuranceRunError("layer diagnostic stream contains a non-object")
                if line_index == 0:
                    header = row
                    continue
                event = row.get("event")
                if not isinstance(event, dict):
                    continue
                schema = event.get("schema")
                if schema == "qwen-r9700.dflash-layer-boundary.v1":
                    layer = event.get("layer_index")
                    event_positions = event.get("positions")
                    if isinstance(layer, int) and isinstance(event_positions, list):
                        flattened = {
                            item
                            for position in event_positions
                            if isinstance(position, list)
                            for item in position
                            if isinstance(item, int)
                        }
                        observed_boundaries.update((layer, position) for position in flattened)
                elif schema in {
                    "qwen-r9700.quest-m8-q1-micro.v2",
                    "qwen-r9700.quest-m8-q1-micro.v3",
                }:
                    layer_name = event.get("layer_name")
                    position = event.get("logical_position")
                    if isinstance(layer_name, str) and isinstance(position, int):
                        match = re.search(r"\.layers\.(\d+)\.", layer_name)
                        if match is not None:
                            observed_quest.add((int(match.group(1)), position))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssuranceRunError(f"layer diagnostic stream is invalid: {error}") from error

    if (
        header is None
        or header.get("schema") != "qwen-r9700.dflash-lossless-capture-header.v1"
        or header.get("request_id") != spec.request.request_id
        or header.get("layer_indices") != sorted(layers)
    ):
        raise AssuranceRunError("layer diagnostic header does not match the request contract")
    expected = {(layer, position) for layer in layers for position in positions}
    missing_boundaries = sorted(expected - observed_boundaries)
    if missing_boundaries:
        raise AssuranceRunError(
            f"layer diagnostic omitted requested decoder boundaries: {missing_boundaries}"
        )
    expected_quest = {item for item in expected if item[0] % 4 == 3}
    missing_quest = sorted(expected_quest - observed_quest)
    if missing_quest:
        raise AssuranceRunError(
            f"layer diagnostic omitted requested Quest evidence: {missing_quest}"
        )

    non_promotable = (
        environment.get("QWEN_DFLASH_ASSURANCE_NON_PROMOTABLE_DIAGNOSTIC", "0") == "1"
    )
    diagnostic_identity_sha256 = (
        _require_non_promotable_diagnostic_identity(spec, environment)
        if non_promotable
        else None
    )
    output = capture_dir / "oracle-source.json"
    payload = {
        "schema": "urn:qwen-r9700:layer-diagnostic-source:v1",
        "classification": (
            "non_promotable_diagnostic" if non_promotable else "qualified_layer_diagnostic"
        ),
        "promotable": False,
        "request_id": spec.request.request_id,
        "positions": list(positions),
        "layer_indices": list(layers),
        "diagnostic_identity_sha256": diagnostic_identity_sha256,
        "layer_stream_sha256": _sha256_file(layer_stream),
        "round_stream_sha256": _sha256_file(round_stream),
        "result_sha256": _sha256_file(spec.request.output),
    }
    descriptor = os.open(
        output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.write(descriptor, json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        os.write(descriptor, b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    output.chmod(0o400)
    return output


def _require_state_capture_provenance(bindings: dict[str, FileBinding]) -> None:
    """Authenticate the complete state-export dependency closure before startup."""

    exporter = bindings["state_exporter"]
    module_name = f"qwen_state_exporter_preflight_{exporter.sha256}"
    module_spec = importlib.util.spec_from_file_location(module_name, exporter.path)
    if module_spec is None or module_spec.loader is None:
        raise AssuranceRunError("cannot load the authenticated state exporter for preflight")
    module = importlib.util.module_from_spec(module_spec)
    parent = str(exporter.path.parent)
    sys.path.insert(0, parent)
    try:
        module_spec.loader.exec_module(module)
        load_receipt = getattr(module, "_load_receipt", None)
        load_transition = getattr(module, "_load_consumer_transition_receipt", None)
        authenticate = getattr(module, "_authenticate_runtime_receipt", None)
        cache_contracts = getattr(module, "_load_cache_contracts", None)
        if not all(
            callable(candidate)
            for candidate in (load_receipt, load_transition, authenticate, cache_contracts)
        ):
            raise AssuranceRunError(
                "state exporter has no complete provenance/cache-contract validators"
            )
        cache_contracts()
        receipt = load_receipt(bindings["state_producer_receipt"].path)
        transition = load_transition(bindings["state_consumer_transition_receipt"].path)
        authenticate(
            receipt,
            transition,
            artifact_manifest=bindings["assurance_manifest"].path,
            snapshot_selection=bindings["snapshot_payload_selection"].path,
            accepted_path_commit=module.ACCEPTED_PATH_COMMIT_SOURCE,
            snapshot_format=module.SNAPSHOT_FORMAT_SOURCE,
            exporter_source=exporter.path,
        )
    except AssuranceRunError:
        raise
    except Exception as error:
        raise AssuranceRunError(f"state-capture provenance failed: {error}") from error
    finally:
        sys.path.remove(parent)
        sys.modules.pop(module_name, None)


def _require_capture_runtime_provenance(
    binding: FileBinding,
    *,
    launcher_sha256: str,
    launcher_path: Path,
    consumer_transition_receipt: FileBinding,
    runtime_binding_environment: dict[str, str],
) -> None:
    """Execute the authenticated capture hook's read-only runtime hash gate."""

    module_name = f"qwen_assurance_preflight_{binding.sha256}"
    module_spec = importlib.util.spec_from_file_location(module_name, binding.path)
    if module_spec is None or module_spec.loader is None:
        raise AssuranceRunError("cannot load the authenticated capture hook for preflight")
    module = importlib.util.module_from_spec(module_spec)
    parent = str(binding.path.parent)
    transition_environment = "QWEN_CODING_TURBO_STATE_CONSUMER_TRANSITION_RECEIPT"
    prior_transition = os.environ.get(transition_environment)
    prior_runtime_bindings = {name: os.environ.get(name) for name in runtime_binding_environment}
    sys.path.insert(0, parent)
    try:
        os.environ[transition_environment] = str(consumer_transition_receipt.path)
        os.environ.update(runtime_binding_environment)
        module_spec.loader.exec_module(module)
        validator = getattr(module, "_verify_runtime_files", None)
        if not callable(validator):
            raise AssuranceRunError("capture hook has no runtime provenance validator")
        validator(launcher_sha256, str(launcher_path))
    except AssuranceRunError:
        raise
    except Exception as error:
        raise AssuranceRunError(f"capture runtime provenance failed: {error}") from error
    finally:
        if prior_transition is None:
            os.environ.pop(transition_environment, None)
        else:
            os.environ[transition_environment] = prior_transition
        for name, prior_value in prior_runtime_bindings.items():
            if prior_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prior_value
        sys.path.remove(parent)
        sys.modules.pop(module_name, None)


def _require_runner_contract(binding: FileBinding) -> None:
    """Reject a runner whose accepted completion range drifted from the controller."""

    try:
        tree = ast.parse(binding.path.read_text(encoding="utf-8"), filename=str(binding.path))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise AssuranceRunError(f"cannot inspect request runner contract: {error}") from error
    values: list[int] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "MAX_TOKENS" for target in targets
        ):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, int):
            values.append(value.value)
    if values != [MAX_COMPLETION_TOKENS]:
        raise AssuranceRunError(
            "request runner MAX_TOKENS must exactly match controller limit "
            f"{MAX_COMPLETION_TOKENS}; observed {values}"
        )


def _live_engine_processes() -> list[int]:
    matches: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if (b"serve" in argv and any(argument.endswith(b"/vllm") for argument in argv)) or (
            argv and argv[0].startswith(b"VLLM::EngineCore")
        ):
            matches.append(int(entry.name))
    return sorted(matches)


def _reap_stale_offload_regions(
    *,
    root: Path = Path("/dev/shm"),
    proc_root: Path = Path("/proc"),
    expected_root_uid: int = 0,
    expected_region_size: int = OFFLOAD_SHM_REGION_SIZE,
) -> OffloadReapResult:
    """Remove only authenticated, closed scratch regions from dead vLLM runs."""

    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise AssuranceRunError("vLLM offload scratch root is unsafe")
    root_info = root.stat()
    if root_info.st_uid != expected_root_uid or stat.S_IMODE(root_info.st_mode) != 0o1777:
        raise AssuranceRunError("vLLM offload scratch root has unexpected ownership or mode")
    root = root.resolve(strict=True)

    try:
        entries = list(os.scandir(root))
    except OSError as error:
        raise AssuranceRunError(f"cannot enumerate vLLM offload scratch root: {error}") from error
    candidates = sorted(
        entry.name
        for entry in entries
        if entry.name.startswith("vllm_offload_") and entry.name.endswith(".mmap")
    )
    if len(candidates) > OFFLOAD_SHM_MAX_STALE:
        raise AssuranceRunError("refusing to reap more than eight stale vLLM offload regions")
    for name in candidates:
        if _OFFLOAD_SHM_NAME.fullmatch(name) is None:
            raise AssuranceRunError(f"unexpected vLLM offload scratch name: {name}")
    if not candidates:
        return OffloadReapResult(count=0, bytes=0, names=())

    root_descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    try:
        identities: dict[str, tuple[int, int, int, int, int, int]] = {}
        total_bytes = 0
        for name in candidates:
            try:
                info = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
            except OSError as error:
                raise AssuranceRunError(
                    f"cannot authenticate vLLM offload scratch {name}: {error}"
                ) from error
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_nlink != 1
                or info.st_size not in {0, expected_region_size}
            ):
                raise AssuranceRunError(
                    f"vLLM offload scratch is not an expected owner-only region: {name}"
                )
            identities[name] = (
                info.st_dev,
                info.st_ino,
                info.st_mode,
                info.st_uid,
                info.st_nlink,
                info.st_size,
            )
            total_bytes += info.st_size

        candidate_paths = {str(root / name) for name in candidates}
        try:
            process_entries = list(proc_root.iterdir())
        except OSError as error:
            raise AssuranceRunError(f"cannot enumerate process table: {error}") from error
        for process_dir in process_entries:
            if not process_dir.name.isdigit():
                continue
            try:
                process_info = process_dir.stat(follow_symlinks=False)
            except OSError:
                continue
            if not stat.S_ISDIR(process_info.st_mode) or process_info.st_uid != os.getuid():
                continue
            fd_dir = process_dir / "fd"
            try:
                descriptors = list(fd_dir.iterdir())
            except (FileNotFoundError, PermissionError):
                # Some same-UID sandboxed/non-dumpable desktop processes deny
                # fd-table reads. The caller separately proves no vLLM API or
                # EngineCore is live; mirror the launcher's treatment here.
                continue
            except OSError as error:
                raise AssuranceRunError(
                    f"cannot inspect file descriptors for PID {process_dir.name}: {error}"
                ) from error
            for descriptor in descriptors:
                try:
                    target = str(descriptor.readlink())
                except OSError:
                    continue
                for candidate_path in candidate_paths:
                    if target in {candidate_path, f"{candidate_path} (deleted)"}:
                        raise AssuranceRunError(
                            f"vLLM offload scratch is still open: {candidate_path}"
                        )

        # Reauthenticate every directory entry after the process-table scan so a
        # replaced name cannot be unlinked under the identity checked above.
        for name in candidates:
            info = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
            observed = (
                info.st_dev,
                info.st_ino,
                info.st_mode,
                info.st_uid,
                info.st_nlink,
                info.st_size,
            )
            if observed != identities[name]:
                raise AssuranceRunError(f"vLLM offload scratch changed during validation: {name}")

        for name in candidates:
            os.unlink(name, dir_fd=root_descriptor)
        os.fsync(root_descriptor)
        for name in candidates:
            try:
                os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise AssuranceRunError(f"failed to remove stale vLLM offload scratch: {name}")
    finally:
        os.close(root_descriptor)

    return OffloadReapResult(
        count=len(candidates),
        bytes=total_bytes,
        names=tuple(candidates),
    )


def _claim_directory(path: Path, label: str) -> None:
    try:
        path.mkdir(mode=0o700)
    except FileExistsError as error:
        raise AssuranceRunError(f"{label} already exists; this run is consumed") from error
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_status(status_path: Path, phase: str, **details: object) -> None:
    payload = (
        json.dumps(
            {"at": datetime.now(UTC).isoformat(), "phase": phase, **details},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    descriptor = os.open(
        status_path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _bounded_failure_log(
    path: Path, *, max_bytes: int = 2 * 1024 * 1024, include_tail: bool = True
) -> dict[str, object]:
    """Classify a bounded log tail without copying unbounded model output."""

    if path.is_symlink() or not path.is_file():
        return {"present": False}
    metadata = path.stat()
    start = max(0, metadata.st_size - max_bytes)
    with path.open("rb") as stream:
        stream.seek(start)
        payload = stream.read(max_bytes)
    text = payload.decode("utf-8", errors="replace")
    markers: dict[str, dict[str, object]] = {}
    exception_lines: list[dict[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        exception_match = _PYTHON_EXCEPTION_LINE.search(stripped)
        if exception_match is not None:
            exception_lines.append(
                {
                    "type": exception_match.group("type"),
                    "message": exception_match.group("message")[:1000],
                    "line": stripped[:1200],
                }
            )
        for name, pattern in _FAILURE_MARKERS:
            if pattern.search(line) is None:
                continue
            entry = markers.setdefault(name, {"count": 0, "last_example": ""})
            entry["count"] = int(entry["count"]) + 1
            entry["last_example"] = line.strip()[:300]
    result: dict[str, object] = {
        "present": True,
        "size_bytes": metadata.st_size,
        "tail_bytes_examined": len(payload),
        "tail_truncated": start > 0,
        "markers": markers,
    }
    if exception_lines:
        bounded_exceptions = exception_lines[-16:]
        result["exception_lines"] = bounded_exceptions
        non_wrapper_exceptions = [
            exception
            for exception in bounded_exceptions
            if exception["type"].rsplit(".", 1)[-1] not in _WRAPPER_EXCEPTION_TYPES
            and not any(
                pattern.search(exception["message"]) for pattern in _WRAPPER_EXCEPTION_MESSAGES
            )
        ]
        result["probable_root_cause"] = (
            non_wrapper_exceptions[-1] if non_wrapper_exceptions else bounded_exceptions[-1]
        )
    if include_tail:
        result["last_lines"] = [line.strip()[:300] for line in text.splitlines()[-12:]]
    return result


def _bounded_release_probe_log(
    path: Path, *, max_bytes: int = 2 * 1024 * 1024
) -> dict[str, object]:
    """Extract the final authenticated probe error without exposing raw events."""

    if path.is_symlink() or not path.is_file():
        return {"present": False}
    metadata = path.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
        return {"present": True, "trusted": False, "size_bytes": metadata.st_size}
    start = max(0, metadata.st_size - max_bytes)
    with path.open("rb") as stream:
        stream.seek(start)
        payload = stream.read(max_bytes)
    if start:
        separator = payload.find(b"\n")
        payload = b"" if separator < 0 else payload[separator + 1 :]

    final_error: dict[str, object] | None = None
    valid_records = 0
    invalid_records = 0
    for raw_line in payload.splitlines():
        try:
            record = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            invalid_records += 1
            continue
        if not isinstance(record, dict) or record.get("schema") != (
            "urn:qwen-r9700:quest-release-activation-probe:v1"
        ):
            invalid_records += 1
            continue
        valid_records += 1
        if record.get("event") != "probe_error":
            continue
        event_index = record.get("event_index")
        exception_type = record.get("exception_type")
        message = record.get("message")
        if (
            not isinstance(event_index, int)
            or isinstance(event_index, bool)
            or event_index <= 0
            or not isinstance(exception_type, str)
            or not exception_type
            or not isinstance(message, str)
        ):
            invalid_records += 1
            continue
        final_error = {
            "schema": record["schema"],
            "type": exception_type[:256],
            "message": message[:1000],
            "event_index": event_index,
            "event_sha256": hashlib.sha256(raw_line).hexdigest(),
        }
    result: dict[str, object] = {
        "present": True,
        "trusted": True,
        "size_bytes": metadata.st_size,
        "tail_bytes_examined": len(payload),
        "tail_truncated": start > 0,
        "valid_records": valid_records,
        "invalid_records": invalid_records,
    }
    if final_error is not None:
        result["last_probe_error"] = final_error
    return result


def _last_lifecycle_phase(path: Path) -> str | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[-1]) if lines else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    phase = record.get("phase") if isinstance(record, dict) else None
    return phase if isinstance(phase, str) else None


def _publish_failure_diagnostic(
    path: Path,
    *,
    error: BaseException,
    spec: RunSpec,
    process: subprocess.Popen[bytes] | None,
    command_environment: dict[str, str],
    release_probe_output: Path | None,
) -> None:
    """Publish a compact create-only failure report for immediate triage."""

    inspected = {
        "server": _bounded_failure_log(spec.qualification_dir / "server.log"),
        "runner_stdout": _bounded_failure_log(
            spec.qualification_dir / "runner.stdout", include_tail=False
        ),
        "runner_stderr": _bounded_failure_log(spec.qualification_dir / "runner.stderr"),
    }
    probable_root_cause: dict[str, object] | None = None
    for log_name in ("server", "runner_stderr", "runner_stdout"):
        candidate = inspected[log_name].get("probable_root_cause")
        if isinstance(candidate, dict):
            probable_root_cause = {"log": log_name, **candidate}
            break
    probe: dict[str, object] | None = None
    if release_probe_output is not None:
        probe = {
            "path": str(release_probe_output),
            **_bounded_release_probe_log(release_probe_output),
        }
        probe_error = probe.get("last_probe_error")
        if isinstance(probe_error, dict):
            probable_root_cause = {"log": "release_probe", **probe_error}
    payload = {
        "schema": FAILURE_DIAGNOSTIC_SCHEMA,
        "at": datetime.now(UTC).isoformat(),
        "artifact_mode": spec.artifact_mode,
        "request_id": spec.request.request_id,
        "error_type": type(error).__name__,
        "error": str(error),
        "process_status": process.poll() if process is not None else None,
        "last_completed_phase": _last_lifecycle_phase(spec.qualification_dir / "lifecycle.jsonl"),
        "active_paths": {
            "cached_gemm_selector": command_environment.get("QWEN_QUEST_CACHED_GEMM_SELECTOR"),
            "cached_gemm_cold_seed": command_environment.get("QWEN_QUEST_CACHED_GEMM_COLD_SEED"),
            "fixed_serial_conv_m8": command_environment.get("QWEN_GDN_FIXED_SLOT_SERIAL_CONV_M8"),
            "fixed_serial_conv_m8_crosscheck": command_environment.get(
                "QWEN_GDN_FIXED_SLOT_SERIAL_CONV_M8_CROSSCHECK"
            ),
            "greedy_m8_verifier": command_environment.get("QWEN_DFLASH_GREEDY_M8_VERIFIER"),
            "quest_m8_page_stripe": command_environment.get("QWEN_QUEST_M8_PAGE_STRIPE"),
            "serial_batched_gdn": command_environment.get("QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED"),
            "release_activation_probe": command_environment.get("QWEN_RELEASE_ACTIVATION_PROBE"),
        },
        "probable_root_cause": probable_root_cause,
        "logs": inspected,
        "release_probe": probe,
    }
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.write(
            descriptor,
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode() + b"\n",
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_runtime_observations(
    path: Path,
    *,
    spec: RunSpec,
    command_environment: dict[str, str],
) -> dict[str, object]:
    """Publish bounded success-path warnings so performance anomalies need no log grep."""

    logs = {
        "server": _bounded_failure_log(spec.qualification_dir / "server.log", include_tail=False),
        "runner_stdout": _bounded_failure_log(
            spec.qualification_dir / "runner.stdout", include_tail=False
        ),
        "runner_stderr": _bounded_failure_log(
            spec.qualification_dir / "runner.stderr", include_tail=False
        ),
    }
    marker_counts: dict[str, int] = {}
    for log in logs.values():
        markers = log.get("markers")
        if not isinstance(markers, dict):
            continue
        for name, marker in markers.items():
            if not isinstance(marker, dict):
                continue
            marker_counts[name] = marker_counts.get(name, 0) + int(marker.get("count", 0))
    payload = {
        "schema": RUNTIME_OBSERVATIONS_SCHEMA,
        "at": datetime.now(UTC).isoformat(),
        "artifact_mode": spec.artifact_mode,
        "request_id": spec.request.request_id,
        "active_paths": {
            "cached_gemm_selector": command_environment.get("QWEN_QUEST_CACHED_GEMM_SELECTOR"),
            "cached_gemm_cold_seed": command_environment.get("QWEN_QUEST_CACHED_GEMM_COLD_SEED"),
            "quest_m8_page_stripe": command_environment.get("QWEN_QUEST_M8_PAGE_STRIPE"),
            "serial_batched_gdn": command_environment.get("QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED"),
            "outer_stage_timing": command_environment.get("QWEN_OUTER_STAGE_TIMING"),
            "release_activation_probe": command_environment.get("QWEN_RELEASE_ACTIVATION_PROBE"),
        },
        "marker_counts": marker_counts,
        "logs": logs,
    }
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.write(
            descriptor,
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode() + b"\n",
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload


def _health(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=timeout) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _wait_health(process: subprocess.Popen[bytes], base_url: str, seconds: int) -> None:
    for _ in range(seconds * 4):
        if _health(base_url):
            return
        code = process.poll()
        if code is not None:
            raise AssuranceRunError(f"API exited before health with status {code}")
        time.sleep(0.25)
    raise AssuranceRunError(f"API did not become healthy within {seconds} seconds")


def _require_log_marker(
    process: subprocess.Popen[bytes], log_path: Path, marker: str, seconds: int = 5
) -> None:
    """Refuse inference unless the initialized worker published its exact readiness proof."""

    for _ in range(max(1, seconds * 20)):
        try:
            if marker in log_path.read_text(encoding="utf-8", errors="replace"):
                return
        except OSError:
            pass
        code = process.poll()
        if code is not None:
            raise AssuranceRunError(
                f"API exited before required runtime marker with status {code}: {marker}"
            )
        time.sleep(0.05)
    raise AssuranceRunError(f"required initialized-runtime marker is absent: {marker}")


def _wait_owned_api_identity(
    process: subprocess.Popen[bytes],
    expected_model: Path = TARGET_MODEL_CONFIG.parent,
    seconds: int = 5,
) -> None:
    for _ in range(seconds * 20):
        if process.poll() is not None:
            raise AssuranceRunError(
                f"API exited before identity validation with status {process.returncode}"
            )
        try:
            argv = (Path("/proc") / str(process.pid) / "cmdline").read_bytes().split(b"\0")
        except OSError:
            time.sleep(0.05)
            continue
        if (
            b"serve" in argv
            and os.fsencode(expected_model) in argv
            and any(argument.endswith(b"/vllm") for argument in argv)
        ):
            return
        time.sleep(0.05)
    raise AssuranceRunError("recorded PID did not become the pinned vLLM API process")


def _stop_owned_api(process: subprocess.Popen[bytes], status_path: Path) -> None:
    if process.poll() is not None:
        _append_status(status_path, "api_already_exited", status=process.returncode)
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired as error:
        raise AssuranceRunError(
            f"API PID {process.pid} did not exit after TERM; no stronger signal was sent"
        ) from error
    _append_status(status_path, "api_stopped", pid=process.pid, status=process.returncode)


def _wait_zero_commit_witness(
    *,
    stream: Path,
    runner: subprocess.Popen[bytes],
    api: subprocess.Popen[bytes],
    timeout: int,
) -> None:
    """Wait only for the fsynced event that terminates a capture-only c=0 request."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stream.is_symlink():
            raise AssuranceRunError("zero-commit witness path became a symlink")
        if stream.is_file():
            try:
                rows = stream.read_text(encoding="utf-8").splitlines()
                if len(rows) >= 2:
                    event_row = json.loads(rows[1])
                    if not isinstance(event_row, dict) or set(event_row) != {"event"}:
                        raise AssuranceRunError("zero-commit witness event envelope is invalid")
                    return
            except UnicodeDecodeError as error:
                raise AssuranceRunError("zero-commit witness is not UTF-8") from error
            except json.JSONDecodeError:
                # A concurrently written final line is not evidence yet. The
                # capture hook fsyncs the complete line before this can pass.
                pass
        if api.poll() is not None:
            raise AssuranceRunError(
                f"API exited before the zero-commit witness was published: {api.returncode}"
            )
        if runner.poll() is not None:
            raise AssuranceRunError(
                "request runner exited before the zero-commit witness was published: "
                f"{runner.returncode}"
            )
        time.sleep(0.05)
    raise AssuranceRunError("timed out waiting for the fsynced zero-commit witness")


def run_once(spec: RunSpec, *, startup_timeout: int = 90, preflight_only: bool = False) -> int:
    qualification_dir = _require_qualification_dir(spec.qualification_dir)
    _require_owned_regular(spec.command, "command")
    _require_owned_regular(spec.runner, "runner")
    _require_runner_contract(spec.runner)
    for name, binding in spec.bindings.items():
        _require_owned_regular(binding, f"binding {name}")
    _require_cross_bindings(spec.bindings)
    command_text = spec.command.path.read_text(encoding="utf-8")
    if len(command_text.splitlines()) != 1 or not command_text.startswith("exec /usr/bin/env "):
        raise AssuranceRunError(
            "command must start with exec /usr/bin/env for stable PID ownership"
        )
    command_environment = _command_environment(command_text)
    if spec.artifact_mode == "release":
        forbidden = (
            "QWEN_CODING_TURBO_ASSURANCE_DIAGNOSTIC",
            "QWEN_DFLASH_ASSURANCE_",
            "QWEN_ROUND_EQUIVALENCE_",
            "/dflash-lossless-assurance/capture_site",
        )
        for fragment in forbidden:
            if fragment in command_text:
                raise AssuranceRunError(
                    f"release command contains assurance-only fragment: {fragment}"
                )
    cached_gemm_cold_seed = command_environment.get("QWEN_QUEST_CACHED_GEMM_COLD_SEED", "0")
    if cached_gemm_cold_seed not in {"0", "1"}:
        raise AssuranceRunError("QWEN_QUEST_CACHED_GEMM_COLD_SEED must be 0 or 1")
    if (
        cached_gemm_cold_seed == "1"
        and command_environment.get("QWEN_QUEST_CACHED_GEMM_SELECTOR") != "1"
    ):
        raise AssuranceRunError(
            "QWEN_QUEST_CACHED_GEMM_COLD_SEED=1 requires QWEN_QUEST_CACHED_GEMM_SELECTOR=1"
        )
    layer_diagnostic_raw = command_environment.get(
        "QWEN_DFLASH_ASSURANCE_LAYER_DIAGNOSTIC_ONLY", "0"
    )
    if layer_diagnostic_raw not in {"0", "1"}:
        raise AssuranceRunError("QWEN_DFLASH_ASSURANCE_LAYER_DIAGNOSTIC_ONLY must be 0 or 1")
    layer_diagnostic_only = layer_diagnostic_raw == "1"
    non_promotable_raw = command_environment.get(
        "QWEN_DFLASH_ASSURANCE_NON_PROMOTABLE_DIAGNOSTIC", "0"
    )
    if non_promotable_raw not in {"0", "1"}:
        raise AssuranceRunError(
            "QWEN_DFLASH_ASSURANCE_NON_PROMOTABLE_DIAGNOSTIC must be 0 or 1"
        )
    non_promotable_diagnostic = non_promotable_raw == "1"
    if non_promotable_diagnostic and not layer_diagnostic_only:
        raise AssuranceRunError(
            "non-promotable diagnostic context requires layer-only capture"
        )
    diagnostic_runtime_files: dict[str, dict[str, str]] | None = None
    target_model_root = TARGET_MODEL_CONFIG.parent
    target_model_config = TARGET_MODEL_CONFIG
    target_model_config_sha256: str | None = None
    if non_promotable_diagnostic:
        diagnostic_positions = _selected_integers(
            command_environment,
            "QWEN_DFLASH_ASSURANCE_LAYER_POSITION",
            "QWEN_DFLASH_ASSURANCE_LAYER_POSITIONS",
        )
        try:
            diagnostic_prompt_tokens = int(
                command_environment["QWEN_DFLASH_ASSURANCE_PROMPT_TOKENS"]
            )
        except (KeyError, ValueError) as error:
            raise AssuranceRunError(
                "non-promotable diagnostic prompt-token boundary is invalid"
            ) from error
        target_only_diagnostic = (
            command_environment.get("QWEN_FIXED_SLOT_TARGET_ONLY_ORACLE", "0") == "1"
        )
        exact_limit = (
            max(diagnostic_positions)
            - diagnostic_prompt_tokens
            + 1
            + int(target_only_diagnostic)
        )
        if exact_limit <= 0 or spec.request.max_tokens != exact_limit:
            raise AssuranceRunError(
                "non-promotable diagnostic is not bounded to its final capture position"
            )
        diagnostic_runtime_files = _require_diagnostic_runtime_file_bindings(
            spec, command_environment
        )
        (
            target_model_root,
            target_model_config,
            target_model_config_sha256,
        ) = _diagnostic_target_model(spec)
    precommit_capture = command_environment.get("QWEN_ROUND_EQUIVALENCE_PRECOMMIT_OUTPUT")
    expected_precommit = qualification_dir / "capture" / "precommit.jsonl"
    if precommit_capture is not None and precommit_capture != str(expected_precommit):
        raise AssuranceRunError("precommit capture output is not bound to this run")
    forced_proposal = _round_equivalence_proposal(spec, command_environment)
    if forced_proposal is not None and precommit_capture is None:
        raise AssuranceRunError(
            "round-equivalence proposal requires the precommit canonical-state witness"
        )
    zero_commit_raw = command_environment.get("QWEN_ROUND_EQUIVALENCE_ZERO_COMMIT", "0")
    if zero_commit_raw not in {"0", "1"}:
        raise AssuranceRunError("QWEN_ROUND_EQUIVALENCE_ZERO_COMMIT must be 0 or 1")
    zero_commit_capture = zero_commit_raw == "1"
    if zero_commit_capture and (
        spec.artifact_mode != "assurance"
        or layer_diagnostic_only
        or precommit_capture is None
        or forced_proposal is None
    ):
        raise AssuranceRunError(
            "zero-commit mode requires a full speculative assurance run, precommit capture, "
            "and an authenticated forced proposal"
        )
    if spec.artifact_mode == "assurance":
        for binding_name, environment_name in RUNTIME_CAPTURE_SHA_ENVS.items():
            if command_environment.get(environment_name) != spec.bindings[binding_name].sha256:
                raise AssuranceRunError(
                    f"command runtime binding differs from authenticated {binding_name}"
                )
        assurance_capture_site, assurance_module_root = _require_assurance_artifact_bindings(
            spec.bindings
        )
        capture_runtime_environment = {
            environment_name: command_environment[environment_name]
            for environment_name in RUNTIME_CAPTURE_SHA_ENVS.values()
        }
        if diagnostic_runtime_files is not None:
            capture_runtime_environment.update(
                {
                    "QWEN_DFLASH_ASSURANCE_NON_PROMOTABLE_DIAGNOSTIC": "1",
                    RUNTIME_FILE_BINDINGS_ENV: command_environment[RUNTIME_FILE_BINDINGS_ENV],
                    RUNTIME_FILE_BINDINGS_SHA_ENV: command_environment[
                        RUNTIME_FILE_BINDINGS_SHA_ENV
                    ],
                }
            )
        _require_capture_runtime_provenance(
            spec.bindings["capture_hook"],
            launcher_sha256=spec.bindings["launcher"].sha256,
            launcher_path=spec.bindings["launcher"].path,
            consumer_transition_receipt=spec.bindings["state_consumer_transition_receipt"],
            runtime_binding_environment=capture_runtime_environment,
        )
        if not layer_diagnostic_only:
            _require_state_capture_provenance(spec.bindings)
    _require_fixed_serial_conv_preflight(
        spec,
        command_environment,
        spec.bindings["launcher"].path.read_text(encoding="utf-8"),
        target_model_config=target_model_config,
        target_model_config_sha256=target_model_config_sha256,
    )
    exact_k_runtime = _require_exact_k_preflight(spec, command_environment)
    _require_dflash_d7_graph_runtime_preflight(command_environment)
    _require_b7_position_stride_admission_preflight(
        spec,
        command_environment,
        spec.bindings["launcher"].path.read_text(encoding="utf-8"),
        command_text,
    )
    _require_b7_active_v2_transport_admission_preflight(
        spec,
        command_environment,
        spec.bindings["launcher"].path.read_text(encoding="utf-8"),
        command_text,
    )
    _require_quest_selector_extension_preflight(
        spec,
        command_environment,
        spec.bindings["launcher"].path.read_text(encoding="utf-8"),
    )
    _require_quest_tree_wmma_preflight(
        command_environment,
        spec.bindings["launcher"].path.read_text(encoding="utf-8"),
    )
    _require_fixed_slot_resume_admission_preflight(
        spec,
        command_environment,
        command_text,
    )
    if spec.request.fixture_dir.is_symlink() or not spec.request.fixture_dir.is_dir():
        raise AssuranceRunError("fixture directory must be a real directory")
    if _live_engine_processes():
        raise AssuranceRunError("a vLLM API or EngineCore process is already running")

    launcher_sha = spec.bindings["launcher"].sha256
    release_probe_output = (
        _release_activation_probe_output(command_text, qualification_dir)
        if spec.artifact_mode == "release"
        else None
    )
    tree_score_outputs = _tree_score_capture_outputs(command_text, qualification_dir, spec)
    if tree_score_outputs is not None and spec.artifact_mode != "release":
        raise AssuranceRunError("DFlash score-lattice capture requires a release artifact")
    required_fragments = [
        f"/vllm serve {target_model_root} ",
        "--host 127.0.0.1 --port 8000",
    ]
    if spec.artifact_mode == "assurance":
        state_exporter_parent = spec.bindings["state_exporter"].path.parent
        if state_exporter_parent.resolve(strict=True) != assurance_module_root:
            raise AssuranceRunError("state exporter is outside the authenticated module root")
        _require_command_module_identity(
            command_text, assurance_capture_site, assurance_module_root
        )
        required_fragments.extend(
            (
                f"QWEN_DFLASH_ASSURANCE_REQUEST_ID={spec.request.request_id}",
                f"QWEN_DFLASH_ASSURANCE_LAUNCHER_SHA256={launcher_sha}",
                f"QWEN_DFLASH_ASSURANCE_LAUNCHER_PATH={spec.bindings['launcher'].path}",
                f"QWEN_DFLASH_ASSURANCE_OUTPUT={qualification_dir}/capture/rounds.jsonl",
                f"QWEN_DFLASH_ASSURANCE_LAYER_OUTPUT={qualification_dir}/capture/layers.jsonl",
                f"PYTHONPATH={assurance_capture_site}:{state_exporter_parent}:",
                "QWEN_CODING_TURBO_STATE_PRODUCER_RECEIPT="
                f"{spec.bindings['state_producer_receipt'].path}",
                "QWEN_CODING_TURBO_STATE_CONSUMER_TRANSITION_RECEIPT="
                f"{spec.bindings['state_consumer_transition_receipt'].path}",
                f"QWEN_CODING_TURBO_ASSURANCE_MANIFEST={spec.bindings['assurance_manifest'].path}",
                "QWEN_CODING_TURBO_SNAPSHOT_SELECTION="
                f"{spec.bindings['snapshot_payload_selection'].path}",
                "QWEN_CODING_TURBO_ASSURANCE_DIAGNOSTIC=1",
                "QWEN_DFLASH_ASSURANCE_CAPTURE=1",
            )
        )
        if layer_diagnostic_only:
            required_fragments.append("QWEN_DFLASH_ASSURANCE_LAYER_DIAGNOSTIC_ONLY=1")
            if non_promotable_diagnostic:
                required_fragments.extend(
                    (
                        "QWEN_DFLASH_ASSURANCE_NON_PROMOTABLE_DIAGNOSTIC=1",
                        f"{RUNTIME_FILE_BINDINGS_ENV}="
                        f"{qualification_dir}/{RUNTIME_FILE_BINDINGS_NAME}",
                        f"{RUNTIME_FILE_BINDINGS_SHA_ENV}="
                        f"{command_environment[RUNTIME_FILE_BINDINGS_SHA_ENV]}",
                    )
                )
            for forbidden in (
                "QWEN_DFLASH_ASSURANCE_DRAFT_OUTPUT=",
                "QWEN_CODING_TURBO_ORACLE_STATE_OUTPUT=",
                "QWEN_ROUND_EQUIVALENCE_PRECOMMIT_OUTPUT=",
            ):
                if forbidden in command_text:
                    raise AssuranceRunError(
                        f"layer-only diagnostic contains expensive full capture: {forbidden}"
                    )
        else:
            required_fragments.extend(
                (
                    f"QWEN_DFLASH_ASSURANCE_DRAFT_OUTPUT={qualification_dir}/capture/draft.jsonl",
                    "QWEN_CODING_TURBO_ORACLE_STATE_OUTPUT="
                    f"{qualification_dir}/capture/states.jsonl",
                )
            )
            if forced_proposal is not None:
                required_fragments.extend(
                    (
                        "QWEN_ROUND_EQUIVALENCE_PROPOSAL_INPUT="
                        f"{qualification_dir}/round-equivalence-proposal.json",
                        "QWEN_ROUND_EQUIVALENCE_PROPOSAL_SHA256="
                        f"{_sha256_file(qualification_dir / 'round-equivalence-proposal.json')}",
                    )
                )
            if zero_commit_capture:
                required_fragments.append("QWEN_ROUND_EQUIVALENCE_ZERO_COMMIT=1")
    for fragment in required_fragments:
        if fragment not in command_text:
            raise AssuranceRunError(f"command is not bound to this run: missing {fragment}")

    if spec.artifact_mode == "assurance":
        capture_dir = qualification_dir / "capture"
        if capture_dir.is_symlink() or not capture_dir.is_dir():
            raise AssuranceRunError("capture directory must be a real directory")
    output_paths = [
        qualification_dir / "api.pid",
        qualification_dir / "server.log",
        qualification_dir / "runner.stdout",
        qualification_dir / "runner.stderr",
        qualification_dir / "request.claimed",
        qualification_dir / "capture" / "oracle-source.json",
        qualification_dir / "failure-diagnostic.json",
        qualification_dir / "runtime-observations.json",
        spec.request.output,
    ]
    if release_probe_output is not None:
        output_paths.append(release_probe_output)
    if tree_score_outputs is not None:
        output_paths.extend(tree_score_outputs)
    if precommit_capture is not None:
        output_paths.extend(
            (
                expected_precommit,
                qualification_dir / "capture" / "precommit-source.json",
            )
        )
    for path in output_paths:
        if path.exists() or path.is_symlink():
            raise AssuranceRunError(f"run output already exists: {path}")
    if preflight_only:
        return 0

    stale_regions = _reap_stale_offload_regions()
    if _live_engine_processes():
        raise AssuranceRunError("a vLLM API or EngineCore process started during scratch cleanup")

    start_claim = qualification_dir / "start.claimed"
    _claim_directory(start_claim, "start claim")
    status_path = qualification_dir / "lifecycle.jsonl"
    _append_status(
        status_path,
        "validated",
        spec_sha256=None,
        stale_offload_regions_removed=stale_regions.count,
        stale_offload_bytes_removed=stale_regions.bytes,
        stale_offload_names=list(stale_regions.names),
    )

    log_descriptor = os.open(
        qualification_dir / "server.log",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            ["bash", str(spec.command.path)],
            stdin=subprocess.DEVNULL,
            stdout=log_descriptor,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        os.close(log_descriptor)
    _append_status(status_path, "api_started", pid=process.pid)
    pid_path = qualification_dir / "api.pid"
    pid_descriptor = os.open(pid_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(pid_descriptor, f"{process.pid}\n".encode())
        os.fsync(pid_descriptor)
    finally:
        os.close(pid_descriptor)

    try:
        _wait_owned_api_identity(process, target_model_root)
        _append_status(status_path, "api_identity_validated", pid=process.pid)
        _wait_health(process, spec.request.base_url, startup_timeout)
        _append_status(status_path, "api_healthy", pid=process.pid)
        if exact_k_runtime:
            _require_log_marker(
                process,
                qualification_dir / "server.log",
                EXACT_K_RUNTIME_MARKER,
            )
            _append_status(status_path, "exact_k_runtime_ready", pid=process.pid)
        if spec.artifact_mode == "assurance" and not layer_diagnostic_only:
            _require_log_marker(
                process,
                qualification_dir / "server.log",
                STATE_EXPORTER_PREFLIGHT_MARKER,
            )
            _append_status(status_path, "state_exporter_preflight_ready", pid=process.pid)
        runner_command = [
            sys.executable,
            str(spec.runner.path),
            "--base-url",
            spec.request.base_url,
            "--fixture-dir",
            str(spec.request.fixture_dir),
            "--output",
            str(spec.request.output),
            "--request-id",
            spec.request.request_id,
            "--timeout",
            str(spec.request.timeout),
            "--max-tokens",
            str(spec.request.max_tokens),
            "--single-request-claim",
            str(qualification_dir / "request.claimed"),
            "--cache-session-id",
            spec.request.cache_session_id,
            "--cache-branch",
            spec.request.cache_branch,
            "--cache-manifest-sha256",
            spec.request.cache_manifest_sha256,
        ]
        stdout_path = qualification_dir / "runner.stdout"
        stderr_path = qualification_dir / "runner.stderr"
        if zero_commit_capture:
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                runner = subprocess.Popen(
                    runner_command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                _wait_zero_commit_witness(
                    stream=expected_precommit,
                    runner=runner,
                    api=process,
                    timeout=spec.request.timeout,
                )
                if runner.poll() is None:
                    runner.send_signal(signal.SIGTERM)
                    try:
                        runner.wait(timeout=30)
                    except subprocess.TimeoutExpired as error:
                        raise AssuranceRunError(
                            "zero-commit request runner did not stop after TERM; "
                            "no stronger signal sent"
                        ) from error
            _append_status(
                status_path,
                "zero_commit_request_cancelled_after_witness",
                status=runner.returncode,
            )
            _stop_owned_api(process, status_path)
        else:
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                completed = subprocess.run(
                    runner_command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=spec.request.timeout + 30,
                    check=False,
                )
            _append_status(status_path, "request_finished", status=completed.returncode)
            if completed.returncode != 0:
                raise AssuranceRunError(f"request runner failed with status {completed.returncode}")
            if spec.request.output.is_symlink() or not spec.request.output.is_file():
                raise AssuranceRunError("request runner did not publish its result")
        if release_probe_output is not None:
            release_probe_event_counts = _require_release_activation_probe_events(
                release_probe_output, command_text
            )
            _append_status(
                status_path,
                "release_activation_probe_reconciled",
                event_counts=release_probe_event_counts,
                cached_gemm_selector=command_environment.get("QWEN_QUEST_CACHED_GEMM_SELECTOR"),
                event_stream=str(release_probe_output),
            )
        if tree_score_outputs is not None:
            _stop_owned_api(process, status_path)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not all(
                path.is_file() for path in tree_score_outputs
            ):
                time.sleep(0.05)
            tree_capture = _require_tree_score_capture_artifacts(*tree_score_outputs)
            _append_status(status_path, "dflash_score_lattice_reconciled", **tree_capture)
        if spec.artifact_mode == "assurance":
            precommit_source = _reconcile_precommit_capture(spec, command_environment)
            oracle_source = None
            if not zero_commit_capture:
                oracle_source = (
                    _reconcile_layer_diagnostic(spec, command_environment)
                    if layer_diagnostic_only
                    else _reduce_authenticated_capture(spec)
                )
            _append_status(
                status_path,
                "evidence_reconciled",
                oracle_source_sha256=(
                    _sha256_file(oracle_source) if oracle_source is not None else None
                ),
                precommit_source_sha256=(
                    _sha256_file(precommit_source) if precommit_source is not None else None
                ),
            )
        runtime_observations = qualification_dir / "runtime-observations.json"
        runtime_observation_record = _publish_runtime_observations(
            runtime_observations,
            spec=spec,
            command_environment=command_environment,
        )
        _append_status(
            status_path,
            "runtime_observations_published",
            path=str(runtime_observations),
            sha256=_sha256_file(runtime_observations),
        )
        marker_counts = runtime_observation_record.get("marker_counts")
        fatal_markers = (
            sorted(set(marker_counts) & FATAL_RUNTIME_MARKERS)
            if isinstance(marker_counts, dict)
            else []
        )
        if fatal_markers:
            raise AssuranceRunError(
                "runtime log contains fatal diagnostic markers: " + ", ".join(fatal_markers)
            )
        return 0
    except Exception as error:
        diagnostic_path = qualification_dir / "failure-diagnostic.json"
        published_diagnostic = False
        try:
            _publish_failure_diagnostic(
                diagnostic_path,
                error=error,
                spec=spec,
                process=process,
                command_environment=command_environment,
                release_probe_output=release_probe_output,
            )
            _append_status(
                status_path,
                "run_failed",
                error_type=type(error).__name__,
                error=str(error),
                diagnostic=str(diagnostic_path),
                diagnostic_sha256=_sha256_file(diagnostic_path),
            )
            published_diagnostic = True
        except Exception as diagnostic_error:
            _append_status(
                status_path,
                "run_failed_diagnostic_error",
                error_type=type(error).__name__,
                error=str(error),
                diagnostic_error_type=type(diagnostic_error).__name__,
                diagnostic_error=str(diagnostic_error),
            )
        if published_diagnostic:
            raise AssuranceRunError(f"{error}; structured diagnostic: {diagnostic_path}") from error
        raise
    finally:
        _stop_owned_api(process, status_path)
        remaining = _live_engine_processes()
        if remaining:
            raise AssuranceRunError(
                f"vLLM processes remain after API shutdown; scratch was not touched: {remaining}"
            )
        stopped_regions = _reap_stale_offload_regions()
        _append_status(
            status_path,
            "post_stop_scratch_reaped",
            stale_offload_regions_removed=stopped_regions.count,
            stale_offload_bytes_removed=stopped_regions.bytes,
            stale_offload_names=list(stopped_regions.names),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "NAME\n"
            "    qwen-assurance-once - execute one manifest-bound assurance request\n\n"
            "SYNOPSIS\n"
            "    qwen-assurance-once --spec FILE [--startup-timeout SECONDS] "
            "[--preflight-only]\n\n"
            "DESCRIPTION\n"
            "    Validate and execute exactly one manifest-bound Qwen assurance process "
            "and request."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "OPERATION\n"
            "    Refuses drift, reused outputs, live vLLM processes, unstable PID commands,\n"
            "    incomplete runtime bindings, and second requests before contacting the API.\n\n"
            "EXAMPLES\n"
            "    qwen-assurance-once --spec /absolute/qualification/one-shot.json\n\n"
            "FILES\n"
            "    The JSON spec names the command, runner, runtime bindings, qualification\n"
            "    directory and request.\n\n"
            "PATHS\n"
            "    Command, result, capture, lifecycle and claim paths are confined to the\n"
            "    qualification directory.\n\n"
            "SECURITY NOTES\n"
            "    No privilege escalation is used. Mutable evidence is owner-only and\n"
            "    create-only.\n\n"
            "EXIT STATUS\n"
            "    0 on a completed request and graceful shutdown; 1 on any fail-closed\n"
            "    violation.\n\n"
            "AUTHORS\n"
            "    Qwen R9700 inference lab maintainers"
        ),
    )
    parser._optionals.title = "OPTIONS"
    parser.add_argument("--spec", required=True, type=Path, help="absolute one-shot JSON spec")
    parser.add_argument("--startup-timeout", type=int, default=90)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate every static source/output contract without starting the model",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec = load_spec(args.spec)
        if args.startup_timeout <= 0:
            raise AssuranceRunError("startup timeout must be positive")
        return run_once(
            spec,
            startup_timeout=args.startup_timeout,
            preflight_only=args.preflight_only,
        )
    except (AssuranceRunError, OSError, subprocess.SubprocessError) as error:
        print(f"qwen-assurance-once: {error}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
