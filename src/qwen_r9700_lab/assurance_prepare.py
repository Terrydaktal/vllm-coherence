"""Construct one self-contained, create-only Qwen assurance run bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import qwen_r9700_lab.assurance_once as assurance_once_module
from qwen_r9700_lab.assurance_once import (
    ARTIFACT_SCHEMA,
    ASSURANCE_BINDINGS,
    FULL_INSTRUMENTATION_ARTIFACT_BINDINGS,
    FULL_INSTRUMENTATION_BINDINGS,
    RELEASE_BINDINGS,
    SCHEMA,
    has_full_instrumentation_contract,
)
from qwen_r9700_lab.coding_turbo_artifacts import verify_artifacts
from qwen_r9700_lab.coding_turbo_oracle import OracleSafetyError, _validate_identity
from qwen_r9700_lab.coding_turbo_state_provider import (
    StateProviderError,
    validate_consumer_transition_receipt,
    validate_producer_receipt,
)
from qwen_r9700_lab.round_equivalence import (
    RoundEquivalenceError,
    normalize_forced_proposal,
)


class PrepareError(RuntimeError):
    """A bundle input cannot produce a deterministic valid assurance run."""


ARTIFACT_OUTPUTS = {
    "capture_hook": "assurance/capture_hook.py",
    "capture_reducer": "assurance/qwen_r9700_lab/coding_turbo_capture.py",
    "capture_site": "assurance/capture_site/sitecustomize.py",
    "oracle_contract": "assurance/qwen_r9700_lab/coding_turbo_oracle.py",
    "round_equivalence_contract": "assurance/qwen_r9700_lab/round_equivalence.py",
    "state_exporter": "assurance/coding_turbo_state_exporter.py",
    "state_provider": "assurance/qwen_r9700_lab/coding_turbo_state_provider.py",
}


def _artifact_outputs(semantic_contract: object) -> dict[str, str]:
    outputs = dict(ARTIFACT_OUTPUTS)
    if has_full_instrumentation_contract(semantic_contract):
        outputs.update(FULL_INSTRUMENTATION_ARTIFACT_BINDINGS)
    return outputs


ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ORACLE_IDENTITY_NAME = "oracle-identity.json"
NON_PROMOTABLE_DIAGNOSTIC_SCHEMA = "urn:qwen-r9700:layer-diagnostic-identity:v1"
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
RUNTIME_REMOTE_ROOT = PurePosixPath(
    "/home/lewis/.local/share/qwen-r9700/runtimes/vllm-rocm/.venv/"
    "lib/python3.12/site-packages/vllm"
)
GEMMA_RMS_FP32_EXTENSION = (
    "/home/lewis/.local/share/qwen-r9700/kernels/gemma-rmsnorm-fp32/mixed_gemma_rms_gfx1201_v1.so"
)
RUNTIME_SOURCE_BINDINGS = (
    "qwen_gdn_linear_attn",
    "qwen3_next",
    "qwen_text_rope",
    "snapshot_selection",
)
RUNTIME_CAPTURE_SHA_ENVS = {
    "qwen_gdn_linear_attn": "QWEN_DFLASH_ASSURANCE_GDN_SHA256",
    "qwen3_next": "QWEN_DFLASH_ASSURANCE_QWEN3_NEXT_SHA256",
    "qwen_text_rope": "QWEN_DFLASH_ASSURANCE_TEXT_ROPE_SHA256",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise PrepareError(f"cannot inspect {label}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise PrepareError(f"{label} must be an owned regular non-symlink file")
    return path.resolve(strict=True)


def _runtime_binding_records(
    bindings: dict[str, Any],
    explicit_sources: dict[str, Path],
    *,
    allow_changes: bool,
) -> dict[str, dict[str, str]]:
    """Validate every installed-runtime binding and report all source drift at once."""

    unknown = set(explicit_sources) - set(RUNTIME_SOURCE_BINDINGS)
    if unknown:
        raise PrepareError(
            "runtime binding sources contain unknown bindings: " + ", ".join(sorted(unknown))
        )
    records: dict[str, dict[str, str]] = {}
    mismatches: list[str] = []
    for name in RUNTIME_SOURCE_BINDINGS:
        inherited = bindings.get(name)
        if not isinstance(inherited, dict) or not isinstance(inherited.get("path"), str):
            raise PrepareError(f"template spec lacks the runtime binding {name}")
        runtime_source = _regular(
            explicit_sources.get(name, Path(inherited["path"])),
            f"runtime binding {name}",
        )
        runtime_sha256 = _sha256(runtime_source)
        inherited_sha256 = inherited.get("sha256")
        if name in explicit_sources and not allow_changes and runtime_sha256 != inherited_sha256:
            mismatches.append(f"{name}: expected {inherited_sha256}, observed {runtime_sha256}")
        records[name] = {
            "path": inherited["path"],
            "sha256": runtime_sha256,
        }
    if mismatches:
        raise PrepareError(
            "runtime binding sources differ from the template; all mismatches: "
            + "; ".join(mismatches)
            + "; use fetched runtime postimages or explicitly authorize one coordinated runtime "
            "binding change"
        )
    return records


def _directory(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise PrepareError(f"cannot inspect {label}: {error}") from error
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise PrepareError(f"{label} must be an owned real directory")
    return path.resolve(strict=True)


def _json(path: Path, label: str) -> dict[str, Any]:
    _regular(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PrepareError(f"{label} is invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise PrepareError(f"{label} must be a JSON object")
    return value


def _runtime_file_binding_table(path: Path) -> tuple[Path, dict[str, Any], str]:
    """Validate one explicit complete runtime identity table without ambient lookup."""

    source = _regular(path, "runtime file binding table")
    document = _json(source, "runtime file binding table")
    if (
        set(document) != {"schema", "files"}
        or document.get("schema") != RUNTIME_FILE_BINDINGS_SCHEMA
    ):
        raise PrepareError("runtime file binding table identity is invalid")
    files = document.get("files")
    if not isinstance(files, dict) or set(files) != RUNTIME_FILE_LABELS:
        raise PrepareError(
            "runtime file binding table labels differ: expected="
            + repr(sorted(RUNTIME_FILE_LABELS))
        )
    normalized: dict[str, dict[str, str]] = {}
    for label in sorted(RUNTIME_FILE_LABELS):
        entry = files[label]
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise PrepareError(f"runtime file binding {label} is malformed")
        raw_path = entry.get("path")
        digest = entry.get("sha256")
        if not isinstance(raw_path, str) or not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
            raise PrepareError(f"runtime file binding {label} path or SHA256 is invalid")
        runtime_path = PurePosixPath(raw_path)
        if not runtime_path.is_absolute() or any(
            part in {"", ".", ".."} for part in runtime_path.parts
        ):
            raise PrepareError(f"runtime file binding {label} path is not normalized")
        try:
            runtime_path.relative_to(RUNTIME_REMOTE_ROOT)
        except ValueError as error:
            raise PrepareError(
                f"runtime file binding {label} is outside {RUNTIME_REMOTE_ROOT}"
            ) from error
        normalized[label] = {"path": str(runtime_path), "sha256": str(digest)}
    normalized_document = {"schema": RUNTIME_FILE_BINDINGS_SCHEMA, "files": normalized}
    if document != normalized_document:
        raise PrepareError("runtime file binding table is not canonical by value")
    return source, normalized_document, _sha256(source)


def _remote_path(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    allowed = PurePosixPath("/home/lewis/projects/qwen-r9700/artifacts/qualifications")
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PrepareError(f"{label} must be a normalized absolute path")
    try:
        path.relative_to(allowed)
    except ValueError as error:
        raise PrepareError(f"{label} must be below {allowed}") from error
    if path == allowed:
        raise PrepareError(f"{label} must not equal {allowed}")
    return path


def _extension_binding(
    path: str | None, sha256: str | None, label: str
) -> tuple[PurePosixPath, str] | None:
    if bool(path) != bool(sha256):
        raise PrepareError(f"{label} path and SHA256 must be supplied together")
    if path is None or sha256 is None:
        return None
    remote_path = _remote_path(path, f"{label} path")
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise PrepareError(f"{label} SHA256 must be 64 lowercase hex characters")
    return remote_path, sha256


def _manifest_members(
    artifact_root: Path, artifact_mode: str = "assurance"
) -> tuple[Path, dict[str, str], dict[str, str]]:
    verify_artifacts(artifact_root)
    if artifact_mode not in {"assurance", "release"}:
        raise PrepareError("artifact mode must be assurance or release")
    manifest_path = artifact_root / artifact_mode / "manifest.json"
    manifest = _json(manifest_path, f"{artifact_mode} manifest")
    if manifest.get("schema") != ARTIFACT_SCHEMA or manifest.get("artifact_kind") != artifact_mode:
        raise PrepareError(f"{artifact_mode} manifest identity is invalid")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise PrepareError("assurance manifest file table is invalid")
    members: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise PrepareError(f"{artifact_mode} manifest contains a malformed entry")
        output = entry.get("path")
        digest = entry.get("sha256")
        if not isinstance(output, str) or not isinstance(digest, str) or output in members:
            raise PrepareError(f"{artifact_mode} manifest paths or hashes are invalid")
        members[output] = digest
    artifact_outputs = _artifact_outputs(manifest.get("semantic_contract"))
    for output in artifact_outputs.values():
        source = artifact_root / artifact_mode / "files" / output
        if members.get(output) != _sha256(_regular(source, output)):
            raise PrepareError(f"{artifact_mode} artifact member is invalid: {output}")
    return manifest_path, members, artifact_outputs


def _rewrite_command(
    payload: str,
    *,
    remote_qualification: PurePosixPath,
    old_qualification: str,
    request_id: str,
    prompt_tokens: int,
    capture_rounds: int,
    launcher_sha256: str,
    force_target_only: bool = False,
    layer_position: int | None = None,
    layer_positions: tuple[int, ...] | None = None,
    layer_indices: tuple[int, ...] | None = None,
    layer_diagnostic_only: bool = False,
    allow_unqualified_layer_diagnostic_context: bool = False,
    runtime_file_bindings_path: PurePosixPath | None = None,
    runtime_file_bindings_sha256: str | None = None,
    state_committed_counts: tuple[int, ...] | None = None,
    round_equivalence_precommit: bool = False,
    round_equivalence_proposal_sha256: str | None = None,
    round_equivalence_zero_commit: bool = False,
    gemma_rms_fp32_kernel: bool = False,
    cached_accepted_commit: bool = False,
    fixed_serial_conv_m8: bool = False,
    fixed_serial_conv_m8_crosscheck: bool = False,
    greedy_m8_verifier: bool = False,
    serial_batched: bool = False,
    grouped_ba_prefix: bool = False,
    grouped_ba_crosscheck: bool = False,
    batched_ba: bool = False,
    batched_ba_crosscheck: bool = False,
    serial_reference_conv: bool = False,
    quest_m8_page_stripe: bool = False,
    cached_gemm_selector: bool = False,
    cached_gemm_cold_seed: bool = False,
    d7_c1_draft_graph: bool = False,
    enable_outer_stage_timing: bool = False,
    disable_outer_stage_timing: bool = False,
    proposal_temperature_scale: str | None = None,
    quest_selector_extension: PurePosixPath | None = None,
    quest_selector_extension_sha256: str | None = None,
    row_local_dualphase_extension: PurePosixPath | None = None,
    row_local_dualphase_extension_sha256: str | None = None,
    release_probe_root: PurePosixPath | None = None,
    release_probe_sha256: str | None = None,
    release_probe_site_sha256: str | None = None,
    tree_score_capture_root: PurePosixPath | None = None,
    tree_score_capture_hashes: dict[str, str] | None = None,
    tree_public_manifest_sha256: str | None = None,
    artifact_mode: str = "assurance",
    runtime_launcher_path: PurePosixPath | None = None,
    runtime_binding_sha256: dict[str, str] | None = None,
) -> str:
    if len(payload.splitlines()) != 1:
        raise PrepareError("template command must be exactly one line")
    try:
        arguments = shlex.split(payload)
    except ValueError as error:
        raise PrepareError(f"template command quoting is invalid: {error}") from error
    if arguments[:3] != ["exec", "/usr/bin/env", "-i"]:
        raise PrepareError("template command must begin with exec /usr/bin/env -i")
    template_environment: dict[str, str] = {}
    for argument in arguments[3:]:
        if "=" not in argument:
            break
        name, value = argument.split("=", 1)
        if not ENVIRONMENT_NAME.fullmatch(name):
            break
        template_environment.setdefault(name, value)
    best_first_b7_raw = template_environment.get("QWEN_DFLASH2_BEST_FIRST_B7_REQUIRED", "0")
    if best_first_b7_raw not in {"0", "1"}:
        raise PrepareError("QWEN_DFLASH2_BEST_FIRST_B7_REQUIRED must be 0 or 1")
    best_first_b7 = best_first_b7_raw == "1"
    if isinstance(prompt_tokens, bool) or prompt_tokens <= 0:
        raise PrepareError("assurance prompt tokens must be a positive integer")
    if isinstance(capture_rounds, bool) or capture_rounds <= 0:
        raise PrepareError("assurance capture rounds must be a positive integer")
    if serial_reference_conv and not serial_batched:
        raise PrepareError("serial reference convolution requires serial-batched GDN")
    if fixed_serial_conv_m8 and not cached_accepted_commit:
        raise PrepareError("fixed serial M8 convolution requires cached accepted commit")
    if fixed_serial_conv_m8 and force_target_only:
        raise PrepareError("fixed serial M8 convolution requires speculative M8 execution")
    if fixed_serial_conv_m8 and serial_reference_conv:
        raise PrepareError(
            "fixed serial M8 and serial reference convolution are mutually exclusive"
        )
    if fixed_serial_conv_m8_crosscheck and not fixed_serial_conv_m8:
        raise PrepareError("fixed serial M8 crosscheck requires fixed serial M8 convolution")
    if greedy_m8_verifier and force_target_only:
        raise PrepareError("greedy M8 verifier requires speculative DFlash execution")
    if grouped_ba_crosscheck and not grouped_ba_prefix:
        raise PrepareError("grouped B/A crosscheck requires grouped B/A prefix execution")
    if batched_ba_crosscheck and not batched_ba:
        raise PrepareError("batched B/A crosscheck requires batched B/A execution")
    if batched_ba and grouped_ba_prefix:
        raise PrepareError("batched and grouped B/A execution are mutually exclusive")
    if cached_gemm_cold_seed and not cached_gemm_selector:
        raise PrepareError("cached-GEMM cold seeding requires the cached-GEMM selector")
    if d7_c1_draft_graph and force_target_only:
        raise PrepareError("D7/C1 draft graph requires speculative DFlash execution")
    if d7_c1_draft_graph and not greedy_m8_verifier:
        raise PrepareError("D7/C1 draft graph requires the exact greedy M8 verifier")
    if d7_c1_draft_graph and not (fixed_serial_conv_m8 or best_first_b7):
        raise PrepareError("D7/C1 draft graph requires exact fixed serial M8 convolution")
    if enable_outer_stage_timing and disable_outer_stage_timing:
        raise PrepareError("outer stage timing cannot be both enabled and disabled")
    if proposal_temperature_scale is not None:
        if force_target_only:
            raise PrepareError("proposal-temperature scale requires speculative DFlash execution")
        if not re.fullmatch(r"([0-9]+([.][0-9]*)?|[.][0-9]+)", proposal_temperature_scale):
            raise PrepareError("proposal-temperature scale must be a positive decimal")
        if re.fullmatch(r"0*([.]0*)?", proposal_temperature_scale):
            raise PrepareError("proposal-temperature scale must be greater than zero")

    remote = str(remote_qualification)
    if artifact_mode not in {"assurance", "release"}:
        raise PrepareError("artifact mode must be assurance or release")
    if fixed_serial_conv_m8_crosscheck and artifact_mode != "assurance":
        raise PrepareError("fixed serial M8 crosscheck requires the assurance artifact")
    artifact = f"{remote}/artifact-pair/{artifact_mode}"
    capture_site = f"{artifact}/files/assurance/capture_site"
    module_root = f"{artifact}/files/assurance"
    runtime_backends = f"{artifact}/files/vllm/v1/attention/backends"
    probe_names = {
        "QWEN_RELEASE_ACTIVATION_PROBE",
        "QWEN_RELEASE_ACTIVATION_PROBE_OUTPUT",
        "QWEN_RELEASE_ACTIVATION_PROBE_SHA256",
        "QWEN_RELEASE_ACTIVATION_PROBE_SITE_SHA256",
        "QWEN_RELEASE_ACTIVATION_PROBE_GPU_TIMING",
        "QWEN_RELEASE_ACTIVATION_PROBE_UNION_STATS",
        "QWEN_RELEASE_ACTIVATION_PROBE_REQUIRED_EVENTS",
    }
    tree_capture_names = {
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
    replacements = {
        "QWEN_CODING_TURBO_QUEST96": "1",
        "QWEN_DFLASH_WHOLE_MODEL_ACCEPTED_REPLAY": "1",
        "QWEN_FIXED_SLOT_TARGET_ONLY_ORACLE": "0",
        "QWEN_GDN_BA_SERIAL_ROW_EXACT": "1",
        "QWEN_LM_HEAD_SERIAL_M8": "1",
        "QWEN_LM_HEAD_PREFIX_SERIAL_M8": ("0" if force_target_only or best_first_b7 else "1"),
        "QWEN_GDN_FIXED_SLOT_CACHED_ACCEPTED_COMMIT": ("1" if cached_accepted_commit else "0"),
        "QWEN_GDN_FIXED_SLOT_SERIAL_CONV_M8": "1" if fixed_serial_conv_m8 else "0",
        "QWEN_GDN_FIXED_SLOT_SERIAL_CONV_M8_CROSSCHECK": (
            "1" if fixed_serial_conv_m8_crosscheck else "0"
        ),
        "QWEN_DFLASH_GREEDY_M8_VERIFIER": "1" if greedy_m8_verifier else "0",
        "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED": ("1" if serial_batched else "0"),
        "QWEN_GDN_BA_GROUPED_PREFIX_EXACT": ("1" if grouped_ba_prefix else "0"),
        "QWEN_GDN_BA_GROUPED_PREFIX_CROSSCHECK": ("1" if grouped_ba_crosscheck else "0"),
        "QWEN_GDN_BA_BATCHED_EXACT": "1" if batched_ba else "0",
        "QWEN_GDN_BA_BATCHED_CROSSCHECK": "1" if batched_ba_crosscheck else "0",
        "QWEN_QUEST_M8_PAGE_STRIPE": ("1" if quest_m8_page_stripe else "0"),
        "QWEN_QUEST_CACHED_GEMM_SELECTOR": ("1" if cached_gemm_selector else "0"),
        "QWEN_QUEST_CACHED_GEMM_COLD_SEED": ("1" if cached_gemm_cold_seed else "0"),
        "QWEN_DFLASH_D7_C1_DRAFT_GRAPH": "1" if d7_c1_draft_graph else "0",
    }
    if allow_unqualified_layer_diagnostic_context:
        replacements["QWEN_DFLASH_ASSURANCE_NON_PROMOTABLE_DIAGNOSTIC"] = "1"
        if runtime_file_bindings_path is None or not re.fullmatch(
            r"[0-9a-f]{64}", runtime_file_bindings_sha256 or ""
        ):
            raise PrepareError(
                "unqualified-context diagnostics require an authenticated complete runtime table"
            )
        replacements[RUNTIME_FILE_BINDINGS_ENV] = str(runtime_file_bindings_path)
        replacements[RUNTIME_FILE_BINDINGS_SHA_ENV] = str(runtime_file_bindings_sha256)
    if best_first_b7:
        replacements.update(
            {
                "QWEN_DFLASH2_BEST_FIRST_B7_REQUIRED": "1",
                "QWEN_GDN_TREE_RDNA": "1",
                "QWEN_GDN_TREE_REQUIRED": "1",
                "QWEN_GDN_BEST_FIRST_B7": "1",
                "QWEN_GDN_BEST_FIRST_B7_BULK_COMMIT": "1",
            }
        )
    if serial_reference_conv or fixed_serial_conv_m8:
        replacements["QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_VERIFY_CONV"] = "0"
    if enable_outer_stage_timing or disable_outer_stage_timing:
        replacements["QWEN_OUTER_STAGE_TIMING"] = "1" if enable_outer_stage_timing else "0"
    if proposal_temperature_scale is not None:
        replacements["QWEN_DFLASH2_PROPOSAL_TEMPERATURE_SCALE"] = proposal_temperature_scale
    if release_probe_root is not None:
        if artifact_mode != "release":
            raise PrepareError("release activation probe requires the release artifact")
        if not release_probe_sha256 or not release_probe_site_sha256:
            raise PrepareError("release activation probe requires both source hashes")
        active_selector_event = (
            "_cached_gemm_select_logical_pages"
            if cached_gemm_selector
            else "_compiled_select_row_local_m8_logical_pages"
        )
        required_probe_events = ["module_patched", active_selector_event]
        if cached_gemm_cold_seed:
            required_probe_events.append("_cached_gemm_seed_exact_row_local_m8")
        required_probe_events.extend(
            (
                "_try_compiled_selected_attention",
                "_load_m8_row_local_dualphase_extension",
                "gpu_timing",
                "union_stats",
            )
        )
        unique_depth_mixed_raw = template_environment.get("QWEN_UNIQUE_DEPTH_MIXED_REQUIRED", "0")
        if unique_depth_mixed_raw not in {"0", "1"}:
            raise PrepareError("QWEN_UNIQUE_DEPTH_MIXED_REQUIRED must be 0 or 1")
        if unique_depth_mixed_raw == "1":
            required_probe_events.append("_try_unique_depth_mixed_attention")
        if best_first_b7:
            required_probe_events.extend(
                (
                    "_qwen_stage_best_first_b7",
                    "verify_best_first_b7_out",
                    "commit_and_clear",
                )
            )
        replacements.update(
            {
                "QWEN_RELEASE_ACTIVATION_PROBE": "1",
                "QWEN_RELEASE_ACTIVATION_PROBE_OUTPUT": f"{remote}/probe/events.jsonl",
                "QWEN_RELEASE_ACTIVATION_PROBE_SHA256": release_probe_sha256,
                "QWEN_RELEASE_ACTIVATION_PROBE_SITE_SHA256": release_probe_site_sha256,
                "QWEN_RELEASE_ACTIVATION_PROBE_GPU_TIMING": "1",
                "QWEN_RELEASE_ACTIVATION_PROBE_UNION_STATS": "1",
                "QWEN_RELEASE_ACTIVATION_PROBE_REQUIRED_EVENTS": ",".join(required_probe_events),
            }
        )
    if tree_score_capture_root is not None:
        if artifact_mode != "release" or force_target_only:
            raise PrepareError(
                "DFlash score-lattice capture requires speculative release execution"
            )
        if release_probe_root is not None:
            raise PrepareError("DFlash score-lattice capture conflicts with the release probe")
        if runtime_launcher_path is None:
            raise PrepareError("DFlash score-lattice capture requires a runtime launcher path")
        expected_hash_names = {
            "capture_contract",
            "capture_hook",
            "capture_site",
            "score_lattice",
        }
        if (
            tree_score_capture_hashes is None
            or set(tree_score_capture_hashes) != expected_hash_names
            or not tree_public_manifest_sha256
        ):
            raise PrepareError("DFlash score-lattice capture source hashes are incomplete")
        replacements.update(
            {
                "QWEN_DFLASH_TREE_RANK_CAPTURE": "1",
                "QWEN_DFLASH_TREE_PUBLIC_FIXTURE": "1",
                "QWEN_DFLASH_TREE_PUBLIC_MANIFEST_SHA256": tree_public_manifest_sha256,
                "QWEN_DFLASH_TREE_LAUNCHER_SHA256": launcher_sha256,
                "QWEN_DFLASH_TREE_LAUNCHER_PATH": str(runtime_launcher_path),
                "QWEN_DFLASH_TREE_PROMPT_TOKENS": str(prompt_tokens),
                "QWEN_DFLASH_TREE_OUTPUT_TOKENS": str(capture_rounds),
                "QWEN_DFLASH_TREE_RANK_OUTPUT": (
                    f"{remote}/dflash-tree-score-output/rank-evidence.json"
                ),
                "QWEN_DFLASH_TREE_SCORE_OUTPUT": (
                    f"{remote}/dflash-tree-score-output/score-lattice.json"
                ),
                "QWEN_DFLASH_TREE_CAPTURE_CONTRACT_SHA256": tree_score_capture_hashes[
                    "capture_contract"
                ],
                "QWEN_DFLASH_TREE_CAPTURE_HOOK_SHA256": tree_score_capture_hashes["capture_hook"],
                "QWEN_DFLASH_TREE_CAPTURE_SITE_SHA256": tree_score_capture_hashes["capture_site"],
                "QWEN_DFLASH_TREE_SCORE_LATTICE_SHA256": tree_score_capture_hashes["score_lattice"],
            }
        )
    extension_pairs = (
        (
            "QWEN_QUEST_SELECTOR_EXTENSION",
            "QWEN_QUEST_SELECTOR_EXTENSION_SHA256",
            quest_selector_extension,
            quest_selector_extension_sha256,
            "Quest selector extension",
        ),
        (
            "QWEN_QUEST_M8_ROW_LOCAL_DUALPHASE_EXTENSION",
            "QWEN_QUEST_M8_ROW_LOCAL_DUALPHASE_EXTENSION_SHA256",
            row_local_dualphase_extension,
            row_local_dualphase_extension_sha256,
            "row-local dual-phase extension",
        ),
    )
    for path_name, hash_name, path, digest, label in extension_pairs:
        if bool(path) != bool(digest):
            raise PrepareError(f"{label} path and SHA256 must be supplied together")
        if path is None or digest is None:
            continue
        if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise PrepareError(f"{label} must be a normalized absolute path")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise PrepareError(f"{label} SHA256 must be 64 lowercase hex characters")
        replacements[path_name] = str(path)
        replacements[hash_name] = digest
    assurance_names = {
        "QWEN_CODING_TURBO_ASSURANCE_DIAGNOSTIC",
        "QWEN_CODING_TURBO_ORACLE_STATE_OUTPUT",
        "QWEN_CODING_TURBO_ORACLE_STATE_COMMITTED_COUNTS",
        "QWEN_ROUND_EQUIVALENCE_PRECOMMIT_OUTPUT",
        "QWEN_ROUND_EQUIVALENCE_PROPOSAL_INPUT",
        "QWEN_ROUND_EQUIVALENCE_PROPOSAL_SHA256",
        "QWEN_ROUND_EQUIVALENCE_ZERO_COMMIT",
        "QWEN_CODING_TURBO_STATE_PRODUCER_RECEIPT",
        "QWEN_CODING_TURBO_STATE_CONSUMER_TRANSITION_RECEIPT",
        "QWEN_CODING_TURBO_ASSURANCE_MANIFEST",
        "QWEN_CODING_TURBO_SNAPSHOT_SELECTION",
        RUNTIME_FILE_BINDINGS_ENV,
        RUNTIME_FILE_BINDINGS_SHA_ENV,
        *RUNTIME_CAPTURE_SHA_ENVS.values(),
    }
    full_capture_names = {
        "QWEN_CODING_TURBO_ORACLE_STATE_OUTPUT",
        "QWEN_DFLASH_ASSURANCE_DRAFT_OUTPUT",
        "QWEN_ROUND_EQUIVALENCE_PRECOMMIT_OUTPUT",
    }
    proposal_calibration_names = {
        "QWEN_DFLASH2_PROPOSAL_TEMPERATURE_SCALE",
        "QWEN_DFLASH2_PROPOSAL_TEMPERATURE_SWEEP",
    }
    if artifact_mode == "assurance":
        replacements.update(
            {
                "QWEN_CODING_TURBO_ASSURANCE_DIAGNOSTIC": "1",
                "QWEN_CODING_TURBO_ORACLE_STATE_OUTPUT": f"{remote}/capture/states.jsonl",
                "QWEN_DFLASH_ASSURANCE_CAPTURE": "1",
                "QWEN_DFLASH_ASSURANCE_DRAFT_OUTPUT": f"{remote}/capture/draft.jsonl",
                "QWEN_DFLASH_ASSURANCE_LAYER_OUTPUT": f"{remote}/capture/layers.jsonl",
                "QWEN_DFLASH_ASSURANCE_REQUEST_ID": request_id,
                "QWEN_DFLASH_ASSURANCE_PROMPT_TOKENS": str(prompt_tokens),
                "QWEN_DFLASH_ASSURANCE_MAX_ROUNDS": str(capture_rounds),
                "QWEN_DFLASH_ASSURANCE_LAUNCHER_SHA256": launcher_sha256,
                "QWEN_DFLASH_ASSURANCE_LAUNCHER_PATH": (
                    str(runtime_launcher_path) if runtime_launcher_path is not None else ""
                ),
                "QWEN_DFLASH_ASSURANCE_OUTPUT": f"{remote}/capture/rounds.jsonl",
                "QWEN_CODING_TURBO_STATE_PRODUCER_RECEIPT": (
                    f"{remote}/state-producer-receipt.json"
                ),
                "QWEN_CODING_TURBO_STATE_CONSUMER_TRANSITION_RECEIPT": (
                    f"{remote}/state-consumer-transition-receipt.json"
                ),
                "QWEN_CODING_TURBO_ASSURANCE_MANIFEST": f"{artifact}/manifest.json",
                "QWEN_CODING_TURBO_SNAPSHOT_SELECTION": f"{remote}/snapshot-selection.json",
            }
        )
        if runtime_binding_sha256 is not None:
            for binding_name, environment_name in RUNTIME_CAPTURE_SHA_ENVS.items():
                digest = runtime_binding_sha256.get(binding_name)
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise PrepareError(
                        f"assurance command lacks runtime binding hash for {binding_name}"
                    )
                replacements[environment_name] = digest
    position_filter_requested = layer_position is not None or layer_positions is not None
    if layer_position is not None and layer_positions is not None:
        raise PrepareError("single and multiple assurance layer positions are mutually exclusive")
    if layer_positions is not None:
        if not 1 <= len(layer_positions) <= 8 or len(set(layer_positions)) != len(layer_positions):
            raise PrepareError("assurance layer positions must contain 1..8 unique values")
        if any(position <= 0 for position in layer_positions):
            raise PrepareError("assurance layer positions must be positive")
        layer_positions = tuple(sorted(layer_positions))
    elif layer_position is not None and layer_position <= 0:
        raise PrepareError("assurance layer position must be positive")
    if layer_indices is not None:
        if (
            not 1 <= len(layer_indices) <= 64
            or len(set(layer_indices)) != len(layer_indices)
            or any(not 0 <= layer_index < 64 for layer_index in layer_indices)
        ):
            raise PrepareError("assurance layer indices must contain 1..64 unique values in 0..63")
        if not position_filter_requested:
            raise PrepareError("assurance layer indices require a layer-position filter")
        layer_indices = tuple(sorted(layer_indices))
    if artifact_mode == "release" and position_filter_requested:
        raise PrepareError("release runs cannot request assurance layer capture")
    if layer_diagnostic_only:
        if artifact_mode != "assurance":
            raise PrepareError("layer-only diagnostics require the assurance artifact")
        if not position_filter_requested or layer_indices is None:
            raise PrepareError("layer-only diagnostics require position and decoder-layer filters")
        replacements["QWEN_DFLASH_ASSURANCE_LAYER_DIAGNOSTIC_ONLY"] = "1"
        for name in full_capture_names:
            replacements.pop(name, None)
    if round_equivalence_precommit:
        if artifact_mode != "assurance" or layer_diagnostic_only:
            raise PrepareError("round-equivalence precommit capture requires a full assurance run")
        replacements["QWEN_ROUND_EQUIVALENCE_PRECOMMIT_OUTPUT"] = (
            f"{remote}/capture/precommit.jsonl"
        )
    if round_equivalence_zero_commit:
        if (
            not round_equivalence_precommit
            or round_equivalence_proposal_sha256 is None
            or force_target_only
        ):
            raise PrepareError(
                "zero-commit capture requires speculative precommit capture and a forced proposal"
            )
        replacements["QWEN_ROUND_EQUIVALENCE_ZERO_COMMIT"] = "1"
    if state_committed_counts is not None:
        if artifact_mode != "assurance" or layer_diagnostic_only:
            raise PrepareError("sparse state checkpoints require a full assurance run")
        if round_equivalence_precommit:
            raise PrepareError("sparse state checkpoints cannot be combined with precommit capture")
        if (
            not 1 <= len(state_committed_counts) <= 64
            or len(set(state_committed_counts)) != len(state_committed_counts)
            or any(not 1 <= count <= 1_000_000 for count in state_committed_counts)
        ):
            raise PrepareError(
                "state committed counts must contain 1..64 unique values in 1..1000000"
            )
        state_committed_counts = tuple(sorted(state_committed_counts))
        replacements["QWEN_CODING_TURBO_ORACLE_STATE_COMMITTED_COUNTS"] = ",".join(
            str(count) for count in state_committed_counts
        )
    if round_equivalence_proposal_sha256 is not None:
        if not round_equivalence_precommit:
            raise PrepareError("round-equivalence proposal input requires precommit state capture")
        if force_target_only:
            raise PrepareError("round-equivalence proposal input requires speculative M8")
        if capture_rounds < 2 and not round_equivalence_zero_commit:
            raise PrepareError(
                "round-equivalence M8 proposal requires at least two API tokens: "
                "one restored-prefix bootstrap anchor and one published transition"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", round_equivalence_proposal_sha256):
            raise PrepareError("round-equivalence proposal SHA256 is invalid")
        replacements.update(
            {
                "QWEN_ROUND_EQUIVALENCE_PROPOSAL_INPUT": (
                    f"{remote}/round-equivalence-proposal.json"
                ),
                "QWEN_ROUND_EQUIVALENCE_PROPOSAL_SHA256": (round_equivalence_proposal_sha256),
            }
        )
    if gemma_rms_fp32_kernel:
        replacements.update(
            {
                "QWEN_GEMMA_RMS_WEIGHT_CACHE": "1",
                "QWEN_GEMMA_RMS_FP32_KERNEL": "1",
                "QWEN_GEMMA_RMS_FP32_SO": GEMMA_RMS_FP32_EXTENSION,
            }
        )
    if force_target_only:
        replacements.update(
            {
                "QWEN_DFLASH_WHOLE_MODEL_ACCEPTED_REPLAY": "0",
                "QWEN_FIXED_SLOT_TARGET_ONLY_ORACLE": "1",
                # These switches select the speculative M8 verification and
                # accepted-path commit machinery.  A target-only oracle runs
                # serial M1 transitions and must not inherit them from its M8
                # template: the runtime deliberately rejects selector
                # sub-switches without the serial-batched master switch.
                "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_VERIFY": "0",
                "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_COMMIT": "0",
                "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_VERIFY_CONV": "0",
                "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_VERIFY_RECURRENCE": "0",
            }
        )
    if layer_position is not None:
        replacements["QWEN_DFLASH_ASSURANCE_LAYER_POSITION"] = str(layer_position)
        replacements["QWEN_DFLASH_ASSURANCE_LAYER_DETAIL"] = "all"
    elif layer_positions is not None:
        replacements["QWEN_DFLASH_ASSURANCE_LAYER_POSITIONS"] = ",".join(
            str(position) for position in layer_positions
        )
        replacements["QWEN_DFLASH_ASSURANCE_LAYER_DETAIL"] = "all"
    if layer_indices is not None:
        replacements["QWEN_DFLASH_ASSURANCE_LAYER_INDICES"] = ",".join(
            str(layer_index) for layer_index in layer_indices
        )
    rewritten: list[str] = []
    pythonpath_seen = False
    environment: dict[str, str] = {}
    for argument in arguments:
        argument = argument.replace(old_qualification, remote)
        if argument.startswith("PYTHONPATH="):
            if pythonpath_seen:
                raise PrepareError("template command contains duplicate PYTHONPATH assignments")
            pythonpath_seen = True
            roots = argument.removeprefix("PYTHONPATH=").split(":")
            retained = [
                root
                for root in roots
                if root
                and "/artifact-pair/assurance/files/assurance" not in root
                and "/artifact-pair/assurance/files/vllm/v1/attention/backends" not in root
                and "/artifact-pair/release/files/vllm/v1/attention/backends" not in root
                and "/dflash-lossless-assurance/capture_site" not in root
                and "/release-activation-probe" not in root
                and "/dflash-tree-score-capture" not in root
            ]
            roots = (
                (capture_site, module_root, runtime_backends, *retained)
                if artifact_mode == "assurance"
                else (
                    *(
                        (f"{tree_score_capture_root}/capture_site",)
                        if tree_score_capture_root is not None
                        else ()
                    ),
                    *((str(release_probe_root),) if release_probe_root is not None else ()),
                    runtime_backends,
                    *retained,
                )
            )
            argument = f"PYTHONPATH={':'.join(roots)}"
        elif "=" in argument:
            name, _value = argument.split("=", 1)
            if name in proposal_calibration_names and not (
                name == "QWEN_DFLASH2_PROPOSAL_TEMPERATURE_SCALE"
                and proposal_temperature_scale is not None
            ):
                # Calibration is a property of the new candidate, never of its
                # template.  Strip inherited single-scale and sweep settings
                # unless this bundle explicitly binds one scale.
                continue
            if artifact_mode == "release" and (
                name in assurance_names or name.startswith("QWEN_DFLASH_ASSURANCE_")
            ):
                continue
            if (
                artifact_mode == "assurance"
                and layer_diagnostic_only
                and name in full_capture_names
            ):
                continue
            if name == "QWEN_DFLASH_ASSURANCE_LAYER_DIAGNOSTIC_ONLY" and not layer_diagnostic_only:
                continue
            if name in probe_names:
                continue
            if name in tree_capture_names:
                continue
            if (
                (name == "QWEN_DFLASH_ASSURANCE_LAYER_POSITION" and layer_position is None)
                or (name == "QWEN_DFLASH_ASSURANCE_LAYER_POSITIONS" and layer_positions is None)
                or (name == "QWEN_DFLASH_ASSURANCE_LAYER_DETAIL" and not position_filter_requested)
                or (name == "QWEN_DFLASH_ASSURANCE_LAYER_INDICES" and layer_indices is None)
                or (
                    name == "QWEN_ROUND_EQUIVALENCE_PRECOMMIT_OUTPUT"
                    and not round_equivalence_precommit
                )
                or (
                    name
                    in {
                        "QWEN_ROUND_EQUIVALENCE_PROPOSAL_INPUT",
                        "QWEN_ROUND_EQUIVALENCE_PROPOSAL_SHA256",
                    }
                    and round_equivalence_proposal_sha256 is None
                )
                or (
                    name == "QWEN_ROUND_EQUIVALENCE_ZERO_COMMIT"
                    and not round_equivalence_zero_commit
                )
                or (
                    name == "QWEN_CODING_TURBO_ORACLE_STATE_COMMITTED_COUNTS"
                    and state_committed_counts is None
                )
            ):
                # A position filter belongs to the new qualification, not to
                # the template it was derived from.  Omitting or changing the
                # filter must therefore remove the inherited assignment.
                continue
            if name in replacements:
                argument = f"{name}={replacements[name]}"
        if "=" in argument:
            name, value = argument.split("=", 1)
            if ENVIRONMENT_NAME.fullmatch(name):
                previous = environment.get(name)
                if previous is not None:
                    if previous != value:
                        raise PrepareError(
                            f"template command conflicts on environment variable {name}"
                        )
                    continue
                environment[name] = value
        rewritten.append(argument)
    if not pythonpath_seen:
        raise PrepareError("template command has no PYTHONPATH assignment")
    insertion = 3
    while insertion < len(rewritten):
        argument = rewritten[insertion]
        if "=" not in argument or not ENVIRONMENT_NAME.fullmatch(argument.split("=", 1)[0]):
            break
        insertion += 1
    for name, value in replacements.items():
        if name not in environment:
            rewritten.insert(insertion, f"{name}={value}")
            environment[name] = value
            insertion += 1
    for name, value in replacements.items():
        if environment.get(name) != value:
            raise PrepareError(f"template command does not contain exactly one {name} assignment")
    if d7_c1_draft_graph:
        rewritten = [argument for argument in rewritten if argument != "--enforce-eager"]
        compilation_options = [
            offset
            for offset, argument in enumerate(rewritten)
            if argument == "--compilation-config"
        ]
        if len(compilation_options) > 1:
            raise PrepareError("template command duplicates --compilation-config")
        exact_target_config = json.dumps(
            {"mode": 0, "cudagraph_mode": "NONE"}, separators=(",", ":")
        )
        if compilation_options:
            compilation_option = compilation_options[0]
            if compilation_option + 1 >= len(rewritten):
                raise PrepareError("template --compilation-config has no value")
            rewritten[compilation_option + 1] = exact_target_config
        else:
            try:
                insertion = rewritten.index("--kv-transfer-config")
            except ValueError as error:
                raise PrepareError("assurance command has no KV-transfer configuration") from error
            rewritten[insertion:insertion] = ["--compilation-config", exact_target_config]
    try:
        option = rewritten.index("--kv-transfer-config")
    except ValueError as error:
        raise PrepareError("assurance command has no KV-transfer configuration") from error
    if option + 1 >= len(rewritten):
        raise PrepareError("assurance KV-transfer configuration has no value")
    try:
        kv_transfer = json.loads(rewritten[option + 1])
    except json.JSONDecodeError as error:
        raise PrepareError(f"assurance KV-transfer configuration is invalid: {error}") from error
    if not isinstance(kv_transfer, dict):
        raise PrepareError("assurance KV-transfer configuration must be an object")
    extra_config = kv_transfer.get("kv_connector_extra_config")
    if not isinstance(extra_config, dict):
        raise PrepareError("assurance command lacks KV-transfer extra configuration")
    extra_config.update(
        {
            "fixed_slot_m8_serial_oracle": False,
            "fixed_slot_target_only_oracle": force_target_only,
            "fixed_slot_trusted_replay": not force_target_only,
            "fixed_slot_whole_model_replay": not force_target_only,
        }
    )
    rewritten[option + 1] = json.dumps(
        kv_transfer, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return shlex.join(rewritten) + "\n"


def _command_contract(payload: str) -> tuple[dict[str, str], bool, bool]:
    """Return launch environment, effective DFlash mode, and target-only mode."""

    try:
        arguments = shlex.split(payload)
    except ValueError as error:  # pragma: no cover - already checked by _rewrite_command
        raise PrepareError(f"prepared command quoting is invalid: {error}") from error
    if arguments[:3] != ["exec", "/usr/bin/env", "-i"]:
        raise PrepareError("prepared command must begin with exec /usr/bin/env -i")

    environment: dict[str, str] = {}
    index = 3
    while index < len(arguments) and ENVIRONMENT_NAME.fullmatch(arguments[index].split("=", 1)[0]):
        argument = arguments[index]
        if "=" not in argument:
            break
        name, value = argument.split("=", 1)
        if name in environment:
            raise PrepareError(f"prepared command duplicates environment variable {name}")
        environment[name] = value
        index += 1

    speculative: dict[str, Any] | None = None
    try:
        option = arguments.index("--speculative-config", index)
    except ValueError:
        option = -1
    if option >= 0:
        if option + 1 >= len(arguments):
            raise PrepareError("prepared command has no --speculative-config value")
        try:
            candidate = json.loads(arguments[option + 1])
        except json.JSONDecodeError as error:
            raise PrepareError(f"prepared speculative config is invalid JSON: {error}") from error
        if not isinstance(candidate, dict):
            raise PrepareError("prepared speculative config must be an object")
        speculative = candidate
    dflash_configured = speculative is not None and speculative.get("method") == "dflash"
    if speculative is not None and not dflash_configured:
        raise PrepareError("assurance supports only absent speculation or method=dflash")
    layer_diagnostic_raw = environment.get("QWEN_DFLASH_ASSURANCE_LAYER_DIAGNOSTIC_ONLY", "0")
    if layer_diagnostic_raw not in {"0", "1"}:
        raise PrepareError("QWEN_DFLASH_ASSURANCE_LAYER_DIAGNOSTIC_ONLY must be 0 or 1")
    layer_diagnostic_only = layer_diagnostic_raw == "1"
    if environment.get("QWEN_DFLASH_ASSURANCE_CAPTURE") == "1":
        for name in (
            "QWEN_DFLASH_ASSURANCE_PROMPT_TOKENS",
            "QWEN_DFLASH_ASSURANCE_MAX_ROUNDS",
        ):
            try:
                value = int(environment.get(name, ""))
            except ValueError as error:
                raise PrepareError(f"{name} must be a positive integer") from error
            if value <= 0:
                raise PrepareError(f"{name} must be a positive integer")
        layer_position = environment.get("QWEN_DFLASH_ASSURANCE_LAYER_POSITION")
        layer_positions = environment.get("QWEN_DFLASH_ASSURANCE_LAYER_POSITIONS")
        layer_indices = environment.get("QWEN_DFLASH_ASSURANCE_LAYER_INDICES")
        layer_detail = environment.get("QWEN_DFLASH_ASSURANCE_LAYER_DETAIL")
        if layer_position is not None and layer_positions is not None:
            raise PrepareError("assurance command contains conflicting layer position filters")
        if (layer_position is not None or layer_positions is not None) and layer_detail != "all":
            raise PrepareError("position-filtered assurance requires all decoder-layer boundaries")
        if layer_positions is not None:
            try:
                parsed_positions = tuple(int(value) for value in layer_positions.split(","))
            except ValueError as error:
                raise PrepareError("assurance layer positions must be integers") from error
            if (
                not 1 <= len(parsed_positions) <= 8
                or len(set(parsed_positions)) != len(parsed_positions)
                or any(position <= 0 for position in parsed_positions)
            ):
                raise PrepareError("assurance layer positions are invalid")
        if layer_indices is not None:
            if layer_position is None and layer_positions is None:
                raise PrepareError("assurance layer indices require a layer-position filter")
            try:
                parsed_indices = tuple(int(value) for value in layer_indices.split(","))
            except ValueError as error:
                raise PrepareError("assurance layer indices must be integers") from error
            if (
                not 1 <= len(parsed_indices) <= 64
                or len(set(parsed_indices)) != len(parsed_indices)
                or any(not 0 <= layer_index < 64 for layer_index in parsed_indices)
            ):
                raise PrepareError("assurance layer indices are invalid")
        if layer_diagnostic_only:
            if layer_position is None and layer_positions is None:
                raise PrepareError("layer-only diagnostics require a layer-position filter")
            if layer_indices is None:
                raise PrepareError("layer-only diagnostics require decoder-layer indices")
            for forbidden in (
                "QWEN_CODING_TURBO_ORACLE_STATE_OUTPUT",
                "QWEN_DFLASH_ASSURANCE_DRAFT_OUTPUT",
            ):
                if forbidden in environment:
                    raise PrepareError(f"layer-only diagnostics must omit {forbidden}")
    elif layer_diagnostic_only:
        raise PrepareError("layer-only diagnostics require authenticated token capture")
    target_only_raw = environment.get("QWEN_FIXED_SLOT_TARGET_ONLY_ORACLE", "0")
    if target_only_raw not in {"0", "1"}:
        raise PrepareError("QWEN_FIXED_SLOT_TARGET_ONLY_ORACLE must be 0 or 1")
    target_only = target_only_raw == "1"
    cached_commit_raw = environment.get("QWEN_GDN_FIXED_SLOT_CACHED_ACCEPTED_COMMIT", "0")
    if cached_commit_raw not in {"0", "1"}:
        raise PrepareError("QWEN_GDN_FIXED_SLOT_CACHED_ACCEPTED_COMMIT must be 0 or 1")
    cached_commit = cached_commit_raw == "1"
    fixed_serial_conv_m8_raw = environment.get("QWEN_GDN_FIXED_SLOT_SERIAL_CONV_M8", "0")
    if fixed_serial_conv_m8_raw not in {"0", "1"}:
        raise PrepareError("QWEN_GDN_FIXED_SLOT_SERIAL_CONV_M8 must be 0 or 1")
    fixed_serial_conv_m8 = fixed_serial_conv_m8_raw == "1"
    fixed_serial_conv_m8_crosscheck_raw = environment.get(
        "QWEN_GDN_FIXED_SLOT_SERIAL_CONV_M8_CROSSCHECK", "0"
    )
    if fixed_serial_conv_m8_crosscheck_raw not in {"0", "1"}:
        raise PrepareError("QWEN_GDN_FIXED_SLOT_SERIAL_CONV_M8_CROSSCHECK must be 0 or 1")
    fixed_serial_conv_m8_crosscheck = fixed_serial_conv_m8_crosscheck_raw == "1"
    greedy_m8_verifier_raw = environment.get("QWEN_DFLASH_GREEDY_M8_VERIFIER", "0")
    if greedy_m8_verifier_raw not in {"0", "1"}:
        raise PrepareError("QWEN_DFLASH_GREEDY_M8_VERIFIER must be 0 or 1")
    greedy_m8_verifier = greedy_m8_verifier_raw == "1"
    best_first_b7_raw = environment.get("QWEN_DFLASH2_BEST_FIRST_B7_REQUIRED", "0")
    if best_first_b7_raw not in {"0", "1"}:
        raise PrepareError("QWEN_DFLASH2_BEST_FIRST_B7_REQUIRED must be 0 or 1")
    best_first_b7 = best_first_b7_raw == "1"
    d7_c1_draft_graph_raw = environment.get("QWEN_DFLASH_D7_C1_DRAFT_GRAPH", "0")
    if d7_c1_draft_graph_raw not in {"0", "1"}:
        raise PrepareError("QWEN_DFLASH_D7_C1_DRAFT_GRAPH must be 0 or 1")
    d7_c1_draft_graph = d7_c1_draft_graph_raw == "1"
    draft_graph_required = environment.get("QWEN_DFLASH_D7_C1_DRAFT_GRAPH_REQUIRED")
    split_k_workspace_slots = environment.get("VLLM_RDNA_W4A16_SPLIT_K_WORKSPACE_SLOTS")
    if d7_c1_draft_graph:
        if draft_graph_required != "1":
            raise PrepareError(
                "D7/C1 draft graph requires launcher-managed "
                "QWEN_DFLASH_D7_C1_DRAFT_GRAPH_REQUIRED=1"
            )
        if split_k_workspace_slots != "3":
            raise PrepareError(
                "D7/C1 draft graph requires launcher-managed "
                "VLLM_RDNA_W4A16_SPLIT_K_WORKSPACE_SLOTS=3"
            )
    elif draft_graph_required is not None or split_k_workspace_slots is not None:
        raise PrepareError(
            "disabled D7/C1 draft graph forbids launcher-managed graph workspace tokens"
        )
    serial_batched_raw = environment.get("QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED", "0")
    if serial_batched_raw not in {"0", "1"}:
        raise PrepareError("QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED must be 0 or 1")
    serial_batched = serial_batched_raw == "1"
    serial_batched_selectors = (
        "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_VERIFY",
        "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_COMMIT",
        "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_VERIFY_CONV",
        "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_VERIFY_RECURRENCE",
    )
    enabled_serial_batched_selectors: list[str] = []
    for name in serial_batched_selectors:
        value = environment.get(name, "0")
        if value not in {"0", "1"}:
            raise PrepareError(f"{name} must be 0 or 1")
        if value == "1":
            enabled_serial_batched_selectors.append(name)
    if enabled_serial_batched_selectors and not serial_batched:
        raise PrepareError(
            "serial-batched verification/commit selectors require "
            "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED=1"
        )
    page_stripe_raw = environment.get("QWEN_QUEST_M8_PAGE_STRIPE", "0")
    if page_stripe_raw not in {"0", "1"}:
        raise PrepareError("QWEN_QUEST_M8_PAGE_STRIPE must be 0 or 1")
    if page_stripe_raw == "1" and environment.get("QWEN_QUEST_M8_ROW_LOCAL") != "1":
        raise PrepareError("page-stripe Quest requires row-local M8 selection")
    if target_only and not dflash_configured:
        raise PrepareError("target-only snapshot oracle requires configured DFlash geometry")
    if target_only and environment.get("QWEN_DFLASH_WHOLE_MODEL_ACCEPTED_REPLAY", "0") == "1":
        raise PrepareError("target-only oracle cannot enable whole-model accepted replay")
    if target_only and cached_commit:
        raise PrepareError("target-only oracle cannot enable cached accepted commit")
    if target_only and fixed_serial_conv_m8:
        raise PrepareError("target-only oracle cannot enable fixed serial M8 convolution")
    if greedy_m8_verifier:
        if not dflash_configured or target_only:
            raise PrepareError("greedy M8 verifier requires speculative DFlash execution")
        if speculative.get("num_speculative_tokens") != 7:
            raise PrepareError("greedy M8 verifier requires exact DFlash7 geometry")
        if environment.get("QWEN_CODING_TURBO_QUEST96") != "1":
            raise PrepareError("greedy M8 verifier requires coding-turbo Quest96")
        expected_prefix_serial = "0" if best_first_b7 else "1"
        if environment.get("QWEN_LM_HEAD_PREFIX_SERIAL_M8") != expected_prefix_serial:
            if best_first_b7:
                raise PrepareError("best-first B7 forbids prefix-serial M8 LM head")
            raise PrepareError("greedy M8 verifier requires prefix-serial M8 LM head")
        if environment.get("QWEN_DFLASH_GREEDY_LOOP_ESCAPE", "0") != "0":
            raise PrepareError("greedy M8 verifier conflicts with greedy-loop escape")
    if best_first_b7:
        if not greedy_m8_verifier or not dflash_configured or target_only:
            raise PrepareError("best-first B7 requires speculative exact greedy M8 verification")
        required_b7_environment = {
            "QWEN_GDN_TREE_RDNA": "1",
            "QWEN_GDN_TREE_REQUIRED": "1",
            "QWEN_GDN_BEST_FIRST_B7": "1",
            "QWEN_GDN_BEST_FIRST_B7_BULK_COMMIT": "1",
        }
        for name, expected in required_b7_environment.items():
            if environment.get(name) != expected:
                raise PrepareError(f"best-first B7 requires {name}={expected}")
        if not environment.get("QWEN_DFLASH2_BEST_FIRST_B7_RECEIPT"):
            raise PrepareError("best-first B7 requires an authenticated consumer receipt path")
        receipt_sha = environment.get("QWEN_DFLASH2_BEST_FIRST_B7_RECEIPT_SHA256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", receipt_sha):
            raise PrepareError("best-first B7 requires an authenticated consumer receipt SHA-256")
    if cached_commit and environment.get("QWEN_DFLASH_WHOLE_MODEL_ACCEPTED_REPLAY") != "1":
        raise PrepareError("cached accepted commit requires accepted-path replay transport")
    if fixed_serial_conv_m8 and not cached_commit:
        raise PrepareError("fixed serial M8 convolution requires cached accepted commit")
    if fixed_serial_conv_m8_crosscheck and not fixed_serial_conv_m8:
        raise PrepareError("fixed serial M8 crosscheck requires fixed serial M8 convolution")
    if d7_c1_draft_graph:
        if not dflash_configured or target_only:
            raise PrepareError("D7/C1 draft graph requires speculative DFlash execution")
        if speculative.get("num_speculative_tokens") != 7:
            raise PrepareError("D7/C1 draft graph requires exact DFlash7 geometry")
        required_environment = {
            "QWEN_CODING_TURBO_QUEST96": "1",
            "QWEN_DFLASH_GREEDY_M8_VERIFIER": "1",
            "QWEN_QUEST_M8_DUALPHASE": "1",
            "QWEN_QUEST_M8_ROW_LOCAL": "1",
            "QWEN_QUEST_PAGE_BUDGET": "96",
            "QWEN_RDNA_ENABLE_EXPERIMENTAL_SPLIT_K": "1",
            "QWEN_RDNA_W4A16_COMMON_SPLIT_K": "1",
            "QWEN_RDNA_W4A16_ACTIVATION_PRECAST": "1",
        }
        if not best_first_b7:
            required_environment["QWEN_GDN_FIXED_SLOT_SERIAL_CONV_M8"] = "1"
        for name, expected in required_environment.items():
            if environment.get(name) != expected:
                raise PrepareError(f"D7/C1 draft graph requires {name}={expected}")
        if environment.get("QWEN_DFLASH_FIXED_GRAPH", "0") != "0":
            raise PrepareError("D7/C1 draft graph conflicts with the legacy D8/M9 graph")
        try:
            max_num_seqs_option = arguments.index("--max-num-seqs", index)
        except ValueError as error:
            raise PrepareError("D7/C1 draft graph requires --max-num-seqs 1") from error
        if max_num_seqs_option + 1 >= len(arguments) or arguments[max_num_seqs_option + 1] != "1":
            raise PrepareError("D7/C1 draft graph requires --max-num-seqs 1")
        if "--enforce-eager" in arguments:
            raise PrepareError("D7/C1 draft graph target cannot use --enforce-eager")
        compilation_options = [
            offset
            for offset, argument in enumerate(arguments)
            if argument == "--compilation-config"
        ]
        if len(compilation_options) != 1 or compilation_options[0] + 1 >= len(arguments):
            raise PrepareError("D7/C1 draft graph requires one target compilation config")
        try:
            target_compilation = json.loads(arguments[compilation_options[0] + 1])
        except json.JSONDecodeError as error:
            raise PrepareError("D7/C1 draft graph target compilation config is invalid") from error
        if target_compilation != {"mode": 0, "cudagraph_mode": "NONE"}:
            raise PrepareError("D7/C1 draft graph requires target mode 0/cudagraph NONE")
    if (
        fixed_serial_conv_m8
        and environment.get("QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_VERIFY_CONV", "0") == "1"
    ):
        raise PrepareError(
            "fixed serial M8 convolution conflicts with generic serial-batched convolution"
        )
    if serial_batched and not cached_commit:
        raise PrepareError("serial-batched GDN requires cached accepted commit")
    if dflash_configured:
        try:
            option = arguments.index("--kv-transfer-config", index)
        except ValueError as error:
            raise PrepareError("assurance command has no KV-transfer configuration") from error
        if option + 1 >= len(arguments):
            raise PrepareError("assurance KV-transfer configuration has no value")
        try:
            kv_transfer = json.loads(arguments[option + 1])
        except json.JSONDecodeError as error:
            raise PrepareError(
                f"assurance KV-transfer configuration is invalid: {error}"
            ) from error
        extra_config = (
            kv_transfer.get("kv_connector_extra_config") if isinstance(kv_transfer, dict) else None
        )
        expected_transport = {
            "fixed_slot_m8_serial_oracle": False,
            "fixed_slot_target_only_oracle": target_only,
            "fixed_slot_trusted_replay": not target_only,
            "fixed_slot_whole_model_replay": not target_only,
        }
        if (
            not isinstance(extra_config, dict)
            or {name: extra_config.get(name) for name in expected_transport} != expected_transport
        ):
            raise PrepareError("assurance environment and KV-transfer transport are inconsistent")
        expected_replay = "0" if target_only else "1"
        if environment.get("QWEN_DFLASH_WHOLE_MODEL_ACCEPTED_REPLAY") != expected_replay:
            raise PrepareError("assurance replay environment and engine transport are inconsistent")
    return environment, dflash_configured and not target_only, target_only


def _oracle_identity(
    *,
    artifact_manifest: dict[str, Any],
    artifact_manifest_sha256: str,
    command_payload: str,
    receipt_document: dict[str, Any],
    selection: dict[str, Any],
    request: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    """Derive one exact oracle identity from authenticated bundle inputs."""

    try:
        receipt = validate_producer_receipt(receipt_document)
    except StateProviderError as error:
        raise PrepareError(f"state producer receipt is invalid: {error}") from error
    environment, dflash_enabled, _target_only = _command_contract(command_payload)
    if environment.get("QWEN_CODING_TURBO_QUEST96") != "1":
        raise PrepareError("oracle identity requires QWEN_CODING_TURBO_QUEST96=1")
    try:
        quest_budget = int(environment.get("QWEN_QUEST_PAGE_BUDGET", ""))
    except ValueError as error:
        raise PrepareError("oracle identity Quest page budget is invalid") from error

    selection_request = selection.get("request")
    bindings = selection.get("bindings")
    if not isinstance(selection_request, dict) or not isinstance(bindings, dict):
        raise PrepareError("snapshot selection lacks request or binding identity")
    target = bindings.get("target_model")
    tokenizer = bindings.get("tokenizer")
    if not isinstance(target, dict) or not isinstance(tokenizer, dict):
        raise PrepareError("snapshot selection lacks target/tokenizer bindings")
    context_tokens = selection_request.get("prompt_tokens")
    prompt_sha256 = selection_request.get("prompt_token_ids_sha256")
    if context_tokens not in {60_298, 249_957}:
        raise PrepareError("oracle identity context is not qualified")
    if prompt_sha256 != selection.get("payload_provenance", {}).get("prompt_token_ids_sha256"):
        raise PrepareError("snapshot prompt identity differs from payload provenance")
    identity = {
        "artifact_kind": "assurance",
        "context_tokens": context_tokens,
        "dflash_enabled": dflash_enabled,
        "lifecycle": "snapshot_restore",
        "model_sha256": target.get("evidence_sha256"),
        "payload_producer_receipt_sha256": receipt.document["snapshot_payload_receipt_sha256"],
        "prompt_token_ids_sha256": prompt_sha256,
        "quest_page_budget": quest_budget,
        "run_id": request_id,
        "runtime_artifact_manifest_sha256": artifact_manifest_sha256,
        "sampling": {"temperature": 0.0, "top_k": 1, "top_p": 1.0},
        "semantic_source_sha256": artifact_manifest.get("semantic_source_sha256"),
        "snapshot_manifest_sha256": request.get("cache_manifest_sha256"),
        "tokenizer_sha256": tokenizer.get("evidence_sha256"),
    }
    try:
        return _validate_identity(identity)
    except OracleSafetyError as error:
        raise PrepareError(f"derived oracle identity is invalid: {error}") from error


def _non_promotable_diagnostic_identity(
    *,
    artifact_manifest: dict[str, Any],
    artifact_manifest_sha256: str,
    command_payload: str,
    receipt_document: dict[str, Any],
    selection: dict[str, Any],
    request: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    """Derive an authenticated identity that promotion code cannot accept.

    Arbitrary-context layer captures are useful for localizing a concrete
    counterexample, but they are not production qualification evidence.  Keep
    their identity deliberately outside the coding-turbo oracle schema while
    retaining the same authenticated model, prompt, snapshot, and artifact
    provenance.
    """

    try:
        receipt = validate_producer_receipt(receipt_document)
    except StateProviderError as error:
        raise PrepareError(f"state producer receipt is invalid: {error}") from error
    environment, dflash_enabled, _target_only = _command_contract(command_payload)
    if environment.get("QWEN_CODING_TURBO_QUEST96") != "1":
        raise PrepareError("diagnostic identity requires QWEN_CODING_TURBO_QUEST96=1")
    try:
        quest_budget = int(environment.get("QWEN_QUEST_PAGE_BUDGET", ""))
    except ValueError as error:
        raise PrepareError("diagnostic identity Quest page budget is invalid") from error
    if quest_budget != 96:
        raise PrepareError("diagnostic identity Quest page budget must be exactly 96")

    selection_request = selection.get("request")
    bindings = selection.get("bindings")
    if not isinstance(selection_request, dict) or not isinstance(bindings, dict):
        raise PrepareError("snapshot selection lacks request or binding identity")
    target = bindings.get("target_model")
    tokenizer = bindings.get("tokenizer")
    if not isinstance(target, dict) or not isinstance(tokenizer, dict):
        raise PrepareError("snapshot selection lacks target/tokenizer bindings")
    context_tokens = selection_request.get("prompt_tokens")
    if (
        isinstance(context_tokens, bool)
        or not isinstance(context_tokens, int)
        or not 1 <= context_tokens <= 253_792
    ):
        raise PrepareError("diagnostic identity context must be within 1..253792 tokens")
    prompt_sha256 = selection_request.get("prompt_token_ids_sha256")
    if prompt_sha256 != selection.get("payload_provenance", {}).get(
        "prompt_token_ids_sha256"
    ):
        raise PrepareError("snapshot prompt identity differs from payload provenance")

    identity = {
        "artifact_kind": "assurance",
        "classification": "non_promotable_diagnostic",
        "context_tokens": context_tokens,
        "dflash_enabled": dflash_enabled,
        "lifecycle": "snapshot_restore",
        "model_sha256": target.get("evidence_sha256"),
        "payload_producer_receipt_sha256": receipt.document[
            "snapshot_payload_receipt_sha256"
        ],
        "promotable": False,
        "prompt_token_ids_sha256": prompt_sha256,
        "quest_page_budget": quest_budget,
        "run_id": request_id,
        "runtime_artifact_manifest_sha256": artifact_manifest_sha256,
        "sampling": {"temperature": 0.0, "top_k": 1, "top_p": 1.0},
        "schema": NON_PROMOTABLE_DIAGNOSTIC_SCHEMA,
        "semantic_source_sha256": artifact_manifest.get("semantic_source_sha256"),
        "snapshot_manifest_sha256": request.get("cache_manifest_sha256"),
        "tokenizer_sha256": tokenizer.get("evidence_sha256"),
    }
    for field in (
        "model_sha256",
        "payload_producer_receipt_sha256",
        "prompt_token_ids_sha256",
        "runtime_artifact_manifest_sha256",
        "semantic_source_sha256",
        "snapshot_manifest_sha256",
        "tokenizer_sha256",
    ):
        if not isinstance(identity[field], str) or not re.fullmatch(
            r"[0-9a-f]{64}", identity[field]
        ):
            raise PrepareError(f"diagnostic identity {field} is invalid")
    return identity


def _capture_stop_token_limit(
    *,
    prompt_tokens: int,
    max_tokens: int,
    force_target_only: bool,
    layer_position: int | None,
    layer_positions: tuple[int, ...] | None,
    enabled: bool,
) -> int:
    """Return the smallest request length that still evaluates the capture target."""

    if not enabled:
        return max_tokens
    capture_positions = (
        tuple(layer_positions)
        if layer_positions is not None
        else ((layer_position,) if layer_position is not None else ())
    )
    if not capture_positions:
        raise PrepareError("capture-triggered early stop requires a layer-position filter")
    required_tokens = max(capture_positions) - prompt_tokens + 1
    if force_target_only:
        required_tokens += 1
    if required_tokens <= 0:
        raise PrepareError("capture-triggered early stop precedes the restored prompt")
    if max_tokens < required_tokens:
        raise PrepareError(
            "max tokens cannot reach the requested capture before the early-stop boundary"
        )
    return required_tokens


def prepare_bundle(
    *,
    template_spec: Path,
    template_command: Path,
    artifact_root: Path,
    launcher: Path,
    state_producer_receipt: Path,
    state_consumer_transition_receipt: Path,
    snapshot_selection: Path,
    output: Path,
    remote_qualification: str,
    request_id: str,
    max_tokens: int,
    timeout: int,
    force_target_only: bool = False,
    layer_position: int | None = None,
    layer_positions: tuple[int, ...] | None = None,
    layer_indices: tuple[int, ...] | None = None,
    stop_after_layer_capture: bool = False,
    layer_diagnostic_only: bool = False,
    allow_unqualified_layer_diagnostic_context: bool = False,
    state_committed_counts: tuple[int, ...] | None = None,
    round_equivalence_precommit: bool = False,
    round_equivalence_proposal: Path | None = None,
    round_equivalence_zero_commit: bool = False,
    gemma_rms_fp32_kernel: bool = False,
    cached_accepted_commit: bool = False,
    fixed_serial_conv_m8: bool = False,
    fixed_serial_conv_m8_crosscheck: bool = False,
    greedy_m8_verifier: bool = False,
    serial_batched: bool = False,
    grouped_ba_prefix: bool = False,
    grouped_ba_crosscheck: bool = False,
    batched_ba: bool = False,
    batched_ba_crosscheck: bool = False,
    serial_reference_conv: bool = False,
    quest_m8_page_stripe: bool = False,
    cached_gemm_selector: bool = False,
    cached_gemm_cold_seed: bool = False,
    d7_c1_draft_graph: bool = False,
    enable_outer_stage_timing: bool = False,
    disable_outer_stage_timing: bool = False,
    proposal_temperature_scale: str | None = None,
    quest_selector_extension: str | None = None,
    quest_selector_extension_sha256: str | None = None,
    row_local_dualphase_extension: str | None = None,
    row_local_dualphase_extension_sha256: str | None = None,
    release_activation_probe: Path | None = None,
    dflash_tree_score_capture: Path | None = None,
    dflash_tree_public_manifest: Path | None = None,
    artifact_mode: str | None = None,
    runner_source: Path | None = None,
    runtime_binding_sources: dict[str, Path] | None = None,
    allow_runtime_binding_changes: bool = False,
    runtime_file_bindings: Path | None = None,
) -> dict[str, Any]:
    """Publish one internally consistent assurance bundle atomically."""

    template = _json(template_spec, "template spec")
    template_mode = template.get("artifact_mode")
    if template_mode not in {"assurance", "release"}:
        raise PrepareError("template artifact_mode must be assurance or release")
    selected_mode = artifact_mode if artifact_mode is not None else template_mode
    if selected_mode not in {"assurance", "release"}:
        raise PrepareError("artifact mode must be assurance or release")
    if allow_unqualified_layer_diagnostic_context and not (
        selected_mode == "assurance"
        and layer_diagnostic_only
        and stop_after_layer_capture
        and (layer_position is not None or layer_positions is not None)
        and layer_indices is not None
    ):
        raise PrepareError(
            "unqualified-context capture requires an assurance layer-only diagnostic "
            "with explicit positions/layers and capture-triggered early stop"
        )
    command_source = _regular(template_command, "template command")
    artifact_root = _directory(artifact_root, "artifact root")
    launcher = _regular(launcher, "launcher")
    receipt = _regular(state_producer_receipt, "state producer receipt")
    transition_receipt = _regular(
        state_consumer_transition_receipt, "state consumer-transition receipt"
    )
    selection = _regular(snapshot_selection, "snapshot selection")
    proposal_source = (
        _regular(round_equivalence_proposal, "round-equivalence proposal")
        if round_equivalence_proposal is not None
        else None
    )
    proposal_document = None
    if proposal_source is not None:
        try:
            proposal_document = normalize_forced_proposal(
                _json(proposal_source, "round-equivalence proposal")
            )
        except RoundEquivalenceError as error:
            raise PrepareError(f"round-equivalence proposal is invalid: {error}") from error
    selection_document = _json(selection, "snapshot selection")
    selection_request = selection_document.get("request")
    if not isinstance(selection_request, dict):
        raise PrepareError("snapshot selection request is invalid")
    prompt_tokens = selection_request.get("prompt_tokens")
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        raise PrepareError("snapshot selection prompt tokens must be a positive integer")
    template_runner = template.get("runner")
    if not isinstance(template_runner, dict) or not isinstance(template_runner.get("path"), str):
        raise PrepareError("template request runner is invalid")
    runner_input = _regular(
        runner_source if runner_source is not None else Path(template_runner["path"]),
        "template request runner",
    )
    if template_runner.get("sha256") != _sha256(runner_input):
        raise PrepareError("template request runner differs from its binding")
    output = output.absolute()
    remote = _remote_path(remote_qualification, "remote qualification directory")
    remote_launcher = PurePosixPath(remote) / "controller" / "stock-vllm-frozenlock"
    runtime_file_binding_source: Path | None = None
    runtime_file_binding_document: dict[str, Any] | None = None
    runtime_file_binding_sha256: str | None = None
    if runtime_file_bindings is not None:
        (
            runtime_file_binding_source,
            runtime_file_binding_document,
            runtime_file_binding_sha256,
        ) = _runtime_file_binding_table(runtime_file_bindings)
    if allow_unqualified_layer_diagnostic_context and runtime_file_binding_source is None:
        raise PrepareError(
            "unqualified-context diagnostics require --runtime-file-bindings"
        )
    selector_binding = _extension_binding(
        quest_selector_extension,
        quest_selector_extension_sha256,
        "Quest selector extension",
    )
    dualphase_binding = _extension_binding(
        row_local_dualphase_extension,
        row_local_dualphase_extension_sha256,
        "row-local dual-phase extension",
    )
    probe_root = None
    probe_source = None
    probe_site_source = None
    probe_sha256 = None
    probe_site_sha256 = None
    if release_activation_probe is not None:
        if selected_mode != "release":
            raise PrepareError("release activation probe is valid only for a release run")
        local_probe_root = _directory(release_activation_probe, "release activation probe")
        probe_source = _regular(local_probe_root / "probe.py", "release activation probe source")
        probe_site_source = _regular(
            local_probe_root / "sitecustomize.py", "release activation probe bootstrap"
        )
        probe_sha256 = _sha256(probe_source)
        probe_site_sha256 = _sha256(probe_site_source)
        probe_root = PurePosixPath(remote) / "release-activation-probe"
    tree_capture_sources: dict[str, Path] | None = None
    tree_capture_hashes: dict[str, str] | None = None
    tree_manifest_source: Path | None = None
    tree_manifest_sha256: str | None = None
    tree_capture_root: PurePosixPath | None = None
    if bool(dflash_tree_score_capture) != bool(dflash_tree_public_manifest):
        raise PrepareError(
            "DFlash tree score capture and public manifest must be supplied together"
        )
    if dflash_tree_score_capture is not None and dflash_tree_public_manifest is not None:
        if selected_mode != "release":
            raise PrepareError("DFlash tree score capture requires a release artifact")
        local_tree_root = _directory(dflash_tree_score_capture, "DFlash tree score capture")
        tree_capture_sources = {
            "capture_contract": _regular(
                local_tree_root / "capture_contract.py", "DFlash tree capture contract"
            ),
            "capture_hook": _regular(
                local_tree_root / "capture_hook.py", "DFlash tree capture hook"
            ),
            "capture_site": _regular(
                local_tree_root / "capture_site" / "sitecustomize.py",
                "DFlash tree capture bootstrap",
            ),
            "score_lattice": _regular(
                local_tree_root / "score_lattice.py", "DFlash tree score contract"
            ),
        }
        tree_capture_hashes = {
            name: _sha256(source) for name, source in tree_capture_sources.items()
        }
        tree_manifest_source = _regular(dflash_tree_public_manifest, "DFlash tree public manifest")
        tree_manifest_sha256 = _sha256(tree_manifest_source)
        tree_capture_root = PurePosixPath(remote) / "dflash-tree-score-capture"
    if not request_id or any(character.isspace() for character in request_id):
        raise PrepareError("request ID must be non-empty and contain no whitespace")
    max_tokens = _capture_stop_token_limit(
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        force_target_only=force_target_only,
        layer_position=layer_position,
        layer_positions=layer_positions,
        enabled=stop_after_layer_capture,
    )
    if not 1 <= max_tokens <= 2049:
        raise PrepareError("max tokens must be within 1..2049")
    if timeout <= 0:
        raise PrepareError("timeout must be positive")
    if output.exists() or output.is_symlink():
        raise PrepareError("output already exists")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    manifest_source, members, artifact_outputs = _manifest_members(artifact_root, selected_mode)

    old_qualification = template.get("qualification_dir")
    if not isinstance(old_qualification, str):
        raise PrepareError("template spec qualification_dir is invalid")
    template_command_payload = command_source.read_text(encoding="utf-8")
    template_bindings = template.get("bindings")
    if not isinstance(template_bindings, dict):
        raise PrepareError("template bindings are invalid")
    explicit_runtime_sources = runtime_binding_sources or {}
    runtime_binding_hashes: dict[str, str] = {}
    for binding_name in RUNTIME_CAPTURE_SHA_ENVS:
        explicit_source = explicit_runtime_sources.get(binding_name)
        if explicit_source is not None:
            runtime_binding_hashes[binding_name] = _sha256(
                _regular(explicit_source, f"runtime binding {binding_name}")
            )
            continue
        template_binding = template_bindings.get(binding_name)
        if not isinstance(template_binding, dict) or not isinstance(
            template_binding.get("sha256"), str
        ):
            raise PrepareError(f"template lacks runtime binding {binding_name}")
        runtime_binding_hashes[binding_name] = template_binding["sha256"]
    # Rewrite first so identical stale assignments in a preserved command are
    # normalized before the strict command contract is evaluated.
    command_payload = _rewrite_command(
        template_command_payload,
        remote_qualification=remote,
        old_qualification=old_qualification,
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        capture_rounds=max_tokens,
        launcher_sha256=_sha256(launcher),
        force_target_only=force_target_only,
        layer_position=layer_position,
        layer_positions=layer_positions,
        layer_indices=layer_indices,
        layer_diagnostic_only=layer_diagnostic_only,
        allow_unqualified_layer_diagnostic_context=(
            allow_unqualified_layer_diagnostic_context
        ),
        runtime_file_bindings_path=(
            PurePosixPath(remote) / RUNTIME_FILE_BINDINGS_NAME
            if runtime_file_binding_source is not None
            else None
        ),
        runtime_file_bindings_sha256=runtime_file_binding_sha256,
        state_committed_counts=state_committed_counts,
        round_equivalence_precommit=round_equivalence_precommit,
        round_equivalence_proposal_sha256=(
            _sha256(proposal_source) if proposal_source is not None else None
        ),
        round_equivalence_zero_commit=round_equivalence_zero_commit,
        gemma_rms_fp32_kernel=gemma_rms_fp32_kernel,
        cached_accepted_commit=cached_accepted_commit,
        fixed_serial_conv_m8=fixed_serial_conv_m8,
        fixed_serial_conv_m8_crosscheck=fixed_serial_conv_m8_crosscheck,
        greedy_m8_verifier=greedy_m8_verifier,
        serial_batched=serial_batched,
        grouped_ba_prefix=grouped_ba_prefix,
        grouped_ba_crosscheck=grouped_ba_crosscheck,
        batched_ba=batched_ba,
        batched_ba_crosscheck=batched_ba_crosscheck,
        serial_reference_conv=serial_reference_conv,
        quest_m8_page_stripe=quest_m8_page_stripe,
        cached_gemm_selector=cached_gemm_selector,
        cached_gemm_cold_seed=cached_gemm_cold_seed,
        d7_c1_draft_graph=d7_c1_draft_graph,
        enable_outer_stage_timing=enable_outer_stage_timing,
        disable_outer_stage_timing=disable_outer_stage_timing,
        proposal_temperature_scale=proposal_temperature_scale,
        quest_selector_extension=(selector_binding[0] if selector_binding else None),
        quest_selector_extension_sha256=(selector_binding[1] if selector_binding else None),
        row_local_dualphase_extension=(dualphase_binding[0] if dualphase_binding else None),
        row_local_dualphase_extension_sha256=(dualphase_binding[1] if dualphase_binding else None),
        release_probe_root=probe_root,
        release_probe_sha256=probe_sha256,
        release_probe_site_sha256=probe_site_sha256,
        tree_score_capture_root=tree_capture_root,
        tree_score_capture_hashes=tree_capture_hashes,
        tree_public_manifest_sha256=tree_manifest_sha256,
        artifact_mode=selected_mode,
        runtime_launcher_path=remote_launcher,
        runtime_binding_sha256=runtime_binding_hashes,
    )
    _environment, dflash_enabled, target_only = _command_contract(command_payload)
    if target_only and max_tokens < 2:
        raise PrepareError("target-only oracle requires one committed token plus one lookahead")
    if dflash_enabled and max_tokens > 2048:
        raise PrepareError("DFlash assurance is limited to 2048 externally committed tokens")
    capture_rounds = max_tokens - 1 if target_only else max_tokens
    if capture_rounds != max_tokens:
        command_payload = _rewrite_command(
            template_command_payload,
            remote_qualification=remote,
            old_qualification=old_qualification,
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            capture_rounds=capture_rounds,
            launcher_sha256=_sha256(launcher),
            force_target_only=force_target_only,
            layer_position=layer_position,
            layer_positions=layer_positions,
            layer_indices=layer_indices,
            layer_diagnostic_only=layer_diagnostic_only,
            state_committed_counts=state_committed_counts,
            round_equivalence_precommit=round_equivalence_precommit,
            round_equivalence_proposal_sha256=(
                _sha256(proposal_source) if proposal_source is not None else None
            ),
            round_equivalence_zero_commit=round_equivalence_zero_commit,
            gemma_rms_fp32_kernel=gemma_rms_fp32_kernel,
            cached_accepted_commit=cached_accepted_commit,
            fixed_serial_conv_m8=fixed_serial_conv_m8,
            fixed_serial_conv_m8_crosscheck=fixed_serial_conv_m8_crosscheck,
            greedy_m8_verifier=greedy_m8_verifier,
            serial_batched=serial_batched,
            grouped_ba_prefix=grouped_ba_prefix,
            grouped_ba_crosscheck=grouped_ba_crosscheck,
            batched_ba=batched_ba,
            batched_ba_crosscheck=batched_ba_crosscheck,
            serial_reference_conv=serial_reference_conv,
            quest_m8_page_stripe=quest_m8_page_stripe,
            cached_gemm_selector=cached_gemm_selector,
            cached_gemm_cold_seed=cached_gemm_cold_seed,
            d7_c1_draft_graph=d7_c1_draft_graph,
            enable_outer_stage_timing=enable_outer_stage_timing,
            disable_outer_stage_timing=disable_outer_stage_timing,
            proposal_temperature_scale=proposal_temperature_scale,
            quest_selector_extension=(selector_binding[0] if selector_binding else None),
            quest_selector_extension_sha256=(selector_binding[1] if selector_binding else None),
            row_local_dualphase_extension=(dualphase_binding[0] if dualphase_binding else None),
            row_local_dualphase_extension_sha256=(
                dualphase_binding[1] if dualphase_binding else None
            ),
            release_probe_root=probe_root,
            release_probe_sha256=probe_sha256,
            release_probe_site_sha256=probe_site_sha256,
            tree_score_capture_root=tree_capture_root,
            tree_score_capture_hashes=tree_capture_hashes,
            tree_public_manifest_sha256=tree_manifest_sha256,
            artifact_mode=selected_mode,
            runtime_launcher_path=remote_launcher,
            runtime_binding_sha256=runtime_binding_hashes,
        )
    request = template.get("request")
    if not isinstance(request, dict):
        raise PrepareError("template request is invalid")
    request = json.loads(json.dumps(request))
    request.update(
        {
            "output": f"{remote}/result-{max_tokens}.json",
            "request_id": request_id,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
    )
    artifact_manifest = _json(manifest_source, "assurance manifest")
    if proposal_document is not None:
        proposal_bindings = {
            "prompt_tokens": prompt_tokens,
            "request_id": request_id,
            "runtime_artifact_manifest_sha256": _sha256(manifest_source),
            "semantic_source_sha256": artifact_manifest.get("semantic_source_sha256"),
            "snapshot_manifest_sha256": request.get("cache_manifest_sha256"),
        }
        for field, expected in proposal_bindings.items():
            if proposal_document[field] != expected:
                raise PrepareError(f"round-equivalence proposal differs at {field}")
    receipt_document = _json(receipt, "state producer receipt")
    transition_document = _json(transition_receipt, "state consumer-transition receipt")
    try:
        producer = validate_producer_receipt(receipt_document)
        transition = validate_consumer_transition_receipt(transition_document)
    except StateProviderError as error:
        raise PrepareError(f"state receipt chain is invalid: {error}") from error
    if runtime_file_binding_document is not None:
        runtime_files = runtime_file_binding_document["files"]
        expected_runtime_links = {
            "gdn_adapter": runtime_binding_hashes["qwen_gdn_linear_attn"],
            "qwen3_next": runtime_binding_hashes["qwen3_next"],
            "qwen_text_rope": runtime_binding_hashes["qwen_text_rope"],
            "model_runner": transition.document["consumer_accepted_path_commit_sha256"],
        }
        for label, expected_digest in expected_runtime_links.items():
            if runtime_files[label]["sha256"] != expected_digest:
                raise PrepareError(
                    f"runtime file binding table differs from authenticated {label}"
                )
    if selected_mode == "assurance" and not layer_diagnostic_only:
        transition_links = {
            "producer_receipt_sha256": producer.digest,
            "producer_runtime_artifact_manifest_sha256": producer.document[
                "runtime_artifact_manifest_sha256"
            ],
            "producer_semantic_source_sha256": producer.document["semantic_source_sha256"],
            "producer_accepted_path_commit_sha256": producer.document[
                "accepted_path_commit_sha256"
            ],
            "producer_snapshot_format_sha256": producer.document["snapshot_format_sha256"],
            "snapshot_payload_receipt_sha256": producer.document["snapshot_payload_receipt_sha256"],
            "snapshot_selection_sha256": _sha256(selection),
            "consumer_runtime_artifact_manifest_sha256": _sha256(manifest_source),
            "consumer_semantic_source_sha256": artifact_manifest.get("semantic_source_sha256"),
            "consumer_state_exporter_sha256": members[ARTIFACT_OUTPUTS["state_exporter"]],
        }
        for field, expected in transition_links.items():
            if transition.document[field] != expected:
                raise PrepareError(f"consumer-transition receipt differs at {field}")
    oracle_identity = None
    if selected_mode == "assurance":
        identity_builder = (
            _non_promotable_diagnostic_identity
            if allow_unqualified_layer_diagnostic_context
            else _oracle_identity
        )
        oracle_identity = identity_builder(
            artifact_manifest=artifact_manifest,
            artifact_manifest_sha256=_sha256(manifest_source),
            command_payload=command_payload,
            receipt_document=receipt_document,
            selection=selection_document,
            request=request,
            request_id=request_id,
        )

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    staging.chmod(0o700)
    try:
        shutil.copytree(artifact_root, staging / "artifact-pair")
        if probe_source is not None and probe_site_source is not None:
            local_probe = staging / "release-activation-probe"
            local_probe.mkdir(mode=0o700)
            shutil.copy2(probe_source, local_probe / "probe.py")
            shutil.copy2(probe_site_source, local_probe / "sitecustomize.py")
            (staging / "probe").mkdir(mode=0o700)
        if tree_capture_sources is not None and tree_manifest_source is not None:
            local_tree = staging / "dflash-tree-score-capture"
            (local_tree / "capture_site").mkdir(mode=0o700, parents=True)
            local_tree.chmod(0o700)
            shutil.copy2(
                tree_capture_sources["capture_contract"], local_tree / "capture_contract.py"
            )
            shutil.copy2(tree_capture_sources["capture_hook"], local_tree / "capture_hook.py")
            shutil.copy2(
                tree_capture_sources["capture_site"], local_tree / "capture_site/sitecustomize.py"
            )
            shutil.copy2(tree_capture_sources["score_lattice"], local_tree / "score_lattice.py")
            shutil.copy2(tree_manifest_source, local_tree / "public-manifest.json")
            (staging / "dflash-tree-score-output").mkdir(mode=0o700)
        if selected_mode == "assurance":
            shutil.copy2(receipt, staging / "state-producer-receipt.json")
            shutil.copy2(transition_receipt, staging / "state-consumer-transition-receipt.json")
            shutil.copy2(selection, staging / "snapshot-selection.json")
            if proposal_source is not None:
                shutil.copy2(proposal_source, staging / "round-equivalence-proposal.json")
            (staging / "capture").mkdir(mode=0o700)
            (staging / "capture" / ".keep").write_bytes(b"")
            if runtime_file_binding_source is not None:
                shutil.copy2(
                    runtime_file_binding_source,
                    staging / RUNTIME_FILE_BINDINGS_NAME,
                )
        command = staging / "command.sh"
        command.write_text(command_payload, encoding="utf-8")
        command.chmod(0o600)
        if oracle_identity is not None:
            identity_path = staging / ORACLE_IDENTITY_NAME
            identity_path.write_text(
                json.dumps(oracle_identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            identity_path.chmod(0o600)
        controller = staging / "controller" / "assurance_once.py"
        controller.parent.mkdir(mode=0o700)
        shutil.copy2(Path(assurance_once_module.__file__).resolve(strict=True), controller)
        bundled_launcher = staging / "controller" / "stock-vllm-frozenlock"
        shutil.copy2(launcher, bundled_launcher)
        bundled_launcher.chmod(0o700)
        request_runner = staging / "controller" / "run_public_fixture.py"
        shutil.copy2(runner_input, request_runner)

        bindings = template.get("bindings")
        if not isinstance(bindings, dict):
            raise PrepareError("template spec bindings are invalid")
        bindings = json.loads(json.dumps(bindings))
        if selected_mode == "release":
            bindings = {
                name: binding for name, binding in bindings.items() if name in RELEASE_BINDINGS
            }
        inherited_launcher = bindings.get("launcher")
        if not isinstance(inherited_launcher, dict) or not isinstance(
            inherited_launcher.get("path"), str
        ):
            raise PrepareError("template spec lacks the runtime binding launcher")
        bindings["launcher"] = {
            "path": str(remote_launcher),
            "sha256": _sha256(launcher),
        }
        explicit_sources = runtime_binding_sources or {}
        bindings.update(
            _runtime_binding_records(
                bindings,
                explicit_sources,
                allow_changes=allow_runtime_binding_changes,
            )
        )
        if selected_mode == "assurance":
            remote_artifact = f"{remote}/artifact-pair/assurance"
            for name, output_path in artifact_outputs.items():
                bindings[name] = {
                    "path": f"{remote_artifact}/files/{output_path}",
                    "sha256": members[output_path],
                }
            bindings["assurance_manifest"] = {
                "path": f"{remote_artifact}/manifest.json",
                "sha256": _sha256(manifest_source),
            }
            bindings["state_producer_receipt"] = {
                "path": f"{remote}/state-producer-receipt.json",
                "sha256": _sha256(receipt),
            }
            bindings["state_consumer_transition_receipt"] = {
                "path": f"{remote}/state-consumer-transition-receipt.json",
                "sha256": _sha256(transition_receipt),
            }
            bindings["snapshot_payload_selection"] = {
                "path": f"{remote}/snapshot-selection.json",
                "sha256": _sha256(selection),
            }
            expected_assurance_bindings = (
                FULL_INSTRUMENTATION_BINDINGS
                if "full_instrumentation_contract" in artifact_outputs
                else ASSURANCE_BINDINGS
            )
            if set(bindings) != expected_assurance_bindings:
                raise PrepareError("constructed assurance bindings are not dependency-closed")
        elif set(bindings) != RELEASE_BINDINGS:
            raise PrepareError("constructed release bindings are not dependency-closed")

        spec = {
            "schema": SCHEMA,
            "artifact_mode": selected_mode,
            "qualification_dir": str(remote),
            "command": {"path": f"{remote}/command.sh", "sha256": _sha256(command)},
            "runner": {
                "path": f"{remote}/controller/run_public_fixture.py",
                "sha256": _sha256(request_runner),
            },
            "bindings": bindings,
            "request": request,
        }
        spec_path = staging / "one-shot.json"
        spec_path.write_text(
            json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        spec_path.chmod(0o600)
        shutil.copy2(Path(__file__), staging / "controller" / "assurance_prepare.py")
        bundled_artifacts = staging / "artifact-pair"
        for path in staging.rglob("*"):
            if path.is_file() and not path.is_relative_to(bundled_artifacts):
                path.chmod(0o600)
        staging.rename(output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        "bundle": str(output),
        "command_sha256": _sha256(output / "command.sh"),
        "manifest_sha256": _sha256(output / f"artifact-pair/{selected_mode}/manifest.json"),
        "oracle_identity_sha256": (
            _sha256(output / ORACLE_IDENTITY_NAME) if oracle_identity is not None else None
        ),
        "request_id": request_id,
        "remote_qualification": str(remote),
        "spec_sha256": _sha256(output / "one-shot.json"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-assurance-prepare",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""NAME
    qwen-assurance-prepare - construct one self-contained Qwen assurance bundle

SYNOPSIS
    qwen-assurance-prepare --template-spec FILE --template-command FILE
      --artifact-root DIR --launcher FILE --state-producer-receipt FILE
      --state-consumer-transition-receipt FILE
      --snapshot-selection FILE --output DIR --remote-qualification DIR
      --request-id ID --max-tokens N --timeout SECONDS [--target-only]
      [--round-equivalence-precommit]
      [--round-equivalence-proposal FILE]
      [--layer-position ABSOLUTE_POSITION | --layer-positions P1,P2,...]
      [--layer-indices L0,L1,...] [--stop-after-layer-capture]
      [--gemma-rms-fp32-kernel]
      [--cached-accepted-commit]
      [--fixed-serial-conv-m8]
      [--fixed-serial-conv-m8-crosscheck]
      [--greedy-m8-verifier]
      [--d7-c1-draft-graph]
      [--serial-batched]
      [--serial-reference-conv]
      [--cached-gemm-selector] [--cached-gemm-cold-seed]
      [--enable-outer-stage-timing | --disable-outer-stage-timing]
      [--quest-selector-extension FILE --quest-selector-extension-sha256 SHA256]
      [--row-local-dualphase-extension FILE
       --row-local-dualphase-extension-sha256 SHA256]
      [--runner-source FILE]
      [--qwen-gdn-source FILE --qwen3-next-source FILE --qwen-text-rope-source FILE
       --runtime-snapshot-selection-source FILE]

DESCRIPTION
    Builds one create-only qualification tree whose command, artifact members,
    hashes, import paths, receipt, snapshot, and one-shot spec agree by construction.

OPTIONS
    All paths and run identity fields are required. DFlash is limited to 2048
    externally committed tokens. A target-only oracle accepts 2049 API tokens so
    its final lookahead call can certify 2048 canonical state transitions.
    --target-only derives that serial command from the authenticated M8 template.
    --layer-position selects one absolute prefix length for paired attention capture.
    --layer-positions selects up to eight absolute positions from one restored run.
    --layer-indices limits hashing and Quest micro-capture to selected decoder
    layers. --stop-after-layer-capture clamps generation to the last requested
    absolute position (plus the target-only lookahead when required).
    --gemma-rms-fp32-kernel selects the launcher's pinned cross-shape exact
    gfx1201 Gemma RMSNorm implementation for M1/M8 transition assurance.
    --cached-accepted-commit replaces whole-model accepted replay with the
    exact cached per-layer GDN/convolution commit path.
    --fixed-serial-conv-m8 selects the exact fixed-width one-launch M8
    convolution and eight prefix checkpoints; it requires cached accepted commit.
    --fixed-serial-conv-m8-crosscheck runs the qualified serial M1 convolution
    from an independent state clone and bit-compares every output and checkpoint.
    --greedy-m8-verifier enables the exact default-off D7/C1 target-argmax
    verifier. Unsupported geometry or sampling policy stays on the general sampler.
    --d7-c1-draft-graph enables the qualified private D7/C1 draft-only HIP graph.
    It requires the greedy M8 verifier and fixed serial M8 convolution, removes
    target --enforce-eager, and binds the target to mode 0 / cudagraph NONE.
    --serial-batched selects the bit-exact one-launch serial recurrence/conv path.
    --grouped-ba-prefix evaluates the B/A projection in two exact M4 groups.
    --grouped-ba-crosscheck compares every grouped B/A result to serial M1 in assurance.
    --serial-reference-conv keeps batched recurrence/commit enabled but verifies
    convolution history through the exact ordered serial transition.
    --cached-gemm-selector enables exact cached row-local Quest ranking.
    --cached-gemm-cold-seed populates that cache from the exact first-round scorer.
    --round-equivalence-precommit captures canonical logical state after target
    execution and immediately before serial or M8 accepted-path commit. It is valid
    only for a full assurance run and is compiled out of release artifacts.
    --round-equivalence-proposal supplies the exact authenticated D7 proposal set
    evaluated by that run and requires --round-equivalence-precommit.
    Both extension paths require their exact SHA-256 and are loaded only after
    startup authentication. The selected artifact projection supplies the runtime
    Quest adapter directly, before any mutable project path.
    The runner and four runtime source options inspect local postimages while
    preserving authenticated remote paths inherited from the template.

OPERATION
    Verifies the paired artifact, rewrites the command structurally, derives every
    diagnostic binding from the assurance manifest, and publishes atomically.

EXAMPLES
    qwen-assurance-prepare --template-spec prior.json --template-command prior.sh \\
      --artifact-root pair --launcher stock-vllm-frozenlock \\
      --state-producer-receipt receipt.json \\
      --state-consumer-transition-receipt transition.json \\
      --snapshot-selection snapshot.json --output /tmp/run \\
      --remote-qualification /home/lewis/projects/qwen-r9700/artifacts/qualifications/run \\
      --request-id run --max-tokens 2048 --timeout 1800

FILES
    Produces artifact-pair/, capture/, command.sh, oracle-identity.json,
    one-shot.json, controller/, the immutable snapshot selection, producer receipt,
    and exact consumer-transition receipt.

PATHS
    The remote directory must be below the project's qualifications root.

SECURITY NOTES
    Inputs must be owned regular files/directories. Existing outputs and symlinks
    are refused; the complete bundle is published by one rename.

EXIT STATUS
    0 on success; 2 when construction or validation fails.

AUTHORS
    Qwen R9700 inference lab maintainers.
""",
    )
    parser.add_argument("--template-spec", required=True, type=Path)
    parser.add_argument("--template-command", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--launcher", required=True, type=Path)
    parser.add_argument("--state-producer-receipt", required=True, type=Path)
    parser.add_argument("--state-consumer-transition-receipt", required=True, type=Path)
    parser.add_argument("--snapshot-selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--remote-qualification", required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--max-tokens", required=True, type=int)
    parser.add_argument("--timeout", required=True, type=int)
    position_group = parser.add_mutually_exclusive_group()
    position_group.add_argument(
        "--layer-position",
        type=int,
        help="absolute prefix length for one paired Quest attention capture",
    )
    position_group.add_argument(
        "--layer-positions",
        type=lambda value: tuple(int(item) for item in value.split(",")),
        help="comma-separated absolute positions for one bounded transition capture",
    )
    parser.add_argument(
        "--layer-indices",
        type=lambda value: tuple(int(item) for item in value.split(",")),
        help="comma-separated decoder layer indices to hash at the selected position",
    )
    parser.add_argument(
        "--stop-after-layer-capture",
        action="store_true",
        help="clamp generation to the final requested capture position",
    )
    parser.add_argument(
        "--layer-diagnostic-only",
        action="store_true",
        help=(
            "retain authenticated token/layer evidence but omit expensive draft and "
            "full-state streams"
        ),
    )
    parser.add_argument(
        "--allow-unqualified-layer-diagnostic-context",
        action="store_true",
        help=(
            "allow one explicitly bounded, non-promotable layer diagnostic outside "
            "the production 60298/249957 qualification contexts"
        ),
    )
    parser.add_argument(
        "--target-only",
        action="store_true",
        help="derive a serial target-only command from the authenticated DFlash template",
    )
    parser.add_argument(
        "--state-committed-counts",
        type=lambda value: tuple(int(item) for item in value.split(",")),
        help="hash authoritative state only at these exact committed-token boundaries",
    )
    parser.add_argument(
        "--round-equivalence-precommit",
        action="store_true",
        help="capture the assurance-only post-M8/pre-commit canonical-state witness",
    )
    parser.add_argument(
        "--round-equivalence-proposal",
        type=Path,
        help="owner-only self-authenticating seven-token proposal fixture",
    )
    parser.add_argument(
        "--round-equivalence-zero-commit",
        action="store_true",
        help="capture one real M8 round while suppressing every external/canonical commit",
    )
    parser.add_argument(
        "--gemma-rms-fp32-kernel",
        action="store_true",
        help="enable the launcher's pinned M1/M8-exact gfx1201 Gemma RMSNorm kernel",
    )
    parser.add_argument(
        "--cached-accepted-commit",
        action="store_true",
        help="enable exact cached per-layer accepted-state commit",
    )
    parser.add_argument(
        "--fixed-serial-conv-m8",
        action="store_true",
        help="enable exact one-launch fixed-width M8 convolution and checkpoints",
    )
    parser.add_argument(
        "--fixed-serial-conv-m8-crosscheck",
        action="store_true",
        help="bit-compare fixed M8 convolution to independent serial M1 transitions",
    )
    parser.add_argument(
        "--greedy-m8-verifier",
        action="store_true",
        help="enable the exact default-off greedy D7/C1/M8 target-argmax verifier",
    )
    parser.add_argument(
        "--d7-c1-draft-graph",
        action="store_true",
        help="enable the qualified exact D7/C1 draft-only HIP graph",
    )
    parser.add_argument(
        "--serial-batched",
        action="store_true",
        help="enable the bit-exact one-launch serial GDN/conv path",
    )
    parser.add_argument(
        "--grouped-ba-prefix",
        action="store_true",
        help="enable prefix-grouped exact B/A projections",
    )
    parser.add_argument(
        "--grouped-ba-crosscheck",
        action="store_true",
        help="compare grouped B/A projections to serial M1 during assurance",
    )
    parser.add_argument(
        "--batched-ba",
        action="store_true",
        help="enable one-call M8 B/A projection after exact cross-shape qualification",
    )
    parser.add_argument(
        "--batched-ba-crosscheck",
        action="store_true",
        help="compare batched B/A projections to serial M1 during assurance",
    )
    parser.add_argument(
        "--serial-reference-conv",
        action="store_true",
        help="verify M8 convolution through ordered serial target transitions",
    )
    parser.add_argument(
        "--quest-m8-page-stripe",
        action="store_true",
        help="enable the experimental globally deduplicated Quest M8 page-stripe schedule",
    )
    timing_group = parser.add_mutually_exclusive_group()
    timing_group.add_argument(
        "--enable-outer-stage-timing",
        action="store_true",
        help="enable bounded outer HIP-event stage instrumentation in the run",
    )
    timing_group.add_argument(
        "--disable-outer-stage-timing",
        action="store_true",
        help="disable outer HIP-event stage instrumentation in the release run",
    )
    parser.add_argument(
        "--proposal-temperature-scale",
        help=(
            "bind one positive DFlash2 proposal-temperature scale; inherited "
            "single-scale and sweep calibration is otherwise removed"
        ),
    )
    parser.add_argument(
        "--cached-gemm-selector",
        action="store_true",
        help="enable exact cached row-local Quest selector ranking",
    )
    parser.add_argument(
        "--cached-gemm-cold-seed",
        action="store_true",
        help="seed immutable cached centroids from the exact first-round Quest scorer",
    )
    parser.add_argument("--quest-selector-extension")
    parser.add_argument("--quest-selector-extension-sha256")
    parser.add_argument("--row-local-dualphase-extension")
    parser.add_argument("--row-local-dualphase-extension-sha256")
    parser.add_argument(
        "--release-activation-probe",
        type=Path,
        help="inject a hash-bound external dispatch observer into a release-only run",
    )
    parser.add_argument(
        "--dflash-tree-score-capture",
        type=Path,
        help="inject the external create-only DFlash rank/score capture into a release run",
    )
    parser.add_argument(
        "--dflash-tree-public-manifest",
        type=Path,
        help="public fixture manifest authenticated by the DFlash rank/score capture",
    )
    parser.add_argument(
        "--artifact-mode",
        choices=("assurance", "release"),
        help="override the template projection; defaults to the template artifact_mode",
    )
    parser.add_argument("--qwen-gdn-source", type=Path)
    parser.add_argument("--qwen3-next-source", type=Path)
    parser.add_argument("--qwen-text-rope-source", type=Path)
    parser.add_argument("--runtime-snapshot-selection-source", type=Path)
    parser.add_argument(
        "--runtime-file-bindings",
        type=Path,
        help=(
            "complete explicit runtime path/SHA table required by unqualified-context "
            "diagnostics"
        ),
    )
    parser.add_argument(
        "--allow-runtime-binding-changes",
        action="store_true",
        help="authorize bindings for a coordinated installed-runtime source swap",
    )
    parser.add_argument("--runner-source", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    runtime_sources = {
        name: path
        for name, path in {
            "qwen_gdn_linear_attn": args.qwen_gdn_source,
            "qwen3_next": args.qwen3_next_source,
            "qwen_text_rope": args.qwen_text_rope_source,
            "snapshot_selection": args.runtime_snapshot_selection_source,
        }.items()
        if path is not None
    }
    try:
        result = prepare_bundle(
            template_spec=args.template_spec,
            template_command=args.template_command,
            artifact_root=args.artifact_root,
            launcher=args.launcher,
            state_producer_receipt=args.state_producer_receipt,
            state_consumer_transition_receipt=args.state_consumer_transition_receipt,
            snapshot_selection=args.snapshot_selection,
            output=args.output,
            remote_qualification=args.remote_qualification,
            request_id=args.request_id,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            force_target_only=args.target_only,
            layer_position=args.layer_position,
            layer_positions=args.layer_positions,
            layer_indices=args.layer_indices,
            stop_after_layer_capture=args.stop_after_layer_capture,
            layer_diagnostic_only=args.layer_diagnostic_only,
            allow_unqualified_layer_diagnostic_context=(
                args.allow_unqualified_layer_diagnostic_context
            ),
            state_committed_counts=args.state_committed_counts,
            round_equivalence_precommit=args.round_equivalence_precommit,
            round_equivalence_proposal=args.round_equivalence_proposal,
            round_equivalence_zero_commit=args.round_equivalence_zero_commit,
            gemma_rms_fp32_kernel=args.gemma_rms_fp32_kernel,
            cached_accepted_commit=args.cached_accepted_commit,
            fixed_serial_conv_m8=args.fixed_serial_conv_m8,
            fixed_serial_conv_m8_crosscheck=args.fixed_serial_conv_m8_crosscheck,
            greedy_m8_verifier=args.greedy_m8_verifier,
            d7_c1_draft_graph=args.d7_c1_draft_graph,
            serial_batched=args.serial_batched,
            grouped_ba_prefix=args.grouped_ba_prefix,
            grouped_ba_crosscheck=args.grouped_ba_crosscheck,
            batched_ba=args.batched_ba,
            batched_ba_crosscheck=args.batched_ba_crosscheck,
            serial_reference_conv=args.serial_reference_conv,
            quest_m8_page_stripe=args.quest_m8_page_stripe,
            cached_gemm_selector=args.cached_gemm_selector,
            cached_gemm_cold_seed=args.cached_gemm_cold_seed,
            enable_outer_stage_timing=args.enable_outer_stage_timing,
            disable_outer_stage_timing=args.disable_outer_stage_timing,
            proposal_temperature_scale=args.proposal_temperature_scale,
            quest_selector_extension=args.quest_selector_extension,
            quest_selector_extension_sha256=args.quest_selector_extension_sha256,
            row_local_dualphase_extension=args.row_local_dualphase_extension,
            row_local_dualphase_extension_sha256=args.row_local_dualphase_extension_sha256,
            release_activation_probe=args.release_activation_probe,
            dflash_tree_score_capture=args.dflash_tree_score_capture,
            dflash_tree_public_manifest=args.dflash_tree_public_manifest,
            artifact_mode=args.artifact_mode,
            runner_source=args.runner_source,
            runtime_binding_sources=runtime_sources,
            allow_runtime_binding_changes=args.allow_runtime_binding_changes,
            runtime_file_bindings=args.runtime_file_bindings,
        )
    except (PrepareError, OSError) as error:
        print(f"qwen-assurance-prepare: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
