"""Fast, evidence-labelled projections for the occupied-250K DFlash lane.

This module deliberately separates two questions which were previously
coupled:

* Does a fixed-shape kernel implementation run faster at the exact production
  geometry?
* Does a semantic change preserve model quality with an occupied 249,957-token
  prefix?

The first question can normally be answered by standalone GPU-event controls
in seconds.  The second sometimes requires the expensive occupied-prefix
qualification.  A projection is never reported as an occupied-context
measurement unless the input bundle records that qualification explicitly.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_CONTEXT_TOKENS = 249_957
DEFAULT_VERIFICATION_ROWS = 9
DEFAULT_ATTENTION_LAYERS = 16
DEFAULT_GDN_LAYERS = 48
NATIVE_B_SCHEDULE_CONTRACT = "qwen3.8-27b-native-b-w4-v2"
NATIVE_B_W4_PROJECTION_COUNT = 256
NATIVE_B_EXCLUDED_BF16_GDN_BA_COUNT = 48
NATIVE_B_W4_SCHEDULE = (
    ("mlp_gate_up", 34_816, 5_120, 64),
    ("mlp_down", 5_120, 17_408, 64),
    ("gdn_qkvz", 16_384, 5_120, 48),
    ("gdn_out", 5_120, 6_144, 48),
    ("attention_qkv", 14_336, 5_120, 16),
    ("attention_out", 5_120, 6_144, 16),
)
DEFAULT_BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "kernel-250k"
    / "r9700-dflash8-baseline-v1.json"
)

STAGE_NAMES = (
    "native_b",
    "quest",
    "gdn",
    "draft",
    "runtime_other",
)


@dataclass(frozen=True)
class ChangePolicy:
    occupied_prefix_required: bool
    required_gates: tuple[str, ...]
    rationale: str


CHANGE_POLICIES: dict[str, ChangePolicy] = {
    "native_b_implementation": ChangePolicy(
        False,
        (
            "randomized output parity at every affected M/N/K/dtype",
            "exact 256-projection M=9 W4 schedule timing",
            "bounded live fast-path activation check",
        ),
        "A numerically identical projection kernel is independent of prefix occupancy.",
    ),
    "quest_kernel_implementation": ChangePolicy(
        False,
        (
            "selected-token attention parity at the affected page budgets",
            "exact 249,957-token physical-layout timing",
            "bounded live compiled-path activation check",
        ),
        "An implementation-only consumer/selector rewrite can use synthetic exact-layout pages.",
    ),
    "gdn_kernel_implementation": ChangePolicy(
        False,
        (
            "two-round RecoverSSM parity for accepted counts 1,5,9",
            "exact M=9, 48-layer weighted schedule timing",
            "bounded live M=9 dispatch check",
        ),
        "A fixed-shape recurrence implementation can be checked against the materialized oracle.",
    ),
    "draft_kernel_implementation": ChangePolicy(
        False,
        (
            "proposal-logit/token parity",
            "fixed depth-8 proposal timing",
            "complete-output short-context acceptance benchmark",
        ),
        "Kernel parity and proposal cadence do not require a cold 250K cache fill.",
    ),
    "runtime_wiring": ChangePolicy(
        False,
        (
            "hash-pinned build/install check",
            "bounded live fast-path activation check",
            "complete-output greedy/speculative parity",
        ),
        "Wiring changes need a live smoke test, but not an occupied prefix if semantics are fixed.",
    ),
    "attention_policy": ChangePolicy(
        True,
        (
            "exact 249,957-token page-selection quality",
            "long-range retrieval qualification",
            "occupied-prefix decode timing",
        ),
        "Page budget, recent/ranked split, or ranking changes alter which history is visible.",
    ),
    "kv_cache_contract": ChangePolicy(
        True,
        (
            "occupied-prefix cache-layout/capacity gate",
            "prefix-hit equivalence",
            "long-range retrieval qualification",
        ),
        "KV layout, mapping, dtype, or quantization changes are context-dependent semantics.",
    ),
    "position_or_mask_semantics": ChangePolicy(
        True,
        (
            "occupied-prefix causal-position parity",
            "long-range retrieval qualification",
            "complete-output parity",
        ),
        (
            "Position and mask changes cannot be qualified from a synthetic selected-page "
            "timing alone."
        ),
    ),
    "gdn_state_semantics": ChangePolicy(
        True,
        (
            "multi-round state/rollback oracle",
            "prefix-cache migration/rebind qualification",
            "occupied-prefix complete-output parity",
        ),
        (
            "State ownership, commit, rollback, or prefix migration changes span requests and "
            "cache hits."
        ),
    ),
    "speculation_semantics": ChangePolicy(
        True,
        (
            "greedy target-token parity",
            "complete-output quality qualification",
            "occupied-prefix acceptance and decode timing",
        ),
        "Proposal selection, verification, acceptance, or commit changes can alter emitted tokens.",
    ),
    "model_quantization": ChangePolicy(
        True,
        (
            "full-model numerical/quality qualification",
            "long-context retrieval qualification",
            "occupied-prefix capacity and decode timing",
        ),
        "Changing weight or activation quantization changes model numerics and usually memory fit.",
    ),
}


class ProjectionError(ValueError):
    """The measurement bundle does not satisfy the projection contract."""


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProjectionError(f"{label} must be a JSON object")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProjectionError(f"{label} must be a positive integer")
    return value


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProjectionError(f"{label} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ProjectionError(f"{label} must be a finite non-negative number")
    return result


def load_bundle(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProjectionError(f"cannot load measurement bundle {path}: {error}") from error
    return _object(value, "measurement bundle")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_component_result(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProjectionError(f"cannot load component result {resolved}: {error}") from error
    return resolved, _object(document, f"component result {resolved}")


def apply_component_result(bundle: dict[str, Any], stage_name: str, path: Path) -> None:
    """Replace one canonical stage from a known exact-layout benchmark JSON."""

    if stage_name not in {"native_b", "quest", "gdn"}:
        raise ProjectionError(
            "--component-result supports native_b, quest, and gdn; use --stage-ms for other stages"
        )
    scenario = _validate_scenario(bundle)
    stages = _object(bundle.get("stages"), "stages")
    stage = _object(stages.get(stage_name), f"stages.{stage_name}")
    resolved, document = _load_component_result(path)
    evidence = {
        "kind": "exact_layout_standalone",
        "source": str(resolved),
        "sha256": _sha256_file(resolved),
    }
    correctness: dict[str, Any] = {"status": "not_recorded"}

    if stage_name == "native_b":
        if document.get("rows") != scenario["verification_rows"]:
            raise ProjectionError("Native-B component result must use rows=9")
        projection_count = document.get("w4_projection_count")
        if projection_count != NATIVE_B_W4_PROJECTION_COUNT:
            legacy = (
                " (the legacy 304-call schedule overcounted 48 BF16 GDN b/a linears)"
                if projection_count == 304
                else ""
            )
            raise ProjectionError(
                "Native-B component result must contain the exact 256-call W4 schedule" + legacy
            )
        excluded_bf16 = document.get("excluded_bf16_gdn_ba_count")
        if excluded_bf16 != NATIVE_B_EXCLUDED_BF16_GDN_BA_COUNT:
            raise ProjectionError(
                "Native-B component result must record 48 excluded BF16 GDN b/a linears"
            )
        if document.get("schedule_contract") != NATIVE_B_SCHEDULE_CONTRACT:
            raise ProjectionError(
                f"Native-B component result must use schedule_contract={NATIVE_B_SCHEDULE_CONTRACT}"
            )
        experimental_split_k = document.get("experimental_split_k_enabled")
        if not isinstance(experimental_split_k, bool):
            raise ProjectionError(
                "Native-B component result must record experimental_split_k_enabled as a boolean"
            )
        raw_schedule = document.get("stages")
        if (
            not isinstance(raw_schedule, list)
            or len(raw_schedule) != len(NATIVE_B_W4_SCHEDULE)
            or not all(isinstance(item, dict) for item in raw_schedule)
            or tuple(
                (item.get("name"), item.get("n"), item.get("k"), item.get("count"))
                for item in raw_schedule
            )
            != NATIVE_B_W4_SCHEDULE
        ):
            raise ProjectionError(
                "Native-B component result does not match the canonical W4 stage inventory"
            )
        evidence["rows"] = scenario["verification_rows"]
        milliseconds = _finite_nonnegative(
            document.get("weighted_schedule_median_ms"),
            "Native-B weighted_schedule_median_ms",
        )
        evidence.update(
            {
                "projection_count": projection_count,
                "excluded_bf16_gdn_ba_count": excluded_bf16,
                "schedule_contract": NATIVE_B_SCHEDULE_CONTRACT,
                "experimental_split_k_enabled": experimental_split_k,
            }
        )
        parity = document.get("parity")
        if isinstance(parity, dict):
            max_abs = _finite_nonnegative(parity.get("max_abs_delta"), "Native-B max_abs_delta")
            nonidentical = parity.get("nonidentical_bf16_elements")
            passed = max_abs <= 0.5 and isinstance(nonidentical, int) and nonidentical == 0
            correctness = {
                "status": "pass" if passed else "fail",
                "max_abs_delta": max_abs,
                "nonidentical_bf16_elements": nonidentical,
            }
        stage.update(
            {
                "milliseconds": milliseconds,
                "scope": "per_round",
                "count": 1,
                "evidence": evidence,
                "correctness": correctness,
            }
        )
        return

    if stage_name == "quest":
        if document.get("context") != scenario["context_tokens"]:
            raise ProjectionError("Quest component result must use context=249957")
        if document.get("physical_block_size") != 1_648:
            raise ProjectionError("Quest component result must use 1648-token physical blocks")
        if document.get("output_rows") != scenario["verification_rows"]:
            raise ProjectionError("Quest component result must cover the exact 9 output rows")
        timings = _object(document.get("timings"), "Quest timings")
        timing = _object(timings.get(str(scenario["page_budget"])), "Quest selected-page timing")
        milliseconds = _finite_nonnegative(timing.get("median_ms"), "Quest median_ms")
        parity = _object(document.get("parity"), "Quest parity")
        max_abs = _finite_nonnegative(parity.get("max_abs"), "Quest parity.max_abs")
        cosine = _finite_nonnegative(parity.get("cosine"), "Quest parity.cosine")
        finite = parity.get("finite") is True
        passed = finite and max_abs <= 0.01 and cosine >= 0.9999
        evidence.update(
            {
                "context_tokens": scenario["context_tokens"],
                "rows": document["output_rows"],
                "page_budget": scenario["page_budget"],
                "physical_block_tokens": 1_648,
            }
        )
        stage.update(
            {
                "milliseconds": milliseconds,
                "scope": "per_layer",
                "count": scenario["attention_layers"],
                "evidence": evidence,
                "correctness": {
                    "status": "pass" if passed else "fail",
                    "max_abs": max_abs,
                    "cosine": cosine,
                    "finite": finite,
                },
            }
        )
        return

    if document.get("rows") != scenario["verification_rows"]:
        raise ProjectionError("GDN component result must use rows=9")
    if document.get("layer_count") != scenario["gdn_layers"]:
        raise ProjectionError("GDN component result must time all 48 layers")
    milliseconds = _finite_nonnegative(document.get("schedule_median_ms"), "GDN schedule_median_ms")
    finite = document.get("output_finite") is True
    evidence.update({"rows": 9, "layer_count": 48})
    stage.update(
        {
            "milliseconds": milliseconds,
            "scope": "per_round",
            "count": 1,
            "evidence": evidence,
            "correctness": {
                "status": "not_recorded" if finite else "fail",
                "finite": finite,
                "note": "two-round parity artifact is separate; attest it with --qualified-stage",
            },
        }
    )


def _validate_scenario(bundle: Mapping[str, Any]) -> dict[str, Any]:
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ProjectionError(f"schema_version must be {SCHEMA_VERSION}")
    scenario = _object(bundle.get("scenario"), "scenario")
    context = _positive_int(scenario.get("context_tokens"), "scenario.context_tokens")
    rows = _positive_int(scenario.get("verification_rows"), "scenario.verification_rows")
    attention_layers = _positive_int(scenario.get("attention_layers"), "scenario.attention_layers")
    gdn_layers = _positive_int(scenario.get("gdn_layers"), "scenario.gdn_layers")
    page_budget = _positive_int(scenario.get("page_budget"), "scenario.page_budget")
    page_tokens = _positive_int(scenario.get("page_tokens"), "scenario.page_tokens")
    if context != DEFAULT_CONTEXT_TOKENS:
        raise ProjectionError(
            f"scenario.context_tokens must be the exact occupied layout {DEFAULT_CONTEXT_TOKENS}"
        )
    if rows != DEFAULT_VERIFICATION_ROWS:
        raise ProjectionError(
            "scenario.verification_rows must be the DFlash8 target width "
            f"{DEFAULT_VERIFICATION_ROWS}"
        )
    if attention_layers != DEFAULT_ATTENTION_LAYERS:
        raise ProjectionError(f"scenario.attention_layers must be {DEFAULT_ATTENTION_LAYERS}")
    if gdn_layers != DEFAULT_GDN_LAYERS:
        raise ProjectionError(f"scenario.gdn_layers must be {DEFAULT_GDN_LAYERS}")
    return {
        **scenario,
        "context_tokens": context,
        "verification_rows": rows,
        "attention_layers": attention_layers,
        "gdn_layers": gdn_layers,
        "page_budget": page_budget,
        "page_tokens": page_tokens,
        "selected_tokens": page_budget * page_tokens,
    }


def _validate_stage(name: str, raw: object) -> dict[str, Any]:
    stage = _object(raw, f"stages.{name}")
    milliseconds = _finite_nonnegative(stage.get("milliseconds"), f"stages.{name}.milliseconds")
    count = _positive_int(stage.get("count"), f"stages.{name}.count")
    scope = stage.get("scope")
    if scope not in {"per_round", "per_layer"}:
        raise ProjectionError(f"stages.{name}.scope must be per_round or per_layer")
    if scope == "per_round" and count != 1:
        raise ProjectionError(f"stages.{name}.count must be 1 for per_round timing")
    evidence = _object(stage.get("evidence"), f"stages.{name}.evidence")
    evidence_kind = evidence.get("kind")
    if not isinstance(evidence_kind, str) or not evidence_kind:
        raise ProjectionError(f"stages.{name}.evidence.kind must be a non-empty string")
    correctness = _object(stage.get("correctness"), f"stages.{name}.correctness")
    status = correctness.get("status")
    if status not in {"pass", "fail", "not_recorded", "not_applicable"}:
        raise ProjectionError(
            f"stages.{name}.correctness.status must be pass, fail, not_recorded, or not_applicable"
        )
    return {
        **stage,
        "milliseconds": milliseconds,
        "count": count,
        "scope": scope,
        "evidence": evidence,
        "correctness": correctness,
        "weighted_round_ms": milliseconds * count,
    }


def _validate_qualification(bundle: Mapping[str, Any]) -> dict[str, Any]:
    qualification = _object(bundle.get("qualification", {}), "qualification")
    occupied = qualification.get("occupied_prefix_249957", "not_run")
    if occupied not in {"pass", "fail", "not_run"}:
        raise ProjectionError("qualification.occupied_prefix_249957 must be pass, fail, or not_run")
    quality = qualification.get("long_context_quality", "not_run")
    if quality not in {"pass", "fail", "not_run"}:
        raise ProjectionError("qualification.long_context_quality must be pass, fail, or not_run")
    return {
        **qualification,
        "occupied_prefix_249957": occupied,
        "long_context_quality": quality,
    }


def _change_readiness(
    change_kind: str,
    changed_stages: Sequence[str],
    stages: Mapping[str, Mapping[str, Any]],
    exact_layout_coverage: Mapping[str, bool],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for name in changed_stages:
        if name not in stages:
            reasons.append(f"changed stage {name!r} is not present")
            continue
        stage = stages[name]
        if stage["correctness"]["status"] != "pass":
            reasons.append(f"changed stage {name} has no passing correctness result")
        evidence_kind = stage["evidence"]["kind"]
        if evidence_kind not in {
            "exact_layout_standalone",
            "attested_exact_layout",
            "live_gpu_event",
        }:
            reasons.append(
                f"changed stage {name} uses {evidence_kind!r}, not exact-layout/live GPU timing"
            )
        if evidence_kind == "exact_layout_standalone" and not exact_layout_coverage.get(name, True):
            reasons.append(f"changed stage {name} does not match the exact production geometry")

    if change_kind == "runtime_wiring" and not changed_stages:
        reasons.append("runtime_wiring requires at least one named changed stage")
    return not reasons, reasons


def _exact_layout_coverage(
    scenario: Mapping[str, Any], stages: Mapping[str, Mapping[str, Any]]
) -> dict[str, bool]:
    native = stages["native_b"]
    quest = stages["quest"]
    gdn = stages["gdn"]
    native_evidence = native["evidence"]
    quest_evidence = quest["evidence"]
    gdn_evidence = gdn["evidence"]
    return {
        "native_b": (
            native_evidence["kind"] == "exact_layout_standalone"
            and native_evidence.get("rows") == scenario["verification_rows"]
            and native_evidence.get("projection_count") == NATIVE_B_W4_PROJECTION_COUNT
            and native_evidence.get("excluded_bf16_gdn_ba_count")
            == NATIVE_B_EXCLUDED_BF16_GDN_BA_COUNT
            and native_evidence.get("schedule_contract") == NATIVE_B_SCHEDULE_CONTRACT
        ),
        "quest": (
            quest_evidence["kind"] == "exact_layout_standalone"
            and quest_evidence.get("context_tokens") == scenario["context_tokens"]
            and quest_evidence.get("rows") == scenario["verification_rows"]
            and quest_evidence.get("page_budget") == scenario["page_budget"]
            and quest_evidence.get("physical_block_tokens") == 1_648
        ),
        "gdn": (
            gdn_evidence["kind"] == "exact_layout_standalone"
            and gdn_evidence.get("rows") == scenario["verification_rows"]
            and gdn_evidence.get("layer_count") == scenario["gdn_layers"]
        ),
    }


def project_bundle(
    bundle: Mapping[str, Any],
    *,
    change_kind: str,
    changed_stages: Sequence[str] = (),
    target_tps: float = 100.0,
    acceptance_values: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Validate and project a canonical fixed-layout measurement bundle."""

    if change_kind not in CHANGE_POLICIES:
        choices = ", ".join(sorted(CHANGE_POLICIES))
        raise ProjectionError(f"unknown change_kind {change_kind!r}; choose one of: {choices}")
    target_tps = _finite_nonnegative(target_tps, "target_tps")
    if target_tps == 0:
        raise ProjectionError("target_tps must be greater than zero")
    scenario = _validate_scenario(bundle)
    raw_stages = _object(bundle.get("stages"), "stages")
    missing = [name for name in STAGE_NAMES if name not in raw_stages]
    if missing:
        raise ProjectionError(f"measurement bundle is missing stages: {', '.join(missing)}")
    unknown_changed = sorted(set(changed_stages).difference(STAGE_NAMES))
    if unknown_changed:
        raise ProjectionError(f"unknown changed stages: {', '.join(unknown_changed)}")
    stages = {name: _validate_stage(name, raw_stages[name]) for name in STAGE_NAMES}
    round_ms = sum(stage["weighted_round_ms"] for stage in stages.values())
    if round_ms <= 0:
        raise ProjectionError("projected round latency must be positive")

    if acceptance_values is None:
        raw_values = bundle.get("emitted_tokens_per_round")
        if not isinstance(raw_values, list) or not raw_values:
            raise ProjectionError("emitted_tokens_per_round must be a non-empty JSON array")
        acceptance_values = [
            _finite_nonnegative(value, "emitted_tokens_per_round") for value in raw_values
        ]
    else:
        acceptance_values = [
            _finite_nonnegative(value, "acceptance_values") for value in acceptance_values
        ]
    if any(value <= 0 for value in acceptance_values):
        raise ProjectionError("every emitted-tokens-per-round value must be greater than zero")

    scenarios = [
        {
            "emitted_tokens_per_round": emitted,
            "projected_tps": emitted * 1_000.0 / round_ms,
            "meets_target": emitted * 1_000.0 / round_ms >= target_tps,
            "round_budget_ms_at_target": emitted * 1_000.0 / target_tps,
        }
        for emitted in acceptance_values
    ]
    policy = CHANGE_POLICIES[change_kind]
    exact_layout_coverage = _exact_layout_coverage(scenario, stages)
    change_ready, change_blockers = _change_readiness(
        change_kind, changed_stages, stages, exact_layout_coverage
    )
    qualification = _validate_qualification(bundle)
    occupied_complete = (
        qualification["occupied_prefix_249957"] == "pass"
        and qualification["long_context_quality"] == "pass"
    )
    production_claim_ready = occupied_complete and change_ready

    return {
        "schema_version": SCHEMA_VERSION,
        "projection_kind": "component_sum_not_live_occupied_measurement",
        "scenario": scenario,
        "change": {
            "kind": change_kind,
            "changed_stages": list(changed_stages),
            "occupied_prefix_required": policy.occupied_prefix_required,
            "required_gates": list(policy.required_gates),
            "rationale": policy.rationale,
            "candidate_component_ready": change_ready,
            "candidate_component_blockers": change_blockers,
        },
        "stages": stages,
        "round": {
            "projected_ms": round_ms,
            "target_tps": target_tps,
            "required_emitted_tokens_per_round": target_tps * round_ms / 1_000.0,
            "scenarios": scenarios,
        },
        "evidence": {
            "exact_layout_standalone_coverage": exact_layout_coverage,
            "all_core_stages_exact_layout": all(exact_layout_coverage.values()),
            "occupied_prefix_249957": qualification["occupied_prefix_249957"],
            "long_context_quality": qualification["long_context_quality"],
        },
        "decision": {
            "cold_fill_needed_for_this_iteration": policy.occupied_prefix_required,
            "component_projection_ready": change_ready,
            "production_250k_tps_claim_ready": production_claim_ready,
            "may_publish_as": (
                "occupied-250k measured throughput"
                if production_claim_ready
                else "exact-layout component projection only"
                if change_ready
                else "unqualified component projection only"
            ),
        },
    }


def _parse_stage_override(value: str) -> tuple[str, float]:
    try:
        name, raw_ms = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("stage override must have STAGE=MILLISECONDS") from error
    if name not in STAGE_NAMES:
        raise argparse.ArgumentTypeError(f"unknown stage {name!r}")
    try:
        milliseconds = float(raw_ms)
    except ValueError as error:
        raise argparse.ArgumentTypeError("stage milliseconds must be numeric") from error
    if not math.isfinite(milliseconds) or milliseconds < 0:
        raise argparse.ArgumentTypeError("stage milliseconds must be finite and non-negative")
    return name, milliseconds


def _parse_component_result(value: str) -> tuple[str, Path]:
    try:
        name, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("component result must have STAGE=PATH") from error
    if name not in {"native_b", "quest", "gdn"}:
        raise argparse.ArgumentTypeError("component result stage must be native_b, quest, or gdn")
    if not raw_path:
        raise argparse.ArgumentTypeError("component result path cannot be empty")
    return name, Path(raw_path)


def _parse_emitted(value: str) -> list[float]:
    try:
        values = [float(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("--emitted must be comma-separated numbers") from error
    if not values or any(not math.isfinite(item) or item <= 0 for item in values):
        raise argparse.ArgumentTypeError("--emitted values must be finite and positive")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Project DFlash8 throughput from exact-layout 249,957-token component timings and "
            "decide whether the current change requires an occupied-prefix qualification."
        )
    )
    parser.add_argument("--measurements", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--change-kind", choices=sorted(CHANGE_POLICIES), default="runtime_wiring")
    parser.add_argument(
        "--changed-stage",
        action="append",
        choices=STAGE_NAMES,
        default=[],
        help="stage changed in this iteration; repeat for multiple stages",
    )
    parser.add_argument(
        "--stage-ms",
        action="append",
        type=_parse_stage_override,
        default=[],
        metavar="STAGE=MILLISECONDS",
        help="replace a timing; the candidate becomes unqualified until --qualified-stage is set",
    )
    parser.add_argument(
        "--component-result",
        action="append",
        type=_parse_component_result,
        default=[],
        metavar="STAGE=PATH",
        help="ingest a Native-B, Quest, or GDN exact-layout benchmark JSON",
    )
    parser.add_argument(
        "--qualified-stage",
        action="append",
        choices=STAGE_NAMES,
        default=[],
        help=(
            "attest that this candidate stage passed its policy gates and timing used the exact "
            "layout; repeat for multiple stages"
        ),
    )
    parser.add_argument(
        "--emitted",
        type=_parse_emitted,
        help="comma-separated emitted-token/round scenarios; defaults to the bundle",
    )
    parser.add_argument("--target-tps", type=float, default=100.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-component-ready",
        action="store_true",
        help="exit 2 unless changed stages have passing exact-layout/live GPU evidence",
    )
    parser.add_argument(
        "--require-100-tps",
        action="store_true",
        help="exit 3 unless at least one supplied emitted-token scenario reaches the target",
    )
    parser.add_argument(
        "--require-production-250k",
        action="store_true",
        help="exit 4 unless occupied-prefix speed and long-context quality are recorded as passed",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        bundle = copy.deepcopy(load_bundle(args.measurements.expanduser().resolve()))
        stages = _object(bundle.get("stages"), "stages")
        component_stage_names = {name for name, _ in args.component_result}
        override_stage_names = {name for name, _ in args.stage_ms}
        duplicates = sorted(component_stage_names.intersection(override_stage_names))
        if duplicates:
            raise ProjectionError(
                "cannot use both --component-result and --stage-ms for: " + ", ".join(duplicates)
            )
        if len(component_stage_names) != len(args.component_result):
            raise ProjectionError("each --component-result stage may be specified only once")
        for name, path in args.component_result:
            apply_component_result(bundle, name, path)
        for name, milliseconds in args.stage_ms:
            stage = _object(stages.get(name), f"stages.{name}")
            stage["milliseconds"] = milliseconds
            evidence = _object(stage.get("evidence"), f"stages.{name}.evidence")
            evidence["kind"] = "manual_candidate_timing"
            evidence["source"] = "command-line --stage-ms (no bound result artifact)"
            evidence.pop("sha256", None)
            evidence["note"] = "command-line candidate timing; qualification not yet attested"
            correctness = _object(stage.get("correctness"), f"stages.{name}.correctness")
            correctness["status"] = "not_recorded"
        for name in args.qualified_stage:
            stage = _object(stages.get(name), f"stages.{name}")
            evidence = _object(stage.get("evidence"), f"stages.{name}.evidence")
            evidence["kind"] = (
                "attested_exact_layout"
                if evidence.get("kind") == "manual_candidate_timing"
                else "exact_layout_standalone"
            )
            evidence["note"] = "candidate gates explicitly attested with --qualified-stage"
            correctness = _object(stage.get("correctness"), f"stages.{name}.correctness")
            correctness["status"] = "pass"
        result = project_bundle(
            bundle,
            change_kind=args.change_kind,
            changed_stages=args.changed_stage,
            target_tps=args.target_tps,
            acceptance_values=args.emitted,
        )
    except ProjectionError as error:
        parser.exit(1, f"error: {error}\n")

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")

    if args.require_component_ready and not result["decision"]["component_projection_ready"]:
        return 2
    if args.require_100_tps and not any(
        scenario["meets_target"] for scenario in result["round"]["scenarios"]
    ):
        return 3
    if args.require_production_250k and not result["decision"]["production_250k_tps_claim_ready"]:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
