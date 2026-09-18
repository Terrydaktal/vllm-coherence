# QWEN_ASSURANCE_ONLY_BEGIN: full-assurance-instrumentation
"""Fail-closed coverage contract for the complete Quest96/W4A16 debug build.

This module does not claim that a finite trace proves universal equivalence.  It
defines the minimum evidence a *fully instrumented* serial/M8 qualification must
contain before any comparator or promotion controller may call that trace
complete.  Sparse diagnostic captures are useful for localization, but they do
not satisfy this contract.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_r9700_lab.assurance_fault_contract import (
    EXPECTED_PROJECTION_SITES,
    FAULT_SITES,
    GDN_LAYERS,
    LAYER_COUNT,
    QUEST_LAYERS,
)

TRACE_HEADER_SCHEMA = "urn:qwen-r9700:full-assurance-trace-header:v2"
TRACE_EVENT_SCHEMA = "urn:qwen-r9700:full-assurance-event:v2"
COVERAGE_SCHEMA = "urn:qwen-r9700:full-assurance-coverage:v2"
COMPARISON_SCHEMA = "urn:qwen-r9700:full-assurance-comparison:v2"
INVARIANT_REPORT_SCHEMA = "urn:qwen-r9700:full-assurance-invariants:v2"
FRAGMENT_HEADER_SCHEMA = "urn:qwen-r9700:full-assurance-fragment-header:v2"
FRAGMENT_COMPLETE_SCHEMA = "urn:qwen-r9700:full-assurance-fragment-complete:v2"
CAMPAIGN_ASSEMBLY_SCHEMA = "urn:qwen-r9700:full-assurance-campaign-assembly:v2"
COUNTEREXAMPLE_SCHEMA = "urn:qwen-r9700:full-assurance-counterexample:v2"
PAIR_QUALIFICATION_SCHEMA = "urn:qwen-r9700:full-assurance-pair-qualification:v2"
RUNTIME_SCOPES_SCHEMA = "urn:qwen-r9700:full-assurance-runtime-scopes:v2"
FRAGMENT_PLAN_SCHEMA = "urn:qwen-r9700:full-assurance-fragment-plan:v2"
ARMS = ("serial-m1", "speculative-m8")
COMMIT_COUNTS = tuple(range(9))

# Ordered by causal execution.  A comparator reports the first difference in
# this order, never a later aggregate hash that merely reflects it.
UNIT_ORDER = (
    "snapshot_restore",
    "m8_construction_positions_rope",
    "w4a16_projections",
    "full_attention_qk_norm_rope",
    "quest_scoring",
    "quest_top96_selection",
    "quest_union_visibility",
    "quest_historical_attention",
    "quest_causal_tail_attention",
    "quest_softmax_reduction",
    "gdn_convolution",
    "gdn_recurrence",
    "gdn_gated_rmsnorm",
    "residual_stream",
    "swiglu_ffn",
    "final_rmsnorm",
    "lm_head",
    "target_verification",
    "sampler_rng_decoding",
    "provisional_isolation",
    "atomic_commit",
    "fault_injection",
    "lifecycle",
    "observer_integrity",
    "assurance_release_equivalence",
    "structured_outcome",
)

# Every semantic unit has one canonical observation boundary.  Operations,
# injected fault sites, and lifecycle scenarios use their exact identity as the
# phase so coverage cannot be satisfied by an observation taken too early or
# too late.
UNIT_PHASES = {
    "snapshot_restore": "post_restore",
    "m8_construction_positions_rope": "post_construct",
    "full_attention_qk_norm_rope": "post_qk_norm_rope",
    "quest_scoring": "post_score",
    "quest_top96_selection": "post_select",
    "quest_union_visibility": "post_union_visibility",
    "quest_historical_attention": "post_historical_attention",
    "quest_causal_tail_attention": "post_causal_tail_attention",
    "quest_softmax_reduction": "post_softmax_reduction",
    "gdn_convolution": "post_convolution",
    "gdn_recurrence": "post_recurrence",
    "gdn_gated_rmsnorm": "post_gated_rmsnorm",
    "residual_stream": "post_residual",
    "swiglu_ffn": "post_ffn",
    "final_rmsnorm": "post_final_rmsnorm",
    "lm_head": "post_logits",
    "target_verification": "post_verify",
    "sampler_rng_decoding": "post_sample",
    "provisional_isolation": "post_provisional",
    "atomic_commit": "post_publish",
    "observer_integrity": "final",
    "assurance_release_equivalence": "complete",
    "structured_outcome": "provider_end",
}

LIFECYCLE_SCENARIOS = (
    "fresh_60298",
    "restored_60298",
    "fresh_249957",
    "restored_249957",
    "restart_60298",
    "restart_249957",
    "offload_reload_60298",
    "offload_reload_249957",
    "page_reuse",
    "missing_snapshot",
    "corrupt_snapshot",
    "two_sessions",
    "two_branches",
    "rapid_switch_60298",
    "rapid_switch_249957",
)

# The values are semantic witnesses, not logging suggestions.  Every listed
# field must be present.  Individual probes may add richer fields.
REQUIRED_EVIDENCE_FIELDS: dict[str, frozenset[str]] = {
    "snapshot_restore": frozenset(
        {
            "snapshot_manifest_sha256",
            "prompt_token_sha256",
            "logical_target_kv_sha256",
            "physical_target_kv_sha256",
            "logical_target_kv_sha256_by_layer",
            "physical_target_kv_sha256_by_layer",
            "logical_draft_kv_sha256",
            "physical_draft_kv_sha256",
            "logical_draft_kv_sha256_by_layer",
            "physical_draft_kv_sha256_by_layer",
            "gdn_state_sha256_by_layer",
            "convolution_state_sha256_by_layer",
            "positions_sha256",
            "rope_indices_sha256",
            "quant_scales_sha256",
            "cache_mapping_sha256",
            "allocator_generation",
            "allocation_topology_sha256",
            "ownership_sha256",
            "pins_sha256",
            "refcounts_sha256",
            "commit_epoch",
            "transaction_state_sha256",
            "device_error_word",
        }
    ),
    "m8_construction_positions_rope": frozenset(
        {
            "input_token_id",
            "embedding_sha256",
            "position",
            "rope_index_sha256",
            "rope_cos_sin_sha256",
            "row_mapping_sha256",
            "tensor_layout_sha256",
        }
    ),
    "w4a16_projections": frozenset(
        {
            "operation",
            "input_sha256",
            "quantized_weight_sha256",
            "scale_sha256",
            "mapping_sha256",
            "workspace_before_sha256",
            "workspace_after_sha256",
            "input_layout_sha256",
            "output_layout_sha256",
            "output_sha256",
            "nonfinite_count",
        }
    ),
    "full_attention_qk_norm_rope": frozenset(
        {
            "qkv_input_sha256",
            "qkv_fused_output_sha256",
            "q_raw_sha256",
            "k_raw_sha256",
            "v_raw_sha256",
            "q_norm_weight_sha256",
            "k_norm_weight_sha256",
            "rope_cos_sin_sha256",
            "q_output_sha256",
            "k_output_sha256",
            "v_output_sha256",
            "serial_q_output_sha256",
            "serial_k_output_sha256",
            "serial_v_output_sha256",
            "input_layout_sha256",
            "split_layout_sha256",
            "output_layout_sha256",
            "storage_relations_sha256",
            "cuda_stream",
            "input_unchanged",
            "nonfinite_count",
        }
    ),
    "quest_scoring": frozenset(
        {
            "query_sha256",
            "centroids_sha256",
            "ordered_scores_sha256",
            "score_count",
        }
    ),
    "quest_top96_selection": frozenset(
        {"ordered_scores_sha256", "ordered_pages_sha256", "selected_count"}
    ),
    "quest_union_visibility": frozenset(
        {
            "selected_pages_sha256",
            "union_pages_sha256",
            "row_visibility_sha256",
            "union_counts_sha256",
        }
    ),
    "quest_historical_attention": frozenset(
        {
            "query_sha256",
            "historical_key_sha256",
            "historical_value_sha256",
            "scores_sha256",
            "softmax_statistics_sha256",
            "output_sha256",
        }
    ),
    "quest_causal_tail_attention": frozenset(
        {
            "query_sha256",
            "tail_key_sha256",
            "tail_value_sha256",
            "causal_mask_sha256",
            "scores_sha256",
            "output_sha256",
        }
    ),
    "quest_softmax_reduction": frozenset(
        {
            "historical_statistics_sha256",
            "tail_statistics_sha256",
            "reduction_order_sha256",
            "output_sha256",
        }
    ),
    "gdn_convolution": frozenset(
        {
            "input_sha256",
            "state_index",
            "state_before_sha256",
            "output_sha256",
            "state_after_sha256",
            "nonfinite_count",
        }
    ),
    "gdn_recurrence": frozenset(
        {
            "q_sha256",
            "k_sha256",
            "v_sha256",
            "a_sha256",
            "b_sha256",
            "state_index",
            "state_before_sha256",
            "output_sha256",
            "state_after_sha256",
            "nonfinite_count",
        }
    ),
    "gdn_gated_rmsnorm": frozenset(
        {"input_sha256", "gate_sha256", "weight_sha256", "output_sha256"}
    ),
    "residual_stream": frozenset(
        {
            "input_hidden_sha256",
            "input_residual_sha256",
            "post_attention_sha256",
            "output_hidden_sha256",
            "output_residual_sha256",
        }
    ),
    "swiglu_ffn": frozenset(
        {
            "input_sha256",
            "gate_up_sha256",
            "activation_sha256",
            "down_projection_sha256",
            "output_sha256",
        }
    ),
    "final_rmsnorm": frozenset(
        {"input_sha256", "residual_sha256", "weight_sha256", "output_sha256"}
    ),
    "lm_head": frozenset(
        {
            "input_sha256",
            "weight_sha256",
            "full_logits_sha256",
            "full_logits_chunk_digests",
            "argmax_token_id",
            "nonfinite_count",
        }
    ),
    "target_verification": frozenset(
        {
            "proposal_token_ids_sha256",
            "target_logits_sha256",
            "target_logits_chunk_digests",
            "target_argmax_token_id",
            "accepted_draft_count",
            "externally_capped_commit_count",
        }
    ),
    "sampler_rng_decoding": frozenset(
        {
            "sampling_parameters_sha256",
            "rng_state_before_sha256",
            "rng_state_after_sha256",
            "stop_state_sha256",
            "grammar_state_sha256",
            "selected_token_id",
        }
    ),
    "provisional_isolation": frozenset(
        {
            "commit_count",
            "canonical_before_sha256",
            "canonical_after_provisional_sha256",
            "provisional_state_sha256",
            "allocation_topology_before_sha256",
            "allocation_topology_after_sha256",
            "ownership_before_sha256",
            "ownership_after_sha256",
            "pins_before_sha256",
            "pins_after_sha256",
            "refcounts_before_sha256",
            "refcounts_after_sha256",
            "target_kv_before_sha256_by_layer",
            "target_kv_after_provisional_sha256_by_layer",
            "draft_kv_before_sha256_by_layer",
            "draft_kv_after_provisional_sha256_by_layer",
            "gdn_state_before_sha256_by_layer",
            "gdn_state_after_provisional_sha256_by_layer",
            "convolution_state_before_sha256_by_layer",
            "convolution_state_after_provisional_sha256_by_layer",
            "device_error_word",
        }
    ),
    "atomic_commit": frozenset(
        {
            "commit_count",
            "canonical_before_sha256",
            "provisional_sha256",
            "serial_reference_sha256",
            "published_sha256",
            "root_pointer_before_sha256",
            "root_pointer_after_sha256",
            "commit_epoch_before",
            "commit_epoch_after",
            "journal_sha256",
            "allocator_generation",
            "serial_allocation_topology_sha256",
            "published_allocation_topology_sha256",
            "serial_ownership_sha256",
            "published_ownership_sha256",
            "serial_pins_sha256",
            "published_pins_sha256",
            "serial_refcounts_sha256",
            "published_refcounts_sha256",
            "serial_target_kv_sha256_by_layer",
            "published_target_kv_sha256_by_layer",
            "serial_draft_kv_sha256_by_layer",
            "published_draft_kv_sha256_by_layer",
            "serial_gdn_state_sha256_by_layer",
            "published_gdn_state_sha256_by_layer",
            "serial_convolution_state_sha256_by_layer",
            "published_convolution_state_sha256_by_layer",
            "device_error_word",
        }
    ),
    "fault_injection": frozenset(
        {
            "site",
            "commit_count",
            "injected",
            "failure_observed",
            "canonical_before_sha256",
            "canonical_after_sha256",
            "allocation_topology_before_sha256",
            "allocation_topology_after_sha256",
            "ownership_before_sha256",
            "ownership_after_sha256",
            "pins_before_sha256",
            "pins_after_sha256",
            "refcounts_before_sha256",
            "refcounts_after_sha256",
            "target_kv_before_sha256_by_layer",
            "target_kv_after_sha256_by_layer",
            "draft_kv_before_sha256_by_layer",
            "draft_kv_after_sha256_by_layer",
            "gdn_state_before_sha256_by_layer",
            "gdn_state_after_sha256_by_layer",
            "convolution_state_before_sha256_by_layer",
            "convolution_state_after_sha256_by_layer",
            "exception_sha256",
            "device_error_word",
        }
    ),
    "lifecycle": frozenset(
        {
            "scenario",
            "expected_outcome",
            "observed_outcome",
            "state_before_sha256",
            "state_after_sha256",
            "reference_state_sha256",
            "prompt_tokens",
            "cold_fill_occurred",
            "cross_session_contamination",
            "device_error_word",
        }
    ),
    "observer_integrity": frozenset(
        {
            "installed_sites_sha256",
            "required_sites_sha256",
            "events_attempted",
            "events_written",
            "events_dropped",
            "stream_fsync_sha256",
            "device_synchronize_count",
            "capture_failures",
            "device_error_word",
        }
    ),
    "assurance_release_equivalence": frozenset(
        {
            "semantic_source_sha256",
            "assurance_binary_sha256",
            "release_binary_sha256",
            "assurance_trace_sha256",
            "release_output_sha256",
            "serial_reference_sha256",
            "instrumentation_removed_sha256",
            "passed",
        }
    ),
    "structured_outcome": frozenset(
        {
            "raw_token_ids_sha256",
            "parser_input_sha256",
            "parser_output_sha256",
            "finish_reason",
            "tool_ledger_sha256",
            "outcome",
        }
    ),
}

GLOBAL_UNITS = frozenset(
    {
        "snapshot_restore",
        "observer_integrity",
        "assurance_release_equivalence",
        "structured_outcome",
    }
)
GLOBAL_PHASES = {
    "snapshot_restore": UNIT_PHASES["snapshot_restore"],
    "observer_integrity": UNIT_PHASES["observer_integrity"],
    "assurance_release_equivalence": UNIT_PHASES["assurance_release_equivalence"],
    "structured_outcome": UNIT_PHASES["structured_outcome"],
}
POSITION_UNITS = frozenset(
    {
        "m8_construction_positions_rope",
        "final_rmsnorm",
        "lm_head",
        "target_verification",
        "sampler_rng_decoding",
    }
)
ALL_LAYER_UNITS = frozenset({"residual_stream", "swiglu_ffn"})
QUEST_UNITS = frozenset(
    {
        "quest_scoring",
        "full_attention_qk_norm_rope",
        "quest_top96_selection",
        "quest_union_visibility",
        "quest_historical_attention",
        "quest_causal_tail_attention",
        "quest_softmax_reduction",
    }
)
GDN_UNITS = frozenset({"gdn_convolution", "gdn_recurrence", "gdn_gated_rmsnorm"})
COMMIT_UNITS = frozenset({"provisional_isolation", "atomic_commit"})
POSITION_SCOPED_UNITS = (
    POSITION_UNITS | ALL_LAYER_UNITS | QUEST_UNITS | GDN_UNITS | {"w4a16_projections"}
)

# Every non-synthetic probe has exactly one producer class.  A campaign may be
# split across processes, prompts, fault cases, or lifecycle cases, but it may
# not silently substitute one producer for another.  In particular, model
# hooks cannot claim snapshot, transaction, provider, or release evidence.
PRODUCER_UNITS: dict[str, frozenset[str]] = {
    "engine_model_capture": frozenset(
        POSITION_SCOPED_UNITS | {"target_verification", "sampler_rng_decoding"}
    ),
    "snapshot_controller": frozenset({"snapshot_restore"}),
    "transaction_controller": frozenset(
        {"provisional_isolation", "atomic_commit", "fault_injection"}
    ),
    "lifecycle_controller": frozenset({"lifecycle"}),
    "release_controller": frozenset({"assurance_release_equivalence"}),
    "provider_controller": frozenset({"structured_outcome"}),
}
PRODUCER_KINDS = tuple(PRODUCER_UNITS)
UNIT_INSTRUMENTATION_SITES: dict[str, tuple[str, ...]] = {
    "snapshot_restore": ("snapshot.restore.boundary",),
    "m8_construction_positions_rope": ("model.input.positions-rope",),
    "w4a16_projections": ("model.decoder.w4a16-projections",),
    "full_attention_qk_norm_rope": ("model.full-attention.qk-norm-rope",),
    "quest_scoring": ("model.quest96.scoring",),
    "quest_top96_selection": ("model.quest96.top96-selection",),
    "quest_union_visibility": ("model.quest96.union-visibility",),
    "quest_historical_attention": ("model.quest96.historical-attention",),
    "quest_causal_tail_attention": ("model.quest96.causal-tail-attention",),
    "quest_softmax_reduction": ("model.quest96.softmax-reduction",),
    "gdn_convolution": ("model.gdn.convolution",),
    "gdn_recurrence": ("model.gdn.recurrence",),
    "gdn_gated_rmsnorm": ("model.gdn.gated-rmsnorm",),
    "residual_stream": ("model.decoder.residual-stream",),
    "swiglu_ffn": ("model.decoder.swiglu-ffn",),
    "final_rmsnorm": ("model.final-rmsnorm",),
    "lm_head": ("model.lm-head",),
    "target_verification": ("sampler.target-verification",),
    "sampler_rng_decoding": ("sampler.rng-decoding",),
    "provisional_isolation": ("transaction.private.provisional",),
    "atomic_commit": ("transaction.atomic.publish",),
    "fault_injection": ("transaction.fault.injector",),
    "lifecycle": ("lifecycle.scenario.controller",),
    "observer_integrity": (),
    "assurance_release_equivalence": ("assurance.release.differential",),
    "structured_outcome": ("provider.raw-token-outcome",),
}
if set(UNIT_INSTRUMENTATION_SITES) != set(UNIT_ORDER):
    raise RuntimeError("full-assurance unit instrumentation inventory is incomplete")
if any(
    not sites
    for unit, sites in UNIT_INSTRUMENTATION_SITES.items()
    if unit != "observer_integrity"
):
    raise RuntimeError("full-assurance semantic unit has no executable instrumentation boundary")

ENGINE_ALWAYS_INSTRUMENTATION_SITES = ("engine.request.gate",)
ENGINE_SPECULATIVE_INSTRUMENTATION_SITES = ("sampler.rejection-parser",)
ENGINE_BASE_INSTRUMENTATION_SITES = tuple(
    sorted(
        {
            *ENGINE_ALWAYS_INSTRUMENTATION_SITES,
            *(
                site
                for unit in PRODUCER_UNITS["engine_model_capture"]
                for site in UNIT_INSTRUMENTATION_SITES[unit]
            ),
        }
    )
)

_producer_unit_union = frozenset().union(*PRODUCER_UNITS.values())
if _producer_unit_union != frozenset(UNIT_ORDER) - {"observer_integrity"}:
    raise RuntimeError("full-assurance producer routing omits or invents a semantic unit")
if sum(len(units) for units in PRODUCER_UNITS.values()) != len(_producer_unit_union):
    raise RuntimeError("full-assurance semantic unit has more than one producer")
LAYER_DIGEST_COUNTS = {
    "gdn_state_sha256_by_layer": len(GDN_LAYERS),
    "convolution_state_sha256_by_layer": len(GDN_LAYERS),
    "logical_target_kv_sha256_by_layer": len(QUEST_LAYERS),
    "physical_target_kv_sha256_by_layer": len(QUEST_LAYERS),
    "logical_draft_kv_sha256_by_layer": 5,
    "physical_draft_kv_sha256_by_layer": 5,
    "target_kv_before_sha256_by_layer": len(QUEST_LAYERS),
    "target_kv_after_provisional_sha256_by_layer": len(QUEST_LAYERS),
    "target_kv_after_sha256_by_layer": len(QUEST_LAYERS),
    "serial_target_kv_sha256_by_layer": len(QUEST_LAYERS),
    "published_target_kv_sha256_by_layer": len(QUEST_LAYERS),
    "draft_kv_before_sha256_by_layer": 5,
    "draft_kv_after_provisional_sha256_by_layer": 5,
    "draft_kv_after_sha256_by_layer": 5,
    "serial_draft_kv_sha256_by_layer": 5,
    "published_draft_kv_sha256_by_layer": 5,
    "gdn_state_before_sha256_by_layer": len(GDN_LAYERS),
    "gdn_state_after_provisional_sha256_by_layer": len(GDN_LAYERS),
    "gdn_state_after_sha256_by_layer": len(GDN_LAYERS),
    "serial_gdn_state_sha256_by_layer": len(GDN_LAYERS),
    "published_gdn_state_sha256_by_layer": len(GDN_LAYERS),
    "convolution_state_before_sha256_by_layer": len(GDN_LAYERS),
    "convolution_state_after_provisional_sha256_by_layer": len(GDN_LAYERS),
    "convolution_state_after_sha256_by_layer": len(GDN_LAYERS),
    "serial_convolution_state_sha256_by_layer": len(GDN_LAYERS),
    "published_convolution_state_sha256_by_layer": len(GDN_LAYERS),
}
# These witnesses are mandatory because they explain aliasing, workspace, and
# launch-shape failures, but a serial M1 tensor is not required to have the same
# physical stride, stream, or scratch allocation as its logical M8 row.  They
# are validated within each arm and retained in counterexamples rather than
# incorrectly treated as cross-arm semantic equality.
ARM_LOCAL_EVIDENCE_FIELDS: dict[str, frozenset[str]] = {
    "observer_integrity": REQUIRED_EVIDENCE_FIELDS["observer_integrity"],
    "m8_construction_positions_rope": frozenset(
        {"row_mapping_sha256", "tensor_layout_sha256"}
    ),
    "w4a16_projections": frozenset(
        {
            "workspace_before_sha256",
            "workspace_after_sha256",
            "input_layout_sha256",
            "output_layout_sha256",
        }
    ),
    "full_attention_qk_norm_rope": frozenset(
        {
            "input_layout_sha256",
            "split_layout_sha256",
            "output_layout_sha256",
            "storage_relations_sha256",
            "cuda_stream",
        }
    ),
    "target_verification": frozenset(
        {
            "proposal_token_ids_sha256",
            "accepted_draft_count",
            "externally_capped_commit_count",
        }
    ),
    "sampler_rng_decoding": frozenset(
        {
            "rng_state_before_sha256",
            "rng_state_after_sha256",
            "selected_token_id",
        }
    ),
    "gdn_convolution": frozenset({"state_index"}),
    "gdn_recurrence": frozenset({"state_index"}),
}

HEADER_KEYS = {
    "schema",
    "run_id",
    "arm",
    "semantic_source_sha256",
    "assurance_manifest_sha256",
    "release_manifest_sha256",
    "binary_sha256",
    "dependencies_sha256",
    "compiler_sha256",
    "hardware_runtime_sha256",
    "model_sha256",
    "prompt_sha256",
    "prompt_tokens",
    "snapshot_sha256",
    "positions",
    "position_rows",
    "projection_sites",
    "fault_sites",
    "lifecycle_scenarios",
    "required_units_sha256",
}
EVENT_KEYS = {
    "schema",
    "sequence",
    "arm",
    "unit",
    "phase",
    "position",
    "layer_index",
    "row",
    "evidence",
    "previous_event_sha256",
    "event_sha256",
}


class InstrumentationError(RuntimeError):
    """A debug trace is incomplete, unauthenticated, or internally inconsistent."""


_SCOPE_LENGTH = 6


def _scope(event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        event["unit"],
        event["position"],
        event["layer_index"],
        event["row"],
        event["phase"],
        event["evidence"].get("commit_count"),
    )


def _normalize_scope(value: object) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != _SCOPE_LENGTH:
        raise InstrumentationError("full-assurance fragment scope is malformed")
    scope = tuple(value)
    unit, position, layer, row, phase, commit_count = scope
    if not isinstance(unit, str) or unit not in REQUIRED_EVIDENCE_FIELDS:
        raise InstrumentationError("full-assurance fragment scope unit is invalid")
    if not isinstance(phase, str) or not phase:
        raise InstrumentationError("full-assurance fragment scope phase is invalid")
    for field, maximum, label in (
        (position, 1_000_000, "position"),
        (layer, 63, "layer"),
        (row, 7, "row"),
        (commit_count, 8, "commit count"),
    ):
        if field is not None and (
            isinstance(field, bool) or not isinstance(field, int) or not 0 <= field <= maximum
        ):
            raise InstrumentationError(f"full-assurance fragment scope {label} is invalid")
    return scope


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    except (TypeError, ValueError) as error:
        raise InstrumentationError(
            f"full-assurance value is not canonical JSON: {error}"
        ) from error


def _json_clone(value: object) -> Any:
    return json.loads(_canonical(value))


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise InstrumentationError(f"{label} must be a lowercase SHA-256")
    return value


def required_units_sha256() -> str:
    return _sha(
        {
            "expected_projection_sites": EXPECTED_PROJECTION_SITES,
            "fault_sites": FAULT_SITES,
            "lifecycle_scenarios": LIFECYCLE_SCENARIOS,
            "order": UNIT_ORDER,
            "phases": UNIT_PHASES,
            "arm_local_fields": {
                unit: sorted(fields) for unit, fields in ARM_LOCAL_EVIDENCE_FIELDS.items()
            },
            "required_fields": {
                unit: sorted(fields) for unit, fields in REQUIRED_EVIDENCE_FIELDS.items()
            },
            "producer_units": {
                producer: sorted(units) for producer, units in PRODUCER_UNITS.items()
            },
            "unit_instrumentation_sites": UNIT_INSTRUMENTATION_SITES,
            "engine_always_instrumentation_sites": ENGINE_ALWAYS_INSTRUMENTATION_SITES,
            "engine_speculative_instrumentation_sites": (
                ENGINE_SPECULATIVE_INSTRUMENTATION_SITES
            ),
        }
    )


def producer_kind_for_scope(scope_value: object) -> str:
    """Return the sole authenticated producer class for one non-observer scope."""

    scope = _normalize_scope(scope_value)
    unit = scope[0]
    if unit == "observer_integrity":
        raise InstrumentationError("observer integrity is derived only by campaign assembly")
    matches = [producer for producer, units in PRODUCER_UNITS.items() if unit in units]
    if len(matches) != 1:  # guarded at import too; retain a fail-closed runtime check
        raise InstrumentationError(f"semantic unit has no unique producer: {unit}")
    return matches[0]


def expected_instrumentation_sites(
    producer_kind: str, scopes_value: object, *, arm: str
) -> tuple[str, ...]:
    """Return the exact code-boundary inventory one fragment must prove installed."""

    if producer_kind not in PRODUCER_UNITS or arm not in ARMS:
        raise InstrumentationError("fragment producer kind or arm is invalid")
    if not isinstance(scopes_value, (list, tuple, set, frozenset)) or not scopes_value:
        raise InstrumentationError("fragment instrumentation inventory has no scopes")
    scopes = tuple(_normalize_scope(scope) for scope in scopes_value)
    if any(producer_kind_for_scope(scope) != producer_kind for scope in scopes):
        raise InstrumentationError("fragment mixes semantic scopes from different producers")
    sites = {
        site
        for scope in scopes
        for site in UNIT_INSTRUMENTATION_SITES[scope[0]]
    }
    if producer_kind == "transaction_controller":
        sites.update(
            f"transaction.fault.injector:{scope[4]}"
            for scope in scopes
            if scope[0] == "fault_injection"
        )
    if producer_kind == "lifecycle_controller":
        sites.update(
            f"lifecycle.scenario.controller:{scope[4]}"
            for scope in scopes
            if scope[0] == "lifecycle"
        )
    if producer_kind != "engine_model_capture":
        return tuple(sorted(sites))
    sites.update(ENGINE_ALWAYS_INSTRUMENTATION_SITES)
    if arm == "speculative-m8":
        sites.update(ENGINE_SPECULATIVE_INSTRUMENTATION_SITES)
    return tuple(sorted(sites))


def required_probe_sites_sha256(header_value: object) -> str:
    """Bind observer integrity to the campaign-specific, non-synthetic probe sites."""

    header = normalize_header(header_value)
    observer_scope = (
        "observer_integrity",
        None,
        None,
        None,
        UNIT_PHASES["observer_integrity"],
        None,
    )
    return _sha(sorted(expected_probe_scopes(header) - {observer_scope}, key=repr))


def partition_probe_scopes_by_producer(
    header_value: object,
) -> dict[str, tuple[tuple[Any, ...], ...]]:
    """Partition the entire campaign into disjoint, exhaustively routed producers."""

    header = normalize_header(header_value)
    partition: dict[str, list[tuple[Any, ...]]] = {producer: [] for producer in PRODUCER_KINDS}
    for scope in expected_probe_scopes(header):
        if scope[0] == "observer_integrity":
            continue
        partition[producer_kind_for_scope(scope)].append(scope)
    result = {
        producer: tuple(sorted(scopes, key=repr))
        for producer, scopes in partition.items()
        if scopes
    }
    routed = [scope for scopes in result.values() for scope in scopes]
    if len(routed) != len(set(routed)):
        raise InstrumentationError("full-assurance producer partition overlaps")
    if set(routed) != (
        expected_probe_scopes(header)
        - {
            (
                "observer_integrity",
                None,
                None,
                None,
                UNIT_PHASES["observer_integrity"],
                None,
            )
        }
    ):
        raise InstrumentationError("full-assurance producer partition is incomplete")
    return result


def instrumentation_coverage_matrix(header_value: object) -> list[dict[str, Any]]:
    """Expose the complete unit→producer→boundary routing for audit and build gates."""

    header = normalize_header(header_value)
    partition = partition_probe_scopes_by_producer(header)
    rows: list[dict[str, Any]] = []
    for unit in UNIT_ORDER:
        if unit == "observer_integrity":
            rows.append(
                {
                    "unit": unit,
                    "producer_kind": "campaign_assembler",
                    "scope_count": 1,
                    "instrumentation_sites": ["campaign.observer.derived"],
                    "derived": True,
                }
            )
            continue
        producer_kind = next(
            producer for producer, units in PRODUCER_UNITS.items() if unit in units
        )
        scopes = tuple(scope for scope in partition[producer_kind] if scope[0] == unit)
        sites = expected_instrumentation_sites(producer_kind, scopes, arm=header["arm"])
        unit_sites = set(UNIT_INSTRUMENTATION_SITES[unit])
        if unit == "fault_injection":
            unit_sites.update(f"transaction.fault.injector:{scope[4]}" for scope in scopes)
        elif unit == "lifecycle":
            unit_sites.update(f"lifecycle.scenario.controller:{scope[4]}" for scope in scopes)
        if not unit_sites.issubset(sites):
            raise InstrumentationError(f"semantic unit lacks its runtime boundary: {unit}")
        rows.append(
            {
                "unit": unit,
                "producer_kind": producer_kind,
                "scope_count": len(scopes),
                "instrumentation_sites": sorted(unit_sites),
                "derived": False,
            }
        )
    if [row["unit"] for row in rows] != list(UNIT_ORDER):
        raise InstrumentationError("instrumentation coverage matrix order differs")
    return rows


def runtime_scopes_document(header_value: object, scopes_value: object) -> dict[str, Any]:
    """Seal the exact scope document consumed by one authenticated producer."""

    header = normalize_header(header_value)
    if not isinstance(scopes_value, (list, tuple, set, frozenset)) or not scopes_value:
        raise InstrumentationError("runtime scope document has no scopes")
    scopes = tuple(sorted((_normalize_scope(scope) for scope in scopes_value), key=repr))
    if len({producer_kind_for_scope(scope) for scope in scopes}) != 1:
        raise InstrumentationError("runtime scope document mixes producer classes")
    return {
        "schema": RUNTIME_SCOPES_SCHEMA,
        "campaign_header_sha256": _sha(header),
        "scopes": [list(scope) for scope in scopes],
        "scopes_sha256": _sha(scopes),
    }


def build_fragment_plan(header_value: object) -> dict[str, Any]:
    """Build an exhaustive runnable plan, splitting EngineCore probes into <=8 positions."""

    header = normalize_header(header_value)
    partition = partition_probe_scopes_by_producer(header)
    fragments: list[dict[str, Any]] = []
    for producer_kind, producer_scopes in partition.items():
        groups: list[tuple[tuple[Any, ...], ...]]
        if producer_kind == "engine_model_capture":
            positions = sorted({int(scope[1]) for scope in producer_scopes if scope[1] is not None})
            groups = []
            for start in range(0, len(positions), 8):
                selected = frozenset(positions[start : start + 8])
                groups.append(tuple(scope for scope in producer_scopes if scope[1] in selected))
        else:
            groups = [producer_scopes]
        for index, scopes in enumerate(groups):
            fragment_id = f"{producer_kind}-{index:03d}"
            scope_document = runtime_scopes_document(header, scopes)
            sites = expected_instrumentation_sites(producer_kind, scopes, arm=header["arm"])
            fragments.append(
                {
                    "fragment_id": fragment_id,
                    "producer_kind": producer_kind,
                    "scopes": scope_document["scopes"],
                    "scopes_sha256": scope_document["scopes_sha256"],
                    "installed_sites": list(sites),
                    "installed_sites_sha256": _sha(sites),
                }
            )
    routed = [tuple(scope) for fragment in fragments for scope in fragment["scopes"]]
    observer = (
        "observer_integrity",
        None,
        None,
        None,
        UNIT_PHASES["observer_integrity"],
        None,
    )
    if len(routed) != len(set(routed)) or set(routed) != expected_probe_scopes(header) - {observer}:
        raise InstrumentationError("constructed full-assurance fragment plan is not exhaustive")
    plan = {
        "schema": FRAGMENT_PLAN_SCHEMA,
        "campaign_header_sha256": _sha(header),
        "required_probe_sites_sha256": required_probe_sites_sha256(header),
        "fragments": fragments,
    }
    plan["plan_sha256"] = _sha(plan)
    return plan


def normalize_header(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != HEADER_KEYS:
        raise InstrumentationError("full-assurance trace header keys are incomplete")
    header = _json_clone(value)
    if header["schema"] != TRACE_HEADER_SCHEMA:
        raise InstrumentationError("full-assurance trace header schema differs")
    if header["arm"] not in ARMS:
        raise InstrumentationError("full-assurance trace arm is invalid")
    if not isinstance(header["run_id"], str) or not header["run_id"]:
        raise InstrumentationError("full-assurance run_id is invalid")
    structured_keys = {
        "positions",
        "position_rows",
        "projection_sites",
        "fault_sites",
        "lifecycle_scenarios",
    }
    for key in HEADER_KEYS - {"schema", "run_id", "arm"} - structured_keys:
        if key != "prompt_tokens":
            _require_digest(header[key], f"trace header.{key}")
    if header["required_units_sha256"] != required_units_sha256():
        raise InstrumentationError("debug build uses an incomplete instrumentation specification")
    positions = header["positions"]
    prompt_tokens = header["prompt_tokens"]
    if (
        isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
        or not 1 <= prompt_tokens <= 253_792
    ):
        raise InstrumentationError("trace header prompt_tokens is invalid")
    if (
        not isinstance(positions, list)
        or not positions
        or len(positions) > 64
        or positions != sorted(set(positions))
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in positions
        )
    ):
        raise InstrumentationError("trace header positions are invalid")
    if any(position < prompt_tokens for position in positions):
        raise InstrumentationError("trace header position precedes the declared prompt")
    position_rows = header["position_rows"]
    if (
        not isinstance(position_rows, list)
        or len(position_rows) != len(positions)
        or any(
            isinstance(row, bool) or not isinstance(row, int) or not 0 <= row <= 7
            for row in position_rows
        )
    ):
        raise InstrumentationError("trace header position_rows are invalid")
    if header["arm"] == "serial-m1" and any(row != 0 for row in position_rows):
        raise InstrumentationError("serial-m1 trace rows must all be zero")
    projection_sites = header["projection_sites"]
    expected_projection_sites = [
        {"layer_index": layer, "operation": operation}
        for layer, operation in EXPECTED_PROJECTION_SITES
    ]
    if projection_sites != expected_projection_sites:
        raise InstrumentationError("trace header does not enumerate every W4A16 projection site")
    if header["fault_sites"] != list(FAULT_SITES):
        raise InstrumentationError("trace header does not enumerate every transactional fault site")
    if header["lifecycle_scenarios"] != list(LIFECYCLE_SCENARIOS):
        raise InstrumentationError("trace header does not enumerate every lifecycle scenario")
    return header


def seal_event(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != EVENT_KEYS - {"event_sha256"}:
        raise InstrumentationError("full-assurance event keys are incomplete")
    event = _json_clone(value)
    event["event_sha256"] = _sha(event)
    return normalize_event(event)


def normalize_event(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != EVENT_KEYS:
        raise InstrumentationError("full-assurance event keys are incomplete")
    event = _json_clone(value)
    if event["schema"] != TRACE_EVENT_SCHEMA or event["arm"] not in ARMS:
        raise InstrumentationError("full-assurance event identity is invalid")
    if (
        isinstance(event["sequence"], bool)
        or not isinstance(event["sequence"], int)
        or event["sequence"] < 0
    ):
        raise InstrumentationError("full-assurance event sequence is invalid")
    previous = event["previous_event_sha256"]
    if previous is not None:
        _require_digest(previous, "full-assurance previous-event hash")
    unit = event["unit"]
    if unit not in REQUIRED_EVIDENCE_FIELDS:
        raise InstrumentationError(f"unknown full-assurance unit: {unit!r}")
    if not isinstance(event["phase"], str) or not event["phase"]:
        raise InstrumentationError("full-assurance event phase is invalid")
    for key, maximum in (("position", 1_000_000), ("layer_index", 63), ("row", 7)):
        field = event[key]
        if field is not None and (
            isinstance(field, bool) or not isinstance(field, int) or not 0 <= field <= maximum
        ):
            raise InstrumentationError(f"full-assurance event {key} is invalid")
    evidence = event["evidence"]
    if not isinstance(evidence, dict):
        raise InstrumentationError("full-assurance event evidence is not an object")
    missing = REQUIRED_EVIDENCE_FIELDS[unit] - set(evidence)
    if missing:
        raise InstrumentationError(
            f"{unit} instrumentation is missing required evidence: {sorted(missing)}"
        )
    for field in REQUIRED_EVIDENCE_FIELDS[unit]:
        field_value = evidence[field]
        if field.endswith("_sha256_by_layer"):
            expected_count = LAYER_DIGEST_COUNTS.get(field)
            if expected_count is None:
                raise InstrumentationError(f"{unit}.{field} has no canonical layer inventory")
            if not isinstance(field_value, list) or len(field_value) != expected_count:
                raise InstrumentationError(
                    f"{unit}.{field} must contain {expected_count} layer digests"
                )
            for index, digest in enumerate(field_value):
                _require_digest(digest, f"{unit}.{field}[{index}]")
        elif field.endswith("_sha256_by_row"):
            if not isinstance(field_value, list) or len(field_value) != 8:
                raise InstrumentationError(f"{unit}.{field} must contain 8 row digests")
            for index, digest in enumerate(field_value):
                _require_digest(digest, f"{unit}.{field}[{index}]")
        elif field.endswith("_chunk_digests"):
            if (
                not isinstance(field_value, list)
                or not field_value
                or len(field_value) > 4096
            ):
                raise InstrumentationError(
                    f"{unit}.{field} must contain 1..4096 ordered chunk digests"
                )
            for index, digest in enumerate(field_value):
                _require_digest(digest, f"{unit}.{field}[{index}]")
        elif "sha256" in field:
            _require_digest(field_value, f"{unit}.{field}")
        elif field in {"input_token_id", "argmax_token_id", "target_argmax_token_id"}:
            if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value < 0:
                raise InstrumentationError(f"{unit}.{field} must be a token ID")
        elif field == "selected_token_id" and field_value is not None:
            if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value < 0:
                raise InstrumentationError(f"{unit}.{field} must be null or a token ID")
        elif field == "accepted_draft_count":
            if (
                isinstance(field_value, bool)
                or not isinstance(field_value, int)
                or not 0 <= field_value <= 7
            ):
                raise InstrumentationError(f"{unit}.{field} must be in 0..7")
        elif field in {"externally_capped_commit_count", "commit_count"}:
            if (
                isinstance(field_value, bool)
                or not isinstance(field_value, int)
                or not 0 <= field_value <= 8
            ):
                raise InstrumentationError(f"{unit}.{field} must be in 0..8")
        elif field == "selected_count":
            if field_value != 96:
                raise InstrumentationError(f"{unit}.selected_count must be exactly 96")
        elif field in {
            "score_count",
            "state_index",
            "nonfinite_count",
            "allocator_generation",
            "commit_epoch",
            "commit_epoch_before",
            "commit_epoch_after",
            "device_error_word",
            "position",
        }:
            if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value < 0:
                raise InstrumentationError(f"{unit}.{field} must be a nonnegative integer")
        elif field in {
            "operation",
            "finish_reason",
            "outcome",
            "site",
            "scenario",
            "expected_outcome",
            "observed_outcome",
        }:
            if not isinstance(field_value, str) or not field_value:
                raise InstrumentationError(f"{unit}.{field} must be a nonempty string")
        elif field in {
            "cuda_stream",
            "events_attempted",
            "events_written",
            "events_dropped",
            "device_synchronize_count",
            "capture_failures",
            "prompt_tokens",
        }:
            if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value < 0:
                raise InstrumentationError(f"{unit}.{field} must be a nonnegative integer")
        elif field in {
            "input_unchanged",
            "injected",
            "failure_observed",
            "cold_fill_occurred",
            "cross_session_contamination",
            "passed",
        } and not isinstance(field_value, bool):
            raise InstrumentationError(f"{unit}.{field} must be boolean")

    if unit in POSITION_SCOPED_UNITS and (event["position"] is None or event["row"] is None):
        raise InstrumentationError(f"{unit} requires an exact position and row")
    if (
        unit in ALL_LAYER_UNITS | QUEST_UNITS | GDN_UNITS | {"w4a16_projections"}
        and event["layer_index"] is None
    ):
        raise InstrumentationError(f"{unit} requires an exact layer index")
    if unit == "w4a16_projections" and evidence["operation"] != event["phase"]:
        raise InstrumentationError("W4A16 projection phase must equal its operation")
    canonical_phase = UNIT_PHASES.get(unit)
    if canonical_phase is not None and event["phase"] != canonical_phase:
        raise InstrumentationError(f"{unit} phase differs from its canonical phase")
    if unit == "fault_injection" and (
        event["phase"] != evidence["site"] or evidence["site"] not in FAULT_SITES
    ):
        raise InstrumentationError("fault-injection phase/site is invalid")
    if unit == "lifecycle":
        scenario = evidence["scenario"]
        if event["phase"] != scenario or scenario not in LIFECYCLE_SCENARIOS:
            raise InstrumentationError("lifecycle phase/scenario is invalid")
    if unit == "observer_integrity" and (
        evidence["installed_sites_sha256"] != evidence["required_sites_sha256"]
        or evidence["events_attempted"] != evidence["events_written"]
        or evidence["events_dropped"] != 0
        or evidence["capture_failures"] != 0
    ):
        raise InstrumentationError("debug observer did not capture every required event")
    _require_digest(event["event_sha256"], "full-assurance event hash")
    unsigned = dict(event)
    unsigned.pop("event_sha256")
    if event["event_sha256"] != _sha(unsigned):
        raise InstrumentationError("full-assurance event self-hash differs")
    return event


def _identity(event: dict[str, Any]) -> tuple[Any, ...]:
    commit_count = event["evidence"].get("commit_count", -1)
    return (
        UNIT_ORDER.index(event["unit"]),
        commit_count,
        -1 if event["position"] is None else event["position"],
        -1 if event["layer_index"] is None else event["layer_index"],
        event["phase"],
    )


def expected_probe_scopes(header_value: object) -> frozenset[tuple[Any, ...]]:
    """Return every exact probe scope required by one declared trace campaign."""

    header = normalize_header(header_value)
    scopes: set[tuple[Any, ...]] = set()
    for unit, phase in GLOBAL_PHASES.items():
        scopes.add((unit, None, None, None, phase, None))
    for position, row in zip(header["positions"], header["position_rows"], strict=True):
        for unit in POSITION_UNITS:
            scopes.add((unit, position, None, row, UNIT_PHASES[unit], None))
        for unit in ALL_LAYER_UNITS:
            for layer in range(LAYER_COUNT):
                scopes.add((unit, position, layer, row, UNIT_PHASES[unit], None))
        for site in header["projection_sites"]:
            scopes.add(
                (
                    "w4a16_projections",
                    position,
                    site["layer_index"],
                    row,
                    site["operation"],
                    None,
                )
            )
        for unit in QUEST_UNITS:
            for layer in QUEST_LAYERS:
                scopes.add((unit, position, layer, row, UNIT_PHASES[unit], None))
        for unit in GDN_UNITS:
            for layer in GDN_LAYERS:
                scopes.add((unit, position, layer, row, UNIT_PHASES[unit], None))
    for unit in COMMIT_UNITS:
        for count in COMMIT_COUNTS:
            scopes.add((unit, None, None, None, UNIT_PHASES[unit], count))
    for site in header["fault_sites"]:
        for count in COMMIT_COUNTS:
            scopes.add(("fault_injection", None, None, None, site, count))
    for scenario in header["lifecycle_scenarios"]:
        scopes.add(("lifecycle", None, None, None, scenario, None))
    return frozenset(scopes)


def validate_coverage(header_value: object, event_values: object) -> dict[str, Any]:
    """Reject any trace that does not observe the complete declared campaign."""

    header = normalize_header(header_value)
    if not isinstance(event_values, list):
        raise InstrumentationError("full-assurance events must be a list")
    events = [normalize_event(value) for value in event_values]
    if [event["sequence"] for event in events] != list(range(len(events))):
        raise InstrumentationError("full-assurance event sequence has a gap or reordering")
    for sequence, event in enumerate(events):
        expected_previous = None if sequence == 0 else events[sequence - 1]["event_sha256"]
        if event["previous_event_sha256"] != expected_previous:
            raise InstrumentationError("full-assurance event hash chain differs")
    if any(event["arm"] != header["arm"] for event in events):
        raise InstrumentationError("full-assurance event arm differs from its header")
    identities = [_identity(event) for event in events]
    duplicates = [identity for identity, count in Counter(identities).items() if count > 1]
    if duplicates:
        raise InstrumentationError(f"full-assurance events duplicate identities: {duplicates[:3]}")

    observed = set(identities)
    scope_index = {_scope(event) for event in events}
    expected_scopes = expected_probe_scopes(header)
    missing = sorted(expected_scopes - scope_index, key=repr)
    unexpected = sorted(scope_index - expected_scopes, key=repr)
    if missing or unexpected:
        raise InstrumentationError(
            "debug trace is not fully instrumented; "
            f"missing {len(missing)} probes: {missing[:16]}; "
            f"unexpected {len(unexpected)} probes: {unexpected[:16]}"
        )

    arm = header["arm"]
    observer = next(event for event in events if event["unit"] == "observer_integrity")
    observer_evidence = observer["evidence"]
    probe_sites_sha256 = required_probe_sites_sha256(header)
    if (
        observer_evidence["required_sites_sha256"] != probe_sites_sha256
        or observer_evidence["installed_sites_sha256"] != probe_sites_sha256
        or observer_evidence["events_attempted"] != len(events)
        or observer_evidence["events_written"] != len(events)
    ):
        raise InstrumentationError(
            "debug observer inventory or event accounting differs from complete coverage"
        )
    result = {
        "schema": COVERAGE_SCHEMA,
        "arm": arm,
        "run_id": header["run_id"],
        "event_count": len(events),
        "header_sha256": _sha(header),
        "positions": list(header["positions"]),
        "required_units_sha256": required_units_sha256(),
        "event_identity_sha256": _sha(sorted(observed)),
        "event_stream_sha256": _sha([event["event_sha256"] for event in events]),
        "passed": True,
    }
    result["coverage_sha256"] = _sha(result)
    return result


def evaluate_trace_invariants(header_value: object, event_values: object) -> dict[str, Any]:
    """Evaluate correctness without discarding a complete divergent counterexample."""

    header = normalize_header(header_value)
    coverage = validate_coverage(header, event_values)
    assert isinstance(event_values, list)
    events = sorted((normalize_event(value) for value in event_values), key=_identity)
    violations: list[dict[str, Any]] = []

    def violate(event: dict[str, Any], invariant: str, observed: object, expected: object) -> None:
        violations.append(
            {
                "identity": _identity(event),
                "unit": event["unit"],
                "phase": event["phase"],
                "position": event["position"],
                "layer_index": event["layer_index"],
                "row": event["row"],
                "invariant": invariant,
                "observed": observed,
                "expected": expected,
            }
        )

    for event in events:
        evidence = event["evidence"]
        unit = event["unit"]
        if evidence.get("nonfinite_count", 0) != 0:
            violate(event, "no_nonfinite_values", evidence["nonfinite_count"], 0)
        if evidence.get("device_error_word", 0) != 0:
            violate(event, "device_error_word_clear", evidence["device_error_word"], 0)
        if unit == "full_attention_qk_norm_rope":
            if evidence["input_unchanged"] is not True:
                violate(event, "qkv_input_immutable", evidence["input_unchanged"], True)
            for component in ("q", "k", "v"):
                observed = evidence[f"{component}_output_sha256"]
                expected = evidence[f"serial_{component}_output_sha256"]
                if observed != expected:
                    violate(event, f"batched_{component}_equals_serial_rows", observed, expected)
        elif unit == "provisional_isolation":
            pairs = [
                ("canonical_before_sha256", "canonical_after_provisional_sha256"),
                ("allocation_topology_before_sha256", "allocation_topology_after_sha256"),
                ("ownership_before_sha256", "ownership_after_sha256"),
                ("pins_before_sha256", "pins_after_sha256"),
                ("refcounts_before_sha256", "refcounts_after_sha256"),
                ("target_kv_before_sha256_by_layer", "target_kv_after_provisional_sha256_by_layer"),
                ("draft_kv_before_sha256_by_layer", "draft_kv_after_provisional_sha256_by_layer"),
                ("gdn_state_before_sha256_by_layer", "gdn_state_after_provisional_sha256_by_layer"),
                (
                    "convolution_state_before_sha256_by_layer",
                    "convolution_state_after_provisional_sha256_by_layer",
                ),
            ]
            for before, after in pairs:
                if evidence[before] != evidence[after]:
                    violate(
                        event, f"provisional_preserves_{before}", evidence[after], evidence[before]
                    )
        elif unit == "atomic_commit":
            pairs = [
                ("serial_reference_sha256", "published_sha256"),
                ("serial_allocation_topology_sha256", "published_allocation_topology_sha256"),
                ("serial_ownership_sha256", "published_ownership_sha256"),
                ("serial_pins_sha256", "published_pins_sha256"),
                ("serial_refcounts_sha256", "published_refcounts_sha256"),
                ("serial_target_kv_sha256_by_layer", "published_target_kv_sha256_by_layer"),
                ("serial_draft_kv_sha256_by_layer", "published_draft_kv_sha256_by_layer"),
                ("serial_gdn_state_sha256_by_layer", "published_gdn_state_sha256_by_layer"),
                (
                    "serial_convolution_state_sha256_by_layer",
                    "published_convolution_state_sha256_by_layer",
                ),
            ]
            for expected_name, observed_name in pairs:
                if evidence[expected_name] != evidence[observed_name]:
                    violate(
                        event,
                        f"atomic_{observed_name}_equals_serial",
                        evidence[observed_name],
                        evidence[expected_name],
                    )
            expected_epoch = evidence["commit_epoch_before"] + (
                1 if evidence["commit_count"] else 0
            )
            if evidence["commit_epoch_after"] != expected_epoch:
                violate(
                    event,
                    "atomic_commit_epoch",
                    evidence["commit_epoch_after"],
                    expected_epoch,
                )
            pointer_should_change = evidence["commit_count"] != 0
            pointer_changed = (
                evidence["root_pointer_before_sha256"] != evidence["root_pointer_after_sha256"]
            )
            if pointer_changed != pointer_should_change:
                violate(
                    event, "atomic_root_pointer_transition", pointer_changed, pointer_should_change
                )
        elif unit == "fault_injection":
            if evidence["injected"] is not True:
                violate(event, "fault_was_injected", evidence["injected"], True)
            if evidence["failure_observed"] is not True:
                violate(event, "injected_failure_observed", evidence["failure_observed"], True)
            for before, after in (
                ("canonical_before_sha256", "canonical_after_sha256"),
                ("allocation_topology_before_sha256", "allocation_topology_after_sha256"),
                ("ownership_before_sha256", "ownership_after_sha256"),
                ("pins_before_sha256", "pins_after_sha256"),
                ("refcounts_before_sha256", "refcounts_after_sha256"),
                ("target_kv_before_sha256_by_layer", "target_kv_after_sha256_by_layer"),
                ("draft_kv_before_sha256_by_layer", "draft_kv_after_sha256_by_layer"),
                ("gdn_state_before_sha256_by_layer", "gdn_state_after_sha256_by_layer"),
                (
                    "convolution_state_before_sha256_by_layer",
                    "convolution_state_after_sha256_by_layer",
                ),
            ):
                if evidence[before] != evidence[after]:
                    violate(event, f"fault_preserves_{before}", evidence[after], evidence[before])
        elif unit == "lifecycle":
            if evidence["observed_outcome"] != evidence["expected_outcome"]:
                violate(
                    event,
                    "lifecycle_outcome",
                    evidence["observed_outcome"],
                    evidence["expected_outcome"],
                )
            if evidence["state_after_sha256"] != evidence["reference_state_sha256"]:
                violate(
                    event,
                    "lifecycle_state_equals_serial",
                    evidence["state_after_sha256"],
                    evidence["reference_state_sha256"],
                )
            if evidence["cross_session_contamination"] is not False:
                violate(
                    event,
                    "no_cross_session_contamination",
                    evidence["cross_session_contamination"],
                    False,
                )
            scenario = evidence["scenario"]
            if (
                scenario.startswith(("restored_", "restart_", "offload_", "rapid_switch_"))
                and evidence["cold_fill_occurred"] is not False
            ):
                violate(event, "resume_without_cold_fill", evidence["cold_fill_occurred"], False)
        elif unit == "assurance_release_equivalence":
            if evidence["passed"] is not True:
                violate(event, "assurance_release_gate_passed", evidence["passed"], True)
            if evidence["release_output_sha256"] != evidence["serial_reference_sha256"]:
                violate(
                    event,
                    "release_output_equals_serial",
                    evidence["release_output_sha256"],
                    evidence["serial_reference_sha256"],
                )
        elif unit == "structured_outcome" and evidence["outcome"] not in {
            "tool_call",
            "final_answer",
        }:
            violate(
                event,
                "structured_outcome_complete",
                evidence["outcome"],
                "tool_call|final_answer",
            )

    result = {
        "schema": INVARIANT_REPORT_SCHEMA,
        "arm": header["arm"],
        "run_id": header["run_id"],
        "coverage_sha256": coverage["coverage_sha256"],
        "violation_count": len(violations),
        "first_violation": None if not violations else violations[0],
        "violations": violations,
        "passed": not violations,
    }
    result["invariants_sha256"] = _sha(result)
    return result


def _first_nested_difference(
    left: object, right: object, path: tuple[object, ...] = ()
) -> tuple[tuple[object, ...], object, object] | None:
    """Locate the first canonical leaf difference inside a semantic witness."""

    if type(left) is not type(right):
        return path, left, right
    if isinstance(left, dict):
        assert isinstance(right, dict)
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                return (*path, key), left.get(key), right.get(key)
            difference = _first_nested_difference(left[key], right[key], (*path, key))
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        assert isinstance(right, list)
        for index in range(max(len(left), len(right))):
            if index >= len(left) or index >= len(right):
                return (
                    (*path, index),
                    None if index >= len(left) else left[index],
                    None if index >= len(right) else right[index],
                )
            difference = _first_nested_difference(left[index], right[index], (*path, index))
            if difference is not None:
                return difference
        return None
    if left != right:
        return path, left, right
    return None


def _comparison_identity(event: dict[str, Any]) -> tuple[Any, ...]:
    """Pair a serial row with the M8 logical row for the same absolute position."""

    commit_count = event["evidence"].get("commit_count", -1)
    return (
        UNIT_ORDER.index(event["unit"]),
        commit_count,
        -1 if event["position"] is None else event["position"],
        -1 if event["layer_index"] is None else event["layer_index"],
        event["phase"],
    )


def compare_complete_traces(
    serial_header: object,
    serial_events: object,
    speculative_header: object,
    speculative_events: object,
) -> dict[str, Any]:
    """Compare two already complete traces and identify the first semantic difference."""

    left_header = normalize_header(serial_header)
    right_header = normalize_header(speculative_header)
    if left_header["arm"] != "serial-m1" or right_header["arm"] != "speculative-m8":
        raise InstrumentationError("trace comparison requires serial-m1 then speculative-m8")
    for key in HEADER_KEYS - {"schema", "run_id", "arm", "position_rows"}:
        if left_header[key] != right_header[key]:
            raise InstrumentationError(f"trace identities differ at {key}")
    left_coverage = validate_coverage(left_header, serial_events)
    right_coverage = validate_coverage(right_header, speculative_events)
    left_invariants = evaluate_trace_invariants(left_header, serial_events)
    right_invariants = evaluate_trace_invariants(right_header, speculative_events)
    assert isinstance(serial_events, list) and isinstance(speculative_events, list)
    normalized_left = [normalize_event(event) for event in serial_events]
    normalized_right = [normalize_event(event) for event in speculative_events]
    left = {_comparison_identity(event): event for event in normalized_left}
    right = {_comparison_identity(event): event for event in normalized_right}
    first_difference: dict[str, Any] | None = None
    for identity in sorted(set(left) | set(right)):
        left_event = left.get(identity)
        right_event = right.get(identity)
        if left_event is None or right_event is None:
            first_difference = {
                "identity": identity,
                "field": "event_presence",
                "field_path": ["event_presence"],
                "serial_value": left_event is not None,
                "speculative_value": right_event is not None,
            }
            break
        # Extra diagnostic fields may contain process-local addresses or timing.
        # Only the required semantic witnesses participate in equivalence.
        fields = sorted(
            REQUIRED_EVIDENCE_FIELDS[left_event["unit"]]
            - ARM_LOCAL_EVIDENCE_FIELDS.get(left_event["unit"], frozenset())
        )
        for field in fields:
            difference = _first_nested_difference(
                left_event["evidence"].get(field), right_event["evidence"].get(field)
            )
            if difference is not None:
                nested_path, serial_value, speculative_value = difference
                first_difference = {
                    "identity": identity,
                    "unit": left_event["unit"],
                    "phase": left_event["phase"],
                    "position": left_event["position"],
                    "layer_index": left_event["layer_index"],
                    "serial_row": left_event["row"],
                    "speculative_row": right_event["row"],
                    "field": field,
                    "field_path": [field, *nested_path],
                    "serial_value": serial_value,
                    "speculative_value": speculative_value,
                }
                break
        if first_difference is not None:
            break
    result = {
        "schema": COMPARISON_SCHEMA,
        "classification": "bounded_qualification",
        "universal_equivalence_proven": False,
        "serial_coverage_sha256": left_coverage["coverage_sha256"],
        "speculative_coverage_sha256": right_coverage["coverage_sha256"],
        "serial_invariants_sha256": left_invariants["invariants_sha256"],
        "speculative_invariants_sha256": right_invariants["invariants_sha256"],
        "serial_first_invariant_violation": left_invariants["first_violation"],
        "speculative_first_invariant_violation": right_invariants["first_violation"],
        "first_difference": first_difference,
        "passed": (
            first_difference is None and left_invariants["passed"] and right_invariants["passed"]
        ),
    }
    result["comparison_sha256"] = _sha(result)
    return result


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
                raise InstrumentationError(f"short write while creating {path.name}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns


def _directory_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_mode, value.st_uid


def _require_owned_private_directory(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise InstrumentationError(f"cannot inspect {label}: {error}") from error
    if (
        path.is_symlink()
        or not path.is_dir()
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
    ):
        raise InstrumentationError(f"{label} must be an owned private non-symlink directory")
    return metadata


def _read_stable_private_file(path: Path, label: str) -> bytes:
    try:
        before = path.lstat()
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise InstrumentationError(f"cannot read {label}: {error}") from error
    if (
        _stat_identity(before) != _stat_identity(after)
        or path.is_symlink()
        or not path.is_file()
        or before.st_uid != os.getuid()
        or before.st_mode & 0o077
    ):
        raise InstrumentationError(f"{label} changed or is not an owned private regular file")
    return payload


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


FRAGMENT_HEADER_KEYS = {
    "schema",
    "fragment_id",
    "campaign_header_sha256",
    "full_header",
    "scopes",
    "producer_kind",
    "producer_sha256",
}


def _normalize_fragment_header(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != FRAGMENT_HEADER_KEYS:
        raise InstrumentationError("full-assurance fragment header keys are incomplete")
    header = _json_clone(value)
    if header["schema"] != FRAGMENT_HEADER_SCHEMA:
        raise InstrumentationError("full-assurance fragment header schema differs")
    fragment_id = header["fragment_id"]
    if (
        not isinstance(fragment_id, str)
        or not fragment_id
        or len(fragment_id) > 128
        or any(character.isspace() for character in fragment_id)
    ):
        raise InstrumentationError("full-assurance fragment ID is invalid")
    full_header = normalize_header(header["full_header"])
    if header["campaign_header_sha256"] != _sha(full_header):
        raise InstrumentationError("full-assurance fragment campaign binding differs")
    _require_digest(header["producer_sha256"], "full-assurance fragment producer")
    producer_kind = header["producer_kind"]
    if producer_kind not in PRODUCER_UNITS:
        raise InstrumentationError("full-assurance fragment producer kind is invalid")
    raw_scopes = header["scopes"]
    if not isinstance(raw_scopes, list) or not raw_scopes:
        raise InstrumentationError("full-assurance fragment declares no scopes")
    scopes = [_normalize_scope(scope) for scope in raw_scopes]
    if len(scopes) != len(set(scopes)):
        raise InstrumentationError("full-assurance fragment declares duplicate scopes")
    if scopes != sorted(scopes, key=repr):
        raise InstrumentationError("full-assurance fragment scopes are not canonical")
    fragment_positions = {scope[1] for scope in scopes if scope[1] is not None}
    if len(fragment_positions) > 8:
        raise InstrumentationError(
            "one runtime fragment may instrument at most eight exact positions"
        )
    observer_scope = (
        "observer_integrity",
        None,
        None,
        None,
        UNIT_PHASES["observer_integrity"],
        None,
    )
    expected = expected_probe_scopes(full_header)
    unexpected = set(scopes) - (expected - {observer_scope})
    if unexpected or observer_scope in scopes:
        raise InstrumentationError(
            f"full-assurance fragment declares invalid scopes: {sorted(unexpected, key=repr)[:8]}"
        )
    if any(producer_kind_for_scope(scope) != producer_kind for scope in scopes):
        raise InstrumentationError(
            "full-assurance fragment mixes scopes from different producer classes"
        )
    header["full_header"] = full_header
    header["scopes"] = [list(scope) for scope in scopes]
    return header


class FullAssuranceFragmentWriter:
    """Capture one honest subset of a multi-process assurance campaign.

    A fragment cannot claim global completeness.  Its completion marker proves
    only that every scope declared before execution was observed exactly once
    and durably. Engine fragments additionally require device synchronization;
    controller fragments must report zero device synchronizations because they
    authenticate already-settled external evidence. The campaign assembler is
    the sole component allowed to derive the global observer-integrity event.
    """

    def __init__(
        self,
        root: Path,
        full_header: object,
        *,
        fragment_id: str,
        scopes: object,
        producer_kind: str,
        producer_sha256: str,
        sync_interval: int = 32,
    ) -> None:
        normalized_full_header = normalize_header(full_header)
        header = _normalize_fragment_header(
            {
                "schema": FRAGMENT_HEADER_SCHEMA,
                "fragment_id": fragment_id,
                "campaign_header_sha256": _sha(normalized_full_header),
                "full_header": normalized_full_header,
                "scopes": [
                    list(scope)
                    for scope in sorted(
                        (_normalize_scope(scope) for scope in scopes), key=repr
                    )
                ]
                if isinstance(scopes, (list, tuple, set, frozenset))
                else scopes,
                "producer_kind": producer_kind,
                "producer_sha256": producer_sha256,
            }
        )
        if (
            not root.is_absolute()
            or root != root.resolve(strict=False)
            or root.exists()
            or root.is_symlink()
        ):
            raise InstrumentationError("full-assurance fragment root must be a new absolute path")
        if (
            isinstance(sync_interval, bool)
            or not isinstance(sync_interval, int)
            or sync_interval < 1
        ):
            raise InstrumentationError("full-assurance fragment sync interval must be positive")
        parent = root.parent.resolve(strict=True)
        if parent != root.parent:
            raise InstrumentationError("full-assurance fragment path must not traverse symlinks")
        _require_owned_private_directory(parent, "full-assurance fragment parent")
        root.mkdir(mode=0o700)
        self._root_identity = _directory_identity(
            _require_owned_private_directory(root, "full-assurance fragment root")
        )
        self.root = root
        self.header = header
        self.events: list[dict[str, Any]] = []
        self._allowed_scopes = {tuple(scope) for scope in header["scopes"]}
        self._descriptor: int | None = None
        self._closed = False
        self._lock = threading.RLock()
        self._sync_interval = sync_interval
        try:
            _write_create_only(root / "header.json", _canonical(header) + b"\n")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            self._descriptor = os.open(root / "events.jsonl", flags, 0o600)
            self._event_identity = _stat_identity(os.fstat(self._descriptor))[:3]
            _fsync_directory(root)
            _fsync_directory(parent)
        except BaseException:
            if self._descriptor is not None:
                os.close(self._descriptor)
                self._descriptor = None
            raise

    def append(
        self,
        *,
        unit: str,
        phase: str,
        position: int | None,
        layer_index: int | None,
        row: int | None,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            if self._closed or self._descriptor is None:
                raise InstrumentationError("full-assurance fragment writer is closed")
            event = seal_event(
                {
                    "schema": TRACE_EVENT_SCHEMA,
                    "sequence": len(self.events),
                    "arm": self.header["full_header"]["arm"],
                    "unit": unit,
                    "phase": phase,
                    "position": position,
                    "layer_index": layer_index,
                    "row": row,
                    "evidence": evidence,
                    "previous_event_sha256": (
                        None if not self.events else self.events[-1]["event_sha256"]
                    ),
                }
            )
            scope = _scope(event)
            if scope not in self._allowed_scopes:
                raise InstrumentationError(f"event is outside its declared fragment scope: {scope}")
            if any(_scope(existing) == scope for existing in self.events):
                raise InstrumentationError(f"fragment captured a scope more than once: {scope}")
            if (
                _directory_identity(
                    _require_owned_private_directory(self.root, "full-assurance fragment root")
                )
                != self._root_identity
                or _stat_identity(os.fstat(self._descriptor))[:3] != self._event_identity
            ):
                self.abort()
                raise InstrumentationError("full-assurance fragment identity changed")
            payload = _canonical(event) + b"\n"
            view = memoryview(payload)
            try:
                while view:
                    written = os.write(self._descriptor, view)
                    if written <= 0:
                        raise InstrumentationError("short write while appending fragment event")
                    view = view[written:]
                self.events.append(event)
                if len(self.events) % self._sync_interval == 0:
                    os.fsync(self._descriptor)
            except BaseException:
                self.abort()
                raise
            return _json_clone(event)

    def finalize(
        self,
        *,
        device_synchronize_count: int,
        installed_instrumentation_sites: object,
    ) -> dict[str, Any]:
        with self._lock:
            if self._closed or self._descriptor is None:
                raise InstrumentationError("full-assurance fragment writer is already closed")
            observed = {_scope(event) for event in self.events}
            if observed != self._allowed_scopes:
                missing = sorted(self._allowed_scopes - observed, key=repr)
                raise InstrumentationError(
                    "full-assurance fragment is incomplete; "
                    f"missing {len(missing)} scopes: {missing[:8]}"
                )
            is_engine_fragment = self.header["producer_kind"] == "engine_model_capture"
            if (
                isinstance(device_synchronize_count, bool)
                or not isinstance(device_synchronize_count, int)
                or device_synchronize_count < 0
                or (is_engine_fragment and device_synchronize_count < len(self.events))
                or (not is_engine_fragment and device_synchronize_count != 0)
            ):
                requirement = (
                    "at least one device synchronization per captured event"
                    if is_engine_fragment
                    else "zero device synchronizations for a controller-only fragment"
                )
                raise InstrumentationError(f"fragment must prove {requirement}")
            if not isinstance(installed_instrumentation_sites, (list, tuple, set, frozenset)):
                raise InstrumentationError("fragment installed-site inventory is malformed")
            installed_sites = tuple(sorted(installed_instrumentation_sites))
            if any(not isinstance(site, str) or not site for site in installed_sites):
                raise InstrumentationError("fragment installed-site inventory is malformed")
            expected_sites = expected_instrumentation_sites(
                self.header["producer_kind"],
                self._allowed_scopes,
                arm=self.header["full_header"]["arm"],
            )
            if installed_sites != tuple(sorted(expected_sites)):
                raise InstrumentationError(
                    "fragment did not install every required producer boundary; "
                    f"expected {sorted(expected_sites)}, observed {list(installed_sites)}"
                )
            os.fsync(self._descriptor)
            os.close(self._descriptor)
            self._descriptor = None
            self._closed = True
            events_payload = _read_stable_private_file(
                self.root / "events.jsonl", "full-assurance fragment events"
            )
            persisted = [
                normalize_event(json.loads(line)) for line in events_payload.splitlines()
            ]
            if persisted != self.events:
                raise InstrumentationError("persisted fragment events differ from captured memory")
            complete = {
                "schema": FRAGMENT_COMPLETE_SCHEMA,
                "header_sha256": _sha(self.header),
                "events_sha256": hashlib.sha256(events_payload).hexdigest(),
                "event_count": len(persisted),
                "scope_sha256": _sha(sorted(self._allowed_scopes, key=repr)),
                "producer_kind": self.header["producer_kind"],
                "producer_sha256": self.header["producer_sha256"],
                "installed_instrumentation_sites_sha256": _sha(installed_sites),
                "required_instrumentation_sites_sha256": _sha(tuple(sorted(expected_sites))),
                "events_attempted": len(persisted),
                "events_written": len(persisted),
                "events_dropped": 0,
                "device_synchronize_count": device_synchronize_count,
                "capture_failures": 0,
            }
            complete["complete_sha256"] = _sha(complete)
            _write_create_only(self.root / "complete.json", _canonical(complete) + b"\n")
            _fsync_directory(self.root)
            return complete

    def abort(self) -> None:
        with self._lock:
            if self._descriptor is not None:
                os.fsync(self._descriptor)
                os.close(self._descriptor)
                self._descriptor = None
            self._closed = True


def load_complete_fragment(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if (
        not root.is_absolute()
        or root != root.resolve(strict=False)
        or not root.is_dir()
        or root.is_symlink()
    ):
        raise InstrumentationError("full-assurance fragment root is unsafe or absent")
    before = _require_owned_private_directory(root, "full-assurance fragment root")
    expected_names = {"header.json", "events.jsonl", "complete.json"}
    if {path.name for path in root.iterdir()} != expected_names:
        raise InstrumentationError("full-assurance fragment contains unknown artifacts")
    payloads = {
        name: _read_stable_private_file(root / name, f"full-assurance fragment {name}")
        for name in expected_names
    }
    after = _require_owned_private_directory(root, "full-assurance fragment root")
    if _directory_identity(before) != _directory_identity(after):
        raise InstrumentationError("full-assurance fragment changed during authentication")
    try:
        header = _normalize_fragment_header(json.loads(payloads["header.json"]))
        events = [
            normalize_event(json.loads(line)) for line in payloads["events.jsonl"].splitlines()
        ]
        complete = json.loads(payloads["complete.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InstrumentationError(f"cannot parse full-assurance fragment: {error}") from error
    scopes = {tuple(scope) for scope in header["scopes"]}
    if len(events) != len(scopes) or {_scope(event) for event in events} != scopes:
        raise InstrumentationError("full-assurance fragment event coverage differs")
    for sequence, event in enumerate(events):
        if event["sequence"] != sequence or event["arm"] != header["full_header"]["arm"]:
            raise InstrumentationError("full-assurance fragment event ordering or arm differs")
        previous = None if sequence == 0 else events[sequence - 1]["event_sha256"]
        if event["previous_event_sha256"] != previous:
            raise InstrumentationError("full-assurance fragment hash chain differs")
    if not isinstance(complete, dict) or complete.get("schema") != FRAGMENT_COMPLETE_SCHEMA:
        raise InstrumentationError("full-assurance fragment completion marker is invalid")
    signed = dict(complete)
    complete_sha256 = signed.pop("complete_sha256", None)
    if complete_sha256 != _sha(signed):
        raise InstrumentationError("full-assurance fragment completion self-hash differs")
    expected = {
        "header_sha256": _sha(header),
        "events_sha256": hashlib.sha256(payloads["events.jsonl"]).hexdigest(),
        "event_count": len(events),
        "scope_sha256": _sha(sorted(scopes, key=repr)),
        "producer_kind": header["producer_kind"],
        "producer_sha256": header["producer_sha256"],
        "required_instrumentation_sites_sha256": _sha(
            tuple(
                sorted(
                    expected_instrumentation_sites(
                        header["producer_kind"],
                        scopes,
                        arm=header["full_header"]["arm"],
                    )
                )
            )
        ),
        "events_attempted": len(events),
        "events_written": len(events),
        "events_dropped": 0,
        "capture_failures": 0,
    }
    if any(complete.get(key) != value for key, value in expected.items()):
        raise InstrumentationError("full-assurance fragment completion bindings differ")
    syncs = complete.get("device_synchronize_count")
    is_engine_fragment = header["producer_kind"] == "engine_model_capture"
    if (
        isinstance(syncs, bool)
        or not isinstance(syncs, int)
        or syncs < 0
        or (is_engine_fragment and syncs < len(events))
        or (not is_engine_fragment and syncs != 0)
    ):
        raise InstrumentationError("full-assurance fragment synchronization evidence is incomplete")
    if (
        complete.get("installed_instrumentation_sites_sha256")
        != complete.get("required_instrumentation_sites_sha256")
    ):
        raise InstrumentationError("full-assurance fragment installed-site inventory differs")
    return header, events, complete


def assemble_complete_campaign(output_root: Path, fragment_roots: object) -> dict[str, Any]:
    """Authenticate disjoint run fragments and publish one complete trace."""

    if not isinstance(fragment_roots, (list, tuple)) or not fragment_roots:
        raise InstrumentationError("full-assurance campaign has no fragments")
    loaded = [load_complete_fragment(Path(root)) for root in fragment_roots]
    full_header = loaded[0][0]["full_header"]
    campaign_header_sha256 = _sha(full_header)
    fragment_ids: set[str] = set()
    producer_kinds: set[str] = set()
    observed_scopes: set[tuple[Any, ...]] = set()
    collected: list[dict[str, Any]] = []
    marker_hashes: list[str] = []
    device_synchronize_count = 0
    for fragment_header, events, complete in loaded:
        if (
            fragment_header["campaign_header_sha256"] != campaign_header_sha256
            or fragment_header["full_header"] != full_header
        ):
            raise InstrumentationError("full-assurance fragment belongs to another campaign")
        fragment_id = fragment_header["fragment_id"]
        if fragment_id in fragment_ids:
            raise InstrumentationError("full-assurance campaign repeats a fragment ID")
        fragment_ids.add(fragment_id)
        producer_kinds.add(fragment_header["producer_kind"])
        scopes = {tuple(scope) for scope in fragment_header["scopes"]}
        overlap = observed_scopes & scopes
        if overlap:
            raise InstrumentationError(
                f"full-assurance campaign fragments overlap: {sorted(overlap, key=repr)[:8]}"
            )
        observed_scopes.update(scopes)
        collected.extend(events)
        marker_hashes.append(complete["complete_sha256"])
        device_synchronize_count += complete["device_synchronize_count"]
    observer_scope = (
        "observer_integrity",
        None,
        None,
        None,
        UNIT_PHASES["observer_integrity"],
        None,
    )
    required_fragment_scopes = expected_probe_scopes(full_header) - {observer_scope}
    if observed_scopes != required_fragment_scopes:
        missing = sorted(required_fragment_scopes - observed_scopes, key=repr)
        unexpected = sorted(observed_scopes - required_fragment_scopes, key=repr)
        raise InstrumentationError(
            "full-assurance fragment union is incomplete; "
            f"missing {len(missing)}: {missing[:8]}; unexpected {len(unexpected)}: {unexpected[:8]}"
        )
    expected_producers = set(partition_probe_scopes_by_producer(full_header))
    if producer_kinds != expected_producers:
        raise InstrumentationError(
            "full-assurance campaign producer inventory is incomplete; "
            f"expected {sorted(expected_producers)}, observed {sorted(producer_kinds)}"
        )
    installed_sites_sha256 = _sha(sorted(observed_scopes, key=repr))
    required_sites_sha256 = required_probe_sites_sha256(full_header)
    if installed_sites_sha256 != required_sites_sha256:
        raise InstrumentationError("full-assurance installed probe sites differ from the campaign")
    observer_evidence = {
        "installed_sites_sha256": installed_sites_sha256,
        "required_sites_sha256": required_sites_sha256,
        "events_attempted": len(collected) + 1,
        "events_written": len(collected) + 1,
        "events_dropped": 0,
        "stream_fsync_sha256": _sha(marker_hashes),
        "device_synchronize_count": device_synchronize_count,
        "capture_failures": 0,
        "device_error_word": 0,
        "diagnostic_fragment_ids_sha256": _sha(sorted(fragment_ids)),
        "diagnostic_fragment_complete_sha256": _sha(marker_hashes),
        "diagnostic_producer_kinds_sha256": _sha(sorted(producer_kinds)),
    }
    unsigned_observer = {
        "schema": TRACE_EVENT_SCHEMA,
        "sequence": 0,
        "arm": full_header["arm"],
        "unit": "observer_integrity",
        "phase": UNIT_PHASES["observer_integrity"],
        "position": None,
        "layer_index": None,
        "row": None,
        "evidence": observer_evidence,
        "previous_event_sha256": None,
    }
    collected.append(seal_event(unsigned_observer))
    writer = FullAssuranceTraceWriter(output_root, full_header)
    try:
        for event in sorted(collected, key=_identity):
            writer.append(
                unit=event["unit"],
                phase=event["phase"],
                position=event["position"],
                layer_index=event["layer_index"],
                row=event["row"],
                evidence=event["evidence"],
            )
        complete = writer.finalize()
    except BaseException:
        writer.abort()
        raise
    return {
        "schema": CAMPAIGN_ASSEMBLY_SCHEMA,
        "fragment_count": len(loaded),
        "fragment_complete_sha256": _sha(marker_hashes),
        "trace_complete_sha256": complete["complete_sha256"],
        "passed": True,
    }


class FullAssuranceTraceWriter:
    """Create one append-only, hash-chained, fail-closed debug trace."""

    def __init__(self, root: Path, header_value: object, *, sync_interval: int = 128) -> None:
        if (
            not root.is_absolute()
            or root != root.resolve(strict=False)
            or root.exists()
            or root.is_symlink()
        ):
            raise InstrumentationError("full-assurance trace root must be a new absolute path")
        if (
            isinstance(sync_interval, bool)
            or not isinstance(sync_interval, int)
            or sync_interval < 1
        ):
            raise InstrumentationError("full-assurance sync interval must be positive")
        parent = root.parent.resolve(strict=True)
        if parent != root.parent:
            raise InstrumentationError("full-assurance trace path must not traverse symlinks")
        _require_owned_private_directory(parent, "full-assurance trace parent")
        root.mkdir(mode=0o700)
        self._root_identity = _directory_identity(
            _require_owned_private_directory(root, "full-assurance trace root")
        )
        self.root = root
        self.header = normalize_header(header_value)
        self.sync_interval = sync_interval
        self.events: list[dict[str, Any]] = []
        self._closed = False
        self._descriptor: int | None = None
        self._lock = threading.RLock()
        try:
            _write_create_only(root / "header.json", _canonical(self.header) + b"\n")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            self._descriptor = os.open(root / "events.jsonl", flags, 0o600)
            event_metadata = os.fstat(self._descriptor)
            if (
                not os.path.samestat(event_metadata, (root / "events.jsonl").lstat())
                or event_metadata.st_uid != os.getuid()
                or event_metadata.st_mode & 0o077
            ):
                raise InstrumentationError("full-assurance event stream identity is unsafe")
            self._event_identity = _stat_identity(event_metadata)[:3]
            _fsync_directory(root)
            _fsync_directory(parent)
        except BaseException:
            if self._descriptor is not None:
                os.close(self._descriptor)
                self._descriptor = None
            raise

    def append(
        self,
        *,
        unit: str,
        phase: str,
        position: int | None,
        layer_index: int | None,
        row: int | None,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            if self._closed or self._descriptor is None:
                raise InstrumentationError("full-assurance trace writer is closed")
            if (
                _directory_identity(
                    _require_owned_private_directory(self.root, "full-assurance trace root")
                )
                != self._root_identity
            ):
                self.abort()
                raise InstrumentationError("full-assurance trace root identity changed")
            event_metadata = os.fstat(self._descriptor)
            if _stat_identity(event_metadata)[:3] != self._event_identity:
                self.abort()
                raise InstrumentationError("full-assurance event stream identity changed")
            event = seal_event(
                {
                    "schema": TRACE_EVENT_SCHEMA,
                    "sequence": len(self.events),
                    "arm": self.header["arm"],
                    "unit": unit,
                    "phase": phase,
                    "position": position,
                    "layer_index": layer_index,
                    "row": row,
                    "evidence": evidence,
                    "previous_event_sha256": (
                        None if not self.events else self.events[-1]["event_sha256"]
                    ),
                }
            )
            payload = _canonical(event) + b"\n"
            view = memoryview(payload)
            try:
                while view:
                    written = os.write(self._descriptor, view)
                    if written <= 0:
                        raise InstrumentationError(
                            "short write while appending full-assurance event"
                        )
                    view = view[written:]
                self.events.append(event)
                if len(self.events) % self.sync_interval == 0:
                    os.fsync(self._descriptor)
            except BaseException:
                self.abort()
                raise
            return _json_clone(event)

    def finalize(self) -> dict[str, Any]:
        with self._lock:
            if self._closed or self._descriptor is None:
                raise InstrumentationError("full-assurance trace writer is already closed")
            os.fsync(self._descriptor)
            os.close(self._descriptor)
            self._descriptor = None
            self._closed = True
            events_payload = _read_stable_private_file(
                self.root / "events.jsonl", "full-assurance event stream"
            )
            try:
                persisted_events = [
                    normalize_event(json.loads(line)) for line in events_payload.splitlines()
                ]
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise InstrumentationError(
                    f"persisted full-assurance event stream is invalid: {error}"
                ) from error
            if persisted_events != self.events:
                raise InstrumentationError(
                    "persisted full-assurance event stream differs from captured memory"
                )
            persisted_header = normalize_header(
                json.loads(
                    _read_stable_private_file(self.root / "header.json", "full-assurance header")
                )
            )
            if persisted_header != self.header:
                raise InstrumentationError("persisted full-assurance header differs")
            coverage = validate_coverage(persisted_header, persisted_events)
            _write_create_only(self.root / "coverage.json", _canonical(coverage) + b"\n")
            _fsync_directory(self.root)
            complete = {
                "schema": "urn:qwen-r9700:full-assurance-complete:v1",
                "header_sha256": _sha(persisted_header),
                "coverage_sha256": coverage["coverage_sha256"],
                "events_sha256": hashlib.sha256(events_payload).hexdigest(),
                "event_count": len(persisted_events),
            }
            complete["complete_sha256"] = _sha(complete)
            _write_create_only(self.root / "complete.json", _canonical(complete) + b"\n")
            _fsync_directory(self.root)
            return complete

    def abort(self) -> None:
        """Close an incomplete trace without publishing a completion marker."""

        with self._lock:
            if self._descriptor is not None:
                os.fsync(self._descriptor)
                os.close(self._descriptor)
                self._descriptor = None
            self._closed = True


def load_complete_trace(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Authenticate and load only a trace carrying the final completion marker."""

    if (
        not root.is_absolute()
        or root != root.resolve(strict=False)
        or not root.is_dir()
        or root.is_symlink()
    ):
        raise InstrumentationError("full-assurance trace root is unsafe or absent")
    root_before = _require_owned_private_directory(root, "full-assurance trace root")
    expected_names = {"header.json", "events.jsonl", "coverage.json", "complete.json"}
    if {path.name for path in root.iterdir()} != expected_names:
        raise InstrumentationError("full-assurance trace directory contains unknown artifacts")
    payloads = {
        name: _read_stable_private_file(root / name, f"full-assurance artifact {name}")
        for name in expected_names
    }
    root_after = _require_owned_private_directory(root, "full-assurance trace root")
    if (
        _directory_identity(root_before) != _directory_identity(root_after)
        or {path.name for path in root.iterdir()} != expected_names
    ):
        raise InstrumentationError("full-assurance trace changed during authentication")
    try:
        header = normalize_header(json.loads(payloads["header.json"]))
        events = [
            normalize_event(json.loads(line)) for line in payloads["events.jsonl"].splitlines()
        ]
        coverage = json.loads(payloads["coverage.json"])
        complete = json.loads(payloads["complete.json"])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InstrumentationError(f"cannot parse full-assurance trace: {error}") from error
    recomputed = validate_coverage(header, events)
    if coverage != recomputed:
        raise InstrumentationError("full-assurance coverage artifact differs")
    if not isinstance(complete, dict) or complete.get("schema") != (
        "urn:qwen-r9700:full-assurance-complete:v1"
    ):
        raise InstrumentationError("full-assurance completion marker is invalid")
    signed = dict(complete)
    complete_sha256 = signed.pop("complete_sha256", None)
    if complete_sha256 != _sha(signed):
        raise InstrumentationError("full-assurance completion self-hash differs")
    expected_complete = {
        "header_sha256": _sha(header),
        "coverage_sha256": coverage["coverage_sha256"],
        "events_sha256": hashlib.sha256(payloads["events.jsonl"]).hexdigest(),
        "event_count": len(events),
    }
    if any(complete.get(key) != value for key, value in expected_complete.items()):
        raise InstrumentationError("full-assurance completion bindings differ")
    return header, events, coverage


COUNTEREXAMPLE_DESCRIPTOR_KEYS = {
    "encoding",
    "dtype",
    "shape",
    "strides",
    "storage_offset",
}


def _normalize_counterexample_descriptor(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != COUNTEREXAMPLE_DESCRIPTOR_KEYS:
        raise InstrumentationError(f"{label} counterexample descriptor keys are incomplete")
    descriptor = _json_clone(value)
    for key in ("encoding", "dtype"):
        if not isinstance(descriptor[key], str) or not descriptor[key]:
            raise InstrumentationError(f"{label} counterexample {key} is invalid")
    shape = descriptor["shape"]
    strides = descriptor["strides"]
    if (
        not isinstance(shape, list)
        or not isinstance(strides, list)
        or len(shape) != len(strides)
        or len(shape) > 16
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in strides)
    ):
        raise InstrumentationError(f"{label} counterexample tensor geometry is invalid")
    offset = descriptor["storage_offset"]
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise InstrumentationError(f"{label} counterexample storage offset is invalid")
    return descriptor


def write_counterexample_capsule(
    root: Path,
    comparison_value: object,
    *,
    serial_payload: bytes,
    speculative_payload: bytes,
    serial_descriptor: object,
    speculative_descriptor: object,
) -> dict[str, Any]:
    """Preserve the exact first differing physical values in a create-only capsule."""

    if not isinstance(comparison_value, dict):
        raise InstrumentationError("counterexample comparison is not an object")
    comparison = _json_clone(comparison_value)
    comparison_sha256 = comparison.pop("comparison_sha256", None)
    if comparison_sha256 != _sha(comparison):
        raise InstrumentationError("counterexample comparison is not self-authenticating")
    comparison["comparison_sha256"] = comparison_sha256
    first_difference = comparison.get("first_difference")
    if comparison.get("passed") is not False or not isinstance(first_difference, dict):
        raise InstrumentationError("counterexample capsule requires a failed comparison")
    if (
        not isinstance(serial_payload, bytes)
        or not isinstance(speculative_payload, bytes)
        or not serial_payload
        or not speculative_payload
    ):
        raise InstrumentationError("counterexample payloads must be nonempty immutable bytes")
    if max(len(serial_payload), len(speculative_payload)) > 1 << 30:
        raise InstrumentationError("counterexample payload exceeds the 1 GiB capsule bound")
    serial = _normalize_counterexample_descriptor(serial_descriptor, "serial")
    speculative = _normalize_counterexample_descriptor(speculative_descriptor, "speculative")
    if (
        not root.is_absolute()
        or root != root.resolve(strict=False)
        or root.exists()
        or root.is_symlink()
    ):
        raise InstrumentationError("counterexample root must be a new absolute path")
    parent = root.parent.resolve(strict=True)
    if parent != root.parent:
        raise InstrumentationError("counterexample path must not traverse symlinks")
    _require_owned_private_directory(parent, "counterexample parent")
    root.mkdir(mode=0o700)
    try:
        _write_create_only(root / "serial.bin", serial_payload)
        _write_create_only(root / "speculative.bin", speculative_payload)
        metadata = {
            "schema": COUNTEREXAMPLE_SCHEMA,
            "comparison_sha256": comparison_sha256,
            "first_difference": first_difference,
            "first_difference_sha256": _sha(first_difference),
            "serial": {
                **serial,
                "payload_bytes": len(serial_payload),
                "payload_sha256": hashlib.sha256(serial_payload).hexdigest(),
                "semantic_value_sha256": _sha(first_difference.get("serial_value")),
            },
            "speculative": {
                **speculative,
                "payload_bytes": len(speculative_payload),
                "payload_sha256": hashlib.sha256(speculative_payload).hexdigest(),
                "semantic_value_sha256": _sha(first_difference.get("speculative_value")),
            },
        }
        metadata["metadata_sha256"] = _sha(metadata)
        _write_create_only(root / "metadata.json", _canonical(metadata) + b"\n")
        complete = {
            "schema": "urn:qwen-r9700:full-assurance-counterexample-complete:v1",
            "metadata_sha256": metadata["metadata_sha256"],
            "serial_payload_sha256": metadata["serial"]["payload_sha256"],
            "speculative_payload_sha256": metadata["speculative"]["payload_sha256"],
        }
        complete["complete_sha256"] = _sha(complete)
        _write_create_only(root / "complete.json", _canonical(complete) + b"\n")
        _fsync_directory(root)
        _fsync_directory(parent)
    except BaseException:
        # The directory remains intentionally unpublished/incomplete.  Never
        # delete evidence after a partial create-only write.
        raise
    return complete


def load_counterexample_capsule(
    root: Path, comparison_value: object
) -> tuple[dict[str, Any], dict[str, bytes]]:
    if (
        not root.is_absolute()
        or root != root.resolve(strict=False)
        or not root.is_dir()
        or root.is_symlink()
    ):
        raise InstrumentationError("counterexample root is unsafe or absent")
    before = _require_owned_private_directory(root, "counterexample root")
    expected_names = {"metadata.json", "serial.bin", "speculative.bin", "complete.json"}
    if {path.name for path in root.iterdir()} != expected_names:
        raise InstrumentationError("counterexample capsule contains unknown artifacts")
    payloads = {
        name: _read_stable_private_file(root / name, f"counterexample artifact {name}")
        for name in expected_names
    }
    after = _require_owned_private_directory(root, "counterexample root")
    if _directory_identity(before) != _directory_identity(after):
        raise InstrumentationError("counterexample capsule changed during authentication")
    try:
        metadata = json.loads(payloads["metadata.json"])
        complete = json.loads(payloads["complete.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InstrumentationError(f"counterexample metadata is invalid: {error}") from error
    if not isinstance(metadata, dict) or metadata.get("schema") != COUNTEREXAMPLE_SCHEMA:
        raise InstrumentationError("counterexample metadata schema differs")
    unsigned_metadata = dict(metadata)
    metadata_sha256 = unsigned_metadata.pop("metadata_sha256", None)
    if metadata_sha256 != _sha(unsigned_metadata):
        raise InstrumentationError("counterexample metadata self-hash differs")
    if not isinstance(comparison_value, dict):
        raise InstrumentationError("counterexample comparison is not an object")
    comparison = _json_clone(comparison_value)
    comparison_sha256 = comparison.get("comparison_sha256")
    first_difference = comparison.get("first_difference")
    if (
        metadata.get("comparison_sha256") != comparison_sha256
        or metadata.get("first_difference") != first_difference
        or metadata.get("first_difference_sha256") != _sha(first_difference)
    ):
        raise InstrumentationError("counterexample does not bind the exact first difference")
    for arm, filename in (("serial", "serial.bin"), ("speculative", "speculative.bin")):
        descriptor = metadata.get(arm)
        if not isinstance(descriptor, dict):
            raise InstrumentationError(f"counterexample {arm} descriptor is absent")
        base_descriptor = {
            key: descriptor.get(key) for key in COUNTEREXAMPLE_DESCRIPTOR_KEYS
        }
        _normalize_counterexample_descriptor(base_descriptor, arm)
        payload = payloads[filename]
        expected_semantic_value = (
            first_difference.get(f"{arm}_value") if isinstance(first_difference, dict) else None
        )
        if (
            descriptor.get("payload_bytes") != len(payload)
            or descriptor.get("payload_sha256") != hashlib.sha256(payload).hexdigest()
            or descriptor.get("semantic_value_sha256") != _sha(expected_semantic_value)
        ):
            raise InstrumentationError(f"counterexample {arm} payload binding differs")
    if not isinstance(complete, dict) or complete.get("schema") != (
        "urn:qwen-r9700:full-assurance-counterexample-complete:v1"
    ):
        raise InstrumentationError("counterexample completion marker is invalid")
    signed_complete = dict(complete)
    complete_sha256 = signed_complete.pop("complete_sha256", None)
    if complete_sha256 != _sha(signed_complete):
        raise InstrumentationError("counterexample completion self-hash differs")
    if (
        complete.get("metadata_sha256") != metadata_sha256
        or complete.get("serial_payload_sha256")
        != hashlib.sha256(payloads["serial.bin"]).hexdigest()
        or complete.get("speculative_payload_sha256")
        != hashlib.sha256(payloads["speculative.bin"]).hexdigest()
    ):
        raise InstrumentationError("counterexample completion bindings differ")
    return metadata, {
        "serial": payloads["serial.bin"],
        "speculative": payloads["speculative.bin"],
    }


def qualify_trace_pair(
    serial_root: Path,
    speculative_root: Path,
    *,
    counterexample_root: Path | None = None,
) -> dict[str, Any]:
    """Promotion-grade pair gate: every detected divergence must retain raw values."""

    serial_header, serial_events, _ = load_complete_trace(serial_root)
    speculative_header, speculative_events, _ = load_complete_trace(speculative_root)
    comparison = compare_complete_traces(
        serial_header,
        serial_events,
        speculative_header,
        speculative_events,
    )
    capsule_sha256 = None
    if comparison["first_difference"] is not None:
        if counterexample_root is None:
            raise InstrumentationError(
                "first semantic divergence has no authenticated physical counterexample capsule"
            )
        _metadata, _payloads = load_counterexample_capsule(counterexample_root, comparison)
        complete = json.loads(
            _read_stable_private_file(
                counterexample_root / "complete.json", "counterexample completion marker"
            )
        )
        capsule_sha256 = complete["complete_sha256"]
    elif counterexample_root is not None:
        raise InstrumentationError(
            "equal traces must not carry a misleading counterexample capsule"
        )
    result = {
        "schema": PAIR_QUALIFICATION_SCHEMA,
        "classification": "bounded_qualification",
        "universal_equivalence_proven": False,
        "comparison_sha256": comparison["comparison_sha256"],
        "counterexample_complete_sha256": capsule_sha256,
        "passed": comparison["passed"],
    }
    result["qualification_sha256"] = _sha(result)
    return result


FULL_ASSURANCE_ENABLE_ENV = "QWEN_FULL_ASSURANCE_FRAGMENT"
FULL_ASSURANCE_ROOT_ENV = "QWEN_FULL_ASSURANCE_FRAGMENT_ROOT"
FULL_ASSURANCE_HEADER_ENV = "QWEN_FULL_ASSURANCE_HEADER"
FULL_ASSURANCE_SCOPES_ENV = "QWEN_FULL_ASSURANCE_SCOPES"
FULL_ASSURANCE_FRAGMENT_ID_ENV = "QWEN_FULL_ASSURANCE_FRAGMENT_ID"
FULL_ASSURANCE_PRODUCER_KIND_ENV = "QWEN_FULL_ASSURANCE_PRODUCER_KIND"
FULL_ASSURANCE_PRODUCER_SHA_ENV = "QWEN_FULL_ASSURANCE_PRODUCER_SHA256"
FULL_ASSURANCE_ENVIRONMENT = frozenset(
    {
        FULL_ASSURANCE_ROOT_ENV,
        FULL_ASSURANCE_HEADER_ENV,
        FULL_ASSURANCE_SCOPES_ENV,
        FULL_ASSURANCE_FRAGMENT_ID_ENV,
        FULL_ASSURANCE_PRODUCER_KIND_ENV,
        FULL_ASSURANCE_PRODUCER_SHA_ENV,
    }
)
_RUNTIME_LOCK = threading.RLock()
_RUNTIME_WRITER: FullAssuranceFragmentWriter | None = None
_RUNTIME_SCOPES: frozenset[tuple[Any, ...]] = frozenset()
_RUNTIME_DEVICE_SYNCHRONIZE_COUNT = 0
_RUNTIME_FINALIZED = False
_RUNTIME_INSTALLED_SITES: tuple[str, ...] | None = None


def _load_runtime_json(path_value: str, label: str) -> object:
    path = Path(path_value)
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise InstrumentationError(f"{label} path must be normalized and absolute")
    try:
        return json.loads(_read_stable_private_file(path, label))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InstrumentationError(f"{label} is invalid JSON: {error}") from error


def start_runtime_fragment_from_environment() -> bool:
    """Start the authenticated EngineCore-side recorder for one declared fragment."""

    global _RUNTIME_WRITER, _RUNTIME_SCOPES, _RUNTIME_FINALIZED, _RUNTIME_INSTALLED_SITES
    enabled = os.environ.get(FULL_ASSURANCE_ENABLE_ENV, "0")
    if enabled not in {"0", "1"}:
        raise InstrumentationError(f"{FULL_ASSURANCE_ENABLE_ENV} must be 0 or 1")
    supplied = {name for name in FULL_ASSURANCE_ENVIRONMENT if os.environ.get(name)}
    if enabled == "0":
        if supplied:
            raise InstrumentationError(
                "full-assurance auxiliary environment requires explicit fragment enablement"
            )
        return False
    if supplied != FULL_ASSURANCE_ENVIRONMENT:
        missing = sorted(FULL_ASSURANCE_ENVIRONMENT - supplied)
        raise InstrumentationError(f"full-assurance fragment environment is incomplete: {missing}")
    with _RUNTIME_LOCK:
        if _RUNTIME_WRITER is not None:
            return True
        producer_sha256 = _require_digest(
            os.environ[FULL_ASSURANCE_PRODUCER_SHA_ENV], "runtime producer SHA-256"
        )
        module_payload = _read_stable_private_file(
            Path(__file__).resolve(strict=True), "full-assurance runtime module"
        )
        if hashlib.sha256(module_payload).hexdigest() != producer_sha256:
            raise InstrumentationError(
                "full-assurance runtime module differs from its producer pin"
            )
        full_header = _load_runtime_json(
            os.environ[FULL_ASSURANCE_HEADER_ENV], "full-assurance runtime header"
        )
        scopes_value = _load_runtime_json(
            os.environ[FULL_ASSURANCE_SCOPES_ENV], "full-assurance runtime scopes"
        )
        if not isinstance(scopes_value, dict) or set(scopes_value) != {
            "schema",
            "campaign_header_sha256",
            "scopes",
            "scopes_sha256",
        }:
            raise InstrumentationError("full-assurance runtime scope document is malformed")
        header = normalize_header(full_header)
        if (
            scopes_value["schema"] != RUNTIME_SCOPES_SCHEMA
            or scopes_value["campaign_header_sha256"] != _sha(header)
        ):
            raise InstrumentationError("full-assurance runtime scope identity differs")
        raw_scopes = scopes_value["scopes"]
        if not isinstance(raw_scopes, list):
            raise InstrumentationError("full-assurance runtime scopes are not a list")
        scopes = tuple(sorted((_normalize_scope(scope) for scope in raw_scopes), key=repr))
        if scopes_value["scopes_sha256"] != _sha(scopes):
            raise InstrumentationError("full-assurance runtime scope digest differs")
        root = Path(os.environ[FULL_ASSURANCE_ROOT_ENV])
        _RUNTIME_WRITER = FullAssuranceFragmentWriter(
            root,
            header,
            fragment_id=os.environ[FULL_ASSURANCE_FRAGMENT_ID_ENV],
            scopes=scopes,
            producer_kind=os.environ[FULL_ASSURANCE_PRODUCER_KIND_ENV],
            producer_sha256=producer_sha256,
            sync_interval=1,
        )
        _RUNTIME_SCOPES = frozenset(scopes)
        _RUNTIME_FINALIZED = False
        _RUNTIME_INSTALLED_SITES = None
        atexit.register(finalize_runtime_fragment)
        return True


def mark_runtime_instrumentation_ready(installed_sites_value: object) -> tuple[str, ...]:
    """Authenticate the exact hook inventory before the first debug request may run."""

    global _RUNTIME_INSTALLED_SITES
    with _RUNTIME_LOCK:
        writer = _RUNTIME_WRITER
        if writer is None:
            raise InstrumentationError("full-assurance runtime recorder is not active")
        if _RUNTIME_INSTALLED_SITES is not None:
            raise InstrumentationError("full-assurance runtime hook inventory was already sealed")
        if not isinstance(installed_sites_value, (list, tuple, set, frozenset)):
            raise InstrumentationError("full-assurance runtime hook inventory is malformed")
        installed_sites = tuple(sorted(installed_sites_value))
        if any(not isinstance(site, str) or not site for site in installed_sites):
            raise InstrumentationError("full-assurance runtime hook inventory is malformed")
        expected = expected_instrumentation_sites(
            writer.header["producer_kind"],
            _RUNTIME_SCOPES,
            arm=writer.header["full_header"]["arm"],
        )
        if installed_sites != tuple(sorted(expected)):
            raise InstrumentationError(
                "full-assurance runtime hook inventory is incomplete; "
                f"expected {sorted(expected)}, observed {list(installed_sites)}"
            )
        _RUNTIME_INSTALLED_SITES = installed_sites
        return installed_sites


def runtime_instrumentation_ready() -> bool:
    with _RUNTIME_LOCK:
        return _RUNTIME_WRITER is not None and _RUNTIME_INSTALLED_SITES is not None


def runtime_scope_enabled(
    *,
    unit: str,
    phase: str,
    position: int | None,
    layer_index: int | None,
    row: int | None,
    commit_count: int | None = None,
) -> bool:
    scope = (unit, position, layer_index, row, phase, commit_count)
    with _RUNTIME_LOCK:
        return scope in _RUNTIME_SCOPES


def runtime_campaign_header() -> dict[str, Any] | None:
    """Return the authenticated active header without exposing writer internals."""

    with _RUNTIME_LOCK:
        if _RUNTIME_WRITER is None:
            return None
        return _json_clone(_RUNTIME_WRITER.header)


def runtime_producer_kind() -> str | None:
    """Return the authenticated producer class for the active runtime fragment."""

    with _RUNTIME_LOCK:
        if _RUNTIME_WRITER is None:
            return None
        return str(_RUNTIME_WRITER.header["producer_kind"])


def runtime_declared_positions() -> tuple[int, ...]:
    """Return only positions owned by this fragment, not the whole campaign."""

    with _RUNTIME_LOCK:
        return tuple(
            sorted(
                {
                    int(scope[1])
                    for scope in _RUNTIME_SCOPES
                    if scope[1] is not None
                }
            )
        )


def record_runtime_probe(
    *,
    unit: str,
    phase: str,
    position: int | None,
    layer_index: int | None,
    row: int | None,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Synchronize the device and append one exact declared runtime probe."""

    global _RUNTIME_DEVICE_SYNCHRONIZE_COUNT
    with _RUNTIME_LOCK:
        writer = _RUNTIME_WRITER
        if writer is None:
            raise InstrumentationError("full-assurance runtime recorder is not active")
        if _RUNTIME_INSTALLED_SITES is None:
            raise InstrumentationError("full-assurance runtime hooks are not authenticated")
        commit_count = evidence.get("commit_count")
        scope = (unit, position, layer_index, row, phase, commit_count)
        if scope not in _RUNTIME_SCOPES:
            raise InstrumentationError(f"runtime attempted an undeclared probe: {scope}")
        try:
            import torch

            torch.cuda.synchronize()
        except Exception as error:
            writer.abort()
            raise InstrumentationError(
                f"device synchronization failed before {unit} evidence: {error}"
            ) from error
        _RUNTIME_DEVICE_SYNCHRONIZE_COUNT += 1
        return writer.append(
            unit=unit,
            phase=phase,
            position=position,
            layer_index=layer_index,
            row=row,
            evidence=evidence,
        )


def finalize_runtime_fragment() -> dict[str, Any] | None:
    """Finalize only a complete runtime fragment; incomplete exits stay unpublished."""

    global _RUNTIME_FINALIZED
    with _RUNTIME_LOCK:
        if _RUNTIME_WRITER is None or _RUNTIME_FINALIZED:
            return None
        if _RUNTIME_INSTALLED_SITES is None:
            _RUNTIME_WRITER.abort()
            raise InstrumentationError(
                "full-assurance runtime exited before its hooks were authenticated"
            )
        _RUNTIME_FINALIZED = True
        return _RUNTIME_WRITER.finalize(
            device_synchronize_count=_RUNTIME_DEVICE_SYNCHRONIZE_COUNT,
            installed_instrumentation_sites=_RUNTIME_INSTALLED_SITES,
        )


# QWEN_ASSURANCE_ONLY_END: full-assurance-instrumentation
