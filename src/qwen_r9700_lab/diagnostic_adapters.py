"""Versioned bindings around existing instrumentation, not replacement probes.

These adapters normalize the common decoder boundaries of the older W4A16
diagnostic and the current Radiance capture. Their native, richer comparators
remain available. Importing this module never installs an inference hook.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from qwen_r9700_lab.diagnostic_contract import (
    Boundary,
    DiagnosticError,
    Observation,
    digest,
    integer,
    require_sha,
)

COMMON_DECODER_FIELDS = (
    ("decoder.input.hidden", "input_hidden_sha256", "input_hidden"),
    ("decoder.output.hidden", "output_hidden_sha256", "output.0"),
    ("decoder.output.residual", "output_residual_sha256", "output.1"),
)

# The inventory is an implementation map, not a qualification certificate.
# Every row names existing reusable code and the native binding still required.
INSTRUMENTATION = (
    (
        "artifact_identity",
        "src/qwen_r9700_lab/manifests.py",
        "shared",
        "Collect the actual loaded binaries and runtime in the new environment.",
    ),
    (
        "assurance_release_pair",
        "src/qwen_r9700_lab/coding_turbo_artifacts.py",
        "legacy_binding",
        "Update semantic source manifests and release erasure rules for the new backend.",
    ),
    (
        "authenticated_trace",
        "src/qwen_r9700_lab/assurance_instrumentation.py",
        "legacy_binding",
        "Keep the old Quest96 coverage contract; provide a new backend coverage contract.",
    ),
    (
        "layer_first_difference",
        "src/qwen_r9700_lab/assurance_layer_diagnosis.py",
        "legacy_binding",
        "Map native module names and logical token positions.",
    ),
    (
        "tensor_layout_capsule",
        "experiments/m8-layer-diagnostic/layer_diagnostic.py",
        "shared_helpers",
        "Reuse tensor helpers without installing its old import hooks.",
    ),
    (
        "forced_qk_replay",
        "experiments/dflash-lossless-assurance/replay_qk_capsule.py",
        "legacy_binding",
        "Bind the Q/K/RoPE operator and its tensor layout.",
    ),
    (
        "transaction_state",
        "experiments/full-state-diagnostic/runtime_state_capture.py",
        "legacy_binding",
        "Export all native recurrent, convolution, KV and pending-token state.",
    ),
    (
        "full_state_compare",
        "src/qwen_r9700_lab/full_state_diagnostic_compare.py",
        "legacy_binding",
        "Replace fixed 48/16/5 layer lists through a versioned state adapter.",
    ),
    (
        "snapshot_equivalence",
        "src/qwen_r9700_lab/assurance_snapshot_producer.py",
        "legacy_binding",
        "Bind snapshot save/load and all semantically required state.",
    ),
    (
        "cache_lifecycle",
        "experiments/full-state-diagnostic/kv_lifecycle_capture.py",
        "legacy_binding",
        "Bind scheduler, worker and allocator methods; verify ownership locally.",
    ),
    (
        "transaction_faults",
        "src/qwen_r9700_lab/assurance_transaction_producer.py",
        "legacy_binding",
        "Bind each accept, reject, cancel and publication boundary.",
    ),
    (
        "provider_tools",
        "src/qwen_r9700_lab/assurance_provider_producer.py",
        "legacy_binding",
        "Use the installed tokenizer, chat template, tool parser and actual tool schemas.",
    ),
    (
        "radiance_decoder_capture",
        "experiments/radiance-public/capture_radiance_layers.py",
        "radiance_binding",
        "Requalify decoder source hashes and native module outputs.",
    ),
    (
        "radiance_tensor_compare",
        "experiments/radiance-public/compare_radiance_layers.py",
        "shared_comparator",
        "Map boundary names; preserve private capsules and numerical metrics.",
    ),
    (
        "native_dispatch",
        "experiments/radiance-public/r4d_dispatch_audit.py",
        "radiance_binding",
        "Enumerate the actual native exports and attest every inference process.",
    ),
    (
        "sampling_distribution",
        "experiments/radiance-public/probe_dflash_sampling_rng.py",
        "radiance_binding",
        "Bind target/proposal/rejection sampling and independently test distributions.",
    ),
    (
        "gdn_numerics",
        "experiments/radiance-public/probe_gdn_numerics.py",
        "radiance_binding",
        "Reuse the serial recurrence oracle; bind native GDN shape and dtype contracts.",
    ),
    (
        "attention_numerics",
        "experiments/radiance-public/probe_r4d_attention_numerics.py",
        "radiance_binding",
        "Bind dense or explicitly accepted sparse attention and page/scale layouts.",
    ),
    (
        "linear_numerics",
        "experiments/radiance-public/probe_mxfp4_numerics.py",
        "radiance_binding",
        "Bind exact dequantization, activation rounding and supported shapes.",
    ),
    (
        "normalization_rope",
        "experiments/radiance-public/probe_norm_rope_numerics.py",
        "radiance_binding",
        "Bind native normalization, RoPE and strides to the independent reference.",
    ),
    (
        "loop_replay",
        "experiments/radiance-public/diagnose_loop_replay.py",
        "radiance_binding",
        "Keep private input hashes, censoring, installed parser and runtime attestation.",
    ),
    (
        "forced_token_driver",
        "src/qwen_r9700_lab/diagnostic_contract.py",
        "shared",
        "Implement reset/step/close; consume the supplied token and expose complete boundaries.",
    ),
    (
        "independent_quantized_reference",
        "src/qwen_r9700_lab/conformance_model.py",
        "shared",
        "Declare checkpoint, quantizers, finite arithmetic and a supported architecture.",
    ),
    (
        "logical_tensor_replay",
        "src/qwen_r9700_lab/conformance_replay.py",
        "shared",
        "Keep forced inputs, initial prefill, complete state and causal observation order.",
    ),
    (
        "durable_checked_authority",
        "src/qwen_r9700_lab/conformance_session.py",
        "shared",
        "Submit independent reference and tentative candidate state before publication.",
    ),
    (
        "radiance_v2_native_capture",
        "src/qwen_r9700_lab/conformance_radiance.py",
        "radiance_binding",
        "Experimental source-bound adapter; GPU state extraction remains unqualified.",
    ),
    (
        "runtime_mapped_artifacts",
        "src/qwen_r9700_lab/conformance_artifacts.py",
        "shared",
        "Add device code objects and compiler dispatch evidence; mapped files are insufficient.",
    ),
)

ADAPTERS = {
    "radiance-v2-conformance-20260914": {
        "attention": "dense",
        "implemented": ["cpu_logical_layout_tests", "source_binding_check"],
        "experimental_source": "src/qwen_r9700_lab/conformance_radiance.py",
        "unsupported": {
            "native_full_state": "Written but not yet GPU-qualified; explicit arming required.",
            "portable_forced_token_full_state": "Native M1/D7 capture has CPU coverage only.",
            "native_snapshot_checks": "Connector restore/cancellation still needs GPU coverage.",
            "formal_control_plane_proof": "Implementation-bound proofs cover small pure helpers.",
        },
    },
    "w4a16-quest96-v1": {
        "attention": "quest96",
        "common_decoder_schema": "qwen-r9700.m1-m8-layer-boundary.v1",
        "normalizer": "legacy_decoder_observations",
        "implemented": [
            "common_decoder_hashes",
            "native_layer_detail",
            "native_full_state",
            "native_quest_pages",
            "native_transaction_faults",
            "native_snapshot_checks",
        ],
        "unsupported": {"portable_forced_token_full_state": "Native step adapter not yet ported."},
    },
    "radiance-0.28-qwen3-next-v1": {
        "attention": "dense",
        "common_decoder_schema": "qwen-radiance-layer-capsule-v1",
        "normalizer": "radiance_decoder_observations",
        "implemented": [
            "common_decoder_hashes",
            "native_layer_detail",
            "private_tensor_capsules",
            "native_dispatch_audit",
            "operator_reference_probes",
            "sampling_probe",
        ],
        "unsupported": {
            "portable_forced_token_full_state": "Native step/state adapter not yet ported.",
            "native_full_state": "Decoder capture lacks persistent KV/GDN/conv state.",
            "native_transaction_faults": "Old transaction hooks do not bind this scheduler.",
            "formal_control_plane_proof": "No checked model-to-implementation refinement.",
        },
    },
}


def inventory(project_root: Path) -> dict[str, Any]:
    from qwen_r9700_lab.manifests import sha256_file

    rows = []
    for name, relative, portability, adaptation in INSTRUMENTATION:
        source = project_root / relative
        rows.append(
            {
                "id": name,
                "source": relative,
                "portability": portability,
                "required_adaptation": adaptation,
                "exists": source.is_file(),
                "source_sha256": sha256_file(source) if source.is_file() else None,
            }
        )
    return {
        "schema": "urn:qwen:diagnostic-inventory:v1",
        "components": rows,
        "adapters": ADAPTERS,
        "missing_source_files": sum(not r["exists"] for r in rows),
        "inventory_sha256": digest(rows),
        "qualification_inherited": False,
    }


def require_capability(adapter_id: str, capability: str) -> None:
    adapter = ADAPTERS.get(adapter_id)
    if adapter is None:
        raise DiagnosticError("unknown backend adapter; an explicit binding is required")
    if capability not in adapter["implemented"]:
        raise DiagnosticError("backend adapter does not implement the requested observation")


def common_decoder_boundaries(layer_count: int) -> list[Boundary]:
    return [
        Boundary(layer, name)
        for layer in range(integer(layer_count, minimum=1))
        for name, _, _ in COMMON_DECODER_FIELDS
    ]


def legacy_decoder_observations(
    records: Iterable[Mapping[str, Any]],
    *,
    position: int,
    pass_index: int,
    layer_count: int,
) -> list[Observation]:
    """Adapt preserved old-format boundaries without reinstalling old kernels.

    The caller authenticates the containing stream against its execution
    manifest. No finite-value test existed in these rows, so it remains unknown.
    """
    position, pass_index = integer(position), integer(pass_index)
    selected = {}
    for row in records:
        if row.get("schema") != ADAPTERS["w4a16-quest96-v1"]["common_decoder_schema"]:
            continue
        if row.get("position") != position or row.get("pass_index") != pass_index:
            continue
        layer = integer(row["layer_index"])
        if layer in selected:
            raise DiagnosticError("legacy decoder pass contains duplicate layers")
        selected[layer] = row
    if set(selected) != set(range(integer(layer_count, minimum=1))):
        raise DiagnosticError("legacy decoder pass does not cover every declared layer")
    return [
        Observation(position, Boundary(layer, name), require_sha(selected[layer].get(field)))
        for layer in range(layer_count)
        for name, field, _ in COMMON_DECODER_FIELDS
    ]


def radiance_decoder_observations(
    records: Mapping[str, Mapping[str, Any]],
    *,
    position: int,
    layer_count: int,
) -> list[Observation]:
    """Adapt already authenticated Radiance capsule records.

    Capsule loading, raw tensor hash validation and rich numerical comparison
    stay in compare_radiance_layers.py. This common view intentionally covers
    only input/output hashes; it cannot establish equality of internal state.
    """
    position = integer(position)
    observations = []
    for layer in range(integer(layer_count, minimum=1)):
        for name, _, stage in COMMON_DECODER_FIELDS:
            row = records.get(f"{layer}:{stage}:{position}")
            if row is None:
                raise DiagnosticError("Radiance capsule is missing a declared decoder boundary")
            observations.append(
                Observation(
                    position,
                    Boundary(layer, name),
                    require_sha(row.get("sha256")),
                    integer(row.get("nonfinite")),
                )
            )
    return observations
