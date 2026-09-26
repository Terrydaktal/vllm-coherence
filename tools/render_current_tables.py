#!/usr/bin/env python3
"""Render current-only README evidence from the committed aggregate measurements."""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path

from compute_stage26_residual import compute as audit_stage26

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- COHERENCE_CURRENT_RESULTS -->"
END = "<!-- /COHERENCE_CURRENT_RESULTS -->"
REPOSITORY = "https://github.com/Terrydaktal/vllm-coherence"
# Keep captured checkout identities intact; link to the same tree after rewording.
REWORDED_COMMITS = {
    "7386d835a32e4be3549e87cfff9bb9998e0a91c3": "9bb795d2e612c76087d16932841131edf4834d5e",
}
def render_head_candidate_benchmark():
    current = json.loads((ROOT / "benchmarks/results/coherence-current.json").read_text())
    reference = current.get("head_candidate_result", "benchmarks/results/head-candidate-depth-20260924.json")
    evidence = json.loads((ROOT / reference).read_text())
    requests = evidence["requests"]["stages"]
    workloads = "; ".join(f"{row['label']}: {row['input_tokens']:,} input and {row['output_tokens']:,} output tokens ({row['finish_reason']})" for row in requests)
    rows = evidence["modes"]["full"]["rows"]
    samples = evidence["timings_m8"]["full"]["samples"]
    lines = [
        "## Global-512 target-head", "",
        "Global-512 is the serving default. This paired comparison used identical hidden inputs for every head method from two natural completions starting with the retained 60K Pi prefix: " + workloads + ". Sampling was temperature 1, top-p 0.95 and top-k 40.", "",
        "| Target path | Median M8 head time | Same top-1 token | Complete reference top-20 retained | Complete reference top-40 retained |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, label in (("global256", "Global INT2 top-256 + BF16 rerank"), ("global512", "Global INT2 top-512 + BF16 rerank (default)"), ("full", "Full BF16 reference")):
        result = evidence["modes"][mode]
        cells = []
        for key in ("argmax_equal", "top20_complete_including_ties", "top40_complete_including_ties"):
            count = result[key]
            percent = "100%" if count == rows else f"{100 * count / rows:.4f}%"
            cells.append(f"{count:,}/{rows:,} ({percent})")
        timing = evidence["timings_m8"][mode]["median_ms"]
        lines.append(f"| {label} | {timing:.3f} ms | " + " | ".join(cells) + " |")
    misses = [rows - evidence["modes"][mode]["top40_complete_including_ties"] for mode in ("global256", "global512")]
    added = evidence["timings_m8"]["global512"]["median_ms"] - evidence["timings_m8"]["global256"]["median_ms"]
    lines += ["", f"The comparison covers **{rows:,} prediction rows**, including prefill and rejected speculative rows, not {rows:,} generated tokens. Timing uses {samples // 5} eight-row hidden inputs with five randomized-order repetitions: {samples} measurements per method. These isolated head timings include native dispatch gaps and exclude comparison/reporting; they do not measure whole-round time or tok/s.", "",
        f"Incomplete top-40 retention occurred in {misses[0]:,} rows with Global-256 and {misses[1]:,} with Global-512; the added median head time was {added:.3f} ms. Retention includes cutoff ties and does not establish score equality, ordering or identical sampling probabilities. Both shortlists remain approximate; the full BF16 head is the reference for this comparison, not an independently proved model. The drafter is unchanged.", "",
        f"[Methodology and limits](docs/HEAD_CANDIDATE_DEPTH.md) · [Numeric results]({reference}) · [Earlier Global-256 study](docs/VERIFY_HEAD_GLOBAL_TOPK.md)."]
    return lines


def measurement_marker(data):
    commit = data.get("measurement_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("current measurements must identify a full commit hash")
    return f"[`{commit[:7]}`]({REPOSITORY}/commit/{commit})"


def provenance_marker(data, stage):
    provenance = data.get("stage_provenance")
    if not isinstance(provenance, dict):
        raise TypeError("compiled stages must identify their code provenance")
    entry = provenance.get(stage)
    if not isinstance(entry, dict):
        raise TypeError(f"missing code provenance for compiled stage: {stage}")
    commit = entry.get("commit")
    change = entry.get("change")
    document = entry.get("document")
    if document:
        if document != data.get("current_qualification_document") or not change:
            raise ValueError(f"invalid qualification document for compiled stage: {stage}")
        return f"[Qualified speed changes]({document}): {change}"
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or not isinstance(change, str)
        or not change.strip()
    ):
        raise ValueError(f"invalid code provenance for compiled stage: {stage}")
    return f"[`{commit[:7]}`]({REPOSITORY}/commit/{commit}): {change}"


def commit_marker(commit):
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("evidence must identify a full commit hash")
    commit = REWORDED_COMMITS.get(commit, commit)
    return f"[`{commit[:7]}`]({REPOSITORY}/commit/{commit})"


def evidence_with_commits(data, stage, evidence):
    configured = data.get("evidence_commits", {}).get(stage)
    commits = configured or [data["measurement_commit"]]
    markers = [commit_marker(commit) for commit in commits]
    parts = evidence.split("; ")
    if len(commits) > 1:
        if len(parts) != len(commits):
            raise ValueError(
                f"evidence commit count does not match semicolon-separated evidence: {stage}"
            )
        return "; ".join(f"{part} · {marker}" for part, marker in zip(parts, markers))
    return f"{evidence} · {markers[0]}"


def current_stage_profile(data):
    """Return the old grouped 26-row view populated by the new profile.

    The README historically exposed a grouped 26-row table.  The replacement
    capture uses the original compiled profiler's 25 disjoint scopes, so this
    adapter maps those scopes back into the historical rows without reviving
    the old eager/full-BF16 timings.  Fused scopes are charged to one row and
    called out in the row note; rows with no corresponding profiler scope are
    rendered with an explicit inclusion or measurement-status label rather than
    being presented as measured zero milliseconds.
    """
    evidence = ROOT / data.get("matched_stage_profile", "benchmarks/results/compiled-global256-stage-profile-1200.json")
    base = data.get("stage_profile_2k")
    if not evidence.exists() or not isinstance(base, dict):
        return base
    raw = json.loads(evidence.read_text())
    matched = raw.get("status") == "matched_estimate"

    # This is the commit that packages the retained capture and the renderer;
    # the artifact itself records the older source checkout used to take the
    # measurement.  The table link therefore identifies the reproducible
    # evidence bundle rather than pretending the capture ran after a later
    # source change.
    evidence_commit = "b8d681001cc726089c387eeddfc7c78e2e74ac3c"
    grouped = copy.deepcopy(base)
    if matched:
        evidence_commit = data.get(
            "current_qualification_commit", raw["stage26_execution"]["measurement_commit"]
        )
    grouped["measurement_commit"] = evidence_commit
    grouped["matched"] = matched
    grouped["measurement_date"] = raw.get("measurement_date", "2026-09-23")
    grouped["optimized_manifest_sha256"] = raw.get("binding", {}).get("optimized_manifest_sha256")
    head = raw.get("target_head", "global256")
    depth = head.removeprefix("global")
    grouped["target_head"] = head
    grouped["scope"] = (
        "Archived diagnostic compiled Global-256 stage attribution, with BF16 attention arithmetic. "
        "The 0K / 60K / 200K cells sum **GPU kernel activity durations**, excluding CPU annotations, "
        "Python hooks and trace-export time. They do not certify that profiling left kernel execution "
        "unchanged: tracing can alter clocks, dispatch and overlap. Requested rounds were "
        "2,183 / 1,191 / 1,191; 2,062 / 1,132 / 1,133 complete cycles were retained. "
        "The capture remains stage attribution only: first-use Triton JIT compilation was observed "
        "in its preserved slow window, so its wall timings are not steady-state measurements. "
        "The old profiler result is retired from the production metric set. "
        "**Row 26 is not qualified as real runtime gaps.** The old control retained forced-replay "
        "copies, scalar reads and sample writes; its host-step boundary and accepted-token schedule "
        "were not independently matched to the profiled GPU cycles. Subtracting those captures "
        "cannot remove their observer effects. [Timing audit](benchmarks/results/"
        "stage-timing-audit-20260923.json); [measurement contract](docs/STAGE_TIMING.md)."
    )
    grouped["context_order"] = list(raw["context_order"])
    if matched:
        counts = " / ".join(str(raw["contexts"][c]["included_rounds"]) for c in raw["context_order"])
        outputs = " / ".join(f"{raw['contexts'][c]['generated_tokens_per_arm']:,}" for c in raw["context_order"])
        deltas = " / ".join(f"{raw['observer_comparison']['contexts'][c]['mean_delta_ms']:.3f}" for c in raw["context_order"])
        grouped["scope"] = (
            f"Measured on {grouped['measurement_date']} using the compiled, optimized Global-{depth} serving backend"
            + (" with the pinned-RAM huge-page promotion repair" if raw["binding"].get("host_runtime") else "")
            + "; sampling is temperature 1.0, top-p 0.95 and top-k 40. Context labels are starting prefixes: "
            "0K, the private 60K Pi fixture, and the public synthetic 200K fixture. Each context ran "
            "a natural warmup followed by clean control, trace, and clean control; each arm generated "
            + outputs + " tokens respectively. Generated-token hashes and accepted-token schedules matched. "
            "The stage means retain " + counts + " complete M8 cycles (0K / 60K / 200K), and controls "
            "use exactly those same decode indices. Trace setup/export boundaries and incomplete or inconsistent trace "
            "inventories are excluded by structure, never by duration; complete native round logs retain "
            "all rounds and stalls. GPU activity timestamps supply the stage times; CPU annotations, "
            "Python hooks and export time are excluded. No per-stage event probes or forced-token replay "
            "are used. The measured tracing slowdown was " + deltas + " ms per retained round; it is "
            "reported separately and is **not charged to row 26**. Row 26 is the clean control mean minus "
            "the union of GPU activity intervals. This remains an estimate: tracing can indirectly affect "
            "clocks and scheduling. Overlap is counted once in the total. "
            + ("The header links the qualification bundled with these repairs; its captures retain their original source identities. "
               if data.get("current_qualification_document") else
               "The header identifies the base revision and marks the uncommitted source changes. "
               if data.get("uncommitted_qualification") else
               "The header identifies the commit containing the measured backend repairs and captured results. ")
            + "The capture retains its original checkout and source identities. "
            f"[Capture and source identities]({data['matched_stage_profile']}) "
            f"· [controls]({data['matched_stage_control']}) · [method and uncertainty](docs/STAGE_TIMING.md)."
        )
        if data.get("current_qualification_document"):
            grouped["scope"] += (
                "\n\nThis candidate rerun includes the full-graph cache-preparation repair and "
                "measurement adapters in the [qualified speed refresh]("
                + data["current_qualification_document"] + "). Captured checkout identities "
                "remain unchanged after the history rewrite; the installed source and binary "
                "hashes bind the measured implementation. "
                + (
                    "The lifecycle-qualified worker is now selected by the live Pi backend; "
                    f"its [deployment receipt]({data['current_deployment']}) preserves the "
                    "installed source identities."
                    if data.get("current_deployment") else
                    "The experimental worker is separate from the normal Pi deployment."
                )
            )
        elif data.get("uncommitted_qualification"):
            grouped["scope"] += (
                "\n\nThis candidate rerun uses the banked speed changes plus the uncommitted "
                "full-graph cache-preparation repair and measurement adapters. The commit link "
                "identifies the base revision; the capture binds the actual installed source "
                "and binary hashes. It is not yet a deployed production release."
            )
        long_context = raw["observer_comparison"]["contexts"].get("200K", {})
        later = long_context.get("control_after_round_ms", [])
        pauses = [value for value in later if value > 100]
        if pauses:
            low, high = long_context["remainder_before_after_range_ms"]
            grouped["scope"] += (
                f"\n\nThe later 200K clean control retained {len(pauses)} intervals above 100 ms, "
                f"the longest **{max(pauses):,.3f} ms**, although its median was {statistics.median(later):.3f} ms. "
                f"These pauses raise the displayed mean and row 26. The before/after remainder range is "
                f"{low:.3f}–{high:.3f} ms; their cause remains unresolved. This is repeat variability, "
                "not a confidence interval or proof that earlier tracing had no indirect effect."
            )
    context_tokens = {"0K": 0, "60K": 60_000, "200K": 200_000}
    grouped["stage26"] = copy.deepcopy(base["stage26"])
    control_ref = (data["matched_stage_control"] if matched else
                   grouped.get("stage26_benchmark", {}).get("artifact"))
    if control_ref:
        control_path = ROOT / control_ref
        if not control_path.exists():
            raise ValueError(f"stage-26 control artifact is missing: {control_ref}")
        grouped["stage26_benchmark"] = json.loads(control_path.read_text())
        control = grouped["stage26_benchmark"]
        grouped["stage26_audit"] = audit_stage26(raw, control)
        if control.get("status") == "complete" and not matched:
            control_contexts = control.get("contexts", {})
            counts = [
                len(control_contexts[context].get("round_samples", []))
                for context in grouped["context_order"]
            ]
            means = [
                control_contexts[context]["full_uninstrumented_round_ms"]
                for context in grouped["context_order"]
            ]
            grouped["scope"] += (
                " The historical harness control retained "
                + " / ".join(f"{count:,}" for count in counts)
                + " complete unprofiled rounds (0K / 60K / 200K) and its host-step "
                + "mean was "
                + " / ".join(f"{mean:.3f}" for mean in means)
                + " ms before the 25-stage subtraction."
            )
    # Preserve the row taxonomy, but never promote archival arithmetic to a
    # qualified production-gap measurement merely because a control completed.
    grouped["stage26"]["label"] = "Estimated runtime overhead"
    grouped["stage26"]["definition"] = (
        "Requires a matched natural-serving control and GPU activity union, with observer "
        "effect assessed separately. Historical subtraction is not a measurement of real gaps."
    )
    if matched:
        grouped["stage26"]["definition"] = "Matched clean-control mean minus GPU occupied time; CPU tracing/export cost is excluded. Indirect observer effects remain an uncertainty."

    # Keep the exact historical row structure from cbbf495.  The current
    # profiler emits these as separate disjoint scopes; the renderer changes
    # only the timing column, not the published row taxonomy.
    historical_rows = (
        (
            "Drafter",
            "Drafter",
            "Suggests up to seven tokens for the target model to check.",
        ),
        (
            "Embedding + first input normalization + FP8 production",
            "Embedding + first input normalization",
            "Looks up token vectors, normalizes them and creates FP8 inputs in one kernel, preserving intermediate rounding.",
        ),
        (
            "Layer input residual/normalization + FP8 production",
            "Layer input residual/normalization",
            "Adds the residual, normalizes the result and creates FP8 inputs in one kernel, preserving intermediate rounding.",
        ),
        (
            "GDN input projection",
            "GDN input projection",
            "Projects hidden vectors into the inputs and gates for the recurrent layer.",
        ),
        (
            "GDN layout/copies and buffer initialization",
            "GDN layout/copies and buffer initialization",
            "Arranges inputs and clears temporary buffers throughout each GDN layer.",
        ),
        (
            "GDN convolution",
            "GDN convolution",
            "Updates recent-token history using the corrected multiply/add order.",
        ),
        (
            "GDN recurrence and gates",
            "GDN recurrence and gates",
            "Updates recurrent memory in token order. Faster prefill retains the existing nine-slot state layout.",
        ),
        (
            "GDN output gated normalization + FP8 production",
            "GDN output gated normalization",
            "Normalizes and gates GDN output, then produces FP8 inputs in one kernel; intermediate BF16 rounding is preserved.",
        ),
        (
            "GDN output projection",
            "GDN output projection",
            "Maps the recurrent-layer result back to the model's hidden-vector width.",
        ),
        (
            "Attention input projection",
            "Attention input projection",
            "Projects hidden vectors into the inputs for attention.",
        ),
        (
            "Attention Q/K normalization, RoPE and layout",
            "Attention Q/K normalization, RoPE and layout",
            "Normalizes queries and keys and applies positional rotation (RoPE), retaining the corrected rounding.",
        ),
        (
            "Attention KV write",
            "Attention KV write",
            "Stores new keys and values in the cache for reuse by later tokens.",
        ),
        (
            "Attention decode",
            "Attention decode",
            "Attends to the current and earlier tokens using corrected arithmetic and shared cache loads.",
        ),
        (
            "Attention split-KV merge",
            "Attention split-KV merge",
            "Combines attention results from cache partitions in the corrected arithmetic order.",
        ),
        (
            "Attention output gating",
            "Attention output gating",
            "Applies learned gates to the attention output.",
        ),
        (
            "Attention output activation FP8 quantization",
            "Attention output activation FP8 quantization",
            "Converts the attention output to FP8 for its output projection.",
        ),
        (
            "Attention output projection",
            "Attention output projection",
            "Maps the attention result back to the model's hidden-vector width.",
        ),
        (
            "Post-attention/GDN residual/normalization + FP8 production",
            "Post-attention/GDN residual/normalization",
            "Adds the layer result to the residual, normalizes it and creates FP8 inputs for the MLP in one kernel.",
        ),
        (
            "MLP gate/up projection",
            "MLP gate/up projection",
            "Computes both MLP input projections. Wider decode dispatch avoids the slower prefill kernel for qualified shapes.",
        ),
        (
            "MLP SiLU and gating",
            "MLP SiLU and gating",
            "Applies SiLU and combines the two MLP branches, retaining intermediate BF16 rounding.",
        ),
        (
            "MLP down input FP8 quantization",
            "MLP down input FP8 quantization",
            "Converts MLP activations to FP8 for the down projection.",
        ),
        (
            "MLP down projection",
            "MLP down projection",
            "Projects the MLP result back to the model's hidden-vector width.",
        ),
        (
            "Final normalization/layout",
            "Final normalization/layout",
            "Normalizes the final hidden vector before vocabulary scoring.",
        ),
        (
            "Global-256 target head",
            f"Target head ({head})",
            f"Scores the vocabulary with INT2, selects {depth} candidates and rescores them with BF16 weights. Selection remains approximate.",
        ),
        (
            "Other GPU bookkeeping",
            "Other GPU bookkeeping",
            "Runs sampling and cache/state update kernels outside the named model stages.",
        ),
    )
    grouped["stage_order"] = [row[0] for row in historical_rows]
    grouped["stage_labels"] = {row[0]: row[0].replace("Global-256", f"Global-{depth}") for row in historical_rows}
    grouped["evidence_sources"] = {row[0]: row[0] for row in historical_rows}
    grouped["historical_raw_scopes"] = {row[0]: row[1] for row in historical_rows}
    grouped["stage_notes"] = {row[0]: row[2] for row in historical_rows}
    grouped["contexts"] = {}
    for context in raw["context_order"]:
        source = raw["contexts"][context]
        source_stages = source["stages_ms"]
        stages = {
            label: source_stages[raw_scope]
            for label, raw_scope, _note in historical_rows
        }
        grouped["contexts"][context] = {
            "context_tokens": context_tokens[context],
            "output_positions": source["profile_rounds"],
            "profile_steps": source["included_rounds"],
            "kernel_subtotal_ms": source["stage_sum_ms"],
            "profile_coverage": (
                f"{source['included_rounds']:,}/{source['profile_rounds']:,} "
                "complete retained cycles"
            ),
            "profile_exit_code": 0,
            "same_final_logits_all_passes": None,
            "stages": stages,
            "unattributed_kernel_ms": 0.0,
        }

    return grouped


def _render_stage_profile_table(data):
    """Render the archived 26-stage attribution as one 0K/60K/200K column.

    The older aggregate ``stages`` table is retained in the JSON for the
    compiled-trace/layer evidence below. This table is deliberately sourced
    from the current three-context compiled profiler and rendered as one
    slash-separated timing column. Stage 26 is deliberately sourced from a
    separate unprofiled-round control. Rendering fails if that control is not
    complete; a profiler-cycle remainder is never substituted for it.
    """
    profile = current_stage_profile(data)
    if not isinstance(profile, dict):
        raise ValueError("the current measurements do not contain the 2K stage profile")
    order = profile["stage_order"]
    context_order = profile["context_order"]
    contexts = profile["contexts"]
    commit = (
        f"[qualified speed refresh]({data['current_qualification_document']})"
        if data.get("current_qualification_document")
        else commit_marker(profile["measurement_commit"])
    )
    old_rows = {row["stage"]: row for row in data["stages"] if row["ms"] is not None}
    deployment = json.loads((ROOT / "benchmarks/results/eager-m1-normalization-deployment-20260924.json").read_text())
    confirmations_ref = data.get("current_stage_confirmations")
    confirmations = json.loads((ROOT / confirmations_ref).read_text()) if confirmations_ref else None
    if confirmations and (
        confirmations["status"] != "SAMPLE_CHECKED"
        or confirmations["optimized_manifest_sha256"] != deployment["optimized_manifest_sha256"]
    ):
        raise ValueError("current stage evidence is incomplete or belongs to another release")
    heading = ("Current GPU activity per retained compiled M8 cycle" if profile.get("matched") else
               "Archived diagnostic interval per retained compiled profile cycle")
    run_label = (f"{profile['measurement_date']}; run {commit}" if profile.get("matched") else
                 f"evidence run {commit}")
    if data.get("uncommitted_qualification"):
        run_label = f"{profile['measurement_date']}; base {commit} + recorded working-tree changes"
    lines = [
        f"| Stage | {heading} ({' / '.join(context_order)}; milliseconds unless explicitly marked; {run_label}) | Current M1->M8 correctness and eager->compiled correctness evidence | Last relevant code commit / change | What this stage does |",
        "| --- | ---: | --- | --- | --- |",
    ]
    timing_labels = {
        "input_preparation": "Not separately emitted",
        "rope": "Included in stage 11",
        "target_sampling": "Not separately emitted; see stage 25",
        "target_other": "Not separately emitted; see stage 25",
        "forced_replay_control": "Outside profiled GPU scopes",
    }
    for number, stage in enumerate(order, start=1):
        values = [contexts[key]["stages"][stage] for key in context_order]
        timing = timing_labels.get(
            stage, " / ".join(f"{value:.3f}" for value in values)
        )
        evidence_source = profile.get("evidence_sources", {}).get(stage)
        if evidence_source in old_rows:
            source_row = old_rows[evidence_source]
            evidence = evidence_with_commits(data, evidence_source, source_row["evidence"])
            provenance = provenance_marker(data, evidence_source)
        else:
            evidence = f"Timing-only diagnostic profile; no isolated correctness claim · {commit}"
            provenance = ("2026-09-23: native serving timing capture" if profile.get("matched") else
                          f"{commit}: 2K forced-output stage profile; no backend code change")
        if profile.get("matched") and stage == "Attention decode":
            evidence = "Paired natural outputs match at 0K/60K/200K · [attention repair evidence](benchmarks/results/attention-page-boundary-20260923.json); earlier isolated alignment evidence remains in the report."
            provenance = "[September 23 attention-page repair](experiments/radiance-public/build_stock_m1_attention_shared.py): reuse the context traversal when M8 queries cross a 16-token attention-page boundary; source/binary hashes are in the capture."
        if profile.get("optimized_manifest_sha256") == deployment["optimized_manifest_sha256"]:
            norm_rows = {
                "Embedding + first input normalization + FP8 production",
                "Layer input residual/normalization + FP8 production",
                "Post-attention/GDN residual/normalization + FP8 production",
            }
            if stage in norm_rows or stage == "GDN output gated normalization + FP8 production":
                sites = 128 if stage in norm_rows else 48
                evidence = f"Released row-invariant normalization gate: {sites} sites × 1,000 rows; finite operator checks, not a new full-model alignment run · [24 September evidence](docs/eager-m1-contract-qualification.md)."
                provenance = provenance_marker(data, stage)
            elif stage in {"Attention decode", "Attention split-KV merge"}:
                provenance = provenance_marker(data, stage)
        if profile.get("target_head") == "global512" and stage == "Global-256 target head":
            head_ref = data["head_candidate_result"]
            measured_head = json.loads((ROOT / head_ref).read_text())["modes"]["global512"]
            evidence = (
                f"Same top-1: {measured_head['argmax_equal']:,}/{measured_head['rows']:,}; "
                f"complete reference top-20 retained: {measured_head['top20_complete_including_ties']:,}/{measured_head['rows']:,}. "
                f"Includes M1/M8; not an M1-versus-M8 ordering test · [current head study]({head_ref})."
            )
            previous_head = commit_marker(data["stage_provenance"][stage]["commit"])
            current_head = commit_marker(data["target_head_change_commit"])
            provenance = current_head + ": increase target shortlist to 512; drafter unchanged. Extends the Global-256 method from " + previous_head + "."
        if confirmations:
            paired_stage = (
                "Attention decode and split-KV merge"
                if stage in {"Attention decode", "Attention split-KV merge"} else stage
            )
            measured = confirmations["stages"].get(paired_stage)
            if measured:
                fixed, modes = measured["fixed"], measured["final_modes"]
                count = fixed["positions"]
                evidence = (
                    f"M1/M8: {fixed['top20_set_exact']}/{count}; {fixed['top20_order_exact']}/{count}. "
                    f"Eager/compiled M8: {modes['top20_set_exact']}/{count}; {modes['top20_order_exact']}/{count} "
                    f"· [current 320-token run]({confirmations_ref})."
                )
                if paired_stage != stage:
                    evidence += " Decode and merge checked together."
            elif stage == "GDN layout/copies and buffer initialization":
                evidence = (
                    f"Authoritative cache/state restored in all {confirmations['cache_restored_groups']} "
                    f"eight-token groups; state/output corruption controls detected "
                    f"· [current replay]({confirmations_ref}). No separate layout top-20 attribution."
                )
            elif stage == "Drafter":
                evidence = "Separate proposal model; target M1/M8 comparisons do not independently qualify it."
            elif stage == "Other GPU bookkeeping":
                evidence = "No isolated top-20 operator claim; sampling/state controls have separate evidence."
        lines.append(
            f"| **{number}. {profile['stage_labels'][stage]}** | {timing} | {evidence} | {provenance} | {profile['stage_notes'][stage]} |"
        )
        detail_rows = {
            "Layer input residual/normalization + FP8 production": (
                (
                    "GDN input activation FP8 quantization",
                    "Uses the FP8 output produced by the same input-normalization kernel.",
                ),
                (
                    "Attention input activation FP8 quantization",
                    "Uses the FP8 output produced by the same input-normalization kernel.",
                ),
            ),
            "GDN output gated normalization + FP8 production": (
                (
                    "GDN output activation FP8 quantization",
                    "Uses the FP8 output produced by the same gated-normalization kernel.",
                ),
            ),
            "Post-attention/GDN residual/normalization + FP8 production": (
                (
                    "MLP gate/up input FP8 quantization",
                    "Uses the FP8 output produced by the same post-normalization kernel.",
                ),
            ),
        }.get(stage, ())
        if detail_rows:
            for detail_label, detail_note in detail_rows:
                lines.append(
                    f"| ↳ {detail_label} | Included in **stage {number}** | "
                    f"Exact fused FP8 bytes/scales; see stage {number} and its evidence scope | "
                    f"{provenance} | {detail_note} |"
                )
    audit = profile.get("stage26_audit")
    if not isinstance(audit, dict):
        raise ValueError("stage-26 evidence has not been audited")
    if audit["status"] == "diagnostic_only":
        lines.extend([
            f"| **26. {profile['stage26']['label']}** | **Not qualified** | "
            f"[Timing audit](benchmarks/results/stage-timing-audit-20260923.json) | "
            f"{commit}: archived control; not a runtime-gap measurement | {profile['stage26']['definition']} |",
            "| **Total reconstructed round (stages 1–26)** | **Not qualified** | "
            "No valid production-gap value to add to these archived stages | "
            "— | Current natural-serving full-round means are in Benchmarks below. |",
        ])
    else:
        # Even a matched difference is an estimate. No per-stage correction
        # can be inferred merely from an aggregate profiler-on/off delta.
        estimates = [audit["contexts"][key]["union_corrected_difference_ms"] for key in context_order]
        timing = " / ".join(f"{value:.3f}" for value in estimates)
        totals = " / ".join(f"{audit['contexts'][key]['full_uninstrumented_round_ms']:.3f}" for key in context_order)
        lines.extend([
            f"| **26. {profile['stage26']['label']}** | {timing} (estimate) | "
            f"[Matched control minus GPU activity union]({data.get('matched_stage_residual', 'benchmarks/results/matched-stage-residual-20260923.json')}) | "
            f"{commit}: record matched unprofiled controls and overlap-corrected residuals | Indirect observer effects are not proved zero. |",
            f"| **Total reconstructed round (stages 1–26)** | **{totals}** | "
            "GPU activity union plus the estimated remainder | "
            "— | Overlapping stages are counted once in the total. |",
        ])
    return lines


def render_stage_profile_table(data):
    """Keep independently scoped eager-M1 evidence when refreshing timings."""
    import re

    evidence = json.loads((ROOT / "benchmarks/results/eager-m1-readme-evidence.json").read_text())
    lines = _render_stage_profile_table(data)
    enriched = []
    observed = set()
    for line in lines:
        if not line.startswith("|"):
            enriched.append(line)
            continue
        cells = line.strip("|").strip().split(" | ")
        if cells[0] == "Stage":
            value = "Current Eager M1 correctness evidence"
        elif cells[0].startswith("---"):
            value = "---"
        else:
            label = re.sub(r"^\*\*\d+\. ", "", cells[0]).removesuffix("**")
            key = "Global-256 target head" if label == "Global-512 target head" else label
            value = evidence["rows"][key]
            observed.add(key)
        cells.insert(3, value)
        enriched.append("| " + " | ".join(cells) + " |")
    if observed != set(evidence["rows"]):
        raise ValueError("eager-M1 evidence does not cover every compiled stage row")
    return evidence["paragraphs"].splitlines() + [""] + enriched


CHAINED_RESULTS = ROOT / "benchmarks/results/pi-coding-json-compaction.json"


def workload_target_head(report):
    """Bind shared-workload labels to the measured runtime, never a default."""
    if report.get("suite_capture_id"):
        current = json.loads((ROOT / "benchmarks/results/coherence-current.json").read_text())
        profile = json.loads((ROOT / current["matched_stage_profile"]).read_text())
        shared = profile.get("binding", {}).get("shared_suite", {})
        for report_key, binding_key in (
            ("suite_capture_id", "capture_id"),
            ("contract_sha256", "contract_sha256"),
            ("runtime_manifest_sha256", "runtime_manifest_sha256"),
        ):
            if not report.get(report_key) or report[report_key] != shared.get(binding_key):
                raise ValueError(f"shared workload/profile identity mismatch: {report_key}")
        head = profile.get("target_head")
        declared = report.get("runtime", {}).get("target_head")
        if declared and declared != head:
            raise ValueError("shared workload target head disagrees with measured profile")
    else:
        head = report.get("runtime", {}).get("target_head")
    if head not in ("global256", "global512", "full"):
        raise ValueError("workload target head is not recorded")
    return head


def render_chained_workload_results(report=None):
    if report is None:
        report = json.loads(CHAINED_RESULTS.read_text())
    sampling = report["sampling"]
    run_date = datetime.fromtimestamp(report["started_at"], tz=UTC).date().isoformat()
    stages = {row["stage"]: row for row in report["stages"]}
    compaction = stages.get("compaction", {})
    checkpoint = compaction.get("checkpoint_validation", {})
    compaction_temperature = compaction.get("sampling", {}).get("temperature", 0.3)
    head = workload_target_head(report)
    runner = (
        "[benchmark_pi_coding_contexts.py --suite]"
        "(experiments/radiance-public/benchmark_pi_coding_contexts.py)"
        if report.get("suite_capture_id") else
        "[benchmark_pi_coding_json_compaction.py]"
        "(experiments/radiance-public/benchmark_pi_coding_json_compaction.py)"
    )
    lines = [
        "## Benchmarks",
        "",
        (
            "This benchmark uses the retained 60,000-input-token Pi prefix and the compiled "
            f"Coherence backend with {head}, the attention-page-boundary repair and the "
            "pinned-RAM huge-page promotion repair. Five "
            "requests are chained in one context: code, prose about code measurement, JSON, "
            "thinking/prose and checkpoint generation. Code and JSON disable thinking; both "
            "prose requests enable it. All requests stop naturally. "
            f"Generation uses temperature {sampling['temperature']:g}, top-p {sampling['top_p']:g}, "
            f"top-k {sampling['top_k']} and seed {sampling['seed']}; compaction uses temperature "
            f"{compaction_temperature:g}. The checkpoint request forces a snapshot-tail flush. "
            "Private fixture text was not decoded or inspected. Public results contain "
            "aggregates and hashes, not chat text or token arrays. "
            + ("The shared suite retains sealed continuation tokens privately for resumability. "
               if report.get("suite_capture_id") else "")
            + f"Runner: {runner}."
        ),
        "",
        "| Stage | Thinking | Prompt tokens | Generated tokens | Classified output | First data | Mean round | Post-first | Peak 3s | Acceptance |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = [("coding", "Coding task"), ("prose_code", "Prose about code measurement"),
              ("json", "JSON task"), ("thinking", "Thinking/prose task"),
              ("compaction", "Compaction checkpoint")]
    for key, label in labels:
        row = stages.get(key)
        if row is None:
            lines.append(f"| {label} | — | — | — | pending run | — | — | — | — | — |")
            continue
        phase = row["phase_blocks"]
        if key == "coding":
            classified = (f"{phase['file_edit_code_tokens']:,} code; {phase['prose_tokens']:,} prose; "
                          f"{phase['reasoning_tokens']:,} separately observed reasoning")
        elif key == "json":
            classified = f"{phase['file_edit_json_tokens']:,} JSON; " + ("valid JSON" if phase['json_valid'] else "invalid JSON")
        elif key == "compaction":
            classified = f"{phase['checkpoint_tokens']:,} checkpoint tokens; "
            classified += "completion marker valid" if checkpoint.get("marker_valid") else "completion marker invalid"
            classified += "; required headings valid" if checkpoint.get("headings_valid") else "; required headings missing or duplicated"
        else:
            classified = f"{phase['prose_tokens']:,} prose; {phase['reasoning_tokens']:,} separately observed reasoning"
        values = [label, "on" if row["thinking_enabled"] else "off",
                  f"{row['prompt_tokens']:,}", f"{row['generated_tokens']:,}", classified,
                  _coding_context_value(row.get("first_token_seconds"), " s"),
                  _coding_context_value(row.get("mean_generation_round_ms"), " ms"),
                  _coding_context_value(row.get("post_first_tokens_per_second"), " tok/s"),
                  _coding_context_value(row.get("peak_3s_tokens_per_second"), " tok/s"),
                  f"{100 * row['acceptance_rate']:.2f}%" if row.get("acceptance_rate") is not None else "—"]
        lines.append("| " + " | ".join(values) + " |")
    cache = ", ".join(f"{label}: {stages[key]['cached_prompt_tokens']:,}/{stages[key]['prompt_tokens']:,}"
                      for key, label in labels if key in stages)
    lines.extend(["", f"Cached/total prompt tokens: {cache}.", ""])
    unclassified = [label for key, label in labels if key in stages and not stages[key].get("phase_token_counts_cover_output")]
    missing_reasoning = [label for key, label in labels if key in stages and stages[key]["thinking_enabled"] and not stages[key].get("reasoning_channel_observed")]
    lines.append(
        "The phase counters retokenize classified text, so their totals can differ from the backend's emitted-token count. "
        + ("`phase_token_counts_cover_output=false` for " + ", ".join(unclassified) + ". " if unclassified else "All streams report complete phase-token coverage. ")
        + ("No separate reasoning channel was exposed for " + ", ".join(missing_reasoning) + "; those streams are reported as prose, not relabelled as reasoning. " if missing_reasoning else "")
        + "The compaction row measures checkpoint generation with a requested cache flush, not a full Pi transcript commit or old-snapshot retirement. "
        + ("The checkpoint format passed its heading and completion checks. " if checkpoint.get("passed") else "The checkpoint format failed validation and must not be treated as a committed compaction. ")
        + "`peak_3s_tokens_per_second` is the maximum completed three-second sliding-window rate after first data, never a single-frame burst."
    )
    lines.extend([
        "",
        (
            f"{run_date} rerun status: `{report['status']}`. [Numeric results and release identity]"
            f"(benchmarks/results/pi-coding-json-compaction.json). These results use top-k {sampling['top_k']}, "
            "while the older September 20 table used top-k 20, so the output "
            "and acceptance changes are not a controlled before/after comparison."
        ),
        "",
    ])
    return lines


def render_coding_json_compaction_benchmark():
    return [
        *render_chained_workload_results(),
        *render_coding_context_benchmark(),
        "",
        *render_round_capture_summary(),
        "",
        *render_round_histogram(),
        "",
        *render_known_remaining_symptoms(),
    ]


def render_known_remaining_symptoms():
    current = json.loads((ROOT / "benchmarks/results/coherence-current.json").read_text())
    head = json.loads((ROOT / current["head_candidate_result"]).read_text())["modes"]["global512"]
    profile = json.loads((ROOT / current["matched_stage_profile"]).read_text())
    control = profile["observer_comparison"]["contexts"]["200K"]
    later = control["control_after_round_ms"]
    pauses = [value for value in later if value > 100]
    low, high = control["remainder_before_after_range_ms"]
    if current.get("full_graph_repair"):
        shared = bool(profile["binding"].get("shared_suite"))
        workload = json.loads(CODING_CONTEXT_RESULTS.read_text())
        feed_pauses = []
        control_pauses = []
        for context in profile["context_order"]:
            records = workload["contexts"][context]["round_capture"]["records"]
            slow = [row for row in records if (row.get("round_ms") or 0) > 100]
            if slow:
                feed_pauses.append(
                    f"{context}: {len(slow)} interval(s), longest {max(row['round_ms'] for row in slow):,.3f} ms"
                )
            observed_controls = profile["observer_comparison"]["contexts"][context]
            for arm in ("before", "after"):
                slow = [ms for ms in observed_controls[f"control_{arm}_round_ms"] if ms > 100]
                if slow:
                    control_pauses.append(
                        f"{context} {arm}: {len(slow)} interval(s), longest {max(slow):,.3f} ms"
                    )
        pause_note = (
            "**Isolated pauses remain.** The complete coding/histogram feeds retain "
            + ("; ".join(feed_pauses) if feed_pauses else "no intervals above 100 ms")
            + ". The retained paired controls separately contain "
            + ("; ".join(control_pauses) if control_pauses else "no intervals above 100 ms")
            + ". Their cause has not been localized. The histogram uses the predetermined after-control; "
            "row 26 uses both controls at the trace's retained indices. Warmup and trace-boundary "
            "exclusions are structural, not duration-based; every coding round remains in the public feed."
        )
        if pauses:
            observed = (
                f"The later 200K clean control retained {len(pauses)} of {len(later):,} "
                f"matched intervals above 100 ms, reaching {max(pauses):,.3f} ms; "
                f"its median was {statistics.median(later):.3f} ms. "
            )
        else:
            observed = (
                f"None of the {len(later):,} matched intervals in the later 200K clean control "
                f"exceeded 100 ms; its median was {statistics.median(later):.3f} ms. "
            )
        return [
            "### Known remaining symptoms and likely causes", "",
            ("**The missing full-graph preparation hook is repaired.** The first integrated speed run "
            "took 59.44 ms at 60K because FULL replay bypassed the existing post-cache-preparation "
            "synchronization and recovery hook. After installing it on both execution branches, "
            "the matched comparison measured 43.38 ms control versus 41.14 ms optimized, with "
            "identical output and acceptance. Fresh-chat reproductions also recovered. "
            "This fixes a specific integration defect; it does not prove that all HIP/ROCr queue "
            "stalls are impossible. "
            f"[Reproduction and repair evidence]({current['full_graph_repair']})."), "",
            observed
            + ("The histogram includes this same predetermined control's entire round feed; the stage "
               "residual uses its subset of exactly matched M8 cycles plus the preceding control. "
               if shared else "The context histogram and matched controls are separate captures. ")
            + f"The paired residual estimates span {low:.3f}–{high:.3f} ms. "
            "Profiling overhead is reported separately. Indirect changes in clocks, execution duration "
            "and scheduling remain measurement uncertainty, so the residual is an estimate rather "
            "than a proved exact sum of runtime gaps. "
            f"[Current controls]({current['matched_stage_control']}).", "",
            ("**The completions stream still lacks a separate reasoning channel.** Thinking-enabled "
            "requests are reported as observed prose; their reasoning throughput cannot be isolated "
            "from this stream. The checkpoint benchmark measures generation and requested tail "
            "flushing, not a complete Pi transcript commit. Its format-validation result is reported "
            "in the table. Natural completions shorter than the coding target remain explicit "
            "validation failures, even when their timing and round captures are complete."), "",
            pause_note, "",
            ("**Global-512 candidate selection remains approximate.** The current head study matched "
            f"reference top-1 in {head['argmax_equal']:,}/{head['rows']:,} rows and retained the complete "
            f"reference top-20 in {head['top20_complete_including_ties']:,}/{head['rows']:,}. "
            "Full-head M1/M8 and eager/compiled agreement does not certify shortlist completeness "
            "or eliminate model-generated loops. "
            f"[Current head evidence]({current['head_candidate_result']})."),
        ]
    return [
        "### Known remaining symptoms and likely causes",
        "",
        (
            "**Occasional long-context pauses remain.** The evidence comes from the separate "
            "compiled-stage timing experiment. Its later unprofiled 200K control, run after "
            f"the profiling arm, recorded {len(pauses)} of {len(later):,} retained intervals "
            "above 100 ms: " + ", ".join(f"{value:,.3f} ms" for value in pauses) + ". "
            f"Its median was {statistics.median(later):.3f} ms. The coding-context histogram "
            "above covers a different run; these control intervals are recorded separately. "
            f"The paired control residuals span {low:.3f}–{high:.3f} ms. "
            "This is an intermittent stall, not a sustained increase in every kernel's cost. "
            "Its cause is not localized: host scheduling, cache/dependency waits and HIP/ROCr "
            "queue state remain candidates. Correlated per-round host and GPU event records "
            "are needed to distinguish them. All measured intervals, including the pauses, "
            f"remain in the [current controls]({current['matched_stage_control']})."
        ),
        "",
        (
            "**Stage attribution has a separate trace limitation.** One 200K cycle recorded "
            "normalization after its consuming projection. That inconsistent timestamp order "
            "is excluded from per-stage attribution, while the original records are retained. "
            "It does not establish that the GPU executed the dependency incorrectly. "
            "Profiling can also affect clocks and scheduling indirectly; subtracting traced "
            "activity from the clean control is not proof of an exact, observer-free gap total. "
            "[Trace witness](benchmarks/results/trace-stage-order-witness-20260924.json) · "
            "[timing method](docs/STAGE_TIMING.md)."
        ),
        "",
        (
            "**The benchmark stream still lacks a separate reasoning channel.** "
            "Thinking-enabled requests exposed only prose, so the table cannot isolate their "
            "reasoning throughput. The unresolved boundary is the provider/parser metadata path; "
            "this observation alone does not show whether internal reasoning was absent. "
            "The latest checkpoint passed its section and completion-marker checks. "
            "That benchmark covers checkpoint generation and tail flushing; full Pi transcript "
            "commit, retirement, cancellation and concurrent-chat recovery need their own "
            "integration checks. [Current chained run](benchmarks/results/pi-coding-json-compaction.json)."
        ),
        "",
        (
            "**Global-512 candidate selection remains approximate.** The current study retained "
            f"the reference top-1 in {head['argmax_equal']:,}/{head['rows']:,} rows and the complete "
            f"top-20 in {head['top20_complete_including_ties']:,}/{head['rows']:,}. "
            "An excluded vocabulary token can still belong in the reference sampling support. "
            "The new exact M1/M8 and eager/compiled confirmations use the full BF16 comparison "
            "head; they do not certify shortlist completeness or eliminate model-generated loops. "
            f"[Current head evidence]({current['head_candidate_result']})."
        ),
    ]


def render_hugepage_comparison():
    data = json.loads((ROOT / "benchmarks/results/huge-page-promotion-20260923.json").read_text())
    before = next(row["before"] for row in data["intervention_comparisons"] if row["context"] == "200K")
    protected = data["repair"]["worker_status"]["host_page_policy_bytes"] / 2**30
    rerun = data["rerun"]["contexts"]
    rounds = sum(row["timed_rounds"] for row in rerun.values())
    spikes = sum(row["spikes_over100ms"] for row in rerun.values())
    tokens = sum(row["generated_tokens"] for row in rerun.values())
    diagnostic = data["stage_capture_diagnostics"]
    warmup_maxima = " / ".join(f"{diagnostic[c]['arms']['warmup']['max_ms']:,.0f}" for c in ("60K", "200K"))
    control_spikes = [spike for context in diagnostic.values()
                      for arm in ("control_before", "control_after")
                      for spike in context["arms"][arm]["spikes_over100ms"]]
    return [
        "**2026-09-23: the separate periodic huge-page stalls are repaired.** Kernel tracing "
        "caught `khugepaged` collapsing the registered host-memory arena and invoking "
        "`amdgpu_hmm_invalidate_hsa`. At 200K this produced 189–206 ms rounds about every "
        "10.24 seconds. Pinned chat-handover buffers now apply `MADV_NOHUGEPAGE` to their "
        "whole anonymous backing mappings, including unused allocator space, before any "
        f"cache transfer. The live worker reported {protected:g} GiB protected. This runs "
        "once during allocation; it adds no per-round scan or syscall and changes neither "
        "the global huge-page policy nor model arithmetic or snapshot contents.",
        "",
        f"The targeted 200K control had {len(before['spikes_over100ms'])} rounds over 100 ms "
        f"among {before['rounds']:,}; both repaired 200K replays had none, including a return "
        "from RAM. The September 23 rerun preserved all three coding output-token hashes "
        f"({tokens:,} tokens) and all five chained-task hashes. Across its {rounds:,} timed "
        f"coding rounds, {spikes} exceeded 100 ms; maximum rounds were "
        + " / ".join(f"{rerun[c]['max_ms']:.3f}" for c in ("0K", "60K", "200K"))
        + " ms. These are sampled intervention results, not a proof that every possible "
        "driver or scheduler pause is eliminated. First-use allocation/prefill costs were "
        "not a controlled comparison. [Diagnosis, mapping policy, RAM-return checks and "
        "rerun evidence](benchmarks/results/huge-page-promotion-20260923.json).",
        "",
        "The separate stage-capture restart/warmup sequence still recorded isolated "
        + warmup_maxima + " ms warmup pauses at 60K / 200K, and "
        + str(len(control_spikes)) + " clean-control round over 100 ms ("
        + ", ".join(f"{spike['ms']:.3f}" for spike in control_spikes) + " ms). "
        "These remain in the evidence. Their cause is unresolved; this repair does not "
        "claim to eliminate those isolated pauses or the initial pinned-allocation cost.",
    ]


def render_attention_boundary_comparison():
    path = ROOT / "benchmarks/results/attention-page-boundary-20260923.json"
    data = json.loads(path.read_text())
    lines = [
        "**2026-09-23: a repeating attention-page-boundary slowdown is diagnosed and repaired.** "
        "The shared M8 attention kernel used two independent GPU work groups whenever its eight "
        "verification queries crossed a 16-token KV page. Both groups reread the full context. "
        "This added about 3 ms per round at 60K and 10 ms at 200K. The repair shares one traversal "
        "while preserving each query's original split range, softmax accumulation and rounding. "
        "Results below use compiled Global-256 natural Pi coding completions with temperature 1, "
        "top-p 0.95, top-k 40 and seed 0. The event profiler is disabled; ordinary numeric round "
        "telemetry remains enabled. All timed rounds, including outliers, contribute to the means.",
        "",
        "| Starting history | Before, mean round ms | Fixed, mean round ms | Timed rounds per run | Output tokens per run | Same generated output |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for context in ("0K", "60K", "200K"):
        row = data["contexts"][context]
        before, after = row["before"], row["after"]
        lines.append(
            f"| {context} | {before['mean_ms']:.3f} | {after['mean_ms']:.3f} | "
            f"{after['timed_rounds']:,} | {row['generated_tokens']:,} | "
            f"{'Yes' if row['output_exact'] else 'No'} |"
        )
    lines.extend([
        "",
        "The 0K task starts at 203 prompt tokens; the longer tasks start at 60,208 and 200,208. "
        "Each run also retains its first, untimed prefill event. Across all three pairs, all "
        "16,632 generated tokens match. Separately, 264 native operator cases check 2,112 query "
        "rows against serial M1 with no differing output bytes, covering every page offset, "
        "split boundaries, FP8/BF16/padded-byte KV layouts, graph replay and corruption controls. "
        "These are sampled checks, not a universal arithmetic proof or full-vocabulary comparison.",
        "",
        "Fixed per-offset round medians span 42.511–42.594 ms at 60K and 50.350–50.418 ms at 200K. "
        "A few isolated spikes remain. The old 200K control also entered an additional persistent "
        "slow state near an adaptive recovery fence; that state was absent from the fixed run, "
        "but its separate queue mechanism is not independently proved. The repair establishes "
        "the cause of the repeating page-boundary mode, not that all possible scheduling or "
        "driver jitter has been eliminated. [Complete numeric comparison and histograms]"
        "(benchmarks/results/attention-page-boundary-20260923.json).",
        "",
        "The normal release worker was then restarted with the frozen repair and repeated the "
        "60K task: all 5,686 output tokens still matched, with 1,320 timed rounds, a 42.560 ms "
        "median and a 42.996 ms mean. All page-offset medians were within 42.528–42.591 ms. "
        "The mean includes the first verification round's 566.892 ms startup spike; it is not "
        "removed from the record or mistaken for a recurring latency mode. "
        "[Post-deployment verification](benchmarks/results/attention-page-boundary-deployment-20260923.json).",
    ])
    return lines


CODING_CONTEXT_RESULTS = ROOT / "benchmarks/results/pi-coding-contexts.json"


def _coding_context_value(value, suffix=""):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value:,}{suffix}" if isinstance(value, int) else str(value)


def render_coding_context_benchmark():
    """Render the same coding task at independent 0K/60K/200K prefixes."""

    report = {}
    if CODING_CONTEXT_RESULTS.is_file():
        try:
            candidate = json.loads(CODING_CONTEXT_RESULTS.read_text())
        except (OSError, json.JSONDecodeError):
            candidate = {}
        if isinstance(candidate, dict):
            report = candidate
    contexts = report.get("contexts") if isinstance(report.get("contexts"), dict) else {}
    rows = []
    for context in ("0K", "60K", "200K"):
        row = contexts.get(context)
        if not isinstance(row, dict):
            rows.append(f"| {context} | — | — | — | — | — | — | — | pending run |")
            continue
        minimum_met = row.get("minimum_output_met")
        validation = "pass" if minimum_met else "short natural stop"
        capture = row.get("round_capture")
        if isinstance(capture, dict):
            capture_status = capture.get("status", "unknown")
            capture_text = (
                f"{capture.get('record_count', 0):,} total; "
                f"{capture.get('speculative_round_count', '—')}/{capture.get('expected_rounds', '—')} "
                f"speculative; {capture.get('measured_round_count', 0):,} timed; {capture_status}"
            )
        else:
            capture_text = "pending rerun with complete round capture"
        rows.append(
            "| {context} | {prompt} | {generated} | {round_ms} | {post} | {peak} | {acceptance} | {capture} | {validation} |".format(
                context=context,
                prompt=_coding_context_value(row.get("prompt_tokens")),
                generated=_coding_context_value(row.get("generated_tokens")),
                round_ms=_coding_context_value(
                    row.get("mean_generation_round_ms"), " ms"
                ),
                post=_coding_context_value(
                    row.get("post_first_tokens_per_second"), " tok/s"
                ),
                peak=_coding_context_value(
                    row.get("peak_3s_tokens_per_second"), " tok/s"
                ),
                acceptance=(
                    f"{100 * row['acceptance_rate']:.2f}%"
                    if isinstance(row.get("acceptance_rate"), (int, float))
                    else "—"
                ),
                capture=capture_text,
                validation=validation,
            )
        )
    report_state = report.get("status") if report else "not_run"
    sampling = report.get("sampling", {})
    provenance = report.get("fixture_provenance") if isinstance(report, dict) else None
    provenance_line = None
    if isinstance(provenance, dict):
        provenance_line = (
            "Fixture provenance: "
            + "; ".join(
                f"{context} — {provenance[context]}"
                for context in ("0K", "60K", "200K")
                if context in provenance
            )
            + "."
        )
    return [
        (
            "This is the same natural-stop coding task run independently at empty, 60K and "
            "200K input context. Thinking is disabled, EOS remains enabled, and each arm uses "
            f"temperature {sampling.get('temperature', 1):g}, top-p {sampling.get('top_p', 0.95):g} "
            f"and top-k {sampling.get('top_k', 20)}. The non-empty arms use operator-supplied "
            "token-prefix fixtures; only their hashes are published. The three-second peak is "
            "the maximum completed sliding-window rate, not a single-frame burst. Runner: "
            "[benchmark_pi_coding_contexts.py](experiments/radiance-public/"
            "benchmark_pi_coding_contexts.py)."
        ),
        "",
        "| Context | Prompt tokens | Generated tokens | Mean round | Post-first | Peak 3s | Acceptance | Round events (total; speculative/expected; timed) | Validation |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        *rows,
        "",
        f"Report status: `{report_state}`. [Numeric results and every round](benchmarks/results/pi-coding-contexts.json). Each completed row stores every scheduler event under `contexts.<context>.round_capture.records`; a count mismatch is a validation failure. The target is {report.get('coding_min_tokens', 5000):,} output tokens, with shorter natural completions reported explicitly.",
        "",
        *( [provenance_line, ""] if provenance_line else [] ),
        (
            ("These results use " if report.get("suite_capture_id") else "The next refresh can use ")
            + "[the shared benchmark suite](docs/BENCHMARK_SUITE.md): "
            "`benchmark_pi_coding_contexts.py --suite` reuses each context's predetermined "
            "unprofiled control for its coding row and complete histogram, and continues the "
            "same 60K output through prose, JSON, thinking and compaction. It keeps both "
            "controls around each stage trace for the residual calculation. "
            + ("The 60K coding row above and the chained coding row are the same measured request."
               if report.get("suite_capture_id") else
               "Existing numbers above retain their original capture provenance.")
        ),
    ]


def render_round_capture_summary():
    """Show whether the context benchmark retained every scheduler event."""

    report = {}
    if CODING_CONTEXT_RESULTS.is_file():
        try:
            candidate = json.loads(CODING_CONTEXT_RESULTS.read_text())
        except (OSError, json.JSONDecodeError):
            candidate = {}
        if isinstance(candidate, dict):
            report = candidate
    contexts = report.get("contexts") if isinstance(report.get("contexts"), dict) else {}
    rows = []
    for context in ("0K", "60K", "200K"):
        row = contexts.get(context)
        capture = row.get("round_capture") if isinstance(row, dict) else None
        if not isinstance(capture, dict):
            rows.append(f"| {context} | — | — | — | — | — | — | pending rerun |")
            continue
        missing = ", ".join(str(value) for value in capture.get("missing_round_numbers", [])) or "none"
        rows.append(
            f"| {context} | {capture.get('record_count', 0):,} | "
            f"{capture.get('speculative_round_count', '—')} | "
            f"{capture.get('expected_rounds', '—')} | "
            f"{capture.get('measured_round_count', 0):,} | "
            f"{capture.get('unmeasured_round_count', 0):,} | {missing} | "
            f"{capture.get('status', 'unknown')} |"
        )
    return [
        "### Complete per-round capture",
        "",
        (
            "The context benchmark now retains every content-free scheduler event for each arm, "
            "including an unmeasured first event. The full numeric records are stored under "
            "`contexts.<context>.round_capture.records` in the result JSON; this table is a "
            "coverage check rather than another latency aggregate. A count mismatch is a "
            "validation failure. Expected rounds come from the speculative-round counter; "
            "the first prefill event is retained in logged events but excluded from that counter."
        ),
        "",
        "| Context | Logged events | Speculative rounds | Expected speculative rounds | Timed round values | Untimed events | Missing round numbers | Capture status |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        *rows,
        "",
    ]


def render_round_histogram():
    """Render the histogram produced by the context benchmark itself."""

    report = {}
    if CODING_CONTEXT_RESULTS.is_file():
        try:
            candidate = json.loads(CODING_CONTEXT_RESULTS.read_text())
        except (OSError, json.JSONDecodeError):
            candidate = {}
        if isinstance(candidate, dict):
            report = candidate
    contexts = report.get("contexts") if isinstance(report.get("contexts"), dict) else {}
    captures = {
        context: contexts.get(context, {}).get("round_capture")
        if isinstance(contexts.get(context), dict)
        else None
        for context in ("0K", "60K", "200K")
    }
    histograms = {
        context: capture.get("histogram")
        for context, capture in captures.items()
        if isinstance(capture, dict)
        and capture.get("status") == "captured"
        and isinstance(capture.get("histogram"), dict)
    }
    if len(histograms) == 3:
        context_order = ("0K", "60K", "200K")
        bins = histograms[context_order[0]].get("bins", [])
        def mean_median(context):
            mean = histograms[context].get("mean_ms")
            median = histograms[context].get("median_ms")
            if mean is None or median is None:
                return "— / — ms"
            return f"{mean:.2f} / {median:.2f} ms"

        lines = [
            "The histogram below is generated from the complete per-round records of the "
            "[coding-context benchmark](benchmarks/results/pi-coding-contexts.json). Every measured "
            "`round_ms` value appears in exactly one bin; untimed events are reported separately. "
            + ("The shared suite uses this same predetermined clean control for the coding row and "
             "histogram; the stage residual matches only structurally admitted M8 cycles from both "
             "controls, while this histogram retains every measured round."
             if report.get("suite_capture_id") else
             "The compiled-stage timing experiment has separate unprofiled control runs; their "
             "pauses are discussed below and are not part of this histogram."),
            "",
            "| Round time | 0K arm | 60K arm | 200K arm |",
            "|---|---:|---:|---:|",
        ]
        for index, first_bin in enumerate(bins):
            cells = []
            for context in context_order:
                row = histograms[context]["bins"][index]
                percentage = row.get("percentage")
                cells.append(
                    f"{row.get('count', 0):,}"
                    + (f" ({percentage:.1f}%)" if percentage is not None else "")
                )
            lines.append(f"| `{first_bin['label']}` | {' | '.join(cells)} |")
        lines.extend(
            [
                "| **Timed rounds** | "
                + " | ".join(
                    f"**{histograms[context].get('measured_round_count', 0):,}**"
                    for context in context_order
                )
                + " |",
                "| **Untimed events** | "
                + " | ".join(
                    f"**{histograms[context].get('unmeasured_round_count', 0):,}**"
                    for context in context_order
                )
                + " |",
                "| **Mean / median** | "
                + " | ".join(
                    f"**{mean_median(context)}**"
                    for context in context_order
                )
                + " |",
                "",
            ]
        )
        return lines

    return [
        (
            "The earlier retained round log gives this historical partial latency histogram. "
            "Its events did not contain a context-token field, so the columns used the matching "
            "0K, 60K and 200K fixture streams as proxies. The retained spans were approximately "
            "0–1.7K, 60–61.7K and 200–201.5K generated tokens, not complete 20K-wide bands. "
            "The current context benchmark now owns histogram generation and will replace this "
            "table once all three arms have a complete per-round capture."
        ),
        "",
        "| Round time | 0–20K* | 60–80K* | 200–220K* |",
        "|---|---:|---:|---:|",
        "| `<35` | 0 | 0 | 0 |",
        "| `35–37` | 0 | 0 | 0 |",
        "| `37–39` | 0 | 0 | 0 |",
        "| `39–40` | 0 | 0 | 0 |",
        "| `40–40.5` | 23 (8.9%) | 0 | 0 |",
        "| `40.5–41` | 75 (29.0%) | 0 | 0 |",
        "| `41–41.5` | 111 (42.9%) | 0 | 0 |",
        "| `41.5–42` | 22 (8.5%) | 0 | 0 |",
        "| `42–43` | 3 (1.2%) | 0 | 0 |",
        "| `43–45` | 0 | 0 | 0 |",
        "| `45–46` | 0 | 158 (50.0%) | 0 |",
        "| `46–47` | 1 (0.4%) | 3 (0.9%) | 0 |",
        "| `47–48` | 0 | 1 (0.3%) | 0 |",
        "| `48–48.5` | 0 | 114 (36.1%) | 0 |",
        "| `48.5–49` | 2 (0.8%) | 16 (5.1%) | 0 |",
        "| `49–51` | 20 (7.7%) | 2 (0.6%) | 0 |",
        "| `51–52.5` | 0 | 0 | 0 |",
        "| `52.5–53` | 0 | 0 | 98 (20.1%) |",
        "| `53–53.5` | 0 | 0 | 157 (32.2%) |",
        "| `53.5–54` | 0 | 1 (0.3%) | 6 (1.2%) |",
        "| `54–55` | 0 | 19 (6.0%) | 0 |",
        "| `55–56` | 0 | 0 | 1 (0.2%) |",
        "| `56–60` | 0 | 2 (0.6%) | 0 |",
        "| `60–62.5` | 0 | 0 | 0 |",
        "| `62.5–63` | 0 | 0 | 79 (16.2%) |",
        "| `63–63.5` | 0 | 0 | 103 (21.1%) |",
        "| `63.5–64` | 0 | 0 | 42 (8.6%) |",
        "| `64–65` | 0 | 0 | 1 (0.2%) |",
        "| `65–70` | 0 | 0 | 0 |",
        "| `70–100` | 1 (0.4%) | 0 | 0 |",
        "| `100–250` | 0 | 0 | 0 |",
        "| `250–500` | 1 (0.4%) | 0 | 0 |",
        "| `≥500` | 0 | 0 | 0 |",
        "| **Rounds** | **259** | **316** | **487** |",
        "| **Mean / median** | **43.52 / 41.12 ms** | **47.28 / 46.02 ms** | **57.72 / 53.16 ms** |",
        "",
    ]


def current_fine_detail(data):
    reference = data.get("matched_stage_profile")
    if not reference:
        return None
    capture = json.loads((ROOT / reference).read_text())
    context = capture["contexts"]["60K"]
    if "layers_ms" not in context:
        return None
    if set(context["layers_ms"]) != {str(i) for i in range(64)}:
        raise ValueError("current layer detail must cover all 64 layers")
    layers = []
    for i in range(64):
        values = context["layers_ms"][str(i)]
        kind = "Attention" if i % 4 == 3 else "GDN"
        row = {"layer": i, "type": kind, "total": sum(values.values()),
               "input": values[f"{kind} input projection"],
               "output": values[f"{kind} output projection"],
               "gate_up": values["MLP gate/up projection"],
               "down": values["MLP down projection"]}
        row["other"] = row["total"] - sum(row[k] for k in ("input", "output", "gate_up", "down"))
        layers.append(row)
    kernels = [{"stage": k["stage"], "kernel": k["kernel"], "calls": k["activity_records"],
                "ms": k["ms_per_round"]} for k in context["kernel_groups"]]
    kernels.sort(key=lambda row: (capture["stage_order"].index(row["stage"]), row["kernel"]))
    if not math.isclose(sum(k["ms"] for k in kernels), context["stage_sum_ms"], abs_tol=1e-8):
        raise ValueError("current granular detail does not reconcile to the stage table")
    return {"layers": layers, "kernels": kernels, "cycles": context["included_rounds"],
            "date": capture["measurement_date"]}


def render(data):
    timing = data["round_timing"]
    mean = timing["mean"]
    retained_cycles = len(timing["rounds"])
    detail = current_fine_detail(data)
    if detail:
        retained_cycles = detail["cycles"]
    commit = measurement_marker(data)
    stage_profile = current_stage_profile(data)
    serving_overhead = data["serving_overhead"]
    if serving_overhead["status"] != "estimated_from_separate_runs":
        raise ValueError(
            "Serving overhead must be labelled as an estimate from separate runs"
        )
    round_ms = serving_overhead["unprofiled_round_ms"]
    overhead_ms = round_ms - data["gpu_ms"]
    if (
        not math.isfinite(round_ms)
        or round_ms <= 0
        or overhead_ms < 0
        or serving_overhead["method"] != "unprofiled_round_ms - gpu_ms"
        or not math.isclose(
            serving_overhead["ms"], overhead_ms, rel_tol=0, abs_tol=1e-8
        )
    ):
        raise ValueError("Serving overhead estimate does not match its source timings")
    if not math.isclose(data["gpu_ms"], mean["kernel_sum_ms"], rel_tol=0, abs_tol=1e-8):
        raise ValueError("Stage and overhead timing must cover the same rounds")
    if not math.isclose(
        data["gpu_ms"] - mean["gpu_overlap_ms"] + mean["overhead_ms"],
        mean["elapsed_ms"],
        rel_tol=0,
        abs_tol=1e-8,
    ):
        raise ValueError("GPU work and overhead do not reconcile to elapsed time")
    lines = [
        START,
        "## Current numerical results",
        "",
        data["correctness_scope"],
        "",
        "| Prediction | Same token set | Same ordering | Mean shared tokens |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in data["predictions"]:
        n = row["rows"]
        k = row["k"]
        lines.append(
            f"| Top {k} | {row['set']:,} / {n:,} ({100 * row['set'] / n:.2f}%) | {row['order']:,} / {n:,} ({100 * row['order'] / n:.2f}%) | {row['overlap']:.4f} / {k} |"
        )
    lines += [
        "",
        data["correctness_limits"],
        "",
        "## Compiled backend stages",
        "",
        stage_profile["scope"] if stage_profile else data["timing_scope"],
        "",
        *render_stage_profile_table(data),
        "",
        (
            "The table restores the historical grouped 26 measured-row layout and "
            "adds a total row. Each timing cell is ordered **0K / 60K / 200K**. "
            "Fused kernels are charged once to their containing stage; "
            "the ↳ rows are detail-only inclusion records and add no timing; "
            "rows without a separate profiler scope are labelled in the timing "
            "cell rather than displayed as 0.000. The total counts overlapping GPU "
            "activity once. The old forced-replay subtraction is superseded; "
            "[its audit](benchmarks/results/stage-timing-audit-20260923.json) remains available."
        ),
        "",
        (
            "Each **set/order** pair means the same top-20 token set, followed by the same "
            "ranking. Current stage confirmations identify M1/M8 and eager/compiled M8 separately. "
            "Each position passes only if every layer instance passes on the same captured inputs. "
            "These diagnostic stage replays are checked against the compiled graph control; "
            "their times are not used in this table. Timing and correctness captures record their "
            "own exact source hashes. The provenance column identifies the last relevant code change."
        ),
    ]
    lines += [
        "",
        (
            "Fusion still permits correctness instrumentation: a diagnostic kernel can expose "
            "intermediate values, and fused outputs can be compared with an unfused reference. "
            "The normal GPU profile measures the combined kernel. Internal probes or splitting "
            "the kernel can change its performance, so those measurements are not an additive "
            "breakdown of the production kernel's time."
        ),
        "",
        (
            (f"The expandable layer and kernel tables use the same {retained_cycles:,} retained "
             f"60K cycles as the main table ({detail['date']}). GPU activity crossing a worker "
             "boundary is clipped to that boundary; activity-record counts include these fragments. "
             "CPU profiling work is excluded. Cycle counts are not output-token counts."
             if detail else
             f"The lower layer and kernel detail remains the separate historical compiled 60K trace: "
             f"it contains {retained_cycles} retained complete cycles and is not the 0K/60K/200K "
             "profile table above. Its cycle count must not be read as an output-token count.")
        ),
        "",
        "<details>",
        "<summary>" + ("Current" if detail else "Historical") + " compiled 60K decoder-layer detail: projection and remaining-work timings</summary>",
        "",
        (
            "Finish each row's layer, including its MLP, before moving to the next row. "
            "Each layer has four projections. Gate and up are one joint GEMM; there is no "
            "separately measured gate/up split. The remaining-work column combines normalization, "
            "mixing and other operations between those projections."
        ),
        "",
        "| Layer | Type | All layer work ms | Input projection ms | Output projection ms | Gate/up projection ms | Down projection ms | Other work ms |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in (detail["layers"] if detail else data["layers"]):
        lines.append(
            f"| {row['layer']} | {row['type']} | {row['total']:.4f} | {row['input']:.4f} | {row['output']:.4f} | {row['gate_up']:.4f} | {row['down']:.4f} | {row['other']:.4f} |"
        )
    lines += [
        "",
        "</details>",
        "",
        "<details>",
        "<summary>" + ("Current 60K GPU activity" if detail else "Historical recorded kernels") + ", grouped by stage</summary>",
        "",
        (f"| Stage / compiled kernel | Activity records in {retained_cycles:,} retained cycles | Current GPU ms per retained 60K cycle |"
         if detail else f"| Stage / compiled kernel | Calls in {retained_cycles} retained cycles | Historical GPU ms per profile cycle |"),
        "| --- | ---: | ---: |",
    ]
    for row in (detail["kernels"] if detail else data["kernels"]):
        name = row["kernel"].replace("|", "&#124;").replace("`", "")
        lines.append(
            f"| {row['stage']} / `{name}` | {row['calls']} | {row['ms']:.6f} |"
        )
    lines += [
        "",
        "</details>",
        "",
        *render_head_candidate_benchmark(),
        "",
        *render_coding_json_compaction_benchmark(),
        "",
        (
            "Aggregate data: [current measurements](benchmarks/results/coherence-current.json). "
            "Historical methodology and detailed numerical evidence: [technical report](reports/d7-rdna4-2026-09-17/REPORT.md)."
        ),
        "",
        END,
    ]
    return "\n".join(lines)


def update(text, data):
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError("README measurement markers are missing or duplicated")
    a = text.index(START)
    b = text.index(END) + len(END)
    return text[:a] + render(data) + text[b:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = ROOT / "README.md"
    actual = path.read_text()
    expected = update(
        actual,
        json.loads((ROOT / "benchmarks/results/coherence-current.json").read_text()),
    )
    if args.check:
        if actual != expected:
            raise SystemExit("README does not match current measured evidence")
    else:
        path.write_text(expected)


if __name__ == "__main__":
    main()
