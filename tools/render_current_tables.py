#!/usr/bin/env python3
"""Render current-only README evidence from the committed aggregate measurements."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- COHERENCE_CURRENT_RESULTS -->"
END = "<!-- /COHERENCE_CURRENT_RESULTS -->"
REPOSITORY = "https://github.com/Terrydaktal/vllm-coherence"
GLOBAL_BENCHMARK_LINES = (
    "## Global-256 target-head",
    "",
    "This is the table from [the top-256 PR](https://github.com/magiccodingman/vllm-radiance/pull/9).",
    "It is a **separate, earlier 60K-generated-token benchmark per method**, predating the",
    "M1/M8 and eager/compiled repairs and subsequent performance backports. Its old",
    "end-to-end throughput figures are intentionally omitted because they do not",
    "describe the current complete backend.",
    "",
    "| Target path | Median M8 head time | Top-1 match | Complete reference top-20 retained |",
    "|---|---:|---:|---:|",
    "| Full BF16 fallback | 4.122 ms | 119,988/119,988 (100.0000%) | 119,988/119,988 (100.0000%) |",
    "| Original block-8/64 + rerank-80 | 1.085 ms | 119,956/119,988 (99.9733%) | 98,452/119,988 (82.0515%) |",
    "| Global INT2 top-128 + BF16 rerank | 1.114 ms | 119,986/119,988 (99.9983%) | 118,254/119,988 (98.5549%) |",
    "| Global INT2 top-256 + BF16 rerank (default) | 1.128 ms | 119,986/119,988 (99.9983%) | 119,786/119,988 (99.8316%) |",
    "",
    "There were 115 natural completions per method on 11 private Pi request boundaries",
    "with 57,008–65,527 input tokens. Output totals were 60,598 / 60,075 / 60,348 /",
    "60,675 tokens for full / block / global-128 / global-256 respectively. Tools were",
    "not executed. Head timings are median eight-row GPU-event measurements on the",
    "same captured hidden vectors; 119,988 prediction rows were compared.",
    "",
    "Global-256 removes the eight-per-tile capacity limit, but remains approximate.",
    "The two changed final argmax IDs and 202 incomplete top-20 sets are observed",
    "misses. Complete top-20 retention does not certify score equality, ordering or",
    "sampling probabilities. The full BF16 path is the reference in this head study,",
    "not an independent proof of the model. [Methodology](docs/VERIFY_HEAD_GLOBAL_TOPK.md)",
    "· [Aggregate evidence](benchmarks/results/20260916-verify-head-global-topk-long/summary.json).",
)


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
    evidence = ROOT / "benchmarks/results/compiled-global256-stage-profile-1200.json"
    base = data.get("stage_profile_2k")
    if not evidence.exists() or not isinstance(base, dict):
        return base
    raw = json.loads(evidence.read_text())

    # This is the commit that packages the retained capture and the renderer;
    # the artifact itself records the older source checkout used to take the
    # measurement.  The table link therefore identifies the reproducible
    # evidence bundle rather than pretending the capture ran after a later
    # source change.
    evidence_commit = "40486363738eae43373fa350464ce1d5d0fd069b"
    grouped = copy.deepcopy(base)
    grouped["measurement_commit"] = evidence_commit
    grouped["scope"] = (
        "Original compiled Global-256 profiler in the fixed-BF16 lane. The "
        "single timing column shows 0K / 60K / 200K in that order; each value "
        "is the mean of complete retained cycles from one long capture, after "
        "the asynchronous profiler-window boundary cycles were removed. "
        "Requested profiler rounds were 2,183 / 1,191 / 1,191, with 2,062 / "
        "1,132 / 1,133 complete retained cycles. This restores the historical "
        "grouped table layout without reusing its old eager/full-BF16 timings. "
        "Row 26 carries the measured profile-cycle residual previously shown as "
        "Cycle overhead after stage sum: elapsed cycle boundary minus the 25 "
        "named stage sums. A separately controlled uninstrumented full-round "
        "measurement remains distinct and is not claimed here. These are "
        "diagnostic timings, not production throughput."
    )
    grouped["context_order"] = list(raw["context_order"])
    context_tokens = {"0K": 0, "60K": 60_000, "200K": 200_000}
    grouped["stage26"] = copy.deepcopy(base["stage26"])
    # Keep the historical row name from cbbf495; the definition below records
    # that its current value is the retained uninstrumented-round residual.
    grouped["stage26"]["label"] = "Estimated runtime overhead"
    grouped["stage26"]["definition"] = (
        "the displayed value is the measured profile-cycle residual; the "
        "separate uninstrumented full-round mean minus the sum of the 25 named "
        "instrumented-stage means remains the definitive control and must use "
        "the same build, mode, fixture and round window"
    )

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
            "Target head (global256)",
            "Scores the vocabulary with INT2, selects 256 candidates and rescores them with BF16 weights. Selection remains approximate.",
        ),
        (
            "Other GPU bookkeeping",
            "Other GPU bookkeeping",
            "Runs sampling and cache/state update kernels outside the named model stages.",
        ),
    )
    grouped["stage_order"] = [row[0] for row in historical_rows]
    grouped["stage_labels"] = {row[0]: row[0] for row in historical_rows}
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
            "profile_wall_ms_per_cycle": source["round_timing_ms"]["elapsed_ms"],
            "kernel_subtotal_ms": source["stage_sum_ms"],
            "profile_cycle_residual_ms": source["round_timing_ms"]["overhead_ms"],
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


def render_stage_profile_table(data):
    """Render the historical 26-stage profile as one 0K/60K/200K column.

    The older aggregate ``stages`` table is retained in the JSON for the
    compiled-trace/layer evidence below. This table is deliberately sourced
    from the current three-context compiled profiler and rendered as one
    slash-separated timing column. Stage 26 is deliberately sourced from a
    separate uninstrumented-round control when that control has been retained;
    the existing profile-cycle residual is shown when that is the only retained
    measurement, and is labelled as such rather than being presented as an
    independent uninstrumented control.
    """
    profile = current_stage_profile(data)
    if not isinstance(profile, dict):
        raise ValueError("the current measurements do not contain the 2K stage profile")
    order = profile["stage_order"]
    context_order = profile["context_order"]
    contexts = profile["contexts"]
    commit = commit_marker(profile["measurement_commit"])
    old_rows = {row["stage"]: row for row in data["stages"] if row["ms"] is not None}
    lines = [
        f"| Stage | Current timing per retained compiled profile cycle ({' / '.join(context_order)}; milliseconds unless explicitly marked; evidence run {commit}) | Current correctness evidence | Last relevant code commit / change | What this stage does |",
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
            provenance = f"{commit}: 2K forced-output stage profile; no backend code change"
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
            detail_commit = (
                data.get("stage_provenance", {})
                .get(evidence_source, {})
                .get("commit", profile["measurement_commit"])
            )
            detail_marker = commit_marker(detail_commit)
            for detail_label, detail_note in detail_rows:
                lines.append(
                    f"| ↳ {detail_label} | Included in **stage {number}** | "
                    f"Exact fused FP8 bytes/scales; see stage {number} · "
                    f"{detail_marker} | {provenance} | {detail_note} |"
                )
    benchmark = profile.get("stage26_benchmark", {})
    if benchmark.get("status") == "complete":
        benchmark_contexts = benchmark.get("contexts", {})
        residual = [benchmark_contexts[key]["residual_ms"] for key in context_order]
        timing = " / ".join(f"{value:.3f}" for value in residual)
        evidence = (
            "Separate uninstrumented-round residual; no isolated correctness claim "
            f"· {commit}"
        )
        provenance = f"{commit}: separate uninstrumented-round residual; no backend code change"
    else:
        residual = [contexts[key].get("profile_cycle_residual_ms") for key in context_order]
        if all(value is not None for value in residual):
            timing = " / ".join(f"{value:.3f}" for value in residual)
            evidence = (
                "Measured profile-cycle residual (elapsed boundary minus the 25 "
                "named stage sums); a separate uninstrumented control is not "
                f"claimed · {commit}"
            )
            provenance = (
                f"{commit}: restored the existing Cycle overhead after stage sum; "
                "no separate uninstrumented control"
            )
        else:
            timing = "— (separate benchmark pending)"
            evidence = (
                "Timing-only diagnostic profile; the required matching uninstrumented control "
                f"has not been retained · {commit}"
            )
            provenance = f"{commit}: separate uninstrumented-round benchmark is pending"
    lines.append(
        f"| **26. {profile['stage26']['label']}** | {timing} | {evidence} | {provenance} | {profile['stage26']['definition']}. |"
    )
    return lines


def render_coding_json_compaction_benchmark():
    lines = [
        "## Benchmarks",
        "",
        (
            "This content-free benchmark uses one retained 60,000-input-token Pi prefix and "
            "the compiled Coherence backend with the repaired arithmetic paths, Global-256 "
            "target head and qualified performance backports. The five requests are chained "
            "in one context, so only the first request pays the fresh-prefix preparation. All "
            "requests use temperature 1, top-p 0.95 and top-k 20 with natural stopping. The "
            "coding request disables thinking and treats fenced file edits as code; the short "
            "code-measurement prose request enables thinking but forbids code and JSON; the JSON "
            "request disables thinking and treats file-edit content as JSON; the fourth request "
            "enables thinking and asks for engineering prose; the last request generates a "
            "checkpoint after a forced cache flush. Prompt, response and token arrays were "
            "not read or saved; the run retained only hashes and numeric measurements. The "
            "runner is [benchmark_pi_coding_json_compaction.py](experiments/radiance-public/"
            "benchmark_pi_coding_json_compaction.py). The coding request began with 0 cached "
            "of 60,208 prompt tokens; later stages reused 64,272, 69,216, 70,864 and 77,456 "
            "cached tokens respectively, so this run did not reproduce a repeated full cold prefill."
        ),
        "",
        "| Stage | Thinking | Prompt tokens | Generated tokens | Classified output | First data | Mean round | Post-first | Peak 3s | Acceptance |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        "| Coding task | off | 60,208 | 6,994 | 6,382 code; 605 prose; 0 reasoning | 30.58 s | 44.80 ms | 101.61 tok/s | 138.45 tok/s | 50.72% |",
        "| Prose about code measurement | on | 67,343 | 3,753 | 3,752 prose; 0 separately observed reasoning; 0 code | 2.23 s | 45.43 ms | 82.63 tok/s | 153.49 tok/s | 39.33% |",
        "| JSON task | off | 71,215 | 2,690 | 2,684 JSON; 0 prose; 0 reasoning | 2.14 s | 46.30 ms | 78.03 tok/s | 102.43 tok/s | 37.30% |",
        "| Thinking/prose task | on | 74,052 | 5,754 | 5,753 prose; 0 separately observed reasoning | 2.70 s | 45.18 ms | 65.09 tok/s | 154.11 tok/s | 27.72% |",
        "| Compaction checkpoint | off | 79,958 | 3,389 | 3,389 checkpoint tokens; completion marker valid, required headings missing | 2.10 s | 45.68 ms | 71.64 tok/s | 116.18 tok/s | 32.45% |",
        "",
        (
            "The phase counters are content classifications, not a proof-level partition of "
            "backend token IDs: the coding, prose, JSON and thinking streams each retain one "
            "protocol-boundary token outside the classified content and report "
            "`phase_token_counts_cover_output=false`. Thinking was enabled for both prose "
            "stages, but this provider stream exposed no separate reasoning channel, so their "
            "3,752 and 5,753 observed tokens are reported as prose rather than being relabelled "
            "as reasoning. The compaction row "
            "measures checkpoint generation and cache flushing; it is not a claim that a full "
            "Pi transcript commit and old-snapshot retirement succeeded. The completion marker "
            "passed, but the required checkpoint headings did not, so that checkpoint must be "
            "treated as validation failure rather than a committed compaction. Each new run "
            "also records `peak_3s_tokens_per_second`: the maximum completed three-second "
            "sliding-window rate after first data, never a single-frame burst."
        ),
        "",
        *render_coding_context_benchmark(),
        "",
        *render_round_capture_summary(),
        "",
        *render_round_histogram(),
        "",
        "### Known remaining symptoms and likely causes",
        "",
        (
            "The situation recorded in `cbbf495` had warm rounds around 43.6--43.8 ms but a "
            "repeatable fresh-cache state around 53.5--53.9 ms. A stream or device "
            "synchronization recovered roughly 6.4 ms, which narrowed the evidence to residual "
            "HIP/ROCr queue or dependency state but did not identify a permanent repair."
        ),
        "",
        (
            "Since that diagnosis, [`abb7668`](https://github.com/Terrydaktal/vllm-coherence/commit/"
            "abb76682e96e1600e9b28ff404c36fa294244c54) made the runtime behavior and observation "
            "path explicit. It now drains pending device work after cache/mamba preparation, "
            "periodically re-arms the long-response recovery fence, records each decode round "
            "without charging another chat's GPU time to it, and gives Pi one shared snapshot "
            "for scheduler, cache, temperature, round and acceptance telemetry. Its scheduler "
            "also preserves response ownership through generation, makes tool-call handover "
            "decisions at the intended boundary, and retires superseded cache generations safely."
        ),
        "",
        (
            "[`279e0fe`](https://github.com/Terrydaktal/vllm-coherence/commit/"
            "279e0fe81ef63718914ab439530ffca5312fa3d8) added the missing HIP event-gap and "
            "round-latency records and made backend failures retain a content-safe, expandable "
            "diagnostic report. These changes fix the previous lack of evidence and misleading "
            "Pi status; they do not make the underlying asynchronous queue issue mathematically "
            "solved. The M1/M8 and eager/compiled arithmetic repairs, Global-256 target-head "
            "change and GEMM performance backports were already present in the baseline documented "
            "by `cbbf495`; they are not new fixes after that commit."
        ),
        "",
        (
            "The old 53--54 ms state was not reproduced by this chained run: the measured "
            "generation intervals were 44.60--47.01 ms. That is evidence that the recovery and "
            "scheduling changes are helping, not proof that the slow state is impossible. The "
            "remaining latency risk is still the same class of defect: an asynchronous HIP/ROCr "
            "stream or queue dependency left behind by cache restore, handover or a long response. "
            "The new event-gap feed can distinguish a GPU queue gap from host dispatch, cache "
            "transfer and telemetry wait; further live evidence is required before calling that "
            "root cause fixed. Round means also depend on workload and speculative acceptance, so "
            "the 47.01 ms thinking row alone is not a new kernel regression."
        ),
        "",
        (
            "Two independent correctness/observability issues remain visible in this run: the "
            "provider did not expose a reasoning channel even when requested, and the checkpoint "
            "marker was present while the required section contract was absent. The first points "
            "to the provider/stream adapter's reasoning metadata path; the second is a checkpoint-"
            "format or model-compliance failure, not evidence of a cache-timing failure. Both "
            "should remain explicit failures in qualification."
        ),
    ]
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
                f"{capture.get('record_count', 0):,}/{capture.get('expected_rounds', '—')} "
                f"events; {capture.get('measured_round_count', 0):,} timed; {capture_status}"
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
            "temperature 1, top-p 0.95 and top-k 20. The non-empty arms use operator-supplied "
            "token-prefix fixtures; only their hashes are retained. The three-second peak is "
            "the maximum completed sliding-window rate, not a single-frame burst. Runner: "
            "[benchmark_pi_coding_contexts.py](experiments/radiance-public/"
            "benchmark_pi_coding_contexts.py)."
        ),
        "",
        "| Context | Prompt tokens | Generated tokens | Mean round | Post-first | Peak 3s | Acceptance | Round events (logged/expected; timed) | Validation |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        *rows,
        "",
        f"Report status: `{report_state}`. A pending row has not been measured and carries no fabricated performance value. Each completed row stores every content-free scheduler event under `contexts.<context>.round_capture.records`; a count mismatch is a validation failure.",
        *( [provenance_line, ""] if provenance_line else [] ),
        (
            "Run all three arms against an active backend with `uv run python "
            "experiments/radiance-public/benchmark_pi_coding_contexts.py --fixture-60k "
            "PATH_TO_60K_FIXTURE --fixture-200k PATH_TO_200K_FIXTURE "
            "--tokenizer-json PATH_TO_TOKENIZER --abi SNAPSHOT_ABI`."
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
            rows.append(f"| {context} | — | — | — | — | — | pending rerun |")
            continue
        missing = ", ".join(str(value) for value in capture.get("missing_round_numbers", [])) or "none"
        rows.append(
            f"| {context} | {capture.get('record_count', 0):,} | "
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
            "validation failure."
        ),
        "",
        "| Context | Logged events | Expected rounds | Timed round values | Untimed events | Missing round numbers | Capture status |",
        "| ---: | ---: | ---: | ---: | ---: | --- | --- |",
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
            "The histogram below is generated from the complete per-round records in this benchmark. Every measured `round_ms` value appears in exactly one bin; untimed events are reported separately.",
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


def render(data):
    timing = data["round_timing"]
    mean = timing["mean"]
    retained_cycles = len(timing["rounds"])
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
            "The table restores the historical grouped 26-row layout. Each timing "
            "cell is ordered **0K / 60K / 200K** and comes from the retained "
            "compiled profiler cycles in the evidence run named in the header. "
            "The original profiler recorded 2,183 / 1,191 / 1,191 requested "
            "rounds and retained 2,062 / 1,132 / 1,133 complete cycles. Fused "
            "scopes are charged to one historical row and called out in its note; "
            "the ↳ rows are detail-only inclusion records and add no timing; "
            "rows without a separate profiler scope are labelled in the timing "
            "cell rather than displayed as 0.000. Row 26 includes the "
            "measured profile-cycle residual; the definitive uninstrumented "
            "control remains separate."
        ),
        "",
        (
            "**Set/order** in the numerical section means the same top-20 token set, "
            "followed by the same ranking. The timing-table header and each "
            "correctness result link to the commit that produced or packaged that "
            "evidence. The correctness column refers to the corresponding "
            "production stage; the profile itself is timing-only. The provenance "
            "column links the last relevant implementation commit and describes "
            "its change."
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
            f"The lower layer and kernel detail remains the separate compiled 60K trace: it "
            f"contains {retained_cycles} retained complete cycles and is not the 0K/60K/200K "
            "profile table above. Its cycle count must not be read as an output-token count."
        ),
        "",
        "<details>",
        "<summary>Historical compiled 60K decoder-layer detail: projection and remaining-work timings</summary>",
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
    for row in data["layers"]:
        lines.append(
            f"| {row['layer']} | {row['type']} | {row['total']:.4f} | {row['input']:.4f} | {row['output']:.4f} | {row['gate_up']:.4f} | {row['down']:.4f} | {row['other']:.4f} |"
        )
    lines += [
        "",
        "</details>",
        "",
        "<details>",
        "<summary>Every recorded kernel, grouped by stage</summary>",
        "",
        f"| Stage / compiled kernel | Calls in {retained_cycles} retained cycles | Current GPU ms per profile cycle |",
        "| --- | ---: | ---: |",
    ]
    for row in data["kernels"]:
        name = row["kernel"].replace("|", "&#124;").replace("`", "")
        lines.append(
            f"| {row['stage']} / `{name}` | {row['calls']} | {row['ms']:.6f} |"
        )
    lines += [
        "",
        "</details>",
        "",
        *GLOBAL_BENCHMARK_LINES,
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
