# vLLM Coherence

**Fast Qwen inference for the AMD R9700, with numerical repairs, checked
optimizations and persistent Pi coding sessions.**

Coherence is a model-serving backend, verification toolkit and session layer for
running a local coding agent. Developed and maintained by
[Terrydaktal](https://github.com/Terrydaktal), it builds on Radiance and selected
GGZ14 optimizations, adding repairs for demonstrated numerical errors,
instrumentation to locate new ones, and durable chat caches.

| Current supported setup | What it runs |
| --- | --- |
| GPU and host | **One AMD Radeon AI PRO R9700, 32 GiB VRAM**, RDNA4 / `gfx1201`, Linux x86-64 |
| Target model | **Qwen3.8-27B-Uncensored-MXFP4-awq**, with MXFP4 weights |
| Speculative drafter | **Qwen3.8-27B-DFlash2-FP8**; seven proposed tokens checked in an eight-row target pass (D7 / M8) |
| Coding agent | **[Pi](https://github.com/earendil-works/pi) 0.84.2**, with Coherence's runtime patches and extensions |
| Serving precision | MXFP4 weights, FP8 activation and KV-cache quantization, with BF16/FP32 arithmetic where specified by the implementation |

Pi supplies the terminal agent and tool loop. Coherence supplies its local model
backend, cache persistence, scheduling and progress reporting. The backend also
exposes vLLM's API for other clients; the integrated workflow and published
qualification focus on the setup above.

[Quick start](#quick-start) · [Numerical report](reports/d7-rdna4-2026-09-17/REPORT.md)
· [Verification](docs/VERIFICATION.md) · [Architecture](docs/ARCHITECTURE.md)
· [Development and private lab](docs/DEVELOPMENT.md)
· [Pi features](#what-changes-in-pi) · [Pi setup](#pi)
· [Releases](https://github.com/Terrydaktal/vllm-coherence/releases)

## Why Coherence exists

Radiance and GGZ14 provide the fast RDNA4 serving and kernel work this project
depends on. Coherence addresses a further problem: execution paths that implement
the same model can produce different results. The investigation found numerical
disagreements between ordinary one-token target decoding (M1) and eight-row
speculative verification (M8), and between eager and compiled execution.

Coherence repairs those demonstrated discrepancies and adds a reusable way to
find the first differing operation or state update. Performance backports are
adapted and checked against the repaired path, with guards and fallbacks where
needed. Long-running Pi sessions also get persistent caches, reliable compaction
and visibility into what the backend is doing.

**The goal is to be as faithful and lossless as possible to the original
quantized model.** Optimizations, speculation and cache save/restore should
preserve its outputs and committed model state. The long-term objective is
equivalence for every supported input and reachable state under a precisely
defined arithmetic and quantization contract. Weight, activation and KV
quantization must be declared separately; an optimization must not silently
introduce another approximation.

**Current evidence is finite, not a universal correctness proof.** The alignment
study and later backport checks establish exact agreement in their tested
domains; they do not independently prove the reference path correct. The default
global-512 target head remains an explicitly approximate speed option.
`--head full-bf16` removes that candidate-shortlist approximation, while retaining
the rest of the quantized backend. See the [results below](#current-numerical-results)
and [verification contract](docs/VERIFICATION.md) for the exact scope.

The [latest eager-M1 repairs](docs/eager-m1-followup-audit.md) remove demonstrated
attention range and intermediate-rounding losses and reject invalid GDN/cache
layouts. The deployed release now passes fresh 320-token comparisons between
M1/M8 and eager/compiled execution, including every full-logit hash and the
prefill prediction. Independent attention oracles are recorded separately.

The [independent all-stage M1 audit](docs/eager-m1-all-stages-audit.md), now
[rerun with 2,817 passing checks](benchmarks/results/current-operator-confirmations-20260925.json), extends this
to all 64 target layers: public-weight projections, normalization, recurrent and
convolution state, RoPE, attention, cache writes and sampling. It distinguishes
exact checks from numerical-error checks. The [normalization follow-up](docs/eager-m1-contract-qualification.md) eliminates the demonstrated prefill/M1 FP8 counterexamples under a declared arithmetic contract; full-model prefill/decode equivalence and approximate-head limitations remain open.

## Upstream foundations

Coherence is a downstream fork of **magiccodingman's Radiance**. That Radiance
line builds on **StillDeadcode's Radiance and libr4d**, and incorporates
**GGZ14's MXFP4/W4A8 work**. These projects in turn use vLLM and the AMD software
stack. Coherence preserves their attribution and adds its own repair,
verification and session work on top.

| Upstream project | What Coherence builds on |
| --- | --- |
| [vLLM](https://github.com/vllm-project/vllm) | Model execution, serving API, scheduling, paged KV cache and speculative-decoding integration |
| [StillDeadcode's Radiance](https://codeberg.org/StillDeadcode/vllm-radiance) and [libr4d](https://codeberg.org/StillDeadcode/libr4d) | Original Radiance runtime and hand-written RDNA4 attention, GDN and other GPU kernels |
| [GGZ14 / Brian's MXFP4 work](https://github.com/GGZ14/vllm-mxfp4) | MXFP4/W4A8 kernels and performance optimizations, inherited through Radiance and selectively backported here |
| [magiccodingman's Radiance](https://github.com/magiccodingman/vllm-radiance) | Coherence's direct source and image base: pinned vLLM/ROCm integration, reviewed backports and deployment work |
| [ROCm](https://github.com/ROCm/rocm-systems), [AMD PyTorch](https://github.com/ROCm/pytorch), [AMD Triton](https://github.com/ROCm/triton), [AITER](https://github.com/ROCm/aiter) and [FLA](https://github.com/fla-org/flash-linear-attention) | GPU runtime, tensor execution, compilers and underlying operator implementations |
| [Qwen](https://github.com/QwenLM), [DFlash](https://github.com/z-lab/dflash) and the target/drafter checkpoint authors | Model architecture, weights and speculative-decoding method; weights are obtained separately |
| [Pi](https://github.com/earendil-works/pi) | The coding agent, terminal interface, provider API and extension system |

The first commit is the unchanged Radiance 1.0.16 source at
`f295b9ef51ad413a68e4192371e0377741a354ce`. Later commits group Coherence's additions
by feature. This is a standalone GitHub repository with that documented fork
lineage. [Attribution and individual upstream fixes](ATTRIBUTION.md) retain the
exact source links and pins; the inherited [Radiance README](docs/upstream/RADIANCE-README.md)
and [Dockerfile](Dockerfile) document its base stack.

## What Terrydaktal adds in Coherence

- **Numerical repairs and execution alignment:** GDN convolution, recurrent-state
  updates and prefill; alignment of normalization, attention and vocabulary-head
  arithmetic; preservation of intermediate BF16 rounding during compilation.
- **Performance backports that preserve the tested results:** guarded GGZ14 GEMM
  dispatch, normalization/FP8 fusion, GDN prefill scan tiling and tiled prefill
  inputs. The backports retain the corrected arithmetic and existing state layout,
  with operator and complete-model comparisons documented below.
- **Global-512 target-head improvement:** search the complete INT2 score row, then
  rescore 512 candidates using BF16 weights. This removes the old eight-candidates-
  per-tile restriction and improves measured candidate recall; it remains approximate.
- **The Pi workflow below:** a patched agent runtime and extensions, backed by
  persistent chat state, transactional compaction, shared-GPU scheduling and live
  progress and cache telemetry.
- **Reusable verification:** forced-token replay, logical-state comparison,
  first-divergence capture, operator checks, deliberate fault injection and small,
  explicitly scoped machine-checked obligations.

Existing upstream fixes, including DFlash RNG separation and the Triton RoPE
rounding repair, are integrated and credited as backports. The drafter head also
[defers INT2 initialization until its real shared weights are available](https://github.com/GGZ14/vllm-mxfp4/commit/1d76c82699c24ffe537e543dc4da575500d4e639):
nonzero allocator leftovers must not be mistaken for loaded weights. This startup
repair preserves the existing quantizer and adds no work to subsequent rounds.
CPU regressions cover dirty placeholders, weight sharing, one-time packing and
the hash-checked release installer. The original model,
agent and kernel foundations retain their upstream authorship.

## What changes in Pi

The Pi integration is a substantial part of Coherence. It connects the agent's
interface and session lifecycle to the backend's cache and scheduler, so users
can see what their chat is doing and resume work with compatible cached state.

### Live status in the terminal

| Feature | What you see |
| --- | --- |
| Exact context counter | Used tokens / total capacity, with thousands separators and a percentage. The workspace, counters, temperatures and model form one continuous footer that wraps to the terminal width. |
| Per-chat cache counters | `GPU` and `RAM` show available cached context; `Disk` shows verified tokens saved to disk; `Cold` estimates tokens that must be recomputed. Disk backup can overlap GPU/RAM residency, so these four figures are not a partition to add together. |
| Shared hardware monitoring | GPU junction temperature, edge temperature and fan percentage, followed by Pi's input/output/cache usage figures and model name. One shared probe serves multiple windows: temperatures update every second, cache/scheduler metadata every 0.5 seconds. |
| Specific working spinner | Separate timed states for request preparation, admission, waiting for another chat, GPU/RAM handover, cache lookup, snapshot restore, uncached prefill, reasoning and answer generation. A queued chat identifies the blocking chat and its activity when telemetry is available. |
| Honest generation rates | A three-second rolling rate followed by the average, latest round time, three-second acceptance, token counts, time to first data and elapsed time. Generation and compaction displays refresh every 100 ms from shared telemetry. Time spent waiting for another chat is excluded from generation rates; missing reasoning counts are omitted. |
| Tool-call visibility | Token usage continues updating while tool arguments are buffered. `generating edit arguments` and `applying edit` distinguish model work from tool execution, with an execution timer. |
| Backend error details | Engine failures can carry the actual recorded traceback, expandable with `Ctrl+O`; `/backend-error` retrieves the latest diagnostic. |

Unavailable telemetry is reported as unavailable rather than presented as zero
cached tokens or an invented explanation for a wait.

### Compaction, snapshots and history

| Feature | What Coherence adds |
| --- | --- |
| Transactional compaction | Manual and automatic compaction preserve the original transcript unless the checkpoint passes section, completion-marker and finish-status checks. The prompt preserves the existing token prefix for cache reuse, and checkpoint output can use the remaining context capacity. |
| Visible compaction steps | An in-place progress display times prompt preparation, queueing, cache loading, prefill, checkpoint generation, validation, tail flush, commit and cleanup. Checkpoint generation shows tokens, three-second and post-first average t/s, round time and three-second acceptance. The completed compaction entry retains total elapsed time. Typed editor input survives compaction. |
| Snapshot restore | Cache identity binds the chat, compaction generation, exact prefix and runtime compatibility. Compatible state can be reused from GPU, RAM or compressed disk snapshots; only the missing suffix needs prefill. |
| Less disk rewriting | Immutable blocks are reused and the changing tail stays in memory. It normally flushes after about 8,192 new tokens, and on explicit flush, RAM eviction, successful compaction and clean shutdown. The previous complete disk head remains valid until its replacement is verified. |
| Cleanup across compactions | A committed compaction supersedes the old generation, retires its snapshots and prevents stale requests from bringing it back. Cleanup failures are reported and retried. The superseded GPU bank is discarded without parking the obsolete generation in RAM. |
| Inspectable cache storage | The cache CLI shows each chat's saved coverage, disk size, cumulative disk traffic, RAM usage and active Pi processes/ports, and flags duplicate/incomplete snapshots and failed garbage collection. |
| Project-local history | Model-neutral transcripts live in the workspace's `.pi/sessions`; `--session last` resumes its latest chat. Transcript reuse across models is separate from model-specific KV-cache compatibility. |
| Bounded tool output with retrieval | Large tool results get compact context views while the full output remains in an authenticated archive. The agent can retrieve exact line ranges or matching passages. |
| Reproducible local/remote setup | The launcher installs and checks the patched Pi runtime, loads the operating prompt and extensions, and automatically reserves a free SSH forwarding port. A custom search extension can supply a VM-specific tool. |

### Two chats sharing one GPU

At equal priority, a chat finishes its current model response before the other
chat can take over. A **two-second tool-call grace period** lets short tools return
without a needless GPU handover. Longer tools give another waiting chat an
opportunity to run while the first is occupied. Cached state can be parked in
system RAM for the return trip; the waiting window explains who owns the GPU.

`/priority` shows the current setting, which defaults to zero and is saved per chat:

| Setting | Scheduling behavior |
| --- | --- |
| `/priority 0` | Normal response-boundary scheduling and tool-call grace. |
| `/priority 1` | Acquire the GPU at a normal handover, then retain it through tool calls until the answer finishes, ahead of lower-priority chats. |
| `/priority 2` | Request takeover from a lower-priority chat at the next safe GPU step, then retain the GPU through tools until the answer finishes. |

Equal priorities keep normal scheduling. An **answer** includes its model
responses and intervening tool calls; finishing it releases the ownership hold.
See [Pi setup, counter definitions and recovery](docs/PI.md) and
[cache inspection commands](#cache-inspection).

<!-- COHERENCE_CURRENT_RESULTS -->
## Current numerical results

September 25 candidate rerun: **320 forced decode tokens after a fresh 60,000-token Pi prefill**, plus the prefill prediction, in each of four arms. Compiled M1 versus compiled M8, eager M1 versus compiled M1, eager M8 versus compiled M8, and eager M1 versus compiled M8 each produced the results below. Compiled M8 includes the three banked speed candidates and uses the FULL graph; compiled M1 retains the original PIECEWISE path. Eager controls retain the existing RoPE rounding repair and original operators. The **full BF16 comparison head** exposes target-body differences; serving performance uses approximate Global-512, measured separately. [Run identities and all four comparisons](benchmarks/results/current-320-confirmations-20260925.json). The [qualified speed refresh](docs/SPEED_INVESTIGATION_20260925.md) includes the measurement adapters and FULL-graph preparation repair. Captures retain their original checkout identities and measured source hashes after the history rewrite. The experimental worker remains separate from the normal Pi deployment.

| Prediction | Same token set | Same ordering | Mean shared tokens |
| --- | ---: | ---: | ---: |
| Top 1 | 320 / 320 (100.00%) | 320 / 320 (100.00%) | 1.0000 / 1 |
| Top 10 | 320 / 320 (100.00%) | 320 / 320 (100.00%) | 10.0000 / 10 |
| Top 20 | 320 / 320 (100.00%) | 320 / 320 (100.00%) | 20.0000 / 20 |

Each comparison also matched all 320 full-vocabulary logit hashes and the prefill prediction. This is finite execution consistency, not independent model certification, an arbitrary-input proof or a natural-completion quality test. The slice comes from the retained Pi corpus; it is not the deleted historical standalone 320-token fixture. The earlier 10K study remains historical and was not rerun.

## Compiled backend stages

Measured on 2026-09-25 using the compiled, optimized Global-512 serving backend with the pinned-RAM huge-page promotion repair; sampling is temperature 1.0, top-p 0.95 and top-k 40. Context labels are starting prefixes: 0K, the private 60K Pi fixture, and the public synthetic 200K fixture. Each context ran a natural warmup followed by clean control, trace, and clean control; each arm generated 4,913 / 5,173 / 3,912 tokens respectively. Generated-token hashes and accepted-token schedules matched. The stage means retain 849 / 1122 / 756 complete M8 cycles (0K / 60K / 200K), and controls use exactly those same decode indices. Trace setup/export boundaries and incomplete or inconsistent trace inventories are excluded by structure, never by duration; complete native round logs retain all rounds and stalls. GPU activity timestamps supply the stage times; CPU annotations, Python hooks and export time are excluded. No per-stage event probes or forced-token replay are used. The measured tracing slowdown was 4.288 / 3.134 / 3.478 ms per retained round; it is reported separately and is **not charged to row 26**. Row 26 is the clean control mean minus the union of GPU activity intervals. This remains an estimate: tracing can indirectly affect clocks and scheduling. Overlap is counted once in the total. The header links the qualification bundled with these repairs; its captures retain their original source identities. The capture retains its original checkout and source identities. [Capture and source identities](benchmarks/results/compiled-global512-stage-profile-20260925.json) · [controls](benchmarks/results/stage26-control-20260925.json) · [method and uncertainty](docs/STAGE_TIMING.md).

This candidate rerun includes the full-graph cache-preparation repair and measurement adapters in the [qualified speed refresh](docs/SPEED_INVESTIGATION_20260925.md). Captured checkout identities remain unchanged after the history rewrite; the installed source and binary hashes bind the measured implementation. The experimental worker is separate from the normal Pi deployment.

The **eager M1** column combines the [current independent operator rerun](benchmarks/results/current-operator-confirmations-20260925.json) (**2,817 checks, zero failures**, 320 rows at the audited sites) with the broader [arithmetic-contract qualification](docs/eager-m1-contract-qualification.md). The [declared arithmetic](docs/M1_ARITHMETIC_CONTRACT.md) separates accepted weight, activation and KV quantization from implementation defects. Confidence scores are editorial assessments: **1/5** untested, **2/5** limited, **3/5** moderate, **4/5** high, **5/5** very high for the tested exact encoding/index/state properties. They are **not probabilities of being bug-free**; 5/5 is not an arbitrary-input proof. N/A identifies a separate model or a timing-only row. FP64 oracle checks use declared error criteria, with numerical differences retained in the evidence.

The candidate rerun also passes all four [320-token whole-model comparisons](benchmarks/results/current-320-confirmations-20260925.json), including eager M1 versus compiled M8: exact full-logit hashes and prefill prediction. This establishes consistency on the sample, not independent correctness of the model reference. The normalization release gates additionally cover 128 hidden-normalization sites and 48 GDN sites with 1,000 rows, and remain historical operator evidence. The timing column measures the new compiled Global-512 speed candidate, whose source and binary identities are bound to the September 25 capture; the candidate is not yet the Pi deployment. [Numerical release identities](benchmarks/results/eager-m1-normalization-deployment-20260924.json). Complete-model prefill-versus-serial-decode equality, candidate completeness and arbitrary concurrency remain open. A fresh [native snapshot lifecycle test](benchmarks/results/snapshot-lifecycle-20260925.json) did pass A-to-B-to-A handover, disk restore after restarting the backend and three verified replacement cycles. This is finite lifecycle coverage, not an exhaustive concurrency proof.

The fresh [isolated stage run](benchmarks/results/current-stage-confirmations-20260925.json) covers 22 boundaries and 770 layer instances over the same 320 Pi tokens. Current M1/M8 and eager/compiled M8 match at every stage: local tensors and state, full-logit hashes, and top-1/10/20 sets and ordering. All 40 eight-token groups pass deliberate corruption controls and restore authoritative state. This adds integration evidence to the independent operator checks; shared implementations still are not independent mathematical references.

| Stage | Current GPU activity per retained compiled M8 cycle (0K / 60K / 200K; milliseconds unless explicitly marked; 2026-09-25; run [qualified speed refresh](docs/SPEED_INVESTIGATION_20260925.md)) | Current M1->M8 correctness and eager->compiled correctness evidence | Current Eager M1 correctness evidence | Last relevant code commit / change | What this stage does |
| --- | ---: | --- | --- | --- | --- |
| **1. Drafter** | 4.832 / 4.939 / 4.988 | Separate proposal model; target M1/M8 comparisons do not independently qualify it. | **N/A: separate proposal model.** Independent proposal/target RNG controls passed. **Uncertainty:** this target-M1 audit does not independently qualify drafter arithmetic or every speculative commit path. | [Qualified speed changes](docs/SPEED_INVESTIGATION_20260925.md): Use the qualified unit_w4_s1_occ2 drafter attention launch; target attention is unchanged and deferred head initialization is retained. | Suggests up to seven tokens for the target model to check. |
| **2. Embedding + first input normalization + FP8 production** | 0.009 / 0.009 / 0.009 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260925.json). | **4/5 High.** Exact sampled embeddings and independent RMS checks; released fused norm preserves M1 FP8 bytes/scales at tested batch boundaries from 1 to 2,048 rows. **Uncertainty:** finite inputs and indices; prefill versus a fully serial reference is not qualified. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 reduction and BF16/FP8 rounding at every admitted prefill width, removing batch-dependent activation differences. | Looks up token vectors, normalizes them and creates FP8 inputs in one kernel, preserving intermediate rounding. |
| **3. Layer input residual/normalization + FP8 production** | 0.496 / 0.507 / 0.526 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260925.json). | **4/5 High.** Qualified release matches native M1 plus independent FP8 encoding at 129 hidden-norm weights × 320 rows; original 55 byte differences eliminated. CPU arithmetic reproduces the saved failing rows. **Uncertainty:** finite inputs, intrinsic/denormal behavior and untested inputs; deployed 24 September. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 reduction and BF16/FP8 rounding at every admitted prefill width, removing batch-dependent activation differences. | Adds the residual, normalizes the result and creates FP8 inputs in one kernel, preserving intermediate rounding. |
| ↳ GDN input activation FP8 quantization | Included in **stage 3** | Exact fused FP8 bytes/scales; see stage 3 and its evidence scope | **5/5 Very high for tested encoding.** Qualified release fused bytes/scales match independent encoding; stage 3 prefill/M1 counterexamples now pass. **Uncertainty:** inherits normalization arithmetic; FP8 is an explicitly accepted approximation; deployed 24 September. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 reduction and BF16/FP8 rounding at every admitted prefill width, removing batch-dependent activation differences. | Uses the FP8 output produced by the same input-normalization kernel. |
| ↳ Attention input activation FP8 quantization | Included in **stage 3** | Exact fused FP8 bytes/scales; see stage 3 and its evidence scope | **5/5 Very high for tested encoding.** Qualified release fused bytes/scales match independent encoding; stage 3 prefill/M1 counterexamples now pass. **Uncertainty:** inherits normalization arithmetic; FP8 is an explicitly accepted approximation; deployed 24 September. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 reduction and BF16/FP8 rounding at every admitted prefill width, removing batch-dependent activation differences. | Uses the FP8 output produced by the same input-normalization kernel. |
| **4. GDN input projection** | 4.040 / 3.972 / 3.993 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260925.json). | **4/5 High.** Current 496-matrix × 320-input sampled-channel rerun, plus all output channels of 30 selected matrices on 32 inputs each (7,739,392 outputs); 12 saved operator-activation replays add 3,276,800 outputs. **Uncertainty:** other matrices retain sampled-channel coverage; FP64 error criteria are not exact-arithmetic proofs; the separate 320-token stage replay uses full-model activations but shares the native operator, rather than an independent FP64 oracle. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve stored MXFP4 coefficients in the folded lookup and repair fragment/tiled per-block fallbacks while retaining the qualified GEMM dispatch. | Projects hidden vectors into the inputs and gates for the recurrent layer. |
| **5. GDN layout/copies and buffer initialization** | 0.193 / 0.200 / 0.205 | Authoritative cache/state restored in all 40 eight-token groups; state/output corruption controls detected · [current replay](benchmarks/results/current-stage-confirmations-20260925.json). No separate layout top-20 attribution. | **4/5 High in the tested replay.** Unsupported strides/overlap rejected; final histories and untouched slots exact across 48 GDN sites. All 40 current eight-token replay groups restore authoritative cache/state, including corruption controls. **Uncertainty:** the separate synthetic snapshot lifecycle passes handover and disk restore, but not every cancellation or concurrent-ownership history. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Reject unsupported packed GDN inner strides and overlapping state slots; retain the existing nine-slot layout. | Arranges inputs and clears temporary buffers throughout each GDN layer. |
| **6. GDN convolution** | 0.180 / 0.185 / 0.190 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260925.json). | **4/5 High.** 48 layers × 320 consecutive inputs pass FP64 convolution/SiLU criteria; histories and untouched slots exact. **Uncertainty:** finite numerical/state samples; full-session integration remains unqualified by this audit. | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Aligned GDN convolution product and accumulation arithmetic across M1/M8 and eager/compiled execution. | Updates recent-token history using the corrected multiply/add order. |
| **7. GDN recurrence and gates** | 1.348 / 1.202 / 1.231 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260925.json). | **4/5 High.** 48 layers × 320 transitions match independent state/output equations within tolerance; 32,768-step decay passes. **Uncertainty:** no exact full-model recurrence proof or coverage of every reachable state. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Use stable log1p(exp(x)) in M1, M8 and prefill so small nonzero recurrent gates survive FP32 cancellation. | Updates recurrent memory in token order. Faster prefill retains the existing nine-slot state layout. |
| **8. GDN output gated normalization + FP8 production** | 0.148 / 0.151 / 0.155 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260925.json). | **4/5 High numerically.** Explicit FP32 norm/weight/gate contract with one final BF16 boundary; released path matches native M1 at all 48 sites × 320 audit rows, plus 1,000-row release gates and batch-boundary checks. **Uncertainty:** this deliberately differs from Hugging Face intermediate casts; intrinsic behavior and arbitrary inputs remain unproved; deployed 24 September. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 gated-normalization reduction layout during prefill and retain one final BF16 rounding before FP8 production. | Normalizes and gates GDN output, then produces FP8 inputs in one kernel; intermediate BF16 rounding is preserved. |
| ↳ GDN output activation FP8 quantization | Included in **stage 8** | Exact fused FP8 bytes/scales; see stage 8 and its evidence scope | **5/5 Very high for tested encoding.** Qualified M1/prefill bytes and scales match independent encoding of native BF16 output at 48 sites × 320 rows. **Uncertainty:** inherits the declared stage 8 arithmetic; no weight-only model equivalence; deployed 24 September. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 gated-normalization reduction layout during prefill and retain one final BF16 rounding before FP8 production. | Uses the FP8 output produced by the same gated-normalization kernel. |
| **9. GDN output projection** | 1.833 / 1.816 / 1.826 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260925.json). | **4/5 High.** Current 496-matrix × 320-input sampled-channel rerun, plus all output channels of 30 selected matrices on 32 inputs each (7,739,392 outputs); 12 saved operator-activation replays add 3,276,800 outputs. **Uncertainty:** other matrices retain sampled-channel coverage; FP64 error criteria are not exact-arithmetic proofs; the separate 320-token stage replay uses full-model activations but shares the native operator, rather than an independent FP64 oracle. | [Qualified speed changes](docs/SPEED_INVESTIGATION_20260925.md): Use qualified local-split GEMM for M1/M8 down/output shapes; preserve the repaired MXFP4 coefficients and reduction order. | Maps the recurrent-layer result back to the model's hidden-vector width. |
| **10. Attention input projection** | 1.157 / 1.152 / 1.159 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260925.json). | **4/5 High.** Current 496-matrix × 320-input sampled-channel rerun, plus all output channels of 30 selected matrices on 32 inputs each (7,739,392 outputs); 12 saved operator-activation replays add 3,276,800 outputs. **Uncertainty:** other matrices retain sampled-channel coverage; FP64 error criteria are not exact-arithmetic proofs; the separate 320-token stage replay uses full-model activations but shares the native operator, rather than an independent FP64 oracle. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve stored MXFP4 coefficients in the folded lookup and repair fragment/tiled per-block fallbacks while retaining the qualified GEMM dispatch. | Projects hidden vectors into the inputs for attention. |
| **11. Attention Q/K normalization, RoPE and layout** | 0.240 / 0.245 / 0.252 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260925.json). | **4/5 High.** All 32 Q/K norm sites; both position paths at nine ranges pass independent arithmetic checks; repaired eager MRoPE launches verified. **Uncertainty:** rounding tolerances and compact RoPE tables; full-table addressing not covered. | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Added live-row handling for compiled M8 graph padding while retaining the repaired RoPE arithmetic. | Normalizes queries and keys and applies positional rotation (RoPE), retaining the corrected rounding. |
| **12. Attention KV write** | 0.034 / 0.035 / 0.036 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260925.json). | **5/5 Very high within tested cases.** 393,216 stored values/destinations exact; BF16/OCP FP8 scales, masked slots, page edges and every finite FP8 code checked. **Uncertainty:** arbitrary aliasing, asynchronous access and native snapshot restoration remain unproved. | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Aligned causal attention and cache writes with the M1/M8 arithmetic contract. | Stores new keys and values in the cache for reuse by later tokens. |
| **13. Attention decode** | 0.387 / 4.823 / 14.315 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260925.json). Decode and merge checked together. | **4/5 High within tested cases.** Earlier 88 cases plus 96 independent FP64 cases with unique shuffled pages, holes, poisoned tails, per-head scales and varied contents through 60,001 tokens; guards intact. **Uncertainty:** longest 253K checks still use structured/repeated pages; exact softmax and arbitrary layouts remain unproved. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve BF16 query/key/value range, scale after the FP32 dot product, retain probability residuals and reject incompatible FP8 formats; keep the page-boundary traversal repair. | Attends to the current and earlier tokens using corrected arithmetic and shared cache loads. |
| **14. Attention split-KV merge** | 0.263 / 0.132 / 0.107 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260925.json). Decode and merge checked together. | **4/5 High for repaired arithmetic.** Four exact split-mean failures repaired; 96 additional varied-layout attention cases pass, including exact BF16 cancellation controls. **Uncertainty:** integrated evidence, not an exhaustive independent merge-stage proof. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Keep split outputs and normalization sums in FP32 until the final BF16 merge, removing intermediate FP16 rounding loss. | Combines attention results from cache partitions in the corrected arithmetic order. |
| **15. Attention output gating** | 0.028 / 0.033 / 0.034 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260925.json). | **4/5 High.** Every finite BF16 sigmoid input tested with a bounded multiplier against FP64 equations. **Uncertainty:** numerical tolerances; not every gate/output pair or full attention-to-gate integration. | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Preserved attention output gating and intermediate BF16 rounding in aligned M1/M8 execution. | Applies learned gates to the attention output. |
| **16. Attention output activation FP8 quantization** | 0.037 / 0.043 / 0.044 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260925.json). | **4/5 High for tested encoding and integration.** Independent FP8 codec checks plus current 320-token isolated M1/M8 and eager/compiled quantizer agreement at every attention layer. **Uncertainty:** the independent oracle checks remain sampled; arbitrary gate/quantization inputs and layouts are unproved. | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Enabled qualified native-rounding FP8 output production and rejected approximate traced quantization. | Converts the attention output to FP8 for its output projection. |
| **17. Attention output projection** | 0.588 / 0.536 / 0.516 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260925.json). | **4/5 High.** Current 496-matrix × 320-input sampled-channel rerun, plus all output channels of 30 selected matrices on 32 inputs each (7,739,392 outputs); 12 saved operator-activation replays add 3,276,800 outputs. **Uncertainty:** other matrices retain sampled-channel coverage; FP64 error criteria are not exact-arithmetic proofs; the separate 320-token stage replay uses full-model activations but shares the native operator, rather than an independent FP64 oracle. | [Qualified speed changes](docs/SPEED_INVESTIGATION_20260925.md): Use qualified local-split GEMM for M1/M8 down/output shapes; preserve the repaired MXFP4 coefficients and reduction order. | Maps the attention result back to the model's hidden-vector width. |
| **18. Post-attention/GDN residual/normalization + FP8 production** | 0.508 / 0.523 / 0.542 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260925.json). | **4/5 High.** Qualified release matches native M1 plus independent FP8 encoding in the 129-site × 320-row hidden-norm audit; original 55 byte differences eliminated. **Uncertainty:** finite inputs and prefill-versus-serial-decode behavior; deployed 24 September. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 reduction and BF16/FP8 rounding at every admitted prefill width, removing batch-dependent activation differences. | Adds the layer result to the residual, normalizes it and creates FP8 inputs for the MLP in one kernel. |
| ↳ MLP gate/up input FP8 quantization | Included in **stage 18** | Exact fused FP8 bytes/scales; see stage 18 and its evidence scope | **5/5 Very high for tested encoding.** Released bytes/scales match independent encoding of M1 BF16 norm; stage 18 prefill counterexamples now pass. **Uncertainty:** inherits normalization arithmetic and accepted FP8 loss; deployed 24 September. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 reduction and BF16/FP8 rounding at every admitted prefill width, removing batch-dependent activation differences. | Uses the FP8 output produced by the same post-normalization kernel. |
| **19. MLP gate/up projection** | 11.445 / 11.137 / 11.159 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260925.json). | **4/5 High.** Current 496-matrix × 320-input sampled-channel rerun, plus all output channels of 30 selected matrices on 32 inputs each (7,739,392 outputs); 12 saved operator-activation replays add 3,276,800 outputs. **Uncertainty:** other matrices retain sampled-channel coverage; FP64 error criteria are not exact-arithmetic proofs; the separate 320-token stage replay uses full-model activations but shares the native operator, rather than an independent FP64 oracle. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve stored MXFP4 coefficients in the folded lookup and repair fragment/tiled per-block fallbacks while retaining the qualified GEMM dispatch. | Computes both MLP input projections. Wider decode dispatch avoids the slower prefill kernel for qualified shapes. |
| **20. MLP SiLU and gating** | 0.172 / 0.186 / 0.190 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260925.json). | **4/5 High.** Every finite BF16 gate encoding checked with a bounded multiplier; 320 fused M1 cases pass. **Uncertainty:** tiny subnormal differences from FP64 remain; not all gate/up pairs are covered. | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Preserved intermediate BF16 rounding in aligned M1/M8 and eager/compiled arithmetic; the slower fused SiLU remains disabled. | Applies SiLU and combines the two MLP branches, retaining intermediate BF16 rounding. |
| **21. MLP down input FP8 quantization** | 0.250 / 0.263 / 0.271 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260925.json). | **5/5 Very high for tested encoding.** 320 fused M1 cases exactly match independent FP8 encoding of the native BF16 SiLU/gate pipeline. **Uncertainty:** inherits stage 20 numerical behavior; no exhaustive input-pair or shape proof. | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Enabled qualified native per-token FP8 production for the down projection input. | Converts MLP activations to FP8 for the down projection. |
| **22. MLP down projection** | 6.895 / 5.293 / 5.311 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260925.json). | **4/5 High.** Current 496-matrix × 320-input sampled-channel rerun, plus all output channels of 30 selected matrices on 32 inputs each (7,739,392 outputs); 12 saved operator-activation replays add 3,276,800 outputs. **Uncertainty:** other matrices retain sampled-channel coverage; FP64 error criteria are not exact-arithmetic proofs; the separate 320-token stage replay uses full-model activations but shares the native operator, rather than an independent FP64 oracle. | [Qualified speed changes](docs/SPEED_INVESTIGATION_20260925.md): Use qualified local-split GEMM for M1/M8 down/output shapes; preserve the repaired MXFP4 coefficients and reduction order. | Projects the MLP result back to the model's hidden-vector width. |
| **23. Final normalization/layout** | 0.012 / 0.013 / 0.013 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260925.json). | **4/5 High.** 320 inputs at the actual final BF16 norm pass independent RMS equations. **Uncertainty:** sampled numerical criteria; current complete-model mode agreement is sampled, not an independent reference proof. The extra FP8 control is not serving behavior. | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Aligned final normalization and full BF16 head precision boundaries while preserving the rounding contract. | Normalizes the final hidden vector before vocabulary scoring. |
| **24. Global-512 target head** | 1.032 / 1.045 / 1.079 | Same top-1: 12,015/12,015; complete reference top-20 retained: 12,015/12,015. Includes M1/M8; not an M1-versus-M8 ordering test · [current head study](benchmarks/results/head-candidate-depth-20260925.json). | **4/5 for score arithmetic; selection known approximate.** Three 512-row weight slabs × 320 inputs pass FP64 criteria. Current Global-512 study: top-1 12,015/12,015; complete top-20 12,015/12,015; top-40 12,000/12,015. **Uncertainty:** sampled scores, retained-logit differences and uncertified excluded tokens; these recall rows include M1 and M8, not M1 alone. [Head evidence](docs/HEAD_CANDIDATE_DEPTH.md). | [`9bb795d`](https://github.com/Terrydaktal/vllm-coherence/commit/9bb795d2e612c76087d16932841131edf4834d5e): increase target shortlist to 512; drafter unchanged. Extends the Global-256 method from [`31add3d`](https://github.com/Terrydaktal/vllm-coherence/commit/31add3d06080be6e4f5198c6e8afcc6276e1edaf). | Scores the vocabulary with INT2, selects 512 candidates and rescores them with BF16 weights. Selection remains approximate. |
| **25. Other GPU bookkeeping** | 0.479 / 0.489 / 0.506 | No isolated top-20 operator claim; sampling/state controls have separate evidence. | **3/5 Moderate for sampled decision/state checks.** 600,000 sampling trials; maximum probability error 0.118 percentage points; faulty shared-RNG control detected. **Uncertainty:** generic bookkeeping kernels and every asynchronous commit/rollback path are not independently qualified. | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Integrated qualified sampling, cache/state bookkeeping and graph-capture admission; no isolated correctness claim is made for this aggregate row. | Runs sampling and cache/state update kernels outside the named model stages. |
| **26. Estimated runtime overhead** | 0.819 / 2.336 / 2.353 (estimate) | [Matched control minus GPU activity union](benchmarks/results/matched-stage-residual-20260925.json) | **N/A: timing estimate.** No model arithmetic to score. **Uncertainty:** performance residual is not evidence of scheduler or session-state correctness. | [qualified speed refresh](docs/SPEED_INVESTIGATION_20260925.md): record matched unprofiled controls and overlap-corrected residuals | Indirect observer effects are not proved zero. |
| **Total reconstructed round (stages 1–26)** | **37.424 / 41.262 / 51.008** | GPU activity union plus the estimated remainder | **N/A: no aggregate correctness score.** Operator scores cannot be averaged into a model guarantee. **Uncertainty:** full-model prefill/M1 equality, candidate completeness and exhaustive restore/concurrency qualification remain open. | — | Overlapping stages are counted once in the total. |

The table restores the historical grouped 26 measured-row layout and adds a total row. Each timing cell is ordered **0K / 60K / 200K**. Fused kernels are charged once to their containing stage; the ↳ rows are detail-only inclusion records and add no timing; rows without a separate profiler scope are labelled in the timing cell rather than displayed as 0.000. The total counts overlapping GPU activity once. The old forced-replay subtraction is superseded; [its audit](benchmarks/results/stage-timing-audit-20260923.json) remains available.

Each **set/order** pair means the same top-20 token set, followed by the same ranking. Current stage confirmations identify M1/M8 and eager/compiled M8 separately. Each position passes only if every layer instance passes on the same captured inputs. These diagnostic stage replays are checked against the compiled graph control; their times are not used in this table. Timing and correctness captures record their own exact source hashes. The provenance column identifies the last relevant code change.

Fusion still permits correctness instrumentation: a diagnostic kernel can expose intermediate values, and fused outputs can be compared with an unfused reference. The normal GPU profile measures the combined kernel. Internal probes or splitting the kernel can change its performance, so those measurements are not an additive breakdown of the production kernel's time.

The expandable layer and kernel tables use the same 1,122 retained 60K cycles as the main table (2026-09-25). GPU activity crossing a worker boundary is clipped to that boundary; activity-record counts include these fragments. CPU profiling work is excluded. Cycle counts are not output-token counts.

<details>
<summary>Current compiled 60K decoder-layer detail: projection and remaining-work timings</summary>

Finish each row's layer, including its MLP, before moving to the next row. Each layer has four projections. Gate and up are one joint GEMM; there is no separately measured gate/up split. The remaining-work column combines normalization, mixing and other operations between those projections.

| Layer | Type | All layer work ms | Input projection ms | Output projection ms | Gate/up projection ms | Down projection ms | Other work ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | GDN | 0.4247 | 0.0768 | 0.0367 | 0.1723 | 0.0798 | 0.0591 |
| 1 | GDN | 0.4431 | 0.0820 | 0.0387 | 0.1807 | 0.0858 | 0.0559 |
| 2 | GDN | 0.4515 | 0.0850 | 0.0391 | 0.1819 | 0.0861 | 0.0593 |
| 3 | Attention | 0.7048 | 0.0717 | 0.0335 | 0.1640 | 0.0797 | 0.3559 |
| 4 | GDN | 0.4301 | 0.0827 | 0.0369 | 0.1721 | 0.0804 | 0.0580 |
| 5 | GDN | 0.4419 | 0.0819 | 0.0383 | 0.1779 | 0.0858 | 0.0579 |
| 6 | GDN | 0.4518 | 0.0841 | 0.0383 | 0.1815 | 0.0878 | 0.0600 |
| 7 | Attention | 0.7023 | 0.0720 | 0.0334 | 0.1642 | 0.0797 | 0.3530 |
| 8 | GDN | 0.4305 | 0.0825 | 0.0368 | 0.1728 | 0.0800 | 0.0584 |
| 9 | GDN | 0.4415 | 0.0817 | 0.0382 | 0.1797 | 0.0845 | 0.0574 |
| 10 | GDN | 0.4499 | 0.0843 | 0.0390 | 0.1803 | 0.0860 | 0.0603 |
| 11 | Attention | 0.7035 | 0.0718 | 0.0335 | 0.1642 | 0.0798 | 0.3543 |
| 12 | GDN | 0.4281 | 0.0817 | 0.0369 | 0.1722 | 0.0794 | 0.0579 |
| 13 | GDN | 0.4407 | 0.0814 | 0.0384 | 0.1792 | 0.0842 | 0.0575 |
| 14 | GDN | 0.4493 | 0.0845 | 0.0378 | 0.1814 | 0.0858 | 0.0598 |
| 15 | Attention | 0.7036 | 0.0720 | 0.0336 | 0.1641 | 0.0801 | 0.3537 |
| 16 | GDN | 0.4300 | 0.0818 | 0.0369 | 0.1727 | 0.0800 | 0.0586 |
| 17 | GDN | 0.4434 | 0.0812 | 0.0384 | 0.1806 | 0.0855 | 0.0576 |
| 18 | GDN | 0.4477 | 0.0836 | 0.0389 | 0.1792 | 0.0859 | 0.0601 |
| 19 | Attention | 0.7017 | 0.0708 | 0.0337 | 0.1636 | 0.0794 | 0.3542 |
| 20 | GDN | 0.4314 | 0.0821 | 0.0367 | 0.1726 | 0.0803 | 0.0598 |
| 21 | GDN | 0.4417 | 0.0817 | 0.0378 | 0.1790 | 0.0854 | 0.0579 |
| 22 | GDN | 0.4479 | 0.0843 | 0.0389 | 0.1795 | 0.0846 | 0.0607 |
| 23 | Attention | 0.7039 | 0.0727 | 0.0336 | 0.1640 | 0.0797 | 0.3540 |
| 24 | GDN | 0.4301 | 0.0822 | 0.0368 | 0.1724 | 0.0799 | 0.0588 |
| 25 | GDN | 0.4458 | 0.0816 | 0.0383 | 0.1813 | 0.0862 | 0.0584 |
| 26 | GDN | 0.4484 | 0.0840 | 0.0381 | 0.1796 | 0.0865 | 0.0600 |
| 27 | Attention | 0.7053 | 0.0716 | 0.0334 | 0.1636 | 0.0803 | 0.3564 |
| 28 | GDN | 0.4306 | 0.0825 | 0.0368 | 0.1725 | 0.0805 | 0.0584 |
| 29 | GDN | 0.4398 | 0.0828 | 0.0377 | 0.1763 | 0.0846 | 0.0584 |
| 30 | GDN | 0.4470 | 0.0840 | 0.0388 | 0.1793 | 0.0847 | 0.0602 |
| 31 | Attention | 0.7052 | 0.0721 | 0.0336 | 0.1644 | 0.0796 | 0.3555 |
| 32 | GDN | 0.4308 | 0.0820 | 0.0369 | 0.1724 | 0.0804 | 0.0592 |
| 33 | GDN | 0.4453 | 0.0822 | 0.0377 | 0.1812 | 0.0855 | 0.0586 |
| 34 | GDN | 0.4490 | 0.0843 | 0.0381 | 0.1783 | 0.0854 | 0.0629 |
| 35 | Attention | 0.7078 | 0.0717 | 0.0335 | 0.1643 | 0.0800 | 0.3583 |
| 36 | GDN | 0.4311 | 0.0824 | 0.0368 | 0.1724 | 0.0809 | 0.0586 |
| 37 | GDN | 0.4406 | 0.0828 | 0.0380 | 0.1772 | 0.0835 | 0.0590 |
| 38 | GDN | 0.4469 | 0.0834 | 0.0380 | 0.1792 | 0.0861 | 0.0601 |
| 39 | Attention | 0.7079 | 0.0730 | 0.0336 | 0.1642 | 0.0800 | 0.3572 |
| 40 | GDN | 0.4318 | 0.0829 | 0.0369 | 0.1727 | 0.0804 | 0.0589 |
| 41 | GDN | 0.4459 | 0.0817 | 0.0379 | 0.1817 | 0.0856 | 0.0590 |
| 42 | GDN | 0.4522 | 0.0859 | 0.0383 | 0.1804 | 0.0867 | 0.0609 |
| 43 | Attention | 0.7054 | 0.0717 | 0.0336 | 0.1644 | 0.0801 | 0.3556 |
| 44 | GDN | 0.4315 | 0.0829 | 0.0368 | 0.1726 | 0.0801 | 0.0590 |
| 45 | GDN | 0.4428 | 0.0818 | 0.0377 | 0.1784 | 0.0860 | 0.0590 |
| 46 | GDN | 0.4487 | 0.0841 | 0.0379 | 0.1794 | 0.0867 | 0.0606 |
| 47 | Attention | 0.7065 | 0.0728 | 0.0334 | 0.1647 | 0.0798 | 0.3557 |
| 48 | GDN | 0.4339 | 0.0826 | 0.0369 | 0.1726 | 0.0812 | 0.0606 |
| 49 | GDN | 0.4461 | 0.0829 | 0.0382 | 0.1815 | 0.0847 | 0.0588 |
| 50 | GDN | 0.4499 | 0.0853 | 0.0384 | 0.1797 | 0.0866 | 0.0599 |
| 51 | Attention | 0.7067 | 0.0717 | 0.0334 | 0.1644 | 0.0799 | 0.3575 |
| 52 | GDN | 0.4286 | 0.0817 | 0.0367 | 0.1721 | 0.0792 | 0.0588 |
| 53 | GDN | 0.4413 | 0.0820 | 0.0384 | 0.1781 | 0.0846 | 0.0583 |
| 54 | GDN | 0.4469 | 0.0839 | 0.0389 | 0.1797 | 0.0836 | 0.0608 |
| 55 | Attention | 0.7061 | 0.0725 | 0.0335 | 0.1639 | 0.0799 | 0.3564 |
| 56 | GDN | 0.4306 | 0.0820 | 0.0368 | 0.1729 | 0.0798 | 0.0592 |
| 57 | GDN | 0.4435 | 0.0816 | 0.0383 | 0.1805 | 0.0841 | 0.0589 |
| 58 | GDN | 0.4499 | 0.0851 | 0.0390 | 0.1797 | 0.0850 | 0.0611 |
| 59 | Attention | 0.7071 | 0.0720 | 0.0334 | 0.1642 | 0.0797 | 0.3577 |
| 60 | GDN | 0.4315 | 0.0823 | 0.0369 | 0.1726 | 0.0801 | 0.0597 |
| 61 | GDN | 0.4417 | 0.0819 | 0.0375 | 0.1785 | 0.0851 | 0.0588 |
| 62 | GDN | 0.4509 | 0.0840 | 0.0389 | 0.1801 | 0.0842 | 0.0638 |
| 63 | Attention | 0.7064 | 0.0725 | 0.0335 | 0.1640 | 0.0798 | 0.3566 |

</details>

<details>
<summary>Current 60K GPU activity, grouped by stage</summary>

| Stage / compiled kernel | Activity records in 1,122 retained cycles | Current GPU ms per retained 60K cycle |
| --- | ---: | ---: |
| Drafter / `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x32_MI16x16x1_SN_LDSB1_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_CD1_1_CLR0_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB1_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA1201_ICIW0_IU1_K1_LDSTI0_LBSPPA128_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB0_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB2_ONLL1_PAP0_PGL0_PGR1_PLR1_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1.kd` | 1122 | 0.006418 |
| Drafter / `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x32x256_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB1_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA1201_ICIW0_IU1_K1_LDSTI0_LBSPPA512_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB0_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB16_ONLL0_PAP0_PGL0_PGR1_PLR1_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_4_1.kd` | 1124 | 0.177106 |
| Drafter / `__amd_rocclr_copyBuffer.kd` | 1122 | 0.001909 |
| Drafter / `__amd_rocclr_fillBufferAligned.kd` | 3366 | 0.003653 |
| Drafter / `_cache_draft_logits_kernel.kd` | 1122 | 0.001878 |
| Drafter / `_draft_head_int2.kd` | 1122 | 0.836469 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_17408_EVEN_K_1_GRID_MN_40_cache_modifier_NONE.kd` | 5649 | 0.734742 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_25600_EVEN_K_1_GRID_MN_40_cache_modifier_NONE.kd` | 1123 | 0.235520 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_40_cache_modifier_NONE.kd` | 5718 | 0.198023 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_5120_EVEN_K_1_GRID_MN_272_cache_modifier_NONE.kd` | 5788 | 1.444349 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_5120_EVEN_K_1_GRID_MN_48_cache_modifier_NONE.kd` | 5629 | 0.269691 |
| Drafter / `_prepare_dflash_inputs_kernel.kd` | 1122 | 0.007746 |
| Drafter / `_rerank_exact.kd` | 1122 | 0.008138 |
| Drafter / `_selector_walk_kernel.kd` | 1122 | 0.008154 |
| Drafter / `kernel_unified_attention.kd` | 6096 | 0.503309 |
| Drafter / `reshape_and_cache_kernel_flash.kd` | 11230 | 0.023862 |
| Drafter / `triton_per_fused_4.kd` | 1127 | 0.001490 |
| Drafter / `triton_per_fused_8.kd` | 4488 | 0.005076 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_preshuffle_gemm_squeeze_view_0.kd` | 5618 | 0.007212 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_preshuffle_gemm_squeeze_view_2.kd` | 1122 | 0.001434 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_preshuffle_gemm_squeeze_view_4.kd` | 10104 | 0.011095 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_view_0.kd` | 1122 | 0.005637 |
| Drafter / `triton_poi_fused_0.kd` | 1122 | 0.001938 |
| Drafter / `triton_poi_fused_5.kd` | 1134 | 0.001705 |
| Drafter / `triton_poi_fused_9.kd` | 4488 | 0.007436 |
| Drafter / `triton_poi_fused__to_copy_clamp_div_preshuffle_gemm_squeeze_view_1.kd` | 5616 | 0.008729 |
| Drafter / `triton_poi_fused__to_copy_clamp_div_preshuffle_gemm_squeeze_view_3.kd` | 1122 | 0.001528 |
| Drafter / `triton_poi_fused__to_copy_clamp_div_preshuffle_gemm_squeeze_view_5.kd` | 10100 | 0.011847 |
| Drafter / `triton_poi_fused_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_3.kd` | 10103 | 0.026028 |
| Drafter / `triton_poi_fused_add_arange_bitwise_and_constant_pad_nd_ge_mul_rms_norm_select_slice_unsqueeze_view_1.kd` | 1122 | 0.002257 |
| Drafter / `triton_poi_fused_add_permute_unsqueeze_view_2.kd` | 1122 | 0.001287 |
| Drafter / `triton_poi_fused_cat_expand_index_mul_slice_unsqueeze_view_1.kd` | 1122 | 0.001917 |
| Drafter / `triton_red_fused__to_copy_abs_clamp_div_max_mul_preshuffle_gemm_silu_slice_squeeze_view_6.kd` | 5613 | 0.015681 |
| Drafter / `triton_red_fused__to_copy_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_w4_gemm_2.kd` | 5629 | 0.028762 |
| Drafter / `triton_red_fused__to_copy_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_w4_gemm_7.kd` | 4488 | 0.023947 |
| Drafter / `triton_red_fused__to_copy_embedding_mul_rms_norm_w4_gemm_0.kd` | 1122 | 0.003880 |
| Drafter / `triton_red_fused_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_7.kd` | 1122 | 0.005905 |
| Drafter / `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) [clone .kd]` | 2244 | 0.008901 |
| Drafter / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 1122 | 0.003013 |
| Drafter / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 1122 | 0.002670 |
| Drafter / `void at::native::bitonicSortKVInPlace<2, -1, 16, 16, c10::BFloat16, long, at::native::GTOp<c10::BFloat16, true>, unsigned int>(at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, at::native::GTOp<c10::BFloat16, true>) [clone .kd]` | 1122 | 0.002652 |
| Drafter / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 1122 | 0.004488 |
| Drafter / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 1122 | 0.002652 |
| Drafter / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 1122 | 0.003589 |
| Drafter / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 1122 | 0.001939 |
| Drafter / `void at::native::mbtopk::computeBlockDigitCounts<c10::BFloat16, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 2244 | 0.037886 |
| Drafter / `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 4488 | 0.051837 |
| Drafter / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, c10::BFloat16>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 2244 | 0.017835 |
| Drafter / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, float>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, float*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 4488 | 0.026701 |
| Drafter / `void at::native::mbtopk::fill<unsigned int, unsigned int>(unsigned int*, unsigned int, unsigned int) [clone .kd]` | 2244 | 0.002004 |
| Drafter / `void at::native::mbtopk::gatherTopK<c10::BFloat16, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 1122 | 0.020436 |
| Drafter / `void at::native::mbtopk::gatherTopK<float, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, float*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 1122 | 0.010197 |
| Drafter / `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4>) [clone .kd]` | 1122 | 0.002555 |
| Drafter / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul>) [clone .kd]` | 1122 | 0.001188 |
| Drafter / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) [clone .kd]` | 1122 | 0.001953 |
| Drafter / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>) [clone .kd]` | 2244 | 0.002545 |
| Drafter / `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>) [clone .kd]` | 2244 | 0.004110 |
| Drafter / `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool) [clone .kd]` | 1122 | 0.001590 |
| Drafter / `void at::native::warpMergeSortKVInPlace<2, -1, 128, 16, float, long, at::native::GTOp<float, true>, unsigned int, 32>(at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, at::native::GTOp<float, true>, float) [clone .kd]` | 1122 | 0.004388 |
| Drafter / `void r4d_gemm_w4a16_nt_m64_kernel<1, 1, false>(unsigned short const*, unsigned int const*, unsigned int const*, __hip_bfloat16*, int, int, int, int, int) [clone .kd]` | 1122 | 0.003998 |
| Drafter / `void r4d_gemm_w4a16_nt_m64_kernel<1, 1, true>(unsigned short const*, unsigned int const*, unsigned int const*, __hip_bfloat16*, int, int, int, int, int) [clone .kd]` | 11240 | 0.075219 |
| Drafter / `void vllm::rms_norm_kernel<c10::BFloat16, 8, 2, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, long, long, long, c10::BFloat16 const*, long, float, int, int) [clone .kd]` | 1122 | 0.003069 |
| Drafter / `void vllm::rms_norm_kernel<c10::BFloat16, 8, 4, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, long, long, long, c10::BFloat16 const*, long, float, int, int) [clone .kd]` | 1122 | 0.003027 |
| Drafter / `void vllm::rotary_embedding_kernel<c10::BFloat16, c10::BFloat16, true>(long const*, c10::BFloat16*, c10::BFloat16*, c10::BFloat16 const*, int, long, long, long, int, int, int, long, bool) [clone .kd]` | 1122 | 0.002838 |
| Embedding + first input normalization / `triton_poi_fused__to_copy_embedding_0.kd` | 1122 | 0.002241 |
| Embedding + first input normalization / `void norm_quant<false, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned char*, float*, unsigned short*, long, long, float) [clone .kd]` | 1122 | 0.007022 |
| GDN layout/copies and buffer initialization / `triton_poi_fused_add_1.kd` | 3366 | 0.003886 |
| GDN layout/copies and buffer initialization / `triton_poi_fused_add_2.kd` | 2244 | 0.002588 |
| GDN layout/copies and buffer initialization / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 53856 | 0.088503 |
| GDN layout/copies and buffer initialization / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) [clone .kd]` | 53856 | 0.060545 |
| GDN layout/copies and buffer initialization / `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>) [clone .kd]` | 53856 | 0.044336 |
| GDN convolution / `_causal_conv1d_update_kernel.kd` | 53856 | 0.184674 |
| GDN recurrence and gates / `stock_gdn_scan_kernel.kd` | 53856 | 1.202298 |
| GDN output gated normalization / `gdn_norm_quant_kernel.kd` | 53856 | 0.150874 |
| Post-attention/GDN residual/normalization / `void norm_quant<true, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned char*, float*, unsigned short*, long, long, float) [clone .kd]` | 71808 | 0.523027 |
| MLP SiLU and gating / `triton_poi_fused_dynamic_per_token_scaled_fp8_quant_mul_silu_slice_0.kd` | 53856 | 0.131564 |
| MLP SiLU and gating / `triton_poi_fused_dynamic_per_token_scaled_fp8_quant_mul_silu_slice_1.kd` | 17952 | 0.054345 |
| MLP down input FP8 quantization / `void vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<c10::BFloat16, c10::Float8_e4m3fn>(c10::Float8_e4m3fn*, float*, c10::BFloat16 const*, float const*, int, long, long) [clone .kd]` | 71808 | 0.262962 |
| GDN input projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 1, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 53856 | 3.972270 |
| GDN output projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 4, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 53856 | 1.815575 |
| MLP gate/up projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 1, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 71808 | 11.136968 |
| MLP down projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 4, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 71808 | 5.292604 |
| Layer input residual/normalization / `void norm_quant<true, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned char*, float*, unsigned short*, long, long, float) [clone .kd]` | 70685 | 0.507439 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_1.kd` | 17952 | 0.021487 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_3.kd` | 17952 | 0.026732 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_4.kd` | 17952 | 0.054170 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_arange_bitwise_and_eq_index_lt_remainder_select_split_where_2.kd` | 17952 | 0.028810 |
| Attention Q/K normalization, RoPE and layout / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 17952 | 0.028985 |
| Attention Q/K normalization, RoPE and layout / `void stock_m1_gemma_norm<false, 256, 32>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, long, long, float, float*) [clone .kd]` | 17952 | 0.047233 |
| Attention Q/K normalization, RoPE and layout / `void stock_m1_gemma_norm<false, 256, 64>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, long, long, float, float*) [clone .kd]` | 17952 | 0.037267 |
| Attention KV write / `reshape_and_cache_kernel_flash.kd` | 17952 | 0.035042 |
| Attention decode / `void qwen_stock_m1_shared_decode<4, 16, 256, 6, 16, 0, 3430931>(R4DArgs, int) [clone .kd]` | 17952 | 4.822588 |
| Attention split-KV merge / `void qwen_stock_m1_shared_merge<256, 4, 0>(R4DArgs, int, int) [clone .kd]` | 17952 | 0.131960 |
| Attention output gating / `triton_poi_fused_dynamic_per_token_scaled_fp8_quant_mul_sigmoid_view_0.kd` | 17952 | 0.033038 |
| Attention output activation FP8 quantization / `void vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<c10::BFloat16, c10::Float8_e4m3fn>(c10::Float8_e4m3fn*, float*, c10::BFloat16 const*, float const*, int, long, long) [clone .kd]` | 17952 | 0.042852 |
| Attention input projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 1, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 17952 | 1.152481 |
| Attention output projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 4, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 17952 | 0.536164 |
| Final normalization/layout / `__amd_rocclr_copyBuffer.kd` | 6732 | 0.007596 |
| Final normalization/layout / `void stock_m1_gemma_norm<true, 5120, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, long, long, float, float*) [clone .kd]` | 1122 | 0.005185 |
| Other GPU bookkeeping / `__amd_rocclr_copyBuffer.kd` | 43765 | 0.078032 |
| Other GPU bookkeeping / `__amd_rocclr_fillBufferAligned.kd` | 1122 | 0.002453 |
| Other GPU bookkeeping / `_apply_write_kernel.kd` | 5 | 0.000026 |
| Other GPU bookkeeping / `_combine_sampled_and_draft_tokens_kernel.kd` | 1122 | 0.002987 |
| Other GPU bookkeeping / `_compute_local_logits_stats_kernel.kd` | 1122 | 0.028410 |
| Other GPU bookkeeping / `_compute_slot_mappings_kernel.kd` | 1122 | 0.003018 |
| Other GPU bookkeeping / `_expand_idx_mapping_kernel.kd` | 1122 | 0.001978 |
| Other GPU bookkeeping / `_gather_block_tables_kernel.kd` | 1122 | 0.004907 |
| Other GPU bookkeeping / `_get_num_sampled_and_rejected_kernel.kd` | 1122 | 0.003010 |
| Other GPU bookkeeping / `_insert_resampled_kernel.kd` | 1122 | 0.003502 |
| Other GPU bookkeeping / `_post_update_kernel.kd` | 1122 | 0.005248 |
| Other GPU bookkeeping / `_prepare_pos_seq_lens_kernel.kd` | 1122 | 0.002140 |
| Other GPU bookkeeping / `_prepare_rope_positions_kernel.kd` | 1122 | 0.002927 |
| Other GPU bookkeeping / `_rejection_kernel.kd` | 1122 | 0.007538 |
| Other GPU bookkeeping / `_resample_kernel.kd` | 1122 | 0.014577 |
| Other GPU bookkeeping / `_scatter_num_accepted_kernel.kd` | 1122 | 0.002109 |
| Other GPU bookkeeping / `_zero_kv_blocks_kernel.kd` | 2 | 0.000119 |
| Other GPU bookkeeping / `postprocess_mamba_fused_kernel.kd` | 1122 | 0.004124 |
| Other GPU bookkeeping / `precopy_mamba_align_fused_kernel.kd` | 1122 | 0.004024 |
| Other GPU bookkeeping / `preprocess_mamba_align_fused_kernel.kd` | 1122 | 0.002555 |
| Other GPU bookkeeping / `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}>(int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}, function_traits<at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}>::result_type*) [clone .kd]` | 6732 | 0.009223 |
| Other GPU bookkeeping / `void (anonymous namespace)::softmax_warp_forward<float, float, float, 6, false, false, 32>(float*, float const*, int, int, int, bool const*, int, bool) [clone .kd]` | 1122 | 0.002281 |
| Other GPU bookkeeping / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 7853 | 0.013458 |
| Other GPU bookkeeping / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 1122 | 0.002582 |
| Other GPU bookkeeping / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 7854 | 0.014161 |
| Other GPU bookkeeping / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<int> const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<int> const&)::{lambda(int, bool)#1}) [clone .kd]` | 6732 | 0.014904 |
| Other GPU bookkeeping / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<float> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<float> const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<float> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<float> const&)::{lambda(int, bool)#1}) [clone .kd]` | 3366 | 0.019272 |
| Other GPU bookkeeping / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 5610 | 0.017244 |
| Other GPU bookkeeping / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 2244 | 0.006175 |
| Other GPU bookkeeping / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 1122 | 0.003146 |
| Other GPU bookkeeping / `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 4488 | 0.060045 |
| Other GPU bookkeeping / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, float>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, float*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 4488 | 0.037303 |
| Other GPU bookkeeping / `void at::native::mbtopk::fill<unsigned int, unsigned int>(unsigned int*, unsigned int, unsigned int) [clone .kd]` | 1122 | 0.001595 |
| Other GPU bookkeeping / `void at::native::mbtopk::gatherTopK<float, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, float*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 1122 | 0.026314 |
| Other GPU bookkeeping / `void at::native::tensor_kernel_scan_innermost_dim<float, std::plus<float> >(float*, float const*, unsigned int, unsigned int, unsigned int, float, std::plus<float>) [clone .kd]` | 1122 | 0.002645 |
| Other GPU bookkeeping / `void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>, 4, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast) [clone .kd]` | 1122 | 0.001751 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<16, at::native::BinaryFunctor<bool, bool, bool, at::native::BitwiseOrFunctor<bool> >, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<bool, bool, bool, at::native::BitwiseOrFunctor<bool> >, std::array<char*, 3ul>) [clone .kd]` | 1122 | 0.002602 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<16, at::native::FillFunctor<bool>, std::array<char*, 1ul> >(int, at::native::FillFunctor<bool>, std::array<char*, 1ul>) [clone .kd]` | 1121 | 0.001855 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<16, at::native::bitwise_not_kernel_cuda(at::TensorIteratorBase&)::{lambda(bool)#1}, std::array<char*, 2ul> >(int, at::native::bitwise_not_kernel_cuda(at::TensorIteratorBase&)::{lambda(bool)#1}, std::array<char*, 2ul>) [clone .kd]` | 1122 | 0.002398 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}, std::array<char*, 2ul>) [clone .kd]` | 6732 | 0.010300 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}, std::array<char*, 2ul>) [clone .kd]` | 1122 | 0.001831 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}, std::array<char*, 3ul> >(int, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}, std::array<char*, 3ul>) [clone .kd]` | 1122 | 0.010106 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool, float, float)#1}, std::array<char*, 4ul> >(int, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool, float, float)#1}, std::array<char*, 4ul>) [clone .kd]` | 2244 | 0.004689 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 2ul> >(int, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 2ul>) [clone .kd]` | 6732 | 0.011140 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, 2ul>) [clone .kd]` | 7854 | 0.012245 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul>) [clone .kd]` | 1122 | 0.002051 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>) [clone .kd]` | 1122 | 0.001987 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>) [clone .kd]` | 2244 | 0.003280 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int, at::native::FillFunctor<int>, std::array<char*, 1ul>) [clone .kd]` | 1122 | 0.001644 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>) [clone .kd]` | 1122 | 0.009804 |
| Other GPU bookkeeping / `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool) [clone .kd]` | 1122 | 0.002580 |
| Other GPU bookkeeping / `void at::native::warpMergeSortKVInPlace<2, -1, 128, 16, float, long, at::native::GTOp<float, true>, unsigned int, 32>(at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, at::native::GTOp<float, true>, float) [clone .kd]` | 1122 | 0.004737 |
| Target head (global512) / `__amd_rocclr_fillBufferAligned.kd` | 1122 | 0.002569 |
| Target head (global512) / `_draft_head_int2.kd` | 1122 | 0.775613 |
| Target head (global512) / `_rerank_exact.kd` | 1122 | 0.054367 |
| Target head (global512) / `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) [clone .kd]` | 1122 | 0.004833 |
| Target head (global512) / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 1122 | 0.003128 |
| Target head (global512) / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 1122 | 0.002909 |
| Target head (global512) / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 1122 | 0.005268 |
| Target head (global512) / `void at::native::mbtopk::computeBlockDigitCounts<c10::BFloat16, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 2244 | 0.091936 |
| Target head (global512) / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, c10::BFloat16>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 2244 | 0.041150 |
| Target head (global512) / `void at::native::mbtopk::fill<unsigned int, unsigned int>(unsigned int*, unsigned int, unsigned int) [clone .kd]` | 1122 | 0.001733 |
| Target head (global512) / `void at::native::mbtopk::gatherTopK<c10::BFloat16, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 1122 | 0.042890 |
| Target head (global512) / `void at::native::radixSortKVInPlace<2, -1, 128, 8, c10::BFloat16, long, unsigned int>(at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, bool) [clone .kd]` | 1122 | 0.005823 |
| Target head (global512) / `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4>) [clone .kd]` | 1122 | 0.003920 |
| Target head (global512) / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) [clone .kd]` | 1122 | 0.001737 |
| Target head (global512) / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>) [clone .kd]` | 1122 | 0.002075 |
| Target head (global512) / `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>) [clone .kd]` | 2244 | 0.005280 |

</details>

## Global-512 target-head

Global-512 is the serving default. This paired comparison used identical hidden inputs for every head method from two natural completions starting with the retained 60K Pi prefix: coding: 60,208 input and 6,234 output tokens (stop); reasoning: 66,589 input and 6,746 output tokens (stop). Sampling was temperature 1, top-p 0.95 and top-k 40.

| Target path | Median M8 head time | Same top-1 token | Complete reference top-20 retained | Complete reference top-40 retained |
|---|---:|---:|---:|---:|
| Global INT2 top-256 + BF16 rerank | 1.187 ms | 12,015/12,015 (100%) | 12,006/12,015 (99.9251%) | 11,811/12,015 (98.3021%) |
| Global INT2 top-512 + BF16 rerank (default) | 1.213 ms | 12,015/12,015 (100%) | 12,015/12,015 (100%) | 12,000/12,015 (99.8752%) |
| Full BF16 reference | 4.059 ms | 12,015/12,015 (100%) | 12,015/12,015 (100%) | 12,015/12,015 (100%) |

The comparison covers **12,015 prediction rows**, including prefill and rejected speculative rows, not 12,015 generated tokens. Timing uses 47 eight-row hidden inputs with five randomized-order repetitions: 235 measurements per method. These isolated head timings include native dispatch gaps and exclude comparison/reporting; they do not measure whole-round time or tok/s.

Incomplete top-40 retention occurred in 204 rows with Global-256 and 15 with Global-512; the added median head time was 0.026 ms. Retention includes cutoff ties and does not establish score equality, ordering or identical sampling probabilities. Both shortlists remain approximate; the full BF16 head is the reference for this comparison, not an independently proved model. The drafter is unchanged.

[Methodology and limits](docs/HEAD_CANDIDATE_DEPTH.md) · [Numeric results](benchmarks/results/head-candidate-depth-20260925.json) · [Earlier Global-256 study](docs/VERIFY_HEAD_GLOBAL_TOPK.md).

## Benchmarks

This benchmark uses the retained 60,000-input-token Pi prefix and the compiled Coherence backend with global512, the attention-page-boundary repair and the pinned-RAM huge-page promotion repair. Five requests are chained in one context: code, prose about code measurement, JSON, thinking/prose and checkpoint generation. Code and JSON disable thinking; both prose requests enable it. All requests stop naturally. Generation uses temperature 1, top-p 0.95, top-k 40 and seed 0; compaction uses temperature 0.3. The checkpoint request forces a snapshot-tail flush. Private fixture text was not decoded or inspected. Public results contain aggregates and hashes, not chat text or token arrays. The shared suite retains sealed continuation tokens privately for resumability. Runner: [benchmark_pi_coding_contexts.py --suite](experiments/radiance-public/benchmark_pi_coding_contexts.py).

| Stage | Thinking | Prompt tokens | Generated tokens | Classified output | First data | Mean round | Post-first | Peak 3s | Acceptance |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Coding task | off | 60,208 | 5,173 | 4,641 code; 531 prose; 0 separately observed reasoning | 2.01 s | 41.16 ms | 103.02 tok/s | 143.06 tok/s | 46.29% |
| Prose about code measurement | on | 65,522 | 2,845 | 2,846 prose; 0 separately observed reasoning | 2.49 s | 41.98 ms | 74.58 tok/s | 121.49 tok/s | 30.43% |
| JSON task | off | 68,486 | 2,409 | 2,408 JSON; valid JSON | 2.36 s | 41.67 ms | 89.26 tok/s | 103.24 tok/s | 38.82% |
| Thinking/prose task | on | 71,042 | 6,613 | 6,612 prose; 0 separately observed reasoning | 1.90 s | 41.99 ms | 59.81 tok/s | 80.14 tok/s | 21.59% |
| Compaction checkpoint | off | 77,841 | 4,282 | 4,282 checkpoint tokens; completion marker valid; required headings valid | 2.11 s | 42.43 ms | 67.03 tok/s | 93.83 tok/s | 26.33% |

Cached/total prompt tokens: Coding task: 57,680/60,208, Prose about code measurement: 62,624/65,522, JSON task: 65,920/68,486, Thinking/prose task: 69,216/71,042, Compaction checkpoint: 75,808/77,841.

The phase counters retokenize classified text, so their totals can differ from the backend's emitted-token count. `phase_token_counts_cover_output=false` for Coding task, Prose about code measurement, JSON task, Thinking/prose task, Compaction checkpoint. No separate reasoning channel was exposed for Prose about code measurement, Thinking/prose task; those streams are reported as prose, not relabelled as reasoning. The compaction row measures checkpoint generation with a requested cache flush, not a full Pi transcript commit or old-snapshot retirement. The checkpoint format passed its heading and completion checks. `peak_3s_tokens_per_second` is the maximum completed three-second sliding-window rate after first data, never a single-frame burst.

2026-09-25 rerun status: `complete`. [Numeric results and release identity](benchmarks/results/pi-coding-json-compaction.json). These results use top-k 40, while the older September 20 table used top-k 20, so the output and acceptance changes are not a controlled before/after comparison.

This is the same natural-stop coding task run independently at empty, 60K and 200K input context. Thinking is disabled, EOS remains enabled, and each arm uses temperature 1, top-p 0.95 and top-k 40. The non-empty arms use operator-supplied token-prefix fixtures; only their hashes are published. The three-second peak is the maximum completed sliding-window rate, not a single-frame burst. Runner: [benchmark_pi_coding_contexts.py](experiments/radiance-public/benchmark_pi_coding_contexts.py).

| Context | Prompt tokens | Generated tokens | Mean round | Post-first | Peak 3s | Acceptance | Round events (total; speculative/expected; timed) | Validation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0K | 203 | 4,913 | 37.65 ms | 137.25 tok/s | 171.50 tok/s | 59.59% | 951 total; 950/950 speculative; 950 timed; captured | short natural stop |
| 60K | 60,208 | 5,173 | 41.16 ms | 103.02 tok/s | 143.06 tok/s | 46.29% | 1,221 total; 1220/1220 speculative; 1,220 timed; captured | pass |
| 200K | 200,208 | 3,912 | 51.00 ms | 91.62 tok/s | 115.18 tok/s | 52.40% | 839 total; 838/838 speculative; 838 timed; captured | short natural stop |

Report status: `complete_with_validation_failure`. [Numeric results and every round](benchmarks/results/pi-coding-contexts.json). Each completed row stores every scheduler event under `contexts.<context>.round_capture.records`; a count mismatch is a validation failure. The target is 5,000 output tokens, with shorter natural completions reported explicitly.

These results use [the shared benchmark suite](docs/BENCHMARK_SUITE.md): `benchmark_pi_coding_contexts.py --suite` reuses each context's predetermined unprofiled control for its coding row and complete histogram, and continues the same 60K output through prose, JSON, thinking and compaction. It keeps both controls around each stage trace for the residual calculation. The 60K coding row above and the chained coding row are the same measured request.

### Complete per-round capture

The context benchmark now retains every content-free scheduler event for each arm, including an unmeasured first event. The full numeric records are stored under `contexts.<context>.round_capture.records` in the result JSON; this table is a coverage check rather than another latency aggregate. A count mismatch is a validation failure. Expected rounds come from the speculative-round counter; the first prefill event is retained in logged events but excluded from that counter.

| Context | Logged events | Speculative rounds | Expected speculative rounds | Timed round values | Untimed events | Missing round numbers | Capture status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0K | 951 | 950 | 950 | 950 | 1 | none | captured |
| 60K | 1,221 | 1220 | 1220 | 1,220 | 1 | none | captured |
| 200K | 839 | 838 | 838 | 838 | 1 | none | captured |


The histogram below is generated from the complete per-round records of the [coding-context benchmark](benchmarks/results/pi-coding-contexts.json). Every measured `round_ms` value appears in exactly one bin; untimed events are reported separately. The shared suite uses this same predetermined clean control for the coding row and histogram; the stage residual matches only structurally admitted M8 cycles from both controls, while this histogram retains every measured round.

| Round time | 0K arm | 60K arm | 200K arm |
|---|---:|---:|---:|
| `<35` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `35–37` | 230 (24.2%) | 0 (0.0%) | 0 (0.0%) |
| `37–39` | 717 (75.5%) | 0 (0.0%) | 0 (0.0%) |
| `39–40` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `40–40.5` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `40.5–41` | 0 (0.0%) | 271 (22.2%) | 0 (0.0%) |
| `41–41.5` | 0 (0.0%) | 921 (75.5%) | 0 (0.0%) |
| `41.5–42` | 0 (0.0%) | 15 (1.2%) | 0 (0.0%) |
| `42–43` | 0 (0.0%) | 11 (0.9%) | 0 (0.0%) |
| `43–45` | 0 (0.0%) | 1 (0.1%) | 0 (0.0%) |
| `45–46` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `46–47` | 0 (0.0%) | 1 (0.1%) | 0 (0.0%) |
| `47–48` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `48–48.5` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `48.5–49` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `49–51` | 1 (0.1%) | 0 (0.0%) | 437 (52.1%) |
| `51–52.5` | 0 (0.0%) | 0 (0.0%) | 399 (47.6%) |
| `52.5–53` | 0 (0.0%) | 0 (0.0%) | 1 (0.1%) |
| `53–53.5` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `53.5–54` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `54–55` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `55–56` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `56–60` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `60–62.5` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `62.5–63` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `63–63.5` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `63.5–64` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `64–65` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `65–70` | 0 (0.0%) | 0 (0.0%) | 1 (0.1%) |
| `70–100` | 1 (0.1%) | 0 (0.0%) | 0 (0.0%) |
| `100–250` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `250–500` | 1 (0.1%) | 0 (0.0%) | 0 (0.0%) |
| `≥500` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| **Timed rounds** | **950** | **1,220** | **838** |
| **Untimed events** | **1** | **1** | **1** |
| **Mean / median** | **37.65 / 37.26 ms** | **41.16 / 41.15 ms** | **51.00 / 50.99 ms** |


### Known remaining symptoms and likely causes

**The missing full-graph preparation hook is repaired.** The first integrated speed run took 59.44 ms at 60K because FULL replay bypassed the existing post-cache-preparation synchronization and recovery hook. After installing it on both execution branches, the matched comparison measured 43.38 ms control versus 41.14 ms optimized, with identical output and acceptance. Fresh-chat reproductions also recovered. This fixes a specific integration defect; it does not prove that all HIP/ROCr queue stalls are impossible. [Reproduction and repair evidence](benchmarks/results/full-graph-cache-prepare-20260925.json).

None of the 756 matched intervals in the later 200K clean control exceeded 100 ms; its median was 51.005 ms. The histogram includes this same predetermined control's entire round feed; the stage residual uses its subset of exactly matched M8 cycles plus the preceding control. The paired residual estimates span 2.346–2.361 ms. Profiling overhead is reported separately. Indirect changes in clocks, execution duration and scheduling remain measurement uncertainty, so the residual is an estimate rather than a proved exact sum of runtime gaps. [Current controls](benchmarks/results/stage26-control-20260925.json).

**The completions stream still lacks a separate reasoning channel.** Thinking-enabled requests are reported as observed prose; their reasoning throughput cannot be isolated from this stream. The checkpoint benchmark measures generation and requested tail flushing, not a complete Pi transcript commit. Its format-validation result is reported in the table. Natural completions shorter than the coding target remain explicit validation failures, even when their timing and round captures are complete.

**Isolated pauses remain.** The complete coding/histogram feeds retain 0K: 1 interval(s), longest 402.752 ms. The retained paired controls separately contain 0K before: 1 interval(s), longest 377.192 ms; 60K before: 1 interval(s), longest 285.164 ms. Their cause has not been localized. The histogram uses the predetermined after-control; row 26 uses both controls at the trace's retained indices. Warmup and trace-boundary exclusions are structural, not duration-based; every coding round remains in the public feed.

**Global-512 candidate selection remains approximate.** The current head study matched reference top-1 in 12,015/12,015 rows and retained the complete reference top-20 in 12,015/12,015. Full-head M1/M8 and eager/compiled agreement does not certify shortlist completeness or eliminate model-generated loops. [Current head evidence](benchmarks/results/head-candidate-depth-20260925.json).

Aggregate data: [current measurements](benchmarks/results/coherence-current.json). Historical methodology and detailed numerical evidence: [technical report](reports/d7-rdna4-2026-09-17/REPORT.md).

<!-- /COHERENCE_CURRENT_RESULTS -->

## Quick start

The initial serving profile targets **one R9700, Linux x86-64, Qwen3.8-27B-
Uncensored-MXFP4-awq and Qwen3.8-27B-DFlash2-FP8**. Obtain the target and matching
drafter separately. Model weights and private benchmark inputs are not included.

The GPU host needs ROCm device access, Python 3.12+, rootless Podman,
at least 18 GiB free `/dev/shm`, and space for compiler caches and snapshots. The
profile uses 10 GB of GPU KV memory, an 18 GiB CPU offload arena, and additional
per-chat handover/tail RAM; allow ample system RAM.

```sh
git clone https://github.com/Terrydaktal/vllm-coherence.git
cd vllm-coherence
tools/coherence doctor
tools/coherence prepare
tools/coherence serve --model /path/to/target --draft /path/to/drafter
```

`prepare` verifies the versioned source/kernel bundle without opening the GPU.
`serve` starts the digest-pinned image and installs the verified additions before
graph capture. Initial compilation can take several minutes. The API binds to
`127.0.0.1:8080`.

```sh
curl http://127.0.0.1:8080/v1/models
```

Use `--dry-run` to inspect the complete command, or `--head full-bf16` for the
complete target vocabulary head. The default
`global512` profile uses faster, approximate candidate selection; `--head global256`
retains the smaller shortlist.

Keep the pinned compiler/image stack together. Rebuilding numerical code needs
fresh qualification; replacing hashes does not transfer evidence.
[BUILDING.md](docs/BUILDING.md) describes the source-to-bundle pipeline.

## Pi

Install uv, Node.js/npm, Git and jq on the client. From the workspace whose history
you want to use:

```sh
/path/to/vllm-coherence/tools/coherence pi -- --thinking xhigh
/path/to/vllm-coherence/tools/coherence pi -- --session last
```

For a remote GPU host:

```sh
/path/to/vllm-coherence/tools/coherence pi --ssh gpu-host -- --session last
```

The launcher installs patched Pi 0.84.2 into private Coherence state, verifies its
patches on reuse, chooses an available SSH forwarding port, and loads the operating
prompt and extensions. Sessions live in the current workspace's `.pi/sessions`.
Pi does not load workspace `AGENTS.md` as context. Supply a bespoke search tool
with `--search-extension /path/to/index.ts`, including a VM-specific extension.

`/compact` commits only a validated checkpoint. `/priority` shows/changes answer
ownership. Cache/temperature monitoring shares probes across windows.
See [Pi setup and recovery](docs/PI.md).

## Cache inspection

On the GPU host:

```sh
tools/coherence cache -- status --details
tools/coherence cache -- audit
tools/coherence cache -- watch --interval 5
```

The inventory distinguishes disk usage, cumulative disk traffic, published token
coverage, handover RAM and buffered tail RAM. It flags incomplete/duplicate
snapshots and failed cleanup. Dirty tails normally flush after about 8,192 new
tokens, and on explicit flush, eviction, successful compaction and clean shutdown.
Allow the server's clean shutdown timeout to finish. A crash may require prefilling
the unflushed tail.

## Verification and development

```sh
uv sync --frozen --extra cpu-tests
uv run --frozen --extra cpu-tests pytest -q
uv run --frozen coherence-conformance --help
python3 tools/check_publication.py
```

CPU tests use CPU-only Torch. Native qualification requires an explicit isolated
GPU run with the lease. Original Pi fixtures stay private; use the synthetic tests
or your own owner-only fixture. See [coverage and reference scope](docs/VERIFICATION.md).

The [25 September speed investigation](docs/SPEED_INVESTIGATION_20260925.md)
preserves three sample-qualified experimental improvements and the rejected
trials. These changes are not deployed; their measurements do not replace the
production benchmark tables above.
The subsequent integration run exposed and repaired a missing full-graph
cache-preparation hook: the matched 60K comparison now measures 43.38 ms control
versus 41.14 ms with all three candidates, instead of the failed 59.44 ms result.
[Diagnosis and verification](docs/SPEED_INVESTIGATION_20260925.md#full-graph-cache-preparation-regression-and-repair).

## Repository map

```text
tools/                         Portable launcher, Pi connection, packaging and audits
releases/                      Versioned archive/image identities and inventories
src/qwen_r9700_lab/             Conformance, references, logical state and diagnostics
experiments/radiance-public/    Runtime adapters, HIP kernels and build/probe/replay drivers
  upstream-correctness/        Attributed upstream backports and component licenses
  rocr-poll-backoff/            CPU idle-wait repair and build recipe
integrations/pi/               Client lock, operating prompt and extensions
scripts/                       Pi installer/patchers and shared cache/telemetry helpers
configs/profiles/              Finite-precision numerical contracts
tests/                         Synthetic regressions and negative controls
reports/                       Public numerical report and aggregate evidence
docs/                          Setup, architecture, verification and performance records
benchmarks/                    Current Coherence results and retained upstream harness/fixtures
Dockerfile, patch_*, radiance_* Original base/build and focused upstream-facing changes
```

The `qwen_r9700_lab` namespace and wire-format names remain for evidence/snapshot
compatibility. Root Radiance Dockerfiles support upstream reproduction;
**`tools/coherence` launches the assembled Coherence profile**. Research drivers
require explicit inputs and do not run automatically during serving.

Inherited Radiance run artifacts remain in the
[original base commit](https://github.com/Terrydaktal/vllm-coherence/tree/4e604dcccb311ca709858a4bd57a11fc636e7f43/benchmarks/results).
The working tree keeps Coherence's measurements. Root Compose files and
`.env.example` are labelled upstream reproduction examples, not Coherence launchers.

## Contributing

Submit a minimized failure or measured optimization with state/output comparisons,
negative controls and precise hardware/profile scope. Read
[CONTRIBUTING.md](CONTRIBUTING.md). Existing upstream PRs remain independent and are
linked from [ATTRIBUTION.md](ATTRIBUTION.md). See [LICENSE](LICENSE) for component terms.
