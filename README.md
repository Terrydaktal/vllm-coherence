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
[rerun with 2,817 passing checks](benchmarks/results/current-operator-confirmations-20260924.json), extends this
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

Current repaired release: **320 forced decode tokens after a fresh 60,000-token Pi prefill**, plus the prefill prediction, in each of four arms. Compiled M1 versus compiled M8, eager M1 versus compiled M1, eager M8 versus compiled M8, and eager M1 versus compiled M8 each produced the results below. Compiled controls use production piecewise graphs; eager controls include the existing RoPE rounding repair. The **full BF16 comparison head** exposes target-body differences; production uses approximate Global-512, measured separately. [Run identities and all four comparisons](benchmarks/results/current-320-confirmations-20260924.json).

| Prediction | Same token set | Same ordering | Mean shared tokens |
| --- | ---: | ---: | ---: |
| Top 1 | 320 / 320 (100.00%) | 320 / 320 (100.00%) | 1.0000 / 1 |
| Top 10 | 320 / 320 (100.00%) | 320 / 320 (100.00%) | 10.0000 / 10 |
| Top 20 | 320 / 320 (100.00%) | 320 / 320 (100.00%) | 20.0000 / 20 |

Each comparison also matched all 320 full-vocabulary logit hashes and the prefill prediction. This is finite execution consistency, not independent model certification, an arbitrary-input proof or a natural-completion quality test. This fresh slice comes from the retained Pi corpus; it is not the deleted historical standalone 320-token fixture. The earlier 10K study remains historical and was not rerun.

## Compiled backend stages

Measured on 2026-09-24 using the compiled, optimized Global-512 serving backend with the pinned-RAM huge-page promotion repair; sampling is temperature 1.0, top-p 0.95 and top-k 40. Context labels are starting prefixes: 0K, the private 60K Pi fixture, and the public synthetic 200K fixture. Each context ran a natural warmup followed by clean control, trace, and clean control; each arm generated 4,512 / 4,788 / 4,030 tokens respectively. Generated-token hashes and accepted-token schedules matched. The stage means retain 870 / 903 / 752 complete M8 cycles (0K / 60K / 200K), and controls use exactly those same decode indices. Trace setup/export boundaries and incomplete or inconsistent trace inventories are excluded by structure, never by duration; complete native round logs retain all rounds and stalls. GPU activity timestamps supply the stage times; CPU annotations, Python hooks and export time are excluded. No per-stage event probes or forced-token replay are used. The measured tracing slowdown was 8.802 / 5.621 / 1.626 ms per retained round; it is reported separately and is **not charged to row 26**. Row 26 is the clean control mean minus the union of GPU activity intervals. This remains an estimate: tracing can indirectly affect clocks and scheduling. Overlap is counted once in the total. The header identifies the commit containing the measured backend repairs and captured results; the capture retains its original checkout and source identities. [Capture and source identities](benchmarks/results/compiled-global512-stage-profile-20260924.json) · [controls](benchmarks/results/stage26-control-20260924.json) · [method and uncertainty](docs/STAGE_TIMING.md).

The later 200K clean control retained 3 intervals above 100 ms, the longest **2,555.045 ms**, although its median was 52.520 ms. These pauses raise the displayed mean and row 26. The before/after remainder range is 2.685–6.200 ms; their cause remains unresolved. This is repeat variability, not a confidence interval or proof that earlier tracing had no indirect effect.

The **eager M1** column combines the [current independent operator rerun](benchmarks/results/current-operator-confirmations-20260924.json) (**2,817 checks, zero failures**, 320 rows at the audited sites) with the broader [arithmetic-contract qualification](docs/eager-m1-contract-qualification.md). The [declared arithmetic](docs/M1_ARITHMETIC_CONTRACT.md) separates accepted weight, activation and KV quantization from implementation defects. Confidence scores are editorial assessments: **1/5** untested, **2/5** limited, **3/5** moderate, **4/5** high, **5/5** very high for the tested exact encoding/index/state properties. They are **not probabilities of being bug-free**; 5/5 is not an arbitrary-input proof. N/A identifies a separate model or a timing-only row. FP64 oracle checks use declared error criteria, with numerical differences retained in the evidence.

The current release also passes all four [320-token whole-model comparisons](benchmarks/results/current-320-confirmations-20260924.json), including eager M1 versus compiled M8: exact full-logit hashes and prefill prediction. This establishes consistency on the sample, not independent correctness of the model reference. The normalization release gates additionally cover 128 hidden-normalization sites and 48 GDN sites with 1,000 rows, and the timing column measures this deployed compiled Global-512 build. [Deployment identities](benchmarks/results/eager-m1-normalization-deployment-20260924.json). Complete-model prefill-versus-serial-decode equality, candidate completeness, and fresh native restore/concurrency qualification remain open.

The fresh [isolated stage run](benchmarks/results/current-stage-confirmations-20260924.json) covers 22 boundaries and 770 layer instances over the same 320 Pi tokens. Current M1/M8 and eager/compiled M8 match at every stage: local tensors and state, full-logit hashes, and top-1/10/20 sets and ordering. All 40 eight-token groups pass deliberate corruption controls and restore authoritative state. This adds integration evidence to the independent operator checks; shared implementations still are not independent mathematical references.

| Stage | Current GPU activity per retained compiled M8 cycle (0K / 60K / 200K; milliseconds unless explicitly marked; 2026-09-24; run [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed)) | Current M1->M8 correctness and eager->compiled correctness evidence | Current Eager M1 correctness evidence | Last relevant code commit / change | What this stage does |
| --- | ---: | --- | --- | --- | --- |
| **1. Drafter** | 5.985 / 6.301 / 6.350 | Separate proposal model; target M1/M8 comparisons do not independently qualify it. | **N/A: separate proposal model.** Independent proposal/target RNG controls passed. **Uncertainty:** this target-M1 audit does not independently qualify drafter arithmetic or every speculative commit path. | [`146c682`](https://github.com/Terrydaktal/vllm-coherence/commit/146c6824c530acd1a627a99db67b0159c10cf0ac): Defer INT2 draft-head packing until the real shared target weights are bound, avoiding uninitialized placeholder weights. | Suggests up to seven tokens for the target model to check. |
| **2. Embedding + first input normalization + FP8 production** | 0.010 / 0.009 / 0.009 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260924.json). | **4/5 High.** Exact sampled embeddings and independent RMS checks; released fused norm preserves M1 FP8 bytes/scales at tested batch boundaries from 1 to 2,048 rows. **Uncertainty:** finite inputs and indices; prefill versus a fully serial reference is not qualified. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 reduction and BF16/FP8 rounding at every admitted prefill width, removing batch-dependent activation differences. | Looks up token vectors, normalizes them and creates FP8 inputs in one kernel, preserving intermediate rounding. |
| **3. Layer input residual/normalization + FP8 production** | 0.490 / 0.492 / 0.504 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260924.json). | **4/5 High.** Qualified release matches native M1 plus independent FP8 encoding at 129 hidden-norm weights × 320 rows; original 55 byte differences eliminated. CPU arithmetic reproduces the saved failing rows. **Uncertainty:** finite inputs, intrinsic/denormal behavior and untested inputs; deployed 24 September. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 reduction and BF16/FP8 rounding at every admitted prefill width, removing batch-dependent activation differences. | Adds the residual, normalizes the result and creates FP8 inputs in one kernel, preserving intermediate rounding. |
| ↳ GDN input activation FP8 quantization | Included in **stage 3** | Exact fused FP8 bytes/scales; see stage 3 and its evidence scope | **5/5 Very high for tested encoding.** Qualified release fused bytes/scales match independent encoding; stage 3 prefill/M1 counterexamples now pass. **Uncertainty:** inherits normalization arithmetic; FP8 is an explicitly accepted approximation; deployed 24 September. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 reduction and BF16/FP8 rounding at every admitted prefill width, removing batch-dependent activation differences. | Uses the FP8 output produced by the same input-normalization kernel. |
| ↳ Attention input activation FP8 quantization | Included in **stage 3** | Exact fused FP8 bytes/scales; see stage 3 and its evidence scope | **5/5 Very high for tested encoding.** Qualified release fused bytes/scales match independent encoding; stage 3 prefill/M1 counterexamples now pass. **Uncertainty:** inherits normalization arithmetic; FP8 is an explicitly accepted approximation; deployed 24 September. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 reduction and BF16/FP8 rounding at every admitted prefill width, removing batch-dependent activation differences. | Uses the FP8 output produced by the same input-normalization kernel. |
| **4. GDN input projection** | 4.060 / 3.975 / 3.986 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260924.json). | **4/5 High.** Current 496-matrix × 320-input sampled-channel rerun, plus all output channels of 30 selected matrices on 32 inputs each (7,739,392 outputs); 12 saved operator-activation replays add 3,276,800 outputs. **Uncertainty:** other matrices retain sampled-channel coverage; FP64 error criteria are not exact-arithmetic proofs; the separate 320-token stage replay uses full-model activations but shares the native operator, rather than an independent FP64 oracle. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve stored MXFP4 coefficients in the folded lookup and repair fragment/tiled per-block fallbacks while retaining the qualified GEMM dispatch. | Projects hidden vectors into the inputs and gates for the recurrent layer. |
| **5. GDN layout/copies and buffer initialization** | 0.316 / 0.318 / 0.319 | Authoritative cache/state restored in all 40 eight-token groups; state/output corruption controls detected · [current replay](benchmarks/results/current-stage-confirmations-20260924.json). No separate layout top-20 attribution. | **4/5 High in the tested replay.** Unsupported strides/overlap rejected; final histories and untouched slots exact across 48 GDN sites. All 40 current eight-token replay groups restore authoritative cache/state, including corruption controls. **Uncertainty:** replay restoration is not a new native disk snapshot, cancellation or concurrent-ownership qualification. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Reject unsupported packed GDN inner strides and overlapping state slots; retain the existing nine-slot layout. | Arranges inputs and clears temporary buffers throughout each GDN layer. |
| **6. GDN convolution** | 0.200 / 0.209 / 0.212 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260924.json). | **4/5 High.** 48 layers × 320 consecutive inputs pass FP64 convolution/SiLU criteria; histories and untouched slots exact. **Uncertainty:** finite numerical/state samples; full-session integration remains unqualified by this audit. | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Aligned GDN convolution product and accumulation arithmetic across M1/M8 and eager/compiled execution. | Updates recent-token history using the corrected multiply/add order. |
| **7. GDN recurrence and gates** | 1.423 / 1.232 / 1.246 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260924.json). | **4/5 High.** 48 layers × 320 transitions match independent state/output equations within tolerance; 32,768-step decay passes. **Uncertainty:** no exact full-model recurrence proof or coverage of every reachable state. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Use stable log1p(exp(x)) in M1, M8 and prefill so small nonzero recurrent gates survive FP32 cancellation. | Updates recurrent memory in token order. Faster prefill retains the existing nine-slot state layout. |
| **8. GDN output gated normalization + FP8 production** | 0.159 / 0.160 / 0.163 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260924.json). | **4/5 High numerically.** Explicit FP32 norm/weight/gate contract with one final BF16 boundary; released path matches native M1 at all 48 sites × 320 audit rows, plus 1,000-row release gates and batch-boundary checks. **Uncertainty:** this deliberately differs from Hugging Face intermediate casts; intrinsic behavior and arbitrary inputs remain unproved; deployed 24 September. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 gated-normalization reduction layout during prefill and retain one final BF16 rounding before FP8 production. | Normalizes and gates GDN output, then produces FP8 inputs in one kernel; intermediate BF16 rounding is preserved. |
| ↳ GDN output activation FP8 quantization | Included in **stage 8** | Exact fused FP8 bytes/scales; see stage 8 and its evidence scope | **5/5 Very high for tested encoding.** Qualified M1/prefill bytes and scales match independent encoding of native BF16 output at 48 sites × 320 rows. **Uncertainty:** inherits the declared stage 8 arithmetic; no weight-only model equivalence; deployed 24 September. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 gated-normalization reduction layout during prefill and retain one final BF16 rounding before FP8 production. | Uses the FP8 output produced by the same gated-normalization kernel. |
| **9. GDN output projection** | 1.924 / 1.902 / 1.916 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260924.json). | **4/5 High.** Current 496-matrix × 320-input sampled-channel rerun, plus all output channels of 30 selected matrices on 32 inputs each (7,739,392 outputs); 12 saved operator-activation replays add 3,276,800 outputs. **Uncertainty:** other matrices retain sampled-channel coverage; FP64 error criteria are not exact-arithmetic proofs; the separate 320-token stage replay uses full-model activations but shares the native operator, rather than an independent FP64 oracle. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve stored MXFP4 coefficients in the folded lookup and repair fragment/tiled per-block fallbacks while retaining the qualified GEMM dispatch. | Maps the recurrent-layer result back to the model's hidden-vector width. |
| **10. Attention input projection** | 1.157 / 1.156 / 1.157 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260924.json). | **4/5 High.** Current 496-matrix × 320-input sampled-channel rerun, plus all output channels of 30 selected matrices on 32 inputs each (7,739,392 outputs); 12 saved operator-activation replays add 3,276,800 outputs. **Uncertainty:** other matrices retain sampled-channel coverage; FP64 error criteria are not exact-arithmetic proofs; the separate 320-token stage replay uses full-model activations but shares the native operator, rather than an independent FP64 oracle. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve stored MXFP4 coefficients in the folded lookup and repair fragment/tiled per-block fallbacks while retaining the qualified GEMM dispatch. | Projects hidden vectors into the inputs for attention. |
| **11. Attention Q/K normalization, RoPE and layout** | 0.234 / 0.214 / 0.218 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260924.json). | **4/5 High.** All 32 Q/K norm sites; both position paths at nine ranges pass independent arithmetic checks; repaired eager MRoPE launches verified. **Uncertainty:** rounding tolerances and compact RoPE tables; full-table addressing not covered. | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Added live-row handling for compiled M8 graph padding while retaining the repaired RoPE arithmetic. | Normalizes queries and keys and applies positional rotation (RoPE), retaining the corrected rounding. |
| **12. Attention KV write** | 0.044 / 0.045 / 0.045 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260924.json). | **5/5 Very high within tested cases.** 393,216 stored values/destinations exact; BF16/OCP FP8 scales, masked slots, page edges and every finite FP8 code checked. **Uncertainty:** arbitrary aliasing, asynchronous access and native snapshot restoration remain unproved. | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Aligned causal attention and cache writes with the M1/M8 arithmetic contract. | Stores new keys and values in the cache for reuse by later tokens. |
| **13. Attention decode** | 0.397 / 4.765 / 13.895 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260924.json). Decode and merge checked together. | **4/5 High within tested cases.** Earlier 88 cases plus 96 independent FP64 cases with unique shuffled pages, holes, poisoned tails, per-head scales and varied contents through 60,001 tokens; guards intact. **Uncertainty:** longest 253K checks still use structured/repeated pages; exact softmax and arbitrary layouts remain unproved. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve BF16 query/key/value range, scale after the FP32 dot product, retain probability residuals and reject incompatible FP8 formats; keep the page-boundary traversal repair. | Attends to the current and earlier tokens using corrected arithmetic and shared cache loads. |
| **14. Attention split-KV merge** | 0.261 / 0.142 / 0.116 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260924.json). Decode and merge checked together. | **4/5 High for repaired arithmetic.** Four exact split-mean failures repaired; 96 additional varied-layout attention cases pass, including exact BF16 cancellation controls. **Uncertainty:** integrated evidence, not an exhaustive independent merge-stage proof. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Keep split outputs and normalization sums in FP32 until the final BF16 merge, removing intermediate FP16 rounding loss. | Combines attention results from cache partitions in the corrected arithmetic order. |
| **15. Attention output gating** | 0.025 / 0.029 / 0.030 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260924.json). | **4/5 High.** Every finite BF16 sigmoid input tested with a bounded multiplier against FP64 equations. **Uncertainty:** numerical tolerances; not every gate/output pair or full attention-to-gate integration. | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Preserved attention output gating and intermediate BF16 rounding in aligned M1/M8 execution. | Applies learned gates to the attention output. |
| **16. Attention output activation FP8 quantization** | 0.037 / 0.042 / 0.043 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260924.json). | **4/5 High for tested encoding and integration.** Independent FP8 codec checks plus current 320-token isolated M1/M8 and eager/compiled quantizer agreement at every attention layer. **Uncertainty:** the independent oracle checks remain sampled; arbitrary gate/quantization inputs and layouts are unproved. | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Enabled qualified native-rounding FP8 output production and rejected approximate traced quantization. | Converts the attention output to FP8 for its output projection. |
| **17. Attention output projection** | 0.617 / 0.578 / 0.553 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260924.json). | **4/5 High.** Current 496-matrix × 320-input sampled-channel rerun, plus all output channels of 30 selected matrices on 32 inputs each (7,739,392 outputs); 12 saved operator-activation replays add 3,276,800 outputs. **Uncertainty:** other matrices retain sampled-channel coverage; FP64 error criteria are not exact-arithmetic proofs; the separate 320-token stage replay uses full-model activations but shares the native operator, rather than an independent FP64 oracle. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve stored MXFP4 coefficients in the folded lookup and repair fragment/tiled per-block fallbacks while retaining the qualified GEMM dispatch. | Maps the attention result back to the model's hidden-vector width. |
| **18. Post-attention/GDN residual/normalization + FP8 production** | 0.505 / 0.510 / 0.523 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260924.json). | **4/5 High.** Qualified release matches native M1 plus independent FP8 encoding in the 129-site × 320-row hidden-norm audit; original 55 byte differences eliminated. **Uncertainty:** finite inputs and prefill-versus-serial-decode behavior; deployed 24 September. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 reduction and BF16/FP8 rounding at every admitted prefill width, removing batch-dependent activation differences. | Adds the layer result to the residual, normalizes it and creates FP8 inputs for the MLP in one kernel. |
| ↳ MLP gate/up input FP8 quantization | Included in **stage 18** | Exact fused FP8 bytes/scales; see stage 18 and its evidence scope | **5/5 Very high for tested encoding.** Released bytes/scales match independent encoding of M1 BF16 norm; stage 18 prefill counterexamples now pass. **Uncertainty:** inherits normalization arithmetic and accepted FP8 loss; deployed 24 September. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 reduction and BF16/FP8 rounding at every admitted prefill width, removing batch-dependent activation differences. | Uses the FP8 output produced by the same post-normalization kernel. |
| **19. MLP gate/up projection** | 11.539 / 11.273 / 11.296 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260924.json). | **4/5 High.** Current 496-matrix × 320-input sampled-channel rerun, plus all output channels of 30 selected matrices on 32 inputs each (7,739,392 outputs); 12 saved operator-activation replays add 3,276,800 outputs. **Uncertainty:** other matrices retain sampled-channel coverage; FP64 error criteria are not exact-arithmetic proofs; the separate 320-token stage replay uses full-model activations but shares the native operator, rather than an independent FP64 oracle. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve stored MXFP4 coefficients in the folded lookup and repair fragment/tiled per-block fallbacks while retaining the qualified GEMM dispatch. | Computes both MLP input projections. Wider decode dispatch avoids the slower prefill kernel for qualified shapes. |
| **20. MLP SiLU and gating** | 0.163 / 0.181 / 0.184 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260924.json). | **4/5 High.** Every finite BF16 gate encoding checked with a bounded multiplier; 320 fused M1 cases pass. **Uncertainty:** tiny subnormal differences from FP64 remain; not all gate/up pairs are covered. | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Preserved intermediate BF16 rounding in aligned M1/M8 and eager/compiled arithmetic; the slower fused SiLU remains disabled. | Applies SiLU and combines the two MLP branches, retaining intermediate BF16 rounding. |
| **21. MLP down input FP8 quantization** | 0.247 / 0.256 / 0.262 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260924.json). | **5/5 Very high for tested encoding.** 320 fused M1 cases exactly match independent FP8 encoding of the native BF16 SiLU/gate pipeline. **Uncertainty:** inherits stage 20 numerical behavior; no exhaustive input-pair or shape proof. | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Enabled qualified native per-token FP8 production for the down projection input. | Converts MLP activations to FP8 for the down projection. |
| **22. MLP down projection** | 5.504 / 5.335 / 5.339 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260924.json). | **4/5 High.** Current 496-matrix × 320-input sampled-channel rerun, plus all output channels of 30 selected matrices on 32 inputs each (7,739,392 outputs); 12 saved operator-activation replays add 3,276,800 outputs. **Uncertainty:** other matrices retain sampled-channel coverage; FP64 error criteria are not exact-arithmetic proofs; the separate 320-token stage replay uses full-model activations but shares the native operator, rather than an independent FP64 oracle. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve stored MXFP4 coefficients in the folded lookup and repair fragment/tiled per-block fallbacks while retaining the qualified GEMM dispatch. | Projects the MLP result back to the model's hidden-vector width. |
| **23. Final normalization/layout** | 0.005 / 0.005 / 0.005 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [current 320-token run](benchmarks/results/current-stage-confirmations-20260924.json). | **4/5 High.** 320 inputs at the actual final BF16 norm pass independent RMS equations. **Uncertainty:** sampled numerical criteria; current complete-model mode agreement is sampled, not an independent reference proof. The extra FP8 control is not serving behavior. | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Aligned final normalization and full BF16 head precision boundaries while preserving the rounding contract. | Normalizes the final hidden vector before vocabulary scoring. |
| **24. Global-512 target head** | 1.035 / 1.019 / 1.048 | Same top-1: 12,015/12,015; complete reference top-20 retained: 12,011/12,015. Includes M1/M8; not an M1-versus-M8 ordering test · [current head study](benchmarks/results/head-candidate-depth-normalization-20260924.json). | **4/5 for score arithmetic; selection known approximate.** Three 512-row weight slabs × 320 inputs pass FP64 criteria. Current Global-512 study: top-1 12,015/12,015; complete top-20 12,011/12,015; top-40 11,983/12,015. **Uncertainty:** sampled scores, retained-logit differences and uncertified excluded tokens; these recall rows include M1 and M8, not M1 alone. [Head evidence](docs/HEAD_CANDIDATE_DEPTH.md). | [`9bb795d`](https://github.com/Terrydaktal/vllm-coherence/commit/9bb795d2e612c76087d16932841131edf4834d5e): increase target shortlist to 512; drafter unchanged. Extends the Global-256 method from [`31add3d`](https://github.com/Terrydaktal/vllm-coherence/commit/31add3d06080be6e4f5198c6e8afcc6276e1edaf). | Scores the vocabulary with INT2, selects 512 candidates and rescores them with BF16 weights. Selection remains approximate. |
| **25. Other GPU bookkeeping** | 0.568 / 0.525 / 0.524 | No isolated top-20 operator claim; sampling/state controls have separate evidence. | **3/5 Moderate for sampled decision/state checks.** 600,000 sampling trials; maximum probability error 0.118 percentage points; faulty shared-RNG control detected. **Uncertainty:** generic bookkeeping kernels and every asynchronous commit/rollback path are not independently qualified. | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Integrated qualified sampling, cache/state bookkeeping and graph-capture admission; no isolated correctness claim is made for this aggregate row. | Runs sampling and cache/state update kernels outside the named model stages. |
| **26. Estimated runtime overhead** | 2.107 / 2.426 / 4.442 (estimate) | [Matched control minus GPU activity union](benchmarks/results/matched-stage-residual-20260924.json) | **N/A: timing estimate.** No model arithmetic to score. **Uncertainty:** performance residual is not evidence of scheduler or session-state correctness. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): record matched unprofiled controls and overlap-corrected residuals | Indirect observer effects are not proved zero. |
| **Total reconstructed round (stages 1–26)** | **39.008 / 43.095 / 54.367** | GPU activity union plus the estimated remainder | **N/A: no aggregate correctness score.** Operator scores cannot be averaged into a model guarantee. **Uncertainty:** full-model prefill/M1 equality, candidate completeness and new restore/concurrency qualification remain open. | — | Overlapping stages are counted once in the total. |

The table restores the historical grouped 26 measured-row layout and adds a total row. Each timing cell is ordered **0K / 60K / 200K**. Fused kernels are charged once to their containing stage; the ↳ rows are detail-only inclusion records and add no timing; rows without a separate profiler scope are labelled in the timing cell rather than displayed as 0.000. The total counts overlapping GPU activity once. The old forced-replay subtraction is superseded; [its audit](benchmarks/results/stage-timing-audit-20260923.json) remains available.

Each **set/order** pair means the same top-20 token set, followed by the same ranking. Current stage confirmations identify M1/M8 and eager/compiled M8 separately. Each position passes only if every layer instance passes on the same captured inputs. These diagnostic stage replays are checked against the compiled graph control; their times are not used in this table. Timing and correctness captures record their own exact source hashes. The provenance column identifies the last relevant code change.

Fusion still permits correctness instrumentation: a diagnostic kernel can expose intermediate values, and fused outputs can be compared with an unfused reference. The normal GPU profile measures the combined kernel. Internal probes or splitting the kernel can change its performance, so those measurements are not an additive breakdown of the production kernel's time.

The expandable layer and kernel tables use the same 903 retained 60K cycles as the main table (2026-09-24). GPU activity crossing a worker boundary is clipped to that boundary; activity-record counts include these fragments. CPU profiling work is excluded. Cycle counts are not output-token counts.

<details>
<summary>Current compiled 60K decoder-layer detail: projection and remaining-work timings</summary>

Finish each row's layer, including its MLP, before moving to the next row. Each layer has four projections. Gate and up are one joint GEMM; there is no separately measured gate/up split. The remaining-work column combines normalization, mixing and other operations between those projections.

| Layer | Type | All layer work ms | Input projection ms | Output projection ms | Gate/up projection ms | Down projection ms | Other work ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | GDN | 0.4353 | 0.0767 | 0.0391 | 0.1749 | 0.0801 | 0.0645 |
| 1 | GDN | 0.4533 | 0.0819 | 0.0403 | 0.1832 | 0.0874 | 0.0604 |
| 2 | GDN | 0.4595 | 0.0851 | 0.0399 | 0.1829 | 0.0881 | 0.0636 |
| 3 | Attention | 0.7093 | 0.0733 | 0.0361 | 0.1676 | 0.0801 | 0.3522 |
| 4 | GDN | 0.4383 | 0.0823 | 0.0389 | 0.1746 | 0.0803 | 0.0622 |
| 5 | GDN | 0.4505 | 0.0822 | 0.0402 | 0.1801 | 0.0863 | 0.0617 |
| 6 | GDN | 0.4592 | 0.0842 | 0.0403 | 0.1834 | 0.0879 | 0.0633 |
| 7 | Attention | 0.7059 | 0.0728 | 0.0361 | 0.1674 | 0.0803 | 0.3493 |
| 8 | GDN | 0.4391 | 0.0826 | 0.0388 | 0.1747 | 0.0806 | 0.0623 |
| 9 | GDN | 0.4513 | 0.0818 | 0.0401 | 0.1826 | 0.0863 | 0.0605 |
| 10 | GDN | 0.4560 | 0.0839 | 0.0400 | 0.1821 | 0.0866 | 0.0635 |
| 11 | Attention | 0.7062 | 0.0712 | 0.0362 | 0.1672 | 0.0801 | 0.3515 |
| 12 | GDN | 0.4357 | 0.0816 | 0.0387 | 0.1736 | 0.0800 | 0.0617 |
| 13 | GDN | 0.4492 | 0.0813 | 0.0396 | 0.1819 | 0.0858 | 0.0606 |
| 14 | GDN | 0.4561 | 0.0845 | 0.0406 | 0.1823 | 0.0859 | 0.0628 |
| 15 | Attention | 0.7041 | 0.0712 | 0.0361 | 0.1671 | 0.0803 | 0.3494 |
| 16 | GDN | 0.4377 | 0.0820 | 0.0388 | 0.1747 | 0.0801 | 0.0621 |
| 17 | GDN | 0.4525 | 0.0817 | 0.0404 | 0.1822 | 0.0872 | 0.0610 |
| 18 | GDN | 0.4554 | 0.0846 | 0.0396 | 0.1807 | 0.0876 | 0.0629 |
| 19 | Attention | 0.7035 | 0.0719 | 0.0363 | 0.1675 | 0.0798 | 0.3480 |
| 20 | GDN | 0.4384 | 0.0820 | 0.0386 | 0.1740 | 0.0805 | 0.0632 |
| 21 | GDN | 0.4464 | 0.0816 | 0.0398 | 0.1793 | 0.0849 | 0.0608 |
| 22 | GDN | 0.4555 | 0.0841 | 0.0396 | 0.1810 | 0.0878 | 0.0631 |
| 23 | Attention | 0.7044 | 0.0722 | 0.0361 | 0.1671 | 0.0801 | 0.3490 |
| 24 | GDN | 0.4377 | 0.0820 | 0.0389 | 0.1748 | 0.0801 | 0.0619 |
| 25 | GDN | 0.4528 | 0.0814 | 0.0404 | 0.1827 | 0.0873 | 0.0611 |
| 26 | GDN | 0.4536 | 0.0845 | 0.0403 | 0.1805 | 0.0858 | 0.0625 |
| 27 | Attention | 0.7059 | 0.0725 | 0.0361 | 0.1674 | 0.0801 | 0.3498 |
| 28 | GDN | 0.4378 | 0.0824 | 0.0390 | 0.1746 | 0.0800 | 0.0618 |
| 29 | GDN | 0.4483 | 0.0827 | 0.0394 | 0.1791 | 0.0857 | 0.0615 |
| 30 | GDN | 0.4565 | 0.0844 | 0.0398 | 0.1818 | 0.0877 | 0.0629 |
| 31 | Attention | 0.7066 | 0.0720 | 0.0362 | 0.1676 | 0.0799 | 0.3508 |
| 32 | GDN | 0.4375 | 0.0820 | 0.0389 | 0.1739 | 0.0802 | 0.0625 |
| 33 | GDN | 0.4539 | 0.0824 | 0.0403 | 0.1824 | 0.0872 | 0.0615 |
| 34 | GDN | 0.4581 | 0.0840 | 0.0402 | 0.1811 | 0.0876 | 0.0652 |
| 35 | Attention | 0.7073 | 0.0724 | 0.0362 | 0.1670 | 0.0800 | 0.3517 |
| 36 | GDN | 0.4386 | 0.0822 | 0.0387 | 0.1746 | 0.0809 | 0.0622 |
| 37 | GDN | 0.4476 | 0.0822 | 0.0405 | 0.1783 | 0.0850 | 0.0615 |
| 38 | GDN | 0.4540 | 0.0837 | 0.0403 | 0.1815 | 0.0860 | 0.0625 |
| 39 | Attention | 0.7078 | 0.0727 | 0.0359 | 0.1677 | 0.0800 | 0.3516 |
| 40 | GDN | 0.4400 | 0.0825 | 0.0391 | 0.1748 | 0.0810 | 0.0625 |
| 41 | GDN | 0.4536 | 0.0819 | 0.0403 | 0.1826 | 0.0870 | 0.0618 |
| 42 | GDN | 0.4588 | 0.0859 | 0.0400 | 0.1825 | 0.0870 | 0.0634 |
| 43 | Attention | 0.7050 | 0.0723 | 0.0358 | 0.1676 | 0.0800 | 0.3493 |
| 44 | GDN | 0.4400 | 0.0832 | 0.0388 | 0.1752 | 0.0806 | 0.0622 |
| 45 | GDN | 0.4477 | 0.0820 | 0.0393 | 0.1791 | 0.0857 | 0.0615 |
| 46 | GDN | 0.4536 | 0.0836 | 0.0399 | 0.1815 | 0.0845 | 0.0641 |
| 47 | Attention | 0.7086 | 0.0724 | 0.0363 | 0.1676 | 0.0801 | 0.3522 |
| 48 | GDN | 0.4407 | 0.0827 | 0.0392 | 0.1750 | 0.0802 | 0.0637 |
| 49 | GDN | 0.4532 | 0.0827 | 0.0397 | 0.1822 | 0.0872 | 0.0613 |
| 50 | GDN | 0.4568 | 0.0850 | 0.0402 | 0.1816 | 0.0877 | 0.0624 |
| 51 | Attention | 0.7059 | 0.0723 | 0.0361 | 0.1669 | 0.0799 | 0.3507 |
| 52 | GDN | 0.4370 | 0.0824 | 0.0388 | 0.1740 | 0.0798 | 0.0620 |
| 53 | GDN | 0.4466 | 0.0826 | 0.0393 | 0.1788 | 0.0850 | 0.0609 |
| 54 | GDN | 0.4557 | 0.0844 | 0.0395 | 0.1817 | 0.0863 | 0.0637 |
| 55 | Attention | 0.7071 | 0.0723 | 0.0361 | 0.1676 | 0.0800 | 0.3511 |
| 56 | GDN | 0.4393 | 0.0827 | 0.0388 | 0.1744 | 0.0806 | 0.0627 |
| 57 | GDN | 0.4523 | 0.0819 | 0.0406 | 0.1831 | 0.0859 | 0.0609 |
| 58 | GDN | 0.4563 | 0.0854 | 0.0398 | 0.1802 | 0.0873 | 0.0637 |
| 59 | Attention | 0.7078 | 0.0719 | 0.0365 | 0.1676 | 0.0800 | 0.3519 |
| 60 | GDN | 0.4383 | 0.0825 | 0.0388 | 0.1741 | 0.0803 | 0.0625 |
| 61 | GDN | 0.4490 | 0.0817 | 0.0395 | 0.1802 | 0.0861 | 0.0615 |
| 62 | GDN | 0.4563 | 0.0843 | 0.0406 | 0.1809 | 0.0854 | 0.0651 |
| 63 | Attention | 0.7069 | 0.0725 | 0.0360 | 0.1671 | 0.0799 | 0.3515 |

</details>

<details>
<summary>Current 60K GPU activity, grouped by stage</summary>

| Stage / compiled kernel | Activity records in 903 retained cycles | Current GPU ms per retained 60K cycle |
| --- | ---: | ---: |
| Drafter / `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x32_MI16x16x1_SN_LDSB1_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_CD1_1_CLR0_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB1_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA1201_ICIW0_IU1_K1_LDSTI0_LBSPPA128_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB0_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB2_ONLL1_PAP0_PGL0_PGR1_PLR1_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1.kd` | 903 | 0.007167 |
| Drafter / `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x32x256_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB1_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA1201_ICIW0_IU1_K1_LDSTI0_LBSPPA512_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB0_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB16_ONLL0_PAP0_PGL0_PGR1_PLR1_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_4_1.kd` | 903 | 0.176932 |
| Drafter / `__amd_rocclr_copyBuffer.kd` | 903 | 0.001859 |
| Drafter / `__amd_rocclr_fillBufferAligned.kd` | 2709 | 0.007444 |
| Drafter / `_cache_draft_logits_kernel.kd` | 903 | 0.002423 |
| Drafter / `_draft_head_int2.kd` | 936 | 0.810002 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_17408_EVEN_K_1_GRID_MN_40_cache_modifier_NONE.kd` | 4583 | 0.737678 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_25600_EVEN_K_1_GRID_MN_40_cache_modifier_NONE.kd` | 903 | 0.234704 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_40_cache_modifier_NONE.kd` | 4532 | 0.199968 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_5120_EVEN_K_1_GRID_MN_272_cache_modifier_NONE.kd` | 4653 | 1.449178 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_5120_EVEN_K_1_GRID_MN_48_cache_modifier_NONE.kd` | 4804 | 0.274304 |
| Drafter / `_prepare_dflash_inputs_kernel.kd` | 903 | 0.007627 |
| Drafter / `_rerank_exact.kd` | 903 | 0.008884 |
| Drafter / `_selector_walk_kernel.kd` | 903 | 0.007968 |
| Drafter / `kernel_unified_attention.kd` | 4717 | 1.796761 |
| Drafter / `reshape_and_cache_kernel_flash.kd` | 9033 | 0.026959 |
| Drafter / `triton_per_fused_4.kd` | 901 | 0.001954 |
| Drafter / `triton_per_fused_8.kd` | 3615 | 0.007698 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_preshuffle_gemm_squeeze_view_0.kd` | 4516 | 0.009519 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_preshuffle_gemm_squeeze_view_2.kd` | 905 | 0.002066 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_preshuffle_gemm_squeeze_view_4.kd` | 8130 | 0.015752 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_view_0.kd` | 903 | 0.004385 |
| Drafter / `triton_poi_fused_0.kd` | 903 | 0.002558 |
| Drafter / `triton_poi_fused_5.kd` | 901 | 0.002254 |
| Drafter / `triton_poi_fused_9.kd` | 3616 | 0.008102 |
| Drafter / `triton_poi_fused__to_copy_clamp_div_preshuffle_gemm_squeeze_view_1.kd` | 4516 | 0.010848 |
| Drafter / `triton_poi_fused__to_copy_clamp_div_preshuffle_gemm_squeeze_view_3.kd` | 909 | 0.002104 |
| Drafter / `triton_poi_fused__to_copy_clamp_div_preshuffle_gemm_squeeze_view_5.kd` | 8130 | 0.016551 |
| Drafter / `triton_poi_fused_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_3.kd` | 8134 | 0.025778 |
| Drafter / `triton_poi_fused_add_arange_bitwise_and_constant_pad_nd_ge_mul_rms_norm_select_slice_unsqueeze_view_1.kd` | 913 | 0.022671 |
| Drafter / `triton_poi_fused_add_permute_unsqueeze_view_2.kd` | 903 | 0.001893 |
| Drafter / `triton_poi_fused_cat_expand_index_mul_slice_unsqueeze_view_1.kd` | 903 | 0.002488 |
| Drafter / `triton_red_fused__to_copy_abs_clamp_div_max_mul_preshuffle_gemm_silu_slice_squeeze_view_6.kd` | 4519 | 0.018334 |
| Drafter / `triton_red_fused__to_copy_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_w4_gemm_2.kd` | 4520 | 0.030958 |
| Drafter / `triton_red_fused__to_copy_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_w4_gemm_7.kd` | 3615 | 0.025746 |
| Drafter / `triton_red_fused__to_copy_embedding_mul_rms_norm_w4_gemm_0.kd` | 905 | 0.004372 |
| Drafter / `triton_red_fused_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_7.kd` | 904 | 0.006358 |
| Drafter / `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) [clone .kd]` | 1807 | 0.009910 |
| Drafter / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 903 | 0.003570 |
| Drafter / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 903 | 0.003478 |
| Drafter / `void at::native::bitonicSortKVInPlace<2, -1, 16, 16, c10::BFloat16, long, at::native::GTOp<c10::BFloat16, true>, unsigned int>(at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, at::native::GTOp<c10::BFloat16, true>) [clone .kd]` | 903 | 0.003169 |
| Drafter / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 903 | 0.004952 |
| Drafter / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 904 | 0.002626 |
| Drafter / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 903 | 0.003570 |
| Drafter / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 903 | 0.003183 |
| Drafter / `void at::native::mbtopk::computeBlockDigitCounts<c10::BFloat16, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 1806 | 0.038868 |
| Drafter / `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 3613 | 0.052710 |
| Drafter / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, c10::BFloat16>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 1808 | 0.020435 |
| Drafter / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, float>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, float*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 3613 | 0.028519 |
| Drafter / `void at::native::mbtopk::fill<unsigned int, unsigned int>(unsigned int*, unsigned int, unsigned int) [clone .kd]` | 1806 | 0.003142 |
| Drafter / `void at::native::mbtopk::gatherTopK<c10::BFloat16, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 903 | 0.021213 |
| Drafter / `void at::native::mbtopk::gatherTopK<float, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, float*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 903 | 0.010946 |
| Drafter / `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4>) [clone .kd]` | 904 | 0.003896 |
| Drafter / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul>) [clone .kd]` | 903 | 0.001809 |
| Drafter / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) [clone .kd]` | 903 | 0.002509 |
| Drafter / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>) [clone .kd]` | 1807 | 0.003803 |
| Drafter / `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>) [clone .kd]` | 1808 | 0.005310 |
| Drafter / `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool) [clone .kd]` | 904 | 0.002170 |
| Drafter / `void at::native::warpMergeSortKVInPlace<2, -1, 128, 16, float, long, at::native::GTOp<float, true>, unsigned int, 32>(at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, at::native::GTOp<float, true>, float) [clone .kd]` | 903 | 0.004887 |
| Drafter / `void r4d_gemm_w4a16_nt_m64_kernel<1, 1, false>(unsigned short const*, unsigned int const*, unsigned int const*, __hip_bfloat16*, int, int, int, int, int) [clone .kd]` | 903 | 0.004622 |
| Drafter / `void r4d_gemm_w4a16_nt_m64_kernel<1, 1, true>(unsigned short const*, unsigned int const*, unsigned int const*, __hip_bfloat16*, int, int, int, int, int) [clone .kd]` | 9042 | 0.080353 |
| Drafter / `void vllm::rms_norm_kernel<c10::BFloat16, 8, 2, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, long, long, long, c10::BFloat16 const*, long, float, int, int) [clone .kd]` | 903 | 0.003271 |
| Drafter / `void vllm::rms_norm_kernel<c10::BFloat16, 8, 4, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, long, long, long, c10::BFloat16 const*, long, float, int, int) [clone .kd]` | 903 | 0.003044 |
| Drafter / `void vllm::rotary_embedding_kernel<c10::BFloat16, c10::BFloat16, true>(long const*, c10::BFloat16*, c10::BFloat16*, c10::BFloat16 const*, int, long, long, long, int, int, int, long, bool) [clone .kd]` | 903 | 0.002961 |
| Embedding + first input normalization / `triton_poi_fused__to_copy_embedding_0.kd` | 903 | 0.002583 |
| Embedding + first input normalization / `void norm_quant<false, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned char*, float*, unsigned short*, long, long, float) [clone .kd]` | 903 | 0.006780 |
| GDN layout/copies and buffer initialization / `triton_poi_fused_add_1.kd` | 2709 | 0.003817 |
| GDN layout/copies and buffer initialization / `triton_poi_fused_add_2.kd` | 1806 | 0.002487 |
| GDN layout/copies and buffer initialization / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 43344 | 0.151686 |
| GDN layout/copies and buffer initialization / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) [clone .kd]` | 43343 | 0.087853 |
| GDN layout/copies and buffer initialization / `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>) [clone .kd]` | 43343 | 0.072319 |
| GDN convolution / `_causal_conv1d_update_kernel.kd` | 43344 | 0.209290 |
| GDN recurrence and gates / `stock_gdn_scan_kernel.kd` | 43344 | 1.231679 |
| GDN output gated normalization / `gdn_norm_quant_kernel.kd` | 43343 | 0.159882 |
| Post-attention/GDN residual/normalization / `void norm_quant<true, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned char*, float*, unsigned short*, long, long, float) [clone .kd]` | 57792 | 0.510119 |
| MLP SiLU and gating / `triton_poi_fused_dynamic_per_token_scaled_fp8_quant_mul_silu_slice_0.kd` | 43344 | 0.127671 |
| MLP SiLU and gating / `triton_poi_fused_dynamic_per_token_scaled_fp8_quant_mul_silu_slice_1.kd` | 14448 | 0.053163 |
| MLP down input FP8 quantization / `void vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<c10::BFloat16, c10::Float8_e4m3fn>(c10::Float8_e4m3fn*, float*, c10::BFloat16 const*, float const*, int, long, long) [clone .kd]` | 57792 | 0.255893 |
| GDN input projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 1, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 43344 | 3.975389 |
| GDN output projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 4, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 43344 | 1.902024 |
| MLP gate/up projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 1, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 57792 | 11.273337 |
| MLP down projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 4, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 57792 | 5.334867 |
| Layer input residual/normalization / `void norm_quant<true, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned char*, float*, unsigned short*, long, long, float) [clone .kd]` | 56889 | 0.491601 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_1.kd` | 14448 | 0.021158 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_3.kd` | 14447 | 0.027012 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_4.kd` | 14448 | 0.026441 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_arange_bitwise_and_eq_index_lt_remainder_select_split_where_2.kd` | 14448 | 0.028067 |
| Attention Q/K normalization, RoPE and layout / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 14448 | 0.028851 |
| Attention Q/K normalization, RoPE and layout / `void stock_m1_gemma_norm<false, 256, 32>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, long, long, float, float*) [clone .kd]` | 14448 | 0.045865 |
| Attention Q/K normalization, RoPE and layout / `void stock_m1_gemma_norm<false, 256, 64>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, long, long, float, float*) [clone .kd]` | 14448 | 0.036312 |
| Attention KV write / `reshape_and_cache_kernel_flash.kd` | 14448 | 0.044625 |
| Attention decode / `void qwen_stock_m1_shared_decode<4, 16, 256, 6, 16, 0, 3430931>(R4DArgs, int) [clone .kd]` | 14448 | 4.764989 |
| Attention split-KV merge / `void qwen_stock_m1_shared_merge<256, 4, 0>(R4DArgs, int, int) [clone .kd]` | 14448 | 0.142420 |
| Attention output gating / `triton_poi_fused_dynamic_per_token_scaled_fp8_quant_mul_sigmoid_view_0.kd` | 14448 | 0.028713 |
| Attention output activation FP8 quantization / `void vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<c10::BFloat16, c10::Float8_e4m3fn>(c10::Float8_e4m3fn*, float*, c10::BFloat16 const*, float const*, int, long, long) [clone .kd]` | 14448 | 0.041861 |
| Attention input projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 1, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 14448 | 1.155984 |
| Attention output projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 4, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 14448 | 0.578146 |
| Final normalization/layout / `void stock_m1_gemma_norm<true, 5120, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, long, long, float, float*) [clone .kd]` | 903 | 0.005004 |
| Other GPU bookkeeping / `__amd_rocclr_copyBuffer.kd` | 22584 | 0.069718 |
| Other GPU bookkeeping / `__amd_rocclr_fillBufferAligned.kd` | 903 | 0.002492 |
| Other GPU bookkeeping / `_apply_write_kernel.kd` | 6 | 0.000031 |
| Other GPU bookkeeping / `_combine_sampled_and_draft_tokens_kernel.kd` | 903 | 0.003059 |
| Other GPU bookkeeping / `_compute_local_logits_stats_kernel.kd` | 903 | 0.028301 |
| Other GPU bookkeeping / `_compute_slot_mappings_kernel.kd` | 903 | 0.002981 |
| Other GPU bookkeeping / `_expand_idx_mapping_kernel.kd` | 903 | 0.001982 |
| Other GPU bookkeeping / `_gather_block_tables_kernel.kd` | 903 | 0.005100 |
| Other GPU bookkeeping / `_get_num_sampled_and_rejected_kernel.kd` | 903 | 0.002757 |
| Other GPU bookkeeping / `_insert_resampled_kernel.kd` | 903 | 0.003508 |
| Other GPU bookkeeping / `_post_update_kernel.kd` | 903 | 0.005465 |
| Other GPU bookkeeping / `_prepare_pos_seq_lens_kernel.kd` | 903 | 0.002116 |
| Other GPU bookkeeping / `_prepare_rope_positions_kernel.kd` | 903 | 0.002895 |
| Other GPU bookkeeping / `_rejection_kernel.kd` | 903 | 0.007279 |
| Other GPU bookkeeping / `_resample_kernel.kd` | 903 | 0.013748 |
| Other GPU bookkeeping / `_scatter_num_accepted_kernel.kd` | 903 | 0.002089 |
| Other GPU bookkeeping / `_zero_kv_blocks_kernel.kd` | 3 | 0.000220 |
| Other GPU bookkeeping / `postprocess_mamba_fused_kernel.kd` | 903 | 0.004473 |
| Other GPU bookkeeping / `precopy_mamba_align_fused_kernel.kd` | 903 | 0.004348 |
| Other GPU bookkeeping / `preprocess_mamba_align_fused_kernel.kd` | 903 | 0.002520 |
| Other GPU bookkeeping / `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}>(int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}, function_traits<at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}>::result_type*) [clone .kd]` | 5418 | 0.010128 |
| Other GPU bookkeeping / `void (anonymous namespace)::softmax_warp_forward<float, float, float, 6, false, false, 32>(float*, float const*, int, int, int, bool const*, int, bool) [clone .kd]` | 903 | 0.002293 |
| Other GPU bookkeeping / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 6321 | 0.014001 |
| Other GPU bookkeeping / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 903 | 0.002546 |
| Other GPU bookkeeping / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 6321 | 0.015387 |
| Other GPU bookkeeping / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<int> const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<int> const&)::{lambda(int, bool)#1}) [clone .kd]` | 5418 | 0.016349 |
| Other GPU bookkeeping / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<float> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<float> const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<float> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<float> const&)::{lambda(int, bool)#1}) [clone .kd]` | 2709 | 0.018553 |
| Other GPU bookkeeping / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 14448 | 0.049182 |
| Other GPU bookkeeping / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 1806 | 0.006077 |
| Other GPU bookkeeping / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 903 | 0.003158 |
| Other GPU bookkeeping / `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 3612 | 0.058564 |
| Other GPU bookkeeping / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, float>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, float*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 3612 | 0.036426 |
| Other GPU bookkeeping / `void at::native::mbtopk::fill<unsigned int, unsigned int>(unsigned int*, unsigned int, unsigned int) [clone .kd]` | 903 | 0.001526 |
| Other GPU bookkeeping / `void at::native::mbtopk::gatherTopK<float, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, float*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 903 | 0.025708 |
| Other GPU bookkeeping / `void at::native::tensor_kernel_scan_innermost_dim<float, std::plus<float> >(float*, float const*, unsigned int, unsigned int, unsigned int, float, std::plus<float>) [clone .kd]` | 903 | 0.002482 |
| Other GPU bookkeeping / `void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>, 4, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast) [clone .kd]` | 5418 | 0.010603 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<16, at::native::BinaryFunctor<bool, bool, bool, at::native::BitwiseOrFunctor<bool> >, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<bool, bool, bool, at::native::BitwiseOrFunctor<bool> >, std::array<char*, 3ul>) [clone .kd]` | 903 | 0.002465 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<16, at::native::FillFunctor<bool>, std::array<char*, 1ul> >(int, at::native::FillFunctor<bool>, std::array<char*, 1ul>) [clone .kd]` | 903 | 0.001842 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<16, at::native::bitwise_not_kernel_cuda(at::TensorIteratorBase&)::{lambda(bool)#1}, std::array<char*, 2ul> >(int, at::native::bitwise_not_kernel_cuda(at::TensorIteratorBase&)::{lambda(bool)#1}, std::array<char*, 2ul>) [clone .kd]` | 903 | 0.002358 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}, std::array<char*, 2ul>) [clone .kd]` | 5418 | 0.012103 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}, std::array<char*, 2ul>) [clone .kd]` | 903 | 0.001803 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}, std::array<char*, 3ul> >(int, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}, std::array<char*, 3ul>) [clone .kd]` | 903 | 0.009871 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool, float, float)#1}, std::array<char*, 4ul> >(int, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool, float, float)#1}, std::array<char*, 4ul>) [clone .kd]` | 1806 | 0.004729 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 2ul> >(int, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 2ul>) [clone .kd]` | 5418 | 0.013593 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, 2ul>) [clone .kd]` | 6321 | 0.012527 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul>) [clone .kd]` | 903 | 0.001904 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>) [clone .kd]` | 903 | 0.002048 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>) [clone .kd]` | 1806 | 0.003233 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int, at::native::FillFunctor<int>, std::array<char*, 1ul>) [clone .kd]` | 903 | 0.001692 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>) [clone .kd]` | 903 | 0.009921 |
| Other GPU bookkeeping / `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool) [clone .kd]` | 903 | 0.002319 |
| Other GPU bookkeeping / `void at::native::warpMergeSortKVInPlace<2, -1, 128, 16, float, long, at::native::GTOp<float, true>, unsigned int, 32>(at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, at::native::GTOp<float, true>, float) [clone .kd]` | 903 | 0.004630 |
| Target head (global512) / `__amd_rocclr_fillBufferAligned.kd` | 903 | 0.002559 |
| Target head (global512) / `_draft_head_int2.kd` | 903 | 0.752175 |
| Target head (global512) / `_rerank_exact.kd` | 903 | 0.054634 |
| Target head (global512) / `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) [clone .kd]` | 903 | 0.004659 |
| Target head (global512) / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 903 | 0.003775 |
| Target head (global512) / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 903 | 0.002963 |
| Target head (global512) / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 903 | 0.005193 |
| Target head (global512) / `void at::native::mbtopk::computeBlockDigitCounts<c10::BFloat16, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 1806 | 0.089550 |
| Target head (global512) / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, c10::BFloat16>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 1806 | 0.041316 |
| Target head (global512) / `void at::native::mbtopk::fill<unsigned int, unsigned int>(unsigned int*, unsigned int, unsigned int) [clone .kd]` | 903 | 0.001607 |
| Target head (global512) / `void at::native::mbtopk::gatherTopK<c10::BFloat16, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 903 | 0.042186 |
| Target head (global512) / `void at::native::radixSortKVInPlace<2, -1, 128, 8, c10::BFloat16, long, unsigned int>(at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, bool) [clone .kd]` | 903 | 0.005655 |
| Target head (global512) / `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4>) [clone .kd]` | 903 | 0.003874 |
| Target head (global512) / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) [clone .kd]` | 903 | 0.001748 |
| Target head (global512) / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>) [clone .kd]` | 903 | 0.002073 |
| Target head (global512) / `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>) [clone .kd]` | 1806 | 0.005175 |

</details>

## Global-512 target-head

Global-512 is the serving default. This paired comparison used identical hidden inputs for every head method from two natural completions starting with the retained 60K Pi prefix: coding: 60,208 input and 10,047 output tokens (stop); reasoning: 70,402 input and 2,598 output tokens (stop). Sampling was temperature 1, top-p 0.95 and top-k 40.

| Target path | Median M8 head time | Same top-1 token | Complete reference top-20 retained | Complete reference top-40 retained |
|---|---:|---:|---:|---:|
| Global INT2 top-256 + BF16 rerank | 1.155 ms | 12,015/12,015 (100%) | 11,988/12,015 (99.7753%) | 11,742/12,015 (97.7278%) |
| Global INT2 top-512 + BF16 rerank (default) | 1.178 ms | 12,015/12,015 (100%) | 12,011/12,015 (99.9667%) | 11,983/12,015 (99.7337%) |
| Full BF16 reference | 4.061 ms | 12,015/12,015 (100%) | 12,015/12,015 (100%) | 12,015/12,015 (100%) |

The comparison covers **12,015 prediction rows**, including prefill and rejected speculative rows, not 12,015 generated tokens. Timing uses 47 eight-row hidden inputs with five randomized-order repetitions: 235 measurements per method. These isolated head timings include native dispatch gaps and exclude comparison/reporting; they do not measure whole-round time or tok/s.

Incomplete top-40 retention occurred in 273 rows with Global-256 and 32 with Global-512; the added median head time was 0.023 ms. Retention includes cutoff ties and does not establish score equality, ordering or identical sampling probabilities. Both shortlists remain approximate; the full BF16 head is the reference for this comparison, not an independently proved model. The drafter is unchanged.

[Methodology and limits](docs/HEAD_CANDIDATE_DEPTH.md) · [Numeric results](benchmarks/results/head-candidate-depth-normalization-20260924.json) · [Earlier Global-256 study](docs/VERIFY_HEAD_GLOBAL_TOPK.md).

## Benchmarks

This benchmark uses the retained 60,000-input-token Pi prefix and the compiled Coherence backend with global512, the attention-page-boundary repair and the pinned-RAM huge-page promotion repair. Five requests are chained in one context: code, prose about code measurement, JSON, thinking/prose and checkpoint generation. Code and JSON disable thinking; both prose requests enable it. All requests stop naturally. Generation uses temperature 1, top-p 0.95, top-k 40 and seed 0; compaction uses temperature 0.3. The checkpoint request forces a snapshot-tail flush. Private fixture text was not decoded or inspected, and no generated text or token arrays were saved. Runner: [benchmark_pi_coding_json_compaction.py](experiments/radiance-public/benchmark_pi_coding_json_compaction.py).

| Stage | Thinking | Prompt tokens | Generated tokens | Classified output | First data | Mean round | Post-first | Peak 3s | Acceptance |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Coding task | off | 60,208 | 4,788 | 4,240 code; 547 prose; 0 separately observed reasoning | 34.24 s | 43.14 ms | 109.93 tok/s | 148.88 tok/s | 53.44% |
| Prose about code measurement | on | 65,137 | 2,044 | 2,043 prose; 0 separately observed reasoning | 2.08 s | 43.31 ms | 87.14 tok/s | 172.89 tok/s | 39.59% |
| JSON task | off | 67,300 | 2,623 | 2,622 JSON; valid JSON | 2.46 s | 43.55 ms | 85.49 tok/s | 118.55 tok/s | 38.87% |
| Thinking/prose task | on | 70,070 | 4,748 | 4,747 prose; 0 separately observed reasoning | 2.13 s | 43.82 ms | 59.41 tok/s | 84.15 tok/s | 22.90% |
| Compaction checkpoint | off | 75,004 | 2,790 | 2,790 checkpoint tokens; completion marker valid; required headings valid | 2.29 s | 44.10 ms | 74.02 tok/s | 94.09 tok/s | 32.33% |

Cached/total prompt tokens: Coding task: 0/60,208, Prose about code measurement: 62,624/65,137, JSON task: 64,272/67,300, Thinking/prose task: 67,568/70,070, Compaction checkpoint: 72,512/75,004.

The phase counters retokenize classified text, so their totals can differ from the backend's emitted-token count. `phase_token_counts_cover_output=false` for Coding task, Prose about code measurement, JSON task, Thinking/prose task, Compaction checkpoint. No separate reasoning channel was exposed for Prose about code measurement, Thinking/prose task; those streams are reported as prose, not relabelled as reasoning. The compaction row measures checkpoint generation with a requested cache flush, not a full Pi transcript commit or old-snapshot retirement. The checkpoint format passed its heading and completion checks. `peak_3s_tokens_per_second` is the maximum completed three-second sliding-window rate after first data, never a single-frame burst.

2026-09-24 rerun status: `complete`. [Numeric results and release identity](benchmarks/results/pi-coding-json-compaction.json). These results use top-k 40, while the older September 20 table used top-k 20, so the output and acceptance changes are not a controlled before/after comparison.

This is the same natural-stop coding task run independently at empty, 60K and 200K input context. Thinking is disabled, EOS remains enabled, and each arm uses temperature 1, top-p 0.95 and top-k 40. The non-empty arms use operator-supplied token-prefix fixtures; only their hashes are retained. The three-second peak is the maximum completed sliding-window rate, not a single-frame burst. Runner: [benchmark_pi_coding_contexts.py](experiments/radiance-public/benchmark_pi_coding_contexts.py).

| Context | Prompt tokens | Generated tokens | Mean round | Post-first | Peak 3s | Acceptance | Round events (total; speculative/expected; timed) | Validation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0K | 203 | 4,512 | 38.91 ms | 121.34 tok/s | 152.75 tok/s | 53.14% | 957 total; 956/956 speculative; 956 timed; captured | short natural stop |
| 60K | 60,208 | 4,788 | 43.16 ms | 110.00 tok/s | 148.79 tok/s | 53.44% | 1,011 total; 1010/1010 speculative; 1,010 timed; captured | short natural stop |
| 200K | 200,208 | 4,030 | 52.69 ms | 90.00 tok/s | 119.81 tok/s | 53.29% | 853 total; 852/852 speculative; 852 timed; captured | short natural stop |

Report status: `complete_with_validation_failure`. [Numeric results and every round](benchmarks/results/pi-coding-contexts.json). Each completed row stores every scheduler event under `contexts.<context>.round_capture.records`; a count mismatch is a validation failure. The target is 5,000 output tokens, with shorter natural completions reported explicitly.

For the next full refresh, use [the shared benchmark suite](docs/BENCHMARK_SUITE.md): `benchmark_pi_coding_contexts.py --suite` reuses each context's predetermined unprofiled control for its coding row and complete histogram, and continues the same 60K output through prose, JSON, thinking and compaction. It keeps both controls around each stage trace for the residual calculation. Existing numbers above retain their original capture provenance.

### Complete per-round capture

The context benchmark now retains every content-free scheduler event for each arm, including an unmeasured first event. The full numeric records are stored under `contexts.<context>.round_capture.records` in the result JSON; this table is a coverage check rather than another latency aggregate. A count mismatch is a validation failure. Expected rounds come from the speculative-round counter; the first prefill event is retained in logged events but excluded from that counter.

| Context | Logged events | Speculative rounds | Expected speculative rounds | Timed round values | Untimed events | Missing round numbers | Capture status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0K | 957 | 956 | 956 | 956 | 1 | none | captured |
| 60K | 1,011 | 1010 | 1010 | 1,010 | 1 | none | captured |
| 200K | 853 | 852 | 852 | 852 | 1 | none | captured |


The histogram below is generated from the complete per-round records of the [coding-context benchmark](benchmarks/results/pi-coding-contexts.json). Every measured `round_ms` value appears in exactly one bin; untimed events are reported separately. The compiled-stage timing experiment has separate unprofiled control runs; their pauses are discussed below and are not part of this histogram.

| Round time | 0K arm | 60K arm | 200K arm |
|---|---:|---:|---:|
| `<35` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `35–37` | 2 (0.2%) | 0 (0.0%) | 0 (0.0%) |
| `37–39` | 427 (44.7%) | 0 (0.0%) | 0 (0.0%) |
| `39–40` | 520 (54.4%) | 0 (0.0%) | 0 (0.0%) |
| `40–40.5` | 2 (0.2%) | 0 (0.0%) | 0 (0.0%) |
| `40.5–41` | 1 (0.1%) | 0 (0.0%) | 0 (0.0%) |
| `41–41.5` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `41.5–42` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `42–43` | 0 (0.0%) | 164 (16.2%) | 0 (0.0%) |
| `43–45` | 1 (0.1%) | 842 (83.4%) | 0 (0.0%) |
| `45–46` | 1 (0.1%) | 0 (0.0%) | 0 (0.0%) |
| `46–47` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `47–48` | 0 (0.0%) | 2 (0.2%) | 0 (0.0%) |
| `48–48.5` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `48.5–49` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `49–51` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `51–52.5` | 1 (0.1%) | 1 (0.1%) | 123 (14.4%) |
| `52.5–53` | 0 (0.0%) | 0 (0.0%) | 704 (82.6%) |
| `53–53.5` | 0 (0.0%) | 1 (0.1%) | 14 (1.6%) |
| `53.5–54` | 0 (0.0%) | 0 (0.0%) | 4 (0.5%) |
| `54–55` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `55–56` | 0 (0.0%) | 0 (0.0%) | 1 (0.1%) |
| `56–60` | 0 (0.0%) | 0 (0.0%) | 1 (0.1%) |
| `60–62.5` | 0 (0.0%) | 0 (0.0%) | 3 (0.4%) |
| `62.5–63` | 0 (0.0%) | 0 (0.0%) | 2 (0.2%) |
| `63–63.5` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `63.5–64` | 1 (0.1%) | 0 (0.0%) | 0 (0.0%) |
| `64–65` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `65–70` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `70–100` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `100–250` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `250–500` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `≥500` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| **Timed rounds** | **956** | **1,010** | **852** |
| **Untimed events** | **1** | **1** | **1** |
| **Mean / median** | **38.91 / 39.17 ms** | **43.16 / 43.14 ms** | **52.69 / 52.61 ms** |


### Known remaining symptoms and likely causes

**Occasional long-context pauses remain.** The evidence comes from the separate compiled-stage timing experiment. Its later unprofiled 200K control, run after the profiling arm, recorded 3 of 752 retained intervals above 100 ms: 2,555.045 ms, 131.085 ms, 177.172 ms. Its median was 52.520 ms. The coding-context histogram above covers a different run; these control intervals are recorded separately. The paired control residuals span 2.685–6.200 ms. This is an intermittent stall, not a sustained increase in every kernel's cost. Its cause is not localized: host scheduling, cache/dependency waits and HIP/ROCr queue state remain candidates. Correlated per-round host and GPU event records are needed to distinguish them. All measured intervals, including the pauses, remain in the [current controls](benchmarks/results/stage26-control-20260924.json).

**Stage attribution has a separate trace limitation.** One 200K cycle recorded normalization after its consuming projection. That inconsistent timestamp order is excluded from per-stage attribution, while the original records are retained. It does not establish that the GPU executed the dependency incorrectly. Profiling can also affect clocks and scheduling indirectly; subtracting traced activity from the clean control is not proof of an exact, observer-free gap total. [Trace witness](benchmarks/results/trace-stage-order-witness-20260924.json) · [timing method](docs/STAGE_TIMING.md).

**The benchmark stream still lacks a separate reasoning channel.** Thinking-enabled requests exposed only prose, so the table cannot isolate their reasoning throughput. The unresolved boundary is the provider/parser metadata path; this observation alone does not show whether internal reasoning was absent. The latest checkpoint passed its section and completion-marker checks. That benchmark covers checkpoint generation and tail flushing; full Pi transcript commit, retirement, cancellation and concurrent-chat recovery need their own integration checks. [Current chained run](benchmarks/results/pi-coding-json-compaction.json).

**Global-512 candidate selection remains approximate.** The current study retained the reference top-1 in 12,015/12,015 rows and the complete top-20 in 12,011/12,015. An excluded vocabulary token can still belong in the reference sampling support. The new exact M1/M8 and eager/compiled confirmations use the full BF16 comparison head; they do not certify shortlist completeness or eliminate model-generated loops. [Current head evidence](benchmarks/results/head-candidate-depth-normalization-20260924.json).

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
