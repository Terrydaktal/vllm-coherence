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

The [eager-M1 repairs](docs/eager-m1-followup-audit.md) remove demonstrated
attention range and intermediate-rounding losses and reject invalid GDN/cache
layouts. The [September 25 qualification](benchmarks/results/current-320-confirmations-20260925.json)
passed 320-token comparisons between M1/M8 and eager/compiled execution,
including every full-logit hash and the prefill prediction. The October 9 refresh
remeasures speed; those correctness captures were not rerun. Independent
attention oracles are recorded separately.

The [independent all-stage M1 audit](docs/eager-m1-all-stages-audit.md), including the
[September 25 rerun with 2,817 passing checks](benchmarks/results/current-operator-confirmations-20260925.json), extends this
to all 64 target layers: public-weight projections, normalization, recurrent and
convolution state, RoPE, attention, cache writes and sampling. It distinguishes
exact checks from numerical-error checks. The [normalization follow-up](docs/eager-m1-contract-qualification.md) eliminates the demonstrated prefill/M1 FP8 counterexamples under a declared arithmetic contract. In the September 27 [prefill alignment study](docs/PREFILL_ALIGNMENT.md), all 2,320 sampled positions matched in their complete hidden and BF16-logit vectors after repairs to three full-model discrepancies. Those numerical checks were not rerun for the October 9 speed refresh. Arbitrary-input equivalence and approximate-head limitations remain open.

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
| Context counter | Used tokens / total capacity, with thousands separators and a percentage. Fresh streamed usage updates it before the answer is saved; checkpoint generation also counts its growing output. After compaction, a prepared prompt estimate is identified until current provider usage arrives. The workspace, counters, temperatures and model form one continuous footer that wraps to the terminal width. |
| Per-chat cache counters | `VRAM` and `RAM` show available cached context; `Disk` shows verified tokens saved to disk; `Cold` estimates tokens that must be recomputed. Disk backup can overlap GPU/RAM residency, so these four figures are not a partition to add together. |
| Shared hardware monitoring | GPU junction temperature, edge temperature and fan percentage, then used / total VRAM in GiB, Pi's input/output/cache usage figures and model name. One shared probe serves multiple windows: hardware readings update every second, cache/scheduler metadata every 0.5 seconds. |
| Specific working spinner | One line updates through request preparation, admission, GPU queueing, RAM allocation, cache-bank handover, cache updates, reusable-context checks, snapshot loading, prefill and first output. Tools and continuation preparation also have timed states. Queued chats identify the blocking chat when telemetry is available. The line refreshes every 100 ms using the existing shared telemetry reader; normal generation retains its rate display. |
| Honest generation rates | A three-second rolling rate followed by the average, latest round time, three-second acceptance, token counts, time to first data and elapsed time. Generation and compaction displays refresh every 100 ms from shared telemetry. Time spent waiting for another chat is excluded from generation rates; missing reasoning counts are omitted. |
| Tool-call visibility | Token usage continues updating while tool arguments are buffered. `generating edit arguments` and `applying edit` distinguish model work from tool execution, with an execution timer. Tool-continuation preparation remains visible until the next request. |
| Backend error details | Failed requests automatically collect bounded backend and connection evidence. `Ctrl+O` expands the recorded traceback, container state, host boot identity and disk-space evidence when available; `/backend-error` repeats the lookup. |
| Backend control | `/backend status`, `/backend start` and `/backend stop` show shared backend state and timed startup/shutdown stages. Stop drains active requests and flushes cached tails. |
| Sampling settings | `/sampling` shows configured temperature, top-p and top-k, and the settings observed on the latest request when available. |
| Loaded tools | `/tools` lists registered tool names, descriptions and enabled status from the current Pi session, without a model request. |

Unavailable telemetry is reported as unavailable rather than presented as zero
cached tokens or an invented explanation for a wait.

### Compaction, snapshots and history

| Feature | What Coherence adds |
| --- | --- |
| Transactional compaction | Manual and automatic compaction preserve the original transcript unless the checkpoint passes section, completion-marker and finish-status checks. The prompt preserves the existing token prefix for cache reuse, and checkpoint output can use the remaining context capacity. `Esc` cancels; `Alt+C` explicitly accepts the exact partial summary already generated, without a replacement request. |
| Interrupted answers | Received Qwen thinking and prose survive cancellation in the session and active history. Unfinished tool calls remain archival records and never execute. |
| Compaction during generation | Fresh streamed usage interrupts a Qwen answer at 240,000 context tokens or earlier to preserve the configured reserve (237,408 with the default reserve). Received thinking and prose are saved before checkpoint generation; unfinished tool calls never execute. A committed checkpoint resumes the task without a new user message. Cancellation or failure leaves the partial answer saved. |
| Working-state continuity | Checkpoints lead with the current objective, latest correction and unfinished action, preserve an incremental decision/constraint ledger, and carry bounded source excerpts, tool-outcome pointers and fresh Git state. An atomic recent tail keeps tool calls with results; `pi_session_search` recovers older evidence. No extra model call or background GPU work. [Memory design and limits](docs/COMPACTION_MEMORY.md). |
| Purge earlier thinking | `/purge-thinking` excludes current thinking from future prompts and compaction while retaining thinking generated afterward. Its cutoff is saved per chat branch; prose, tools and the original transcript are preserved. `/purge-thinking status` shows the policy. |
| Select active context | `/context` selects messages, turns or ranges to exclude or restore, with an exact CPU-tokenized preview. Changes apply to ordinary requests and checkpoints; the original transcript remains intact. Tool calls and results stay paired. |
| Visible compaction steps | An in-place progress display times prompt preparation, queueing, cache loading, prefill, checkpoint generation, validation, tail flush, commit and cleanup. Checkpoint generation shows tokens, three-second and post-first average t/s, round time and three-second acceptance. The completed compaction entry retains total elapsed time. Typed editor input survives compaction. |
| Snapshot restore | Cache identity binds the chat, compaction generation, exact prefix and runtime compatibility. Compatible state can be reused from GPU, RAM or compressed disk snapshots; only the missing suffix needs prefill. |
| Response-end reuse | DFlash retains the exact processed response end, including target/drafter KV and matching GDN/convolution state. An unchanged next prompt reuses the partial block too, avoiding the inherited extra-block replay. Ownership transfers to the continuation, and optional historical GDN snapshots release space under allocation pressure. An emitted but unprocessed token still needs processing. [State contract and verification](docs/RESPONSE_END_CACHE.md). |
| Generated-token history | Unchanged chat turns retain the exact tokens previously generated, avoiding multi-thousand-token replay when re-encoding identical text produces different IDs. Unchanged history is checked before tokenization and only the new rendered suffix is encoded. Strict history, delivered-output, full tokenizer-backend, template and decoded-text checks guard reuse; edited or unsupported inputs use canonical rendering. This preserves generated history rather than promising equality to full-history text re-encoding. [Continuation contract and private journal](docs/CACHE_ROUND_DIAGNOSTICS.md#generated-token-continuation). |
| Less disk rewriting | Immutable blocks are reused and the changing tail stays in memory. It normally flushes after about 8,192 new tokens, and on explicit flush, RAM eviction, successful compaction and clean shutdown. The previous complete disk head remains valid until its replacement is verified. |
| Cleanup across compactions | A committed compaction supersedes the old generation, retires its snapshots and prevents stale requests from bringing it back. Cleanup failures are reported and retried. The superseded GPU bank is discarded without parking the obsolete generation in RAM. |
| Inspectable cache storage | The live cache dashboard reads shared telemetry every 100 ms and shows each chat's residency, context coverage, disk size and checkpoint state. Disk inventory refreshes separately. Lifetime disk traffic includes recorded writes from deleted chats; synthetic test chats can be purged. |
| Project-local history | Model-neutral transcripts live in the workspace's `.pi/sessions`; `--session last` resumes its latest chat. Transcript reuse across models is separate from model-specific KV-cache compatibility. |
| Searchable original history | `pi_session_search` retrieves bounded excerpts from original JSONL messages, including history removed by compaction. A lean CPU SQLite index keeps word positions and source offsets without another transcript-text copy. The current branch is the default; other project sessions and alternate branches require explicit scope. [Search and storage contract](docs/SESSION_SEARCH.md). |
| Persistent plans and planning mode | `manage_task_plan` maintains the objective, step status, relevant files and constraints as structured session state. Pi restores it after compaction without repeated plan-file reads. `/plan` toggles read-only investigation; `/plan execute` resumes implementation. [Commands and limits](docs/PLAN_MODE.md). |
| Working-state restoration | Compaction restores bounded source evidence, fresh Git state, up to five relevant files and recorded task handles, prioritizing unfinished work. Loaded instructions stay in the system prompt; historical process handles remain unverified. [Restoration contract](docs/COMPACTION_MEMORY.md). |
| Bounded tool output with retrieval | Large tool results get compact context views while their original available text remains in a SHA-256-verified archive. `rehydrate_tool_result` retrieves line ranges or literal-match excerpts, preserving matches ahead of context under its limits. [Archive and retrieval contract](docs/TOOL_REHYDRATION.md). |
| Reproducible local/remote setup | The launcher installs and checks the patched Pi runtime, loads the operating prompt and extensions, and automatically reserves a free SSH forwarding port. A custom search extension can supply a VM-specific tool. |
| Reusable startup warmup | A warmup receipt binds the running container and release identity. Valid receipts avoid repeating warmup when resuming another chat; missing or changed identities require verification. |

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

The speed measurements below are dated **2026-10-09** and use the compiled serving build at [`99d5fcf`](https://github.com/Terrydaktal/vllm-coherence/commit/99d5fcfd1bcd0fe6f70445d154e63b7092b02f24), with the approximate Global-512 target head. The table below retains the **2026-09-25 correctness study**, which was not rerun for this speed refresh: 320 forced decode positions after a fresh 60,000-token Pi prefill, plus the prefill prediction, in each of four arms. Compiled M1 versus compiled M8, eager M1 versus compiled M1, eager M8 versus compiled M8, and eager M1 versus compiled M8 all matched. Those comparisons use the full BF16 head to expose target-body differences. [Correctness capture and tested identities](benchmarks/results/current-320-confirmations-20260925.json).

| Prediction | Same token set | Same ordering | Mean shared tokens |
| --- | ---: | ---: | ---: |
| Top 1 | 320 / 320 (100.00%) | 320 / 320 (100.00%) | 1.0000 / 1 |
| Top 10 | 320 / 320 (100.00%) | 320 / 320 (100.00%) | 10.0000 / 10 |
| Top 20 | 320 / 320 (100.00%) | 320 / 320 (100.00%) | 20.0000 / 20 |

Each comparison also matched all 320 full-vocabulary logit hashes and the prefill prediction. This establishes finite execution consistency on the retained Pi corpus, not independent mathematical certification or arbitrary-input equivalence. The earlier 10K study and the cache/prefill correctness captures remain dated historical evidence.

## Compiled backend stages

Measured on 2026-10-09 using the compiled, optimized Global-512 serving backend with the pinned-RAM huge-page promotion repair; sampling is temperature 1.0, top-p 0.95 and top-k 40. Context labels are starting prefixes: 0K, the private 60K Pi fixture, and the public synthetic 200K fixture. Each context ran a natural warmup followed by clean control, trace, and clean control; each arm generated 4,912 / 5,675 / 7,948 tokens respectively. Generated-token hashes and accepted-token schedules matched. The stage means retain 1003 / 1134 / 1134 complete M8 cycles (0K / 60K / 200K), and controls use exactly those same decode indices. The shared suite has a nominal 1,152-call trace budget after 64 warmup calls, plus one closing boundary per 128-call chunk; only complete eight-row target cycles enter the stage means. Each chunk's closing boundary and first two trace-activation cycles are excluded; natural EOS can end the capture earlier. Trace setup/export boundaries and incomplete or inconsistent trace inventories are excluded by structure, never by duration; complete native round logs retain all rounds and stalls. GPU activity timestamps supply the stage times; CPU annotations, Python hooks and export time are excluded. No per-stage event probes or forced-token replay are used. The measured tracing slowdown was 2.769 / 2.665 / 3.278 ms per retained round; it is reported separately and is **not charged to row 26**. Row 26 is the clean control mean minus the union of GPU activity intervals. This remains an estimate: tracing can indirectly affect clocks and scheduling. Overlap is counted once in the total. The header identifies the measured source commit. The capture retains its original checkout and source identities. [Capture and source identities](benchmarks/results/compiled-global512-stage-profile-20261009.json) · [controls](benchmarks/results/stage26-control-20261009.json) · [method and uncertainty](docs/STAGE_TIMING.md).

The backend includes the full-graph cache-preparation repair and measurement adapters described in the [September 25 speed investigation](docs/SPEED_INVESTIGATION_20260925.md). Captured checkout identities and installed source and binary hashes bind the measured implementation. The separately dated [Pi deployment receipt](benchmarks/results/token-continuation-deployment-20261008.json) records its installed identities; the isolated profiling worker is not the live Pi server.

**Correctness comparisons were not rerun.** The speed measurements dated 2026-10-09 use [`99d5fcf`](https://github.com/Terrydaktal/vllm-coherence/commit/99d5fcfd1bcd0fe6f70445d154e63b7092b02f24); the M1/M8 and eager/compiled correctness cells retain the 2026-09-25 study, its original release and source bindings. Those historical passes do not establish correctness of the newly timed build. User requested speed measurements only.

The **eager M1** column combines the historical [25 September independent operator audit](benchmarks/results/current-operator-confirmations-20260925.json) (**2,817 checks, zero failures**, 320 synthetic rows at the audited sites) with the broader [24 September arithmetic-contract qualification](docs/eager-m1-contract-qualification.md). The [declared arithmetic](docs/M1_ARITHMETIC_CONTRACT.md) separates accepted weight, activation and KV quantization from implementation defects. Confidence scores are editorial assessments: **1/5** untested, **2/5** limited, **3/5** moderate, **4/5** high, **5/5** very high for the tested exact encoding/index/state properties. They are **not probabilities of being bug-free**; 5/5 is not an arbitrary-input proof. Independent FP64 equations use declared error criteria; numerical differences are retained and a tolerance pass is not byte equality.

The historical [25 September whole-model comparisons](benchmarks/results/current-320-confirmations-20260925.json) cover 320 forced decode positions and the prefill prediction in each of four path comparisons, including eager M1 versus compiled M8. Exact full-logit hashes establish execution consistency on those inputs, not independent mathematical correctness of the complete model. The [24 September normalization release gates](benchmarks/results/eager-m1-normalization-deployment-20260924.json) additionally cover 128 hidden-normalization sites and 48 GDN sites with 1,000 rows. The [27 September prefill alignment study](docs/PREFILL_ALIGNMENT.md) adds exact complete hidden/logit-vector comparisons on 2,320 synthetic positions. Arbitrary-input equality, approximate-head completeness and exhaustive concurrency remain open. The [25 September native snapshot lifecycle test](benchmarks/results/snapshot-lifecycle-20260925.json) passed A-to-B-to-A handover, disk restore after restarting the backend and three verified replacement cycles; those are finite historical lifecycle checks.

The historical [26 September cancellation and concurrent-chat qualification](docs/SPEED_LIFECYCLE_20260926.md) passed all 11 cases both with dispatch observation and on the plain worker, including cold-prefill cancellation, queued cancellation, priority handover and sampled replay. It repaired queued-cancellation cleanup and uninitialized GDN history for fresh one-token prompts. This covers the tested serial GPU scheduler with two resident chat banks, not numerical two-request GPU batching. The later prefill arithmetic repair deliberately changed the snapshot data ABI; historical lifecycle evidence is not a new qualification of a later build.

The historical [25 September isolated stage run](benchmarks/results/current-stage-confirmations-20260925.json) covers 22 boundaries and 770 layer instances over the same 320 Pi tokens. Its M1/M8 and eager/compiled M8 pairs match local tensors and state, full-logit hashes, and top-1/10/20 sets and ordering at every tested stage. All 40 eight-token groups pass deliberate corruption controls and restore authoritative state. These full-model activation replays add integration evidence to the independent operator checks, but share native implementations and are not independent mathematical references.

The [9 October head comparison](benchmarks/results/head-candidate-depth-20261009.json) measures the frozen [`99d5fcf`](https://github.com/Terrydaktal/vllm-coherence/commit/99d5fcfd1bcd0fe6f70445d154e63b7092b02f24) source tree. Its recall and head timings remain separate from historical operator coverage and complete-model path equality.

| Stage | Current GPU activity per retained compiled M8 cycle (0K / 60K / 200K; milliseconds unless explicitly marked; 2026-10-09; run [`99d5fcf`](https://github.com/Terrydaktal/vllm-coherence/commit/99d5fcfd1bcd0fe6f70445d154e63b7092b02f24)) | Current M1->M8 correctness and eager->compiled correctness evidence | Current Eager M1 correctness evidence | Last relevant code commit / change | What this stage does |
| --- | ---: | --- | --- | --- | --- |
| **1. Drafter** | 4.799 / 4.912 / 4.946 | Separate proposal model; target M1/M8 comparisons do not independently qualify it. | **N/A: separate proposal model.** Independent proposal/target RNG controls passed. **Uncertainty:** this target-M1 audit does not independently qualify drafter arithmetic or every speculative commit path. | [Qualified speed changes](docs/SPEED_INVESTIGATION_20260925.md): Use the qualified unit_w4_s1_occ2 drafter attention launch; target attention is unchanged and deferred head initialization is retained. | Suggests up to seven tokens for the target model to check. |
| **2. Embedding + first input normalization + FP8 production** | 0.009 / 0.009 / 0.009 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [September 25 study](benchmarks/results/current-stage-confirmations-20260925.json) · source binding `6a392be8004d`. | **4/5 High.** Historical 24 September exact sampled embeddings and independent RMS checks; released fused norm preserves M1 FP8 bytes/scales at tested batch boundaries from 1 to 2,048 rows. **Uncertainty:** finite inputs and indices. Later complete-model prefill/continuation samples match; arbitrary-input equivalence to fully serial decode is not proved. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 reduction and BF16/FP8 rounding at every admitted prefill width, removing batch-dependent activation differences. | Looks up token vectors, normalizes them and creates FP8 inputs in one kernel, preserving intermediate rounding. |
| **3. Layer input residual/normalization + FP8 production** | 0.483 / 0.495 / 0.511 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [September 25 study](benchmarks/results/current-stage-confirmations-20260925.json) · source binding `6a392be8004d`. | **4/5 High.** The 24 September qualified release matches native M1 plus independent FP8 encoding at 129 hidden-norm weights × 320 rows; original 55 byte differences eliminated. CPU arithmetic reproduces the saved failing rows. **Uncertainty:** finite inputs, intrinsic/denormal behavior and untested inputs; deployed 24 September. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 reduction and BF16/FP8 rounding at every admitted prefill width, removing batch-dependent activation differences. | Adds the residual, normalizes the result and creates FP8 inputs in one kernel, preserving intermediate rounding. |
| ↳ GDN input activation FP8 quantization | Included in **stage 3** | Exact fused FP8 bytes/scales; see stage 3 and its evidence scope | **5/5 Very high for tested encoding.** The 24 September qualified release fused bytes/scales match independent encoding; stage 3 prefill/M1 counterexamples pass in that historical qualification. **Uncertainty:** inherits normalization arithmetic; FP8 is an explicitly accepted approximation; deployed 24 September. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 reduction and BF16/FP8 rounding at every admitted prefill width, removing batch-dependent activation differences. | Uses the FP8 output produced by the same input-normalization kernel. |
| ↳ Attention input activation FP8 quantization | Included in **stage 3** | Exact fused FP8 bytes/scales; see stage 3 and its evidence scope | **5/5 Very high for tested encoding.** The 24 September qualified release fused bytes/scales match independent encoding; stage 3 prefill/M1 counterexamples pass in that historical qualification. **Uncertainty:** inherits normalization arithmetic; FP8 is an explicitly accepted approximation; deployed 24 September. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 reduction and BF16/FP8 rounding at every admitted prefill width, removing batch-dependent activation differences. | Uses the FP8 output produced by the same input-normalization kernel. |
| **4. GDN input projection** | 4.030 / 3.964 / 3.983 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [September 25 study](benchmarks/results/current-stage-confirmations-20260925.json) · source binding `6a392be8004d`. | **4/5 High.** Shared audit-wide coverage: the 25 September audit checks 496 matrices with 320 synthetic inputs each at sampled output channels. The separate 24 September qualification checks all channels of 30 selected matrices on 32 inputs each (7,739,392 outputs), plus 12 operator-generated activation replays (3,276,800 outputs). These are shared totals, not additional totals for each projection row. **Uncertainty:** other matrices retain sampled-channel coverage; FP64 tolerance passes are not exact-arithmetic proofs. Complete-model stage replays use model activations but share the native operator, not an independent FP64 oracle. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve stored MXFP4 coefficients in the folded lookup and repair fragment/tiled per-block fallbacks while retaining the qualified GEMM dispatch. | Projects hidden vectors into the inputs and gates for the recurrent layer. |
| **5. GDN layout/copies and buffer initialization** | 0.193 / 0.200 / 0.204 | Authoritative cache/state restored in all 40 eight-token groups; state/output corruption controls detected · [September 25 study](benchmarks/results/current-stage-confirmations-20260925.json) · source binding `6a392be8004d`. No separate layout top-20 attribution. | **4/5 High in the tested replay.** Unsupported strides/overlap rejected; final histories and untouched slots exact across 48 GDN sites. The historical 25 September replay restores authoritative cache/state in all 40 eight-token groups, including corruption controls. **Uncertainty:** separately dated snapshot and cancellation studies cover finite handover/restore/ownership cases, not every possible history. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Reject unsupported packed GDN inner strides and overlapping state slots; retain the existing nine-slot layout. | Arranges inputs and clears temporary buffers throughout each GDN layer. |
| **6. GDN convolution** | 0.178 / 0.182 / 0.187 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [September 25 study](benchmarks/results/current-stage-confirmations-20260925.json) · source binding `6a392be8004d`. | **4/5 High.** Historical 25 September audit: 48 layers × 320 consecutive inputs pass FP64 convolution/SiLU criteria; histories and untouched slots exact. **Uncertainty:** finite numerical/state samples; full-session integration remains unqualified by this audit. | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Aligned GDN convolution product and accumulation arithmetic across M1/M8 and eager/compiled execution. | Updates recent-token history using the corrected multiply/add order. |
| **7. GDN recurrence and gates** | 1.336 / 1.177 / 1.209 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [September 25 study](benchmarks/results/current-stage-confirmations-20260925.json) · source binding `6a392be8004d`. | **4/5 High.** Historical 25 September audit: 48 layers × 320 transitions match independent state/output equations within tolerance; 32,768-step decay passes. **Uncertainty:** no exact full-model recurrence proof or coverage of every reachable state. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Use stable log1p(exp(x)) in M1, M8 and prefill so small nonzero recurrent gates survive FP32 cancellation. | Updates recurrent memory in token order. Faster prefill retains the existing nine-slot state layout. |
| **8. GDN output gated normalization + FP8 production** | 0.145 / 0.147 / 0.151 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [September 25 study](benchmarks/results/current-stage-confirmations-20260925.json) · source binding `6a392be8004d`. | **4/5 High numerically.** Explicit FP32 norm/weight/gate contract with one final BF16 boundary; the 24 September released path matches native M1 at all 48 sites × 320 audit rows, plus 1,000-row release gates and batch-boundary checks. **Uncertainty:** this deliberately differs from Hugging Face intermediate casts; intrinsic behavior and arbitrary inputs remain unproved; deployed 24 September. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 gated-normalization reduction layout during prefill and retain one final BF16 rounding before FP8 production. | Normalizes and gates GDN output, then produces FP8 inputs in one kernel; intermediate BF16 rounding is preserved. |
| ↳ GDN output activation FP8 quantization | Included in **stage 8** | Exact fused FP8 bytes/scales; see stage 8 and its evidence scope | **5/5 Very high for tested encoding.** The 24 September qualified M1/prefill bytes and scales match independent encoding of native BF16 output at 48 sites × 320 rows. **Uncertainty:** inherits the declared stage 8 arithmetic; no weight-only model equivalence; deployed 24 September. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 gated-normalization reduction layout during prefill and retain one final BF16 rounding before FP8 production. | Uses the FP8 output produced by the same gated-normalization kernel. |
| **9. GDN output projection** | 1.827 / 1.810 / 1.815 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [September 25 study](benchmarks/results/current-stage-confirmations-20260925.json) · source binding `6a392be8004d`. | **4/5 High.** Shared audit-wide coverage: the 25 September audit checks 496 matrices with 320 synthetic inputs each at sampled output channels. The separate 24 September qualification checks all channels of 30 selected matrices on 32 inputs each (7,739,392 outputs), plus 12 operator-generated activation replays (3,276,800 outputs). These are shared totals, not additional totals for each projection row. **Uncertainty:** other matrices retain sampled-channel coverage; FP64 tolerance passes are not exact-arithmetic proofs. Complete-model stage replays use model activations but share the native operator, not an independent FP64 oracle. | [Qualified speed changes](docs/SPEED_INVESTIGATION_20260925.md): Use qualified local-split GEMM for M1/M8 down/output shapes; preserve the repaired MXFP4 coefficients and reduction order. | Maps the recurrent-layer result back to the model's hidden-vector width. |
| **10. Attention input projection** | 1.159 / 1.159 / 1.155 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [September 25 study](benchmarks/results/current-stage-confirmations-20260925.json) · source binding `6a392be8004d`. | **4/5 High.** Shared audit-wide coverage: the 25 September audit checks 496 matrices with 320 synthetic inputs each at sampled output channels. The separate 24 September qualification checks all channels of 30 selected matrices on 32 inputs each (7,739,392 outputs), plus 12 operator-generated activation replays (3,276,800 outputs). These are shared totals, not additional totals for each projection row. **Uncertainty:** other matrices retain sampled-channel coverage; FP64 tolerance passes are not exact-arithmetic proofs. Complete-model stage replays use model activations but share the native operator, not an independent FP64 oracle. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve stored MXFP4 coefficients in the folded lookup and repair fragment/tiled per-block fallbacks while retaining the qualified GEMM dispatch. | Projects hidden vectors into the inputs for attention. |
| **11. Attention Q/K normalization, RoPE and layout** | 0.208 / 0.212 / 0.217 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [September 25 study](benchmarks/results/current-stage-confirmations-20260925.json) · source binding `6a392be8004d`. | **4/5 High.** Historical 25 September audit: All 32 Q/K norm sites; both position paths at nine ranges pass independent arithmetic checks; repaired eager MRoPE launches verified. **Uncertainty:** rounding tolerances and compact RoPE tables; full-table addressing not covered. | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Added live-row handling for compiled M8 graph padding while retaining the repaired RoPE arithmetic. | Normalizes queries and keys and applies positional rotation (RoPE), retaining the corrected rounding. |
| **12. Attention KV write** | 0.034 / 0.035 / 0.035 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [September 25 study](benchmarks/results/current-stage-confirmations-20260925.json) · source binding `6a392be8004d`. | **5/5 Very high within tested cases.** Historical 25 September audit: 393,216 stored values/destinations exact; BF16/OCP FP8 scales, masked slots, page edges and every finite FP8 code checked. **Uncertainty:** arbitrary aliasing, asynchronous access and native snapshot restoration remain unproved. | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Aligned causal attention and cache writes with the M1/M8 arithmetic contract. | Stores new keys and values in the cache for reuse by later tokens. |
| **13. Attention decode** | 0.371 / 4.754 / 13.960 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [September 25 study](benchmarks/results/current-stage-confirmations-20260925.json) · source binding `6a392be8004d`. Decode and merge checked together. | **4/5 High within tested cases.** Historical operator evidence: Earlier 88 cases plus 96 independent FP64 cases with unique shuffled pages, holes, poisoned tails, per-head scales and varied contents through 60,001 tokens; guards intact. **Uncertainty:** longest 253K checks still use structured/repeated pages; exact softmax and arbitrary layouts remain unproved. | [September 23 attention-page repair](experiments/radiance-public/build_stock_m1_attention_shared.py): reuse the context traversal when M8 queries cross a 16-token attention-page boundary; source/binary hashes are in the capture. | Attends to the current and earlier tokens using corrected arithmetic and shared cache loads. |
| **14. Attention split-KV merge** | 0.260 / 0.133 / 0.107 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [September 25 study](benchmarks/results/current-stage-confirmations-20260925.json) · source binding `6a392be8004d`. Decode and merge checked together. | **4/5 High for repaired arithmetic.** Historical operator evidence: Four exact split-mean failures repaired; 96 additional varied-layout attention cases pass, including exact BF16 cancellation controls. **Uncertainty:** integrated evidence, not an exhaustive independent merge-stage proof. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Keep split outputs and normalization sums in FP32 until the final BF16 merge, removing intermediate FP16 rounding loss. | Combines attention results from cache partitions in the corrected arithmetic order. |
| **15. Attention output gating** | 0.021 / 0.025 / 0.025 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [September 25 study](benchmarks/results/current-stage-confirmations-20260925.json) · source binding `6a392be8004d`. | **4/5 High.** Historical 25 September audit: Every finite BF16 sigmoid input tested with a bounded multiplier against FP64 equations. **Uncertainty:** numerical tolerances; not every gate/output pair or full attention-to-gate integration. | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Preserved attention output gating and intermediate BF16 rounding in aligned M1/M8 execution. | Applies learned gates to the attention output. |
| **16. Attention output activation FP8 quantization** | 0.036 / 0.042 / 0.043 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [September 25 study](benchmarks/results/current-stage-confirmations-20260925.json) · source binding `6a392be8004d`. | **4/5 High for tested encoding and integration.** Historical independent FP8 codec checks plus the 25 September 320-token M1/M8 and eager/compiled quantizer replay agree at every attention layer. **Uncertainty:** independent oracle checks remain sampled; arbitrary gate/quantization inputs and layouts are unproved. | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Enabled qualified native-rounding FP8 output production and rejected approximate traced quantization. | Converts the attention output to FP8 for its output projection. |
| **17. Attention output projection** | 0.585 / 0.535 / 0.514 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [September 25 study](benchmarks/results/current-stage-confirmations-20260925.json) · source binding `6a392be8004d`. | **4/5 High.** Shared audit-wide coverage: the 25 September audit checks 496 matrices with 320 synthetic inputs each at sampled output channels. The separate 24 September qualification checks all channels of 30 selected matrices on 32 inputs each (7,739,392 outputs), plus 12 operator-generated activation replays (3,276,800 outputs). These are shared totals, not additional totals for each projection row. **Uncertainty:** other matrices retain sampled-channel coverage; FP64 tolerance passes are not exact-arithmetic proofs. Complete-model stage replays use model activations but share the native operator, not an independent FP64 oracle. | [Qualified speed changes](docs/SPEED_INVESTIGATION_20260925.md): Use qualified local-split GEMM for M1/M8 down/output shapes; preserve the repaired MXFP4 coefficients and reduction order. | Maps the attention result back to the model's hidden-vector width. |
| **18. Post-attention/GDN residual/normalization + FP8 production** | 0.495 / 0.510 / 0.527 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [September 25 study](benchmarks/results/current-stage-confirmations-20260925.json) · source binding `6a392be8004d`. | **4/5 High.** The 24 September qualified release matches native M1 plus independent FP8 encoding in the 129-site × 320-row hidden-norm audit; original 55 byte differences eliminated. **Uncertainty:** finite inputs and prefill-versus-serial-decode behavior; deployed 24 September. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 reduction and BF16/FP8 rounding at every admitted prefill width, removing batch-dependent activation differences. | Adds the layer result to the residual, normalizes it and creates FP8 inputs for the MLP in one kernel. |
| ↳ MLP gate/up input FP8 quantization | Included in **stage 18** | Exact fused FP8 bytes/scales; see stage 18 and its evidence scope | **5/5 Very high for tested encoding.** The 24 September released bytes/scales match independent encoding of M1 BF16 norm; stage 18 prefill counterexamples pass in that historical qualification. **Uncertainty:** inherits normalization arithmetic and accepted FP8 loss; deployed 24 September. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 reduction and BF16/FP8 rounding at every admitted prefill width, removing batch-dependent activation differences. | Uses the FP8 output produced by the same post-normalization kernel. |
| **19. MLP gate/up projection** | 11.431 / 11.150 / 11.144 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [September 25 study](benchmarks/results/current-stage-confirmations-20260925.json) · source binding `6a392be8004d`. | **4/5 High.** Shared audit-wide coverage: the 25 September audit checks 496 matrices with 320 synthetic inputs each at sampled output channels. The separate 24 September qualification checks all channels of 30 selected matrices on 32 inputs each (7,739,392 outputs), plus 12 operator-generated activation replays (3,276,800 outputs). These are shared totals, not additional totals for each projection row. **Uncertainty:** other matrices retain sampled-channel coverage; FP64 tolerance passes are not exact-arithmetic proofs. Complete-model stage replays use model activations but share the native operator, not an independent FP64 oracle. | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve stored MXFP4 coefficients in the folded lookup and repair fragment/tiled per-block fallbacks while retaining the qualified GEMM dispatch. | Computes both MLP input projections. Wider decode dispatch avoids the slower prefill kernel for qualified shapes. |
| **20. MLP SiLU and gating** | 0.122 / 0.134 / 0.137 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [September 25 study](benchmarks/results/current-stage-confirmations-20260925.json) · source binding `6a392be8004d`. | **4/5 High.** Historical 25 September audit: Every finite BF16 gate encoding checked with a bounded multiplier; 320 fused M1 cases pass. **Uncertainty:** tiny subnormal differences from FP64 remain; not all gate/up pairs are covered. | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Preserved intermediate BF16 rounding in aligned M1/M8 and eager/compiled arithmetic; the slower fused SiLU remains disabled. | Applies SiLU and combines the two MLP branches, retaining intermediate BF16 rounding. |
| **21. MLP down input FP8 quantization** | 0.242 / 0.255 / 0.262 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [September 25 study](benchmarks/results/current-stage-confirmations-20260925.json) · source binding `6a392be8004d`. | **5/5 Very high for tested encoding.** Historical 25 September audit: 320 fused M1 cases exactly match independent FP8 encoding of the native BF16 SiLU/gate pipeline. **Uncertainty:** inherits stage 20 numerical behavior; no exhaustive input-pair or shape proof. | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Enabled qualified native per-token FP8 production for the down projection input. | Converts MLP activations to FP8 for the down projection. |
| **22. MLP down projection** | 5.464 / 5.313 / 5.322 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [September 25 study](benchmarks/results/current-stage-confirmations-20260925.json) · source binding `6a392be8004d`. | **4/5 High.** Shared audit-wide coverage: the 25 September audit checks 496 matrices with 320 synthetic inputs each at sampled output channels. The separate 24 September qualification checks all channels of 30 selected matrices on 32 inputs each (7,739,392 outputs), plus 12 operator-generated activation replays (3,276,800 outputs). These are shared totals, not additional totals for each projection row. **Uncertainty:** other matrices retain sampled-channel coverage; FP64 tolerance passes are not exact-arithmetic proofs. Complete-model stage replays use model activations but share the native operator, not an independent FP64 oracle. | [Qualified speed changes](docs/SPEED_INVESTIGATION_20260925.md): Use qualified local-split GEMM for M1/M8 down/output shapes; preserve the repaired MXFP4 coefficients and reduction order. | Projects the MLP result back to the model's hidden-vector width. |
| **23. Final normalization/layout** | 0.012 / 0.012 / 0.012 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [September 25 study](benchmarks/results/current-stage-confirmations-20260925.json) · source binding `6a392be8004d`. | **4/5 High.** The historical 25 September audit checks 320 inputs at the actual final BF16 norm against independent RMS equations. **Uncertainty:** sampled numerical criteria; complete-model mode agreement is execution consistency, not an independent reference proof. The extra FP8 control is not serving behavior. | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Aligned final normalization and full BF16 head precision boundaries while preserving the rounding contract. | Normalizes the final hidden vector before vocabulary scoring. |
| **24. Global-512 target head** | 1.015 / 1.020 / 1.050 | Same top-1: 12,148/12,148; complete reference top-20 retained: 12,143/12,148. Includes M1/M8; not an M1-versus-M8 ordering test · [current head study](benchmarks/results/head-candidate-depth-20261009.json). | **4/5 for score arithmetic; selection known approximate.** The historical 25 September operator audit checks three 512-row weight slabs × 320 synthetic inputs against FP64 criteria. The 9 October Global-512 study retains top-1 in 12,148/12,148 rows, complete top-20 in 12,143/12,148, and complete top-40 in 12,118/12,148. **Uncertainty:** sampled scores, retained-logit differences and uncertified excluded tokens; recall includes M1 and M8 and is distinct from independent M1 score checks. [Head evidence](docs/HEAD_CANDIDATE_DEPTH.md). | [`9bb795d`](https://github.com/Terrydaktal/vllm-coherence/commit/9bb795d2e612c76087d16932841131edf4834d5e): increase target shortlist to 512; drafter unchanged. Extends the Global-256 method from [`31add3d`](https://github.com/Terrydaktal/vllm-coherence/commit/31add3d06080be6e4f5198c6e8afcc6276e1edaf). | Scores the vocabulary with INT2, selects 512 candidates and rescores them with BF16 weights. Selection remains approximate. |
| **25. Other GPU bookkeeping** | 0.471 / 0.480 / 0.493 | No isolated top-20 operator claim; sampling/state controls have separate evidence. | **3/5 Moderate for sampled decision/state checks.** Historical 24 September sampling evidence: 600,000 trials; maximum observed probability error 0.118 percentage points; deliberately shared proposal noise detected by the negative control. **Uncertainty:** this does not independently qualify every bookkeeping kernel or asynchronous commit/rollback path. | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Integrated qualified sampling, cache/state bookkeeping and graph-capture admission; no isolated correctness claim is made for this aggregate row. | Runs sampling and cache/state update kernels outside the named model stages. |
| **26. Estimated runtime overhead** | 2.279 / 2.465 / 2.269 (estimate) | [Matched control minus GPU activity union](benchmarks/results/matched-stage-residual-20261009.json) | **N/A: timing estimate.** No model arithmetic to score. **Uncertainty:** performance residual is not evidence of scheduler or session-state correctness. | [`99d5fcf`](https://github.com/Terrydaktal/vllm-coherence/commit/99d5fcfd1bcd0fe6f70445d154e63b7092b02f24): record matched unprofiled controls and overlap-corrected residuals | Indirect observer effects are not proved zero. |
| **Total reconstructed round (stages 1–26)** | **37.201 / 41.126 / 50.282** | GPU activity union plus the estimated remainder | **N/A: no aggregate correctness score.** Operator scores cannot be averaged into a model guarantee. **Uncertainty:** arbitrary-input prefill/M1 equality, approximate-head completeness and exhaustive restore/concurrency qualification remain open; finite complete-model equality and lifecycle checks are reported separately. | — | Overlapping stages are counted once in the total. |

The table groups the backend into 26 stages. Each timing cell is ordered **0K / 60K / 200K**. Fused kernels are charged once to their containing stage; the ↳ rows are detail-only inclusion records and add no timing; rows without a separate profiler scope are labelled in the timing cell rather than displayed as 0.000. The total counts overlapping GPU activity once.

Each **set/order** pair means the same top-20 token set, followed by the same ranking. Current stage confirmations identify M1/M8 and eager/compiled M8 separately. Each position passes only if every layer instance passes on the same captured inputs. These diagnostic stage replays are checked against the compiled graph control; their times are not used in this table. Timing and correctness captures record their own exact source hashes. The provenance column identifies the last relevant code change.

Fusion still permits correctness instrumentation: a diagnostic kernel can expose intermediate values, and fused outputs can be compared with an unfused reference. The normal GPU profile measures the combined kernel. Internal probes or splitting the kernel can change its performance, so those measurements are not an additive breakdown of the production kernel's time.

### Cache-state equivalence and prefill/restore timings

The latest recorded result for each path is shown once, with its capture date. These synthetic cache/equality checks were not rerun in the October 9 speed refresh; their timings describe the captured builds. Current cold-prefill speeds are shown below.

| State / execution comparison | Workload and cache coverage | Latest correctness evidence | Latest measured speed / latency | Capture |
| --- | --- | --- | --- | --- |
| Cold prefill ↔ retained decode history | 1,651-token prefix + 1,000 forced token positions | Byte-exact hidden rows 1,000/1,000; full BF16-logit rows 1,000/1,000.<br>Top-1/10/20 sets and ordering all **1,000/1,000 (100%)** | Not separately benchmarked | [2026-09-27 alignment](benchmarks/results/prefill-alignment-20260927.json) |
| Cold prefill ↔ retained decode history | 60,000-token prefix + 1,000 forced token positions | Byte-exact hidden rows 1,000/1,000; full BF16-logit rows 1,000/1,000.<br>Top-1/10/20 sets and ordering all **1,000/1,000 (100%)** | Vector replay not separately timed; current cold-prefill speeds below | [2026-09-27 alignment](benchmarks/results/prefill-alignment-20260927.json) |
| Cold prefill ↔ retained decode history | 200,000-token prefix + 320 forced token positions | Byte-exact hidden rows 320/320; full BF16-logit rows 320/320.<br>Top-1/10/20 sets and ordering all **320/320 (100%)** | Vector replay not separately timed; current cold-prefill speeds below | [2026-09-27 alignment](benchmarks/results/prefill-alignment-20260927.json) |
| Response-end state copy ↔ source state | 48 GDN layers; 576 state/history checks; 144 physical-page copies | Exact: 1,450,967,040 bytes; observed accepted offsets 1, 2, 5, 6, 7 | Not a production timing capture | [2026-09-27 cache paths](benchmarks/results/response-end-cache-20260927.json) |
| Resident GPU reuse across block/response boundaries | 1,647, 1,653, 1,657, 1,663, 1,665, 1,675, 1,709, 1,719, 3,307, 3,313, 3,325, 3,369 input tokens; 5, 6 uncached | 12/12 exact continuations, 64 generated tokens each | First data **0.075–1.069 s** | [2026-09-27 cache paths](benchmarks/results/response-end-cache-20260927.json) |
| Resident GPU reuse, long context | 60,025 input tokens; 5 uncached | 1/1 exact continuations, 64 generated tokens each | First data **0.221 s** | [2026-09-27 cache paths](benchmarks/results/response-end-cache-20260927.json) |
| Resident GPU reuse, long context | 200,025 input tokens; 5 uncached | 1/1 exact continuations, 64 generated tokens each | First data **1.026 s** | [2026-09-27 cache paths](benchmarks/results/response-end-cache-20260927.json) |
| GPU → RAM → GPU chat handover | 3,313, 3,325 input tokens; 5 uncached | 2/2 exact continuations, 64 generated tokens each | First data **0.272–0.320 s** | [2026-09-27 cache paths](benchmarks/results/response-end-cache-20260927.json) |
| Reuse after GPU-bank eviction | 1,653, 3,313, 3,325 input tokens; 5 uncached | 3/3 exact continuations, 64 generated tokens each | First data **0.343–0.935 s** | [2026-09-27 cache paths](benchmarks/results/response-end-cache-20260927.json) |
| Disk restore after backend restart, short context | 1,675, 3,325 input tokens; 5 uncached | 2/2 exact continuations, 64 generated tokens each | First data **0.940–14.286 s** | [2026-09-27 cache paths](benchmarks/results/response-end-cache-20260927.json) |
| Disk restore after backend restart, long context | 60,094 input tokens; 5 uncached | 1/1 exact continuations, 64 generated tokens each | First data **4.593 s** | [2026-09-27 cache paths](benchmarks/results/response-end-cache-20260927.json) |
| Disk restore after backend restart, long context | 200,094 input tokens; 5 uncached | 1/1 exact continuations, 64 generated tokens each | First data **23.468 s** | [2026-09-27 cache paths](benchmarks/results/response-end-cache-20260927.json) |
| Damaged snapshot → reject and cold rebuild | 3,325 input tokens; 3,325 uncached | 1/1 exact continuations, 64 generated tokens each; corrupted endpoint rejected; zero cached tokens used | First data **1.912 s** | [2026-09-27 cache paths](benchmarks/results/response-end-cache-20260927.json) |
| Reuse after explicit stop boundary | 3,313, 3,325, 3,369 input tokens; 5 uncached | 3/3 exact continuations, 64 generated tokens each | First data **0.147–0.170 s** | [2026-09-27 cache paths](benchmarks/results/response-end-cache-20260927.json) |
| Cached tool continuation ↔ cold full prompt (full BF16) | 1,701, 1,711, 1,745, 1,755, 3,361, 3,405 input tokens; 41 uncached | 6/6 exact continuations, 64 generated tokens each; greedy; same appended 41-token tool suffix | First data **0.189–1.240 s** | [2026-09-27 alignment](benchmarks/results/prefill-alignment-20260927.json) |
| Cached tool continuation ↔ cold full prompt (Global-512) | 1,701, 1,711, 1,745, 1,755, 3,361, 3,405 input tokens; 41 uncached | 6/6 exact continuations, 64 generated tokens each; greedy; same appended 41-token tool suffix | First data **0.183–1.057 s** | [2026-09-27 alignment](benchmarks/results/prefill-alignment-20260927.json) |
| Repeated sampled tool continuation | 1,711, 3,361 input tokens; 41, 42 uncached | 2/2 exact continuations, 64 generated tokens each; T=1, top-p=0.95, top-k=40, seed=0; same prefill/decode boundaries, not a cold comparison | First data **0.469–0.501 s** | [2026-09-27 cache paths](benchmarks/results/response-end-cache-20260927.json) |
| Cancelled continuation ↔ uninterrupted control | 1,675, 1,719 input tokens; 5, 6 uncached | 2/2 exact continuations, 64 generated tokens each | First data **0.049–0.065 s** | [2026-09-27 pressure checks](benchmarks/results/response-end-pressure-20260927.json) |
| Cancel during decode → replay | Interrupted after 1, 65, 258 received tokens | 3/3 replays equal uninterrupted control | Cancellation to scheduler release: **25.401–50.736 ms** | [2026-09-27 lifecycle](benchmarks/results/prefill-alignment-deployment-20260927.json) |
| Cancel during cold prefill → replay | 60K input; 3,296 processed at cancellation | Replay equal | Cancellation to scheduler release: **809.836 ms** | [2026-09-27 lifecycle](benchmarks/results/prefill-alignment-deployment-20260927.json) |
| Cancel queued chat → replay | Two chats; cancelled request had not generated | Replay equal | Cancellation to scheduler release: **25.453 ms** | [2026-09-27 lifecycle](benchmarks/results/prefill-alignment-deployment-20260927.json) |
| Sampled cancellation with another chat waiting → replay | Two chats; T=1, top-p=0.95, top-k=40, seed=113 | Replay equal | Cancellation to scheduler release: **25.466 ms** | [2026-09-27 lifecycle](benchmarks/results/prefill-alignment-deployment-20260927.json) |
| Concurrent chats and priority handovers ↔ uninterrupted controls | Equal priority, immediate priority-2 takeover/cancellation, priority-1 tool-boundary hold | 5/5 lifecycle cases pass their output/ownership checks | Isolated handover latency not measured | [2026-09-27 lifecycle](benchmarks/results/prefill-alignment-deployment-20260927.json) |
| Near-limit endpoint ownership and cache reclamation | 239,720 input; 233,182 cached; 1,024 generated | 0 allocator preemptions; forward progress (not numerical-equivalence evidence) | First data **12.104 s** | [2026-09-27 pressure checks](benchmarks/results/response-end-pressure-20260927.json) |
| Near-limit endpoint ownership and cache reclamation | 252,000 input; 240,744 cached; 512 generated | 0 allocator preemptions; forward progress (not numerical-equivalence evidence) | First data **20.378 s** | [2026-09-27 pressure checks](benchmarks/results/response-end-pressure-20260927.json) |
| Compaction-generation snapshot replacement | 3 publication/retirement cycles | 3/3 keep the previous disk head until replacement verification, then retain one generation; lifecycle safety, not old/new summary equivalence | Not separately benchmarked | [2026-09-25 snapshots](benchmarks/results/snapshot-lifecycle-20260925.json) |

**Timing boundaries:** first data is request-to-first-output wall time, including admission, handover, restore and any required prefill; it is not pure disk or RAM transfer time. Restart measurements start after the backend is ready. Cancellation times measure scheduler release, not completion of the replay. No qualification-suite runtime is substituted for an operation benchmark.

**Equality boundaries:** set/order lists identical token sets followed by identical ordering. The full-vector prefill checks cover 2,320 distinct forced-token positions; the packaged 1,000-position repeat does not add new positions. Matching generated tokens alone does not establish equality of every latent state or logit. Compaction creates a new history, so its storage test checks safe replacement rather than equivalence to the uncompressed conversation. These finite checks do not prove arbitrary-input correctness, exhaustive scheduling interleavings or completeness of the approximate Global-512 head.

### Cold-prefill speeds (2026-10-09 measurements)

The serving build uses packed attention, register-resident projection partials, prepared GDN inputs and compiled kernels reusable across prompt lengths. Its 4,096-row admission limit permits 3,296-row scheduler chunks while preserving the corrected arithmetic.

| Cold context | Mean prefill | Prefill tokens/s | Mean first data | Samples |
| --- | ---: | ---: | ---: | ---: |
| 60,000 tokens | **33.25 s** | **1,805** | 33.50 s | 2 |
| 200,000 tokens | **200.11 s** | **999** | 200.67 s | 2 |

Current values are means of 2 / 2 unprofiled cold requests at 60K / 200K; prefill ranges were 33.05–33.45 s / 199.94–200.28 s. Each request reused zero prompt tokens and generated one token with greedy sampling; the log-probability request selects the full BF16 head for that output. Prefill tokens/s is prompt length divided by mean backend-prefill time; first data also includes request setup and the first generated token. [Current measurement](benchmarks/results/prefill-speed-timings-20261009.json).

[Historical vector qualification (2026-09-28)](benchmarks/results/prefill-speed-qualification-20260928.json) records complete hidden/logit agreement at 1,000 sampled positions at 60K and 320 at 200K. Those vector checks were not rerun for the newly timed build. [Implementation and numerical scope](docs/PREFILL_SPEED.md).

The expandable layer and kernel tables use the same 1,134 retained 60K cycles as the main table (2026-10-09). GPU activity crossing a worker boundary is clipped to that boundary; activity-record counts include these fragments. CPU profiling work is excluded. Cycle counts are not output-token counts.

<details>
<summary>Current compiled 60K decoder-layer detail: projection and remaining-work timings</summary>

Finish each row's layer, including its MLP, before moving to the next row. Each layer has four projections. Gate and up are one joint GEMM; there is no separately measured gate/up split. The remaining-work column combines normalization, mixing and other operations between those projections.

| Layer | Type | All layer work ms | Input projection ms | Output projection ms | Gate/up projection ms | Down projection ms | Other work ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | GDN | 0.4232 | 0.0766 | 0.0365 | 0.1718 | 0.0798 | 0.0585 |
| 1 | GDN | 0.4427 | 0.0818 | 0.0385 | 0.1809 | 0.0858 | 0.0556 |
| 2 | GDN | 0.4490 | 0.0839 | 0.0388 | 0.1818 | 0.0859 | 0.0586 |
| 3 | Attention | 0.7006 | 0.0730 | 0.0334 | 0.1646 | 0.0795 | 0.3501 |
| 4 | GDN | 0.4275 | 0.0827 | 0.0367 | 0.1714 | 0.0801 | 0.0567 |
| 5 | GDN | 0.4412 | 0.0817 | 0.0387 | 0.1804 | 0.0845 | 0.0558 |
| 6 | GDN | 0.4468 | 0.0838 | 0.0390 | 0.1788 | 0.0857 | 0.0596 |
| 7 | Attention | 0.6976 | 0.0714 | 0.0334 | 0.1642 | 0.0798 | 0.3487 |
| 8 | GDN | 0.4273 | 0.0825 | 0.0367 | 0.1714 | 0.0800 | 0.0566 |
| 9 | GDN | 0.4408 | 0.0814 | 0.0380 | 0.1806 | 0.0852 | 0.0556 |
| 10 | GDN | 0.4475 | 0.0840 | 0.0385 | 0.1791 | 0.0871 | 0.0588 |
| 11 | Attention | 0.6953 | 0.0725 | 0.0335 | 0.1641 | 0.0796 | 0.3456 |
| 12 | GDN | 0.4254 | 0.0812 | 0.0368 | 0.1716 | 0.0796 | 0.0563 |
| 13 | GDN | 0.4409 | 0.0815 | 0.0380 | 0.1803 | 0.0858 | 0.0553 |
| 14 | GDN | 0.4482 | 0.0849 | 0.0378 | 0.1804 | 0.0879 | 0.0572 |
| 15 | Attention | 0.6965 | 0.0726 | 0.0335 | 0.1640 | 0.0799 | 0.3465 |
| 16 | GDN | 0.4273 | 0.0818 | 0.0367 | 0.1715 | 0.0801 | 0.0572 |
| 17 | GDN | 0.4411 | 0.0814 | 0.0389 | 0.1799 | 0.0853 | 0.0557 |
| 18 | GDN | 0.4471 | 0.0845 | 0.0379 | 0.1802 | 0.0873 | 0.0572 |
| 19 | Attention | 0.6952 | 0.0723 | 0.0332 | 0.1640 | 0.0794 | 0.3461 |
| 20 | GDN | 0.4283 | 0.0818 | 0.0368 | 0.1716 | 0.0799 | 0.0581 |
| 21 | GDN | 0.4426 | 0.0814 | 0.0381 | 0.1816 | 0.0865 | 0.0551 |
| 22 | GDN | 0.4471 | 0.0844 | 0.0379 | 0.1799 | 0.0868 | 0.0581 |
| 23 | Attention | 0.6973 | 0.0716 | 0.0335 | 0.1647 | 0.0800 | 0.3475 |
| 24 | GDN | 0.4268 | 0.0820 | 0.0367 | 0.1719 | 0.0794 | 0.0567 |
| 25 | GDN | 0.4421 | 0.0817 | 0.0380 | 0.1807 | 0.0858 | 0.0559 |
| 26 | GDN | 0.4476 | 0.0837 | 0.0384 | 0.1808 | 0.0875 | 0.0573 |
| 27 | Attention | 0.6975 | 0.0722 | 0.0334 | 0.1641 | 0.0800 | 0.3478 |
| 28 | GDN | 0.4278 | 0.0823 | 0.0368 | 0.1722 | 0.0802 | 0.0563 |
| 29 | GDN | 0.4431 | 0.0823 | 0.0378 | 0.1808 | 0.0865 | 0.0558 |
| 30 | GDN | 0.4451 | 0.0840 | 0.0384 | 0.1789 | 0.0858 | 0.0580 |
| 31 | Attention | 0.6974 | 0.0721 | 0.0332 | 0.1647 | 0.0795 | 0.3479 |
| 32 | GDN | 0.4270 | 0.0818 | 0.0366 | 0.1715 | 0.0804 | 0.0567 |
| 33 | GDN | 0.4417 | 0.0817 | 0.0380 | 0.1801 | 0.0854 | 0.0564 |
| 34 | GDN | 0.4505 | 0.0845 | 0.0386 | 0.1807 | 0.0869 | 0.0598 |
| 35 | Attention | 0.6981 | 0.0731 | 0.0336 | 0.1644 | 0.0798 | 0.3472 |
| 36 | GDN | 0.4280 | 0.0821 | 0.0367 | 0.1721 | 0.0805 | 0.0566 |
| 37 | GDN | 0.4438 | 0.0828 | 0.0381 | 0.1816 | 0.0859 | 0.0554 |
| 38 | GDN | 0.4443 | 0.0840 | 0.0388 | 0.1780 | 0.0852 | 0.0582 |
| 39 | Attention | 0.6983 | 0.0724 | 0.0334 | 0.1642 | 0.0795 | 0.3488 |
| 40 | GDN | 0.4290 | 0.0825 | 0.0367 | 0.1720 | 0.0804 | 0.0575 |
| 41 | GDN | 0.4433 | 0.0815 | 0.0384 | 0.1811 | 0.0860 | 0.0563 |
| 42 | GDN | 0.4458 | 0.0849 | 0.0385 | 0.1792 | 0.0856 | 0.0576 |
| 43 | Attention | 0.6982 | 0.0721 | 0.0334 | 0.1651 | 0.0798 | 0.3479 |
| 44 | GDN | 0.4290 | 0.0826 | 0.0367 | 0.1725 | 0.0803 | 0.0569 |
| 45 | GDN | 0.4420 | 0.0812 | 0.0375 | 0.1804 | 0.0862 | 0.0567 |
| 46 | GDN | 0.4480 | 0.0844 | 0.0379 | 0.1807 | 0.0872 | 0.0578 |
| 47 | Attention | 0.6975 | 0.0725 | 0.0334 | 0.1645 | 0.0798 | 0.3473 |
| 48 | GDN | 0.4309 | 0.0826 | 0.0367 | 0.1722 | 0.0807 | 0.0587 |
| 49 | GDN | 0.4423 | 0.0822 | 0.0384 | 0.1811 | 0.0839 | 0.0568 |
| 50 | GDN | 0.4494 | 0.0851 | 0.0378 | 0.1809 | 0.0873 | 0.0583 |
| 51 | Attention | 0.6982 | 0.0726 | 0.0335 | 0.1641 | 0.0797 | 0.3483 |
| 52 | GDN | 0.4265 | 0.0814 | 0.0369 | 0.1716 | 0.0797 | 0.0569 |
| 53 | GDN | 0.4416 | 0.0816 | 0.0379 | 0.1804 | 0.0860 | 0.0556 |
| 54 | GDN | 0.4475 | 0.0842 | 0.0379 | 0.1804 | 0.0870 | 0.0579 |
| 55 | Attention | 0.6981 | 0.0727 | 0.0336 | 0.1641 | 0.0796 | 0.3482 |
| 56 | GDN | 0.4266 | 0.0814 | 0.0366 | 0.1716 | 0.0798 | 0.0571 |
| 57 | GDN | 0.4423 | 0.0815 | 0.0384 | 0.1810 | 0.0847 | 0.0567 |
| 58 | GDN | 0.4475 | 0.0845 | 0.0381 | 0.1803 | 0.0863 | 0.0582 |
| 59 | Attention | 0.6989 | 0.0725 | 0.0332 | 0.1643 | 0.0797 | 0.3492 |
| 60 | GDN | 0.4286 | 0.0817 | 0.0366 | 0.1721 | 0.0806 | 0.0576 |
| 61 | GDN | 0.4415 | 0.0815 | 0.0377 | 0.1808 | 0.0854 | 0.0561 |
| 62 | GDN | 0.4514 | 0.0846 | 0.0377 | 0.1803 | 0.0877 | 0.0610 |
| 63 | Attention | 0.6992 | 0.0734 | 0.0335 | 0.1640 | 0.0799 | 0.3483 |

</details>

<details>
<summary>Current 60K GPU activity, grouped by stage</summary>

| Stage / compiled kernel | Activity records in 1,134 retained cycles | Current GPU ms per retained 60K cycle |
| --- | ---: | ---: |
| Drafter / `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x32_MI16x16x1_SN_LDSB1_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_CD1_1_CLR0_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB1_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA1201_ICIW0_IU1_K1_LDSTI0_LBSPPA128_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB0_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB2_ONLL1_PAP0_PGL0_PGR1_PLR1_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1.kd` | 1134 | 0.006307 |
| Drafter / `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x32x256_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB1_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA1201_ICIW0_IU1_K1_LDSTI0_LBSPPA512_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB0_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB16_ONLL0_PAP0_PGL0_PGR1_PLR1_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_4_1.kd` | 1135 | 0.176919 |
| Drafter / `__amd_rocclr_copyBuffer.kd` | 1134 | 0.002024 |
| Drafter / `__amd_rocclr_fillBufferAligned.kd` | 3402 | 0.003626 |
| Drafter / `_cache_draft_logits_kernel.kd` | 1134 | 0.001776 |
| Drafter / `_draft_head_int2.kd` | 1134 | 0.807286 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_17408_EVEN_K_1_GRID_MN_40_cache_modifier_NONE.kd` | 5674 | 0.733660 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_25600_EVEN_K_1_GRID_MN_40_cache_modifier_NONE.kd` | 1136 | 0.235401 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_40_cache_modifier_NONE.kd` | 5741 | 0.196852 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_5120_EVEN_K_1_GRID_MN_272_cache_modifier_NONE.kd` | 5765 | 1.443186 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_5120_EVEN_K_1_GRID_MN_48_cache_modifier_NONE.kd` | 5674 | 0.269281 |
| Drafter / `_prepare_dflash_inputs_kernel.kd` | 1134 | 0.010122 |
| Drafter / `_rerank_exact.kd` | 1134 | 0.007939 |
| Drafter / `_selector_walk_kernel.kd` | 1134 | 0.007909 |
| Drafter / `kernel_unified_attention.kd` | 6476 | 0.489514 |
| Drafter / `reshape_and_cache_kernel_flash.kd` | 11341 | 0.023306 |
| Drafter / `triton_per_fused_4.kd` | 1136 | 0.001498 |
| Drafter / `triton_per_fused_8.kd` | 4536 | 0.005111 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_preshuffle_gemm_squeeze_view_0.kd` | 5672 | 0.007014 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_preshuffle_gemm_squeeze_view_2.kd` | 1134 | 0.001435 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_preshuffle_gemm_squeeze_view_4.kd` | 10206 | 0.010854 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_view_0.kd` | 1134 | 0.013058 |
| Drafter / `triton_poi_fused_0.kd` | 1134 | 0.002019 |
| Drafter / `triton_poi_fused_5.kd` | 1135 | 0.001671 |
| Drafter / `triton_poi_fused_9.kd` | 4536 | 0.006114 |
| Drafter / `triton_poi_fused__to_copy_clamp_div_preshuffle_gemm_squeeze_view_1.kd` | 5671 | 0.007823 |
| Drafter / `triton_poi_fused__to_copy_clamp_div_preshuffle_gemm_squeeze_view_3.kd` | 1134 | 0.001516 |
| Drafter / `triton_poi_fused__to_copy_clamp_div_preshuffle_gemm_squeeze_view_5.kd` | 10207 | 0.011283 |
| Drafter / `triton_poi_fused_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_3.kd` | 10209 | 0.025090 |
| Drafter / `triton_poi_fused_add_arange_bitwise_and_constant_pad_nd_ge_mul_rms_norm_select_slice_unsqueeze_view_1.kd` | 1134 | 0.022543 |
| Drafter / `triton_poi_fused_add_permute_unsqueeze_view_2.kd` | 1134 | 0.001268 |
| Drafter / `triton_poi_fused_cat_expand_index_mul_slice_unsqueeze_view_1.kd` | 1134 | 0.001948 |
| Drafter / `triton_red_fused__to_copy_abs_clamp_div_max_mul_preshuffle_gemm_silu_slice_squeeze_view_6.kd` | 5670 | 0.014839 |
| Drafter / `triton_red_fused__to_copy_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_w4_gemm_2.kd` | 5672 | 0.028126 |
| Drafter / `triton_red_fused__to_copy_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_w4_gemm_7.kd` | 4536 | 0.023340 |
| Drafter / `triton_red_fused__to_copy_embedding_mul_rms_norm_w4_gemm_0.kd` | 1134 | 0.003854 |
| Drafter / `triton_red_fused_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_7.kd` | 1134 | 0.005720 |
| Drafter / `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) [clone .kd]` | 2268 | 0.008743 |
| Drafter / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 1134 | 0.002950 |
| Drafter / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 1134 | 0.002589 |
| Drafter / `void at::native::bitonicSortKVInPlace<2, -1, 16, 16, c10::BFloat16, long, at::native::GTOp<c10::BFloat16, true>, unsigned int>(at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, at::native::GTOp<c10::BFloat16, true>) [clone .kd]` | 1134 | 0.002575 |
| Drafter / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 1134 | 0.004419 |
| Drafter / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 1134 | 0.002654 |
| Drafter / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 1134 | 0.003564 |
| Drafter / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 1134 | 0.001956 |
| Drafter / `void at::native::mbtopk::computeBlockDigitCounts<c10::BFloat16, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 2268 | 0.037060 |
| Drafter / `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 4536 | 0.053220 |
| Drafter / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, c10::BFloat16>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 2268 | 0.017556 |
| Drafter / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, float>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, float*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 4536 | 0.026227 |
| Drafter / `void at::native::mbtopk::fill<unsigned int, unsigned int>(unsigned int*, unsigned int, unsigned int) [clone .kd]` | 2268 | 0.001992 |
| Drafter / `void at::native::mbtopk::gatherTopK<c10::BFloat16, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 1134 | 0.019973 |
| Drafter / `void at::native::mbtopk::gatherTopK<float, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, float*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 1134 | 0.009934 |
| Drafter / `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4>) [clone .kd]` | 1134 | 0.002532 |
| Drafter / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul>) [clone .kd]` | 1134 | 0.001161 |
| Drafter / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) [clone .kd]` | 1134 | 0.001912 |
| Drafter / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>) [clone .kd]` | 2268 | 0.002500 |
| Drafter / `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>) [clone .kd]` | 2268 | 0.004130 |
| Drafter / `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool) [clone .kd]` | 1134 | 0.001561 |
| Drafter / `void at::native::warpMergeSortKVInPlace<2, -1, 128, 16, float, long, at::native::GTOp<float, true>, unsigned int, 32>(at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, at::native::GTOp<float, true>, float) [clone .kd]` | 1134 | 0.004289 |
| Drafter / `void r4d_gemm_w4a16_nt_m64_kernel<1, 1, false>(unsigned short const*, unsigned int const*, unsigned int const*, __hip_bfloat16*, int, int, int, int, int) [clone .kd]` | 1134 | 0.003989 |
| Drafter / `void r4d_gemm_w4a16_nt_m64_kernel<1, 1, true>(unsigned short const*, unsigned int const*, unsigned int const*, __hip_bfloat16*, int, int, int, int, int) [clone .kd]` | 11345 | 0.075343 |
| Drafter / `void vllm::rms_norm_kernel<c10::BFloat16, 8, 2, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, long, long, long, c10::BFloat16 const*, long, float, int, int) [clone .kd]` | 1134 | 0.002836 |
| Drafter / `void vllm::rms_norm_kernel<c10::BFloat16, 8, 4, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, long, long, long, c10::BFloat16 const*, long, float, int, int) [clone .kd]` | 1134 | 0.002710 |
| Drafter / `void vllm::rotary_embedding_kernel<c10::BFloat16, c10::BFloat16, true>(long const*, c10::BFloat16*, c10::BFloat16*, c10::BFloat16 const*, int, long, long, long, int, int, int, long, bool) [clone .kd]` | 1134 | 0.002542 |
| Embedding + first input normalization / `triton_poi_fused__to_copy_embedding_0.kd` | 1134 | 0.002117 |
| Embedding + first input normalization / `void norm_quant<false, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned char*, float*, unsigned short*, long, long, float) [clone .kd]` | 1134 | 0.006787 |
| GDN layout/copies and buffer initialization / `triton_poi_fused_add_1.kd` | 3402 | 0.003680 |
| GDN layout/copies and buffer initialization / `triton_poi_fused_add_2.kd` | 2268 | 0.002438 |
| GDN layout/copies and buffer initialization / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 54432 | 0.090043 |
| GDN layout/copies and buffer initialization / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) [clone .kd]` | 54432 | 0.059548 |
| GDN layout/copies and buffer initialization / `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>) [clone .kd]` | 54432 | 0.043939 |
| GDN convolution / `_causal_conv1d_update_kernel.kd` | 54432 | 0.181833 |
| GDN recurrence and gates / `stock_gdn_scan_kernel.kd` | 54432 | 1.177023 |
| GDN output gated normalization / `gdn_norm_quant_kernel.kd` | 54432 | 0.146886 |
| Post-attention/GDN residual/normalization / `void norm_quant<true, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned char*, float*, unsigned short*, long, long, float) [clone .kd]` | 72576 | 0.510157 |
| MLP SiLU and gating / `triton_poi_fused_dynamic_per_token_scaled_fp8_quant_mul_silu_slice_0.kd` | 54432 | 0.090734 |
| MLP SiLU and gating / `triton_poi_fused_dynamic_per_token_scaled_fp8_quant_mul_silu_slice_1.kd` | 18144 | 0.043746 |
| MLP down input FP8 quantization / `void vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<c10::BFloat16, c10::Float8_e4m3fn>(c10::Float8_e4m3fn*, float*, c10::BFloat16 const*, float const*, int, long, long) [clone .kd]` | 72576 | 0.254980 |
| GDN input projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 1, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 54432 | 3.963580 |
| GDN output projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 4, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 54432 | 1.809744 |
| MLP gate/up projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 1, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 72576 | 11.150031 |
| MLP down projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 4, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 72576 | 5.313300 |
| Layer input residual/normalization / `void norm_quant<true, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned char*, float*, unsigned short*, long, long, float) [clone .kd]` | 71442 | 0.494705 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_1.kd` | 18144 | 0.021805 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_3.kd` | 18144 | 0.025892 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_4.kd` | 18144 | 0.024796 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_arange_bitwise_and_eq_index_lt_remainder_select_split_where_2.kd` | 18144 | 0.027742 |
| Attention Q/K normalization, RoPE and layout / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 18144 | 0.028445 |
| Attention Q/K normalization, RoPE and layout / `void stock_m1_gemma_norm<false, 256, 32>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, long, long, float, float*) [clone .kd]` | 18144 | 0.046337 |
| Attention Q/K normalization, RoPE and layout / `void stock_m1_gemma_norm<false, 256, 64>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, long, long, float, float*) [clone .kd]` | 18144 | 0.036602 |
| Attention KV write / `reshape_and_cache_kernel_flash.kd` | 18144 | 0.034558 |
| Attention decode / `void qwen_stock_m1_shared_decode<4, 16, 256, 6, 16, 0, 3430931>(R4DArgs, int) [clone .kd]` | 18144 | 4.753961 |
| Attention split-KV merge / `void qwen_stock_m1_shared_merge<256, 4, 0>(R4DArgs, int, int) [clone .kd]` | 18144 | 0.132544 |
| Attention output gating / `triton_poi_fused_dynamic_per_token_scaled_fp8_quant_mul_sigmoid_view_0.kd` | 18144 | 0.025042 |
| Attention output activation FP8 quantization / `void vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<c10::BFloat16, c10::Float8_e4m3fn>(c10::Float8_e4m3fn*, float*, c10::BFloat16 const*, float const*, int, long, long) [clone .kd]` | 18144 | 0.041968 |
| Attention input projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 1, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 18144 | 1.158912 |
| Attention output projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 4, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 18144 | 0.534779 |
| Final normalization/layout / `__amd_rocclr_copyBuffer.kd` | 6804 | 0.007099 |
| Final normalization/layout / `void stock_m1_gemma_norm<true, 5120, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, long, long, float, float*) [clone .kd]` | 1134 | 0.005007 |
| Other GPU bookkeeping / `__amd_rocclr_copyBuffer.kd` | 44235 | 0.077152 |
| Other GPU bookkeeping / `__amd_rocclr_fillBufferAligned.kd` | 1134 | 0.002506 |
| Other GPU bookkeeping / `_apply_write_kernel.kd` | 6 | 0.000025 |
| Other GPU bookkeeping / `_combine_sampled_and_draft_tokens_kernel.kd` | 1134 | 0.003055 |
| Other GPU bookkeeping / `_compute_local_logits_stats_kernel.kd` | 1134 | 0.028107 |
| Other GPU bookkeeping / `_compute_slot_mappings_kernel.kd` | 1134 | 0.002892 |
| Other GPU bookkeeping / `_expand_idx_mapping_kernel.kd` | 1134 | 0.001965 |
| Other GPU bookkeeping / `_gather_block_tables_kernel.kd` | 1134 | 0.004860 |
| Other GPU bookkeeping / `_get_num_sampled_and_rejected_kernel.kd` | 1134 | 0.002787 |
| Other GPU bookkeeping / `_insert_resampled_kernel.kd` | 1134 | 0.003641 |
| Other GPU bookkeeping / `_post_update_kernel.kd` | 1134 | 0.005027 |
| Other GPU bookkeeping / `_prepare_pos_seq_lens_kernel.kd` | 1134 | 0.002094 |
| Other GPU bookkeeping / `_prepare_rope_positions_kernel.kd` | 1134 | 0.002924 |
| Other GPU bookkeeping / `_rejection_kernel.kd` | 1134 | 0.007453 |
| Other GPU bookkeeping / `_resample_kernel.kd` | 1134 | 0.014309 |
| Other GPU bookkeeping / `_scatter_num_accepted_kernel.kd` | 1134 | 0.002094 |
| Other GPU bookkeeping / `_zero_kv_blocks_kernel.kd` | 3 | 0.000178 |
| Other GPU bookkeeping / `postprocess_mamba_fused_kernel.kd` | 1134 | 0.004193 |
| Other GPU bookkeeping / `precopy_mamba_align_fused_kernel.kd` | 1134 | 0.004002 |
| Other GPU bookkeeping / `preprocess_mamba_align_fused_kernel.kd` | 1134 | 0.002514 |
| Other GPU bookkeeping / `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}>(int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}, function_traits<at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}>::result_type*) [clone .kd]` | 6804 | 0.008619 |
| Other GPU bookkeeping / `void (anonymous namespace)::softmax_warp_forward<float, float, float, 6, false, false, 32>(float*, float const*, int, int, int, bool const*, int, bool) [clone .kd]` | 1134 | 0.002242 |
| Other GPU bookkeeping / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 7938 | 0.013385 |
| Other GPU bookkeeping / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 1134 | 0.002567 |
| Other GPU bookkeeping / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 7938 | 0.014076 |
| Other GPU bookkeeping / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<int> const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<int> const&)::{lambda(int, bool)#1}) [clone .kd]` | 6804 | 0.014996 |
| Other GPU bookkeeping / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<float> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<float> const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<float> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<float> const&)::{lambda(int, bool)#1}) [clone .kd]` | 3402 | 0.018639 |
| Other GPU bookkeeping / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 5670 | 0.016479 |
| Other GPU bookkeeping / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 2268 | 0.006073 |
| Other GPU bookkeeping / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 1134 | 0.003157 |
| Other GPU bookkeeping / `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 4536 | 0.058600 |
| Other GPU bookkeeping / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, float>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, float*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 4536 | 0.036529 |
| Other GPU bookkeeping / `void at::native::mbtopk::fill<unsigned int, unsigned int>(unsigned int*, unsigned int, unsigned int) [clone .kd]` | 1134 | 0.001586 |
| Other GPU bookkeeping / `void at::native::mbtopk::gatherTopK<float, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, float*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 1134 | 0.025557 |
| Other GPU bookkeeping / `void at::native::tensor_kernel_scan_innermost_dim<float, std::plus<float> >(float*, float const*, unsigned int, unsigned int, unsigned int, float, std::plus<float>) [clone .kd]` | 1134 | 0.002546 |
| Other GPU bookkeeping / `void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>, 4, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast) [clone .kd]` | 1134 | 0.001746 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<16, at::native::BinaryFunctor<bool, bool, bool, at::native::BitwiseOrFunctor<bool> >, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<bool, bool, bool, at::native::BitwiseOrFunctor<bool> >, std::array<char*, 3ul>) [clone .kd]` | 1134 | 0.002776 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<16, at::native::FillFunctor<bool>, std::array<char*, 1ul> >(int, at::native::FillFunctor<bool>, std::array<char*, 1ul>) [clone .kd]` | 1134 | 0.001781 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<16, at::native::bitwise_not_kernel_cuda(at::TensorIteratorBase&)::{lambda(bool)#1}, std::array<char*, 2ul> >(int, at::native::bitwise_not_kernel_cuda(at::TensorIteratorBase&)::{lambda(bool)#1}, std::array<char*, 2ul>) [clone .kd]` | 1134 | 0.002315 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}, std::array<char*, 2ul>) [clone .kd]` | 6804 | 0.009963 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}, std::array<char*, 2ul>) [clone .kd]` | 1134 | 0.001801 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}, std::array<char*, 3ul> >(int, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}, std::array<char*, 3ul>) [clone .kd]` | 1134 | 0.009996 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool, float, float)#1}, std::array<char*, 4ul> >(int, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool, float, float)#1}, std::array<char*, 4ul>) [clone .kd]` | 2268 | 0.004700 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 2ul> >(int, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 2ul>) [clone .kd]` | 6804 | 0.010946 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, 2ul>) [clone .kd]` | 7938 | 0.012112 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul>) [clone .kd]` | 1134 | 0.001869 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>) [clone .kd]` | 1134 | 0.001935 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>) [clone .kd]` | 2268 | 0.003262 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int, at::native::FillFunctor<int>, std::array<char*, 1ul>) [clone .kd]` | 1134 | 0.001696 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>) [clone .kd]` | 1134 | 0.009689 |
| Other GPU bookkeeping / `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool) [clone .kd]` | 1134 | 0.002282 |
| Other GPU bookkeeping / `void at::native::warpMergeSortKVInPlace<2, -1, 128, 16, float, long, at::native::GTOp<float, true>, unsigned int, 32>(at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, at::native::GTOp<float, true>, float) [clone .kd]` | 1134 | 0.004697 |
| Target head (global512) / `__amd_rocclr_fillBufferAligned.kd` | 1134 | 0.002579 |
| Target head (global512) / `_draft_head_int2.kd` | 1134 | 0.754538 |
| Target head (global512) / `_rerank_exact.kd` | 1134 | 0.054066 |
| Target head (global512) / `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) [clone .kd]` | 1134 | 0.004634 |
| Target head (global512) / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 1134 | 0.003069 |
| Target head (global512) / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 1134 | 0.002920 |
| Target head (global512) / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 1134 | 0.005135 |
| Target head (global512) / `void at::native::mbtopk::computeBlockDigitCounts<c10::BFloat16, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 2268 | 0.089509 |
| Target head (global512) / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, c10::BFloat16>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 2268 | 0.040943 |
| Target head (global512) / `void at::native::mbtopk::fill<unsigned int, unsigned int>(unsigned int*, unsigned int, unsigned int) [clone .kd]` | 1134 | 0.001643 |
| Target head (global512) / `void at::native::mbtopk::gatherTopK<c10::BFloat16, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 1134 | 0.042111 |
| Target head (global512) / `void at::native::radixSortKVInPlace<2, -1, 128, 8, c10::BFloat16, long, unsigned int>(at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, bool) [clone .kd]` | 1134 | 0.005793 |
| Target head (global512) / `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4>) [clone .kd]` | 1134 | 0.003885 |
| Target head (global512) / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) [clone .kd]` | 1134 | 0.001721 |
| Target head (global512) / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>) [clone .kd]` | 1134 | 0.002237 |
| Target head (global512) / `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>) [clone .kd]` | 2268 | 0.005229 |

</details>

## Global-512 target-head

Global-512 is the serving default. This paired comparison used identical hidden inputs for every head method from two natural completions starting with the retained 60K Pi prefix: coding: 60,208 input and 7,748 output tokens (stop); reasoning: 68,103 input and 6,825 output tokens (stop). Sampling was temperature 1, top-p 0.95 and top-k 40.

| Target path | Median M8 head time | Same top-1 token | Complete reference top-20 retained | Complete reference top-40 retained |
|---|---:|---:|---:|---:|
| Global INT2 top-256 + BF16 rerank | 1.139 ms | 12,148/12,148 (100%) | 12,130/12,148 (99.8518%) | 11,893/12,148 (97.9009%) |
| Global INT2 top-512 + BF16 rerank (default) | 1.168 ms | 12,148/12,148 (100%) | 12,143/12,148 (99.9588%) | 12,118/12,148 (99.7530%) |
| Full BF16 reference | 4.057 ms | 12,148/12,148 (100%) | 12,148/12,148 (100%) | 12,148/12,148 (100%) |

The comparison covers **12,148 prediction rows**, including prefill and rejected speculative rows, not 12,148 generated tokens. Timing uses 48 eight-row hidden inputs with five randomized-order repetitions: 240 measurements per method. These isolated head timings include native dispatch gaps and exclude comparison/reporting; they do not measure whole-round time or tok/s.

Incomplete top-40 retention occurred in 255 rows with Global-256 and 30 with Global-512; the added median head time was 0.029 ms. Retention includes cutoff ties and does not establish score equality, ordering or identical sampling probabilities. Both shortlists remain approximate; the full BF16 head is the reference for this comparison, not an independently proved model. The drafter is unchanged.

Global-512 reranked values differed from the full-head reference in 39/6,219,776 retained scores (maximum absolute difference 0.0625); the diagnostic filtered probabilities differed in 13/12,148 rows. Candidate recall and retained-score fidelity are separate checks.

[Methodology and limits](docs/HEAD_CANDIDATE_DEPTH.md) · [Numeric results](benchmarks/results/head-candidate-depth-20261009.json) · [Earlier Global-256 study](docs/VERIFY_HEAD_GLOBAL_TOPK.md).

## Benchmarks

This benchmark uses the retained 60,000-input-token Pi prefix and the compiled Coherence backend with global512, the attention-page-boundary repair and the pinned-RAM huge-page promotion repair. Five requests are chained in one context: code, prose about code measurement, JSON, thinking/prose and checkpoint generation. Code and JSON disable thinking; both prose requests enable it. All requests stop naturally. Generation uses temperature 1, top-p 0.95, top-k 40 and seed 0; compaction uses temperature 0.3. The checkpoint request forces a snapshot-tail flush. Private fixture text was not decoded or inspected. Public results contain aggregates and hashes, not chat text or token arrays. The shared suite retains sealed continuation tokens privately for resumability. Runner: [benchmark_pi_coding_contexts.py --suite](experiments/radiance-public/benchmark_pi_coding_contexts.py).

| Stage | Thinking | Prompt tokens | Generated tokens | Classified output | First data | Mean round | Post-first | Peak 3s | Acceptance |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Coding task | off | 60,208 | 5,675 | 5,051 code; 623 prose; 0 separately observed reasoning | 0.80 s | 41.32 ms | 99.32 tok/s | 141.19 tok/s | 44.33% |
| Prose about code measurement | on | 66,024 | 4,199 | 4,204 prose; 0 separately observed reasoning | 1.31 s | 41.05 ms | 86.49 tok/s | 150.46 tok/s | 36.42% |
| JSON task | off | 70,342 | 2,650 | 2,649 JSON; valid JSON | 0.26 s | 41.40 ms | 88.59 tok/s | 105.02 tok/s | 38.08% |
| Thinking/prose task | on | 73,139 | 6,328 | 6,327 prose; 0 separately observed reasoning | 0.67 s | 41.66 ms | 62.67 tok/s | 94.46 tok/s | 23.01% |
| Compaction checkpoint | off | 79,653 | 4,730 | 4,730 checkpoint tokens; completion marker valid; required headings valid | 0.31 s | 42.13 ms | 70.02 tok/s | 96.25 tok/s | 27.84% |

Cached/total prompt tokens: Coding task: 59,328/60,208, Prose about code measurement: 65,883/66,024, JSON task: 70,223/70,342, Thinking/prose task: 72,992/73,139, Compaction checkpoint: 79,467/79,653.

The phase counters retokenize classified text, so their totals can differ from the backend's emitted-token count. `phase_token_counts_cover_output=false` for Coding task, Prose about code measurement, JSON task, Thinking/prose task, Compaction checkpoint. No separate reasoning channel was exposed for Prose about code measurement, Thinking/prose task; those streams are reported as prose, not relabelled as reasoning. The compaction row measures checkpoint generation with a requested cache flush, not a full Pi transcript commit or old-snapshot retirement. The checkpoint format passed its heading and completion checks. `peak_3s_tokens_per_second` is the maximum completed three-second sliding-window rate after first data, never a single-frame burst.

2026-10-09 rerun status: `complete`. [Numeric results and release identity](benchmarks/results/pi-coding-json-compaction.json).

This is the same natural-stop coding task run independently at empty, 60K and 200K input context. Thinking is disabled, EOS remains enabled, and each arm uses temperature 1, top-p 0.95 and top-k 40. The non-empty arms use operator-supplied token-prefix fixtures; only their hashes are published. The three-second peak is the maximum completed sliding-window rate, not a single-frame burst. Runner: [benchmark_pi_coding_contexts.py](experiments/radiance-public/benchmark_pi_coding_contexts.py).

| Context | Prompt tokens | Generated tokens | Mean round | Post-first | Peak 3s | Acceptance | Round events (total; speculative/expected; timed) | Validation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0K | 203 | 4,912 | 37.44 ms | 120.06 tok/s | 158.86 tok/s | 50.03% | 1,092 total; 1091/1091 speculative; 1,091 timed; captured | short natural stop |
| 60K | 60,208 | 5,675 | 41.32 ms | 99.32 tok/s | 141.19 tok/s | 44.33% | 1,384 total; 1383/1383 speculative; 1,383 timed; captured | pass |
| 200K | 200,208 | 7,948 | 50.34 ms | 92.12 tok/s | 129.21 tok/s | 51.96% | 1,715 total; 1714/1714 speculative; 1,714 timed; captured | pass |

Report status: `complete_with_validation_failure`. [Numeric results and every round](benchmarks/results/pi-coding-contexts.json). Each completed row stores every scheduler event under `contexts.<context>.round_capture.records`; a count mismatch is a validation failure. The target is 5,000 output tokens, with shorter natural completions reported explicitly.

These results use [the shared benchmark suite](docs/BENCHMARK_SUITE.md): `benchmark_pi_coding_contexts.py --suite` reuses each context's predetermined unprofiled control for its coding row and complete histogram, and continues the same 60K output through prose, JSON, thinking and compaction. It keeps both controls around each stage trace for the residual calculation. The 60K coding row above and the chained coding row are the same measured request.

### Complete per-round capture

The context benchmark now retains every content-free scheduler event for each arm, including an unmeasured first event. The full numeric records are stored under `contexts.<context>.round_capture.records` in the result JSON; this table is a coverage check rather than another latency aggregate. A count mismatch is a validation failure. Expected rounds come from the speculative-round counter; the first prefill event is retained in logged events but excluded from that counter.

| Context | Logged events | Speculative rounds | Expected speculative rounds | Timed round values | Untimed events | Missing round numbers | Capture status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0K | 1,092 | 1091 | 1091 | 1,091 | 1 | none | captured |
| 60K | 1,384 | 1383 | 1383 | 1,383 | 1 | none | captured |
| 200K | 1,715 | 1714 | 1714 | 1,714 | 1 | none | captured |


The histogram below is generated from the complete per-round records of the [histogram-only rerun](benchmarks/results/pi-round-histogram.json). Every measured `round_ms` value appears in exactly one bin; untimed events are reported separately. This is a separate unprofiled capture. The coding and chained workload tables, compiled-stage timings and residual retain their original measurement captures.
Half-millisecond display bins resolve the current dense clusters at 35–42 and 49–53 ms. These bins are recomputed from the original records; the captured bins, round counts and measurement identity are unchanged.

2026-10-09 rerun after [profiler cleanup](experiments/radiance-public/matched_stage_profile_worker.py). Captured source: base `a16fa00` plus benchmark changes identified by exact file hashes in the artifact. The diagnostic filename-prefix repair was added after that source was frozen.

Output tokens at 0K / 60K / 200K: 4,912 / 5,675 / 7,948. 0K stopped naturally below the requested 5,000-token minimum. Round coverage is complete; overall report status remains `complete_with_validation_failure`.

The original per-arm diagnostic copies missed the cache-job feeds. Bounded supplements retained 7 files; the recorder reports 1 generic dropped record(s).
CPU and HIP diagnostics match all 4,188 selected timed rounds, with no missing or invalid round records. The generic drop is separate from that complete round coverage. All 26 profiler cleanups took 103.33–168.97 ms at excluded trace boundaries, outside the measured controls. Full lifetime archive coverage is not claimed.

| Round time | 0K arm | 60K arm | 200K arm |
|---|---:|---:|---:|
| `<35` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `35–35.5` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `35.5–36` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `36–36.5` | 66 (6.0%) | 0 (0.0%) | 0 (0.0%) |
| `36.5–37` | 450 (41.2%) | 0 (0.0%) | 0 (0.0%) |
| `37–37.5` | 559 (51.2%) | 0 (0.0%) | 0 (0.0%) |
| `37.5–38` | 13 (1.2%) | 0 (0.0%) | 0 (0.0%) |
| `38–38.5` | 2 (0.2%) | 0 (0.0%) | 0 (0.0%) |
| `38.5–39` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `39–39.5` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `39.5–40` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `40–40.5` | 0 (0.0%) | 3 (0.2%) | 0 (0.0%) |
| `40.5–41` | 0 (0.0%) | 1,287 (93.1%) | 0 (0.0%) |
| `41–41.5` | 0 (0.0%) | 75 (5.4%) | 0 (0.0%) |
| `41.5–42` | 0 (0.0%) | 17 (1.2%) | 0 (0.0%) |
| `42–43` | 1 (0.1%) | 0 (0.0%) | 0 (0.0%) |
| `43–45` | 0 (0.0%) | 1 (0.1%) | 0 (0.0%) |
| `45–46` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `46–47` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `47–48` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `48–48.5` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `48.5–49` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `49–49.5` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `49.5–50` | 0 (0.0%) | 0 (0.0%) | 3 (0.2%) |
| `50–50.5` | 0 (0.0%) | 0 (0.0%) | 1,072 (62.5%) |
| `50.5–51` | 0 (0.0%) | 0 (0.0%) | 611 (35.6%) |
| `51–51.5` | 0 (0.0%) | 0 (0.0%) | 19 (1.1%) |
| `51.5–52` | 0 (0.0%) | 0 (0.0%) | 8 (0.5%) |
| `52–52.5` | 0 (0.0%) | 0 (0.0%) | 1 (0.1%) |
| `52.5–53` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
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
| `65–70` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `70–100` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `100–250` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `250–500` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `≥500` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| **Timed rounds** | **1,091** | **1,383** | **1,714** |
| **Untimed events** | **1** | **1** | **1** |
| **Mean / median** | **36.95 / 37.01 ms** | **40.81 / 40.80 ms** | **50.43 / 50.43 ms** |


### Known remaining symptoms and likely causes

**The latest histogram did not reproduce the large stalls.** The separate unprofiled controls recorded 0K: 0/1,091 timed rounds at least 90 ms, longest 42.639 ms; 60K: 0/1,383 timed rounds at least 90 ms, longest 43.133 ms; 200K: 0/1,714 timed rounds at least 90 ms, longest 52.475 ms. This is bounded evidence, not a guarantee against future stalls. [Fresh complete round records](benchmarks/results/pi-round-histogram.json).

**The earlier 591.203 ms and 751.876 ms pauses remain unattributed.** A CPU-only PyTorch check confirmed that stopped profiler cycles can retain native trace results until later garbage collection. The [benchmark cleanup](experiments/radiance-public/matched_stage_profile_worker.py) reclaims those cycles outside measured rounds. The original detailed feeds were not archived, so that mechanism cannot be assigned to either historical pause. [Original round records](benchmarks/results/pi-coding-contexts.json) and [matched controls](benchmarks/results/stage26-control-20261009.json).

**Reasoning throughput could not be isolated.** The completions stream exposed no separate reasoning channel for prose_code, thinking. Those thinking-enabled requests are reported as observed prose; the remaining gap is in stream classification, not a demonstrated target-model arithmetic error.

**Global-512 remains approximate.** The current head study matched reference top-1 in 12,148/12,148 rows, but missed complete reference top-20 support in 5 rows. Retained-score differences are also reported in the head section. Full-head M1/M8 and eager/compiled agreement does not certify the shortlist or its rescoring arithmetic. Use the full BF16 head to remove these head approximations. Neither mode guarantees freedom from model-generated loops. [Current head evidence](benchmarks/results/head-candidate-depth-20261009.json).

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
snapshots and failed cleanup. Reopening the live dashboard reuses the last private
disk inventory immediately, with its age shown; live counters remain current and
older inventories refresh in the background. Local chat names and context counts
appear independently of the full disk scan. Dirty tails normally flush after about 8,192 new
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

The [October 9 speed refresh](benchmarks/results/coherence-current.json) covers
compiled stage timings, matched clean controls, natural coding at 0K/60K/200K,
the chained 60K tasks, target-head latency and cold prefill. The
[shared suite](docs/BENCHMARK_SUITE.md) reuses captures across the tables and
retains every coding round, including stalls. The earlier numerical, cancellation
and cache-state checks remain dated evidence; they were not rerun for this
speed-only refresh. The [speed investigation](docs/SPEED_INVESTIGATION_20260925.md)
retains the original experiments, rejected trials and full-graph preparation repair.

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
