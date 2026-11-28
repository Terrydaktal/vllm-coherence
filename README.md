# vLLM Coherence

Coherence builds on Radiance and GGZ14's RDNA4 work. Its goal is to preserve the
quantized model's outputs and committed state across optimized inference,
speculation and cache reuse. The arithmetic contract specifies the weight,
activation and KV quantization separately. [Arithmetic contract](docs/M1_ARITHMETIC_CONTRACT.md)

- **327,952 tokens on one R9700.** Pinned host embeddings free VRAM for a
  larger KV cache. The authenticated Pi profile completed a cold **327,888-token
  prompt plus 64 speculative output tokens** and published all **407** expected
  snapshot blocks on clean shutdown. [Context qualification](docs/BUILDING.md#327952-token-authenticated-pi-lane)
- **Numerical repairs.** Align ordinary one-token decoding (M1), eight-row
  speculative verification (M8), prefill, and eager and compiled execution.
  Repairs cover GDN convolution and recurrent state, attention and intermediate
  rounding. [Eager-M1 repair audit](docs/eager-m1-followup-audit.md)
- **Checked performance improvements.** Guarded GEMM dispatch, fused
  normalization and FP8 encoding, and tiled GDN prefill retain the tested
  arithmetic. The default Global-512 head ranks the full INT2 score row and
  reranks 512 candidates with BF16. The full BF16 comparison head scores the
  full vocabulary. [Head comparison](#global-512-target-head)
- **Persistent Pi sessions.** Resume from compatible GPU, RAM or disk state.
  Transactional compaction carries forward the active task, and live telemetry
  shows cache reuse, prefill and GPU scheduling. [Pi features](#what-changes-in-pi)
- **Reusable verification.** Locate the first difference using forced-token
  replay, logical-state comparison, independent operator checks and injected
  faults. The checks distinguish execution agreement from independent numerical
  evidence. [Verification scope](docs/VERIFICATION.md)

The November candidate's four execution comparisons matched all **320 full
decode-logit vectors** and the prefill prediction in each pair. Prefill and fresh
M8 also matched full hidden and BF16-logit vectors at **2,320 sampled positions**.
The checks cover finite fixtures. Arbitrary-input equivalence and the approximate
Global-512 head's completeness remain open. [Current numerical results](#current-numerical-results)
· [Independent M1 methodology](docs/eager-m1-all-stages-audit.md)

| Current supported setup | What it runs |
| --- | --- |
| GPU and host | **One AMD Radeon AI PRO R9700, 32 GiB VRAM**, RDNA4 / `gfx1201`, Linux x86-64 |
| Context window | **327,952 total tokens**, including the prompt and generated output |
| Target model | **Qwen3.8-27B-Uncensored-MXFP4-awq**, with MXFP4 weights |
| Speculative drafter | **Qwen3.8-27B-DFlash2-FP8**, with up to seven proposed tokens checked in an eight-row target pass (D7 / M8) |
| Coding agent | **[Pi](https://github.com/earendil-works/pi) 0.84.2**, with Coherence's runtime patches and extensions |
| Serving precision | MXFP4 weights, FP8 activation and KV-cache quantization, with BF16/FP32 arithmetic where specified by the implementation |

[Quick start](#quick-start) · [Pi setup](#pi) · [Benchmarks](#benchmarks)
· [Verification](docs/VERIFICATION.md) · [Architecture](docs/ARCHITECTURE.md)
· [Pi features](#what-changes-in-pi) · [Development](docs/DEVELOPMENT.md)
· [Releases](https://github.com/Terrydaktal/vllm-coherence/releases)

<!-- COHERENCE_BENCHMARK_SUMMARY -->
One R9700 ran the same coding task at three starting context lengths on the 2026-11-26 isolated candidate `621b999a`.

| Starting context | Generated tokens | Coding tok/s after first output | Peak 3-second tok/s | Mean decode round |
| ---: | ---: | ---: | ---: | ---: |
| 0K | 6,248 | **141.89** | 202.88 | 36.66 ms |
| 60K | 7,402 | **109.09** | 154.85 | 40.54 ms |
| 200K | 5,074 | **102.84** | 131.47 | 49.99 ms |

The slowest of 3,863 timed coding rounds took 52.214 ms. Separate cold-prefill runs averaged 32.86 s at 60K and 198.97 s at 200K. [Full benchmark tables](#benchmarks) · [Cold-prefill and cache results](benchmarks/results/retained-20261126/REPORT.md)
<!-- /COHERENCE_BENCHMARK_SUMMARY -->

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

This repository's first commit imports the unchanged Radiance 1.0.16 source
from upstream commit `f295b9ef51ad413a68e4192371e0377741a354ce`. Later commits
group Coherence's additions by feature. Backports include DFlash RNG separation,
the Triton RoPE rounding repair and GGZ14's startup fix that defers INT2 drafter
head packing until the shared weights are ready. [Attribution and source pins](ATTRIBUTION.md)
· [Inherited Radiance README](docs/upstream/RADIANCE-README.md)
· [Archived Radiance Dockerfile](https://github.com/Terrydaktal/vllm-coherence/blob/3cbf9c17e8cef4736563c8ddab514b6874fd96ad/Dockerfile)

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
| Transactional compaction | Manual and automatic compaction preserve the original transcript and replace active context after the checkpoint passes section, completion-marker and finish-status checks. Checkpoint prompts reuse the unchanged token prefix, and checkpoint output can use the remaining context capacity. `Esc` cancels. `Alt+C` explicitly accepts the exact partial summary already generated. |
| Interrupted answers | Received Qwen thinking and prose survive cancellation in the session and active history. Unfinished tool calls remain archival records and never execute. |
| Compaction during generation | With automatic compaction enabled and the default 16,384-token reserve, streamed usage triggers compaction at **311,568 tokens** in the 327,952-token window. Pi saves received thinking and prose before generating a checkpoint. It resumes the task after the checkpoint commits. Cancellation or failure leaves the received answer saved. [Threshold and recovery details](docs/COMPACTION_MEMORY.md#automatic-compaction-during-generation) |
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
| Searchable original history | `pi_session_search` (provided by [`pi-search-tools`](https://github.com/Terrydaktal/pi-search-tools)) retrieves bounded excerpts from original JSONL messages, including history removed by compaction. A lean CPU SQLite index keeps word positions and source offsets without another transcript-text copy. The current branch is the default; other project sessions and alternate branches require explicit scope. [Search and storage contract](https://github.com/Terrydaktal/pi-search-tools/blob/main/docs/SESSION_SEARCH.md). |
| Persistent plans and planning mode | `manage_task_plan` maintains the objective, step status, relevant files and constraints as structured session state. Pi restores it after compaction without repeated plan-file reads. `/plan` toggles read-only investigation; `/plan execute` resumes implementation. [Commands and limits](docs/PLAN_MODE.md). |
| Working-state restoration | Compaction restores bounded source evidence, fresh Git state, up to five relevant files and recorded task handles, prioritizing unfinished work. Loaded instructions stay in the system prompt; historical process handles remain unverified. [Restoration contract](docs/COMPACTION_MEMORY.md). |
| Bounded tool output with retrieval | Large tool results get compact context views while their original available text remains in a SHA-256-verified archive. `rehydrate_tool_result` retrieves line ranges or literal-match excerpts, preserving matches ahead of context under its limits. [Archive and retrieval contract](docs/TOOL_REHYDRATION.md). |
| Reproducible local/remote setup | The launcher installs and checks the patched Pi runtime, loads the operating prompt and extensions, and automatically reserves a free SSH forwarding port. A custom search extension can supply a VM-specific tool. |
| Reusable startup warmup | A warmup receipt binds the running container and release identity. Valid receipts avoid repeating warmup when resuming another chat; missing or changed identities require verification. |

### Two chats sharing one GPU

`/priority` shows the chat's setting. It defaults to `0` and is saved per chat.
Changing an idle chat's priority takes effect when it next queues a model request.

| Setting | When the waiting chat gets the GPU | GPU ownership during its tool calls |
| --- | --- | --- |
| `/priority 0` | Waits for a response boundary with another priority-0 chat. Waits through the whole answer while a higher-priority chat holds the GPU. | Normally gets two seconds for the tool to return before another priority-0 chat can take over. |
| `/priority 1` | Takes over from priority 0 as soon as its current model response ends. If that response ends in a tool call, priority 1 skips priority 0's two-second grace. The tool can keep running. | Holds the GPU against priority 0 until the whole answer finishes. Another priority-1 chat can take over under the equal-priority rules below. |
| `/priority 2` | Can pause a priority-0 or priority-1 response at the next safe GPU step, even mid-sentence. The paused response can resume when it gets the GPU back. | Holds the GPU against priorities 0 and 1 until the whole answer finishes. |

A **model response** ends when the model finishes its output or hands off to a
tool. The **whole answer** includes the tools and any model responses that follow.
Chats at the **same priority** always wait for the current model response to end.
At a tool boundary, the active chat usually gets two seconds for its tool to
return and continue. A longer tool gives a waiting chat a chance to run.

At a handover, the scheduler tries the highest-priority eligible chat first.
Within the same priority, requests waiting at least 30 seconds go first, oldest
first. Otherwise, the chat served least recently goes first. The waiting window
shows who owns the GPU, and a chat's cached state can be parked in system RAM
between turns. [Scheduler code](experiments/radiance-public/radiance_fair_scheduler.py)
· [Pi setup and recovery](docs/PI.md) · [Cache inspection](#cache-inspection)

## Measurement scope

The tables below use the isolated 26 November 2026 worker `621b999a` on
one Radeon AI PRO R9700. The October release and earlier November candidate
comparisons are in the benchmark report. Each cache-path row identifies its
capture so readers can see which builds and conditions it covers.
[November candidate report](benchmarks/results/retained-20261126/REPORT.md).

<!-- COHERENCE_CURRENT_RESULTS -->
## Current numerical results

The 2026-11-26 isolated worker `621b999a` was checked at 320 forced decode positions after a fresh 60,000-token prefill. Each arm uses the full BF16 comparison head to expose differences inside the target model. [Whole-model capture](benchmarks/results/retained-20261126/whole-model-confirmations.json).

| Execution pair | Exact decode logits | Top-1 set/order | Top-10 set/order | Top-20 set/order | Exact prefill logits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compiled M1 → compiled M8 | 320/320 | 320/320 · 320/320 | 320/320 · 320/320 | 320/320 · 320/320 | Yes |
| Eager M1 → compiled M1 | 320/320 | 320/320 · 320/320 | 320/320 · 320/320 | 320/320 · 320/320 | Yes |
| Eager M8 → compiled M8 | 320/320 | 320/320 · 320/320 | 320/320 · 320/320 | 320/320 · 320/320 | Yes |
| Eager M1 → compiled M8 | 320/320 | 320/320 · 320/320 | 320/320 · 320/320 | 320/320 · 320/320 | Yes |

Against stock release arithmetic on the same fixture, candidate compiled M8 matched top-1 at 320/320 positions and full logits at 0/320. Ordered top-20 matched at 0/320, with mean overlap 18.525/20. The table establishes consistency between execution paths in the measured candidate. Approximate head recall and independent arithmetic have separate coverage below. [Stock comparison](benchmarks/results/retained-20261126/results.json).

## Compiled backend stages

Measured on 2026-11-26 using the compiled Global-512 candidate on an isolated worker. Sampling uses temperature 1.0, top-p 0.95 and top-k 40. Context labels are starting prefixes 0K, the private 60K Pi fixture, and the public synthetic 200K fixture. Each context ran a natural warmup followed by clean control, trace, and clean control; each arm generated 6,248 / 7,402 / 5,074 tokens respectively. Generated-token hashes and accepted-token schedules matched. The stage means retain 618 / 961 / 760 complete M8 cycles (0K / 60K / 200K), and controls use exactly those same decode indices. The shared suite has a nominal 1,152-call trace budget after 64 warmup calls, plus one closing boundary per 128-call chunk; only complete eight-row target cycles enter the stage means. Each chunk's closing boundary and first two trace-activation cycles are excluded; natural EOS can end the capture earlier. Trace setup/export boundaries and incomplete or inconsistent trace inventories are excluded by structure, never by duration; complete native round logs retain all rounds and stalls. GPU activity timestamps supply the stage times; CPU annotations, Python hooks and export time are excluded. No per-stage event probes or forced-token replay are used. The measured tracing slowdown was 2.659 / 2.548 / 2.448 ms per retained round; it is reported separately and is **not charged to row 26**. Row 26 is the clean control mean minus the union of GPU activity intervals. This remains an estimate: tracing can indirectly affect clocks and scheduling. Overlap is counted once in the total. The header identifies the measured source commit. The capture retains its original checkout and source identities. [Capture and source identities](benchmarks/results/retained-20261126/stage-detail.json) · [method and uncertainty](docs/STAGE_TIMING.md).

The benchmark ran the isolated November worker `621b999abb59` at context capacity 327,952. The header identifies the captured base commit and measured worker overlay. [Candidate report](benchmarks/results/retained-20261126/REPORT.md).

The **eager M1** column uses independent operator evidence at the named dates and links each claim to its scope. Confidence scores are editorial assessments. **1/5** is untested, **2/5** limited, **3/5** moderate, **4/5** high and **5/5** very high for the tested properties. These scores are not probabilities of being bug-free. Independent FP64 checks use declared error criteria. [Arithmetic contract](docs/M1_ARITHMETIC_CONTRACT.md) · [September independent operator audit](benchmarks/results/current-operator-confirmations-20260925.json).

The independent repeat passed 2,817/2,817 checks at 320 rows per sampled site on base release `08f7c01c`. Projection checks sample output channels across all tested matrices. The long attention checks use structured pages. [Independent repeat](benchmarks/results/retained-20261126/operator-confirmations.json).

The November worker `621b999a` stage replays checked 22 boundaries and 770 layer instances over 320 decode positions. The resulting 123,200 local output and state checks passed. All 40 eight-token groups restored authoritative state and detected deliberate corruption. The replays share native operators and establish execution equivalence on the sampled inputs. [Stage replay capture](benchmarks/results/retained-20261126/stage-confirmations.json).

| Stage | Current GPU activity per retained compiled M8 cycle (0K / 60K / 200K; milliseconds unless explicitly marked; 2026-11-26; base [`21c7e84`](https://github.com/Terrydaktal/vllm-coherence/commit/21c7e84ccfe0fafb3991b02e304171837a7e553a) + worker `621b999a`) | Current M1->M8 correctness and eager->compiled correctness evidence | Current Eager M1 correctness evidence | Last relevant code commit / change | What this stage does |
| --- | ---: | --- | --- | --- | --- |
| **1. Drafter** | 4.815 / 4.860 / 4.921 | Separate proposal model; target M1/M8 comparisons do not independently qualify it. | **N/A: separate proposal model.** Independent proposal/target RNG controls passed. **Uncertainty:** this target-M1 audit does not independently qualify drafter arithmetic or every speculative commit path. | [`9842b30`](https://github.com/Terrydaktal/vllm-coherence/commit/9842b30b53c9b7b32fa30e4562cefbef1b5ff5ad) · skip draft queries on confirmed distant-match streaks. | Suggests up to seven tokens for the target model to check. |
| **2. Embedding + first input normalization + FP8 production** | 0.012 / 0.012 / 0.012 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [2026-11-26 stage replay](benchmarks/results/retained-20261126/stage-confirmations.json) · worker `621b999a`. | **4/5 High.** Historical 24 September exact sampled embeddings and independent RMS checks; released fused norm preserves M1 FP8 bytes/scales at tested batch boundaries from 1 to 2,048 rows. **Uncertainty:** finite inputs and indices. Later complete-model prefill/continuation samples match; arbitrary-input equivalence to fully serial decode is not proved. Independent base-release repeat passed 133/133 checks in these operator groups. [Repeat](benchmarks/results/retained-20261126/operator-confirmations.json). | [`6c34f50`](https://github.com/Terrydaktal/vllm-coherence/commit/6c34f50ef02260159fcf6059e9637bad1f9bd79a) · preserve FP8 rounding in wide prefill. | Looks up token vectors, normalizes them and creates FP8 inputs in one kernel, preserving intermediate rounding. |
| **3. Layer input residual/normalization + FP8 production** | 0.478 / 0.492 / 0.511 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [2026-11-26 stage replay](benchmarks/results/retained-20261126/stage-confirmations.json) · worker `621b999a`. | **4/5 High.** The 24 September qualified release matches native M1 plus independent FP8 encoding at 129 hidden-norm weights × 320 rows; original 55 byte differences eliminated. CPU arithmetic reproduces the saved failing rows. **Uncertainty:** finite inputs, intrinsic/denormal behavior and untested inputs; deployed 24 September. Independent base-release repeat passed 258/258 checks in these operator groups. [Repeat](benchmarks/results/retained-20261126/operator-confirmations.json). | [`6c34f50`](https://github.com/Terrydaktal/vllm-coherence/commit/6c34f50ef02260159fcf6059e9637bad1f9bd79a) · preserve FP8 rounding in wide prefill. | Adds the residual, normalizes the result and creates FP8 inputs in one kernel, preserving intermediate rounding. |
| ↳ GDN input activation FP8 quantization | Included in **stage 3** | Exact fused FP8 bytes/scales; see stage 3 and its evidence scope | **5/5 Very high for tested encoding.** The 24 September qualified release fused bytes/scales match independent encoding; stage 3 prefill/M1 counterexamples pass in that historical qualification. **Uncertainty:** inherits normalization arithmetic; FP8 is an explicitly accepted approximation; deployed 24 September. Independent base-release repeat passed 129/129 checks in these operator groups. [Repeat](benchmarks/results/retained-20261126/operator-confirmations.json). | [`6c34f50`](https://github.com/Terrydaktal/vllm-coherence/commit/6c34f50ef02260159fcf6059e9637bad1f9bd79a) · preserve FP8 rounding in wide prefill. | Uses the FP8 output produced by the same input-normalization kernel. |
| ↳ Attention input activation FP8 quantization | Included in **stage 3** | Exact fused FP8 bytes/scales; see stage 3 and its evidence scope | **5/5 Very high for tested encoding.** The 24 September qualified release fused bytes/scales match independent encoding; stage 3 prefill/M1 counterexamples pass in that historical qualification. **Uncertainty:** inherits normalization arithmetic; FP8 is an explicitly accepted approximation; deployed 24 September. Independent base-release repeat passed 129/129 checks in these operator groups. [Repeat](benchmarks/results/retained-20261126/operator-confirmations.json). | [`6c34f50`](https://github.com/Terrydaktal/vllm-coherence/commit/6c34f50ef02260159fcf6059e9637bad1f9bd79a) · preserve FP8 rounding in wide prefill. | Uses the FP8 output produced by the same input-normalization kernel. |
| **4. GDN input projection** | 3.984 / 3.935 / 3.958 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [2026-11-26 stage replay](benchmarks/results/retained-20261126/stage-confirmations.json) · worker `621b999a`. | **4/5 High.** Shared audit-wide coverage: the 25 September audit checks 496 matrices with 320 synthetic inputs each at sampled output channels. The separate 24 September qualification checks all channels of 30 selected matrices on 32 inputs each (7,739,392 outputs), plus 12 operator-generated activation replays (3,276,800 outputs). These are shared totals, not additional totals for each projection row. **Uncertainty:** other matrices retain sampled-channel coverage; FP64 tolerance passes are not exact-arithmetic proofs. Complete-model stage replays use model activations but share the native operator, not an independent FP64 oracle. Independent base-release repeat across all audited matrices passed 992/992 checks in these operator groups. [Repeat](benchmarks/results/retained-20261126/operator-confirmations.json). | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve stored MXFP4 coefficients in the folded lookup and repair fragment/tiled per-block fallbacks while retaining the qualified GEMM dispatch. | Projects hidden vectors into the inputs and gates for the recurrent layer. |
| **5. GDN layout/copies and buffer initialization** | 0.132 / 0.138 / 0.140 | Authoritative cache/state restored in all 40 eight-token groups; state/output corruption controls detected · [current replay](benchmarks/results/retained-20261126/stage-confirmations.json) · worker `621b999a`. No separate layout top-20 attribution. | **4/5 High in the tested replay.** Unsupported strides/overlap rejected; final histories and untouched slots exact across 48 GDN sites. The historical 25 September replay restores authoritative cache/state in all 40 eight-token groups, including corruption controls. **Uncertainty:** separately dated snapshot and cancellation studies cover finite handover/restore/ownership cases, not every possible history. | [`d43d9a7`](https://github.com/Terrydaktal/vllm-coherence/commit/d43d9a74e713e7d283075fd6032ebef37d3be08e) · initialize qualified prefill convolution slots in place. | Arranges inputs and clears temporary buffers throughout each GDN layer. |
| **6. GDN convolution** | 0.176 / 0.180 / 0.186 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [2026-11-26 stage replay](benchmarks/results/retained-20261126/stage-confirmations.json) · worker `621b999a`. | **4/5 High.** Historical 25 September audit: 48 layers × 320 consecutive inputs pass FP64 convolution/SiLU criteria; histories and untouched slots exact. **Uncertainty:** finite numerical/state samples; full-session integration remains unqualified by this audit. Independent base-release repeat passed 144/144 checks in these operator groups. [Repeat](benchmarks/results/retained-20261126/operator-confirmations.json). | [`d43d9a7`](https://github.com/Terrydaktal/vllm-coherence/commit/d43d9a74e713e7d283075fd6032ebef37d3be08e) · initialize qualified prefill convolution slots in place. | Updates recent-token history using the corrected multiply/add order. |
| **7. GDN recurrence and gates** | 1.303 / 1.175 / 1.217 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [2026-11-26 stage replay](benchmarks/results/retained-20261126/stage-confirmations.json) · worker `621b999a`. | **4/5 High.** Historical 25 September audit: 48 layers × 320 transitions match independent state/output equations within tolerance; 32,768-step decay passes. **Uncertainty:** no exact full-model recurrence proof or coverage of every reachable state. Independent base-release repeat passed 149/149 checks in these operator groups. [Repeat](benchmarks/results/retained-20261126/operator-confirmations.json). The retained indexed bias wrapper passed 480/480 state, output and guard checks against an FP64 oracle across the sampled accepted-state slots. Maximum absolute error was 0.000244 for output and 3.06e-07 for state. [Retained bias audit](benchmarks/results/retained-20261126/retained-gdn-bias.json). | [`b312bf1`](https://github.com/Terrydaktal/vllm-coherence/commit/b312bf1552882ed066b960307dd4df4316cb76b7) · preserve lazy GDN alignment alongside the indexed-state and packed-bias changes. | Updates recurrent memory in token order. Faster prefill retains the existing nine-slot state layout. |
| **8. GDN output gated normalization + FP8 production** | 0.144 / 0.147 / 0.152 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [2026-11-26 stage replay](benchmarks/results/retained-20261126/stage-confirmations.json) · worker `621b999a`. | **4/5 High numerically.** Explicit FP32 norm/weight/gate contract with one final BF16 boundary; the 24 September released path matches native M1 at all 48 sites × 320 audit rows, plus 1,000-row release gates and batch-boundary checks. **Uncertainty:** this deliberately differs from Hugging Face intermediate casts; intrinsic behavior and arbitrary inputs remain unproved; deployed 24 September. Independent base-release repeat passed 96/96 checks in these operator groups. [Repeat](benchmarks/results/retained-20261126/operator-confirmations.json). | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 gated-normalization reduction layout during prefill and retain one final BF16 rounding before FP8 production. | Normalizes and gates GDN output, then produces FP8 inputs in one kernel; intermediate BF16 rounding is preserved. |
| ↳ GDN output activation FP8 quantization | Included in **stage 8** | Exact fused FP8 bytes/scales; see stage 8 and its evidence scope | **5/5 Very high for tested encoding.** The 24 September qualified M1/prefill bytes and scales match independent encoding of native BF16 output at 48 sites × 320 rows. **Uncertainty:** inherits the declared stage 8 arithmetic; no weight-only model equivalence; deployed 24 September. Independent base-release repeat passed 48/48 checks in these operator groups. [Repeat](benchmarks/results/retained-20261126/operator-confirmations.json). | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve the M1 gated-normalization reduction layout during prefill and retain one final BF16 rounding before FP8 production. | Uses the FP8 output produced by the same gated-normalization kernel. |
| **9. GDN output projection** | 1.801 / 1.799 / 1.809 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [2026-11-26 stage replay](benchmarks/results/retained-20261126/stage-confirmations.json) · worker `621b999a`. | **4/5 High.** Shared audit-wide coverage: the 25 September audit checks 496 matrices with 320 synthetic inputs each at sampled output channels. The separate 24 September qualification checks all channels of 30 selected matrices on 32 inputs each (7,739,392 outputs), plus 12 operator-generated activation replays (3,276,800 outputs). These are shared totals, not additional totals for each projection row. **Uncertainty:** other matrices retain sampled-channel coverage; FP64 tolerance passes are not exact-arithmetic proofs. Complete-model stage replays use model activations but share the native operator, not an independent FP64 oracle. Independent base-release repeat across all audited matrices passed 992/992 checks in these operator groups. [Repeat](benchmarks/results/retained-20261126/operator-confirmations.json). | [`41730ab`](https://github.com/Terrydaktal/vllm-coherence/commit/41730ab2594b50969aae1799b94d357b4d73502a) · produce FP8 tiles directly for wide input. | Maps the recurrent-layer result back to the model's hidden-vector width. |
| **10. Attention input projection** | 1.135 / 1.146 / 1.143 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [2026-11-26 stage replay](benchmarks/results/retained-20261126/stage-confirmations.json) · worker `621b999a`. | **4/5 High.** Shared audit-wide coverage: the 25 September audit checks 496 matrices with 320 synthetic inputs each at sampled output channels. The separate 24 September qualification checks all channels of 30 selected matrices on 32 inputs each (7,739,392 outputs), plus 12 operator-generated activation replays (3,276,800 outputs). These are shared totals, not additional totals for each projection row. **Uncertainty:** other matrices retain sampled-channel coverage; FP64 tolerance passes are not exact-arithmetic proofs. Complete-model stage replays use model activations but share the native operator, not an independent FP64 oracle. Independent base-release repeat across all audited matrices passed 992/992 checks in these operator groups. [Repeat](benchmarks/results/retained-20261126/operator-confirmations.json). | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve stored MXFP4 coefficients in the folded lookup and repair fragment/tiled per-block fallbacks while retaining the qualified GEMM dispatch. | Projects hidden vectors into the inputs for attention. |
| **11. Attention Q/K normalization, RoPE and layout** | 0.209 / 0.213 / 0.220 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [2026-11-26 stage replay](benchmarks/results/retained-20261126/stage-confirmations.json) · worker `621b999a`. Separate Q and K checks cover 32 sites. | **4/5 High.** Historical 25 September audit: All 32 Q/K norm sites; both position paths at nine ranges pass independent arithmetic checks; repaired eager MRoPE launches verified. **Uncertainty:** rounding tolerances and compact RoPE tables; full-table addressing not covered. Independent base-release repeat passed 108/108 checks in these operator groups. [Repeat](benchmarks/results/retained-20261126/operator-confirmations.json). | [`2aaa0d3`](https://github.com/Terrydaktal/vllm-coherence/commit/2aaa0d389033ba71a5f0abf35b2974459aa5ffff) · use the M1 reduction in wide Q/K normalization. | Normalizes queries and keys and applies positional rotation (RoPE), retaining the corrected rounding. |
| **12. Attention KV write** | 0.033 / 0.034 / 0.035 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [2026-11-26 stage replay](benchmarks/results/retained-20261126/stage-confirmations.json) · worker `621b999a`. | **5/5 Very high within tested cases.** Historical 25 September audit: 393,216 stored values/destinations exact; BF16/OCP FP8 scales, masked slots, page edges and every finite FP8 code checked. **Uncertainty:** arbitrary aliasing, asynchronous access and native snapshot restoration remain unproved. Independent base-release repeat passed 5/5 checks in these operator groups. [Repeat](benchmarks/results/retained-20261126/operator-confirmations.json). | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Aligned causal attention and cache writes with the M1/M8 arithmetic contract. | Stores new keys and values in the cache for reuse by later tokens. |
| **13. Attention decode** | 0.495 / 4.702 / 13.895 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [2026-11-26 stage replay](benchmarks/results/retained-20261126/stage-confirmations.json) · worker `621b999a`. Decode and merge checked together. | **4/5 High within tested cases.** Historical operator evidence: Earlier 88 cases plus 96 independent FP64 cases with unique shuffled pages, holes, poisoned tails, per-head scales and varied contents through 60,001 tokens; guards intact. **Uncertainty:** longest 253K checks still use structured/repeated pages; exact softmax and arbitrary layouts remain unproved. Independent base-release repeat passed 176/176 checks in these operator groups. [Repeat](benchmarks/results/retained-20261126/operator-confirmations.json). | [`f3b5778`](https://github.com/Terrydaktal/vllm-coherence/commit/f3b5778b5b1e4e4c2d9bd751bf3882ead13db84a) · preserve logical splits in short FULL graphs. | Attends to the current and earlier tokens using corrected arithmetic and shared cache loads. |
| **14. Attention split-KV merge** | 0.266 / 0.131 / 0.106 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [2026-11-26 stage replay](benchmarks/results/retained-20261126/stage-confirmations.json) · worker `621b999a`. Decode and merge checked together. | **4/5 High for repaired arithmetic.** Historical operator evidence: Four exact split-mean failures repaired; 96 additional varied-layout attention cases pass, including exact BF16 cancellation controls. **Uncertainty:** integrated evidence, not an exhaustive independent merge-stage proof. Independent base-release repeat passed 176/176 checks in these operator groups. [Repeat](benchmarks/results/retained-20261126/operator-confirmations.json). | [`f3b5778`](https://github.com/Terrydaktal/vllm-coherence/commit/f3b5778b5b1e4e4c2d9bd751bf3882ead13db84a) · preserve logical splits in short FULL graphs. | Combines attention results from cache partitions in the corrected arithmetic order. |
| **15. Attention output gating** | 0.021 / 0.026 / 0.026 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [2026-11-26 stage replay](benchmarks/results/retained-20261126/stage-confirmations.json) · worker `621b999a`. | **4/5 High.** Historical 25 September audit: Every finite BF16 sigmoid input tested with a bounded multiplier against FP64 equations. **Uncertainty:** numerical tolerances; not every gate/output pair or full attention-to-gate integration. Independent base-release repeat passed 4/4 checks in these operator groups. [Repeat](benchmarks/results/retained-20261126/operator-confirmations.json). | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Preserved attention output gating and intermediate BF16 rounding in aligned M1/M8 execution. | Applies learned gates to the attention output. |
| **16. Attention output activation FP8 quantization** | 0.036 / 0.042 / 0.043 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [2026-11-26 stage replay](benchmarks/results/retained-20261126/stage-confirmations.json) · worker `621b999a`. | **4/5 High for tested encoding and integration.** Historical independent FP8 codec checks plus the 25 September 320-token M1/M8 and eager/compiled quantizer replay agree at every attention layer. **Uncertainty:** independent oracle checks remain sampled; arbitrary gate/quantization inputs and layouts are unproved. | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Enabled qualified native-rounding FP8 output production and rejected approximate traced quantization. | Converts the attention output to FP8 for its output projection. |
| **17. Attention output projection** | 0.579 / 0.528 / 0.509 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [2026-11-26 stage replay](benchmarks/results/retained-20261126/stage-confirmations.json) · worker `621b999a`. | **4/5 High.** Shared audit-wide coverage: the 25 September audit checks 496 matrices with 320 synthetic inputs each at sampled output channels. The separate 24 September qualification checks all channels of 30 selected matrices on 32 inputs each (7,739,392 outputs), plus 12 operator-generated activation replays (3,276,800 outputs). These are shared totals, not additional totals for each projection row. **Uncertainty:** other matrices retain sampled-channel coverage; FP64 tolerance passes are not exact-arithmetic proofs. Complete-model stage replays use model activations but share the native operator, not an independent FP64 oracle. Independent base-release repeat across all audited matrices passed 992/992 checks in these operator groups. [Repeat](benchmarks/results/retained-20261126/operator-confirmations.json). | [Qualified speed changes](docs/SPEED_INVESTIGATION_20260925.md): Use qualified local-split GEMM for M1/M8 down/output shapes; preserve the repaired MXFP4 coefficients and reduction order. | Maps the attention result back to the model's hidden-vector width. |
| **18. Post-attention/GDN residual/normalization + FP8 production** | 0.491 / 0.508 / 0.525 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [2026-11-26 stage replay](benchmarks/results/retained-20261126/stage-confirmations.json) · worker `621b999a`. | **4/5 High.** The 24 September qualified release matches native M1 plus independent FP8 encoding in the 129-site × 320-row hidden-norm audit; original 55 byte differences eliminated. **Uncertainty:** finite inputs and prefill-versus-serial-decode behavior; deployed 24 September. Independent base-release repeat passed 258/258 checks in these operator groups. [Repeat](benchmarks/results/retained-20261126/operator-confirmations.json). | [`6c34f50`](https://github.com/Terrydaktal/vllm-coherence/commit/6c34f50ef02260159fcf6059e9637bad1f9bd79a) · preserve FP8 rounding in wide prefill. | Adds the layer result to the residual, normalizes it and creates FP8 inputs for the MLP in one kernel. |
| ↳ MLP gate/up input FP8 quantization | Included in **stage 18** | Exact fused FP8 bytes/scales; see stage 18 and its evidence scope | **5/5 Very high for tested encoding.** The 24 September released bytes/scales match independent encoding of M1 BF16 norm; stage 18 prefill counterexamples pass in that historical qualification. **Uncertainty:** inherits normalization arithmetic and accepted FP8 loss; deployed 24 September. Independent base-release repeat passed 129/129 checks in these operator groups. [Repeat](benchmarks/results/retained-20261126/operator-confirmations.json). | [`6c34f50`](https://github.com/Terrydaktal/vllm-coherence/commit/6c34f50ef02260159fcf6059e9637bad1f9bd79a) · preserve FP8 rounding in wide prefill. | Uses the FP8 output produced by the same post-normalization kernel. |
| **19. MLP gate/up projection** | 11.274 / 11.057 / 11.069 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [2026-11-26 stage replay](benchmarks/results/retained-20261126/stage-confirmations.json) · worker `621b999a`. | **4/5 High.** Shared audit-wide coverage: the 25 September audit checks 496 matrices with 320 synthetic inputs each at sampled output channels. The separate 24 September qualification checks all channels of 30 selected matrices on 32 inputs each (7,739,392 outputs), plus 12 operator-generated activation replays (3,276,800 outputs). These are shared totals, not additional totals for each projection row. **Uncertainty:** other matrices retain sampled-channel coverage; FP64 tolerance passes are not exact-arithmetic proofs. Complete-model stage replays use model activations but share the native operator, not an independent FP64 oracle. Independent base-release repeat across all audited matrices passed 992/992 checks in these operator groups. [Repeat](benchmarks/results/retained-20261126/operator-confirmations.json). | [`2f0c571`](https://github.com/Terrydaktal/vllm-coherence/commit/2f0c571a8847c8059c8aa549b60978a1d9722fed): Preserve stored MXFP4 coefficients in the folded lookup and repair fragment/tiled per-block fallbacks while retaining the qualified GEMM dispatch. | Computes both MLP input projections. Wider decode dispatch avoids the slower prefill kernel for qualified shapes. |
| **20. MLP SiLU and gating** | 0.118 / 0.124 / 0.126 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [2026-11-26 stage replay](benchmarks/results/retained-20261126/stage-confirmations.json) · worker `621b999a`. | **4/5 High.** Historical 25 September audit: Every finite BF16 gate encoding checked with a bounded multiplier; 320 fused M1 cases pass. **Uncertainty:** tiny subnormal differences from FP64 remain; not all gate/up pairs are covered. Independent base-release repeat passed 4/4 checks in these operator groups. [Repeat](benchmarks/results/retained-20261126/operator-confirmations.json). | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Preserved intermediate BF16 rounding in aligned M1/M8 and eager/compiled arithmetic; the slower fused SiLU remains disabled. | Applies SiLU and combines the two MLP branches, retaining intermediate BF16 rounding. |
| **21. MLP down input FP8 quantization** | 0.240 / 0.254 / 0.262 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [2026-11-26 stage replay](benchmarks/results/retained-20261126/stage-confirmations.json) · worker `621b999a`. | **5/5 Very high for tested encoding.** Historical 25 September audit: 320 fused M1 cases exactly match independent FP8 encoding of the native BF16 SiLU/gate pipeline. **Uncertainty:** inherits stage 20 numerical behavior; no exhaustive input-pair or shape proof. Independent base-release repeat passed 320/320 checks in these operator groups. [Repeat](benchmarks/results/retained-20261126/operator-confirmations.json). | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Enabled qualified native per-token FP8 production for the down projection input. | Converts MLP activations to FP8 for the down projection. |
| **22. MLP down projection** | 5.384 / 5.266 / 5.284 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [2026-11-26 stage replay](benchmarks/results/retained-20261126/stage-confirmations.json) · worker `621b999a`. | **4/5 High.** Shared audit-wide coverage: the 25 September audit checks 496 matrices with 320 synthetic inputs each at sampled output channels. The separate 24 September qualification checks all channels of 30 selected matrices on 32 inputs each (7,739,392 outputs), plus 12 operator-generated activation replays (3,276,800 outputs). These are shared totals, not additional totals for each projection row. **Uncertainty:** other matrices retain sampled-channel coverage; FP64 tolerance passes are not exact-arithmetic proofs. Complete-model stage replays use model activations but share the native operator, not an independent FP64 oracle. Independent base-release repeat across all audited matrices passed 992/992 checks in these operator groups. [Repeat](benchmarks/results/retained-20261126/operator-confirmations.json). | [Qualified speed changes](docs/SPEED_INVESTIGATION_20260925.md): Use qualified local-split GEMM for M1/M8 down/output shapes; preserve the repaired MXFP4 coefficients and reduction order. | Projects the MLP result back to the model's hidden-vector width. |
| **23. Final normalization/layout** | 0.011 / 0.011 / 0.012 | M1/M8: 320/320; 320/320. Eager/compiled M8: 320/320; 320/320 · [2026-11-26 stage replay](benchmarks/results/retained-20261126/stage-confirmations.json) · worker `621b999a`. | **4/5 High.** The historical 25 September audit checks 320 inputs at the actual final BF16 norm against independent RMS equations. **Uncertainty:** sampled numerical criteria; complete-model mode agreement is execution consistency, not an independent reference proof. The extra FP8 control is not serving behavior. Independent base-release repeat passed 161/161 checks in these operator groups. [Repeat](benchmarks/results/retained-20261126/operator-confirmations.json). | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Aligned final normalization and full BF16 head precision boundaries while preserving the rounding contract. | Normalizes the final hidden vector before vocabulary scoring. |
| **24. Global-512 target head** | 0.928 / 0.931 / 0.961 | Same top-1: 12,141/12,141; complete reference top-20 retained: 12,137/12,141. Includes M1/M8; not an M1-versus-M8 ordering test · [current head study](benchmarks/results/retained-20261126/results.json). | **4/5 for independently sampled score arithmetic.** Three 512-row weight slabs were checked against FP64 criteria in the 25 September operator audit. The November head study retains top-1 in 12,141/12,141 rows, complete top-20 in 12,137/12,141 and complete top-40 in 12,120/12,141. **Uncertainty:** scores are sampled, shortlist selection is approximate, and the head study combines M1 and M8. [November head evidence](benchmarks/results/retained-20261126/results.json). Independent base-release repeat passed 9/9 checks in these operator groups. [Repeat](benchmarks/results/retained-20261126/operator-confirmations.json). | [`dd1aa90`](https://github.com/Terrydaktal/vllm-coherence/commit/dd1aa906f9e795bd4008224ce6661b5fdb69fa16) · rerank live Global-512 rows only. | Scores the vocabulary with INT2, selects 512 candidates and rescores them with BF16 weights. Selection remains approximate. |
| **25. Other GPU bookkeeping** | 0.460 / 0.467 / 0.477 | No isolated top-20 operator claim; sampling/state controls have separate evidence. | **3/5 Moderate for sampled decision/state checks.** Historical 24 September sampling evidence: 600,000 trials; maximum observed probability error 0.118 percentage points; deliberately shared proposal noise detected by the negative control. **Uncertainty:** this does not independently qualify every bookkeeping kernel or asynchronous commit/rollback path. | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Integrated qualified sampling, cache/state bookkeeping and graph-capture admission; no isolated correctness claim is made for this aggregate row. | Runs sampling and cache/state update kernels outside the named model stages. |
| **26. Estimated runtime overhead** | 2.524 / 2.403 / 2.601 (estimate) | [Matched control minus GPU activity union](benchmarks/results/retained-20261126/stage-detail.json) | **N/A: timing estimate.** No model arithmetic to score. **Uncertainty:** performance residual is not evidence of scheduler or session-state correctness. | [`21c7e84`](https://github.com/Terrydaktal/vllm-coherence/commit/21c7e84ccfe0fafb3991b02e304171837a7e553a): record matched unprofiled controls and overlap-corrected residuals | Indirect observer effects are not proved zero. |
| **Total reconstructed round (stages 1–26)** | **36.806 / 40.468 / 50.020** | GPU activity union plus the estimated remainder | **N/A: no aggregate correctness score.** Operator scores cannot be averaged into a model guarantee. **Uncertainty:** arbitrary-input prefill/M1 equality, approximate-head completeness and exhaustive restore/concurrency qualification remain open; finite complete-model equality and lifecycle checks are reported separately. | — | Overlapping stages are counted once in the total. |

The table groups the backend into 26 stages. Each timing cell is ordered **0K / 60K / 200K**. Fused kernels are charged once to their containing stage; the ↳ rows are detail-only inclusion records and add no timing; rows without a separate profiler scope are labelled in the timing cell rather than displayed as 0.000. The total counts overlapping GPU activity once.

Each **set/order** pair means the same top-20 token set, followed by the same ranking. Current stage confirmations identify M1/M8 and eager/compiled M8 separately. Each position passes only if every layer instance passes on the same captured inputs. These diagnostic stage replays are checked against the compiled graph control; their times are not used in this table. Timing and correctness captures record their own exact source hashes. The provenance column identifies the last relevant code change.

Fusion still permits correctness instrumentation: a diagnostic kernel can expose intermediate values, and fused outputs can be compared with an unfused reference. The normal GPU profile measures the combined kernel. Internal probes or splitting the kernel can change its performance, so those measurements are not an additive breakdown of the production kernel's time.

### Cache-state equivalence and prefill/restore timings

The isolated 2026-11-26 candidate covers fresh prefill/decode equality, cache-boundary replay, tool continuation and restarted disk restore. Additional paths retain the dates and scopes of their earlier measurements. Incomplete candidate probes include pressure (`FAIL`). Both candidate pressure requests finished. The old-policy negative control did not reproduce repeated allocator preemption, so the wider comparison remains incomplete. [2026-11-26 candidate lifecycle](benchmarks/results/retained-20261126/lifecycle.json).

| State / execution comparison | Workload and cache coverage | Correctness evidence | Measured speed / latency | Capture |
| --- | --- | --- | --- | --- |
| Short cold prefill ↔ fresh M8 decode history | Cold prompt rows around 127/128, 255/256, 1,023/1,024, 1,647/1,648 and 3,296 | 20/20 cases passed, with byte-exact hidden rows 608/608 and exact full BF16 logits 608/608 | Vector replay not separately timed | [2026-11-26 candidate](benchmarks/results/retained-20261126/REPORT.md) |
| Cold prefill ↔ fresh M8 decode history | 1,651-token prefix + 1,000 forced-token positions | Byte-exact hidden rows 1,000/1,000; exact full BF16-logit rows 1,000/1,000; identical top-1 1,000/1,000 | Not separately benchmarked | [2026-11-26 candidate](benchmarks/results/retained-20261126/REPORT.md) |
| Cold prefill ↔ fresh M8 decode history | 60,000-token prefix + 1,000 forced-token positions | Byte-exact hidden rows 1,000/1,000; exact full BF16-logit rows 1,000/1,000; identical top-1 1,000/1,000 | Vector replay not separately timed; current cold-prefill speeds below | [2026-11-26 candidate](benchmarks/results/retained-20261126/REPORT.md) |
| Cold prefill ↔ fresh M8 decode history | 200,000-token prefix + 320 forced-token positions | Byte-exact hidden rows 320/320; exact full BF16-logit rows 320/320; identical top-1 320/320 | Vector replay not separately timed; current cold-prefill speeds below | [2026-11-26 candidate](benchmarks/results/retained-20261126/REPORT.md) |
| Warm coding ↔ cold coding | The same natural-stop workload at 0K / 60K / 200K starting context | 3/3 exact outputs. 6,248 / 7,402 / 5,074 generated tokens | Throughput and first data in the coding table | [2026-11-26 coding capture](benchmarks/results/retained-20261126/REPORT.md#coding-and-chained-requests) |
| Cached continuation ↔ fresh execution | 60,208-token prompt, comparison from position 57,680; 256- and 1,024-token chunk boundaries | Byte-exact hidden rows 2,528/2,528; 17/17 checks including 128-token greedy and sampled replay, native and forced graph/KV boundaries, and sampled cancellation with a waiter | Paths not separately timed | [2026-11-26 candidate](benchmarks/results/retained-20261126/REPORT.md) |
| Natural, raw, compressed and damaged disk checkpoints ↔ cold controls | 164, 1,684 context tokens; 0 tokens reused after damage | 4/4 exact 64-token outputs | Paths not separately timed | [2026-11-26 candidate](benchmarks/results/retained-20261126/REPORT.md) |
| Cached tool continuation across GPU/RAM handover ↔ cold controls | 256- and 1,648-token prefixes; resident GPU and parked RAM paths; 267, 1,659 tokens reused | 4/4 exact 64-token continuations | Paths not separately timed | [2026-11-26 candidate](benchmarks/results/retained-20261126/REPORT.md) |
| Response-end state copy ↔ source state | 48 GDN layers, 576 state/history checks and 144 physical-page copies | 720/720 byte-exact copies across 1,450,967,040 bytes checked. Status `PASS_FOR_BYTE_PRESERVATION_ONLY`. | Instrumented byte-copy check | [2026-11-26 candidate lifecycle](benchmarks/results/retained-20261126/lifecycle.json) |
| Resident GPU reuse across block/response boundaries | 1,657, 1,675, 1,719, 3,307, 3,325, 3,369 input tokens, 5, 6 uncached | 6/6 exact continuations, 64 generated tokens each | First data **0.109–0.259 s** | [2026-11-26 candidate lifecycle](benchmarks/results/retained-20261126/lifecycle.json) |
| Resident GPU append, long context | 60,024 input tokens, 60,019 cached, 5 uncached | Exact resident prefix reported | First data **0.261 s** | [2026-11-26 candidate lifecycle](benchmarks/results/retained-20261126/lifecycle.json) |
| Resident GPU append, long context | 200,024 input tokens, 200,019 cached, 5 uncached | Exact resident prefix reported | First data **0.304 s** | [2026-11-26 candidate lifecycle](benchmarks/results/retained-20261126/lifecycle.json) |
| Resident GPU continuation ↔ uninterrupted control, long context | 60,025 input tokens, 60,020 cached, 5 uncached | 1/1 exact continuations, 64 generated tokens each | First data **0.221 s** | [2026-09-27 cache paths](benchmarks/results/response-end-cache-20260927.json) |
| Resident GPU continuation ↔ uninterrupted control, long context | 200,025 input tokens, 200,020 cached, 5 uncached | 1/1 exact continuations, 64 generated tokens each | First data **1.026 s** | [2026-09-27 cache paths](benchmarks/results/response-end-cache-20260927.json) |
| GPU → RAM → GPU chat handover | 1,657, 1,675, 1,719, 3,307, 3,325, 3,369 input tokens, 5, 6 uncached | 6/6 exact continuations, 64 generated tokens each | First data **0.210–0.360 s** | [2026-11-26 candidate lifecycle](benchmarks/results/retained-20261126/lifecycle.json) |
| Reuse after GPU-bank eviction | 1,657, 1,675, 1,719, 3,307, 3,325, 3,369 input tokens, 5, 6 uncached | 6/6 exact continuations, 64 generated tokens each | First data **0.145–0.317 s** | [2026-11-26 candidate lifecycle](benchmarks/results/retained-20261126/lifecycle.json) |
| Disk restore after backend restart | 1,675 input tokens, 1,669 cached, 6 uncached | 1/1 exact continuations, 64 generated tokens each | First data **0.491 s** | [2026-11-26 candidate](benchmarks/results/retained-20261126/REPORT.md) |
| Disk restore after backend restart | 3,325 input tokens, 3,320 cached, 5 uncached | 1/1 exact continuations, 64 generated tokens each | First data **0.556 s** | [2026-11-26 intermediate worker](benchmarks/results/retained-20261126/REPORT.md) |
| Disk restore after backend restart | 60,024 input tokens, 60,018 cached, 6 uncached | 1/1 exact continuations, 64 generated tokens each | First data **1.794 s** | [2026-11-26 candidate](benchmarks/results/retained-20261126/REPORT.md) |
| Disk restore after backend restart | 200,024 input tokens, 200,019 cached, 5 uncached | 1/1 exact continuations, 64 generated tokens each | First data **5.616 s** | [2026-11-26 candidate](benchmarks/results/retained-20261126/REPORT.md) |
| Damaged snapshot → reject and cold rebuild | 3,325 input tokens, 0 cached, 3,325 uncached | 1/1 exact continuations, 64 generated tokens each; corrupted endpoint rejected; zero cached tokens used | First data **1.912 s** | [2026-09-27 cache paths](benchmarks/results/response-end-cache-20260927.json) |
| Reuse after explicit stop boundary | 1,657, 1,675, 1,721, 3,307, 3,325, 3,369 input tokens, 5, 6 uncached | 6/6 exact continuations, 64 generated tokens each | First data **0.118–0.264 s** | [2026-11-26 candidate lifecycle](benchmarks/results/retained-20261126/lifecycle.json) |
| Cached tool continuation ↔ cold full prompt (full BF16) | 1,701, 1,711, 1,745, 1,755, 3,361, 3,405 input tokens, 41 uncached | 6/6 exact continuations, 64 generated tokens each; greedy; same appended 41-token tool suffix | First data **0.189–1.240 s** | [2026-09-27 alignment](benchmarks/results/prefill-alignment-20260927.json) |
| Cached tool continuation ↔ cold full prompt (Global-512) | 1,701, 1,711, 1,745, 1,755, 3,361, 3,405 input tokens, 41 uncached | 6/6 exact continuations, 64 generated tokens each; greedy; same appended 41-token tool suffix | First data **0.183–1.057 s** | [2026-09-27 alignment](benchmarks/results/prefill-alignment-20260927.json) |
| Repeated sampled tool continuation | 1,711, 3,361 input tokens, 41, 42 uncached | 2/2 exact continuations, 64 generated tokens each; T=1, top-p=0.95, top-k=40, seed=0; same prefill/decode boundaries, not a cold comparison | First data **0.469–0.501 s** | [2026-09-27 cache paths](benchmarks/results/response-end-cache-20260927.json) |
| Cancelled continuation ↔ uninterrupted control | 1,657, 1,675, 1,719, 3,307, 3,325, 3,369 input tokens, 5, 6 uncached | 6/6 exact continuations, 64 generated tokens each | First data **0.053–0.068 s** | [2026-11-26 candidate lifecycle](benchmarks/results/retained-20261126/lifecycle.json) |
| Cancel during decode → replay | Interrupted after 1, 67, 262 received tokens | 3/3 replays equal uninterrupted control | Cancellation to scheduler release: **25.370–50.727 ms** | [2026-11-26 candidate lifecycle](benchmarks/results/retained-20261126/lifecycle.json) |
| Cancel during cold prefill → replay | 60K input; 3,296 processed at cancellation | Replay equal | Cancellation to scheduler release: **1,236.117 ms** | [2026-11-26 candidate lifecycle](benchmarks/results/retained-20261126/lifecycle.json) |
| Cancel queued chat → replay | Two chats; cancelled request had not generated | Replay equal | Cancellation to scheduler release: **26.151 ms** | [2026-11-26 candidate lifecycle](benchmarks/results/retained-20261126/lifecycle.json) |
| Sampled cancellation with another chat waiting → replay | Two chats; T=1, top-p=0.95, top-k=40, seed=113 | Replay equal | Cancellation to scheduler release: **25.437 ms** | [2026-11-26 candidate lifecycle](benchmarks/results/retained-20261126/lifecycle.json) |
| Concurrent chats and priority handovers ↔ uninterrupted controls | Equal priority, immediate priority-2 takeover/cancellation, priority-1 tool-boundary hold | 5/5 lifecycle cases pass their output/ownership checks | Isolated handover latency not measured | [2026-11-26 candidate lifecycle](benchmarks/results/retained-20261126/lifecycle.json) |
| Near-limit endpoint ownership and cache reclamation | 313,880 input; 307,342 cached; 1,024 generated | 0 allocator preemptions; forward progress (not numerical-equivalence evidence) | First data **15.633 s** | [2026-11-26 candidate lifecycle](benchmarks/results/retained-20261126/lifecycle.json) |
| Near-limit endpoint ownership and cache reclamation | 326,160 input; 307,342 cached; 512 generated | 0 allocator preemptions; forward progress (not numerical-equivalence evidence) | First data **45.161 s** | [2026-11-26 candidate lifecycle](benchmarks/results/retained-20261126/lifecycle.json) |
| Compaction-generation snapshot replacement | 3 synthetic storage cycles. A 10,000-token initial prompt preceded 4,000-token prompts requesting 8 output tokens each. Published lengths 10,007 → 4,008 → 4,008 → 4,008. | 3/3 cycles retained one published generation and left the other chat unchanged. Status `PASS_FOR_DECLARED_SCOPE`. | Activation **5.271–6.489 ms**. Request to durable snapshot **2.844–2.998 s**. Synthetic publication follows generation. Full Pi compaction was not timed. | [2026-11-26 candidate snapshot replacement](benchmarks/results/retained-20261126/snapshot-replacement.json) |

**Timing boundaries:** first data is request-to-first-output wall time, including admission, handover, restore and any required prefill; it is not pure disk or RAM transfer time. Restart measurements start after the backend is ready. Cancellation times measure scheduler release, not completion of the replay. No qualification-suite runtime is substituted for an operation benchmark.

**Equality boundaries:** the candidate full-vector prefill checks cover 2,320 distinct forced-token positions. Matching generated tokens alone does not establish equality of every latent state or logit. Compaction creates a new history, so its storage test checks safe replacement rather than equivalence to the uncompressed conversation. These finite checks do not prove arbitrary-input correctness, exhaustive scheduling interleavings or completeness of the approximate Global-512 head.

### Cold-prefill speeds (2026-11-26 candidate)

| Cold context | Mean prefill | Prefill tokens/s | Mean first data | Samples |
| --- | ---: | ---: | ---: | ---: |
| 60,000 tokens | **32.86 s** | **1,826** | 33.27 s | 2 |
| 200,000 tokens | **198.97 s** | **1,005** | 199.66 s | 2 |

Each unprofiled request reused zero tokens and requested one generated token. Prefill tokens/s divides prompt length by mean backend-prefill time. First data measures request start through first output. Prefill ranges were 32.59–33.14 s / 198.82–199.13 s at 60K / 200K. The same-date stock workers averaged 33.35 s / 200.25 s with the same prompts and settings. [Measurement and fresh vector comparisons](benchmarks/results/retained-20261126/REPORT.md).

The expandable layer and kernel tables use the same 961 retained 60K cycles as the main table (2026-11-26). GPU activity crossing a worker boundary is clipped to that boundary; activity-record counts include these fragments. CPU profiling work is excluded. Cycle counts are not output-token counts.

<details>
<summary>Current compiled 60K decoder-layer detail: projection and remaining-work timings</summary>

Finish each row's layer, including its MLP, before moving to the next row. Each layer has four projections. Gate and up are one joint GEMM; there is no separately measured gate/up split. The remaining-work column combines normalization, mixing and other operations between those projections.

| Layer | Type | All layer work ms | Input projection ms | Output projection ms | Gate/up projection ms | Down projection ms | Other work ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | GDN | 0.4245 | 0.0766 | 0.0366 | 0.1718 | 0.0796 | 0.0598 |
| 1 | GDN | 0.4401 | 0.0818 | 0.0383 | 0.1807 | 0.0854 | 0.0539 |
| 2 | GDN | 0.4500 | 0.0846 | 0.0383 | 0.1817 | 0.0875 | 0.0579 |
| 3 | Attention | 0.7006 | 0.0729 | 0.0336 | 0.1643 | 0.0797 | 0.3502 |
| 4 | GDN | 0.4266 | 0.0825 | 0.0367 | 0.1718 | 0.0806 | 0.0551 |
| 5 | GDN | 0.4393 | 0.0815 | 0.0386 | 0.1812 | 0.0834 | 0.0546 |
| 6 | GDN | 0.4467 | 0.0836 | 0.0377 | 0.1808 | 0.0868 | 0.0577 |
| 7 | Attention | 0.6978 | 0.0727 | 0.0332 | 0.1642 | 0.0797 | 0.3481 |
| 8 | GDN | 0.4273 | 0.0826 | 0.0366 | 0.1720 | 0.0804 | 0.0557 |
| 9 | GDN | 0.4420 | 0.0814 | 0.0383 | 0.1816 | 0.0860 | 0.0547 |
| 10 | GDN | 0.4474 | 0.0843 | 0.0382 | 0.1800 | 0.0876 | 0.0573 |
| 11 | Attention | 0.6951 | 0.0724 | 0.0333 | 0.1641 | 0.0797 | 0.3455 |
| 12 | GDN | 0.4242 | 0.0813 | 0.0366 | 0.1720 | 0.0796 | 0.0546 |
| 13 | GDN | 0.4407 | 0.0814 | 0.0387 | 0.1800 | 0.0864 | 0.0542 |
| 14 | GDN | 0.4462 | 0.0857 | 0.0377 | 0.1805 | 0.0866 | 0.0557 |
| 15 | Attention | 0.6952 | 0.0724 | 0.0335 | 0.1640 | 0.0800 | 0.3453 |
| 16 | GDN | 0.4254 | 0.0819 | 0.0366 | 0.1719 | 0.0799 | 0.0550 |
| 17 | GDN | 0.4351 | 0.0811 | 0.0384 | 0.1768 | 0.0843 | 0.0545 |
| 18 | GDN | 0.4448 | 0.0835 | 0.0385 | 0.1804 | 0.0857 | 0.0567 |
| 19 | Attention | 0.6948 | 0.0723 | 0.0333 | 0.1641 | 0.0794 | 0.3457 |
| 20 | GDN | 0.4268 | 0.0816 | 0.0368 | 0.1717 | 0.0798 | 0.0570 |
| 21 | GDN | 0.4409 | 0.0813 | 0.0385 | 0.1803 | 0.0868 | 0.0540 |
| 22 | GDN | 0.4451 | 0.0828 | 0.0380 | 0.1811 | 0.0865 | 0.0567 |
| 23 | Attention | 0.6960 | 0.0727 | 0.0335 | 0.1640 | 0.0799 | 0.3459 |
| 24 | GDN | 0.4260 | 0.0818 | 0.0367 | 0.1722 | 0.0798 | 0.0555 |
| 25 | GDN | 0.4367 | 0.0815 | 0.0376 | 0.1774 | 0.0844 | 0.0558 |
| 26 | GDN | 0.4451 | 0.0829 | 0.0380 | 0.1809 | 0.0864 | 0.0569 |
| 27 | Attention | 0.6967 | 0.0719 | 0.0332 | 0.1637 | 0.0802 | 0.3477 |
| 28 | GDN | 0.4265 | 0.0823 | 0.0368 | 0.1722 | 0.0800 | 0.0551 |
| 29 | GDN | 0.4426 | 0.0827 | 0.0386 | 0.1796 | 0.0869 | 0.0547 |
| 30 | GDN | 0.4462 | 0.0844 | 0.0381 | 0.1800 | 0.0869 | 0.0568 |
| 31 | Attention | 0.6956 | 0.0721 | 0.0332 | 0.1646 | 0.0794 | 0.3463 |
| 32 | GDN | 0.4262 | 0.0818 | 0.0366 | 0.1716 | 0.0801 | 0.0560 |
| 33 | GDN | 0.4363 | 0.0816 | 0.0385 | 0.1769 | 0.0838 | 0.0555 |
| 34 | GDN | 0.4482 | 0.0825 | 0.0379 | 0.1816 | 0.0869 | 0.0593 |
| 35 | Attention | 0.6970 | 0.0719 | 0.0332 | 0.1643 | 0.0799 | 0.3477 |
| 36 | GDN | 0.4268 | 0.0820 | 0.0367 | 0.1723 | 0.0806 | 0.0552 |
| 37 | GDN | 0.4420 | 0.0828 | 0.0388 | 0.1797 | 0.0865 | 0.0542 |
| 38 | GDN | 0.4451 | 0.0831 | 0.0381 | 0.1805 | 0.0869 | 0.0565 |
| 39 | Attention | 0.6971 | 0.0722 | 0.0331 | 0.1641 | 0.0798 | 0.3479 |
| 40 | GDN | 0.4276 | 0.0827 | 0.0368 | 0.1721 | 0.0803 | 0.0556 |
| 41 | GDN | 0.4360 | 0.0814 | 0.0385 | 0.1768 | 0.0837 | 0.0556 |
| 42 | GDN | 0.4459 | 0.0826 | 0.0382 | 0.1813 | 0.0866 | 0.0571 |
| 43 | Attention | 0.6962 | 0.0718 | 0.0331 | 0.1642 | 0.0794 | 0.3477 |
| 44 | GDN | 0.4266 | 0.0821 | 0.0367 | 0.1726 | 0.0798 | 0.0555 |
| 45 | GDN | 0.4406 | 0.0823 | 0.0381 | 0.1799 | 0.0853 | 0.0549 |
| 46 | GDN | 0.4446 | 0.0824 | 0.0383 | 0.1807 | 0.0857 | 0.0575 |
| 47 | Attention | 0.6986 | 0.0721 | 0.0334 | 0.1645 | 0.0798 | 0.3487 |
| 48 | GDN | 0.4291 | 0.0828 | 0.0368 | 0.1720 | 0.0797 | 0.0577 |
| 49 | GDN | 0.4416 | 0.0821 | 0.0380 | 0.1801 | 0.0857 | 0.0557 |
| 50 | GDN | 0.4474 | 0.0847 | 0.0382 | 0.1802 | 0.0867 | 0.0576 |
| 51 | Attention | 0.6969 | 0.0724 | 0.0334 | 0.1641 | 0.0795 | 0.3474 |
| 52 | GDN | 0.4257 | 0.0820 | 0.0367 | 0.1720 | 0.0794 | 0.0555 |
| 53 | GDN | 0.4426 | 0.0826 | 0.0388 | 0.1799 | 0.0867 | 0.0546 |
| 54 | GDN | 0.4470 | 0.0850 | 0.0380 | 0.1795 | 0.0873 | 0.0573 |
| 55 | Attention | 0.6978 | 0.0725 | 0.0332 | 0.1642 | 0.0801 | 0.3478 |
| 56 | GDN | 0.4277 | 0.0827 | 0.0368 | 0.1721 | 0.0798 | 0.0563 |
| 57 | GDN | 0.4326 | 0.0816 | 0.0376 | 0.1772 | 0.0810 | 0.0553 |
| 58 | GDN | 0.4185 | 0.0803 | 0.0356 | 0.1682 | 0.0803 | 0.0541 |
| 59 | Attention | 0.6413 | 0.0668 | 0.0305 | 0.1513 | 0.0739 | 0.3187 |
| 60 | GDN | 0.3943 | 0.0760 | 0.0339 | 0.1585 | 0.0738 | 0.0521 |
| 61 | GDN | 0.4049 | 0.0749 | 0.0356 | 0.1652 | 0.0787 | 0.0505 |
| 62 | GDN | 0.4136 | 0.0776 | 0.0350 | 0.1665 | 0.0800 | 0.0545 |
| 63 | Attention | 0.6431 | 0.0670 | 0.0308 | 0.1511 | 0.0734 | 0.3208 |

</details>

<details>
<summary>Current 60K GPU activity, grouped by stage</summary>

| Stage / compiled kernel | Activity records in 961 retained cycles | Current GPU ms per retained 60K cycle |
| --- | ---: | ---: |
| Drafter / `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x32_MI16x16x1_SN_LDSB1_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_CD1_1_CLR0_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB1_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA1201_ICIW0_IU1_K1_LDSTI0_LBSPPA128_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB0_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB2_ONLL1_PAP0_PGL0_PGR1_PLR1_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1.kd` | 961 | 0.006358 |
| Drafter / `__amd_rocclr_copyBuffer.kd` | 1920 | 0.004434 |
| Drafter / `__amd_rocclr_fillBufferAligned.kd` | 3844 | 0.004822 |
| Drafter / `_cache_draft_logits_kernel.kd` | 961 | 0.001774 |
| Drafter / `_draft_head_int2.kd` | 961 | 0.803009 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_17408_EVEN_K_1_GRID_MN_40_cache_modifier_NONE.kd` | 4805 | 0.733325 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_25600_EVEN_K_1_GRID_MN_40_cache_modifier_NONE.kd` | 960 | 0.230003 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_40_cache_modifier_NONE.kd` | 4805 | 0.197539 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_5120_EVEN_K_1_GRID_MN_272_cache_modifier_NONE.kd` | 4805 | 1.443517 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_5120_EVEN_K_1_GRID_MN_48_cache_modifier_NONE.kd` | 4805 | 0.269269 |
| Drafter / `_prepare_dflash_inputs_kernel.kd` | 961 | 0.009905 |
| Drafter / `_project.kd` | 960 | 0.198036 |
| Drafter / `_rerank_exact.kd` | 961 | 0.006638 |
| Drafter / `_walk.kd` | 961 | 0.007992 |
| Drafter / `apply.kd` | 961 | 0.002575 |
| Drafter / `kernel_unified_attention.kd` | 4805 | 0.491161 |
| Drafter / `reshape_and_cache_kernel_flash.kd` | 9607 | 0.022876 |
| Drafter / `scan.kd` | 960 | 0.014156 |
| Drafter / `triton_per_fused_4.kd` | 961 | 0.001468 |
| Drafter / `triton_per_fused_8.kd` | 3844 | 0.005096 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_preshuffle_gemm_squeeze_view_0.kd` | 4805 | 0.007156 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_preshuffle_gemm_squeeze_view_2.kd` | 961 | 0.001416 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_preshuffle_gemm_squeeze_view_4.kd` | 8649 | 0.010873 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_view_0.kd` | 960 | 0.013012 |
| Drafter / `triton_poi_fused_0.kd` | 961 | 0.001809 |
| Drafter / `triton_poi_fused_5.kd` | 961 | 0.001666 |
| Drafter / `triton_poi_fused_9.kd` | 3844 | 0.006645 |
| Drafter / `triton_poi_fused__to_copy_clamp_div_preshuffle_gemm_squeeze_view_1.kd` | 4805 | 0.008493 |
| Drafter / `triton_poi_fused__to_copy_clamp_div_preshuffle_gemm_squeeze_view_3.kd` | 961 | 0.001502 |
| Drafter / `triton_poi_fused__to_copy_clamp_div_preshuffle_gemm_squeeze_view_5.kd` | 8649 | 0.011247 |
| Drafter / `triton_poi_fused_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_3.kd` | 8649 | 0.018662 |
| Drafter / `triton_poi_fused_add_arange_bitwise_and_constant_pad_nd_ge_mul_rms_norm_select_slice_unsqueeze_view_1.kd` | 961 | 0.002269 |
| Drafter / `triton_poi_fused_add_permute_unsqueeze_view_2.kd` | 961 | 0.001285 |
| Drafter / `triton_poi_fused_cat_expand_index_mul_slice_unsqueeze_view_1.kd` | 961 | 0.001903 |
| Drafter / `triton_red_fused__to_copy_abs_clamp_div_max_mul_preshuffle_gemm_silu_slice_squeeze_view_6.kd` | 4805 | 0.015470 |
| Drafter / `triton_red_fused__to_copy_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_w4_gemm_2.kd` | 4805 | 0.028164 |
| Drafter / `triton_red_fused__to_copy_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_w4_gemm_7.kd` | 3844 | 0.023321 |
| Drafter / `triton_red_fused__to_copy_embedding_mul_rms_norm_w4_gemm_0.kd` | 961 | 0.005556 |
| Drafter / `triton_red_fused_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_7.kd` | 961 | 0.005762 |
| Drafter / `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) [clone .kd]` | 1921 | 0.008596 |
| Drafter / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 961 | 0.002477 |
| Drafter / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 961 | 0.002112 |
| Drafter / `void at::native::bitonicSortKVInPlace<2, -1, 16, 16, c10::BFloat16, long, at::native::GTOp<c10::BFloat16, true>, unsigned int>(at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, at::native::GTOp<c10::BFloat16, true>) [clone .kd]` | 961 | 0.002584 |
| Drafter / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 961 | 0.004502 |
| Drafter / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 961 | 0.002513 |
| Drafter / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 960 | 0.003532 |
| Drafter / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 961 | 0.001961 |
| Drafter / `void at::native::mbtopk::computeBlockDigitCounts<c10::BFloat16, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 1922 | 0.038374 |
| Drafter / `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 3844 | 0.010695 |
| Drafter / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, c10::BFloat16>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 1922 | 0.017686 |
| Drafter / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, float>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, float*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 3844 | 0.014745 |
| Drafter / `void at::native::mbtopk::fill<unsigned int, unsigned int>(unsigned int*, unsigned int, unsigned int) [clone .kd]` | 1922 | 0.002008 |
| Drafter / `void at::native::mbtopk::gatherTopK<c10::BFloat16, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 961 | 0.020008 |
| Drafter / `void at::native::mbtopk::gatherTopK<float, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, float*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 961 | 0.004955 |
| Drafter / `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4>) [clone .kd]` | 961 | 0.002534 |
| Drafter / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul>) [clone .kd]` | 961 | 0.001171 |
| Drafter / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) [clone .kd]` | 961 | 0.001548 |
| Drafter / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>) [clone .kd]` | 1922 | 0.002505 |
| Drafter / `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>) [clone .kd]` | 1922 | 0.002872 |
| Drafter / `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool) [clone .kd]` | 961 | 0.001556 |
| Drafter / `void at::native::warpMergeSortKVInPlace<2, -1, 128, 16, float, long, at::native::GTOp<float, true>, unsigned int, 32>(at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, at::native::GTOp<float, true>, float) [clone .kd]` | 961 | 0.004298 |
| Drafter / `void r4d_gemm_w4a16_nt_m64_kernel<1, 1, false>(unsigned short const*, unsigned int const*, unsigned int const*, __hip_bfloat16*, int, int, int, int, int) [clone .kd]` | 961 | 0.003478 |
| Drafter / `void r4d_gemm_w4a16_nt_m64_kernel<1, 1, true>(unsigned short const*, unsigned int const*, unsigned int const*, __hip_bfloat16*, int, int, int, int, int) [clone .kd]` | 9610 | 0.075541 |
| Drafter / `void vllm::rms_norm_kernel<c10::BFloat16, 8, 2, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, long, long, long, c10::BFloat16 const*, long, float, int, int) [clone .kd]` | 960 | 0.002749 |
| Drafter / `void vllm::rms_norm_kernel<c10::BFloat16, 8, 4, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, long, long, long, c10::BFloat16 const*, long, float, int, int) [clone .kd]` | 961 | 0.002673 |
| Drafter / `void vllm::rotary_embedding_kernel<c10::BFloat16, c10::BFloat16, true>(long const*, c10::BFloat16*, c10::BFloat16*, c10::BFloat16 const*, int, long, long, long, int, int, int, long, bool) [clone .kd]` | 960 | 0.002344 |
| Embedding + first input normalization / `triton_poi_fused__to_copy_embedding_0.kd` | 961 | 0.004861 |
| Embedding + first input normalization / `void norm_quant<false, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned char*, float*, unsigned short*, long, long, float) [clone .kd]` | 961 | 0.006792 |
| GDN layout/copies and buffer initialization / `triton_poi_fused_add_1.kd` | 2807 | 0.004226 |
| GDN layout/copies and buffer initialization / `triton_poi_fused_add_2.kd` | 1922 | 0.002898 |
| GDN layout/copies and buffer initialization / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 45848 | 0.088773 |
| GDN layout/copies and buffer initialization / `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>) [clone .kd]` | 45846 | 0.041646 |
| GDN convolution / `_causal_conv1d_update_kernel.kd` | 45849 | 0.180169 |
| GDN recurrence and gates / `stock_gdn_scan_kernel.kd` | 45849 | 1.174561 |
| GDN output gated normalization / `gdn_norm_quant_kernel.kd` | 45844 | 0.146826 |
| Post-attention/GDN residual/normalization / `void norm_quant<true, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned char*, float*, unsigned short*, long, long, float) [clone .kd]` | 61058 | 0.507561 |
| MLP SiLU and gating / `triton_poi_fused_dynamic_per_token_scaled_fp8_quant_mul_silu_slice_0.kd` | 45807 | 0.088842 |
| MLP SiLU and gating / `triton_poi_fused_dynamic_per_token_scaled_fp8_quant_mul_silu_slice_1.kd` | 15224 | 0.034699 |
| MLP down input FP8 quantization / `void vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<c10::BFloat16, c10::Float8_e4m3fn>(c10::Float8_e4m3fn*, float*, c10::BFloat16 const*, float const*, int, long, long) [clone .kd]` | 61032 | 0.254101 |
| GDN input projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 1, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 45878 | 3.935011 |
| GDN output projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 4, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 45846 | 1.799274 |
| MLP gate/up projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 1, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 61705 | 11.056846 |
| MLP down projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 4, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 61146 | 5.266459 |
| Layer input residual/normalization / `void norm_quant<true, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned char*, float*, unsigned short*, long, long, float) [clone .kd]` | 60135 | 0.492187 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_1.kd` | 15224 | 0.020972 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_3.kd` | 15224 | 0.033902 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_4.kd` | 15224 | 0.021142 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_arange_bitwise_and_eq_index_lt_remainder_select_split_where_2.kd` | 15224 | 0.027242 |
| Attention Q/K normalization, RoPE and layout / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 15224 | 0.028232 |
| Attention Q/K normalization, RoPE and layout / `void stock_m1_gemma_norm<false, 256, 32>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, long, long, float, float*) [clone .kd]` | 15224 | 0.045720 |
| Attention Q/K normalization, RoPE and layout / `void stock_m1_gemma_norm<false, 256, 64>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, long, long, float, float*) [clone .kd]` | 15224 | 0.035912 |
| Attention KV write / `reshape_and_cache_kernel_flash.kd` | 15224 | 0.034014 |
| Attention decode / `void qwen_stock_m1_shared_decode<4, 16, 256, 6, 16, 0, 3430931>(R4DArgs, int) [clone .kd]` | 15224 | 4.702220 |
| Attention split-KV merge / `void qwen_stock_m1_shared_merge<256, 4, 0>(R4DArgs, int, int) [clone .kd]` | 15224 | 0.130599 |
| Attention output gating / `triton_poi_fused_dynamic_per_token_scaled_fp8_quant_mul_sigmoid_view_0.kd` | 15224 | 0.025852 |
| Attention output activation FP8 quantization / `void vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<c10::BFloat16, c10::Float8_e4m3fn>(c10::Float8_e4m3fn*, float*, c10::BFloat16 const*, float const*, int, long, long) [clone .kd]` | 15224 | 0.041568 |
| Attention input projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 1, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 15224 | 1.145936 |
| Attention output projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 4, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 15224 | 0.527553 |
| Final normalization/layout / `__amd_rocclr_copyBuffer.kd` | 5310 | 0.006621 |
| Final normalization/layout / `void stock_m1_gemma_norm<true, 5120, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, long, long, float, float*) [clone .kd]` | 885 | 0.004605 |
| Other GPU bookkeeping / `__amd_rocclr_copyBuffer.kd` | 37480 | 0.074860 |
| Other GPU bookkeeping / `__amd_rocclr_fillBufferAligned.kd` | 960 | 0.002367 |
| Other GPU bookkeeping / `_apply_write_kernel.kd` | 3 | 0.000016 |
| Other GPU bookkeeping / `_combine_sampled_and_draft_tokens_kernel.kd` | 961 | 0.002958 |
| Other GPU bookkeeping / `_compute_local_logits_stats_kernel.kd` | 961 | 0.028530 |
| Other GPU bookkeeping / `_compute_slot_mappings_kernel.kd` | 961 | 0.002866 |
| Other GPU bookkeeping / `_expand_idx_mapping_kernel.kd` | 961 | 0.001865 |
| Other GPU bookkeeping / `_gather_block_tables_kernel.kd` | 961 | 0.004954 |
| Other GPU bookkeeping / `_get_num_sampled_and_rejected_kernel.kd` | 960 | 0.002668 |
| Other GPU bookkeeping / `_insert_resampled_kernel.kd` | 960 | 0.003329 |
| Other GPU bookkeeping / `_post_update_kernel.kd` | 961 | 0.005221 |
| Other GPU bookkeeping / `_prepare_pos_seq_lens_kernel.kd` | 961 | 0.002037 |
| Other GPU bookkeeping / `_prepare_rope_positions_kernel.kd` | 961 | 0.002937 |
| Other GPU bookkeeping / `_rejection_kernel.kd` | 960 | 0.006899 |
| Other GPU bookkeeping / `_resample_kernel.kd` | 961 | 0.013974 |
| Other GPU bookkeeping / `_scatter_num_accepted_kernel.kd` | 960 | 0.001873 |
| Other GPU bookkeeping / `_zero_kv_blocks_kernel.kd` | 1 | 0.000070 |
| Other GPU bookkeeping / `postprocess_mamba_fused_kernel.kd` | 960 | 0.003699 |
| Other GPU bookkeeping / `precopy_mamba_align_fused_kernel.kd` | 961 | 0.003622 |
| Other GPU bookkeeping / `preprocess_mamba_align_fused_kernel.kd` | 961 | 0.002624 |
| Other GPU bookkeeping / `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}>(int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}, function_traits<at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}>::result_type*) [clone .kd]` | 5766 | 0.008427 |
| Other GPU bookkeeping / `void (anonymous namespace)::softmax_warp_forward<float, float, float, 6, false, false, 32>(float*, float const*, int, int, int, bool const*, int, bool) [clone .kd]` | 960 | 0.002268 |
| Other GPU bookkeeping / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 6726 | 0.012810 |
| Other GPU bookkeeping / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 961 | 0.002303 |
| Other GPU bookkeeping / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 6727 | 0.013697 |
| Other GPU bookkeeping / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<int> const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<int> const&)::{lambda(int, bool)#1}) [clone .kd]` | 5766 | 0.014420 |
| Other GPU bookkeeping / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<float> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<float> const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<float> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<float> const&)::{lambda(int, bool)#1}) [clone .kd]` | 2883 | 0.018000 |
| Other GPU bookkeeping / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 4808 | 0.015770 |
| Other GPU bookkeeping / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 1923 | 0.005918 |
| Other GPU bookkeeping / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 961 | 0.002912 |
| Other GPU bookkeeping / `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 3841 | 0.057985 |
| Other GPU bookkeeping / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, float>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, float*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 3841 | 0.036298 |
| Other GPU bookkeeping / `void at::native::mbtopk::fill<unsigned int, unsigned int>(unsigned int*, unsigned int, unsigned int) [clone .kd]` | 960 | 0.001530 |
| Other GPU bookkeeping / `void at::native::mbtopk::gatherTopK<float, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, float*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 960 | 0.025085 |
| Other GPU bookkeeping / `void at::native::tensor_kernel_scan_innermost_dim<float, std::plus<float> >(float*, float const*, unsigned int, unsigned int, unsigned int, float, std::plus<float>) [clone .kd]` | 961 | 0.002402 |
| Other GPU bookkeeping / `void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>, 4, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast) [clone .kd]` | 961 | 0.001706 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<16, at::native::BinaryFunctor<bool, bool, bool, at::native::BitwiseOrFunctor<bool> >, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<bool, bool, bool, at::native::BitwiseOrFunctor<bool> >, std::array<char*, 3ul>) [clone .kd]` | 961 | 0.002380 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<16, at::native::FillFunctor<bool>, std::array<char*, 1ul> >(int, at::native::FillFunctor<bool>, std::array<char*, 1ul>) [clone .kd]` | 961 | 0.001872 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<16, at::native::bitwise_not_kernel_cuda(at::TensorIteratorBase&)::{lambda(bool)#1}, std::array<char*, 2ul> >(int, at::native::bitwise_not_kernel_cuda(at::TensorIteratorBase&)::{lambda(bool)#1}, std::array<char*, 2ul>) [clone .kd]` | 961 | 0.002222 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}, std::array<char*, 2ul>) [clone .kd]` | 5766 | 0.009829 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}, std::array<char*, 2ul>) [clone .kd]` | 960 | 0.001734 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}, std::array<char*, 3ul> >(int, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}, std::array<char*, 3ul>) [clone .kd]` | 962 | 0.009999 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool, float, float)#1}, std::array<char*, 4ul> >(int, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool, float, float)#1}, std::array<char*, 4ul>) [clone .kd]` | 1921 | 0.004487 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 2ul> >(int, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 2ul>) [clone .kd]` | 5766 | 0.010802 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, 2ul>) [clone .kd]` | 6727 | 0.011677 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul>) [clone .kd]` | 961 | 0.001811 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>) [clone .kd]` | 960 | 0.001837 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>) [clone .kd]` | 1923 | 0.003145 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int, at::native::FillFunctor<int>, std::array<char*, 1ul>) [clone .kd]` | 961 | 0.001577 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>) [clone .kd]` | 963 | 0.007779 |
| Other GPU bookkeeping / `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool) [clone .kd]` | 939 | 0.002197 |
| Other GPU bookkeeping / `void at::native::warpMergeSortKVInPlace<2, -1, 128, 16, float, long, at::native::GTOp<float, true>, unsigned int, 32>(at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, at::native::GTOp<float, true>, float) [clone .kd]` | 960 | 0.004570 |
| Target head (global512) / `__amd_rocclr_fillBufferAligned.kd` | 963 | 0.002443 |
| Target head (global512) / `_draft_head_int2.kd` | 965 | 0.761730 |
| Target head (global512) / `_rerank_exact.kd` | 969 | 0.050881 |
| Target head (global512) / `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) [clone .kd]` | 965 | 0.004553 |
| Target head (global512) / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 973 | 0.003850 |
| Target head (global512) / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 984 | 0.002819 |
| Target head (global512) / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 973 | 0.005280 |
| Target head (global512) / `void at::native::mbtopk::computeBlockDigitCounts<c10::BFloat16, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 1942 | 0.031747 |
| Target head (global512) / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, c10::BFloat16>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 1947 | 0.020059 |
| Target head (global512) / `void at::native::mbtopk::fill<unsigned int, unsigned int>(unsigned int*, unsigned int, unsigned int) [clone .kd]` | 961 | 0.001541 |
| Target head (global512) / `void at::native::mbtopk::gatherTopK<c10::BFloat16, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 979 | 0.028133 |
| Target head (global512) / `void at::native::radixSortKVInPlace<2, -1, 128, 8, c10::BFloat16, long, unsigned int>(at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, bool) [clone .kd]` | 986 | 0.006044 |
| Target head (global512) / `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4>) [clone .kd]` | 977 | 0.003851 |
| Target head (global512) / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) [clone .kd]` | 975 | 0.001699 |
| Target head (global512) / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>) [clone .kd]` | 970 | 0.002220 |
| Target head (global512) / `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>) [clone .kd]` | 1935 | 0.003934 |

</details>

## Global-512 target-head

Global-512 is the serving default. The 2026-11-26 isolated candidate comparison used identical hidden inputs for every head method, collected from two natural completions starting with the retained 60K Pi prefix for coding (60,208 input, 7,276 output) and reasoning (67,631 input, 6,029 output). Temperature was 1, top-p 0.95 and top-k 40.

| Target path | Median M8 head time | Same top-1 token | Complete reference top-20 retained | Complete reference top-40 retained |
|---|---:|---:|---:|---:|
| Global INT2 top-256 + BF16 rerank | 1.037 ms | 12,141/12,141 (100%) | 12,127/12,141 (99.8847%) | 11,955/12,141 (98.4680%) |
| Global INT2 top-512 + BF16 rerank (default) | 1.059 ms | 12,141/12,141 (100%) | 12,137/12,141 (99.9671%) | 12,120/12,141 (99.8270%) |
| Full BF16 reference | 4.068 ms | 12,141/12,141 (100%) | 12,141/12,141 (100%) | 12,141/12,141 (100%) |

The comparison covers **12,141 prediction rows**, including prefill and rejected speculative rows, not 12,141 generated tokens. Timing uses 48 eight-row hidden inputs with five randomized-order repetitions: 240 measurements per method. These isolated head timings include native dispatch gaps and exclude comparison/reporting; they do not measure whole-round time or tok/s.

Incomplete top-40 retention occurred in 186 rows with Global-256 and 21 with Global-512; the added median head time was 0.022 ms. Retention includes cutoff ties and does not establish score equality, ordering or identical sampling probabilities. Both shortlists remain approximate; the full BF16 head is the reference for this comparison, not an independently proved model. The drafter is unchanged.

Global-512 reranked values differed from the full-head reference in 39/6,216,192 retained scores (maximum absolute difference 0.0625); the diagnostic filtered probabilities differed in 14/12,141 rows. Candidate recall and retained-score fidelity are separate checks.

[Methodology and limits](docs/HEAD_CANDIDATE_DEPTH.md) · [Numeric results](benchmarks/results/retained-20261126/results.json) · [Earlier Global-256 study](docs/VERIFY_HEAD_GLOBAL_TOPK.md).

The 2026-10-09 Global-512 control took 1.168 ms. The two dates use different input trajectories. [October release evidence](benchmarks/results/head-candidate-depth-20261009.json) · [November comparison](benchmarks/results/retained-20261126/REPORT.md#head-latency-and-numerical-results).

## Benchmarks

The 2026-11-26 isolated candidate uses Global-512 and FP8 DFlash2 with proposal temperature 0.55. The same 60K prefix and task sequence were used for the October control. Later prompts incorporate each run's earlier output. [October release control](benchmarks/results/pi-coding-json-compaction.json) · [November comparison](benchmarks/results/retained-20261126/REPORT.md#coding-and-chained-requests). Five requests are chained in one context: code, prose about code measurement, JSON, thinking/prose and checkpoint generation. Code and JSON disable thinking; both prose requests enable it. All requests stop naturally. Generation uses temperature 1, top-p 0.95, top-k 40 and seed 0; compaction uses temperature 0.3. The checkpoint request forces a snapshot-tail flush. Public results contain aggregate counters and source identities. The shared suite retains sealed continuation tokens privately for resumability. Runner: [benchmark_pi_coding_contexts.py --suite](experiments/radiance-public/benchmark_pi_coding_contexts.py).

| Stage | Thinking | Prompt tokens | Generated tokens | Classified output | First data | Mean round | Post-first | Peak 3s | Acceptance |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Coding task | off | 60,208 | 7,402 | 6,859 code; 542 prose; 0 separately observed reasoning | 0.79 s | 40.54 ms | 109.09 tok/s | 154.85 tok/s | 48.88% |
| Prose about code measurement | on | 67,751 | 2,194 | 2,193 prose; 0 separately observed reasoning | 0.55 s | 40.95 ms | 79.76 tok/s | 140.84 tok/s | 32.36% |
| JSON task | off | 70,064 | 2,305 | 2,304 JSON; invalid JSON | 0.25 s | 41.57 ms | 97.69 tok/s | 116.69 tok/s | 43.69% |
| Thinking/prose task | on | 72,516 | 4,648 | 4,647 prose; 0 separately observed reasoning | 0.79 s | 41.27 ms | 64.41 tok/s | 87.81 tok/s | 23.67% |
| Compaction checkpoint | off | 77,350 | 4,049 | 4,049 checkpoint tokens; completion marker valid; required headings valid | 0.31 s | 41.70 ms | 69.03 tok/s | 83.62 tok/s | 26.83% |

Cached/total prompt tokens: Coding task: 59,328/60,208, Prose about code measurement: 67,610/67,751, JSON task: 69,945/70,064, Thinking/prose task: 72,369/72,516, Compaction checkpoint: 77,163/77,350.

The phase counters retokenize classified text, so their totals can differ from the backend's emitted-token count. `phase_token_counts_cover_output=false` for Coding task, Prose about code measurement, JSON task, Thinking/prose task, Compaction checkpoint. No separate reasoning channel was exposed for Prose about code measurement, Thinking/prose task; those streams are reported as prose, not relabelled as reasoning. The compaction row measures checkpoint generation with a requested cache flush, not a full Pi transcript commit or old-snapshot retirement. The checkpoint format passed its heading and completion checks. `peak_3s_tokens_per_second` is the maximum completed three-second sliding-window rate after first data, never a single-frame burst.

2026-11-26 rerun status: `complete_with_validation_failure`. [Numeric results and tested identity](benchmarks/results/retained-20261126/results.json).

This is the same natural-stop coding task run independently at empty, 60K and 200K input context. Thinking is disabled, EOS remains enabled, and each arm uses temperature 1, top-p 0.95 and top-k 40. The non-empty arms use operator-supplied token-prefix fixtures; only their hashes are published. The three-second peak is the maximum completed sliding-window rate, not a single-frame burst. Runner: [benchmark_pi_coding_contexts.py](experiments/radiance-public/benchmark_pi_coding_contexts.py).

These are 2026-11-26 isolated candidate measurements. All three arms generated the same token sequences from warm and cold starts within this build. The earlier November candidate used the same prompts and sampling settings. October used a different drafter proposal policy. [Paired throughput comparison](benchmarks/results/retained-20261126/REPORT.md#coding-and-chained-requests).

| Context | Prompt tokens | Generated tokens | Mean round | Post-first | Peak 3s | Acceptance | Round events (total; speculative/expected; timed) | Output minimum |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0K | 203 | 6,248 | 36.66 ms | 141.89 tok/s | 202.88 tok/s | 60.03% | 1,202 total; 1201/1201 speculative; 1,201 timed; captured | pass |
| 60K | 60,208 | 7,402 | 40.54 ms | 109.09 tok/s | 154.85 tok/s | 48.88% | 1,675 total; 1674/1674 speculative; 1,674 timed; captured | pass |
| 200K | 200,208 | 5,074 | 49.99 ms | 102.84 tok/s | 131.47 tok/s | 59.08% | 989 total; 988/988 speculative; 988 timed; captured | pass |

Report status: `complete_with_validation_failure`. [Numeric results and every round](benchmarks/results/retained-20261126/round-records.json). The round records appear under `contexts.<context>.round_capture.records`. The target is 5,000 output tokens, with shorter natural completions reported explicitly.

These results use [the shared benchmark suite](docs/BENCHMARK_SUITE.md): `benchmark_pi_coding_contexts.py --suite` reuses each context's predetermined unprofiled control for its coding row and complete histogram, and continues the same 60K output through prose, JSON, thinking and compaction. It keeps both controls around each stage trace for the residual calculation. The 60K coding row above and the chained coding row are the same measured request.

### Complete per-round capture

The context benchmark now retains every content-free scheduler event for each arm, including an unmeasured first event. The full numeric records are stored under `contexts.<context>.round_capture.records` in the result JSON; this table is a coverage check rather than another latency aggregate. A count mismatch is a validation failure. Expected rounds come from the speculative-round counter; the first prefill event is retained in logged events but excluded from that counter.

| Context | Logged events | Speculative rounds | Expected speculative rounds | Timed round values | Untimed events | Missing round numbers | Capture status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0K | 1,202 | 1201 | 1201 | 1,201 | 1 | none | captured |
| 60K | 1,675 | 1674 | 1674 | 1,674 | 1 | none | captured |
| 200K | 989 | 988 | 988 | 988 | 1 | none | captured |

The 2026-11-26 public round records cover all expected speculative rounds. The separate diagnostic recorder reported event loss for 60K, 200K, chained coding, prose about code, JSON, thinking/prose, checkpoint generation. Archive rotation did not settle for 200K, prose about code, JSON. [Timing and diagnostic coverage](benchmarks/results/retained-20261126/REPORT.md#round-latency).



The histogram below is generated from the complete per-round records of the [shared-suite coding controls](benchmarks/results/retained-20261126/round-records.json). Every measured `round_ms` value appears in exactly one bin; untimed events are reported separately. The shared suite uses this same predetermined clean control for the coding row and histogram; the stage residual matches only structurally admitted M8 cycles from both controls, while this histogram retains every measured round.
Half-millisecond display bins resolve the current dense clusters at 35–42 and 49–53 ms. These bins are recomputed from the original records; the captured bins, round counts and measurement identity are unchanged.

| Round time | 0K arm | 60K arm | 200K arm |
|---|---:|---:|---:|
| `<35` | 2 (0.2%) | 0 (0.0%) | 0 (0.0%) |
| `35–35.5` | 1 (0.1%) | 0 (0.0%) | 0 (0.0%) |
| `35.5–36` | 7 (0.6%) | 0 (0.0%) | 0 (0.0%) |
| `36–36.5` | 294 (24.5%) | 0 (0.0%) | 0 (0.0%) |
| `36.5–37` | 838 (69.8%) | 0 (0.0%) | 0 (0.0%) |
| `37–37.5` | 49 (4.1%) | 0 (0.0%) | 0 (0.0%) |
| `37.5–38` | 8 (0.7%) | 0 (0.0%) | 0 (0.0%) |
| `38–38.5` | 1 (0.1%) | 0 (0.0%) | 0 (0.0%) |
| `38.5–39` | 1 (0.1%) | 0 (0.0%) | 0 (0.0%) |
| `39–39.5` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `39.5–40` | 0 (0.0%) | 2 (0.1%) | 0 (0.0%) |
| `40–40.5` | 0 (0.0%) | 814 (48.6%) | 0 (0.0%) |
| `40.5–41` | 0 (0.0%) | 825 (49.3%) | 0 (0.0%) |
| `41–41.5` | 0 (0.0%) | 22 (1.3%) | 0 (0.0%) |
| `41.5–42` | 0 (0.0%) | 10 (0.6%) | 0 (0.0%) |
| `42–43` | 0 (0.0%) | 1 (0.1%) | 0 (0.0%) |
| `43–45` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `45–46` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `46–47` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `47–48` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `48–48.5` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `48.5–49` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `49–49.5` | 0 (0.0%) | 0 (0.0%) | 2 (0.2%) |
| `49.5–50` | 0 (0.0%) | 0 (0.0%) | 529 (53.5%) |
| `50–50.5` | 0 (0.0%) | 0 (0.0%) | 438 (44.3%) |
| `50.5–51` | 0 (0.0%) | 0 (0.0%) | 12 (1.2%) |
| `51–51.5` | 0 (0.0%) | 0 (0.0%) | 6 (0.6%) |
| `51.5–52` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
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
| **Timed rounds** | **1,201** | **1,674** | **988** |
| **Untimed events** | **1** | **1** | **1** |
| **Mean / median** | **36.66 / 36.73 ms** | **40.54 / 40.51 ms** | **49.99 / 49.99 ms** |


### Known remaining symptoms and likely causes

**November round latency.** The candidate recorded 0/3,863 timed rounds at least 90 ms. The October profiler investigation recorded separate 591.203 ms and 751.876 ms pauses. Their cause remains unattributed because the original diagnostic feeds were incomplete. [November per-round records](benchmarks/results/retained-20261126/round-records.json) · [October matched controls](benchmarks/results/stage26-control-20261009.json).

**Reasoning throughput could not be isolated.** The prose and thinking-enabled requests exposed no separate reasoning channel. The JSON request reached a natural stop and failed the format check. [Workload evidence](benchmarks/results/retained-20261126/results.json).

**Global-512 remains approximate.** The November candidate matched reference top-1 in 12,141/12,141 rows and retained all reference top-20 candidates in 12,137. Retained-score differences are reported above. [Head evidence](benchmarks/results/retained-20261126/REPORT.md#head-latency-and-numerical-results).

[November aggregate results](benchmarks/results/retained-20261126/results.json) · [October release baseline](benchmarks/results/coherence-current.json) · [Historical technical report](reports/d7-rdna4-2026-09-17/REPORT.md).

<!-- /COHERENCE_CURRENT_RESULTS -->

## Quick start

The Pi launcher expects an R9700 host with the authenticated **327,952-token**
release installed. The host uses rootless Podman, ROCm device access, at least
18 GiB free in `/dev/shm`, and disk space for compiler caches and snapshots.
Obtain the **Qwen3.8-27B-Uncensored-MXFP4-awq** target and matching
**Qwen3.8-27B-DFlash2-FP8** drafter separately.
[Deployment profile and build details](docs/BUILDING.md#327952-token-authenticated-pi-lane)

Install uv, Node.js/npm, Git and jq on the client, and configure SSH access to
the GPU host. Run the launcher from your workspace. It authenticates the
backend and starts it if needed.

```sh
/path/to/vllm-coherence/scripts/pi-remote-qwen-radiance --remote user@gpu-host
```

Resume the latest conversation in that workspace with `--session last`.
Use `--reuse-existing` to open Pi while the launcher checks backend readiness
in the background.

```sh
/path/to/vllm-coherence/scripts/pi-remote-qwen-radiance --remote user@gpu-host --reuse-existing --session last
```

## Pi

The launcher verifies patched Pi 0.84.2, reserves a forwarding port and loads
Coherence's operating prompt and extensions. Sessions live in the workspace's
`.pi/sessions`. Workspace context-file discovery is disabled.

Use `/compact` to generate and validate a checkpoint, `/priority` to control
GPU handovers, and `/backend status` to inspect the connection and backend.
The authenticated launcher's `--reuse-existing` mode opens saved sessions even
while the backend is offline. [Pi setup and recovery](docs/PI.md)

## Cache inspection

From a Pi client:

```sh
qwen-radiance-cache --host gpu-host
qwen-radiance-cache --host gpu-host show CHAT
qwen-radiance-cache --host gpu-host audit
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

The shared benchmark suite reuses one coding control per prefix for natural
throughput and the complete per-round histogram. The 60K control also supplies
the starting turn of the chained workload. Matched controls around the GPU
trace supply the estimated runtime remainder. The numerical qualification
records its own fixture and worker identities.
[Shared suite](docs/BENCHMARK_SUITE.md) ·
[November evidence](benchmarks/results/retained-20261126/REPORT.md) ·
[October baseline](benchmarks/results/coherence-current.json).

## Repository map

```text
tools/                         Runtime packaging, launch tooling and audits
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
benchmarks/                    Published Coherence results and verification fixtures
verification/                  Behavior and evidence coverage inventory
radiance_drafthead.py          Drafter head source used in the combined candidate build
radiance_verifyhead.py         Target head source used in CPU verification
radiance_mxfp4*                 MXFP4 control source for kernel builds and experiments
```

The `qwen_r9700_lab` namespace and wire-format names preserve evidence and snapshot
compatibility. The authenticated launcher checks the serving image, context window
and snapshot compatibility against the pinned release. Research drivers run
separately from serving and require explicit inputs.

The inherited Docker builds, Compose examples and TP2 benchmarks are available
in the [archived Radiance source tree at `3cbf9c1`](https://github.com/Terrydaktal/vllm-coherence/tree/3cbf9c17e8cef4736563c8ddab514b6874fd96ad).
The [benchmark index](benchmarks/README.md) links to earlier run evidence and
the current suite.

## Contributing

Submit a minimized failure or measured optimization with state/output comparisons,
negative controls and precise hardware/profile scope. Read
[CONTRIBUTING.md](CONTRIBUTING.md). Existing upstream PRs remain independent and are
linked from [ATTRIBUTION.md](ATTRIBUTION.md). See [LICENSE](LICENSE) for component terms.
