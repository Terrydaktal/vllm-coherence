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
global-256 target head remains an explicitly approximate speed option.
`--head full-bf16` removes that candidate-shortlist approximation, while retaining
the rest of the quantized backend. See the [results below](#current-numerical-results)
and [verification contract](docs/VERIFICATION.md) for the exact scope.

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
- **Global-256 target-head improvement:** search the complete INT2 score row, then
  rescore 256 candidates using BF16 weights. This removes the old eight-candidates-
  per-tile restriction and improves measured candidate recall; it remains approximate.
- **The Pi workflow below:** a patched agent runtime and extensions, backed by
  persistent chat state, transactional compaction, shared-GPU scheduling and live
  progress and cache telemetry.
- **Reusable verification:** forced-token replay, logical-state comparison,
  first-divergence capture, operator checks, deliberate fault injection and small,
  explicitly scoped machine-checked obligations.

Existing upstream fixes, including DFlash RNG separation and the Triton RoPE
rounding repair, are integrated and credited as backports. The original model,
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
| Honest generation rates | A three-second rolling rate followed by the average, token counts, time to first data and elapsed time. Time spent waiting for another chat is excluded from generation rates; missing reasoning counts are omitted. |
| Tool-call visibility | Token usage continues updating while tool arguments are buffered. `generating edit arguments` and `applying edit` distinguish model work from tool execution, with an execution timer. |
| Backend error details | Engine failures can carry the actual recorded traceback, expandable with `Ctrl+O`; `/backend-error` retrieves the latest diagnostic. |

Unavailable telemetry is reported as unavailable rather than presented as zero
cached tokens or an invented explanation for a wait.

### Compaction, snapshots and history

| Feature | What Coherence adds |
| --- | --- |
| Transactional compaction | Manual and automatic compaction preserve the original transcript unless the checkpoint passes section, completion-marker and finish-status checks. The prompt preserves the existing token prefix for cache reuse, and checkpoint output can use the remaining context capacity. |
| Visible compaction steps | An in-place progress display times prompt preparation, queueing, cache loading, prefill, checkpoint generation, validation, tail flush, commit and cleanup. It shows checkpoint tokens and a three-second generation rate; the completed compaction entry retains total elapsed time. Typed editor input survives compaction. |
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

Current backported compiled M8 versus the aligned pre-backport compiled M8 control, using the **full BF16 target head**: 320 forced decode tokens on the same 60K Pi prefix, plus one prefill prediction. The 10K eager-M1/compiled-M8 study belongs to the preceding alignment revision; it was not rerun after these backports.

| Prediction | Same token set | Same ordering | Mean shared tokens |
| --- | ---: | ---: | ---: |
| Top 1 | 320 / 320 (100.00%) | 320 / 320 (100.00%) | 1.0000 / 1 |
| Top 10 | 320 / 320 (100.00%) | 320 / 320 (100.00%) | 10.0000 / 10 |
| Top 20 | 320 / 320 (100.00%) | 320 / 320 (100.00%) | 20.0000 / 20 |

All 320 full-vocabulary hashes and the prefill prediction matched. These are finite consistency checks, not model task accuracy, an arbitrary-input proof, or certification of approximate global-256 selection.

## Compiled backend stages

Measured on 2026-09-23 using the current compiled, optimized Global-256 serving backend with the pinned-RAM huge-page promotion repair; sampling is temperature 1.0, top-p 0.95 and top-k 40. Context labels are starting prefixes: 0K, the private 60K Pi fixture, and the public synthetic 200K fixture. Each context ran a natural warmup followed by clean control, trace, and clean control; each arm generated 4,981 / 8,213 / 4,233 tokens respectively. Generated-token hashes and accepted-token schedules matched. The stage means retain 917 / 1130 / 774 complete M8 cycles (0K / 60K / 200K), and controls use exactly those same decode indices. Trace setup/export boundaries and incomplete trace inventories are excluded by structure, never by duration; complete native round logs retain all rounds and stalls. GPU activity timestamps supply the stage times; CPU annotations, Python hooks and export time are excluded. No per-stage event probes or forced-token replay are used. The measured tracing slowdown was 8.717 / 5.154 / 5.383 ms per retained round; it is reported separately and is **not charged to row 26**. Row 26 is the clean control mean minus the union of GPU activity intervals. This remains an estimate: tracing can indirectly affect clocks and scheduling. Overlap is counted once in the total. [Capture and source identities](benchmarks/results/compiled-global256-stage-profile-20260923.json) · [controls](benchmarks/results/stage26-control-20260923.json) · [method and uncertainty](docs/STAGE_TIMING.md).

| Stage | Current GPU activity per retained compiled M8 cycle (0K / 60K / 200K; milliseconds unless explicitly marked; 2026-09-23; exact source hashes in capture) | Current correctness evidence | Last relevant code commit / change | What this stage does |
| --- | ---: | --- | --- | --- |
| **1. Drafter** | 6.006 / 6.268 / 6.324 | N/A: no isolated target top-20 prediction · [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2) | [`2a2175a`](https://github.com/Terrydaktal/vllm-coherence/commit/2a2175a2cfe64e666b32df0db3c39f562ca96082): Separated DFlash proposal and target replacement RNG streams so drafting and target verification cannot share random state. | Suggests up to seven tokens for the target model to check. |
| **2. Embedding + first input normalization + FP8 production** | 0.009 / 0.009 / 0.009 | 320 rows + adversarial values: exact FP8/scales/BF16 residual; prefill 1,000 rows per site · [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2) | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Enabled the qualified native-rounding norm/FP8 producer while preserving the BF16 residual interface. | Looks up token vectors, normalizes them and creates FP8 inputs in one kernel, preserving intermediate rounding. |
| **3. Layer input residual/normalization + FP8 production** | 0.485 / 0.484 / 0.491 | 320 rows + adversarial values: exact FP8/scales/BF16 residual; prefill 1,000 rows per site · [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2) | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Enabled qualified TP1 FP8 norm/quant fusion while retaining the required residual and rounding boundaries. | Adds the residual, normalizes the result and creates FP8 inputs in one kernel, preserving intermediate rounding. |
| ↳ GDN input activation FP8 quantization | Included in **stage 3** | Exact fused FP8 bytes/scales; see stage 3 · [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2) | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Enabled qualified TP1 FP8 norm/quant fusion while retaining the required residual and rounding boundaries. | Uses the FP8 output produced by the same input-normalization kernel. |
| ↳ Attention input activation FP8 quantization | Included in **stage 3** | Exact fused FP8 bytes/scales; see stage 3 · [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2) | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Enabled qualified TP1 FP8 norm/quant fusion while retaining the required residual and rounding boundaries. | Uses the FP8 output produced by the same input-normalization kernel. |
| **4. GDN input projection** | 4.003 / 3.924 / 3.929 | 320/320; 320/320 · [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2) | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Added guarded wide-N MXFP4 decode dispatch for the 34,816-column projection, with exact GEMM checks. | Projects hidden vectors into the inputs and gates for the recurrent layer. |
| **5. GDN layout/copies and buffer initialization** | 0.312 / 0.321 / 0.321 | State/layout checked with convolution and recurrence · [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2) | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Recovered packed GDN transport and retained-prefix state handling without changing the nine-slot state layout. | Arranges inputs and clears temporary buffers throughout each GDN layer. |
| **6. GDN convolution** | 0.203 / 0.208 / 0.209 | 320/320; 320/320 · [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2) | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Aligned GDN convolution product and accumulation arithmetic across M1/M8 and eager/compiled execution. | Updates recent-token history using the corrected multiply/add order. |
| **7. GDN recurrence and gates** | 1.354 / 1.157 / 1.174 | Decode unchanged; tiled prefill output/state exact at 1/8/64/320/1,000/1,648/2,048 rows · [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2) | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Aligned recurrence, gates, chronological prefill and retained-prefix state semantics across execution paths. | Updates recurrent memory in token order. Faster prefill retains the existing nine-slot state layout. |
| **8. GDN output gated normalization + FP8 production** | 0.158 / 0.158 / 0.159 | 1,000 rows × 48 sites × M1/M8: exact bytes and scales · [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2) | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Enabled all 48 qualified GDN norm/quant fusion sites with native rounding and exact operator checks. | Normalizes and gates GDN output, then produces FP8 inputs in one kernel; intermediate BF16 rounding is preserved. |
| ↳ GDN output activation FP8 quantization | Included in **stage 8** | Exact fused FP8 bytes/scales; see stage 8 · [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2) | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Enabled all 48 qualified GDN norm/quant fusion sites with native rounding and exact operator checks. | Uses the FP8 output produced by the same gated-normalization kernel. |
| **9. GDN output projection** | 1.915 / 1.901 / 1.904 | 320/320; 320/320 · [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2) | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Applied guarded wide-N MXFP4 decode dispatch to the recurrent output projection. | Maps the recurrent-layer result back to the model's hidden-vector width. |
| **10. Attention input projection** | 1.158 / 1.155 / 1.157 | 320/320; 320/320 · [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2) | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Routed the attention input projection through the qualified FP8 stream and guarded MXFP4 dispatch. | Projects hidden vectors into the inputs for attention. |
| **11. Attention Q/K normalization, RoPE and layout** | 0.228 / 0.209 / 0.212 | 320/320; 320/320 · [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2) | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Added live-row handling for compiled M8 graph padding while retaining the repaired RoPE arithmetic. | Normalizes queries and keys and applies positional rotation (RoPE), retaining the corrected rounding. |
| **12. Attention KV write** | 0.044 / 0.044 / 0.045 | 320/320; 320/320 · [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2) | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Aligned causal attention and cache writes with the M1/M8 arithmetic contract. | Stores new keys and values in the cache for reuse by later tokens. |
| **13. Attention decode** | 0.398 / 4.481 / 12.080 | Paired natural outputs match at 0K/60K/200K · [attention repair evidence](benchmarks/results/attention-page-boundary-20260923.json); earlier isolated alignment evidence remains in the report. | [September 23 attention-page repair](experiments/radiance-public/build_stock_m1_attention_shared.py): reuse the context traversal when M8 queries cross a 16-token attention-page boundary; source/binary hashes are in the capture. | Attends to the current and earlier tokens using corrected arithmetic and shared cache loads. |
| **14. Attention split-KV merge** | 0.166 / 0.130 / 0.097 | 320/320; 320/320 · [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2) | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Recovered shared attention reads and arithmetic-preserving split-KV merging. | Combines attention results from cache partitions in the corrected arithmetic order. |
| **15. Attention output gating** | 0.028 / 0.033 / 0.033 | 320/320; 320/320 · [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2) | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Preserved attention output gating and intermediate BF16 rounding in aligned M1/M8 execution. | Applies learned gates to the attention output. |
| **16. Attention output activation FP8 quantization** | 0.037 / 0.042 / 0.042 | 320/320; 320/320 · [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2) | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Enabled qualified native-rounding FP8 output production and rejected approximate traced quantization. | Converts the attention output to FP8 for its output projection. |
| **17. Attention output projection** | 0.614 / 0.540 / 0.512 | 320/320; 320/320 · [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2) | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Routed the attention output through the qualified FP8 stream and guarded MXFP4 dispatch. | Maps the attention result back to the model's hidden-vector width. |
| **18. Post-attention/GDN residual/normalization + FP8 production** | 0.499 / 0.501 / 0.509 | 320 rows + adversarial values: exact FP8/scales/BF16 residual; prefill 1,000 rows per site · [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2) | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Enabled qualified decoder residual norm/FP8 fusion while preserving BF16 residual and rounding boundaries. | Adds the layer result to the residual, normalizes it and creates FP8 inputs for the MLP in one kernel. |
| ↳ MLP gate/up input FP8 quantization | Included in **stage 18** | Exact fused FP8 bytes/scales; see stage 18 · [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2) | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Enabled qualified decoder residual norm/FP8 fusion while preserving BF16 residual and rounding boundaries. | Uses the FP8 output produced by the same post-normalization kernel. |
| **19. MLP gate/up projection** | 11.723 / 11.370 / 11.314 | 115,841,664 elements: 0 differences; 320 rows on 3 checkpoint matrices plus boundary cases · [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2) | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Enabled guarded wide-N MXFP4 decode dispatch for Qwen's 34,816-column gate/up projection. | Computes both MLP input projections. Wider decode dispatch avoids the slower prefill kernel for qualified shapes. |
| **20. MLP SiLU and gating** | 0.168 / 0.179 / 0.181 | 320/320; 320/320 · [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2) | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Preserved intermediate BF16 rounding in aligned M1/M8 and eager/compiled arithmetic; the slower fused SiLU remains disabled. | Applies SiLU and combines the two MLP branches, retaining intermediate BF16 rounding. |
| **21. MLP down input FP8 quantization** | 0.244 / 0.252 / 0.257 | 320/320; 320/320 · [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2) | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Enabled qualified native per-token FP8 production for the down projection input. | Converts MLP activations to FP8 for the down projection. |
| **22. MLP down projection** | 5.522 / 5.348 / 5.348 | 320/320; 320/320 · [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2) | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Enabled guarded wide-N MXFP4 decode dispatch for the down projection. | Projects the MLP result back to the model's hidden-vector width. |
| **23. Final normalization/layout** | 0.005 / 0.005 / 0.005 | 320/320; 320/320 · [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2) | [`1fecdfe`](https://github.com/Terrydaktal/vllm-coherence/commit/1fecdfe4c6891fe028d68c6cf8705322baa320f2): Aligned final normalization and full BF16 head precision boundaries while preserving the rounding contract. | Normalizes the final hidden vector before vocabulary scoring. |
| **24. Global-256 target head** | 0.999 / 0.981 / 1.002 | Approximate: head-study top-20 retained 119,786/119,988; ranking/probabilities not certified · [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2) | [`31add3d`](https://github.com/Terrydaktal/vllm-coherence/commit/31add3d06080be6e4f5198c6e8afcc6276e1edaf): Removed the eight-per-tile target capacity limit with global INT2 top-256 selection and BF16 reranking; selection remains approximate. | Scores the vocabulary with INT2, selects 256 candidates and rescores them with BF16 weights. Selection remains approximate. |
| **25. Other GPU bookkeeping** | 0.558 / 0.534 / 0.524 | N/A: no isolated target top-20 prediction · [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2) | [`dd89218`](https://github.com/Terrydaktal/vllm-coherence/commit/dd892187327914c02ce8852665eb6ee5a119bca2): Integrated qualified sampling, cache/state bookkeeping and graph-capture admission; no isolated correctness claim is made for this aggregate row. | Runs sampling and cache/state update kernels outside the named model stages. |
| **26. Estimated runtime overhead** | 2.006 / 2.596 / 2.473 (estimate) | [Matched control minus GPU activity union](benchmarks/results/matched-stage-residual-20260923.json) | 2026-09-23: matched natural-serving measurement; source hashes in capture | Indirect observer effects are not proved zero. |
| **Total reconstructed round (stages 1–26)** | **38.839 / 42.826 / 50.300** | GPU activity union plus the estimated remainder | — | Overlapping stages are counted once in the total. |

The table restores the historical grouped 26 measured-row layout and adds a total row. Each timing cell is ordered **0K / 60K / 200K**. Fused kernels are charged once to their containing stage; the ↳ rows are detail-only inclusion records and add no timing; rows without a separate profiler scope are labelled in the timing cell rather than displayed as 0.000. The total counts overlapping GPU activity once. The old forced-replay subtraction is superseded; [its audit](benchmarks/results/stage-timing-audit-20260923.json) remains available.

**Set/order** in the numerical section means the same top-20 token set, followed by the same ranking. Historical correctness results link to their evidence commits; the current timing capture records exact source hashes and is archived with this repair. The profile checks repeatability, not new reference equality. The provenance column links the last relevant implementation commit or links the attention-page repair source.

Fusion still permits correctness instrumentation: a diagnostic kernel can expose intermediate values, and fused outputs can be compared with an unfused reference. The normal GPU profile measures the combined kernel. Internal probes or splitting the kernel can change its performance, so those measurements are not an additive breakdown of the production kernel's time.

The lower layer and kernel detail remains the separate compiled 60K trace: it contains 6 retained complete cycles and is not the 0K/60K/200K profile table above. Its cycle count must not be read as an output-token count.

<details>
<summary>Historical compiled 60K decoder-layer detail: projection and remaining-work timings</summary>

Finish each row's layer, including its MLP, before moving to the next row. Each layer has four projections. Gate and up are one joint GEMM; there is no separately measured gate/up split. The remaining-work column combines normalization, mixing and other operations between those projections.

| Layer | Type | All layer work ms | Input projection ms | Output projection ms | Gate/up projection ms | Down projection ms | Other work ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | GDN | 0.4345 | 0.0758 | 0.0390 | 0.1748 | 0.0800 | 0.0649 |
| 1 | GDN | 0.4526 | 0.0805 | 0.0399 | 0.1833 | 0.0871 | 0.0617 |
| 2 | GDN | 0.4564 | 0.0835 | 0.0396 | 0.1829 | 0.0880 | 0.0623 |
| 3 | Attention | 0.7453 | 0.0732 | 0.0329 | 0.1705 | 0.0807 | 0.3880 |
| 4 | GDN | 0.4377 | 0.0820 | 0.0383 | 0.1743 | 0.0809 | 0.0620 |
| 5 | GDN | 0.4497 | 0.0812 | 0.0403 | 0.1827 | 0.0853 | 0.0601 |
| 6 | GDN | 0.4564 | 0.0844 | 0.0396 | 0.1816 | 0.0885 | 0.0623 |
| 7 | Attention | 0.7418 | 0.0719 | 0.0330 | 0.1722 | 0.0801 | 0.3845 |
| 8 | GDN | 0.4371 | 0.0821 | 0.0386 | 0.1734 | 0.0806 | 0.0624 |
| 9 | GDN | 0.4527 | 0.0807 | 0.0398 | 0.1824 | 0.0885 | 0.0612 |
| 10 | GDN | 0.4612 | 0.0842 | 0.0402 | 0.1855 | 0.0887 | 0.0628 |
| 11 | Attention | 0.7384 | 0.0729 | 0.0329 | 0.1703 | 0.0804 | 0.3819 |
| 12 | GDN | 0.4375 | 0.0815 | 0.0393 | 0.1753 | 0.0800 | 0.0615 |
| 13 | GDN | 0.4514 | 0.0805 | 0.0398 | 0.1829 | 0.0880 | 0.0602 |
| 14 | GDN | 0.4561 | 0.0844 | 0.0399 | 0.1813 | 0.0881 | 0.0623 |
| 15 | Attention | 0.7394 | 0.0732 | 0.0332 | 0.1705 | 0.0806 | 0.3819 |
| 16 | GDN | 0.4367 | 0.0817 | 0.0387 | 0.1739 | 0.0805 | 0.0620 |
| 17 | GDN | 0.4496 | 0.0812 | 0.0400 | 0.1815 | 0.0859 | 0.0610 |
| 18 | GDN | 0.4567 | 0.0834 | 0.0392 | 0.1823 | 0.0883 | 0.0634 |
| 19 | Attention | 0.7393 | 0.0728 | 0.0330 | 0.1711 | 0.0805 | 0.3819 |
| 20 | GDN | 0.4398 | 0.0816 | 0.0390 | 0.1754 | 0.0808 | 0.0629 |
| 21 | GDN | 0.4499 | 0.0814 | 0.0399 | 0.1818 | 0.0858 | 0.0611 |
| 22 | GDN | 0.4554 | 0.0837 | 0.0397 | 0.1819 | 0.0878 | 0.0623 |
| 23 | Attention | 0.7413 | 0.0720 | 0.0330 | 0.1703 | 0.0805 | 0.3855 |
| 24 | GDN | 0.4371 | 0.0813 | 0.0392 | 0.1741 | 0.0802 | 0.0623 |
| 25 | GDN | 0.4536 | 0.0811 | 0.0404 | 0.1838 | 0.0874 | 0.0610 |
| 26 | GDN | 0.4550 | 0.0841 | 0.0394 | 0.1811 | 0.0872 | 0.0632 |
| 27 | Attention | 0.7386 | 0.0731 | 0.0328 | 0.1703 | 0.0803 | 0.3821 |
| 28 | GDN | 0.4374 | 0.0813 | 0.0392 | 0.1747 | 0.0804 | 0.0617 |
| 29 | GDN | 0.4528 | 0.0825 | 0.0398 | 0.1839 | 0.0859 | 0.0608 |
| 30 | GDN | 0.4584 | 0.0832 | 0.0399 | 0.1849 | 0.0877 | 0.0626 |
| 31 | Attention | 0.7377 | 0.0721 | 0.0329 | 0.1692 | 0.0804 | 0.3830 |
| 32 | GDN | 0.4376 | 0.0814 | 0.0389 | 0.1739 | 0.0808 | 0.0626 |
| 33 | GDN | 0.4504 | 0.0815 | 0.0404 | 0.1807 | 0.0870 | 0.0609 |
| 34 | GDN | 0.4591 | 0.0833 | 0.0393 | 0.1831 | 0.0874 | 0.0660 |
| 35 | Attention | 0.7389 | 0.0736 | 0.0332 | 0.1699 | 0.0804 | 0.3817 |
| 36 | GDN | 0.4416 | 0.0822 | 0.0393 | 0.1763 | 0.0813 | 0.0624 |
| 37 | GDN | 0.4493 | 0.0818 | 0.0396 | 0.1815 | 0.0860 | 0.0604 |
| 38 | GDN | 0.4581 | 0.0841 | 0.0398 | 0.1840 | 0.0880 | 0.0621 |
| 39 | Attention | 0.7342 | 0.0727 | 0.0328 | 0.1697 | 0.0806 | 0.3785 |
| 40 | GDN | 0.4400 | 0.0815 | 0.0388 | 0.1761 | 0.0814 | 0.0621 |
| 41 | GDN | 0.4498 | 0.0807 | 0.0396 | 0.1826 | 0.0869 | 0.0600 |
| 42 | GDN | 0.4600 | 0.0849 | 0.0404 | 0.1846 | 0.0878 | 0.0622 |
| 43 | Attention | 0.7339 | 0.0721 | 0.0328 | 0.1707 | 0.0805 | 0.3778 |
| 44 | GDN | 0.4354 | 0.0816 | 0.0385 | 0.1747 | 0.0798 | 0.0608 |
| 45 | GDN | 0.4510 | 0.0813 | 0.0397 | 0.1822 | 0.0875 | 0.0603 |
| 46 | GDN | 0.4545 | 0.0844 | 0.0400 | 0.1815 | 0.0873 | 0.0613 |
| 47 | Attention | 0.7395 | 0.0719 | 0.0329 | 0.1704 | 0.0806 | 0.3837 |
| 48 | GDN | 0.4399 | 0.0820 | 0.0386 | 0.1755 | 0.0806 | 0.0632 |
| 49 | GDN | 0.4499 | 0.0818 | 0.0392 | 0.1816 | 0.0873 | 0.0601 |
| 50 | GDN | 0.4593 | 0.0848 | 0.0405 | 0.1850 | 0.0874 | 0.0617 |
| 51 | Attention | 0.7333 | 0.0725 | 0.0329 | 0.1703 | 0.0801 | 0.3775 |
| 52 | GDN | 0.4360 | 0.0817 | 0.0388 | 0.1746 | 0.0795 | 0.0614 |
| 53 | GDN | 0.4512 | 0.0826 | 0.0398 | 0.1816 | 0.0874 | 0.0599 |
| 54 | GDN | 0.4537 | 0.0843 | 0.0397 | 0.1822 | 0.0855 | 0.0621 |
| 55 | Attention | 0.7349 | 0.0707 | 0.0328 | 0.1706 | 0.0799 | 0.3809 |
| 56 | GDN | 0.4371 | 0.0818 | 0.0385 | 0.1748 | 0.0807 | 0.0614 |
| 57 | GDN | 0.4508 | 0.0806 | 0.0402 | 0.1832 | 0.0870 | 0.0597 |
| 58 | GDN | 0.4555 | 0.0832 | 0.0398 | 0.1827 | 0.0876 | 0.0623 |
| 59 | Attention | 0.7340 | 0.0724 | 0.0328 | 0.1707 | 0.0804 | 0.3776 |
| 60 | GDN | 0.4366 | 0.0818 | 0.0390 | 0.1741 | 0.0805 | 0.0612 |
| 61 | GDN | 0.4516 | 0.0809 | 0.0398 | 0.1830 | 0.0883 | 0.0596 |
| 62 | GDN | 0.4577 | 0.0846 | 0.0396 | 0.1836 | 0.0870 | 0.0629 |
| 63 | Attention | 0.7354 | 0.0728 | 0.0330 | 0.1708 | 0.0800 | 0.3789 |

</details>

<details>
<summary>Every recorded kernel, grouped by stage</summary>

| Stage / compiled kernel | Calls in 6 retained cycles | Current GPU ms per profile cycle |
| --- | ---: | ---: |
| Drafter / `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x32_MI16x16x1_SN_LDSB1_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_CD1_1_CLR0_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB1_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA1201_ICIW0_IU1_K1_LDSTI0_LBSPPA128_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB0_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB2_ONLL1_PAP0_PGL0_PGR1_PLR1_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1.kd` | 6 | 0.007046 |
| Drafter / `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x32x256_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB1_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA1201_ICIW0_IU1_K1_LDSTI0_LBSPPA512_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB0_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB16_ONLL0_PAP0_PGL0_PGR1_PLR1_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_4_1.kd` | 6 | 0.176626 |
| Drafter / `__amd_rocclr_copyBuffer.kd` | 6 | 0.001932 |
| Drafter / `__amd_rocclr_fillBufferAligned.kd` | 18 | 0.006884 |
| Drafter / `_cache_draft_logits_kernel.kd` | 6 | 0.002426 |
| Drafter / `_draft_head_int2.kd` | 6 | 0.830296 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_17408_EVEN_K_1_GRID_MN_40_cache_modifier_NONE.kd` | 30 | 0.738051 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_25600_EVEN_K_1_GRID_MN_40_cache_modifier_NONE.kd` | 6 | 0.235780 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_40_cache_modifier_NONE.kd` | 30 | 0.202370 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_5120_EVEN_K_1_GRID_MN_272_cache_modifier_NONE.kd` | 30 | 1.447241 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_5120_EVEN_K_1_GRID_MN_48_cache_modifier_NONE.kd` | 30 | 0.272916 |
| Drafter / `_prepare_dflash_inputs_kernel.kd` | 6 | 0.007519 |
| Drafter / `_rerank_exact.kd` | 6 | 0.009086 |
| Drafter / `_selector_walk_kernel.kd` | 6 | 0.008006 |
| Drafter / `kernel_unified_attention.kd` | 30 | 1.837015 |
| Drafter / `reshape_and_cache_kernel_flash.kd` | 60 | 0.026323 |
| Drafter / `triton_per_fused_4.kd` | 6 | 0.001946 |
| Drafter / `triton_per_fused_8.kd` | 24 | 0.007556 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_preshuffle_gemm_squeeze_view_0.kd` | 30 | 0.009728 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_preshuffle_gemm_squeeze_view_2.kd` | 6 | 0.001992 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_preshuffle_gemm_squeeze_view_4.kd` | 54 | 0.016071 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_view_0.kd` | 6 | 0.004959 |
| Drafter / `triton_poi_fused_0.kd` | 6 | 0.002713 |
| Drafter / `triton_poi_fused_5.kd` | 6 | 0.002219 |
| Drafter / `triton_poi_fused_9.kd` | 24 | 0.010056 |
| Drafter / `triton_poi_fused__to_copy_clamp_div_preshuffle_gemm_squeeze_view_1.kd` | 30 | 0.012388 |
| Drafter / `triton_poi_fused__to_copy_clamp_div_preshuffle_gemm_squeeze_view_3.kd` | 6 | 0.002012 |
| Drafter / `triton_poi_fused__to_copy_clamp_div_preshuffle_gemm_squeeze_view_5.kd` | 54 | 0.016664 |
| Drafter / `triton_poi_fused_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_3.kd` | 54 | 0.022558 |
| Drafter / `triton_poi_fused_add_arange_bitwise_and_constant_pad_nd_ge_mul_rms_norm_select_slice_unsqueeze_view_1.kd` | 6 | 0.023039 |
| Drafter / `triton_poi_fused_add_permute_unsqueeze_view_2.kd` | 6 | 0.001799 |
| Drafter / `triton_poi_fused_cat_expand_index_mul_slice_unsqueeze_view_1.kd` | 6 | 0.002519 |
| Drafter / `triton_red_fused__to_copy_abs_clamp_div_max_mul_preshuffle_gemm_silu_slice_squeeze_view_6.kd` | 30 | 0.017248 |
| Drafter / `triton_red_fused__to_copy_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_w4_gemm_2.kd` | 30 | 0.031342 |
| Drafter / `triton_red_fused__to_copy_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_w4_gemm_7.kd` | 24 | 0.025796 |
| Drafter / `triton_red_fused__to_copy_embedding_mul_rms_norm_w4_gemm_0.kd` | 6 | 0.004226 |
| Drafter / `triton_red_fused_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_7.kd` | 6 | 0.006579 |
| Drafter / `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) [clone .kd]` | 12 | 0.009231 |
| Drafter / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 6 | 0.003572 |
| Drafter / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 6 | 0.003252 |
| Drafter / `void at::native::bitonicSortKVInPlace<2, -1, 16, 16, c10::BFloat16, long, at::native::GTOp<c10::BFloat16, true>, unsigned int>(at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, at::native::GTOp<c10::BFloat16, true>) [clone .kd]` | 6 | 0.003119 |
| Drafter / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 6 | 0.004959 |
| Drafter / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 6 | 0.002526 |
| Drafter / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 6 | 0.003486 |
| Drafter / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 6 | 0.003112 |
| Drafter / `void at::native::mbtopk::computeBlockDigitCounts<c10::BFloat16, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 12 | 0.038252 |
| Drafter / `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 24 | 0.057416 |
| Drafter / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, c10::BFloat16>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 12 | 0.022158 |
| Drafter / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, float>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, float*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 24 | 0.028703 |
| Drafter / `void at::native::mbtopk::fill<unsigned int, unsigned int>(unsigned int*, unsigned int, unsigned int) [clone .kd]` | 12 | 0.002898 |
| Drafter / `void at::native::mbtopk::gatherTopK<c10::BFloat16, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 6 | 0.024292 |
| Drafter / `void at::native::mbtopk::gatherTopK<float, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, float*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 6 | 0.008919 |
| Drafter / `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4>) [clone .kd]` | 6 | 0.003806 |
| Drafter / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul>) [clone .kd]` | 6 | 0.001712 |
| Drafter / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) [clone .kd]` | 6 | 0.002446 |
| Drafter / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>) [clone .kd]` | 12 | 0.003878 |
| Drafter / `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>) [clone .kd]` | 12 | 0.005205 |
| Drafter / `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool) [clone .kd]` | 6 | 0.002099 |
| Drafter / `void at::native::warpMergeSortKVInPlace<2, -1, 128, 16, float, long, at::native::GTOp<float, true>, unsigned int, 32>(at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, at::native::GTOp<float, true>, float) [clone .kd]` | 6 | 0.004932 |
| Drafter / `void r4d_gemm_w4a16_nt_m64_kernel<1, 1, false>(unsigned short const*, unsigned int const*, unsigned int const*, __hip_bfloat16*, int, int, int, int, int) [clone .kd]` | 6 | 0.004526 |
| Drafter / `void r4d_gemm_w4a16_nt_m64_kernel<1, 1, true>(unsigned short const*, unsigned int const*, unsigned int const*, __hip_bfloat16*, int, int, int, int, int) [clone .kd]` | 60 | 0.081617 |
| Drafter / `void vllm::rms_norm_kernel<c10::BFloat16, 8, 2, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, long, long, long, c10::BFloat16 const*, long, float, int, int) [clone .kd]` | 6 | 0.002959 |
| Drafter / `void vllm::rms_norm_kernel<c10::BFloat16, 8, 4, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, long, long, long, c10::BFloat16 const*, long, float, int, int) [clone .kd]` | 6 | 0.002759 |
| Drafter / `void vllm::rotary_embedding_kernel<c10::BFloat16, c10::BFloat16, true>(long const*, c10::BFloat16*, c10::BFloat16*, c10::BFloat16 const*, int, long, long, long, int, int, int, long, bool) [clone .kd]` | 6 | 0.002406 |
| Attention KV write / `reshape_and_cache_kernel_flash.kd` | 96 | 0.043691 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_1.kd` | 96 | 0.020891 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_3.kd` | 96 | 0.028504 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_4.kd` | 96 | 0.025304 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_arange_bitwise_and_eq_index_lt_remainder_select_split_where_2.kd` | 96 | 0.028065 |
| Attention Q/K normalization, RoPE and layout / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 96 | 0.029691 |
| Attention Q/K normalization, RoPE and layout / `void stock_m1_gemma_norm<false, 256, 32>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, long, long, float, float*) [clone .kd]` | 96 | 0.047164 |
| Attention Q/K normalization, RoPE and layout / `void stock_m1_gemma_norm<false, 256, 64>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, long, long, float, float*) [clone .kd]` | 96 | 0.037204 |
| Attention decode / `void qwen_stock_m1_shared_decode<4, 16, 256, 6, 16, 0, 3430971>(R4DArgs, int) [clone .kd]` | 96 | 5.256132 |
| Attention input projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 1, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 96 | 1.159962 |
| Attention output activation FP8 quantization / `void vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<c10::BFloat16, c10::Float8_e4m3fn>(c10::Float8_e4m3fn*, float*, c10::BFloat16 const*, float const*, int, long, long) [clone .kd]` | 96 | 0.043097 |
| Attention output gating / `triton_poi_fused_dynamic_per_token_scaled_fp8_quant_mul_sigmoid_view_0.kd` | 96 | 0.031084 |
| Attention output projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 4, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 96 | 0.526833 |
| Attention split-KV merge / `void qwen_stock_m1_shared_merge<256, 4, 1>(R4DArgs, int, int) [clone .kd]` | 96 | 0.128212 |
| Embedding + first input normalization + FP8 production / `triton_poi_fused__to_copy_embedding_0.kd` | 6 | 0.002419 |
| Embedding + first input normalization + FP8 production / `void norm_quant<false, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned char*, float*, unsigned short*, long, long, float) [clone .kd]` | 6 | 0.007192 |
| Final normalization/layout / `void stock_m1_gemma_norm<true, 5120, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, long, long, float, float*) [clone .kd]` | 6 | 0.005099 |
| GDN convolution / `_causal_conv1d_update_kernel.kd` | 288 | 0.209313 |
| GDN input projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 1, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 288 | 3.945794 |
| GDN layout/copies and buffer initialization / `triton_poi_fused_add_1.kd` | 18 | 0.003897 |
| GDN layout/copies and buffer initialization / `triton_poi_fused_add_2.kd` | 12 | 0.002578 |
| GDN layout/copies and buffer initialization / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 288 | 0.145980 |
| GDN layout/copies and buffer initialization / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) [clone .kd]` | 288 | 0.085379 |
| GDN layout/copies and buffer initialization / `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>) [clone .kd]` | 288 | 0.068732 |
| GDN output gated normalization + FP8 production / `gdn_norm_quant_kernel.kd` | 288 | 0.160199 |
| GDN output projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 4, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 288 | 1.896700 |
| GDN recurrence and gates / `stock_gdn_scan_kernel.kd` | 288 | 1.181616 |
| Layer input residual/normalization + FP8 production / `void norm_quant<true, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned char*, float*, unsigned short*, long, long, float) [clone .kd]` | 378 | 0.509859 |
| MLP SiLU and gating / `triton_poi_fused_dynamic_per_token_scaled_fp8_quant_mul_silu_slice_0.kd` | 288 | 0.125460 |
| MLP SiLU and gating / `triton_poi_fused_dynamic_per_token_scaled_fp8_quant_mul_silu_slice_1.kd` | 96 | 0.053631 |
| MLP down input FP8 quantization / `void vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<c10::BFloat16, c10::Float8_e4m3fn>(c10::Float8_e4m3fn*, float*, c10::BFloat16 const*, float const*, int, long, long) [clone .kd]` | 384 | 0.264137 |
| MLP down projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 4, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 384 | 5.367781 |
| MLP gate/up projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 1, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 384 | 11.370361 |
| Post-attention/GDN residual/normalization + FP8 production / `void norm_quant<true, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned char*, float*, unsigned short*, long, long, float) [clone .kd]` | 384 | 0.530712 |
| Global-256 target head / `__amd_rocclr_fillBufferAligned.kd` | 6 | 0.002432 |
| Global-256 target head / `_draft_head_int2.kd` | 6 | 0.771209 |
| Global-256 target head / `_rerank_exact.kd` | 6 | 0.029826 |
| Global-256 target head / `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) [clone .kd]` | 6 | 0.004506 |
| Global-256 target head / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 6 | 0.003492 |
| Global-256 target head / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 6 | 0.002779 |
| Global-256 target head / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 6 | 0.005179 |
| Global-256 target head / `void at::native::mbtopk::computeBlockDigitCounts<c10::BFloat16, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 12 | 0.090052 |
| Global-256 target head / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, c10::BFloat16>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 12 | 0.038398 |
| Global-256 target head / `void at::native::mbtopk::fill<unsigned int, unsigned int>(unsigned int*, unsigned int, unsigned int) [clone .kd]` | 6 | 0.001526 |
| Global-256 target head / `void at::native::mbtopk::gatherTopK<c10::BFloat16, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 6 | 0.040066 |
| Global-256 target head / `void at::native::radixSortKVInPlace<2, -1, 128, 8, c10::BFloat16, long, unsigned int>(at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, bool) [clone .kd]` | 6 | 0.005379 |
| Global-256 target head / `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4>) [clone .kd]` | 6 | 0.003819 |
| Global-256 target head / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) [clone .kd]` | 6 | 0.001632 |
| Global-256 target head / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>) [clone .kd]` | 6 | 0.002232 |
| Global-256 target head / `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>) [clone .kd]` | 12 | 0.005191 |
| Other GPU bookkeeping / `__amd_rocclr_copyBuffer.kd` | 150 | 0.065649 |
| Other GPU bookkeeping / `__amd_rocclr_fillBufferAligned.kd` | 6 | 0.002306 |
| Other GPU bookkeeping / `_combine_sampled_and_draft_tokens_kernel.kd` | 6 | 0.002992 |
| Other GPU bookkeeping / `_compute_local_logits_stats_kernel.kd` | 6 | 0.028246 |
| Other GPU bookkeeping / `_compute_slot_mappings_kernel.kd` | 6 | 0.002906 |
| Other GPU bookkeeping / `_expand_idx_mapping_kernel.kd` | 6 | 0.001886 |
| Other GPU bookkeeping / `_gather_block_tables_kernel.kd` | 6 | 0.004639 |
| Other GPU bookkeeping / `_get_num_sampled_and_rejected_kernel.kd` | 6 | 0.002552 |
| Other GPU bookkeeping / `_insert_resampled_kernel.kd` | 6 | 0.003412 |
| Other GPU bookkeeping / `_post_update_kernel.kd` | 6 | 0.005159 |
| Other GPU bookkeeping / `_prepare_pos_seq_lens_kernel.kd` | 6 | 0.002039 |
| Other GPU bookkeeping / `_prepare_rope_positions_kernel.kd` | 6 | 0.002726 |
| Other GPU bookkeeping / `_rejection_kernel.kd` | 6 | 0.007452 |
| Other GPU bookkeeping / `_resample_kernel.kd` | 6 | 0.013832 |
| Other GPU bookkeeping / `_scatter_num_accepted_kernel.kd` | 6 | 0.002119 |
| Other GPU bookkeeping / `_temperature_kernel.kd` | 6 | 0.013779 |
| Other GPU bookkeeping / `postprocess_mamba_fused_kernel.kd` | 6 | 0.002526 |
| Other GPU bookkeeping / `precopy_mamba_align_fused_kernel.kd` | 6 | 0.002679 |
| Other GPU bookkeeping / `preprocess_mamba_align_fused_kernel.kd` | 6 | 0.002579 |
| Other GPU bookkeeping / `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}>(int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}, function_traits<at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}>::result_type*) [clone .kd]` | 36 | 0.011374 |
| Other GPU bookkeeping / `void (anonymous namespace)::softmax_warp_forward<float, float, float, 6, false, false, 32>(float*, float const*, int, int, int, bool const*, int, bool) [clone .kd]` | 6 | 0.002259 |
| Other GPU bookkeeping / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 42 | 0.013413 |
| Other GPU bookkeeping / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 6 | 0.002412 |
| Other GPU bookkeeping / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 42 | 0.013860 |
| Other GPU bookkeeping / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<int> const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<int> const&)::{lambda(int, bool)#1}) [clone .kd]` | 36 | 0.014521 |
| Other GPU bookkeeping / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<float> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<float> const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<float> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<float> const&)::{lambda(int, bool)#1}) [clone .kd]` | 18 | 0.018624 |
| Other GPU bookkeeping / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 96 | 0.046478 |
| Other GPU bookkeeping / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 12 | 0.005938 |
| Other GPU bookkeeping / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 6 | 0.003072 |
| Other GPU bookkeeping / `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 24 | 0.060523 |
| Other GPU bookkeeping / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, float>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, float*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 24 | 0.033636 |
| Other GPU bookkeeping / `void at::native::mbtopk::fill<unsigned int, unsigned int>(unsigned int*, unsigned int, unsigned int) [clone .kd]` | 6 | 0.001452 |
| Other GPU bookkeeping / `void at::native::mbtopk::gatherTopK<float, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, float*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 6 | 0.026926 |
| Other GPU bookkeeping / `void at::native::tensor_kernel_scan_innermost_dim<float, std::plus<float> >(float*, float const*, unsigned int, unsigned int, unsigned int, float, std::plus<float>) [clone .kd]` | 6 | 0.002459 |
| Other GPU bookkeeping / `void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>, 4, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast) [clone .kd]` | 36 | 0.010181 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<16, at::native::BinaryFunctor<bool, bool, bool, at::native::BitwiseOrFunctor<bool> >, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<bool, bool, bool, at::native::BitwiseOrFunctor<bool> >, std::array<char*, 3ul>) [clone .kd]` | 6 | 0.002466 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<16, at::native::FillFunctor<bool>, std::array<char*, 1ul> >(int, at::native::FillFunctor<bool>, std::array<char*, 1ul>) [clone .kd]` | 6 | 0.001812 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<16, at::native::bitwise_not_kernel_cuda(at::TensorIteratorBase&)::{lambda(bool)#1}, std::array<char*, 2ul> >(int, at::native::bitwise_not_kernel_cuda(at::TensorIteratorBase&)::{lambda(bool)#1}, std::array<char*, 2ul>) [clone .kd]` | 6 | 0.002306 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}, std::array<char*, 2ul>) [clone .kd]` | 36 | 0.012547 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}, std::array<char*, 2ul>) [clone .kd]` | 6 | 0.001759 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}, std::array<char*, 3ul> >(int, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}, std::array<char*, 3ul>) [clone .kd]` | 6 | 0.009699 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool, float, float)#1}, std::array<char*, 4ul> >(int, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool, float, float)#1}, std::array<char*, 4ul>) [clone .kd]` | 12 | 0.004251 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 2ul> >(int, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 2ul>) [clone .kd]` | 36 | 0.011427 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, 2ul>) [clone .kd]` | 42 | 0.011613 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul>) [clone .kd]` | 6 | 0.001992 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>) [clone .kd]` | 6 | 0.001866 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>) [clone .kd]` | 12 | 0.003418 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int, at::native::FillFunctor<int>, std::array<char*, 1ul>) [clone .kd]` | 6 | 0.001666 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>) [clone .kd]` | 6 | 0.009692 |
| Other GPU bookkeeping / `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool) [clone .kd]` | 6 | 0.001992 |
| Other GPU bookkeeping / `void at::native::warpMergeSortKVInPlace<2, -1, 128, 16, float, long, at::native::GTOp<float, true>, unsigned int, 32>(at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, at::native::GTOp<float, true>, float) [clone .kd]` | 6 | 0.004652 |

</details>

## Global-256 target-head

This is the table from [the top-256 PR](https://github.com/magiccodingman/vllm-radiance/pull/9).
It is a **separate, earlier 60K-generated-token benchmark per method**, predating the
M1/M8 and eager/compiled repairs and subsequent performance backports. Its old
end-to-end throughput figures are intentionally omitted because they do not
describe the current complete backend.

| Target path | Median M8 head time | Top-1 match | Complete reference top-20 retained |
|---|---:|---:|---:|
| Full BF16 fallback | 4.122 ms | 119,988/119,988 (100.0000%) | 119,988/119,988 (100.0000%) |
| Original block-8/64 + rerank-80 | 1.085 ms | 119,956/119,988 (99.9733%) | 98,452/119,988 (82.0515%) |
| Global INT2 top-128 + BF16 rerank | 1.114 ms | 119,986/119,988 (99.9983%) | 118,254/119,988 (98.5549%) |
| Global INT2 top-256 + BF16 rerank (default) | 1.128 ms | 119,986/119,988 (99.9983%) | 119,786/119,988 (99.8316%) |

There were 115 natural completions per method on 11 private Pi request boundaries
with 57,008–65,527 input tokens. Output totals were 60,598 / 60,075 / 60,348 /
60,675 tokens for full / block / global-128 / global-256 respectively. Tools were
not executed. Head timings are median eight-row GPU-event measurements on the
same captured hidden vectors; 119,988 prediction rows were compared.

Global-256 removes the eight-per-tile capacity limit, but remains approximate.
The two changed final argmax IDs and 202 incomplete top-20 sets are observed
misses. Complete top-20 retention does not certify score equality, ordering or
sampling probabilities. The full BF16 path is the reference in this head study,
not an independent proof of the model. [Methodology](docs/VERIFY_HEAD_GLOBAL_TOPK.md)
· [Aggregate evidence](benchmarks/results/20260916-verify-head-global-topk-long/summary.json).

## Benchmarks

This benchmark uses the retained 60,000-input-token Pi prefix and the compiled Coherence backend with Global-256, the attention-page-boundary repair and the pinned-RAM huge-page promotion repair. Five requests are chained in one context: code, prose about code measurement, JSON, thinking/prose and checkpoint generation. Code and JSON disable thinking; both prose requests enable it. All requests stop naturally. Generation uses temperature 1, top-p 0.95, top-k 40 and seed 0; compaction uses temperature 0.3. The checkpoint request forces a snapshot-tail flush. Private fixture text was not decoded or inspected, and no generated text or token arrays were saved. Runner: [benchmark_pi_coding_json_compaction.py](experiments/radiance-public/benchmark_pi_coding_json_compaction.py).

| Stage | Thinking | Prompt tokens | Generated tokens | Classified output | First data | Mean round | Post-first | Peak 3s | Acceptance |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Coding task | off | 60,208 | 8,213 | 7,693 code; 519 prose; 0 separately observed reasoning | 30.50 s | 42.69 ms | 103.78 tok/s | 140.70 tok/s | 49.00% |
| Prose about code measurement | on | 68,562 | 5,071 | 5,072 prose; 0 separately observed reasoning | 1.90 s | 43.04 ms | 89.96 tok/s | 176.26 tok/s | 41.01% |
| JSON task | off | 73,752 | 3,245 | 3,244 JSON; valid JSON | 2.12 s | 43.33 ms | 84.56 tok/s | 114.64 tok/s | 38.02% |
| Thinking/prose task | on | 77,144 | 6,248 | 6,247 prose; 0 separately observed reasoning | 2.33 s | 43.60 ms | 59.03 tok/s | 83.16 tok/s | 22.48% |
| Compaction checkpoint | off | 83,546 | 3,045 | 3,045 checkpoint tokens; completion marker valid; required headings missing or duplicated | 2.62 s | 43.91 ms | 67.81 tok/s | 96.02 tok/s | 28.24% |

Cached/total prompt tokens: Coding task: 0/60,208, Prose about code measurement: 65,920/68,562, JSON task: 70,864/73,752, Thinking/prose task: 74,160/77,144, Compaction checkpoint: 80,752/83,546.

The phase counters retokenize classified text, so their totals can differ from the backend's emitted-token count. `phase_token_counts_cover_output=false` for Coding task, Prose about code measurement, JSON task, Thinking/prose task, Compaction checkpoint. No separate reasoning channel was exposed for Prose about code measurement, Thinking/prose task; those streams are reported as prose, not relabelled as reasoning. The compaction row measures checkpoint generation with a requested cache flush, not a full Pi transcript commit or old-snapshot retirement. The checkpoint format failed validation and must not be treated as a committed compaction. `peak_3s_tokens_per_second` is the maximum completed three-second sliding-window rate after first data, never a single-frame burst.

2026-09-23 rerun status: `complete_with_validation_failure`. [Numeric results and release identity](benchmarks/results/pi-coding-json-compaction.json). These results use top-k 40, while the older September 20 table used top-k 20, so the output and acceptance changes are not a controlled before/after comparison.

This is the same natural-stop coding task run independently at empty, 60K and 200K input context. Thinking is disabled, EOS remains enabled, and each arm uses temperature 1, top-p 0.95 and top-k 40. The non-empty arms use operator-supplied token-prefix fixtures; only their hashes are retained. The three-second peak is the maximum completed sliding-window rate, not a single-frame burst. Runner: [benchmark_pi_coding_contexts.py](experiments/radiance-public/benchmark_pi_coding_contexts.py).

| Context | Prompt tokens | Generated tokens | Mean round | Post-first | Peak 3s | Acceptance | Round events (total; speculative/expected; timed) | Validation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0K | 203 | 4,981 | 38.66 ms | 128.06 tok/s | 157.62 tok/s | 56.45% | 1,007 total; 1006/1006 speculative; 1,006 timed; captured | short natural stop |
| 60K | 60,208 | 8,213 | 42.68 ms | 103.80 tok/s | 140.20 tok/s | 49.00% | 1,855 total; 1854/1854 speculative; 1,854 timed; captured | pass |
| 200K | 200,208 | 4,233 | 50.37 ms | 96.12 tok/s | 136.38 tok/s | 54.75% | 877 total; 876/876 speculative; 876 timed; captured | short natural stop |

Report status: `complete_with_validation_failure`. [Numeric results and every round](benchmarks/results/pi-coding-contexts.json). Each completed row stores every scheduler event under `contexts.<context>.round_capture.records`; a count mismatch is a validation failure. The target is 5,000 output tokens, with shorter natural completions reported explicitly.

Fixture provenance: 0K — empty prefix; 60K — operator-supplied private 60K token fixture; content not decoded; 200K — deterministic public synthetic 200K token prefix; no chat content.

Run all three arms against an active backend with `uv run python experiments/radiance-public/benchmark_pi_coding_contexts.py --fixture-60k PATH_TO_60K_FIXTURE --fixture-200k PATH_TO_200K_FIXTURE --tokenizer-json PATH_TO_TOKENIZER --abi SNAPSHOT_ABI --top-k 40`.

### Complete per-round capture

The context benchmark now retains every content-free scheduler event for each arm, including an unmeasured first event. The full numeric records are stored under `contexts.<context>.round_capture.records` in the result JSON; this table is a coverage check rather than another latency aggregate. A count mismatch is a validation failure. Expected rounds come from the speculative-round counter; the first prefill event is retained in logged events but excluded from that counter.

| Context | Logged events | Speculative rounds | Expected speculative rounds | Timed round values | Untimed events | Missing round numbers | Capture status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0K | 1,007 | 1006 | 1006 | 1,006 | 1 | none | captured |
| 60K | 1,855 | 1854 | 1854 | 1,854 | 1 | none | captured |
| 200K | 877 | 876 | 876 | 876 | 1 | none | captured |


The histogram below is generated from the complete per-round records in this benchmark. Every measured `round_ms` value appears in exactly one bin; untimed events are reported separately.

| Round time | 0K arm | 60K arm | 200K arm |
|---|---:|---:|---:|
| `<35` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `35–37` | 13 (1.3%) | 0 (0.0%) | 0 (0.0%) |
| `37–39` | 750 (74.6%) | 0 (0.0%) | 0 (0.0%) |
| `39–40` | 228 (22.7%) | 1 (0.1%) | 0 (0.0%) |
| `40–40.5` | 1 (0.1%) | 3 (0.2%) | 0 (0.0%) |
| `40.5–41` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `41–41.5` | 0 (0.0%) | 2 (0.1%) | 0 (0.0%) |
| `41.5–42` | 1 (0.1%) | 1 (0.1%) | 0 (0.0%) |
| `42–43` | 2 (0.2%) | 1,795 (96.8%) | 0 (0.0%) |
| `43–45` | 3 (0.3%) | 38 (2.0%) | 0 (0.0%) |
| `45–46` | 2 (0.2%) | 4 (0.2%) | 0 (0.0%) |
| `46–47` | 3 (0.3%) | 1 (0.1%) | 0 (0.0%) |
| `47–48` | 2 (0.2%) | 1 (0.1%) | 0 (0.0%) |
| `48–48.5` | 0 (0.0%) | 0 (0.0%) | 1 (0.1%) |
| `48.5–49` | 1 (0.1%) | 0 (0.0%) | 0 (0.0%) |
| `49–51` | 0 (0.0%) | 2 (0.1%) | 858 (97.9%) |
| `51–52.5` | 0 (0.0%) | 5 (0.3%) | 12 (1.4%) |
| `52.5–53` | 0 (0.0%) | 1 (0.1%) | 0 (0.0%) |
| `53–53.5` | 0 (0.0%) | 0 (0.0%) | 1 (0.1%) |
| `53.5–54` | 0 (0.0%) | 0 (0.0%) | 1 (0.1%) |
| `54–55` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `55–56` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `56–60` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `60–62.5` | 0 (0.0%) | 0 (0.0%) | 3 (0.3%) |
| `62.5–63` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `63–63.5` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `63.5–64` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `64–65` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `65–70` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `70–100` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `100–250` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `250–500` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `≥500` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| **Timed rounds** | **1,006** | **1,854** | **876** |
| **Untimed events** | **1** | **1** | **1** |
| **Mean / median** | **38.66 / 38.85 ms** | **42.68 / 42.63 ms** | **50.37 / 50.31 ms** |


### Known remaining symptoms and likely causes

**2026-09-23: the separate periodic huge-page stalls are repaired.** Kernel tracing caught `khugepaged` collapsing the registered host-memory arena and invoking `amdgpu_hmm_invalidate_hsa`. At 200K this produced 189–206 ms rounds about every 10.24 seconds. Pinned chat-handover buffers now apply `MADV_NOHUGEPAGE` to their whole anonymous backing mappings, including unused allocator space, before any cache transfer. The live worker reported 16.25 GiB protected. This runs once during allocation; it adds no per-round scan or syscall and changes neither the global huge-page policy nor model arithmetic or snapshot contents.

The targeted 200K control had 4 rounds over 100 ms among 876; both repaired 200K replays had none, including a return from RAM. The full rerun above preserved all three coding output-token hashes (17,427 tokens) and all five chained-task hashes. Across its 3,736 timed coding rounds, 0 exceeded 100 ms; maximum rounds were 48.600 / 52.974 / 62.057 ms. These are sampled intervention results, not a proof that every possible driver or scheduler pause is eliminated. First-use allocation/prefill costs were not a controlled comparison. [Diagnosis, mapping policy, RAM-return checks and rerun evidence](benchmarks/results/huge-page-promotion-20260923.json).

The separate stage-capture restart/warmup sequence still recorded isolated 6,073 / 1,025 ms warmup pauses at 60K / 200K, and 1 clean-control round over 100 ms (660.499 ms). These remain in the evidence. Their cause is unresolved; this repair does not claim to eliminate those isolated pauses or the initial pinned-allocation cost.

**2026-09-23: a repeating attention-page-boundary slowdown is diagnosed and repaired.** The shared M8 attention kernel used two independent GPU work groups whenever its eight verification queries crossed a 16-token KV page. Both groups reread the full context. This added about 3 ms per round at 60K and 10 ms at 200K. The repair shares one traversal while preserving each query's original split range, softmax accumulation and rounding. Results below use compiled Global-256 natural Pi coding completions with temperature 1, top-p 0.95, top-k 40 and seed 0. The event profiler is disabled; ordinary numeric round telemetry remains enabled. All timed rounds, including outliers, contribute to the means.

| Starting history | Before, mean round ms | Fixed, mean round ms | Timed rounds per run | Output tokens per run | Same generated output |
| --- | ---: | ---: | ---: | ---: | --- |
| 0K | 38.768 | 38.754 | 1,006 | 4,981 | Yes |
| 60K | 44.551 | 42.585 | 1,320 | 5,686 | Yes |
| 200K | 62.741 | 50.440 | 1,296 | 5,965 | Yes |

The 0K task starts at 203 prompt tokens; the longer tasks start at 60,208 and 200,208. Each run also retains its first, untimed prefill event. Across all three pairs, all 16,632 generated tokens match. Separately, 264 native operator cases check 2,112 query rows against serial M1 with no differing output bytes, covering every page offset, split boundaries, FP8/BF16/padded-byte KV layouts, graph replay and corruption controls. These are sampled checks, not a universal arithmetic proof or full-vocabulary comparison.

Fixed per-offset round medians span 42.511–42.594 ms at 60K and 50.350–50.418 ms at 200K. A few isolated spikes remain. The old 200K control also entered an additional persistent slow state near an adaptive recovery fence; that state was absent from the fixed run, but its separate queue mechanism is not independently proved. The repair establishes the cause of the repeating page-boundary mode, not that all possible scheduling or driver jitter has been eliminated. [Complete numeric comparison and histograms](benchmarks/results/attention-page-boundary-20260923.json).

The normal release worker was then restarted with the frozen repair and repeated the 60K task: all 5,686 output tokens still matched, with 1,320 timed rounds, a 42.560 ms median and a 42.996 ms mean. All page-offset medians were within 42.528–42.591 ms. The mean includes the first verification round's 566.892 ms startup spike; it is not removed from the record or mistaken for a recurring latency mode. [Post-deployment verification](benchmarks/results/attention-page-boundary-deployment-20260923.json).

The situation recorded in `cbbf495` had warm rounds around 43.6--43.8 ms but a repeatable fresh-cache state around 53.5--53.9 ms. A stream or device synchronization recovered roughly 6.4 ms, which narrowed the evidence to a residual HIP/ROCr stream or queue dependency but did not identify a permanent repair.

Since that diagnosis, [`abb7668`](https://github.com/Terrydaktal/vllm-coherence/commit/abb76682e96e1600e9b28ff404c36fa294244c54) made the runtime behavior and observation path explicit. It now drains pending device work after cache/mamba preparation, records each decode round without charging another chat's GPU time to it, and gives Pi one shared snapshot for scheduler, cache, temperature, round and acceptance telemetry. Its scheduler also preserves response ownership through generation, makes tool-call handover decisions at the intended boundary, and retires superseded cache generations safely.

[`b8d6810`](https://github.com/Terrydaktal/vllm-coherence/commit/b8d681001cc726089c387eeddfc7c78e2e74ac3c) carries the missing HIP event-gap and round-latency records and made backend failures retain a content-safe, expandable diagnostic report. These changes fix the previous lack of evidence and misleading Pi status; they do not make the underlying asynchronous queue issue mathematically solved. The M1/M8 and eager/compiled arithmetic repairs, Global-256 target-head change and GEMM performance backports were already present in the baseline documented by `cbbf495`; they are not new fixes after that commit.

The event collector now reclaims completed asynchronous HIP-event pairs and uses a separately managed marker pool. The previous monotonic ring and an unclosed sample-boundary marker could exhaust after about 64 rounds, causing later gap records to disappear; the repair is covered by an 80-round CPU telemetry test. Asynchronous rows are now held until their already-recorded HIP end events complete, then written with a round span and all available stage gaps; this adds no device synchronization to the serving path. The scheduler's recovery fence is adaptive: after the transition fence it triggers only when a previous round exceeds the recent baseline by at least 4 ms and 8%, so it does not manufacture a fixed-cadence spike. The generic and Radiance launchers both bound JIT checks to the current warmup log tail and repeat a warmup that compiled a new shape. Even `--reuse-existing` now performs that non-session warmup before Pi attaches, because a restarted backend must not expose first-use compilation to a chat; `QWEN_PI_SKIP_WARMUP=1` is an explicit diagnostic opt-out. Native-runtime validation of the repaired collector is still required for these paths.

The September 20 chained run did not reproduce the old 53--54 ms state: its measured generation intervals were 44.60--47.01 ms. That is evidence that the transition and adaptive recovery changes help, not proof that the slow state is impossible. The archived cbbf495 latency mode remains historical evidence; the current per-round results and the later page-boundary diagnosis are reported above. The completed event feed can now distinguish a GPU queue gap from host dispatch, cache transfer and telemetry wait; the new analyzer rejects dropped or incomplete rows. Round means also depend on workload and speculative acceptance, so the 47.01 ms thinking row alone is not a new kernel regression.

A September 21 content-free synthetic token-ID capture after the non-session warm-up is recorded in [round-steady-state-20260921.json](benchmarks/results/round-steady-state-20260921.json). Excluding the first verification row after each request boundary, the GPU round spans were 53.52 ms at 0K, 58.88 ms at 60K and 67.20 ms at 200K; the corresponding target-forward means were 44.39, 48.09 and 56.44 ms. The 200K rows ranged only from 67.15 to 67.25 ms, all three captures used the same PIECEWISE eight-token runtime descriptor, and no new inference-time JIT or dropped telemetry record occurred after warm-up. This capture excluded new first-use compilation during its measured window. It did not establish that natural Pi requests cannot alternate between fast and slow modes; the later page-boundary comparison above reproduces and repairs one such cause. In this synthetic capture, the event feed measured every named inter-stage GPU gap below 0.02 ms. This is a separate historical diagnostic, not a replacement for the matched stage-26 control: its disposable runtime and synthetic token-ID fixture are intentionally different from the authenticated stage-profile execution identity.

Two independent correctness/observability issues remain visible in this run: the provider did not expose a reasoning channel even when requested, and the checkpoint marker was present while the required section contract was absent. The first points to the provider/stream adapter's reasoning metadata path; the second is a checkpoint-format or model-compliance failure, not evidence of a cache-timing failure. Both should remain explicit failures in qualification.

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
`global256` profile uses faster, approximate candidate selection.

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
benchmarks/                    Original Radiance fixtures and upstream historical results
Dockerfile, patch_*, radiance_* Original base/build and focused upstream-facing changes
```

The `qwen_r9700_lab` namespace and wire-format names remain for evidence/snapshot
compatibility. Root Radiance Dockerfiles support upstream reproduction;
**`tools/coherence` launches the assembled Coherence profile**. Research drivers
require explicit inputs and do not run automatically during serving.

## Contributing

Submit a minimized failure or measured optimization with state/output comparisons,
negative controls and precise hardware/profile scope. Read
[CONTRIBUTING.md](CONTRIBUTING.md). Existing upstream PRs remain independent and are
linked from [ATTRIBUTION.md](ATTRIBUTION.md). See [LICENSE](LICENSE) for component terms.
