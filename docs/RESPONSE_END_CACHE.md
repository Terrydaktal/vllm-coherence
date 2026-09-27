# Exact response-end cache reuse

The DFlash cache now distinguishes a **processed prefix** from emitted tokens
and speculative queries. This removes repeated prefill after a completed tool
call when the next prompt extends the same token prefix. It does not reduce the
computation required for genuinely new tool output or a completely cold prompt.

## State contract

For a terminal target execution, let `s` be its starting processed-token count,
`n` the scheduled target rows, `d` the speculative rows, `g` the sampler's raw
output count before EOS/length trimming, and `v` the visible sequence length.

The committed endpoint is `p = min(s + n - d + g - 1, v)`. The final emitted
correction/bonus token may not have passed through the model and is excluded.
An inherited EAGLE rule excluded another complete 1,648-token block. DFlash V2
does not need that exclusion: it masks rejected target context and places query
KV strictly after the processed context. The EAGLE/MTP rules stay unchanged.

One endpoint per chat bank retains:

- The target and drafter KV blocks covering exactly `[0, p)`.
- A private canonical GDN state after the accepted prefix, selected from the
  existing speculative state blocks.
- Matching convolution history, shifted by the accepted-row offset.
- The processed-token count, salted full-prefix hash chain and exact partial
  token suffix. Token IDs are never added to telemetry or disk manifests.

The next request must extend that unchanged prefix. Its writable tails use the
existing copy-on-write path. Source and destination blocks stay pinned through
copy completion, including cancellation or cache reset. A matching GPU endpoint
does not wait for disk publication; existing transfer fences still protect
blocks from being overwritten while a store is reading them.

After the successor completes an advancing worker step, allocation pressure can
release the old endpoint's additional GPU references. Its own page and copy
references remain valid. Without pressure the endpoint stays available for
cancellation/retry. Keeping both sets pinned unconditionally throughout a long
successor caused repeated allocation failure and replay near the context limit.
The previous durable disk head remains valid until its replacement is verified.

For this exact local hand-off, admission reserves the first execution's space
rather than requiring the temporary checkpoint/copy pages to coexist with every
future prompt block. Each later execution still checks its allocation. Without
that distinction, a fitting near-limit continuation could wait forever before
its first step, even after the ownership-lifetime fix.

Under allocation pressure, the scheduler also releases optional historical GDN
snapshot states older than the native recurrence working set, then retries the
allocation once. It preserves the last processed state, speculative successors
and pages with pending offload reads. This can reduce available old GPU restore
points; it does not change attention, arithmetic, the current response or the
previous complete disk snapshot. Reclamation emits a bounded, content-free
`-cache-pressure.json` diagnostic only when it frees space.

The copy changes no floating-point arithmetic. Physical page sizes, the nine-slot
GDN layout, numerical kernels, sampling parameters and snapshot data ABI remain
unchanged. This is qualified for the pinned synchronous TP1 DFlash V2 profile,
with token-only input, full/sliding attention and aligned Mamba caching.

## RAM and disk

Exact endpoint keys have a separate hash domain. The manifest binds their count,
prefix identity, cache-group count and block/hash sizes. A complete head consists
of immutable full-attention prefix blocks, the required drafter window, and the
new endpoint blocks. It does not require obsolete aligned GDN checkpoints.

Changing endpoint blocks remain in the bounded RAM tail. Existing flush rules
still apply: about 8,192 new tokens, explicit flush, RAM eviction and clean
shutdown. Every dependency is verified before atomically publishing a replacement
disk head. Failed writes preserve the preceding complete head. Missing, damaged,
incompatible or prefix-mismatched state is a cache miss, never a usable checkpoint.
Older snapshots remain readable; they gain exact endpoint coverage after a new
successful response.

Fallback remains conservative for unsupported state layouts, insufficient free
blocks, cancellation, changed prefixes, and the rare stop inside a speculative
batch whose earlier convolution window was already overwritten by in-place
aligned postprocessing. Such a case reuses the older safe prefix instead.
Queued cancellation never tries to create or resolve a response-end bank: only
a completed execution with a successful stop/length boundary can publish one.

## Verification

`tests/test_response_end_cache.py` checks all D7 accepted widths, processed versus
emitted positions, stop truncation, namespace/prefix identity, block ownership,
reset-before-copy lifetime, both convolution layouts, exact state-copy bytes,
manifest validation and complete dependency lookup. Storage tests cover atomic
publication and rollback of endpoint metadata together with its blocks.
CPU regressions also cover successor ownership transfer, protection of a newly
published replacement, exact-boundary reclamation and pending-transfer lifetimes.

The `pressure` qualification mode uses synthetic 233K/239K/252K prompts. It first
disables the ownership repairs as a negative control, requires repeated native
allocator preemption, and then checks forward progress with the repairs enabled.
It reports capacity and liveness separately from numerical-equivalence evidence.

The [September 27 pressure regression](../benchmarks/results/response-end-pressure-20260927.json)
reproduced two allocator preemptions with the old policy and no output. With the
repair, 239,720- and 252,000-token prompts reached first output in 12.104 and
20.378 seconds, then generated 1,024 and 512 tokens without preemption. Each
started from a resident synthetic endpoint. Two warm and two cancelled/resumed
64-token continuations matched their uninterrupted controls exactly, reusing
all but five or six input tokens. These checks bind the installed source hashes.
If pressure evicts an old GPU endpoint, cancellation may still require restoring
its disk checkpoint or recomputing a suffix; the last complete disk head remains
valid.

Internal endpoint counters and host-memory-policy diagnostics remain in their
native phase and worker records but are excluded from Pi's stable v2 public
records. A producer-to-exporter-to-Pi regression checks the full scheduler/worker
and phase payloads, so round milliseconds and three-second acceptance still render. This
avoids breaking already-running host and VM clients when diagnostics expand.

`experiments/radiance-public/qualify_response_end.py` runs synthetic requests
against the installed compiled release. It compares interrupted/resumed output
with an uninterrupted control, and asserts actual cached-token counts. Modes
cover warm reuse, RAM handover, GPU-bank eviction, cancellation and a separate
container restart. Restart fixtures contain only synthetic token arrays; public
reports contain counts, hashes and timing measurements.

The September 27 native run reused 60,020 of 60,025 prompt tokens and 200,020 of
200,025 tokens: only the five appended tokens needed computation. First data
arrived in 0.221 and 1.026 seconds respectively on the resident GPU path. These
are synthetic continuation measurements, not a general latency guarantee.
Restarting the process still requires loading the snapshot; exact reuse removes
recomputation, not disk reads or allocation of a second chat's RAM bank.

The test-only `response_end_probe.ResponseEndProbe` extension checks canonical
GDN/convolution state and copy-on-write pages byte for byte. It is excluded from
the serving package. `qualify_response_end_state.py` also compares exact reuse,
aligned reuse and the legacy extra-block rule on independent synthetic requests.

### Separate numerical finding

Cold-prefilling an entire prompt is **not established as bit-equivalent** to
continuing state previously produced by decode. The arbitrary tool-suffix test
found its first generated-token difference at offset 15; a corresponding
difference also occurs with response-end reuse disabled. The cache-copy oracle
passed. No numerical kernel or sampling change was made to force this comparison
to pass, and the failing cold-prefill comparison remains in the evidence.

`--mode tool` deliberately checks that stronger, currently failing cold-prefill
equivalence. `--mode tool-repeat` checks independent executions with the same
prefill/decode boundaries, including production sampling (temperature 1,
top-p 0.95, top-k 40). Its result must not be substituted for the cold comparison.
The repair preserves the processed native state; it does not claim that all
alternative ways of computing that state have identical floating-point results.

Aggregate results, installed source identities, negative controls and limitations
are in [the qualification record](../benchmarks/results/response-end-cache-20260927.json).
The [production deployment receipt](../benchmarks/results/response-end-deployment-20260927.json)
records the installed compiled worker, five API smoke requests and a passing
exact-end continuation on the live backend. Existing Pi sessions use this backend
on their next request; an old snapshot gains the new endpoint after a successful
response. No numerical kernel or sampler change is needed.

These are finite regression and lifecycle checks. They do not prove arbitrary
model inputs, all asynchronous interleavings or the model's independent numerical
correctness.
