# Eager-M1 follow-up repairs — 24 September 2026

All five findings below now have implemented repairs. The original evidence is
retained as the negative control. [Repair results and identities](../benchmarks/results/eager-m1-followup-repairs-20260924.json)
record the new operator qualification separately from the historical full-model
alignment study.

| Repair | Verification |
| --- | --- |
| Keep split outputs and normalization sums in FP32 until final BF16 output | All four constructed split-mean failures now give the correct BF16 result. |
| Preserve BF16 query/key range; apply scaling after the FP32 dot product | The tiny-query counterexample is repaired in both decode and prefill. |
| Preserve BF16 cache range in the value path | Single-key `2^-30` and `2^20` values are returned exactly in both phases. |
| Enforce packed inner GDN strides and non-overlapping slots | Actual wrapper rejects padded/overlapping layouts; packed and outer-padded layouts remain admitted. No recurrence arithmetic or state-layout change. |
| Separate FNUZ from OCP in the serving selector and validate tensor dtypes on every call | FNUZ is refused, including after geometry was cached. Valid OCP/BF16 calls still work. |

The probability/value multiplication uses a high term plus a residual term,
avoiding loss of the small probability difference after the query repair. FP8
cache values widen exactly to FP16; BF16 cache values keep their BF16 range.
This remains finite-precision attention, not a universal exact softmax.

Qualification of the optimized binaries:

- **22/22 independent witness/phase cases pass**, including all original failures
  and controls, with output and scratch guards intact.
- **2,112/2,112 synthetic query rows match corrected M1 exactly in M8**, including
  graph replay, every page offset, split-size boundaries, and three cache layouts
  at short, 60K and 200K contexts. Rejected/future queries cannot alter the first row.
- **640 further query rows**, each run through decode and prefill, pass the
  independent dense CPU FP64 comparison. Every tested case has lower RMS error
  than the original kernel. These use synthetic activations, not private chats.
- **105 focused CPU tests pass**, covering admission, prior repairs, snapshot
  identities, packaging and attention adapters. An isolated installation checks
  the actual four installed bindings and their 6,340,608-byte eight-query scratch.

| Context | Previous shared M8 attention, ms/layer | Repaired shared M8 attention, ms/layer | Estimated change across 16 attention layers |
| --- | ---: | ---: | ---: |
| 60K | 0.2489 | 0.2813 | +0.518 ms/round |
| 200K | 0.7537 | 0.8782 | +1.992 ms/round |

These are paired optimized graph measurements: 30 repetitions at each of 16 page
offsets, averaged over the per-offset medians. The final column is a sum estimate,
not a newly measured full-model round time or t/s result. The prior page-boundary
traversal repair, GEMM backports, existing GDN layout and Global-512 head remain.

The repaired arithmetic has a new snapshot data identity. Existing chats need
one fresh prefill; prior snapshots are retained for rollback. Earlier 10K
full-model results do not qualify this new arithmetic.

## Original counterexamples

Native R9700 checks confirmed the additional precision losses in attention.
CPU checks also found two conditional input-handling defects. This does not establish that these caused
chat loops or measure their frequency in real activations. The earlier
[MXFP4 and GDN-softplus repairs](eager-m1-independent-audit.md) are already
present in the inspected runtime and are not counted again.

| Finding | Evidence obtained | Relevance to the current backend |
| --- | --- | --- |
| Attention narrows normalized split results to FP16, with rounding toward zero | Native GPU execution returns `1.0` instead of the correctly rounded BF16 `1.0078125` in four cases, differing in all 6,144 output elements per case; three controls agree | This conversion is in the active FP8-cache M1 attention path. It is a deliberate extra approximation, not loss from storing the input KV in FP8. |
| Attention scales queries and then narrows them to FP16 | The GPU loses a representable BF16 query: a two-key example returns `0` rather than BF16 `0.0008544921875` in all 6,144 output elements | Same active arithmetic path; a synthetic range counterexample, with no evidence yet of its occurrence in real model activations. |
| Packed M1 GDN admits unsupported inner state strides | The actual installed Python wrapper accepts a padded state view; the kernel's address formula selects 393,216 wrong values out of 786,432. Packed control: zero wrong values. | Conditional bug. The qualified packed layout avoids this case; this audit did not observe a padded state in a live request. No GPU kernel was launched. |
| R4D's selector aliases FNUZ FP8 to OCP FP8 | The installed native selector chooses the same kernel for both formats, although byte `0x38` means `0.5` in FNUZ and `1.0` in OCP | Conditional format bug. The current gfx1201 platform selects OCP E4M3, so this is not evidence of a live cache-format error. |
| BF16-cache attention loses BF16 exponent range internally | Native GPU execution maps `2^-30` to zero and `2^20` to a final BF16 value of `65,536`; the `1.0` control matches | Conditional range defect in the advertised BF16-cache path; the current server uses FP8 KV. |

## Why the attention examples are independent

For the mean witnesses, every query and key is zero. Every attention weight is
therefore equal, and the correct answer is simply the arithmetic mean of the
values. The values are `1` and `1.125`, both exactly representable in the stored
FP8 format. There is no quantization error in these inputs.

The kernel computes a normalized result for each KV split, stores that result
as FP16 with rounding toward zero, and combines the stored results into BF16.
For lengths 31, 511, 1,023 and 4,095, the chosen exact mean is just above a BF16
rounding midpoint. The extra intermediate narrowing loses enough information
to round the final answer down. Lengths 16, 32 and 512 provide matching controls.

This is a one-BF16-step output difference in a constructed example, not a
0.78% estimate of model-quality loss. The source explicitly chooses reduced
precision for speed. These results refute an exact-arithmetic claim for that
choice; they do not refute an independently specified tolerance contract.

The query witness uses BF16 `q = 2^-22`, a zero key and a key whose 256 values
are all `448`, with values `-1` and `+1`. These are also exact stored inputs.
The expected output is `tanh((256 × 2^-22 × 448 / 16) / 2)`. Folding the query
scale into a later FP16 conversion makes every query component zero instead.

M1/M8 agreement alone could not detect these shared conversions. The repair
checks both paths against the independent cases and gives them a new cache identity.

## Conditional input-handling defects

The packed GDN wrapper checks only `initial_state.stride(-1) == 1`. Its kernel
uses an outer slot stride but assumes inner strides `(V*K, K, 1)`. A view with
inner strides `(32768, 256, 1)` passes the wrapper for the production head
geometry even though the kernel assumes `(16384, 128, 1)`. The reproducer
executes the actual wrapper with a launch recorder, then compares the kernel's
address calculation with the tensor's logical values. It does not emulate a
whole GPU recurrence or claim that the live allocator produced this layout.

The repair rejects unsupported inner strides. The current packed layout needs
no arithmetic change.

The R4D selector's FNUZ alias can similarly be rejected without changing valid
OCP requests. FNUZ also interprets byte `0x80` as NaN, whereas OCP interprets it
as negative zero; the formats cannot safely share a raw-byte decoder.

## Validation and limits

- Fifteen CPU diagnostic tests passed, including exact rational mean oracles,
  representable-input checks, rounding boundaries, packed/padded layout
  controls, FP8 format distinctions and refusal of busy/unknown server state.
  The standalone mode accepts connection refusal, not an arbitrary timeout or
  API error, as evidence that the model server is stopped.
- The native selector and repaired M1 source hashes were read from the live
  container. The packed recurrence hash matches the repaired source under test.
- The server was gracefully stopped while idle, with no pending snapshot tails.
  The probe then ran the exact installed `r4d.so` in an isolated container. All
  seven counterexamples reproduced and all four controls matched, with intact
  output/scratch canaries. The eleven cases completed in 0.359 seconds; this is
  diagnostic elapsed time, not a kernel-performance benchmark. The same
  Global-512 release was restarted afterwards.
- Each case has 6,144 output elements (24 heads × 256 values). The same witness
  is repeated across those elements; this is not thousands of independent
  natural model inputs or a measured chat error rate.
- No private transcripts, generated chat content or model prompts were read.

No additional recurrence-equation defect was established by this pass. The
previous gated-normalization/HF difference remains a choice of finite-precision
contract, not automatically a reason to add the HF rounding steps. Global-512
also remains an approximate shortlist with the previously published recall
measurements; increasing it did not establish universal full-head equivalence.

## Reproduction

[Numeric results and source identities](../benchmarks/results/eager-m1-followup-20260924.json)
contain no chat data. The [captured wrapper](../benchmarks/fixtures/eager-m1-followup-20260924/packed_decode.py.txt)
retains its upstream notices and has a hash-bound extraction manifest.
The [native attention results](../benchmarks/results/eager-m1-attention-native-20260924.json)
record every case, expected/actual values, guard checks and the tested binary hash.

The CPU audit expects the pinned libr4d source directory and the repaired
installed `fused_recurrent.py`; it refuses different source hashes:

```bash
uv run --no-sync python experiments/radiance-public/probe_m1_followup_cpu.py \
  --attention-source /path/to/pinned/libr4d \
  --packed-source /path/to/installed/fused_recurrent.py \
  --output /tmp/m1-followup-new
uv run --no-sync pytest -q tests/test_m1_followup_audit.py
```

The [native attention probe](../experiments/radiance-public/probe_m1_attention_precision.py)
needs a running idle metrics endpoint and sufficient free VRAM, or an explicitly
reserved window with `--backend-stopped`. It uses private
allocations of less than 9 MiB per case, plus the separate HIP context, checks
output/scratch canaries, and verifies the installed `r4d.so` hash. CPU-only mode
is the default; native execution requires `--allow-gpu` and `--library`.
An idle observation does not reserve the production scheduler, so run its
native arm in a reserved qualification window before relying on the result.

The completed native run used the pinned image and ROCr overlay, with the
model server stopped, and this command inside the isolated container:

```bash
/opt/vllm/bin/python /evidence/probe.py --allow-gpu --backend-stopped \
  --library /opt/vllm/lib/python3.12/site-packages/r4d.so \
  --output /evidence/native-v1
```
