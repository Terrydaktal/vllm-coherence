# M1 arithmetic contract and prefill normalization repair

The 24 September 2026 operator qualification reproduced all **55 hidden-norm
FP8 differences**, traced their downstream effects, and removed them by using
the same arithmetic for prefill and M1. It also found and repaired the analogous
GDN gated-normalization reduction difference. Both repairs passed the expanded
release gates and were **deployed on 24 September**. The current release also
passes the fresh [320-token whole-model and stage confirmations](CURRENT_CONFIRMATIONS.md).
[Deployment identities](../benchmarks/results/eager-m1-normalization-deployment-20260924.json).

[Arithmetic contract](M1_ARITHMETIC_CONTRACT.md) ·
[machine-readable profile](../configs/profiles/m1-arithmetic-contract-v2.json) ·
[554-check result and timings](../benchmarks/results/eager-m1-contract-20260924.json) ·
[norm compiler/source/binary identity](../benchmarks/results/eager-m1-contract-norm-build-20260924.json)

## Cause and chosen arithmetic

The old hidden norm selected 512 logical reduction lanes for up to eight rows,
64 lanes for 9–15 rows, and 32 lanes for larger batches. Changing the FP32
addition order occasionally crosses a BF16 rounding boundary, which can then
cross an FP8 code boundary. Scales and residual carries were unchanged in the
original failures. The repair keeps **512-lane M1 arithmetic at every admitted
row count**, while still executing rows in parallel.

GDN gated normalization had a separate batch-dependent layout: four adjacent
components per lane in M1 versus eight in prefill. Its wrapper now retains the
M1 layout. Normalization, weight multiplication and SiLU gating stay in FP32,
with one final BF16 rounding before FP8. This deliberately preserves repaired
native M1 arithmetic; it does not insert Hugging Face's intermediate BF16 casts.

The contract explicitly accepts MXFP4 weights, FP8 activations and FP8 KV.
It specifies residual, normalization, activation and state rounding boundaries.
It is not a claim of equality to a weight-only quantized model. Compiler
intrinsics and unproved arithmetic details are identified rather than hidden
inside a numerical tolerance. Global-512 remains a separate approximate policy.

## Reproduction and propagation

| Check | Old path | Repaired candidate |
| --- | --- | --- |
| Hidden norm, 129 weights × 320 rows | 55 differing FP8 bytes at 40 sites, out of 211,353,600 | 0 differing bytes; scales and residual carries match |
| GDN norm, 48 weights × 320 rows | 9 differing FP8 bytes at 9 sites in this run, out of 94,371,840 | 0 differing bytes; scales match |
| Hidden-norm batch boundaries | Original divergence retained as a negative control | 24 checks pass: 12 row counts from 1 to 2,048, with/without residual |
| GDN-norm batch boundaries | Original divergence retained as a negative control | 6 checks pass: 1, 8, 9, 16, 320 and 2,048 rows |
| Reduced single-row counterexample | Adding 15 unrelated zero rows changes an FP8 byte | Batched and single-row bytes/scales match |

The 129 hidden-norm weights include the final norm as a diagnostic control;
serving does not FP8-quantize the final norm. The independent CPU implementation
reproduces the native M1 BF16 result for saved failing rows at all 40 sites.
The single-row reproducer retains 5,072 nonzero input coordinates after 48
coordinates were removed; it is a reduced example, not a globally minimal one.
Different seeded input schedules can produce different GDN failure counts; the
table reports the one combined run linked above.

The saved counterexamples were replayed through their actual checkpoint
projections. **All 14 traced GDN input sites changed convolution history,
recurrent state and recurrent output; 17 traced MLP sites changed the final
MLP output.** For example, the two-row layer-1 post-norm witness changes 7,905
of 10,240 MLP output values. These counts establish propagation, not its
task-level severity. No full-model logits, generated answers or loop frequency
were measured in this run.

The GDN trace uses active state slot 1, poisons output before execution and
requires the reserved slot 0 to remain untouched. An earlier probe used the
null slot and its state observations were discarded; the published result is
the complete rerun with these checks.

## Broader independent checks

| Coverage | Result and scope |
| --- | --- |
| Complete projections | All output channels of 30 matrices across layers 0, 15, 60 and 63; 32 inputs per matrix; 7,739,392 output values compared with independently reconstructed coefficients and CPU FP64 dot products |
| Saved numerical activation replay | 12 operator-output captures, 32 inputs each; 3,276,800 complete projection outputs checked |
| Attention contents/layouts | 96 cases: BF16 and OCP FP8 KV, 12 lengths through 60,001 tokens, random/uniform/peaked/cancellation contents, unique shuffled physical pages, holes, poisoned tails and per-head descales |
| Memory isolation | All 96 attention guard checks pass; unused pages and padded values remain invisible |

Projection inputs combine public checkpoint embeddings, seeded values,
one-hot controls and gated products. Saved activation replays use those
**operator-generated numerical outputs**, not private chats or complete-model
hidden-state captures. Full-model activation distributions remain a coverage
gap. The earlier longest-context attention tests still use structured pages;
the new unique-page cases extend through 60K, not the full 250K window.

Projection and general attention checks use relative-L2 limits of 0.004 and
0.005 respectively. They are diagnostic limits, not proved error bounds or
bitwise equivalence. Maximum observed relative-L2 errors were 0.001674 for
complete projections, 0.001661 for saved-activation replays and 0.002229 for
attention. Against a BF16-rounded FP64 oracle, 97 / 7,739,392, 490 / 3,276,800
and 547 / 589,824 output values respectively differ. Thus a numerical pass
must not be read as exact score agreement. Cancellation controls instead require
the exact BF16 rounding of the independent mean; comparison with an unrounded real-number
mean would wrongly count unavoidable output rounding as a defect.

The combined run reports **554 checks, zero failures**, in 63.5 seconds, with
252.6 MiB peak probe allocation. Forty of the checks record successful causal
traces; they are not equality assertions. The idle serving process was neither
restarted nor used for inference by the probe. Checkpoint tensor, source,
binary, profile and numerical-input hashes accompany the result.

## Operator cost before adoption

Milliseconds per call, median of 15 HIP graph replays containing 32 calls each,
on the same idle R9700. Inputs and output buffers are fixed; these are isolated
operator timings, not a full-model round or Pi throughput benchmark. Events
surround the graph, not individual calls. Old/candidate samples are retained
in the result; clock/scheduling effects remain possible.

| Operator | Rows | Residual | Old ms | Repaired ms |
| --- | ---: | --- | ---: | ---: |
| Hidden norm + FP8 | 1 | No | 0.008801 | 0.008781 |
| Hidden norm + FP8 | 1 | Yes | 0.010181 | 0.010160 |
| Hidden norm + FP8 | 8 | No | 0.008806 | 0.008783 |
| Hidden norm + FP8 | 8 | Yes | 0.010180 | 0.010160 |
| Hidden norm + FP8 | 320 | No | 0.042109 | 0.033460 |
| Hidden norm + FP8 | 320 | Yes | 0.065036 | 0.034520 |
| Hidden norm + FP8 | 2,048 | No | 0.148161 | 0.163360 |
| Hidden norm + FP8 | 2,048 | Yes | 0.259164 | 0.181614 |
| GDN gated norm + FP8 | 1 | — | 0.004674 | 0.004697 |
| GDN gated norm + FP8 | 8 | — | 0.004700 | 0.004746 |
| GDN gated norm + FP8 | 320 | — | 0.020258 | 0.021495 |
| GDN gated norm + FP8 | 2,048 | — | 0.107094 | 0.113628 |

M1/M8 arithmetic is unchanged by these two repairs; their measured differences
are tiny. The 2,048-row GDN case adds about 0.0065 ms per call (6.1%), and hidden
norm without residual adds 0.0152 ms (10.3%). Hidden norm with residual becomes
faster on the measured prefill cases. These mixed operator results do not
establish an end-to-end prefill speedup or slowdown. The README reports the
separate full-model measurements of the deployed release.

## Reusing the probe

Build the candidate hidden-norm library using the recorded compiler flags and
keep the old release available as the negative control. Inside that pinned
runtime, provide the public model and candidate build paths. The following
regenerates the broader suite and its operator captures; the archived reduced
witness can separately be supplied to `--stages minimal_norm`.

```bash
mkdir -p /tmp/m1-contract-run/numeric-activations
python experiments/radiance-public/probe_m1_contract.py \
  --release /qualification \
  --model /models/Qwen3.8-27B-Uncensored-MXFP4-awq \
  --norm-candidate /path/to/candidate-build \
  --gdn-candidate experiments/radiance-public/stock_gdn_norm_quant.py \
  --contract configs/profiles/m1-arithmetic-contract-v2.json \
  --stages norm_partition gdn_partition projection_full attention_varied norm_trace captured_replay \
  --witness-dir /tmp/m1-contract-run \
  --activation-dir /tmp/m1-contract-run/numeric-activations \
  --memory-mib 320 --allow-gpu \
  --output /tmp/m1-contract-run/result.json
```

The probe requires a fresh result path, idle backend admission and sufficient
free memory. Private numeric witnesses and binaries stay in the lab; public
results contain aggregate comparisons and identities. The CPU suite checks
the independent arithmetic, boundary handling and deliberate negative controls;
all 92 focused CPU tests pass.

Deployment used a rebuilt frozen release and a new snapshot arithmetic
identity. These operator repairs do not automatically inherit the older 10K M1/M8 alignment result or
prove full-model, snapshot, cancellation or concurrent-session correctness.

## Deployment

The qualified hidden and GDN normalization paths were deployed on 24 September. Additional native release gates passed at 128 hidden-normalization sites and 48 GDN sites with 1,000-row coverage under the declared row-invariant contract. The live server passed a public smoke request. [Release identities](../benchmarks/results/eager-m1-normalization-deployment-20260924.json). Full-model performance is measured separately in the README rerun; these operator checks do not establish arbitrary-input equality.
