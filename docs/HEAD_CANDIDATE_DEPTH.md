# Global-256 versus Global-512 target head

Measured on 2026-09-24 on the R9700 using the current corrected, compiled serving
release. Enlarging the shortlist reduced incomplete top-40 retention by 79.2%,
with 0.022 ms added median M8 head time in this sample. Global-512 is now the
configured serving default. This comparison used identical hidden inputs from
Global-256 generation; it is not an end-to-end Global-512 throughput measurement.

| Target head | Median M8 head time | Same top-1 token | Complete reference top-20 retained | Complete reference top-40 retained |
| --- | ---: | ---: | ---: | ---: |
| Global-256 | 1.156 ms | 12,015/12,015 (100%) | 11,975/12,015 (99.6671%) | 11,620/12,015 (96.7124%) |
| Global-512 | 1.178 ms | 12,015/12,015 (100%) | 11,996/12,015 (99.8419%) | 11,933/12,015 (99.3175%) |
| Installed full BF16 head | 4.061 ms | 12,015/12,015 (100%) | 12,015/12,015 (100%) | 12,015/12,015 (100%) |

Retention includes every token tied at the reference cutoff. It differs from
equality of the arbitrarily selected top-k set or its order; all three metrics
are retained in the [numeric results](../benchmarks/results/head-candidate-depth-20260924.json).
Neither candidate depth establishes full-model correctness or a reduction in
reasoning loops.

## Workload and measurement

The existing private 60,000-token Pi prefix was consumed programmatically, without
decoding or displaying the conversation. A coding task began at 60,208 input
tokens and completed naturally after 5,812 output tokens. A following engineering
reasoning task began at 66,167 input tokens and completed naturally after 4,720
output tokens. Both used temperature 1, top-p 0.95 and top-k 40, with seed 24512.
The task labels describe the requests, not a certified partition of reasoning and
prose channels.

The observer checked the first 768 target-head calls of each task: 5,885 coding
and 6,130 reasoning prediction rows, including prefill and rejected speculative
rows. These **12,015 prediction rows are not 12,015 generated output tokens**.
There were 39 M1 invocations and 1,497 M8 invocations. The observed generation
returned the original Global-256 logits unchanged; all variants received exactly
the same hidden inputs. Each recomputed Global-256 result had to match the actual
returned target logits bit for bit. A deliberately removed winning token was
detected before accepting any measurements.

Timing uses 47 sampled M8 inputs, with five repetitions of each method per input:
235 measurements per method. All methods were warmed and their order randomized
within each repetition. HIP events surround the actual head calls and include
their native dispatch gaps. Accuracy checks, reporting and observer overhead
are outside the timed intervals. The target backbone retains production
compilation and graph settings. This is an isolated head comparison; it does not
measure a whole generation round or end-to-end tokens per second.

The mean paired increment was 0.02276 ms. A bootstrap resampling the 47 input
groups gives a descriptive 95% interval of 0.01837–0.02687 ms for that mean.
The 0.022 ms median increment would be about 0.05% of a 45 ms round if all other
costs and acceptance remained unchanged. This is an estimate, not a measured
throughput change.

## Score and sampling differences

Global-256 had 22 unequal retained logits among 3,075,840 rescored values;
Global-512 had 41 among 6,151,680. Both had a maximum absolute difference of 0.125
against the installed full head. The larger candidate set rescores twice as many
values, so those raw mismatch counts are not directly comparable error rates.

Using a diagnostic temperature-1/top-k-40/top-p-0.95 distribution, mean total
variation distance fell from 0.000009817 to 0.000000322. Global-512 omitted no
token with positive probability under that diagnostic reference distribution in
this sample. Its remaining score differences still prevent an exact-distribution
claim. Native sampler boundary tie handling is not certified by this diagnostic.

## Reproduction and scope

[benchmark_head_candidate_depth.py](../experiments/radiance-public/benchmark_head_candidate_depth.py)
contains the observer, sampler diagnostic and natural-task client.
[test_head_candidate_depth.py](../tests/test_head_candidate_depth.py) covers
missing candidates, reranking errors, cutoff ties, invalid support and aggregate
accounting. CPU validation: 123 passed, 19 GPU-only tests skipped; the native
measurements above are separate from those CPU tests.

Use an isolated serving instance with the exact frozen release, explicitly selecting
`--head global256` for the observed generation. Its early Python startup must call `benchmark_head_candidate_depth.install()` after preserving the
release's existing `sitecustomize.py`, and set `QWEN_HEAD_DEPTH_BENCH_ROOT` to an
owner-only `/dev/shm/qwen-head-depth-*` directory. Keep its snapshot/cache storage
and control paths separate from production. The client accepts `--base-url`,
`--root`, `--fixture`, `--tokenizer-json` and `--abi`. The numeric report records
source hashes, image digest, runtime ABI, workload boundaries and raw timing
samples. No chat text, token arrays, hidden tensors or logits are exported.

During the comparison, the observer changed candidate depth around a direct call
to the existing global-head implementation, restoring 256 in a `finally` block.
The serving configuration now admits 0/128/256/512 and defaults to 512. Its numeric
head implementation is unchanged. Existing processed-prefix snapshots remain
compatible because candidate selection happens after the backbone. The retained
comparison evidence is sampled qualification, not a universal correctness proof.

Deployment on 2026-09-24 passed 224 focused CPU checks and a public inference
request through the VM bridge. The worker reported `target global-512` and
executed that path; the snapshot data ABI was preserved. [Deployment receipt](../benchmarks/results/head-candidate-depth-deployment-20260924.json).
