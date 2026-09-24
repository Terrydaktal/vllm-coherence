# Global-256 versus Global-512 target head

Measured on 24 September after deploying the normalization-consistency release. All three methods received identical hidden inputs from **Global-512 generation**. This is an isolated head comparison, not a measurement of whole-model throughput or reasoning quality.

| Target head | Median M8 time | Same top-1 | Complete top-20 retained | Complete top-40 retained |
| --- | ---: | ---: | ---: | ---: |
| Global-256 | 1.155 ms | 12,015/12,015 (100.0000%) | 11,988/12,015 (99.7753%) | 11,742/12,015 (97.7278%) |
| Global-512 (default) | 1.178 ms | 12,015/12,015 (100.0000%) | 12,011/12,015 (99.9667%) | 11,983/12,015 (99.7337%) |
| Installed full BF16 head | 4.061 ms | 12,015/12,015 (100.0000%) | 12,015/12,015 (100.0000%) | 12,015/12,015 (100.0000%) |

Global-512 missed complete top-40 retention in 32 rows, versus 273 for Global-256. The added median head time was 0.023 ms. Retention includes cutoff ties; it does not establish identical ranking or sampling probabilities.

## Workload and measurement

- Coding: 60,208 input tokens, 10,047 output tokens, finish reason `stop`.
- Reasoning: 70,402 input tokens, 2,598 output tokens, finish reason `stop`.

The existing private 60K Pi prefix was consumed without decoding or displaying the conversation. Sampling was temperature 1, top-p 0.95, top-k 40, seed 24512. The first 768 head calls of each request supplied **12,015 prediction rows**, including prefill and rejected speculative rows. These are not generated-token counts. There were 39 M1 and 1497 M8 calls. Every recomputed Global-512 result matched the actual returned target logits; removing a winning token was detected by the negative control.

Timing uses 47 M8 hidden inputs and five warmed, randomized-order repetitions: 235 samples per method. HIP events surround the installed head functions, including native dispatch gaps. Comparison, reporting and observer work are outside those intervals. The target backbone retains the production compilation and graph settings.

## Score fidelity remains a separate limitation

- global256: 20 unequal retained logits; maximum absolute difference 0.0625; mean diagnostic total-variation distance 1.58592999e-05.
- global512: 39 unequal retained logits; maximum absolute difference 0.0625; mean diagnostic total-variation distance 1.11891741e-06.

The diagnostic distribution uses temperature 1, top-k 40 and top-p 0.95. Neither shortlist is certified complete for arbitrary inputs, and retained logits can differ from the installed full BF16 head. The full head is the reference for this comparison; it is not an independently proved complete model. This study does not measure changes in reasoning loops.

[Current numeric results](../benchmarks/results/head-candidate-depth-normalization-20260924.json) · [Earlier Global-256-generation comparison](../benchmarks/results/head-candidate-depth-20260924.json). Different continuations prevent treating recall differences between those studies as a causal model-quality score.

## Reproduction

Use an isolated copy of the exact frozen release. Preserve the release startup hooks, call `benchmark_head_candidate_depth.install()` from the additional startup hook, and supply an owner-only `/dev/shm/qwen-head-depth-*` directory in `QWEN_HEAD_DEPTH_BENCH_ROOT`. The client and its `benchmark_pi_coding_json_compaction.py` and `benchmark_pi_task_workloads.py` dependencies must be available together. The observer supports Global-256 or Global-512 generation and verifies that its recomputation matches the configured serving head. It restores the original depth after every comparison.

Keep snapshot storage, telemetry paths and startup receipts separate from production. Invoke [benchmark_head_candidate_depth.py](../experiments/radiance-public/benchmark_head_candidate_depth.py) with `--base-url`, `--root`, `--fixture`, `--tokenizer-json` and `--abi`. Reports contain source identities, aggregate comparisons and timing samples; no chat text, token arrays, hidden tensors or logits are exported.
