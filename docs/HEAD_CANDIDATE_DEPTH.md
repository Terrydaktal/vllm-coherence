# Global-256 versus Global-512 target head

Measured on 9 October 2026 using the frozen [`99d5fcf`](https://github.com/Terrydaktal/vllm-coherence/commit/99d5fcfd1bcd0fe6f70445d154e63b7092b02f24) source tree and its FULL-graph target path. All three methods received identical hidden inputs from **Global-512 generation**. This is an isolated head comparison, not a measurement of whole-model throughput or reasoning quality.

| Target head | Median M8 time | Same top-1 | Complete top-20 retained | Complete top-40 retained |
| --- | ---: | ---: | ---: | ---: |
| Global-256 | 1.139 ms | 12,148/12,148 (100.0000%) | 12,130/12,148 (99.8518%) | 11,893/12,148 (97.9009%) |
| Global-512 (default) | 1.168 ms | 12,148/12,148 (100.0000%) | 12,143/12,148 (99.9588%) | 12,118/12,148 (99.7530%) |
| Installed full BF16 head | 4.057 ms | 12,148/12,148 (100.0000%) | 12,148/12,148 (100.0000%) | 12,148/12,148 (100.0000%) |

Global-512 missed complete top-20 retention in 5 rows and complete top-40 retention in 30 rows. Global-256 missed complete top-40 retention in 255 rows. The added median head time was 0.029 ms. Retention includes cutoff ties; it does not establish identical ranking or sampling probabilities.

## Workload and measurement

- Coding: 60,208 input tokens, 7,748 output tokens, finish reason `stop`.
- Reasoning: 68,103 input tokens, 6,825 output tokens, finish reason `stop`.

The existing private 60K Pi prefix was consumed without decoding or displaying the conversation. Sampling was temperature 1, top-p 0.95, top-k 40, seed 24512. The first 768 head calls of each request supplied **12,148 prediction rows**, including prefill and rejected speculative rows. These are not generated-token counts. There were 20 M1 and 1,516 M8 calls. The observer requires recomputed Global-512 logits to match the actual returned target logits; removing a winning token was detected by the negative control.

Timing uses 48 M8 hidden inputs and five warmed, randomized-order repetitions: 240 samples per method. HIP events surround the installed head functions, including native dispatch gaps. Comparison, reporting and observer work are outside those intervals. The target backbone uses the compiled FULL graph; the report records its frozen source and release identities. The normal Pi server was not modified by this head comparison.

## Score fidelity remains a separate limitation

- global256: 15 unequal retained logits; maximum absolute difference 0.0625; mean diagnostic total-variation distance 6.5078482e-06.
- global512: 39 unequal retained logits; maximum absolute difference 0.0625; mean diagnostic total-variation distance 6.210417e-07.

The diagnostic distribution uses temperature 1, top-k 40 and top-p 0.95. Neither shortlist is certified complete for arbitrary inputs, and retained logits can differ from the installed full BF16 head. The full head is the reference for this comparison; it is not an independently proved complete model. This study does not measure changes in reasoning loops.

[Current numeric results](../benchmarks/results/head-candidate-depth-20261009.json) · [25 September comparison](../benchmarks/results/head-candidate-depth-20260925.json) · [Earlier Global-256-generation comparison](../benchmarks/results/head-candidate-depth-20260924.json). Different continuations prevent treating recall differences between those studies as a causal model-quality score.

## Reproduction

Use an isolated copy of the exact frozen release. Preserve the release startup hooks, call `benchmark_head_candidate_depth.install()` from the additional startup hook, and supply an owner-only `/dev/shm/qwen-head-depth-*` directory in `QWEN_HEAD_DEPTH_BENCH_ROOT`. The client and its `benchmark_pi_coding_json_compaction.py` and `benchmark_pi_task_workloads.py` dependencies must be available together. The observer supports Global-256 or Global-512 generation and verifies that its recomputation matches the configured serving head. It restores the original depth after every comparison.

Keep snapshot storage, telemetry paths and startup receipts separate from production. Invoke [benchmark_head_candidate_depth.py](../experiments/radiance-public/benchmark_head_candidate_depth.py) with `--base-url`, `--root`, `--fixture`, `--tokenizer-json` and `--abi`. Reports contain source identities, aggregate comparisons and timing samples; no chat text, token arrays, hidden tensors or logits are exported.
