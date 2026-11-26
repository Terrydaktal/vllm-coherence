# R9700 retained candidate benchmark, 26 November 2026

On one 32 GiB Radeon AI PRO R9700, the candidate generated **141.89, 109.09 and 102.84 tokens/s** after first output at 0K, 60K and 200K starting context. The published 9 October controls reached 120.06, 99.32 and 92.12 tokens/s. Against the earlier November candidate on identical input, throughput increased at 0K and 60K and fell from **109.92 to 102.84 tokens/s at 200K**. The final build produced the same output from cold and warm caches at all three lengths.

The measured worker manifest is `621b999abb596ae6ff572ae039be8b1856172719e1a6cccf72b71bf2f49ab780`. This was an isolated candidate run. The October serving measurements remain separately documented. The aggregate contains the full-precision timings, sample counts, histogram bins and correctness counts used in these tables. [Measured worker manifest](measured-worker-manifest.json) · [Numeric results](results.json) · [October release benchmarks](https://github.com/Terrydaktal/vllm-coherence/blob/206db15fdea3c0f85aad74bddd4f2a4b8c6a4562/README.md#benchmarks) · [Reproduction procedure](../../../docs/BENCHMARK_SUITE.md)

## Coding and chained requests

Generation used the compiled MXFP4 target with a Global-512 vocabulary head, an FP8 DFlash2 drafter and seven proposed tokens per round. The target used temperature 1, top-p 0.95, top-k 40 and seed 0. Both November runs used proposal temperature 0.55. Context labels denote the starting prefix before the task suffix. The post-first rate excludes the wait for the first output. Peak is the highest completed sliding three-second rate.

| Context | Final prompt / output tokens | Oct post-first tok/s | Nov earlier tok/s | Nov final tok/s | Final peak 3 s tok/s | Mean round ms, Oct / final | Acceptance, Oct / final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0K | 203 / 6,248 | 120.06 | 130.55 | 141.89 | 202.88 | 37.44 / 36.66 | 50.03% / 60.03% |
| 60K | 60,208 / 7,402 | 99.32 | 105.46 | 109.09 | 154.85 | 41.32 / 40.54 | 44.33% / 48.88% |
| 200K | 200,208 / 5,074 | 92.12 | 109.92 | 102.84 | 131.47 | 50.34 / 49.99 | 51.96% / 59.08% |

The October comparison spans changes to both serving code and proposal policy. The two November builds used the same compiler configuration, head, proposal temperature, seed and prompt tokens. The output trajectories differed across those builds at every length, so the tok/s comparison also reflects changes in speculative acceptance. The earlier November worker reproduced its cold output at 0K and diverged at 60K and 200K. Its manifest is `ecf694935ff6adad86f97dbfa26226b4a025f27192640e4bf14ba72cf98bd85a`.

| Chained stage | Final input / output tokens | First data s, final | Post-first tok/s, Oct / final | Peak 3 s tok/s, Oct / final | Mean round ms, Oct / final | Final acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Coding | 60,208 / 7,402 | 0.79 | 99.32 / 109.09 | 141.19 / 154.85 | 41.32 / 40.54 | 48.88% |
| Prose about code | 67,751 / 2,194 | 0.55 | 86.49 / 79.76 | 150.46 / 140.84 | 41.05 / 40.95 | 32.36% |
| JSON | 70,064 / 2,305 | 0.25 | 88.59 / 97.69 | 105.02 / 116.69 | 41.40 / 41.57 | 43.69% |
| Thinking/prose | 72,516 / 4,648 | 0.79 | 62.67 / 64.41 | 94.46 / 87.81 | 41.66 / 41.27 | 23.67% |
| Checkpoint | 77,350 / 4,049 | 0.31 | 70.02 / 69.03 | 96.25 / 83.62 | 42.13 / 41.70 | 26.83% |

The chain continued from the 60K coding request, so later inputs incorporated each build’s own prior output. The November JSON output failed parsing; October’s passed. The November checkpoint passed its completion-marker and heading checks. The checkpoint timing ends at generation and does not include a Pi session commit. Thinking-enabled requests exposed their output as prose in this capture.

## Round latency

November histograms cover every timed speculative round in the same unprofiled coding controls shown above. October used its separately published histogram run. Each context had one untimed event. The November coding run recorded 3,863 timed rounds, all below 90 ms.

| Context | Rounds, Oct / final | Median ms, Oct / final | p99 ms, Oct / final | Max ms, Oct / final | Final ≥90 ms | Final ≥500 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0K | 1,091 / 1,201 | 37.013 / 36.735 | 37.628 / 37.432 | 42.639 / 38.597 | 0 | 0 |
| 60K | 1,383 / 1,674 | 40.803 / 40.507 | 41.566 / 41.354 | 43.133 / 42.666 | 0 | 0 |
| 200K | 1,714 / 988 | 50.430 / 49.986 | 51.230 / 50.862 | 52.475 / 52.214 | 0 | 0 |

Separate diagnostic archives reported lost recorder events for 60K, 200K and all five chained stages. Archive rotation also failed to settle in the 200K, prose-about-code and JSON captures. The independent coding round counters matched all expected speculative rounds; the auxiliary archives provide incomplete coverage for diagnosing individual events. This run measures one workload and duration per context.

The [per-round record](round-records.json) contains all 3,866 round events, including the three untimed events.

## Cold prefill and cache restoration

Each prefill repeat started with zero cached tokens and requested one output token. Throughput divides the prompt length by client elapsed time, including first-output overhead. The November stock and candidate workers ran in separate processes with the same model, fixture, scheduler limits and 327,952-token maximum. All four runs per length returned the same first token.

| Prompt tokens | Oct mean tok/s | Nov stock tok/s, both runs | Nov final tok/s, both runs | Mean rate change, stock to final | Backend prefill s, stock | Backend prefill s, final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 60,000 | 1,791.4 | 1,806.5 / 1,754.4 | 1,837.1 / 1,770.6 | +1.3% | 33.14 / 33.55 | 32.59 / 33.14 |
| 200,000 | 996.7 | 994.3 / 996.2 | 1,000.7 / 1,002.6 | +0.6% | 200.40 / 200.10 | 199.13 / 198.82 |

For restart testing, a fresh server loaded a durable chat checkpoint and generated 64 tokens. All measured restarts matched their own independent cold continuation. Each latency below is a single request. The two shorter matched runs reused one fewer token in the final build.

| Prompt tokens | Reused, earlier / final | First output s, earlier / final |
| --- | ---: | ---: |
| 1,675 | 1,670 / 1,669 | 0.526 / 0.491 |
| 3,325 | 3,320 / — | 0.556 / — |
| 60,024 | 60,019 / 60,018 | 1.975 / 1.794 |
| 200,024 | 200,019 / 200,019 | 5.122 / 5.616 |

The earlier November build supplied the only 3,325-token restart. In the final encoding tests both raw and Zstd restored 147 tokens. A deliberately damaged payload reused zero and regenerated the expected output. Four tool continuations, with the chat active or RAM-parked at 256 and 1,648 tokens, also matched all 64 cold-control output tokens. The generated-token API admitted saved history on the second request.

## Compiled stage times

Each cell gives the mean in milliseconds at **0K / 60K / 200K**. Rows 1–25 measure GPU kernels on complete eight-row verify cycles. The matched sample counts were 1,003 / 1,134 / 1,134 in October and 618 / 961 / 760 in November. Row 26 estimates the remaining time by subtracting the union of profiled GPU activity from the worker interval in separate unprofiled controls over the same decode indices. Kernel overlap makes this different from subtracting the sum of rows 1–25. Profiling can also affect scheduling and clocks.

| Stage | October ms (0K / 60K / 200K) | November candidate ms (0K / 60K / 200K) |
| --- | ---: | ---: |
| 1. Drafter | 4.799 / 4.912 / 4.946 | 4.815 / 4.860 / 4.921 |
| 2. Embedding + first input normalization + FP8 production | 0.009 / 0.009 / 0.009 | 0.012 / 0.012 / 0.012 |
| 3. Layer input residual/normalization + FP8 production | 0.483 / 0.495 / 0.511 | 0.478 / 0.492 / 0.511 |
| 4. GDN input projection | 4.030 / 3.964 / 3.983 | 3.984 / 3.935 / 3.958 |
| 5. GDN layout/copies and buffer initialization | 0.193 / 0.200 / 0.204 | 0.132 / 0.138 / 0.140 |
| 6. GDN convolution | 0.178 / 0.182 / 0.187 | 0.176 / 0.180 / 0.186 |
| 7. GDN recurrence and gates | 1.336 / 1.177 / 1.209 | 1.303 / 1.175 / 1.217 |
| 8. GDN output gated normalization + FP8 production | 0.145 / 0.147 / 0.151 | 0.144 / 0.147 / 0.152 |
| 9. GDN output projection | 1.827 / 1.810 / 1.815 | 1.801 / 1.799 / 1.809 |
| 10. Attention input projection | 1.159 / 1.159 / 1.155 | 1.135 / 1.146 / 1.143 |
| 11. Attention Q/K normalization, RoPE and layout | 0.208 / 0.212 / 0.217 | 0.209 / 0.213 / 0.220 |
| 12. Attention KV write | 0.034 / 0.035 / 0.035 | 0.033 / 0.034 / 0.035 |
| 13. Attention decode | 0.371 / 4.754 / 13.960 | 0.495 / 4.702 / 13.895 |
| 14. Attention split-KV merge | 0.260 / 0.133 / 0.107 | 0.266 / 0.131 / 0.106 |
| 15. Attention output gating | 0.021 / 0.025 / 0.025 | 0.021 / 0.026 / 0.026 |
| 16. Attention output activation FP8 quantization | 0.036 / 0.042 / 0.043 | 0.036 / 0.042 / 0.043 |
| 17. Attention output projection | 0.585 / 0.535 / 0.514 | 0.579 / 0.528 / 0.509 |
| 18. Post-attention/GDN residual/normalization + FP8 production | 0.495 / 0.510 / 0.527 | 0.491 / 0.508 / 0.525 |
| 19. MLP gate/up projection | 11.431 / 11.150 / 11.144 | 11.274 / 11.057 / 11.069 |
| 20. MLP SiLU and gating | 0.122 / 0.134 / 0.137 | 0.118 / 0.124 / 0.126 |
| 21. MLP down input FP8 quantization | 0.242 / 0.255 / 0.262 | 0.240 / 0.254 / 0.262 |
| 22. MLP down projection | 5.464 / 5.313 / 5.322 | 5.384 / 5.266 / 5.284 |
| 23. Final normalization/layout | 0.012 / 0.012 / 0.012 | 0.011 / 0.011 / 0.012 |
| 24. Global-512 target head | 1.015 / 1.020 / 1.050 | 0.928 / 0.931 / 0.961 |
| 25. Other GPU bookkeeping | 0.471 / 0.480 / 0.493 | 0.460 / 0.467 / 0.477 |
| 26. Estimated runtime overhead | 2.279 / 2.465 / 2.269 | 2.524 / 2.403 / 2.601 |
| GPU kernel sum, rows 1–25 | 34.924 / 38.663 / 48.016 | 34.523 / 38.176 / 47.599 |
| Union of GPU activity | 34.921 / 38.661 / 48.013 | 34.282 / 38.066 / 47.419 |
| Full unprofiled worker interval | 37.201 / 41.126 / 50.282 | 36.806 / 40.468 / 50.020 |

All three November speed arms at each context produced matching token IDs and accepted-token schedules. Stage matching retained complete eight-row cycles with a closing worker-entry timestamp. The round histogram includes partial-width and untimed events separately. Shape counters for synthetic 129-, 320- and 1,024-token requests recorded respectively 80 thin-tile, 80 thin-tile and 80 retiled launches.

The [stage capture](stage-detail.json) includes the matched control samples and the timings for all 64 layers and 168 kernel groups per context.

## Head latency and numerical results

The isolated head benchmark timed 240 eight-row calls per method. Candidate accuracy was evaluated on 12,141 prediction rows, including speculative rows that generation later rejected. The reference for this comparison was the full BF16 head on the same hidden vectors.

| Head | M8 median ms, Oct / final | Final top-1 exact | Final top-20 order exact | Final top-40 order exact |
| --- | ---: | ---: | ---: | ---: |
| Global-256 | 1.139 / 1.037 | 12,141 / 12,141 | 12,127 / 12,141 | 11,969 / 12,141 |
| Global-512 | 1.168 / 1.059 | 12,141 / 12,141 | 12,137 / 12,141 | 12,120 / 12,141 |
| Full BF16 | 4.057 / 4.068 | 12,141 / 12,141 | 12,141 / 12,141 | 12,141 / 12,141 |

Global-512 matched top-1 on all 12,141 rows, missed the reference top-20 order on four, and had 39 differences in individual retained logit values. Those counts limit what can be inferred about native sampled output from exact body comparisons.

| Numerical comparison | Observed result |
| --- | --- |
| Final full-head eager and compiled, M1 and M8 | All four mode comparisons matched the prefill prediction and all 320 full decode-logit vectors on the same trajectory. All 22 target boundaries passed across 40 eight-row groups, including negative controls. |
| November stock release vs final full head | On the same 320 forced positions, top-1 matched 320 times. Full-vocabulary logits matched 0 times. Ordered top-10 matched 27 times and ordered top-20 matched 0 times, with mean overlap 18.525 of 20. |
| Final cold prefill vs final forced M8 | All 20 short-width cases passed. The three longer cases at 1,651, 60,000 and 200,000 context matched every hidden vector and full logit row, across 1,000 + 1,000 + 320 positions. |
| Final M8 vs archived M8 | Full logit rows matched 0 of 2,320 on the archived trajectory. Top-1 matched 991/1,000, 989/1,000 and 317/320 at those three lengths. |
| Final cache and partition boundaries | All 17 checks passed. The cold/warm target compared 2,528 hidden rows starting at token 57,680. Sampled and greedy runs generated 128 matching IDs each. Short native/forced graph and logical KV checks, plus sampled cancellation and the queued request, passed. |

The source-bound operator receipts record the measured Q/K normalization and fixed-tile DFlash context projection used by the candidate. Both files retain the original seals accepted by the numerical amendment tool. Source and build digests identify what ran; rerunning the controlled comparisons requires the corresponding model, ROCm build and fixtures. [Q/K normalization receipt](qk-norm-qualification.json) · [DFlash context projection receipt](draft-context-qualification.json)

The [whole-model](whole-model-confirmations.json) and [stage replay](stage-confirmations.json) captures preserve the candidate's source and fixture bindings. The independent repeat passed 2,817 checks on base release `08f7c01c` with 320 rows per sampled site. Its checks cover the release primitives with declared tolerances, sampled projection channels and structured long-context attention pages. [Independent operator results](operator-confirmations.json).

The retained indexed GDN bias wrapper passed 480 independent checks across 48 layers, including 384 state comparisons and guard-slot checks. Maximum absolute error against the FP64 oracle was 0.000244 for output and 3.06 × 10⁻⁷ for state. [Retained GDN bias audit](retained-gdn-bias.json).

## Additional cache qualification

The same worker passed all 10 scheduler cases, including sampled cancellation and replay with another request queued. Six 64-token continuations matched their controls in each of five modes covering resident reuse, RAM handover, bank eviction, explicit stop and cancellation. These cases use synthetic prompts. The capture records first-data and scheduler-release times separately. [Cache and scheduler results](lifecycle.json).

GPU-resident continuations with five new prompt tokens reached first data in 0.261 s at 60K and 0.304 s at 200K. The instrumented state-copy check matched all 576 recurrent and convolution buffers and 144 physical page copies. [Cache timing and byte-copy results](lifecycle.json).

The near-limit requests at 313,880 and 326,160 input tokens completed with zero observed allocator preemptions. Each reused 307,342 tokens. First data took 15.633 s and 45.161 s. The old-policy control did not reproduce repeated preemption, so the overall comparison did not qualify. [Pressure results](lifecycle.json).

All three generation replacements kept the previous disk head until verification and collected it after publication. The second chat's stored head stayed intact. Activation took 5.27–6.49 ms and request-to-durable publication took 2.844–2.998 s for the 4,000-token replacement prompts. This measures the snapshot transaction used by compaction. [Snapshot replacement](snapshot-replacement.json).
