# Stage timing and runtime gaps

The current stage, layer and kernel tables use the September 24 compiled **Global-512** release with the eager-M1 arithmetic and normalization repairs. The separate coding histogram retains all **2,818 timed rounds**, plus three explicitly untimed first-token events. Traced rounds are never presented as ordinary Pi latency.

## Matched native measurements

All values below use the same retained decode indices in each arm. A full natural warmup precedes control → trace → control. Output-token hashes, accepted-token sequences and scheduled shapes agree across the three arms. Sampling is temperature 1.0, top-p 0.95, top-k 40, seed 0; EOS remains enabled. The existing manual GPU mode, −50 mV offset and 300 W cap were retained.

| Starting context | Retained M8 cycles per arm | Clean before, ms | Traced, ms | Clean after, ms | GPU occupied union, ms | Estimated remainder, ms | Before/after remainder range, ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0K | 870 | 39.002 | 47.810 | 39.015 | 36.901 | 2.107 | 2.101–2.114 |
| 60K | 903 | 43.096 | 48.716 | 43.095 | 40.669 | 2.426 | 2.425–2.426 |
| 200K | 752 | 52.609 | 55.993 | 56.125 | 49.925 | 4.442 | 2.685–6.200 |

Tracing added 8.802 / 5.621 / 1.626 ms per retained round relative to the paired clean mean. This observed difference is reported separately and is not added to the clean-control residual. The before/after range measures repeat variability; it is **not** a confidence interval or a bound on every possible observer effect.

The later 200K control retained three consecutive long intervals: **2,555.045, 131.085 and 177.172 ms**. Its median was **52.519 ms**, compared with **52.620 ms** before tracing. These pauses raise its retained mean to 56.125 ms and the paired residual estimate to 4.442 ms. They were not removed by a duration cutoff. Their cause is unresolved: this observation does not establish a new persistent kernel regression, nor prove that earlier tracing had no indirect effect on the later control. The earlier control alone gives a 2.685 ms remainder; both controls remain visible.

GPU activity is clipped at the worker-entry boundaries also timed by the controls. Mean trace-marker/host-clock differences were at most 0.000007 ms; the largest individual difference was 0.033359 ms. CPU annotations, Python hooks and trace export are never added to a stage. GPU overlap is counted once: its mean was 0.002 / 0.004 / 0.017 ms. Controls retain ordinary production telemetry and one host boundary clock/record; they contain no tensor reads, forced sampling, stage events or additional synchronization.

The capture requested 1,152 traced rounds per context in bounded chunks. Natural completions contained 4,512 / 4,788 / 4,030 output tokens. First/last chunk boundaries, partial-width steps and structurally incomplete or inconsistent trace inventories are excluded from M8 stage attribution independently of duration. Both valid short-context attention variants are included. One 200K cycle had inconsistent normalization/projection timestamp order; its [minimal numeric witness](../benchmarks/results/trace-stage-order-witness-20260924.json) is retained. Every omitted inventory is recorded, all native rounds remain captured, and no missing operation is assigned zero time.

The expandable 60K layer and kernel tables use the **same 903 retained cycles** as the 60K stage table. Kernel activity counts include boundary-clipped fragments; their durations reconcile exactly to the stage sum. This replaces the former six-cycle detail without another GPU run.

The page-policy observations show no pinned handover allocation at 0K/60K and a protected allocation at 200K. Runtime source hashes and the actual page-policy byte counts accompany the numerical build identity.

Indirect profiler effects on clock speed, kernel execution and scheduling remain an uncertainty. GPU activity durations exclude direct CPU profiling cost, but cannot establish zero disturbance for each kernel. [PyTorch documents profiler overhead](https://docs.pytorch.org/tutorials/beginner/profiler.html).

[Profile and identities](../benchmarks/results/compiled-global512-stage-profile-20260924.json) · [clean controls](../benchmarks/results/stage26-control-20260924.json) · [residual calculation](../benchmarks/results/matched-stage-residual-20260924.json). Raw traces and private token fixtures remain outside the public repository. The normal backend was restored after capture.

## Superseded measurement

The [archived audit](../benchmarks/results/stage-timing-audit-20260923.json) still rejects the earlier row-26 result. That control retained forced-replay copies, scalar reads and output writes, used a different round boundary, and was not paired by accepted-token schedule. The old stage capture also predates the attention-page repair. Its arithmetic remains preserved, not promoted to a production gap.

## Accounting contract

For a traced round with duration `T`, let `K` be all its GPU kernel and transfer
intervals, with complete stage attribution:

```text
sum of stage durations = sum(duration(k) for k in K)
GPU occupied time      = duration(union(K))
overlap                = sum of stage durations - GPU occupied time
observed traced gaps   = T - GPU occupied time
```

Observed traced gaps can include profiler effects. They must not be labelled
as unprofiled gaps. A separate control supports only this estimate:

```text
estimated unprofiled remainder = mean(unprofiled round) - mean(traced GPU occupied time)
```

This requires matching runtime artifacts, the same compiled target-head execution,
prefixes, sampling, round boundaries, positions and accepted-token schedules.
Use ordinary natural generation without forced replay, added synchronization
or per-stage event probes. Warm every measured shape before capture and retain
every round in the declared window. Record clean controls before and after
the trace, plus the profiled round samples; keep the observer delta separate.

A total observer delta cannot be allocated to individual kernels or used to
prove that each kernel was unaffected: changes can cancel. Do not distribute
it across stages or subtract a guessed per-event cost. Report this limit even
when the paired round totals agree.

The HIP scope-event diagnostic is not a substitute for kernel activity
timestamps. An event interval around a Python scope can contain dispatch gaps
and event-recording effects even when no device synchronization is inserted.

## Enforced checks

`tools/compute_stage26_residual.py` exits unsuccessfully for unqualified evidence.
`--diagnostic-only` permits historical arithmetic but retains the failed
qualification and never labels the result as real gaps. The README renderer
recomputes this audit instead of trusting an old `status=complete` label.

| Requirement | Regression coverage |
| --- | --- |
| Every recorded coding round reaches the histogram | `test_current_context_results_account_for_every_round` |
| CPU annotation duration is excluded from GPU totals | `test_overhead_counts_gaps_once_and_subtracts_concurrent_gpu_work` |
| Concurrent GPU work is counted once | `test_stage26_uses_union_instead_of_double_counting_overlapping_kernels` |
| A fabricated interval union is rejected | `test_stage26_rejects_a_fabricated_union` |
| Archived control/profile subtraction is not certified as real gaps | `test_published_compiled_profile_and_control_are_not_qualified_runtime_gaps` |
| Scope events cannot masquerade as kernel-only time | `test_scope_events_cannot_be_declared_profiler_free_kernel_time` |
| A claimed observer comparison requires samples | `test_measured_label_without_observer_samples_is_rejected` |

These checks validate accounting and evidence eligibility. They are not a new
GPU qualification or proof of zero measurement disturbance.
