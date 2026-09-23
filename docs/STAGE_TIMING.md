# Stage timing and runtime gaps

The current stage table uses the September 23 compiled Global-256 serving capture with the pinned-RAM huge-page promotion repair. The separate coding histogram retains every timed round (3,736) and counts its three untimed first-token events separately. Neither includes traced rounds as ordinary Pi latency.

## Matched native measurements

All values below use the same retained decode indices in each arm. A full natural warmup precedes control → trace → control. Output-token hashes, accepted-token sequences and scheduled shapes agree across the three arms. Sampling is temperature 1.0, top-p 0.95, top-k 40, seed 0; EOS remains enabled. The existing manual GPU mode, −50 mV offset and 300 W cap were retained.

| Starting context | Retained M8 cycles per arm | Clean before, ms | Traced, ms | Clean after, ms | GPU occupied union, ms | Estimated remainder, ms | Before/after remainder range, ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0K | 917 | 38.915 | 47.556 | 38.764 | 36.833 | 2.006 | 1.931–2.082 |
| 60K | 1,130 | 43.113 | 47.980 | 42.538 | 40.230 | 2.596 | 2.308–2.883 |
| 200K | 774 | 50.318 | 55.683 | 50.282 | 47.827 | 2.473 | 2.456–2.491 |

Tracing added 8.717 / 5.154 / 5.383 ms per retained round. That observed slowdown is separate from the GPU kernel durations and is not included in the clean-control residual. The before/after range measures repeat variability; it is **not** a confidence interval or a bound on every possible observer effect.

The whole 60K controls averaged 42.978 and 42.628 ms. The earlier control retained one 660.499 ms pause; the later control's maximum was 46.526 ms. The separate first warmup contained a 6,072.966 ms pause. These isolated outliers remain unresolved and are preserved in the native records and [huge-page intervention evidence](../benchmarks/results/huge-page-promotion-20260923.json); removing the repeating huge-page stalls does not establish that every startup/cache/driver delay is gone.

GPU activity is clipped at the worker-entry boundaries also timed by the controls. Mean trace-marker/host-clock boundary differences were at most 0.000014 ms, with a largest individual difference of 0.029411 ms. CPU annotations, Python hooks and trace export are never added to a stage. GPU overlap is counted once: its mean was 0.003 / 0.003 / 0.011 ms. Controls retain ordinary production telemetry and one host boundary clock/record; they contain no tensor reads, forced sampling, stage events or additional synchronization.

The capture requested 1,152 traced rounds per context in bounded chunks. Natural completions contained 4,981 / 8,213 / 4,233 output tokens. First/last chunk boundaries, partial-width steps and structurally incomplete trace inventories are excluded from M8 stage attribution, independently of duration. Both valid short-context attention variants are included: a page crossing can execute two decode/merge pairs. Every omitted inventory is recorded in the aggregate; no missing operation is assigned zero time.

Indirect profiler effects on clock speed, kernel execution and scheduling remain an uncertainty. Kernel activity durations exclude direct CPU profiling cost, but cannot establish zero disturbance for each kernel. [PyTorch documents profiler overhead](https://docs.pytorch.org/tutorials/beginner/profiler.html).

[Profile and identities](../benchmarks/results/compiled-global256-stage-profile-20260923.json) · [clean controls](../benchmarks/results/stage26-control-20260923.json) · [residual calculation](../benchmarks/results/matched-stage-residual-20260923.json). Runtime source hashes and observed pinned-memory page policies accompany the numerical build identity. Raw traces and private token fixtures remain outside the public repository. The normal backend was restored after capture.

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

This requires matching runtime artifacts, compiled Global-256 execution,
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
