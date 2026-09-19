# Intermittent 44 → 54 ms generation slowdown

The slowdown is reproducible on the same backend, prompt and output. It persists
after prefill or disk restoration and disappears after a stream synchronization.
The remaining defect is narrowed to HIP stream/queue handling; the specific
internal dependency, fence or queue resource has **not** been established.
No permanent fix was deployed during this diagnosis.

## Workload and reproduction

The retained private Pi fixture supplies 60,000 prefix tokens. The benchmark's
Chat framing makes the request 60,077 tokens. Each natural completion produces
530 output tokens over 196 generation rounds, with 24.344% draft acceptance.
Sampling is temperature 1, top-p 0.95, top-k 20, seed 0. The compiled PIECEWISE
backend, corrected arithmetic and Global-256 head remain enabled.

On the original running process:

| Operation | Mean generation round |
| --- | ---: |
| Three warm repeats | 43.59 / 43.67 / 43.75 ms |
| Same prompt with a fresh cache identity | 53.51 ms |
| Immediate repeat on that cache | 53.71 ms |
| Switch to another RAM-backed chat | 43.62 / 43.70 ms |
| Return to the previously slow chat through RAM | 43.64 ms |
| Another fresh cache | 53.86 ms |

Creating a new generation of the same diagnostic chat also reproduced the
slowdown, without allocating a second chat's RAM bank. Therefore the fresh-chat
RAM allocation is not required to trigger it.

The captured run contains 29 requests. The 23 responses recorded with both text
and token-chunk hashes have identical hashes, output counts and acceptance
patterns. The other six predate the combined text/token-hash capture. Chat text
was not read or saved in the diagnostic evidence.

## Controlled interventions

Interventions occur after the 30th streamed output chunk. Before/after values
average 29 pre-intervention and 110 later per-round arrival intervals. The
intervention and its surrounding intervals are excluded. These are not
whole-request timings or GPU-only timings.

| Intervention | Before | After | Intervention duration |
| --- | ---: | ---: | ---: |
| No-op | 53.95 ms | 52.89 ms | <0.01 ms |
| Pause the host for 50 ms | 53.10 ms | 53.03 ms | 50.06 ms |
| Record and synchronize timing events | 52.23 ms | 53.15 ms | 6.04 ms |
| Synchronize the current stream | 52.52 ms | 43.88 ms | 6.38 ms |
| Repeat stream synchronization after a fresh generation | 52.33 ms | 44.11 ms | 6.39 ms |
| Synchronize the device | 52.19 ms | 43.82 ms | 6.41 ms |
| Stream synchronization with original ROCr | 53.18 ms | 43.64 ms | 6.38 ms |

The original ROCr library was verified against its recorded SHA-256. Removing
our CPU polling-backoff patch did not prevent the slowdown or change the
recovery mechanism. The normal backoff library was restored afterward.

Event completion and a host pause do not produce the stream-synchronization
effect. This distinguishes completion of submitted GPU work from the additional
stream/queue cleanup performed by HIP's synchronization path. The relevant
source is [`HostQueue::finish`](https://github.com/ROCm/rocm-systems/blob/2b22ab0195cc1461cd9abf3b969e9dd7c10af350/projects/clr/rocclr/platform/commandqueue.cpp)
and [`VirtualGPU::releaseGpuMemoryFence`](https://github.com/ROCm/rocm-systems/blob/2b22ab0195cc1461cd9abf3b969e9dd7c10af350/projects/clr/rocclr/device/rocm/rocvirtual.cpp).
Which cleanup action is decisive remains an inference to test, not a confirmed
source-level diagnosis.

## Native trace evidence

The existing profiler hook synchronizes before capture, which clears the slow
state. A diagnostic worker extension therefore starts profiling without that
explicit synchronization. Profiling can still perturb execution, and stopping
the profiler synchronizes internally. Unprofiled controls remain the release
performance evidence.

The comparison below uses six complete cycles per trace, with 1,326 kernels per
cycle. An incomplete fast cycle was excluded. The gap sample spans 889 slow and
1,016 fast normalization-to-GEMM boundaries.

| Measurement | Slow capture | Fast capture |
| --- | ---: | ---: |
| Profiled cycle elapsed | 51.72 ms | 47.37 ms |
| Sum of kernel execution durations | 43.26 ms | 41.78 ms |
| Target GEMM execution | 24.27 ms | 24.21 ms |
| Normalization → GEMM gap | 7.40 µs | 3.89 µs |

The optimized GEMMs remain active with almost unchanged execution time. Extra
inter-kernel gaps and some small-kernel overhead appear in the slow state.
Profiler overhead prevents attributing the entire unprofiled 10 ms difference
by subtracting these trace totals.

## Per-round synchronization telemetry

The backend now has a bounded, content-free round feed for the next diagnostic
capture. Enable it only in the outer stage-timing run:

```text
QWEN_OUTER_STAGE_TIMING=1 \
QWEN_ROUND_EVENT_TELEMETRY=1 \
QWEN_ROUND_EVENT_TELEMETRY_SYNC=1
```

The feed is written to
`<fair-scheduler-status-path>-gpu-rounds.jsonl` (normally
`/dev/shm/qwen-radiance-fair-public-gpu-rounds.jsonl`) and rotates at 8 MiB.
Each row records the total scheduled tokens, request count, tokens per request,
speculative widths, device and stream labels, ordered HIP GPU-event markers,
duration for each measured boundary, and the HIP event gap between every
adjacent marker. It also records the host dispatch interval and whether the
round crossed a recorded device/current-stream synchronization, including that
fence's elapsed time. The `dropped_records_before` counter makes a failed or
rotated diagnostic write visible on the next successful row. No prompt, token
ID, tool argument or chat text is written. The separate `-rounds.jsonl` feed
also carries the scheduler shape so shape can still be correlated if GPU
events are unavailable.

`QWEN_ROUND_EVENT_TELEMETRY_SYNC=1` waits on the round-end event to complete
the event measurements and reports the wait separately; this is intentionally a
diagnostic overhead. With it unset, the capture never adds a synchronization:
unavailable HIP event values are reported as `null` rather than being guessed.

The implementation uses PyTorch's `torch.cuda` compatibility namespace because
PyTorch exposes the ROCm device API there; the events are HIP events and this
does not require an NVIDIA CUDA runtime or GPU.

In the original process, fast/slow decode captures have unchanged VRAM and GTT
usage and no engine-process disk-read/write increment. The fresh slow case has
slightly higher average GPU clocks than the fast baseline, so lower clocks do
not explain that comparison. A force-flush check also found the diagnostic tail
already durable while the slowdown persisted.

## Evidence and remaining work

[Machine-readable measurements](round-latency-20260919.json) include source
hashes, telemetry aggregates and the restored configuration receipt.
[The reusable worker extension](../experiments/radiance-public/round_latency_probe.py)
is byte-identical to the tested second diagnostic module. It is for a disposable
backend with developer RPC bound to loopback; it is not installed in the normal
backend. Two CPU checks protect the absence of explicit profiler
presynchronization and reject unsupported capture modes.

The normal backend is restored and healthy, with developer RPC disabled and all
original model/runtime flags preserved. Raw metadata and compressed traces are
retained privately. Temporary token fixtures are removed after collection.

A permanent fix must prevent or retire the residual stream/queue state at the
correct transfer/prefill boundary. It still needs cold-prefill, disk-restore,
RAM-handover and tail-flush coverage, plus unchanged-output checks. The measured
one-time synchronization is a recovery control, not yet that qualified fix.
