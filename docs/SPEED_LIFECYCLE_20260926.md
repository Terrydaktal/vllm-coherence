# Cancellation and concurrent-chat qualification — 26 September 2026

This qualification exercises the September 25 speed combination with the real
snapshot connector, response-boundary scheduler, Global-512 head and ROCr
polling backoff. Every prompt is synthetic. No private session was read.

## Results

| Check | Result |
| --- | --- |
| Native lifecycle suite with dispatch observation | 11 / 11 cases passed |
| Same suite with the plain speed worker, without diagnostic hooks | 11 / 11 cases passed |
| Targeted CPU regression and release-binding tests | 226 passed |
| Identical fresh one-token requests after the initialization repair | 6 / 6 identical hidden-state, full-logit and output hashes |

Both native runs include the sampled replay case at temperature 1.0, top-p 0.95
and top-k 40. The observed run exercised 2,872 FULL, 338 PIECEWISE and 157 eager
dispatches. These counts describe coverage, not a speed measurement.
[Aggregate results and exact source identities](../benchmarks/results/speed-lifecycle-20260926.json)
bind this result to the tested implementation. This completes the requested
cancellation and concurrent-chat qualification for the configuration below.
Arbitrary schedules and numerical two-request GPU batching remain unqualified.

The scheduler and GDN repairs were first deployed with the normal worker,
recorded in the [initial deployment receipt](../benchmarks/results/lifecycle-repairs-deployment-20260926.json).
The live Pi backend was then switched to the qualified speed worker: FULL target
graphs, local-split GEMM and qualified drafter attention, with Global-512 retained.
The [speed deployment receipt](../benchmarks/results/qualified-speed-deployment-20260926.json)
records the exact installed worker, scheduler and GDN hashes, successful graph
capture and the loaded qualified GEMM binary. Two Pi warmups completed, three
fresh one-token requests produced identical 17-token outputs, and the API was
healthy with an empty scheduler. Pi and pi-opsec use this shared backend. The
snapshot data ABI is unchanged, so existing saved chats remain compatible.

## Defects found and repaired

**Cancelling a chat before its first GPU admission crashed the engine.** Its
request already had a chat owner, but that chat did not yet have a KV allocator.
vLLM's normal cancellation cleanup tried to look up the nonexistent allocator.
`CacheBanks` now explicitly tracks requests that have never acquired a bank and
returns their empty block tables during cleanup. Missing banks belonging to
previously admitted requests remain errors; they are not silently treated as
empty.

**The first cleanup repair exposed a stalled replay.** Removing the bankless
request's owner also caused the scheduler to withhold its completion record.
The snapshot connector then waited indefinitely for a finished predecessor to
settle. Bankless completions now reach the connector on the next scheduler
step, even while a different chat owns the GPU. They cannot refer to GPU blocks.
Completions for real parked banks remain isolated until their own bank is active.

**A fresh one-token prompt read uninitialized GDN history.** The metadata builder
classified it as ordinary decode solely because its query length was one.
That path assumes an existing convolution and recurrent state; the generic KV
zeroer explicitly skips Mamba state. The defect occurred with the speed changes
both enabled and disabled. The repair routes a batch whose CPU maximum sequence
length is one through existing prefill initialization. Ordinary decode and
longer-prefill classification remain unchanged.

Before that repair, six identical fresh one-token requests produced six distinct
hidden-state hashes and six distinct full-logit hashes. After it, all six match
exactly across control/candidate switches. The two-token control's hidden states
and logits are unchanged. This is a reproduced initialization defect and repair,
not a new claim of arbitrary-input model equivalence.

## Test scope

`experiments/radiance-public/qualify_speed_lifecycle.py` exercises:

1. Prompt lengths 1–8, output stops around D7 boundaries, and prefill sizes
   2,047 / 2,048 / 2,049.
2. Uninterrupted reference/candidate outputs at 2K and 4K, plus a 60K control.
3. Disconnects after the first token and during longer generation, then replay.
4. Disconnect during a cold 60K prefill, then replay.
5. Cancellation of a queued chat while another chat continues, then replay.
6. Two equal-priority requests: response completion before GPU handover.
7. Priority-2 preemption and exact resumption of the parked chat.
8. Cancellation of the parked chat during priority-2 service, then replay.
9. Cancellation of the urgent chat and resumption of the parked chat.
10. Priority-1 ownership across a simulated tool pause and model continuation.
11. Sampled cancellation and concurrent-waiter replay at temperature 1.0,
    top-p 0.95, top-k 40 and seed 113.

Greedy controls use temperature zero. Tests deliberately request bounded output
with EOS ignored to exercise cancellation and length boundaries; these are
functional tests, not natural-completion throughput measurements. Replays must
match token count, token-sequence hash and finish reason. Tests also require
waiter progress and an empty scheduler at completion.

The supported deployment schedules one request on the GPU at a time, with two
resident chat banks. FULL capture is admitted only for one eight-token D7
verification request. Small shapes retain PIECEWISE captures at 1 / 2 / 4 / 8.
The diagnostic observer records actual selected descriptors and checks the real
dispatcher on 18 combinations of one/two requests and widths 1–9. That metadata
check is not a numerical qualification of two requests batched on the GPU;
such batching remains outside the scheduler's supported contract.

## Reproduction and evidence

Use an isolated pinned-image server with `speed_candidate_worker.SpeedCandidateWorker`,
the authenticated September 25 GEMM/drafter artifacts, FULL_AND_PIECEWISE captures
at 1 / 2 / 4 / 8, and the current integration patches. Run inside its IPC namespace:

```sh
python /path/to/qualify_speed_lifecycle.py \
  --base-url http://127.0.0.1:8081 \
  --abi "$SNAPSHOT_DATA_ABI" --run-id unique-run-name \
  --status-prefix /dev/shm/qwen-stage-timing \
  --output /path/to/new-result.json
```

For dispatch observation, use `speed_lifecycle_worker.SpeedLifecycleWorker` and
add `--observe`. Its normal observer reads CPU metadata only. The optional
single-token tensor probe is a separate diagnosis mode and must not be used for
timing. Qualification is repeated with the plain worker to exclude those hooks.

Reports contain hashes, counts, timings and configuration identities, never
prompt text or token arrays. CPU tests exercise the actual cancellation routing
and installed GDN metadata branch, plus deliberately invalid dispatch and output
records to verify that the qualification checker rejects them. The patch installer
checks both pinned preimage and postimage hashes.
