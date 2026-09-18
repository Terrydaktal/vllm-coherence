# Exact GDN B/A M4-pair out-parameter screen

This is a stopped-GPU, component-first experiment. It is deliberately not connected to the vLLM
runtime or launcher, so the candidate is default-off until both exactness and a measured whole-round
latency gate pass.

## Why the v416 convolution projection did not transfer

The original v416 component result projected a 19.195343 ms whole-round saving by multiplying one
isolated layer's 0.399903 ms paired saving by 48. Its timed reference did not have the production
state ABI:

```text
v416 component reference: shape [2,10240,10], stride [1687552,1,10240]
authenticated release:    shape [2,10240,10], stride [0,1,10240]
                          backed by one private [1,10240,10] slot
```

The live path constructs the zero-slot-stride view in
`_ensure_recoverssm_trusted_replay_buffers`, then passes logical state index one to
`causal_conv1d_update`. Index one therefore aliases the one private scratch slot. The old benchmark
instead sent index one to a physically separate slot 1,687,552 elements away. It timed a different
kernel memory contract and its 48-layer multiplication was not a production projection.

The corrected component harness now uses `_production_reference_state` to reproduce the zero-stride
alias. Independent whole-runtime evidence bounds the actual transfer: the matched uncached-selector
v422 fixed-convolution run and v424 fixed-convolution-off control recorded these counters:

```text
                                      v422 on       v424 off
nonverification.round.gpu total      13195.351      13717.957 ms
counter                                      124            124
paired whole-round difference                        4.214565 ms

steady target.forward M8 total         8313.085       8834.011 ms
counter                                      121            121
steady target-forward difference                    4.305174 ms
```

Thus the old 19.195343 ms claim overstated measured whole-round transfer by about 4.55x. This is why
the B/A candidate below requires a full 48-layer component replay and a 2 ms promotion floor; a
single-layer latency multiplied by layer count is insufficient.

## Candidate

The authenticated ROCm target computes `in_proj_ba` as two ordered M4 projections and concatenates
their results. On gfx1201, vLLM's `rocm_unquantized_gemm_impl` routes each M4 projection to
`_rocm_C.wvSplitK`; it is not an ordinary `F.linear` GEMM:

```python
torch.cat((wvSplitK(weight, hidden[0:4]), wvSplitK(weight, hidden[4:8])), dim=0)
```

The candidate is source-bound to vLLM commit
`d626108b1841888ec90aced33367149a6bbc7e4b`, which is the installed release's commit. For the R9700's
runtime-reported 32 CUs and fixed weight/input geometry `[96,5120] @ [4,5120]`, upstream `wvSplitK` selects
`wvSplitK_hf_sml_<BF16,32,1,16,8,4,4>` with four active waves. The candidate copies that executed
arithmetic path into `gdn_ba_m4_pair_kernel.cu` and launches it twice in the original order. The only
intentional change is that rows 0–3 and 4–7 are written directly into a caller-owned `[8,96]` BF16
output. This removes the two result allocations and `torch.cat` kernel without changing M4 reduction
order. It does not substitute a generic M8 or batched GEMM; v363 already proved that generic M8
arithmetic is not bit-identical.

The complete upstream `skinny_gemms.cu` source used for that extraction has SHA-256
`013f14b570cd8f25e254bf47643ba2802ab7d5fdd2069adb111bc6ff560f6682`; the benchmark also rejects an
installed vLLM whose version is not bound to commit `d626108b1`.

The native ABI accepts only:

```text
hidden_states  BF16 contiguous [8,5120]
weight         BF16 contiguous [96,5120]
output         BF16 contiguous [8,96], non-overlapping
```

All tensors must share one ROCm device. The checkpoint's separate B and A weights are merged once in
the same B-then-A order as `MergedColumnParallelLinear(output_sizes=[48,48])`.

## Hard gate

`benchmark.py` loads all 48 real GDN B/A weight pairs. It compares every physical BF16 output bit for
all eight rows and every layer across random, zero, signed-zero, alternating, and scale-boundary
inputs. It then mutates every input in place, crosses the M4 boundary with a row permutation, checks
repeatability and output canaries, rejects overlapping buffers, and verifies that steady candidate
execution retains no new device allocation.

Promotion is rejected unless both paired medians over a complete 48-layer replay save at least 2 ms:

```text
baseline = two installed `_rocm_C.wvSplitK` M4 calls + torch.cat, repeated for 48 layers
candidate = one native entry with the same two ordered M4 kernels writing an M8 out parameter

median paired GPU saving           >= 2.0 ms
median paired dispatch-wall saving >= 2.0 ms
```

Full runtime integration is intentionally absent. A passing component result authorizes a separate
default-off runtime overlay and authenticated assurance/release A/B; it does not promote itself.

## Build and run

Only run this with vLLM stopped and after coordinating ownership of the R9700:

```bash
with_rocm=/home/lewis/.local/share/qwen-r9700/runtimes/vllm-rocm/bin/with-rocm
python_bin=/home/lewis/.local/share/qwen-r9700/runtimes/vllm-rocm/.venv/bin/python
candidate=/home/lewis/projects/qwen-r9700/artifacts/qualifications/CANDIDATE

VLLM_ROCM_USE_AITER_LINEAR=0 "$with_rocm" "$python_bin" \
  "$candidate/build_extension.py" --build-directory "$candidate/build"

VLLM_ROCM_USE_AITER_LINEAR=0 "$with_rocm" "$python_bin" \
  "$candidate/benchmark.py" \
  --extension "$candidate/build/qwen_gdn_ba_m4_pair_out_gfx1201_v2.so" \
  --output "$candidate/result.json"
```

The benchmark exits nonzero on either a one-bit mismatch or a saving below 2 ms.
