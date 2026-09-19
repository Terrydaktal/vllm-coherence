# Pi generation and prefill qualification — 18 September 2026

**Deployed and healthy on the normal Pi backend.** The final release contains all 48 GDN fusion sites, prefill norm fusion, spatial GDN scan tiling and a qualified tiled-activation GEMM path. Local/remote release preflight passed; the startup receipt confirms installation before compilation and graph capture. No benchmark inference was submitted after the final production restart. Nine task-created benchmark caches were removed, freeing 16.52 GiB; original chat caches were retained.

The existing nine-slot recurrent layout, corrected D7 arithmetic, guarded GEMM dispatch, global-256 target head and durable snapshots remain in the qualified configuration.

## Serving measurements

The before/after comparison uses the normal compiled, graph-enabled serving launcher, with snapshots enabled and no diagnostic worker or timing hooks. Input is the same existing private **60,000-token Pi prefix**. Three natural completions use seeds 0, 17 and 42, temperature 0.6, top-p 0.95 and top-k 20. Each response has a 1,024-token safety budget; all ended naturally. These are **2,375 generated tokens per variant**, not 60K generated tokens. The before release already contains the faster GEMM and global-256 head. The final measurement uses a fresh server with an empty KV bank and previously populated compiler caches.

| Seed | Before round ms | Final round ms | Before tok/s | Final tok/s | Output tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 44.58 | 44.52 | 120.1 | 120.2 | 791 |
| 17 | 43.92 | 43.77 | 127.3 | 127.7 | 759 |
| 42 | 43.92 | 43.80 | 124.7 | 125.1 | 825 |

| Measurement | Before | Norm fusion + scan tiling | Final, also tiled GEMM |
| --- | ---: | ---: | ---: |
| Cold 60K time to first token | 35.09 s | 33.02 s | 31.98 s |
| Effective cold prompt throughput, including first-token overhead | 1710 tok/s | 1817 tok/s | 1876 tok/s |
| Weighted mean generation round | 44.144 ms | 43.939 ms | 44.037 ms |
| Pooled generation after first token | 123.93 tok/s | 124.41 tok/s | 124.24 tok/s |

All three output hashes, output counts and acceptance counts matched across the releases. Cold time to first token fell **8.9%** overall. Decode is essentially unchanged within this small sample; these additions are not another 17 ms decode improvement. First-token time includes CPU preparation, cold prefill, drafting and initial sampling. It is not a pure GPU prefill timer. Later requests reused 57,680 prompt tokens and are excluded from the cold-prefill comparison. These are single cold observations, not a repeated statistical prefill study.

Pi itself uses **temperature 1.0**, confirmed by capturing both the installed host and VM clients' request construction with networking disabled. A separate backend replay used those sampling settings on the same 60K prefix:

| Seed | Generation tok/s | Mean round ms | Draft acceptance | Output tokens |
| --- | ---: | ---: | ---: | ---: |
| 0 | 117.3 | 43.93 | 59.20% | 787 |
| 17 | 122.3 | 43.74 | 62.23% | 782 |
| 42 | 122.5 | 43.64 | 61.33% | 794 |

Pooled rate: **120.65 tok/s**, 2,363 generated tokens, all three responses ending naturally. This checks Pi's sampling settings through the backend completions API; it is not an end-to-end Pi UI or VM transport benchmark, and does not predict every chat's acceptance rate.

## What was fixed

- **48 GDN normalization/quantization sites:** fuse gated RMSNorm with per-token FP8 quantization while preserving native reduction order and the intermediate BF16 rounding. Explicit Gluon layouts are required: an initial automatically selected layout failed one FP8 byte and was rejected. The accepted implementation measured 6.656 → 4.546 microseconds per M8 site, approximately **0.101 ms across 48 sites**.
- **128 decoder norm/FP8 sites during prefill:** the earlier backport only admitted batches up to eight rows. Larger batches now use the native prefill reduction layout, which differs from the single-row layout. This avoids changing arithmetic at rounding ties.
- **GDN prefill recurrence:** retain chronological updates and change only the spatial tile. At 1,648 tokens, the isolated scan measured 2.706 → 2.343 ms per layer. This preserves the existing state layout.
- **Tiled prefill GEMM:** reorder existing FP8 bytes without changing values or scales, then use the existing native tiled consumer. Only four qualified projection shapes at M=1,000/1,648/2,048 are admitted; other shapes, including decode, use the original consumer. At M=1,648 the largest projection measured 3.262 → 2.815 ms including the reorder. The full-model receipt recorded 7,104 tiled calls; the optimization actually executed.
- **Deployment:** freeze the tested sources, binaries and evidence into a new authenticated payload. The runtime ABI changes; the canonical snapshot data ABI remains unchanged. Existing snapshots are retained. Production is restarted after benchmarking, with no synthetic inference left owning its GPU bank.

## Correctness checks

| Check | Result |
| --- | --- |
| GDN fusion, all 48 actual checkpoint weights, M1 and M8 | 1,000/1,000 rows per site and width; exact FP8 bytes and scales |
| GDN fusion prefill | Exact at 9, 320, 1,000, 1,648 and 2,048 rows |
| Decoder norm/FP8 prefill | 1,000 rows at each of 128 sites; residual and non-residual cases; additional boundary widths |
| GDN scan tiling | Exact outputs and final recurrent state at 1, 8, 64, 320, 1,000, 1,648 and 2,048 rows |
| Tiled prefill GEMM | Exact native output and actual custom-op dispatch for four projection shapes at 1,000/1,648/2,048 rows; byte-layout, graph replay and output canary checks passed |
| Full model, same private 60K prefix, compiled M8 before/after | 320/320 complete vocabulary-logit hashes; prefill prediction also exact |
| Full-model top-1 / top-10 / top-20 set and ordering | 320/320 for every measure |
| Fault-injection controls | Detected deliberate corruption |
| CPU release/admission regression checks | 66 passed |

The full-model equality check was repeated after adding tiled GEMM and again passed all 320 rows and the prefill prediction. It uses the corrected full BF16 head to expose backbone differences; serving measurements use the requested approximate global-256 head. This is sampled equivalence to the corrected baseline, not a universal mathematical proof or independent validation of the base model.

## Generation diagnosis and remaining limits

Native traces confirm that the faster GEMM executes. An earlier controlled original-versus-candidate GEMM comparison reduced the complete median round from 56.76 to 42.85 ms. The earlier roughly 17 ms estimate summed isolated projection savings; it was not a measured complete serving-round reduction.

Diagnostic workers sometimes produced roughly 54 ms rounds. Removing offload appeared to remove that delay in one diagnostic control, but the final normal launchers both achieve approximately 44 ms **with offload enabled**. Consequently, the evidence does not establish a sustained offload defect, an idle-backoff defect, or the cause of the historical roughly 56 ms Pi observation. The diagnostic timing results are kept separately from the release comparison. Profiling also produced a 91 ms outlier at capture shutdown; that trace is not a release timing result.

The final payload's first serving response measured 47.44 ms/round, followed by roughly 44 ms; after a restart with compiler caches populated, the first response measured 44.52 ms. Startup/first-use work can affect a short benchmark. Kernel JIT warnings also appeared after that restart, so the warnings alone do not quantify fresh compilation time. Pi's temperature difference reduces measured acceptance somewhat here, but does not explain a 56 ms round.

The GPU reaches 110–112°C junction during sustained prefill; its driver reports a 110°C critical threshold. The hardware metric reports the hotspot-throttle bit during the old-profile control, and the new-profile clock capture also reaches the temperature threshold. Cooling/fan response is therefore an additional performance limit to investigate; its exact throughput cost was not measured. The throttle flag can remain set after cooling and must not be interpreted as an instantaneous duty-cycle counter. The board power cap was 300 W. Hardware settings were not changed.

Metric interpretation uses the kernel's [GPU metrics v1.3 layout](https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/include/kgd_pp_interface.h) and [independent throttle-bit definitions](https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/pm/swsmu/inc/amdgpu_smu.h).

Results apply to this fixture, context length and sampling configuration. A different acceptance rate or longer context can lower visible tokens/s. No private chat text was read or added to the report. [Machine-readable results](pi-prefill-generation-20260918.json).
