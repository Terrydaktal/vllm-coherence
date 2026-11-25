# Building and qualification

`releases/0.1.0.json` binds the runtime archive, internal file manifest, CPU support
library and pinned container digest. `prepare` rejects unsafe archive members and
verifies all artifacts before installation. The archive preserves the exact
qualified binaries, their sources and receipts beneath `/qualification`; keeping
these container paths avoids rewriting hash-bound evidence. No model weights or
private token/tensor captures are included.

The release also hashes every integration file copied into the serving directory.
Preparation and serving hold an exclusive state lock so a second preparation
cannot replace code underneath a running backend. The initial portable launcher
supports rootless Podman; other container runtimes are not qualified here.

Build changed kernels inside the pinned image without GPU devices. Drivers emit
compiler commands, source/binary hashes and **BUILT_UNTESTED** receipts:

| Driver in `experiments/radiance-public` | Input → output |
| --- | --- |
| `build_stock_m1_norm.py` | HIP source and reduction options → normalization library |
| `build_stock_m1_head_pair.py` | Pinned head source → interleaved M4-pair library |
| `build_stock_m1_attention_shared.py` | Source/headers → serial-contract attention library |
| `build_packed_gdn_transport.py` | Repair manifest → packed transport implementation |
| `build_mxfp4_dispatch.py` | Native/Python preimages → unchanged control and guarded candidate |
| `build_stock_fp8_epilogue.py` | Epilogue source → fused norm/quant library |
| `rocr-poll-backoff/build.sh` | Patched pinned ROCr source → CPU wait library and tests |

Consult each driver's `--help` for required paths. Compilation is not qualification.
Run the matching `probe_*` on an isolated GPU, checking outputs, state, boundary
shapes, graph replay and injected faults. Follow with forced-token whole-model
comparison and separate uninstrumented performance controls.

`prepare_mxfp4_dispatch.py`, `prepare_tp1_lazy_backports.py`,
`prepare_pi_prefill_bundle.py` and `prepare_pi_prefill_release.py` bind successful
receipts into immutable bundles. The production profile retains nine state slots;
the lazy-layout experiment is excluded. `prepare_optimized_pi_release.py` exports
the serving dependency closure. `tools/package_runtime.py` packages its allowlisted
files and the authenticated CPU support library.

For publication: run CPU/install checks, qualify changed native code, freeze the
bundle, package it, update the pinned release manifest, then publish source and
archive together. Never overwrite an existing versioned asset. A changed compiler,
source, model, state layout or arithmetic contract needs a new identity. CPU checks
of portability tooling do not extend native evidence to other hardware/profiles.

## Experimental host embedding storage

The opt-in worker maps the 2.37 GiB BF16 input embedding table from pinned host
memory on a single gfx1201 GPU. Startup verifies the mapped bytes and checks
that the FP8 drafter shares the same table. The configured KV pool stays fixed.

```sh
tools/coherence serve --model /models/target --draft /models/dflash2-fp8 \
  --state /path/to/host-embedding-state --port 18081 --experimental-host-embedding
```

Use a fresh state directory. Its snapshot and compiler cache identity includes
both host embedding source files.

## 327,952-token authenticated Pi lane

The authenticated R9700 launcher binds the context limit to `extended-context.json`.
It uses the original FP8 drafter and fixed-32 target attention with pinned host
embeddings and a 12,447,383,552-byte KV allocation. The target retains its
existing MRoPE table. The launcher verifies the original draft configuration
and generates a read-only override with 327,960 rotary rows. Rotary frequencies
stay the same.

The launcher authenticates the frozen speed release before selecting the updated
worker. That worker checks target and draft rotary capacity and extends both
prefill admission guards using the same fixed-split HIP kernel. The sizing test
keeps the previous profile's spare KV blocks and sets aside 168 MiB of GPU
memory for allocation peaks. On one R9700, the 329,600-token profile dropped
to 151 MiB free during the live checkpoint check, so the launcher uses the
next lower allocation step.

The 327,952-token production profile kept 259 MiB free across startup,
cache handover, checkpoint generation and a 327,888-token cold prompt followed
by 64 output tokens with speculative decoding. Graceful shutdown published
all 407 expected blocks for the long request. The final startup and continuation
check kept 206 MiB free.

The authenticated profile uses a separate snapshot data namespace. Pi reads the
limit from the matching manifest. Preserve the previous namespace for rollback
before switching because the active engine collects incompatible namespaces.
Restart existing Pi sessions to pick up the new capacity and authenticated lane.
