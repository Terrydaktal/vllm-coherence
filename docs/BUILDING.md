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
