# Native check of the upstream M8 head port

This supplement records the isolated native check for
[vLLM #57286](https://github.com/vllm-project/vllm/pull/57286), based on vLLM
`9854b580dfe518171eed3a6ad76cbd402ca2bdfa` with the interleaved-M4 patch.
The original report/release evidence remains unchanged.

- `current_main_head.hip` is the compiled extraction of the patched kernel, with
  minimal BF16 arithmetic helpers and a direct launch entry point.
- `probe.py` preserves the executed synthetic probe, including its original
  `/qualification/preflight/publication-20260917` paths. The run was admitted
  through the external shared GPU lease before this script executed.
- `result.json` preserves its measurements; `sha256.json` binds those files.

Compilation used ROCm HIP, `-O3 -std=c++17 --offload-arch=gfx1201 -shared -fPIC
-ffp-contract=off`. The script expects `head-current.so` and `head-current.hip`
in its recorded working directory. Reproduction must use an isolated output
directory and acquire the applicable GPU lease before launching the probe.
It allocates a full BF16 head and must not be run over an active inference job.

Four scales (0, 0.01, 1 and 16), eight positions each, compared all 248,320
vocabulary values per position. M1/M8, two M4 groups/M8 and graph replay all
reported zero differing values: 7,946,240 values per comparison. This probe uses
tensor value comparison, not raw-byte hashing, so it does not distinguish signed
zeros. The separate 10K downstream campaign also checked full-logit digests.

This proves neither arbitrary-input numerical equivalence nor the unbuilt
current-main C++ extension/Python dispatch integration. The complete compiled
**Radiance** model already passed the separate 10,000-position qualification.

The extracted upstream code is derived from vLLM's `csrc/rocm/skinny_gemms.cu`
and distributed under the accompanying Apache-2.0 license. The patch adds group
interleaving; the helper scaffold permits compiling that kernel in isolation.
