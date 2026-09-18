# Provenance

Coherence is a standalone downstream fork of Radiance, maintained by **Terrydaktal**.
Its root commit imports the exact Radiance 1.0.16 tree; subsequent feature commits
are curated public imports, not reconstructed private experiment chronology.

| Component | Origin |
| --- | --- |
| Radiance | [magiccodingman/vllm-radiance](https://github.com/magiccodingman/vllm-radiance), `f295b9ef51ad413a68e4192371e0377741a354ce`; canonical [Lance Wright GitLab](https://gitlab.sayou.io/lance-wright/vllm-radiance) |
| Serving/model infrastructure | [vllm-project/vllm](https://github.com/vllm-project/vllm) |
| RDNA4 kernels | [StillDeadcode/libr4d](https://codeberg.org/StillDeadcode/libr4d) |
| Performance backports | [GGZ14 3f542b7](https://github.com/GGZ14/vllm-mxfp4/commit/3f542b7cbfce3fa0d01dc55665af4c77a7093ce8), guarded/adapted to the repaired arithmetic |
| DFlash RNG separation | [vLLM #54282](https://github.com/vllm-project/vllm/pull/54282), backported rather than newly discovered |
| BF16 RoPE multiply lowering | [Triton #11227](https://github.com/triton-lang/triton/pull/11227), existing upstream repair reproduced/adapted for the pinned stack |
| CPU wait backoff | [ROCm/rocm-systems](https://github.com/ROCm/rocm-systems), `46558b7af4dc79b8b8014619c1afdb82db079a9f` |
| Pi client | [earendil-works/pi](https://github.com/earendil-works/pi), pinned 0.84.2, plus Coherence patches/extensions |

Existing independent submissions remain available:

- [Radiance #8](https://github.com/magiccodingman/vllm-radiance/pull/8): RNG backport and extreme-decay GDN repair.
- [Radiance #9](https://github.com/magiccodingman/vllm-radiance/pull/9): global-256 target head.
- [Radiance #10](https://github.com/magiccodingman/vllm-radiance/pull/10): conformance support.
- [Radiance #11](https://github.com/magiccodingman/vllm-radiance/pull/11): D7/execution-mode alignment.
- [vLLM #57280](https://github.com/vllm-project/vllm/pull/57280): gated-normalization tiling.
- [vLLM #57286](https://github.com/vllm-project/vllm/pull/57286): gfx1201 M8 head performance.

The final three feature commits introduce global-256, M1/M8 plus eager/compiled
alignment, then GGZ14 performance backports. Original component headers and license
texts remain; see `experiments/radiance-public/upstream-correctness/licenses` and
[LICENSE](LICENSE).
