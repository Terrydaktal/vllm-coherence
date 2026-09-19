# Provenance

Coherence is a standalone downstream fork of Radiance, maintained by
**Terrydaktal**. Its root commit imports the exact Radiance 1.0.16 tree;
subsequent feature commits identify their authors and upstream basis in their
commit titles and summaries. Original authorship is retained for inherited code
and upstream fixes.

The direct base is magiccodingman's Radiance, which continues StillDeadcode's
Radiance/libr4d work and incorporates GGZ14's MXFP4/W4A8 optimizations. vLLM and
the AMD runtime/compiler/operator stack underpin those projects. Coherence's
contribution is the additional repair, instrumentation, qualification and agent
session work described in the [README](README.md#what-terrydaktal-adds-in-coherence).

| Component | Origin |
| --- | --- |
| Radiance | [magiccodingman/vllm-radiance](https://github.com/magiccodingman/vllm-radiance), `f295b9ef51ad413a68e4192371e0377741a354ce`; canonical [Lance Wright GitLab](https://gitlab.sayou.io/lance-wright/vllm-radiance) |
| Original Radiance | [StillDeadcode/vllm-radiance](https://codeberg.org/StillDeadcode/vllm-radiance), the runtime lineage continued by the direct base |
| Serving/model infrastructure | [vllm-project/vllm](https://github.com/vllm-project/vllm) |
| RDNA4 kernels | [StillDeadcode/libr4d](https://codeberg.org/StillDeadcode/libr4d) |
| MXFP4/W4A8 foundation | [Brian / GGZ14](https://github.com/GGZ14/vllm-mxfp4) and the earlier [ggz14/radiance-vllm-mxfp4](https://codeberg.org/ggz14/radiance-vllm-mxfp4), adapted by Radiance before Coherence's own backports |
| Performance backports | [GGZ14 `3f542b7`](https://github.com/GGZ14/vllm-mxfp4/commit/3f542b7cbfce3fa0d01dc55665af4c77a7093ce8), guarded/adapted to the repaired arithmetic |
| GPU runtime and compiler stack | [ROCm](https://github.com/ROCm/rocm-systems), [AMD PyTorch](https://github.com/ROCm/pytorch), [AMD Triton](https://github.com/ROCm/triton); exact inherited build pins in [Dockerfile](Dockerfile) |
| Operator libraries | [AITER](https://github.com/ROCm/aiter) and [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention), used through the serving stack |
| Model and drafting method | [Qwen](https://github.com/QwenLM), [DFlash](https://github.com/z-lab/dflash), and the authors of the separately supplied target/drafter checkpoints |
| Pi client | [earendil-works/pi](https://github.com/earendil-works/pi), pinned 0.84.2, plus Coherence patches/extensions |

The [upstream-correctness manifest](experiments/radiance-public/upstream-correctness/manifest.json)
retains exact source and post-patch file hashes for every imported backport. The
human-readable PR and commit links are carried by the corresponding historical
commit summaries, so the history remains the primary provenance record.

Existing independent submissions remain available:

- [Radiance #8](https://github.com/magiccodingman/vllm-radiance/pull/8): DFlash RNG and extreme-decay GDN repair.
- [Radiance #9](https://github.com/magiccodingman/vllm-radiance/pull/9): global-256 target head.
- [Radiance #10](https://github.com/magiccodingman/vllm-radiance/pull/10): conformance support.
- [Radiance #11](https://github.com/magiccodingman/vllm-radiance/pull/11): D7/execution-mode alignment.
- [vLLM #57280](https://github.com/vllm-project/vllm/pull/57280): gated-normalization tiling.
- [vLLM #57286](https://github.com/vllm-project/vllm/pull/57286): gfx1201 M8 head performance.

Original component headers and license texts remain; see
`experiments/radiance-public/upstream-correctness/licenses` and [LICENSE](LICENSE).
