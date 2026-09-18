# Upstream submissions

With both fixes applied, eager M1 and compiled M8 match at all 10,000 decode
positions and 23 initial-prefill predictions, including full-vocabulary hashes.
The report compares the original and final pairs on the same Pi corpus; its
separate 320-position measurements isolate individual stages and execution-mode
differences. The integration limits below concern ports to other upstream builds.

| Scope | Owning project | Submission state |
| --- | --- | --- |
| GDN gated-normalization row tiling | vLLM, vendored FLA | [vLLM #57280](https://github.com/vllm-project/vllm/pull/57280), draft; 32 native synthetic cases passed. The complete server built from this new upstream branch still needs testing. |
| Arithmetic-preserving interleaved BF16 head | vLLM, ROCm skinny GEMMs | [vLLM #57286](https://github.com/vllm-project/vllm/pull/57286), draft; 32 native positions / 7,946,240 values per comparison passed, including graph replay. The new upstream extension/binding and model integration still need testing. [Probe supplement](publication/head/README.md). |
| Query-specific attention reduction with shared KV | libr4d | [libr4d #5](https://codeberg.org/StillDeadcode/libr4d/pulls/5), draft; current-header CPU compilation passed. Current-header native execution and public dispatch remain to qualify. |
| Causal GDN prefill and chunk independence | libr4d / Radiance adapters | [libr4d #6](https://codeberg.org/StillDeadcode/libr4d/pulls/6), draft; current-header CPU compilation passed. This R4D-M1 reference is distinct from the final stock-FLA contract. |
| Serial-contract speculative convolution and recurrence | libr4d / Radiance adapters | [libr4d #7](https://codeberg.org/StillDeadcode/libr4d/pulls/7), draft; 78 CPU checks passed, one compiler-artifact check skipped. A production HIP port remains separate work. The tested Radiance implementation is in #11 below. |
| Compiled numerical-contract integration | Radiance | [Radiance #11](https://github.com/magiccodingman/vllm-radiance/pull/11), draft; covers both the M1/M8 repair and the later eager/compiled rounding alignment, the four isolated comparison columns and fresh compiled timings. Depends on #10. |
| Forced-token conformance and stage qualification | Radiance | [Radiance #10](https://github.com/magiccodingman/vllm-radiance/pull/10), draft; includes execution-mode admission, native stage replay and evidence checks. Combined CPU validation is recorded in the PR descriptions; retained compiler-artifact checks are separate from native qualification. |
| Native BF16 RoPE multiplication rounding | Triton | [#11227](https://github.com/triton-lang/triton/pull/11227) merged August 14, 2026. The pinned compiler still uses the affected lowering; #11 contains the pinned eager intervention. No duplicate Triton PR is needed. |

The three [libr4d PRs, patches and descriptions](publication/libr4d/README.md)
are published with commit identities and hashes. They remain experimental drafts,
with the outstanding integration work documented individually.

Existing submissions remain separate:

- [Radiance #8: numerical corrections](https://github.com/magiccodingman/vllm-radiance/pull/8).
- [Radiance #9: target-head candidate selection](https://github.com/magiccodingman/vllm-radiance/pull/9).
- [vLLM #57007](https://github.com/vllm-project/vllm/pull/57007) was closed in
  favor of [#52905](https://github.com/vllm-project/vllm/pull/52905); no duplicate
  convolution-tail submission is planned.
- Residual-normalization overlap is tracked against
  [vLLM #49639](https://github.com/vllm-project/vllm/pull/49639) and
  [#52243](https://github.com/vllm-project/vllm/pull/52243).

The report and aggregate evidence are published independently of whether these
ports have been accepted. No universal correctness or production deployment is
implied by opening a PR.

Radiance's [README](https://github.com/magiccodingman/vllm-radiance/blob/main/README.md)
links GitLab as its source repository. The submissions above are on GitHub;
GitLab forwarding has not been performed. If the maintainer requires it, forward
the same #10/#11 branches and descriptions rather than creating another fix.
