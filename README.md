# vLLM Coherence

**Fast, observable inference with validated execution and persistent sessions.**

Coherence is an independently maintained **downstream fork of
[Radiance](https://github.com/magiccodingman/vllm-radiance)** for the AMD Radeon AI
PRO R9700. It combines numerical repairs, measured performance backports,
reusable conformance instrumentation and durable Pi sessions.

This is a standalone GitHub repository. Its first commit is the unchanged
Radiance 1.0.16 source at `f295b9ef51ad413a68e4192371e0377741a354ce`; subsequent
commits describe Coherence's additions. See [attribution](ATTRIBUTION.md).

[Quick start](#quick-start) · [Numerical report](reports/d7-rdna4-2026-09-17/REPORT.md)
· [Verification](docs/VERIFICATION.md) · [Architecture](docs/ARCHITECTURE.md)
· [Pi](docs/PI.md) · [Releases](https://github.com/Terrydaktal/vllm-coherence/releases)

## What is included

- **Consistent target execution:** repaired GDN convolution, recurrence and
  prefill; matching normalization, attention and vocabulary-head arithmetic;
  preserved intermediate BF16 rounding in compiled execution; the pinned native
  RoPE rounding correction.
- **Measured performance work:** guarded wide GEMM dispatch, normalization/FP8
  fusion, GDN spatial scan tiling and tiled prefill activation layout, retaining
  the existing recurrent-state layout.
- **Global-256 target head:** search the complete INT2 score row, then rescore
  256 candidates using BF16 weights. Removes the eight-candidates-per-tile capacity
  defect; unsupported sampling modes retain a full-head fallback.
- **Persistent conversations:** compressed incremental snapshots, buffered tails,
  explicit flushes, verified publication before retiring old heads, generation-
  aware garbage collection and cumulative disk-write accounting.
- **Shared-GPU scheduling:** hand over at response/tool boundaries, retain short
  tool calls with a grace period, and support per-answer `/priority`.
- **Transparent Pi:** prefill/restore/queue phases, a three-second token-rate
  window, context/residency counters, shared temperature/fan probes, expandable
  backend errors and transactional compaction that preserves editor input.
- **Reusable verification:** forced-token replay, logical-state comparison,
  first-divergence capture, operator checks, deliberate fault injection and small,
  explicitly scoped machine-checked obligations.

<!-- COHERENCE_CURRENT_RESULTS -->
## Current numerical results

Current backported compiled M8 versus the aligned pre-backport compiled M8 control, using the **full BF16 target head**: 320 forced decode tokens on the same 60K Pi prefix, plus one prefill prediction. The 10K eager-M1/compiled-M8 study belongs to the preceding alignment revision; it was not rerun after these backports.

| Prediction | Same token set | Same ordering | Mean shared tokens |
| --- | ---: | ---: | ---: |
| Top 1 | 320 / 320 (100.00%) | 320 / 320 (100.00%) | 1.0000 / 1 |
| Top 10 | 320 / 320 (100.00%) | 320 / 320 (100.00%) | 10.0000 / 10 |
| Top 20 | 320 / 320 (100.00%) | 320 / 320 (100.00%) | 20.0000 / 20 |

All 320 full-vocabulary hashes and the prefill prediction matched. These are finite consistency checks, not model task accuracy, an arbitrary-input proof, or certification of approximate global-256 selection.

## Compiled backend stages

Latest retained **compiled, piecewise-graph** decode profile, 60K-input Pi fixture, global-256 head, both alignment repairs and the GGZ14-derived backports. Seven of eight rounds have a complete modal kernel inventory; the incomplete round is excluded by inventory, never by its duration. The subsequent tiled-activation change is prefill-only and does not change these decode kernels. Every observed dispatch in the retained rounds is counted once; fused constituents have no separately measurable time.

**Set/order** means the same top-20 token set, followed by the same ranking. For example, `320/320; 320/320` means both checks passed at every tested token. Operator-byte checks and whole-model checks are labelled separately; each result retains its stated test scope.

| Stage | Current GPU ms per round | Current correctness evidence | Implementation / measurement boundary |
| --- | ---: | --- | --- |
| Embedding + first input normalization + FP8 production | 0.010 | 320 rows + adversarial values: exact FP8/scales/BF16 residual; prefill 1,000 rows per site | Native rounding retained; fused FP8 output replaces a separate quantizer. Prefill uses its own admitted reduction layout. |
| Layer input residual/normalization + FP8 production | 0.512 | 320 rows + adversarial values: exact FP8/scales/BF16 residual; prefill 1,000 rows per site | Native rounding retained; fused FP8 output replaces a separate quantizer. Prefill uses its own admitted reduction layout. |
| GDN input activation FP8 quantization | Included in input norm | Exact fused FP8 bytes/scales; see norm row | FP8 production is fused into layer input normalization. |
| GDN input projection | 3.948 | 320/320; 320/320 in alignment study; unchanged operator | Same corrected operator; measured again in the retained backport trace. |
| GDN layout/copies and buffer initialization | 0.307 | State/layout checked with convolution and recurrence | Packed QKV views and existing state layout. |
| GDN convolution | 0.210 | 320/320; 320/320 in alignment study; unchanged operator | Corrected serial product/accumulation order and rolling history. |
| GDN recurrence and gates | 1.183 | Decode unchanged; tiled prefill output/state exact at 1/8/64/320/1,000/1,648/2,048 rows | Corrected chronological transitions; spatial tiling changes prefill only. Existing nine-slot state layout. |
| GDN output gated normalization + FP8 production | 0.161 | 1,000 rows × 48 sites × M1/M8: exact bytes and scales | All 48 sites fused; native intermediate BF16 rounding and Gluon layout retained. |
| GDN output activation FP8 quantization | Included in GDN output norm | 1,000 rows × 48 sites × M1/M8: exact bytes and scales | One fused gated-normalization/quantization launch. |
| GDN output projection | 1.899 | 320/320; 320/320 in alignment study; unchanged operator | Same corrected operator; measured again in the retained backport trace. |
| Attention input activation FP8 quantization | Included in input norm | Exact fused FP8 bytes/scales; see norm row | FP8 production is fused into layer input normalization. |
| Attention input projection | 1.161 | 320/320; 320/320 in alignment study; unchanged operator | Same corrected operator; measured again in the retained backport trace. |
| Attention Q/K normalization, RoPE and layout | 0.218 | 320/320; 320/320 in alignment study; unchanged operator | Same corrected operator; measured again in the retained backport trace. |
| Attention KV write | 0.044 | 320/320; 320/320 in alignment study; unchanged operator | Same corrected operator; measured again in the retained backport trace. |
| Attention decode | 5.119 | 320/320; 320/320 in alignment study; unchanged operator | Corrected causal tile/reduction policy; shared KV loads. |
| Attention split-KV merge | 0.128 | 320/320; 320/320 in alignment study; unchanged operator | Corrected serial split/merge arithmetic. |
| Attention output gating | 0.031 | 320/320; 320/320 in alignment study; unchanged operator | Same corrected operator; measured again in the retained backport trace. |
| Attention output activation FP8 quantization | 0.043 | 320/320; 320/320 in alignment study; unchanged operator | Same corrected operator; measured again in the retained backport trace. |
| Attention output projection | 0.528 | 320/320; 320/320 in alignment study; unchanged operator | Same corrected operator; measured again in the retained backport trace. |
| Post-attention/GDN residual/normalization + FP8 production | 0.533 | 320 rows + adversarial values: exact FP8/scales/BF16 residual; prefill 1,000 rows per site | Native rounding retained; fused FP8 output replaces a separate quantizer. Prefill uses its own admitted reduction layout. |
| MLP gate/up input FP8 quantization | Included in post norm | Exact fused FP8 bytes/scales; see norm row | FP8 production is fused into post-attention/GDN normalization. |
| MLP gate/up projection | 11.375 | 115,841,664 elements: 0 differences; 320 rows on 3 checkpoint matrices plus boundary cases | Guarded wide-N decode dispatch replaces folded GEMM; four projections per layer still validated. |
| MLP SiLU and gating | 0.180 | 320/320; 320/320 in alignment study; unchanged operator | BF16 intermediate preserved; no speculative slower SiLU backport. |
| MLP down input FP8 quantization | 0.265 | 320/320; 320/320 in alignment study; unchanged operator | Same corrected operator; measured again in the retained backport trace. |
| MLP down projection | 5.369 | 320/320; 320/320 in alignment study; unchanged operator | Same corrected operator; measured again in the retained backport trace. |
| Final normalization/layout | 0.005 | 320/320; 320/320 in alignment study; unchanged operator | Same corrected operator; measured again in the retained backport trace. |
| Global-256 target head | 1.012 | Approximate: head-study top-20 retained 119,786/119,988; ranking/probabilities not certified | Current serving head; whole-vocabulary INT2 selection then BF16 rerank. Full-head correctness controls are separate. |
| Drafter | 6.368 | N/A: no isolated target top-20 prediction | Same corrected operator; measured again in the retained backport trace. |
| Other GPU bookkeeping | 0.527 | N/A: no isolated target top-20 prediction | Sampling/state bookkeeping outside the model scopes; kernel list below. |

**Sum of measured GPU dispatch durations: 41.133 ms per profiled round.** This sum excludes host gaps and queue time and is not the uninstrumented round timer.

<details>
<summary>Every decoder layer: current projection and remaining-work timings</summary>

Each layer has four projections. Gate and up are one joint GEMM; there is no separately measured gate/up split.

| Layer | Type | All layer work ms | Input projection ms | Output projection ms | Gate/up projection ms | Down projection ms | Other work ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | GDN | 0.4341 | 0.0758 | 0.0391 | 0.1749 | 0.0800 | 0.0644 |
| 1 | GDN | 0.4523 | 0.0807 | 0.0398 | 0.1834 | 0.0871 | 0.0612 |
| 2 | GDN | 0.4567 | 0.0835 | 0.0396 | 0.1834 | 0.0880 | 0.0623 |
| 3 | Attention | 0.7360 | 0.0733 | 0.0330 | 0.1704 | 0.0806 | 0.3787 |
| 4 | GDN | 0.4372 | 0.0820 | 0.0385 | 0.1738 | 0.0809 | 0.0620 |
| 5 | GDN | 0.4498 | 0.0812 | 0.0404 | 0.1828 | 0.0853 | 0.0601 |
| 6 | GDN | 0.4576 | 0.0845 | 0.0397 | 0.1824 | 0.0886 | 0.0624 |
| 7 | Attention | 0.7335 | 0.0720 | 0.0330 | 0.1724 | 0.0802 | 0.3759 |
| 8 | GDN | 0.4375 | 0.0821 | 0.0386 | 0.1732 | 0.0809 | 0.0628 |
| 9 | GDN | 0.4526 | 0.0807 | 0.0397 | 0.1824 | 0.0885 | 0.0613 |
| 10 | GDN | 0.4618 | 0.0842 | 0.0402 | 0.1856 | 0.0887 | 0.0629 |
| 11 | Attention | 0.7294 | 0.0731 | 0.0330 | 0.1701 | 0.0804 | 0.3728 |
| 12 | GDN | 0.4380 | 0.0815 | 0.0393 | 0.1756 | 0.0800 | 0.0616 |
| 13 | GDN | 0.4522 | 0.0805 | 0.0398 | 0.1835 | 0.0879 | 0.0604 |
| 14 | GDN | 0.4571 | 0.0846 | 0.0400 | 0.1819 | 0.0881 | 0.0624 |
| 15 | Attention | 0.7318 | 0.0732 | 0.0332 | 0.1711 | 0.0807 | 0.3736 |
| 16 | GDN | 0.4367 | 0.0815 | 0.0387 | 0.1738 | 0.0806 | 0.0622 |
| 17 | GDN | 0.4497 | 0.0813 | 0.0399 | 0.1814 | 0.0859 | 0.0612 |
| 18 | GDN | 0.4572 | 0.0835 | 0.0392 | 0.1826 | 0.0885 | 0.0634 |
| 19 | Attention | 0.7307 | 0.0729 | 0.0331 | 0.1711 | 0.0804 | 0.3733 |
| 20 | GDN | 0.4398 | 0.0817 | 0.0390 | 0.1753 | 0.0807 | 0.0631 |
| 21 | GDN | 0.4499 | 0.0815 | 0.0400 | 0.1816 | 0.0857 | 0.0610 |
| 22 | GDN | 0.4560 | 0.0837 | 0.0397 | 0.1826 | 0.0878 | 0.0623 |
| 23 | Attention | 0.7329 | 0.0721 | 0.0330 | 0.1702 | 0.0805 | 0.3771 |
| 24 | GDN | 0.4373 | 0.0816 | 0.0392 | 0.1738 | 0.0802 | 0.0625 |
| 25 | GDN | 0.4541 | 0.0811 | 0.0404 | 0.1840 | 0.0873 | 0.0614 |
| 26 | GDN | 0.4554 | 0.0840 | 0.0394 | 0.1814 | 0.0873 | 0.0633 |
| 27 | Attention | 0.7300 | 0.0731 | 0.0328 | 0.1703 | 0.0803 | 0.3735 |
| 28 | GDN | 0.4377 | 0.0813 | 0.0393 | 0.1746 | 0.0805 | 0.0620 |
| 29 | GDN | 0.4530 | 0.0824 | 0.0398 | 0.1837 | 0.0859 | 0.0612 |
| 30 | GDN | 0.4583 | 0.0832 | 0.0400 | 0.1846 | 0.0876 | 0.0630 |
| 31 | Attention | 0.7293 | 0.0722 | 0.0329 | 0.1691 | 0.0804 | 0.3747 |
| 32 | GDN | 0.4375 | 0.0814 | 0.0390 | 0.1736 | 0.0807 | 0.0628 |
| 33 | GDN | 0.4506 | 0.0814 | 0.0403 | 0.1808 | 0.0870 | 0.0611 |
| 34 | GDN | 0.4597 | 0.0833 | 0.0395 | 0.1833 | 0.0875 | 0.0661 |
| 35 | Attention | 0.7303 | 0.0737 | 0.0332 | 0.1702 | 0.0803 | 0.3730 |
| 36 | GDN | 0.4415 | 0.0824 | 0.0394 | 0.1758 | 0.0814 | 0.0626 |
| 37 | GDN | 0.4504 | 0.0819 | 0.0397 | 0.1821 | 0.0861 | 0.0605 |
| 38 | GDN | 0.4581 | 0.0841 | 0.0398 | 0.1841 | 0.0880 | 0.0622 |
| 39 | Attention | 0.7265 | 0.0727 | 0.0328 | 0.1701 | 0.0805 | 0.3703 |
| 40 | GDN | 0.4402 | 0.0816 | 0.0389 | 0.1761 | 0.0814 | 0.0623 |
| 41 | GDN | 0.4501 | 0.0808 | 0.0398 | 0.1825 | 0.0868 | 0.0602 |
| 42 | GDN | 0.4608 | 0.0848 | 0.0404 | 0.1853 | 0.0879 | 0.0624 |
| 43 | Attention | 0.7260 | 0.0721 | 0.0330 | 0.1705 | 0.0806 | 0.3698 |
| 44 | GDN | 0.4357 | 0.0815 | 0.0386 | 0.1750 | 0.0798 | 0.0609 |
| 45 | GDN | 0.4512 | 0.0814 | 0.0397 | 0.1821 | 0.0875 | 0.0605 |
| 46 | GDN | 0.4556 | 0.0845 | 0.0400 | 0.1821 | 0.0873 | 0.0617 |
| 47 | Attention | 0.7316 | 0.0719 | 0.0330 | 0.1707 | 0.0806 | 0.3753 |
| 48 | GDN | 0.4406 | 0.0823 | 0.0387 | 0.1755 | 0.0806 | 0.0635 |
| 49 | GDN | 0.4504 | 0.0820 | 0.0391 | 0.1818 | 0.0872 | 0.0603 |
| 50 | GDN | 0.4597 | 0.0849 | 0.0405 | 0.1849 | 0.0873 | 0.0620 |
| 51 | Attention | 0.7252 | 0.0726 | 0.0329 | 0.1702 | 0.0801 | 0.3694 |
| 52 | GDN | 0.4369 | 0.0817 | 0.0388 | 0.1751 | 0.0796 | 0.0617 |
| 53 | GDN | 0.4514 | 0.0825 | 0.0398 | 0.1816 | 0.0874 | 0.0600 |
| 54 | GDN | 0.4538 | 0.0844 | 0.0399 | 0.1818 | 0.0855 | 0.0623 |
| 55 | Attention | 0.7274 | 0.0709 | 0.0330 | 0.1704 | 0.0800 | 0.3732 |
| 56 | GDN | 0.4376 | 0.0818 | 0.0385 | 0.1747 | 0.0808 | 0.0618 |
| 57 | GDN | 0.4507 | 0.0806 | 0.0402 | 0.1830 | 0.0870 | 0.0598 |
| 58 | GDN | 0.4559 | 0.0831 | 0.0398 | 0.1829 | 0.0876 | 0.0625 |
| 59 | Attention | 0.7261 | 0.0724 | 0.0328 | 0.1708 | 0.0805 | 0.3696 |
| 60 | GDN | 0.4367 | 0.0819 | 0.0391 | 0.1737 | 0.0806 | 0.0614 |
| 61 | GDN | 0.4519 | 0.0808 | 0.0399 | 0.1832 | 0.0882 | 0.0598 |
| 62 | GDN | 0.4579 | 0.0845 | 0.0398 | 0.1834 | 0.0869 | 0.0632 |
| 63 | Attention | 0.7279 | 0.0728 | 0.0331 | 0.1710 | 0.0801 | 0.3709 |

</details>

<details>
<summary>Every recorded kernel, grouped by stage</summary>

| Stage / compiled kernel | Calls in retained rounds | Current GPU ms per round |
| --- | ---: | ---: |
| Drafter / `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x32_MI16x16x1_SN_LDSB1_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_CD1_1_CLR0_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB1_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA1201_ICIW0_IU1_K1_LDSTI0_LBSPPA128_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB0_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB2_ONLL1_PAP0_PGL0_PGR1_PLR1_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1.kd` | 7 | 0.007090 |
| Drafter / `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x32x256_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB1_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA1201_ICIW0_IU1_K1_LDSTI0_LBSPPA512_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB0_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB16_ONLL0_PAP0_PGL0_PGR1_PLR1_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_4_1.kd` | 7 | 0.176760 |
| Drafter / `__amd_rocclr_copyBuffer.kd` | 7 | 0.001948 |
| Drafter / `__amd_rocclr_fillBufferAligned.kd` | 21 | 0.006877 |
| Drafter / `_cache_draft_logits_kernel.kd` | 7 | 0.002416 |
| Drafter / `_draft_head_int2.kd` | 7 | 0.826031 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_17408_EVEN_K_1_GRID_MN_40_cache_modifier_NONE.kd` | 35 | 0.738124 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_25600_EVEN_K_1_GRID_MN_40_cache_modifier_NONE.kd` | 7 | 0.235794 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_40_cache_modifier_NONE.kd` | 35 | 0.202362 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_5120_EVEN_K_1_GRID_MN_272_cache_modifier_NONE.kd` | 35 | 1.447715 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_5120_EVEN_K_1_GRID_MN_48_cache_modifier_NONE.kd` | 35 | 0.272956 |
| Drafter / `_prepare_dflash_inputs_kernel.kd` | 7 | 0.007542 |
| Drafter / `_rerank_exact.kd` | 7 | 0.008850 |
| Drafter / `_selector_walk_kernel.kd` | 7 | 0.007993 |
| Drafter / `kernel_unified_attention.kd` | 35 | 1.843762 |
| Drafter / `reshape_and_cache_kernel_flash.kd` | 70 | 0.026356 |
| Drafter / `triton_per_fused_4.kd` | 7 | 0.001942 |
| Drafter / `triton_per_fused_8.kd` | 28 | 0.007527 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_preshuffle_gemm_squeeze_view_0.kd` | 35 | 0.009738 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_preshuffle_gemm_squeeze_view_2.kd` | 7 | 0.001999 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_preshuffle_gemm_squeeze_view_4.kd` | 63 | 0.016128 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_view_0.kd` | 7 | 0.004988 |
| Drafter / `triton_poi_fused_0.kd` | 7 | 0.002662 |
| Drafter / `triton_poi_fused_5.kd` | 7 | 0.002228 |
| Drafter / `triton_poi_fused_9.kd` | 28 | 0.010059 |
| Drafter / `triton_poi_fused__to_copy_clamp_div_preshuffle_gemm_squeeze_view_1.kd` | 35 | 0.012252 |
| Drafter / `triton_poi_fused__to_copy_clamp_div_preshuffle_gemm_squeeze_view_3.kd` | 7 | 0.002028 |
| Drafter / `triton_poi_fused__to_copy_clamp_div_preshuffle_gemm_squeeze_view_5.kd` | 63 | 0.016614 |
| Drafter / `triton_poi_fused_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_3.kd` | 63 | 0.022545 |
| Drafter / `triton_poi_fused_add_arange_bitwise_and_constant_pad_nd_ge_mul_rms_norm_select_slice_unsqueeze_view_1.kd` | 7 | 0.023102 |
| Drafter / `triton_poi_fused_add_permute_unsqueeze_view_2.kd` | 7 | 0.001810 |
| Drafter / `triton_poi_fused_cat_expand_index_mul_slice_unsqueeze_view_1.kd` | 7 | 0.002519 |
| Drafter / `triton_red_fused__to_copy_abs_clamp_div_max_mul_preshuffle_gemm_silu_slice_squeeze_view_6.kd` | 35 | 0.017275 |
| Drafter / `triton_red_fused__to_copy_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_w4_gemm_2.kd` | 35 | 0.031390 |
| Drafter / `triton_red_fused__to_copy_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_w4_gemm_7.kd` | 28 | 0.025836 |
| Drafter / `triton_red_fused__to_copy_embedding_mul_rms_norm_w4_gemm_0.kd` | 7 | 0.004245 |
| Drafter / `triton_red_fused_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_7.kd` | 7 | 0.006593 |
| Drafter / `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) [clone .kd]` | 14 | 0.009198 |
| Drafter / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 7 | 0.003570 |
| Drafter / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 7 | 0.003256 |
| Drafter / `void at::native::bitonicSortKVInPlace<2, -1, 16, 16, c10::BFloat16, long, at::native::GTOp<c10::BFloat16, true>, unsigned int>(at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, at::native::GTOp<c10::BFloat16, true>) [clone .kd]` | 7 | 0.003102 |
| Drafter / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 7 | 0.004953 |
| Drafter / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 7 | 0.002519 |
| Drafter / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 7 | 0.003496 |
| Drafter / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 7 | 0.003096 |
| Drafter / `void at::native::mbtopk::computeBlockDigitCounts<c10::BFloat16, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 14 | 0.038553 |
| Drafter / `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 28 | 0.055408 |
| Drafter / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, c10::BFloat16>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 14 | 0.022135 |
| Drafter / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, float>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, float*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 28 | 0.028647 |
| Drafter / `void at::native::mbtopk::fill<unsigned int, unsigned int>(unsigned int*, unsigned int, unsigned int) [clone .kd]` | 14 | 0.002901 |
| Drafter / `void at::native::mbtopk::gatherTopK<c10::BFloat16, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 7 | 0.023502 |
| Drafter / `void at::native::mbtopk::gatherTopK<float, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, float*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 7 | 0.008902 |
| Drafter / `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4>) [clone .kd]` | 7 | 0.003793 |
| Drafter / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul>) [clone .kd]` | 7 | 0.001719 |
| Drafter / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) [clone .kd]` | 7 | 0.002450 |
| Drafter / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>) [clone .kd]` | 14 | 0.003855 |
| Drafter / `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>) [clone .kd]` | 14 | 0.005192 |
| Drafter / `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool) [clone .kd]` | 7 | 0.002090 |
| Drafter / `void at::native::warpMergeSortKVInPlace<2, -1, 128, 16, float, long, at::native::GTOp<float, true>, unsigned int, 32>(at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, at::native::GTOp<float, true>, float) [clone .kd]` | 7 | 0.004913 |
| Drafter / `void r4d_gemm_w4a16_nt_m64_kernel<1, 1, false>(unsigned short const*, unsigned int const*, unsigned int const*, __hip_bfloat16*, int, int, int, int, int) [clone .kd]` | 7 | 0.004462 |
| Drafter / `void r4d_gemm_w4a16_nt_m64_kernel<1, 1, true>(unsigned short const*, unsigned int const*, unsigned int const*, __hip_bfloat16*, int, int, int, int, int) [clone .kd]` | 70 | 0.081664 |
| Drafter / `void vllm::rms_norm_kernel<c10::BFloat16, 8, 2, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, long, long, long, c10::BFloat16 const*, long, float, int, int) [clone .kd]` | 7 | 0.002965 |
| Drafter / `void vllm::rms_norm_kernel<c10::BFloat16, 8, 4, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, long, long, long, c10::BFloat16 const*, long, float, int, int) [clone .kd]` | 7 | 0.002753 |
| Drafter / `void vllm::rotary_embedding_kernel<c10::BFloat16, c10::BFloat16, true>(long const*, c10::BFloat16*, c10::BFloat16*, c10::BFloat16 const*, int, long, long, long, int, int, int, long, bool) [clone .kd]` | 7 | 0.002382 |
| Attention KV write / `reshape_and_cache_kernel_flash.kd` | 112 | 0.043841 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_1.kd` | 112 | 0.020961 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_3.kd` | 112 | 0.028590 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_4.kd` | 112 | 0.025390 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_arange_bitwise_and_eq_index_lt_remainder_select_split_where_2.kd` | 112 | 0.028150 |
| Attention Q/K normalization, RoPE and layout / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 112 | 0.029818 |
| Attention Q/K normalization, RoPE and layout / `void stock_m1_gemma_norm<false, 256, 32>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, long, long, float, float*) [clone .kd]` | 112 | 0.047344 |
| Attention Q/K normalization, RoPE and layout / `void stock_m1_gemma_norm<false, 256, 64>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, long, long, float, float*) [clone .kd]` | 112 | 0.037333 |
| Attention decode / `void qwen_stock_m1_shared_decode<4, 16, 256, 6, 16, 0, 3430971>(R4DArgs, int) [clone .kd]` | 112 | 5.119284 |
| Attention input projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 1, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 112 | 1.160880 |
| Attention output activation FP8 quantization / `void vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<c10::BFloat16, c10::Float8_e4m3fn>(c10::Float8_e4m3fn*, float*, c10::BFloat16 const*, float const*, int, long, long) [clone .kd]` | 112 | 0.043224 |
| Attention output gating / `triton_poi_fused_dynamic_per_token_scaled_fp8_quant_mul_sigmoid_view_0.kd` | 112 | 0.031133 |
| Attention output projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 4, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 112 | 0.527620 |
| Attention split-KV merge / `void qwen_stock_m1_shared_merge<256, 4, 1>(R4DArgs, int, int) [clone .kd]` | 112 | 0.128076 |
| Embedding + first input normalization / `triton_poi_fused__to_copy_embedding_0.kd` | 7 | 0.002405 |
| Embedding + first input normalization / `void norm_quant<false, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned char*, float*, unsigned short*, long, long, float) [clone .kd]` | 7 | 0.007176 |
| Final normalization/layout / `void stock_m1_gemma_norm<true, 5120, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, long, long, float, float*) [clone .kd]` | 7 | 0.005130 |
| GDN convolution / `_causal_conv1d_update_kernel.kd` | 336 | 0.209805 |
| GDN input projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 1, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 336 | 3.947534 |
| GDN layout/copies and buffer initialization / `triton_poi_fused_add_1.kd` | 21 | 0.003906 |
| GDN layout/copies and buffer initialization / `triton_poi_fused_add_2.kd` | 14 | 0.002581 |
| GDN layout/copies and buffer initialization / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 336 | 0.146159 |
| GDN layout/copies and buffer initialization / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) [clone .kd]` | 336 | 0.085546 |
| GDN layout/copies and buffer initialization / `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>) [clone .kd]` | 336 | 0.068815 |
| GDN output gated normalization / `gdn_norm_quant_kernel.kd` | 336 | 0.160655 |
| GDN output projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 4, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 336 | 1.898577 |
| GDN recurrence and gates / `stock_gdn_scan_kernel.kd` | 336 | 1.183110 |
| Layer input residual/normalization / `void norm_quant<true, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned char*, float*, unsigned short*, long, long, float) [clone .kd]` | 441 | 0.512087 |
| MLP SiLU and gating / `triton_poi_fused_dynamic_per_token_scaled_fp8_quant_mul_silu_slice_0.kd` | 336 | 0.125907 |
| MLP SiLU and gating / `triton_poi_fused_dynamic_per_token_scaled_fp8_quant_mul_silu_slice_1.kd` | 112 | 0.053767 |
| MLP down input FP8 quantization / `void vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<c10::BFloat16, c10::Float8_e4m3fn>(c10::Float8_e4m3fn*, float*, c10::BFloat16 const*, float const*, int, long, long) [clone .kd]` | 448 | 0.265069 |
| MLP down projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 4, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 448 | 5.368755 |
| MLP gate/up projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 1, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 448 | 11.374764 |
| Post-attention/GDN residual/normalization / `void norm_quant<true, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned char*, float*, unsigned short*, long, long, float) [clone .kd]` | 448 | 0.533053 |
| Target head (global256) / `__amd_rocclr_fillBufferAligned.kd` | 7 | 0.002422 |
| Target head (global256) / `_draft_head_int2.kd` | 7 | 0.774574 |
| Target head (global256) / `_rerank_exact.kd` | 7 | 0.029519 |
| Target head (global256) / `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) [clone .kd]` | 7 | 0.004508 |
| Target head (global256) / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 7 | 0.003490 |
| Target head (global256) / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 7 | 0.002776 |
| Target head (global256) / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 7 | 0.005182 |
| Target head (global256) / `void at::native::mbtopk::computeBlockDigitCounts<c10::BFloat16, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 14 | 0.090736 |
| Target head (global256) / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, c10::BFloat16>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 14 | 0.038444 |
| Target head (global256) / `void at::native::mbtopk::fill<unsigned int, unsigned int>(unsigned int*, unsigned int, unsigned int) [clone .kd]` | 7 | 0.001530 |
| Target head (global256) / `void at::native::mbtopk::gatherTopK<c10::BFloat16, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 7 | 0.040611 |
| Target head (global256) / `void at::native::radixSortKVInPlace<2, -1, 128, 8, c10::BFloat16, long, unsigned int>(at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, bool) [clone .kd]` | 7 | 0.005410 |
| Target head (global256) / `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4>) [clone .kd]` | 7 | 0.003833 |
| Target head (global256) / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) [clone .kd]` | 7 | 0.001633 |
| Target head (global256) / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>) [clone .kd]` | 7 | 0.002228 |
| Target head (global256) / `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>) [clone .kd]` | 14 | 0.005169 |
| Other GPU bookkeeping / `__amd_rocclr_copyBuffer.kd` | 175 | 0.066467 |
| Other GPU bookkeeping / `__amd_rocclr_fillBufferAligned.kd` | 7 | 0.002296 |
| Other GPU bookkeeping / `_combine_sampled_and_draft_tokens_kernel.kd` | 7 | 0.003010 |
| Other GPU bookkeeping / `_compute_local_logits_stats_kernel.kd` | 7 | 0.028256 |
| Other GPU bookkeeping / `_compute_slot_mappings_kernel.kd` | 7 | 0.002942 |
| Other GPU bookkeeping / `_expand_idx_mapping_kernel.kd` | 7 | 0.001896 |
| Other GPU bookkeeping / `_gather_block_tables_kernel.kd` | 7 | 0.004685 |
| Other GPU bookkeeping / `_get_num_sampled_and_rejected_kernel.kd` | 7 | 0.002542 |
| Other GPU bookkeeping / `_insert_resampled_kernel.kd` | 7 | 0.003410 |
| Other GPU bookkeeping / `_post_update_kernel.kd` | 7 | 0.004913 |
| Other GPU bookkeeping / `_prepare_pos_seq_lens_kernel.kd` | 7 | 0.002068 |
| Other GPU bookkeeping / `_prepare_rope_positions_kernel.kd` | 7 | 0.002759 |
| Other GPU bookkeeping / `_rejection_kernel.kd` | 7 | 0.007245 |
| Other GPU bookkeeping / `_resample_kernel.kd` | 7 | 0.014068 |
| Other GPU bookkeeping / `_scatter_num_accepted_kernel.kd` | 7 | 0.002073 |
| Other GPU bookkeeping / `_temperature_kernel.kd` | 7 | 0.013805 |
| Other GPU bookkeeping / `postprocess_mamba_fused_kernel.kd` | 7 | 0.002519 |
| Other GPU bookkeeping / `precopy_mamba_align_fused_kernel.kd` | 7 | 0.002770 |
| Other GPU bookkeeping / `preprocess_mamba_align_fused_kernel.kd` | 7 | 0.002622 |
| Other GPU bookkeeping / `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}>(int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}, function_traits<at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}>::result_type*) [clone .kd]` | 42 | 0.013063 |
| Other GPU bookkeeping / `void (anonymous namespace)::softmax_warp_forward<float, float, float, 6, false, false, 32>(float*, float const*, int, int, int, bool const*, int, bool) [clone .kd]` | 7 | 0.002273 |
| Other GPU bookkeeping / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 49 | 0.013667 |
| Other GPU bookkeeping / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 7 | 0.002428 |
| Other GPU bookkeeping / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 49 | 0.014016 |
| Other GPU bookkeeping / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<int> const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<int> const&)::{lambda(int, bool)#1}) [clone .kd]` | 42 | 0.014634 |
| Other GPU bookkeeping / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<float> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<float> const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<float> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<float> const&)::{lambda(int, bool)#1}) [clone .kd]` | 21 | 0.018666 |
| Other GPU bookkeeping / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 112 | 0.048521 |
| Other GPU bookkeeping / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 14 | 0.005895 |
| Other GPU bookkeeping / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 7 | 0.003056 |
| Other GPU bookkeeping / `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 28 | 0.060642 |
| Other GPU bookkeeping / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, float>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, float*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 28 | 0.033745 |
| Other GPU bookkeeping / `void at::native::mbtopk::fill<unsigned int, unsigned int>(unsigned int*, unsigned int, unsigned int) [clone .kd]` | 7 | 0.001468 |
| Other GPU bookkeeping / `void at::native::mbtopk::gatherTopK<float, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, float*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 7 | 0.026502 |
| Other GPU bookkeeping / `void at::native::tensor_kernel_scan_innermost_dim<float, std::plus<float> >(float*, float const*, unsigned int, unsigned int, unsigned int, float, std::plus<float>) [clone .kd]` | 7 | 0.002468 |
| Other GPU bookkeeping / `void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>, 4, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast) [clone .kd]` | 42 | 0.010285 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<16, at::native::BinaryFunctor<bool, bool, bool, at::native::BitwiseOrFunctor<bool> >, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<bool, bool, bool, at::native::BitwiseOrFunctor<bool> >, std::array<char*, 3ul>) [clone .kd]` | 7 | 0.002468 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<16, at::native::FillFunctor<bool>, std::array<char*, 1ul> >(int, at::native::FillFunctor<bool>, std::array<char*, 1ul>) [clone .kd]` | 7 | 0.001833 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<16, at::native::bitwise_not_kernel_cuda(at::TensorIteratorBase&)::{lambda(bool)#1}, std::array<char*, 2ul> >(int, at::native::bitwise_not_kernel_cuda(at::TensorIteratorBase&)::{lambda(bool)#1}, std::array<char*, 2ul>) [clone .kd]` | 7 | 0.002313 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}, std::array<char*, 2ul>) [clone .kd]` | 42 | 0.012428 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}, std::array<char*, 2ul>) [clone .kd]` | 7 | 0.001759 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}, std::array<char*, 3ul> >(int, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}, std::array<char*, 3ul>) [clone .kd]` | 7 | 0.009730 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool, float, float)#1}, std::array<char*, 4ul> >(int, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool, float, float)#1}, std::array<char*, 4ul>) [clone .kd]` | 14 | 0.004272 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 2ul> >(int, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 2ul>) [clone .kd]` | 42 | 0.013468 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, 2ul>) [clone .kd]` | 49 | 0.011862 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul>) [clone .kd]` | 7 | 0.001999 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>) [clone .kd]` | 7 | 0.001913 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>) [clone .kd]` | 14 | 0.003404 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int, at::native::FillFunctor<int>, std::array<char*, 1ul>) [clone .kd]` | 7 | 0.001679 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>) [clone .kd]` | 7 | 0.009685 |
| Other GPU bookkeeping / `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool) [clone .kd]` | 7 | 0.001999 |
| Other GPU bookkeeping / `void at::native::warpMergeSortKVInPlace<2, -1, 128, 16, float, long, at::native::GTOp<float, true>, unsigned int, 32>(at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, at::native::GTOp<float, true>, float) [clone .kd]` | 7 | 0.004656 |

</details>

## Uninstrumented performance

Normal compiled serving with snapshots enabled, no diagnostic worker or timing hooks. Three natural completions on a 60,000-input-token private Pi fixture; 2,375 generated tokens, temperature 0.6, top-p 0.95, top-k 20. Current global-256 head and all selected backports.

| Measurement | Current result |
| --- | ---: |
| Weighted mean generation round | 44.037 ms |
| Pooled rate after first output | 124.24 tok/s |
| Output tokens / natural responses | 2,375 / 3 |
| Cold 60K time to first token | 31.98 s (single observation) |
| Pi-temperature-1.0 pooled backend rate | 120.65 tok/s; 2,363 tokens across 3 natural completions |

These are brief 60K-input controls, **not** 60K-output throughput and not an end-to-end Pi UI/VM benchmark. Cold first-token time includes CPU/request overhead and initial generation. Throughput varies with draft acceptance, context and GPU clocks. [Measurements and limits](docs/pi-prefill-generation-20260918.md).

Aggregate data: [current measurements](benchmarks/results/coherence-current.json). Historical methodology and detailed numerical evidence: [technical report](reports/d7-rdna4-2026-09-17/REPORT.md).

<!-- /COHERENCE_CURRENT_RESULTS -->

## Global-256 target-head benchmark

This is the table from [the top-256 PR](https://github.com/magiccodingman/vllm-radiance/pull/9).
It is a **separate, earlier 60K-generated-token benchmark per method**, predating the
M1/M8 and eager/compiled repairs and subsequent performance backports. Its tok/s
figures do not describe the current complete backend.

| Target path | Median M8 head time | Top-1 match | Complete reference top-20 retained | Measured tok/s | Estimated tok/s |
|---|---:|---:|---:|---:|---:|
| Full BF16 fallback | 4.122 ms | 119,988/119,988 (100.0000%) | 119,988/119,988 (100.0000%) | 63.9 | 63.9 |
| Original block-8/64 + rerank-80 | 1.085 ms | 119,956/119,988 (99.9733%) | 98,452/119,988 (82.0515%) | 67.2 | 67.2 |
| Global INT2 top-128 + BF16 rerank | 1.114 ms | 119,986/119,988 (99.9983%) | 118,254/119,988 (98.5549%) | 67.5 | 67.2 |
| Global INT2 top-256 + BF16 rerank (default) | 1.128 ms | 119,986/119,988 (99.9983%) | 119,786/119,988 (99.8316%) | 66.8 | 67.1 |

There were 115 natural completions per method on 11 private Pi request boundaries
with 57,008–65,527 input tokens. Output totals were 60,598 / 60,075 / 60,348 /
60,675 tokens for full / block / global-128 / global-256 respectively. Tools were
not executed. Head timings are median eight-row GPU-event measurements on the
same captured hidden vectors; 119,988 prediction rows were compared.

Global-256 removes the eight-per-tile capacity limit, but remains approximate.
The two changed final argmax IDs and 202 incomplete top-20 sets are observed
misses. Complete top-20 retention does not certify score equality, ordering or
sampling probabilities. The full BF16 path is the reference in this head study,
not an independent proof of the model. [Methodology](docs/VERIFY_HEAD_GLOBAL_TOPK.md)
· [Aggregate evidence](benchmarks/results/20260916-verify-head-global-topk-long/summary.json).

## Quick start

The initial serving profile targets **one R9700, Linux x86-64, Qwen3.8-27B-
Uncensored-MXFP4-awq and Qwen3.8-27B-DFlash2-FP8**. Obtain the target and matching
drafter separately. Model weights and private benchmark inputs are not included.

The GPU host needs ROCm device access, Python 3.12+, rootless Podman,
at least 18 GiB free `/dev/shm`, and space for compiler caches and snapshots. The
profile uses 10 GB of GPU KV memory, an 18 GiB CPU offload arena, and additional
per-chat handover/tail RAM; allow ample system RAM.

```sh
git clone https://github.com/Terrydaktal/vllm-coherence.git
cd vllm-coherence
tools/coherence doctor
tools/coherence prepare
tools/coherence serve --model /path/to/target --draft /path/to/drafter
```

`prepare` verifies the versioned source/kernel bundle without opening the GPU.
`serve` starts the digest-pinned image and installs the verified additions before
graph capture. Initial compilation can take several minutes. The API binds to
`127.0.0.1:8080`.

```sh
curl http://127.0.0.1:8080/v1/models
```

Use `--dry-run` to inspect the complete command, or `--head full-bf16` for the
complete target vocabulary head. The default
`global256` profile uses faster, approximate candidate selection.

Keep the pinned compiler/image stack together. Rebuilding numerical code needs
fresh qualification; replacing hashes does not transfer evidence.
[BUILDING.md](docs/BUILDING.md) describes the source-to-bundle pipeline.

## Pi

Install uv, Node.js/npm, Git and jq on the client. From the workspace whose history
you want to use:

```sh
/path/to/vllm-coherence/tools/coherence pi -- --thinking xhigh
/path/to/vllm-coherence/tools/coherence pi -- --session last
```

For a remote GPU host:

```sh
/path/to/vllm-coherence/tools/coherence pi --ssh gpu-host -- --session last
```

The launcher installs patched Pi 0.84.2 into private Coherence state, verifies its
patches on reuse, chooses an available SSH forwarding port, and loads the operating
prompt and extensions. Sessions live in the current workspace's `.pi/sessions`.
Pi does not load workspace `AGENTS.md` as context. Supply a bespoke search tool
with `--search-extension /path/to/index.ts`, including a VM-specific extension.

`/compact` commits only a validated checkpoint. `/priority` shows/changes answer
ownership. Cache/temperature monitoring shares probes across windows.
See [Pi setup and recovery](docs/PI.md).

## Cache inspection

On the GPU host:

```sh
tools/coherence cache -- status --details
tools/coherence cache -- audit
tools/coherence cache -- watch --interval 5
```

The inventory distinguishes disk usage, cumulative disk traffic, published token
coverage, handover RAM and buffered tail RAM. It flags incomplete/duplicate
snapshots and failed cleanup. Dirty tails normally flush after about 8,192 new
tokens, and on explicit flush, eviction, successful compaction and clean shutdown.
Allow the server's clean shutdown timeout to finish. A crash may require prefilling
the unflushed tail.

## Verification and development

```sh
uv sync --frozen --extra cpu-tests
uv run --frozen --extra cpu-tests pytest -q
uv run --frozen coherence-conformance --help
python3 tools/check_publication.py
```

CPU tests use CPU-only Torch. Native qualification requires an explicit isolated
GPU run with the lease. Original Pi fixtures stay private; use the synthetic tests
or your own owner-only fixture. See [coverage and reference scope](docs/VERIFICATION.md).

## Repository map

```text
tools/                         Portable launcher, Pi connection, packaging and audits
releases/                      Versioned archive/image identities and inventories
src/qwen_r9700_lab/             Conformance, references, logical state and diagnostics
experiments/radiance-public/    Runtime adapters, HIP kernels and build/probe/replay drivers
  upstream-correctness/        Attributed upstream backports and component licenses
  rocr-poll-backoff/            CPU idle-wait repair and build recipe
integrations/pi/               Client lock, operating prompt and extensions
scripts/                       Pi installer/patchers and shared cache/telemetry helpers
configs/profiles/              Finite-precision numerical contracts
tests/                         Synthetic regressions and negative controls
reports/                       Public numerical report and aggregate evidence
docs/                          Setup, architecture, verification and performance records
benchmarks/                    Original Radiance fixtures and upstream historical results
Dockerfile, patch_*, radiance_* Original base/build and focused upstream-facing changes
```

The `qwen_r9700_lab` namespace and wire-format names remain for evidence/snapshot
compatibility. Root Radiance Dockerfiles support upstream reproduction;
**`tools/coherence` launches the assembled Coherence profile**. Research drivers
require explicit inputs and do not run automatically during serving.

## Contributing

Submit a minimized failure or measured optimization with state/output comparisons,
negative controls and precise hardware/profile scope. Read
[CONTRIBUTING.md](CONTRIBUTING.md). Existing upstream PRs remain independent and are
linked from [ATTRIBUTION.md](ATTRIBUTION.md). See [LICENSE](LICENSE) for component terms.
