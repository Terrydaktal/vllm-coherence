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

- **Target conformance diagnostics:** forced-token replay and isolated stage
  checks expose M1/M8 and eager/compiled differences. This revision retains the
  original arithmetic; alignment is introduced in the following commit.
- **Measured baseline:** full compiled stage timings and isolated correctness
  checks identify the numerical repairs introduced by the following commit.
- **Target-head diagnostics:** capture candidate recall and numerical differences.
  The global-256 replacement is introduced in the next feature commit.
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

Original eager M1 versus original compiled M8, using the **full BF16 target head**: 10,000 forced decode tokens across 23 Pi continuations, with 23 separately checked prefill predictions.

| Prediction | Same token set | Same ordering | Mean shared tokens |
| --- | ---: | ---: | ---: |
| Top 1 | 9,838 / 10,000 (98.38%) | 9,838 / 10,000 (98.38%) | 0.9838 / 1 |
| Top 10 | 4,878 / 10,000 (48.78%) | 761 / 10,000 (7.61%) | 9.3516 / 10 |
| Top 20 | 2,313 / 10,000 (23.13%) | 4 / 10,000 (0.04%) | 18.6883 / 20 |

The mismatches above are unresolved in this revision. These are finite consistency checks, not model task accuracy, an arbitrary-input proof, or certification of approximate global-256 selection.

## Compiled backend stages

Original **compiled, piecewise-graph M8**, full BF16 head, same 60K-input Pi fixture as the report. Values are the report's existing measurements, summed across all layer instances per round. Six matching complete-inventory rounds are retained. This is not eager timing, and these GPU durations must not be added to host/queue timers.

**Set/order** means the same top-20 token set, followed by the same ranking. For example, `320/320; 320/320` means both checks passed at every tested token. Operator-byte checks and whole-model checks are labelled separately; each result retains its stated test scope.

| Stage | Current GPU ms per round | Current correctness evidence | Implementation / measurement boundary |
| --- | ---: | --- | --- |
| Embedding + first input normalization | 0.004 | 320/320; 320/320 | Original pinned implementation. |
| Layer input residual/normalization | 0.155 | 320/320; 320/320 | Original pinned implementation. |
| GDN input activation FP8 quantization | 0.127 | 320/320; 320/320 | Original pinned implementation. |
| GDN input projection | 3.875 | 320/320; 320/320 | Original pinned implementation. |
| GDN layout/copies and buffer initialization | 0.284 | State/layout checked with convolution and recurrence | Original pinned implementation. |
| GDN convolution | 0.494 | 9/320; 0/320 | Original pinned implementation. |
| GDN recurrence and gates | 1.162 | 13/320; 0/320 | Original pinned implementation. |
| GDN output gated normalization | 0.138 | 320/320; 320/320 | Original pinned implementation. |
| GDN output activation FP8 quantization | 0.128 | 320/320; 320/320 | Original pinned implementation. |
| GDN output projection | 1.884 | 320/320; 320/320 | Original pinned implementation. |
| Attention input activation FP8 quantization | 0.043 | 320/320; 320/320 | Original pinned implementation. |
| Attention input projection | 1.099 | 320/320; 320/320 | Original pinned implementation. |
| Attention Q/K normalization, RoPE and layout | 0.079 | 320/320; 320/320 | Original pinned implementation. |
| Attention KV write | 0.046 | 320/320; 320/320 | Original pinned implementation. |
| Attention decode | 4.224 | 22/320; 0/320 | Original pinned implementation. |
| Attention split-KV merge | 0.138 | 22/320; 0/320 | Original pinned implementation. |
| Attention output gating | 0.028 | 320/320; 320/320 | Original pinned implementation. |
| Attention output activation FP8 quantization | 0.046 | 320/320; 320/320 | Original pinned implementation. |
| Attention output projection | 0.557 | 320/320; 320/320 | Original pinned implementation. |
| Post-attention/GDN residual/normalization | 0.142 | 320/320; 320/320 | Original pinned implementation. |
| MLP gate/up input FP8 quantization | 0.165 | 320/320; 320/320 | Original pinned implementation. |
| MLP gate/up projection | 25.271 | 320/320; 320/320 | Original pinned implementation. |
| MLP SiLU and gating | 0.156 | 320/320; 320/320 | Original pinned implementation. |
| MLP down input FP8 quantization | 0.297 | 320/320; 320/320 | Original pinned implementation. |
| MLP down projection | 5.147 | 320/320; 320/320 | Original pinned implementation. |
| Final normalization/layout | 0.002 | 320/320; 320/320 | Original pinned implementation. |
| Full BF16 target head | 4.024 | 320/320; 319/320 | Original pinned implementation. |
| Drafter | 6.328 | N/A: no isolated target top-20 prediction | Original pinned implementation. |
| Other GPU bookkeeping | 0.551 | N/A: no isolated target top-20 prediction | Original pinned implementation. |

**Sum of measured GPU dispatch durations: 56.596 ms per profiled round.** This sum excludes host gaps and queue time and is not the uninstrumented round timer.

<details>
<summary>Every decoder layer: current projection and remaining-work timings</summary>

Each layer has four projections. Gate and up are one joint GEMM; there is no separately measured gate/up split.

| Layer | Type | All layer work ms | Input projection ms | Output projection ms | Gate/up projection ms | Down projection ms | Other work ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | GDN | 0.6255 | 0.0756 | 0.0388 | 0.3712 | 0.0802 | 0.0597 |
| 1 | GDN | 0.6274 | 0.0797 | 0.0387 | 0.3712 | 0.0802 | 0.0577 |
| 2 | GDN | 0.6321 | 0.0798 | 0.0385 | 0.3716 | 0.0806 | 0.0616 |
| 3 | Attention | 0.8453 | 0.0683 | 0.0352 | 0.3747 | 0.0797 | 0.2876 |
| 4 | GDN | 0.6372 | 0.0798 | 0.0390 | 0.3797 | 0.0805 | 0.0582 |
| 5 | GDN | 0.6423 | 0.0805 | 0.0392 | 0.3828 | 0.0799 | 0.0599 |
| 6 | GDN | 0.6501 | 0.0811 | 0.0389 | 0.3868 | 0.0803 | 0.0630 |
| 7 | Attention | 0.8685 | 0.0690 | 0.0347 | 0.3884 | 0.0796 | 0.2968 |
| 8 | GDN | 0.6468 | 0.0797 | 0.0392 | 0.3876 | 0.0802 | 0.0600 |
| 9 | GDN | 0.6481 | 0.0805 | 0.0395 | 0.3875 | 0.0802 | 0.0604 |
| 10 | GDN | 0.6543 | 0.0807 | 0.0391 | 0.3902 | 0.0802 | 0.0641 |
| 11 | Attention | 0.8717 | 0.0684 | 0.0343 | 0.3912 | 0.0799 | 0.2979 |
| 12 | GDN | 0.6526 | 0.0805 | 0.0394 | 0.3912 | 0.0799 | 0.0615 |
| 13 | GDN | 0.6521 | 0.0811 | 0.0393 | 0.3904 | 0.0800 | 0.0612 |
| 14 | GDN | 0.6536 | 0.0818 | 0.0391 | 0.3881 | 0.0803 | 0.0643 |
| 15 | Attention | 0.8739 | 0.0687 | 0.0354 | 0.3909 | 0.0798 | 0.2991 |
| 16 | GDN | 0.6519 | 0.0806 | 0.0388 | 0.3909 | 0.0804 | 0.0612 |
| 17 | GDN | 0.6531 | 0.0808 | 0.0394 | 0.3911 | 0.0807 | 0.0610 |
| 18 | GDN | 0.6587 | 0.0807 | 0.0401 | 0.3910 | 0.0808 | 0.0662 |
| 19 | Attention | 0.8778 | 0.0684 | 0.0345 | 0.3926 | 0.0802 | 0.3020 |
| 20 | GDN | 0.6580 | 0.0809 | 0.0394 | 0.3955 | 0.0803 | 0.0620 |
| 21 | GDN | 0.6583 | 0.0806 | 0.0390 | 0.3971 | 0.0803 | 0.0614 |
| 22 | GDN | 0.6630 | 0.0804 | 0.0386 | 0.3977 | 0.0808 | 0.0654 |
| 23 | Attention | 0.8782 | 0.0682 | 0.0348 | 0.3938 | 0.0800 | 0.3014 |
| 24 | GDN | 0.6550 | 0.0806 | 0.0391 | 0.3926 | 0.0809 | 0.0619 |
| 25 | GDN | 0.6546 | 0.0803 | 0.0396 | 0.3926 | 0.0806 | 0.0615 |
| 26 | GDN | 0.6622 | 0.0817 | 0.0393 | 0.3948 | 0.0805 | 0.0659 |
| 27 | Attention | 0.8835 | 0.0694 | 0.0352 | 0.3948 | 0.0801 | 0.3040 |
| 28 | GDN | 0.6579 | 0.0815 | 0.0397 | 0.3946 | 0.0803 | 0.0617 |
| 29 | GDN | 0.6606 | 0.0813 | 0.0390 | 0.3970 | 0.0805 | 0.0628 |
| 30 | GDN | 0.6652 | 0.0809 | 0.0392 | 0.3983 | 0.0808 | 0.0660 |
| 31 | Attention | 0.8834 | 0.0693 | 0.0348 | 0.3966 | 0.0803 | 0.3025 |
| 32 | GDN | 0.6588 | 0.0812 | 0.0393 | 0.3970 | 0.0805 | 0.0608 |
| 33 | GDN | 0.6600 | 0.0811 | 0.0396 | 0.3965 | 0.0806 | 0.0622 |
| 34 | GDN | 0.6646 | 0.0806 | 0.0392 | 0.3981 | 0.0811 | 0.0656 |
| 35 | Attention | 0.8900 | 0.0688 | 0.0346 | 0.4003 | 0.0801 | 0.3062 |
| 36 | GDN | 0.6646 | 0.0814 | 0.0398 | 0.4001 | 0.0804 | 0.0628 |
| 37 | GDN | 0.6616 | 0.0806 | 0.0396 | 0.3983 | 0.0808 | 0.0622 |
| 38 | GDN | 0.6671 | 0.0811 | 0.0390 | 0.4007 | 0.0809 | 0.0654 |
| 39 | Attention | 0.8897 | 0.0685 | 0.0345 | 0.4000 | 0.0801 | 0.3066 |
| 40 | GDN | 0.6629 | 0.0809 | 0.0393 | 0.3994 | 0.0806 | 0.0627 |
| 41 | GDN | 0.6629 | 0.0812 | 0.0392 | 0.3993 | 0.0804 | 0.0627 |
| 42 | GDN | 0.6651 | 0.0804 | 0.0398 | 0.3992 | 0.0803 | 0.0655 |
| 43 | Attention | 0.8903 | 0.0684 | 0.0351 | 0.4013 | 0.0801 | 0.3053 |
| 44 | GDN | 0.6671 | 0.0820 | 0.0394 | 0.4023 | 0.0802 | 0.0632 |
| 45 | GDN | 0.6669 | 0.0808 | 0.0392 | 0.4030 | 0.0807 | 0.0632 |
| 46 | GDN | 0.6671 | 0.0802 | 0.0389 | 0.4009 | 0.0814 | 0.0658 |
| 47 | Attention | 0.8922 | 0.0685 | 0.0347 | 0.4026 | 0.0802 | 0.3062 |
| 48 | GDN | 0.6687 | 0.0814 | 0.0396 | 0.4040 | 0.0809 | 0.0628 |
| 49 | GDN | 0.6713 | 0.0812 | 0.0393 | 0.4064 | 0.0811 | 0.0633 |
| 50 | GDN | 0.6709 | 0.0805 | 0.0395 | 0.4037 | 0.0809 | 0.0663 |
| 51 | Attention | 0.8912 | 0.0685 | 0.0349 | 0.4003 | 0.0809 | 0.3065 |
| 52 | GDN | 0.6629 | 0.0802 | 0.0398 | 0.3994 | 0.0805 | 0.0630 |
| 53 | GDN | 0.6618 | 0.0802 | 0.0394 | 0.3997 | 0.0808 | 0.0617 |
| 54 | GDN | 0.6692 | 0.0822 | 0.0386 | 0.4012 | 0.0810 | 0.0663 |
| 55 | Attention | 0.8885 | 0.0691 | 0.0345 | 0.3995 | 0.0802 | 0.3051 |
| 56 | GDN | 0.6586 | 0.0803 | 0.0387 | 0.3974 | 0.0803 | 0.0619 |
| 57 | GDN | 0.6636 | 0.0808 | 0.0395 | 0.3996 | 0.0809 | 0.0628 |
| 58 | GDN | 0.6692 | 0.0814 | 0.0397 | 0.4010 | 0.0807 | 0.0664 |
| 59 | Attention | 0.8904 | 0.0685 | 0.0347 | 0.4009 | 0.0804 | 0.3059 |
| 60 | GDN | 0.6652 | 0.0817 | 0.0392 | 0.4008 | 0.0806 | 0.0630 |
| 61 | GDN | 0.6649 | 0.0813 | 0.0395 | 0.4011 | 0.0802 | 0.0627 |
| 62 | GDN | 0.6688 | 0.0809 | 0.0392 | 0.4018 | 0.0808 | 0.0661 |
| 63 | Attention | 0.8917 | 0.0691 | 0.0350 | 0.4010 | 0.0799 | 0.3067 |

</details>

<details>
<summary>Every recorded kernel, grouped by stage</summary>

| Stage / compiled kernel | Calls in retained rounds | Current GPU ms per round |
| --- | ---: | ---: |
| Drafter / `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x32_MI16x16x1_SN_LDSB1_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_CD1_1_CLR0_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB1_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA1201_ICIW0_IU1_K1_LDSTI0_LBSPPA128_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB0_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB2_ONLL1_PAP0_PGL0_PGR1_PLR1_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1.kd` | 6 | 0.007179 |
| Drafter / `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x32x256_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB1_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA1201_ICIW0_IU1_K1_LDSTI0_LBSPPA512_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB0_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB16_ONLL0_PAP0_PGL0_PGR1_PLR1_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_4_1.kd` | 6 | 0.177347 |
| Drafter / `__amd_rocclr_copyBuffer.kd` | 6 | 0.001919 |
| Drafter / `__amd_rocclr_fillBufferAligned.kd` | 18 | 0.007077 |
| Drafter / `_cache_draft_logits_kernel.kd` | 6 | 0.002359 |
| Drafter / `_draft_head_int2.kd` | 6 | 0.824683 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_17408_EVEN_K_1_GRID_MN_40_cache_modifier_NONE.kd` | 30 | 0.730700 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_25600_EVEN_K_1_GRID_MN_40_cache_modifier_NONE.kd` | 6 | 0.218080 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_40_cache_modifier_NONE.kd` | 30 | 0.202844 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_5120_EVEN_K_1_GRID_MN_272_cache_modifier_NONE.kd` | 30 | 1.446181 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_5120_EVEN_K_1_GRID_MN_48_cache_modifier_NONE.kd` | 30 | 0.273222 |
| Drafter / `_prepare_dflash_inputs_kernel.kd` | 6 | 0.007819 |
| Drafter / `_rerank_exact.kd` | 6 | 0.008606 |
| Drafter / `_selector_walk_kernel.kd` | 6 | 0.007866 |
| Drafter / `kernel_unified_attention.kd` | 30 | 1.852924 |
| Drafter / `reshape_and_cache_kernel_flash.kd` | 60 | 0.026984 |
| Drafter / `triton_per_fused_4.kd` | 6 | 0.001926 |
| Drafter / `triton_per_fused_9.kd` | 24 | 0.007490 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_preshuffle_gemm_squeeze_view_0.kd` | 30 | 0.009582 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_preshuffle_gemm_squeeze_view_2.kd` | 6 | 0.002272 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_preshuffle_gemm_squeeze_view_4.kd` | 54 | 0.014478 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_view_0.kd` | 6 | 0.004099 |
| Drafter / `triton_poi_fused_0.kd` | 6 | 0.002686 |
| Drafter / `triton_poi_fused_10.kd` | 24 | 0.007503 |
| Drafter / `triton_poi_fused_5.kd` | 6 | 0.002232 |
| Drafter / `triton_poi_fused__to_copy_clamp_div_mul_preshuffle_gemm_silu_slice_squeeze_view_7.kd` | 30 | 0.015428 |
| Drafter / `triton_poi_fused__to_copy_clamp_div_preshuffle_gemm_squeeze_view_1.kd` | 30 | 0.011795 |
| Drafter / `triton_poi_fused__to_copy_clamp_div_preshuffle_gemm_squeeze_view_3.kd` | 6 | 0.002059 |
| Drafter / `triton_poi_fused__to_copy_clamp_div_preshuffle_gemm_squeeze_view_5.kd` | 54 | 0.016371 |
| Drafter / `triton_poi_fused_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_3.kd` | 54 | 0.019358 |
| Drafter / `triton_poi_fused_add_arange_bitwise_and_constant_pad_nd_ge_mul_rms_norm_select_slice_unsqueeze_view_1.kd` | 6 | 0.002306 |
| Drafter / `triton_poi_fused_add_permute_unsqueeze_view_2.kd` | 6 | 0.001846 |
| Drafter / `triton_poi_fused_cat_expand_index_mul_slice_unsqueeze_view_1.kd` | 6 | 0.002546 |
| Drafter / `triton_red_fused__to_copy_abs_clamp_div_max_mul_preshuffle_gemm_silu_slice_squeeze_view_6.kd` | 30 | 0.013422 |
| Drafter / `triton_red_fused__to_copy_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_w4_gemm_2.kd` | 30 | 0.030876 |
| Drafter / `triton_red_fused__to_copy_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_w4_gemm_8.kd` | 24 | 0.025756 |
| Drafter / `triton_red_fused__to_copy_embedding_mul_rms_norm_w4_gemm_0.kd` | 6 | 0.003846 |
| Drafter / `triton_red_fused_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_8.kd` | 6 | 0.006719 |
| Drafter / `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) [clone .kd]` | 12 | 0.009285 |
| Drafter / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 6 | 0.003526 |
| Drafter / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 6 | 0.002879 |
| Drafter / `void at::native::bitonicSortKVInPlace<2, -1, 16, 16, c10::BFloat16, long, at::native::GTOp<c10::BFloat16, true>, unsigned int>(at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, at::native::GTOp<c10::BFloat16, true>) [clone .kd]` | 6 | 0.003232 |
| Drafter / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 6 | 0.005012 |
| Drafter / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 6 | 0.002586 |
| Drafter / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 6 | 0.003506 |
| Drafter / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 6 | 0.003146 |
| Drafter / `void at::native::mbtopk::computeBlockDigitCounts<c10::BFloat16, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 12 | 0.038345 |
| Drafter / `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 24 | 0.055463 |
| Drafter / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, c10::BFloat16>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 12 | 0.022351 |
| Drafter / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, float>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, float*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 24 | 0.028736 |
| Drafter / `void at::native::mbtopk::fill<unsigned int, unsigned int>(unsigned int*, unsigned int, unsigned int) [clone .kd]` | 12 | 0.003018 |
| Drafter / `void at::native::mbtopk::gatherTopK<c10::BFloat16, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 6 | 0.021039 |
| Drafter / `void at::native::mbtopk::gatherTopK<float, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, float*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 6 | 0.008879 |
| Drafter / `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4>) [clone .kd]` | 6 | 0.003759 |
| Drafter / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul>) [clone .kd]` | 6 | 0.001732 |
| Drafter / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) [clone .kd]` | 6 | 0.002479 |
| Drafter / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>) [clone .kd]` | 12 | 0.003785 |
| Drafter / `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>) [clone .kd]` | 12 | 0.005351 |
| Drafter / `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool) [clone .kd]` | 6 | 0.002099 |
| Drafter / `void at::native::warpMergeSortKVInPlace<2, -1, 128, 16, float, long, at::native::GTOp<float, true>, unsigned int, 32>(at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, at::native::GTOp<float, true>, float) [clone .kd]` | 6 | 0.004952 |
| Drafter / `void r4d_gemm_w4a16_nt_m64_kernel<1, 1, false>(unsigned short const*, unsigned int const*, unsigned int const*, __hip_bfloat16*, int, int, int, int, int) [clone .kd]` | 6 | 0.004506 |
| Drafter / `void r4d_gemm_w4a16_nt_m64_kernel<1, 1, true>(unsigned short const*, unsigned int const*, unsigned int const*, __hip_bfloat16*, int, int, int, int, int) [clone .kd]` | 60 | 0.080050 |
| Drafter / `void vllm::rms_norm_kernel<c10::BFloat16, 8, 2, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, long, long, long, c10::BFloat16 const*, long, float, int, int) [clone .kd]` | 6 | 0.003059 |
| Drafter / `void vllm::rms_norm_kernel<c10::BFloat16, 8, 4, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, long, long, long, c10::BFloat16 const*, long, float, int, int) [clone .kd]` | 6 | 0.002752 |
| Drafter / `void vllm::rotary_embedding_kernel<c10::BFloat16, c10::BFloat16, true>(long const*, c10::BFloat16*, c10::BFloat16*, c10::BFloat16 const*, int, long, long, long, int, int, int, long, bool) [clone .kd]` | 6 | 0.002499 |
| Attention KV write / `reshape_and_cache_kernel_flash.kd` | 96 | 0.046324 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_6.kd` | 96 | 0.022437 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_8.kd` | 96 | 0.030144 |
| Attention Q/K normalization, RoPE and layout / `triton_red_fused_7.kd` | 96 | 0.026091 |
| Attention decode / `void r4d_attn_decode_kernel<3, 16, 256, 6, 16, 0, 3430971>(R4DArgs, int) [clone .kd]` | 96 | 4.224365 |
| Attention input activation FP8 quantization / `void vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<c10::BFloat16, c10::Float8_e4m3fn>(c10::Float8_e4m3fn*, float*, c10::BFloat16 const*, float const*, int, long, long) [clone .kd]` | 96 | 0.042844 |
| Attention input projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 1, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 96 | 1.099262 |
| Attention output activation FP8 quantization / `void vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<c10::BFloat16, c10::Float8_e4m3fn>(c10::Float8_e4m3fn*, float*, c10::BFloat16 const*, float const*, int, long, long) [clone .kd]` | 96 | 0.045851 |
| Attention output gating / `triton_poi_fused_mul_mxfp4_linear_sigmoid_view_0.kd` | 96 | 0.028144 |
| Attention output projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 4, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 96 | 0.556920 |
| Attention split-KV merge / `void r4d_attn_splitkv_combine_kernel<256, 4, 1>(R4DArgs, int, int) [clone .kd]` | 96 | 0.137671 |
| Embedding + first input normalization / `triton_red_fused__to_copy_add_embedding_mxfp4_linear_rms_norm_0.kd` | 6 | 0.004112 |
| Final normalization/layout / `triton_red_fused__to_copy_add_fused_add_rms_norm_3.kd` | 6 | 0.002392 |
| GDN convolution / `void r4d_gdn_conv_update_kernel<1>(unsigned short const*, long, unsigned short const*, unsigned short const*, unsigned short*, long, long, long, int, int const*, long, int const*, unsigned short*, unsigned short*, unsigned short*, int const*, int, int, int) [clone .kd]` | 288 | 0.494015 |
| GDN input activation FP8 quantization / `void vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<c10::BFloat16, c10::Float8_e4m3fn>(c10::Float8_e4m3fn*, float*, c10::BFloat16 const*, float const*, int, long, long) [clone .kd]` | 288 | 0.127500 |
| GDN input projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 1, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 288 | 3.875011 |
| GDN layout/copies and buffer initialization / `triton_per_fused_1.kd` | 96 | 0.022318 |
| GDN layout/copies and buffer initialization / `triton_poi_fused_0.kd` | 96 | 0.031257 |
| GDN layout/copies and buffer initialization / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 288 | 0.156100 |
| GDN layout/copies and buffer initialization / `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>) [clone .kd]` | 288 | 0.074632 |
| GDN output activation FP8 quantization / `void vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<c10::BFloat16, c10::Float8_e4m3fn>(c10::Float8_e4m3fn*, float*, c10::BFloat16 const*, float const*, int, long, long) [clone .kd]` | 288 | 0.128079 |
| GDN output gated normalization / `triton_per_fused__to_copy_mean_pow_view_0.kd` | 192 | 0.050375 |
| GDN output gated normalization / `triton_poi_fused__to_copy_add_mean_mul_mxfp4_linear_pow_rsqrt_silu_view_1.kd` | 192 | 0.058742 |
| GDN output gated normalization / `triton_poi_fused__to_copy_add_mean_mul_mxfp4_linear_pow_rsqrt_silu_view_2.kd` | 96 | 0.028797 |
| GDN output projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 4, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 288 | 1.884129 |
| GDN recurrence and gates / `void r4d_gdn_recurrent_update_kernel<1, 0, 2>(unsigned short const*, unsigned short const*, unsigned short const*, void const*, void const*, long, float const*, float const*, float*, long, long, unsigned short*, int const*, int const*, long, int const*, unsigned short const*, float const*, float, int, int, int, float, float) [clone .kd]` | 288 | 1.161877 |
| Layer input residual/normalization / `triton_red_fused__to_copy_add_fused_add_rms_norm_mxfp4_linear_3.kd` | 90 | 0.037245 |
| Layer input residual/normalization / `triton_red_fused__to_copy_add_fused_add_rms_norm_mxfp4_linear_4.kd` | 192 | 0.078508 |
| Layer input residual/normalization / `triton_red_fused__to_copy_add_fused_add_rms_norm_mxfp4_linear_5.kd` | 96 | 0.039184 |
| MLP SiLU and gating / `triton_poi_fused_mul_mxfp4_linear_silu_slice_2.kd` | 96 | 0.044918 |
| MLP SiLU and gating / `triton_poi_fused_mul_mxfp4_linear_silu_slice_3.kd` | 192 | 0.066262 |
| MLP SiLU and gating / `triton_poi_fused_mul_mxfp4_linear_silu_slice_4.kd` | 96 | 0.045138 |
| MLP down input FP8 quantization / `void vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<c10::BFloat16, c10::Float8_e4m3fn>(c10::Float8_e4m3fn*, float*, c10::BFloat16 const*, float const*, int, long, long) [clone .kd]` | 384 | 0.296558 |
| MLP down projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 4, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 384 | 5.147348 |
| MLP gate/up input FP8 quantization / `void vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<c10::BFloat16, c10::Float8_e4m3fn>(c10::Float8_e4m3fn*, float*, c10::BFloat16 const*, float const*, int, long, long) [clone .kd]` | 384 | 0.164856 |
| MLP gate/up projection / `void radiance_mxfp4_fp8_gemm_folded<2, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, std::bfloat16_t*, int, int, int) [clone .kd]` | 384 | 25.271205 |
| Post-attention/GDN residual/normalization / `triton_red_fused__to_copy_add_fused_add_rms_norm_mxfp4_linear_1.kd` | 96 | 0.035164 |
| Post-attention/GDN residual/normalization / `triton_red_fused__to_copy_add_fused_add_rms_norm_mxfp4_linear_2.kd` | 192 | 0.064875 |
| Post-attention/GDN residual/normalization / `triton_red_fused__to_copy_add_fused_add_rms_norm_mxfp4_linear_3.kd` | 96 | 0.042258 |
| Full BF16 target head / `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x32x256_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB1_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA1201_ICIW0_IU1_K1_LDSTI0_LBSPPA512_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB0_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB16_ONLL0_PAP0_PGL0_PGR1_PLR1_PKA0_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKFDPO0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_4_1.kd` | 6 | 4.024325 |
| Other GPU bookkeeping / `__amd_rocclr_copyBuffer.kd` | 172 | 0.075278 |
| Other GPU bookkeeping / `__amd_rocclr_fillBufferAligned.kd` | 6 | 0.002699 |
| Other GPU bookkeeping / `_combine_sampled_and_draft_tokens_kernel.kd` | 7 | 0.003432 |
| Other GPU bookkeeping / `_compute_local_logits_stats_kernel.kd` | 6 | 0.028659 |
| Other GPU bookkeeping / `_compute_slot_mappings_kernel.kd` | 7 | 0.003386 |
| Other GPU bookkeeping / `_expand_idx_mapping_kernel.kd` | 7 | 0.002185 |
| Other GPU bookkeeping / `_gather_block_tables_kernel.kd` | 7 | 0.005632 |
| Other GPU bookkeeping / `_get_num_sampled_and_rejected_kernel.kd` | 6 | 0.002632 |
| Other GPU bookkeeping / `_insert_resampled_kernel.kd` | 6 | 0.003466 |
| Other GPU bookkeeping / `_post_update_kernel.kd` | 6 | 0.005292 |
| Other GPU bookkeeping / `_prepare_pos_seq_lens_kernel.kd` | 7 | 0.002499 |
| Other GPU bookkeeping / `_prepare_rope_positions_kernel.kd` | 7 | 0.003119 |
| Other GPU bookkeeping / `_rejection_kernel.kd` | 6 | 0.007559 |
| Other GPU bookkeeping / `_resample_kernel.kd` | 6 | 0.014159 |
| Other GPU bookkeeping / `_scatter_num_accepted_kernel.kd` | 6 | 0.001939 |
| Other GPU bookkeeping / `postprocess_mamba_fused_kernel.kd` | 6 | 0.002552 |
| Other GPU bookkeeping / `precopy_mamba_align_fused_kernel.kd` | 7 | 0.003079 |
| Other GPU bookkeeping / `preprocess_mamba_align_fused_kernel.kd` | 7 | 0.003119 |
| Other GPU bookkeeping / `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}>(int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}, function_traits<at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}>::result_type*) [clone .kd]` | 42 | 0.010240 |
| Other GPU bookkeeping / `void (anonymous namespace)::softmax_warp_forward<float, float, float, 6, false, false, 32>(float*, float const*, int, int, int, bool const*, int, bool) [clone .kd]` | 6 | 0.002226 |
| Other GPU bookkeeping / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 48 | 0.015452 |
| Other GPU bookkeeping / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 6 | 0.002579 |
| Other GPU bookkeeping / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 48 | 0.020612 |
| Other GPU bookkeeping / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<int> const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<int> const&)::{lambda(int, bool)#1}) [clone .kd]` | 42 | 0.021400 |
| Other GPU bookkeeping / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<float> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<float> const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<float> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<float> const&)::{lambda(int, bool)#1}) [clone .kd]` | 18 | 0.018944 |
| Other GPU bookkeeping / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 109 | 0.053089 |
| Other GPU bookkeeping / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 12 | 0.005945 |
| Other GPU bookkeeping / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 6 | 0.003059 |
| Other GPU bookkeeping / `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 24 | 0.050263 |
| Other GPU bookkeeping / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, float>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, float*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 24 | 0.037523 |
| Other GPU bookkeeping / `void at::native::mbtopk::fill<unsigned int, unsigned int>(unsigned int*, unsigned int, unsigned int) [clone .kd]` | 6 | 0.001612 |
| Other GPU bookkeeping / `void at::native::mbtopk::gatherTopK<float, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, float*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 6 | 0.026179 |
| Other GPU bookkeeping / `void at::native::tensor_kernel_scan_innermost_dim<float, std::plus<float> >(float*, float const*, unsigned int, unsigned int, unsigned int, float, std::plus<float>) [clone .kd]` | 6 | 0.002646 |
| Other GPU bookkeeping / `void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>, 4, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast) [clone .kd]` | 42 | 0.012020 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<16, at::native::BinaryFunctor<bool, bool, bool, at::native::BitwiseOrFunctor<bool> >, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<bool, bool, bool, at::native::BitwiseOrFunctor<bool> >, std::array<char*, 3ul>) [clone .kd]` | 6 | 0.002759 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<16, at::native::FillFunctor<bool>, std::array<char*, 1ul> >(int, at::native::FillFunctor<bool>, std::array<char*, 1ul>) [clone .kd]` | 7 | 0.002119 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<16, at::native::bitwise_not_kernel_cuda(at::TensorIteratorBase&)::{lambda(bool)#1}, std::array<char*, 2ul> >(int, at::native::bitwise_not_kernel_cuda(at::TensorIteratorBase&)::{lambda(bool)#1}, std::array<char*, 2ul>) [clone .kd]` | 6 | 0.002286 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}, std::array<char*, 2ul>) [clone .kd]` | 42 | 0.012046 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}, std::array<char*, 2ul>) [clone .kd]` | 6 | 0.001772 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}, std::array<char*, 3ul> >(int, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}, std::array<char*, 3ul>) [clone .kd]` | 6 | 0.010073 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool, float, float)#1}, std::array<char*, 4ul> >(int, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool, float, float)#1}, std::array<char*, 4ul>) [clone .kd]` | 12 | 0.004605 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 2ul> >(int, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 2ul>) [clone .kd]` | 42 | 0.019407 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, 2ul>) [clone .kd]` | 48 | 0.013725 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul>) [clone .kd]` | 6 | 0.001966 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>) [clone .kd]` | 6 | 0.001939 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>) [clone .kd]` | 12 | 0.003618 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int, at::native::FillFunctor<int>, std::array<char*, 1ul>) [clone .kd]` | 7 | 0.001932 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>) [clone .kd]` | 6 | 0.010886 |
| Other GPU bookkeeping / `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool) [clone .kd]` | 6 | 0.002086 |
| Other GPU bookkeeping / `void at::native::warpMergeSortKVInPlace<2, -1, 128, 16, float, long, at::native::GTOp<float, true>, unsigned int, 32>(at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, at::native::GTOp<float, true>, float) [clone .kd]` | 6 | 0.004832 |

</details>

## Uninstrumented performance

Separate uninstrumented compiled direct-engine controls with the full BF16 target head. Three natural responses on the same 60,000-input-token Pi prefix; excludes HTTP/Pi, tool execution, snapshot publication, cold prefill and warm-up.

| Measurement | Current result |
| --- | ---: |
| Median round | 59.684 ms |
| Pooled rate after first output | 85.561 tok/s |
| Output tokens / natural responses | 2,582 / 3 |
| Timed post-first output | 30.142 s |

These approximately 80–86 tok/s results are brief **60K-input** controls, not the separate 60K-generated-token head benchmark. No eager execution or instrumented timing is substituted for these complete-round measurements.

Aggregate data: [current measurements](benchmarks/results/coherence-current.json). Historical methodology and detailed numerical evidence: [technical report](reports/d7-rdna4-2026-09-17/REPORT.md).

<!-- /COHERENCE_CURRENT_RESULTS -->

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
