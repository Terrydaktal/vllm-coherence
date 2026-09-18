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
- **Performance recovery:** arithmetic-preserving packed GDN transport, shared
  attention reads and an interleaved full vocabulary head. The GGZ14 backports
  are introduced in a later commit.
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

Corrected eager M1 versus corrected compiled M8, using the **full BF16 target head**: 10,000 forced decode tokens across 23 Pi continuations, with 23 separately checked prefill predictions.

| Prediction | Same token set | Same ordering | Mean shared tokens |
| --- | ---: | ---: | ---: |
| Top 1 | 10,000 / 10,000 (100.00%) | 10,000 / 10,000 (100.00%) | 1.0000 / 1 |
| Top 10 | 10,000 / 10,000 (100.00%) | 10,000 / 10,000 (100.00%) | 10.0000 / 10 |
| Top 20 | 10,000 / 10,000 (100.00%) | 10,000 / 10,000 (100.00%) | 20.0000 / 20 |

All 10,000 full-vocabulary hashes and 23 prefill predictions matched. These are finite consistency checks, not model task accuracy, an arbitrary-input proof, or certification of approximate global-256 selection.

## Compiled backend stages

Aligned **compiled, piecewise-graph M8**, full BF16 head, same 60K-input Pi fixture as the report. Values are the report's existing measurements, summed across all layer instances per round. Six matching complete-inventory rounds are retained. This is not eager timing, and these GPU durations must not be added to host/queue timers.

**Set/order** means the same top-20 token set, followed by the same ranking. For example, `320/320; 320/320` means both checks passed at every tested token. Operator-byte checks and whole-model checks are labelled separately; each result retains its stated test scope.

| Stage | Current GPU ms per round | Current correctness evidence | Implementation / measurement boundary |
| --- | ---: | --- | --- |
| Embedding + first input normalization | 0.008 | 320/320; 320/320; eager/compiled 320/320; 320/320 | Preserve the reference normalization rounding; the original embedding/norm fusion is indivisible. |
| Layer input residual/normalization | 0.346 | 320/320; 320/320; eager/compiled 320/320; 320/320 | Preserve reduction and rounding; retain FP32 residual sums in registers. |
| GDN input activation FP8 quantization | 0.127 | 320/320; 320/320; eager/compiled 320/320; 320/320 | Same quantization kernel and call count; the small timing delta is unassigned. |
| GDN input projection | 3.873 | 320/320; 320/320; eager/compiled 320/320; 320/320 | Unchanged MXFP4 projection arithmetic; timing differences are observations, not a projection optimization. |
| GDN layout/copies and buffer initialization | 0.332 | State/layout checked with convolution and recurrence | Preserve packed QKV views; remove split materializations and repacking. Remaining copies are included. |
| GDN convolution | 0.226 | 320/320; 320/320; eager/compiled 320/320; 320/320 | Match serial product/accumulation order and rolling history; packed transport removes surrounding copies. |
| GDN recurrence and gates | 1.219 | 320/320; 320/320; eager/compiled 320/320; 320/320 | Match gate precision, reduction order, recurrent-state transition and output rounding. |
| GDN output gated normalization | 0.097 | 320/320; 320/320; eager/compiled 320/320; 320/320 | Keep the serial row tile while processing independent rows concurrently; original fused constituents remain grouped. |
| GDN output activation FP8 quantization | 0.131 | 320/320; 320/320; eager/compiled 320/320; 320/320 | Same quantization kernel and call count; the small timing delta is unassigned. |
| GDN output projection | 1.884 | 320/320; 320/320; eager/compiled 320/320; 320/320 | Unchanged MXFP4 arithmetic; no causal speedup claimed. |
| Attention input activation FP8 quantization | 0.042 | 320/320; 320/320; eager/compiled 320/320; 320/320 | Same quantization kernel and call count; timing cause is not isolated. |
| Attention input projection | 1.099 | 320/320; 320/320; eager/compiled 320/320; 320/320 | Unchanged QKV MXFP4 projection; no causal speedup claimed. |
| Attention Q/K normalization, RoPE and layout | 0.236 | 320/320; 320/320; eager/compiled 320/320; 320/320 | Preserve serial normalization and BF16 RoPE product rounding. Fused original constituents share one timing. |
| Attention KV write | 0.047 | 320/320; 320/320; eager/compiled 320/320; 320/320 | Same cache-write kernel and KV format; timing cause is not isolated. |
| Attention decode | 5.285 | 320/320; 320/320; eager/compiled 320/320; 320/320 | Preserve each query's causal tile/softmax decisions. Share KV reads within one or two tile-aligned query groups; two groups repeat context work. |
| Attention split-KV merge | 0.120 | 320/320; 320/320; eager/compiled 320/320; 320/320 | Use each query's serial split/merge arithmetic; the observed reduction has not been isolated from the decode change. |
| Attention output gating | 0.032 | 320/320; 320/320; eager/compiled 320/320; 320/320 | Preserve the BF16 sigmoid result before multiplying the output gate. |
| Attention output activation FP8 quantization | 0.046 | 320/320; 320/320; eager/compiled 320/320; 320/320 | Same quantization kernel and call count; timing cause is not isolated. |
| Attention output projection | 0.521 | 320/320; 320/320; eager/compiled 320/320; 320/320 | Unchanged output MXFP4 projection; no causal speedup claimed. |
| Post-attention/GDN residual/normalization | 0.353 | 320/320; 320/320; eager/compiled 320/320; 320/320 | Preserve serial reduction/rounding and keep residual values in registers. |
| MLP gate/up input FP8 quantization | 0.167 | 320/320; 320/320; eager/compiled 320/320; 320/320 | Same quantization kernel and call count; timing cause is not isolated. |
| MLP gate/up projection | 25.437 | 320/320; 320/320; eager/compiled 320/320; 320/320 | One joint gate/up GEMM per layer, 64 per round. The per-layer table subdivides this total; gate and up have no separate measured durations. |
| MLP SiLU and gating | 0.189 | 320/320; 320/320; eager/compiled 320/320; 320/320 | Preserve the BF16 SiLU result before multiplication; retain one fused pointwise launch. |
| MLP down input FP8 quantization | 0.297 | 320/320; 320/320; eager/compiled 320/320; 320/320 | Same quantization kernel and call count; timing cause is not isolated. |
| MLP down projection | 5.143 | 320/320; 320/320; eager/compiled 320/320; 320/320 | Unchanged MXFP4 down projection; no causal speedup claimed. |
| Final normalization/layout | 0.006 | 320/320; 320/320; eager/compiled 320/320; 320/320 | Preserve final-normalization reduction and the BF16 rounding of its retained residual sum; isolated check includes that fused addition. |
| Full BF16 target head | 4.140 | 320/320; 320/320; eager/compiled 320/320; 320/320 | Interleave two arithmetic-preserving M4 groups in one HIP launch, removing duplicated launch and concatenation overhead. |
| Drafter | 6.518 | N/A: no isolated target top-20 prediction | Unchanged proposal model; its timings and predictions are not target-M1 equivalence measurements. |
| Other GPU bookkeeping | 0.560 | N/A: no isolated target top-20 prediction | Sampling/state bookkeeping outside the model scopes. Exact semantic attribution is unavailable; each kernel remains listed below. |

**Sum of measured GPU dispatch durations: 58.481 ms per profiled round.** This sum excludes host gaps and queue time and is not the uninstrumented round timer.

<details>
<summary>Every decoder layer: current projection and remaining-work timings</summary>

Each layer has four projections. Gate and up are one joint GEMM; there is no separately measured gate/up split.

| Layer | Type | All layer work ms | Input projection ms | Output projection ms | Gate/up projection ms | Down projection ms | Other work ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | GDN | 0.6298 | 0.0760 | 0.0388 | 0.3704 | 0.0798 | 0.0649 |
| 1 | GDN | 0.6246 | 0.0801 | 0.0383 | 0.3664 | 0.0799 | 0.0599 |
| 2 | GDN | 0.6298 | 0.0799 | 0.0389 | 0.3692 | 0.0798 | 0.0620 |
| 3 | Attention | 0.9203 | 0.0679 | 0.0325 | 0.3743 | 0.0796 | 0.3659 |
| 4 | GDN | 0.6387 | 0.0795 | 0.0389 | 0.3784 | 0.0802 | 0.0617 |
| 5 | GDN | 0.6460 | 0.0803 | 0.0387 | 0.3844 | 0.0798 | 0.0627 |
| 6 | GDN | 0.6501 | 0.0799 | 0.0392 | 0.3858 | 0.0804 | 0.0647 |
| 7 | Attention | 0.9506 | 0.0687 | 0.0325 | 0.3909 | 0.0794 | 0.3790 |
| 8 | GDN | 0.6532 | 0.0800 | 0.0387 | 0.3905 | 0.0799 | 0.0641 |
| 9 | GDN | 0.6572 | 0.0806 | 0.0394 | 0.3924 | 0.0805 | 0.0644 |
| 10 | GDN | 0.6575 | 0.0811 | 0.0397 | 0.3913 | 0.0800 | 0.0653 |
| 11 | Attention | 0.9490 | 0.0679 | 0.0323 | 0.3899 | 0.0800 | 0.3788 |
| 12 | GDN | 0.6541 | 0.0805 | 0.0397 | 0.3905 | 0.0799 | 0.0634 |
| 13 | GDN | 0.6551 | 0.0816 | 0.0391 | 0.3897 | 0.0802 | 0.0645 |
| 14 | GDN | 0.6587 | 0.0808 | 0.0386 | 0.3939 | 0.0801 | 0.0653 |
| 15 | Attention | 0.9551 | 0.0681 | 0.0325 | 0.3940 | 0.0802 | 0.3803 |
| 16 | GDN | 0.6576 | 0.0809 | 0.0388 | 0.3931 | 0.0803 | 0.0646 |
| 17 | GDN | 0.6607 | 0.0810 | 0.0392 | 0.3952 | 0.0802 | 0.0651 |
| 18 | GDN | 0.6637 | 0.0805 | 0.0396 | 0.3972 | 0.0803 | 0.0661 |
| 19 | Attention | 0.9606 | 0.0689 | 0.0327 | 0.3961 | 0.0798 | 0.3831 |
| 20 | GDN | 0.6628 | 0.0804 | 0.0398 | 0.3949 | 0.0804 | 0.0673 |
| 21 | GDN | 0.6637 | 0.0815 | 0.0401 | 0.3955 | 0.0806 | 0.0660 |
| 22 | GDN | 0.6630 | 0.0810 | 0.0391 | 0.3966 | 0.0808 | 0.0654 |
| 23 | Attention | 0.9618 | 0.0687 | 0.0329 | 0.3966 | 0.0797 | 0.3839 |
| 24 | GDN | 0.6633 | 0.0809 | 0.0392 | 0.3970 | 0.0807 | 0.0653 |
| 25 | GDN | 0.6629 | 0.0808 | 0.0395 | 0.3959 | 0.0810 | 0.0658 |
| 26 | GDN | 0.6633 | 0.0812 | 0.0392 | 0.3968 | 0.0807 | 0.0653 |
| 27 | Attention | 0.9642 | 0.0690 | 0.0325 | 0.3978 | 0.0799 | 0.3850 |
| 28 | GDN | 0.6658 | 0.0806 | 0.0392 | 0.3989 | 0.0802 | 0.0670 |
| 29 | GDN | 0.6682 | 0.0814 | 0.0392 | 0.4009 | 0.0807 | 0.0661 |
| 30 | GDN | 0.6682 | 0.0807 | 0.0390 | 0.4019 | 0.0807 | 0.0659 |
| 31 | Attention | 0.9707 | 0.0688 | 0.0325 | 0.4007 | 0.0798 | 0.3889 |
| 32 | GDN | 0.6671 | 0.0805 | 0.0390 | 0.3999 | 0.0809 | 0.0668 |
| 33 | GDN | 0.6675 | 0.0806 | 0.0390 | 0.4013 | 0.0809 | 0.0656 |
| 34 | GDN | 0.6704 | 0.0807 | 0.0393 | 0.4022 | 0.0808 | 0.0674 |
| 35 | Attention | 0.9715 | 0.0689 | 0.0325 | 0.4021 | 0.0802 | 0.3880 |
| 36 | GDN | 0.6680 | 0.0802 | 0.0392 | 0.4024 | 0.0803 | 0.0660 |
| 37 | GDN | 0.6701 | 0.0811 | 0.0395 | 0.4033 | 0.0807 | 0.0655 |
| 38 | GDN | 0.6717 | 0.0803 | 0.0396 | 0.4043 | 0.0810 | 0.0665 |
| 39 | Attention | 0.9784 | 0.0687 | 0.0326 | 0.4058 | 0.0803 | 0.3911 |
| 40 | GDN | 0.6723 | 0.0808 | 0.0391 | 0.4064 | 0.0807 | 0.0652 |
| 41 | GDN | 0.6730 | 0.0807 | 0.0394 | 0.4057 | 0.0806 | 0.0666 |
| 42 | GDN | 0.6751 | 0.0816 | 0.0399 | 0.4062 | 0.0813 | 0.0661 |
| 43 | Attention | 0.9743 | 0.0685 | 0.0327 | 0.4038 | 0.0802 | 0.3892 |
| 44 | GDN | 0.6686 | 0.0805 | 0.0388 | 0.4022 | 0.0805 | 0.0665 |
| 45 | GDN | 0.6672 | 0.0807 | 0.0394 | 0.4014 | 0.0805 | 0.0653 |
| 46 | GDN | 0.6699 | 0.0815 | 0.0388 | 0.4026 | 0.0809 | 0.0661 |
| 47 | Attention | 0.9736 | 0.0691 | 0.0328 | 0.4041 | 0.0800 | 0.3877 |
| 48 | GDN | 0.6745 | 0.0810 | 0.0392 | 0.4057 | 0.0807 | 0.0679 |
| 49 | GDN | 0.6746 | 0.0807 | 0.0390 | 0.4075 | 0.0807 | 0.0667 |
| 50 | GDN | 0.6721 | 0.0810 | 0.0397 | 0.4043 | 0.0804 | 0.0667 |
| 51 | Attention | 0.9744 | 0.0686 | 0.0323 | 0.4037 | 0.0801 | 0.3898 |
| 52 | GDN | 0.6704 | 0.0806 | 0.0391 | 0.4040 | 0.0804 | 0.0663 |
| 53 | GDN | 0.6711 | 0.0813 | 0.0390 | 0.4047 | 0.0805 | 0.0656 |
| 54 | GDN | 0.6741 | 0.0814 | 0.0393 | 0.4058 | 0.0811 | 0.0665 |
| 55 | Attention | 0.9742 | 0.0685 | 0.0325 | 0.4032 | 0.0800 | 0.3899 |
| 56 | GDN | 0.6692 | 0.0805 | 0.0393 | 0.4028 | 0.0806 | 0.0660 |
| 57 | GDN | 0.6692 | 0.0818 | 0.0397 | 0.4014 | 0.0804 | 0.0659 |
| 58 | GDN | 0.6695 | 0.0804 | 0.0399 | 0.4021 | 0.0809 | 0.0662 |
| 59 | Attention | 0.9620 | 0.0693 | 0.0327 | 0.4019 | 0.0800 | 0.3781 |
| 60 | GDN | 0.6723 | 0.0805 | 0.0394 | 0.4053 | 0.0805 | 0.0667 |
| 61 | GDN | 0.6802 | 0.0815 | 0.0398 | 0.4099 | 0.0807 | 0.0682 |
| 62 | GDN | 0.6779 | 0.0815 | 0.0393 | 0.4077 | 0.0809 | 0.0686 |
| 63 | Attention | 0.9819 | 0.0694 | 0.0330 | 0.4064 | 0.0801 | 0.3932 |

</details>

<details>
<summary>Every recorded kernel, grouped by stage</summary>

| Stage / compiled kernel | Calls in retained rounds | Current GPU ms per round |
| --- | ---: | ---: |
| Drafter / `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x32_MI16x16x1_SN_LDSB1_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_CD1_1_CLR0_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB1_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA1201_ICIW0_IU1_K1_LDSTI0_LBSPPA128_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB0_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB2_ONLL1_PAP0_PGL0_PGR1_PLR1_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1.kd` | 6 | 0.007186 |
| Drafter / `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x32x256_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB1_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA1201_ICIW0_IU1_K1_LDSTI0_LBSPPA512_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB0_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB16_ONLL0_PAP0_PGL0_PGR1_PLR1_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_4_1.kd` | 6 | 0.177167 |
| Drafter / `__amd_rocclr_copyBuffer.kd` | 6 | 0.002012 |
| Drafter / `__amd_rocclr_fillBufferAligned.kd` | 18 | 0.006977 |
| Drafter / `_cache_draft_logits_kernel.kd` | 6 | 0.002292 |
| Drafter / `_draft_head_int2.kd` | 6 | 0.840683 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_17408_EVEN_K_1_GRID_MN_40_cache_modifier_NONE.kd` | 30 | 0.739805 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_25600_EVEN_K_1_GRID_MN_40_cache_modifier_NONE.kd` | 6 | 0.228567 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_40_cache_modifier_NONE.kd` | 30 | 0.203789 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_5120_EVEN_K_1_GRID_MN_272_cache_modifier_NONE.kd` | 30 | 1.447936 |
| Drafter / `_gemm_a8w8_blockscale_preshuffle_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_8_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_5120_EVEN_K_1_GRID_MN_48_cache_modifier_NONE.kd` | 30 | 0.274569 |
| Drafter / `_prepare_dflash_inputs_kernel.kd` | 6 | 0.008032 |
| Drafter / `_rerank_exact.kd` | 6 | 0.008919 |
| Drafter / `_selector_walk_kernel.kd` | 6 | 0.008092 |
| Drafter / `kernel_unified_attention.kd` | 30 | 1.957785 |
| Drafter / `reshape_and_cache_kernel_flash.kd` | 60 | 0.027483 |
| Drafter / `triton_per_fused_4.kd` | 6 | 0.001986 |
| Drafter / `triton_per_fused_8.kd` | 24 | 0.007603 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_preshuffle_gemm_squeeze_view_0.kd` | 30 | 0.010142 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_preshuffle_gemm_squeeze_view_2.kd` | 6 | 0.002086 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_preshuffle_gemm_squeeze_view_4.kd` | 54 | 0.016224 |
| Drafter / `triton_per_fused__to_copy_abs_clamp_div_max_view_0.kd` | 6 | 0.005499 |
| Drafter / `triton_poi_fused_0.kd` | 6 | 0.002606 |
| Drafter / `triton_poi_fused_5.kd` | 6 | 0.002319 |
| Drafter / `triton_poi_fused_9.kd` | 24 | 0.009156 |
| Drafter / `triton_poi_fused__to_copy_clamp_div_preshuffle_gemm_squeeze_view_1.kd` | 30 | 0.011862 |
| Drafter / `triton_poi_fused__to_copy_clamp_div_preshuffle_gemm_squeeze_view_3.kd` | 6 | 0.002146 |
| Drafter / `triton_poi_fused__to_copy_clamp_div_preshuffle_gemm_squeeze_view_5.kd` | 54 | 0.016211 |
| Drafter / `triton_poi_fused_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_3.kd` | 54 | 0.031305 |
| Drafter / `triton_poi_fused_add_arange_bitwise_and_constant_pad_nd_ge_mul_rms_norm_select_slice_unsqueeze_view_1.kd` | 6 | 0.023693 |
| Drafter / `triton_poi_fused_add_permute_unsqueeze_view_2.kd` | 6 | 0.001846 |
| Drafter / `triton_poi_fused_cat_expand_index_mul_slice_unsqueeze_view_1.kd` | 6 | 0.002539 |
| Drafter / `triton_red_fused__to_copy_abs_clamp_div_max_mul_preshuffle_gemm_silu_slice_squeeze_view_6.kd` | 30 | 0.018855 |
| Drafter / `triton_red_fused__to_copy_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_w4_gemm_2.kd` | 30 | 0.032549 |
| Drafter / `triton_red_fused__to_copy_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_w4_gemm_7.kd` | 24 | 0.027043 |
| Drafter / `triton_red_fused__to_copy_embedding_mul_rms_norm_w4_gemm_0.kd` | 6 | 0.004519 |
| Drafter / `triton_red_fused_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_7.kd` | 6 | 0.006599 |
| Drafter / `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) [clone .kd]` | 12 | 0.009498 |
| Drafter / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 6 | 0.003532 |
| Drafter / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 6 | 0.003066 |
| Drafter / `void at::native::bitonicSortKVInPlace<2, -1, 16, 16, c10::BFloat16, long, at::native::GTOp<c10::BFloat16, true>, unsigned int>(at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, at::native::GTOp<c10::BFloat16, true>) [clone .kd]` | 6 | 0.003306 |
| Drafter / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 6 | 0.005039 |
| Drafter / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 6 | 0.002546 |
| Drafter / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 6 | 0.003692 |
| Drafter / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 6 | 0.003146 |
| Drafter / `void at::native::mbtopk::computeBlockDigitCounts<c10::BFloat16, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 12 | 0.041232 |
| Drafter / `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 24 | 0.057470 |
| Drafter / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, c10::BFloat16>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 12 | 0.022578 |
| Drafter / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, float>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, float*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 24 | 0.029063 |
| Drafter / `void at::native::mbtopk::fill<unsigned int, unsigned int>(unsigned int*, unsigned int, unsigned int) [clone .kd]` | 12 | 0.003005 |
| Drafter / `void at::native::mbtopk::gatherTopK<c10::BFloat16, unsigned int, 2>(at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, c10::BFloat16*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 6 | 0.021259 |
| Drafter / `void at::native::mbtopk::gatherTopK<float, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, float*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 6 | 0.013026 |
| Drafter / `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4>) [clone .kd]` | 6 | 0.003799 |
| Drafter / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul>) [clone .kd]` | 6 | 0.001866 |
| Drafter / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) [clone .kd]` | 6 | 0.002486 |
| Drafter / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>) [clone .kd]` | 12 | 0.003858 |
| Drafter / `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>) [clone .kd]` | 12 | 0.005218 |
| Drafter / `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool) [clone .kd]` | 6 | 0.002139 |
| Drafter / `void at::native::warpMergeSortKVInPlace<2, -1, 128, 16, float, long, at::native::GTOp<float, true>, unsigned int, 32>(at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, at::native::GTOp<float, true>, float) [clone .kd]` | 6 | 0.005019 |
| Drafter / `void r4d_gemm_w4a16_nt_m64_kernel<1, 1, false>(unsigned short const*, unsigned int const*, unsigned int const*, __hip_bfloat16*, int, int, int, int, int) [clone .kd]` | 6 | 0.004599 |
| Drafter / `void r4d_gemm_w4a16_nt_m64_kernel<1, 1, true>(unsigned short const*, unsigned int const*, unsigned int const*, __hip_bfloat16*, int, int, int, int, int) [clone .kd]` | 60 | 0.081410 |
| Drafter / `void vllm::rms_norm_kernel<c10::BFloat16, 8, 2, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, long, long, long, c10::BFloat16 const*, long, float, int, int) [clone .kd]` | 6 | 0.003192 |
| Drafter / `void vllm::rms_norm_kernel<c10::BFloat16, 8, 4, true>(c10::BFloat16*, c10::BFloat16 const*, long, long, long, long, long, c10::BFloat16 const*, long, float, int, int) [clone .kd]` | 6 | 0.002779 |
| Drafter / `void vllm::rotary_embedding_kernel<c10::BFloat16, c10::BFloat16, true>(long const*, c10::BFloat16*, c10::BFloat16*, c10::BFloat16 const*, int, long, long, long, int, int, int, long, bool) [clone .kd]` | 6 | 0.002579 |
| Attention KV write / `reshape_and_cache_kernel_flash.kd` | 96 | 0.046878 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_1.kd` | 96 | 0.024404 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_3.kd` | 96 | 0.027984 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_4.kd` | 96 | 0.028471 |
| Attention Q/K normalization, RoPE and layout / `triton_poi_fused_arange_bitwise_and_eq_index_lt_remainder_select_split_where_2.kd` | 96 | 0.029938 |
| Attention Q/K normalization, RoPE and layout / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 96 | 0.033671 |
| Attention Q/K normalization, RoPE and layout / `void stock_m1_gemma_norm<false, 256, 32>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, long, long, float, float*) [clone .kd]` | 96 | 0.050797 |
| Attention Q/K normalization, RoPE and layout / `void stock_m1_gemma_norm<false, 256, 64>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, long, long, float, float*) [clone .kd]` | 96 | 0.040311 |
| Attention decode / `void qwen_stock_m1_shared_decode<4, 16, 256, 6, 16, 0, 3430971>(R4DArgs, int) [clone .kd]` | 96 | 5.285290 |
| Attention input activation FP8 quantization / `void vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<c10::BFloat16, c10::Float8_e4m3fn>(c10::Float8_e4m3fn*, float*, c10::BFloat16 const*, float const*, int, long, long) [clone .kd]` | 96 | 0.042158 |
| Attention input projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 1, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 96 | 1.098923 |
| Attention output activation FP8 quantization / `void vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<c10::BFloat16, c10::Float8_e4m3fn>(c10::Float8_e4m3fn*, float*, c10::BFloat16 const*, float const*, int, long, long) [clone .kd]` | 96 | 0.045791 |
| Attention output gating / `triton_poi_fused_mul_mxfp4_linear_sigmoid_view_0.kd` | 96 | 0.032364 |
| Attention output projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 4, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 96 | 0.521433 |
| Attention split-KV merge / `void qwen_stock_m1_shared_merge<256, 4, 1>(R4DArgs, int, int) [clone .kd]` | 96 | 0.119631 |
| Embedding + first input normalization / `triton_poi_fused__to_copy_embedding_0.kd` | 6 | 0.002599 |
| Embedding + first input normalization / `void stock_m1_gemma_norm<false, 5120, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, long, long, float, float*) [clone .kd]` | 6 | 0.005159 |
| Final normalization/layout / `void stock_m1_gemma_norm<true, 5120, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, long, long, float, float*) [clone .kd]` | 6 | 0.005612 |
| GDN convolution / `_causal_conv1d_update_kernel.kd` | 288 | 0.226365 |
| GDN input activation FP8 quantization / `void vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<c10::BFloat16, c10::Float8_e4m3fn>(c10::Float8_e4m3fn*, float*, c10::BFloat16 const*, float const*, int, long, long) [clone .kd]` | 288 | 0.127346 |
| GDN input projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 1, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 288 | 3.872797 |
| GDN layout/copies and buffer initialization / `triton_poi_fused_add_1.kd` | 18 | 0.004004 |
| GDN layout/copies and buffer initialization / `triton_poi_fused_add_2.kd` | 12 | 0.002738 |
| GDN layout/copies and buffer initialization / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 288 | 0.157119 |
| GDN layout/copies and buffer initialization / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) [clone .kd]` | 288 | 0.092759 |
| GDN layout/copies and buffer initialization / `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>) [clone .kd]` | 288 | 0.074886 |
| GDN output activation FP8 quantization / `void vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<c10::BFloat16, c10::Float8_e4m3fn>(c10::Float8_e4m3fn*, float*, c10::BFloat16 const*, float const*, int, long, long) [clone .kd]` | 288 | 0.131086 |
| GDN output gated normalization / `layer_norm_fwd_kernel.kd` | 288 | 0.097307 |
| GDN output projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 4, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 288 | 1.883793 |
| GDN recurrence and gates / `stock_gdn_scan_kernel.kd` | 288 | 1.218805 |
| Layer input residual/normalization / `void stock_m1_gemma_norm<true, 5120, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, long, long, float, float*) [clone .kd]` | 378 | 0.345972 |
| MLP SiLU and gating / `triton_poi_fused_mul_mxfp4_linear_silu_slice_0.kd` | 288 | 0.141272 |
| MLP SiLU and gating / `triton_poi_fused_mul_mxfp4_linear_silu_slice_1.kd` | 96 | 0.048204 |
| MLP down input FP8 quantization / `void vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<c10::BFloat16, c10::Float8_e4m3fn>(c10::Float8_e4m3fn*, float*, c10::BFloat16 const*, float const*, int, long, long) [clone .kd]` | 384 | 0.297344 |
| MLP down projection / `void radiance_mxfp4_fp8_gemm_decode<8, 128, 4, 1, true, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, float*, int*, std::bfloat16_t*, int, int, int) [clone .kd]` | 384 | 5.142933 |
| MLP gate/up input FP8 quantization / `void vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<c10::BFloat16, c10::Float8_e4m3fn>(c10::Float8_e4m3fn*, float*, c10::BFloat16 const*, float const*, int, long, long) [clone .kd]` | 384 | 0.166543 |
| MLP gate/up projection / `void radiance_mxfp4_fp8_gemm_folded<2, true, true>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, float const*, std::bfloat16_t*, int, int, int) [clone .kd]` | 384 | 25.437279 |
| Post-attention/GDN residual/normalization / `void stock_m1_gemma_norm<true, 5120, 512>(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, long, long, float, float*) [clone .kd]` | 384 | 0.352599 |
| Full BF16 target head / `(anonymous namespace)::stock_m1_head_pair(__hip_bfloat16 const*, __hip_bfloat16 const*, __hip_bfloat16*) [clone .kd]` | 6 | 4.139846 |
| Other GPU bookkeeping / `__amd_rocclr_copyBuffer.kd` | 172 | 0.076165 |
| Other GPU bookkeeping / `__amd_rocclr_fillBufferAligned.kd` | 6 | 0.002519 |
| Other GPU bookkeeping / `_combine_sampled_and_draft_tokens_kernel.kd` | 7 | 0.003485 |
| Other GPU bookkeeping / `_compute_local_logits_stats_kernel.kd` | 6 | 0.028819 |
| Other GPU bookkeeping / `_compute_slot_mappings_kernel.kd` | 7 | 0.003485 |
| Other GPU bookkeeping / `_expand_idx_mapping_kernel.kd` | 7 | 0.002339 |
| Other GPU bookkeeping / `_gather_block_tables_kernel.kd` | 7 | 0.005852 |
| Other GPU bookkeeping / `_get_num_sampled_and_rejected_kernel.kd` | 6 | 0.002792 |
| Other GPU bookkeeping / `_insert_resampled_kernel.kd` | 6 | 0.003499 |
| Other GPU bookkeeping / `_post_update_kernel.kd` | 6 | 0.005206 |
| Other GPU bookkeeping / `_prepare_pos_seq_lens_kernel.kd` | 7 | 0.002392 |
| Other GPU bookkeeping / `_prepare_rope_positions_kernel.kd` | 7 | 0.003232 |
| Other GPU bookkeeping / `_rejection_kernel.kd` | 6 | 0.007219 |
| Other GPU bookkeeping / `_resample_kernel.kd` | 6 | 0.015366 |
| Other GPU bookkeeping / `_scatter_num_accepted_kernel.kd` | 6 | 0.001986 |
| Other GPU bookkeeping / `postprocess_mamba_fused_kernel.kd` | 6 | 0.002699 |
| Other GPU bookkeeping / `precopy_mamba_align_fused_kernel.kd` | 7 | 0.003099 |
| Other GPU bookkeeping / `preprocess_mamba_align_fused_kernel.kd` | 7 | 0.003012 |
| Other GPU bookkeeping / `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}>(int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}, function_traits<at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}>::result_type*) [clone .kd]` | 42 | 0.014853 |
| Other GPU bookkeeping / `void (anonymous namespace)::softmax_warp_forward<float, float, float, 6, false, false, 32>(float*, float const*, int, int, int, bool const*, int, bool) [clone .kd]` | 6 | 0.002352 |
| Other GPU bookkeeping / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 48 | 0.015559 |
| Other GPU bookkeeping / `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<4>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]` | 6 | 0.003046 |
| Other GPU bookkeeping / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 48 | 0.020886 |
| Other GPU bookkeeping / `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<int> const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<int> const&)::{lambda(int, bool)#1}) [clone .kd]` | 42 | 0.018933 |
| Other GPU bookkeeping / `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<float> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<float> const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<float> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<float> const&)::{lambda(int, bool)#1}) [clone .kd]` | 18 | 0.019577 |
| Other GPU bookkeeping / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 109 | 0.053589 |
| Other GPU bookkeeping / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 12 | 0.005985 |
| Other GPU bookkeeping / `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_put_kernel_impl<at::native::OpaqueType<8> >(at::TensorIterator&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]` | 6 | 0.002979 |
| Other GPU bookkeeping / `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*) [clone .kd]` | 24 | 0.052110 |
| Other GPU bookkeeping / `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, float>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, float*, unsigned int*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 24 | 0.038716 |
| Other GPU bookkeeping / `void at::native::mbtopk::fill<unsigned int, unsigned int>(unsigned int*, unsigned int, unsigned int) [clone .kd]` | 6 | 0.001706 |
| Other GPU bookkeeping / `void at::native::mbtopk::gatherTopK<float, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, float*, unsigned int*, unsigned int*, unsigned int) [clone .kd]` | 6 | 0.026026 |
| Other GPU bookkeeping / `void at::native::tensor_kernel_scan_innermost_dim<float, std::plus<float> >(float*, float const*, unsigned int, unsigned int, unsigned int, float, std::plus<float>) [clone .kd]` | 6 | 0.002726 |
| Other GPU bookkeeping / `void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>, 4, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast) [clone .kd]` | 42 | 0.012020 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<16, at::native::BinaryFunctor<bool, bool, bool, at::native::BitwiseOrFunctor<bool> >, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<bool, bool, bool, at::native::BitwiseOrFunctor<bool> >, std::array<char*, 3ul>) [clone .kd]` | 6 | 0.002659 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<16, at::native::FillFunctor<bool>, std::array<char*, 1ul> >(int, at::native::FillFunctor<bool>, std::array<char*, 1ul>) [clone .kd]` | 7 | 0.002185 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<16, at::native::bitwise_not_kernel_cuda(at::TensorIteratorBase&)::{lambda(bool)#1}, std::array<char*, 2ul> >(int, at::native::bitwise_not_kernel_cuda(at::TensorIteratorBase&)::{lambda(bool)#1}, std::array<char*, 2ul>) [clone .kd]` | 6 | 0.002332 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}, std::array<char*, 2ul>) [clone .kd]` | 42 | 0.014173 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}, std::array<char*, 2ul>) [clone .kd]` | 6 | 0.001859 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}, std::array<char*, 3ul> >(int, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}, std::array<char*, 3ul>) [clone .kd]` | 6 | 0.010066 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool, float, float)#1}, std::array<char*, 4ul> >(int, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool, float, float)#1}, std::array<char*, 4ul>) [clone .kd]` | 12 | 0.004878 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 2ul> >(int, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(int, int)#1}>, std::array<char*, 2ul>) [clone .kd]` | 42 | 0.015160 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, 2ul>) [clone .kd]` | 48 | 0.016085 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul>) [clone .kd]` | 6 | 0.001899 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>) [clone .kd]` | 6 | 0.001992 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>) [clone .kd]` | 12 | 0.003111 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int, at::native::FillFunctor<int>, std::array<char*, 1ul>) [clone .kd]` | 7 | 0.001886 |
| Other GPU bookkeeping / `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>) [clone .kd]` | 6 | 0.009939 |
| Other GPU bookkeeping / `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool) [clone .kd]` | 6 | 0.002252 |
| Other GPU bookkeeping / `void at::native::warpMergeSortKVInPlace<2, -1, 128, 16, float, long, at::native::GTOp<float, true>, unsigned int, 32>(at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, at::native::GTOp<float, true>, float) [clone .kd]` | 6 | 0.004999 |

</details>

## Uninstrumented performance

Separate uninstrumented compiled direct-engine controls with the full BF16 target head. Three natural responses on the same 60,000-input-token Pi prefix; excludes HTTP/Pi, tool execution, snapshot publication, cold prefill and warm-up.

| Measurement | Current result |
| --- | ---: |
| Median round | 60.052 ms |
| Pooled rate after first output | 84.342 tok/s |
| Output tokens / natural responses | 2,525 / 3 |
| Timed post-first output | 29.902 s |
| Separate rounding-control median round | 60.900 ms |
| Rounding-control natural responses / output tokens | 6 / 5,050 |
| Rounding-control mean committed tokens per round | 5.211 |
| Rounding-control pooled rate | 83.070 tok/s |

These approximately 80–86 tok/s results are brief **60K-input** controls, not the separate 60K-generated-token head benchmark. No eager execution or instrumented timing is substituted for these complete-round measurements.

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
