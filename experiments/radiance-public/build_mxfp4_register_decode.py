"""Build an isolated register-fed decode candidate against the qualified release.

Only fragment-ordered, one-M-tile decode changes. The WMMA operands, their
order, split-K boundaries and epilogue stay fixed. This builder does not install
anything. A successful build is not correctness or performance evidence.
"""

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arithmetic_fold_tables(source):
    """Construct the identical 15 fold tables in registers, avoiding indexed loads.

    For shifts -6..5 all four byte lanes are normal E4M3 values. The final
    three shifts use the exact subnormal entries from the qualified table.
    Unsigned word arithmetic is intentional, including negative shifts.
    """
    helper = r"""
static __device__ __forceinline__ unsigned int coherence_fold_lo(int index) {
  const int d = index - 6;
  return d <= 5 ? 0x3c383000u - (unsigned int)d * 0x08080800u
                : 0x0c080400u >> (d - 6);
}
static __device__ __forceinline__ unsigned int coherence_fold_hi(int index) {
  const int d = index - 6;
  return d <= 7 ? 0x4c484440u - (unsigned int)d * 0x08080808u
                : 0x0c080604u;
}
"""
    start = source.index("template <int DWN, int DBK, int DKS, int DTM")
    end = source.index("// Split-K partial slab.", start)
    body = source[start:end]
    for index in ("dsh", "d", "g"):
        body = body.replace(f"kMag[{index}][0]", f"coherence_fold_lo({index})")
        body = body.replace(f"kMag[{index}][1]", f"coherence_fold_hi({index})")
    if body == source[start:end]:
        raise ValueError("no qualified fold tables were replaced")
    return source[:start] + helper + body + source[end:]


def local_split_reduction(source):
    """Group the unchanged K partitions inside one workgroup.

    Each warp executes precisely its original WMMA chain. The four partials
    are added in their original fixed order, using shared rather than global
    memory. The production weight and recurrent-state layouts are unchanged.
    """
    source = patched_source(source, direct_a=True)
    start = source.index("template <int DWN, int DBK, int DKS, int DTM")
    end = source.index("// Split-K partial slab.", start)
    body = source[start:end]
    body = body.replace(
        "  constexpr int BND = DWN * 16;",
        "  constexpr bool LOCAL_K = WPERM && DTM == 1 && DKS > 1;\n"
        "  constexpr int N_WAVES = LOCAL_K ? DWN / DKS : DWN;\n"
        "  constexpr int BND = N_WAVES * 16;\n"
        "  __shared__ float local_partials[DKS * 16 * BND];",
    )
    body = body.replace(
        "  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;",
        "  const int tid = threadIdx.x, lane = tid & 31;\n"
        "  const int wave = (tid >> 5) % N_WAVES;",
    ).replace(
        "  const int ks = blockIdx.z;",
        "  const int ks = LOCAL_K ? (tid >> 5) / N_WAVES : blockIdx.z;",
    )
    anchor = "  // At DKS==1 the accumulator already holds the whole K range, so the partial buffer, the"
    if body.count(anchor) != 1:
        raise ValueError("qualified split reduction anchor changed")
    body = body.replace(
        anchor,
        r"""  if constexpr (LOCAL_K) {
    if (n < N) {
      #pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = kb8 + e;
        if (m < M) local_partials[(ks * 16 + m) * BND + n - n0] = acc[0][e];
      }
    }
    __syncthreads();
    if (ks != 0) return;
    for (int nn = n0 + tid; nn < min(n0 + BND, N); nn += N_WAVES * 32) {
      const float rf = __int_as_float((int)Wref[nn] << 23);
      for (int m = 0; m < M; ++m) {
        float sum = 0.f;
        for (int part = 0; part < DKS; ++part)
          sum += local_partials[(part * 16 + m) * BND + nn - n0];
        C[(size_t)m * N + nn] = (__bf16)(sum * rf * As[m]);
      }
    }
    return;
  }

"""
        + anchor,
    )
    source = source[:start] + body + source[end:]
    grid = "                         dim3(dec_nblk, 1, KS_), dblock,"
    if source.count(grid) != 1:
        raise ValueError("qualified launch grid changed")
    return source.replace(
        grid,
        "                         dim3((N + ((TM_ == 1 && RAD_WPERM) ? DWN / KS_ : DWN) * 16 - 1) / (((TM_ == 1 && RAD_WPERM) ? DWN / KS_ : DWN) * 16), 1, (TM_ == 1 && RAD_WPERM) ? 1 : KS_), dblock,",
    )


def pipelined_staged_weights(source, *, local_barrier=True):
    """Overlap raw weight reads with unchanged WMMA and coalesced loads."""
    anchor = """  if constexpr (WPERM) {
    for (int s = s_lo; s < s_hi; ++s) {
      const int k0 = s * DBK;
      stage_a(k0);
      stage_w_wperm(k0);
      mma_slab();
    }
  } else {"""
    if source.count(anchor) != 1:
        raise ValueError("qualified staged decode loop changed")
    replacement = r"""  if constexpr (WPERM && DTM == 1) {
    constexpr int NI = DBK / 32;
    uint2_t packed[2][NI];
    unsigned short scales[2][NI], references[NI];
    int rows[NI], klocs[NI], globals[NI], lanes[NI], steps[NI];
    int2_t activation[2][DBK / 16];
    const unsigned int* __restrict__ Wp = (const unsigned int*)W;
    #pragma unroll
    for (int it = 0; it < NI; ++it) {
      const int sl = it * DNTHREADS * 2 + tid * 2;
      const int lane_ = sl & 31, rest = sl >> 5;
      steps[it] = rest % (DBK / 16);
      const int ntl = rest / (DBK / 16);
      rows[it] = ntl * 16 + (lane_ & 15);
      klocs[it] = steps[it] * 16 + (lane_ >> 4) * 8;
      const int gn = n0 + rows[it];
      globals[it] = gn < N - 2 ? gn : N - 2;
      lanes[it] = (lane_ & 16) | (globals[it] & 15);
      references[it] = *(const unsigned short*)(Wref + globals[it]);
    }
    auto fetch = [&](int slab, int bank) {
      const int k0 = slab * DBK;
      #pragma unroll
      for (int it = 0; it < NI; ++it) {
        packed[bank][it] = ld_w<NT>((const uint2_t*)&Wp[
            ((size_t)(globals[it] >> 4) * (K / 16) + k0 / 16 + steps[it]) * 32 + lanes[it]]);
        scales[bank][it] = *(const unsigned short*)(
            Ws + (size_t)((k0 + steps[it] * 16) / 32) * N + globals[it]);
      }
      const int ar = col < M - 1 ? col : M - 1;
      #pragma unroll
      for (int step = 0; step < DBK / 16; ++step)
        activation[bank][step] = *(const int2_t*)(A + (size_t)ar * K + k0 + step * 16 + kb8);
    };
    auto consume = [&](int bank) {
      #pragma unroll
      for (int it = 0; it < NI; ++it) {
        #pragma unroll
        for (int q = 0; q < 2; ++q) {
          int d = (int)((references[it] >> (8*q)) & 255) -
                  (int)((scales[bank][it] >> (8*q)) & 255);
          d = (d < -6 ? -6 : (d > 8 ? 8 : d)) + 6;
          const unsigned int t0 = kMag[d][0], t1 = kMag[d][1];
          const unsigned int wv = packed[bank][it][q];
          const unsigned int ev = wv & 0x0F0F0F0Fu, od = (wv >> 4) & 0x0F0F0F0Fu;
          const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u) | ((ev & 0x08080808u) << 4);
          const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u) | ((od & 0x08080808u) << 4);
          *(uint2_t*)(&sW[(rows[it] + q) * DWSTR + klocs[it]]) =
              uint2_t{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                      __builtin_amdgcn_perm(bo, be, 0x07030602u)};
        }
      }
      COHERENCE_STAGE_BARRIER();
      #pragma unroll
      for (int step = 0; step < DBK / 16; ++step) {
        const int2_t wf = *(const int2_t*)(&sW[(wave * 16 + col) * DWSTR + step * 16 + kb8]);
        acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
            activation[bank][step], wf, acc[0]);
      }
      COHERENCE_STAGE_BARRIER();
    };
    if (s_lo < s_hi) fetch(s_lo, 0);
    for (int slab = s_lo; slab < s_hi; slab += 2) {
      if (slab + 1 < s_hi) fetch(slab + 1, 1);
      consume(0);
      if (slab + 1 < s_hi) {
        if (slab + 2 < s_hi) fetch(slab + 2, 0);
        consume(1);
      }
    }
  } else if constexpr (WPERM) {
    for (int s = s_lo; s < s_hi; ++s) {
      const int k0 = s * DBK;
      stage_a(k0);
      stage_w_wperm(k0);
      mma_slab();
    }
  } else {"""
    return source.replace(
        anchor,
        replacement.replace(
            "COHERENCE_STAGE_BARRIER()",
            "radiance_lds_barrier()" if local_barrier else "__syncthreads()",
        ),
    )


def patched_source(source, direct_a=False, waves=8):
    anchor = """  if constexpr (WPERM) {
    for (int s = s_lo; s < s_hi; ++s) {
      const int k0 = s * DBK;
      stage_a(k0);
      stage_w_wperm(k0);
      mma_slab();
    }
  } else {"""
    if source.count(anchor) != 1:
        raise ValueError("qualified decode staging anchor changed")
    if "d = (d < -6 ? -6 : (d > 8 ? 8 : d)) + 6;" not in source:
        raise ValueError("requires the corrected, expanded exact fold window")
    replacement = """  // Coherence experimental register-fed fragment decode. Preserve all
  // arithmetic and split boundaries; eliminate only the sW store/load trip.
  if constexpr (WPERM && DTM == 1) {
    const int gn = n0 + wave * 16 + col;
    const int gc = gn < N - 1 ? gn : N - 1;
    const int lanec = (lane & 16) | (gc & 15);
    const unsigned int *__restrict__ Wp = (const unsigned int *)W;
    const int wr = (int)Wref[gc];
    for (int s = s_lo; s < s_hi; ++s) {
      const int k0 = s * DBK;
      unsigned int packed[DBK / 16];
      unsigned int table0[DBK / 32], table1[DBK / 32];
#pragma unroll
      for (int step = 0; step < DBK / 16; ++step)
        packed[step] = ld_w<NT>(&Wp[
            ((size_t)(gc >> 4) * (K / 16) + k0 / 16 + step) * 32 + lanec]);
#pragma unroll
      for (int g = 0; g < DBK / 32; ++g) {
        int d = wr - (int)Ws[(size_t)(k0 / 32 + g) * N + gc];
        d = (d < -6 ? -6 : (d > 8 ? 8 : d)) + 6;
        table0[g] = kMag[d][0];
        table1[g] = kMag[d][1];
      }
      stage_a(k0);
      __syncthreads();
#pragma unroll
      for (int step = 0; step < DBK / 16; ++step) {
        const unsigned int ev = packed[step] & 0x0F0F0F0Fu;
        const unsigned int od = (packed[step] >> 4) & 0x0F0F0F0Fu;
        const unsigned int be = __builtin_amdgcn_perm(
            table1[step / 2], table0[step / 2], ev & 0x07070707u)
            | ((ev & 0x08080808u) << 4);
        const unsigned int bo = __builtin_amdgcn_perm(
            table1[step / 2], table0[step / 2], od & 0x07070707u)
            | ((od & 0x08080808u) << 4);
        const int2_t wf = {
            (int)__builtin_amdgcn_perm(bo, be, 0x05010400u),
            (int)__builtin_amdgcn_perm(bo, be, 0x07030602u)};
        const unsigned char *pa = &sA[col * DASTR + step * 16 + kb8];
        const int2_t af = {*(const int *)pa, *(const int *)(pa + 4)};
        acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
            af, wf, acc[0]);
      }
      __syncthreads();
    }
  } else if constexpr (WPERM) {
    for (int s = s_lo; s < s_hi; ++s) {
      const int k0 = s * DBK;
      stage_a(k0);
      stage_w_wperm(k0);
      mma_slab();
    }
  } else {"""
    if direct_a:
        replacement = (
            replacement.replace(
                "      stage_a(k0);\n      __syncthreads();",
                """      int2_t afs[DBK / 16];
      const int ar = col < M - 1 ? col : M - 1;
#pragma unroll
      for (int step = 0; step < DBK / 16; ++step) {
        const int *pa = (const int *)(A + (size_t)ar * K + k0 + step * 16 + kb8);
        afs[step] = int2_t{pa[0], pa[1]};
      }""",
            )
            .replace(
                "        const unsigned char *pa = &sA[col * DASTR + step * 16 + kb8];\n"
                "        const int2_t af = {*(const int *)pa, *(const int *)(pa + 4)};",
                "        const int2_t af = afs[step];",
            )
            .replace("      __syncthreads();\n    }\n  } else if", "    }\n  } else if")
        )
    result = source.replace(anchor, replacement)
    if waves != 8:
        # Keep the original split-K selection (based on 128-column blocks).
        # Change only how independent N fragments are assigned to workgroups.
        launch = "      hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_decode<DWN, BK_, KS_, TM_, RAD_WPERM, true,"
        if result.count(launch) != 1:
            raise ValueError("decode launch changed")
        result = result.replace(
            launch,
            f"      hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_decode<(TM_ == 1 && RAD_WPERM ? {waves} : DWN), BK_, KS_, TM_, RAD_WPERM, true,",
        ).replace(
            "                         dim3(dec_nblk, 1, KS_), dblock,",
            f"                         dim3((N + (TM_ == 1 && RAD_WPERM ? {waves} : DWN) * 16 - 1) / ((TM_ == 1 && RAD_WPERM ? {waves} : DWN) * 16), 1, KS_), dim3((TM_ == 1 && RAD_WPERM ? {waves} : DWN) * 32),",
        )
    return result


def pipeline_register_slabs(source, *, scheduling_barrier=False):
    """Prefetch the next exact K slab while the previous slab is consumed.

    Two explicitly indexed register banks preserve the original WMMA order,
    split-K partition and epilogue. No data format or weight values change.
    Only the existing WPERM/DTM=1 path is replaced.
    """
    source = patched_source(source, direct_a=True)
    start = source.index(
        "    for (int s = s_lo; s < s_hi; ++s) {",
        source.index("// Coherence experimental register-fed"),
    )
    end = source.index("  } else if constexpr (WPERM)", start)
    replacement = r"""    unsigned int packed[2][DBK / 16];
    unsigned int table0[2][DBK / 32], table1[2][DBK / 32];
    int2_t afs[2][DBK / 16];
    auto fetch_slab = [&](int slab, int bank) {
      const int k0 = slab * DBK;
#pragma unroll
      for (int step = 0; step < DBK / 16; ++step)
        packed[bank][step] = ld_w<NT>(&Wp[
            ((size_t)(gc >> 4) * (K / 16) + k0 / 16 + step) * 32 + lanec]);
#pragma unroll
      for (int g = 0; g < DBK / 32; ++g) {
        int d = wr - (int)Ws[(size_t)(k0 / 32 + g) * N + gc];
        d = (d < -6 ? -6 : (d > 8 ? 8 : d)) + 6;
        table0[bank][g] = kMag[d][0];
        table1[bank][g] = kMag[d][1];
      }
      const int ar = col < M - 1 ? col : M - 1;
#pragma unroll
      for (int step = 0; step < DBK / 16; ++step) {
        const int *pa = (const int *)(A + (size_t)ar * K + k0 + step * 16 + kb8);
        afs[bank][step] = int2_t{pa[0], pa[1]};
      }
    };
    auto consume_slab = [&](int bank) {
#pragma unroll
      for (int step = 0; step < DBK / 16; ++step) {
        const unsigned int ev = packed[bank][step] & 0x0F0F0F0Fu;
        const unsigned int od = (packed[bank][step] >> 4) & 0x0F0F0F0Fu;
        const unsigned int be = __builtin_amdgcn_perm(
            table1[bank][step / 2], table0[bank][step / 2], ev & 0x07070707u)
            | ((ev & 0x08080808u) << 4);
        const unsigned int bo = __builtin_amdgcn_perm(
            table1[bank][step / 2], table0[bank][step / 2], od & 0x07070707u)
            | ((od & 0x08080808u) << 4);
        const int2_t wf = {
            (int)__builtin_amdgcn_perm(bo, be, 0x05010400u),
            (int)__builtin_amdgcn_perm(bo, be, 0x07030602u)};
        acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
            afs[bank][step], wf, acc[0]);
      }
    };
    if (s_lo < s_hi) fetch_slab(s_lo, 0);
    for (int slab = s_lo; slab < s_hi; slab += 2) {
      if (slab + 1 < s_hi) fetch_slab(slab + 1, 1);
      // PREFETCH_SCHEDULING_BOUNDARY
      consume_slab(0);
      if (slab + 1 < s_hi) {
        if (slab + 2 < s_hi) fetch_slab(slab + 2, 0);
        // PREFETCH_SCHEDULING_BOUNDARY
        consume_slab(1);
      }
    }
"""
    if scheduling_barrier:
        replacement = replacement.replace(
            "// PREFETCH_SCHEDULING_BOUNDARY", "__builtin_amdgcn_sched_barrier(0);"
        )
    return source[:start] + replacement + source[end:]


def tune_decode_storage(
    source, *, lds_table=False, split_interleave=False, slab_factor=1
):
    start = source.index("template <int DWN, int DBK, int DKS, int DTM")
    end = source.index("// Split-K partial slab.", start)
    body = source[start:end]
    if lds_table:
        anchor = "  __shared__ int s_last;"
        body = body.replace(anchor, anchor + "\n  __shared__ unsigned int sMag[32];")
        anchor = "  const int col = lane & 15, kb8 = (lane >> 4) * 8;"
        body = body.replace(
            anchor,
            anchor
            + "\n  if (tid < 32) sMag[tid] = ((const unsigned int *)kMag)[tid];\n  __syncthreads();",
        )
        body = body.replace("kMag[dsh][0]", "sMag[2 * dsh]").replace(
            "kMag[dsh][1]", "sMag[2 * dsh + 1]"
        )
        body = body.replace("kMag[d][0]", "sMag[2 * d]").replace(
            "kMag[d][1]", "sMag[2 * d + 1]"
        )
    if split_interleave:
        anchor = "  const int n0 = blockIdx.x * BND;\n  const int ks = blockIdx.z;"
        if body.count(anchor) != 1:
            raise ValueError("split grid anchor changed")
        body = body.replace(
            anchor,
            """  // Bijection of the original grid; partial slots and reduction order stay fixed.
  const int linear_block = blockIdx.x + blockIdx.z * gridDim.x;
  const int nblock = linear_block / DKS;
  const int n0 = nblock * BND;
  const int ks = linear_block % DKS;""",
        )
        body = body.replace("cnt[blockIdx.x]", "cnt[nblock]")
    result = source[:start] + body + source[end:]
    if slab_factor != 1:
        # A wider slab is legal only if every K split retains its boundary.
        # In particular K=17408 is divisible by 1024, but not by 512*4.
        # The first unguarded 512 experiment changed 21 BF16 output elements;
        # preserve the original dispatch whenever this divisibility fails.
        anchor = "gemm_decode<DWN, BK_, KS_, TM_, RAD_WPERM, true,"
        if result.count(anchor) != 1:
            raise ValueError("slab dispatch anchor changed")
        begin = result.index("#define RAD_DEC_LAUNCH(BK_, KS_, TM_)")
        end = result.index("#define RAD_DEC_BY_TM(BK_, KS_)", begin)
        macro = result[begin:end]
        header, *lines = macro.splitlines()
        call = "\n".join(
            line.rstrip().removesuffix("\\").rstrip() for line in lines
        ).strip()
        wide = call.replace(
            anchor,
            f"gemm_decode<DWN, (TM_ == 1 && RAD_WPERM ? BK_ * {slab_factor} : BK_), KS_, TM_, RAD_WPERM, true,",
        )
        replacement = [
            header.rstrip().removesuffix("\\").rstrip(),
            "do {",
            f"  if (TM_ == 1 && RAD_WPERM && K % (BK_ * {slab_factor} * KS_) == 0) {{",
            wide + ";",
            "  } else {",
            call + ";",
            "  }",
            "} while (false)",
        ]
        macro_lines = "\n".join(replacement).splitlines()
        result = result[:begin] + " \\\n".join(macro_lines) + "\n" + result[end:]
    return result


def paired_lds_source(source, *, pair_loads=True):
    """Load adjacent FP8 WMMA fragments together without changing accumulation.

    Swapping K-address bits 3/4 makes each lane's two eight-byte fragments
    contiguous. Row strides stay 16-byte aligned. Both 16-wide WMMAs still
    execute in the original order, with the original split-K boundaries.
    """
    begin = source.index("template <int DWN, int DBK, int DKS, int DTM,")
    end = source.index("\n#define DEC_KS", begin)
    body = source[begin:end]
    body = body.replace("DBK + DEC_PAD", "DBK + 16")
    if not pair_loads:
        return source[:begin] + body + source[end:]
    anchor = "  floatx8 acc[DTM];"
    body = body.replace(
        anchor,
        "  auto swizzle_k = [](int k) {\n"
        "    return (k & ~24) | ((k & 8) << 1) | ((k & 16) >> 1);\n"
        "  };\n" + anchor,
    )
    old = "*(uint4_t *)(&sA[r * DASTR + c]) = *(const uint4_t *)(A + (size_t)rc * K + k0 + c);"
    if body.count(old) != 1:
        raise ValueError("decode activation staging changed")
    body = body.replace(
        old,
        "const uint4_t value = *(const uint4_t *)(A + (size_t)rc * K + k0 + c);\n"
        "        *(uint2_t *)(&sA[r * DASTR + swizzle_k(c)]) = uint2_t{value[0], value[1]};\n"
        "        *(uint2_t *)(&sA[r * DASTR + swizzle_k(c + 8)]) = uint2_t{value[2], value[3]};",
    )
    old = "*(uint4_t *)(&sW[r * DWSTR + c * 2]) = uint4_t{outw[0], outw[1], outw[2], outw[3]};"
    if body.count(old) != 1:
        raise ValueError("decode row weight staging changed")
    body = body.replace(
        old,
        "*(uint2_t *)(&sW[r * DWSTR + swizzle_k(c * 2)]) = uint2_t{outw[0], outw[1]};\n"
        "      *(uint2_t *)(&sW[r * DWSTR + swizzle_k(c * 2 + 8)]) = uint2_t{outw[2], outw[3]};",
    )
    body = body.replace("(r + q) * DWSTR + kloc", "(r + q) * DWSTR + swizzle_k(kloc)")
    start = body.index("    for (int step = 0; step < DBK / 16; ++step) {")
    stop = body.index("    __syncthreads();", start)
    body = (
        body[:start]
        + """    for (int step = 0; step < DBK / 16; step += 2) {
      const int kk = step * 16 + kb8 * 2;
      const uint4_t weight = *(const uint4_t *)(&sW[(wave * 16 + col) * DWSTR + kk]);
      const int2_t wf0 = {(int)weight[0], (int)weight[1]};
      const int2_t wf1 = {(int)weight[2], (int)weight[3]};
#pragma unroll
      for (int i = 0; i < DTM; ++i) {
        const uint4_t activation = *(const uint4_t *)(&sA[(i * 16 + col) * DASTR + kk]);
        const int2_t af0 = {(int)activation[0], (int)activation[1]};
        const int2_t af1 = {(int)activation[2], (int)activation[3]};
        acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af0, wf0, acc[i]);
        acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af1, wf1, acc[i]);
      }
    }
"""
        + body[stop:]
    )
    return source[:begin] + body + source[end:]


def local_decode_barriers(source):
    """Scope slab barriers to LDS, retaining global split-K publication fences.

    The slab loop only shares local arrays between waves. All global reads
    feed each wave's own LDS stores; no global write precedes these barriers.
    The later partial writes, threadfence and atomic publication are untouched.
    Wider row tiles and non-permuted weights retain the original barriers.
    """
    begin = source.index("template <int DWN, int DBK, int DKS, int DTM,")
    end = source.index("  const int n = n0 + wave * 16 + col;", begin)
    body = source[begin:end]
    count = body.count("__syncthreads();")
    if count < 2:
        raise ValueError("decode slab barriers changed")
    body = body.replace(
        "__syncthreads();",
        "if constexpr (DTM == 1 && WPERM) radiance_lds_barrier(); else __syncthreads();",
    )
    return source[:begin] + body + source[end:]


def decode_wave_geometry(source, waves, *, minimum_occupancy=None):
    """Change independent N grouping, retaining the old K splits.

    The experiment allocates counters for one wave per workgroup. Production
    must not use narrower groups with the old completion-counter capacity.
    """
    if waves not in (1, 2, 4):
        raise ValueError("unsupported decode wave geometry")
    launch = "      hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_decode<DWN, BK_, KS_, TM_, RAD_WPERM, true,"
    grid = "                         dim3(dec_nblk, 1, KS_), dblock,"
    if source.count(launch) != 1 or source.count(grid) != 1:
        raise ValueError("decode dispatch anchor changed")
    selected = f"(TM_ == 1 && RAD_WPERM ? {waves} : DWN)"
    source = source.replace(
        launch,
        f"      hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_decode<{selected}, BK_, KS_, TM_, RAD_WPERM, true,",
    ).replace(
        grid,
        f"                         dim3((N + {selected} * 16 - 1) / ({selected} * 16), 1, KS_), dim3({selected} * 32),",
    )
    if minimum_occupancy is not None:
        if minimum_occupancy not in (2, 4):
            raise ValueError("unsupported occupancy hint")
        anchor = "__global__ __launch_bounds__(DWN * 32) void radiance_mxfp4_fp8_gemm_decode("
        if source.count(anchor) != 1:
            raise ValueError("decode entry point changed")
        source = source.replace(
            anchor,
            f"__attribute__((amdgpu_waves_per_eu({minimum_occupancy})))\n" + anchor,
        )
    return source


def build(source, output, sweep):
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    hip = (source / "radiance_mxfp4_fp8.hip").read_text()
    wrapper = source / "radiance_mxfp4.py"
    original = source / "radiance_mxfp4_fp8.so"
    report = {
        "schema": "coherence-mxfp4-register-decode-v1",
        "status": "BUILT_UNTESTED",
        "builder_sha256": digest(__file__),
        "parent": {
            p.name: digest(p)
            for p in (wrapper, original, source / "radiance_mxfp4_fp8.hip")
        },
        "variants": {},
    }
    includes = shlex.split(
        subprocess.check_output(
            [sys.executable, "-m", "pybind11", "--includes"], text=True
        )
    )
    variants = (
        (
            ("control", hip),
            ("register_a", patched_source(hip, direct_a=True)),
            ("register_a_w4", patched_source(hip, direct_a=True, waves=4)),
            ("register_w4", patched_source(hip, waves=4)),
        )
        if sweep == 2
        else (
            ("control", hip),
            ("lut_lds", tune_decode_storage(hip, lds_table=True)),
            ("ks_interleave", tune_decode_storage(hip, split_interleave=True)),
            (
                "lut_ks_interleave",
                tune_decode_storage(hip, lds_table=True, split_interleave=True),
            ),
            (
                "register_lut",
                tune_decode_storage(patched_source(hip, direct_a=True), lds_table=True),
            ),
            (
                "register_slab256",
                tune_decode_storage(patched_source(hip, direct_a=True), slab_factor=2),
            ),
            (
                "register_slab512",
                tune_decode_storage(patched_source(hip, direct_a=True), slab_factor=4),
            ),
        )
    )
    if sweep == 4:
        variants = (
            ("control", hip),
            ("lds_pad16", paired_lds_source(hip, pair_loads=False)),
            ("paired_lds", paired_lds_source(hip)),
            (
                "paired_lds_slab256",
                tune_decode_storage(paired_lds_source(hip), slab_factor=2),
            ),
        )
    if sweep == 5:
        variants = (
            ("control", hip),
            ("local_barrier", local_decode_barriers(hip)),
            (
                "local_register",
                local_decode_barriers(patched_source(hip, direct_a=True)),
            ),
            ("local_w16", local_decode_barriers(patched_source(hip, waves=16))),
            ("local_paired", local_decode_barriers(paired_lds_source(hip))),
            (
                "local_slab256",
                local_decode_barriers(
                    tune_decode_storage(
                        patched_source(hip, direct_a=True), slab_factor=2
                    )
                ),
            ),
        )
    if sweep == 6:
        variants = (
            ("control", hip),
            ("staged_w4", decode_wave_geometry(hip, 4)),
            ("staged_w2", decode_wave_geometry(hip, 2)),
            ("staged_w1", decode_wave_geometry(hip, 1)),
            (
                "register_w2",
                decode_wave_geometry(patched_source(hip, direct_a=True), 2),
            ),
            (
                "register_w1",
                decode_wave_geometry(patched_source(hip, direct_a=True), 1),
            ),
            (
                "register_w2_occ4",
                decode_wave_geometry(
                    patched_source(hip, direct_a=True), 2, minimum_occupancy=4
                ),
            ),
            (
                "register_w1_occ4",
                decode_wave_geometry(
                    patched_source(hip, direct_a=True), 1, minimum_occupancy=4
                ),
            ),
        )
    if sweep == 7:
        variants = (
            ("control", hip),
            ("pipeline_w8", pipeline_register_slabs(hip)),
            (
                "pipeline_w8_sched",
                pipeline_register_slabs(hip, scheduling_barrier=True),
            ),
            ("pipeline_w4", decode_wave_geometry(pipeline_register_slabs(hip), 4)),
            (
                "pipeline_w4_sched",
                decode_wave_geometry(
                    pipeline_register_slabs(hip, scheduling_barrier=True), 4
                ),
            ),
        )
    if sweep == 8:
        variants = (
            ("control", hip),
            ("arithmetic_tables", arithmetic_fold_tables(hip)),
            (
                "arithmetic_register",
                arithmetic_fold_tables(patched_source(hip, direct_a=True)),
            ),
            (
                "arithmetic_register_w4",
                arithmetic_fold_tables(patched_source(hip, direct_a=True, waves=4)),
            ),
            (
                "arithmetic_pipeline_w4",
                arithmetic_fold_tables(
                    decode_wave_geometry(pipeline_register_slabs(hip), 4)
                ),
            ),
        )
    if sweep == 9:
        variants = (
            ("control", hip),
            ("local_split", local_split_reduction(hip)),
            (
                "local_split_arithmetic",
                arithmetic_fold_tables(local_split_reduction(hip)),
            ),
        )
    if sweep == 10:
        variants = (
            ("control", hip),
            ("staged_pipeline_local", pipelined_staged_weights(hip)),
            (
                "staged_pipeline_full",
                pipelined_staged_weights(hip, local_barrier=False),
            ),
            (
                "staged_pipeline_w4",
                decode_wave_geometry(pipelined_staged_weights(hip), 4),
            ),
        )
    for name, native in variants:
        directory = output / name
        directory.mkdir()
        local = directory / "radiance_mxfp4_fp8.hip"
        local.write_text(native)
        (directory / wrapper.name).write_bytes(wrapper.read_bytes())
        binary = directory / original.name
        command = [
            "/opt/rocm/bin/hipcc",
            "-O3",
            "-std=c++17",
            "-fPIC",
            "-shared",
            "--offload-arch=gfx1201",
            "-Wno-unused-result",
            *includes,
            str(local),
            "-o",
            str(binary),
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=900, check=False
        )
        (directory / "compiler.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
        report["variants"][name] = {
            "command": command,
            "source_sha256": digest(local),
            "binary_sha256": digest(binary),
            "python_sha256": digest(directory / wrapper.name),
        }
        (output / "build.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"built": name, **report["variants"][name]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sweep", type=int, choices=(2, 3, 4, 5, 6, 7, 8, 9, 10), default=2
    )
    args = parser.parse_args()
    build(args.source, args.output, args.sweep)
