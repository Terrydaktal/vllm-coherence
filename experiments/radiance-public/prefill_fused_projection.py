"""Experimental ordered split-K prefill projection without global partials.

Each output still uses the four decode K ranges and four ordered FP32 additions.
Only the lifetime/location of the intermediate partials changes. This builder is
not a serving admission: native and whole-model qualification remain required.
"""

from pathlib import Path

import prefill_gemm_alignment as aligned


def source_for_fused(source, tn=2, slab=128, tm=4):
    if tm not in (1, 2, 4):
        raise ValueError("unsupported row tile")
    generated = aligned.source_for_prefill(source, 2048, tn, slab)
    for name, step, epilogue in (
        ("radiance_mxfp4_fp8_gemm_folded", "BK", "  float rf[TN];"),
        ("radiance_mxfp4_fp8_gemm_atiled", "LBK", "  // Epilogue: identical"),
    ):
        begin = source.index("void " + name + "(")
        end = source.index("\n}\n", begin) + 2
        original = source[begin:end]
        acc = original.index("  floatx8 acc[TM][TN];")
        tail = original.index(epilogue)
        compute = original[acc:tail]
        loop = f"  for (int k0 = 0; k0 < K; k0 += {step}) {{"
        if compute.count(loop) != 1:
            raise ValueError("parent projection loop changed")
        compute = compute.replace(
            loop,
            f"  const int first_k = split * span, end_k = min(K, first_k + span);\n"
            f"  for (int k0 = first_k; k0 < end_k; k0 += {step}) {{",
        )
        fused = (
            original[:acc]
            + """
  floatx8 merged[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j)
      #pragma unroll
      for (int e = 0; e < 8; ++e) merged[i][j][e] = 0.0f;
  const int span = ((K / 128 + 3) / 4) * 128;
  #pragma unroll 1
  for (int split = 0; split < 4; ++split) {
"""
            + compute
            + """
    #pragma unroll
    for (int i = 0; i < TM; ++i)
      #pragma unroll
      for (int j = 0; j < TN; ++j)
        #pragma unroll
        for (int e = 0; e < 8; ++e) merged[i][j][e] += acc[i][j][e];
  }
"""
            + original[tail:].replace("acc[", "merged[")
        )
        start = generated.index("void " + name + "(")
        stop = generated.index("\n}\n", start) + 2
        generated = generated[:start] + fused + generated[stop:]
    generated = generated.replace("#define TM 4", f"#define TM {tm}")
    generated = generated.replace("/ BMF, 4);", "/ BMF, 1);")
    generated = generated.replace(
        ">(a,w,ws,ref,scale,partials,M,N,K);",
        ">(a,w,ws,ref,scale,out,M,N,K);",
    )
    generated = generated.replace(
        "  coherence_prefill_reduce<<<dim3((size_t(M)*N+255)/256),dim3(256),0,stream>>>(partials,ref,scale,out,M,N);\n",
        "",
    )
    return generated


def build(parent, output, expected_sha256, tn=2, slab=128, tm=4):
    # Keep the checked ABI/scratch reservation for the isolated comparison. The
    # experimental kernel does not dereference the partials allocation.
    original = aligned.source_for_prefill
    try:
        aligned.source_for_prefill = lambda source, *_: source_for_fused_original(
            source, original, tn, slab, tm
        )
        return aligned.build(
            Path(parent), Path(output), expected_sha256, 2048, tn, slab
        )
    finally:
        aligned.source_for_prefill = original


def source_for_fused_original(source, generator, tn, slab, tm):
    current = aligned.source_for_prefill
    try:
        aligned.source_for_prefill = generator
        return source_for_fused(source, tn, slab, tm)
    finally:
        aligned.source_for_prefill = current
