"""Generate isolated tracing copies of the two reviewed GDN transitions.

These expose intermediate values, not new model semantics. Native replay must
first show that tracing preserves the untraced output and persistent state.
Importing this module uses neither Torch nor a GPU.
"""

from __future__ import annotations

import ast
import hashlib

from probe_gdn_beta_precision import KERNEL, SOURCE_SHA256, ablation_source

R4D_SOURCE = "0319d41dc89617cb1bfa3aba21be2b8dad7f56b39f80dce7287dbe6e4ef56c42"
SHAPES = {
    "query": (48, 128),
    "key": (48, 128),
    "gate": (48,),
    "beta": (48,),
    "decay": (48,),
    "decayed_state": (48, 128, 128),
    "prediction": (48, 128),
    "residual": (48, 128),
    "output_fp32": (48, 128),
}
OFFSETS = {}
ELEMENTS = 0
for _name, _shape in SHAPES.items():
    OFFSETS[_name] = ELEMENTS
    _size = 1
    for _dimension in _shape:
        _size *= _dimension
    ELEMENTS += _size


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("trace source anchor is missing or ambiguous")
    return text.replace(before, after, 1)


def stock_source(source: bytes, *, beta_fp32=False) -> bytes:
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("unreviewed stock trace source")
    text = (ablation_source(source) if beta_fp32 else source).decode()
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == KERNEL]
    if len(nodes) != 1:
        raise ValueError("ambiguous stock kernel")
    node = nodes[0]
    lines = text.splitlines(keepends=True)
    body = "".join(lines[node.lineno - 1 : node.end_lineno])
    body = replace_once(body, "    mixed_qkv,\n", "    trace,\n    mixed_qkv,\n")
    body = replace_once(
        body,
        "    b_q = b_q * scale\n",
        "    b_q = b_q * scale\n"
        + f"""
    if i_v == 0:
        tl.store(trace + {OFFSETS["query"]} + i_hv * K + o_k, b_q, mask=mask_k)
        tl.store(trace + {OFFSETS["key"]} + i_hv * K + o_k, b_k, mask=mask_k)
""",
    )
    anchor = "    b_h *= exp(g_val)\n"
    body = replace_once(
        body,
        anchor,
        f"""    if i_v == 0:
        tl.store(trace + {OFFSETS["gate"]} + i_hv, g_val)
        tl.store(trace + {OFFSETS["beta"]} + i_hv, beta_val)
        tl.store(trace + {OFFSETS["decay"]} + i_hv, exp(g_val))
"""
        + anchor
        + f"""    tl.store(trace + {OFFSETS["decayed_state"]} + i_hv * V * K
             + o_v[:, None] * K + o_k[None, :], b_h, mask=mask_h)
    tl.store(trace + {OFFSETS["prediction"]} + i_hv * V + o_v,
             tl.sum(b_h * b_k[None, :], 1), mask=mask_v)
""",
    )
    anchor = "    b_v *= beta_val\n"
    body = replace_once(
        body,
        anchor,
        anchor
        + f"    tl.store(trace + {OFFSETS['residual']} + i_hv * V + o_v, b_v, mask=mask_v)\n",
    )
    anchor = "    b_o = tl.sum(b_h * b_q[None, :], 1)\n"
    body = replace_once(
        body,
        anchor,
        anchor
        + f"    tl.store(trace + {OFFSETS['output_fp32']} + i_hv * V + o_v, b_o, mask=mask_v)\n",
    )
    result = "".join(lines[: node.lineno - 1]) + body + "".join(lines[node.end_lineno :])
    compile(result, "gdn_stock_trace.py", "exec")
    return result.encode()


def r4d_source(source: bytes, *, traced: bool) -> bytes:
    if hashlib.sha256(source).hexdigest() != R4D_SOURCE:
        raise ValueError("unreviewed R4D trace source")
    text = source.decode()
    text = replace_once(
        text,
        'extern "C" int r4d_gdn_recurrent_update_k128_v128_bf16_fp32state(',
        'extern "C" int qwen_gdn_recurrent_diagnostic(',
    )
    if not traced:
        return text.encode()
    text = replace_once(
        text,
        "int H, int Hg, float scale, float sp_thr)",
        "int H, int Hg, float scale, float sp_thr, float* trace)",
    )
    text = replace_once(
        text,
        "int N, int H, int Hg, int K, int V, float scale, float softplus_thr, void* stream)",
        "int N, int H, int Hg, int K, int V, float scale, float softplus_thr, "
        "void* trace, void* stream)",
    )
    text = replace_once(
        text,
        "norm_act, H, Hg, scale, softplus_thr)",
        "norm_act, H, Hg, scale, softplus_thr, (float*)trace)",
    )
    # This diagnostic admits precisely one unfused token/sequence with 48/16
    # heads. Reject other shapes rather than writing past its fixed trace.
    text = replace_once(
        text,
        "  if (K != RU_K || V != RU_V) return -1;",
        "  if (N != 1 || H != 48 || Hg != 16 || z_gate || norm_weight || !trace) return -3;\n"
        "  if (K != RU_K || V != RU_V) return -1;",
    )
    text = replace_once(text, "  if (T <= 0) return;", "  if (T != 1) return;")
    anchor = "    for (int j = 0; j < RU_HK; ++j) { qq[j] *= qs; kk[j] *= ks; }\n"
    text = replace_once(
        text,
        anchor,
        anchor
        + f"""    if (row == 0) {{
#pragma unroll
      for (int j = 0; j < RU_HK; ++j) {{
        trace[{OFFSETS["query"]} + hv * RU_K + k0 + j] = qq[j];
        trace[{OFFSETS["key"]} + hv * RU_K + k0 + j] = kk[j];
      }}
    }}
""",
    )
    anchor = "    const float eg  = __expf(g);\n"
    text = replace_once(
        text,
        anchor,
        anchor
        + f"""    if (row == 0 && half == 0) {{
      trace[{OFFSETS["gate"]} + hv] = g;
      trace[{OFFSETS["beta"]} + hv] = bt;
      trace[{OFFSETS["decay"]} + hv] = eg;
    }}
""",
    )
    anchor = "    dot += __shfl_xor(dot, 1, 32);\n"
    text = replace_once(
        text,
        anchor,
        anchor
        + f"""#pragma unroll
    for (int j = 0; j < RU_HK; ++j)
      trace[{OFFSETS["decayed_state"]} + hv * RU_V * RU_K + row * RU_K + k0 + j] = h[j];
    if (half == 0) trace[{OFFSETS["prediction"]} + hv * RU_V + row] = dot;
""",
    )
    anchor = "    const float u = (bf2f(v[((size_t)tok * H + hv) * RU_V + row]) - dot) * bt;\n"
    text = replace_once(
        text,
        anchor,
        anchor + f"    if (half == 0) trace[{OFFSETS['residual']} + hv * RU_V + row] = u;\n",
    )
    anchor = "    on += __shfl_xor(on, 1, 32);\n"
    text = replace_once(
        text,
        anchor,
        anchor + f"    if (half == 0) trace[{OFFSETS['output_fp32']} + hv * RU_V + row] = on;\n",
    )
    return text.encode()
