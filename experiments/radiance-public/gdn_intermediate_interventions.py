"""Controlled cross-feeding of native GDN intermediates, for diagnosis only."""

from __future__ import annotations

import ast

from gdn_intermediate_trace import OFFSETS, replace_once, stock_source
from probe_gdn_beta_precision import KERNEL

STAGES = ("qk", "decay", "prediction", "residual", "state", "output")


def intervention_source(original: bytes, through: str) -> bytes:
    if through not in STAGES:
        raise ValueError("unknown GDN intervention boundary")
    source = stock_source(original, beta_fp32=True).decode()
    node = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == KERNEL
    )
    lines = source.splitlines(keepends=True)
    body = "".join(lines[node.lineno - 1 : node.end_lineno])
    body = replace_once(
        body, "    trace,\n", "    trace,\n    reference_trace,\n    reference_state,\n"
    )
    enabled = STAGES[: STAGES.index(through) + 1]
    body = replace_once(
        body,
        "    b_q = b_q * scale\n",
        "    b_q = b_q * scale\n"
        + f"    b_q = tl.load(reference_trace + {OFFSETS['query']} + i_hv * K + o_k,"
        " mask=mask_k, other=0)\n"
        + f"    b_k = tl.load(reference_trace + {OFFSETS['key']} + i_hv * K + o_k,"
        " mask=mask_k, other=0)\n",
    )
    if "decay" in enabled:
        expression = f"tl.load(reference_trace + {OFFSETS['decay']} + i_hv)"
        # Change both the observed value and the value used by the transition.
        body = replace_once(body, "+ i_hv, exp(g_val))", f"+ i_hv, {expression})")
        body = replace_once(body, "    b_h *= exp(g_val)\n", f"    b_h *= {expression}\n")
    if "prediction" in enabled:
        expression = (
            f"tl.load(reference_trace + {OFFSETS['prediction']} + i_hv * V + o_v,"
            " mask=mask_v, other=0)"
        )
        if body.count("tl.sum(b_h * b_k[None, :], 1)") != 2:
            raise ValueError("ambiguous prediction intervention")
        body = body.replace("tl.sum(b_h * b_k[None, :], 1)", expression)
    if "residual" in enabled:
        body = replace_once(
            body,
            "    b_v *= beta_val\n",
            "    b_v *= beta_val\n"
            + f"    b_v = tl.load(reference_trace + {OFFSETS['residual']} + i_hv * V + o_v,"
            " mask=mask_v, other=0)\n",
        )
    if "state" in enabled:
        body = replace_once(
            body,
            "    b_h += b_v[:, None] * b_k[None, :]\n",
            "    b_h += b_v[:, None] * b_k[None, :]\n"
            "    b_h = tl.load(reference_state + i_hv * V * K + o_v[:, None] * K"
            " + o_k[None, :], mask=mask_h, other=0)\n",
        )
    if "output" in enabled:
        body = replace_once(
            body,
            "    b_o = tl.sum(b_h * b_q[None, :], 1)\n",
            "    b_o = tl.sum(b_h * b_q[None, :], 1)\n"
            + f"    b_o = tl.load(reference_trace + {OFFSETS['output_fp32']} + i_hv * V + o_v,"
            " mask=mask_v, other=0)\n",
        )
    result = "".join(lines[: node.lineno - 1]) + body + "".join(lines[node.end_lineno :])
    compile(result, "gdn_intervention.py", "exec")
    return result.encode()
