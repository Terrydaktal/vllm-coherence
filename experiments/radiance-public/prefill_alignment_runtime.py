"""Qualified, bounded-workspace prefill using the decode arithmetic contract.

This repairs numerical transitions, not cache serialization. The deployment
contract must change so an old, numerically different prefix is not reused.
"""

import functools
import hashlib
import json
import math
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import authenticate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(entry):
    """A successful sample is scoped evidence, never a universal proof."""
    path = Path(entry["qualification"])
    if digest(path) != entry["qualification_sha256"]:
        raise ValueError("prefill alignment qualification changed")
    report = json.loads(path.read_text())
    authenticate(report)
    if (
        report.get("status") != "SAMPLE_CHECKED"
        or report.get("schema") != "urn:coherence:prefill-alignment:v1"
    ):
        raise ValueError("prefill alignment has no successful qualification")
    for name, expected in entry["sources"].items():
        if (
            Path(name).name != name
            or digest(Path(__file__).with_name(name)) != expected
        ):
            raise ValueError("prefill alignment runtime source changed")
    if report.get("sources") != entry["sources"]:
        raise ValueError("prefill qualification covers different runtime sources")
    for name in ("attention", "projection"):
        build = json.loads((Path(entry[name]["build"]) / "build.json").read_text())
        authenticate(build)
        if (
            build["sha256"] != entry[name]["build_sha256"]
            or report["builds"][name] != build["sha256"]
        ):
            raise ValueError("prefill qualification covers a different native build")
    cases = report["attention_cases"]
    if (
        len(cases) < 11
        or not all(
            c["differences"] == 0 and c["elements"] > 0 and c["finite"] for c in cases
        )
        or max(c["context"] for c in cases) < 253792
    ):
        raise ValueError("prefill attention matrix incomplete or divergent")
    cases = report["projection_cases"]
    if len(cases) < 10 or not all(
        c["different"] == 0 and c["elements"] > 0 for c in cases
    ):
        raise ValueError("prefill projection matrix incomplete or divergent")
    models = report["model_comparisons"]
    if not any(c["first_position"] >= 200000 and c["positions"] >= 320 for c in models):
        raise ValueError("missing long-context model comparison")
    if not any(
        60000 <= c["first_position"] < 200000 and c["positions"] >= 1000 for c in models
    ):
        raise ValueError("missing 60K model comparison")
    for case in models:
        n = case["positions"]
        if (
            n < 1
            or case["hidden_exact_rows"] != n
            or case["hidden_different_elements"] != 0
            or case["logits"]["full_logits_exact"] != n
            or any(
                case["logits"][str(k)][field] != n
                for k in (1, 10, 20)
                for field in ("set_exact", "ranked_exact")
            )
        ):
            raise ValueError("prefill model comparison diverged")
    return report


def prefill_admitted(impl, md, output_scale=None, output_block_scale=None):
    plan = getattr(md, "r4d_plan", ())
    return (
        len(plan) == 1
        and tuple(plan[0][:2]) == (0, 1)
        and plan[0][3] == 0
        and 8 < plan[0][2] <= 2048
        and md.causal is True
        and plan[0][2] <= md.r4d_max_ctx <= 253792
        and (impl.num_heads, impl.num_kv_heads, impl.head_size) == (24, 4, 256)
        and impl.scale == 256**-0.5
        and output_scale is None
        and output_block_scale is None
    )


def install(entry, hooks, *, verify=True):
    import torch
    from prefill_attention_alignment import AlignedPrefillAttention
    from prefill_gemm_alignment import AlignedPrefillGemm

    import radiance_mxfp4 as gemm
    import radiance_r4d_attn as native

    if verify:
        validate(entry)
    attention = AlignedPrefillAttention(entry["attention"]["build"])
    projection = AlignedPrefillGemm(entry["projection"]["build"])
    # Allocate lazily on the first prefill. The single admitted sequence uses
    # its current stream for both scratch writes and reads.
    scratch = None
    calls = {"attention_prefill": 0, "projection_prefill": 0}
    original_attention = native.R4DAttentionImpl.forward

    @functools.wraps(original_attention)
    def forward(
        impl,
        layer,
        query,
        key,
        value,
        kv,
        md,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        nonlocal scratch
        plan = getattr(md, "r4d_plan", ())
        if not plan or all(row[2] <= 8 for row in plan):
            return original_attention(
                impl,
                layer,
                query,
                key,
                value,
                kv,
                md,
                output,
                output_scale,
                output_block_scale,
            )
        if not prefill_admitted(impl, md, output_scale, output_block_scale):
            raise ValueError("prefill batch outside the qualified arithmetic contract")
        width = plan[0][2]
        if scratch is None:
            scratch = torch.empty(
                attention.manifest["scratch_bytes"],
                device=query.device,
                dtype=torch.uint8,
            )
        scales = []
        for name in ("_k_scale_float", "_v_scale_float"):
            scale = float(getattr(layer, name, 1.0))
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("invalid attention storage scale")
            scales.append(
                None
                if scale == 1.0
                else torch.full((4,), scale, device=query.device, dtype=torch.float32)
            )
        attention(
            query[:width],
            kv,
            md.block_table[:1],
            md.seq_lens[:1],
            scratch,
            output[:width],
            ks=scales[0],
            vs=scales[1],
        )
        calls["attention_prefill"] += 1
        return output

    hooks.replace(native.R4DAttentionImpl, "forward", forward)
    op = gemm.mxfp4_linear_pq
    previous = op._backend_fns.get("cuda", op._init_fn)
    hooks.replace(op, "_backend_fns", dict(op._backend_fns))

    @op.register_kernel("cuda")
    def aligned(q, scale, weight, weight_scale, ref):
        if q.shape[0] > 8 and weight.shape[0] == 5120 and ref.numel() == 5120:
            calls["projection_prefill"] += 1
            return projection(
                q,
                scale,
                weight,
                weight_scale,
                ref,
                tiled=q.shape[0] >= 64,
                wperm=gemm.WPERM,
            )
        return previous(q, scale, weight, weight_scale, ref)

    return {
        "attention_build": attention.manifest["sha256"],
        "projection_build": projection.manifest["sha256"],
        "workspace_bytes": attention.manifest["scratch_bytes"]
        + projection.manifest["scratch_bytes"],
        "calls": calls,
        "scope": "TP1, one causal sequence, at most 2048 prefill rows, existing decode arithmetic",
    }
