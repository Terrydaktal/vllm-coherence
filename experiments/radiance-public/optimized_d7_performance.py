"""Stage-qualified performance paths layered over the pinned D7 repairs.

Unsupported shapes retain the repaired implementation. Qualification receipts
bind the exact binaries; sample agreement is not a universal equivalence proof.
"""

import functools
import hashlib
from collections import Counter
from pathlib import Path

from qwen_r9700_lab.conformance_instrumentation import HookSet
from qwen_r9700_lab.conformance_topk import require
from qwen_r9700_lab.diagnostic_contract import authenticate, private_json


def qualified_stage(entry):
    build = private_json(Path(entry["build"]) / "build.json")
    evidence = private_json(Path(entry["qualification"]) / "result.json")
    authenticate(build)
    authenticate(evidence)
    require(build["sha256"] == entry["build_sha256"], "performance build changed")
    require(evidence["sha256"] == entry["qualification_sha256"], "stage evidence changed")
    require(evidence["status"] == "SAMPLE_CHECKED", "stage has no successful sample check")
    # Both probes publish the authenticated build seal in their result.
    require(evidence["build"] == build["sha256"], "stage evidence covers a different build")
    return build


def shared_attention_admitted(impl, md, output_scale, output_block_scale):
    """Host metadata gate; no tensor reads/synchronization on the decode path."""
    return (
        getattr(md, "r4d_plan", None) == ((0, 1, 8, 0),)
        and md.causal is True
        and md.r4d_max_ctx >= 1031
        and (impl.num_heads, impl.num_kv_heads, impl.head_size) == (24, 4, 256)
        and impl.scale == 256**-0.5
        and output_scale is None
        and output_block_scale is None
    )


class PerformanceRepairs:
    def __init__(self, path, model, repairs):
        import radiance_r4d_attn as native
        from stock_m1_attention_shared import SharedM1Attention
        from stock_m1_head_pair import StockM1HeadPair

        self.manifest = private_json(Path(path))
        authenticate(self.manifest)
        if self.manifest.get("tp1_fp8") or self.manifest.get("lazy_gdn"):
            from tp1_lazy_backports import validate_runtime_layout

            validate_runtime_layout(self.manifest)
        require(
            self.manifest["reference_repair"] == repairs.manifest["sha256"],
            "performance paths require their qualified numerical reference",
        )
        for name, digest in self.manifest["sources"].items():
            require(
                hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest,
                "performance adapter source changed",
            )
        for entry in self.manifest["stages"].values():
            qualified_stage(entry)
        self.gemm_dispatch = None
        if self.manifest.get("gemm_dispatch"):
            from mxfp4_dispatch import install

            self.gemm_dispatch = install(self.manifest["gemm_dispatch"])
        self.head = StockM1HeadPair(self.manifest["stages"]["head"]["build"])
        self.attention = SharedM1Attention(self.manifest["stages"]["attention"]["build"])
        self.hooks = HookSet()
        self.calls = Counter()
        self.lazy_gdn = None
        self.fp8_stream = None
        self.gdn_norm_quant = None
        self.prefill_scan = None
        self.activation_tiles = None
        self.repairs = repairs
        heads = [m for n, m in model.named_modules() if n.endswith("logits_processor")]
        require(len(heads) == 1, "target head module changed")
        original_head = heads[0]._apply_head

        @functools.wraps(original_head)
        def head(lm_head, hidden, embedding_bias=None):
            if hidden.shape == (8, 5120) and lm_head.tp_size == 1 and embedding_bias is None:
                self.calls["head_m8"] += 1
                return self.head(lm_head.weight, hidden)
            self.calls["head_fallback"] += 1
            return original_head(lm_head, hidden, embedding_bias)

        self.hooks.replace(heads[0], "_apply_head", head)
        original_attention = native.R4DAttentionImpl.forward

        @functools.wraps(original_attention)
        def attention(
            impl,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale=None,
            output_block_scale=None,
        ):
            md = attn_metadata
            if not shared_attention_admitted(impl, md, output_scale, output_block_scale):
                self.calls["attention_fallback"] += 1
                return original_attention(
                    impl,
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    md,
                    output,
                    output_scale,
                    output_block_scale,
                )
            import torch

            scales = []
            for name in ("_k_scale_float", "_v_scale_float"):
                value = float(getattr(layer, name, 1.0))
                scales.append(
                    None
                    if value == 1.0
                    else torch.full(
                        (4,),
                        value,
                        device=query.device,
                        dtype=torch.float32,
                    )
                )
            self.calls["attention_m8"] += 1
            return self.attention(
                query,
                kv_cache,
                md.block_table[:1],
                md.seq_lens[:1],
                md.r4d_scratch,
                out=output,
                ks=scales[0],
                vs=scales[1],
                max_ctx=md.r4d_max_ctx,
            )

        self.hooks.replace(native.R4DAttentionImpl, "forward", attention)
        transport = self.manifest["stages"].get("gdn_transport")
        if transport:
            import torch
            from packed_gdn_transport import load

            conv, recur = load(transport["build"], torch, repairs.convolution.update)
            gdn = repairs.prefill.native
            original_conv, original_recur = gdn.conv_update, gdn.recurrent_update

            @functools.wraps(original_conv)
            def convolution(*args, **kwargs):
                rows = args[9] if len(args) > 9 else kwargs.get("tokens")
                sequences = args[8] if len(args) > 8 else kwargs.get("num_seqs")
                if rows == 8 and sequences == 1:
                    self.calls["packed_convolution_m8"] += 1
                    return conv(*args, **kwargs)
                return original_conv(*args, **kwargs)

            @functools.wraps(original_recur)
            def recurrent(*args, **kwargs):
                q = args[0] if args else kwargs["q"]
                sequences = args[12] if len(args) > 12 else kwargs.get("num_seqs")
                if q.shape[0] == 8 and sequences == 1:
                    self.calls["packed_recurrent_m8"] += 1
                    return recur(*args, **kwargs)
                return original_recur(*args, **kwargs)

            self.hooks.replace(gdn, "conv_update", convolution)
            self.hooks.replace(gdn, "recurrent_update", recurrent)
            if self.manifest.get("lazy_gdn"):
                from tp1_lazy_backports import install_lazy

                self.lazy_gdn = install_lazy(self.manifest["lazy_gdn"], repairs, self.hooks, conv)
        elif self.manifest.get("lazy_gdn"):
            raise ValueError("lazy GDN needs its qualified packed convolution")

    def install_producers(self, model):
        """After corrected compiled norm wrappers, before compilation/capture."""
        if self.manifest.get("tp1_fp8"):
            from tp1_lazy_backports import install_fp8

            self.fp8_stream = install_fp8(self.manifest["tp1_fp8"], model, self.hooks)
        if self.manifest.get("gdn_norm_quant"):
            from pi_prefill_admission import gdn_evidence
            from stock_gdn_norm_quant import install

            gdn_evidence(self.manifest["gdn_norm_quant"])
            self.gdn_norm_quant = install(model, self.hooks)
        if self.manifest.get("prefill_scan"):
            from optimized_prefill_scan import PrefillScan
            from pi_prefill_admission import scan_evidence

            scan_evidence(self.manifest["prefill_scan"])
            previous = self.repairs.prefill.scan
            self.hooks.replace(
                self.repairs.prefill, "scan", PrefillScan(previous.torch, previous.device)
            )
            self.prefill_scan = {"spatial_tiles": [8, 16], "chronological_arithmetic": "unchanged"}
        if self.manifest.get("activation_tiles"):
            from prefill_tiles_admission import install

            self.activation_tiles = install(self.manifest["activation_tiles"])

    def receipt(self):
        return {
            "manifest": self.manifest["sha256"],
            "calls": dict(self.calls),
            "gemm_dispatch": self.gemm_dispatch,
            "lazy_gdn": self.lazy_gdn,
            "tp1_fp8": self.fp8_stream,
            "gdn_norm_quant": self.gdn_norm_quant,
            "prefill_scan": self.prefill_scan,
            "activation_tiles": self.activation_tiles,
        }
