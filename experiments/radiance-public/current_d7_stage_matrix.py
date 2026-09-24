"""Current-release common-input stage checks; no historical operator substitutions."""

import re

import torch
from native_d7_stage_matrix import StageMatrix, shared_versions

from qwen_r9700_lab.diagnostic_contract import DiagnosticError


def current_attention_stages():
    from native_d7_attention_stages import AttentionStages

    class CurrentAttentionStages(AttentionStages):
        def fixed(self, variant, name, cut):
            if variant == "fix1_m1" or variant.endswith("eager_m8"):
                width = 1 if variant == "fix1_m1" else 8
                qn, kn = self.normalized(cut, width)
                cos, sin = cut["rotations"][0][1][1:3]
                query, key = [], []
                for start in range(0, 8, width):
                    q, k = self.rne.triton_mrope(
                        qn[start:start + width].reshape(width, -1),
                        kn[start:start + width].reshape(width, -1),
                        torch.stack([cos[start:start + width]] * 3),
                        torch.stack([sin[start:start + width]] * 3),
                        [11, 11, 10], 256, 64, True, True,
                    )
                    query.append(q.reshape(width, 24, 256))
                    key.append(k.reshape(width, 4, 256))
                return torch.cat(query), torch.cat(key)
            return super().fixed("final_m8", name, cut)

    return CurrentAttentionStages(None)


def serial_rows(function, args, kwargs):
    def sliced(value, i):
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 8:
            return value[i : i + 1]
        return value

    outputs = [function(*(sliced(x, i) for x in args),
                        **{k: sliced(v, i) for k, v in kwargs.items()}) for i in range(8)]

    def join(values):
        if all(v is None for v in values):
            return None
        if isinstance(values[0], torch.Tensor):
            return torch.cat(values)
        if isinstance(values[0], tuple):
            return tuple(join([v[i] for v in values]) for i in range(len(values[0])))
        raise DiagnosticError("unsupported current row-operator result")

    return join(outputs)


class CurrentStageMatrix(StageMatrix):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.history is not None:
            raise DiagnosticError("current confirmation cannot use historical catalogs")
        if self.attention_stages is not None:
            self.attention_stages = current_attention_stages()

    def capture_filter(self, name, args, kwargs):
        return (name.startswith(("qwen_stock_fp8.", "qwen_stock_gdn_fp8.",
                                 "radiance.mxfp4_linear"))
                or name == "_C.dynamic_per_token_scaled_fp8_quant.default"
                or super().capture_filter(name, args, kwargs))

    def current_cases(self, call):
        args, kwargs = call.cut[0][0].thaw(call.cut[0])
        fn = call.function
        if call.name.startswith("radiance.mxfp4_linear"):
            prequantized = "linear_pq" in call.name
            weight = args[2] if prequantized else args[1]
            owners = self.parameters.get(weight.untyped_storage().data_ptr(), [])
            if len(owners) != 1:
                raise DiagnosticError("current projection owner is ambiguous")
            match = re.search(r"layers\.(\d+)\.(.*)", owners[0])
            if match is None:
                raise DiagnosticError("target projection layer is missing")
            instance, role = match.groups()
            if "gate_up_proj" in role:
                stage = "MLP gate/up projection"
            elif "down_proj" in role:
                stage = "MLP down projection"
            elif "linear_attn" in role:
                stage = "GDN output projection" if "out_proj" in role else "GDN input projection"
            elif "self_attn" in role:
                stage = "Attention input projection" if "qkv_proj" in role else "Attention output projection"
            else:
                raise DiagnosticError("unknown current projection role")
            yield stage, instance, shared_versions(fn, lambda *a, **kw: serial_rows(fn, a, kw))
        elif call.name == "qwen_stock_fp8.norm.default":
            layer, role = args[-1].split("/")
            stage = ("Post-attention/GDN residual/normalization + FP8 production"
                     if role == "post_attention_layernorm" else
                     "Embedding + first input normalization + FP8 production" if layer == "0" else
                     "Layer input residual/normalization + FP8 production")
            yield stage, layer, shared_versions(fn, lambda *a, **kw: serial_rows(fn, a, kw))
        elif call.name == "qwen_stock_gdn_fp8.norm_quant.default":
            owners = self.parameters.get(args[2].untyped_storage().data_ptr(), [])
            if len(owners) != 1:
                raise DiagnosticError("current GDN norm owner is ambiguous")
            layer = re.search(r"layers\.(\d+)\.", owners[0]).group(1)
            yield "GDN output gated normalization + FP8 production", layer, shared_versions(
                fn, lambda *a, **kw: serial_rows(fn, a, kw))
        elif call.name == "target.full_bf16_head":
            yield "Full BF16 comparison head", "head", shared_versions(
                fn, lambda *a, **kw: serial_rows(fn, a, kw))
        elif call.name == "_C.dynamic_per_token_scaled_fp8_quant.default":
            if len(args) != 4 or args[3] is not None:
                raise DiagnosticError("unqualified dynamic FP8 quantization boundary")
            shape = tuple(args[1].shape)
            if shape == (8, 17408):
                stage, instance = "MLP down input FP8 quantization", str(self.silu_index - 1)
            elif shape == (8, 6144):
                stage, instance = "Attention output activation FP8 quantization", str((self.sigmoid_index - 1) * 4 + 3)
            else:
                raise DiagnosticError("unassigned current FP8 quantization geometry")
            yield stage, instance, shared_versions(fn, lambda *a, **kw: serial_rows(fn, a, kw))
        elif "silu_slice" in call.name:
            if (len(args) != 3 or tuple(args[0].shape) != (8, 34816)
                    or args[1].numel() != 8 * 17408 or args[2] != 8 * 17408):
                raise DiagnosticError("compiled SiLU boundary changed")
            def eager(x, out, numel, **launch):
                out.copy_((torch.nn.functional.silu(x[:, :17408]) * x[:, 17408:]).reshape_as(out))
            versions = shared_versions(fn, eager)
            versions["final_eager_m8"] = eager
            layer = str(self.silu_index)
            self.silu_index += 1
            yield "MLP SiLU and gating", layer, versions
        elif "sigmoid_view" in call.name:
            if (len(args) != 4 or args[0].numel() != 8 * 6144
                    or args[1].numel() != 8 * 6144 or args[2].numel() != 8 * 6144
                    or args[3] != 8 * 6144):
                raise DiagnosticError("compiled attention gating boundary changed")
            def eager(x, gate, out, numel, **launch):
                out.copy_((x.reshape(8, 6144) * torch.sigmoid(gate.reshape(8, 6144))).reshape_as(out))
            versions = shared_versions(fn, eager)
            versions["final_eager_m8"] = eager
            layer = str(self.sigmoid_index * 4 + 3)
            self.sigmoid_index += 1
            yield "Attention output gating", layer, versions
        else:
            yield from super().cut_cases(call)

    def cut_cases(self, call):
        for stage, instance, versions in self.current_cases(call):
            # Retain current M1/M8 and current eager/compiled comparisons only.
            yield stage, instance, {k: v for k, v in versions.items()
                                    if k in {"fix1_m1", "fix1_m8", "final_eager_m8", "final_m8"}}
