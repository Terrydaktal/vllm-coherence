"""Bounded independent eager-M1 audit of the installed R9700 operator release.

Uses public checkpoint tensors and synthetic inputs only. Each result names its
reference, tolerance, actual operator and tested layer. It does not instantiate a
second model or modify the running worker. An idle metrics check is observational;
the cooperative lease excludes other qualification tools, not future Pi requests.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import m1_stage_oracles as ref
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def target_projection_names(index):
    names = sorted(
        n
        for n in index
        if n.startswith("model.language_model.layers.") and n.endswith(".weight_scale")
    )
    if len(names) != 496:
        raise ValueError(f"expected 496 target projection matrices, found {len(names)}")
    expected = set()
    for layer in range(64):
        prefix = f"model.language_model.layers.{layer}."
        projections = [f"mlp.{part}_proj" for part in ("gate", "up", "down")]
        if layer % 4 == 3:
            projections += [f"self_attn.{part}_proj" for part in ("q", "k", "v", "o")]
        else:
            projections += [
                f"linear_attn.in_proj_{part}" for part in ("qkv", "z", "a", "b")
            ]
            projections += ["linear_attn.out_proj"]
        expected.update(prefix + part + ".weight_scale" for part in projections)
    if set(names) != expected:
        raise ValueError(
            "target projection inventory differs from the pinned 64-layer architecture"
        )
    return names


def result_status(report, requested):
    counts = report.get("stage_counts", {})
    if (
        report["errors"]
        or not requested
        or len(set(requested)) != len(requested)
        or set(counts) != set(requested)
        or any(counts[name] <= 0 for name in requested)
        or sum(counts.values()) != len(report["checks"])
    ):
        return "INCOMPLETE"
    return (
        "SAMPLE_CHECKED"
        if all(r["passed"] for r in report["checks"])
        else "FAILURES_OBSERVED"
    )


class Audit:
    def __init__(self, args):
        import torch
        from probe_eager_m1_independent import idle
        from safetensors import safe_open

        self.args, self.torch, self.safe_open, self.idle = args, torch, safe_open, idle
        self.manifest = json.loads(
            (args.release / "optimized-release.json").read_text()
        )
        self.performance = json.loads(Path(self.manifest["performance"]).read_text())
        self.repair = json.loads(Path(self.manifest["repair"]).read_text())
        for path in reversed(self.manifest["pythonpath"]):
            sys.path.insert(0, path)
        sys.path.append("/patches")
        self.index = json.loads(
            (args.model / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.config = json.loads((args.model / "config.json").read_text())[
            "text_config"
        ]
        self.rng = torch.Generator().manual_seed(args.seed)
        self.report = {
            "schema": "coherence-independent-m1-stage-audit-v1",
            "status": "RUNNING",
            "private_chat_read": False,
            "reference": "independent CPU FP64 equations; explicit exact properties separately",
            "probe_sha256": sha(__file__),
            "oracle_sha256": sha(ref.__file__),
            "release_sha256": sha(args.release / "optimized-release.json"),
            "checkpoint_config_sha256": sha(args.model / "config.json"),
            "checkpoint_index_sha256": sha(args.model / "model.safetensors.index.json"),
            "repair_sha256": sha(self.manifest["repair"]),
            "performance_sha256": sha(self.manifest["performance"]),
            "checkpoint_tensor_bindings": {},
            "seed": args.seed,
            "rows_per_site": args.rows,
            "checks": [],
            "errors": [],
            "sources": {},
            "limits": [
                "Sampled operator evidence; not full-model or universal equivalence.",
                "Tolerances are diagnostic acceptance bounds, not proved forward-error bounds.",
                "Known Global-512 shortlist approximation is not changed by this audit.",
            ],
        }
        self.started = time.monotonic()
        torch.set_num_threads(4)
        torch.set_grad_enabled(False)
        torch.cuda.set_per_process_memory_fraction(
            args.memory_mib * 2**20 / torch.cuda.get_device_properties(0).total_memory
        )

    def check_idle(self):
        if not self.args.backend_stopped:
            self.idle(self.args.api)

    def tensor(self, name, sl=None):
        with self.safe_open(
            self.args.model / self.index[name], framework="pt", device="cpu"
        ) as handle:
            value = (
                handle.get_tensor(name) if sl is None else handle.get_slice(name)[sl]
            )
        key = name if sl is None else f"{name}[{sl.start}:{sl.stop}:{sl.step}]"
        # Bind evidence to the actual public weights, including coefficient/scales
        # read by this process, rather than trusting a model-directory label.
        raw = value.contiguous().view(self.torch.uint8).numpy()
        binding = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(memoryview(raw)).hexdigest(),
        }
        old = self.report["checkpoint_tensor_bindings"].setdefault(key, binding)
        if old != binding:
            raise RuntimeError("checkpoint tensor changed during qualification")
        return value

    def source(self, name, path):
        self.report["sources"][name] = {"name": Path(path).name, "sha256": sha(path)}

    def array(self, x):
        return x.detach().double().cpu().numpy()

    def random(self, shape, amplitude=1):
        return (self.torch.randn(shape, generator=self.rng) * amplitude).bfloat16()

    def record(
        self,
        stage,
        site,
        actual=None,
        expected=None,
        *,
        exact=False,
        relative=0.004,
        absolute=None,
        passed=None,
        **extra,
    ):
        row = {"stage": stage, "site": site, **extra}
        if actual is not None:
            stats = ref.error(self.array(actual), expected)
            row["actual_dtype"] = str(actual.dtype)
            if actual.dtype != self.torch.bfloat16:
                # A recurrent FP32 state is not meant to equal BF16 rounding.
                stats.pop("bf16_mismatches")
            row.update(stats)
            row["criterion"] = {
                "exact": exact,
                "relative_l2": relative if not exact else 0,
                "max_abs": absolute,
            }
            ok = (
                (stats["finite"] and stats["max_abs"] == 0)
                if exact
                else ref.acceptable(stats, relative=relative, absolute=absolute)
            )
        else:
            ok = bool(passed)
        row["passed"] = ok
        self.report["checks"].append(row)
        self.save()
        if not ok:
            print(json.dumps({"failure": row}), flush=True)

    def save(self):
        self.report["elapsed_seconds"] = time.monotonic() - self.started
        tmp = self.args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n")
        os.chmod(tmp, 0o600)
        tmp.replace(self.args.output)

    def norms(self):
        import stock_gdn_norm_quant
        from stock_fp8_epilogue import StockFP8Epilogue
        from stock_m1_gdn_norm import StockM1GdnNorm
        from stock_m1_norm import StockM1Norm
        from vllm import _custom_ops as ops

        t = self.torch
        native = StockM1Norm(self.repair["norm_build"])
        fused = StockFP8Epilogue(self.performance["tp1_fp8"]["build"])
        gated = StockM1GdnNorm()
        self.source("norm_binary", Path(self.repair["norm_build"]) / "candidate.so")
        self.source(
            "fp8_norm_binary",
            Path(self.performance["tp1_fp8"]["build"]) / "candidate.so",
        )
        self.source("gdn_norm", gated.native.__file__)
        self.source("gdn_norm_quant", stock_gdn_norm_quant.__file__)
        names = [
            n
            for n in self.index
            if n.startswith("model.language_model.layers.")
            and n.endswith(".weight")
            and any(
                s in n
                for s in (
                    "input_layernorm",
                    "post_attention_layernorm",
                    "self_attn.q_norm",
                    "self_attn.k_norm",
                )
            )
        ]
        names += ["model.language_model.norm.weight"]
        if len(names) != 161:
            raise ValueError(
                f"expected 161 target normalization sites, found {len(names)}"
            )
        for name in sorted(names):
            self.check_idle()
            weight = self.tensor(name)
            width = weight.numel()
            heads = 24 if "q_norm" in name else 4
            shape = (
                (self.args.rows, width)
                if width == 5120
                else (self.args.rows, heads, width)
            )
            x = self.random(shape)
            x[0].zero_()
            x[1].fill_(2**-20)
            x[2] *= 128
            xg, wg = x.cuda(), weight.cuda()
            expected = ref.rms(
                self.array(x), self.array(weight), self.config["rms_norm_eps"], offset=1
            )
            outputs = t.cat(
                [
                    native(xg[i : i + 1], None, wg, self.config["rms_norm_eps"])
                    for i in range(len(x))
                ]
            )
            self.record(
                "normalization",
                name,
                outputs,
                expected,
                dispatch="StockM1Norm: native M1 hidden or Q/K head geometry",
            )
            if width == 5120:
                residual = self.random(x.shape)
                residual[0].zero_()
                residual[3] = -x[3]
                r = residual.cuda()
                expected_r = ref.rms(
                    self.array(x),
                    self.array(weight),
                    self.config["rms_norm_eps"],
                    residual=self.array(residual),
                    offset=1,
                )
                q, s, carry = fused.norm(xg, r, wg, self.config["rms_norm_eps"])
                # A quantizer adds its explicitly accepted approximation. Assess
                # norm independently through its BF16 counterpart as well.
                actual = t.cat(
                    [
                        native(
                            xg[i : i + 1], r[i : i + 1], wg, self.config["rms_norm_eps"]
                        )[0]
                        for i in range(len(x))
                    ]
                )
                self.record("residual_norm", name, actual, expected_r)
                self.record(
                    "residual_carry",
                    name,
                    carry,
                    ref.bf16(
                        (
                            self.array(x).astype(np.float32)
                            + self.array(residual).astype(np.float32)
                        ).astype(np.float64)
                    ),
                    exact=True,
                )
                eq, es = ref.dynamic_fp8(self.array(actual))
                native_q, native_s = ops.scaled_fp8_quant(
                    actual, scale=None, use_per_token_if_dynamic=True
                )
                self.record(
                    "fused_norm_quant",
                    name,
                    passed=bool(
                        np.array_equal(q.cpu().view(t.uint8).numpy(), eq)
                        and np.array_equal(s.cpu().numpy(), es)
                    ),
                    rows=len(x),
                    unequal_codes=int(
                        np.count_nonzero(q.cpu().view(t.uint8).numpy() != eq)
                    ),
                    unequal_scales=int(np.count_nonzero(s.cpu().numpy() != es)),
                    native_quant_matches_oracle=bool(
                        np.array_equal(native_q.cpu().view(t.uint8).numpy(), eq)
                        and np.array_equal(native_s.cpu().numpy(), es)
                    ),
                    reference="independent FP8 encoding of actual native BF16 normalization",
                )
                # M1 is assessed separately from the prefill-width fused call.
                one = [
                    fused.norm(
                        xg[i : i + 1], r[i : i + 1], wg, self.config["rms_norm_eps"]
                    )
                    for i in range(len(x))
                ]
                oneq, ones = t.cat([v[0] for v in one]), t.cat([v[1] for v in one])
                self.record(
                    "fused_norm_quant_m1",
                    name,
                    passed=bool(
                        np.array_equal(oneq.cpu().view(t.uint8).numpy(), eq)
                        and np.array_equal(ones.cpu().numpy(), es)
                    ),
                    rows=len(x),
                    unequal_codes=int(
                        np.count_nonzero(oneq.cpu().view(t.uint8).numpy() != eq)
                    ),
                    unequal_scales=int(np.count_nonzero(ones.cpu().numpy() != es)),
                )
                if not np.array_equal(
                    oneq.cpu().view(t.uint8).numpy(), eq
                ) or not np.array_equal(ones.cpu().numpy(), es):
                    dest = self.args.output.parent / ("norm-witness-" + name + ".npz")
                    np.savez(
                        dest,
                        x=self.array(x),
                        residual=self.array(residual),
                        weight=self.array(weight),
                        native_bf16=self.array(actual),
                        expected_q=eq,
                        expected_s=es,
                        actual_q=oneq.cpu().view(t.uint8).numpy(),
                        actual_s=ones.cpu().numpy(),
                    )
                # Also exercise native dynamic quantization used at projection boundaries.
                quant, scale = ops.scaled_fp8_quant(
                    xg, scale=None, use_per_token_if_dynamic=True
                )
                cq, cs = ref.dynamic_fp8(self.array(x))
                self.record(
                    "dynamic_fp8",
                    name,
                    passed=bool(
                        np.array_equal(quant.cpu().view(t.uint8).numpy(), cq)
                        and np.array_equal(scale.cpu().numpy(), cs)
                    ),
                    rows=len(x),
                )
            del xg, wg, outputs
        for name in sorted(
            n for n in self.index if re.search(r"linear_attn\.norm\.weight$", n)
        ):
            self.check_idle()
            w = self.tensor(name)
            x, z = (
                self.random((self.args.rows, 48, 128)),
                self.random((self.args.rows, 48, 128), 4),
            )
            x[0].zero_()
            x[1] *= 2**-20
            z[2].fill_(-90)
            expected = ref.rms(
                self.array(x), self.array(w), self.config["rms_norm_eps"]
            ) * ref.silu(self.array(z))
            xg, zg, wg = x.cuda(), z.cuda(), w.cuda()
            actual = t.stack(
                [
                    gated(xg[i], zg[i], wg, self.config["rms_norm_eps"])
                    for i in range(len(x))
                ]
            )
            self.record("gdn_gated_norm", name, actual, expected)
            eq, es = ref.dynamic_fp8(self.array(actual).reshape(self.args.rows, 6144))
            pairs = [
                stock_gdn_norm_quant.fused(
                    xg[i : i + 1], zg[i : i + 1], wg, self.config["rms_norm_eps"]
                )
                for i in range(self.args.rows)
            ]
            aq = t.cat([q for q, _ in pairs]).view(t.uint8).cpu().numpy()
            ass = t.cat([s for _, s in pairs]).cpu().numpy()
            self.record(
                "gdn_fused_quant_m1",
                name,
                passed=bool(np.array_equal(aq, eq) and np.array_equal(ass, es)),
                rows=self.args.rows,
                unequal_codes=int(np.count_nonzero(aq != eq)),
                unequal_scales=int(np.count_nonzero(ass != es)),
                reference="independent FP8 encoding of separately assessed native gated norm",
            )
            del pairs
            del xg, zg, wg, actual

    def pointwise(self):
        t = self.torch
        import vllm.model_executor.layers.rotary_embedding.mrope as original_rope
        from stock_fp8_epilogue import StockFP8Epilogue
        from vllm.model_executor.layers.activation import SiluAndMul

        # Corrected eager M1 uses the qualified RNE intervention, installed only
        # inside its worker process. Importing the on-disk vLLM module alone
        # would accidentally audit the pre-alignment eager kernel.
        self.source("rope_original", original_rope.__file__)
        patch_paths = [
            Path(p) / "qwen_r9700_lab/conformance_rotary_repair.py"
            for p in self.manifest["pythonpath"]
        ]
        patch_path = next((p for p in patch_paths if p.is_file()), None)
        if patch_path is None:
            raise RuntimeError("release lacks the qualified eager RoPE repair source")
        self.source("rope_repair", patch_path)
        patch_source = module(patch_path, "m1_audit_rope_repair").patch_source
        generated = self.args.output.parent / (self.args.output.stem + "-rope-rne.py")
        generated.write_text(patch_source(Path(original_rope.__file__).read_text()))
        corrected_rope = module(
            generated, "vllm.model_executor.layers.rotary_embedding._stage_audit_rne"
        )
        MRotaryEmbedding = corrected_rope.MRotaryEmbedding
        rope_launches = [0]
        original_run = corrected_rope._triton_mrope_forward.run

        def observed_rope(*args, **kwargs):
            rope_launches[0] += 1
            return original_run(*args, **kwargs)

        corrected_rope._triton_mrope_forward.run = observed_rope

        # All finite BF16 encodings, paired with exactly representable small ups.
        bits = t.arange(65536, dtype=t.int32).short().view(t.bfloat16)
        bits = bits[t.isfinite(bits)]
        values = bits.repeat(2)[:34816].reshape(2, 17408)
        # Include both signs and all exponent ranges over four batches.
        for offset in (0, 16384, 32768, 49152):
            gate = bits.roll(offset).repeat(2)[:17408].reshape(1, -1)
            up = t.full_like(gate, 0.25)
            x = t.cat((gate, up), -1).cuda()
            actual = SiluAndMul().forward_native(x)
            expected = ref.bf16(ref.bf16(ref.silu(self.array(gate))) * 0.25)
            self.record("mlp_silu_gate", f"finite-bf16-{offset}", actual, expected)
            self.record(
                "attention_sigmoid_gate",
                f"finite-bf16-{offset}",
                t.sigmoid(x[:, :17408]),
                ref.sigmoid(self.array(gate)),
            )
        fused = StockFP8Epilogue(self.performance["tp1_fp8"]["build"])
        self.source(
            "silu_quant_binary",
            Path(self.performance["tp1_fp8"]["build"]) / "candidate.so",
        )
        gu = self.random((self.args.rows, 34816), 3)
        for i, offset in enumerate((0, 16384, 32768, 49152)):
            gu[i, :17408] = bits.roll(offset).repeat(2)[:17408]
            gu[i, 17408:] = 0.25
        for start in range(self.args.rows):
            self.check_idle()
            x = gu[start : start + 1].cuda()
            native_value = t.nn.functional.silu(x[:, :17408]) * x[:, 17408:]
            eq, es = ref.dynamic_fp8(self.array(native_value))
            cq, cs = fused.silu(x)
            aq, ass = cq.view(t.uint8).cpu().numpy(), cs.cpu().numpy()
            self.record(
                "mlp_fused_quant_m1",
                f"row-{start}",
                passed=bool(np.array_equal(aq, eq) and np.array_equal(ass, es)),
                unequal_codes=int(np.count_nonzero(aq != eq)),
                unequal_scales=int(np.count_nonzero(ass != es)),
                reference="native BF16 SiLU/multiply rounding then independent FP8 encoding",
            )
        del gu, x, native_value, cq, cs
        parameters = self.config["rope_parameters"]
        dim = int(self.config["head_dim"] * parameters["partial_rotary_factor"])
        with t.device("cuda"):
            rope = MRotaryEmbedding(
                self.config["head_dim"],
                dim,
                self.args.rows,
                parameters["rope_theta"],
                True,
                t.bfloat16,
                mrope_section=parameters["mrope_section"],
                mrope_interleaved=parameters["mrope_interleaved"],
            )
        self.source("rope", sys.modules[type(rope).__module__].__file__)
        for start in (0, 15, 16, 1023, 32767, 60000, 131071, 200000, 253760):
            self.check_idle()
            pos = t.arange(start, start + self.args.rows, dtype=t.long)
            with t.device("cuda"):
                inv = rope._compute_inv_freq(rope.base)
                frequencies = t.einsum("i,j->ij", pos.cuda().float(), inv)
                rope.cos_sin_cache = t.cat(
                    (frequencies.cos(), frequencies.sin()), -1
                ).to(rope.cos_sin_cache.dtype)
            lookup = t.arange(len(pos), device="cuda", dtype=t.long)
            for heads in (4, 24):
                x = self.random((len(pos), heads, 256))
                angles = self.array(pos)[:, None] * np.power(
                    float(parameters["rope_theta"]), -np.arange(0, dim, 2) / dim
                )
                cosine, sine = np.cos(angles)[:, None], np.sin(angles)[:, None]
                arr = self.array(x)
                first, second = arr[..., : dim // 2], arr[..., dim // 2 : dim]
                expected = np.concatenate(
                    (
                        first * cosine - second * sine,
                        second * cosine + first * sine,
                        arr[..., dim:],
                    ),
                    -1,
                )
                # Both Q/K arguments are legal flattened arrays; only one is assessed per call.
                q = x.reshape(len(pos), -1).cuda()
                k = self.random((len(pos), 1024)).cuda()
                for layout, indices in (
                    ("text-vector", lookup),
                    ("mrope-three-axis", lookup.repeat(3, 1)),
                ):
                    before = rope_launches[0]
                    # vLLM's forward_cuda is also the ROCm eager implementation.
                    # Call that implementation explicitly; a compilation config
                    # must not silently select forward_native for this probe.
                    actual, _ = rope.forward_cuda(indices, q.clone(), k.clone())
                    actual = actual.reshape(len(pos), heads, 256)
                    site = f"position-{start}-heads-{heads}-{layout}"
                    self.record(
                        "rope",
                        site,
                        actual,
                        expected,
                        relative=0.012,
                        position_table="native GPU frequency construction at real positions; compact lookup indexes to bound VRAM",
                    )
                    self.record(
                        "rope_dispatch",
                        site,
                        passed=(
                            rope_launches[0] - before == (1 if indices.ndim == 2 else 0)
                        ),
                        corrected_mrope_launches=rope_launches[0] - before,
                    )
                    if dim < 256:
                        self.record(
                            "rope_tail",
                            site,
                            actual[..., dim:],
                            arr[..., dim:],
                            exact=True,
                        )
        # Native OCP cast/scale round-trip exercises every finite stored byte.
        codes = t.tensor([i for i in range(256) if i & 127 != 127], dtype=t.uint8)
        encoded = codes.view(t.float8_e4m3fn).cuda()
        self.record(
            "kv_fp8_storage",
            "all-finite-encodings",
            encoded.float(),
            ref.fp8_decode(codes.numpy()),
            exact=True,
        )
        from vllm.v1.attention.backends.triton_attn import (
            triton_reshape_and_cache_flash,
        )

        self.source(
            "kv_write", sys.modules[triton_reshape_and_cache_flash.__module__].__file__
        )
        for dtype, label in ((t.bfloat16, "auto"), (t.float8_e4m3fn, "fp8_e4m3")):
            slots = t.tensor([-1, 0, 15, 16, 31, 46], dtype=t.long, device="cuda")
            key, value = self.random((6, 4, 256), 2), self.random((6, 4, 256), 2)
            for ks, vs in ((1.0, 1.0), (0.125, 16.0)):
                self.check_idle()
                cache = t.zeros((3, 4, 16, 512), device="cuda", dtype=dtype)
                keycache, valcache = cache.transpose(1, 2).split(256, dim=-1)
                triton_reshape_and_cache_flash(
                    key.cuda(),
                    value.cuda(),
                    keycache,
                    valcache,
                    slots,
                    label,
                    t.tensor([ks], device="cuda"),
                    t.tensor([vs], device="cuda"),
                )
                expected = np.zeros((3, 4, 16, 512), dtype=np.float64)
                for i, slot in enumerate([-1, 0, 15, 16, 31, 46]):
                    if slot < 0:
                        continue
                    k, v = self.array(key[i]), self.array(value[i])
                    if dtype == t.float8_e4m3fn:
                        k, v = (
                            ref.fp8_decode(ref.fp8_encode(k / ks)),
                            ref.fp8_decode(ref.fp8_encode(v / vs)),
                        )
                    expected[slot // 16, :, slot % 16, :256] = k
                    expected[slot // 16, :, slot % 16, 256:] = v
                self.record(
                    "kv_write", f"{label}-scale-{ks}-{vs}", cache, expected, exact=True
                )
        for name in ("model.language_model.embed_tokens.weight",):
            for start in (0, 1024, 60000, self.config["vocab_size"] - 64):
                slab = self.tensor(name, slice(start, start + 64))
                ids = t.tensor([0, 63, 1, 32, 0], dtype=t.long)
                actual = t.nn.functional.embedding(ids.cuda(), slab.cuda())
                self.record(
                    "embedding",
                    f"vocab-slab-{start}",
                    actual,
                    self.array(slab[ids]),
                    exact=True,
                )
        del values

    def gemm(self):
        from mxfp4_fold_precision import make_row_ref
        from probe_eager_m1_independent import decode_mxfp4_rows

        import radiance_mxfp4 as wrapper

        t = self.torch
        build = Path(self.performance["gemm_dispatch"]["m1_arithmetic"]["build"])
        binary = build / "candidate/radiance_mxfp4_fp8.so"
        self.source("gemm_binary", binary)
        native = module(binary, "m1_audit.radiance_mxfp4_fp8")
        # The release's split-K mechanism requires the declared persistent scratch.
        scratch = t.zeros(16 * 1024 * 1024 // 4, dtype=t.float32, device="cuda")
        counter = t.zeros(1024 * 1024 // 4, dtype=t.int32, device="cuda")
        native.set_decode_scratch(
            scratch.data_ptr(), scratch.numel() * 4, counter.data_ptr()
        )
        names = target_projection_names(self.index)
        for count, scale_name in enumerate(names):
            self.check_idle()
            weight_name = scale_name.removesuffix("_scale")
            packed, scales = self.tensor(weight_name), self.tensor(scale_name)
            if packed.dtype != t.uint8 or packed.ndim != 2:
                raise ValueError(f"unexpected quantized projection: {weight_name}")
            n, k = packed.shape[0], packed.shape[1] * 2
            selected = sorted(
                set(
                    [0, 1, n // 2, n - 1]
                    + (
                        [9544, 10568, 10711, 12048]
                        if "layers.15.self_attn.q_proj" in weight_name
                        else []
                    )
                )
            )
            expected_weights = decode_mxfp4_rows(
                t, packed[selected], scales[selected]
            ).numpy()
            source = self.random((self.args.rows, k))
            source[0].zero_()
            source[0, 0] = 1
            source[1].zero_()
            source[1, -1] = 1
            codes, scaling = ref.dynamic_fp8(self.array(source))
            exact_input = ref.fp8_decode(codes) * scaling.astype(np.float64)
            expected = exact_input @ expected_weights.T
            x = t.from_numpy(codes).view(t.float8_e4m3fn).cuda()
            xs = t.from_numpy(scaling).cuda()
            w = wrapper.permute_w(packed, n, k).cuda()
            ws = scales.T.contiguous().cuda()
            wr = make_row_ref(scales.T).cuda()
            out = t.empty((len(source), n), dtype=t.bfloat16, device="cuda")
            for i in range(len(source)):
                native.launch(
                    x[i].data_ptr(),
                    w.data_ptr(),
                    ws.data_ptr(),
                    wr.data_ptr() if wr.numel() else 0,
                    xs[i].data_ptr(),
                    out[i].data_ptr(),
                    1,
                    n,
                    k,
                    t.cuda.current_stream().cuda_stream,
                )
            self.record(
                "mxfp4_projection",
                weight_name,
                out[:, selected],
                expected,
                shape=[n, k],
                rows=len(source),
                selected_output_rows=selected,
            )
            # The one-hot rows have no reduction-order ambiguity.
            self.record(
                "mxfp4_coefficient",
                weight_name,
                out[:2, selected],
                ref.bf16(expected[:2]),
                exact=True,
            )
            del packed, scales, source, x, xs, w, ws, wr, out
            if count % 32 == 0:
                print(
                    json.dumps(
                        {"stage": "gemm", "completed": count + 1, "total": len(names)}
                    ),
                    flush=True,
                )

    def attention(self):
        from attention_precision_runtime import PrecisionAttention

        t = self.torch
        build = Path(self.performance["attention_precision"]["build"])
        native = PrecisionAttention(build)
        self.source("attention_binary", build / "native.so")
        # Periodic immutable pages allow a full 253K logical context without
        # allocating a second 0.5-1 GiB KV cache beside the serving worker. The
        # FP64 reference sums the exact multiplicities of those 48 logical slots.
        guard = 512
        for dtype, label in ((t.float8_e4m3fn, "fp8"), (t.bfloat16, "bf16")):
            physical = self.random((3, 4, 16, 512), 0.5).to(dtype)
            permutation = np.array([1, 2, 0])
            logical = (
                self.array(physical)[permutation]
                .transpose(0, 2, 1, 3)
                .reshape(48, 4, 512)
            )
            cache = physical.cuda()
            original_bytes = cache.view(t.uint8).clone()
            for length in (
                1,
                15,
                16,
                17,
                31,
                32,
                47,
                48,
                49,
                511,
                512,
                1023,
                1024,
                8191,
                8192,
                32767,
                60000,
                60001,
                131071,
                200000,
                253791,
                253792,
            ):
                self.check_idle()
                blocks = (length + 15) // 16
                table = t.tensor(
                    permutation[np.arange(blocks) % 3], dtype=t.int32, device="cuda"
                )[None]
                lens = t.tensor([length], dtype=t.int32, device="cuda")
                query = self.random((1, 24, 256), 0.5)
                q = query.cuda()
                q64 = self.array(query)[0]
                size = native.attn_decode_h256_gqa6_scratch_bytes(
                    1, 1, 24, 4, 256, length, 0
                )
                scratch = t.full(
                    (size + guard * 2,), 0xA5, device="cuda", dtype=t.uint8
                )
                slab = t.full((6144 + guard * 2,), 42, device="cuda", dtype=t.bfloat16)
                output = slab[guard:-guard].view(1, 24, 256)
                for kscale, vscale in ((1.0, 1.0), (0.125, 16.0)):
                    # Native ABI indexes descales by KV head, even when every
                    # head uses the same per-tensor value.
                    kd = t.full((4,), kscale, device="cuda", dtype=t.float32)
                    vd = t.full((4,), vscale, device="cuda", dtype=t.float32)
                    expected = ref.periodic_attention(
                        q64, logical, length, kscale=kscale, vscale=vscale
                    )
                    getattr(native, f"attn_decode_h256_gqa6_{label}kv")(
                        q.data_ptr(),
                        cache.data_ptr(),
                        table.data_ptr(),
                        lens.data_ptr(),
                        output.data_ptr(),
                        kd.data_ptr(),
                        vd.data_ptr(),
                        scratch[guard:].data_ptr(),
                        1,
                        1,
                        24,
                        4,
                        256,
                        16,
                        blocks,
                        cache.stride(0),
                        cache.stride(1),
                        1 / 16,
                        0,
                        length,
                        t.cuda.current_stream().cuda_stream,
                    )
                    self.record(
                        "attention_periodic_long_context",
                        f"{label}-{length}-scale-{kscale}-{vscale}",
                        output[0],
                        expected,
                        relative=0.005,
                        logical_context=length,
                        physical_pages=3,
                    )
                    intact = bool(
                        (scratch[:guard] == 0xA5).all()
                        and (scratch[-guard:] == 0xA5).all()
                        and (slab[:guard] == 42).all()
                        and (slab[-guard:] == 42).all()
                        and t.equal(cache.view(t.uint8), original_bytes)
                    )
                    self.record(
                        "attention_memory_isolation",
                        f"{label}-{length}-scale-{kscale}-{vscale}",
                        passed=intact,
                    )
                del table, lens, q, scratch, slab, output
            del cache, original_bytes

    def head(self):
        import radiance_drafthead as dh

        t = self.torch
        self.source("rerank", dh.__file__)
        # Native candidate rescoring, public head rows, independent dot products.
        # This checks score arithmetic, not approximate candidate completeness.
        name = "lm_head.weight"
        if name not in self.index:
            matches = [n for n in self.index if n.endswith("lm_head.weight")]
            if len(matches) != 1:
                raise ValueError("target head weight identity ambiguous")
            name = matches[0]
        for start in (0, 60000, self.config["vocab_size"] - 512):
            self.check_idle()
            weight = self.tensor(name, slice(start, start + 512))
            x = self.random((self.args.rows, 5120))
            x[0].zero_()
            x[0, 0] = 1
            x[1].zero_()
            x[1, -1] = 1
            w, xg = weight.cuda(), x.cuda()
            ids = t.arange(512, device="cuda", dtype=t.int32)[None]
            expected = self.array(x) @ self.array(weight).T
            output = t.empty((self.args.rows, 512), device="cuda", dtype=t.float32)
            for i in range(self.args.rows):
                dh._rerank_exact[(1, 512)](
                    xg[i : i + 1],
                    w,
                    ids,
                    output[i : i + 1],
                    5120,
                    w.stride(0),
                    R=512,
                    BLOCK_K=512,
                    num_warps=4,
                )
            self.record(
                "head_rerank", f"vocab-slab-{start}", output.bfloat16(), expected
            )
            self.record(
                "head_coefficient",
                f"vocab-slab-{start}",
                output[:2].bfloat16(),
                expected[:2],
                exact=True,
            )
            full = t.cat(
                [
                    t.nn.functional.linear(xg[i : i + 1], w)
                    for i in range(self.args.rows)
                ]
            )
            self.record("head_full_m1", f"vocab-slab-{start}", full, expected)

    def gdn(self):
        from vllm.third_party.flash_linear_attention.ops import (
            fused_recurrent as native,
        )

        t = self.torch
        self.source("gdn_recurrent", native.__file__)
        indices = t.tensor([1], device="cuda", dtype=t.int32)
        state = t.zeros((2, 48, 128, 128), device="cuda", dtype=t.float32)
        output = t.empty((1, 1, 48, 128), dtype=t.bfloat16, device="cuda")
        names = sorted(
            n.removesuffix("A_log")
            for n in self.index
            if n.endswith("linear_attn.A_log")
        )
        for name in names:
            self.check_idle()
            al, bias = self.tensor(name + "A_log"), self.tensor(name + "dt_bias")
            mixed = self.random((self.args.rows, 10240))
            a, b = (
                self.random((self.args.rows, 48), 6),
                self.random((self.args.rows, 48), 4),
            )
            initial = self.random((48, 128, 128), 0.025).float()
            state[0].fill_(0.125)
            state[1].copy_(initial.cuda())
            oracle = self.array(initial)
            outputs, expected = [], []
            mg, ag, bg, alg, dg = (
                mixed.cuda(),
                a.cuda(),
                b.cuda(),
                al.float().cuda(),
                bias.cuda(),
            )
            for i in range(self.args.rows):
                oracle, prediction = ref.gdn_step(
                    oracle,
                    self.array(mixed[i]),
                    self.array(a[i]),
                    self.array(b[i]),
                    self.array(al),
                    self.array(bias),
                )
                native.fused_recurrent_gated_delta_rule_packed_decode(
                    mg[i : i + 1],
                    ag[i : i + 1],
                    bg[i : i + 1],
                    alg,
                    dg,
                    128**-0.5,
                    state,
                    output,
                    indices,
                    True,
                )
                outputs.append(output.clone())
                expected.append(prediction)
            self.record(
                "gdn_recurrence",
                name,
                state[1],
                oracle,
                relative=2e-4,
                transitions=self.args.rows,
            )
            self.record(
                "gdn_output",
                name,
                t.cat(outputs).reshape(self.args.rows, 48, 128),
                np.stack(expected),
            )
            self.record(
                "gdn_slot_isolation",
                name,
                state[0],
                np.full((48, 128, 128), 0.125),
                exact=True,
            )
            del mg, ag, bg, alg, dg, outputs
        # Long decay-only trajectory: exact in all untouched coordinates. The
        # scalar closed form is independent of the native recurrence algorithm.
        self.check_idle()
        mixed = t.zeros((1, 10240), device="cuda", dtype=t.bfloat16)
        a = t.full((1, 48), -14.9375, device="cuda", dtype=t.bfloat16)
        b = t.zeros_like(a)
        al = t.full((48,), 4.9375, device="cuda")
        bias = t.full((48,), -3.078125, device="cuda", dtype=t.bfloat16)
        state[1].fill_(1)
        decay = np.exp(-np.exp(4.9375) * np.logaddexp(0, -18.015625))
        checkpoints = (1, 32, 320, 2048, self.args.transitions)
        for i in range(1, self.args.transitions + 1):
            if i % 128 == 1:
                self.check_idle()
            native.fused_recurrent_gated_delta_rule_packed_decode(
                mixed, a, b, al, bias, 128**-0.5, state, output, indices, True
            )
            if i in checkpoints:
                self.record(
                    "gdn_long_decay",
                    f"transitions-{i}",
                    state[1],
                    np.full((48, 128, 128), decay**i),
                    relative=2e-4,
                )

    def convolution(self):
        t = self.torch
        path = Path(self.repair["convolution"])
        conv = module(path, "m1_audit_convolution")
        self.source("convolution", path)
        names = sorted(n for n in self.index if n.endswith("linear_attn.conv1d.weight"))
        for name in names:
            self.check_idle()
            weight = self.tensor(name).squeeze(1)
            channels, width = weight.shape
            x = self.random((self.args.rows, channels), 0.25)
            history = self.random((channels, width - 1), 0.25)
            state = t.zeros((2, channels, width - 1), device="cuda", dtype=t.bfloat16)
            state[1].copy_(history.cuda())
            state[0].fill_(0.25)
            ids = t.tensor([1], device="cuda", dtype=t.int32)
            wg = weight.cuda()
            outputs, expected = [], []
            logical = self.array(history)
            w = self.array(weight)
            for row in x:
                window = np.concatenate((logical, self.array(row)[:, None]), 1)
                # FP64 convolution with BF16 output after SiLU; arithmetic-order
                # effects are quantified rather than mislabeled as exact.
                expected.append(ref.silu(np.sum(window * w, axis=1)))
                value = conv.causal_conv1d_update(
                    row[None].cuda(),
                    state,
                    wg,
                    activation="silu",
                    conv_state_indices=ids,
                )
                outputs.append(value.clone())
                logical = window[:, 1:]
            self.record("gdn_convolution", name, t.cat(outputs), np.stack(expected))
            self.record("convolution_history", name, state[1], logical, exact=True)
            self.record(
                "convolution_slot_isolation",
                name,
                state[0],
                np.full((channels, width - 1), 0.25),
                exact=True,
            )

    def run(self):
        for name in self.args.stages:
            self.check_idle()
            print(json.dumps({"starting": name}), flush=True)
            before = len(self.report["checks"])
            try:
                getattr(self, name)()
            except Exception as exc:  # noqa: BLE001 -- record an incomplete stage, never a pass
                import traceback

                self.report["errors"].append(
                    {
                        "stage": name,
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                self.save()
                print(json.dumps(self.report["errors"][-1]), flush=True)
                if "busy" in str(exc) or "memory" in str(exc).lower():
                    break
            self.report.setdefault("stage_counts", {})[name] = (
                len(self.report["checks"]) - before
            )
            gc.collect()
            self.torch.cuda.empty_cache()
        failures = sum(not r["passed"] for r in self.report["checks"])
        self.report["status"] = result_status(self.report, self.args.stages)
        self.report["failed_checks"] = failures
        self.report["peak_allocator_mib"] = (
            self.torch.cuda.max_memory_allocated() / 2**20
        )
        self.save()
        print(
            json.dumps(
                {
                    k: v
                    for k, v in self.report.items()
                    if k
                    not in ("checks", "sources", "errors", "checkpoint_tensor_bindings")
                }
            ),
            flush=True,
        )
        return 0 if self.report["status"] == "SAMPLE_CHECKED" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=Path("/qualification"))
    parser.add_argument(
        "--model", type=Path, default=Path("/models/Qwen3.8-27B-Uncensored-MXFP4-awq")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=(
            "norms",
            "pointwise",
            "gemm",
            "gdn",
            "convolution",
            "attention",
            "head",
        ),
        default=[
            "norms",
            "pointwise",
            "gemm",
            "gdn",
            "convolution",
            "attention",
            "head",
        ],
    )
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--transitions", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=240926)
    parser.add_argument("--memory-mib", type=int, default=192)
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--backend-stopped", action="store_true")
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    if (
        not args.allow_gpu
        or not 4 <= args.rows <= 320
        or not 320 <= args.transitions <= 65536
        or len(set(args.stages)) != len(args.stages)
    ):
        parser.error("GPU opt-in and bounded nonempty cases required")
    if args.output.exists():
        parser.error("output already exists; preserve previous evidence")
    release = json.loads((args.release / "optimized-release.json").read_text())
    for path in reversed(release["pythonpath"]):
        sys.path.append(path)
    from probe_eager_m1_independent import device_memory, idle

    if not args.backend_stopped:
        idle(args.api)
    _, free = device_memory()
    if free < (args.memory_mib + 256) * 2**20:
        raise RuntimeError(
            "insufficient free VRAM for the bounded audit plus runtime context"
        )
    from vllm.config import VllmConfig, set_current_vllm_config

    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease

    with (
        gpu_lease(args.output.parent / ("lease-" + args.output.stem)),
        set_current_vllm_config(VllmConfig()),
    ):
        return Audit(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
