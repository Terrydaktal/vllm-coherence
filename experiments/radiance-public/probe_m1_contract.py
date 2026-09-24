"""Isolated native qualification of normalization partitions and wider M1 oracles.

Inputs are public checkpoint tensors and seeded numeric arrays. No chat or token
text is read. Original failures, candidate checks and timing have separate keys.
"""

import argparse
import json
import sys
from pathlib import Path

import m1_arithmetic_contract as contract
import m1_stage_oracles as ref
import numpy as np
from probe_m1_stage_audit import Audit, module, sha


class ContractAudit(Audit):
    def __init__(self, args):
        super().__init__(args)
        self.report["schema"] = "coherence-m1-contract-audit-v1"
        self.report["probe_sha256"] = sha(__file__)
        self.report["base_audit_sha256"] = sha(
            Path(__file__).with_name("probe_m1_stage_audit.py")
        )
        self.report["arithmetic_contract_sha256"] = sha(args.contract)
        self.report["contract_oracle_sha256"] = sha(contract.__file__)
        self.report["baseline_differences"] = []
        self.report["timings"] = []
        self.report["witnesses"] = []

    def norm_partition(self):
        from stock_fp8_epilogue import StockFP8Epilogue
        from stock_m1_norm import StockM1Norm

        t = self.torch
        old = StockFP8Epilogue(self.performance["tp1_fp8"]["build"])
        new = StockFP8Epilogue(self.args.norm_candidate)
        native = StockM1Norm(self.repair["norm_build"])
        self.source("old_norm_quant", old.build / "candidate.so")
        self.source("candidate_norm_quant", new.build / "candidate.so")
        names = sorted(
            n
            for n in self.index
            if n == "model.language_model.norm.weight"
            or (
                n.startswith("model.language_model.layers.")
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
            )
        )
        assert len(names) == 161
        total_old = 0
        for name in names:
            self.check_idle()
            w = self.tensor(name)
            width = w.numel()
            shape = (
                (320, width)
                if width == 5120
                else (320, 24 if "q_norm" in name else 4, width)
            )
            x = self.random(shape)
            x[0].zero_()
            x[1].fill_(2**-20)
            x[2] *= 128
            # Keep the original RNG schedule, including the intervening Q/K sites.
            if width != 5120:
                continue
            residual = self.random(shape)
            residual[0].zero_()
            residual[3] = -x[3]
            xg, rg, wg = x.cuda(), residual.cuda(), w.cuda()
            eps = self.config["rms_norm_eps"]
            expected_bf16 = native(xg, rg, wg, eps)[0]
            eq, es = ref.dynamic_fp8(self.array(expected_bf16))
            oldq, olds, oldr = old.norm(xg, rg, wg, eps)
            newq, news, newr = new.norm(xg, rg, wg, eps)
            oq, nq = oldq.view(t.uint8).cpu().numpy(), newq.view(t.uint8).cpu().numpy()
            different = np.argwhere(oq != eq)
            total_old += len(different)
            self.report["baseline_differences"].append(
                {
                    "site": name,
                    "codes": len(different),
                    "coordinates": different.tolist(),
                    "scale_differences": int(
                        np.count_nonzero(olds.cpu().numpy() != es)
                    ),
                }
            )
            self.record(
                "norm_partition",
                name,
                passed=bool(
                    np.array_equal(nq, eq)
                    and np.array_equal(news.cpu().numpy(), es)
                    and t.equal(newr, oldr)
                ),
                rows=320,
                codes=int(nq.size),
                candidate_unequal_codes=int(np.count_nonzero(nq != eq)),
            )
            if len(different):
                selected = np.unique(different[:, 0])
                sx, sr = self.array(x[selected]), self.array(residual[selected])
                wy, _, variance, inverse = contract.hidden_norm(
                    sx, self.array(w), eps, sr
                )
                self.record(
                    "norm_cpu_contract", name, expected_bf16[selected], wy, exact=True
                )
                witness = self.args.output.parent / ("norm-" + name + ".npz")
                np.savez_compressed(
                    witness,
                    x=sx,
                    residual=sr,
                    weight=self.array(w),
                    original_rows=selected,
                    old_q=oq[selected],
                    candidate_q=nq[selected],
                    scales=es[selected],
                    native_bf16=self.array(expected_bf16[selected]),
                    variance=variance,
                    inverse=inverse,
                )
                self.report["witnesses"].append(
                    {
                        "path": witness.name,
                        "sha256": sha(witness),
                        "rows": len(selected),
                        "site": name,
                    }
                )
            del xg, rg, wg, expected_bf16, oldq, olds, oldr, newq, news, newr
        self.report["baseline_code_differences"] = total_old
        self.record(
            "original_failure_reproduced",
            "original-320-row-seed",
            passed=total_old == 55,
            observed=total_old,
            expected=55,
        )
        self.norm_boundaries(old, new, native)

    def norm_boundaries(self, old, new, native):
        t = self.torch
        w = self.tensor("model.language_model.layers.0.input_layernorm.weight").cuda()
        eps = self.config["rms_norm_eps"]
        for rows in (1, 2, 7, 8, 9, 15, 16, 17, 64, 320, 1000, 2048):
            self.check_idle()
            x = self.random((rows, 5120)).cuda()
            r = self.random((rows, 5120)).cuda()
            for residual in (None, r):
                y = native(x, residual, w, eps)
                if residual is not None:
                    y = y[0]
                eq, es = ref.dynamic_fp8(self.array(y))
                q, s, _ = new.norm(x, residual, w, eps)
                self.record(
                    "norm_boundary",
                    f"rows-{rows}-residual-{residual is not None}",
                    passed=bool(
                        np.array_equal(q.view(t.uint8).cpu().numpy(), eq)
                        and np.array_equal(s.cpu().numpy(), es)
                    ),
                )
                if rows in (1, 8, 320, 2048):
                    times = {}
                    for label, operator in (("old", old), ("candidate", new)):
                        for _ in range(5):
                            operator.norm(x, residual, w, eps)
                        t.cuda.synchronize()
                        graph = t.cuda.CUDAGraph()
                        outq, outs, outr = operator.norm(x, residual, w, eps)
                        stream = t.cuda.current_stream().cuda_stream
                        with t.cuda.graph(graph):
                            stream = t.cuda.current_stream().cuda_stream
                            for _ in range(32):
                                rc = operator.norm_launch(
                                    x.data_ptr(),
                                    residual.data_ptr() if residual is not None else 0,
                                    w.data_ptr(),
                                    outq.data_ptr(),
                                    outs.data_ptr(),
                                    outr.data_ptr() if outr is not None else 0,
                                    rows,
                                    x.stride(0),
                                    residual.stride(0) if residual is not None else 0,
                                    eps,
                                    stream,
                                )
                                if rc:
                                    raise RuntimeError(
                                        f"norm timing launch failed: {rc}"
                                    )
                        graph.replay()
                        t.cuda.synchronize()
                        samples = []
                        for _ in range(15):
                            begin, end = (
                                t.cuda.Event(enable_timing=True),
                                t.cuda.Event(enable_timing=True),
                            )
                            begin.record()
                            graph.replay()
                            end.record()
                            end.synchronize()
                            samples.append(begin.elapsed_time(end) / 32)
                        times[label] = {
                            "median_ms": float(np.median(samples)),
                            "samples_ms": samples,
                        }
                        del outq, outs, outr, graph
                    self.report["timings"].append(
                        {
                            "rows": rows,
                            "residual": residual is not None,
                            "graph_replays": 15,
                            "calls_per_replay": 32,
                            **times,
                        }
                    )
                    self.save()
            del x, r

    def minimal_norm(self):
        from stock_fp8_epilogue import StockFP8Epilogue

        t = self.torch
        old = StockFP8Epilogue(self.performance["tp1_fp8"]["build"])
        new = StockFP8Epilogue(self.args.norm_candidate)
        path = self.args.minimal_witness
        with np.load(path, allow_pickle=False) as saved:
            x = t.from_numpy(saved["x"]).bfloat16().cuda()
            r = t.from_numpy(saved["residual"]).bfloat16().cuda()
            w = t.from_numpy(saved["weight"]).bfloat16().cuda()
            eps = float(saved["epsilon"])
            # Padding unrelated rows crosses the old prefill dispatch boundary.
            xx = t.zeros((16, 5120), device="cuda", dtype=t.bfloat16)
            rr = t.zeros_like(xx)
            xx[:1], rr[:1] = x, r
            oq, os, _ = old.norm(xx, rr, w, eps)
            nq, ns, _ = new.norm(xx, rr, w, eps)
            sq, ss, _ = new.norm(x, r, w, eps)
            self.record(
                "minimal_norm",
                "old-failure",
                passed=bool(
                    np.array_equal(oq[:1].view(t.uint8).cpu().numpy(), saved["old_q"])
                    and np.array_equal(os[:1].cpu().numpy(), saved["old_s"])
                    and not t.equal(oq[:1].view(t.uint8), sq.view(t.uint8))
                ),
                witness_sha256=sha(path),
            )
            self.record(
                "minimal_norm",
                "candidate-partition",
                passed=bool(
                    t.equal(nq[:1].view(t.uint8), sq.view(t.uint8))
                    and t.equal(ns[:1], ss)
                    and np.array_equal(sq.view(t.uint8).cpu().numpy(), saved["new_q"])
                ),
            )

    def load_gemm(self, weight_name):
        from mxfp4_fold_precision import make_row_ref

        import radiance_mxfp4 as wrapper

        t = self.torch
        if not hasattr(self, "gemm_kernel"):
            build = Path(self.performance["gemm_dispatch"]["m1_arithmetic"]["build"])
            binary = build / "candidate/radiance_mxfp4_fp8.so"
            self.source("gemm_binary", binary)
            self.gemm_kernel = module(binary, "m1_contract.radiance_mxfp4_fp8")
            self.gemm_scratch = t.zeros(
                16 * 1024 * 1024 // 4, dtype=t.float32, device="cuda"
            )
            self.gemm_counter = t.zeros(1024 * 1024 // 4, dtype=t.int32, device="cuda")
            self.gemm_kernel.set_decode_scratch(
                self.gemm_scratch.data_ptr(),
                self.gemm_scratch.numel() * 4,
                self.gemm_counter.data_ptr(),
            )
        packed, scales = self.tensor(weight_name), self.tensor(weight_name + "_scale")
        n, k = packed.shape[0], packed.shape[1] * 2
        w = wrapper.permute_w(packed, n, k).cuda()
        ws, wr = scales.T.contiguous().cuda(), make_row_ref(scales.T).cuda()

        def execute(codes, scaling):
            x = t.from_numpy(np.ascontiguousarray(codes)).view(t.float8_e4m3fn).cuda()
            xs = t.from_numpy(np.ascontiguousarray(scaling)).cuda()
            out = t.empty((len(codes), n), dtype=t.bfloat16, device="cuda")
            for i in range(len(codes)):
                self.gemm_kernel.launch(
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
            return out

        return packed, scales, execute

    def projection_full(self):
        from probe_eager_m1_independent import decode_mxfp4_rows

        t = self.torch
        names = []
        for layer in (0, 15, 60, 63):
            prefix = f"model.language_model.layers.{layer}."
            names.extend(
                sorted(
                    n
                    for n in self.index
                    if n.startswith(prefix) and n.endswith("weight_scale")
                )
            )
        capture = self.args.output.parent / "numeric-activations"
        capture.mkdir(mode=0o700)
        for name in names:
            self.check_idle()
            weight_name = name.removesuffix("_scale")
            packed, scales, execute = self.load_gemm(weight_name)
            n, k = packed.shape[0], packed.shape[1] * 2
            source = self.random((32, k))
            # Two real public-checkpoint activation chains: embeddings at hidden
            # width, and a stored BF16 gate/up -> SiLU product at MLP down width.
            kind = "seeded-bf16"
            if k == 5120:
                source[:16] = self.tensor(
                    "model.language_model.embed_tokens.weight", slice(1024, 1040)
                )
                kind = "public-embedding-plus-seeded"
            elif k == 17408:
                gu = self.random((16, 2 * k)).float().numpy()
                product = ref.bf16(ref.bf16(ref.silu(gu[:, :k])) * gu[:, k:])
                source[:16] = t.from_numpy(product).bfloat16()
                kind = "captured-pointwise-product-plus-seeded"
            source[30].zero_()
            source[30, 0] = 1
            source[31].zero_()
            source[31, -1] = 1
            codes, scaling = ref.dynamic_fp8(self.array(source))
            expected_input = ref.fp8_decode(codes) * scaling.astype(np.float64)
            actual = execute(codes, scaling)
            # Tiled CPU reference covers every output channel without holding a
            # second full decoded FP64 weight matrix in memory.
            expected = np.empty((32, n), dtype=np.float64)
            for begin in range(0, n, 256):
                self.check_idle()
                decoded = decode_mxfp4_rows(
                    t, packed[begin : begin + 256], scales[begin : begin + 256]
                ).numpy()
                expected[:, begin : begin + 256] = expected_input @ decoded.T
            self.record(
                "projection_full",
                weight_name,
                actual,
                expected,
                rows=32,
                output_channels=n,
                all_output_channels=True,
                input_kind=kind,
            )
            self.record(
                "projection_coefficients_full",
                weight_name,
                actual[30:],
                ref.bf16(expected[30:]),
                exact=True,
            )
            dest = capture / (weight_name + ".npz")
            np.savez_compressed(
                dest,
                input_codes=codes,
                input_scales=scaling,
                output_bf16=self.array(actual),
            )
            self.report.setdefault("activation_captures", []).append(
                {
                    "path": str(dest.relative_to(self.args.output.parent)),
                    "sha256": sha(dest),
                    "origin": kind,
                }
            )
            del execute, actual, packed, scales, expected, source
            t.cuda.empty_cache()

    def attention_varied(self):
        from attention_precision_runtime import PrecisionAttention

        t = self.torch
        build = Path(self.performance["attention_precision"]["build"])
        native = PrecisionAttention(build)
        self.source("attention_binary", build / "native.so")
        # Distinct physical pages, shuffled order, holes and poisoned tail slots.
        # These complement the previous periodic-page 253K boundary checks.
        for dtype, label in ((t.float8_e4m3fn, "fp8"), (t.bfloat16, "bf16")):
            for length in (
                1,
                15,
                16,
                17,
                511,
                512,
                513,
                4095,
                4096,
                8193,
                32769,
                60001,
            ):
                self.check_idle()
                blocks = (length + 15) // 16
                for pattern in ("random", "uniform", "peaked", "cancellation"):
                    self.check_idle()
                    physical = self.random((blocks + 5, 4, 16, 512), 0.5).to(dtype)
                    ids = t.randperm(blocks + 5, generator=self.rng)[:blocks].numpy()
                    query = self.random((1, 24, 256), 0.5)
                    if pattern == "uniform":
                        query.zero_()
                    elif pattern == "peaked":
                        query.fill_(1)
                        physical[
                            int(ids[length // 2 // 16]), :, (length // 2) % 16, :256
                        ] = 8
                    elif pattern == "cancellation":
                        query.zero_()
                        for slot in range(16):
                            physical[:, :, slot, 256:] = 16 if slot % 2 else -16
                    # Make future/padded logical slots conspicuously wrong.
                    if length % 16:
                        physical[int(ids[-1]), :, length % 16 :, :] = 32
                    cache = physical.cuda()
                    table = t.tensor(ids, dtype=t.int32, device="cuda")[None]
                    lens = t.tensor([length], dtype=t.int32, device="cuda")
                    q = query.cuda()
                    kd = t.tensor([0.125, 1, 2, 0.5], device="cuda", dtype=t.float32)
                    vd = t.tensor([16, 0.5, 1, 2], device="cuda", dtype=t.float32)
                    guard = 256
                    size = native.attn_decode_h256_gqa6_scratch_bytes(
                        1, 1, 24, 4, 256, length, 0
                    )
                    scratch = t.full(
                        (size + guard * 2,), 0xA5, dtype=t.uint8, device="cuda"
                    )
                    slab = t.full(
                        (6144 + 2 * guard,), 42, dtype=t.bfloat16, device="cuda"
                    )
                    out = slab[guard:-guard].view(1, 24, 256)
                    getattr(native, f"attn_decode_h256_gqa6_{label}kv")(
                        q.data_ptr(),
                        cache.data_ptr(),
                        table.data_ptr(),
                        lens.data_ptr(),
                        out.data_ptr(),
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
                    expected = contract.dense_paged_attention(
                        self.array(query)[0],
                        self.array(physical),
                        ids,
                        length,
                        kscale=self.array(kd),
                        vscale=self.array(vd),
                    )
                    self.record(
                        "attention_varied",
                        f"{label}-{length}-{pattern}",
                        out[0],
                        ref.bf16(expected) if pattern == "cancellation" else expected,
                        exact=pattern == "cancellation",
                        relative=0.005,
                        logical_tokens=length,
                        unique_pages=blocks,
                        physical_pages=blocks + 5,
                        pattern=pattern,
                    )
                    intact = bool(
                        (scratch[:guard] == 0xA5).all()
                        and (scratch[-guard:] == 0xA5).all()
                        and (slab[:guard] == 42).all()
                        and (slab[-guard:] == 42).all()
                        and t.equal(cache.cpu().view(t.uint8), physical.view(t.uint8))
                    )
                    self.record(
                        "attention_varied_guards",
                        f"{label}-{length}-{pattern}",
                        passed=intact,
                    )
                    del (
                        cache,
                        physical,
                        q,
                        table,
                        lens,
                        kd,
                        vd,
                        scratch,
                        slab,
                        out,
                        expected,
                    )
                    t.cuda.empty_cache()

    def norm_trace(self):
        """Follow saved counterexamples through their actual next target operators."""
        t = self.torch
        witnesses = sorted(self.args.witness_dir.glob("norm-*.npz"))
        if not witnesses:
            raise ValueError("no saved numerical counterexamples")
        self.report["propagation"] = []
        for path in witnesses:
            self.check_idle()
            site = path.name.removeprefix("norm-").removesuffix(".npz")
            if site == "model.language_model.norm.weight":
                continue
            with np.load(path, allow_pickle=False) as witness:
                old, new, scale = (
                    witness["old_q"],
                    witness["candidate_q"],
                    witness["scales"],
                )
                if "post_attention_layernorm" in site:
                    names = [
                        site.replace("post_attention_layernorm", "mlp.gate_proj"),
                        site.replace("post_attention_layernorm", "mlp.up_proj"),
                    ]
                    kind = "mlp"
                else:
                    layer = int(site.split(".layers.")[1].split(".")[0])
                    kind = "attention" if layer % 4 == 3 else "gdn"
                    parts = (
                        ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
                        if kind == "attention"
                        else (
                            "linear_attn.in_proj_qkv",
                            "linear_attn.in_proj_a",
                            "linear_attn.in_proj_b",
                        )
                    )
                    names = [site.replace("input_layernorm", part) for part in parts]
                pairs = []
                entry = {
                    "site": site,
                    "witness_sha256": sha(path),
                    "rows": len(old),
                    "next_operator_differences": [],
                }
                for name in names:
                    packed, scales, execute = self.load_gemm(name)
                    values = execute(
                        np.concatenate((old, new)), np.concatenate((scale, scale))
                    )
                    left, right = values[: len(old)].clone(), values[len(old) :].clone()
                    pairs.append((left, right))
                    entry["next_operator_differences"].append(
                        {
                            "weight": name,
                            "bf16_values": left.numel(),
                            "different": int((left != right).sum()),
                            "max_abs": float(
                                (left.float() - right.float()).abs().max()
                            ),
                        }
                    )
                    del packed, scales, execute, values
                if kind == "mlp":
                    # Preserve the declared BF16 SiLU and product boundaries.
                    left = t.nn.functional.silu(pairs[0][0]) * pairs[1][0]
                    right = t.nn.functional.silu(pairs[0][1]) * pairs[1][1]
                    entry["gated_mlp_differences"] = int((left != right).sum())
                    oq, os = ref.dynamic_fp8(self.array(left))
                    nq, ns = ref.dynamic_fp8(self.array(right))
                    name = site.replace("post_attention_layernorm", "mlp.down_proj")
                    packed, scales, execute = self.load_gemm(name)
                    values = execute(np.concatenate((oq, nq)), np.concatenate((os, ns)))
                    entry["mlp_output_differences"] = int(
                        (values[: len(old)] != values[len(old) :]).sum()
                    )
                    del packed, scales, execute, values, left, right
                elif kind == "gdn":
                    import vllm.third_party.flash_linear_attention.ops.fused_recurrent as recurrent

                    conv = module(self.repair["convolution"], "contract_trace_conv")
                    prefix = site.replace("input_layernorm.weight", "linear_attn.")
                    cw = self.tensor(prefix + "conv1d.weight").squeeze(1).cuda()
                    al = self.tensor(prefix + "A_log").float().cuda()
                    bias = self.tensor(prefix + "dt_bias").cuda()
                    # Block zero is the null convolution state; qualify a real
                    # slot and require an overwritten output and intact guard.
                    ids = t.tensor([1], device="cuda", dtype=t.int32)
                    state_differences = conv_differences = output_differences = 0
                    for row in range(len(old)):
                        initial = self.random((2, 48, 128, 128), 0.025).float().cuda()
                        history = self.random((2, 10240, cw.shape[1] - 1), 0.25).cuda()
                        initial[0].fill_(0.125)
                        history[0].fill_(0.25)
                        results = []
                        for side in (0, 1):
                            hs, state = history.clone(), initial.clone()
                            mixed = conv.causal_conv1d_update(
                                pairs[0][side][row : row + 1],
                                hs,
                                cw,
                                activation="silu",
                                conv_state_indices=ids,
                            )
                            output = t.full(
                                (1, 1, 48, 128),
                                float("nan"),
                                device="cuda",
                                dtype=t.bfloat16,
                            )
                            recurrent.fused_recurrent_gated_delta_rule_packed_decode(
                                mixed,
                                pairs[1][side][row : row + 1],
                                pairs[2][side][row : row + 1],
                                al,
                                bias,
                                128**-0.5,
                                state,
                                output,
                                ids,
                                True,
                            )
                            if not bool(
                                t.isfinite(output).all()
                                and t.isfinite(state).all()
                                and t.equal(hs[0], history[0])
                                and t.equal(state[0], initial[0])
                            ):
                                raise RuntimeError(
                                    "downstream probe skipped output or damaged the control slot"
                                )
                            results.append((hs, state, output))
                        conv_differences += int((results[0][0] != results[1][0]).sum())
                        state_differences += int((results[0][1] != results[1][1]).sum())
                        output_differences += int(
                            (results[0][2] != results[1][2]).sum()
                        )
                        del initial, history, results, hs, state, output, mixed
                    entry.update(
                        conv_history_differences=conv_differences,
                        gdn_state_differences=state_differences,
                        gdn_output_differences=output_differences,
                        active_state_slot=1,
                        outputs_written_and_control_slot_preserved=True,
                    )
                    del cw, al, bias, ids
                entry["full_model_logits_measured"] = False
                self.report["propagation"].append(entry)
                self.record(
                    "norm_trace",
                    site,
                    passed=True,
                    meaning="counterexample and actual downstream numerical differences retained; not an equality assertion",
                )
                del pairs
                t.cuda.empty_cache()

    def gdn_partition(self):
        import stock_gdn_norm_quant as old
        from stock_m1_gdn_norm import StockM1GdnNorm

        t = self.torch
        new = module(self.args.gdn_candidate, "contract_gdn_norm_candidate")
        native = StockM1GdnNorm()
        self.source("old_gdn_norm_quant", old.__file__)
        self.source("candidate_gdn_norm_quant", self.args.gdn_candidate)
        for name in sorted(
            n for n in self.index if n.endswith("linear_attn.norm.weight")
        ):
            self.check_idle()
            w = self.tensor(name).cuda()
            x, z = (
                self.random((320, 48, 128)).cuda(),
                self.random((320, 48, 128), 4).cuda(),
            )
            eps = self.config["rms_norm_eps"]
            expected = t.stack([native(x[i], z[i], w, eps) for i in range(320)])
            eq, es = ref.dynamic_fp8(self.array(expected).reshape(320, 6144))
            oq, os = old.fused(x, z, w, eps)
            nq, ns = new.fused(x, z, w, eps)
            self.report["baseline_differences"].append(
                {
                    "site": name,
                    "codes": int(
                        np.count_nonzero(oq.view(t.uint8).cpu().numpy() != eq)
                    ),
                    "scales": int(np.count_nonzero(os.cpu().numpy() != es)),
                }
            )
            self.record(
                "gdn_partition",
                name,
                passed=bool(
                    np.array_equal(nq.view(t.uint8).cpu().numpy(), eq)
                    and np.array_equal(ns.cpu().numpy(), es)
                ),
                rows=320,
                codes=int(nq.numel()),
            )
            del expected, oq, os, nq, ns, x, z
        # Numerical layout remains M1 at prefill sizes; graph timings avoid
        # Python dispatch and repeated allocation inside the measured interval.
        for rows in (1, 8, 9, 16, 320, 2048):
            self.check_idle()
            x, z = (
                self.random((rows, 48, 128)).cuda(),
                self.random((rows, 48, 128), 4).cuda(),
            )
            expected = t.stack([native(x[i], z[i], w, eps) for i in range(rows)])
            eq, es = ref.dynamic_fp8(self.array(expected).reshape(rows, 6144))
            nq, ns = new.fused(x, z, w, eps)
            self.record(
                "gdn_boundary",
                f"rows-{rows}",
                passed=bool(
                    np.array_equal(nq.view(t.uint8).cpu().numpy(), eq)
                    and np.array_equal(ns.cpu().numpy(), es)
                ),
            )
            if rows in (1, 8, 320, 2048):
                times = {}
                for label, operator in (("old", old), ("candidate", new)):
                    for _ in range(3):
                        operator.fused(x, z, w, eps)
                    q, s = operator.fused(x, z, w, eps)
                    warps = 16 if rows <= 8 else 8
                    graph = t.cuda.CUDAGraph()
                    t.cuda.synchronize()
                    with t.cuda.graph(graph):
                        for _ in range(32):
                            operator.gdn_norm_quant_kernel[(rows,)](
                                x,
                                z,
                                w,
                                q,
                                s,
                                x.stride(0),
                                z.stride(0),
                                eps,
                                prefill=(rows > 8 and label == "old"),
                                warps=warps,
                                num_warps=warps,
                            )
                    graph.replay()
                    t.cuda.synchronize()
                    samples = []
                    for _ in range(15):
                        a, b = (
                            t.cuda.Event(enable_timing=True),
                            t.cuda.Event(enable_timing=True),
                        )
                        a.record()
                        graph.replay()
                        b.record()
                        b.synchronize()
                        samples.append(a.elapsed_time(b) / 32)
                    times[label] = {
                        "median_ms": float(np.median(samples)),
                        "samples_ms": samples,
                    }
                    del graph, q, s
                self.report["timings"].append(
                    {
                        "stage": "gdn_norm",
                        "rows": rows,
                        "graph_replays": 15,
                        "calls_per_replay": 32,
                        **times,
                    }
                )
                self.save()
            del x, z, expected, nq, ns

    def captured_replay(self):
        """Use preserved operator outputs as inputs to another complete GEMM."""
        from probe_eager_m1_independent import decode_mxfp4_rows

        t = self.torch
        captures = sorted(self.args.activation_dir.glob("*.npz"))
        if not captures:
            raise ValueError("numeric activation captures required")
        count = 0
        for path in captures:
            if not any(
                part in path.name
                for part in (
                    "mlp.down_proj",
                    "mlp.up_proj",
                    "out_proj",
                    "self_attn.o_proj",
                )
            ):
                continue
            self.check_idle()
            with np.load(path, allow_pickle=False) as data:
                source = data["output_bf16"]
            if source.ndim != 2 or source.shape[1] not in (5120, 17408):
                continue
            if not np.isfinite(source).all():
                raise ValueError("nonfinite activation capture")
            target = "model.language_model.layers.1." + (
                "mlp.down_proj.weight"
                if source.shape[1] == 17408
                else "linear_attn.in_proj_qkv.weight"
            )
            packed, scales, execute = self.load_gemm(target)
            q, s = ref.dynamic_fp8(source)
            actual = execute(q, s)
            exact_input = ref.fp8_decode(q) * s.astype(np.float64)
            expected = np.empty(tuple(actual.shape), dtype=np.float64)
            for begin in range(0, len(packed), 256):
                decoded = decode_mxfp4_rows(
                    t, packed[begin : begin + 256], scales[begin : begin + 256]
                ).numpy()
                expected[:, begin : begin + 256] = exact_input @ decoded.T
            self.record(
                "captured_replay",
                path.name,
                actual,
                expected,
                input_sha256=sha(path),
                rows=len(source),
                all_output_channels=True,
                origin="saved public-checkpoint/seeded operator outputs; not a live-chat or full-model capture",
            )
            count += 1
            del packed, scales, execute, actual, expected
            t.cuda.empty_cache()
        if not count:
            raise ValueError("no admitted captured activations were replayed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=Path("/qualification"))
    parser.add_argument(
        "--model", type=Path, default=Path("/models/Qwen3.8-27B-Uncensored-MXFP4-awq")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--norm-candidate", type=Path)
    parser.add_argument("--gdn-candidate", type=Path)
    parser.add_argument("--witness-dir", type=Path)
    parser.add_argument("--activation-dir", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--minimal-witness", type=Path)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=(
            "norm_partition",
            "minimal_norm",
            "projection_full",
            "attention_varied",
            "norm_trace",
            "gdn_partition",
            "captured_replay",
        ),
        required=True,
    )
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--memory-mib", type=int, default=256)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    args.rows, args.transitions, args.seed, args.backend_stopped = (
        320,
        32768,
        240926,
        False,
    )
    if (
        not args.allow_gpu
        or args.output.exists()
        or len(args.stages) != len(set(args.stages))
    ):
        parser.error("explicit GPU admission, fresh output and unique stages required")
    required = {
        "norm_partition": ("norm_candidate",),
        "minimal_norm": ("norm_candidate", "minimal_witness"),
        "gdn_partition": ("gdn_candidate",),
        "norm_trace": ("witness_dir",),
        "captured_replay": ("activation_dir",),
    }
    for stage in args.stages:
        for name in required.get(stage, ()):
            value = getattr(args, name)
            if value is None or not value.exists():
                parser.error(f"{stage} requires an existing --{name.replace('_', '-')}")
    profile = json.loads(args.contract.read_text())
    if profile.get("id") != "qwen38-mxfp4-fp8-m1-row-invariant-v2":
        parser.error("arithmetic contract identity changed")
    manifest = json.loads((args.release / "optimized-release.json").read_text())
    for path in reversed(manifest["pythonpath"]):
        sys.path.append(path)
    from probe_eager_m1_independent import device_memory, idle

    idle(args.api)
    _, free = device_memory()
    if free < (args.memory_mib + 256) * 2**20:
        raise RuntimeError("insufficient free VRAM for isolated probe")
    from vllm.config import VllmConfig, set_current_vllm_config

    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease

    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (
        gpu_lease(args.output.parent / ("lease-" + args.output.stem)),
        set_current_vllm_config(VllmConfig()),
    ):
        return ContractAudit(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
