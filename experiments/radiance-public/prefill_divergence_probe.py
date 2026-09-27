"""Isolated-worker capture of identical prefill/decode token positions.

Synthetic fixtures only. Tensor/token payloads stay in the private run directory;
RPC returns counts and identities. This module is not part of the serving bundle.
"""

import hashlib
import json
from pathlib import Path


class PrefillDivergenceProbe:
    def qwen_prefill_restore_original(self):
        if getattr(self, "_prefill_capture", None) is not None:
            raise ValueError("cannot change a live capture")
        for name in ("_prefill_candidate_hooks", "_prefill_norm_intervention"):
            hooks = getattr(self, name, None)
            if hooks:
                hooks.close()
                setattr(self, name, None)
        self.qwen_prefill_intervene_m1(False)
        self.qwen_prefill_intervene_gemm(False)
        # A CUDA registration creates a dispatcher entry even if the original
        # custom op only had a device-agnostic implementation. Restoring the
        # Python dictionary alone leaves that dispatcher looking up a missing
        # key. Explicitly rebind the original fallback as well.
        import torch

        import radiance_mxfp4 as gemm

        for op in (
            gemm.mxfp4_linear_pq,
            torch._library.custom_ops.OPDEFS["qwen_d7_qualified::gemma_residual"],
        ):
            if "cuda" not in op._backend_fns:
                op.register_kernel("cuda", op._backend_fns.get(None, op._init_fn))
        return {"production_baseline_restored": True}

    def qwen_prefill_install_runtime(self, attention_build, projection_build):
        from prefill_alignment_runtime import install

        from qwen_r9700_lab.conformance_instrumentation import HookSet

        self.qwen_prefill_restore_original()
        self.qwen_prefill_intervene_norm()
        hooks = HookSet()
        entry = {
            "attention": {"build": attention_build},
            "projection": {"build": projection_build},
        }
        result = install(entry, hooks, verify=False)
        self._prefill_candidate_hooks = hooks
        return result

    def qwen_prefill_attention_matrix(self, build, capture, cases, destination):
        """Independent M1 workgroups, scrambled physical pages and long contexts."""
        import functools

        import torch
        from prefill_attention_alignment import AlignedPrefillAttention
        from stock_m1_attention import m1_splits

        import radiance_r4d_attn as native

        path = Path(destination)
        if not str(path).startswith("/prefill-diagnosis/") or path.exists():
            raise ValueError("new isolated result path required")
        data = torch.load(
            Path(capture) / "prefill/first-attention.pt", weights_only=True
        )
        candidate = AlignedPrefillAttention(build)
        device = torch.device("cuda")
        gen = torch.Generator(device=device).manual_seed(271828)
        scratch = torch.empty(
            max(32 * 24 * 32 * 1032, candidate.manifest["scratch_bytes"]),
            device=device,
            dtype=torch.uint8,
        )
        results = []
        for case in cases:
            width, length = case["rows"], case["context"]
            pages = (length + 15) // 16
            raw = data["kv_bytes"]
            logical = raw.repeat((pages + raw.shape[0] - 1) // raw.shape[0], 1, 1, 1)[
                :pages
            ].contiguous()
            # Different physical allocations may represent the same logical KV.
            order = torch.randperm(
                pages, generator=torch.Generator().manual_seed(314159 + pages)
            )
            kv = torch.empty_like(logical)
            kv[order] = logical
            if case.get("bf16"):
                kv = kv.view(torch.float8_e4m3fn).to(torch.bfloat16)
            del logical, raw
            torch.cuda.empty_cache()
            kv = kv.to(device)
            table = order.to(device=device, dtype=torch.int32)[None].contiguous()
            lengths = torch.tensor([length], device=device, dtype=torch.int32)
            q = (
                data["query"]
                .to(device)
                .repeat(
                    (width + data["query"].shape[0] - 1) // data["query"].shape[0], 1, 1
                )[:width]
                .contiguous()
            )
            if case.get("random"):
                q = torch.randn(
                    q.shape, device=device, dtype=torch.bfloat16, generator=gen
                ) * case.get("amplitude", 1.0)
            scales = [
                torch.tensor(v, device=device, dtype=torch.float32)
                for v in ([0.5, 1, 2, 1.25], [1, 0.75, 2, 0.25])
            ]
            output, reference, old = [torch.empty_like(q) for _ in range(3)]
            start = 0
            while start < width:
                splits = m1_splits(length - width + start + 1)
                end = start + 1
                while (
                    end < min(width, start + 32)
                    and m1_splits(length - width + end + 1) == splits
                ):
                    end += 1
                rows = end - start
                bt = table.expand(rows, -1).contiguous()
                lens = (
                    lengths
                    - width
                    + torch.arange(start + 1, end + 1, device=device, dtype=torch.int32)
                )
                ks, vs = [s.repeat(rows) for s in scales]
                native._DECODE[int(case.get("bf16", False))](
                    q[start:].data_ptr(),
                    kv.data_ptr(),
                    bt.data_ptr(),
                    lens.data_ptr(),
                    reference[start:].data_ptr(),
                    ks.data_ptr(),
                    vs.data_ptr(),
                    scratch.data_ptr(),
                    rows,
                    1,
                    24,
                    4,
                    256,
                    16,
                    pages,
                    kv.stride(0),
                    kv.stride(1),
                    1 / 16,
                    splits,
                    length,
                    torch.cuda.current_stream().cuda_stream,
                )
                start = end

            aligned = functools.partial(
                candidate,
                q,
                kv,
                table,
                lengths,
                scratch,
                output,
                ks=scales[0],
                vs=scales[1],
            )
            original = functools.partial(
                native._PREFILL[int(case.get("bf16", False))],
                q.data_ptr(),
                kv.data_ptr(),
                table.data_ptr(),
                lengths.data_ptr(),
                old.data_ptr(),
                scales[0].data_ptr(),
                scales[1].data_ptr(),
                scratch.data_ptr(),
                1,
                width,
                24,
                4,
                256,
                16,
                pages,
                kv.stride(0),
                kv.stride(1),
                1 / 16,
                0,
                length,
                torch.cuda.current_stream().cuda_stream,
            )

            aligned()
            result = {
                **case,
                "differences": int((output != reference).sum()),
                "elements": output.numel(),
                "finite": bool(output.isfinite().all()),
            }
            if case.get("timing"):
                for name, fn in (("original", original), ("candidate", aligned)):
                    for _ in range(2):
                        fn()
                    begin, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    begin.record()
                    for _ in range(5):
                        fn()
                    end.record()
                    end.synchronize()
                    result[name + "_milliseconds"] = begin.elapsed_time(end) / 5
            results.append(result)
            path.write_text(
                json.dumps(
                    {"build": candidate.manifest["sha256"], "cases": results}, indent=2
                )
                + "\n"
            )
            del q, kv, table, lengths, reference, output, old, aligned, original
        return results

    def qwen_prefill_install_candidates(self, attention_build, gemm_build):
        import functools

        import torch
        from prefill_attention_alignment import AlignedPrefillAttention
        from prefill_gemm_alignment import AlignedPrefillGemm

        import radiance_mxfp4 as gemm
        import radiance_r4d_attn as native
        from qwen_r9700_lab.conformance_instrumentation import HookSet

        self.qwen_prefill_intervene_m1(False)
        self.qwen_prefill_intervene_gemm(False)
        old = getattr(self, "_prefill_candidate_hooks", None)
        if old:
            old.close()
        attention = AlignedPrefillAttention(attention_build)
        projection = AlignedPrefillGemm(gemm_build)
        scratch = torch.empty(
            attention.manifest["scratch_bytes"], device="cuda", dtype=torch.uint8
        )
        hooks = HookSet()
        original = native.R4DAttentionImpl.forward

        @functools.wraps(original)
        def forward(impl, layer, query, key, value, kv, md, output, *args, **kwargs):
            plan = getattr(md, "r4d_plan", ())
            if not plan or all(row[2] <= 8 for row in plan):
                return original(
                    impl, layer, query, key, value, kv, md, output, *args, **kwargs
                )
            if (
                len(plan) != 1
                or plan[0][:2] != (0, 1)
                or plan[0][3] != 0
                or not md.causal
            ):
                raise ValueError("aligned prefill requires one causal sequence")
            width = plan[0][2]
            scales = [
                None
                if float(getattr(layer, name, 1.0)) == 1.0
                else torch.full(
                    (4,),
                    float(getattr(layer, name)),
                    device=query.device,
                    dtype=torch.float32,
                )
                for name in ("_k_scale_float", "_v_scale_float")
            ]
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
            return output

        hooks.replace(native.R4DAttentionImpl, "forward", forward)
        op = gemm.mxfp4_linear_pq
        previous = op._backend_fns.get("cuda", op._init_fn)
        hooks.replace(op, "_backend_fns", dict(op._backend_fns))

        @op.register_kernel("cuda")
        def aligned(q, scale, weight, weight_scale, ref):
            if q.shape[0] > 8 and weight.shape[0] == 5120 and ref.numel() == 5120:
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

        self._prefill_candidate_hooks = hooks
        return {
            "attention": attention.manifest["sha256"],
            "gemm": projection.manifest["sha256"],
        }

    def qwen_prefill_intervene_norm(self):
        import torch

        from qwen_r9700_lab.conformance_instrumentation import HookSet

        op = torch._library.custom_ops.OPDEFS["qwen_d7_qualified::gemma_residual"]
        original = op._init_fn
        cells = dict(
            zip(
                original.__code__.co_freevars,
                (cell.cell_contents for cell in original.__closure__),
            )
        )
        candidate = cells["residual_candidate"]
        hooks = HookSet()
        hooks.replace(op, "_backend_fns", dict(op._backend_fns))

        @op.register_kernel("cuda")
        def aligned(x, residual, weight, eps, key):
            return (
                candidate(x, residual, weight, eps)
                if x.shape[0] > 8
                else original(x, residual, weight, eps, key)
            )

        self._prefill_norm_intervention = hooks
        return {"enabled": True, "norm_manifest": candidate.manifest["sha256"]}

    def qwen_prefill_aligned_gemm_probe(self, build):
        import torch
        from prefill_gemm_alignment import AlignedPrefillGemm

        import radiance_mxfp4 as gemm

        candidate = AlignedPrefillGemm(build)
        path = Path(
            "/prefill-diagnosis/intervention-stages-1000/prefill/stages/group-0000.pt"
        )
        saved = torch.load(path, weights_only=True)
        model = getattr(
            self.model_runner.model, "language_model", self.model_runner.model
        )
        layer = model.model.layers[20].mlp.down_proj
        results = []
        for rows in (8, 65, 256, 1003, 1648):
            q = saved["252.before.args.0"].view(torch.float8_e4m3fn).cuda()
            scale = saved["252.before.args.1"].reshape(-1).cuda()
            q = (
                q.view(torch.uint8)
                .repeat((rows + 7) // 8, 1)[:rows]
                .contiguous()
                .view(torch.float8_e4m3fn)
            )
            scale = scale.repeat((rows + 7) // 8)[:rows].contiguous()
            reference = torch.empty((rows, 5120), device=q.device, dtype=torch.bfloat16)
            for first in range(0, rows, 8):
                gemm._ext.launch(
                    q[first:].data_ptr(),
                    layer.weight.data_ptr(),
                    layer.weight_scale.data_ptr(),
                    layer.radiance_wref.data_ptr(),
                    scale[first:].data_ptr(),
                    reference[first:].data_ptr(),
                    min(8, rows - first),
                    5120,
                    q.shape[1],
                    torch.cuda.current_stream().cuda_stream,
                )
            for tiled in (False, True):
                out = candidate(
                    q,
                    scale,
                    layer.weight,
                    layer.weight_scale,
                    layer.radiance_wref,
                    tiled=tiled,
                    wperm=gemm.WPERM,
                )
                results.append(
                    {
                        "rows": rows,
                        "tiled": tiled,
                        "different": int((out != reference).sum()),
                        "elements": out.numel(),
                    }
                )
        (Path(build) / "captured-probe.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        return results

    def qwen_prefill_runtime_info(self):
        import inspect

        import radiance_mxfp4 as gemm

        scan = self._qwen_performance_repairs.repairs.prefill.scan
        op = gemm.mxfp4_linear_pq
        init = getattr(op, "_init_fn", None)
        original_ext = init.__globals__.get("_ext") if init else None
        return {
            "gemm_module": gemm.__file__,
            "gemm_launch": repr(gemm._ext.launch),
            "op_ext_same": original_ext is gemm._ext,
            "op_init": repr(init),
            "op_backend_fns": repr(getattr(op, "_backend_fns", None)),
            "gemm_tiled_launch": repr(gemm._ext.launch_at),
            "scan": str(type(scan)),
            "scan_source": inspect.getsourcefile(type(scan)),
            "tiled_min_m": gemm.A_TILED_MIN_M,
            "gemm_probe_calls": getattr(self, "_prefill_gemm_calls", {}),
        }

    def qwen_prefill_aligned_attention_probe(self, build, capture):
        import torch
        from prefill_attention_alignment import AlignedPrefillAttention
        from stock_m1_attention import m1_splits

        import radiance_r4d_attn as native

        data = torch.load(
            Path(capture) / "prefill/first-attention.pt", weights_only=True
        )
        candidate = AlignedPrefillAttention(build)
        q, kv = data["query"].cuda(), data["kv_bytes"].cuda()
        width, length = q.shape[0], data["length"]
        table = torch.arange(kv.shape[0], device=q.device, dtype=torch.int32)[None]
        lengths = torch.tensor([length], device=q.device, dtype=torch.int32)
        scratch = torch.empty(
            max(64 * 24 * 32 * 1032, candidate.manifest["scratch_bytes"]),
            device=q.device,
            dtype=torch.uint8,
        )
        scales = [
            torch.full((4,), data[name], device=q.device, dtype=torch.float32)
            for name in ("k_scale", "v_scale")
        ]
        output, reference, old = [torch.empty_like(q) for _ in range(3)]
        start = 0
        while start < width:
            splits = m1_splits(length - width + start + 1)
            end = start + 1
            while (
                end < min(width, start + 64)
                and m1_splits(length - width + end + 1) == splits
            ):
                end += 1
            rows = end - start
            bt = table.expand(rows, -1).contiguous()
            lens = (
                lengths
                - width
                + torch.arange(start + 1, end + 1, device=q.device, dtype=torch.int32)
            )
            ks, vs = [s.repeat(rows) for s in scales]
            native._DECODE[0](
                q[start:].data_ptr(),
                kv.data_ptr(),
                bt.data_ptr(),
                lens.data_ptr(),
                reference[start:].data_ptr(),
                ks.data_ptr(),
                vs.data_ptr(),
                scratch.data_ptr(),
                rows,
                1,
                24,
                4,
                256,
                16,
                table.shape[1],
                kv.stride(0),
                kv.stride(1),
                1 / 16,
                splits,
                length,
                torch.cuda.current_stream().cuda_stream,
            )
            start = end

        def aligned():
            candidate(
                q, kv, table, lengths, scratch, output, ks=scales[0], vs=scales[1]
            )

        def original():
            native._PREFILL[0](
                q.data_ptr(),
                kv.data_ptr(),
                table.data_ptr(),
                lengths.data_ptr(),
                old.data_ptr(),
                scales[0].data_ptr(),
                scales[1].data_ptr(),
                scratch.data_ptr(),
                1,
                width,
                24,
                4,
                256,
                16,
                table.shape[1],
                kv.stride(0),
                kv.stride(1),
                1 / 16,
                0,
                length,
                torch.cuda.current_stream().cuda_stream,
            )

        aligned()
        original()
        report = {
            "rows": width,
            "context": length,
            "candidate_differences": int((output != reference).sum()),
            "old_differences": int((old != reference).sum()),
            "elements": output.numel(),
        }
        for name, fn in (("original", original), ("candidate", aligned)):
            for _ in range(3):
                fn()
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            begin.record()
            for _ in range(20):
                fn()
            end.record()
            end.synchronize()
            report[name + "_milliseconds"] = begin.elapsed_time(end) / 20
        torch.save(
            {
                "candidate": output.cpu(),
                "reference": reference.cpu(),
                "original": old.cpu(),
            },
            Path(build) / "captured-outputs.pt",
        )
        (Path(build) / "captured-probe.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        return report

    def qwen_prefill_intervene_gemm(self, enabled=True):
        """Diagnostic only: retain the decode split/reduction for output projections."""
        import radiance_mxfp4 as gemm
        from qwen_r9700_lab.conformance_instrumentation import HookSet

        current = getattr(self, "_prefill_gemm_intervention", None)
        if current:
            current.close()
            self._prefill_gemm_intervention = None
        if not enabled:
            return {"enabled": False}
        hooks = HookSet()
        original = gemm._ext.launch
        self._prefill_gemm_calls = {}
        # The production activation-layout override handles selected prefill
        # widths through launch_at. Exercise the numerical intervention there
        # too, rather than accidentally comparing repaired and unrepaired rows.
        op = gemm.mxfp4_linear_pq
        hooks.replace(op, "_backend_fns", dict(op._backend_fns))
        op.register_kernel("cuda", op._init_fn)

        def launch(a, w, ws, wr, scale, out, m, n, k, stream):
            key = f"{m}/{n}/{k}"
            self._prefill_gemm_calls[key] = self._prefill_gemm_calls.get(key, 0) + 1
            if m <= 24 or n != 5120 or not wr:
                return original(a, w, ws, wr, scale, out, m, n, k, stream)
            for start in range(0, m, 16):
                original(
                    a + start * k,
                    w,
                    ws,
                    wr,
                    scale + start * 4,
                    out + start * n * 2,
                    min(16, m - start),
                    n,
                    k,
                    stream,
                )

        hooks.replace(gemm._ext, "launch", launch)
        self._prefill_gemm_intervention = hooks
        return {"enabled": True, "scope": "diagnostic M1 split-K; no production change"}

    def qwen_prefill_gemm_probe(self, capture, layer_index=20, event=252, first=0):
        import torch

        import radiance_mxfp4 as gemm

        saved = torch.load(
            Path(capture) / "prefill/stages/group-0000.pt", weights_only=True
        )
        qkey = (
            f"{event}.before.args.0"
            if f"{event}.before.args.0" in saved
            else f"{event - 1}.after.mutable.result"
        )
        skey = (
            f"{event}.before.args.1"
            if f"{event}.before.args.1" in saved
            else f"{event - 1}.after.mutable.scale"
        )
        x = saved[qkey][first : first + 8].view(torch.float8_e4m3fn).cuda()
        scale = saved[skey][first : first + 8].cuda()
        model = getattr(
            self.model_runner.model, "language_model", self.model_runner.model
        )
        layer = model.model.layers[layer_index].mlp.down_proj
        ref = layer.radiance_wref
        results = {}
        outputs = {}
        for rows in (1, 8, 16, 24, 32, 64, 65, 256, 1003):
            xx = torch.zeros(
                (max(rows, 8), x.shape[1]), device=x.device, dtype=torch.float8_e4m3fn
            )
            xx.view(torch.uint8)[:8].copy_(x.view(torch.uint8))
            ss = torch.ones((max(rows, 8),), device=x.device, dtype=torch.float32)
            ss[:8].copy_(scale.reshape(-1))
            out = torch.empty(
                (8 if rows == 1 else rows, 5120), device=x.device, dtype=torch.bfloat16
            )
            for start in range(8) if rows == 1 else (0,):
                gemm._ext.launch(
                    xx[start:].data_ptr(),
                    layer.weight.data_ptr(),
                    layer.weight_scale.data_ptr(),
                    ref.data_ptr(),
                    ss[start:].data_ptr(),
                    out[start:].data_ptr(),
                    rows,
                    5120,
                    x.shape[1],
                    torch.cuda.current_stream().cuda_stream,
                )
            outputs[rows] = out[:8].cpu()
            results[str(rows)] = {
                "different_from_m1": int((outputs[1] != outputs[rows]).sum()),
                "different_from_capture": int(
                    (
                        saved[f"{event}.after.result"][first : first + 8]
                        != outputs[rows]
                    ).sum()
                ),
            }
        torch.save(outputs, Path(capture) / "gemm-output-by-width.pt")
        return results

    def qwen_prefill_intervene_m1(self, enabled=True):
        """Diagnostic only: use independent M1 workgroups for every prefill row."""
        import functools

        import torch
        from stock_m1_attention import m1_splits

        import radiance_r4d_attn as native
        from qwen_r9700_lab.conformance_instrumentation import HookSet

        current = getattr(self, "_prefill_m1_intervention", None)
        if current:
            current.close()
            self._prefill_m1_intervention = None
        if not enabled:
            return {"enabled": False}
        scratch = torch.empty(64 * 24 * 32 * 1032, device="cuda", dtype=torch.uint8)
        hooks = HookSet()
        original = native.R4DAttentionImpl.forward

        @functools.wraps(original)
        def forward(impl, layer, query, key, value, kv, md, output, *args, **kwargs):
            plan = getattr(md, "r4d_plan", ())
            if not plan or all(row[2] <= 8 for row in plan):
                return original(
                    impl, layer, query, key, value, kv, md, output, *args, **kwargs
                )
            if len(plan) != 1 or plan[0][:2] != (0, 1) or plan[0][3] != 0:
                raise ValueError("diagnostic prefill requires exactly one sequence")
            width = plan[0][2]
            start = 0
            while start < width:
                splits = m1_splits(md.r4d_max_ctx - width + start + 1)
                end = start + 1
                while (
                    end < min(width, start + 64)
                    and m1_splits(md.r4d_max_ctx - width + end + 1) == splits
                ):
                    end += 1
                rows = end - start
                table = md.block_table[:1].expand(rows, -1).contiguous()
                lengths = (
                    md.seq_lens[:1]
                    - width
                    + torch.arange(
                        start + 1, end + 1, device=query.device, dtype=torch.int32
                    )
                )
                scales = [
                    torch.full(
                        (rows, 4),
                        float(getattr(layer, name, 1.0)),
                        device=query.device,
                        dtype=torch.float32,
                    )
                    for name in ("_k_scale_float", "_v_scale_float")
                ]
                native._DECODE[int(kv.dtype == torch.bfloat16)](
                    query[start:end].data_ptr(),
                    kv.data_ptr(),
                    table.data_ptr(),
                    lengths.data_ptr(),
                    output[start:end].data_ptr(),
                    scales[0].data_ptr(),
                    scales[1].data_ptr(),
                    scratch.data_ptr(),
                    rows,
                    1,
                    24,
                    4,
                    256,
                    16,
                    table.shape[1],
                    kv.stride(0),
                    kv.stride(1),
                    impl.scale,
                    splits,
                    md.r4d_max_ctx,
                    torch.cuda.current_stream().cuda_stream,
                )
                start = end
            return output

        hooks.replace(native.R4DAttentionImpl, "forward", forward)
        self._prefill_m1_intervention = hooks
        return {
            "enabled": True,
            "scope": "diagnostic M1 attention workgroups; no production change",
        }

    def qwen_prefill_probe_reload(self):
        import importlib
        import sys
        import types

        if getattr(self, "_prefill_capture", None) is not None:
            raise ValueError("cannot reload an active capture")
        for name in (
            "prefill_attention_alignment",
            "prefill_gemm_alignment",
            "prefill_alignment_runtime",
        ):
            if name in sys.modules:
                importlib.reload(sys.modules[name])
        if "/prefill-diagnosis" not in sys.path:
            sys.path.append("/prefill-diagnosis")
        module = importlib.reload(sys.modules[__name__])
        for name, method in vars(module.PrefillDivergenceProbe).items():
            if name.startswith("qwen_prefill_"):
                setattr(self, name, types.MethodType(method, self))
        return {
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        }

    def qwen_prefill_capture_begin(
        self,
        destination,
        first_position=0,
        corpus=None,
        stages=False,
        attention=False,
        gdn=False,
    ):
        import torch

        from qwen_r9700_lab.conformance_instrumentation import HookSet
        from qwen_r9700_lab.conformance_topk import ReplaySchedule

        if getattr(self, "_prefill_capture", None) is not None:
            raise ValueError("a capture is already active")
        path = Path(destination)
        if not str(path).startswith("/prefill-diagnosis/") or path.exists():
            raise ValueError("capture needs a new isolated run path")
        path.mkdir(mode=0o700, parents=True)
        runner = self.model_runner
        hooks = HookSet()
        state = {
            "path": path,
            "batches": [],
            "hooks": hooks,
            "positions": [],
            "ids": [],
        }
        schedule = None
        if corpus is not None:
            data = json.loads(Path(corpus).read_text())
            if data.get("synthetic") is not True:
                raise ValueError("only explicitly synthetic fixtures are admitted")
            schedule = ReplaySchedule(data["prefix"], data["output"], speculation=True)
        original_prepare = runner.prepare_inputs
        original_sample = runner.sample

        def prepare(*args, **kwargs):
            batch = original_prepare(*args, **kwargs)
            if batch.num_reqs != 1:
                raise ValueError("diagnosis requires exactly one request")
            state["positions"] = batch.positions[: batch.num_tokens].cpu().tolist()
            state["ids"] = batch.input_ids[: batch.num_tokens].cpu().tolist()
            state["gdn_layer"] = 0
            if schedule is not None:
                schedule.check_inputs(state["positions"], state["ids"])
            return batch

        def sample(hidden, batch, grammar):
            if grammar is not None:
                raise ValueError("grammar is outside this comparison")
            selected = [
                i for i, p in enumerate(state["positions"]) if p >= first_position
            ]
            if selected:
                indices = torch.tensor(selected, device=hidden.device)
                payload = {
                    "positions": [state["positions"][i] for i in selected],
                    "input_ids": [state["ids"][i] for i in selected],
                    "hidden": hidden.index_select(0, indices).detach().cpu(),
                }
                filename = f"batch-{len(state['batches']):05d}.pt"
                torch.save(payload, path / filename)
                state["batches"].append({"file": filename, "rows": len(selected)})
            result, ns, nr = original_sample(hidden, batch, grammar)
            if schedule is not None and int(ns[0].item()):
                step = schedule.commit(int(batch.num_draft_tokens))
                result.sampled_token_ids.fill_(-1)
                for j, token in enumerate(step["tokens"]):
                    result.sampled_token_ids[0, j] = token
                ns.fill_(step["count"])
                nr.fill_(step["reject"])
            return result, ns, nr

        hooks.replace(runner, "prepare_inputs", prepare)
        hooks.replace(runner, "sample", sample)
        if schedule is not None:
            original_propose = runner.speculator.propose

            def propose(*args, **kwargs):
                result = original_propose(*args, **kwargs)
                if tuple(result.shape) != (1, 7):
                    raise ValueError("expected one D7 proposal group")
                for j, token in enumerate(schedule.proposals()):
                    result[0, j] = token
                return result

            hooks.replace(runner.speculator, "propose", propose)
        if stages:
            import functools
            from types import SimpleNamespace

            from isolated_d7_capture import BoundaryCapture, row_tensor

            stage_window = stages.get("window", 8) if isinstance(stages, dict) else 8
            stage_limit = (
                stages.get("limit", 10000) if isinstance(stages, dict) else 10000
            )

            class SelectedCapture(BoundaryCapture):
                def positions(inner):
                    actual = state["positions"]
                    if inner.busy or not state.get("target"):
                        return None
                    return (
                        tuple(actual)
                        if any(
                            first_position <= p < first_position + stage_window
                            for p in actual
                        )
                        else None
                    )

                def invoke(
                    inner, name, function, args, kwargs, schema=None, signature=None
                ):
                    if len(inner.events) >= stage_limit:
                        return function(*args, **kwargs)
                    return super().invoke(
                        name, function, args, kwargs, schema, signature
                    )

                def save_tree(inner, value, key, rows, saved):
                    if isinstance(value, torch.Tensor):
                        if not row_tensor(value, rows):
                            return
                        if stage_window > 8 and ".before." in key:
                            return
                        indices = [
                            i
                            for i, p in enumerate(state["positions"])
                            if first_position <= p < first_position + stage_window
                        ]
                        if value.element_size() == 1 and value.ndim == 2:
                            import radiance_mxfp4 as gemm

                            if value.data_ptr() in gemm._A_TILED:
                                m, k = gemm._A_TILED[value.data_ptr()]
                                raw = value.view(torch.uint8).as_strided(
                                    (((m + 15) // 16) * 16 * k,), (1,)
                                )
                                value = (
                                    raw.view(-1, k // 16, 2, 16, 8)
                                    .permute(0, 3, 1, 2, 4)
                                    .reshape(-1, k)[:m]
                                    .view(value.dtype)
                                )
                        if value.shape[0] == rows * 48:
                            value = value.reshape(rows, 48, *value.shape[1:])[indices]
                            value = value.reshape(len(indices) * 48, *value.shape[2:])
                        else:
                            # Index FP8 storage through its byte representation.
                            dtype = value.dtype
                            value = (
                                value.view(torch.uint8)[indices].view(dtype)
                                if value.element_size() == 1
                                else value[indices]
                            )
                        super().save_tree(value, key, len(indices), saved)
                    elif isinstance(value, (tuple, list)):
                        for i, child in enumerate(value):
                            inner.save_tree(child, f"{key}.{i}", rows, saved)
                    elif isinstance(value, dict):
                        for name, child in value.items():
                            inner.save_tree(child, f"{key}.{name}", rows, saved)

            capture = SelectedCapture(
                SimpleNamespace(runner=runner),
                SimpleNamespace(draft=False),
                path / "stages",
            )
            capture.attach()
            original_forward = runner.model.forward

            @functools.wraps(original_forward)
            def target_forward(*args, **kwargs):
                state["target"] = True
                try:
                    return original_forward(*args, **kwargs)
                finally:
                    state["target"] = False

            hooks.replace(runner.model, "forward", target_forward)
            state["stage_capture"] = capture
        if stages or attention or gdn:
            state["graph_candidates"] = runner.cudagraph_manager._candidates
            runner.cudagraph_manager._candidates = {}
        if attention:
            import functools

            import radiance_r4d_attn as native

            original_attention = native.R4DAttentionImpl.forward

            @functools.wraps(original_attention)
            def capture_attention(
                impl, layer, query, key, value, kv, md, output, *args, **kwargs
            ):
                selected = any(p >= first_position for p in state["positions"])
                save = selected and not state.get("attention_saved")
                if save:
                    state["attention_saved"] = True
                    length = int(md.seq_lens[0].item())
                    pages = (length + 15) // 16
                    indices = md.block_table[0, :pages].long()
                    payload = {
                        "query": query.detach().cpu(),
                        "kv_bytes": kv.view(torch.uint8).index_select(0, indices).cpu(),
                        "kv_dtype": str(kv.dtype),
                        "positions": state["positions"],
                        "length": length,
                        "max_context": md.r4d_max_ctx,
                        "plan": md.r4d_plan,
                        "k_scale": float(getattr(layer, "_k_scale_float", 1.0)),
                        "v_scale": float(getattr(layer, "_v_scale_float", 1.0)),
                    }
                result = original_attention(
                    impl, layer, query, key, value, kv, md, output, *args, **kwargs
                )
                if save:
                    payload["output"] = output.detach().cpu()
                    torch.save(payload, path / "first-attention.pt")
                return result

            hooks.replace(native.R4DAttentionImpl, "forward", capture_attention)
        if gdn:
            scan = self._qwen_performance_repairs.repairs.prefill.scan
            original_run = scan.run

            def capture_scan(initial, packed, a, b, a_log, dt_bias, **kwargs):
                selected = any(p >= first_position for p in state["positions"])
                save = (
                    selected and state["gdn_layer"] == 2 and not state.get("gdn_saved")
                )
                state["gdn_layer"] += 1
                if save:
                    state["gdn_saved"] = True
                    payload = {
                        k: v.detach().cpu()
                        for k, v in {
                            "initial": initial,
                            "packed": packed,
                            "a": a,
                            "b": b,
                            "a_log": a_log,
                            "dt_bias": dt_bias,
                        }.items()
                    }
                    payload["positions"] = state["positions"]
                result = original_run(initial, packed, a, b, a_log, dt_bias, **kwargs)
                if save:
                    payload["outputs"] = result.outputs.detach().cpu()
                    payload["final_state"] = result.final_state.detach().cpu()
                    torch.save(payload, path / "third-gdn.pt")
                return result

            hooks.replace(scan, "run", capture_scan)
        state["schedule"] = schedule
        self._prefill_capture = state
        return {
            "status": "CAPTURING",
            "compiled": not runner.model_config.enforce_eager,
        }

    def qwen_prefill_capture_finish(self):
        state = self._prefill_capture
        state["hooks"].close()
        if "stage_capture" in state:
            capture = state["stage_capture"]
            capture.hooks.close()
            capture.flush()
        if "graph_candidates" in state:
            self.model_runner.cudagraph_manager._candidates = state["graph_candidates"]
        self._prefill_capture = None
        schedule = state["schedule"]
        result = {
            "schema": "urn:coherence:prefill-hidden-capture:v1",
            "batches": state["batches"],
            "forced_complete": schedule.done if schedule is not None else None,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        (state["path"] / "capture.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    def qwen_prefill_compare(self, reference, candidate):
        import torch

        from qwen_r9700_lab.conformance_topk import (
            aggregate,
            compare_rows,
            summarize_logits,
        )

        def load(root):
            path = Path(root)
            if not str(path).startswith("/prefill-diagnosis/"):
                raise ValueError("not an isolated capture")
            capture = json.loads((path / "capture.json").read_text())
            rows = {}
            for entry in capture["batches"]:
                batch = torch.load(path / entry["file"], weights_only=True)
                for i, p in enumerate(batch["positions"]):
                    if p in rows:
                        raise ValueError(
                            "repeated position requires explicit accepted-prefix selection"
                        )
                    rows[p] = (batch["input_ids"][i], batch["hidden"][i])
            return rows

        left, right = load(reference), load(candidate)
        positions = sorted(left.keys() & right.keys())
        if not positions:
            raise ValueError("no common positions")
        for p in positions:
            if left[p][0] != right[p][0]:
                raise ValueError("different input tokens")
        a = torch.stack([left[p][1] for p in positions])
        b = torch.stack([right[p][1] for p in positions])
        diff = a.float() - b.float()
        target = getattr(
            self.model_runner.model, "language_model", self.model_runner.model
        )
        head = self._qwen_performance_repairs.head
        compared = []
        for start in range(0, len(positions), 8):
            n = min(8, len(positions) - start)
            tensors = []
            for x in (a, b):
                batch = torch.zeros((8, x.shape[1]), dtype=x.dtype, device="cuda")
                batch[:n].copy_(x[start : start + n])
                logits = head(target.lm_head.weight, batch).float().cpu().numpy()
                tensors.append([summarize_logits(row) for row in logits[:n]])
            compared.extend(compare_rows(x, y) for x, y in zip(*tensors, strict=True))
        result = {
            "positions": len(positions),
            "first_position": positions[0],
            "last_position": positions[-1],
            "hidden_exact_rows": int((a == b).all(dim=1).sum()),
            "hidden_different_elements": int((a != b).sum()),
            "hidden_elements": a.numel(),
            "hidden_max_absolute_error": float(diff.abs().max()),
            "hidden_rmse": float(diff.square().mean().sqrt()),
            "hidden_relative_l2": float(diff.norm() / a.float().norm()),
            "logits": aggregate(compared),
            "head": "same qualified full BF16 M8 head applied to both captured hidden tensors",
        }
        (Path(candidate) / "comparison.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        return result
