"""Disposable-worker metadata for arithmetic-preserving speed investigations."""

import os

from optimized_d7_worker import OptimizedWorker


class SpeedCandidateWorker(OptimizedWorker):
    def load_model(self, **kwargs):
        extended = {}
        if os.environ.get("QWEN_SPEED_TP1_DRAFT_W4") == "1":
            import radiance_w4
            from qwen_r9700_lab.conformance_topk import require

            require(
                self.vllm_config.parallel_config.tensor_parallel_size == 1,
                "experimental drafter shapes require TP1",
            )
            require(radiance_w4.ENABLED, "existing W4 drafter path is disabled")
            extended = {
                (6144, 5120): (3072, 5120),
                (5120, 4096): (5120, 2048),
                (34816, 5120): (17408, 5120),
                (5120, 17408): (5120, 8704),
            }
            for table in (radiance_w4._CFG, radiance_w4._CFG_A8):
                require(
                    not set(extended).intersection(table), "TP1 shapes already exist"
                )
                for shape, original in extended.items():
                    table[shape] = list(table[original])
        result = super().load_model(**kwargs)
        if extended:
            from qwen_r9700_lab.conformance_topk import require

            runner = self.model_runner
            target = [
                name
                for name, module in runner.model.named_modules()
                if hasattr(module, "_radiance_w4")
            ]
            require(not target, "drafter quantization must never claim target layers")
            converted = {
                name: list(module._radiance_w4)
                for name, module in runner.speculator.model.named_modules()
                if getattr(module, "_radiance_w4", None) in extended
            }
            require(
                len(converted) == 20, "not all five TP1 draft layers were converted"
            )
            self._qwen_speed_draft_w4 = {
                "converted_projections": converted,
                "target_layers_changed": target,
                "scope": "proposal model only; acceptance and target qualification required",
            }
        return result

    def qwen_speed_forced_begin(self, root, task_path):
        """Use the current explicit full-head qualification adapter.

        The frozen release predates the adapter's request-level full-head
        admission. Its replacement still rejects masked/non-finite logits.
        The forced-replay driver requests top_k=129, beyond the fast head's
        admitted 128, to obtain full logits without forcing logprob metadata
        to describe substituted tokens. This is an untimed operator check;
        natural performance runs retain Global-512 and production top_k=40.
        """
        import optimized_d7_worker
        from speed_equivalence_adapter import EquivalenceProbe

        from qwen_r9700_lab.conformance_instrumentation import HookSet

        class CompiledProbe(EquivalenceProbe):
            validate_execution = staticmethod(
                optimized_d7_worker.CompiledEquivalenceProbe.validate_execution
            )

        hooks = HookSet()
        try:
            hooks.replace(
                optimized_d7_worker, "CompiledEquivalenceProbe", CompiledProbe
            )
            return self.qwen_optimized_begin(root, False, task_path)
        finally:
            hooks.close()

    def qwen_speed_set_full_graph(self, enabled):
        """Select captured paths between isolated requests for a matched A/B.

        This experimental worker is never used by the production launcher.
        Both paths retain the same weights, allocations and compiled operators.
        """
        from vllm.config import CUDAGraphMode

        from qwen_r9700_lab.conformance_topk import require

        require(type(enabled) is bool, "full-graph selector requires a boolean")
        require(
            not hasattr(self, "_qwen_observation"),
            "cannot change graph route during a qualification capture",
        )
        manager = self.model_runner.cudagraph_manager
        require(manager._graphs_captured, "graphs have not been captured")
        require(
            any(desc.cg_mode == CUDAGraphMode.FULL for desc in manager.graphs),
            "the experimental worker has no full graph",
        )
        if not hasattr(self, "_qwen_speed_graph_candidates"):
            self._qwen_speed_graph_candidates = {
                key: list(value) for key, value in manager._candidates.items()
            }
        candidates = self._qwen_speed_graph_candidates
        manager._candidates = {
            key: [
                desc for desc in value if enabled or desc.cg_mode != CUDAGraphMode.FULL
            ]
            for key, value in candidates.items()
        }
        self._qwen_speed_full_graph = enabled
        return {"full_graph_enabled": enabled, "same_captured_allocations": True}

    def compile_or_warm_up_model(self):
        if os.environ.get("QWEN_SPEED_FULL_GRAPH_CAPTURE") != "1":
            result = super().compile_or_warm_up_model()
            self._install_speed_head_graph()
            return result
        from vllm.config import CUDAGraphMode
        from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

        import radiance_r4d_attn
        from qwen_r9700_lab.conformance_instrumentation import HookSet
        from qwen_r9700_lab.conformance_topk import require

        config = self.vllm_config.compilation_config
        require(
            list(config.cudagraph_capture_sizes) == [8],
            "experimental full capture admits only one D7 batch",
        )
        repairs = self._qwen_persistent_repairs
        require(
            repairs is not None and repairs.prefill is not None,
            "corrected GDN bindings required",
        )
        native = repairs.prefill.native
        target_attention = self._load_qualified_target_attention()
        target_gemm = self._load_qualified_target_gemm()
        gemm_enabled = [True]
        bindings = [
            (native, name, getattr(native, name))
            for name in ("conv_update", "recurrent_update", "forward_core_fused")
        ]
        bindings.append(
            (
                radiance_r4d_attn.R4DAttentionImpl,
                "forward",
                radiance_r4d_attn.R4DAttentionImpl.forward,
            )
        )
        original_capture = CudaGraphManager.capture
        model_manager = self.model_runner.cudagraph_manager
        calls = []

        def capture(manager, create_forward_fn, *args, **kwargs):
            if manager is not model_manager:
                return original_capture(manager, create_forward_fn, *args, **kwargs)

            def create(desc, *factory_args, **factory_kwargs):
                forward = create_forward_fn(desc, *factory_args, **factory_kwargs)
                if desc.cg_mode != CUDAGraphMode.FULL:
                    return forward
                require(
                    desc.num_tokens == 8,
                    "full capture must have exactly eight target rows",
                )

                def repaired_forward(*forward_args, **forward_kwargs):
                    # Keep multi-sequence piecewise startup calls outside the
                    # admitted domain. Only FULL descriptor warmup/capture uses
                    # the repaired bindings, even when cg_mode=NONE is passed
                    # to the inner forward during outer full-graph capture.
                    hooks = HookSet()
                    try:
                        for owner, name, function in bindings:
                            hooks.replace(owner, name, function)
                        if target_attention is not None:
                            hooks.replace(
                                self._qwen_performance_repairs.attention,
                                "launch",
                                target_attention,
                            )
                        if target_gemm is not None and gemm_enabled[0]:
                            owner, launch = target_gemm
                            hooks.replace(owner, "launch", launch)
                        calls.append(
                            {
                                name: function.__module__ + "." + function.__qualname__
                                for _, name, function in bindings
                            }
                        )
                        return forward(*forward_args, **forward_kwargs)
                    finally:
                        hooks.close()

                return repaired_forward

            result = original_capture(manager, create, *args, **kwargs)
            if target_gemm is not None:
                candidate = dict(manager.graphs)
                original_descs = manager._capture_descs
                manager.graphs = {
                    desc: graph
                    for desc, graph in candidate.items()
                    if desc.cg_mode != CUDAGraphMode.FULL
                }
                try:
                    manager._capture_descs = {
                        CUDAGraphMode.FULL: original_descs[CUDAGraphMode.FULL]
                    }
                    gemm_enabled[0] = False
                    original_capture(manager, create, *args, **kwargs)
                    control = dict(manager.graphs)
                    require(set(control) == set(candidate), "GEMM graph domains differ")
                    self._qwen_speed_gemm_graphs = {False: control, True: candidate}
                finally:
                    manager._capture_descs = original_descs
                    manager.graphs = candidate
                    gemm_enabled[0] = True
                self._qwen_speed_gemm_enabled = True
            return result

        hooks = HookSet()
        try:
            hooks.replace(CudaGraphManager, "capture", capture)
            self._observe_draft_attention_launch(hooks)
            result = super().compile_or_warm_up_model()
        finally:
            hooks.close()
        self._qwen_speed_capture_bindings = calls
        self._capture_draft_attention_control()
        self._install_speed_head_graph()
        self._install_speed_sampler_graph()
        return result

    def _load_qualified_target_gemm(self):
        root = os.environ.get("QWEN_SPEED_TARGET_GEMM_BUILD")
        if not root:
            return None
        import hashlib
        import importlib.util
        import json
        from pathlib import Path

        import radiance_mxfp4 as kernel
        from qwen_r9700_lab.conformance_topk import require

        root = Path(root)
        sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        build = json.loads((root / "build.json").read_bytes())
        report_path = root / "probe-001/result.json"
        report = json.loads(report_path.read_bytes())
        name = "local_split"
        binary = root / name / "radiance_mxfp4_fp8.so"
        entry = build["variants"][name]
        require(
            report["status"] == "OPERATOR_SAMPLE_CHECKED"
            and report["negative_control_detected"]
            and report["build_sha256"] == sha(root / "build.json")
            and entry["binary_sha256"] == sha(binary)
            and entry["source_sha256"] == sha(root / name / "radiance_mxfp4_fp8.hip")
            and build["parent"]["radiance_mxfp4_fp8.so"] == sha(kernel._ext.__file__),
            "GEMM qualification binding changed",
        )
        needed = {
            (shape, m) for shape in ("down", "attention_out", "gdn_out") for m in (1, 8)
        }
        cases = {
            (case["shape"], case["M"]): case for case in report["cases"] if "M" in case
        }
        cross_width = {
            case["shape"]: case
            for case in report["cases"]
            if case.get("M1_vs_M8") == name
        }
        require(
            needed <= set(cases)
            and {shape for shape, _ in needed} <= set(cross_width)
            and all(
                cases[key]["rows"] >= 320
                and cases[key]["comparisons"][name]["different"] == 0
                and cross_width[key[0]]["different"] == 0
                for key in needed
            ),
            "local split GEMM lacks complete sampled output evidence",
        )
        spec = importlib.util.spec_from_file_location(
            "speed_local_split.radiance_mxfp4_fp8", binary
        )
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        partial, counters = kernel._decode_scratch
        candidate.set_decode_scratch(
            partial.data_ptr(), partial.numel() * 4, counters.data_ptr()
        )
        original = kernel._ext.launch

        def launch(*args):
            # Only the exact, independently checked DKS=4 output shapes benefit.
            m, n, k = args[6:9]
            selected = m in (1, 8) and n == 5120 and k in (6144, 17408)
            return (candidate.launch if selected else original)(*args)

        self._qwen_speed_target_gemm_module = candidate
        self._qwen_speed_target_gemm_candidate = {
            "variant": name,
            "binary_sha256": sha(binary),
            "qualification_sha256": sha(report_path),
            "scope": "full target graph, M1/M8 down and output projections only",
        }
        return kernel._ext, launch

    def qwen_speed_set_gemm(self, enabled):
        from qwen_r9700_lab.conformance_topk import require

        require(type(enabled) is bool, "GEMM selector requires a boolean")
        require(
            not hasattr(self, "_qwen_observation"),
            "cannot switch GEMM during qualification",
        )
        require(
            hasattr(self, "_qwen_speed_gemm_graphs"), "paired GEMM captures missing"
        )
        self.model_runner.cudagraph_manager.graphs = dict(
            self._qwen_speed_gemm_graphs[enabled]
        )
        self._qwen_speed_gemm_enabled = enabled
        return {"candidate_gemm": enabled, "same_allocations": True}

    def _load_qualified_target_attention(self):
        path = os.environ.get("QWEN_SPEED_TARGET_ATTN_BUILD")
        if not path:
            return None
        import ctypes
        import hashlib
        import json
        from pathlib import Path

        from stock_m1_attention_shared import R4DArgs

        from qwen_r9700_lab.conformance_topk import require

        root = Path(path)
        sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        build = json.loads((root / "build.json").read_bytes())
        report_path = root / "probe-001/result.json"
        report = json.loads(report_path.read_bytes())
        name = "packed3_pf2"
        binary = root / (name + ".so")
        entry = build["variants"][name]
        require(
            report["status"] == "SAMPLE_CHECKED_VARIANTS_RECORDED"
            and report["build_sha256"] == sha(root / "build.json")
            and entry["binary_sha256"] == sha(binary)
            and entry["source_sha256"] == sha(root / (name + ".hip")),
            "target attention qualification binding changed",
        )
        require(
            build["parent_binary_sha256"]
            == self._qwen_performance_repairs.attention.manifest["files"][
                "candidate.so"
            ],
            "target attention qualification used another arithmetic reference",
        )
        require(
            set(report["contexts"]) == {"1024", "60000", "200000"}
            and all(
                case["positions_per_variant"] >= 320
                and all(value == 0 for value in case["comparisons"][name].values())
                and all(report["negative_controls"][context].values())
                for context, case in report["contexts"].items()
            ),
            "target attention candidate has incomplete or failing sample evidence",
        )
        self._qwen_speed_target_attention_library = ctypes.CDLL(str(binary))
        function = (
            self._qwen_speed_target_attention_library.qwen_stock_m1_attention_shared
        )
        function.argtypes = [ctypes.POINTER(R4DArgs), ctypes.c_int, ctypes.c_void_p]
        function.restype = ctypes.c_int
        self._qwen_speed_target_attention_candidate = {
            "variant": name,
            "binary_sha256": sha(binary),
            "qualification_sha256": sha(report_path),
            "scope": "full target graph only; unchanged split and arithmetic contract",
        }
        return function

    def qwen_speed_set_combined(self, enabled):
        # Change only between requests. The control selects the original target
        # piecewise route and the original drafter capture, using the same model.
        full = self.qwen_speed_set_full_graph(enabled)
        draft = self.qwen_speed_set_draft_attention(enabled)
        gemm = (
            self.qwen_speed_set_gemm(enabled)
            if hasattr(self, "_qwen_speed_gemm_graphs")
            else {}
        )
        sampler = (
            {"sampler_graph": self.qwen_speed_set_sampler_graph(enabled)}
            if hasattr(self, "_qwen_speed_sampler_graph")
            else {}
        )
        return {**full, **draft, **gemm, **sampler}

    def _install_speed_sampler_graph(self):
        if os.environ.get("QWEN_SPEED_SAMPLER_GRAPH") != "1":
            return
        from speed_graph_sampler import GraphSampler

        self._qwen_speed_sampler_graph = GraphSampler(self.model_runner)
        self.model_runner.sample = self._qwen_speed_sampler_graph

    def qwen_speed_set_sampler_graph(self, enabled):
        from qwen_r9700_lab.conformance_topk import require

        require(type(enabled) is bool, "sampler graph selector requires a boolean")
        require(
            not hasattr(self, "_qwen_observation"),
            "cannot change sampler graph during a qualification capture",
        )
        self._qwen_speed_sampler_graph.enabled = enabled
        return self._qwen_speed_sampler_graph.receipt()

    def _capture_draft_attention_control(self):
        """Retain both drafter graphs for a matched, between-request A/B.

        Both captures share model weights, input buffers and the graph pool.
        They are never replayed concurrently. The second capture happens only
        during startup, before any real session exists.
        """
        if os.environ.get("QWEN_SPEED_DRAFT_ATTN_AB") != "1":
            return
        from qwen_r9700_lab.conformance_topk import require

        require(
            hasattr(self, "_qwen_speed_draft_attention_candidate"),
            "drafter A/B requires the qualified candidate capture",
        )
        proposal = self.model_runner.speculator
        manager = proposal.query_cudagraph_manager
        require(manager._graphs_captured and manager.graphs, "drafter graphs missing")
        candidate = dict(manager.graphs)
        manager.graphs = {}
        proposal.capture()
        control = dict(manager.graphs)
        require(set(control) == set(candidate), "drafter capture domains differ")
        self._qwen_speed_draft_attention_graphs = {
            False: control,
            True: candidate,
        }
        manager.graphs = dict(candidate)
        self._qwen_speed_draft_attention_enabled = True

    def qwen_speed_set_draft_attention(self, enabled):
        from qwen_r9700_lab.conformance_topk import require

        require(type(enabled) is bool, "drafter graph selector requires a boolean")
        require(
            not hasattr(self, "_qwen_observation"),
            "cannot change drafter graph during a qualification capture",
        )
        require(
            hasattr(self, "_qwen_speed_draft_attention_graphs"),
            "paired drafter graphs were not captured at startup",
        )
        self.model_runner.speculator.query_cudagraph_manager.graphs = dict(
            self._qwen_speed_draft_attention_graphs[enabled]
        )
        self._qwen_speed_draft_attention_enabled = enabled
        return {"candidate_draft_attention": enabled, "same_allocations": True}

    def _observe_draft_attention_launch(self, hooks):
        candidate_path = os.environ.get("QWEN_SPEED_DRAFT_ATTN_SOURCE")
        if (
            os.environ.get("QWEN_SPEED_DRAFT_ATTN_METADATA") != "1"
            and not candidate_path
        ):
            return
        import inspect

        from vllm.v1.attention.backends import triton_attn

        original = triton_attn.unified_attention
        candidate = (
            self._load_qualified_draft_attention(candidate_path)
            if candidate_path
            else None
        )
        signature = inspect.signature(original)
        self._qwen_speed_draft_attention_calls = []

        def observed(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            query = bound.arguments["q"]
            if tuple(query.shape) == (8, 32, 128):
                meta = {}
                for key, value in bound.arguments.items():
                    if hasattr(value, "shape"):
                        meta[key] = {
                            "shape": list(value.shape),
                            "stride": list(value.stride()),
                            "dtype": str(value.dtype),
                        }
                    elif value is None or isinstance(value, (bool, int, float, tuple)):
                        meta[key] = value
                    else:
                        meta[key] = str(value)
                if meta not in self._qwen_speed_draft_attention_calls:
                    self._qwen_speed_draft_attention_calls.append(meta)
                if candidate is not None:
                    from qwen_r9700_lab.conformance_topk import require

                    require(
                        query.is_contiguous()
                        and str(query.dtype) == "torch.bfloat16"
                        and bound.arguments["max_seqlen_q"] == 8
                        and tuple(bound.arguments["window_size"]) == (2047, 0)
                        and int(bound.arguments["kv_quant_mode"]) == 1,
                        "qualified drafter attention domain changed",
                    )
                    return candidate(*args, **kwargs)
            return original(*args, **kwargs)

        hooks.replace(triton_attn, "unified_attention", observed)

    def _load_qualified_draft_attention(self, path):
        import hashlib
        import importlib.util
        import json
        from pathlib import Path

        import vllm.v1.attention.ops.triton_unified_attention as original

        from qwen_r9700_lab.conformance_topk import require

        path = Path(path)
        report = json.loads((path.parent / "result.json").read_text())
        generator_path = Path(os.environ["QWEN_SPEED_DRAFT_ATTN_GENERATOR"])
        sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        require(
            report["status"] == "SAMPLE_CHECKED_VARIANTS_RECORDED"
            and report["source_sha256"] == sha(original.__file__)
            and report["probe_sha256"] == sha(generator_path)
            and report["kv_quant_mode"] == 1,
            "drafter qualification source binding changed",
        )
        name = path.stem
        require(name == "unit_w4_s1_occ2", "unsupported drafter candidate")
        require(
            len(report["cases"]) == 6
            and all(
                case["positions"] >= 320
                and case["negative_control_detected"]
                and case["checks"][name]["different"] == 0
                for case in report["cases"].values()
            ),
            "drafter candidate did not pass all six qualification cases",
        )
        spec = importlib.util.spec_from_file_location(
            "speed_draft_generator", generator_path
        )
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        source = Path(original.__file__).read_text()
        anchor = "    launch_kwargs: dict[str, int] = {}"
        require(source.count(anchor) == 1, "drafter launch anchor changed")
        generated = generator.specialize_unit_scales(
            source.replace(
                anchor,
                f"    launch_kwargs: dict[str, int] = {report['variants'][name]!r}",
            )
        )
        require(path.read_text() == generated, "tested drafter module changed")
        spec = importlib.util.spec_from_file_location(
            "vllm.v1.attention.ops." + name, path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._qwen_speed_draft_attention_candidate = {
            "source_sha256": sha(path),
            "qualified_generator_sha256": sha(generator_path),
            "qualification_sha256": sha(path.parent / "result.json"),
            "variant": name,
            "scope": "drafter only; original target attention is unchanged",
        }
        return module.unified_attention

    def _install_speed_head_graph(self):
        if os.environ.get("QWEN_SPEED_HEAD_GRAPH") != "1":
            return
        from speed_graph_head import GraphHead

        import radiance_verifyhead
        from qwen_r9700_lab.conformance_topk import require

        lp, _ = radiance_verifyhead._find_target_lp(self.model_runner.model)
        require(
            lp is not None and hasattr(lp, "_radiance_fast_head"),
            "Global-512 head must be armed before installing graph experiment",
        )
        require(radiance_verifyhead.GLOBAL_TOPK == 512, "head candidate depth changed")
        self._qwen_speed_head_graph = GraphHead(lp._radiance_fast_head)
        lp._radiance_fast_head = self._qwen_speed_head_graph

    def qwen_speed_set_head_graph(self, enabled):
        from qwen_r9700_lab.conformance_topk import require

        require(type(enabled) is bool, "head graph selector requires a boolean")
        require(
            not hasattr(self, "_qwen_observation"),
            "cannot change head graph during a qualification capture",
        )
        self._qwen_speed_head_graph.enabled = enabled
        return self._qwen_speed_head_graph.receipt()

    def qwen_speed_metadata(self):
        runner = self.model_runner
        proposal = getattr(runner, "speculator", None) or getattr(
            runner, "drafter", None
        )
        result = {
            "runner": type(runner).__name__,
            "proposal": type(proposal).__name__,
            "proposal_attributes": sorted(vars(proposal)) if proposal else [],
            "models": {},
        }
        for label, obj in (
            ("target", getattr(runner, "model", None)),
            ("draft", getattr(proposal, "model", None)),
        ):
            if obj is None or not hasattr(obj, "named_modules"):
                continue
            modules = []
            for name, module in obj.named_modules():
                if not ("attn" in name or "linear" in type(module).__name__.lower()):
                    continue
                entry = {"name": name, "type": type(module).__name__}
                for key in (
                    "sliding_window",
                    "kv_cache_dtype",
                    "head_size",
                    "num_heads",
                    "num_kv_heads",
                ):
                    value = getattr(module, key, None)
                    if value is not None:
                        entry[key] = (
                            value
                            if isinstance(value, (str, int, float, bool))
                            else str(value)
                        )
                for key in ("weight", "kv_cache"):
                    value = getattr(module, key, None)
                    if hasattr(value, "shape"):
                        entry[key] = {
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                            "stride": list(value.stride()),
                        }
                implementation = getattr(module, "impl", None)
                if implementation is not None:
                    entry["impl"] = {
                        "type": type(implementation).__name__,
                        **{
                            key: getattr(implementation, key)
                            for key in ("sliding_window", "kv_cache_dtype")
                            if isinstance(
                                getattr(implementation, key, None),
                                (str, int, list, tuple),
                            )
                        },
                    }
                modules.append(entry)
            result["models"][label] = modules
        result["execution"] = self.qwen_optimized_metadata()
        result["experimental_drafter_w4"] = getattr(self, "_qwen_speed_draft_w4", None)
        head_graph = getattr(self, "_qwen_speed_head_graph", None)
        result["head_graph"] = head_graph.receipt() if head_graph else None
        sampler_graph = getattr(self, "_qwen_speed_sampler_graph", None)
        result["sampler_graph"] = sampler_graph.receipt() if sampler_graph else None
        result["draft_attention_capture_arguments"] = getattr(
            self, "_qwen_speed_draft_attention_calls", []
        )
        result["draft_attention_candidate"] = getattr(
            self, "_qwen_speed_draft_attention_candidate", None
        )
        result["target_attention_candidate"] = getattr(
            self, "_qwen_speed_target_attention_candidate", None
        )
        result["target_gemm_candidate"] = getattr(
            self, "_qwen_speed_target_gemm_candidate", None
        )
        result["target_gemm_enabled"] = getattr(self, "_qwen_speed_gemm_enabled", None)
        result["draft_attention_enabled"] = getattr(
            self, "_qwen_speed_draft_attention_enabled", None
        )
        result["full_capture_bindings"] = getattr(
            self, "_qwen_speed_capture_bindings", []
        )
        result["full_graph_enabled"] = getattr(self, "_qwen_speed_full_graph", None)
        import radiance_verifyhead

        result["head"] = radiance_verifyhead.dispatch_status()
        return result
