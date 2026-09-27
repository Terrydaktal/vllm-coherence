"""Test-only worker extension: byte checks and legacy-cache controls.

Load only in an isolated DEV_MODE server with --worker-extension-cls. This is
not included in the production runtime package. It deliberately synchronizes
state copies; its timings are not production measurements.
"""

import hashlib


class ResponseEndProbe:
    def qwen_response_probe_pressure_mode(self, enabled):
        """Negative control: reproduce the old pin/retention lifetime policy."""
        import qwen_radiance_fair_scheduler as fair
        from qwen_radiance_response_end import ResponseEndCache

        scheduler = fair._phase_scheduler()
        assert not scheduler.requests
        if not enabled:
            assert not hasattr(self, "_pressure_originals")
            self._pressure_originals = (
                ResponseEndCache.release_after_progress, scheduler.banks.pressure_handler
            )
            ResponseEndCache.release_after_progress = lambda *_: False
            scheduler.banks.pressure_handler = None
        else:
            release, handler = self._pressure_originals
            ResponseEndCache.release_after_progress = release
            scheduler.banks.pressure_handler = handler
            del self._pressure_originals
        return {"repair_enabled": bool(enabled)}

    def qwen_response_probe_allocator(self):
        """Numeric allocator progress only; no prompts, token IDs or KV values."""
        import qwen_radiance_fair_scheduler as fair

        scheduler = fair._phase_scheduler()
        rows = []
        for request in scheduler.requests.values():
            manager = scheduler.banks.manager(request)
            endpoint = getattr(manager, "qwen_response_end", None)
            entry = endpoint.entry if endpoint else None
            rows.append({
                "input_tokens": request.num_prompt_tokens,
                "computed_tokens": request.num_computed_tokens,
                "preemptions": request.num_preemptions,
                "free_blocks": manager.block_pool.get_num_free_blocks(),
                "endpoint_tokens": entry["tokens"] if entry else 0,
                "endpoint_pins": len(entry["pins"]) if entry else 0,
            })
        return rows

    def qwen_response_probe_begin(self):
        import qwen_radiance_fair_scheduler as fair
        import qwen_radiance_response_end as endpoint
        import qwen_radiance_response_offload as offload
        import torch
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
        from vllm.v1.kv_cache_interface import SlidingWindowSpec
        from vllm.v1.worker.gpu import model_runner as runner_module

        assert not hasattr(self, "_response_probe")
        runner = self.model_runner
        scheduler = fair._phase_scheduler()
        assert scheduler is not None and not scheduler.requests
        report = {
            "copies": [],
            "cow": [],
            "mode": "exact",
            "scope": "canonical GDN/conv byte preservation",
        }
        self._response_probe = report
        original_copy = endpoint.copy_response_end
        original_manager = fair.CacheBanks.manager
        original_lookup = offload.lookup_response_end
        original_cow = runner_module.copy_kv_cache_blocks_inplace

        def manager(banks, request):
            result = original_manager(banks, request)
            legacy = report["mode"] == "legacy"
            coordinator = result.coordinator
            ids = {
                i
                for i, group in enumerate(runner.kv_cache_config.kv_cache_groups)
                if legacy and isinstance(group.kv_cache_spec, SlidingWindowSpec)
            }
            coordinator.eagle_group_ids = ids
            coordinator.attention_groups = [
                g._replace(use_eagle=bool(ids.intersection(g.group_ids)))
                for g in coordinator.attention_groups
            ]
            for i, child in enumerate(coordinator.single_type_managers):
                child.use_eagle = i in ids
            return result

        def lookup(connector, state):
            if report["mode"] != "exact":
                return False, None
            return original_lookup(connector, state)

        def raw(tensor):
            return (
                tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            )

        def checked_copy(model_runner, copies):
            context = runner.vllm_config.compilation_config.static_forward_context
            expected = []
            for item in copies:
                group = runner.kv_cache_config.kv_cache_groups[item["group"]]
                for name in group.layer_names:
                    conv, state = context[name].kv_cache
                    source = conv[item["conv"]].detach().cpu()
                    history = torch.zeros_like(source)
                    axis = source.ndim - 1 if is_conv_state_dim_first() else 0
                    # Independent scalar-position oracle for the logical history.
                    for position in range(source.shape[axis] - item["offset"]):
                        history.select(axis, position).copy_(
                            source.select(axis, position + item["offset"])
                        )
                    expected.append(
                        (name, item, raw(history), raw(state[item["state"]]))
                    )
            original_copy(model_runner, copies)
            for name, item, history, state in expected:
                actual_conv, actual_state = context[name].kv_cache
                for kind, want, actual in (
                    ("conv", history, raw(actual_conv[item["destination"]])),
                    ("gdn", state, raw(actual_state[item["destination"]])),
                ):
                    row = {
                        "layer": name,
                        "kind": kind,
                        "offset": item["offset"],
                        "bytes": len(want),
                        "same_bytes": want == actual,
                        "source_sha256": hashlib.sha256(want).hexdigest(),
                        "destination_sha256": hashlib.sha256(actual).hexdigest(),
                    }
                    report["copies"].append(row)
                    assert row["same_bytes"], "response-end state copy changed bytes"

        def checked_cow(caches, block_count, copies):
            expected = []
            seen = set()
            for entry in caches:
                for tensor in entry if isinstance(entry, (tuple, list)) else (entry,):
                    storage = tensor.untyped_storage()
                    if storage.data_ptr() in seen:
                        continue
                    seen.add(storage.data_ptr())
                    bytes_view = torch.empty(0, dtype=torch.uint8, device=tensor.device)
                    bytes_view.set_(storage)
                    size, remainder = divmod(storage.nbytes(), block_count)
                    assert remainder == 0
                    for source, destination in copies:
                        expected.append(
                            (
                                bytes_view,
                                destination,
                                size,
                                raw(bytes_view[source * size : (source + 1) * size]),
                            )
                        )
            original_cow(caches, block_count, copies)
            for view, destination, size, want in expected:
                actual = raw(view[destination * size : (destination + 1) * size])
                report["cow"].append({"bytes": size, "same_bytes": want == actual})
                assert want == actual, "copy-on-write changed a cached physical page"

        fair.CacheBanks.manager = manager
        endpoint.copy_response_end = checked_copy
        offload.lookup_response_end = lookup
        runner_module.copy_kv_cache_blocks_inplace = checked_cow
        self._response_probe_originals = (
            original_copy,
            original_manager,
            original_lookup,
            original_cow,
        )
        return {"installed": True}

    def qwen_response_probe_mode(self, mode):
        import qwen_radiance_fair_scheduler as fair
        from vllm.v1.kv_cache_interface import SlidingWindowSpec

        assert mode in {"exact", "aligned", "legacy"}
        scheduler = fair._phase_scheduler()
        assert scheduler is not None and not scheduler.requests
        self._response_probe["mode"] = mode
        scheduler.response_end_enabled = mode == "exact"
        connector = scheduler.connector.connector_scheduler
        groups = self.model_runner.kv_cache_config.kv_cache_groups
        connector.config = connector.config._replace(
            kv_group_configs=tuple(
                group._replace(
                    is_eagle_group=mode == "legacy"
                    and isinstance(
                        groups[group.group_idx].kv_cache_spec, SlidingWindowSpec
                    )
                )
                for group in connector.config.kv_group_configs
            )
        )
        return {"mode": mode}

    def qwen_response_probe_finish(self):
        import qwen_radiance_fair_scheduler as fair
        import qwen_radiance_response_end as endpoint
        import qwen_radiance_response_offload as offload
        from vllm.v1.worker.gpu import model_runner as runner_module

        self.qwen_response_probe_mode("exact")
        original_copy, original_manager, original_lookup, original_cow = (
            self._response_probe_originals
        )
        fair.CacheBanks.manager = original_manager
        endpoint.copy_response_end = original_copy
        offload.lookup_response_end = original_lookup
        runner_module.copy_kv_cache_blocks_inplace = original_cow
        report = self._response_probe
        del self._response_probe, self._response_probe_originals
        return report
