"""Hash the actual recurrent bank selected across a synthetic cache-reuse boundary."""

import hashlib
import json
from pathlib import Path


def call(worker, action, *args):
    import torch

    from qwen_r9700_lab.conformance_instrumentation import HookSet

    if action == "measure_draft_projection":
        from benchmark_dflash_context_projection import run

        return run(worker, *args)

    if action in {"draft_context_begin", "draft_context_finish"}:
        import dflash_context_observer

        if action == "draft_context_begin":
            return dflash_context_observer.begin(worker, *args)
        return dflash_context_observer.finish(worker)

    if action == "proposal_control":
        temperature, lookup = args
        if temperature not in (0, 0.55) or type(lookup) is not bool:
            raise ValueError("expected a measured proposal-temperature and lookup policy")
        speculator = worker.model_runner.speculator
        original = getattr(worker, "_isolated_lookup_wrapper", None)
        if original is None:
            original = speculator.propose
            if Path(original.__code__.co_filename).name != "dflash_prompt_lookup.py":
                raise ValueError("expected the isolated retained lookup wrapper")
            worker._isolated_lookup_wrapper = original
        speculator.propose = original if lookup else original.__wrapped__
        worker._fourteen_temperature(temperature)
        return {"lookup": lookup, "proposal_temperature": temperature}

    if action == "measure_wide_norm":
        from benchmark_wide_input_norm import run

        return run(worker, *args)
    if action == "enable_wide_norm":
        from wide_input_norm import install

        return install(worker, *args)
    if action == "disable_wide_norm":
        from wide_input_norm import remove

        return remove(worker)

    if action == "end":
        hooks, path, records = worker._target_boundary_check
        hooks.close()
        worker._target_boundary_check = None
        path.write_text(json.dumps(records, indent=2) + "\n")
        return {"records": len(records)}
    if action != "begin" or getattr(worker, "_target_boundary_check", None):
        raise ValueError("expected a fresh isolated target-boundary observation")
    raw_path, disable_880 = args
    path = Path(raw_path).resolve()
    if (
        not path.is_relative_to("/prefill-diagnosis/refresh")
        or path.exists()
        or not path.parent.is_dir()
    ):
        raise ValueError("private synthetic capture destination required")
    runner = worker.model_runner
    state = runner.model_state
    records = []
    hooks = HookSet()
    worker._target_boundary_check = (hooks, path, records)

    def digest(t):
        return hashlib.sha256(
            t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest()

    original = state.preprocess_state

    def observe(batch, tables, config, computed):
        if batch.num_reqs != 1:
            raise ValueError("isolated single request required")
        slot = int(batch.idx_mapping[0].item())
        start = int(computed[slot].item())
        rows = batch.num_tokens
        if start not in (56032, 57680, 59328):
            return original(batch, tables, config, computed)
        gids, spec = state._get_mamba_group_info(config)
        record = {
            "start": start,
            "rows": rows,
            "cache_block_size": state.cache_config.block_size,
            "state_block_size": spec.block_size,
        }

        def capture(phase):
            col = int(state._mamba_state_idx_gpu[slot].item())
            group_list = []
            record[phase] = {
                "state_column": col,
                "accepted": int(state.num_accepted_tokens_gpu[slot].item()),
                "groups": group_list,
            }
            if phase == "after":
                record[phase]["source_column"] = int(state._mamba_src_col_gpu[0].item())
                record[phase]["source_offset"] = int(state._mamba_src_off_gpu[0].item())
            if col < 0:
                return
            for gid in gids:
                if col >= tables[gid].shape[1]:
                    group_list.append({"group": gid, "column_outside_table": True})
                    continue
                physical = int(tables[gid][0, col].item())
                layers = []
                if physical <= 0:
                    group_list.append({"group": gid, "physical": physical, "layers": []})
                    continue
                for name in config.kv_cache_groups[gid].layer_names:
                    conv, ssm = runner.vllm_config.compilation_config.static_forward_context[
                        name
                    ].kv_cache
                    layers.append(
                        {
                            "layer": name,
                            "conv": digest(conv[physical]),
                            "ssm": digest(ssm[physical]),
                        }
                    )
                group_list.append({"group": gid, "physical": physical, "layers": layers})

        capture("before")
        out = original(batch, tables, config, computed)
        capture("after")
        records.append(record)
        return out

    hooks.replace(state, "preprocess_state", observe)
    if disable_880:
        import radiance_gdn

        current = radiance_gdn.forward_core_fused
        globals_ = getattr(current, "__func__", current).__globals__
        if "indexed_scan_into" not in globals_ or "previous" not in globals_:
            hooks.close()
            worker._target_boundary_check = None
            raise ValueError("the indexed GDN route is not installed")
        previous = globals_["previous"]

        def fallback(layer, mixed_qkv, b, a, out):
            if mixed_qkv.shape[0] == 880:
                records.append({"disabled_indexed_prefill_rows": 880})
                return previous(layer, mixed_qkv, b, a, out)
            return current(layer, mixed_qkv, b, a, out)

        hooks.replace(radiance_gdn, "forward_core_fused", fallback)
    return {"active": True, "disable_indexed_880": bool(disable_880)}
