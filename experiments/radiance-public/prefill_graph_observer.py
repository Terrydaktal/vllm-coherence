"""Bounded, post-forward observation of an unmodified DFlash FULL graph."""

import hashlib
import json
import re
from pathlib import Path

from qwen_r9700_lab.conformance_instrumentation import HookSet

CAPTURE_ROOT = Path("/prefill-diagnosis/refresh")


def target_kv_layouts(runner):
    import torch

    result = {}
    for (
        name,
        module,
    ) in runner.vllm_config.compilation_config.static_forward_context.items():
        if re.search(r"(?:^|\.)layers\.(\d+)\.self_attn\.attn$", name):
            cache = getattr(module, "kv_cache", None)
            result[name] = (
                {
                    "shape": list(cache.shape),
                    "strides": list(cache.stride()),
                    "dtype": str(cache.dtype),
                }
                if isinstance(cache, torch.Tensor)
                else {"type": type(cache).__name__}
            )
    return result


def target_kv_rows(runner, state, start, stop):
    import torch

    if not 0 <= start < stop <= 2048:
        raise ValueError("target KV diagnostic exceeds its synthetic window")
    context = runner.vllm_config.compilation_config.static_forward_context
    rows = {}
    for name, metadata in state.attn_metadata.items():
        layer = re.search(r"(?:^|\.)layers\.(\d+)\.self_attn\.attn$", name)
        if (
            layer is None
            or "draft" in name
            or int(layer.group(1)) not in range(3, 64, 4)
        ):
            continue
        cache = context[name].kv_cache
        if cache.shape[1:] != (4, 16, 512) or cache.element_size() != 1:
            raise ValueError(
                f"unexpected target FP8 KV layout at {name}: {tuple(cache.shape)} {cache.dtype} {cache.stride()}"
            )
        physical = metadata.block_table[0, start // 16 : (stop + 15) // 16].long()
        if (
            physical.numel() != (stop + 15) // 16 - start // 16
            or int(physical.min()) < 0
            or int(physical.max()) >= cache.shape[0]
        ):
            raise ValueError(f"invalid physical block table for {name}")
        pages = cache.view(torch.uint8).index_select(0, physical)
        slots = (
            pages.transpose(1, 2)
            .reshape(-1, 4, 512)[start % 16 : start % 16 + stop - start]
            .cpu()
            .numpy()
        )
        key = int(layer.group(1))
        if key in rows:
            raise ValueError("duplicate target KV layer")
        rows[key] = [hashlib.sha256(slot.tobytes(order="C")).hexdigest() for slot in slots]
    if set(rows) != set(range(3, 64, 4)):
        raise ValueError(f"incomplete target attention coverage: {sorted(rows)}")
    return rows


def install_replay_boundaries(worker, layers=(49, 52, 55, 58, 62)):
    """Select same-count target boundary outputs on a disposable forced-input clone."""
    from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
        get_eagle3_aux_layers_from_config,
    )

    runner = worker.model_runner
    original = get_eagle3_aux_layers_from_config(runner.vllm_config.speculative_config)
    if tuple(original or ()) != (6, 20, 34, 48, 62):
        raise ValueError("expected the five qualified DFlash target boundaries")
    if (
        len(layers) != len(original)
        or tuple(sorted(set(layers))) != tuple(layers)
        or not all(type(layer) is int and 0 < layer <= 64 for layer in layers)
        or hasattr(runner, "_diagnostic_replay_boundaries")
    ):
        raise ValueError("expected five unique, increasing diagnostic target layers")
    runner.model.set_aux_hidden_state_layers(tuple(layers))
    runner._diagnostic_replay_boundaries = tuple(layers)
    return {"original": original, "observed": tuple(layers), "forced_input_only": True}


class GraphOutputObserver:
    def __init__(self, runner, destination, first, last, mode, capture_kv=False):
        import torch
        from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
            get_eagle3_aux_layers_from_config,
        )

        destination = Path(destination)
        if (
            mode not in {"native", "prefill", "forced"}
            or first < 0
            or not 0 < last - first <= 128
            or not destination.resolve().is_relative_to(CAPTURE_ROOT)
        ):
            raise ValueError("require a bounded, isolated graph-output diagnostic")
        destination.mkdir(mode=0o700)
        self.root = destination
        self.hooks = HookSet()
        self.batches = []
        self.rows = set()
        self.pending = None
        self.route = None
        self.graph_shape = None
        self.target_kv = {} if capture_kv else None
        self.kv_status = (
            {"status": "collecting", "layouts": target_kv_layouts(runner)}
            if capture_kv
            else None
        )
        config = runner.vllm_config.speculative_config
        diagnostic = getattr(runner, "_diagnostic_replay_boundaries", None)
        if diagnostic and mode == "native":
            raise ValueError("altered drafter boundary selection needs a forced decode")
        self.layers = diagnostic or get_eagle3_aux_layers_from_config(config)
        if not self.layers:
            self.layers = runner.model.get_eagle3_default_aux_hidden_state_layers()
        self.layers = list(self.layers)
        if not self.layers:
            raise ValueError("DFlash auxiliary layer configuration missing")
        self.mode = mode

        execute = runner.execute_model

        def execute_model(*args, **kwargs):
            self.route = None
            self.graph_shape = None
            return execute(*args, **kwargs)

        manager = runner.cudagraph_manager

        def observe_route(name, original):
            def wrapped(*args, **kwargs):
                self.route = name
                if name == "full":
                    desc = args[0] if args else kwargs["desc"]
                    self.graph_shape = {
                        key: getattr(desc, key, None)
                        for key in (
                            "num_tokens",
                            "num_reqs",
                            "uniform_token_count",
                            "max_query_len",
                        )
                    }
                return original(*args, **kwargs)

            return wrapped

        sample_tokens = runner.sample_tokens

        def observe_outputs(*args, **kwargs):
            state = runner.execute_model_state
            if state is not None:
                if self.pending is not None:
                    raise ValueError("previous forward was not postprocessed")
                batch = state.input_batch
                if batch.num_reqs != 1:
                    raise ValueError("graph observer needs a single synthetic request")
                n = batch.num_tokens
                positions = batch.positions[:n].cpu().tolist()
                selected = [i for i, p in enumerate(positions) if first <= p < last]
                if selected:
                    features = [*state.aux_hidden_states, state.hidden_states]
                    if len(features) != len(self.layers) + 1:
                        raise ValueError("graph auxiliary-output coverage changed")
                    indices = torch.tensor(selected, device=batch.positions.device)
                    static = runner.input_buffers
                    static_differences = {
                        key: int(
                            torch.count_nonzero(
                                getattr(batch, key)[:n] != getattr(static, key)[:n]
                            )
                        )
                        for key in ("input_ids", "positions")
                    }
                    self.pending = {
                        "indices": selected,
                        "positions": [positions[i] for i in selected],
                        "input_ids": batch.input_ids.index_select(0, indices).cpu(),
                        "features": [
                            out.index_select(0, indices).detach().cpu()
                            for out in features
                        ],
                        "route": self.route or "plain",
                        "width": n,
                        "first": positions[0],
                        "graph_shape": self.graph_shape,
                        "static_differences": static_differences,
                        "is_prefill": bool(batch.is_prefilling_np[0]),
                        "drafts": batch.num_draft_tokens,
                    }
                    if self.target_kv is not None:
                        try:
                            row_positions = self.pending["positions"]
                            if row_positions != list(
                                range(row_positions[0], row_positions[-1] + 1)
                            ):
                                raise ValueError(
                                    "nonconsecutive synthetic KV observation"
                                )
                            if not self.target_kv and first > 0:
                                self.target_kv = {
                                    layer: dict(enumerate(hashes))
                                    for layer, hashes in target_kv_rows(
                                        runner, state, 0, first
                                    ).items()
                                }
                            self.pending["kv_digests"] = target_kv_rows(
                                runner, state, row_positions[0], row_positions[-1] + 1
                            )
                        except (
                            ValueError,
                            TypeError,
                            KeyError,
                            AttributeError,
                            IndexError,
                            RuntimeError,
                        ) as error:
                            self.kv_status["status"] = "unavailable"
                            self.kv_status["reason"] = str(error)
                            self.target_kv = None
            return sample_tokens(*args, **kwargs)

        post = runner.postprocess_sampled

        def record_commit(
            idx_mapping, sampled_tokens, num_sampled, num_rejected, *args, **kwargs
        ):
            frame = self.pending
            self.pending = None
            if frame is not None:
                count = int(num_sampled[0])
                rejected = int(num_rejected[0])
                indices = [
                    n
                    for n, index in enumerate(frame.pop("indices"))
                    if mode == "prefill" or frame["is_prefill"] or index < count
                ]
                if indices:
                    if self.target_kv is not None:
                        latest = frame.pop("kv_digests")
                        for layer, digests in latest.items():
                            saved = self.target_kv.setdefault(layer, {})
                            for i in indices:
                                p = frame["positions"][i]
                                if p in saved:
                                    raise ValueError("duplicate committed target KV")
                                saved[p] = digests[i]
                    frame["positions"] = [frame["positions"][i] for i in indices]
                    if self.rows.intersection(frame["positions"]):
                        raise ValueError("overlapping committed output positions")
                    self.rows.update(frame["positions"])
                    frame["features"] = [v[indices] for v in frame["features"]]
                    frame["input_ids"] = frame["input_ids"][indices]
                    f = f"aux-{len(self.batches):04d}.pt"
                    torch.save(frame, self.root / f)
                    self.batches.append(
                        {
                            "file": f,
                            "route": frame["route"],
                            "first": frame["first"],
                            "width": frame["width"],
                            "count": count,
                            "rejected": rejected,
                            "graph_shape": frame["graph_shape"],
                            "static_differences": frame["static_differences"],
                            "is_prefill": frame["is_prefill"],
                            "draft_count": frame["drafts"],
                        }
                    )
            return post(
                idx_mapping, sampled_tokens, num_sampled, num_rejected, *args, **kwargs
            )

        try:
            self.hooks.replace(runner, "execute_model", execute_model)
            for name, attr in (
                ("full", "run_fullgraph"),
                ("piecewise", "run_pw_graph"),
            ):
                self.hooks.replace(
                    manager, attr, observe_route(name, getattr(manager, attr))
                )
            self.hooks.replace(runner, "sample_tokens", observe_outputs)
            self.hooks.replace(runner, "postprocess_sampled", record_commit)
        except BaseException:
            self.hooks.close()
            raise

    def finish(self):
        self.hooks.close()
        if self.pending is not None:
            raise ValueError("missing post-forward commit boundary")
        if self.target_kv is not None:
            self.kv_status["status"] = "complete"
            (self.root / "target-kv.json").write_text(json.dumps(self.target_kv) + "\n")
        receipt = {
            "mode": self.mode,
            "target_kv_status": self.kv_status,
            "positions": len(self.rows),
            "layers": self.layers,
            "batches": self.batches,
            "observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        (self.root / "observer.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
