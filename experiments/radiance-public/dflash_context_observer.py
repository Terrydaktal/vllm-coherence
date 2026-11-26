"""Observe the draft context producers while a paired synthetic target capture is active."""

import hashlib
import json
from functools import wraps
from pathlib import Path


def begin(worker, destination):
    import torch
    from torch.nn import functional as functional

    from qwen_r9700_lab.conformance_instrumentation import HookSet

    root = Path(destination).resolve()
    if not root.is_relative_to("/prefill-diagnosis/refresh") or root.exists():
        raise ValueError("expected a new isolated synthetic capture path")
    if not getattr(worker, "_prefill_capture", None) or hasattr(worker, "_draft_context_stages"):
        raise ValueError("draft context observation needs one active target capture")
    root.mkdir(mode=0o700)
    hooks = HookSet()
    spec = worker.model_runner.speculator.model
    model = spec.model
    rows, shapes = {}, {}
    worker._draft_context_stages = hooks, root, rows, shapes
    input_shape = None

    def snapshot(name, t):
        positions = worker._prefill_capture["positions"]
        if not positions or positions[-1] >= 2049:
            return
        if positions != list(range(positions[0], positions[-1] + 1)) or t.shape[0] < len(positions):
            raise ValueError("unexpected prefill position mapping")
        cpu = t[: len(positions)].detach().contiguous().cpu()
        raw = cpu.view(torch.uint8).reshape(len(positions), -1).numpy()
        cap = rows.setdefault(name, {})
        if any(p in cap for p in positions):
            raise ValueError("prefill stage was executed twice for the same position")
        cap.update({p: hashlib.sha256(v).hexdigest() for p, v in zip(positions, raw, strict=True)})
        shapes.setdefault(name, []).append([*t.shape])
        # Enough bounded actual input to reproduce any divergent row at either batch width.
        if name in {"aux", "combined", "normed_context"}:
            torch.save(cpu, root / f"{name}-{positions[0]}-{positions[-1]}.pt")

    original_combine = spec.combine_hidden_states
    original_linear = functional.linear
    original_project = model._project_context_kv
    original_normalize = model._normalize_context_k

    @wraps(original_combine)
    def combine(hidden):
        snapshot("aux", hidden)
        result = original_combine(hidden)
        snapshot("combined", result)
        return result

    @wraps(original_project)
    def project(context, *args, **kwargs):
        nonlocal input_shape
        input_shape = context.shape
        try:
            key, value = original_project(context, *args, **kwargs)
        finally:
            input_shape = None
        snapshot("projected_k", key.transpose(0, 1))
        snapshot("projected_v", value.transpose(0, 1))
        return key, value

    @wraps(original_linear)
    def linear(act, weight, *args, **kwargs):
        if input_shape is not None and weight is model._fused_kv_weight:
            snapshot("normed_context", act)
        return original_linear(act, weight, *args, **kwargs)

    @wraps(original_normalize)
    def normalize(key):
        out = original_normalize(key)
        snapshot("normalized_k", out.transpose(0, 1))
        return out

    hooks.replace(spec, "combine_hidden_states", combine)
    hooks.replace(model, "_project_context_kv", project)
    hooks.replace(functional, "linear", linear)
    hooks.replace(model, "_normalize_context_k", normalize)
    return {"active": True}


def finish(worker):
    hooks, root, rows, shapes = worker._draft_context_stages
    hooks.close()
    del worker._draft_context_stages
    (root / "row-digests.json").write_text(json.dumps(rows))
    (root / "shapes.json").write_text(json.dumps(shapes))
    return {k: len(v) for k, v in rows.items()}
