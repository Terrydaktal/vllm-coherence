#!/usr/bin/env python3
"""Install an opt-in semantic-boundary trace around the Qwen GDN core.

This is diagnostic-only.  It records exact SHA-256 fingerprints of the GDN
inputs and outputs both as complete tensors and row by row.  Row hashes let a
speculative M8 call be compared with the corresponding serial M1 calls without
persisting model data.  Persistent convolution/recurrent state is optional
because copying it is much more expensive.  Capture does not begin until the
sentinel named by RADIANCE_GDN_STATE_TRACE_SENTINEL exists, so model loading and
warm-up are excluded from an incident trace.  Prefill calls are excluded by
default; set RADIANCE_GDN_STATE_TRACE_PREFILL=1 when they are the subject of the
investigation.
"""

from pathlib import Path


TARGET = Path(
    "/opt/vllm/lib/python3.12/site-packages/vllm/model_executor/layers/"
    "mamba/gdn/qwen_gdn_linear_attn.py"
)
MARKER = "# radiance diagnostic: exact GDN semantic-boundary trace v2"

APPEND = r'''

# radiance diagnostic: exact GDN semantic-boundary trace v2
if __import__("os").environ.get("RADIANCE_GDN_STATE_TRACE", "0") == "1":
    import hashlib as _rad_trace_hashlib
    import json as _rad_trace_json
    import os as _rad_trace_os
    import threading as _rad_trace_threading
    from pathlib import Path as _RadTracePath

    _rad_trace_lock = _rad_trace_threading.Lock()
    _rad_trace_sequence = 0
    _rad_trace_original_forward_core = QwenGatedDeltaNetAttention._forward_core

    def _rad_trace_tensor(value):
        if value is None:
            return None
        tensor = value.detach().contiguous().cpu()
        payload = tensor.view(torch.uint8).numpy().tobytes()
        return {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": _rad_trace_hashlib.sha256(payload).hexdigest(),
        }

    def _rad_trace_rows(value):
        if value is None:
            return None
        tensor = value.detach().contiguous().cpu()
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        records = []
        for index in range(tensor.shape[0]):
            row = tensor[index].contiguous()
            payload = row.view(torch.uint8).numpy().tobytes()
            records.append(
                {
                    "index": index,
                    "dtype": str(row.dtype),
                    "shape": list(row.shape),
                    "sha256": _rad_trace_hashlib.sha256(payload).hexdigest(),
                }
            )
        return records

    def _rad_trace_values(value):
        if value is None:
            return None
        return value.detach().cpu().tolist()

    def _rad_trace_selected_state(instance, metadata):
        if _rad_trace_os.environ.get("RADIANCE_GDN_STATE_TRACE_STATE", "0") != "1":
            return None
        indices = metadata.prefill_state_indices
        if indices is None:
            indices = metadata.non_spec_state_indices_tensor
        if indices is None:
            indices = metadata.spec_state_indices_tensor
        if indices is None:
            return {"indices": None, "conv": None, "ssm": None}
        flat = indices.detach().reshape(-1)
        flat = flat[flat >= 0]
        if flat.numel() == 0:
            return {"indices": [], "conv": None, "ssm": None}
        unique = torch.unique(flat, sorted=True)
        conv_state, ssm_state = instance.kv_cache
        return {
            "indices": _rad_trace_values(unique),
            "conv": _rad_trace_tensor(conv_state.index_select(0, unique)),
            "ssm": _rad_trace_tensor(ssm_state.index_select(0, unique)),
        }

    def _rad_trace_forward_core(self, mixed_qkv, b, a, core_attn_out):
        global _rad_trace_sequence
        sentinel = _RadTracePath(
            _rad_trace_os.environ.get(
                "RADIANCE_GDN_STATE_TRACE_SENTINEL", "/cache/gdn-trace-enable"
            )
        )
        if not sentinel.is_file():
            return _rad_trace_original_forward_core(self, mixed_qkv, b, a, core_attn_out)

        context = get_forward_context()
        raw = context.attn_metadata
        metadata = raw[self.prefix]
        if (
            metadata.num_prefills > 0
            and _rad_trace_os.environ.get("RADIANCE_GDN_STATE_TRACE_PREFILL", "0")
            != "1"
        ):
            return _rad_trace_original_forward_core(self, mixed_qkv, b, a, core_attn_out)
        before = {
            "mixed_qkv": _rad_trace_tensor(mixed_qkv),
            "mixed_qkv_rows": _rad_trace_rows(mixed_qkv),
            "b": _rad_trace_tensor(b),
            "b_rows": _rad_trace_rows(b),
            "a": _rad_trace_tensor(a),
            "a_rows": _rad_trace_rows(a),
            "state": _rad_trace_selected_state(self, metadata),
        }
        result = _rad_trace_original_forward_core(self, mixed_qkv, b, a, core_attn_out)
        after = {
            "core_attn_out": _rad_trace_tensor(core_attn_out),
            "core_attn_out_rows": _rad_trace_rows(core_attn_out),
            "state": _rad_trace_selected_state(self, metadata),
        }
        record = {
            "prefix": self.prefix,
            "num_actual_tokens": metadata.num_actual_tokens,
            "num_prefills": metadata.num_prefills,
            "num_prefill_tokens": metadata.num_prefill_tokens,
            "num_decodes": metadata.num_decodes,
            "num_spec_decodes": metadata.num_spec_decodes,
            "prefill_query_start_loc": _rad_trace_values(
                metadata.prefill_query_start_loc
            ),
            "prefill_state_indices": _rad_trace_values(
                metadata.prefill_state_indices
            ),
            "prefill_has_initial_state": _rad_trace_values(
                metadata.prefill_has_initial_state
            ),
            "non_spec_state_indices": _rad_trace_values(
                metadata.non_spec_state_indices_tensor
            ),
            "spec_state_indices": _rad_trace_values(
                metadata.spec_state_indices_tensor
            ),
            "num_accepted_tokens": _rad_trace_values(
                metadata.num_accepted_tokens
            ),
            "before": before,
            "after": after,
        }
        root = _RadTracePath(
            _rad_trace_os.environ.get("RADIANCE_GDN_STATE_TRACE_ROOT", "/cache/gdn-traces")
        )
        arm = _rad_trace_os.environ.get("RADIANCE_GDN_STATE_TRACE_ARM", "unnamed")
        output = root / f"{arm}.jsonl"
        root.mkdir(parents=True, exist_ok=True)
        with _rad_trace_lock:
            record["sequence"] = _rad_trace_sequence
            _rad_trace_sequence += 1
            with output.open("a", encoding="utf-8") as handle:
                handle.write(
                    _rad_trace_json.dumps(record, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
                handle.flush()
                _rad_trace_os.fsync(handle.fileno())
        return result

    QwenGatedDeltaNetAttention._forward_core = _rad_trace_forward_core
    print("[gdn-state-trace] installed; capture waits for sentinel")
'''


def main() -> None:
    source = TARGET.read_text()
    if MARKER in source:
        print(f"[gdn-state-trace] already applied: {TARGET}")
        return
    TARGET.write_text(source + APPEND)
    print(f"[gdn-state-trace] applied: {TARGET}")


if __name__ == "__main__":
    main()
