"""Isolated fast/full target-head comparison on identical actual hidden states.

Returns the original fast logits unchanged. This is a diagnostic observer, not
an output guard. Raw logits/hidden states remain in private RAM; public reports
contain only counts, aggregate errors and capsule hashes.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path

PREIMAGE = "7fc05ceb76672aa9eae3cd7715046873b6294857c1bf56abbfda322ef4536789"
LEGACY_SHA256 = "96c6cafb9fdef9f99b256b5977fd9068f99f8d60d0b1ffd0a5478eeed9a05694"
APPENDIX = """

# Explicitly armed isolated target-head numerical audit.
from qwen_verify_head_audit import wrap as _qwen_wrap_verify_head
_apply_head_gated = _qwen_wrap_verify_head(_apply_head_gated)
"""


def install(package, source):
    package, source = Path(package), Path(source)
    path = package / "radiance_verifyhead.py"
    old = path.read_text()
    original = old[: -len(APPENDIX)] if old.endswith(APPENDIX) else old
    if hashlib.sha256(original.encode()).hexdigest() != PREIMAGE:
        raise ValueError("target-head source does not match the audit binding")
    updated = original + APPENDIX
    compile(updated, str(path), "exec")
    shutil.copyfile(source, package / "qwen_verify_head_audit.py")
    path.write_text(updated)
    return {
        "target_source_sha256": hashlib.sha256(updated.encode()).hexdigest(),
        "audit_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def distribution(logits, top_k=20, top_p=0.95):
    """Diagnostic top-k/top-p distribution; tie handling is not certified here."""
    import torch

    values, indices = logits.float().topk(top_k, dim=-1)
    probs = values.softmax(-1)
    keep = (probs.cumsum(-1) - probs) < top_p
    probs = probs * keep
    probs = probs / probs.sum(-1, keepdim=True)
    result = torch.zeros_like(logits, dtype=torch.float32)
    return result.scatter(-1, indices, probs)


def compare_logits(fast, exact):
    import torch

    fast, exact = (
        fast.float().reshape(-1, fast.shape[-1]),
        exact.float().reshape(-1, exact.shape[-1]),
    )
    if fast.shape != exact.shape or fast.shape[-1] < 20 or not torch.isfinite(exact).all():
        raise ValueError("target-head audit received invalid exact logits")
    finite = torch.isfinite(fast)
    if torch.isnan(fast).any() or torch.isposinf(fast).any() or not (finite.sum(-1) >= 20).all():
        raise ValueError("fast target head has insufficient finite support")
    values, top = exact.topk(20, dim=-1)
    included = finite.gather(-1, top)
    # Strictly above the kth value avoids blaming an arbitrary boundary tie.
    strict_missing = (~included) & (values > values[:, -1:])
    retained_max = exact.masked_fill(~finite, float("-inf")).max(-1).values
    greedy_missing = exact.max(-1).values > retained_max
    delta = torch.where(finite, (fast - exact).abs(), 0)
    return {
        "rows": fast.shape[0],
        "vocabulary": fast.shape[1],
        "missing_full_top20_entries": int((~included).sum()),
        "missing_strict_top20_entries": int(strict_missing.sum()),
        "rows_missing_strict_top20": int(strict_missing.any(-1).sum()),
        "rows_missing_all_exact_maxima": int(greedy_missing.sum()),
        "different_reranked_values": int(((fast != exact) & finite).sum()),
        "max_reranked_logit_difference": delta.max().item(),
        "max_diagnostic_distribution_tv": (
            (distribution(fast) - distribution(exact)).abs().sum(-1).max().item() / 2
        ),
        "topk_boundary_tie_semantics_verified": False,
    }


def verify_helpers(root, manifest):
    for name, expected in (
        ("legacy_layer_diagnostic.py", LEGACY_SHA256),
        ("capture_radiance_layers.py", manifest["layer_capture_source_sha256"]),
    ):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("target-head capture helper binding differs")


def wrap(original, root=Path("/benchmark"), private=Path("/private-fixtures")):
    current, summary, helpers, manifest = None, None, None, None

    @functools.wraps(original)
    def checked(self, lm_head, hidden_states, embedding_bias=None):
        nonlocal current, summary, helpers, manifest
        import torch

        fast = original(self, lm_head, hidden_states, embedding_bias)
        marker = root / "head-audit-request.json"
        if not marker.exists() or not getattr(self, "_radiance_fast_ok", False):
            return fast
        request = json.loads(marker.read_text())
        label = request["label"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,160}", label):
            raise ValueError("invalid target-head audit label")
        if manifest is None:
            manifest = json.loads((root / "manifest.json").read_text())
            if not re.fullmatch(r"qwen-runtime-ab-[a-z0-9]+", manifest["experiment_id"]):
                raise ValueError("invalid target-head experiment identity")
            verify_helpers(root, manifest)
        if current != label:
            current = label
            summary = {
                "schema": "qwen-target-head-audit-v1",
                "label": label,
                "experiment_id": manifest["experiment_id"],
                "calls_checked": 0,
                "rows_checked": 0,
                "coverage": "first 128 armed target-head calls per request",
                "rows_missing_strict_top20": 0,
                "rows_missing_all_exact_maxima": 0,
                "different_reranked_values": 0,
                "max_reranked_logit_difference": 0.0,
                "max_diagnostic_distribution_tv": 0.0,
                "capsules": {},
                "returns_candidate_logits_unchanged": True,
            }
        if summary["calls_checked"] >= 128:
            return fast
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("target-head CPU audit is not supported inside graph capture")
        exact = self._radiance_exact_head(lm_head, hidden_states, embedding_bias)
        vocabulary = getattr(self, "org_vocab_size", fast.shape[-1])
        fast_cpu, exact_cpu = (
            v[..., :vocabulary].detach().to("cpu", copy=True).contiguous() for v in (fast, exact)
        )
        row = compare_logits(fast_cpu, exact_cpu)
        summary["calls_checked"] += 1
        summary["rows_checked"] += row["rows"]
        for key in (
            "rows_missing_strict_top20",
            "rows_missing_all_exact_maxima",
            "different_reranked_values",
        ):
            summary[key] += row[key]
        for key in ("max_reranked_logit_difference", "max_diagnostic_distribution_tv"):
            summary[key] = max(summary[key], row[key])
        capture_kind = (
            "support_miss"
            if row["rows_missing_strict_top20"]
            else "rounding_difference"
            if row["different_reranked_values"]
            else None
        )
        if capture_kind and capture_kind not in summary["capsules"]:
            if helpers is None:
                spec = importlib.util.spec_from_file_location(
                    "head_capture_adapter", root / "capture_radiance_layers.py"
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                helpers = module.load_legacy_helpers(root / "legacy_layer_diagnostic.py")
            directory = private / "layer-capsules"
            directory.mkdir(mode=0o700, exist_ok=True)
            if directory.is_symlink() or directory.stat().st_mode & 0o077:
                raise ValueError("target-head capsule directory is not private")
            name = f"{manifest['experiment_id']}-{label}-{capture_kind}.pt"
            payload = {
                "schema": "qwen-target-head-capsule-v1",
                "label": label,
                "experiment_id": manifest["experiment_id"],
                "call_index": summary["calls_checked"] - 1,
                "measurements": row,
                "hidden": hidden_states.detach().to("cpu", copy=True).contiguous(),
                "fast_logits": fast_cpu,
                "exact_logits": exact_cpu,
            }
            payload["tensor_sha256"] = {
                key: helpers._tensor_sha256(payload[key])
                for key in ("hidden", "fast_logits", "exact_logits")
            }
            identity = helpers._write_tensor_capsule(directory / name, payload)
            summary["capsules"][capture_kind] = {
                "name": name,
                "sha256": identity,
                "call_index": payload["call_index"],
                "measurements": row,
            }
        temporary = root / f"head-audit-{label}.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(root / f"head-audit-{label}.json")
        return fast

    return checked
