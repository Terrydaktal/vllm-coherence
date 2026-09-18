"""CPU negative controls for the isolated target-head observer, without model data."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import audit_verify_head as audit
import torch


def qualify(source_root):
    torch.set_num_threads(2)
    cases = []
    exact = torch.arange(100, 0, -1, dtype=torch.float32).unsqueeze(0)
    fast = exact.clone()
    fast[:, 80:] = float("-inf")
    clean = audit.compare_logits(fast, exact)
    assert clean["missing_strict_top20_entries"] == 0
    assert clean["different_reranked_values"] == 0
    assert clean["max_diagnostic_distribution_tv"] == 0
    cases.append("identical retained support and logits")

    missing = fast.clone()
    missing[:, 0] = float("-inf")
    bad = audit.compare_logits(missing, exact)
    assert bad["rows_missing_strict_top20"] == 1
    assert bad["rows_missing_all_exact_maxima"] == 1
    assert bad["max_diagnostic_distribution_tv"] > 0
    cases.append("true winning token omitted from shortlist")

    ties = exact.clone()
    ties[:, 19:21] = 81
    tie_fast = ties.clone()
    tie_fast[:, 19] = float("-inf")
    assert audit.compare_logits(tie_fast, ties)["missing_strict_top20_entries"] == 0
    cases.append("top-k boundary tie is not a strict support miss")

    rounded = fast.clone()
    rounded[:, 0] += 0.125
    bad = audit.compare_logits(rounded, exact)
    assert bad["different_reranked_values"] == 1
    assert bad["max_reranked_logit_difference"] == 0.125
    assert bad["max_diagnostic_distribution_tv"] > 0
    cases.append("reranking changes a retained logit")

    for invalid in (float("nan"), float("inf")):
        damaged = fast.clone()
        damaged[:, 0] = invalid
        try:
            audit.compare_logits(damaged, exact)
        except ValueError:
            pass
        else:
            raise AssertionError("nonfinite corruption passed")
    cases.append("nonfinite candidate corruption rejected")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        private = root / "private"
        private.mkdir(mode=0o700)
        for name in ("capture_radiance_layers.py", "legacy_layer_diagnostic.py"):
            shutil.copyfile(source_root / name, root / name)
        manifest = {
            "experiment_id": "qwen-runtime-ab-headobserverqualification",
            "layer_capture_source_sha256": hashlib.sha256(
                (root / "capture_radiance_layers.py").read_bytes()
            ).hexdigest(),
        }
        (root / "manifest.json").write_text(json.dumps(manifest))
        calls = {"candidate": 0, "reference": 0}

        def original(self, weight, hidden, bias):
            calls["candidate"] += 1
            return rounded

        def reference(weight, hidden, bias):
            calls["reference"] += 1
            return exact

        observer = audit.wrap(original, root, private)
        state = SimpleNamespace(_radiance_fast_ok=True, _radiance_exact_head=reference)
        hidden = torch.ones(1, 4)
        assert observer(state, None, hidden) is rounded
        assert calls == {"candidate": 1, "reference": 0}
        cases.append("unarmed observer does no reference work")

        (root / "head-audit-request.json").write_text(json.dumps({"label": "synthetic"}))
        before = rounded.clone()
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=False):
            for _ in range(130):
                assert observer(state, None, hidden) is rounded
        assert torch.equal(rounded, before)
        assert calls == {"candidate": 131, "reference": 128}
        report = json.loads((root / "head-audit-synthetic.json").read_text())
        assert report["calls_checked"] == 128
        capsule = report["capsules"]["rounding_difference"]
        path = private / "layer-capsules" / capsule["name"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == capsule["sha256"]
        assert path.stat().st_mode & 0o077 == 0
        assert capsule["name"].startswith(manifest["experiment_id"])
        cases.append("observer preserves candidate output and enforces declared capture coverage")
        cases.append("counterexample uses private authenticated reusable tensor capsule")

        package = root / "package"
        package.mkdir()
        installed = Path("/opt/vllm/lib/python3.12/site-packages/radiance_verifyhead.py")
        shutil.copyfile(installed, package / installed.name)
        audit.install(package, source_root / "audit_verify_head.py")
        assert (package / installed.name).read_text().endswith(audit.APPENDIX)
        audit.install(package, source_root / "audit_verify_head.py")
        (package / installed.name).write_text("raise AssertionError('unknown backend')\n")
        try:
            audit.install(package, source_root / "audit_verify_head.py")
        except ValueError:
            pass
        else:
            raise AssertionError("changed target source passed")
        cases.append("installer binds the pinned source and rejects another backend")

    return {
        "schema": "qwen-target-head-observer-qualification-v1",
        "passed": True,
        "cases": cases,
        "torch": torch.__version__,
        "device": "cpu",
        "source_sha256": hashlib.sha256(
            (source_root / "audit_verify_head.py").read_bytes()
        ).hexdigest(),
        "model_or_gpu_correctness_proven": False,
    }


if __name__ == "__main__":
    try:
        print(json.dumps(qualify(Path(__file__).resolve().parent), indent=2))
    except Exception as error:
        print(json.dumps({"passed": False, "error_type": type(error).__name__}))
        raise SystemExit(1) from None
