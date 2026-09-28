"""Isolated larger-batch experiment; never selected by the serving launcher."""

import hashlib
import inspect
import os
from pathlib import Path

from speed_candidate_worker import SpeedCandidateWorker


class WidePrefillWorker(SpeedCandidateWorker):
    def load_model(self, **kwargs):
        result = super().load_model(**kwargs)
        import stock_gdn_norm_quant as norm

        source = inspect.getsource(norm.fused)
        old = "or not 1 <= x.shape[0] <= 2048"
        if source.count(old) != 1:
            raise ValueError("experimental GDN norm admission anchor changed")
        namespace = dict(norm.fused.__globals__)
        exec(  # noqa: S102 - trusted native function in an isolated experiment
            compile(
                source.replace(old, "or not 1 <= x.shape[0] <= 4096"),
                "<isolated-wide-prefill-norm>",
                "exec",
            ),
            namespace,
        )
        norm.fused = namespace["fused"]
        root = Path(os.environ["QWEN_PREFILL_SPEED_EXPERIMENT"])
        if not root.is_relative_to("/prefill-diagnosis"):
            raise ValueError("isolated experiment mount required")
        # Current frozen releases already install prefill adapters. Replace that
        # ownership once; never stack experimental and released wrappers.
        self.qwen_prefill_legacy_timing()
        self.qwen_prefill_load_candidate_sources(str(root / "source-wide"))
        self.qwen_prefill_install_runtime(
            str(root / "attention-wide-batch"),
            str(root / "projection-wide-batch"),
            True,
        )
        self._wide_prefill_native_norm_sha256 = hashlib.sha256(
            source.encode()
        ).hexdigest()
        return result
