"""Diagnostic-only original/candidate GEMM dispatch before graph capture."""

import hashlib
import os
from collections import Counter
from pathlib import Path

from optimized_d7_worker import OptimizedWorker


class GemmAuditWorker(OptimizedWorker):
    def load_model(self, **kwargs):
        import radiance_mxfp4 as kernel

        original = kernel._ext
        super().load_model(**kwargs)
        variant = os.environ["QWEN_GEMM_AUDIT_VARIANT"]
        if variant not in ("original", "candidate"):
            raise ValueError("unsupported diagnostic GEMM variant")
        selected = original if variant == "original" else kernel._ext
        partial, counters = kernel._decode_scratch
        selected.set_decode_scratch(partial.data_ptr(), partial.numel() * 4, counters.data_ptr())
        self._gemm_calls = Counter()
        self._gemm_selection = {
            "variant": variant,
            "binary_sha256": hashlib.sha256(Path(selected.__file__).read_bytes()).hexdigest(),
            "installed_before_capture": True,
        }
        calls = self._gemm_calls

        class DispatchCounter:
            def __getattr__(self, name):
                return getattr(selected, name)

            def launch(self, *args):
                calls[str(tuple(int(v) for v in args[6:9]))] += 1
                return selected.launch(*args)

        kernel._ext = DispatchCounter()

    def qwen_gemm_audit(self):
        return {**self._gemm_selection, "python_launch_shapes": dict(self._gemm_calls)}
