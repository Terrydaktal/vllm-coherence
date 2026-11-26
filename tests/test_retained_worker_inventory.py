"""Check optional imports reached only inside a worker after model loading."""

import ast
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "experiments" / "radiance-public"
sys.path.insert(0, str(SOURCE))
from build_retained_worker import MODULES

# These are supplied by the authenticated baseline, or by its stage bootstrap.
RELEASE = {
    "build_mxfp4_dispatch",
    "extended_context_worker",
    "host_embedding",
    "optimized_prefill_scan",
    "prefill_gemm_alignment",
    "prepared_prefill_scan",
    "speed_matched_stage_worker",
    "stock_fp8_epilogue",
    "stock_gdn_prefill_adapter",
    "stock_gdn_sequence",
}


def test_retained_operators_have_their_runtime_helpers():
    missing = {}
    for module in MODULES:
        tree = ast.parse((SOURCE / f"{module}.py").read_text())
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        uncovered = sorted(
            name
            for name in imports - set(MODULES) - RELEASE
            if (SOURCE / f"{name}.py").is_file()
        )
        if uncovered:
            missing[module] = uncovered
    assert not missing, missing
