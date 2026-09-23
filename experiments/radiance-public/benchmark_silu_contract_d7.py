"""Controlled diagnostic: retain native BF16 SiLU rounding in compiled target MLPs.

The pinned benchmark, repaired kernels, sampling and cache settings are reused.
Only the compiler's silu_and_mul custom-op choice changes. This is an experiment,
not a production default or a release-speed qualification.
"""

import hashlib
import sys
from pathlib import Path

import benchmark_optimized_d7 as benchmark

from qwen_r9700_lab.diagnostic_contract import DiagnosticError

BASE_CONFIG = benchmark.make_config


def make_config(spec, lane, **kwargs):
    if lane != "fixed-bf16" or kwargs.get("execution_mode", "compiled") == "eager":
        raise DiagnosticError("SiLU intervention requires the fixed compiled lane")
    result = BASE_CONFIG(spec, lane, **kwargs)
    if "custom_ops" in result["compilation_config"]:
        raise DiagnosticError("review an existing custom-op override before this intervention")
    result["compilation_config"]["custom_ops"] = ["none", "+silu_and_mul"]
    return result


def main():
    if "--correctness" not in sys.argv or "--profile" in sys.argv:
        raise DiagnosticError(
            "SiLU intervention requires correctness replay and excludes profiling"
        )
    # Bind the reused driver's contents as well as this wrapper. The wrapper is
    # the subprocess entry point, so the same intervention applies in workers.
    source = Path(benchmark.__file__).read_bytes()
    if hashlib.sha256(source).hexdigest() != BASE_DRIVER_SHA256:
        raise DiagnosticError(
            "underlying benchmark driver changed; review and rebind the experiment"
        )
    benchmark.make_config = make_config
    benchmark.__file__ = __file__
    benchmark.main()


# Filled from the reviewed driver; intentional changes require a new binding.
BASE_DRIVER_SHA256 = "f43594823dfe3a34be199059bf045232f7e397020889851b9b2d206a7875af71"


if __name__ == "__main__":
    main()
