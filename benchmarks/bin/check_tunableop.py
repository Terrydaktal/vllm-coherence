#!/usr/bin/env python3
"""CPU-only checks for persisted TunableOp manifest validation."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import radiance_tunableop


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "results0.csv").write_text(
            "Validator,PT_VERSION,test\nGemmTunableOp_float_NT,nt_1_1_1,Default,1.0\n",
            encoding="utf-8",
        )
        (root / "results1.csv").write_text(
            "Validator,PT_VERSION,test\nGemmTunableOp_float_NT,nt_1_1_1,Default,1.0\n",
            encoding="utf-8",
        )
        radiance_tunableop.write_manifest(root, input_pattern="untuned?.csv", num_gpus=2)
        assert radiance_tunableop.verify(root)["num_gpus"] == 2
        (root / "results1.csv").write_text("tampered\n", encoding="utf-8")
        try:
            radiance_tunableop.verify(root)
        except ValueError as exc:
            assert "checksum mismatch" in str(exc)
        else:
            raise AssertionError("tampered TunableOp result was accepted")
    print("TunableOp manifest CPU checks: PASS")


if __name__ == "__main__":
    main()
