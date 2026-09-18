#!/usr/bin/env python3
"""CPU-only checks for FP8-KV aggregation, artifact validation, and tamper gates."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import radiance_kv_calibration as calibration


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    reports = [
        {
            "rank": 0,
            "layers": {
                "model.layers.3.self_attn.attn": {
                    "q": 2,
                    "k": 4,
                    "v": 8,
                    "calls": 2,
                }
            },
        },
        {
            "rank": 1,
            "layers": {
                "model.layers.3.self_attn.attn": {
                    "q": 3,
                    "k": 2,
                    "v": 9,
                    "calls": 2,
                }
            },
        },
    ]
    merged = calibration.aggregate_worker_stats(reports)
    assert merged["model.layers.3.self_attn.attn"] == {
        "q": 3.0,
        "k": 4.0,
        "v": 9.0,
        "calls": 4,
    }
    scales = calibration.make_scale_values(merged, fp8_max=448, margin=1.05)
    assert scales["model.layers.3.self_attn.attn.v_scale"] == 9 * 1.05 / 448
    assert scales["model.layers.3.self_attn.attn.prob_scale"] == 1.05 / 448

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text('{"model_type":"fixture"}\n', encoding="utf-8")
        save_file({"weight": np.ones((1,), dtype=np.float32)}, model / "model.safetensors")

        sidecar = root / "scales.safetensors"
        save_file(
            {name: np.asarray(value, dtype=np.float32) for name, value in scales.items()},
            sidecar,
            metadata={"format": "radiance-fp8-kv-scales-v1"},
        )
        manifest = {
            "schema_version": calibration.SCHEMA_VERSION,
            "artifact": {"sha256": sha256(sidecar)},
            "source_model": calibration.model_identity(model),
        }
        calibration.manifest_path(sidecar).write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        keys = calibration.validate_scale_sidecar(sidecar, source_model=model)
        assert len(keys) == 4
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None:
            loaded = dict(
                calibration.iter_scale_sidecar(sidecar, source_model=model)
            )
            assert set(loaded) == set(keys)
            assert all(value.dtype == torch.float32 for value in loaded.values())

        # Model binding is strict.
        (model / "config.json").write_text('{"model_type":"changed"}\n', encoding="utf-8")
        try:
            calibration.validate_scale_sidecar(sidecar, source_model=model)
        except ValueError as exc:
            assert "different checkpoint" in str(exc)
        else:
            raise AssertionError("model mismatch was accepted")

        # Artifact tampering is independently rejected.
        manifest["source_model"] = calibration.model_identity(model)
        calibration.manifest_path(sidecar).write_text(json.dumps(manifest), encoding="utf-8")
        sidecar.write_bytes(sidecar.read_bytes() + b"tamper")
        try:
            calibration.validate_scale_sidecar(sidecar, source_model=model)
        except ValueError as exc:
            assert "checksum mismatch" in str(exc)
        else:
            raise AssertionError("tampered sidecar was accepted")

    print("FP8-KV calibration CPU checks: PASS")


if __name__ == "__main__":
    main()
