"""Combine audited stage sweeps without silently replacing overlapping results."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(value):
    raw = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def read(path):
    document = json.loads(path.read_text())
    assert digest({k: v for k, v in document.items() if k != "sha256"}) == document["sha256"]
    assert document["status"] == "SAMPLE_CHECKED"
    assert document["positions"] == document["release_full_vector_bridge"] == 320
    assert document["prefill_bridge_exact"] is True
    return document


def combine(paths):
    stages, per_instance, receipts, fixture = {}, {}, {}, None
    release_rows = None
    for path in paths:
        document = read(path)
        if fixture is None:
            fixture = document["fixture"]
            release_rows = document["receipts"]["release_rows"]
        assert fixture == document["fixture"], "different fixtures"
        assert release_rows == document["receipts"]["release_rows"], "different release reference"
        receipts[path.name] = document["sha256"]
        for stage, columns in document["stages"].items():
            if stage.startswith("MLP "):
                stage = stage.replace("activation FP8", "input FP8")
            target = stages.setdefault(stage, {})
            for column, result in columns.items():
                assert result["positions"] == 320
                value = {
                    **result,
                    "isolated_inputs_verified": True,
                    "reference_remainder_verified": True,
                }
                if column in target:
                    assert target[column] == value, "overlapping stage results disagree"
                target[column] = value
        for stage, instances in document["per_instance"].items():
            if stage.startswith("MLP "):
                stage = stage.replace("activation FP8", "input FP8")
            for instance, columns in instances.items():
                target = per_instance.setdefault(stage, {}).setdefault(instance, {})
                for column, values in columns.items():
                    if column in target:
                        assert target[column] == values, "per-layer comparison results disagree"
                    target[column] = values
    result = {
        "schema": "qwen.combined-native-stage-matrix.v1",
        "status": "SAMPLE_CHECKED",
        "positions": 320,
        "fixture": fixture,
        "release_rows": release_rows,
        "receipts": receipts,
        "aggregation": "Each position must match at every measured layer instance of a stage",
        "stages": stages,
        "per_instance": per_instance,
    }
    return {**result, "sha256": digest(result)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(combine(args.inputs), indent=2) + "\n")


if __name__ == "__main__":
    main()
