"""Locate an attention discrepancy in captured synthetic M1/M8 activations on CPU."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from analyze_gdn_decode_transition import array, stats

from qwen_r9700_lab.conformance_state import read_frame
from qwen_r9700_lab.diagnostic_contract import authenticate, private_json, seal, write_private


def analyze(root, layer):
    model = private_json(root / "probe-result.json")
    authenticate(model)
    if not model["initial_prefill_exact"] or not model["scope"].startswith("Synthetic "):
        raise ValueError("attention attribution requires an equal synthetic initial state")
    prefix = f"target.language_model.model.layers.{layer}.self_attn"
    arms = []
    for arm in ("serial", "d7"):
        base = root / arm / "run/capture/calls"
        manifest = private_json(base / "calls.json")
        authenticate(manifest)
        selected = {}
        for row in manifest["calls"]:
            if (row["site"] == prefix or row["site"].startswith(prefix + ".")) and row["before"][
                "frame"
            ]:
                if row["site"] in selected:
                    raise ValueError("diagnostic requires exactly one captured call per site")
                selected[row["site"]] = (base / f"call-{row['index']:09d}", row)
        arms.append(selected)
    if not arms[0] or set(arms[0]) != set(arms[1]):
        raise ValueError("attention call coverage differs between the arms")
    comparisons = []
    for site, (left, left_call) in arms[0].items():
        right, right_call = arms[1][site]
        pos = [r["logical"]["positions"] for r in (left_call, right_call)]
        if len(pos[0]) != 1 or not pos[1] or pos[0][0] != pos[1][0]:
            raise ValueError("captured first input positions are not aligned")
        stages = {}
        for stage in ("before", "after"):
            frames = [read_frame(path / stage) for path in (left, right)]
            if set(frames[0]["components"]) != set(frames[1]["components"]):
                raise ValueError("captured tensor components differ")
            components = {}
            for name in frames[0]["components"]:
                values = [array(path / stage, name) for path in (left, right)]
                if values[0].ndim and values[0].shape[0] == 1:
                    values = [x[:1] for x in values]
                elif "positions" in name and values[0].shape == (3, 1):
                    values = [x[:, :1] for x in values]
                record = stats(*values)
                payloads = [np.ascontiguousarray(v).tobytes() for v in values]
                record["exact_equal"] = payloads[0] == payloads[1]
                record["sha256"] = [hashlib.sha256(v).hexdigest() for v in payloads]
                components[name] = record
            stages[stage] = components
        comparisons.append({"site": site, "position": pos[0][0], **stages})
    return seal(
        {
            "model": model["sha256"],
            "comparisons": comparisons,
            "scope": "One aligned synthetic position; CPU analysis of saved attention activations.",
            "formal_equivalence": "UNPROVED",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--layer", type=int, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.layer)
    write_private(args.root / "attention-transition-analysis.json", result)
    for row in result["comparisons"]:
        print(
            row["site"],
            {
                phase: [
                    (key, x["exact_equal"], x["different_elements"], x["max_abs"])
                    for key, x in row[phase].items()
                ]
                for phase in ("before", "after")
            },
        )
    print(result["sha256"])
