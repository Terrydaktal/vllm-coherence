import hashlib
import json
from pathlib import Path

import torch

from qwen_r9700_lab.diagnostic_contract import seal


def test_prior_mutable_buffer_values_do_not_classify_as_a_changed_output(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "experiments/radiance-public")
    )
    from compare_prefill_stage_outputs import compare

    for label, number in [("prefill", 1), ("decode", 2)]:
        root = tmp_path / label
        directory = root / "stages"
        directory.mkdir(parents=True)
        torch.save({"positions": [1081, 1082]}, root / "batch.pt")
        (root / "capture.json").write_text(
            json.dumps({"batches": [{"file": "batch.pt"}]})
        )
        tensors = {}
        events = []
        for idx in range(2):
            event = {"operation": "linear", "index": idx}
            for phase, val in [
                ("before", number),
                ("after", 1 if idx == 0 else number),
            ]:
                key = f"{idx}.{phase}.out"
                event[phase] = [{"key": key, "dtype": "torch.bfloat16"}]
                tensors[key] = torch.tensor([[val]], dtype=torch.bfloat16)
            events.append(event)
        filename = directory / "group-0000.pt"
        torch.save(tensors, filename)
        meta = seal(
            {
                "positions": list(range(1080, 1088)),
                "events": events,
                "tensor_file": filename.name,
                "tensor_sha256": hashlib.sha256(filename.read_bytes()).hexdigest(),
            }
        )
        filename.with_suffix(".json").write_text(json.dumps(meta))
    result = compare(tmp_path, 1081)
    assert result["aligned_calls"] == result["compared_output_tensors"] == 2
    assert [event["prefill_index"] for event in result["different_events"]] == [1]
    assert (
        result["different_events"][0]["outputs"]["differences"]["out"]["different"] == 1
    )
