from copy import deepcopy

import numpy as np
import pytest

from qwen_r9700_lab.conformance_serial_boundaries import admit_serial_pair, pack_serial_groups
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal


def groups():
    result = []
    for row in range(8):
        before, after = f"{row}.before.args.0", f"{row}.after.result"
        x = np.full((1, 4), row, dtype=np.int16)
        y = x + 1
        result.append(
            (
                seal(
                    {
                        "positions": [60000 + row],
                        "events": [
                            {
                                "index": row,
                                "operation": "example.linear",
                                "logical_identities": ["layer.0.weight"],
                                "before": [
                                    {"key": before, "shape": [1, 4], "dtype": "torch.bfloat16"}
                                ],
                                "after": [
                                    {"key": after, "shape": [1, 4], "dtype": "torch.bfloat16"}
                                ],
                            }
                        ],
                    }
                ),
                {before: x, after: y},
            )
        )
    return result


def test_packs_logical_rows_despite_different_event_ids_without_mutating_sources():
    source = groups()
    metadata = deepcopy([m for m, _ in source])
    result, tensors = pack_serial_groups(source)
    assert result["positions"] == list(range(60000, 60008))
    assert result["source_captures"] == [m["sha256"] for m in metadata]
    assert np.array_equal(tensors["0.after.result"][:, 0], np.arange(1, 9))
    assert metadata == [m for m, _ in source]


@pytest.mark.parametrize(
    "fault", ["reorder", "duplicate", "missing", "changed_owner", "changed_role", "layout"]
)
def test_rejects_incomplete_or_non_corresponding_serial_rows(fault):
    source = groups()
    if fault == "reorder":
        source[0], source[1] = source[1], source[0]
    elif fault == "duplicate":
        source[1] = source[0]
    elif fault == "missing":
        source.pop()
    else:
        m, t = source[-1]
        m.pop("sha256")
        event = m["events"][0]
        if fault == "changed_owner":
            event["logical_identities"] = ["layer.1.weight"]
        elif fault == "changed_role":
            event["before"][0]["key"] = "7.before.args.1"
            t["7.before.args.1"] = t.pop("7.before.args.0")
        elif fault == "layout":
            t["7.before.args.0"] = np.zeros((2, 4), dtype=np.int16)
        source[-1] = seal(m), t
    with pytest.raises(DiagnosticError):
        pack_serial_groups(source)


def serial_experiment():
    from test_conformance_execution_modes import reseal, saved_rows, side

    serial = side("compiled-no-graphs", saved_rows(), capture=True)
    batched = deepcopy(serial)
    serial["measurement"]["m1"] = True
    serial["measurement"] = reseal(serial["measurement"])
    batched["config"]["speculative_config"] = {"method": "dflash", "num_speculative_tokens": 7}
    batched["config"] = reseal(batched["config"])
    serial["runtime"]["effective_capacity"]["block_size"] = 1568
    serial["runtime"]["effective_capacity"]["num_gpu_blocks"] = 194
    serial["runtime"] = reseal(serial["runtime"])
    return serial, batched


def test_serial_admission_retains_different_physical_capacity():
    serial, batched = serial_experiment()
    result = admit_serial_pair(serial, batched)
    assert result["capacity"][0] != result["capacity"][1]
    assert result["receipts"][0][2] == serial["runtime"]["sha256"]


@pytest.mark.parametrize("field", ["max_num_seqs", "max_num_batched_tokens", "cache_dtype"])
def test_serial_admission_rejects_other_capacity_changes(field):
    from test_conformance_execution_modes import reseal

    serial, batched = serial_experiment()
    serial["runtime"]["effective_capacity"][field] = "changed"
    serial["runtime"] = reseal(serial["runtime"])
    with pytest.raises(DiagnosticError):
        admit_serial_pair(serial, batched)
