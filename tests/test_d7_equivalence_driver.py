import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    private_json,
    seal,
    write_private,
)


@pytest.fixture
def driver():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/radiance-public/benchmark_d7_equivalence.py"
    )
    spec = importlib.util.spec_from_file_location("d7_equivalence_driver", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize("wrapped", [False, True])
def test_bf16_target_head_inside_the_conditional_model_wrapper(driver, wrapped):
    target = SimpleNamespace(
        lm_head=SimpleNamespace(weight=SimpleNamespace(dtype="torch.bfloat16", shape=(128, 64)))
    )
    model = SimpleNamespace(language_model=target) if wrapped else target
    assert driver.target_head_identity(model) == {"dtype": "torch.bfloat16", "shape": [128, 64]}


def test_non_bf16_or_missing_vocabulary_head_is_rejected(driver):
    with pytest.raises(DiagnosticError, match="missing target"):
        driver.target_head_identity(SimpleNamespace(language_model=SimpleNamespace()))
    target = SimpleNamespace(
        lm_head=SimpleNamespace(weight=SimpleNamespace(dtype="torch.int8", shape=(128, 64)))
    )
    with pytest.raises(DiagnosticError, match="not BF16"):
        driver.target_head_identity(target)


@pytest.fixture
def reusable_m1(tmp_path):
    corpus = tmp_path / "corpus"
    output = tmp_path / "candidate-report"
    source = tmp_path / "reference-report"
    for p in (corpus, output, source, corpus / "candidate", corpus / "reference"):
        p.mkdir()
    (corpus / "reference/m1/000").mkdir(parents=True)
    spec = {"binding": {"sha256": "a" * 64}}
    spec_path = tmp_path / "spec.json"
    write_private(spec_path, spec)
    before, after = seal({"kind": "old"}), seal({"kind": "new"})
    repair = tmp_path / "repair.json"
    write_private(repair, after)
    fixture = {"sha256": "b" * 64, "prefix_tokens": 413, "evaluate_positions": 2}
    manifest = seal({"positions": 2, "continuations": [fixture]})
    write_private(corpus / "manifest.json", manifest)
    row = seal(
        {
            "continuation": fixture["sha256"],
            "repair_receipt": {
                "bundle": before["sha256"],
                "target_arithmetic": {"norm_calls": 322, "norm_rows": 322},
            },
            "rows": [
                {"target_rows": 1, "position": i, "absolute_position": 413 + i} for i in range(2)
            ],
        }
    )
    write_private(corpus / "reference/m1/000/rows.json", row)
    complete = seal({"corpus": manifest["sha256"], "receipts": [[{"sha256": row["sha256"]}]]})
    write_private(source / "m1-complete.json", complete)
    write_private(source / "m1-process-result.json", {"returncode": 0})
    write_private(
        source / "measurement.json",
        seal(
            {
                "revision": "reference",
                "native_binding": spec["binding"]["sha256"],
                "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                "repair_hooks": [before],
            }
        ),
    )
    write_private(
        source / "m1-config.json",
        seal(
            {
                "max_num_batched_tokens": 2048,
                "enforce_eager": True,
                "max_num_seqs": 1,
                "async_scheduling": False,
            }
        ),
    )
    decision = seal(
        {
            "schema": "urn:qwen:d7-m1-reuse:v1",
            "corpus": manifest["sha256"],
            "candidate_bundle": after["sha256"],
            "native_binding": spec["binding"]["sha256"],
            "reference_revision": "reference",
            "reference_report": str(source),
            "reference_complete": complete["sha256"],
            "reference_bundle": before["sha256"],
        }
    )
    path = tmp_path / "decision.json"
    write_private(path, decision)
    return (
        SimpleNamespace(
            reuse_m1=path,
            corpus=corpus,
            output=output,
            repair_manifest=repair,
            spec=spec_path,
            revision="candidate",
        ),
        spec,
        row,
        source,
    )


def test_m1_reuse_preserves_published_identity_and_labels_old_measurement(driver, reusable_m1):
    args, spec, row, _ = reusable_m1
    report = driver.reuse_m1_reference(args, spec)
    assert report["positions"] == 2
    assert report["reference_rows"] == [row["sha256"]]
    assert private_json(args.corpus / "candidate/m1/000/rows.json") == row
    assert private_json(args.corpus / "reference/m1/000/rows.json") == row


@pytest.mark.parametrize(
    "fault", ["backend", "bundle", "incomplete", "payload", "batch-size", "small-prefill"]
)
def test_m1_reuse_rejects_changed_or_incomplete_reference(driver, reusable_m1, fault):
    args, spec, row, source = reusable_m1

    def damage(path, value):
        path.write_text(json.dumps(value))

    if fault in ("backend", "bundle"):
        doc = private_json(args.reuse_m1)
        doc.pop("sha256")
        doc["native_binding" if fault == "backend" else "candidate_bundle"] = "c" * 64
        damage(args.reuse_m1, seal(doc))
    elif fault == "incomplete":
        damage(source / "m1-process-result.json", {"returncode": 130})
    elif fault == "payload":
        row["rows"][0]["absolute_position"] += 1
        damage(args.corpus / "reference/m1/000/rows.json", row)
    elif fault == "small-prefill":
        # Keep every published identity consistent, but record an additional
        # two-token prefill that enters the changed normalization branch.
        row["repair_receipt"]["target_arithmetic"].update(norm_calls=483, norm_rows=644)
        row.pop("sha256")
        row = seal(row)
        damage(args.corpus / "reference/m1/000/rows.json", row)
        complete = private_json(source / "m1-complete.json")
        complete.pop("sha256")
        complete["receipts"][0][0]["sha256"] = row["sha256"]
        complete = seal(complete)
        damage(source / "m1-complete.json", complete)
        decision = private_json(args.reuse_m1)
        decision.pop("sha256")
        decision["reference_complete"] = complete["sha256"]
        damage(args.reuse_m1, seal(decision))
    else:
        config = private_json(source / "m1-config.json")
        config.pop("sha256")
        config["max_num_batched_tokens"] = 1024
        damage(source / "m1-config.json", seal(config))
    with pytest.raises(DiagnosticError):
        driver.reuse_m1_reference(args, spec)
    assert not (args.corpus / "candidate/m1").exists()
