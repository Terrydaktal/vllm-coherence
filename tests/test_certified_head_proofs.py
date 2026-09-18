from __future__ import annotations

import hashlib
import inspect

import z3

from qwen_r9700_lab.certified_head import strictly_excluded
from qwen_r9700_lab.certified_head_proofs import run


def test_actual_exclusion_predicate_and_negative_controls(tmp_path):
    root = tmp_path / "proofs"
    report = run(root)
    assert report["all_expected_results"]
    assert report["native_status"] == "UNPROVED"
    assert (
        report["predicate_sha256"]
        == hashlib.sha256(inspect.getsource(strictly_excluded).encode()).hexdigest()
    )
    assert len(report["cases"]) == 14
    for case in report["cases"]:
        assert case["domain_result"] == "sat"
        assert case["result"] == ("sat" if case["name"].startswith("mutant_") else "unsat")
        query = (root / (case["name"] + ".smt2")).read_bytes()
        assert hashlib.sha256(query).hexdigest() == case["query_sha256"]
        # Re-run the saved artifact rather than only trusting report fields.
        solver = z3.Solver()
        solver.from_string(query.decode())
        assert str(solver.check()) == case["result"]


def test_solver_unknown_cannot_become_a_certificate(tmp_path, monkeypatch):
    original = z3.Solver

    def unknown():
        solver = original()
        solver.check = lambda: z3.unknown
        solver.reason_unknown = lambda: "injected timeout"
        return solver

    monkeypatch.setattr(z3, "Solver", unknown)
    report = run(tmp_path / "unknown")
    assert not report["all_expected_results"]
    assert all(c["status"] == "UNPROVED" for c in report["cases"])
