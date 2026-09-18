from __future__ import annotations

import hashlib
import inspect

from qwen_r9700_lab.conformance_gate import publication_allowed
from qwen_r9700_lab.conformance_invariants import unresolved_frontiers_separated
from qwen_r9700_lab.conformance_proofs import check_obligation, run_obligations


def test_actual_helpers_have_scoped_proofs_and_negative_controls(tmp_path):
    result = run_obligations(tmp_path / "proofs")
    assert result["all_expected_results"]
    assert (
        result["functions_sha256"]["publication_allowed"]
        == hashlib.sha256(inspect.getsource(publication_allowed).encode()).hexdigest()
    )
    cases = {row["name"]: row for row in result["cases"]}
    assert len(cases) == 59
    for name in (
        "mxfp4_nibble_roundtrip",
        "canonical_slot_inverse_block",
        "canonical_slot_inverse_offset",
        "accepted_state_column",
        "accepted_conv_window_range",
        "transaction_requires_completed_writes",
        "transaction_rejects_stale_revision",
        "all_state_versions_match",
        "snapshot_requires_verified_durable_current",
        "argmax_margin_bound",
        "two_frontier_topk_bound",
        "published_prefix_induction_step",
        "bf16_rounding_matches_rne",
        "bf16_halfway_ties_even",
    ):
        assert cases[name]["status"] == "PROVED"
    assert cases["mutant_unshifted_conv_window"]["result"] == "sat"
    assert cases["gate_requires_state"]["status"] == "PROVED"
    assert cases["mutant_output_only_gate"]["result"] == "sat"
    assert cases["mutant_commit_all"]["result"] == "sat"
    assert cases["float32_reassociation_counterexample"]["result"] == "sat"
    for name in (
        "mutant_local_only_omits_global_candidate",
        "mutant_nonstrict_cutoff_accepts_omitted_tie",
        "mutant_frontier_coverage_gap",
    ):
        assert cases[name]["result"] == "sat"
    assert (
        result["functions_sha256"]["unresolved_frontiers_separated"]
        == hashlib.sha256(inspect.getsource(unresolved_frontiers_separated).encode()).hexdigest()
    )
    assert result["radiance_status"] == "UNPROVED"
    for name, row in cases.items():
        payload = (tmp_path / "proofs" / (name + ".smt2")).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == row["query_sha256"]
        assert row["domain_result"] == "sat"
        if row["result"] == "sat":
            witness = (tmp_path / "proofs" / (name + ".counterexample.smt2")).read_bytes()
            assert hashlib.sha256(witness).hexdigest() == row["counterexample_sha256"]


def test_two_frontiers_reject_omitted_global_winner_and_ties():
    # A local frontier alone would approve both incomplete candidate sets.
    assert not unresolved_frontiers_separated(20, 3, 21, True, True)
    assert not unresolved_frontiers_separated(20, 3, 20, True, True)
    assert unresolved_frontiers_separated(20, 3, 19, True, True)
    # Only an actually empty partition may be omitted from the comparison.
    assert unresolved_frontiers_separated(20, 300, 19, False, True)
    assert not unresolved_frontiers_separated(20, 300, 19, True, True)


def test_depth_twenty_local_certificate_misses_global_eighty_first():
    # Five 64-token blocks each emit 20 candidates. The global coarse top 80
    # discards a token that really belongs to the exact top 20. Its local
    # block did emit it, so the local discarded frontier cannot detect it.
    emitted = [block * 64 + offset for block in range(5) for offset in range(20)]
    coarse = [-1000] * 320
    reference = [-1000] * 320
    for rank, token in enumerate(emitted):
        coarse[token] = 100 - rank
        reference[token] = 300 - rank if rank < 19 else 0
    reference[emitted[80]] = 100
    error = 400
    assert all(abs(a - b) <= error for a, b in zip(reference, coarse, strict=True))

    rescored = set(sorted(emitted, key=lambda token: coarse[token], reverse=True)[:80])
    cutoff = sorted((reference[token] for token in rescored), reverse=True)[19]
    local = set(range(320)) - set(emitted)
    global_discarded = set(emitted) - rescored
    local_upper = max(coarse[token] + error for token in local)
    global_upper = max(coarse[token] + error for token in global_discarded)
    assert cutoff > local_upper  # The proposed local-only check would accept.
    assert not unresolved_frontiers_separated(cutoff, local_upper, global_upper, True, True)
    true_top_twenty = set(sorted(range(320), key=lambda token: reference[token], reverse=True)[:20])
    assert emitted[80] in true_top_twenty - rescored


def test_unknown_solver_result_never_becomes_proved(tmp_path, monkeypatch):
    import z3

    original = z3.Solver

    def unknown_solver():
        solver = original()
        solver.check = lambda: z3.unknown
        solver.reason_unknown = lambda: "injected timeout"
        return solver

    monkeypatch.setattr(z3, "Solver", unknown_solver)
    result = run_obligations(tmp_path / "unproved")
    assert not result["all_expected_results"]
    assert all(row["status"] == "UNPROVED" for row in result["cases"])


def test_inconsistent_assumptions_are_not_vacuous_proof(tmp_path):
    import z3

    n = z3.Int("n")
    result = check_obligation(
        tmp_path, "empty", n != n, expected=z3.unsat, assumptions=[n > 0, n < 0]
    )
    assert result["result"] == "unsat"
    assert result["status"] == "UNPROVED"
    assert result["domain_result"] == "unsat"
