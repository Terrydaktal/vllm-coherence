import contextlib
import importlib.util
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from qwen_r9700_lab.conformance_instrumentation import HookSet


def test_full_replay_scoped_once_and_piecewise_and_drafter_not_duplicated(monkeypatch):
    scopes = []

    class Graph:
        def replay(self):
            return "unchanged"

    class Base:
        pass

    class Observer:
        def __init__(self):
            self.hooks = HookSet()
            self.counts = Counter()
            self.draft = False
            original = Graph.replay
            def counted(graph):
                self.counts["parent_replays"] += 1
                return original(graph)
            self.hooks.replace(Graph, "replay", counted)

        def close(self):
            self.hooks.close()

        @contextlib.contextmanager
        def scope(self, name):
            scopes.append(name)
            yield

    class Matched(Base):
        pass

    class Candidate(Base):
        pass

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(CUDAGraph=Graph)))
    monkeypatch.setitem(sys.modules, "optimized_d7_worker", SimpleNamespace(GraphObservation=Observer))
    monkeypatch.setitem(sys.modules, "matched_stage_profile_worker", SimpleNamespace(MatchedStageWorker=Matched))
    monkeypatch.setitem(sys.modules, "speed_candidate_worker", SimpleNamespace(SpeedCandidateWorker=Candidate))
    path = Path(__file__).parents[1] / "experiments/radiance-public/speed_matched_stage_worker.py"
    spec = importlib.util.spec_from_file_location("speed_matched_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = Graph.replay
    observer = module.FullGraphObservation()
    graph = Graph()
    try:
        assert graph.replay() == "unchanged"
        assert scopes == ["target_body"]
        with observer.scope("target_body"):
            for _ in range(65):
                assert graph.replay() == "unchanged"
        assert scopes == ["target_body", "target_body"]
        observer.draft = True
        graph.replay()
        assert len(scopes) == 2
        assert observer.counts["full_target_graph_replays"] == 1
        assert observer.target_depth == 0
    finally:
        observer.close()
    assert Graph.replay is original


def test_stage_reference_launch_is_original_and_candidate_is_restored(monkeypatch):
    original = lambda *a: "original"
    candidate = lambda *a: "candidate"
    owner = SimpleNamespace(launch=candidate)
    function = lambda *a: owner.launch(*a)
    matrix = SimpleNamespace(current_cases=lambda call: iter([
        ("MLP down projection", "0", {k: function for k in
                                     ("fix1_m1", "fix1_m8", "final_eager_m8", "final_m8")})]))

    class Parent:
        def qwen_optimized_begin(self):
            return {"installed": True}

    monkeypatch.setitem(sys.modules, "native_d7_tape_worker", SimpleNamespace(NativeTapeWorker=Parent))
    monkeypatch.setitem(sys.modules, "speed_candidate_worker", SimpleNamespace(SpeedCandidateWorker=object))
    path = Path(__file__).parents[1] / "experiments/radiance-public/speed_confirmation_worker.py"
    spec = importlib.util.spec_from_file_location("speed_confirmation_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    worker = module.SpeedTapeWorker()
    worker._tape_observer = SimpleNamespace(matrix=matrix)
    worker._speed_original_launch, worker._speed_launch_owner = original, owner
    worker.qwen_optimized_begin()
    _, _, versions = next(matrix.current_cases(SimpleNamespace(name="radiance.mxfp4_linear_pq.default")))
    assert {k: f() for k, f in versions.items()} == {
        "fix1_m1": "original", "final_eager_m8": "original",
        "fix1_m8": "candidate", "final_m8": "candidate"}
    assert owner.launch is candidate
def test_published_runtime_identity_distinguishes_speed_candidates(monkeypatch):
    import copy
    import importlib
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "tools"))
    package = importlib.import_module("package_matched_stage_timings")
    metadata = {
        "enforce_eager": False,
        "compilation_mode": 3,
        "graph_mode": "FULL_AND_PIECEWISE",
        "effective_capacity": 253792,
        "diagnostic_sources": {},
        "repair": {"bundle": "same-base"},
        "performance": {"manifest": "same-base", "gemm_dispatch": {"binary_sha256": "same-base"}},
        "speed_candidate": {"full_graph_enabled": True, "target_gemm": {"binary_sha256": "candidate-A"}},
    }
    other = copy.deepcopy(metadata)
    other["speed_candidate"]["target_gemm"]["binary_sha256"] = "candidate-B"
    assert package.digest(package.runtime(metadata)) != package.digest(package.runtime(other))
    other = copy.deepcopy(metadata)
    other.pop("speed_candidate")
    assert package.digest(package.runtime(metadata)) != package.digest(package.runtime(other))
