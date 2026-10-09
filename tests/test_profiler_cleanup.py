"""Profiler native resources must not survive into an unprofiled control arm."""

import ast
import contextlib
import gc
import time
import weakref
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_r9700_lab.conformance_instrumentation import HookSet

SOURCE = (
    Path(__file__).parents[1]
    / "experiments/radiance-public/optimized_d7_worker.py"
)
TIMING_SOURCE = SOURCE.with_name("matched_stage_profile_worker.py")


@pytest.fixture(params=[True, False], ids=["gc-enabled", "gc-disabled"])
def gc_policy(request):
    """Keep automatic GC from accidentally masking a deferred native release."""
    original_enabled, original_threshold = gc.isenabled(), gc.get_threshold()
    gc.collect()
    gc.set_threshold(1_000_000, 1_000_000, 1_000_000)
    (gc.enable if request.param else gc.disable)()
    expected = (gc.isenabled(), gc.get_threshold())
    try:
        yield expected
    finally:
        gc.set_threshold(*original_threshold)
        (gc.enable if original_enabled else gc.disable)()
        gc.collect()


def observation_class(torch, gc_module=gc):
    """Execute the real observer class without importing torch or vLLM."""
    tree = ast.parse(SOURCE.read_text())
    node = next(
        item for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "GraphObservation"
    )

    def require(condition, message):
        if not condition:
            raise AssertionError(message)

    namespace = {
        "Counter": Counter,
        "HookSet": HookSet,
        "Path": Path,
        "contextlib": contextlib,
        "gc": gc_module,
        "require": require,
        "time": time,
        "torch": torch,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)  # noqa: S102 -- Only the local observer AST is executed, without GPU imports.
    timing_tree = ast.parse(TIMING_SOURCE.read_text())
    timing_node = next(
        (
            item for item in timing_tree.body
            if isinstance(item, ast.ClassDef)
            and item.name == "TimingGraphObservation"
        ),
        None,
    )
    if timing_node is None:
        # Executing the old base proves the original retained-cycle defect
        # without checking out an earlier revision or changing its source.
        return namespace["GraphObservation"]
    exec(  # noqa: S102 -- Only the local benchmark observer AST is executed.
        compile(ast.Module(body=[timing_node], type_ignores=[]), str(TIMING_SOURCE), "exec"),
        namespace,
    )
    return namespace["TimingGraphObservation"]


def armed_observer(tmp_path, calls, *, failure=None, gc_module=gc):
    def synchronize():
        calls.append("synchronize")
        if failure == "synchronize":
            raise RuntimeError("injected synchronize failure")

    class NativeTrace:
        def __del__(self):
            calls.append("native trace released")

    class Profiler:
        def __init__(self):
            # A cycle models the profiler's native result retained by Python
            # objects until a cyclic collection, rather than reference counting.
            self.cycle = self
            self.native_trace = NativeTrace()

        def stop(self):
            calls.append("stop")
            if failure == "stop":
                raise RuntimeError("injected stop failure")

        def export_chrome_trace(self, path):
            calls.append("export")
            if failure == "export":
                raise OSError("injected export failure")
            Path(path).write_text("{}\n")

    cls = observation_class(
        SimpleNamespace(cuda=SimpleNamespace(synchronize=synchronize)), gc_module
    )
    observer = cls.__new__(cls)
    observer.root = tmp_path
    observer.profile_requested = observer.profile_active = True
    observer.profile_files = []
    observer.profiler = Profiler()
    references = (
        weakref.ref(observer.profiler),
        weakref.ref(observer.profiler.native_trace),
    )
    return observer, references


def test_completed_trace_is_released_before_unprofiled_control(tmp_path, gc_policy):
    calls = []
    observer, references = armed_observer(tmp_path, calls)
    observer.stop_profile()
    calls.append("unprofiled control entry")

    assert all(reference() is None for reference in references), (
        "profiler/native trace survived stop_profile and entered the control arm"
    )
    assert observer.profiler is None and not observer.profile_active
    assert (gc.isenabled(), gc.get_threshold()) == gc_policy
    assert calls[:3] == ["synchronize", "stop", "export"]
    assert calls.index("native trace released") < calls.index("unprofiled control entry")
    assert calls.count("synchronize") == 1
    assert observer.profile_files == [str(tmp_path / "profile-trace.json")]


def test_every_completed_trace_chunk_is_released(tmp_path, gc_policy):
    calls = []
    for index in range(3):
        observer, references = armed_observer(tmp_path / str(index), calls)
        observer.root.mkdir()
        observer.stop_profile()
        assert all(reference() is None for reference in references)
        assert (gc.isenabled(), gc.get_threshold()) == gc_policy
    assert calls.count("native trace released") == 3
    assert calls.count("synchronize") == 3


def test_close_retains_cleanup_receipt_without_native_result(tmp_path, gc_policy):
    calls = []
    observer, references = armed_observer(tmp_path, calls)
    observer.hooks = SimpleNamespace(close=lambda: calls.append("hooks closed"))
    observer.counts = observer.head_shapes = Counter()
    observer.stop_profile()
    report = observer.close()

    assert all(reference() is None for reference in references)
    assert len(report["profile_cleanup"]) == 1
    cleanup = report["profile_cleanup"][0]
    assert cleanup["boundary"] == "excluded_profile_teardown"
    assert cleanup["elapsed_ms"] >= 0
    assert isinstance(cleanup["collected_objects"], int)
    assert cleanup["collected_objects"] >= 0
    assert calls.count("synchronize") == 1  # close must not stop an inactive trace again.
    assert (gc.isenabled(), gc.get_threshold()) == gc_policy


@pytest.mark.parametrize("failure", ["synchronize", "stop", "export"])
def test_failed_trace_clears_observer_and_preserves_original_error(
    tmp_path, gc_policy, failure
):
    calls, collections = [], []

    def collect(*args):
        collections.append(args)
        return gc.collect(*args)

    observer, _references = armed_observer(
        tmp_path, calls, failure=failure,
        gc_module=SimpleNamespace(collect=collect),
    )
    expected_error = OSError if failure == "export" else RuntimeError
    with pytest.raises(expected_error, match=f"injected {failure} failure"):
        observer.stop_profile()

    # The propagated traceback can retain the faulting profiler method frame.
    # No control arm may start after failure. Export fails after the profiler
    # stopped, so discard it; synchronize/stop failures leave it active and must
    # retain it for teardown rather than pretending a safe control can start.
    if failure == "export":
        assert observer.profiler is None and not observer.profile_active
        assert collections, "failed export bypassed out-of-round cleanup"
    else:
        assert observer.profiler is not None and observer.profile_active
        assert not collections, "cleanup ran before the profiler safely stopped"
    assert observer.profile_files == []
    assert (gc.isenabled(), gc.get_threshold()) == gc_policy


def test_cleanup_is_wired_into_benchmark_observers_only():
    timing_tree = ast.parse(TIMING_SOURCE.read_text())
    timing_class = next(
        item for item in timing_tree.body
        if isinstance(item, ast.ClassDef) and item.name == "MatchedStageWorker"
    )
    assignment = next(
        item for item in timing_class.body
        if isinstance(item, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "observation_class"
                for target in item.targets)
    )
    assert isinstance(assignment.value, ast.Name)
    assert assignment.value.id == "TimingGraphObservation"

    speed_tree = ast.parse(SOURCE.with_name("speed_matched_stage_worker.py").read_text())
    full_observer = next(
        item for item in speed_tree.body
        if isinstance(item, ast.ClassDef) and item.name == "FullGraphObservation"
    )
    assert [base.id for base in full_observer.bases] == ["TimingGraphObservation"]
    production_tree = ast.parse(SOURCE.read_text())
    production_observer = next(
        item for item in production_tree.body
        if isinstance(item, ast.ClassDef) and item.name == "GraphObservation"
    )
    production_stop = next(
        item for item in production_observer.body
        if isinstance(item, ast.FunctionDef) and item.name == "stop_profile"
    )
    assert not any(
        isinstance(item, ast.Attribute)
        and isinstance(item.value, ast.Name)
        and item.value.id == "gc"
        for item in ast.walk(production_stop)
    )
