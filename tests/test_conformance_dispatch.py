from types import ModuleType

import pytest

from qwen_r9700_lab.conformance_dispatch import DispatchRecorder
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, digest, seal


def setup(tmp_path):
    import hashlib

    module = ModuleType("fixture_native")
    library = tmp_path / "fixture.so"
    library.write_bytes(b"synthetic library fixture; never loaded")
    module.__file__ = str(library)

    def kernel(value):
        return value + 1

    def registry():
        pytest.fail("metadata registry should not be executed while attaching")

    module.kernel, module.registry = kernel, registry
    aliases = ModuleType("fixture_glue")
    aliases.direct = kernel
    aliases.table = [kernel, {"decode": kernel}]
    binding = seal(
        {
            "schema": "urn:qwen:native-entrypoint-binding:v1",
            "module": module.__name__,
            "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "exports": {"kernel": "kernel", "registry": "metadata"},
            "required": ["kernel"],
        }
    )
    recorder = DispatchRecorder(tmp_path / "dispatch", execution=digest("synthetic CPU test"))
    return recorder, module, aliases, binding


def test_actual_calls_through_direct_and_container_aliases_are_observed_and_restored(tmp_path):
    recorder, module, aliases, binding = setup(tmp_path)
    original, table = module.kernel, aliases.table
    recorder.bind(module, binding, aliases=[aliases])
    assert module.kernel(1) == 2
    assert aliases.direct(2) == 3
    assert aliases.table[0](3) == 4
    assert aliases.table[1]["decode"](4) == 5
    report = recorder.finish()
    assert len(report["calls"]) == 4
    assert all(row["completed"] for row in report["calls"])
    assert report["exact_device_binary_attested"] is False
    assert report["argument_and_state_equivalence"].startswith("UNPROVED")
    assert module.kernel is aliases.direct is original
    assert aliases.table is table and table[0] is original
    assert all("args" not in row for row in report["calls"])


@pytest.mark.parametrize(
    "fault", ["new_export", "changed_library", "missing_required", "metadata_required"]
)
def test_changed_or_incomplete_dispatch_binding_is_refused_before_mutation(tmp_path, fault):
    recorder, module, aliases, binding = setup(tmp_path)
    original = module.kernel
    if fault == "new_export":
        module.new_kernel = lambda: None
    elif fault == "changed_library":
        binding["library_sha256"] = "0" * 64
    elif fault == "missing_required":
        binding["required"] = []
    else:
        binding["required"] = ["registry"]
    binding = seal({k: v for k, v in binding.items() if k != "sha256"})
    with pytest.raises(DiagnosticError):
        recorder.bind(module, binding, aliases=[aliases])
    assert module.kernel is aliases.direct is original


def test_unobserved_dispatch_cannot_finish_and_hooks_are_removed(tmp_path):
    recorder, module, aliases, binding = setup(tmp_path)
    original = module.kernel
    recorder.bind(module, binding, aliases=[aliases])
    with pytest.raises(DiagnosticError, match="incomplete"):
        recorder.finish()
    assert module.kernel is aliases.direct is original


def test_library_changed_during_capture_invalidates_observation(tmp_path):
    from pathlib import Path

    recorder, module, aliases, binding = setup(tmp_path)
    original = module.kernel
    recorder.bind(module, binding, aliases=[aliases])
    module.kernel(1)
    Path(module.__file__).write_bytes(b"future optimization")
    with pytest.raises(DiagnosticError, match="changed during"):
        recorder.finish()
    assert module.kernel is original


def test_cyclic_alias_container_fails_before_any_hook_mutation(tmp_path):
    recorder, module, aliases, binding = setup(tmp_path)
    aliases.cycle = []
    aliases.cycle.append(aliases.cycle)
    original = module.kernel
    with pytest.raises(DiagnosticError, match="cyclic"):
        recorder.bind(module, binding, aliases=[aliases])
    assert module.kernel is original and not recorder.hooks.entries


def test_failed_entrypoint_keeps_evidence_without_exposing_arguments_or_exception_text(tmp_path):
    from qwen_r9700_lab.diagnostic_contract import private_json

    recorder, module, _, binding = setup(tmp_path)

    def fail(_value):
        raise ValueError("private data must not be copied to the entrypoint log")

    module.kernel = fail
    recorder.bind(module, binding)
    with pytest.raises(ValueError):
        module.kernel("private input")
    path = tmp_path / "dispatch" / "entry-000000000.finished.json"
    report = private_json(path)
    assert report["completed"] is False and report["exception_type"] == "ValueError"
    assert "private" not in path.read_text()
    assert (path.parent / "entry-000000000.started.json").exists()
    with pytest.raises(DiagnosticError, match="incomplete"):
        recorder.finish()
    assert module.kernel is fail
