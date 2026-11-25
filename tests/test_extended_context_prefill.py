"""Test bound context admission without importing GPU libraries."""

import hashlib
import importlib
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_r9700_lab.diagnostic_contract import seal, write_private

RUNTIME = Path(__file__).resolve().parents[1] / "experiments/radiance-public"


@pytest.fixture
def parent(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(RUNTIME))
    overlay = importlib.import_module("extended_context_prefill")
    routing = importlib.import_module("prefill_alignment_runtime")
    attention = importlib.import_module("prefill_attention_alignment")
    monkeypatch.setattr(overlay, "_installed", None)
    monkeypatch.setattr(routing, "prefill_admitted", routing.prefill_admitted)
    monkeypatch.setattr(
        attention.AlignedPrefillAttention,
        "__call__",
        attention.AlignedPrefillAttention.__call__,
    )

    native = tmp_path / "attention"
    native.mkdir()
    source = native / "aligned-prefill.hip"
    source.write_text("fixture for the authenticated fixed-32 parent\n")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    build = seal({"files": {source.name: source_sha256}})
    write_private(native / "build.json", build)
    entry = {
        "sources": dict(overlay.SOURCE_HASHES),
        "attention": {"build": str(native), "build_sha256": build["sha256"]},
    }
    manifest = tmp_path / "performance.json"
    write_private(manifest, seal({"prefill_alignment": entry}))
    validations = []
    monkeypatch.setenv("QWEN_OPTIMIZED_PERFORMANCE", str(manifest))
    monkeypatch.setattr(overlay, "ATTENTION_BUILD_SHA256", build["sha256"])
    monkeypatch.setattr(overlay, "ATTENTION_HIP_SHA256", source_sha256)
    monkeypatch.setattr(routing, "validate", validations.append)
    return overlay, routing, attention, source, validations


class Tensor:
    def __init__(self, shape, dtype, device):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.ndim = len(shape)

    def stride(self, dim):
        return math.prod(self.shape[dim + 1 :])

    def is_contiguous(self):
        return True

    def numel(self):
        return math.prod(self.shape)

    def data_ptr(self):
        return id(self)


@pytest.mark.parametrize("context_limit", (319_000, 334_048, 344_000))
def test_both_authenticated_guards_pass_the_requested_bound_to_the_same_kernel(
    parent, monkeypatch, context_limit
):
    overlay, routing, attention, _, validations = parent
    original = routing.prefill_admitted
    original_launch = attention.AlignedPrefillAttention.__call__
    implementation = SimpleNamespace(
        num_heads=24, num_kv_heads=4, head_size=256, scale=1 / 16
    )
    metadata = SimpleNamespace(
        r4d_plan=((0, 1, 9, 0),), causal=True, r4d_max_ctx=context_limit
    )
    assert not original(implementation, metadata)

    receipt = (
        overlay.install()
        if context_limit == 319_000
        else overlay.install(context_limit)
    )
    assert (
        receipt["previous_context"] == 253_792
        and receipt["extended_context"] == context_limit
    )
    assert receipt["kernel_changed"] is False
    assert overlay.install(context_limit) == receipt
    assert len(validations) == 1
    assert routing.prefill_admitted(implementation, metadata)
    metadata.r4d_max_ctx = context_limit + 1
    assert not routing.prefill_admitted(implementation, metadata)
    metadata.r4d_max_ctx = context_limit
    assert not routing.prefill_admitted(implementation, metadata, output_scale=1)

    # Each compiled callable has the same instructions and only the context
    # constant changed. The runtime symbol still receives a 32-split launch.
    for before, after in (
        (original, routing.prefill_admitted),
        (original_launch, attention.AlignedPrefillAttention.__call__),
    ):
        assert before.__code__.co_code == after.__code__.co_code
        assert after.__code__.co_consts == tuple(
            context_limit if x == 253_792 else x for x in before.__code__.co_consts
        )

    torch = SimpleNamespace(
        bfloat16="bf16",
        float8_e4m3fn="fp8",
        uint8="uint8",
        int32="int32",
        _C=SimpleNamespace(_overlaps=lambda *_: False),
        cuda=SimpleNamespace(current_stream=lambda: SimpleNamespace(cuda_stream=7)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    device = SimpleNamespace(type="cuda")
    query = Tensor((9, 24, 256), torch.bfloat16, device)
    output = Tensor(query.shape, query.dtype, device)
    kv = Tensor((1, 4, 16, 512), torch.float8_e4m3fn, device)
    table = Tensor((1, (context_limit + 15) // 16), torch.int32, device)
    lengths = Tensor((1,), torch.int32, device)
    scratch = Tensor((4096,), torch.uint8, device)
    launches = []

    def launch(args, kvp, stream):
        launches.append((args._obj.max_ctx, args._obj.splits, kvp, stream))
        return 0

    instance = object.__new__(attention.AlignedPrefillAttention)
    instance.manifest, instance.launch = {"scratch_bytes": 4096}, launch
    for bound in (253_792, 262_144, context_limit):
        assert (
            instance(query, kv, table, lengths, scratch, output, context_limit=bound)
            is output
        )
    with pytest.raises(ValueError, match="outside admitted contract"):
        instance(
            query, kv, table, lengths, scratch, output, context_limit=context_limit + 1
        )
    assert launches == [
        (253_792, 32, 0, 7),
        (262_144, 32, 0, 7),
        (context_limit, 32, 0, 7),
    ]


@pytest.mark.parametrize("bound", (True, "319000", 319_000.0, 253_792, 344_001))
def test_unsupported_selected_bound_fails_before_binding(parent, bound):
    overlay, routing, attention, _, validations = parent
    originals = routing.prefill_admitted, attention.AlignedPrefillAttention.__call__
    with pytest.raises(ValueError, match="context must be between"):
        overlay.install(bound)
    assert originals == (
        routing.prefill_admitted,
        attention.AlignedPrefillAttention.__call__,
    )
    assert not validations


def test_process_cannot_change_the_bound_after_installing(parent):
    overlay, routing, attention, _, validations = parent
    overlay.install(326_400)
    installed = routing.prefill_admitted, attention.AlignedPrefillAttention.__call__
    with pytest.raises(ValueError, match="different extended prefill contract"):
        overlay.install(344_000)
    assert installed == (
        routing.prefill_admitted,
        attention.AlignedPrefillAttention.__call__,
    )
    assert len(validations) == 1


def test_native_source_mismatch_does_not_partially_install_the_overlay(parent):
    overlay, routing, attention, source, _ = parent
    before = routing.prefill_admitted, attention.AlignedPrefillAttention.__call__
    source.write_text("different native kernel")
    with pytest.raises(ValueError, match="audited fixed-32 native source and build"):
        overlay.install()
    assert before == (
        routing.prefill_admitted,
        attention.AlignedPrefillAttention.__call__,
    )


def test_loaded_function_fingerprint_mismatch_leaves_other_guard_intact(
    parent, monkeypatch
):
    overlay, routing, attention, _, _ = parent
    original = routing.prefill_admitted
    monkeypatch.setattr(attention.AlignedPrefillAttention, "__call__", lambda *_: None)
    with pytest.raises(ValueError, match="loaded prefill function changed"):
        overlay.install()
    assert routing.prefill_admitted is original
