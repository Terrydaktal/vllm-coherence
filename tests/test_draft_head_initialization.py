"""CPU regression for a DFlash head shared after drafter load_weights returns."""

import ast
import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_patcher():
    spec = importlib.util.spec_from_file_location(
        "patch_draft_head_initialization",
        ROOT / "experiments/radiance-public/patch_draft_head_initialization.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pinned_source(patch):
    repaired = (ROOT / patch.SOURCE).read_text()
    assert repaired.count(patch.NEW) == 1
    original = repaired.replace(patch.NEW, patch.OLD)
    assert hashlib.sha256(original.encode()).hexdigest() == patch.PREIMAGE
    return original


class LogitsProcessor:
    def _apply_head(self, head, hidden, bias):
        return (hidden.float() @ head.weight.float().T) + bias


def load_initialization(source=None):
    # Execute the production initialization and packing functions on CPU. Only
    # the final Triton projection is replaced; no model or GPU is needed.
    tree = ast.parse(source or (ROOT / "radiance_drafthead.py").read_text())
    names = {"_quantize_draft_head", "_apply_head_lazy", "_quantize_head_now"}
    functions = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {n.name for n in functions} == names
    namespace = {
        "torch": torch,
        "types": types,
        "sys": sys,
        "GROUP": 128,
        "BITS": 2,
        "BLOCK_N": 64,
        "KCAND": 8,
        "RERANK": 80,
        "_HEAD_CACHE": {},
    }

    def projected(self, head, hidden, bias):
        return self._radiance_wq, self._radiance_scale, self._radiance_zs

    namespace["_apply_head_int2"] = projected
    exec(  # noqa: S102 -- execute only the selected repository-owned CPU functions.
        compile(
            ast.Module(body=functions, type_ignores=[]), "radiance_drafthead.py", "exec"
        ),
        namespace,
    )
    return namespace


def head(weight):
    return types.SimpleNamespace(weight=weight)


@pytest.mark.parametrize("placeholder", [0.0, 7.0, float("nan")])
@pytest.mark.parametrize("replacement", ["parameter", "head"])
def test_only_real_shared_weights_are_packed_once(
    placeholder, replacement, monkeypatch
):
    namespace = load_initialization()
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    candidate = LogitsProcessor()
    target = LogitsProcessor()
    draft = types.SimpleNamespace(
        lm_head=head(torch.full((64, 256), placeholder)),
        candidate_logits_processor=candidate,
        logits_processor=target,
    )
    namespace["_quantize_draft_head"](draft, "candidate_logits_processor")
    assert not namespace["_HEAD_CACHE"], "load_weights packed the placeholder"
    assert not hasattr(candidate, "_radiance_wq")
    assert candidate._apply_head.__func__ is namespace["_apply_head_lazy"]
    assert target._apply_head.__func__ is LogitsProcessor._apply_head

    actual = torch.linspace(-2, 3, 64 * 256).reshape(64, 256).to(torch.bfloat16)
    if replacement == "head":
        draft.lm_head = head(actual)
    else:
        draft.lm_head.weight = actual
    original = actual.clone()
    packed = candidate._apply_head(draft.lm_head, None, None)
    assert candidate._apply_head.__func__ is namespace["_apply_head_int2"]
    assert set(namespace["_HEAD_CACHE"]) == {(actual.data_ptr(), 64, 256)}
    torch.testing.assert_close(actual, original, rtol=0, atol=0)

    # Independent eager initialization of the known real weights must produce
    # identical packed bytes/scales. No changes to quantization are permitted.
    control = load_initialization()
    expected = LogitsProcessor()
    control["_quantize_head_now"](expected, head(actual))
    for got, want in zip(
        packed, expected._apply_head(head(actual), None, None), strict=True
    ):
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    def unexpected_repack(*args):
        pytest.fail("head was quantized again after first use")

    namespace["_quantize_head_now"] = unexpected_repack
    again = candidate._apply_head(draft.lm_head, None, None)
    assert all(a is b for a, b in zip(packed, again, strict=True))


def test_zero_weight_on_first_use_keeps_stock_fallback_then_can_initialize(monkeypatch):
    namespace = load_initialization()
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    processor = LogitsProcessor()
    draft = types.SimpleNamespace(
        lm_head=head(torch.zeros(64, 256)), logits_processor=processor
    )
    namespace["_quantize_draft_head"](draft)
    got = processor._apply_head(draft.lm_head, torch.ones(1, 256), 2)
    torch.testing.assert_close(got, torch.full((1, 64), 2.0), rtol=0, atol=0)
    assert not namespace["_HEAD_CACHE"]
    draft.lm_head.weight.fill_(1)
    processor._apply_head(draft.lm_head, None, None)
    assert processor._apply_head.__func__ is namespace["_apply_head_int2"]
    assert not processor._radiance_topk_only


@pytest.mark.parametrize(
    "weight", [torch.zeros(256), torch.zeros(64, 256, dtype=torch.int8)]
)
def test_unsupported_weight_keeps_stock_processor(weight):
    namespace = load_initialization()
    processor = LogitsProcessor()
    status = namespace["_quantize_draft_head"](
        types.SimpleNamespace(lm_head=head(weight), logits_processor=processor)
    )
    assert "unsupported" in status
    assert processor._apply_head.__func__ is LogitsProcessor._apply_head
    assert not namespace["_HEAD_CACHE"]


def test_missing_head_keeps_stock_processor():
    namespace = load_initialization()
    processor = LogitsProcessor()
    status = namespace["_quantize_draft_head"](
        types.SimpleNamespace(logits_processor=processor)
    )
    assert status == "no lm_head/logits_processor"
    assert processor._apply_head.__func__ is LogitsProcessor._apply_head


def test_runtime_patch_matches_source_and_is_idempotent(tmp_path, monkeypatch):
    patch = load_patcher()
    module_path = tmp_path / patch.SOURCE
    module_path.write_text(pinned_source(patch))
    receipt = patch.install(tmp_path)
    assert receipt["before"] == patch.PREIMAGE
    assert receipt["after"] == patch.POSTIMAGE
    assert module_path.read_bytes() == (ROOT / patch.SOURCE).read_bytes()
    before_stat = module_path.stat()
    assert patch.install(tmp_path)["before"] == patch.POSTIMAGE
    assert module_path.stat().st_mtime_ns == before_stat.st_mtime_ns

    # Exercise the installed artifact with a dirty placeholder and shared head.
    namespace = load_initialization(module_path.read_text())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    processor = LogitsProcessor()
    draft = types.SimpleNamespace(
        lm_head=head(torch.full((64, 256), 7.0)), candidate_logits_processor=processor
    )
    namespace["_quantize_draft_head"](draft, "candidate_logits_processor")
    assert not namespace["_HEAD_CACHE"]
    shared = head(torch.arange(64 * 256).reshape(64, 256).float())
    processor._apply_head(shared, None, None)
    assert set(namespace["_HEAD_CACHE"]) == {(shared.weight.data_ptr(), 64, 256)}


@pytest.mark.parametrize("original", [True, False])
def test_runtime_patch_refuses_unknown_source_without_writing(tmp_path, original):
    patch = load_patcher()
    source = pinned_source(patch) if original else (ROOT / patch.SOURCE).read_text()
    source += "\n# unrelated unreviewed modification\n"
    module_path = tmp_path / patch.SOURCE
    module_path.write_text(source)
    with pytest.raises(ValueError, match="preimage"):
        patch.install(tmp_path)
    assert module_path.read_text() == source


def test_runtime_patch_checks_result_before_writing(tmp_path, monkeypatch):
    patch = load_patcher()
    source = pinned_source(patch)
    module_path = tmp_path / patch.SOURCE
    module_path.write_text(source)
    monkeypatch.setattr(patch, "POSTIMAGE", "0" * 64)
    with pytest.raises(ValueError, match="postimage"):
        patch.install(tmp_path)
    assert module_path.read_text() == source


def test_packaged_release_includes_guarded_initializer():
    import json

    release = json.loads((ROOT / "releases/0.1.0.json").read_text())
    name = "experiments/radiance-public/patch_draft_head_initialization.py"
    assert name in release["patch_files"]
    assert (
        release["integration_sha256"][name]
        == hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    )
    profile = json.loads(
        (ROOT / "experiments/radiance-public/runtime-radiance-1.0.16.json").read_text()
    )
    patch = load_patcher()
    assert profile["source_preimages"][patch.SOURCE] == patch.PREIMAGE
    assert (
        profile["draft_head_initialization"]["source_postimage_sha256"]
        == patch.POSTIMAGE
    )
