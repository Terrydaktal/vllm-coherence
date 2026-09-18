"""CPU regressions against exact pinned upstream source, without importing Torch.

The surrounding allocator/runner objects are synthetic. These tests establish
Python state transitions, not that native kernels or a full server are qualified.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import gzip
import hashlib
import importlib.util
import itertools
import json
import runpy
import sys
import types
from _hashlib import UnsupportedDigestmodError
from collections import deque
from pathlib import Path
from types import SimpleNamespace as Namespace

import numpy as np
import pytest
import regex

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "experiments/radiance-public"
SPEC = importlib.util.spec_from_file_location(
    "upstream_backports", RELEASE / "patch_upstream_correctness.py"
)
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)
ORIGINAL = json.loads(
    gzip.decompress((ROOT / "tests/fixtures/radiance_1_0_16_upstream_sources.json.gz").read_bytes())
)
MANIFEST = json.loads((PATCHER.BUNDLE / "manifest.json").read_text())
NATIVE = json.loads(
    gzip.decompress((ROOT / "tests/fixtures/radiance_upstream_native_sources.json.gz").read_bytes())
)
CORRECTED = dict(ORIGINAL)
for ENTRY in MANIFEST["components"]["python"]["files"]:
    CORRECTED[ENTRY["path"]] = PATCHER.apply_exact_patch(
        ORIGINAL[ENTRY["path"]], (PATCHER.BUNDLE / ENTRY["patch"]).read_text()
    )


def extract(source, name, *, kind=ast.FunctionDef, owner=None):
    tree = ast.parse(source)
    if owner:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    matches = [n for n in ast.walk(tree) if isinstance(n, kind) and n.name == name]
    assert len(matches) == 1, (name, len(matches))
    return matches[0]


def execute(nodes, namespace=None):
    ns = {} if namespace is None else namespace
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "<pinned-backport-fixture>", "exec"), ns)
    return ns


def tree(root):
    for name, source in ORIGINAL.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return root


def test_bundle_is_exact_idempotent_and_validates_every_file_before_writing(tmp_path):
    root = tree(tmp_path)
    receipt = PATCHER.install(root, check_only=True)
    assert all(x["status"] == "would_apply" for x in receipt["files"])
    assert (root / "vllm/config/speculative.py").read_text() == ORIGINAL[
        "vllm/config/speculative.py"
    ]
    PATCHER.install(root)
    assert all((root / p).read_text() == source for p, source in CORRECTED.items())
    assert all(x["status"] == "already_applied" for x in PATCHER.install(root)["files"])
    tree(root)
    last = MANIFEST["components"]["python"]["files"][-1]["path"]
    (root / last).write_text("# unknown source\n")
    with pytest.raises(ValueError, match="unknown source preimage"):
        PATCHER.install(root)
    first = MANIFEST["components"]["python"]["files"][0]["path"]
    assert (root / first).read_text() == ORIGINAL[first]


def test_bundle_reverts_completed_writes_on_io_failure(tmp_path, monkeypatch):
    root = tree(tmp_path)
    replace = PATCHER._replace
    calls = 0

    def failing(path, data):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected storage failure")
        return replace(path, data)

    monkeypatch.setattr(PATCHER, "_replace", failing)
    with pytest.raises(OSError, match="injected"):
        PATCHER.install(root)
    assert all((root / p).read_text() == text for p, text in ORIGINAL.items())


def test_bundle_rejects_symlink_and_tampered_payload(tmp_path):
    root = tree(tmp_path / "source")
    item = MANIFEST["components"]["python"]["files"][0]
    target = root / item["path"]
    original = target.read_bytes()
    target.unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(original)
    target.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        PATCHER.install(root)
    assert outside.read_bytes() == original


@pytest.mark.parametrize("parallel", [True, False])
def test_draft_depth_compilation_identity(parallel):
    results = []
    for sources in (ORIGINAL, CORRECTED):
        ns = execute(
            [
                extract(sources["vllm/utils/hashing.py"], "safe_hash"),
                extract(sources["vllm/config/speculative.py"], "compute_hash"),
            ],
            {"hashlib": hashlib, "UnsupportedDigestmodError": UnsupportedDigestmodError},
        )
        draft = Namespace(
            compute_hash=lambda: "unchanged-model",
            hf_config=Namespace(eagle_aux_hidden_state_layer_ids=[1, 23, 45]),
        )
        hashes = {
            ns["compute_hash"](
                Namespace(
                    method="dflash",
                    parallel_drafting=parallel,
                    num_speculative_tokens=k,
                    draft_model_config=draft,
                )
            )
            for k in (1, 7, 15)
        }
        results.append(len(hashes))
    assert results == ([1, 3] if parallel else [1, 1])


@pytest.mark.parametrize("drafts", [None, [0, 7, 0, 0]])
def test_zero_draft_decode_keeps_previous_state_offset_and_prefill_does_not(drafts):
    path = "vllm/v1/worker/gpu/model_states/mamba_hybrid.py"
    f = execute([extract(CORRECTED[path], "compute_num_decode_draft_tokens")], {"np": np})[
        "compute_num_decode_draft_tokens"
    ]
    scheduled = np.array([1, 8 if drafts else 1, 1, 2])
    actual = f(
        6,
        scheduled,
        None if drafts is None else np.array(drafts),
        np.array([False, False, True, True]),
    )
    assert actual.tolist() == [0, 7 if drafts else 0, -1, -1, -1, -1]
    # Execute the old classifier as a negative control, using the same input.
    prepare = extract(ORIGINAL[path], "prepare_attn")
    branch = next(
        n
        for n in ast.walk(prepare)
        if isinstance(n, ast.If) and "not for_capture" in ast.unparse(n.test)
    )
    begin = next(
        i
        for i, n in enumerate(branch.body)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "num_decode_draft_tokens_np"
    )
    end = next(
        i
        for i, n in enumerate(branch.body)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "num_decode_draft_tokens_cpu"
    )
    old = execute(
        branch.body[begin:end],
        {
            "np": np,
            "num_reqs": 4,
            "input_batch": Namespace(
                num_reqs=4,
                num_scheduled_tokens=scheduled,
                num_draft_tokens_per_req=None if drafts is None else np.array(drafts),
                is_prefilling_np=np.array([False, False, True, True]),
            ),
        },
    )
    assert old["num_decode_draft_tokens_np"][0] == -1


@pytest.mark.parametrize(
    "invalid,expected,evicted",
    [
        ({22}, 32, {13, 14, 22}),
        ({12}, 0, {11, 12, 13, 14, 21, 22}),
        ({0}, 64, set()),
    ],
)
def test_hybrid_load_recovery_uses_all_groups_and_common_alignment(invalid, expected, evicted):
    path = "vllm/v1/core/sched/scheduler.py"
    method = "_update_requests_with_invalid_blocks"
    groups = ([11, 12, 13, 14], [21, 22])
    manager = Namespace(
        get_block_ids=lambda request: groups, group_block_sizes=(16, 32), null_block_id=0
    )
    scheduler = Namespace(kv_cache_manager=manager, block_size=32)
    request = Namespace(request_id="synthetic", num_computed_tokens=64)
    fixed = execute([extract(CORRECTED[path], method)])[method]
    affected, count, blocks = fixed(scheduler, [request], invalid, {}, True)
    assert request.num_computed_tokens == expected
    assert count == 64 - expected
    assert blocks == evicted
    assert affected == ({"synthetic"} if invalid != {0} else set())
    old = execute([extract(ORIGINAL[path], method)])[method]
    with pytest.raises(ValueError, match="too many values"):
        old(
            scheduler,
            [Namespace(request_id="synthetic", num_computed_tokens=64)],
            invalid,
            {},
            True,
        )


class Block:
    def __init__(self, block_id):
        self.block_id, self.block_hash, self.block_hash_num_tokens = block_id, None, None
        self.is_null, self.ref_cnt = False, 0

    def set_block_hash(self, value, *, num_tokens=None):
        self.block_hash, self.block_hash_num_tokens = value, num_tokens

    def reset_hash(self):
        self.block_hash, self.block_hash_num_tokens = None, None


def make_pool():
    source = CORRECTED["vllm/v1/core/block_pool.py"]
    ns = execute(
        [
            extract(source, "BlockHashToBlockMap", kind=ast.ClassDef),
            extract(source, "BlockPool", kind=ast.ClassDef),
        ],
        {
            "KVCacheBlock": Block,
            "FreeKVCacheBlockQueue": deque,
            "make_block_hash_with_group_id": lambda h, g: (h, g),
        },
    )
    return ns["BlockPool"](8, True, 16)


def test_publication_waits_for_gpu_completion_across_empty_passes():
    pool = make_pool()
    block = pool.blocks[1]
    pool.begin_publication_step(1)
    pool._insert_block_hash(("prefix", 0), block, 16)
    assert pool.get_cached_block("prefix", [0]) is None
    pool.begin_publication_step(1)  # empty pass is not a GPU completion
    assert pool.get_cached_block("prefix", [0]) is None
    pool.begin_publication_step(2)  # scheduling ahead also is not completion
    assert pool.get_cached_block("prefix", [0]) is None
    pool.commit_publication_step(1)
    assert pool.get_cached_block("prefix", [0]) == [block]
    with pytest.raises(ValueError, match="completion fence"):
        pool.commit_publication_step(3)


def test_abort_removes_only_tentative_pages_and_cannot_evict_recycled_owner():
    pool = make_pool()
    complete, tentative = pool.blocks[1:3]
    pool._insert_block_hash(("complete", 0), complete, 16)
    pool.begin_publication_step(1)
    pool._insert_block_hash(("tentative", 0), tentative, 16)
    pool.rollback_uncommitted([complete, tentative])
    assert pool.get_cached_block("complete", [0]) == [complete]
    assert pool.get_cached_block("tentative", [0]) is None
    assert not pool._pending_publications
    pool.commit_publication_step(1)
    pool._insert_block_hash(("new-owner", 0), tentative, 16)
    pool.rollback_uncommitted([tentative])
    assert pool.get_cached_block("new-owner", [0]) == [tentative]


def test_copy_on_write_destination_stays_hidden_until_copy_completes():
    pool = make_pool()
    source, target = pool.blocks[1:3]
    pool._insert_block_hash(("prefix", 0), source, 16)
    pool.begin_publication_step(1)
    pool.move_block_hashes(source, target)
    assert source.block_hash is None
    assert pool.get_cached_block("prefix", [0]) is None
    pool.commit_publication_step(1)
    assert pool.get_cached_block("prefix", [0]) == [target]


def test_mamba_binding_precedes_first_resumed_request():
    path = "vllm/v1/worker/gpu/model_states/mamba_hybrid.py"
    node = extract(CORRECTED[path], "add_request")
    # Execute the indexing expression from the actual function on unequal units.
    expressions = [
        n for n in ast.walk(node) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.FloorDiv)
    ]
    assert len(expressions) == 1
    expression = compile(ast.Expression(expressions[0]), "<resume-index>", "eval")
    for tokens, expected in [(1, 0), (64, 0), (65, 1), (128, 1), (129, 2)]:
        assert (
            eval(
                expression,
                {
                    "self": Namespace(
                        _mamba_spec=Namespace(block_size=64), cache_config=Namespace(block_size=16)
                    ),
                    "new_req_data": Namespace(num_computed_tokens=tokens),
                },
            )
            == expected
        )
    runner = extract(CORRECTED["vllm/v1/worker/gpu/model_runner.py"], "initialize_kv_cache")
    assert "self.model_state.set_kv_cache_config(kv_cache_config)" in ast.unparse(runner)


@pytest.mark.parametrize("new_field", [None, [], ["<new>"]])
def test_legacy_tokenizer_tokens_survive_empty_new_field(new_field):
    path = "transformers/tokenization_utils_base.py"
    constructor = extract(CORRECTED[path], "__init__", owner="PreTrainedTokenizerBase")
    node = next(
        n
        for n in constructor.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "'additional_special_tokens' in kwargs"
    )
    kwargs = {"additional_special_tokens": ["<tool>"], "extra_special_tokens": new_field}
    execute([node], {"kwargs": kwargs})
    assert kwargs == {"extra_special_tokens": new_field or ["<tool>"]}


def test_composes_with_existing_snapshot_and_scheduler_hooks():
    spec = importlib.util.spec_from_file_location(
        "chat_snapshot_backport_test", RELEASE / "patch_chat_snapshot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mamba = module.retain_settled_mamba_tail(
        CORRECTED["vllm/v1/core/single_type_kv_cache_manager.py"]
    )
    runner = module.add_fair_runner_hooks(CORRECTED["vllm/v1/worker/gpu/model_runner.py"])
    compile(mamba, "mamba-with-chat-overlay", "exec")
    compile(runner, "runner-with-chat-overlay", "exec")


def parser_engine(sources, monkeypatch):
    """Load the actual standalone parser engine without loading a GPU runtime."""
    for name in ("vllm", "vllm.parser", "vllm.parser.engine"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    modules = {}
    for name in (
        "events",
        "incremental_lexer",
        "token_id_scanner",
        "parser_engine_config",
        "streaming_parser_engine",
    ):
        fullname = "vllm.parser.engine." + name
        module = types.ModuleType(fullname)
        monkeypatch.setitem(sys.modules, fullname, module)
        exec(compile(sources[f"vllm/parser/engine/{name}.py"], fullname, "exec"), module.__dict__)
        modules[name] = module
    cfg = modules["parser_engine_config"]
    ns = {
        "functools": functools,
        "json": json,
        "re": regex,
        "EventType": modules["events"].EventType,
        "ParserEngineConfig": cfg.ParserEngineConfig,
        "ParserState": cfg.ParserState,
        "Transition": cfg.Transition,
    }
    nodes = [
        n
        for n in ast.parse(sources["vllm/parser/qwen3.py"]).body
        if isinstance(n, (ast.Assign, ast.FunctionDef))
    ]
    execute(nodes, ns)
    return modules["streaming_parser_engine"].StreamingParserEngine(ns["qwen3_config"](), None)


@pytest.mark.parametrize("width", [1, 2, 7, 4096])
@pytest.mark.parametrize(
    "text",
    ["A literal <tool_call> in a sentence", "A literal <tool_call>", "A literal <tool_call>\n  "],
)
def test_literal_tool_opener_is_not_silently_swallowed_at_any_fragmentation(
    monkeypatch, width, text
):
    engine = parser_engine(CORRECTED, monkeypatch)
    events = []
    for start in range(0, len(text), width):
        events.extend(engine.feed(text[start : start + width], []))
    events.extend(engine.finish())
    assert "".join(e.value for e in events if e.type.name == "REASONING_CHUNK") == text
    assert not any(e.type.name == "TOOL_CALL_START" for e in events)


@pytest.mark.parametrize("width", [1, 2, 7, 4096])
def test_real_tool_boundary_survives_reasoning_close_in_same_delta(monkeypatch, width):
    engine = parser_engine(CORRECTED, monkeypatch)
    text = (
        "consider</think><tool_call><function=bash>"
        "<parameter=command>printf ok</parameter></function></tool_call>"
    )
    events = []
    for start in range(0, len(text), width):
        events.extend(engine.feed(text[start : start + width], []))
    events.extend(engine.finish())
    assert "".join(e.value for e in events if e.type.name == "REASONING_CHUNK") == "consider"
    assert "".join(e.value for e in events if e.type.name == "TOOL_NAME") == "bash"
    args = "".join(e.value for e in events if e.type.name == "ARG_VALUE_CHUNK")
    assert args == "<parameter=command>printf ok</parameter>"
    assert sum(e.type.name == "TOOL_CALL_START" for e in events) == 1
    assert sum(e.type.name == "TOOL_CALL_END" for e in events) == 1


def test_delegating_parser_hands_off_complete_post_reasoning_text():
    path = "vllm/parser/abstract_parser.py"

    class Delta:
        def __init__(self, reasoning=None, content=None, tool_calls=None):
            self.reasoning, self.content, self.tool_calls = reasoning, content, tool_calls

    handoffs = []
    for sources in (ORIGINAL, CORRECTED):
        ns = execute(
            [
                extract(sources[path], "StreamState", kind=ast.ClassDef),
                extract(sources[path], "parse_delta", owner="DelegatingParser"),
            ],
            {"dataclass": dataclasses.dataclass, "field": dataclasses.field, "DeltaMessage": Delta},
        )
        state = ns["StreamState"](engine_based=True)
        seen = []

        def tool(_seen=seen, **kwargs):
            _seen.append(kwargs)
            return Delta(tool_calls=[Namespace(id=None)]), True

        parser = Namespace(
            _stream_state=state,
            _engine_based=True,
            _tool_parser=object(),
            _initialize_history_tool_call_cnt=lambda request: None,
            _in_reasoning_phase=lambda st: not st.reasoning_ended,
            _in_tool_call_phase=lambda st: st.reasoning_ended,
            _reasoning_parser=Namespace(
                engine_based_streaming=True,
                has_engine_confirmed_reasoning_end=lambda: True,
                finish_streaming=lambda: Delta(content="<"),
            ),
            extract_reasoning_streaming=lambda **kwargs: Delta(reasoning="consider", content=""),
            extract_content_ids=lambda ids: [2, 3],
            _extract_tool_calls_streaming=tool,
        )
        tail = "<tool_call><function=bash><parameter=command>ok</parameter></function></tool_call>"
        result = ns["parse_delta"](
            parser,
            "consider</think>" + tail,
            [1, 2, 3],
            Namespace(include_reasoning=True),
            finished=False,
        )
        assert result.reasoning == "consider"
        handoffs.append(seen[0]["delta_text"])
    assert handoffs[0] != tail  # preserved baseline failure
    assert handoffs[1] == tail


@pytest.mark.parametrize("component", ["triton", "rocr", "xgrammar"])
def test_native_source_bundle_applies_exactly_and_idempotently(tmp_path, component):
    for name, source in NATIVE[component].items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    result = PATCHER.install(tmp_path, component)
    assert all(row["status"] == "applied" for row in result["files"])
    assert result["native_qualification"] == "NOT_RUN"
    assert all(
        row["status"] == "already_applied" for row in PATCHER.install(tmp_path, component)["files"]
    )


def test_snapshot_namespace_binds_native_binaries_without_binding_evidence_labels(tmp_path):
    parent = "a" * 64
    native = {"compiler": {"libtriton.so": "b" * 64}}
    first = PATCHER.candidate_data_abi(parent, native_bindings=native)
    assert first != parent
    assert first != PATCHER.candidate_data_abi(
        parent, native_bindings={"compiler": {"libtriton.so": "c" * 64}}
    )
    manifest = json.loads(json.dumps(MANIFEST))
    manifest["components"]["python"]["gpu_qualification"] = "TESTED"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert first == PATCHER.candidate_data_abi(parent, bundle=tmp_path, native_bindings=native)


def test_malformed_function_tool_is_not_accepted_as_a_builtin(monkeypatch):
    from pydantic import TypeAdapter, ValidationError

    name = "python/xgrammar/openai_tool_call_schema.py"
    original = NATIVE["xgrammar"][name]
    entry = next(e for e in MANIFEST["components"]["xgrammar"]["files"] if e["path"] == name)
    corrected = PATCHER.apply_exact_patch(original, (PATCHER.BUNDLE / entry["patch"]).read_text())
    for source, broken in [(original, True), (corrected, False)]:
        module = types.ModuleType("qwen_test_xgrammar_schema")
        monkeypatch.setitem(sys.modules, module.__name__, module)
        exec(compile(source, name, "exec"), module.__dict__)
        adapter = TypeAdapter(module.ToolParam)
        if broken:
            assert adapter.validate_python({"type": "function"}).type == "function"
        else:
            with pytest.raises(ValidationError):
                adapter.validate_python({"type": "function"})
        assert adapter.validate_python({"type": "web_search"}).type == "web_search"
        assert (
            adapter.validate_python(
                {"type": "function", "function": {"name": "bash", "parameters": {}}}
            ).type
            == "function"
        )


@pytest.mark.parametrize("use_numpy", [True, False])
@pytest.mark.parametrize("counts,expected", [([0], 1), ([-1], 0), ([-1, 0, 7], 2)])
def test_zero_draft_gdn_metadata_keeps_speculative_state_rows(use_numpy, counts, expected):
    path = "vllm/v1/attention/backends/gdn_attn.py"
    observed = []
    for sources in (ORIGINAL, CORRECTED):
        fn = extract(sources[path], "build")
        branch = next(
            n
            for n in fn.body
            if isinstance(n, ast.If) and "not self.use_spec_decode" in ast.unparse(n.test)
        )
        values = np.array(counts)
        ns = execute(
            [branch],
            {
                "self": Namespace(use_spec_decode=True),
                "num_decode_draft_tokens_cpu": values,
                "_r_np": use_numpy,
                "_ndt_np": values,
                "_mask_np": values >= 0,
                "torch": Namespace(from_numpy=lambda a: a),
                "query_start_loc": Namespace(device="synthetic"),
                "async_tensor_h2d": lambda a, device: a,
            },
        )
        observed.append(ns["num_spec_decodes"])
    assert observed[1] == expected
    if counts == [0]:
        assert observed[0] == 0


def test_mamba_zeroer_uses_both_state_tensors_and_preserves_block_strides():
    path = "vllm/v1/worker/utils.py"

    class Tensor:
        def __init__(self, address, shape, strides):
            self.address, self.shape, self.strides, self.ndim = address, shape, strides, len(shape)

        def data_ptr(self):
            return self.address

        def element_size(self):
            return 4

        def stride(self, dimension=None):
            return self.strides if dimension is None else self.strides[dimension]

    class Attention:
        pass

    class Mamba:
        pass

    tensors = (Tensor(1024, (4, 2, 3), (6, 3, 1)), Tensor(2048, (4, 2, 2), (4, 2, 1)))
    results = []
    for sources in (ORIGINAL, CORRECTED):
        ns = execute(
            [extract(sources[path], "KVBlockZeroer", kind=ast.ClassDef)],
            {
                "torch": Namespace(
                    Tensor=Tensor, tensor=lambda value, **kwargs: value, uint64="u64", int64="i64"
                ),
                "AttentionSpec": Attention,
                "MambaSpec": Mamba,
                "iprod": itertools.product,
            },
        )
        group = Namespace(kv_cache_spec=Mamba(), kv_cache_group_id=0, layer_names=["state"])
        zeroer = ns["KVBlockZeroer"](
            "synthetic", [group], [64], "auto", {"state": Namespace(kv_cache=tensors)}
        )
        results.append(zeroer._meta)
    assert results[0] is None
    addresses, strides, spans, *_ = results[1]
    assert addresses == [1024, 2048]
    assert strides == [6, 4]
    assert spans == [6, 4]


def test_native_probe_gate_rejects_skips_and_missing_cases(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(RELEASE))
    spec = importlib.util.spec_from_file_location(
        "test_upstream_probe", RELEASE / "probe_upstream_correctness.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "tests.xml"
    path.write_text("<testsuites><testsuite><testcase/><testcase/></testsuite></testsuites>")
    assert module.junit_status(path, 2)["status"] == "TESTED"
    assert module.junit_status(path, 3)["status"] == "FAILED"
    path.write_text(
        "<testsuites><testsuite><testcase/><testcase><skipped/></testcase></testsuite></testsuites>"
    )
    assert module.junit_status(path, 2)["status"] == "FAILED"
    with pytest.raises(ValueError, match="compiler and sources"):
        module.verify_probe_binding({"files": {"/unrelated.so": "0" * 64}}, "triton_guarded_loop")
    monkeypatch.setattr(module, "sysconfig", Namespace(get_path=lambda _: str(tmp_path)))
    compiler = tmp_path / "triton/_C/libtriton.so"
    compiler.parent.mkdir(parents=True)
    compiler.write_bytes(b"synthetic compiler")
    bound = {"files": {str(compiler): PATCHER.sha256(compiler.read_bytes())}}
    module.verify_probe_binding(bound, "triton_guarded_loop")
    with pytest.raises(ValueError, match="compiler and sources"):
        module.verify_probe_binding(bound, "short_prefill_convolution")


def test_native_binding_rejects_wrong_library_and_changed_binary(tmp_path):
    source = (RELEASE / "bootstrap_radiance_upstream_candidate.py").read_text()
    paths = {name: str(tmp_path / f"{name}.so") for name in ("triton", "rocr", "xgrammar")}
    for path in paths.values():
        Path(path).write_bytes(b"synthetic native artifact")
    components = {
        name: {path: PATCHER.sha256(Path(path).read_bytes())} for name, path in paths.items()
    }
    binding = {
        "manifest_sha256": PATCHER.sha256((PATCHER.BUNDLE / "manifest.json").read_bytes()),
        "components": components,
    }
    receipt = tmp_path / "native.json"
    receipt.write_text(json.dumps(binding))
    check = execute(
        [extract(source, "verify_native_bindings")],
        {
            "json": json,
            "hashlib": hashlib,
            "Path": Path,
            "BUNDLE": PATCHER.BUNDLE,
            "sha256": PATCHER.sha256,
            "NATIVE_PATHS": paths,
        },
    )["verify_native_bindings"]
    assert check(receipt) == components
    moved = dict(components["triton"])
    components["triton"] = {str(tmp_path / "unused.so"): next(iter(moved.values()))}
    receipt.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="installed runtime library"):
        check(receipt)
    components["triton"] = moved
    receipt.write_text(json.dumps(binding))
    Path(paths["triton"]).write_bytes(b"modified after build")
    with pytest.raises(ValueError, match="artifact differs"):
        check(receipt)
    binding["manifest_sha256"] = "0" * 64
    receipt.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="different backport bundle"):
        check(receipt)


def test_candidate_isolates_compiler_caches_without_changing_parent_paths(tmp_path):
    source = (RELEASE / "bootstrap_radiance_upstream_candidate.py").read_text()
    isolate = execute([extract(source, "isolate_compile_caches")], {"Path": Path})[
        "isolate_compile_caches"
    ]
    env = {"TRITON_CACHE_DIR": str(tmp_path), "UNRELATED": "preserved"}
    isolate(env, "a" * 64)
    assert env["TRITON_CACHE_DIR"] == str(tmp_path / ("upstream-" + "a" * 64))
    assert env["UNRELATED"] == "preserved"
    assert "upstream-" in env["TORCHINDUCTOR_CACHE_DIR"]
    assert "upstream-" in env["VLLM_CACHE_ROOT"]
    with pytest.raises(ValueError, match="absolute path"):
        isolate({"TRITON_CACHE_DIR": "relative"}, "b" * 64)


def test_convolution_destination_backport_preserves_prior_precision_repair():
    path = "vllm/model_executor/layers/mamba/ops/causal_conv1d.py"
    for kernel in ("_causal_conv1d_fwd_kernel", "_causal_conv1d_update_kernel"):
        before, after = [extract(sources[path], kernel) for sources in (ORIGINAL, CORRECTED)]
        products = [
            next(
                n.value
                for n in ast.walk(body)
                if isinstance(n, ast.AugAssign)
                and ast.unparse(n.target) == "acc"
                and "matrix_x" in ast.unparse(n.value)
            )
            for body in (before, after)
        ]
        # The two operands must be promoted before multiplication; casting the
        # already-rounded product afterwards would reproduce the original bug.
        assert ast.unparse(products[0]) == "matrix_x * matrix_w"
        assert ast.unparse(products[1]) == "matrix_x.to(tl.float32) * matrix_w.to(tl.float32)"


def test_full_existing_chat_overlay_composes_with_candidate(tmp_path, monkeypatch):
    package = tree(tmp_path)
    scheduler = package / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
    monkeypatch.setenv("QWEN_SNAPSHOT_SCHEDULER", str(scheduler))
    monkeypatch.syspath_prepend(str(RELEASE))
    bootstrap = (RELEASE / "bootstrap_radiance_upstream_candidate.py").read_text()
    compose = execute(
        [extract(bootstrap, "install_before_streaming")],
        {
            "Path": Path,
            "__file__": str(RELEASE / "bootstrap_radiance_upstream_candidate.py"),
            "json": json,
            "install": PATCHER.install,
        },
    )["install_before_streaming"]
    composed = compose(package, runpy.run_path)
    with pytest.raises(ValueError, match="unexpected release"):
        composed(RELEASE / "unexpected_script.py")
    composed(str(RELEASE / "patch_streaming_snapshot.py"))
    assert (package / "qwen_upstream_backports_receipt.json").is_file()
    spec = importlib.util.spec_from_file_location(
        "candidate_chat_overlay", RELEASE / "patch_chat_snapshot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sources = module.transformed_sources(
        package,
        ROOT / "src/qwen_r9700_lab/radiance_cache.py",
        RELEASE / "radiance_chat_tier.py",
        RELEASE / "radiance_fair_scheduler.py",
    )
    for path, text in sources.items():
        compile(text, str(path), "exec")
    assert len(sources) == 12
    assert "rollback_uncommitted" in (package / "vllm/v1/core/sched/scheduler.py").read_text()
