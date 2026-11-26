import asyncio
import json
from pathlib import Path
from types import SimpleNamespace


def test_installed_prefill_timing_preserves_dispatch(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments/radiance-public"))
    import benchmark_prefill_speed as bench

    calls = []

    class Suite:
        def __init__(self, cfg, client):
            pass

        async def rpc(self, name):
            assert name == "qwen_prefill_runtime_info"
            calls.append(name)
            return {"candidate": "existing operator"}

        def chat(self, name, generation):
            return {"id": name, "generation": generation}

        def stream(self, label, prompt, count, chat):
            async def finish():
                return {
                    "sha256": "synthetic-answer",
                    "usage": {"prompt_tokens_details": {"cached_tokens": 0}},
                }

            return SimpleNamespace(
                task=asyncio.create_task(finish()),
                result=finish,
                chat=chat,
                first=2,
                started=1,
            )

        async def absent(self, chat):
            pass

    monkeypatch.setattr(bench, "Suite", Suite)
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"synthetic": True, "prefix": [1, 2, 3]}))
    output = tmp_path / "result.json"
    args = SimpleNamespace(
        fixture=fixture,
        output=output,
        installed=True,
        candidate_only=False,
        legacy=False,
        sources=None,
        baseline_attention=None,
        baseline_projection=None,
        attention=None,
        projection=None,
        base_url="http://127.0.0.1:1",
        abi="synthetic",
        repeats=2,
        contexts=[256],
    )
    asyncio.run(bench.run(args))
    report = json.loads(output.read_text())
    assert calls == ["qwen_prefill_runtime_info"]
    assert report["status"] == "MEASURED" and report["as_installed"]
    assert [row["installed"] for row in report["cases"]] == [{"candidate": "existing operator"}] * 2


def test_installed_prefill_comparison_keeps_hooks_and_rejects_warm_control(tmp_path, monkeypatch):
    import pytest

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments/radiance-public"))
    import diagnose_prefill_decode as bench

    calls = []

    class Suite:
        def __init__(self, cfg, client):
            pass

        async def rpc(self, name, args=None):
            calls.append(name)
            return {}

        def stream(self, label, prompt, count):
            assert label == "full-prefill"

            async def finish():
                return {"usage": {"prompt_tokens_details": {"cached_tokens": 1}}}

            return SimpleNamespace(task=asyncio.create_task(finish()), result=finish)

    monkeypatch.setattr(bench, "Suite", Suite)
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"prompt": [1, 2]}))
    (tmp_path / "synthetic-corpus.json").write_text(
        json.dumps({"synthetic": True, "prefix": [1, 2], "output": [3, 4]})
    )
    args = SimpleNamespace(
        output=tmp_path,
        fixture=fixture,
        abi="synthetic",
        base_url="http://127.0.0.1:1",
        rows=1,
        prefix_rows=2,
        installed=True,
        stages=False,
        gdn=False,
        attention=False,
        first=None,
    )
    with pytest.raises(ValueError, match="unexpectedly reused a prefix"):
        asyncio.run(bench.run(args))
    assert calls == ["qwen_prefill_capture_begin", "qwen_prefill_capture_finish"]


def test_prefill_hidden_comparison_detects_first_signed_zero(tmp_path, monkeypatch):
    import pytest

    torch = pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments/radiance-public"))
    from qualify_combined_continuity import compare_hidden_bits

    for name, value in (("left", 0.0), ("right", -0.0)):
        root = tmp_path / name
        root.mkdir()
        (root / "capture.json").write_text(json.dumps({"batches": [{"file": "rows.pt"}]}))
        torch.save(
            {
                "positions": [6, 7],
                "hidden": torch.tensor([[0.0], [value]], dtype=torch.bfloat16),
            },
            root / "rows.pt",
        )
    result = compare_hidden_bits(tmp_path / "left", tmp_path / "right", 6, 2)
    assert result["hidden_bit_exact_rows"] == 1
    assert result["first_hidden_bit_mismatch"]["position"] == 7
    assert (
        result["first_hidden_bit_mismatch"]["left_sha256"]
        != result["first_hidden_bit_mismatch"]["right_sha256"]
    )


def test_changed_sealed_reference_stays_failed_after_matching_current_replay(tmp_path, monkeypatch):
    import pytest

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments/radiance-public"))
    import qualify_combined_continuity as bench

    archive = tmp_path / "archive"
    archive.mkdir()
    output = [4, 5, 6]
    (archive / "synthetic-corpus.json").write_text(
        json.dumps({"synthetic": True, "prefix": [1, 2, 3], "output": output})
    )
    streams = []

    class Suite:
        def __init__(self, cfg, client):
            pass

        async def rpc(self, name, args=None):
            if name == "qwen_prefill_compare":
                return {
                    "positions": 2,
                    "hidden_exact_rows": 2,
                    "logits": {"full_logits_exact": 2},
                    "first_position": 3,
                    "last_position": 4,
                }
            if name == "qwen_prefill_capture_begin" and len(args) == 3:
                assert args[2] == str(archive / "synthetic-corpus.json")
            return {"forced_complete": True}

        def stream(self, label, prompt, count):
            streams.append((label, prompt, count))

            async def finish():
                return {"usage": {"prompt_tokens_details": {"cached_tokens": 0}}}

            return SimpleNamespace(
                task=asyncio.create_task(finish()), result=finish, ids=output, chat={}
            )

        async def absent(self, chat):
            pass

    monkeypatch.setattr(bench, "Suite", Suite)
    monkeypatch.setattr(
        bench,
        "compare_hidden_bits",
        lambda left, right, first, count: {
            "hidden_bit_exact_rows": 1 if left.parent == archive else 2
        },
    )
    cfg = SimpleNamespace(
        abi="test", references=[archive], root=tmp_path, replay_changed_references=True
    )
    with pytest.raises(RuntimeError, match="differs from archived M8"):
        asyncio.run(bench.sealed(cfg, None))
    assert streams == [("new-3", [1, 2, 3, 4, 5], 1), ("forced-3", [1, 2, 3], 3)]
    (case,) = json.loads((tmp_path / "sealed-results.json").read_text())
    assert not case["pass"]
    assert case["current_prefill_vs_current_m8"]["pass"]
    assert not case["current_m8_vs_archived_m8"]["pass"]


def test_forced_snapshot_encoding_preserves_identity_and_fails_closed(tmp_path, monkeypatch):
    import pytest

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments/radiance-public"))
    import qualify_combined_continuity as bench

    cache = bench.radiance_cache
    monkeypatch.setattr(bench, "SNAPSHOT_ROOT", tmp_path)
    run = "direct-disk-831"
    abi = "c" * 64
    root = tmp_path / abi / "data"
    root.mkdir(parents=True)
    payload = b"synthetic0123456" * 128
    key = "g0-" + "f" * 32 + ".qkv"
    for mode in ("raw", "compressed", "damage"):
        label = f"durable-{mode}-128"
        owner = {
            "id": bench.sha([run, label]),
            "generation": "a" * 64,
            "title": "Synthetic lifecycle " + label,
            "cwd": "/qualification",
            "session_file": "",
        }
        store = cache.ChatStore(root, owner)
        store.activate()
        assert store.write(key, memoryview(payload))
        assert store.publish([key], 128, len(payload))
        before = store.path(key).read_bytes()
        with pytest.raises(ValueError, match="exact preparation"):
            bench.prepare_checkpoint_encoding(owner, abi, run + "wrong", mode)
        assert store.path(key).read_bytes() == before
        record = bench.prepare_checkpoint_encoding(owner, abi, run, mode)
        assert record["mode"] == mode
        encoded = store.path(key).read_bytes()
        if mode == "damage":
            with pytest.raises(cache.SnapshotIntegrityError):
                cache.decode_block(encoded, len(payload))
        else:
            assert cache.HEADER.unpack_from(encoded)[0] == (
                cache.RAW if mode == "raw" else cache.COMPRESSED
            )
            assert cache.decode_block(encoded, len(payload)) == payload
            assert store.metadata()["verified_head"][key] == store._fingerprint(store.path(key))


def test_sampled_replay_omits_matching_tokens_after_rejected_speculation(tmp_path, monkeypatch):
    import torch

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments/radiance-public"))
    from qualify_sampled_cache import select_accepted_rows

    source = tmp_path / "natural"
    source.mkdir()
    packets = [
        ([2], [99], [100]),
        ([3, 4, 5, 6], [10, 11, 666, 13], [101, 102, 998, 999]),
        ([5, 6], [12, 13], [103, 104]),
    ]
    for i, (positions, inputs, hidden) in enumerate(packets):
        torch.save(
            {
                "positions": positions,
                "input_ids": inputs,
                "hidden": torch.tensor(hidden),
            },
            source / f"{i}.pt",
        )
    (source / "capture.json").write_text(
        json.dumps({"batches": [{"file": f"{i}.pt"} for i in range(3)]})
    )
    dst = tmp_path / "accepted"
    select_accepted_rows(source, dst, [97, 98, 99], [10, 11, 12, 13], 4)
    result = torch.load(dst / "selected.pt", weights_only=True)
    assert result["input_ids"] == [99, 10, 11, 12, 13]
    assert result["hidden"].tolist() == [100, 101, 102, 103, 104]


def test_sampled_replay_logical_cache_hash_is_independent_of_page_permutation(
    monkeypatch,
):
    import torch

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments/radiance-public"))
    from sampled_cache_probe import logical_draft_cache

    block_size = 16
    backing = torch.arange(7 * 2 * block_size * 2, dtype=torch.int32).view(7, 2, block_size, 1, 2)

    def describe(kv, table):
        layer = SimpleNamespace(self_attn=SimpleNamespace(attn=SimpleNamespace(kv_cache=kv)))
        tables = SimpleNamespace(
            input_block_tables=[torch.tensor([table])], kernel_block_sizes=[block_size]
        )
        spec = SimpleNamespace(
            model=SimpleNamespace(model=SimpleNamespace(layers=[layer])),
            _layer_group_idx=[0],
            draft_kv_cache_group_ids=[0],
            block_tables=tables,
        )
        return logical_draft_cache(spec, 5, 52)

    baseline = describe(backing, [1, 2, 4, 3])
    permuted = backing[[0, 2, 1, 3, 4, 6, 5]]
    assert baseline == describe(permuted.transpose(0, 1), [2, 1, 4, 3])
    permuted[1, 0, 8, 0, 0] += 1
    changed = describe(permuted, [2, 1, 4, 3])
    assert baseline[0]["chunks_of_32"][0] != changed[0]["chunks_of_32"][0]


def test_sampled_replay_reads_actual_triton_packed_kv_at_logical_positions(monkeypatch):
    import torch

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments/radiance-public"))
    from sampled_cache_probe import logical_draft_cache, tensor_digest

    # NHD physical strides, BHND logical tensor; pages 3 and 1 back the window.
    kv = torch.arange(4 * 16 * 3 * 4, dtype=torch.int32).view(4, 16, 3, 4).transpose(1, 2)
    layer = SimpleNamespace(self_attn=SimpleNamespace(attn=SimpleNamespace(kv_cache=kv)))
    tables = SimpleNamespace(input_block_tables=[torch.tensor([[3, 1]])], kernel_block_sizes=[16])
    spec = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(layers=[layer])),
        _layer_group_idx=None,
        draft_kv_cache_group_ids=[0],
        block_tables=tables,
    )
    expected = torch.stack([kv[3 if p < 16 else 1, :, p % 16].reshape(-1) for p in range(14, 19)])
    assert logical_draft_cache(spec, 14, 19)[0]["chunks_of_32"] == [tensor_digest(expected)]


def test_symbolic_graph_obeys_gdn_actual_runtime_rows(tmp_path, monkeypatch):
    import pytest
    import torch

    pytest.importorskip("triton.experimental.gluon")
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments/radiance-public"))
    from gdn_prefill_direct import _project_for_runtime_rows

    calls = {"projection_direct": 0, "direct_rows": {}, "fallback_rows": {}}
    weight = torch.tensor([1.0])
    originals = {weight.data_ptr(): lambda x, z: x + z}

    @torch.library.custom_op("coherence_test::symbolic_projection_dispatch", mutates_args=())
    def project(x: torch.Tensor, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return _project_for_runtime_rows(x, z, w, originals, calls, lambda: x - z)

    @project.register_fake
    def fake(x, z, w):
        return x.new_empty(x.shape)

    # Trace a prefill first, then execute decode and both sides of the cutoff
    # through the same dynamically shaped compiled entry point.
    compiled = torch.compile(
        lambda x, z: project(x, z, weight),
        backend="eager",
        dynamic=True,
        fullgraph=True,
    )
    for m in (128, 8, 63, 64, 1, 1648):
        x, z = torch.full((m, 4), 3.0), torch.ones(m, 4)
        assert torch.equal(compiled(x, z), x - z if m >= 64 else x + z)
    assert calls["fallback_rows"] == {1: 1, 8: 1, 63: 1}
    assert calls["direct_rows"] == {64: 1, 128: 1, 1648: 1}


def test_wide_norm_build_changes_only_native_admission(tmp_path, monkeypatch):
    import hashlib

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments/radiance-public"))
    import build_stock_fp8_epilogue as builder
    from wide_input_norm import WIDE_SOURCE

    def compile_stub(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"compiled-test-output")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(builder.subprocess, "run", compile_stub)
    for rows in (2048, 4096):
        out = tmp_path / str(rows)
        assert builder.build(out, rows) == 0
    default = (tmp_path / "2048/stock_fp8_epilogue.hip").read_bytes()
    wide = (tmp_path / "4096/stock_fp8_epilogue.hip").read_bytes()
    assert wide == default.replace(b"rows > 2048", b"rows > 4096")
    assert hashlib.sha256(wide).hexdigest() == WIDE_SOURCE


def test_wide_input_norm_routes_real_widths_and_restores_exact_callable(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments/radiance-public"))
    import wide_input_norm as repair

    def original(*args):
        return "baseline"

    op = SimpleNamespace(_init_fn=original, _backend_fns={None: original})
    narrow = SimpleNamespace(manifest={"binary_sha256": "old"})
    wide = SimpleNamespace(manifest={"binary_sha256": "new"}, norm=lambda *args: "wide")
    monkeypatch.setattr(repair, "operators", lambda build: (op, narrow, wide))
    worker = SimpleNamespace()
    repair.install(worker, "frozen")
    for rows, expected in (
        (8, "baseline"),
        (880, "baseline"),
        (1648, "baseline"),
        (2048, "baseline"),
        (2049, "wide"),
        (2528, "wide"),
        (4096, "wide"),
        (4097, "baseline"),
    ):
        assert (
            op._backend_fns[None](
                SimpleNamespace(shape=(rows, 5120), device=SimpleNamespace(type="cuda")),
                None,
                None,
                1e-6,
                "layer",
            )
            == expected
        )
    assert repair.remove(worker)["wide_rows"] == {2049: 1, 2528: 1, 4096: 1}
    assert op._backend_fns[None] is original


def test_boundary_gates_require_full_graph_features_and_all_target_kv(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments/radiance-public"))
    from qualify_numerical_gates import FEATURES, KV_LAYERS, full_graph, target_kv

    graph = {
        "mode": "native", "first": None, "total_positions": 65,
        "rows": [{"position": p} for p in range(255, 320)],
        "exact_positions": dict.fromkeys(FEATURES, 65),
        "actual_decode_batches": [{"route": "full"}],
        "target_kv": {key: {"positions": 320, "different": 0, "first": None} for key in KV_LAYERS},
    }
    assert full_graph(graph, 256, "native") and target_kv(graph, 256, "native")
    assert not full_graph(graph, 256, "forced")
    assert not full_graph(dict(graph, actual_decode_batches=[{"route": "eager"}]), 256, "native")
    assert not full_graph(dict(graph, exact_positions={}), 256, "native")
    assert not full_graph(dict(graph, rows=graph["rows"][1:]), 256, "native")
    assert not target_kv(dict(graph, target_kv={"3": graph["target_kv"]["3"]}), 256, "native")
    graph["target_kv"]["63"]["positions"] -= 1
    assert not target_kv(graph, 256, "native")


def test_boundary_gates_reject_partial_sampled_replay(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments/radiance-public"))
    from qualify_numerical_gates import replay_matches

    observed = {"tokens": 384, "sha256": "a" * 64}
    pair = {"first_differing_output_index": None, "observed": observed, "control": dict(observed)}
    assert replay_matches(pair, 384)
    assert not replay_matches(pair, 512)
    observed["sha256"] = "b" * 64
    assert not replay_matches(pair, 384)
