"""FULL graph post-forward observer must retain only the rows actually committed."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def observe(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "experiments/radiance-public")
    )
    import prefill_graph_observer

    monkeypatch.setattr(prefill_graph_observer, "CAPTURE_ROOT", tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils",
        SimpleNamespace(get_eagle3_aux_layers_from_config=lambda _: (1, 4)),
    )
    return prefill_graph_observer.GraphOutputObserver


def test_acceptance_and_real_graph_metadata(tmp_path, observe):
    import torch
    from diagnose_prefill_graph import compare

    desc = SimpleNamespace(
        num_tokens=8, num_reqs=1, uniform_token_count=8, max_query_len=8
    )
    manager = SimpleNamespace(run_fullgraph=lambda _: None, run_pw_graph=lambda: None)
    position = torch.arange(256, 264)
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(speculative_config=None),
        input_buffers=SimpleNamespace(input_ids=position, positions=position),
        cudagraph_manager=manager,
    )

    def execute(prefill=False, change=False):
        if prefill:
            manager.run_pw_graph()
        else:
            manager.run_fullgraph(desc)
        n = 3 if prefill else 8
        batch = SimpleNamespace(
            num_reqs=1,
            num_tokens=n,
            positions=position[:n],
            input_ids=position[:n],
            is_prefilling_np=[prefill],
            num_draft_tokens=0 if prefill else 7,
        )
        output = torch.arange(n * 4, dtype=torch.bfloat16).reshape(n, 4)
        final = output.clone()
        if change:
            final[2, 1] += 1
        runner.execute_model_state = SimpleNamespace(
            input_batch=batch, aux_hidden_states=[output, output], hidden_states=final
        )

    def sample():
        runner.postprocess_sampled(
            torch.tensor([0]), None, torch.tensor([3]), torch.tensor([5])
        )

    runner.execute_model = execute
    runner.sample_tokens = sample
    runner.postprocess_sampled = lambda *args: None
    for name, mode, change in (
        ("decode-outputs", "native", False),
        ("full-prefill", "prefill", True),
    ):
        observer = observe(runner, tmp_path / name, 256, 259, mode)
        runner.execute_model(mode == "prefill", change)
        runner.sample_tokens()
        receipt = observer.finish()
        assert receipt["positions"] == 3 and runner.execute_model is execute
        if mode == "native":
            row = receipt["batches"][0]
            assert row["route"] == "full" and row["graph_shape"]["num_tokens"] == 8
            assert row["rejected"] == 5 and row["static_differences"] == {
                "input_ids": 0,
                "positions": 0,
            }
    compare(tmp_path)
    import json

    result = json.loads((tmp_path / "graph-output-comparison.json").read_text())
    assert result["first"] == {
        "position": 258,
        "boundary": "final_hidden",
        "differing_elements": 1,
    }
    assert result["exact_positions"] == {
        "aux_layer_1": 3,
        "aux_layer_4": 3,
        "final_hidden": 2,
    }


def test_logical_kv_handles_reordered_physical_pages(observe):
    import hashlib

    import torch
    from prefill_graph_observer import target_kv_rows

    cache = (
        torch.arange(3 * 4 * 16 * 512, dtype=torch.int32)
        .reshape(3, 4, 16, 512)
        .div(513, rounding_mode="floor")
        .to(torch.uint8)
    )
    table = torch.tensor([[2, 0, 1]])
    names = [f"model.layers.{i}.self_attn.attn" for i in range(3, 64, 4)]
    state = SimpleNamespace(
        attn_metadata={name: SimpleNamespace(block_table=table) for name in names}
    )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context={
                    name: SimpleNamespace(kv_cache=cache) for name in names
                }
            )
        )
    )
    rows = target_kv_rows(runner, state, 14, 35)
    expected = [
        hashlib.sha256(
            cache[table[0, p // 16], :, p % 16].contiguous().numpy()
        ).hexdigest()
        for p in range(14, 35)
    ]
    assert len(rows) == 16 and all(v == expected for v in rows.values())
    # A single physical page keeps its strided (head, token, width) view, unlike
    # the materialized multi-page prefill above. Native decode observes this case.
    single = target_kv_rows(runner, state, 16, 20)
    assert all(v == expected[2:6] for v in single.values())



def test_finer_layer_observation_preserves_drafter_output_count(observe, monkeypatch):
    from prefill_graph_observer import install_replay_boundaries

    module = sys.modules["vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils"]
    monkeypatch.setattr(
        module, "get_eagle3_aux_layers_from_config", lambda _: (6, 20, 34, 48, 62)
    )
    applied = []
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(speculative_config=None),
        model=SimpleNamespace(set_aux_hidden_state_layers=applied.append),
    )
    worker = SimpleNamespace(model_runner=runner)
    with pytest.raises(ValueError, match="five unique"):
        install_replay_boundaries(worker, (4, 3, 2, 1))
    receipt = install_replay_boundaries(worker)
    assert len(applied[0]) == len(receipt["original"]) == 5
    assert applied[0] == runner._diagnostic_replay_boundaries
    with pytest.raises(ValueError, match="five unique"):
        install_replay_boundaries(worker)


def test_unsupported_optional_kv_cannot_abort_a_request(observe, tmp_path):
    import torch

    names = ["model.layers.3.self_attn.attn"]
    tiny = torch.arange(4, dtype=torch.int32)
    cfg = SimpleNamespace(
        speculative_config=None,
        compilation_config=SimpleNamespace(
            static_forward_context={names[0]: SimpleNamespace(kv_cache=tiny)}
        ),
    )
    manager = SimpleNamespace(
        run_fullgraph=lambda desc: None, run_pw_graph=lambda: None
    )
    batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=1,
        positions=tiny[1:2],
        input_ids=tiny[1:2],
        is_prefilling_np=[True],
        num_draft_tokens=0,
    )
    state = SimpleNamespace(
        input_batch=batch,
        aux_hidden_states=[tiny[1:2, None]] * 2,
        hidden_states=tiny[1:2, None],
        attn_metadata={names[0]: None},
    )
    runner = SimpleNamespace(
        vllm_config=cfg,
        input_buffers=batch,
        cudagraph_manager=manager,
        execute_model_state=state,
        execute_model=lambda: None,
        postprocess_sampled=lambda *args: 19,
    )
    runner.sample_tokens = lambda: runner.postprocess_sampled(
        tiny, None, torch.tensor([1]), torch.tensor([0])
    )
    root = tmp_path / "obs"
    observer = observe(runner, root, 1, 2, "prefill", True)
    assert runner.sample_tokens() == 19
    record = observer.finish()
    status = record["target_kv_status"]
    assert status["status"] == "unavailable" and names[0] in status["reason"]
    assert status["layouts"][names[0]]["shape"] == [4] and record["positions"] == 1
    assert not (root / "target-kv.json").exists()
