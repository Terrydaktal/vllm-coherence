import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/radiance-public"))
import host_embedding


class Unquantized:
    pass


class Embedding(torch.nn.Embedding):
    quant_method = Unquantized()


class Head(Embedding):
    pass


def make_target():
    target = torch.nn.Module()
    target.config = SimpleNamespace(tie_word_embeddings=False)
    target.model = torch.nn.Module()
    target.model.embed_tokens = Embedding(4, 4)
    target.lm_head = Head(4, 4)
    return target


def test_output_head_cannot_be_moved_even_if_it_uses_a_different_parameter():
    target = make_target()
    input_weight = target.model.embed_tokens.weight
    target.lm_head.weight = torch.nn.Parameter(input_weight.detach())
    with pytest.raises(RuntimeError, match="dedicated input table.*lm_head.weight"):
        host_embedding._input_embedding(target, Embedding, Head, Unquantized)


@pytest.mark.parametrize("corrupt_late_slice", [False, True])
def test_readback_checks_bits_before_rebinding_the_shared_parameter(
    monkeypatch, corrupt_late_slice
):
    monkeypatch.setattr(host_embedding, "CHECK_ROWS", 1)
    data = torch.tensor([[0, 32704], [16256, -32768]], dtype=torch.int16)
    weight = torch.nn.Parameter(data.clone().view(torch.bfloat16), requires_grad=False)
    alias = weight
    original_pointer = weight.data_ptr()
    mapped = data.clone()
    if corrupt_late_slice:
        # These are both zero as floating-point values, with different sign bits.
        mapped[1, 1] = 0

    def to_view(pageable):
        assert pageable.device.type == "cpu"
        assert torch.equal(pageable.view(torch.int16), data)
        return mapped.view(torch.bfloat16)

    if corrupt_late_slice:
        with pytest.raises(RuntimeError, match="readback differs at row 1"):
            host_embedding._replace_storage(weight, to_view)
        assert weight.data_ptr() == original_pointer
    else:
        host_embedding._replace_storage(weight, to_view)
        assert weight.data_ptr() == mapped.data_ptr()
        assert alias is weight
    assert torch.equal(weight.view(torch.int16), data)


def test_uses_only_the_target_loader_and_preserves_drafter_sharing(monkeypatch, capsys):
    target = make_target()
    target_config = SimpleNamespace(enable_sleep_mode=False)
    draft_config = SimpleNamespace()

    class Draft:
        def __init__(self):
            self.model = SimpleNamespace(embed_tokens=target.model.embed_tokens)
            self.lm_head = target.lm_head

    class Loader:
        def load_model(self, vllm_config, model_config, prefix=""):
            return target if model_config is target_config else Draft()

    original_load_model = Loader.load_model
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.base_loader",
        SimpleNamespace(BaseModelLoader=Loader),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.models.qwen3_dflash2",
        SimpleNamespace(DFlash2Qwen3ForCausalLM=Draft),
    )

    def offload(model):
        assert model is target
        return model.model.embed_tokens, 32, 32, 0.1

    monkeypatch.setattr(host_embedding, "_offload_target", offload)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        model_config=target_config,
        speculative_config=SimpleNamespace(),
        lora_config=None,
    )
    speculator = SimpleNamespace()
    worker = SimpleNamespace(
        vllm_config=config, model_runner=SimpleNamespace(speculator=speculator)
    )

    with host_embedding.host_embedding_during_load(worker):
        loader = Loader()
        assert loader.load_model(config, target_config) is target
        speculator.model = loader.load_model(config, draft_config, prefix="drafter")
    assert Loader.load_model is original_load_model
    assert "released 32 GPU tensor bytes" in capsys.readouterr().out

    with (
        pytest.raises(RuntimeError, match="did not observe the target model load"),
        host_embedding.host_embedding_during_load(worker),
    ):
        loader.load_model(config, draft_config)
    assert Loader.load_model is original_load_model

    with (
        pytest.raises(RuntimeError, match="did not share the target"),
        host_embedding.host_embedding_during_load(worker),
    ):
        loader.load_model(config, target_config)
        speculator.model.model.embed_tokens = Embedding(4, 4)
    assert Loader.load_model is original_load_model
