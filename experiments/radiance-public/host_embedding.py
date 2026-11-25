"""GGZ14 host embedding offload for the Coherence TP1 model."""

from contextlib import contextmanager
from functools import wraps
from time import perf_counter

import torch

EMBEDDING_SHAPE = (248320, 5120)
CHECK_ROWS = 4096


def _input_embedding(model, embedding_type, head_type, unquantized_type):
    language_model = (
        model.get_language_model() if hasattr(model, "get_language_model") else model
    )
    embedding = getattr(getattr(language_model, "model", None), "embed_tokens", None)
    if (
        not isinstance(embedding, embedding_type)
        or isinstance(embedding, head_type)
        or not isinstance(getattr(embedding, "quant_method", None), unquantized_type)
    ):
        raise TypeError("host embedding needs the unquantized target input embedding")

    storage = embedding.weight.untyped_storage().data_ptr()
    aliases = [
        name
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if parameter.untyped_storage().data_ptr() == storage
    ]
    if (
        getattr(language_model.config, "tie_word_embeddings", False)
        or len(aliases) != 1
    ):
        raise RuntimeError(
            f"host embedding needs a dedicated input table, found {aliases}"
        )
    return embedding


@torch.no_grad()
def _replace_storage(weight, to_view):
    # vLLM maps an exactly sized allocation for a pageable CPU tensor. The
    # allocator used by pin_memory() would reserve 4 GiB for this 2.37 GiB table.
    view = to_view(weight.to("cpu"))
    if (
        view.device != weight.device
        or view.shape != weight.shape
        or view.dtype != weight.dtype
        or view.stride() != weight.stride()
        or view.data_ptr() == weight.data_ptr()
    ):
        raise RuntimeError("pinned host embedding view does not match the input table")

    checked_at = perf_counter()
    # Compare through the mapped device pointer using the exact BF16 bits.
    # This includes signed zeros and NaNs and bounds the comparison workspace.
    for row in range(0, weight.shape[0], CHECK_ROWS):
        end = row + CHECK_ROWS
        if not torch.equal(
            view[row:end].view(torch.int16), weight[row:end].view(torch.int16)
        ):
            raise RuntimeError(
                f"pinned host embedding readback differs at row {row} or later"
            )

    # Preserve the Parameter's identity and vLLM attributes. The mapped
    # storage's deleter owns the pinned allocation after the swap.
    weight.data = view
    return perf_counter() - checked_at


def _offload_target(model):
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        UnquantizedEmbeddingMethod,
        VocabParallelEmbedding,
    )
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    embedding = _input_embedding(
        model, VocabParallelEmbedding, ParallelLMHead, UnquantizedEmbeddingMethod
    )
    weight = embedding.weight
    if (
        weight.shape != EMBEDDING_SHAPE
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or weight.requires_grad
        or not weight.is_cuda
        or not torch.version.hip
    ):
        raise RuntimeError(
            "host embedding requires the pinned Qwen3.8-27B BF16 input table on ROCm"
        )
    device = weight.device
    if (
        torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
        != "gfx1201"
    ):
        raise RuntimeError("host embedding requires gfx1201")

    torch.cuda.synchronize(device)
    table_bytes = weight.numel() * weight.element_size()
    allocated_before = torch.cuda.memory_allocated(device)
    readback_seconds = _replace_storage(weight, get_accelerator_view_from_cpu_tensor)
    torch.cuda.empty_cache()
    released_bytes = allocated_before - torch.cuda.memory_allocated(device)
    if released_bytes < table_bytes:
        raise RuntimeError(
            f"host embedding released {released_bytes:,} GPU bytes for a {table_bytes:,}-byte table"
        )
    return embedding, table_bytes, released_bytes, readback_seconds


@contextmanager
def host_embedding_during_load(worker):
    """Map the target table before drafter sharing and weight memory accounting."""
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM

    config = worker.vllm_config
    parallel = config.parallel_config
    target_config = config.model_config
    if (
        parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or config.speculative_config is None
        or config.lora_config is not None
        or target_config.enable_sleep_mode
    ):
        raise RuntimeError(
            "host embedding needs the Coherence TP1 DFlash2 serving profile"
        )

    load_model = BaseModelLoader.load_model
    report = None

    @wraps(load_model)
    def load_with_host_embedding(loader, vllm_config, model_config, prefix=""):
        nonlocal report
        model = load_model(loader, vllm_config, model_config, prefix)
        if model_config is target_config:
            if report is not None:
                raise RuntimeError("host embedding encountered multiple target loads")
            report = _offload_target(model)
        return model

    BaseModelLoader.load_model = load_with_host_embedding
    try:
        yield
    finally:
        BaseModelLoader.load_model = load_model

    if report is None:
        raise RuntimeError("host embedding did not observe the target model load")
    embedding, table_bytes, released_bytes, readback_seconds = report
    draft = worker.model_runner.speculator.model
    if (
        not isinstance(draft, DFlash2Qwen3ForCausalLM)
        or draft.model.embed_tokens is not embedding
    ):
        raise RuntimeError(
            "the DFlash2 drafter did not share the target's pinned host embedding"
        )
    if (
        draft.lm_head.weight.untyped_storage().data_ptr()
        == embedding.weight.untyped_storage().data_ptr()
    ):
        raise RuntimeError(
            "the DFlash2 LM head aliases the pinned host input embedding"
        )
    print(
        f"Host embedding candidate: verified {table_bytes:,} bytes through pinned host mapping "
        f"in {readback_seconds:.3f}s; released {released_bytes:,} GPU tensor bytes; "
        "target and DFlash2 share the input table",
        flush=True,
    )
