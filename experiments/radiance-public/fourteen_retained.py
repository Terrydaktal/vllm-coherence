"""Common model-load hooks for the isolated performance and numerical qualification workers."""

import hashlib
import json
import os
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import authenticate

ROOT = Path("/retained")


def manifest():
    bound = json.loads((ROOT / "manifest.json").read_text())
    authenticate(bound)
    for rel, expected in bound["files"].items():
        path = ROOT / rel
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise ValueError("invalid retained build path")
        with path.open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"retained source or artifact changed: {rel}")
    return bound


def before_load(worker):
    """Run before model parameters, including the drafter's head, are bound."""
    import host_embedding
    from embedding_readback_candidate import install as install_readback

    if hasattr(worker, "_fourteen_prepared"):
        raise RuntimeError("retained worker already initialized")
    bound = manifest()
    install_readback(host_embedding)
    full_head = os.environ.get("RADIANCE_VERIFY_HEAD") != "1"
    if not full_head:
        from head_live_rows import install as install_target_rows

        install_target_rows()
    draft_bound = worker.vllm_config.speculative_config is not None
    if draft_bound:
        import radiance_drafthead_candidate as candidate

        import radiance_drafthead as stock

        stock_digest = hashlib.sha256(Path(stock.__file__).read_bytes()).hexdigest()
        if (
            stock_digest != bound["drafter_head_source"]
            or int(os.environ.get("RADIANCE_DRAFT_RERANK", "80")) != 80
        ):
            raise ValueError("unexpected installed drafter head")
        stock._apply_head_int2 = candidate._apply_head_int2
    worker._fourteen_prepared = {
        "binding": bound["sha256"],
        "full_head": full_head,
        "drafter_bound": draft_bound,
    }


def after_load(worker):
    """Install all retained operator dispatchers after release hooks and before graph capture."""
    from gdn_prefill_direct import install_projection, install_scan
    from mxfp4_prefill_tiles import install as install_tiles
    from prefill_bank_norm import install_bank, install_bias
    from prefill_conv_slot import initialize_conv_slot

    bound = manifest()
    if getattr(worker, "_fourteen_prepared", {}).get("binding") != bound["sha256"]:
        raise ValueError("model load does not share the prepared build")
    if hasattr(worker, "_fourteen_results"):
        raise RuntimeError("retained operators already installed")
    worker._fourteen_results = {
        "input_tiles": install_tiles(
            worker,
            ROOT / "prefill-tiles.so",
            max_preprocessed_bytes=96 << 20,
            slab_ahead=True,
        ),
        "direct_scan": install_scan(worker, conv_initializer=initialize_conv_slot),
        "direct_output_projection": install_projection(worker),
        "bank": install_bank(worker, conv_initializer=initialize_conv_slot),
        "bias": install_bias(worker),
    }
    runner = worker.model_runner
    enabled = (
        worker._fourteen_prepared["drafter_bound"]
        and not worker._fourteen_prepared["full_head"]
        and getattr(getattr(runner, "speculator", None), "draft_logits", None)
        is not None
    )
    worker._fourteen_results["sampled_dflash"] = enabled
    if enabled:
        from dflash_prompt_lookup import install as install_lookup
        from dflash_proposal_temperature import install as install_temperature

        worker._fourteen_lookup = install_lookup(
            runner, skip_draft=True, streak_only=True
        )
        worker._fourteen_temperature = install_temperature(runner)
        worker._fourteen_temperature(0.55)
    print(
        "Retained estimation and qualification binding "
        + json.dumps(worker._fourteen_prepared, sort_keys=True),
        flush=True,
    )
    return worker._fourteen_prepared
