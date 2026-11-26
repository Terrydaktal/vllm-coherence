"""Extend a previously measured worker with explicitly selected numerical corrections."""

import json

from fourteen_base_retained import ROOT, before_load, manifest
from fourteen_base_retained import after_load as install_parent

__all__ = ["after_load", "before_load", "manifest"]


def after_load(worker):
    result = install_parent(worker)
    corrections = manifest()["numerical_corrections"]
    if "wide_input_norm" in corrections:
        from wide_input_norm import install

        installed = install(worker, ROOT / "wide-norm-build")
        if installed["binary_sha256"] != corrections["wide_input_norm"]:
            raise ValueError("installed wide norm differs from the selected correction")
        worker._fourteen_results["wide_input_norm"] = installed
    if "attention_runtime_splits" in corrections:
        from install_attention_runtime_splits import install

        install(worker, ROOT / "attention-runtime-splits")
        worker._fourteen_results["attention_runtime_splits"] = corrections[
            "attention_runtime_splits"
        ]
    if "wide_qk_norm" in corrections:
        from wide_qk_norm import install

        installed = install(worker)
        if installed["binary_sha256"] != corrections["wide_qk_norm"]:
            raise ValueError("installed Q/K normalization differs from its operator run")
        worker._fourteen_results["wide_qk_norm"] = installed
    if "draft_context_projection" in corrections and result["drafter_bound"]:
        from dflash_context_projection import install

        qualified = json.loads((ROOT / "draft-context-qualification.json").read_bytes())
        installed = install(worker)
        if any(installed[k] != qualified[k] for k in ("producer_sha256", "tile")):
            raise ValueError("loaded draft context projection differs from its qualification")
        worker._fourteen_results["draft_context_projection"] = installed
    return result
