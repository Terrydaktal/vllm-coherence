"""Install a qualified per-query split kernel before an isolated worker captures graphs."""

import os
from pathlib import Path

from attention_runtime_splits import PARENT_BINARY, load


def install(worker, directory):
    if os.environ.get("QWEN_SPEED_TARGET_ATTN_BUILD"):
        raise ValueError("target attention already has an independent candidate")
    attention = worker._qwen_performance_repairs.attention
    if attention.manifest["files"]["candidate.so"] != PARENT_BINARY:
        raise ValueError("unexpected shared target attention binary")
    if hasattr(attention, "_runtime_splits_original"):
        raise ValueError("runtime target attention was already installed")
    launch = load(Path(directory))
    attention._runtime_splits_original = attention.launch
    attention.launch = launch
