#!/usr/bin/env python3
"""Permit fixed-shape specialization in diagnostic DFlash-width runs.

vLLM marks model token dimensions strictly dynamic.  DFlash block sizes below
the checkpoint's native eight-row block legitimately specialize those
dimensions during the profile AOT compile, so Torch rejects the graph while
building guards.  ``maybe_mark_dynamic`` retains dynamic compilation when the
graph supports it and permits specialization when the graph requires it.

This is a diagnostic-only patch.  The launcher invokes it only when
RADIANCE_DFLASH_SMALL_WIDTH_COMPILE=1 is explicitly set.
"""

from __future__ import annotations

import os
import sysconfig
from pathlib import Path


SITE = Path(os.environ.get("RADIANCE_PATCH_SITE", sysconfig.get_paths()["purelib"]))
DECORATORS = SITE / "vllm/compilation/decorators.py"
MARKER = "radiance diagnostic small-width DFlash specialization"

text = DECORATORS.read_text()
if MARKER in text:
    print(f"[dflash-small-width] already applied: {DECORATORS}")
    raise SystemExit(0)

old = """            else:
                dims = [dim for dim, _ in dim_shape_pairs]
                torch._dynamo.mark_dynamic(arg, dims)
"""
new = """            else:
                dims = [dim for dim, _ in dim_shape_pairs]
                # radiance diagnostic small-width DFlash specialization:
                # allow DFlash's fixed proposal block to specialize this dim.
                torch._dynamo.maybe_mark_dynamic(arg, dims)
"""

count = text.count(old)
if count != 1:
    raise SystemExit(
        f"dflash-small-width: expected one anchor in {DECORATORS}, found {count}"
    )

DECORATORS.write_text(text.replace(old, new, 1))
print(f"[dflash-small-width] applied: {DECORATORS}")
