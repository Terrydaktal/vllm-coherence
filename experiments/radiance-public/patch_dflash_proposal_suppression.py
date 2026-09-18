#!/usr/bin/env python3
"""Add an exact-geometry target-only control at the draft publication boundary.

The DFlash model still executes, so model loading, cache geometry, compilation,
and all target-side state layouts remain identical to the K7 candidate.  Its
proposals are withheld from the scheduler, forcing the target to consume one
token per transition.  This is a diagnostic oracle and is never enabled in the
production lane.
"""

from __future__ import annotations

import os
import sysconfig
from pathlib import Path


SITE = Path(os.environ.get("RADIANCE_PATCH_SITE", sysconfig.get_paths()["purelib"]))
UTILS = SITE / "vllm/v1/worker/gpu/spec_decode/utils.py"
MODEL_RUNNER = SITE / "vllm/v1/worker/gpu/model_runner.py"
MARKER = "radiance exact-geometry target-only control"

text = UTILS.read_text()
if MARKER in text:
    print(f"[dflash-proposal-suppression] already applied: {UTILS}")
    raise SystemExit(0)

old_import = "import numpy as np\nimport torch\n"
new_import = "import os\n\nimport numpy as np\nimport torch\n"
if text.count(old_import) != 1:
    raise SystemExit("dflash-proposal-suppression: import anchor mismatch")
text = text.replace(old_import, new_import, 1)

old_body = """        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]
        if not input_batch.has_structured_output_reqs:
"""
new_body = """        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]
        # radiance exact-geometry target-only control: the drafter ran, but
        # publish no proposal tokens, so the next target transition is serial.
        if os.environ.get("RADIANCE_DFLASH_SUPPRESS_PROPOSALS", "0") == "1":
            self.num_draft_tokens = 0
            draft_tokens = draft_tokens[:, :0]
        if not input_batch.has_structured_output_reqs:
"""
if text.count(old_body) != 1:
    raise SystemExit("dflash-proposal-suppression: handler anchor mismatch")
UTILS.write_text(text.replace(old_body, new_body, 1))
print(f"[dflash-proposal-suppression] applied: {UTILS}")

runner = MODEL_RUNNER.read_text()
runner_marker = "radiance exact-geometry target-only auxiliary suppression"
if runner_marker in runner:
    print(f"[dflash-proposal-suppression] already applied: {MODEL_RUNNER}")
    raise SystemExit(0)

old_runner_import = "import functools\nimport gc\n"
new_runner_import = "import functools\nimport gc\nimport os\n"
if runner.count(old_runner_import) != 1:
    raise SystemExit("dflash-proposal-suppression: model-runner import anchor mismatch")
runner = runner.replace(old_runner_import, new_runner_import, 1)

old_aux = """                self.use_aux_hidden_state_outputs = True
                if self.use_pp:
"""
new_aux = """                self.use_aux_hidden_state_outputs = True
                # radiance exact-geometry target-only auxiliary suppression:
                # isolate DFlash hidden-state capture from speculative cache geometry.
                if os.environ.get("RADIANCE_DFLASH_SUPPRESS_TARGET_AUX", "0") == "1":
                    self.use_aux_hidden_state_outputs = False
                if self.use_pp:
"""
if runner.count(old_aux) != 1:
    raise SystemExit("dflash-proposal-suppression: auxiliary-state anchor mismatch")
runner = runner.replace(old_aux, new_aux, 1)

old_profile = """        # dummy run the eagle speculator's propose to ensure DP/EP sync.
        if self.speculator is not None:
"""
new_profile = """        # dummy run the eagle speculator's propose to ensure DP/EP sync.
        if self.speculator is not None and os.environ.get(
            "RADIANCE_DFLASH_SUPPRESS_TARGET_AUX", "0"
        ) != "1":
"""
if runner.count(old_profile) != 1:
    raise SystemExit("dflash-proposal-suppression: profile proposer anchor mismatch")
runner = runner.replace(old_profile, new_profile, 1)

old_live = """        if self.speculator is not None:
            assert self.sampler is not None
            # Let the target override the hidden state fed to the drafter
"""
new_live = """        if self.speculator is not None and os.environ.get(
            "RADIANCE_DFLASH_SUPPRESS_TARGET_AUX", "0"
        ) != "1":
            assert self.sampler is not None
            # Let the target override the hidden state fed to the drafter
"""
if runner.count(old_live) != 1:
    raise SystemExit("dflash-proposal-suppression: live proposer anchor mismatch")
runner = runner.replace(old_live, new_live, 1)

MODEL_RUNNER.write_text(runner)
print(f"[dflash-proposal-suppression] applied: {MODEL_RUNNER}")
