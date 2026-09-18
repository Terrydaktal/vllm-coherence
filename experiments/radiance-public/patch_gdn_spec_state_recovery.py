#!/usr/bin/env python3
"""Backport vLLM PR #40738's single-request GDN spec-state recovery.

After a speculative step accepts N tokens, vLLM can schedule one ordinary
decode token before the next draft.  The unpatched metadata path discards N,
reads recurrent state from block-table column zero, and reads convolution
state at offset zero.  This patch gathers the accepted recurrent state into
the ordinary decode destination and supplies N to the already-present
causal-conv update kernel.

The production lane is deliberately MAXSEQS=1.  Mixed speculative/non-
speculative batches remain outside this patch's declared contract and must be
qualified separately before increasing concurrency.
"""

from __future__ import annotations

import os
import sysconfig
from pathlib import Path


SITE = Path(os.environ.get("RADIANCE_PATCH_SITE", sysconfig.get_paths()["purelib"]))
GDN_META = SITE / "vllm/v1/attention/backends/gdn_attn.py"
GDN_LAYER = (
    SITE
    / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
)


def replace_once(path: Path, old: str, new: str, marker: str) -> None:
    text = path.read_text()
    if marker in text:
        print(f"[gdn-spec-state] already applied: {path}")
        return
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"gdn-spec-state: expected one anchor in {path}, found {count}: {old[:80]!r}"
        )
    path.write_text(text.replace(old, new, 1))
    print(f"[gdn-spec-state] applied: {path}")


replace_once(
    GDN_META,
    "    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]\n",
    "    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]\n"
    "\n"
    "    # radiance backport of vLLM PR #40738: source recurrent-state blocks\n"
    "    # selected by the preceding speculative step's accepted width.\n"
    "    spec_decode_src_indices: torch.Tensor | None = None\n",
    "radiance backport of vLLM PR #40738: source recurrent-state blocks",
)

replace_once(
    GDN_META,
    "        if spec_sequence_masks is None:\n"
    "            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (\n",
    "        # radiance backport of vLLM PR #40738: preserve accepted-state provenance.\n"
    "        spec_decode_src_indices = None\n"
    "        if spec_sequence_masks is None:\n"
    "            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (\n",
    "radiance backport of vLLM PR #40738: preserve accepted-state provenance",
)

replace_once(
    GDN_META,
    "            non_spec_state_indices_tensor = block_table_tensor[:, 0]\n"
    "            spec_query_start_loc = None\n"
    "            non_spec_query_start_loc = query_start_loc\n"
    "            non_spec_query_start_loc_cpu = query_start_loc_cpu\n"
    "            num_accepted_tokens = None\n",
    "            non_spec_state_indices_tensor = block_table_tensor[:, 0]\n"
    "            spec_query_start_loc = None\n"
    "            non_spec_query_start_loc = query_start_loc\n"
    "            non_spec_query_start_loc_cpu = query_start_loc_cpu\n"
    "            if (\n"
    "                self.use_spec_decode\n"
    "                and num_accepted_tokens is not None\n"
    "                and num_decodes > 0\n"
    "            ):\n"
    "                # MAXSEQS=1 production contract: the accepted state is in\n"
    "                # column N-1, while this ordinary decode writes column zero.\n"
    "                col_indices = (num_accepted_tokens[:num_decodes] - 1).clamp(min=0)\n"
    "                spec_decode_src_indices = block_table_tensor[\n"
    "                    torch.arange(num_decodes, device=block_table_tensor.device),\n"
    "                    col_indices,\n"
    "                ]\n"
    "                num_accepted_tokens = num_accepted_tokens[:num_decodes]\n"
    "                if num_prefills > 0:\n"
    "                    raise RuntimeError(\n"
    "                        'mixed GDN prefill/decode state recovery is outside the MAXSEQS=1 contract'\n"
    "                    )\n"
    "            else:\n"
    "                num_accepted_tokens = None\n",
    "MAXSEQS=1 production contract: the accepted state is in",
)

replace_once(
    GDN_META,
    "            num_accepted_tokens=num_accepted_tokens,\n"
    "            nums_dict=nums_dict,\n",
    "            num_accepted_tokens=num_accepted_tokens,\n"
    "            spec_decode_src_indices=spec_decode_src_indices,\n"
    "            nums_dict=nums_dict,\n",
    "spec_decode_src_indices=spec_decode_src_indices",
)

replace_once(
    GDN_LAYER,
    "        num_actual_tokens = attn_metadata.num_actual_tokens\n"
    "        num_accepted_tokens = attn_metadata.num_accepted_tokens\n"
    "\n"
    "        mixed_qkv = mixed_qkv[:num_actual_tokens]\n",
    "        num_actual_tokens = attn_metadata.num_actual_tokens\n"
    "        num_accepted_tokens = attn_metadata.num_accepted_tokens\n"
    "\n"
    "        # radiance backport of vLLM PR #40738: publish accepted SSM state\n"
    "        # into the ordinary-decode destination before either GDN kernel reads it.\n"
    "        spec_decode_src_indices = attn_metadata.spec_decode_src_indices\n"
    "        if spec_decode_src_indices is not None:\n"
    "            assert non_spec_state_indices_tensor is not None\n"
    "            n_correct = spec_decode_src_indices.shape[0]\n"
    "            dst_indices = non_spec_state_indices_tensor[:n_correct]\n"
    "            ssm_state[dst_indices] = ssm_state[spec_decode_src_indices]\n"
    "\n"
    "        mixed_qkv = mixed_qkv[:num_actual_tokens]\n",
    "radiance backport of vLLM PR #40738: publish accepted SSM state",
)

replace_once(
    GDN_LAYER,
    "                conv_state_indices=non_spec_state_indices_tensor[  # type: ignore[index]\n"
    "                    : attn_metadata.num_actual_tokens  # type: ignore[attr-defined]\n"
    "                ],\n"
    "                validate_data=True,\n",
    "                conv_state_indices=non_spec_state_indices_tensor[  # type: ignore[index]\n"
    "                    : attn_metadata.num_actual_tokens  # type: ignore[attr-defined]\n"
    "                ],\n"
    "                num_accepted_tokens=(\n"
    "                    num_accepted_tokens\n"
    "                    if spec_decode_src_indices is not None\n"
    "                    else None\n"
    "                ),\n"
    "                validate_data=True,\n",
    "if spec_decode_src_indices is not None\n                    else None",
)

replace_once(
    GDN_LAYER,
    "        ssm_state = self_kv_cache[1]\n"
    "        num_actual_tokens = attn_metadata.num_actual_tokens\n"
    "\n"
    "        mixed_qkv = mixed_qkv[:num_actual_tokens]\n",
    "        ssm_state = self_kv_cache[1]\n"
    "        num_actual_tokens = attn_metadata.num_actual_tokens\n"
    "        num_accepted_tokens = attn_metadata.num_accepted_tokens\n"
    "        spec_decode_src_indices = attn_metadata.spec_decode_src_indices\n"
    "        if spec_decode_src_indices is not None:\n"
    "            assert non_spec_state_indices_tensor is not None\n"
    "            n_correct = spec_decode_src_indices.shape[0]\n"
    "            dst_indices = non_spec_state_indices_tensor[:n_correct]\n"
    "            ssm_state[dst_indices] = ssm_state[spec_decode_src_indices]\n"
    "\n"
    "        mixed_qkv = mixed_qkv[:num_actual_tokens]\n",
    "num_accepted_tokens = attn_metadata.num_accepted_tokens\n"
    "        spec_decode_src_indices = attn_metadata.spec_decode_src_indices",
)

replace_once(
    GDN_LAYER,
    "            conv_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]\n"
    "            validate_data=False,\n",
    "            conv_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]\n"
    "            num_accepted_tokens=(\n"
    "                num_accepted_tokens if spec_decode_src_indices is not None else None\n"
    "            ),\n"
    "            validate_data=False,\n",
    "num_accepted_tokens if spec_decode_src_indices is not None else None",
)
