"""Exercise the exact installed branch, including the stale-state counterexample."""

import importlib.util
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "patch_gdn_initial_prefill",
    ROOT / "experiments/radiance-public/patch_gdn_initial_prefill.py",
)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)


def classify(source, counts, context):
    scope = {
        "m": SimpleNamespace(max_seq_len=context),
        "split_decodes_and_prefills": lambda m, decode_threshold: counts,
    }
    exec(  # noqa: S102 -- execute only this repository's installed metadata branch.
        compile(textwrap.dedent(source), "installed_gdn_branch", "exec"), scope
    )
    return tuple(
        scope[k]
        for k in (
            "num_decodes",
            "num_prefills",
            "num_decode_tokens",
            "num_prefill_tokens",
        )
    )


def test_fresh_one_token_prompt_uses_initializing_prefill_not_stale_history():
    assert classify(patch.OLD, (1, 0, 1, 0), 1) == (1, 0, 1, 0)
    assert classify(patch.NEW, (1, 0, 1, 0), 1) == (0, 1, 0, 1)


@pytest.mark.parametrize("length", [2, 8, 2048, 60000, 253792])
def test_existing_decode_and_prefill_classifications_are_unchanged(length):
    for counts in [(1, 0, 1, 0), (0, 1, 0, length), (0, 0, 0, 0)]:
        assert classify(patch.NEW, counts, length) == counts


def test_all_fresh_single_token_rows_and_padding():
    assert classify(patch.NEW, (2, 0, 2, 0), 1) == (0, 2, 0, 2)
    assert classify(patch.NEW, (0, 0, 0, 0), 1) == (0, 0, 0, 0)


def test_unrecognized_source_fails_closed():
    with pytest.raises(ValueError, match="pinned preimage"):
        patch.patched_source(patch.OLD)
