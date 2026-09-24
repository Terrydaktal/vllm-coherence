"""CPU-only checks for the long coding/JSON/checkpoint benchmark."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/radiance-public/benchmark_pi_coding_json_compaction.py"
SPEC = importlib.util.spec_from_file_location("pi_coding_json_compaction", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return SimpleNamespace(ids=list(range(max(1, len(text.split())))))


def test_coding_phase_accounting_treats_fenced_edits_as_code():
    result = MODULE._coding_phases(
        FakeTokenizer(),
        "Plan it.\n```python\ndef f():\n    return 1\n```\nFinish.",
        "Check the invariants first.",
    )
    assert result["reasoning_tokens"] > 0
    assert result["prose_tokens"] > 0
    assert result["file_edit_code_tokens"] > 0
    assert result["file_edit_code_blocks"] == 1


def test_json_phase_accounting_accepts_a_document_as_a_file_edit():
    result = MODULE._json_phases(FakeTokenizer(), '{"items":[{"id":1}]}', "")
    assert result["json_valid"] is True
    assert result["file_edit_json_tokens"] > 0
    assert result["file_edit_json_blocks"] == 1


def test_prose_code_measurement_prompt_is_small_and_code_free():
    prompt = MODULE.PROSE_CODE_PROMPT.lower()
    assert "tokens per second" in prompt
    assert "acceptance rate" in prompt
    assert "markdown fences" in prompt
    result = MODULE._thinking_phases(
        FakeTokenizer(), "A short explanation of code measurement.", ""
    )
    assert result["prose_tokens"] > 0
    assert result["file_edit_code_tokens"] == 0


def test_checkpoint_validation_requires_one_marker_and_all_headings():
    headings = [
        "Goal",
        "Current Authoritative State",
        "Constraints & Invariants",
        "Progress",
        "Measurements & Evidence",
        "Key Decisions",
        "Rejected / Failed Approaches",
        "Unresolved Questions & Hypotheses",
        "Next Steps",
        "Critical Context",
    ]
    text = "\n".join(f"### {heading}\nNone" for heading in headings)
    result = MODULE._checkpoint_validation(
        f"{text}\n{MODULE.COMPACTION_MARKER}", "stop"
    )
    assert result["passed"] is True


def test_checkpoint_validation_rejects_a_missing_marker():
    assert MODULE._checkpoint_validation("### Goal\nNone", "stop")["passed"] is False


def test_compaction_prompt_requests_the_heading_syntax_its_validator_requires():
    import re

    headings = re.findall(r"^### .+$", MODULE.COMPACTION_PROMPT, re.MULTILINE)
    assert len(headings) == 10
    contract = MODULE._checkpoint_validation(
        "\n".join(headings + [MODULE.COMPACTION_MARKER]), "stop"
    )
    assert contract["passed"] is True
    pi_prompt = (
        ROOT / "integrations/pi/qwen-loss-sensitive-compact-prompt.md"
    ).read_text()
    for heading in headings:
        assert heading in pi_prompt.splitlines()
