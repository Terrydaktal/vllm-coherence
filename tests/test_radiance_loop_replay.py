from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "experiments" / "radiance-public"
sys.path.insert(0, str(SOURCE))
spec = importlib.util.spec_from_file_location("diagnose_loop_replay", SOURCE / "diagnose_loop_replay.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_concurrent_operator_stop_does_not_prevent_restoration(monkeypatch):
    from types import SimpleNamespace

    for replies in ([(0, "")], [(125, ""), (1, "")],
                    [(125, ""), (0, ""), (0, "false\n")]):
        pending = iter(replies)

        def run(*args, **kwargs):
            code, output = next(pending)
            return SimpleNamespace(returncode=code, stdout=output)

        monkeypatch.setattr(module.subprocess, "run", run)
        module.stop_isolated_backend("qwen-runtime-ab-example")
        assert list(pending) == []


def test_cleanup_refuses_to_restore_over_a_running_or_unknown_backend(monkeypatch):
    from types import SimpleNamespace
    import pytest

    for replies in ([(125, ""), (0, ""), (0, "true\n")], [(125, ""), (125, "")]):
        pending = iter(replies)

        def run(*args, **kwargs):
            code, output = next(pending)
            return SimpleNamespace(returncode=code, stdout=output)

        monkeypatch.setattr(module.subprocess, "run", run)
        with pytest.raises(RuntimeError, match="stopped state"):
            module.stop_isolated_backend("qwen-runtime-ab-example")


def test_cleanup_cannot_target_a_production_container():
    import pytest

    with pytest.raises(ValueError, match="experiment namespace"):
        module.stop_isolated_backend("production")


def test_repeated_paragraphs_are_detected_without_returning_text_or_tokens():
    text = ("I will now perform the requested example operation.\n\n" * 12)
    result = module.repetition(text, list(range(80)) * 12)
    assert result["loop_candidate"]
    assert result["max_identical_paragraphs"] == 12
    assert result["repeated_64_token_window_fraction"] > 0.8
    assert all(isinstance(value, (int, float, bool)) for value in result.values())


def test_long_nonrepeated_reasoning_is_not_classified_as_a_loop():
    text = "\n\n".join(f"A distinct observation about component number {index}." for index in range(300))
    result = module.repetition(text, list(range(10000)))
    assert not result["loop_candidate"]
    assert result["repeated_64_token_window_fraction"] == 0


def test_repeated_blocks_are_detected_when_paragraph_spacing_changes():
    result = module.repetition("Words without repeated paragraph boundaries.", list(range(120)) * 10)
    assert result["loop_candidate"]
    assert result["max_identical_64_token_windows"] >= 9


def test_short_tool_emission_and_empty_output_are_not_loops():
    for text, ids in (("", []), ("<tool_call>example</tool_call>", list(range(16)))):
        result = module.repetition(text, ids)
        assert not result["loop_candidate"]


def test_compiler_cache_isolation_handles_legacy_and_runtime_bound_paths():
    for base in ("/cache/runtime-1.0.16", '"/cache/runtime/${abi_id}'):
        quote = '"' if base.startswith('"') else ''
        variables = [("VLLM_CACHE_ROOT", "vllm"),
                     ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                     ("TRITON_CACHE_DIR", "triton")]
        original = " ".join(f"-e {key}={base}/{kind}{quote}" for key, kind in variables)
        original += " -e AITER_ROOT_DIR=/cache/shared/aiter "
        isolated = module.isolate_compiler_cache(original, Path("/tmp/qwen-runtime-ab-example"))
        for key, kind in variables:
            assert f"-e {key}=/cache/benchmarks/qwen-runtime-ab-example/{kind} " in isolated
        assert "-e AITER_ROOT_DIR=/cache/shared/aiter " in isolated


def test_compiler_cache_isolation_rejects_ambiguous_environment():
    import pytest

    with pytest.raises(ValueError, match="missing or ambiguous"):
        module.isolate_compiler_cache(
            "-e VLLM_CACHE_ROOT=/cache/one -e VLLM_CACHE_ROOT=/cache/two ",
            Path("/tmp/qwen-runtime-ab-example"),
        )


def test_replay_scoring_rejects_stale_output_even_if_a_legacy_copy_matches(tmp_path):
    import json
    import pytest
    from score_loop_replay_tools import authenticated_output

    row = {"label": "00-example", "output_sha256": module.bench.digest([1, 2, 3])}
    (tmp_path / "output-arm-00-example.json").write_text(json.dumps({"token_ids": [4, 5, 6]}))
    (tmp_path / "output-00-example.json").write_text(json.dumps({"token_ids": [1, 2, 3]}))
    with pytest.raises(ValueError, match="recorded trial hash"):
        authenticated_output(tmp_path, "arm", row)
    (tmp_path / "output-arm-00-example.json").unlink()
    assert authenticated_output(tmp_path, "arm", row) == [1, 2, 3]


def test_gdn_audit_request_marker_is_removed_after_a_failed_request(tmp_path):
    import json
    import pytest

    marker = tmp_path / "gdn-audit-request.json"
    with pytest.raises(RuntimeError):
        with module.gdn_audit_request(tmp_path, "opaque-test-fixture", True):
            assert json.loads(marker.read_text()) == {"label": "opaque-test-fixture"}
            raise RuntimeError("diagnostic request interrupted")
    assert not marker.exists()
    with module.gdn_audit_request(tmp_path, "disabled", False):
        assert not marker.exists()


def test_repeated_code_is_not_reported_as_a_thinking_loop():
    text = "A brief plan.\n</think>\n<tool_call>" + "A repeated line of valid example source code.\n\n" * 30
    ids = [1, 2, 3, 99] + list(range(100, 180)) * 30
    assert module.repetition(text, ids)["loop_candidate"]
    result = module.thinking_repetition(text, ids, 99)
    assert result["closed"]
    assert result["output_tokens"] == 3
    assert not result["loop_candidate"]


def test_history_counterfactual_preserves_every_answer_and_tool_argument():
    from prepare_loop_counterfactuals import without_thinking
    payload = {"messages": [
        {"role": "user", "content": "Keep the repeated values: a a a a."},
        {"role": "assistant", "reasoning": "Prior internal reasoning.",
         "content": "An answer with significant repetition: x x x x.",
         "tool_calls": [{"function": {"name": "example", "arguments": '{"x":"x x x x"}'}}]},
        {"role": "tool", "content": "Exact tool response, including duplicates."},
    ], "temperature": 1}
    cleaned, metadata = without_thinking(payload)
    assert "reasoning" in payload["messages"][1]
    assert cleaned["messages"][0] == payload["messages"][0]
    assert cleaned["messages"][2] == payload["messages"][2]
    assert cleaned["messages"][1] == {k: v for k, v in payload["messages"][1].items() if k != "reasoning"}
    assert metadata == {"removed_thinking_characters": 25, "changed_assistant_messages": 1}


def test_scrambled_control_preserves_length_and_every_token_outside_thinking(tmp_path):
    from prepare_loop_counterfactuals import scrambled_thinking

    class CharacterTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "".join("<" + m["role"] + ">" + m.get("reasoning_content", "")
                           + "</reasoning>" + m.get("content", "") for m in messages)

        def __call__(self, text, **kwargs):
            return {"input_ids": list(map(ord, text)),
                    "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    thinking = "A deliberately distinct scratchpad with a unicode character: Δ."
    payload = {"messages": [{"role": "user", "content": "Do not change this instruction."},
                           {"role": "assistant", "reasoning": thinking, "content": "Exact answer."},
                           {"role": "tool", "content": "Exact tool data."}]}
    tokenizer = CharacterTokenizer()
    from prepare_loop_counterfactuals import normalized_messages
    text = tokenizer.apply_chat_template(normalized_messages(payload))
    reference = list(map(ord, text))
    template = tmp_path / "template.jinja"
    template.write_text("Unused by this character-level test tokenizer.")
    ids, metadata = scrambled_thinking(payload, tokenizer, template, reference)
    low = text.index(thinking)
    high = low + len(thinking)
    assert ids[:low] == reference[:low] and ids[high:] == reference[high:]
    assert sorted(ids[low:high]) == sorted(reference[low:high])
    assert ids != reference and len(ids) == len(reference)
    assert metadata["nonthinking_tokens_preserved"] and metadata["length_matched"]


def test_failed_history_control_preserves_healthy_thinking_and_all_other_fields():
    from prepare_loop_counterfactuals import without_looping_thinking

    class CharacterTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": list(map(ord, text))}

    repeated = "Now I will carry out the requested example operation.\n\n" * 12
    healthy = "The operation succeeded; the remaining step is a single check."
    payload = {"messages": [
        {"role": "user", "content": repeated},
        {"role": "assistant", "reasoning": repeated, "content": repeated,
         "tool_calls": [{"function": {"name": "example", "arguments": repeated}}]},
        {"role": "tool", "content": repeated},
        {"role": "assistant", "reasoning_content": healthy, "content": "Finished."},
    ]}
    cleaned, stats = without_looping_thinking(payload, CharacterTokenizer())
    assert stats["changed_assistant_messages"] == 1
    assert stats["removed_thinking_characters"] == len(repeated)
    assert stats["retained_thinking_characters"] == len(healthy)
    for index in (0, 2, 3):
        assert cleaned["messages"][index] == payload["messages"][index]
    assert "reasoning" not in cleaned["messages"][1]
    assert cleaned["messages"][1] == {k: v for k, v in payload["messages"][1].items()
                                       if k != "reasoning"}
    assert payload["messages"][1]["reasoning"] == repeated


def test_stream_replay_preserves_ids_but_only_exposes_numeric_progress(tmp_path):
    import io
    import json

    events = [
        {"choices": [{"index": 0, "text": "Private example", "token_ids": [101, 102], "finish_reason": None}]},
        {"choices": [{"index": 0, "text": ".", "token_ids": [103], "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3}},
    ]
    wire = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + b"data: [DONE]\n\n"
    result = module.stream_replay("http://example.invalid", {"return_token_ids": True},
        tmp_path / "private.json", tmp_path / "progress.json", open_url=lambda *a, **k: io.BytesIO(wire))
    assert result["choices"][0]["token_ids"] == [101, 102, 103]
    assert result["choices"][0]["text"] == "Private example."
    assert result["usage"]["completion_tokens"] == 3
    evidence = json.loads((tmp_path / "private.json").read_text())
    assert evidence == {"token_ids": [101, 102, 103], "complete": True}
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["output_tokens"] == 3 and progress["complete"]
    assert all(isinstance(value, (int, float, bool)) for value in progress.values())


def test_incomplete_stream_preserves_private_partial_evidence(tmp_path):
    import io
    import json
    import pytest

    wire = b'data: {"choices":[{"text":"Private partial","token_ids":[71,72]}]}\n\n'
    with pytest.raises(ValueError, match="without complete token evidence"):
        module.stream_replay("http://example.invalid", {}, tmp_path / "private.json",
            tmp_path / "progress.json", open_url=lambda *a, **k: io.BytesIO(wire))
    assert json.loads((tmp_path / "private.json").read_text()) == {
        "token_ids": [71, 72], "complete": False}
