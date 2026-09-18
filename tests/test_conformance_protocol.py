import copy
import json

import pytest

from qwen_r9700_lab.conformance_protocol import compare_protocol_pair
from qwen_r9700_lab.conformance_transport import Completion, ProtocolError, SSEDecoder
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, authenticate, private_json


def response():
    return {
        "prompt_token_ids": [11, 12],
        "token_ids": [21, 22, 23],
        "content": "Let me record it.",
        "reasoning": "The number is 37.",
        "tools": [
            {
                "id": "call-a",
                "name": "record",
                "arguments": '{"number":37}',
                "parsed_arguments": {"number": 37},
            }
        ],
        "finish_reason": "tool_calls",
        "usage": {"prompt_tokens": 2, "completion_tokens": 3},
    }


def test_matching_raw_tokens_and_parsed_output_accept_transport_assigned_tool_ids(tmp_path):
    a, b = response(), response()
    b["tools"][0]["id"] = "call-b"
    report = compare_protocol_pair(a, b, tmp_path / "comparison.json")
    authenticate(report)
    assert report["classification"] == "EXACT_OBSERVED_MATCH"
    assert report["prompt_complete"] and report["output_complete"]
    assert report["proof"] == "UNPROVED"


@pytest.mark.parametrize(
    "damage,classification",
    [
        ("missing_prompt", "MISSING_RAW_TOKEN_OBSERVATION"),
        ("missing_output", "MISSING_RAW_TOKEN_OBSERVATION"),
        ("partial_prompt", "MISSING_RAW_TOKEN_OBSERVATION"),
        ("partial_output", "MISSING_RAW_TOKEN_OBSERVATION"),
        ("missing_usage", "MISSING_RAW_TOKEN_OBSERVATION"),
        ("bool_token", "MISSING_RAW_TOKEN_OBSERVATION"),
        ("prompt", "PROMPT_TOKEN_DIFFERENCE"),
        ("generation", "GENERATED_TOKEN_DIFFERENCE"),
        ("whitespace", "PARSED_OUTPUT_DIFFERENCE"),
        ("reasoning", "PARSED_OUTPUT_DIFFERENCE"),
        ("tool", "PARSED_OUTPUT_DIFFERENCE"),
        ("finish", "PARSED_OUTPUT_DIFFERENCE"),
    ],
)
def test_injected_differences_are_preserved_and_classified_without_false_parser_blame(
    tmp_path, damage, classification
):
    a, b = response(), response()
    if damage == "missing_prompt":
        b.pop("prompt_token_ids")
    elif damage == "missing_output":
        b["token_ids"] = []
    elif damage == "partial_prompt":
        a["prompt_token_ids"] = b["prompt_token_ids"] = [11]
    elif damage == "partial_output":
        a["token_ids"] = b["token_ids"] = [21, 22]
    elif damage == "missing_usage":
        b["usage"] = None
    elif damage == "bool_token":
        b["token_ids"][0] = True
    elif damage == "prompt":
        b["prompt_token_ids"][1] = 13
    elif damage == "generation":
        b["token_ids"][1] = 24
        # Parsed strings still match: text-only comparison would miss the fault.
    elif damage == "whitespace":
        b["content"] = "\n\n " + b["content"]
    elif damage == "reasoning":
        b["reasoning"] += "!"
    elif damage == "tool":
        b["tools"][0]["arguments"] = '{"number":38}'
        b["tools"][0]["parsed_arguments"] = {"number": 38}
    elif damage == "finish":
        b["finish_reason"] = "stop"
    evidence = tmp_path / "comparison.json"
    before = copy.deepcopy([a, b])
    with pytest.raises(DiagnosticError, match=classification):
        compare_protocol_pair(a, b, evidence)
    report = private_json(evidence)
    authenticate(report)
    assert [a, b] == before
    assert report["classification"] == classification
    assert not report["whitespace_accepted_as_equal"]
    assert "Let me record" not in evidence.read_text()
    assert "The number is" not in evidence.read_text()
    if damage == "whitespace":
        assert report["content_equal_after_strip"] and not report["content_equal"]
    if damage in {"generation", "prompt"}:
        key = "first_output_difference" if damage == "generation" else "first_prompt_difference"
        assert report[key] == 1


@pytest.mark.parametrize("width", [1, 7, 10000])
def test_buffered_raw_tokens_and_first_chunk_prompt_survive_sse_fragmentation(width):
    events = [
        {
            "prompt_token_ids": [11, 12],
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "token_ids": []}],
        },
        # Buffered reasoning/tool arguments may have token IDs without display text.
        {"choices": [{"index": 0, "delta": {}, "token_ids": [21, 22]}]},
        {
            "choices": [
                {"index": 0, "delta": {"content": "λ"}, "token_ids": [23], "finish_reason": "stop"}
            ]
        },
        {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 3}},
    ]
    raw = (
        "".join("data: " + json.dumps(event, ensure_ascii=False) + "\n\n" for event in events)
        + "data: [DONE]\n\n"
    ).encode()
    decoder, completion = SSEDecoder(), Completion()
    for offset in range(0, len(raw), width):
        for event in decoder.feed(raw[offset : offset + width]):
            completion.accept(event)
    decoder.feed(b"", final=True)
    result = completion.result()
    assert result["prompt_token_ids"] == [11, 12]
    assert result["token_ids"] == [21, 22, 23]
    assert result["content"] == "λ"


@pytest.mark.parametrize("replacement", [[11, 13], [True, 12], "11,12", [-1]])
def test_changed_or_malformed_prompt_metadata_is_rejected(replacement):
    completion = Completion()
    completion.accept({"prompt_token_ids": [11, 12], "choices": []})
    with pytest.raises(ProtocolError, match="prompt token IDs"):
        completion.accept({"prompt_token_ids": replacement, "choices": []})


@pytest.mark.parametrize("top_level", [None, [11, 12]])
def test_plain_completions_prompt_ids_are_on_choice(top_level):
    completion = Completion()
    completion.accept(
        {
            "prompt_token_ids": top_level,
            "choices": [
                {
                    "index": 0,
                    "text": "done",
                    "prompt_token_ids": [11, 12],
                    "token_ids": [21, 22, 23],
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        }
    )
    assert completion.result()["prompt_token_ids"] == [11, 12]


def test_conflicting_chat_and_completions_prompt_ids_are_rejected():
    completion = Completion()
    with pytest.raises(ProtocolError, match="prompt token IDs changed"):
        completion.accept(
            {
                "prompt_token_ids": [11, 12],
                "choices": [
                    {
                        "index": 0,
                        "text": "done",
                        "prompt_token_ids": [11, 13],
                        "token_ids": [21],
                        "finish_reason": "stop",
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    "variant",
    [
        "repeat_release_full_head",
        "minimal_release_target_only",
        "repeat_release_target_only",
        "minimal_release_target_only_eager",
        "repeat_release_target_only_eager",
    ],
)
def test_release_isolation_uses_fresh_servers_and_detects_same_text_different_tokens(
    tmp_path, monkeypatch, variant
):
    from qwen_r9700_lab import conformance_scenarios as scenarios

    servers = []

    class Server:
        def __init__(self, _spec, root, **kwargs):
            self.root = root
            self.options = kwargs
            servers.append(self)

        def __enter__(self):
            self.root.mkdir(mode=0o700)
            return self

        def __exit__(self, *_args):
            self.closed = True

    def generate(server, *_args):
        result = response()
        if len(servers) == 2 and server is servers[1]:
            result["token_ids"][1] += 1
        return result

    monkeypatch.setattr(scenarios, "NativeServer", Server)
    monkeypatch.setattr(scenarios, "tokens", lambda *_args: [11, 12])
    monkeypatch.setattr(scenarios, "generate", generate)
    observations = []
    monkeypatch.setattr(
        scenarios, "observed", lambda server, *names: observations.append((server, names))
    )
    case = {"variant": variant, "context": 64, "seed": 0}
    with pytest.raises(DiagnosticError, match="GENERATED_TOKEN_DIFFERENCE"):
        scenarios.protocol({}, case, tmp_path)
    assert len(servers) == 2 and all(server.closed for server in servers)
    assert servers[0].root != servers[1].root
    assert not any(server.options["head"] for server in servers)
    if variant.startswith("repeat_"):
        assert not observations
        assert not any(server.options["observe"] for server in servers)
    else:
        names = ("runner.commit.return",)
        if not variant.endswith("_eager"):
            names += ("graph.replay.return",)
        assert observations == [(servers[0], names)]
        assert servers[0].options["observe"] and not servers[1].options["observe"]
    for server in servers:
        assert server.options["speculation"] is (variant == "repeat_release_full_head")
        assert server.options["graphs"] is (not variant.endswith("_eager"))
    report = private_json(tmp_path / "release-comparison.json")
    assert report["comparison_kind"] == "fresh_release_instances"
    assert report["first_output_difference"] == 1
