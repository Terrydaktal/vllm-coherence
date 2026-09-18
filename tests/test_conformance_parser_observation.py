import pytest

from qwen_r9700_lab.conformance_observer import (
    Events,
    instrument_qwen_parser,
    read_events,
)
from qwen_r9700_lab.diagnostic_contract import DiagnosticError


def test_deployed_extraction_paths_are_observed_without_calling_parse_delta(tmp_path):
    returned = object()

    class Engine:
        def parse_delta(self, *_args, **_kwargs):
            raise AssertionError("the deployed adapter does not use this path")

        def extract_tool_calls_streaming(self, *args, **kwargs):
            assert args == ("private fixture",) and kwargs == {"finished": True}
            return returned

        def extract_tool_calls_from_content(self, *_args, **_kwargs):
            return returned

        def extract_reasoning_streaming(self, *_args, **_kwargs):
            return returned

        def extract_reasoning(self, *_args, **_kwargs):
            return returned

        def finish_streaming(self, *_args, **_kwargs):
            raise ValueError("native failure")

    class Qwen(Engine):
        pass

    events = Events(tmp_path, "a" * 64)
    instrument_qwen_parser(Qwen, events)
    parser = Qwen()
    assert parser.extract_tool_calls_streaming("private fixture", finished=True) is returned
    assert parser.extract_tool_calls_from_content("private fixture") is returned
    assert parser.extract_reasoning_streaming("private fixture") is returned
    assert parser.extract_reasoning("private fixture") is returned
    with pytest.raises(ValueError, match="native failure"):
        parser.finish_streaming()
    rows = read_events(tmp_path, "a" * 64)
    names = [r["event"] for r in rows]
    assert "parser.delta.return" not in names
    for name in ("tool_stream", "tool_nonstream", "reasoning_stream", "reasoning_nonstream"):
        assert names.count(f"parser.{name}.enter") == 1
        assert names.count(f"parser.{name}.return") == 1
    assert names.count("parser.finish.error") == 1
    assert "parser.finish.return" not in names
    assert "private fixture" not in str(rows)
    # Instrumenting the concrete grammar must not change other engine classes.
    assert not hasattr(Engine.extract_tool_calls_streaming, "__wrapped__")


def test_removed_native_entrypoint_is_an_explicit_coverage_failure(tmp_path):
    class Missing:
        def parse_delta(self):
            pass

    with pytest.raises(DiagnosticError, match=r"parser\.tool_stream"):
        instrument_qwen_parser(Missing, Events(tmp_path, "a" * 64))
