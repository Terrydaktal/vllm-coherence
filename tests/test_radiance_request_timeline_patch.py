from __future__ import annotations

import ast
import asyncio
import collections.abc
import gzip
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def patcher():
    spec = importlib.util.spec_from_file_location(
        "timeline_patch", ROOT / "experiments/radiance-public/patch_chat_snapshot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "name,function",
    [
        ("serving", "stream_buffered_tool_usage"),
        ("async_llm", "request_timeline_async"),
        ("gpu_worker", "memory_report_worker"),
    ],
)
def test_installed_pinned_source_is_patched_idempotently_and_unknown_rejected(
    patcher, name, function
):
    original = gzip.decompress(
        (ROOT / f"tests/fixtures/vllm_028_request_timeline_{name}.py.gz").read_bytes()
    ).decode()
    transform = getattr(patcher, function)
    updated = transform(original)
    compile(updated, name, "exec")
    assert transform(updated) == updated
    with pytest.raises(ValueError, match="source differs"):
        transform(updated + "\n# unreviewed source change\n")


def test_async_input_retains_request_id_assignment_and_submit_await(patcher):
    original = gzip.decompress(
        (ROOT / "tests/fixtures/vllm_028_request_timeline_async_llm.py.gz").read_bytes()
    ).decode()
    updated = patcher.request_timeline_async(original)
    assert updated.count(
        "self.input_processor.assign_request_id(request)"
    ) == original.count("self.input_processor.assign_request_id(request)")
    assert updated.index(
        "self.input_processor.assign_request_id(request)"
    ) < updated.index("request_timeline.internal_id_bridge(request)")
    assert updated.count(
        "await self.engine_core.add_request_async(request)"
    ) == original.count("await self.engine_core.add_request_async(request)")
    tree = ast.parse(updated)
    scopes = [node for node in ast.walk(tree) if isinstance(node, ast.With)]
    input_scope = next(
        node for node in scopes if "async_input_process" in ast.unparse(node.items[0])
    )
    assert "self.input_processor.assign_request_id(request)" in ast.unparse(input_scope)
    assert "self._run_output_handler()" not in ast.unparse(input_scope)


def test_api_marker_is_for_delta_not_initial_role_chunk(patcher):
    original = gzip.decompress(
        (ROOT / "tests/fixtures/vllm_028_request_timeline_serving.py.gz").read_bytes()
    ).decode()
    updated = patcher.stream_buffered_tool_usage(original)
    assert updated.count("request_timeline.first_api_content(choice_data.delta)") == 1
    assert updated.index("request_timeline.bind_external(request_id)") > updated.index(
        "with request_timeline.api_render():"
    )
    marker = updated.index("request_timeline.first_api_content(choice_data.delta)")
    assert updated.rfind("data = chunk.model_dump_json", 0, marker) > updated.index(
        "if first_iteration:"
    )
    assert updated[marker:].index("yield") < updated[marker:].index(
        "# once the final token"
    )


def test_actual_pinned_async_methods_keep_submission_and_output_identity(
    patcher, monkeypatch
):
    """Execute the patched production methods with CPU-only engine stubs."""
    module_spec = importlib.util.spec_from_file_location(
        "timeline_execution", ROOT / "src/qwen_r9700_lab/radiance_request_timeline.py"
    )
    timeline = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(timeline)
    events = []
    monkeypatch.setattr(
        timeline.telemetry,
        "emit_at",
        lambda stage, a, b, **values: events.append(
            {"stage": stage, "start_ns": a, "end_ns": b, **values}
        ),
    )
    original = gzip.decompress(
        (ROOT / "tests/fixtures/vllm_028_request_timeline_async_llm.py.gz").read_bytes()
    ).decode()
    tree = ast.parse(patcher.request_timeline_async(original))
    target = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AsyncLLM"
    )
    methods = [
        node
        for node in target.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name in {"add_request", "_add_request", "generate"}
    ]

    class EngineCoreRequest:
        pass

    class RequestOutput:
        finished = True
        outputs = (SimpleNamespace(token_ids=(123,)),)

    output = RequestOutput()
    queues = []

    class Collector:
        def __init__(self, _kind, request_id):
            self.request_id = request_id
            queues.append(self)

        def get_nowait(self):
            return output

        def close(self):
            return None

    namespace = {
        "request_timeline": timeline,
        "AsyncGenerator": collections.abc.AsyncGenerator,
        "EngineCoreRequest": EngineCoreRequest,
        "PoolingParams": type("PoolingParams", (), {}),
        "RequestOutputCollector": Collector,
        "extract_prompt_components": lambda _model, _prompt: (None, None, None),
        "RequestOutput": RequestOutput,
        "STREAM_FINISHED": object(),
    }
    isolated = ast.ClassDef(
        name="Probe", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    exec("from __future__ import annotations\n" + ast.unparse(isolated), namespace)  # noqa: S102 - execute pinned code with CPU stubs
    probe = namespace["Probe"]()
    params = SimpleNamespace(n=1, output_kind="delta")
    request = EngineCoreRequest()
    request.request_id = "external"
    request.external_req_id = None
    request.params = params
    submitted = []
    local_outputs = []

    def assign(value):
        value.external_req_id = value.request_id
        value.request_id = value.external_req_id + "-8randoms"

    async def submit(value):
        submitted.append(value)
        await asyncio.sleep(0)

    async def tasks():
        return ("generate",)

    probe.errored = probe.log_requests = False
    probe.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False)
    )
    probe.input_processor = SimpleNamespace(
        process_inputs=lambda *_args, **_kw: request, assign_request_id=assign
    )
    probe.model_config = object()
    probe._run_output_handler = lambda: None
    probe.get_supported_tasks = tasks
    probe.output_processor = SimpleNamespace(
        add_request=lambda value, *_args: local_outputs.append(value)
    )
    probe.engine_core = SimpleNamespace(add_request_async=submit)
    context_token = timeline._request.set(
        {"identities": {"http_request_id": "a" * 64}, "observed": set()}
    )

    async def run():
        return [
            value
            async for value in probe.generate({"type": "tokens"}, params, "external")
        ]

    try:
        assert asyncio.run(run()) == [output]
    finally:
        timeline._request.reset(context_token)
    assert submitted == local_outputs == [request]
    assert queues[0].request_id == "external-8randoms"
    assert request.external_req_id == "external"
    assert [row["stage"] for row in events] == [
        "async_input_process",
        "internal_id_bridge",
        "engine_submit",
        "first_engine_output",
    ]
    assert events[-1]["request_id"] == timeline._hash("external-8randoms")
    assert events[-1]["external_request_id"] == timeline._hash("external")
    assert all(row.get("success", True) for row in events)
