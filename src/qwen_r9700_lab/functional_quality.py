"""Non-executing functional-quality checks for fixed benchmark responses."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping
from typing import Any

SUPPORTED_QUALITY_KINDS = frozenset(
    {
        "integer-sequence-v1",
        "python-condition-reasoning-v1",
        "python-merge-intervals-v1",
        "single-tool-call-v1",
    }
)
_SINGLE_PYTHON_FENCE = re.compile(
    r"\s*```(?:python|py)?[ \t]*\n(?P<code>.*?)\n```\s*",
    flags=re.DOTALL | re.IGNORECASE,
)
_SEQUENCE_TEXT = re.compile(r"\s*\d+(?:\s*,\s*\d+)*,?\s*")
_THINK_TAG = re.compile(r"</?think>", flags=re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", flags=re.IGNORECASE)
_PYTHON_FENCE = re.compile(
    r"```(?:python|py)?[ \t]*\n(?P<code>.*?)\n```",
    flags=re.DOTALL | re.IGNORECASE,
)


def _check(check_id: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"id": check_id, "passed": passed, "detail": detail}


def _common_checks(
    evaluation: Mapping[str, Any], response: Mapping[str, Any]
) -> list[dict[str, Any]]:
    allowed = evaluation.get("allowed_finish_reasons", ["stop"])
    finish_reason = response.get("finish_reason")
    checks = [
        _check(
            "complete-finish",
            finish_reason in allowed,
            f"finish_reason={finish_reason!r}; allowed={allowed!r}",
        )
    ]
    if evaluation.get("require_no_reasoning") is True:
        reasoning = response.get("reasoning_content") or ""
        content = response.get("content") or ""
        checks.extend(
            (
                _check(
                    "no-reasoning-channel",
                    reasoning == "",
                    f"reasoning_content_chars={len(reasoning)}",
                ),
                _check(
                    "no-think-tags",
                    _THINK_TAG.search(content) is None,
                    "content must not contain think tags",
                ),
            )
        )
    return checks


def _sequence_checks(
    evaluation: Mapping[str, Any], response: Mapping[str, Any]
) -> list[dict[str, Any]]:
    content = response.get("content") or ""
    start = evaluation["expected_start"]
    end = evaluation["expected_end"]
    syntax_ok = _SEQUENCE_TEXT.fullmatch(content) is not None
    parsed: list[int] | None = None
    if syntax_ok:
        raw = content.strip().removesuffix(",")
        try:
            parsed = [int(item.strip()) for item in raw.split(",")]
        except ValueError:
            parsed = None
    expected = list(range(start, end + 1))
    return [
        _check(
            "sequence-only-output",
            syntax_ok,
            "output must contain only comma-separated decimal integers",
        ),
        _check(
            "sequence-exact-values",
            parsed == expected,
            f"expected {start} through {end}; parsed_count={len(parsed) if parsed else 0}",
        ),
    ]


def _python_source(content: str) -> tuple[str | None, str]:
    fence = _SINGLE_PYTHON_FENCE.fullmatch(content)
    if fence:
        return fence.group("code"), "single Python fence"
    if "```" in content:
        return None, "malformed or additional Markdown fence"
    return content.strip(), "raw Python"


def _is_value_error_raise(node: ast.AST) -> bool:
    if not isinstance(node, ast.Raise) or node.exc is None:
        return False
    exception = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
    return isinstance(exception, ast.Name) and exception.id == "ValueError"


def _has_touching_boundary(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for descendant in ast.walk(node):
            if not isinstance(descendant, ast.BinOp):
                continue
            if not isinstance(descendant.op, (ast.Add, ast.Sub)):
                continue
            operands = (descendant.left, descendant.right)
            if any(isinstance(item, ast.Constant) and item.value == 1 for item in operands):
                return True
    return False


def _python_checks(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    content = response.get("content") or ""
    source, envelope = _python_source(content)
    checks = [
        _check(
            "code-only-output",
            source is not None and bool(source.strip()),
            f"accepted envelope: {envelope}",
        )
    ]
    if source is None:
        return checks

    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        checks.append(_check("python-syntax", False, f"{error.msg} at line {error.lineno}"))
        return checks
    checks.append(_check("python-syntax", True, "ast.parse succeeded"))

    allowed_top_level = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom)
    unexpected_top_level = [
        type(node).__name__
        for index, node in enumerate(tree.body)
        if not isinstance(node, allowed_top_level)
        and not (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
    ]
    checks.append(
        _check(
            "module-shape",
            not unexpected_top_level,
            "unexpected top-level statements=" + repr(unexpected_top_level),
        )
    )

    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "merge_intervals"
    ]
    function = functions[0] if len(functions) == 1 else None
    checks.append(
        _check(
            "merge-intervals-function",
            function is not None,
            f"top-level definitions found={len(functions)}",
        )
    )
    if function is None:
        return checks

    parameter = function.args.args[0] if function.args.args else None
    checks.extend(
        (
            _check(
                "type-hints",
                parameter is not None
                and parameter.annotation is not None
                and function.returns is not None,
                "first parameter and return value must be annotated",
            ),
            _check(
                "docstring",
                bool(ast.get_docstring(function)),
                "merge_intervals must have a non-empty docstring",
            ),
            _check(
                "value-error-validation",
                any(_is_value_error_raise(node) for node in ast.walk(function)),
                "function must contain an explicit ValueError raise",
            ),
            _check(
                "sorting-step",
                any(
                    isinstance(node, ast.Call)
                    and (
                        (isinstance(node.func, ast.Name) and node.func.id == "sorted")
                        or (isinstance(node.func, ast.Attribute) and node.func.attr == "sort")
                    )
                    for node in ast.walk(function)
                ),
                "function must sort the input or a validated copy",
            ),
            _check(
                "touching-boundary-step",
                _has_touching_boundary(function),
                "a comparison must account for the adjacent-integer +1 boundary",
            ),
            _check(
                "return-step",
                any(isinstance(node, ast.Return) for node in ast.walk(function)),
                "function must return a result",
            ),
        )
    )
    return checks


def _final_answer(response: Mapping[str, Any]) -> tuple[str, str]:
    content = response.get("content") or ""
    reasoning = response.get("reasoning_content") or ""
    if reasoning:
        return content.strip(), "separate reasoning_content"
    closing_tags = list(_THINK_CLOSE.finditer(content))
    if closing_tags:
        return content[closing_tags[-1].end() :].strip(), "embedded think tags"
    if _THINK_TAG.search(content):
        return "", "unterminated embedded think block"
    return content.strip(), "unparsed or reasoning-free content"


def _call_leaf_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _assigned_constructor_names(tree: ast.AST, constructors: set[str]) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call) or _call_leaf_name(value) not in constructors:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names.update(target.id for target in targets if isinstance(target, ast.Name))
    return names


def _condition_bindings(tree: ast.AST, lock_names: set[str]) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if (
            not isinstance(value, ast.Call)
            or _call_leaf_name(value) != "Condition"
            or not value.args
            or not isinstance(value.args[0], ast.Name)
            or value.args[0].id not in lock_names
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                bindings[target.id] = value.args[0].id
    return bindings


class _ConditionUseVisitor(ast.NodeVisitor):
    def __init__(self, condition_name: str) -> None:
        self.condition_name = condition_name
        self.condition_depth = 0
        self.synchronization_calls: list[tuple[str, bool]] = []

    def visit_With(self, node: ast.With) -> None:
        enters_condition = any(
            isinstance(item.context_expr, ast.Name) and item.context_expr.id == self.condition_name
            for item in node.items
        )
        if enters_condition:
            self.condition_depth += 1
        for statement in node.body:
            self.visit(statement)
        if enters_condition:
            self.condition_depth -= 1

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == self.condition_name
            and node.func.attr in {"notify", "notify_all", "wait"}
        ):
            self.synchronization_calls.append((node.func.attr, self.condition_depth > 0))
        self.generic_visit(node)


def _wait_is_in_while(tree: ast.AST, condition_name: str) -> bool:
    return any(
        isinstance(node, ast.While)
        and any(
            isinstance(descendant, ast.Call)
            and isinstance(descendant.func, ast.Attribute)
            and isinstance(descendant.func.value, ast.Name)
            and descendant.func.value.id == condition_name
            and descendant.func.attr == "wait"
            for descendant in ast.walk(node)
        )
        for node in ast.walk(tree)
    )


def _condition_reasoning_checks(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    final_answer, answer_mode = _final_answer(response)
    checks = [
        _check(
            "final-answer-present",
            bool(final_answer),
            f"mode={answer_mode}; final_answer_chars={len(final_answer)}",
        )
    ]
    code_blocks = list(_PYTHON_FENCE.finditer(final_answer))
    source = code_blocks[-1].group("code") if code_blocks else None
    checks.append(
        _check(
            "corrected-code-present",
            source is not None,
            f"python_code_blocks={len(code_blocks)}",
        )
    )
    if source is None:
        checks.extend(
            (
                _check("corrected-code-syntax", False, "no corrected Python code found"),
                _check(
                    "shared-lock-condition-invariant",
                    False,
                    "no corrected syntax tree available",
                ),
                _check("wait-in-while", False, "no corrected syntax tree available"),
            )
        )
        return checks

    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        checks.append(_check("corrected-code-syntax", False, f"{error.msg} at line {error.lineno}"))
        checks.extend(
            (
                _check(
                    "shared-lock-condition-invariant",
                    False,
                    "corrected code did not parse",
                ),
                _check("wait-in-while", False, "corrected code did not parse"),
            )
        )
        return checks
    checks.append(_check("corrected-code-syntax", True, "ast.parse succeeded"))

    lock_names = _assigned_constructor_names(tree, {"Lock", "RLock"})
    bindings = _condition_bindings(tree, lock_names)
    condition_name = next(iter(bindings), None)
    uses = _ConditionUseVisitor(condition_name) if condition_name else None
    if uses is not None:
        uses.visit(tree)
    synchronization_calls = uses.synchronization_calls if uses is not None else []
    operations = {operation for operation, _protected in synchronization_calls}
    required_calls_present = "wait" in operations and bool(
        operations.intersection({"notify", "notify_all"})
    )
    protected_calls = required_calls_present and all(
        protected for _operation, protected in synchronization_calls
    )
    checks.append(
        _check(
            "shared-lock-condition-invariant",
            condition_name is not None and protected_calls,
            f"bindings={bindings!r}; synchronization_calls={synchronization_calls!r}",
        )
    )
    checks.append(
        _check(
            "wait-in-while",
            condition_name is not None and _wait_is_in_while(tree, condition_name),
            "condition.wait() must be nested in a while loop",
        )
    )
    return checks


def _single_tool_call_checks(
    evaluation: Mapping[str, Any], response: Mapping[str, Any]
) -> list[dict[str, Any]]:
    content = response.get("content") or ""
    tool_calls = response.get("tool_calls")
    call = tool_calls[0] if isinstance(tool_calls, list) and len(tool_calls) == 1 else None
    call_mapping = call if isinstance(call, Mapping) else {}
    function = call_mapping.get("function")
    function_mapping = function if isinstance(function, Mapping) else {}
    raw_arguments = function_mapping.get("arguments")
    parsed_arguments: Any = None
    argument_error: str | None = None
    if isinstance(raw_arguments, str):
        try:
            parsed_arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as error:
            argument_error = f"{error.msg} at character {error.pos}"
    else:
        argument_error = "function.arguments is not a JSON string"

    expected_name = evaluation.get("expected_tool_name")
    expected_arguments = evaluation.get("expected_arguments")
    return [
        _check(
            "no-prose-content",
            isinstance(content, str) and not content.strip(),
            f"content_chars={len(content) if isinstance(content, str) else 'non-string'}",
        ),
        _check(
            "tool-call-count",
            isinstance(tool_calls, list) and len(tool_calls) == 1,
            f"tool_call_count={len(tool_calls) if isinstance(tool_calls, list) else 'non-list'}",
        ),
        _check(
            "tool-call-id",
            isinstance(call_mapping.get("id"), str) and bool(call_mapping.get("id")),
            "the single tool call must have a non-empty id",
        ),
        _check(
            "tool-call-type",
            call_mapping.get("type") == "function",
            f"type={call_mapping.get('type')!r}",
        ),
        _check(
            "tool-name",
            function_mapping.get("name") == expected_name,
            f"expected={expected_name!r}; actual={function_mapping.get('name')!r}",
        ),
        _check(
            "tool-arguments-json",
            argument_error is None and isinstance(parsed_arguments, dict),
            argument_error or f"parsed_type={type(parsed_arguments).__name__}",
        ),
        _check(
            "tool-arguments-exact",
            parsed_arguments == expected_arguments,
            f"expected={expected_arguments!r}; actual={parsed_arguments!r}",
        ),
    ]


def evaluate_response(
    evaluation: Mapping[str, Any] | None,
    response: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Evaluate a response without executing model-generated source code."""

    if evaluation is None:
        return None
    kind = evaluation["kind"]
    checks = _common_checks(evaluation, response)
    if kind == "integer-sequence-v1":
        checks.extend(_sequence_checks(evaluation, response))
    elif kind == "python-condition-reasoning-v1":
        checks.extend(_condition_reasoning_checks(response))
    elif kind == "python-merge-intervals-v1":
        checks.extend(_python_checks(response))
    elif kind == "single-tool-call-v1":
        checks.extend(_single_tool_call_checks(evaluation, response))
    else:  # Fixture validation rejects this before a request is made.
        raise ValueError(f"unsupported quality evaluation kind: {kind}")
    result = {
        "kind": kind,
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "generated_code_executed": False,
    }
    if kind == "single-tool-call-v1":
        result["tool_executed"] = False
    return result
