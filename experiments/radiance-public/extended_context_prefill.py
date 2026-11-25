"""Extend the text context bound in the pinned fixed-32 attention release."""

import ast
import functools
import hashlib
import json
import os
from pathlib import Path
from types import CodeType, FunctionType

from qwen_r9700_lab.diagnostic_contract import authenticate, private_json

ORIGINAL_CONTEXT = 253_792
EXTENDED_CONTEXT = 319_000
MAX_EXTENDED_CONTEXT = 344_000

SOURCE_HASHES = {
    "prefill_alignment_runtime.py": "5ab7d95abdb5fa05e63d7f74b738cc01055d6cd0b268fa0666cb499d95363c60",
    "prefill_attention_alignment.py": "9342dd68eb180b9abaf2afb74419a19298eb846f561f84ec8c9515616c2cd8e4",
}
ATTENTION_BUILD_SHA256 = (
    "261be6002cd21629dc9b5f20e5ba6d449a81bcc5fced4ec6b31c9fe60c2f6711"
)
ATTENTION_HIP_SHA256 = (
    "dc3f7467f7cc628266947d0164bd5e6fdfa068425ecd083aa5ca7fbb1e83715a"
)

_installed = None


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _code_at(root, names):
    for name in names:
        matches = [
            child
            for child in root.co_consts
            if isinstance(child, CodeType) and child.co_name == name
        ]
        if len(matches) != 1:
            raise ValueError("qualified prefill function code was not found")
        root = matches[0]
    return root


def _node_at(root, names):
    for name in names:
        matches = [
            child
            for child in root.body
            if isinstance(child, (ast.ClassDef, ast.FunctionDef)) and child.name == name
        ]
        if len(matches) != 1:
            raise ValueError("qualified prefill function source was not found")
        root = matches[0]
    return root


def _prepare_guard(module, names, comparison, context_limit):
    path = Path(module.__file__)
    raw = path.read_bytes()
    if _sha256(raw) != SOURCE_HASHES.get(path.name):
        raise ValueError(f"qualified prefill source changed: {path.name}")
    owner = module if len(names) == 1 else getattr(module, names[0])
    original = getattr(owner, names[-1])
    file = str(path)
    original_code = _code_at(compile(raw, file, "exec", dont_inherit=True), names)
    if original.__code__ != original_code or original.__globals__ is not vars(module):
        raise ValueError(f"loaded prefill function changed: {original.__qualname__}")

    tree = ast.parse(raw, filename=file)
    guard = _node_at(tree, names)
    matches = [
        node
        for node in ast.walk(guard)
        if isinstance(node, ast.Compare) and ast.unparse(node) == comparison
    ]
    if len(matches) != 1:
        raise ValueError(
            f"qualified prefill context check changed: {original.__qualname__}"
        )
    bound = matches[0].comparators[-1]
    if not isinstance(bound, ast.Constant) or bound.value != ORIGINAL_CONTEXT:
        raise ValueError("qualified prefill context check has a different upper bound")
    bound.value = context_limit
    expanded = _code_at(compile(tree, file, "exec", dont_inherit=True), names)
    replacement = FunctionType(
        expanded,
        original.__globals__,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    replacement.__kwdefaults__ = original.__kwdefaults__
    return owner, names[-1], functools.update_wrapper(replacement, original)


def install(context_limit=EXTENDED_CONTEXT):
    """Run after authenticated bootstrap and before the optimized worker loads."""
    global _installed

    if (
        type(context_limit) is not int
        or not ORIGINAL_CONTEXT < context_limit <= MAX_EXTENDED_CONTEXT
    ):
        raise ValueError(
            f"extended text prefill context must be between {ORIGINAL_CONTEXT + 1} and {MAX_EXTENDED_CONTEXT}"
        )

    import prefill_alignment_runtime as routing
    import prefill_attention_alignment as attention

    performance = private_json(Path(os.environ["QWEN_OPTIMIZED_PERFORMANCE"]))
    authenticate(performance)
    entry = performance["prefill_alignment"]
    if _installed:
        if (
            _installed["parent_performance"] != performance["sha256"]
            or _installed["extended_context"] != context_limit
        ):
            raise ValueError(
                "a different extended prefill contract is already installed"
            )
        return dict(_installed)

    if any(entry["sources"].get(name) != sha for name, sha in SOURCE_HASHES.items()):
        raise ValueError(
            "extended prefill requires the audited fixed-32 Python adapters"
        )
    native = Path(entry["attention"]["build"])
    native_build = json.loads((native / "build.json").read_bytes())
    authenticate(native_build)
    if (
        entry["attention"]["build_sha256"] != ATTENTION_BUILD_SHA256
        or native_build["sha256"] != ATTENTION_BUILD_SHA256
        or _sha256((native / "aligned-prefill.hip").read_bytes())
        != ATTENTION_HIP_SHA256
    ):
        raise ValueError(
            "extended prefill requires the audited fixed-32 native source and build"
        )
    routing.validate(entry)
    replacements = (
        _prepare_guard(
            routing,
            ("prefill_admitted",),
            "plan[0][2] <= md.r4d_max_ctx <= 253792",
            context_limit,
        ),
        _prepare_guard(
            attention,
            ("AlignedPrefillAttention", "__call__"),
            "q.shape[0] <= context_limit <= 253792",
            context_limit,
        ),
    )
    for owner, name, replacement in replacements:
        setattr(owner, name, replacement)
    _installed = {
        "previous_context": ORIGINAL_CONTEXT,
        "extended_context": context_limit,
        "parent_performance": performance["sha256"],
        "native_build_sha256": ATTENTION_BUILD_SHA256,
        "native_hip_sha256": ATTENTION_HIP_SHA256,
        "original_sources": dict(SOURCE_HASHES),
        "kernel_changed": False,
    }
    return dict(_installed)
