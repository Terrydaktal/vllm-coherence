"""Release gates and installation for the sampled eager-M1 attention repairs."""

import hashlib
import json
from pathlib import Path

from attention_precision import digest
from m1_followup_guards import patch_packed

from qwen_r9700_lab.diagnostic_contract import authenticate

WRAPPER_SHA = "6d4b4571c938003d79e2edc322d1eadc674e30781597845a59b59b8065a6fd0a"
WRAPPER_FOOTER = """
# Coherence: install the qualified arithmetic before metadata/scratch allocation.
from attention_precision_runtime import install_bindings as _coherence_bind_attention
from attention_precision_runtime import guard_geometry as _coherence_guard_geometry
_coherence_bind_attention(r4d, os.environ["QWEN_ATTENTION_PRECISION_BUILD"])
_PREFILL = tuple(_bind("attn_prefill_paged", kv_dtype=d, **_PAGED) for d in _KV_DTYPES)
_DECODE = tuple(_bind("attn_decode_paged", kv_dtype=d, q_len=1, **_PAGED) for d in _KV_DTYPES)
_coherence_guard_geometry(R4DAttentionImpl)
"""


def wrapper_source(source):
    original = source.removesuffix(WRAPPER_FOOTER)
    if hashlib.sha256(original.encode()).hexdigest() != WRAPPER_SHA:
        raise ValueError("attention wrapper differs from pinned source")
    return original + WRAPPER_FOOTER


def validate(entry):
    root = Path(entry["build"])
    build = json.loads((root / "build.json").read_text())
    authenticate(build)
    if (
        digest(root / "build.json") != entry["build_sha256"]
        or build["kernel_abi"] != "coherence-attention-precision-v1"
    ):
        raise ValueError("attention precision build identity changed")
    if (
        build["generator_sha256"] != entry["sources"]["attention_precision.py"]
        or build["shared_generator_sha256"]
        != entry["sources"]["build_stock_m1_attention_shared.py"]
    ):
        raise ValueError("attention generators differ from the compiled build")
    for name, expected in build["files"].items():
        if Path(name).name != name or digest(root / name) != expected:
            raise ValueError("attention precision artifact changed")
    reports = {}
    for name in ("witnesses", "alignment", "oracle"):
        path = Path(entry[name])
        if digest(path) != entry[name + "_sha256"]:
            raise ValueError("attention precision evidence changed")
        reports[name] = json.loads(path.read_text())
    witness, alignment = reports["witnesses"], reports["alignment"]
    if (
        witness["probe_sha256"]
        != entry["sources"]["probe_attention_precision_repair.py"]
        or alignment["source_sha256"]
        != entry["sources"]["probe_stock_m1_attention_shared.py"]
        or reports["oracle"]["probe_sha256"]
        != entry["sources"]["probe_attention_precision_oracle.py"]
    ):
        raise ValueError("attention probes differ from their recorded evidence")
    authenticate(alignment)
    if (
        witness["status"] != "SAMPLE_CHECKED"
        or not witness["negative_control_detected"]
        or witness["build_sha256"] != entry["build_sha256"]
    ):
        raise ValueError("attention counterexamples have not been repaired")
    cases = witness.get("cases", [])
    from probe_m1_attention_precision import fixtures

    expected = {
        (phase, row[0]) for phase in ("decode", "prefill") for row in fixtures()
    }
    if (
        {(c["phase"], c["case"]) for c in cases} != expected
        or len(cases) != len(expected)
        or {c["phase"] for c in cases} != {"prefill", "decode"}
        or any(
            c["fixed"]["different_elements"] != 0
            or not c["fixed"]["guard_regions_intact"]
            for c in cases
        )
    ):
        raise ValueError("attention witness coverage incomplete")
    if (
        alignment["status"] != "SAMPLE_CHECKED"
        or alignment["build"] != build["sha256"]
        or alignment.get("precision_repair") != build["sha256"]
        or not alignment["graph_checks"]
        or not alignment["negative_control_detected"]
    ):
        raise ValueError("attention alignment failed or used another arithmetic")
    required = {
        (dtype, base + offset)
        for dtype in ("torch.float8_e4m3fn", "torch.bfloat16", "torch.uint8")
        for base in (0, 1024, 60000, 200000)
        for offset in range(16)
    }
    checks = alignment["checks"]
    if not required <= {(c["dtype"], c["prefix"]) for c in checks} or any(
        c["rows"] != 8
        or c["candidate_mismatches"] != 0
        or c["baseline_mismatches"] != 0
        for c in checks
    ):
        raise ValueError("attention alignment coverage incomplete")
    for name, expected in entry["sources"].items():
        if (
            Path(name).name != name
            or digest(Path(__file__).with_name(name)) != expected
        ):
            raise ValueError("attention runtime differs from tested source: " + name)
    oracle = reports["oracle"]
    if (
        oracle["status"] != "SAMPLE_CHECKED"
        or oracle["build_sha256"] != entry["build_sha256"]
        or len(oracle["checks"]) != 20
        or any(
            c["rows"] != 32
            or set(c["phases"]) != {"decode", "prefill"}
            or any(p["passed"] is not True for p in c["phases"].values())
            for c in oracle["checks"]
        )
    ):
        raise ValueError("independent FP64 oracle checks failed or are incomplete")
    return build, reports


def install(entry, package):
    validate(entry)
    package = Path(package)
    wrapper = package / "radiance_r4d_attn.py"
    packed = package / "vllm/third_party/flash_linear_attention/ops/fused_recurrent.py"
    # Validate everything before writing anything; installation precedes imports.
    after_wrapper = wrapper_source(wrapper.read_text())
    after_packed = patch_packed(packed.read_text())
    for path, source in ((wrapper, after_wrapper), (packed, after_packed)):
        path.write_text(source)
    return {
        "wrapper_sha256": digest(wrapper),
        "packed_sha256": digest(packed),
        "build_sha256": entry["build_sha256"],
    }
