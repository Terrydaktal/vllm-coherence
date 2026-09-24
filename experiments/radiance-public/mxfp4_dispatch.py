"""Install the sampled, exact GEMM dispatch backport before graph capture."""

import importlib.util
import json
from pathlib import Path

from build_mxfp4_dispatch import ORIGINAL_BINARY_SHA256, PYTHON_SHA256, digest

from qwen_r9700_lab.conformance_topk import require


def validate(entry):
    root = Path(entry["build"])
    require(digest(root / "build.json") == entry["build_sha256"], "GEMM build changed")
    build = json.loads((root / "build.json").read_text())
    require(
        build["original_binary_sha256"] == ORIGINAL_BINARY_SHA256, "GEMM oracle changed"
    )
    for name, metadata in build["variants"].items():
        for file, key in (
            ("radiance_mxfp4_fp8.so", "binary_sha256"),
            ("radiance_mxfp4_fp8.hip", "source_sha256"),
            ("radiance_mxfp4.py", "python_sha256"),
        ):
            require(
                digest(root / name / file) == metadata[key], "GEMM artifact changed"
            )
    require(
        set(entry["qualifications"]) == {"automatic", "split4"}, "GEMM checks missing"
    )
    for name, evidence in entry["qualifications"].items():
        path = Path(evidence["path"])
        require(digest(path) == evidence["sha256"], "GEMM evidence changed")
        report = json.loads(path.read_text())
        require(
            report["build"] == entry["build_sha256"], "GEMM checked a different build"
        )
        require(report["status"] == "SAMPLE_CHECKED", "GEMM sample failed")
        require(report["negative_control_detected"], "GEMM negative control missing")
        require(
            report["split_k"] == ("automatic" if name == "automatic" else 4),
            "GEMM split-K evidence differs",
        )
        require(
            report["probe_sha256"]
            == digest(Path(__file__).with_name("probe_mxfp4_dispatch.py")),
            "GEMM probe changed",
        )
        require(
            len(report["cases"]) == (35 if name == "automatic" else 20),
            "GEMM coverage incomplete",
        )
        for case in report["cases"]:
            if "cross_width" in case:
                require(
                    all(case[k]["equal"] for k in ("original", "control", "candidate")),
                    "GEMM M1/M8 mismatch",
                )
            else:
                require(
                    all(case[k] for k in ("finite", "canaries", "counters_zero"))
                    and case["candidate_vs_original"]["equal"]
                    and case["control_vs_original"]["equal"],
                    "GEMM mismatch or memory error",
                )
    return build


def install(entry):
    import torch
    from vllm.distributed import get_tensor_model_parallel_world_size

    import radiance_mxfp4 as kernel

    build = validate(entry)
    precision = entry.get("m1_arithmetic")
    if precision:
        from m1_arithmetic_release import validate as validate_precision

        build, _ = validate_precision(precision)
    require(
        not torch.cuda.is_current_stream_capturing(), "GEMM must precede graph capture"
    )
    require(
        get_tensor_model_parallel_world_size() == 1, "GEMM adapter was checked on TP1"
    )
    expected = build["variants"]["candidate"] if precision else None
    require(
        digest(kernel.__file__)
        == (expected["python_sha256"] if expected else PYTHON_SHA256),
        "GEMM wrapper differs from frozen baseline",
    )
    require(
        digest(kernel._ext.__file__)
        == (expected["binary_sha256"] if expected else ORIGINAL_BINARY_SHA256),
        "GEMM baseline changed",
    )
    require(
        kernel.DECODE_MAX_M == 64 and kernel.WPERM,
        "GEMM profile differs from checked scope",
    )
    require(
        kernel._decode_scratch_ready[0] and kernel._decode_scratch[0] is not None,
        "GEMM must follow weight loading",
    )
    device = kernel._decode_scratch[0].device
    partial = torch.empty(4 * 64 * 36864, dtype=torch.float32, device=device)
    counters = torch.zeros(36864 // 128 + 8, dtype=torch.int32, device=device)
    if precision:
        # Weight loading already used the repaired fold reference. Never replace
        # that kernel with the old table while allocating the dispatch scratch.
        candidate = kernel._ext
    else:
        binary = Path(entry["build"]) / "candidate/radiance_mxfp4_fp8.so"
        spec = importlib.util.spec_from_file_location(
            "qualified_dispatch.radiance_mxfp4_fp8", binary
        )
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
    candidate.set_decode_scratch(
        partial.data_ptr(), partial.numel() * 4, counters.data_ptr()
    )
    # There are no live graphs yet. Publish the new kernel and retain its
    # allocator-owned scratch for the same lifetime as the original wrapper.
    kernel._decode_scratch[:] = [partial, counters]
    kernel._ext = candidate
    return {
        "build_sha256": (precision or entry)["build_sha256"],
        "binary_sha256": build["variants"]["candidate"]["binary_sha256"],
        "scratch_bytes": partial.numel() * 4,
        "scope": "TP1 operator samples; M1 precision repairs"
        if precision
        else "TP1 decode width; exact sampled native/graph outputs, not a universal proof",
    }
