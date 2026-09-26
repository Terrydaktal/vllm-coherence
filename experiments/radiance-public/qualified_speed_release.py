"""Install the exact lifecycle-qualified speed worker and its immutable payload."""

import hashlib
import json
import os
import shutil
from pathlib import Path

WORKER = "speed_candidate_worker.SpeedCandidateWorker"
CAPTURE = {
    "cudagraph_mode": "FULL_AND_PIECEWISE",
    "cudagraph_capture_sizes": [1, 2, 4, 8],
}
ENVIRONMENT = {
    "QWEN_SPEED_FULL_GRAPH_CAPTURE": "1",
    "QWEN_SPEED_TARGET_GEMM_BUILD": "/work/gemm-tuning-v9",
    "QWEN_SPEED_DRAFT_ATTN_AB": "1",
    "QWEN_SPEED_DRAFT_ATTN_SOURCE": "/work/draft-attention-v4-mode1/unit_w4_s1_occ2.py",
    "QWEN_SPEED_DRAFT_ATTN_GENERATOR": "/work/probe_draft_attention_tuning_v4.py",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_file(root, name, expected):
    relative = Path(name)
    path = root / relative
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or path.is_symlink()
        or not path.is_file()
        or not path.resolve().is_relative_to(root.resolve())
        or digest(path) != expected
    ):
        raise ValueError(f"qualified speed artifact missing or changed: {name}")
    return path


def install(patches, package, args, *, root=Path("/work"), environ=None):
    """Validate everything before changing the environment or installed worker."""
    environ = os.environ if environ is None else environ
    manifest = json.loads((patches / "qualified-speed-release.json").read_text())
    if (
        manifest.get("schema") != "urn:coherence:qualified-speed-release:v1"
        or manifest.get("worker") != WORKER
        or manifest.get("compilation_config") != CAPTURE
        or manifest.get("environment") != ENVIRONMENT
        or not manifest.get("files")
    ):
        raise ValueError("unsupported qualified speed configuration")
    if digest(patches / "optimized-release.json") != manifest["parent_manifest_sha256"]:
        raise ValueError("qualified speed parent release changed")
    for name, expected in manifest["files"].items():
        verify_file(root, name, expected)
    evidence = json.loads(
        verify_file(
            root, "lifecycle-qualification.json", manifest["qualification_sha256"]
        ).read_text()
    )
    for name, observed in (("observed", True), ("plain", False)):
        run = evidence["runs"][name]
        if (
            evidence["status"] != "PASS_FOR_DECLARED_SCOPE"
            or run["status"] != "PASS"
            or run["observer_enabled"] is not observed
            or run["case_group"] != "all"
            or len(run["cases"]) != 11
            or any(case["status"] != "PASS" for case in run["cases"])
        ):
            raise ValueError("incomplete native speed qualification")
    worker = verify_file(
        patches, "speed_candidate_worker.py", manifest["worker_sha256"]
    )
    if evidence["sources"][worker.name] != manifest["worker_sha256"]:
        raise ValueError("speed worker differs from lifecycle qualification")
    if args != ["--check-only"]:

        def option(name):
            if args.count(name) != 1:
                raise ValueError(f"qualified speed launch requires {name} exactly once")
            return args[args.index(name) + 1]

        if (
            option("--worker-cls") != WORKER
            or json.loads(option("--compilation-config")) != CAPTURE
        ):
            raise ValueError(
                "launcher does not select the qualified speed worker and graphs"
            )
    if any(
        key.startswith("QWEN_SPEED_") and ENVIRONMENT.get(key) != value
        for key, value in environ.items()
    ):
        raise ValueError("unqualified speed experiment is enabled")
    shutil.copyfile(worker, package / worker.name)
    environ.update(ENVIRONMENT)
    return {
        "worker": WORKER,
        "compilation_config": CAPTURE,
        "manifest_sha256": digest(patches / "qualified-speed-release.json"),
    }
