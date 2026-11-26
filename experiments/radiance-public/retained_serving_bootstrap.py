"""Apply the pinned release in this disposable container, then add its measured overlays."""

import os
import runpy
import shutil
import sys
from hashlib import sha256
from pathlib import Path

args = sys.argv[1:]
PACKAGE = Path("/opt/vllm/lib/python3.12/site-packages")
R = Path("/retained")
sys.path.insert(0, "/patches")
sys.argv = ["/patches/bootstrap_radiance_release.py", "--check-only"]
runpy.run_path(sys.argv[0], run_name="__main__")
S = Path("/bench/source/experiments/radiance-public")
for f in (
    "speed_matched_stage_worker.py",
    "matched_stage_profile_worker.py",
    "benchmark_head_candidate_depth.py",
    "benchmark_optimized_d7.py",
    "benchmark_d7_equivalence.py",
    "benchmark_current_confirmations.py",
    "prefill_divergence_probe.py",
    "response_end_probe.py",
):
    shutil.copy2(S / f, PACKAGE / f)
if "sampled_cache_probe.SampledCacheProbe" in args:
    for f in (
        "cache_target_boundary_probe.py",
        "isolated_d7_capture.py",
        "prefill_graph_observer.py",
        "sampled_cache_probe.py",
    ):
        shutil.copy2(S / f, PACKAGE / f)
shutil.copy2(
    S / "benchmark_d7_equivalence.py", PACKAGE / "speed_equivalence_adapter.py"
)
# /patches is the authentic installation. Restrict other injected modules to the sealed selection.
release_path = os.environ["PYTHONPATH"]
os.environ["PYTHONPATH"] = os.pathsep.join(
    [
        "/retained",
        "/retained/direct-build",
        str(PACKAGE),
        "/bench/source/src",
        release_path,
        "/patches",
    ]
)
sys.path.insert(0, "/retained")
from fourteen_retained import manifest

m = manifest()
for name, target in m.get("overlay", {}).items():
    dest = PACKAGE / target["path"]
    if not str(dest.resolve()).startswith(str(PACKAGE.resolve()) + "/"):
        raise ValueError("overlay destination outside private Python installation")
    actual = sha256(dest.read_bytes()).hexdigest() if dest.exists() else None
    if actual != target["preimage"]:
        raise ValueError(f"overlay source preimage mismatch for {name}: {actual}")
    shutil.copyfile(R / name, dest)
if os.environ.get("QWEN_HEAD_DEPTH_BENCH_ROOT"):
    Path(os.environ["QWEN_HEAD_DEPTH_BENCH_ROOT"]).mkdir(mode=0o700)
    (PACKAGE / "qwen_head_depth_benchmark.pth").write_text(
        "import os; os.getenv('QWEN_HEAD_DEPTH_BENCH_ROOT') and __import__('benchmark_head_candidate_depth').install()\n"
    )
os.execv("/opt/radiance_entrypoint.sh", ["/opt/radiance_entrypoint.sh", *args])
