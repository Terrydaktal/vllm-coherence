"""Replay the same private requests on the preserved 0.9.3 deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import statistics
import subprocess
from pathlib import Path

import benchmark_kernel_matrix as matrix
import benchmark_runtime_flags as bench

OLD_FLAGS = {
    "OLD": {
        "RADIANCE_VERIFY_HEAD": "0",
        "RADIANCE_DYNAMIC_WIDTH": "0",
        "GPU_MAX_HW_QUEUES": "1",
        "HSA_ENABLE_MWAITX": "1",
    }
}
# This is the preserved data ABI, independent of the newer runtime's data ABI.
OLD_ABI = "74aef30706ffab186ee2fb89d3827c9895496918bdfb3d61188196daeced5b94"
OLD_R4D = "/home/lewis/.cache/radiance-libr4d/b9e42ab-rx5"
OLD_R4D_SHA256 = "616ffdcb84469c98901977747ac3ee663651c97a2048dd622d1a3735eb6add51"


def legacy_launcher(original, root):
    replacements = {
        "set -euo pipefail\n": (
            "set -euo pipefail\n"
            f"[[ $(sha256sum {OLD_R4D}/r4d.so | awk '{{print $1}}') == {OLD_R4D_SHA256} ]] || "
            "{ printf 'Preserved libr4d hash mismatch\\n' >&2; exit 64; }\n"
        ),
        "readonly container_name=qwen38-27b-uncensored-mxfp4-public-snapshot-candidate": (
            'readonly container_name="${QWEN_QUALIFICATION_CONTAINER:?}"'
        ),
        '--arg root_dir "/cache/snapshots/${data_abi}/data"': (
            f'--arg root_dir "/cache/benchmarks/{root.name}/data"'
        ),
        "\tPORT=8080 \\\n": '\tPORT="${QWEN_QUALIFICATION_PORT:-18080}" \\\n',
        "\tPYTHONHASHSEED=0 \\\n": f'\tPATH={shlex.quote(str(root / "bin"))}:"$PATH" \\\n'
        f"\tRUNTIME=podman AUTO_R4D=0 R4D_SO={shlex.quote(OLD_R4D)} \\\n"
        "\tPYTHONHASHSEED=0 \\\n",
    }
    for old, new in replacements.items():
        if original.count(old) != 1:
            raise ValueError("old launcher isolation anchor changed")
        original = original.replace(old, new)
    return original


def runtime_shim(root, private_directory):
    if not re.fullmatch(r"/dev/shm/qwen-private-replay-[a-z0-9_]+", private_directory):
        raise ValueError("expected a private RAM fixture directory")
    # This shim is on PATH only inside the experimental launcher's environment.
    # The untouched old startup script therefore keeps its normal podman code path.
    return f"""#!/usr/bin/env bash
set -euo pipefail
if [[ ${{1:-}} == run ]]; then
    shift
    owned=0
    previous=
    for argument in "$@"; do
        if [[ $previous == --name && $argument == {shlex.quote(root.name)} ]]; then owned=1; fi
        previous=$argument
    done
    ((owned == 1)) || {{ printf 'Refusing an unowned benchmark container\\n' >&2; exit 64; }}
    exec /usr/bin/podman run \\
        -v {shlex.quote(str(root))}:/benchmark:rw \\
        -v {shlex.quote(private_directory)}:/private-fixtures:ro \\
        -e QWEN_BENCHMARK_PRIVATE_FIXTURES=/private-fixtures \\
        -e QWEN_BENCHMARK_LEGACY_METRICS=1 \\
        -e QWEN_RADIANCE_CACHE_ABI={OLD_ABI} "$@"
fi
exec /usr/bin/podman "$@"
"""


def prepare(root, after):
    manifest = bench.read_json(root / "manifest.json")
    original = (root / "old-production-launcher.sh").read_text()
    manifest.update(
        after_run=str(after),
        worker_script=Path(__file__).name,
        variants=OLD_FLAGS,
        continue_on_variant_failure=False,
        baseline_failures_are_fatal=True,
        allow_preceding_variant_failures=True,
        sampling=matrix.SAMPLING,
        tool_cases=matrix.TOOL_CASES,
        old_launcher_sha256=hashlib.sha256(original.encode()).hexdigest(),
        old_r4d_directory=OLD_R4D,
        old_r4d_sha256=OLD_R4D_SHA256,
        old_data_abi=OLD_ABI,
        workload="Same private token fixtures as the new-stack kernel matrix",
        timing_method="Native generation counters on both stacks; old first-token wait "
        "includes queue, cache, prefill and the first generation step",
    )
    bench.write_json(root / "manifest.json", manifest)
    (root / "bin").mkdir(exist_ok=True)
    launcher = root / "launch-OLD.sh"
    launcher.write_text(legacy_launcher(original, root))
    shim = root / "bin/podman"
    shim.write_text(runtime_shim(root, manifest["private_fixture_directory"]))
    shim.chmod(0o700)
    for script in (launcher, shim):
        subprocess.run(["shfmt", "-w", str(script)], check=True)
        subprocess.run(["shellcheck", str(script)], check=True)
        subprocess.run(["shfmt", "-d", str(script)], check=True)
    bench.write_json(
        root / "launcher-validation.json",
        {
            "OLD": {
                "sha256": hashlib.sha256(launcher.read_bytes()).hexdigest(),
                "shellcheck": True,
                "shfmt": True,
            }
        },
    )
    bench.write_json(
        root / "runtime-shim-validation.json",
        {"sha256": hashlib.sha256(shim.read_bytes()).hexdigest()},
    )


def comparison(old_root, new_root):
    """Compare the stored new baseline and old deployment using identical counter rules."""
    old_manifest = bench.read_json(old_root / "manifest.json")
    new_manifest = bench.read_json(new_root / "manifest.json")
    fixtures = old_manifest["private_replay_metadata"]
    if fixtures != new_manifest["private_replay_metadata"]:
        raise ValueError("old and new private fixtures differ")
    rows = []
    for root, variant in ((new_root, "K000"), (old_root, "OLD")):
        status = bench.read_json(root / "status.json")
        if status.get("stage") not in ("complete", "complete_with_failures") or not status.get(
            "production_restored"
        ):
            raise ValueError(f"{variant} has not completed and restored production")
        config = bench.read_json(root / f"{variant}-configuration.json")
        expected_flags = OLD_FLAGS["OLD"] if variant == "OLD" else matrix.variants()["K000"]
        if config.get("flags") != expected_flags:
            raise ValueError(f"mismatched {variant} deployment configuration")
        checks = bench.read_json(root / f"{variant}-correctness.json")
        if checks.get("tool_cases") != matrix.TOOL_CASES:
            raise ValueError(f"incomplete {variant} correctness checks")
        for fixture in fixtures:
            context = fixture["actual_context"]
            records = [
                bench.read_json(root / f"{variant}-{context}-measured-{i}.json")
                for i in range(1, 4)
            ]
            for i, record in enumerate(records, 1):
                if (
                    record.get("variant") != variant
                    or record.get("context") != context
                    or record.get("trial") != f"measured-{i}"
                    or record.get("output_tokens") != 1024
                    or record.get("prompt_sha256") != fixture["sha256"]
                    or record.get("sampling") != matrix.SAMPLING
                    or record.get("usage", {}).get("prompt_tokens") != context
                ):
                    raise ValueError(f"mismatched old/new request {variant}/{context}/{i}")
            windows = [
                bench.steady_summary(bench.counter_generation_window(r["samples"])) for r in records
            ]
            rows.append(
                {
                    "variant": variant,
                    "image": config["image"],
                    "context": context,
                    "trials": len(records),
                    "median": {
                        key: statistics.median(w[key] for w in windows) for key in windows[0]
                    },
                    "range": {
                        key: [min(w[key] for w in windows), max(w[key] for w in windows)]
                        for key in ("round_ms", "tokens_per_second")
                    },
                    "output_hashes": sorted({r["output_sha256"] for r in records}),
                    "tool_passed": checks["tool_passed"],
                    "tool_cases": checks["tool_cases"],
                    "correctness": next(
                        c for c in checks["contexts"] if c["actual_context"] == context
                    ),
                    "allocator_errors": max(
                        r["allocator"]["allocation_retries"]
                        + r["allocator"]["out_of_memory_events"]
                        for r in records
                    ),
                }
            )
    if len({row["image"] for row in rows}) != 2:
        raise ValueError("old/new comparison requires two distinct pinned images")
    for row in rows:
        old = next(r for r in rows if r["variant"] == "OLD" and r["context"] == row["context"])
        row["token_rate_change_percent_vs_old"] = (
            row["median"]["tokens_per_second"] / old["median"]["tokens_per_second"] - 1
        ) * 100
        row["round_time_change_percent_vs_old"] = (
            row["median"]["round_ms"] / old["median"]["round_ms"] - 1
        ) * 100
    return {
        "fixtures": fixtures,
        "sampling": matrix.SAMPLING,
        "rows": rows,
        "measurement": "Both stacks use the same native-counter generation window and three "
        "warmed 1,024-token runs per historical private request. The existing new baseline was "
        "reanalysed without rerunning it, as requested.",
        "scope": "Original 0.9.3 deployment versus current 1.0.16 production settings, including "
        "their original kernel and verify-head choices. This compares deployed stacks rather "
        "than attributing any difference to an individual library. Private outputs were not read.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "queue", "run", "worker", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--new-root", type=Path)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080")
    parser.add_argument("--variant", choices=("OLD",), default="OLD")
    args = parser.parse_args()
    os.umask(0o077)
    bench.FLAGS = OLD_FLAGS
    if args.mode == "summarize":
        result = comparison(args.root, args.new_root)
        bench.write_json(args.root / "comparison.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.mode == "prepare":
        prepare(args.root, args.after)
        return 0
    if args.mode == "worker":
        return matrix.worker(args)
    if args.mode == "queue":
        try:
            return matrix.queue(args)
        except Exception as error:
            failure = {"stage": "queue_failed", "type": type(error).__name__, "message": str(error)}
            bench.write_json(args.root / "queue-failure.json", failure)
            bench.write_json(args.root / "status.json", failure)
            bench.emit(event="old_stack_queue_failed", **failure)
            return 1
    return bench.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
