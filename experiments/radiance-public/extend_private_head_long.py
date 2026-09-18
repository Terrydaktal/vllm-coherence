"""Top up a completed benchmark after uniformly excluding its first warmup pair.

The original evidence is immutable. Reuse its independent head replay and add
matched natural completions until every variant has >=60K measured output tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from benchmark_private_head import digest, prepare_environment, request, write
from benchmark_private_head_long import MODES, complete, plan_pair, tokenize, totals


def exclude_initial_pair(records):
    excluded = [r for r in records if r["pair"] == 0]
    if len(excluded) != len(MODES) or {r["mode"] for r in excluded} != set(MODES):
        raise ValueError("warmup exclusion requires the entire first pair")
    return [r for r in records if r["pair"] != 0], excluded


def trial_number(value):
    if value == "startup":
        return None
    if type(value) is not int or value < 0:
        raise ValueError("invalid measured trial identifier")
    return value


def next_trial_offset(calls):
    identifiers = [row["trial"] for row in calls]
    numeric = [number for value in identifiers if (number := trial_number(value)) is not None]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate head-call trial identifier")
    return max(numeric, default=-1) + 1


def shift_head_call(row, offset):
    number = trial_number(row["trial"])
    return {**row, "trial": f"extension-startup-{offset}" if number is None else number + offset}


def main(args):
    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease
    from qwen_r9700_lab.conformance_runtime import NativeServer

    os.umask(0o077)
    root, reports, previous = args.private_root, args.report_root, args.previous
    if not str(root).startswith("/dev/shm/qwen-private-head-") or root.stat().st_mode & 0o077:
        raise ValueError("private root must be private tmpfs")
    if json.loads((previous / "completed.json").read_text())["status"] != "MEASURED":
        raise ValueError("the original capture and replay must finish before extension")
    original = json.loads((previous / "generation.json").read_text())
    records, excluded = exclude_initial_pair(original)
    calls = json.loads((previous / "head-calls.json").read_text())
    offset = next_trial_offset(calls)
    pair = max(r["pair"] for r in original) + 1
    expected_fixtures = json.loads((previous / "fixtures.json").read_text())
    reports.mkdir(mode=0o700)
    for name in (
        "capture-generation.json",
        "capture-counts.json",
        "head-stage.json",
        "fixtures.json",
    ):
        write(reports / name, json.loads((previous / name).read_text()))
    method = json.loads((previous / "methodology.json").read_text())
    method.update(
        previous_report=str(previous),
        exclusion="The complete first four-variant comparison exposed first-use JIT compilation.",
        extension_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        extension_warmup="32 output tokens per variant, all four before timing.",
    )
    write(reports / "methodology.json", method)
    write(reports / "excluded-warmup-pair.json", excluded)
    write(reports / "generation.json", records)
    spec = json.loads(args.spec.read_text())
    with gpu_lease(reports / "gpu-lease"):
        prepare_environment(spec, root, reports)
        write(root / "control.json", {"mode": "full", "capture": False})
        with NativeServer(
            spec, root / "server", allow_gpu=True, head=True, observe=False
        ) as server:
            fixtures = [
                tokenize(server, json.loads(p.read_text()))
                for p in sorted(root.glob("payload-*.json"))
            ]
            if [digest(t) for t in fixtures] != [r["token_sha256"] for r in expected_fixtures]:
                raise ValueError("extension workload differs from original tokenized prompts")
            common = {"chat_id": digest(str(root)), "progress": reports / "current-request.json"}
            warm = [
                request(server, root, fixtures[0], mode, 0, limit=32, **common) for mode in MODES
            ]
            write(reports / "warmup.json", warm[0])
            write(reports / "all-warmups.json", warm)
            if not list(root.glob("hook-*.json")):
                raise ValueError("extension target hook absent")
            while not complete(records, 60000):
                fixture, seed, order = plan_pair(pair, len(fixtures))
                tokens = fixtures[fixture]
                limit = server.settings["config"]["max_model_len"] - len(tokens) - 1
                for mode in order:
                    result = request(server, root, tokens, mode, seed, limit=limit, **common)
                    result.update(fixture=fixture, pair=pair, trial=result["trial"] + offset)
                    records.append(result)
                    write(reports / "generation.json", records)
                    status = {"phase": "top_up", "completed": totals(records), "last": result}
                    write(reports / "checkpoint.json", status)
                    print(json.dumps(status), flush=True)
                pair += 1
            request(server, root, fixtures[0], "full", 0, limit=1, **common)
            for path in sorted(root.glob("trial-*.json")):
                row = json.loads(path.read_text())
                calls.append(shift_head_call(row, offset))
            write(reports / "head-calls.json", calls)
    write(reports / "completed.json", {"status": "MEASURED", "previous_report": str(previous)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    main(parser.parse_args())
