"""Shared coding, chained-workload and matched-stage benchmark captures.

Run via benchmark_pi_coding_contexts.py --suite --help. Requires an isolated,
compiled MatchedStageWorker with the production numerical release loaded.
The output directory is private: continuation token IDs must not be published.
Only the reports in public/ are content-free. No server is started or restarted.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

import benchmark_matched_stage_timings as matched
import benchmark_pi_coding_contexts as coding
import benchmark_pi_coding_json_compaction as chain

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("warmup", "control_before", "profile", "control_after")
SELECTED_ARM = "control_after"  # Fixed before measuring, never selected by speed.
STAGES = (
    ("prose_code", chain.PROSE_CODE_PROMPT, True),
    ("json", chain.JSON_PROMPT, False),
    ("thinking", chain.THINKING_PROMPT, True),
    ("compaction", chain.COMPACTION_PROMPT, False),
)
SOURCE_FILES = (
    "benchmark_pi_suite.py", "benchmark_matched_stage_timings.py",
    "matched_stage_profile_worker.py", "benchmark_pi_coding_contexts.py",
    "benchmark_pi_coding_json_compaction.py", "benchmark_pi_task_workloads.py",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_private(path, value):
    """Atomic checkpoints, including sensitive continuation IDs, are always 0600."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def read(path):
    return json.loads(path.read_text())


def stable_worker_identity(receipt):
    metadata = receipt["metadata"]
    matched.require_compiled(metadata)
    if "PIECEWISE" not in metadata["graph_mode"]:
        raise ValueError("the compiled PIECEWISE production path is required")
    # Library mappings and dispatch counters grow on first use. They remain in
    # worker.json but are not configuration identities. A new worker UUID always
    # invalidates reuse, even if its release/configuration happens to match.
    return {"instance": receipt["instance"], "model": receipt["model"],
            "speed_candidate": metadata.get("speed_candidate"),
            "target_head_environment": receipt["target_head_environment"],
            "configuration": {key: metadata[key] for key in (
                "enforce_eager", "compilation_mode", "backend", "graph_mode",
                "capture_sizes", "async_scheduling", "effective_capacity",
                "diagnostic_sources", "startup_compatibility")},
            "repair_bundle": metadata["repair"]["bundle"],
            "performance_manifest": metadata["performance"]["manifest"],
            "gemm_binary_sha256": metadata["performance"]["gemm_dispatch"]["binary_sha256"],
            "runtime": {key: metadata["runtime"].get(key) for key in (
                "python", "kernel", "machine", "packages", "flags", "compiler_settings")}}


def build_contract(args, worker, fixtures, rendered):
    settings = {name: value for name, value in vars(args).items() if name in {
        "abi", "temperature", "top_p", "top_k", "seed", "model_context_tokens",
        "warmup_rounds", "chunk_rounds", "profile_rounds", "head",
        "coding_min_tokens", "coding_max_tokens", "prose_code_min_tokens",
        "prose_code_max_tokens", "json_min_tokens", "json_max_tokens",
        "thinking_min_tokens", "thinking_max_tokens", "compaction_max_tokens"}}
    return {
        "schema": "urn:coherence:shared-benchmark-contract:v1",
        "worker": worker, "settings": settings, "model": chain.MODEL,
        "runtime_manifest_sha256": file_hash(args.runtime_manifest),
        "tokenizer_sha256": file_hash(args.tokenizer_json),
        "fixtures": {name: {"file_sha256": sha, "prefix_sha256": coding._prefix_digest(ids)}
                     for name, (ids, sha) in fixtures.items()},
        "rendered_turn_sha256": {stage: digest(ids) for stage, ids in rendered.items()},
        "measurement_sources": {name: file_hash(Path(__file__).with_name(name)) for name in SOURCE_FILES},
        "control_selection": SELECTED_ARM,
        "cache_policy": "Unique context identity; warmup, control_before, profile, control_after; then continue 60K output.",
        "natural_stop": True,
    }


def receipt(root, path):
    return {"path": str(path.relative_to(root)), "sha256": file_hash(path)}


def verify_file(root, item):
    path = root / item["path"]
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or file_hash(path) != item["sha256"]:
        raise ValueError("checkpoint artifact missing or changed; preserve it and use a new output directory")
    return path


def save_ids(root, capture_id, ids):
    if not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("invalid continuation token IDs")
    path = root / "continuations" / f"{capture_id}.json"
    write_private(path, ids)
    return receipt(root, path) | {"tokens": len(ids), "token_ids_sha256": digest(ids)}


def load_ids(root, item, result):
    ids = read(verify_file(root, item))
    if (not isinstance(ids, list) or any(type(t) is not int or t < 0 for t in ids)
            or len(ids) != item["tokens"] or len(ids) != result["generated_tokens"]
            or digest(ids) != item["token_ids_sha256"]
            or digest(ids) != result["output_token_ids_sha256"]):
        raise ValueError("continuation does not match the selected capture")
    return ids


def failures(result, label, *, rounds=False):
    errors = []
    if result.get("minimum_output_met") is False:
        errors.append(f"{label}: short natural completion")
    if result.get("checkpoint_validation", {}).get("passed") is False:
        errors.append(f"{label}: checkpoint contract did not validate")
    if rounds and result["round_capture"]["status"] != "captured":
        errors.append(f"{label}: incomplete round capture")
    return errors


def publish_views(root, state):
    """Different tables are views of the same observations, not extra trials."""
    contract = state["contract"]
    common = {
        "suite_capture_id": state["id"], "contract_sha256": digest(contract),
        "started_at": state["started_at"], "model": chain.MODEL,
        "sampling": {key: contract["settings"][key] for key in ("temperature", "top_p", "top_k", "seed")},
        "tokenizer_sha256": contract["tokenizer_sha256"],
        "runtime_manifest_sha256": contract["runtime_manifest_sha256"],
        "capture_selection": SELECTED_ARM,
        "privacy": "Numeric measurements and hashes only; continuation IDs are separate private files.",
    }
    contexts = common | {"schema": "urn:coherence:pi-coding-contexts:v2",
        "contexts_requested": list(coding.CONTEXTS), "contexts": {}, "fixture_sha256": {},
        "prefix_token_ids_sha256": {}, "validation_failures": [],
        "natural_stop_required": True, "ignore_eos": False,
        "coding_min_tokens": contract["settings"]["coding_min_tokens"],
        "coding_max_tokens": contract["settings"]["coding_max_tokens"],
        "round_telemetry": {"retains_every_event": True}}
    for context, group in state["contexts"].items():
        result = group["result"]
        contexts["contexts"][context] = {"context": context,
            "prefix_tokens": coding.CONTEXT_TOKEN_COUNTS[context], **result}
        fixture = contract["fixtures"][context]
        contexts["fixture_sha256"][context] = fixture["file_sha256"]
        contexts["prefix_token_ids_sha256"][context] = fixture["prefix_sha256"]
        contexts["validation_failures"].extend(failures(result, context, rounds=True))
    chain_report = common | {"schema": "urn:coherence:pi-coding-json-compaction:v1",
        "fixture_sha256": contract["fixtures"]["60K"]["file_sha256"], "prefix_tokens": 60000,
        "stages": [], "validation_failures": []}
    if "60K" in state["contexts"]:
        chain_report["stages"].append(state["contexts"]["60K"]["result"])
    chain_report["stages"].extend(item["result"] for item in state["chain"])
    for result in chain_report["stages"]:
        chain_report["validation_failures"].extend(failures(result, result["stage"]))
    for report, done, name in (
        (contexts, len(contexts["contexts"]) == 3, "pi-coding-contexts.json"),
        (chain_report, len(chain_report["stages"]) == 5, "pi-coding-json-compaction.json"),
    ):
        report["status"] = ("complete_with_validation_failure" if report["validation_failures"]
                            else "complete") if done else "running"
        write_private(root / "public" / name, report)


def check_worker(rpc, state):
    observed = rpc("qwen_timing_status")
    if observed["instance"] != state["contract"]["worker"]["instance"] or observed["arm_active"]:
        raise RuntimeError("worker restarted or timing arm remains active; capture cannot be reused")


def capture_context(args, opener, tokenizer, rpc, state, context, prefix, suffix):
    root = args.output
    directory = root / "runs" / context
    if directory.exists():
        # Never mix a pre-interruption control with a post-interruption trace.
        abandoned = root / "abandoned"
        abandoned.mkdir(exist_ok=True, mode=0o700)
        directory.rename(abandoned / f"{context}-{uuid.uuid4().hex}")
    directory.mkdir(parents=True, mode=0o700)
    attempt = uuid.uuid4().hex
    identity = {"id": digest([state["id"], context, attempt]),
                "generation": digest([attempt, "initial"]),
                "title": "Shared Pi benchmark", "cwd": "/qualification/shared-suite", "session_file": ""}
    section = {"fixture_sha256": state["contract"]["fixtures"][context]["file_sha256"],
               "prefix_sha256": coding._prefix_digest(prefix),
               "prompt_sha256": coding._prefix_digest(prefix + suffix), "arms": {}}
    report = {"schema": "urn:coherence:matched-stage-serving:v1", "status": "running",
              "sampling": {key: getattr(args, key) for key in ("temperature", "top_p", "top_k", "seed")},
              "natural_stop": True, "contexts": {context: section}}
    write_private(directory / "report.json", report)
    continuation = None
    for arm in ARMS:
        print(json.dumps({"context": context, "arm": arm}), flush=True)
        result, ids, _ = matched.capture_arm(
            opener=opener, args=args, rpc=rpc, tokenizer=tokenizer, identity=identity,
            prompt=prefix + suffix, suffix=suffix, arm=arm, root=directory / f"{context}-{arm}")
        check_worker(rpc, state)
        result.update(capture_id=digest([attempt, arm]), measurement_mode=arm,
                      output_token_ids_sha256=digest(ids))
        section["arms"][arm] = result
        if context == "60K" and arm == SELECTED_ARM:
            continuation = save_ids(root, result["capture_id"], ids)
        write_private(directory / "report.json", report)
    section["identical_outputs"] = len({r["output_token_ids_sha256"] for r in section["arms"].values()}) == 1
    report["status"] = "complete"
    write_private(directory / "report.json", report)
    # Hash traces once, outside all timed requests. Detect missing/changed evidence
    # on resume rather than running the GPU again to silently replace it.
    artifacts = [receipt(root, path) for path in sorted(directory.rglob("*")) if path.is_file()]
    return {"identity": identity, "result": section["arms"][SELECTED_ARM],
            "continuation": continuation, "artifacts": artifacts}


def continue_chain(args, opener, tokenizer, rpc, state, prefix, rendered):
    group = state["contexts"]["60K"]
    sequence = prefix + chain._turn_suffix(tokenizer, prefix, rendered["coding"], first=True)
    sequence += load_ids(args.output, group["continuation"], group["result"])
    for index, (stage, _, thinking) in enumerate(STAGES):
        suffix = chain._turn_suffix(tokenizer, sequence, rendered[stage], first=False)
        prompt = sequence + suffix
        if len(prompt) + getattr(args, f"{stage}_max_tokens") > args.model_context_tokens:
            raise ValueError(f"{stage}: prompt plus output ceiling exceeds context capacity")
        if index < len(state["chain"]):
            saved = state["chain"][index]
            if saved["result"]["stage"] != stage or saved["prompt_sha256"] != digest(prompt):
                raise ValueError("saved chain does not continue the selected coding capture")
            ids = load_ids(args.output, saved["continuation"], saved["result"])
        else:
            print(json.dumps({"context": "60K", "stage": stage}), flush=True)
            result, ids, completion = chain._request(
                opener=opener, args=args, identity=group["identity"], tokenizer=tokenizer,
                prompt_tokens=prompt, suffix_tokens=suffix, stage=stage,
                max_tokens=getattr(args, f"{stage}_max_tokens"), thinking=thinking)
            del completion
            check_worker(rpc, state)
            result.update(capture_id=digest([state["id"], group["result"]["capture_id"], stage]),
                          measurement_mode="unprofiled_chain", output_token_ids_sha256=digest(ids))
            state["chain"].append({"result": result, "prompt_sha256": digest(prompt),
                "continuation": save_ids(args.output, result["capture_id"], ids)})
            write_private(args.output / "checkpoint.json", state)
            publish_views(args.output, state)
        sequence = prompt + ids


def analyze_stages(root, head):
    command = [sys.executable, str(ROOT / "tools/analyze_matched_stage_timings.py"),
               "--root", str(root / "runs"), "--head", head, "--output", str(root / "analysis.json")]
    subprocess.run(command, check=True)


def execute(args, opener, tokenizer, rpc):
    root = args.output
    worker_receipt = rpc("qwen_timing_identity")
    if worker_receipt["arm_active"]:
        raise RuntimeError("worker still has a timing arm active; finish that isolated capture before resuming")
    worker = stable_worker_identity(worker_receipt)
    flags = worker["target_head_environment"]
    # Never silently benchmark the full BF16 target head or a different shortlist.
    if (str(flags.get("RADIANCE_VERIFY_HEAD", "0")) != "1"
            or str(flags.get("RADIANCE_VERIFY_HEAD_GLOBAL_TOPK", "0")) != args.head.removeprefix("global")):
        raise ValueError("live target-head flags do not match --head; refusing a different numerical profile")
    fixtures = {context: coding._load_prefix(
        {"0K": None, "60K": args.fixture_60k, "200K": args.fixture_200k}[context],
        coding.CONTEXT_TOKEN_COUNTS[context]) for context in coding.CONTEXTS}
    rendered = {stage: chain._render_user_turn(opener, args.base_url, prompt,
                thinking=thinking, timeout=args.request_timeout)
                for stage, prompt, thinking in (("coding", chain.CODING_PROMPT, False), *STAGES)}
    contract = build_contract(args, worker, fixtures, rendered)
    checkpoint = root / "checkpoint.json"
    if checkpoint.exists():
        state = read(checkpoint)
        if state["contract"] != contract:
            raise ValueError("benchmark identity changed (release/worker/fixture/settings/source); use a new output directory")
        for context, group in state["contexts"].items():
            for item in group["artifacts"]:
                verify_file(root, item)
            if group["continuation"]:
                load_ids(root, group["continuation"], group["result"])
            captured = read(root / "runs" / context / "report.json")["contexts"][context]["arms"][SELECTED_ARM]
            if captured != group["result"]:
                raise ValueError("checkpoint result differs from its captured control")
        for saved in state["chain"]:
            load_ids(root, saved["continuation"], saved["result"])
    else:
        state = {"schema": "urn:coherence:shared-benchmark:v1", "id": uuid.uuid4().hex,
                 "contract": contract, "started_at": time.time(), "contexts": {}, "chain": []}
        write_private(checkpoint, state)
        for name in SOURCE_FILES:
            (root / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    for context in coding.CONTEXTS:
        prefix, _ = fixtures[context]
        suffix = chain._turn_suffix(tokenizer, prefix, rendered["coding"], first=True)
        if len(prefix) + len(suffix) + args.coding_max_tokens > args.model_context_tokens:
            raise ValueError(f"{context}: prompt plus output ceiling exceeds context capacity")
        if context not in state["contexts"]:
            state["contexts"][context] = capture_context(
                args, opener, tokenizer, rpc, state, context, prefix, suffix)
            write_private(checkpoint, state)
        publish_views(root, state)
        if context == "60K":
            continue_chain(args, opener, tokenizer, rpc, state, prefix, rendered)
    # Post-processing never submits inference. It uses exactly matched M8 indices
    # and GPU interval unions, not the whole-request coding mean for row 26.
    analyze_stages(root, args.head)
    state["validation_failures"] = []
    for name in ("pi-coding-contexts.json", "pi-coding-json-compaction.json"):
        state["validation_failures"].extend(read(root / "public" / name)["validation_failures"])
    state["status"] = "complete_with_validation_failure" if state["validation_failures"] else "complete"
    write_private(checkpoint, state)
    return state


def run(args):
    from tokenizers import Tokenizer
    args.output = args.output.expanduser().resolve()
    if args.output.is_relative_to(ROOT) or any((p / ".git").exists() for p in (args.output, *args.output.parents)):
        raise ValueError("--output must be a private directory outside Git repositories (continuation IDs are sensitive)")
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.output.stat().st_mode & 0o077:
        raise ValueError("--output must have private permissions (0700)")
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
        return execute(args, opener, tokenizer,
                       lambda method, *values: matched.rpc_request(opener, args.base_url, method, *values))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "fixture-60k", "fixture-200k", "tokenizer-json", "runtime-manifest"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--abi", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8081")
    p.add_argument("--head", choices=("global256", "global512"), default="global512")
    p.add_argument("--round-log", type=Path, default=Path("/dev/shm/qwen-stage-timing-rounds.jsonl"))
    for name, default in (("warmup-rounds", 64), ("chunk-rounds", 128), ("profile-rounds", 1152),
                          ("model-context-tokens", 253792), ("coding-min-tokens", chain.CODING_MIN_TOKENS),
                          ("prose-code-min-tokens", 500), ("json-min-tokens", 1000),
                          ("thinking-min-tokens", 1000), ("coding-max-tokens", 10000),
                          ("prose-code-max-tokens", 8192), ("json-max-tokens", 4096),
                          ("thinking-max-tokens", 8192), ("compaction-max-tokens", 8192),
                          ("top-k", 40), ("seed", 0)):
        p.add_argument(f"--{name}", type=int, default=default)
    for name, default in (("temperature", 1.), ("top-p", .95), ("request-timeout", 1800.),
                          ("metrics-timeout", 30.), ("round-log-settle-timeout", 10.)):
        p.add_argument(f"--{name}", type=float, default=default)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if (args.top_k < 1 or not 0 < args.top_p <= 1 or args.temperature < 0
            or args.warmup_rounds < 0 or args.chunk_rounds < 1 or args.profile_rounds < 1
            or args.compaction_max_tokens < 1 or args.model_context_tokens < 1
            or args.request_timeout <= 0 or args.metrics_timeout <= 0
            or args.round_log_settle_timeout < 0
            or any(getattr(args, f"{stage}_min_tokens") < 1 or
                   getattr(args, f"{stage}_max_tokens") < getattr(args, f"{stage}_min_tokens")
                   for stage in ("coding", "prose_code", "json", "thinking"))):
        p.error("invalid sampling, timing or token limits")
    run(args)


if __name__ == "__main__":
    main()
