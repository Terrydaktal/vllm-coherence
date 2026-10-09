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
import re
import stat
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
DIAGNOSTIC_MAX_FILES = 32
DIAGNOSTIC_MAX_BYTES = 128 * 1024 * 1024
DIAGNOSTIC_FILE_MAX_BYTES = 16 * 1024 * 1024
DIAGNOSTIC_ATTEMPTS = 3


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


_IDENTITY_UNSET = object()


def latest_round_recorder_identity(round_log):
    """Ignore earlier lifetimes and incomplete final records in a bounded tail."""
    round_log = Path(round_log)
    try:
        descriptor = os.open(round_log, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if stat.S_ISREG(info.st_mode):
                stream.seek(max(0, info.st_size - 65536))
                for line in reversed(stream.read(65536).splitlines()):
                    try:
                        row = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if not isinstance(row, dict):
                        continue
                    trace, pid = row.get("cache_trace_id"), row.get("pid")
                    if (isinstance(trace, str) and trace and isinstance(pid, int)
                            and not isinstance(pid, bool) and pid > 0):
                        return trace, pid
    except OSError:
        pass
    return None


def diagnostic_cache_prefix(round_log, explicit=None, *, current_identity=_IDENTITY_UNSET):
    """Resolve recorder identity without assuming it follows the round writer."""
    round_log = Path(round_log)
    if explicit is not None:
        return Path(explicit), "explicit"
    suffix = "-rounds.jsonl"
    name = round_log.name[:-len(suffix)] if round_log.name.endswith(suffix) else round_log.stem
    expected = round_log.with_name(name)
    # Worker initialization can configure the recorder before the scheduler
    # chooses a different round-log path. Its current health identity is authoritative.
    if current_identity is _IDENTITY_UNSET:
        current_identity = latest_round_recorder_identity(round_log)
    matches = []
    for health in round_log.parent.glob("*-cache-jobs-health.json"):
        try:
            descriptor = os.open(health, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > 65536:
                    continue
                row = json.loads(stream.read(65536))
            if (current_identity is not None and isinstance(row, dict)
                    and (row.get("trace_id"), row.get("pid")) == current_identity):
                matches.append(health.with_name(health.name.removesuffix("-cache-jobs-health.json")))
        except (OSError, ValueError, UnicodeDecodeError):
            continue
    if len(matches) == 1:
        return matches[0], "matching_round_trace_health"
    if expected.with_name(name + "-cache-jobs-health.json").is_file():
        return expected, "round_path"
    return expected, "round_path_unconfirmed"


def retain_diagnostics(round_log, directory, cache_telemetry_prefix=None):
    """Copy existing numeric feeds before their container tmpfs disappears.

    Copies happen between requests, with no recorder/GPU calls. Append-only
    feeds are copied to their initial size; rotations trigger bounded retries.
    The manifest describes a bounded snapshot, never complete lifetime history.
    """
    round_log, directory = Path(round_log), Path(directory)
    directory.mkdir(parents=True, mode=0o700)
    current_identity = latest_round_recorder_identity(round_log)
    cache_prefix, prefix_source = diagnostic_cache_prefix(
        round_log, cache_telemetry_prefix, current_identity=current_identity)
    prefix = cache_prefix.name
    cache_name = prefix + "-cache-jobs.jsonl"
    health_name = prefix + "-cache-jobs-health.json"
    pattern = re.compile(re.escape(prefix) + r"(?:-api-[0-9]+)?-cache-jobs(?:\.jsonl(?:\.1)?|-health\.json)$")

    def inventory():
        paths = {round_log, round_log.with_name(round_log.name + ".1")}
        paths.update(path for path in cache_prefix.parent.iterdir() if pattern.fullmatch(path.name))
        result = {}
        for path in sorted(paths):
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            result[path.name] = (path, info.st_dev, info.st_ino, info.st_size, info.st_mode)
        return result

    manifest = {"schema": "urn:coherence:benchmark-diagnostics:v1", "started_at_ns": time.time_ns(),
                "source_directory": str(round_log.parent), "round_log": round_log.name,
                "cache_telemetry_prefix": str(cache_prefix),
                "cache_telemetry_prefix_source": prefix_source,
                "round_recorder_identity": ({"trace_id": current_identity[0], "pid": current_identity[1]}
                                            if current_identity is not None else None),
                "scope": "Bounded current/previous numeric feeds; not permanent lifetime history.",
                "limits": {"files": DIAGNOSTIC_MAX_FILES, "bytes": DIAGNOSTIC_MAX_BYTES,
                           "file_bytes": DIAGNOSTIC_FILE_MAX_BYTES, "attempts": DIAGNOSTIC_ATTEMPTS},
                "rotation_retries": 0, "files": [], "issues": []}
    for attempt in range(DIAGNOSTIC_ATTEMPTS):
        copied, issues, total = [], [], 0
        try:
            before = inventory()
        except OSError as error:
            issues.append({"code": "inventory_unavailable", "errno": error.errno})
            before = {}
        for name in (round_log.name, cache_name, health_name):
            if name not in before:
                issues.append({"code": "required_feed_missing", "file": name})
        if len(before) > DIAGNOSTIC_MAX_FILES:
            issues.append({"code": "file_count_limit", "available": len(before)})
        for name, (source, device, inode, size, mode) in list(before.items())[:DIAGNOSTIC_MAX_FILES]:
            if not stat.S_ISREG(mode):
                issues.append({"code": "non_regular_file", "file": name})
                continue
            if size > DIAGNOSTIC_FILE_MAX_BYTES or total + size > DIAGNOSTIC_MAX_BYTES:
                issues.append({"code": "byte_limit", "file": name, "source_bytes": size})
                continue
            try:
                fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
                with os.fdopen(fd, "rb") as stream:
                    opened = os.fstat(stream.fileno())
                    if (opened.st_dev, opened.st_ino) != (device, inode):
                        raise FileNotFoundError("diagnostic feed rotated before open")
                    data = stream.read(size)
                    if len(data) != size or os.fstat(stream.fileno()).st_size < size:
                        raise FileNotFoundError("diagnostic feed truncated during copy")
                incomplete_tail = 0
                if name.endswith((".jsonl", ".jsonl.1")) and data and not data.endswith(b"\n"):
                    end = data.rfind(b"\n") + 1
                    incomplete_tail = len(data) - end
                    issues.append({"code": "incomplete_record", "file": name, "tail_bytes": incomplete_tail})
                records = data.splitlines() if name.endswith((".jsonl", ".jsonl.1")) else [data]
                decoded, parse_errors = [], 0
                for row in records:
                    try:
                        value = json.loads(row)
                        if not isinstance(value, dict):
                            raise TypeError("diagnostic record is not an object")
                        decoded.append(value)
                    except (ValueError, TypeError):
                        parse_errors += 1
                if parse_errors:
                    issues.append({"code": "invalid_records", "file": name, "count": parse_errors})
                health = None
                if name.endswith("-health.json") and decoded:
                    health = {key: decoded[0].get(key) for key in (
                        "trace_id", "pid", "started_ns", "updated_at_ms", "written", "dropped",
                        "write_errors", "context_drops", "main_thread_samples", "round_sample_errors",
                        "worker_first_work_hooks", "generation_round_telemetry", "gpu_rounds")}
                    if any(isinstance(health.get(key), (int, float)) and health[key] > 0
                           for key in ("dropped", "write_errors", "context_drops", "round_sample_errors")):
                        issues.append({"code": "recorder_reports_loss", "file": name})
                    gpu = health.get("gpu_rounds")
                    if isinstance(gpu, dict) and any(gpu.get(key, 0) for key in ("dropped", "errors", "quarantined")):
                        issues.append({"code": "gpu_recorder_reports_loss", "file": name})
                    if name == health_name and (health.get("worker_first_work_hooks") is not True
                                               or health.get("generation_round_telemetry") is not True):
                        issues.append({"code": "generation_diagnostics_unconfirmed", "file": name})
                with tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
                    temporary = Path(output.name)
                    output.write(data)
                temporary.replace(directory / name)
                total += len(data)
                copied.append({"file": name, "bytes": len(data), "source_bytes": size,
                               "sha256": hashlib.sha256(data).hexdigest(), "records": len(records),
                               "validated_records": len(decoded), "parse_errors": parse_errors,
                               "incomplete_tail_bytes": incomplete_tail,
                               "source_device": device, "source_inode": inode,
                               **({"health": health} if health is not None else {})})
            except (OSError, ValueError, IndexError) as error:
                issues.append({"code": "copy_failed", "file": name, "errno": getattr(error, "errno", None)})
        try:
            after = inventory()
            rotated = ({name: value[1:3] for name, value in before.items()}
                       != {name: value[1:3] for name, value in after.items()})
        except OSError:
            rotated = True
        manifest.update(files=copied, issues=issues, attempts=attempt + 1)
        if not rotated:
            break
        manifest["rotation_retries"] += 1
        if attempt == DIAGNOSTIC_ATTEMPTS - 1:
            manifest["issues"].append({"code": "rotation_did_not_settle"})
    # File retention and lifecycle agreement are different requirements. Even
    # an explicit path must not qualify another process's otherwise healthy feed.
    selected_health = next((item.get("health") for item in manifest["files"]
                            if item["file"] == health_name), None)
    if current_identity is not None:
        observed_identity = ((selected_health.get("trace_id"), selected_health.get("pid"))
                             if selected_health is not None else None)
        verified = observed_identity == current_identity
        manifest["recorder_identity_verified"] = verified
        if not verified:
            manifest["issues"].append({"code": "recorder_identity_missing" if observed_identity is None
                                      else "recorder_identity_mismatch", "file": health_name})
    else:
        manifest["recorder_identity_verified"] = None
    manifest.update(status="retained_with_gaps" if manifest["issues"] else "retained",
                    finished_at_ns=time.time_ns())
    write_private(directory / "manifest.json", manifest)
    return {"status": manifest["status"], "manifest": str(directory / "manifest.json"),
            "file_count": len(manifest["files"]), "issues": manifest["issues"]}


def retain_request_diagnostics(args, directory):
    """Optional archive failures remain visible and never replace request errors."""
    try:
        result = retain_diagnostics(args.round_log, directory, getattr(args, "cache_telemetry_prefix", None))
        result["manifest"] = str(Path(result["manifest"]).relative_to(args.output))
        return result
    except Exception as error:  # noqa: BLE001 - Preserve an already-raised inference error even if diagnostics fail.
        result = {"status": "retained_with_gaps", "manifest": None, "file_count": 0,
                  "issues": [{"code": "archive_failed", "errno": getattr(error, "errno", None)}]}
        try:
            write_private(directory / "failure.json", result)
        except Exception:  # noqa: BLE001 - Even failure-report I/O must not replace the original request error.
            result["issues"].append({"code": "failure_manifest_unavailable"})
        print(json.dumps({"diagnostic_archive_failed": True, "errno": getattr(error, "errno", None)}),
              file=sys.stderr, flush=True)
        return result


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
    if result.get("diagnostic_capture", {}).get("status") == "retained_with_gaps":
        errors.append(f"{label}: diagnostic archive has explicit coverage gaps")
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
    # Replace any earlier focused histogram with this suite's same control
    # observations, even while capture is incomplete. Never leave an older
    # complete histogram looking current while new context rows are published.
    write_private(root / "public/pi-round-histogram.json", contexts | {
        "measurement_mode": "shared_suite_control_after",
        "coding_context_result": "pi-coding-contexts.json",
    })


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
        arm_root = directory / f"{context}-{arm}"
        try:
            result, ids, _ = matched.capture_arm(
                opener=opener, args=args, rpc=rpc, tokenizer=tokenizer, identity=identity,
                prompt=prefix + suffix, suffix=suffix, arm=arm, root=arm_root)
        finally:
            diagnostic = retain_request_diagnostics(args, arm_root / "diagnostics")
        result["diagnostic_capture"] = diagnostic
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
            diagnostic_root = args.output / "diagnostics" / "chain" / f"{stage}-{uuid.uuid4().hex}"
            try:
                result, ids, completion = chain._request(
                    opener=opener, args=args, identity=group["identity"], tokenizer=tokenizer,
                    prompt_tokens=prompt, suffix_tokens=suffix, stage=stage,
                    max_tokens=getattr(args, f"{stage}_max_tokens"), thinking=thinking)
            finally:
                diagnostic = retain_request_diagnostics(args, diagnostic_root)
            result["diagnostic_capture"] = diagnostic
            del completion
            check_worker(rpc, state)
            result.update(capture_id=digest([state["id"], group["result"]["capture_id"], stage]),
                          measurement_mode="unprofiled_chain", output_token_ids_sha256=digest(ids))
            state["chain"].append({"result": result, "prompt_sha256": digest(prompt),
                "continuation": save_ids(args.output, result["capture_id"], ids),
                "diagnostic_artifacts": [receipt(args.output, path) for path in sorted(diagnostic_root.iterdir())]
                                        if diagnostic_root.is_dir() else []})
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
            for item in saved.get("diagnostic_artifacts", []):
                verify_file(root, item)
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
    p.add_argument("--cache-telemetry-prefix", type=Path,
                   help="Actual cache-job recorder status prefix, if it differs from the round writer")
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
