"""Opaque, isolated replay of long generations to diagnose backend repetition.

Private inputs and outputs stay in the supplied tmpfs directory. Only numeric
measurements and hashes are written to the result directory. This is an offline
experiment, not a serving-time repetition detector or an output limit for Pi.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
from contextlib import contextmanager
import hashlib
import http.server
import json
import os
import re
import shlex
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import benchmark_runtime_flags as bench


def repetition(text, ids):
    paragraphs = [
        re.sub(r"\s+", " ", part).strip()
        for part in re.split(r"\n\s*\n", text)
    ]
    paragraphs = [p for p in paragraphs if len(p) >= 24]
    counts = collections.Counter(paragraphs)
    maximum = max(counts.values(), default=0)
    repeated = sum(count for count in counts.values() if count >= 4)
    # Sliding 64-token windows also catch repeated multi-paragraph blocks whose
    # paragraph boundaries vary. Never retain the windows or their text.
    windows = collections.Counter(tuple(ids[i : i + 64]) for i in range(len(ids) - 63))
    recurring = sum(count for count in windows.values() if count >= 4)
    return {
        "characters": len(text),
        "paragraphs": len(paragraphs),
        "max_identical_paragraphs": maximum,
        "repeated_paragraphs": repeated,
        "repeated_64_token_window_fraction": recurring / max(1, len(ids) - 63),
        "max_identical_64_token_windows": max(windows.values(), default=0),
        "loop_candidate": (maximum >= 4 and repeated >= 8)
        or recurring / max(1, len(ids) - 63) >= 0.15,
        "tool_call_openings": text.count("<tool_call>"),
        "tool_call_closings": text.count("</tool_call>"),
        "thinking_closings": text.count("</think>"),
    }


def thinking_repetition(text, ids, end_token_id):
    """Separate repeated reasoning from legitimate repetition in emitted code."""
    try:
        end = ids.index(end_token_id)
    except ValueError:
        end = len(ids)
    value = repetition(text.split("</think>", 1)[0], ids[:end])
    value.update(output_tokens=end, closed=end < len(ids))
    return value


def stream_replay(endpoint, payload, private_output, progress_path, timeout=1800,
                  open_url=urllib.request.urlopen):
    """Preserve partial diagnostic output privately if a long replay fails.

    No repetition rule changes or terminates generation. The public progress
    file contains only counts and times; the private file contains token IDs.
    """
    payload = dict(payload, stream=True, stream_options={"include_usage": True})
    request = urllib.request.Request(endpoint + "/v1/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    ids, pieces, usage, finish = [], [], None, None
    started = last_saved = time.monotonic()
    complete = False

    def save():
        nonlocal last_saved
        # Atomic replacements also make metadata-only inspection safe while the
        # worker is still receiving a response. These paths belong to this trial.
        for path, value in [
            (private_output, {"token_ids": ids, "complete": complete}),
            (progress_path, {"output_tokens": len(ids), "complete": complete,
                             "elapsed_seconds": time.monotonic() - started}),
        ]:
            temporary = path.with_name(path.name + ".tmp")
            with temporary.open("w") as handle:
                os.fchmod(handle.fileno(), 0o600)
                json.dump(value, handle)
            temporary.replace(path)
        last_saved = time.monotonic()

    try:
        with open_url(request, timeout=timeout) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    complete = finish is not None
                    break
                event = json.loads(data)
                if event.get("usage") is not None:
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    if choice.get("index", 0) != 0:
                        raise ValueError("diagnostic replay returned multiple choices")
                    ids.extend(choice.get("token_ids") or [])
                    pieces.append(choice.get("text") or "")
                    finish = choice.get("finish_reason") or finish
                if time.monotonic() - last_saved >= 5:
                    save()
        if not complete or not ids:
            raise ValueError("diagnostic stream ended without complete token evidence")
        return {"choices": [{"text": "".join(pieces), "token_ids": ids,
                              "finish_reason": finish}], "usage": usage}
    finally:
        save()


@contextmanager
def gdn_audit_request(root, label, enabled, name="gdn-audit-request.json"):
    marker = root / name
    if enabled:
        bench.write_json(marker, {"label": label})
    try:
        yield
    finally:
        if enabled:
            marker.unlink(missing_ok=True)


def trial(root, private, endpoint, fixture, head, seed, index, limit=16384):
    value = bench.read_json(private / f"fixture-{fixture}.json")
    prompt = value["tokens"]
    manifest = bench.read_json(root / "manifest.json")
    experiment = manifest.get("experiment_id", root.name)
    plain = manifest.get("plain_runtime", False)
    chat = {
        "id": bench.digest([experiment, fixture]),
        "generation": bench.digest([experiment, fixture, "replay"]),
    }
    label = f"{index:02d}-{fixture}-{head}-s{seed}"
    payload = {
        "model": bench.MODEL,
        "prompt": prompt,
        "max_tokens": limit,
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 20,
        "seed": seed,
        "ignore_eos": False,
        "skip_special_tokens": False,
        "return_token_ids": True,
        "cache_salt": f"qwen-chat-cache-v1:{chat['id']}:{chat['generation']}",
        "kv_transfer_params": {
            "qwen_chat": chat,
            "qwen_snapshot_abi": os.environ["QWEN_RADIANCE_CACHE_ABI"],
        },
    }
    if plain:
        payload.pop("kv_transfer_params")
    if head == "full":
        # radiance_verifyhead._batch_is_safe explicitly bypasses INT2 when
        # logprobs are requested. Confirm later with an env-disabled boot.
        payload["logprobs"] = 1
    before = bench.parse_metrics(bench.http(endpoint, "/metrics").decode())
    started = time.monotonic()
    samples = []
    bench.emit(event="trial_start", label=label, input_tokens=len(prompt))
    with (gdn_audit_request(root, label, manifest.get("gdn_span_audit", False)),
          gdn_audit_request(root, label, manifest.get("audit_verify_head", False),
                            "head-audit-request.json"),
          concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor):
        if manifest.get("stream_replay"):
            future = executor.submit(stream_replay, endpoint, payload,
                private / f"partial-{experiment}-{label}.json",
                root / "stream-progress.json", manifest.get("request_timeout_seconds", 1800))
        else:
            future = executor.submit(bench.http, endpoint, "/v1/completions", payload, 1800)
        last_report = 0
        while not future.done():
            phase = bench.phase_for(chat)
            sample = {
                "seconds": time.monotonic() - started,
                "phase": {key: phase[key] for key in (
                    "phase", "input_tokens", "computed_tokens", "cached_tokens", "timings_ms"
                ) if key in phase},
                "thermal": bench.thermal(),
            }
            samples.append(sample)
            if sample["seconds"] - last_report >= 15:
                bench.emit(event="trial_progress", label=label, **sample)
                last_report = sample["seconds"]
            time.sleep(1)
        result = future.result()
        if not isinstance(result, dict):
            result = json.loads(result)
    choice = result["choices"][0]
    ids = choice["token_ids"]
    after = bench.parse_metrics(bench.http(endpoint, "/metrics").decode())
    row = {
        "label": label,
        "fixture": fixture,
        "head": head,
        "seed": seed,
        "input_tokens": len(prompt),
        "input_sha256": bench.digest(prompt),
        "output_tokens": len(ids),
        "output_sha256": bench.digest(ids),
        "finish_reason": choice.get("finish_reason"),
        "elapsed_seconds": time.monotonic() - started,
        "usage": result.get("usage"),
        "repetition": repetition(choice.get("text", ""), ids),
        "final_phase": bench.phase_for(chat),
        "metric_delta": {key: after[key] - before.get(key, 0) for key in after},
        "allocator": {} if plain else bench.read_json(bench.MEMORY).get("allocator", {}),
    }
    # Strip vocabulary-normalized logprobs and token strings. Preserve output
    # token IDs only in private RAM so later divergence checks remain possible.
    bench.write_json(private / f"output-{experiment}-{label}.json", {"token_ids": ids})
    bench.write_json(root / f"trial-{label}.json", row)
    bench.write_json(root / f"samples-{label}.json", samples)
    bench.emit(event="trial_complete", **{key: row[key] for key in (
        "label", "output_tokens", "finish_reason", "elapsed_seconds", "repetition"
    )})
    return row


def worker(args):
    private = Path("/private-fixtures")
    experiment = bench.read_json(args.root / "manifest.json").get("experiment_id", args.root.name)
    plan = bench.read_json(args.root / "plan.json")["trials"]
    rows = []
    for index, item in enumerate(plan):
        rows.append(trial(args.root, private, args.endpoint, index=index, **item))
        bench.write_json(args.root / "results.json", rows)
    if bench.read_json(args.root / "manifest.json").get("plain_runtime"):
        return
    from qwen_radiance_cache import request_tail_flush
    for fixture in {item["fixture"] for item in plan}:
        chat = {
            "id": bench.digest([experiment, fixture]),
            "generation": bench.digest([experiment, fixture, "replay"]),
        }
        result = request_tail_flush(chat, timeout=120)
        bench.emit(event="experiment_tail_flush", fixture=fixture, status=result["status"])


def capture_layers(args):
    private = Path("/private-fixtures")
    manifest = bench.read_json(args.root / "manifest.json")
    assert manifest.get("capture_decoder_layers") and manifest.get("plain_runtime")
    assert manifest.get("enforce_eager") and manifest.get("disable_dflash")
    fixture = manifest["layer_capture_fixture"]
    assert re.fullmatch(r"[a-z_]+", fixture)
    prompt = bench.read_json(private / f"fixture-{fixture}.json")["tokens"]
    request = {"schema": "qwen-radiance-layer-capture-request-v1",
               "capture_id": manifest["experiment_id"], "positions": [len(prompt) - 1],
               "expected_layers": 64, "input_sha256": bench.digest(prompt)}
    if manifest.get("gdn_operator_layers"):
        request["gdn_operator_layers"] = manifest["gdn_operator_layers"]
    marker = args.root / "layer-capture-request.json"
    if marker.exists() or (args.root / "layer-capture-result.json").exists():
        raise ValueError("this experiment already armed a decoder capture")
    bench.write_json(marker, request)
    try:
        result = trial(args.root, private, args.endpoint, fixture=fixture, head="full",
                       seed=0, index=len(manifest["trials"]), limit=2)
        bench.write_json(args.root / "layer-capture-trial.json", result)
        report = bench.read_json(args.root / "layer-capture-result.json")
        assert report["complete"] and report["layers"] == 64
        assert report["input_sha256"] == request["input_sha256"]
    finally:
        marker.unlink(missing_ok=True)


def isolate_compiler_cache(launcher, root):
    """Give each experiment fresh compiled code for all three compiler layers."""
    for variable, kind in (("VLLM_CACHE_ROOT", "vllm"),
                           ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                           ("TRITON_CACHE_DIR", "triton")):
        pattern = rf'(?<=-e ){variable}=(?:"[^"\n]+"|[^\s\\"]+)(?=\s)'
        launcher, count = re.subn(
            pattern, f"{variable}=/cache/benchmarks/{root.name}/{kind}", launcher
        )
        if count != 1:
            raise ValueError(f"compiler cache argument is missing or ambiguous: {variable}")
    return launcher


def prepare(args):
    root = args.root.resolve()
    manifest = bench.read_json(root / "manifest.json")
    private = manifest["private_fixture_directory"]
    if not re.fullmatch(r"/dev/shm/qwen-private-replay-[a-z0-9_]+", private):
        raise ValueError("private fixture directory must be isolated tmpfs")
    original = (root / "production-launcher.sh").read_text()
    if hashlib.sha256(original.encode()).hexdigest() != manifest["launcher_sha256"]:
        raise ValueError("production launcher changed")
    launch = bench.make_launcher(original, root, "heads", f"benchmarks/{root.name}", {})
    if manifest.get("isolated_compiler_cache"):
        # These environment switches can affect Python tracing without being
        # part of vLLM's AOT cache key. A fresh namespace makes the actual
        # executed graph independent of previous production compilations.
        launch = isolate_compiler_cache(launch, root)
    overrides = manifest.get("experiment_environment", {})
    if overrides:
        profile = bench.read_json(root / "production-profile.json")
        profile["kernel_environment"].update(overrides)
        bench.write_json(root / "experiment-profile.json", profile)
        source = "radiance-vllm-mxfp4/runtime-radiance-1.0.16.json)"
        assert launch.count(source) == 1
        launch = launch.replace(source, shlex.quote(str(root / "experiment-profile.json")) + ")")
    if manifest.get("disable_dflash"):
        assert launch.count('--speculative-config "$speculative_config"') == 1
        launch = launch.replace('--speculative-config "$speculative_config"', "")
        launch = re.sub(r"^speculative_config=.*\n", "", launch, flags=re.MULTILINE)
    if manifest.get("enforce_eager"):
        launch, count = re.subn(r"--compilation-config '[^']+'", "--enforce-eager", launch)
        if count != 1:
            raise ValueError("eager reference compilation argument is ambiguous")
    if manifest.get("disable_prefix_caching"):
        if launch.count("--enable-prefix-caching") != 1:
            raise ValueError("reference prefix-caching argument is ambiguous")
        launch = launch.replace("--enable-prefix-caching", "--no-enable-prefix-caching")
    if manifest.get("experiment_model"):
        model = manifest["experiment_model"]
        if not re.fullmatch(r"/models/[A-Za-z0-9_.-]+", model):
            raise ValueError("reference model must be in the read-only model mount")
        old = "/models/Qwen3.8-27B-Uncensored-MXFP4-awq --served-model-name"
        if launch.count(old) != 1:
            raise ValueError("reference target model argument is ambiguous")
        launch = launch.replace(old, shlex.quote(model) + " --served-model-name")
    if manifest.get("cpu_offload_gb"):
        amount = manifest["cpu_offload_gb"]
        if not isinstance(amount, int) or not 1 <= amount <= 8:
            raise ValueError("invalid diagnostic CPU weight offload allowance")
        before = "--gpu-memory-utilization 0.97"
        if launch.count(before) != 1:
            raise ValueError("reference memory argument is ambiguous")
        launch = launch.replace(before, before + f" --cpu-offload-gb {amount}")
    for key, value in manifest.get("experiment_serving", {}).items():
        flag = {
            "kv_cache_dtype": "--kv-cache-dtype",
            "max_model_len": "--max-model-len",
            "attention_backend": "--attention-backend",
            "kv_cache_memory": "--kv-cache-memory",
        }[key]
        # Only substitute the explicit target argument, never the drafter's
        # independent configuration or arbitrary shell source.
        pattern = rf"(?<![\w-]){re.escape(flag)} [A-Za-z0-9_]+(?=\s)"
        launch, count = re.subn(pattern, f"{flag} {shlex.quote(str(value))}", launch)
        assert count == 1, key
    if manifest.get("plain_runtime"):
        assert launch.count('"$image" /patches/bootstrap_radiance_release.py') == 1
        launch = launch.replace('"$image" /patches/bootstrap_radiance_release.py',
                                '"$image" /benchmark/bootstrap_plain_experiment.py')
        for argument in (
            '--scheduler-cls qwen_radiance_fair_scheduler.FairScheduler --additional-config "$fair_config"',
            '--kv-transfer-config "$kv_transfer_config"',
            '--middleware qwen_radiance_request_guard.require_snapshot_abi',
        ):
            assert launch.count(argument) == 1
            launch = launch.replace(argument, "")
        launch, count = re.subn(r"^kv_transfer_config=\$\(jq -cn .*?^fair_config=[^\n]*\n", "",
                               launch, flags=re.MULTILINE | re.DOTALL)
        assert count == 1
        launch = re.sub(r"^[ \t]*\\\n", "", launch, flags=re.MULTILINE)
    elif manifest.get("independent_draft_rng"):
        assert launch.count('"$image" /patches/bootstrap_radiance_release.py') == 1
        launch = launch.replace('"$image" /patches/bootstrap_radiance_release.py',
                                '"$image" /benchmark/bootstrap_sampling_experiment.py')
    anchor = '\t"${release_environment[@]}" \\\n'
    assert launch.count(anchor) == 1
    launch = launch.replace(anchor, anchor + f"\t-v {shlex.quote(private)}:/private-fixtures:rw \\\n")
    path = root / "launch-heads.sh"
    path.write_text(launch)
    subprocess.run(["shfmt", "-w", str(path)], check=True)
    subprocess.run(["shellcheck", str(path)], check=True)
    subprocess.run(["shfmt", "-d", str(path)], check=True)
    bench.write_json(root / "launcher-validation.json", {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "shellcheck": True, "shfmt": True})
    trials = manifest.get("trials", [])
    if not trials:
        for fixture, order in (("largest_loop", ("fast", "full")), ("latest_loop", ("full", "fast"))):
            for seed in (0, 1):
                for head in order:
                    trials.append({"fixture": fixture, "head": head, "seed": seed})
        for head in ("fast", "full"):
            trials.append({"fixture": "first_after_compaction", "head": head, "seed": 0})
    bench.write_json(root / "plan.json", {"trials": trials})
    bench.emit(event="prepared", trials=len(trials))


def stop_isolated_backend(name):
    """Allow an operator's concurrent stop without skipping production restore."""
    if not re.fullmatch(r"qwen-runtime-ab-[a-z0-9]+", name):
        raise ValueError("refusing to stop a container outside the experiment namespace")
    result = subprocess.run(["podman", "stop", "--time", "120", name],
                            capture_output=True, text=True, timeout=135)
    if result.returncode == 0:
        return
    exists = subprocess.run(["podman", "container", "exists", name],
                            capture_output=True, text=True, timeout=30)
    if exists.returncode == 1:
        return  # The other stop removed this --rm container.
    if exists.returncode == 0:
        state = subprocess.run(["podman", "inspect", "--format", "{{.State.Running}}", name],
                               capture_output=True, text=True, timeout=30)
        if state.returncode == 0 and state.stdout.strip() == "false":
            return
    raise RuntimeError("isolated backend is still running or its stopped state cannot be verified")


def run(args):
    root = args.root.resolve()
    assert re.fullmatch(r"qwen-runtime-ab-[a-z0-9]+", root.name)
    manifest = bench.read_json(root / "manifest.json")
    launch = root / "launch-heads.sh"
    assert hashlib.sha256(launch.read_bytes()).hexdigest() == bench.read_json(root / "launcher-validation.json")["sha256"]
    if manifest.get("probe_mxfp4_numerics"):
        # Catch missing mounted inputs before interrupting the production serve.
        assert (root / "probe_mxfp4_numerics.py").is_file()
        assert bench.read_json(root / "production-profile.json")["kernel_hashes"]
    if manifest.get("probe_r4d_attention"):
        assert (root / "probe_r4d_attention_numerics.py").is_file()
        assert bench.read_json(root / "production-profile.json")["kernel_hashes"]
    if manifest.get("probe_gdn_numerics"):
        assert (root / "probe_gdn_numerics.py").is_file()
        assert bench.read_json(root / "production-profile.json")["kernel_hashes"]
    if manifest.get("probe_gdn_repair_performance"):
        assert hashlib.sha256((root / "probe_gdn_repair_performance.py").read_bytes()).hexdigest() == manifest["gdn_repair_performance_probe_sha256"]
        assert manifest.get("gdn_extreme_decay_repair")
    if manifest.get("gdn_extreme_decay_repair"):
        assert (root / "patch_gdn_extreme_decay.py").is_file()
        assert hashlib.sha256((root / "gdn_extreme_decay_reference.so").read_bytes()).hexdigest() == manifest["gdn_repair_sha256"]
    if manifest.get("r4d_dispatch_audit"):
        assert manifest.get("plain_runtime")
        assert hashlib.sha256((root / "r4d_dispatch_audit.py").read_bytes()).hexdigest() == manifest["r4d_dispatch_audit_sha256"]
    if manifest.get("capture_decoder_layers"):
        from capture_radiance_layers import LEGACY_SHA256

        assert hashlib.sha256((root / "capture_radiance_layers.py").read_bytes()).hexdigest() == manifest["layer_capture_sha256"]
        assert hashlib.sha256((root / "legacy_layer_diagnostic.py").read_bytes()).hexdigest() == LEGACY_SHA256
    process = None
    server = None
    audit_engine_pids = []
    stopped = False
    failure = None
    try:
        # Explicit user authorization permits interrupting ongoing generation.
        # Flush settled tails first; the backend shutdown hook handles its last
        # settled state. Do not wait indefinitely for a looping client to idle.
        flush_source = '''
import json
from pathlib import Path
from qwen_radiance_cache import request_tail_flush
rows=json.loads(Path('/dev/shm/qwen-radiance-snapshot-tail.json').read_text()).get('chats', [])
out=[]
for row in rows:
 result=request_tail_flush({'id':row['chat_id'],'generation':row['generation']},timeout=120)
 out.append({k:result[k] for k in ('status','tokens') if k in result})
print(json.dumps({'flushed':out}))
'''
        flushed = bench.command(["podman", "exec", bench.PRODUCTION, "/opt/vllm/bin/python", "-c", flush_source], timeout=360)
        bench.write_json(root / "flush.json", json.loads(flushed.stdout))
        bench.command(["podman", "stop", "--time", "120", bench.PRODUCTION], timeout=135)
        stopped = True
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 8080), bench.Maintenance)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        (bench.CACHE_ROOT / "benchmarks" / root.name / "data").mkdir(mode=0o700, parents=True, exist_ok=False)
        if manifest.get("probe_sampling_rng"):
            bench.write_json(root / "status.json", {"stage": "native_sampling_probe", "at": time.time()})
            with (root / "sampling-probe.log").open("w") as log:
                result = subprocess.run([
                    "podman", "run", "--rm", "--pull=never", "--network=none",
                    "--privileged", "--device", "/dev/kfd", "--device", "/dev/dri",
                    "--group-add", "keep-groups", "-v", f"{root}:/benchmark:rw",
                    "--entrypoint", "/opt/vllm/bin/python", manifest["image"],
                    "/benchmark/probe_dflash_sampling_rng.py", "--output", "/benchmark/sampling-probe.json",
                ], stdout=log, stderr=subprocess.STDOUT, timeout=180)
            if result.returncode:
                raise RuntimeError("native sampling distribution gate failed")
        if manifest.get("probe_mxfp4_numerics"):
            bench.write_json(root / "status.json", {"stage": "native_mxfp4_probe", "at": time.time()})
            with (root / "mxfp4-probe.log").open("w") as log:
                result = subprocess.run([
                    "podman", "run", "--rm", "--pull=never", "--network=none",
                    "--privileged", "--device", "/dev/kfd", "--device", "/dev/dri",
                    "--group-add", "keep-groups", "-v", f"{root}:/benchmark:rw",
                    "-v", "/home/lewis/models-radiance:/models:ro",
                    "--entrypoint", "/opt/vllm/bin/python", manifest["image"],
                    "/benchmark/probe_mxfp4_numerics.py", "--root", "/benchmark",
                ], stdout=log, stderr=subprocess.STDOUT, timeout=600)
            if result.returncode:
                raise RuntimeError("native MXFP4 numerical gate failed")
        if manifest.get("probe_r4d_attention"):
            bench.write_json(root / "status.json", {"stage": "native_attention_probe", "at": time.time()})
            with (root / "attention-probe.log").open("w") as log:
                result = subprocess.run([
                    "podman", "run", "--rm", "--pull=never", "--network=none",
                    "--privileged", "--device", "/dev/kfd", "--device", "/dev/dri",
                    "--group-add", "keep-groups", "-v", f"{root}:/benchmark:rw",
                    "--entrypoint", "/opt/vllm/bin/python", manifest["image"],
                    "/benchmark/probe_r4d_attention_numerics.py", "--root", "/benchmark",
                ], stdout=log, stderr=subprocess.STDOUT, timeout=600)
            if result.returncode:
                raise RuntimeError("native paged-attention numerical gate failed")
        if manifest.get("probe_gdn_numerics"):
            bench.write_json(root / "status.json", {"stage": "native_gdn_probe", "at": time.time()})
            with (root / "gdn-probe.log").open("w") as log:
                result = subprocess.run([
                    "podman", "run", "--rm", "--pull=never", "--network=none",
                    "--privileged", "--device", "/dev/kfd", "--device", "/dev/dri",
                    "--group-add", "keep-groups", "-v", f"{root}:/benchmark:rw",
                    "--entrypoint", "/opt/vllm/bin/python", manifest["image"],
                    "/benchmark/probe_gdn_numerics.py", "--root", "/benchmark",
                ], stdout=log, stderr=subprocess.STDOUT, timeout=600)
            if result.returncode:
                raise RuntimeError("native GDN diagnostic did not complete")
        if manifest.get("probe_gdn_repair_performance"):
            bench.write_json(root / "status.json", {"stage": "gdn_repair_timing", "at": time.time()})
            with (root / "gdn-repair-performance.log").open("w") as log:
                result = subprocess.run([
                    "podman", "run", "--rm", "--pull=never", "--network=none",
                    "--privileged", "--device", "/dev/kfd", "--device", "/dev/dri",
                    "--group-add", "keep-groups", "-v", f"{root}:/benchmark:rw",
                    "--entrypoint", "/opt/vllm/bin/python", manifest["image"],
                    "/benchmark/probe_gdn_repair_performance.py", "--root", "/benchmark",
                ], stdout=log, stderr=subprocess.STDOUT, timeout=300)
            if result.returncode:
                raise RuntimeError("GDN correction timing did not complete")
        if manifest.get("probe_norm_rope"):
            bench.write_json(root / "status.json", {"stage": "native_norm_rope_probe", "at": time.time()})
            with (root / "norm-rope-probe.log").open("w") as log:
                result = subprocess.run([
                    "podman", "run", "--rm", "--pull=never", "--network=none",
                    "--privileged", "--device", "/dev/kfd", "--device", "/dev/dri",
                    "--group-add", "keep-groups", "-v", f"{root}:/benchmark:rw",
                    "-v", "/home/lewis/models-radiance:/models:ro",
                    "--entrypoint", "/opt/vllm/bin/python", manifest["image"],
                    "/benchmark/probe_norm_rope_numerics.py", "--root", "/benchmark",
                ], stdout=log, stderr=subprocess.STDOUT, timeout=300)
            if result.returncode:
                raise RuntimeError("normalization/position numerical diagnostic did not pass")
        env = dict(os.environ, QWEN_QUALIFICATION_CONTAINER=root.name, QWEN_QUALIFICATION_PORT="18080")
        with (root / "backend-heads.log").open("w") as log:
            process = subprocess.Popen(["bash", str(launch)], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
        bench.write_json(root / "status.json", {"stage": "startup", "at": time.time()})
        bench.wait_ready("http://127.0.0.1:18080", process)
        if manifest.get("r4d_dispatch_audit"):
            probe = (
                "import json;from pathlib import Path\n"
                "pids=[]\n"
                "for process in Path('/proc').iterdir():\n"
                " if not process.name.isdecimal(): continue\n"
                " try: identity=(process/'cmdline').read_bytes().split(b'\\0')[0]\n"
                " except OSError: continue\n"
                " if identity.startswith(b'VLLM::EngineCore'): pids.append(int(process.name))\n"
                "assert pids, 'no inference-engine process could be attested'\n"
                "print(json.dumps(pids))\n"
            )
            checked = bench.command([
                "podman", "exec", root.name, "/opt/vllm/bin/python", "-c", probe,
            ], timeout=60)
            audit_engine_pids = json.loads(checked.stdout)
        if manifest.get("experiment_environment"):
            expected = manifest["experiment_environment"]
            checks = {
                "RADIANCE_TOPK_COMPOSITE": (
                    "vllm.v1.sample.ops.topk_topp_sampler", "_RADIANCE_TOPK_COMPOSITE"
                ),
                "RADIANCE_GDN_MERGE_INPROJ": ("radiance_gdnmerge", "ENABLED"),
            }
            probe = (
                "import os,json,importlib\n"
                f"expected={expected!r}\nchecks={checks!r}\n"
                "actual={key:os.environ.get(key) for key in expected}\n"
                "assert actual==expected, 'experiment environment mismatch'\n"
                "gates={}\n"
                "for key,(module,attribute) in checks.items():\n"
                " if key in expected:\n"
                "  gates[key]=bool(getattr(importlib.import_module(module),attribute))\n"
                "  assert gates[key]==(expected[key]=='1'), 'experiment gate mismatch'\n"
                "print(json.dumps({'environment':actual,'imported_gates':gates}))\n"
            )
            checked = bench.command([
                "podman", "exec", root.name, "/opt/vllm/bin/python", "-c", probe,
            ], timeout=60)
            bench.write_json(root / "experiment-runtime.json", json.loads(checked.stdout))
        if manifest.get("independent_draft_rng") or manifest.get("plain_runtime"):
            checked = bench.command([
                "podman", "exec", root.name, "/opt/vllm/bin/python", "-c",
                "import hashlib,json,sys;from pathlib import Path;"
                "sys.path.insert(0,'/benchmark');import patch_dflash_sampling_rng as p;"
                "actual=hashlib.sha256((Path('/opt/vllm/lib/python3.12/site-packages')/p.SOURCE).read_bytes()).hexdigest();"
                "assert actual==p.POSTIMAGE;print(json.dumps({'active_source_sha256':actual,'upstream_pr':54282}))",
            ], timeout=30)
            bench.write_json(root / "sampling-runtime.json", json.loads(checked.stdout))
        if manifest.get("reference_linear"):
            checked = bench.command([
                "podman", "exec", root.name, "/opt/vllm/bin/python", "-c",
                "import hashlib,json;from pathlib import Path;import radiance_mxfp4 as r;"
                "actual=hashlib.sha256(Path(r.__file__).read_bytes()).hexdigest();"
                f"assert actual=={manifest['reference_source_sha256']!r} and r.REF_LINEAR and r.WPERM;"
                "print(json.dumps({'active_source_sha256':actual,'BF16_linear_reference':True}))",
            ], timeout=60)
            bench.write_json(root / "reference-runtime.json", json.loads(checked.stdout))
        if manifest.get("gdn_extreme_decay_repair"):
            correction_enabled = manifest.get("experiment_environment", {}).get("RADIANCE_USE_R4D", "1") == "1" and manifest.get("experiment_environment", {}).get("RADIANCE_USE_R4D_GDN", "1") == "1"
            checked = bench.command([
                "podman", "exec", root.name, "/opt/vllm/bin/python", "-c",
                "import torch,hashlib,json;from pathlib import Path;import radiance_gdn as r;" +
                ("assert r._CHUNK_SCAN.__name__=='_qwen_gdn_corrected_scan';"
                 if correction_enabled else "assert r._CHUNK_SCAN is None and not r.USE_R4D;") +
                "library=Path(r._qwen_gdn_library._name);"
                "actual=hashlib.sha256(library.read_bytes()).hexdigest();"
                f"assert actual=={manifest['gdn_repair_sha256']!r};"
                "print(json.dumps({'library_sha256':actual,'active_source_sha256':"
                "hashlib.sha256(Path(r.__file__).read_bytes()).hexdigest(),"
                f"'corrected_prefill_scan_installed':True,'corrected_prefill_scan_enabled':{correction_enabled!r}}}))",
            ], timeout=60)
            bench.write_json(root / "gdn-correction-runtime.json", json.loads(checked.stdout))
        if manifest.get("bf16_gdn_gate_reference"):
            attestation = bench.read_json(root / "gate-loaded-attestation.json")
            if not attestation["all_exact_bf16"] or attestation["gate_modules"] != 48:
                raise ValueError("BF16 gate experiment did not verify all loaded target gates")
        bench.emit(event="isolated_backend_ready")
        bench.write_json(root / "status.json", {"stage": "replays", "at": time.time()})
        with (root / "worker.log").open("w") as log:
            result = subprocess.run(["podman", "exec", root.name, "/opt/vllm/bin/python", "/benchmark/diagnose_loop_replay.py", "worker", "--root", "/benchmark", "--endpoint", "http://127.0.0.1:18080"], stdout=log, stderr=subprocess.STDOUT, timeout=14400)
        if result.returncode:
            raise RuntimeError("isolated replay worker failed")
        if manifest.get("audit_verify_head"):
            audited = []
            for row in bench.read_json(root / "results.json"):
                report = bench.read_json(root / f"head-audit-{row['label']}.json")
                if (report.get("schema") != "qwen-target-head-audit-v1"
                        or report.get("experiment_id") != root.name
                        or report.get("label") != row["label"]
                        or not 1 <= report.get("calls_checked", 0) <= 128
                        or not report.get("returns_candidate_logits_unchanged")):
                    raise ValueError("target-head observation did not cover this request")
                audited.append({"label": row["label"], "calls_checked": report["calls_checked"],
                                "rows_checked": report["rows_checked"]})
            if len(audited) != len(manifest["trials"]):
                raise ValueError("target-head observation has incomplete trial coverage")
            bench.write_json(root / "head-audit-validation.json", {
                "complete": True, "trials": audited,
                "scope": "declared first 128 target-head calls; candidate output unchanged",
                "full_request_equivalence_proven": False})
        if manifest.get("capture_decoder_layers"):
            bench.write_json(root / "status.json", {"stage": "decoder_layer_capture", "at": time.time()})
            with (root / "layer-capture-worker.log").open("w") as log:
                captured = subprocess.run([
                    "podman", "exec", root.name, "/opt/vllm/bin/python", "/benchmark/diagnose_loop_replay.py",
                    "capture", "--root", "/benchmark", "--endpoint", "http://127.0.0.1:18080",
                ], stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            if captured.returncode:
                raise RuntimeError("decoder layer capture did not complete")
        if manifest.get("r4d_dispatch_audit"):
            from r4d_dispatch_audit import validate_reports

            # Reporters publish once a second without touching GPU state. Let
            # the final dispatch become visible before inspecting the counts.
            time.sleep(1.1)
            attestation = validate_reports(root / "r4d-dispatch", manifest["r4d_dispatch_audit"],
                                           engine_pids=audit_engine_pids)
            bench.write_json(root / "r4d-dispatch-validation.json", attestation)
    except Exception as error:
        failure = {"type": type(error).__name__}
        bench.write_json(root / "failure.json", failure)
        bench.emit(event="experiment_failed", **failure)
    finally:
        if stopped:
            bench.write_json(root / "status.json", {"stage": "restoring_production", "at": time.time()})
            stop_isolated_backend(root.name)
            if process is not None:
                process.wait(timeout=20)
            if server is not None:
                server.shutdown()
                server.server_close()
            production = Path(manifest["production_launcher"])
            assert hashlib.sha256(production.read_bytes()).hexdigest() == manifest["launcher_sha256"]
            with (root / "production-restored.log").open("w") as log:
                restored = subprocess.Popen(["bash", str(production)], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            bench.wait_ready("http://127.0.0.1:8080", restored)
            bench.emit(event="production_restored")
        bench.write_json(root / "status.json", {"stage": "failed" if failure else "complete", "production_restored": stopped, "at": time.time()})
    return int(failure is not None)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "worker", "capture"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080")
    args = parser.parse_args()
    return {"prepare": prepare, "run": run, "worker": worker, "capture": capture_layers}[args.mode](args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Request/response bodies and exception messages may contain chat data.
        bench.emit(event="failed", type=type(error).__name__)
        raise SystemExit(1) from None
