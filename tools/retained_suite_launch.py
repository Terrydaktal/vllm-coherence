"""Emit a Podman argv for a source-bound, isolated shared benchmark server."""

import argparse
import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROBES = {
    "prefill": "sampled_cache_probe.SampledCacheProbe",
    "response-end": "response_end_probe.ResponseEndProbe",
}


def replace_flag(argv, flag, value):
    if argv.count(flag) != 1:
        raise ValueError(f"missing or repeated launch flag {flag}")
    argv[argv.index(flag) + 1] = value


def server_argv(inspection, overlay, private_cache, repository, name, port, out, probe=None):
    original = inspection[0] if isinstance(inspection, list) else inspection
    if not re.fullmatch(r"coherence-bench-[a-z0-9-]+", name):
        raise ValueError("use a dedicated coherence-bench-* container name")
    if not 1024 <= port <= 65535:
        raise ValueError("choose an unprivileged isolated port")
    argv = list(original["Config"]["CreateCommand"])
    if argv[:2] != ["podman", "run"] or argv.count(original["ImageName"]) != 1:
        raise ValueError("expected the original recorded Podman run command")
    image = argv.index(original["ImageName"])
    if argv[image + 1] != "/patches/extended_context_bootstrap.py":
        raise ValueError("unexpected baseline bootstrap")
    base_port = argv[argv.index("--port") + 1]
    if str(port) == base_port or name == argv[argv.index("--name") + 1]:
        raise ValueError("choose a separate endpoint and container")
    live = next(m["Source"] for m in original["Mounts"] if m["Destination"] == "/cache")
    if not all(p.is_dir() for p in (overlay, private_cache, repository, out)):
        raise ValueError(
            "overlay, private cache, checkout and output directories must exist"
        )
    if (out / "startup.json").exists():
        raise ValueError("use a fresh output directory for each serving instance")
    live_cache, private_cache = Path(live).resolve(), private_cache.resolve()
    if live_cache.is_relative_to(private_cache) or private_cache.is_relative_to(live_cache):
        raise ValueError("provide a separate cache root")
    mounts = [
        i
        for i, value in enumerate(argv[:image])
        if value == live + ":/cache" or value.startswith(live + ":/cache:")
    ]
    if len(mounts) != 1 or argv.count("--ipc=host") != 1:
        raise ValueError("unexpected cache or shared memory mount")
    argv[mounts[0]] = f"{private_cache}:/cache:rw"
    replace_flag(argv, "--name", name)
    replace_flag(argv, "--port", str(port))
    replace_flag(argv, "--host", "127.0.0.1")
    replace_flag(argv, "--worker-cls", "fourteen_worker.Worker")
    status_path = (
        "/dev/shm/qwen-prefill-diagnosis" if probe == "prefill" else "/dev/shm/qwen-stage-timing"
    )
    cfg = json.loads(argv[argv.index("--additional-config") + 1])
    cfg["qwen_fair"]["status_path"] = status_path
    replace_flag(argv, "--additional-config", json.dumps(cfg))
    cfg = json.loads(argv[argv.index("--kv-transfer-config") + 1])
    cfg["engine_id"] = name
    replace_flag(argv, "--kv-transfer-config", json.dumps(cfg))
    argv[image + 1] = (
        "/bench/source/experiments/radiance-public/retained_serving_bootstrap.py"
    )
    extra = [
        "-v",
        f"{out.resolve()}:/bench:rw",
        "-v",
        f"{repository.resolve()}:/bench/source:ro",
        "-v",
        f"{overlay.resolve()}:/retained:ro",
    ]
    for key, value in {
        "VLLM_SERVER_DEV_MODE": "1",
        "QWEN_ROUND_EVENT_STATUS_PATH": status_path,
        "QWEN_OPTIMIZED_STARTUP_RECEIPT": "/bench/startup.json",
    }.items():
        existing = [i for i, entry in enumerate(argv[:image]) if entry.startswith(key + "=")]
        if len(existing) > 1:
            raise ValueError(f"repeated baseline setting {key}")
        assignment = f"{key}={value}"
        if existing:
            argv[existing[0]] = assignment
        else:
            extra += ["-e", assignment]
    if probe:
        if "--worker-extension-cls" in argv:
            raise ValueError("expected an uninstrumented baseline worker")
        argv += ["--worker-extension-cls", PROBES[probe]]
    if probe == "prefill":
        extra += ["-v", f"{out.resolve()}:/prefill-diagnosis/refresh:rw"]
    argv[image:image] = extra
    i = argv.index("--ipc=host")
    argv[i : i + 1] = ["--ipc=private", "--shm-size=30g"]
    return argv


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inspect", type=Path, required=True)
    p.add_argument("--overlay", type=Path, required=True)
    p.add_argument("--private-cache", type=Path, required=True)
    p.add_argument("--repository", type=Path, default=REPO)
    p.add_argument("--name", default="coherence-bench-retained")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--probe", choices=PROBES, help="enable a dedicated correctness capture lane")
    p.add_argument("--output-directory", type=Path, required=True)
    p.add_argument(
        "--argv", required=True, type=Path, help="new private JSON argv file"
    )
    a = p.parse_args()
    command = server_argv(
        json.loads(a.inspect.read_text()),
        a.overlay,
        a.private_cache,
        a.repository,
        a.name,
        a.port,
        a.output_directory,
        a.probe,
    )
    descriptor = os.open(
        a.argv, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "w") as stream:
        json.dump(command, stream, indent=2)
        stream.write("\n")
