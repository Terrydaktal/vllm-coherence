"""Prepare and run the pinned Coherence payload without importing a GPU runtime."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "experiments/radiance-public"
MODEL = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate"
HELP = """NAME
    coherence - pinned inference, persistent sessions and conformance tooling

SYNOPSIS
    tools/coherence doctor
    tools/coherence prepare [--archive PATH] [--state PATH]
    tools/coherence serve --model PATH --draft PATH [--dry-run] [--state PATH]
    tools/coherence pi [--ssh HOST] [--remote-state PATH] [--state PATH] [-- PI_OPTIONS]
    tools/coherence cache [--state PATH] [-- CACHE_OPTIONS]
    tools/coherence verify-runtime [--state PATH]

DESCRIPTION
    Coherence is a standalone downstream fork of Radiance. Preparation verifies
    the release archive and every payload file. Serving uses the pinned image,
    repaired D7 arithmetic, global-256 target head, GGZ14 performance backports,
    persistent compressed snapshots and response-boundary scheduling.

OPTIONS
    --state PATH       Private local state (default: $XDG_STATE_HOME/vllm-coherence).
    --archive PATH     Use an already downloaded, checksum-verified release archive.
    --model PATH       Qwen3.8-27B-Uncensored-MXFP4-awq directory.
    --draft PATH       Corresponding Qwen3.8-27B-DFlash2-FP8 directory.
    --engine NAME      podman (the supported container runtime).
    --port PORT        Local API port (default: 8080).
    --head MODE        global512 (default), global256 or full-bf16.
    --dry-run          Print the container command; do not prepare or launch it.
    --ssh HOST         Forward a remote Coherence backend over SSH for Pi.
    --remote-state P   Remote state directory (default: ~/.local/state/vllm-coherence).
    --search-extension PATH  Optional user-selected Pi search extension.
    --help             Show this help.

OPERATION
    Run prepare once, then serve on the GPU host, then pi from each workspace.
    Pi installation is pinned and isolated beneath the state directory.
    Each workspace keeps its own .pi/sessions. Use --session last to resume.
    The default head uses approximate INT2 selection followed by BF16 reranking;
    full-bf16 selects the complete target vocabulary head for numerical controls.

EXAMPLES
    tools/coherence prepare
    tools/coherence serve --model /models/target --draft /models/drafter
    tools/coherence pi -- --thinking xhigh --session last
    tools/coherence pi --ssh gpu-host -- --continue
    tools/coherence cache -- status --details

FILES
    releases/0.1.0.json             Artifact identity and qualification scope.
    STATE/runtime/                 Verified frozen sources, kernels and receipts.
    STATE/patches/                  Versioned snapshot and serving integration.
    STATE/cache/snapshots/          Private compressed model state.
    STATE/connection.json           Local connection metadata; no credentials.

PATHS
    Model directories are mounted read-only. State defaults to user-local storage.
    Install tools/coherence into ~/.local/bin using a symbolic link if desired.

SECURITY NOTES
    The API binds to loopback. Use SSH forwarding for remote access. Snapshots,
    token fixtures and diagnostic tensors are private data. Never commit them.
    Container GPU access is required only by serve. doctor and prepare use no GPU.

EXIT STATUS
    0 success; 2 invalid configuration, integrity or preparation failure;
    otherwise the launched program's exit status.

AUTHORS
    Terrydaktal and the attributed Radiance, vLLM, libr4d and GGZ14 contributors.
"""


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"refusing symlink: {path}")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex)
    with temporary.open("x") as out:
        os.chmod(temporary, 0o600)
        json.dump(value, out, sort_keys=True, indent=2)
        out.write("\n")
    temporary.replace(path)


def state_path(value=None):
    root = (
        Path(value)
        if value
        else Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
        / "vllm-coherence"
    )
    root = root.expanduser().absolute()
    if root.is_symlink():
        raise ValueError("state directory must not be a symlink")
    return root


def private_directory(path):
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    st = path.lstat()
    if path.is_symlink() or st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise ValueError(f"directory must be owned and mode 0700: {path}")


@contextlib.contextmanager
def deployment_lock(state):
    """Do not replace integration files underneath a running backend."""
    private_directory(state)
    descriptor = os.open(
        state / "deployment.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "r+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "this state is serving or being prepared; stop its backend before replacing it"
            ) from error
        yield


def verify_payload(root, expected_manifest=None):
    if root.is_symlink() or (root / "optimized-release.json").is_symlink():
        raise ValueError("indirect payload root")
    if (
        expected_manifest
        and digest(root / "optimized-release.json") != expected_manifest
    ):
        raise ValueError("payload manifest checksum mismatch")
    manifest = read_json(root / "optimized-release.json")
    if manifest.get("schema") != "urn:qwen:optimized-pi-release:v1" or not manifest.get(
        "files"
    ):
        raise ValueError("invalid or empty payload manifest")
    for name, expected in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("payload path escapes root")
        path = root / relative
        if any(
            parent.is_symlink()
            for parent in [path, *path.parents]
            if parent != root.parent
        ):
            raise ValueError(f"indirect payload file: {name}")
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"payload file changed or missing: {name}")
    return manifest


def extract_archive(archive, destination):
    private_directory(destination)
    with tarfile.open(archive, "r:xz") as stream:
        members = stream.getmembers()
        if len(members) > 2000 or sum(m.size for m in members) > 128 * 1024**2:
            raise ValueError("oversized runtime archive")
        names = set()
        for member in members:
            name = Path(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or not name.parts
                or member.name in names
            ):
                raise ValueError("unsafe or duplicate archive path")
            names.add(member.name)
            if not member.isfile():
                raise ValueError("archive must contain only regular files")
        for member in members:
            path = destination / member.name
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with stream.extractfile(member) as src, path.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            path.chmod(0o600)


def release():
    return read_json(ROOT / "releases/0.1.0.json")


def verify_runtime(state):
    spec = release()
    root = state / "runtime"
    manifest = verify_payload(root, spec["manifest_sha256"])
    for relative, expected in spec.get("support_files", {}).items():
        path = root / relative
        if path.is_symlink() or path.parent.is_symlink() or digest(path) != expected:
            raise ValueError("support library changed")
    return manifest


def prepare(state, archive=None, head="global512"):
    with deployment_lock(state):
        return _prepare(state, archive, head)


def _prepare(state, archive=None, head="global512"):
    private_directory(state)
    spec = release()
    runtime = state / "runtime"
    if not runtime.exists():
        with tempfile.TemporaryDirectory(prefix="prepare-", dir=state) as temporary:
            temporary = Path(temporary)
            if archive:
                download = Path(archive).expanduser().resolve()
            else:
                download = temporary / "runtime.tar.xz"
                with (
                    urllib.request.urlopen(spec["url"], timeout=60) as response,
                    download.open("xb") as out,
                ):
                    remaining = 32 * 1024**2
                    while chunk := response.read(min(1024**2, remaining + 1)):
                        remaining -= len(chunk)
                        if remaining < 0:
                            raise ValueError("oversized runtime download")
                        out.write(chunk)
            if digest(download) != spec["sha256"]:
                raise ValueError("release archive checksum mismatch")
            extracted = temporary / "runtime"
            extract_archive(download, extracted)
            verify_payload(extracted, spec["manifest_sha256"])
            extracted.rename(runtime)
    verify_runtime(state)
    patches = state / "patches"
    private_directory(patches)
    names = spec["patch_files"]
    for relative in names:
        source = ROOT / relative
        if (
            source.is_symlink()
            or digest(source) != spec["integration_sha256"][relative]
        ):
            raise ValueError(f"release integration changed: {relative}")
        destination = patches / source.name
        if destination.is_symlink():
            raise ValueError("indirect integration file")
        shutil.copyfile(source, destination)
        destination.chmod(0o600)
    profile = read_json(patches / "runtime-radiance-1.0.16.json")
    original = read_json(runtime / "optimized-release.json")
    # Keep the frozen release intact; the selected output head has a separate
    # manifest. An output-head change never claims new backbone qualification.
    manifest = json.loads(json.dumps(original))
    manifest["target_head"] = head
    manifest["environment"]["RADIANCE_VERIFY_HEAD"] = (
        "1" if head in ("global256", "global512") else "0"
    )
    manifest["environment"]["RADIANCE_VERIFY_HEAD_GLOBAL_TOPK"] = (
        head.removeprefix("global") if head.startswith("global") else "512"
    )
    write_json(patches / "optimized-release.json", manifest)
    profile["optimized_d7"]["manifest_sha256"] = digest(
        patches / "optimized-release.json"
    )
    profile["optimized_d7"]["target_head"]["mode"] = head
    profile["optimized_d7"]["target_head"]["candidate_count"] = (
        int(head.removeprefix("global")) if head.startswith("global") else 0
    )
    profile["optimized_d7"]["target_head"]["selection"] = (
        f"full-vocabulary INT2 top-{head.removeprefix('global')} then BF16-weight rerank"
        if head.startswith("global")
        else "complete corrected BF16 head"
    )
    profile["kernel_environment"]["RADIANCE_VERIFY_HEAD"] = manifest["environment"][
        "RADIANCE_VERIFY_HEAD"
    ]
    profile["kernel_environment"]["RADIANCE_VERIFY_HEAD_GLOBAL_TOPK"] = manifest[
        "environment"
    ]["RADIANCE_VERIFY_HEAD_GLOBAL_TOPK"]
    write_json(patches / "runtime-radiance-1.0.16.json", profile)
    # A separate namespace for this exported release; old user cache is never
    # reinterpreted under a new state/arithmetic contract.
    identity = hashlib.sha256(
        json.dumps(
            {
                "release": spec["sha256"],
                "integration": {n: digest(ROOT / n) for n in names},
                "schema": "urn:coherence:snapshot:v1",
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    cache = state / "cache"
    private_directory(cache)
    data = cache / "snapshots" / identity / "data"
    data.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_json(
        data.parent / "abi.json",
        {
            "schema": "urn:coherence:snapshot:v1",
            "storage": {"data_abi": identity},
            "release": spec["sha256"],
        },
    )
    connection = {
        "abi": identity,
        "cache_root": str(cache),
        "state": str(state),
        "port": 8080,
        "model": MODEL,
        "head": head,
    }
    existing = state / "connection.json"
    if existing.exists():
        connection["port"] = read_json(existing).get("port", 8080)
    write_json(existing, connection)
    return connection


def serving_command(args, connection):
    state = state_path(args.state)
    profile = read_json(RUNTIME / "runtime-radiance-1.0.16.json")
    abi = connection["abi"]
    model, draft = (
        Path(args.model).expanduser().absolute(),
        Path(args.draft).expanduser().absolute(),
    )
    for path in (state, model, draft):
        if ":" in str(path) or "\n" in str(path):
            raise ValueError("container mount paths cannot contain a colon or newline")
    environment = {
        **profile["kernel_environment"],
        "PYTHONHASHSEED": "0",
        "ROCR_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
        "QWEN_RADIANCE_CACHE_ABI": abi,
        "VLLM_ROCM_USE_AITER": "1",
        "VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION": "1",
        **{
            f"VLLM_ROCM_USE_AITER_{n}": "0"
            for n in ("MHA", "MLA", "MOE", "LINEAR", "FP8BMM", "FP4BMM", "RMSNORM")
        },
        "NCCL_PROTO": "Simple",
        "R4D_ATTN_FP8": "0",
        "RADIANCE_WEIGHT_QUANTIZATION": "auto",
        "RADIANCE_RUN_BWTEST": "0",
        "RADIANCE_BANNER_PLAIN": "1",
        "VLLM_CACHE_ROOT": f"/cache/runtime/{abi}/vllm",
        "TORCHINDUCTOR_CACHE_DIR": f"/cache/runtime/{abi}/inductor",
        "TRITON_CACHE_DIR": f"/cache/runtime/{abi}/triton",
        "AITER_ROOT_DIR": "/cache/aiter",
        "TRITON_CACHE_AUTOTUNING": "1",
        "QWEN_OPTIMIZED_STARTUP_RECEIPT": f"/cache/runtime/{abi}/startup-{uuid.uuid4().hex}.json",
    }
    environment["RADIANCE_VERIFY_HEAD"] = (
        "1" if args.head in ("global256", "global512") else "0"
    )
    environment["RADIANCE_VERIFY_HEAD_GLOBAL_TOPK"] = (
        args.head.removeprefix("global") if args.head.startswith("global") else "512"
    )
    transfer = {
        "kv_connector": "OffloadingConnector",
        "engine_id": f"coherence-{abi[:16]}",
        "kv_role": "kv_both",
        "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": {
            "spec_name": "TieringOffloadingSpec",
            "cpu_bytes_to_use": 19327352832,
            "offload_prompt_only": False,
            "blocks_per_chunk": 1,
            "eviction_policy": "lru",
            "snapshot_settled_tail_only": True,
            "secondary_tiers": [
                {
                    "type": "qwen_chat_fs",
                    "root_dir": f"/cache/snapshots/{abi}/data",
                    "n_read_threads": 8,
                    "n_write_threads": 8,
                    "tail_flush_tokens": 8192,
                    "tail_ram_max_bytes": 6442450944,
                    "tail_ram_max_chats": 5,
                    "tail_block_limit": 15,
                    "control_directory": "/dev/shm/qwen-radiance-snapshot-control-v1",
                    "tail_status_path": "/dev/shm/qwen-radiance-snapshot-tail.json",
                }
            ],
        },
    }
    fair = {
        "qwen_fair": {
            "policy": "response_boundary",
            "tool_grace_seconds": 2,
            "max_tool_deferral_seconds": 30,
            "max_cached_chats": 2,
            "status_path": "/dev/shm/qwen-radiance-fair-public",
        }
    }
    spec = {
        "method": "dflash",
        "model": "/models/draft",
        "num_speculative_tokens": 7,
        "attention_backend": "TRITON_ATTN",
        "disable_padded_drafter_batch": True,
        "draft_sample_method": "probabilistic",
    }
    command = [
        args.engine,
        "run",
        "--rm",
        "--name",
        "vllm-coherence",
        "--ipc=host",
        "--network=host",
        "--stop-timeout",
        "90",
        "--device",
        "/dev/kfd",
        "--device",
        "/dev/dri",
        "--security-opt",
        "seccomp=unconfined",
    ]
    if args.engine == "podman":
        command += ["--group-add", "keep-groups"]
    for key, value in environment.items():
        command += ["-e", f"{key}={value}"]
    for host, container in (
        (model, "/models/target:ro"),
        (draft, "/models/draft:ro"),
        (state / "cache", "/cache"),
        (state / "patches", "/patches:ro"),
        (state / "runtime", "/qualification:ro"),
    ):
        command += ["-v", f"{host}:{container}"]
    command += [
        "-v",
        f"{state}/runtime/support/libhsa-runtime64.so.1.21.0:/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1.21.0:ro",
    ]
    command += [
        "--entrypoint",
        "/opt/vllm/bin/python",
        release()["image"],
        "/patches/bootstrap_radiance_release.py",
        "/models/target",
        "--served-model-name",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--kv-cache-dtype",
        "fp8",
        "--tensor-parallel-size",
        "1",
        "--shutdown-timeout",
        "60",
        "--gpu-memory-utilization",
        "0.97",
        "--kv-cache-memory",
        "10000000000",
        "--max-model-len",
        "253792",
        "--max-num-seqs",
        "2",
        "--max-num-batched-tokens",
        "2048",
        "--attention-backend",
        "R4D",
        "--speculative-config",
        json.dumps(spec),
        "--no-async-scheduling",
        "--language-model-only",
        "--skip-mm-profiling",
        "--scheduler-cls",
        "qwen_radiance_fair_scheduler.FairScheduler",
        "--additional-config",
        json.dumps(fair),
        "--worker-cls",
        "optimized_d7_worker.OptimizedWorker",
        "--kv-transfer-config",
        json.dumps(transfer),
        "--middleware",
        "qwen_radiance_request_guard.require_snapshot_abi",
        "--enable-prefix-caching",
        "--mamba-cache-mode",
        "align",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml",
        "--reasoning-parser",
        "qwen3",
        "--enable-per-request-metrics",
        "--enable-force-include-usage",
        "--enable-prompt-tokens-details",
        "--override-generation-config",
        '{"temperature":1,"top_p":0.95,"top_k":20}',
        "--chat-template",
        "/patches/qwen-fixed-v22.3.jinja",
        "--default-chat-template-kwargs",
        '{"reasoning_effort":"xhigh"}',
        "--compilation-config",
        '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8]}',
    ]
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "command",
        choices=("doctor", "prepare", "serve", "verify-runtime", "pi", "cache"),
        nargs="?",
    )
    parser.add_argument("--help", "-h", action="store_true")
    parser.add_argument("--state")
    parser.add_argument("--archive")
    parser.add_argument(
        "--head", choices=("global512", "global256", "full-bf16"), default="global512"
    )
    parser.add_argument("--model")
    parser.add_argument("--draft")
    parser.add_argument("--engine", choices=("podman",), default="podman")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ssh")
    parser.add_argument("--remote-state", default="~/.local/state/vllm-coherence")
    parser.add_argument("--search-extension")
    args, remainder = parser.parse_known_args(argv)
    if args.help or not args.command:
        print(HELP)
        return 0
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    if remainder and args.command not in ("pi", "cache"):
        parser.error("unexpected arguments: " + shlex.join(remainder))
    try:
        state = state_path(args.state)
        if args.command == "doctor":
            print(
                json.dumps(
                    {
                        "release": release()["version"],
                        "python": sys.version.split()[0],
                        "programs": {
                            n: bool(shutil.which(n))
                            for n in (
                                "podman",
                                "docker",
                                "uv",
                                "node",
                                "npm",
                                "git",
                                "ssh",
                                "jq",
                            )
                        },
                        "kfd_present": Path("/dev/kfd").exists(),
                        "state": str(state),
                        "gpu_opened": False,
                    },
                    indent=2,
                )
            )
        elif args.command == "prepare":
            print(json.dumps(prepare(state, args.archive, args.head), indent=2))
        elif args.command == "verify-runtime":
            manifest = verify_runtime(state)
            print(
                json.dumps(
                    {"verified_files": len(manifest["files"]), "gpu_used": False}
                )
            )
        elif args.command == "serve":
            if not args.model or not args.draft or not 1 <= args.port <= 65535:
                raise ValueError("serve requires --model, --draft and a valid --port")
            if args.dry_run:
                print(shlex.join(serving_command(args, {"abi": "DRY_RUN_ABI"})))
                return 0
            for model in (args.model, args.draft):
                if not (Path(model).expanduser() / "config.json").is_file():
                    raise ValueError(f"model directory has no config.json: {model}")
            if not shutil.which(args.engine):
                raise ValueError(f"{args.engine} is not installed")
            with socket.socket() as check:
                check.bind(("127.0.0.1", args.port))
            # No delete/reuse of a previous process's mmap. Refuse insufficient RAM.
            if shutil.disk_usage("/dev/shm").free < 19327352832:
                raise ValueError(
                    "at least 18 GiB free /dev/shm is required for the RAM tier"
                )
            with deployment_lock(state):
                existing = subprocess.run(
                    [args.engine, "container", "exists", "vllm-coherence"], check=False
                )
                if existing.returncode != 1:
                    raise ValueError(
                        "cannot start while the Coherence container exists or container inspection fails"
                    )
                connection = _prepare(state, args.archive, args.head)
                connection["port"] = args.port
                write_json(state / "connection.json", connection)
                command = serving_command(args, connection)
                return subprocess.call(command)
        elif args.command == "pi":
            from coherence_pi import launch

            return launch(args, remainder)
        elif args.command == "cache":
            connection = read_json(state / "connection.json")
            command = [
                sys.executable,
                str(ROOT / "scripts/qwen-radiance-cache"),
                "--host",
                "local",
                "--cache-root",
                connection["cache_root"],
                *remainder,
            ]
            return subprocess.call(command)
        return 0
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        print(f"coherence: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
