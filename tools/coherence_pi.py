"""Patched Pi launcher with project-local history and optional SSH transport."""

import fcntl
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

from coherence_cli import (
    MODEL,
    ROOT,
    private_directory,
    read_json,
    state_path,
    write_json,
)


def remote_connection(host, remote_state):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", host):
        raise ValueError("invalid SSH host")
    program = "import pathlib,sys; print((pathlib.Path(sys.argv[1]).expanduser()/'connection.json').read_text())"
    command = shlex.join(["python3", "-c", program, remote_state])
    result = subprocess.run(
        ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    return json.loads(result.stdout)


def reserve_port(state):
    directory = state / "ports"
    private_directory(directory)
    for port in range(8012, 8112):
        lock = (directory / str(port)).open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))
            return port, lock
        except (BlockingIOError, OSError):
            lock.close()
    raise ValueError("no free forwarding port between 8012 and 8111")


def launch(args, pi_args):
    state = state_path(args.state)
    private_directory(state)
    host = args.ssh or "local"
    connection = (
        remote_connection(host, args.remote_state)
        if args.ssh
        else read_json(state / "connection.json")
    )
    if not re.fullmatch(r"[a-f0-9]{64}", connection["abi"]):
        raise ValueError("invalid backend snapshot identity")
    if connection.get("model") != MODEL:
        raise ValueError("unsupported backend model")
    if not isinstance(connection["port"], int) or not 1 <= connection["port"] <= 65535:
        raise ValueError("invalid backend port")
    runtime = state / "pi"
    binary = state / "bin/pi"
    if not binary.is_file():
        subprocess.run(
            [
                str(ROOT / "scripts/install-pi-coding-agent"),
                "--install-root",
                str(runtime),
                "--bin-dir",
                str(binary.parent),
            ],
            check=True,
        )
    # Authenticate the patched files on reuse as well as at install time.
    for patch in sorted((ROOT / "scripts").glob("patch-pi-*")):
        subprocess.run([str(patch), "--check", str(runtime / "0.84.2")], check=True)
    port, lock, tunnel = connection["port"], None, None
    try:
        if args.ssh:
            # Retry a bind race; lock ownership coordinates multiple Coherence windows.
            for _ in range(5):
                port, lock = reserve_port(state)
                tunnel = subprocess.Popen(
                    [
                        "ssh",
                        "-T",
                        "-N",
                        "-o",
                        "BatchMode=yes",
                        "-o",
                        "ExitOnForwardFailure=yes",
                        "-o",
                        "ServerAliveInterval=15",
                        "-o",
                        "ServerAliveCountMax=3",
                        "-L",
                        f"127.0.0.1:{port}:127.0.0.1:{connection['port']}",
                        host,
                    ]
                )
                time.sleep(0.15)
                if tunnel.poll() is None:
                    break
                lock.close()
                lock = None
            if tunnel.poll() is not None:
                raise ValueError("SSH tunnel could not start")
        for attempt in range(40):
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/v1/models", timeout=2
                ) as response:
                    models = json.load(response)
                if MODEL not in [m["id"] for m in models["data"]]:
                    raise ValueError("endpoint advertises another model")
                break
            except OSError:
                if attempt == 39 or (tunnel and tunnel.poll() is not None):
                    raise ValueError(
                        "backend unavailable; run coherence serve on the GPU host"
                    )
                time.sleep(0.2)
        agent = state / "agents" / str(port)
        private_directory(agent)
        model_config = read_json(ROOT / "integrations/pi/models-coherence.json")
        model_config["providers"]["qwen-r9700"]["baseUrl"] = (
            f"http://127.0.0.1:{port}/v1"
        )
        write_json(agent / "models.json", model_config)
        settings_path = agent / "settings.json"
        if not settings_path.exists():
            write_json(
                settings_path,
                {
                    "httpIdleTimeoutMs": 0,
                    "defaultTools": [
                        "read",
                        "bash",
                        "edit",
                        "write",
                        "grep",
                        "find",
                        "ls",
                        "qwen_rehydrate_tool_turn",
                    ],
                },
            )
        telemetry = (
            state
            / "telemetry"
            / hashlib.sha256((host + connection["cache_root"]).encode()).hexdigest()[
                :16
            ]
        )
        private_directory(telemetry)
        private_directory(telemetry / "clients")
        history = Path.cwd() / ".pi/sessions"
        history.mkdir(mode=0o700, parents=True, exist_ok=True)
        extensions = [
            "qwen-progress.mjs",
            "qwen-gpu-temperature.mjs",
            "qwen-tool-output-condense.mjs",
            "qwen-tool-turn-rehydrate.mjs",
            "qwen-radiance-compaction.ts",
            "qwen-radiance-cache.mjs",
        ]
        command = [
            str(binary),
            "--model",
            f"qwen-r9700/{MODEL}",
            "--session-dir",
            str(history),
            "--no-context-files",
            "--append-system-prompt",
            str(ROOT / "integrations/pi/qwen-radiance-operating-prompt.md"),
            "--no-extensions",
        ]
        for name in extensions:
            command += ["--extension", str(ROOT / "integrations/pi" / name)]
        if args.search_extension:
            extension = Path(args.search_extension).expanduser().resolve(strict=True)
            command += ["--extension", str(extension)]
        environment = {
            **os.environ,
            "PI_CODING_AGENT_DIR": str(agent),
            "PI_OFFLINE": "1",
            "QWEN_RADIANCE_CACHE_HOST": host,
            "QWEN_RADIANCE_CONTAINER": "vllm-coherence",
            "QWEN_RADIANCE_CACHE_ROOT": connection["cache_root"],
            "QWEN_RADIANCE_CACHE_ABI": connection["abi"],
            "QWEN_RADIANCE_LOCAL_PORT": str(port),
            "QWEN_RADIANCE_GPU_TEMPERATURE_STATE": str(telemetry),
            "QWEN_RADIANCE_GPU_TEMPERATURE_HELPER": str(
                ROOT / "scripts/qwen-radiance-gpu-temperature"
            ),
            "QWEN_RADIANCE_SCHEDULER_HELPER": str(
                ROOT / "scripts/qwen-radiance-scheduler-status"
            ),
        }
        print(
            f"Coherence: 127.0.0.1:{port} · snapshot {connection['abi'][:12]} · {connection['head']}",
            flush=True,
        )
        return subprocess.call([*command, *pi_args], env=environment)
    finally:
        if tunnel and tunnel.poll() is None:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel.kill()
                tunnel.wait()
        if lock:
            lock.close()
