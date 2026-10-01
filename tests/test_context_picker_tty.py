"""Drive the installed Pi CLI in a PTY using an isolated synthetic session."""

from __future__ import annotations

import fcntl
import json
import os
import pty
import re
import select
import signal
import struct
import subprocess
import termios
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get("QWEN_TEST_PI_RUNTIME", Path.home() / ".local/share/qwen-r9700/pi/0.84.2"))
CLI = RUNTIME / "node_modules/@earendil-works/pi-coding-agent/dist/cli.js"
MODEL = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate"
ANSI = re.compile(rb"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


@pytest.mark.skipif(not CLI.exists(), reason="patched Pi CLI is not installed")
def test_picker_on_installed_cli_saves_only_metadata_and_survives_terminal_resize(tmp_path):
    agent = tmp_path / "agent"
    agent.mkdir()
    provider = {
        "api": "openai-completions", "baseUrl": "http://fixture.invalid/v1", "apiKey": "synthetic",
        "models": [{"id": MODEL, "name": "Synthetic context fixture", "reasoning": True,
                    "input": ["text"], "contextWindow": 253792, "maxTokens": 253792,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}],
    }
    (agent / "models.json").write_text(json.dumps({"providers": {"qwen-r9700": provider}}))
    (agent / "settings.json").write_text(json.dumps({"defaultProvider": "qwen-r9700", "defaultModel": MODEL,
                                                    "compaction": {"enabled": False}, "retry": {"enabled": False}}))
    usage = {"input": 20, "output": 10, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 30,
             "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}}
    timestamp = "2026-09-30T01:00:00.000Z"
    entries = [
        {"type": "session", "version": 3, "id": "synthetic-context-tty", "timestamp": timestamp, "cwd": str(tmp_path)},
        {"type": "message", "id": "aaaa0001", "parentId": None, "timestamp": timestamp,
         "message": {"role": "user", "content": "Synthetic obsolete request", "timestamp": 1}},
        {"type": "message", "id": "aaaa0002", "parentId": "aaaa0001", "timestamp": timestamp,
         "message": {"role": "assistant", "api": "openai-completions", "provider": "qwen-r9700", "model": MODEL,
                     "content": [{"type": "text", "text": "Synthetic answer"}], "timestamp": 2, "stopReason": "stop", "usage": usage}},
    ]
    session = tmp_path / "synthetic.jsonl"
    original_lines = [json.dumps(entry) for entry in entries]
    session.write_text("\n".join(original_lines) + "\n")
    env = {**os.environ, "PI_CODING_AGENT_DIR": str(agent), "PI_OFFLINE": "1", "TERM": "xterm-256color"}
    for key in list(env):
        if key.startswith("QWEN_RADIANCE_") or key in {"TMUX", "TMUX_PANE"}:
            del env[key]
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 35, 110, 0, 0))
    process = subprocess.Popen(["node", str(CLI), "--session", str(session), "--no-extensions", "--no-skills",
                                "--no-prompt-templates", "--no-themes", "--no-context-files", "-e",
                                str(ROOT / "integrations/pi/qwen-radiance-compaction.ts")],
                               cwd=tmp_path, env=env, stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
    os.close(slave)
    captured = bytearray()

    def wait_for(expected: bytes, seconds=8):
        since = len(captured)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], min(0.1, deadline - time.monotonic()))
            if ready:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                captured.extend(chunk)
                if expected in ANSI.sub(b"", captured[since:]):
                    return
            if process.poll() is not None:
                break
        # This output contains only this test's synthetic transcript/config.
        pytest.fail(f"Pi PTY did not show {expected!r}; returncode={process.poll()}\n" +
                    ANSI.sub(b"", captured[-12000:]).decode("utf-8", "replace"))

    try:
        wait_for(b"Synthetic answer")
        os.write(master, b"/context\r")
        wait_for(b"choose what the model remembers")
        os.write(master, b"\x1b[H")
        wait_for(b"Message 1")
        os.write(master, b" ")
        wait_for(b"1 selected")
        os.write(master, b"e")
        wait_for(b"1 pending changes")
        # Resize the real overlay before saving; the key handler must keep focus.
        fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 78, 0, 0))
        os.kill(process.pid, signal.SIGWINCH)
        os.write(master, b"\r")
        wait_for(b"Saved context selection")
        recorded = [json.loads(line) for line in session.read_text().splitlines()]
        decision = next(entry for entry in recorded if entry.get("customType") == "qwen-context-selection-v1")
        assert decision["data"]["changes"] == [{"entryId": "aaaa0001", "part": "message", "excluded": True}]
        assert all(line in session.read_text().splitlines() for line in original_lines)
        assert "Synthetic obsolete request" not in json.dumps(decision)
        os.write(master, b"/context undo\r")
        wait_for(b"Undid the last context selection")
        os.write(master, b"/quit\r")
        process.wait(timeout=5)
        assert process.returncode == 0
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        os.close(master)
