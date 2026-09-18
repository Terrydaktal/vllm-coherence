#!/usr/bin/env python3
"""Audit Coherence additions, leaving the exact public upstream import identifiable."""

import ast
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".npy",
    ".npz",
    ".qkv",
    ".safetensors",
    ".gguf",
    ".hsaco",
    ".so",
    ".jsonl",
}
SECRET = re.compile(
    rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)"
)


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)


def additions():
    base = git("rev-list", "--max-parents=0", "HEAD").strip()
    files = set(git("diff", "--name-only", base).splitlines())
    files.update(git("ls-files", "--others", "--exclude-standard").splitlines())
    return sorted(name for name in files if (ROOT / name).exists())


def audit():
    problems = []
    files = additions()
    for name in files:
        p = ROOT / name
        if p.is_symlink() or not p.is_file():
            problems.append((name, "indirect/non-regular file"))
            continue
        if p.suffix in FORBIDDEN_SUFFIXES or any(
            part in {"private", "sessions", ".pi", "artifacts", "node_modules", ".venv"}
            for part in p.relative_to(ROOT).parts
        ):
            problems.append((name, "private/generated artifact"))
        if p.stat().st_size > 8 * 1024**2:
            problems.append((name, "oversized source file"))
        data = p.read_bytes()
        if SECRET.search(data):
            problems.append((name, "credential/private-key pattern"))
        if p.suffix == ".py":
            try:
                ast.parse(data, filename=name)
            except SyntaxError:
                problems.append((name, "Python syntax"))
        if p.suffix == ".json":
            try:
                json.loads(data)
            except ValueError:
                problems.append((name, "JSON syntax"))
    print(json.dumps({"checked_additions": len(files), "problems": problems}, indent=2))
    return bool(problems)


if __name__ == "__main__":
    raise SystemExit(audit())
