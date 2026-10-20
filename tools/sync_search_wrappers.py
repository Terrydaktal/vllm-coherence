#!/usr/bin/env python3
"""Pin exact standalone search-wrapper runtime sources for Pi deployment.

The standalone repositories remain authoritative. This copies their runtime
files byte for byte and records a digest for every file; --check detects drift.
No user data, npm dependencies, test captures or generated executables are copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "integrations/pi/search-wrappers"


def runtime_files(source: Path) -> dict[str, bytes]:
    package = json.loads((source / "package.json").read_text())
    files = {}
    # npm's shipped directory list is the inventory. Only runtime directories
    # and attribution are pinned here; standalone tests stay in their own repo.
    for entry in package["files"]:
        name = entry.rstrip("/")
        if name in {"tests", "docs", "pi", "README.md"}:
            continue
        path = source / name
        if name == "tools":
            items = [path / builder for builder in ("build-native.mjs", "build-rust.mjs")
                     if (path / builder).exists()]
        elif name == "rust":
            # Cargo targets contain generated metadata as well as ELF binaries.
            # Pin only the locked source inventory, without walking build trees.
            items = [path / "Cargo.toml", path / "Cargo.lock", *sorted((path / "src").rglob("*"))]
        else:
            items = sorted(path.rglob("*")) if path.is_dir() else [path]
        for item in items:
            if not item.is_file() or item.is_symlink():
                continue
            relative = item.relative_to(source)
            if any(part.startswith(".") for part in relative.parts):
                continue
            # Native binaries are rebuilt from pinned source at installation,
            # never silently published from this machine's build directory.
            data = item.read_bytes()
            if data.startswith(b"\x7fELF"):
                continue
            files[str(relative)] = data
    files["package.json"] = (source / "package.json").read_bytes()
    return files


def snapshot(sources: dict[str, Path]) -> tuple[dict[str, bytes], dict]:
    all_files = {}
    packages = {}
    for name, source in sources.items():
        source = source.resolve(strict=True)
        files = runtime_files(source)
        metadata = json.loads(files["package.json"])
        revision = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        packages[name] = {"version": metadata["version"],
                          "repository": metadata["repository"]["url"],
                          "base_commit": revision,
                          "files": {path: hashlib.sha256(data).hexdigest()
                                    for path, data in sorted(files.items())}}
        all_files.update({f"{name}/{path}": data for path, data in files.items()})
    manifest = {"schema": "coherence-search-wrapper-source-v1",
                "policy": "Exact runtime source snapshot; hashes include uncommitted changes.",
                "packages": packages}
    return all_files, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grep-root", type=Path, default=Path.home() / "Dev/grep2rg")
    parser.add_argument("--find-root", type=Path, default=Path.home() / "Dev/find2fd")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files, manifest = snapshot({"grep2rg": args.grep_root, "find2fd": args.find_root})
    files["source-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    if args.check:
        mismatches = [name for name, data in files.items()
                      if not (VENDOR / name).is_file() or (VENDOR / name).read_bytes() != data]
        if mismatches:
            raise SystemExit("Pinned search-wrapper sources differ: " + ", ".join(mismatches))
    else:
        previous = VENDOR / "source-manifest.json"
        if previous.is_file():
            old = json.loads(previous.read_text())
            for package, metadata in old["packages"].items():
                for name in metadata["files"]:
                    relative = Path(package) / name
                    if str(relative) not in files:
                        # Only retire files owned by the preceding generated
                        # snapshot, never local helper builds or other files.
                        if relative.is_absolute() or ".." in relative.parts:
                            raise ValueError("Unsafe preceding snapshot path")
                        path = VENDOR / relative
                        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != metadata["files"][name]:
                            raise ValueError("Locally edited obsolete pinned source: " + str(relative))
                        path.unlink(missing_ok=True)
        for name, data in files.items():
            path = VENDOR / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            if data.startswith(b"#!"):
                path.chmod(0o755)
    print(json.dumps({"checked": args.check, "runtime_files": len(files),
                      "source_manifest_sha256": hashlib.sha256(files["source-manifest.json"]).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
