from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "scripts" / "qwen-snapshot-repair-head"
SESSION = "pi-repair-test"
BRANCH = "main"


def canonical(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def protected_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)


def protected_file(path: Path, data: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(mode)


def manifest(
    root: Path,
    generation: int,
    parent: str | None,
    payload: bytes,
    *,
    prompt_tokens: int = 112,
    replay_tokens: int = 104,
) -> str:
    relative = f"payload/generation-{generation}.bin"
    protected_file(root / relative, payload, 0o600)
    catalog = digest(f"{relative}\0{digest(payload)}\n".encode())
    evidence_records: dict[str, object] = {"offload_abi_sha256": "f" * 64}
    for name in ("target_model", "draft_model", "tokenizer", "runtime"):
        evidence = f"{name}-evidence".encode()
        path = root / "evidence" / f"{name}.json"
        protected_file(path, evidence, 0o600)
        evidence_records[name] = {
            "evidence_path": str(path),
            "evidence_sha256": digest(evidence),
            "identity": name,
        }
    config = b"{}\n"
    config_path = root / "namespace" / "config.json"
    protected_file(config_path, config, 0o600)
    document = {
        "bindings": evidence_records,
        "blobs": [
            {
                "payload_sha256": digest(payload),
                "relative_path": relative,
                "size": len(payload),
            }
        ],
        "generation": generation,
        "namespace": {
            "config_relative_path": "namespace/config.json",
            "config_sha256": digest(config),
        },
        "parent_manifest_sha256": parent,
        "request": {
            "prompt_token_ids_sha256": digest(f"prompt-{generation}".encode()),
            "prompt_tokens": prompt_tokens,
            "replay_boundary_tokens": replay_tokens,
            "request_id": f"request-{generation}",
        },
        "schema": "urn:qwen-r9700:cache-generation:p8:v1",
        "session": {"branch": BRANCH, "session_id": SESSION},
        "source_catalog_sha256": catalog,
        "summary": {"blob_count": 1, "payload_bytes": len(payload)},
    }
    manifest_digest = digest(canonical(document))
    document["manifest_sha256"] = manifest_digest
    protected_file(
        root / ".qwen-250k-cache-v1" / "manifests" / f"{manifest_digest}.json",
        canonical(document) + b"\n",
        0o400,
    )
    return manifest_digest


def fixture(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "cache"
    metadata = root / ".qwen-250k-cache-v1"
    for path in (root, metadata, metadata / "manifests", metadata / "refs"):
        protected_directory(path)
    session_refs = metadata / "refs" / SESSION
    protected_directory(session_refs)
    protected_file(metadata / "LIFECYCLE.lock", b"", 0o600)
    generation_10 = "0" * 64
    generation_11 = manifest(root, 11, generation_10, b"safe-state")
    generation_12 = manifest(root, 12, generation_11, b"poisoned-state")
    reference = {
        "branch": BRANCH,
        "generation": 12,
        "manifest_sha256": generation_12,
        "schema": "urn:qwen-r9700:cache-ref:v1",
        "session_id": SESSION,
        "updated_at": "2026-09-05T00:00:00Z",
    }
    protected_file(session_refs / "main.json", canonical(reference) + b"\n", 0o600)
    return root, generation_11, generation_12


def arguments(root: Path, source: str, current: str) -> list[str]:
    return [
        str(PROGRAM),
        "--root",
        str(root),
        "--session-id",
        SESSION,
        "--branch",
        BRANCH,
        "--expected-current-generation",
        "12",
        "--expected-current-manifest",
        current,
        "--source-generation",
        "11",
        "--source-manifest",
        source,
        "--repair-generation",
        "13",
        "--updated-at",
        "2026-09-05T01:00:00Z",
    ]


def test_repair_is_monotonic_authenticated_and_compare_and_swap_safe(tmp_path: Path) -> None:
    root, source, current = fixture(tmp_path)
    base = arguments(root, source, current)
    dry = subprocess.run(base, check=True, capture_output=True, text=True)
    plan = json.loads(dry.stdout)
    assert plan["current"]["generation"] == 12
    assert plan["source"]["generation"] == 11
    assert plan["repair"]["generation"] == 13
    assert plan["payload"]["blob_count"] == 1

    applied = subprocess.run(
        [*base, "--apply", "--expected-plan-sha256", plan["plan_sha256"]],
        check=True,
        capture_output=True,
        text=True,
    )
    complete = json.loads(applied.stdout)
    assert complete["repair_manifest_sha256"] == plan["repair"]["manifest_sha256"]
    ref_path = root / ".qwen-250k-cache-v1" / "refs" / SESSION / "main.json"
    repaired_ref = json.loads(ref_path.read_bytes())
    assert repaired_ref["generation"] == 13
    assert repaired_ref["manifest_sha256"] == complete["repair_manifest_sha256"]
    repaired_manifest = json.loads(
        (
            root
            / ".qwen-250k-cache-v1"
            / "manifests"
            / f"{complete['repair_manifest_sha256']}.json"
        ).read_bytes()
    )
    assert repaired_manifest["generation"] == 13
    assert repaired_manifest["parent_manifest_sha256"] == current
    assert repaired_manifest["blobs"][0]["relative_path"] == "payload/generation-11.bin"
    assert (root / "payload" / "generation-12.bin").read_bytes() == b"poisoned-state"


def test_repair_rejects_corrupt_source_payload_without_changing_head(tmp_path: Path) -> None:
    root, source, current = fixture(tmp_path)
    ref_path = root / ".qwen-250k-cache-v1" / "refs" / SESSION / "main.json"
    before = ref_path.read_bytes()
    payload = root / "payload" / "generation-11.bin"
    payload.write_bytes(b"corrupt")
    payload.chmod(0o600)

    result = subprocess.run(
        arguments(root, source, current), check=False, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "source payload size differs" in result.stderr
    assert ref_path.read_bytes() == before


def test_repair_rejects_a_stale_current_head(tmp_path: Path) -> None:
    root, source, current = fixture(tmp_path)
    ref_path = root / ".qwen-250k-cache-v1" / "refs" / SESSION / "main.json"
    value = json.loads(ref_path.read_bytes())
    value["generation"] = 13
    protected_file(ref_path, canonical(value) + b"\n", 0o600)

    result = subprocess.run(
        arguments(root, source, current), check=False, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "compare-and-swap head" in result.stderr


def test_repair_rejects_a_same_chain_boundary_rollback(tmp_path: Path) -> None:
    root, source, current = fixture(tmp_path)
    source_path = root / ".qwen-250k-cache-v1" / "manifests" / f"{source}.json"
    source_document = json.loads(source_path.read_bytes())
    source_document["request"]["prompt_tokens"] = 111
    source_document["request"]["replay_boundary_tokens"] = 103
    source_document.pop("manifest_sha256")
    replacement_digest = digest(canonical(source_document))
    source_document["manifest_sha256"] = replacement_digest
    replacement_path = source_path.with_name(f"{replacement_digest}.json")
    protected_file(replacement_path, canonical(source_document) + b"\n", 0o400)

    current_path = root / ".qwen-250k-cache-v1" / "manifests" / f"{current}.json"
    current_document = json.loads(current_path.read_bytes())
    current_document["parent_manifest_sha256"] = replacement_digest
    current_document.pop("manifest_sha256")
    replacement_current = digest(canonical(current_document))
    current_document["manifest_sha256"] = replacement_current
    replacement_current_path = current_path.with_name(f"{replacement_current}.json")
    protected_file(replacement_current_path, canonical(current_document) + b"\n", 0o400)
    ref_path = root / ".qwen-250k-cache-v1" / "refs" / SESSION / "main.json"
    reference = json.loads(ref_path.read_bytes())
    reference["manifest_sha256"] = replacement_current
    protected_file(ref_path, canonical(reference) + b"\n", 0o600)

    result = subprocess.run(
        arguments(root, replacement_digest, replacement_current),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "move the prompt or replay boundary backwards" in result.stderr


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX file modes and flock")
def test_program_is_executable() -> None:
    assert PROGRAM.stat().st_mode & 0o111
