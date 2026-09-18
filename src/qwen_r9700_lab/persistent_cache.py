"""Crash-safe manifests and offline garbage collection for persistent vLLM KV pages.

The production profile in this module is intentionally narrow.  It describes the
qualified 249,957-token Qwen/DFlash lane and refuses to seal any other geometry.
The filesystem tier's names are content-addressed by token-chain hash plus cache
group, but the names do not authenticate the payload bytes.  Sealed manifests
therefore bind both the vLLM key path and an independently computed SHA-256.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import sys
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

SELECTION_SCHEMA = "urn:qwen-r9700:cache-selection:p8:v1"
MANIFEST_SCHEMA = "urn:qwen-r9700:cache-generation:p8:v1"
FIXED_SLOT_PRODUCER_SCHEMA = "urn:qwen-r9700:fixed-slot-payload-producer:v1"
HISTORICAL_FIXED_SLOT_PRODUCER_SCHEMA = (
    "urn:qwen-r9700:fixed-slot-historical-payload-producer-attestation:v1"
)
FIXED_SLOT_PRODUCER_CONTRACT = (
    "qwen-fixed-slot-context-producer-v1/"
    "target=row-local-quest96-complete-causal-tail/"
    "native-b=m1-m8-common-arithmetic/"
    "gdn=fixed-slot-current-conv/"
    "draft=context-from-same-live-forward"
)
FIXED_SLOT_SUFFIX_REPAIR_SOURCE_CONTRACT = (
    "qwen-fixed-slot-suffix-repair-source-v1/"
    "checkpoint=recoverssm-align-57680/"
    "target=predecode-prefill/"
    "draft=context-from-same-live-forward/"
    "purpose=corrected-2618-token-replay"
)
HISTORICAL_FIXED_SLOT_ATTESTABLE_CONTRACTS = (
    FIXED_SLOT_PRODUCER_CONTRACT,
    FIXED_SLOT_SUFFIX_REPAIR_SOURCE_CONTRACT,
)
REF_SCHEMA = "urn:qwen-r9700:cache-ref:v1"
QUARANTINE_PLAN_SCHEMA = "urn:qwen-r9700:cache-quarantine-plan:v1"
QUARANTINE_COMPLETE_SCHEMA = "urn:qwen-r9700:cache-quarantine-complete:v1"
PURGING_SCHEMA = "urn:qwen-r9700:cache-quarantine-purging:v1"
PURGED_SCHEMA = "urn:qwen-r9700:cache-quarantine-purged:v1"
DEFERRED_SCHEMA = "urn:qwen-r9700:cache-quarantine-deferred:v1"
METADATA_DIRECTORY = ".qwen-250k-cache-v1"
FILE_MAPPER_ABI = "vllm-file-mapper-direct-v1"
LIFECYCLE_LOCK_NAME = "LIFECYCLE.lock"
MAX_METADATA_JSON_BYTES = 16 * 1024 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RUN_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$")
_PUBLISH_TEMP_RE = re.compile(r"^\.publish-[0-9a-f]{32}\.tmp$")
_MOVE_TEMP_RE = re.compile(r"^\.move-[0-9a-f]{32}\.tmp$")
_IMPORT_TEMP_RE = re.compile(r"^\.import-[0-9a-f]{32}\.tmp$")
_BLOB_RE = re.compile(
    r"^(?P<rank_root>[^/]+_[0-9a-f]{12}_r(?P<rank>[0-9]+))/"
    r"(?P<prefix1>[0-9a-f]{3})/(?P<prefix2>[0-9a-f]{2})_g"
    r"(?P<group>[0-9]+)/(?P<block_hash>[0-9a-f]{64})\.bin$"
)
_ENGINE_CMDLINE_MARKERS = (
    b"vllm serve",
    b"vllm.entrypoints.openai.api_server",
    b"VLLM::EngineCore",
)
_ENGINE_COMM_MARKERS = ("VLLM::EngineCor", "VLLM::EngineCore", "vllm")
_FICLONE = 0x40049409

FaultHook = Callable[[str], None]


class CacheSafetyError(RuntimeError):
    """Raised whenever safety or integrity cannot be proven."""


@dataclass(frozen=True)
class CacheProfile:
    """Exact page geometry retained by one sealed session generation."""

    name: str
    prompt_tokens: int
    hash_aligned_tokens: int
    replay_boundary_tokens: int
    tokens_per_hash: int
    page_tokens: int
    page_bytes: int
    rank: int
    mamba_groups: tuple[int, ...]
    full_attention_groups: tuple[int, ...]
    sliding_window_groups: tuple[int, ...]
    mamba_retained_pages: int
    sliding_window_retained_pages: int
    group_layer_names: tuple[str, ...] | None = None
    state_contract: str | None = None
    payload_producer_contract: str | None = None

    @property
    def full_attention_pages(self) -> int:
        return (self.replay_boundary_tokens + self.page_tokens - 1) // self.page_tokens

    @property
    def blob_count(self) -> int:
        return (
            len(self.full_attention_groups) * self.full_attention_pages
            + len(self.mamba_groups) * self.mamba_retained_pages
            + len(self.sliding_window_groups) * self.sliding_window_retained_pages
        )

    @property
    def payload_bytes(self) -> int:
        return self.blob_count * self.page_bytes

    @property
    def all_groups(self) -> tuple[int, ...]:
        return self.mamba_groups + self.full_attention_groups + self.sliding_window_groups


@dataclass(frozen=True)
class HistoricalFixedSlotProducer:
    """One reviewed producer identity that may attest pre-receipt payloads.

    This is deliberately an allowlist of immutable historical facts, not a
    compatibility pattern.  A new source hash or snapshot requires a new
    reviewed entry; metadata supplied by an operator cannot extend the set.
    """

    artifact_id: str
    contract: str
    original_selection_sha256: str
    original_selection_canonical_sha256: str
    runtime_evidence_sha256: str
    offload_abi_sha256: str
    source_catalog_sha256: str
    auxiliary_evidence_sha256: tuple[tuple[str, str], ...]
    critical_sources: tuple[tuple[str, str], ...]


QUALIFIED_MAMBA_LAYER_NAMES = tuple(
    f"language_model.model.layers.{layer}.linear_attn"
    for quartet in range(16)
    for layer in range(quartet * 4, quartet * 4 + 3)
)
QUALIFIED_FULL_ATTENTION_LAYER_NAMES = tuple(
    f"language_model.model.layers.{quartet * 4 + 3}.self_attn.attn" for quartet in range(16)
)
QUALIFIED_SLIDING_WINDOW_LAYER_NAMES = tuple(
    f"model.layers.{layer}.self_attn.attn" for layer in range(64, 69)
)
QUALIFIED_GROUP_LAYER_NAMES = (
    QUALIFIED_MAMBA_LAYER_NAMES
    + QUALIFIED_FULL_ATTENTION_LAYER_NAMES
    + QUALIFIED_SLIDING_WINDOW_LAYER_NAMES
)


PRODUCTION_PROFILE = CacheProfile(
    name="qwen-dflash-agent262-p8-249957-v1",
    prompt_tokens=249_957,
    hash_aligned_tokens=249_952,
    replay_boundary_tokens=249_944,
    tokens_per_hash=8,
    page_tokens=1_648,
    page_bytes=3_375_104,
    rank=0,
    mamba_groups=tuple(range(48)),
    full_attention_groups=tuple(range(48, 64)),
    sliding_window_groups=tuple(range(64, 69)),
    mamba_retained_pages=2,
    sliding_window_retained_pages=5,
    group_layer_names=QUALIFIED_GROUP_LAYER_NAMES,
)


FIXED_SLOT_60K_PROFILE = CacheProfile(
    name="qwen-dflash-fixed-slot-p8-60298-v1",
    prompt_tokens=60_298,
    hash_aligned_tokens=60_296,
    replay_boundary_tokens=60_288,
    tokens_per_hash=8,
    page_tokens=1_648,
    page_bytes=3_375_104,
    rank=0,
    mamba_groups=tuple(range(48)),
    full_attention_groups=tuple(range(48, 64)),
    sliding_window_groups=tuple(range(64, 69)),
    mamba_retained_pages=1,
    sliding_window_retained_pages=5,
    group_layer_names=QUALIFIED_GROUP_LAYER_NAMES,
    state_contract="fixed-slot-fp32-gdn-conv-v1",
    payload_producer_contract=FIXED_SLOT_PRODUCER_CONTRACT,
)


FIXED_SLOT_60K_SUFFIX_REPAIR_PROFILE = CacheProfile(
    name="qwen-dflash-fixed-slot-suffix-repair-p8-60298-v1",
    prompt_tokens=60_298,
    hash_aligned_tokens=60_296,
    replay_boundary_tokens=57_680,
    tokens_per_hash=8,
    page_tokens=1_648,
    page_bytes=3_375_104,
    rank=0,
    mamba_groups=tuple(range(48)),
    full_attention_groups=tuple(range(48, 64)),
    sliding_window_groups=tuple(range(64, 69)),
    mamba_retained_pages=1,
    sliding_window_retained_pages=5,
    group_layer_names=QUALIFIED_GROUP_LAYER_NAMES,
    state_contract="fixed-slot-fp32-gdn-conv-v1",
    payload_producer_contract=FIXED_SLOT_SUFFIX_REPAIR_SOURCE_CONTRACT,
)


HISTORICAL_FIXED_SLOT_PRODUCERS = (
    HistoricalFixedSlotProducer(
        artifact_id="fixed-slot-current-conv-v13e",
        contract=FIXED_SLOT_PRODUCER_CONTRACT,
        original_selection_sha256=(
            "b4afecccbba5811bf5e4517fc1964ea3fa3c59bcb46799c9cf230c284188aa73"
        ),
        original_selection_canonical_sha256=(
            "39da4e500587dc2fb47350a1cc5457395a2b75038b204b8d49ff9cbe1cd40d07"
        ),
        runtime_evidence_sha256=(
            "27bb42ace4740f9b44f0f69ddef6dfbfe58575e3abf55ee2ddacb0b3f6c11eee"
        ),
        offload_abi_sha256=("acca22384353e045740a4a5d7318a52fcc6ac27a21819de1b9fcf915614790b1"),
        source_catalog_sha256=("7a568a87a882a208d66d38f76507316a1d77e82137a29e939b1c4a48b4420f73"),
        auxiliary_evidence_sha256=(
            (
                "binding-sha256.txt",
                "e4ea2069b5a183cb2fc3df0c05c13d0f13219792e3b0a174044889bd86b80e78",
            ),
            (
                "dry-run.txt",
                "46f42f26ee4cd2231542f9ee2fb8c8086db355412da8c395679aece77a8a9e77",
            ),
            (
                "server.log",
                "bc243822e379c3d8be200b8783b697639010c1af5b9f667294c85df6ff92e9d4",
            ),
        ),
        critical_sources=(
            (
                "vllm/distributed/kv_transfer/kv_connector/v1/offloading/"
                "qwen_persistent_selection.py",
                "79964a5b695c639183cb7289d3bf75017e6f95610efb023589552f8ad1adf0c6",
            ),
            (
                "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py",
                "847dc1b54d9f1c394ab09858c3f3624331d7655b47fc2a77425d43191a0897e6",
            ),
            (
                "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
                "3d3d41ad168af343c9eec4bee96246a9aec4bd0e7af7590c69f9fcc55eb56dd3",
            ),
            (
                "vllm/model_executor/layers/mamba/mamba_utils.py",
                "b1e57f7f034996be4ed477224141d4caa7df76ad99f558beb71524d048d5cb87",
            ),
            (
                "vllm/model_executor/models/qwen3_dflash2.py",
                "0f8be7dc032134a25a6bb8341dce5f0391e1ab26a816ab85e003249911f2a1ef",
            ),
            (
                "vllm/v1/attention/backends/quest_vllm_attention.py",
                "378e1a1c66192f6d2790516ffd5822f6df48ea5c122d5cc3d829957bebec7edf",
            ),
            (
                "vllm/v1/attention/backends/rocm_attn.py",
                "219611b685603477a1c8cdf9a0878507e06fc8a82f98c73eab9ebb544c1494c8",
            ),
            (
                "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py",
                "bb2975ba2ee6a442c6f9748026b644bb01a26b8181da473c2a1931cae0661ca5",
            ),
            (
                "vllm/v1/worker/gpu/spec_decode/rejection_sampler.py",
                "911de5c8a4477d2823fd54779aabebcfa15fbd1d5bc433b27c6a22f94e8f8389",
            ),
            (
                "vllm/v1/worker/gpu_model_runner.py",
                "c390e485ce1149406d1db5b2473766169f8bc87cddff2eec4e327541979f791f",
            ),
            (
                "vllm/v1/worker/mamba_utils.py",
                "ea6f828efd26da9b66d4fa6c966701184cee6c37a0e2ac46ac75bf264ba0e130",
            ),
        ),
    ),
    HistoricalFixedSlotProducer(
        artifact_id="fixed-slot-align-57680-suffix-repair-a2-v2",
        contract=FIXED_SLOT_SUFFIX_REPAIR_SOURCE_CONTRACT,
        original_selection_sha256=(
            "54378bb5b54c16680c9509323d54b659c1ac2c0035eb4679d6168c776e973c54"
        ),
        original_selection_canonical_sha256=(
            "7e5bc4b6516fd40d313772f06762320de148ce784dfd54c36db07e1ca063e3cd"
        ),
        runtime_evidence_sha256=(
            "ba1d3241b193f2addc0fab8d1d138196256bc375d4e0e24465c9555c41621858"
        ),
        offload_abi_sha256=("7be3f2873e572c8098eb335996c8973a51aa9ea291bb2066dc3975a0421eb705"),
        source_catalog_sha256=("151ab24e22836d4a13bccab50e4ca16093122c0fb162cd9782c293a344d5fe6f"),
        auxiliary_evidence_sha256=(
            (
                "base-runtime-evidence.json",
                "f017488f13ff8a074ffcfaba901fee3eb8cefc86c36abf94403d69771c762ca6",
            ),
            (
                "build-evidence.json",
                "eb89bfb840097a4cf7e9ebb334eb360ac9ee3e6980d01b3e1c5749b5fdd5d31d",
            ),
            (
                "chain-manifest.json",
                "61507af4b7a193508ea2dbfef67610047d0834ea8fdc91fdf4c7f095ba7bdfb2",
            ),
            (
                "launcher.out",
                "f4bc9bd6748ab1a1b910d51831d30c613c9703a3773c506495366ae84c4d27e8",
            ),
            (
                "offload-abi-evidence.json",
                "7be3f2873e572c8098eb335996c8973a51aa9ea291bb2066dc3975a0421eb705",
            ),
            (
                "rounds.jsonl",
                "e21e52b369ab3069c1b2ce24f9c054306d0aacbc68c278d4ddfb83522e30d54f",
            ),
            (
                "runtime-evidence.json",
                "ba1d3241b193f2addc0fab8d1d138196256bc375d4e0e24465c9555c41621858",
            ),
            (
                "server.log",
                "c4988df39ce52c414bfe7dcbc8add58ab98538c2632726342c8c01dbd6b6a14e",
            ),
            (
                "source-catalog.tsv",
                "151ab24e22836d4a13bccab50e4ca16093122c0fb162cd9782c293a344d5fe6f",
            ),
        ),
        critical_sources=(
            (
                "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py",
                "3d18ca7f690cdc426a58c04e844a8aa8c5448d15e392a21e1c2d6ffa478be160",
            ),
            (
                "vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py",
                "055d664f9bc845b30b3f86ed1ac1529889d2bcb64fc55f2988c99bd0437f4e71",
            ),
            (
                "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
                "3c63017f8e24b341b068b00a83121dc0f726f17b6f210b13239ab58282692ac9",
            ),
            (
                "vllm/model_executor/models/qwen3_dflash2.py",
                "0f8be7dc032134a25a6bb8341dce5f0391e1ab26a816ab85e003249911f2a1ef",
            ),
            (
                "vllm/v1/attention/backends/quest_vllm_attention.py",
                "c0e13cda00f9d69c360c05ae81b862b118148314410b1355554f3260428c35ea",
            ),
            (
                "vllm/v1/attention/backends/rocm_attn.py",
                "c27f7131b66d3c284d8f1fa826cfc5058b94823b1fb708d6d9345a35ae1ed36b",
            ),
            (
                "vllm/v1/worker/gpu/model_runner.py",
                "3d23c01d84ccd37d7fcc5defa9d7912e3f67db6922d7532815629d36cdca16d5",
            ),
            (
                "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py",
                "bb2975ba2ee6a442c6f9748026b644bb01a26b8181da473c2a1931cae0661ca5",
            ),
            (
                "vllm/v1/worker/gpu/spec_decode/rejection_sampler.py",
                "911de5c8a4477d2823fd54779aabebcfa15fbd1d5bc433b27c6a22f94e8f8389",
            ),
            (
                "vllm/v1/worker/mamba_utils.py",
                "d535c85b70edb4a297fafa032535057a68bc0a2f463173fa89187da48edad4dd",
            ),
        ),
    ),
)


PROFILES = {
    "250k": PRODUCTION_PROFILE,
    "fixed-slot-60k": FIXED_SLOT_60K_PROFILE,
    "fixed-slot-60k-suffix-repair": FIXED_SLOT_60K_SUFFIX_REPAIR_PROFILE,
}


@dataclass(frozen=True)
class BlobSpec:
    relative_path: str
    group_id: int
    role: str
    page_ordinal: int
    block_hash: str


@dataclass(frozen=True)
class BlobRecord:
    relative_path: str
    group_id: int
    role: str
    page_ordinal: int
    block_hash: str
    size: int
    payload_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "block_hash": self.block_hash,
            "group_id": self.group_id,
            "page_ordinal": self.page_ordinal,
            "payload_sha256": self.payload_sha256,
            "relative_path": self.relative_path,
            "role": self.role,
            "size": self.size,
        }


@dataclass(frozen=True)
class ActiveState:
    refs: tuple[dict[str, Any], ...]
    manifests: tuple[dict[str, Any], ...]
    blobs: Mapping[str, BlobRecord]
    logical_blob_count: int
    logical_payload_bytes: int


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _pretty_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CacheSafetyError(f"{label} must be a lowercase SHA-256")
    return value


def _require_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise CacheSafetyError(f"{label} is not a safe identifier")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise CacheSafetyError(f"{label} keys differ: missing={missing}, extra={extra}")


def _as_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise CacheSafetyError(f"{label} must be a JSON object")
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CacheSafetyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_symlink_components(path: Path) -> None:
    # Resolving here would follow the very symlinks this guard must detect.
    absolute = Path(os.path.abspath(path))  # noqa: PTH100
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise CacheSafetyError(f"symlink path component is forbidden: {current}")


def _reject_group_other_write(info: os.stat_result, path: Path) -> None:
    mode = stat.S_IMODE(info.st_mode)
    if mode & 0o022:
        raise CacheSafetyError(
            f"group/other-writable mode {mode:04o} is forbidden for cache lifecycle: {path}"
        )


def _guard_root(root: Path, *, expected_uid: int | None = None) -> Path:
    uid = os.getuid() if expected_uid is None else expected_uid
    # Keep the unresolved spelling until every component has been lstat'd.
    root = Path(os.path.abspath(root.expanduser()))  # noqa: PTH100
    if root == Path(root.anchor) or len(root.parts) < 4:
        raise CacheSafetyError(f"refusing dangerously broad cache root: {root}")
    _reject_symlink_components(root)
    try:
        info = root.lstat()
    except FileNotFoundError as error:
        raise CacheSafetyError(f"cache root does not exist: {root}") from error
    if not stat.S_ISDIR(info.st_mode):
        raise CacheSafetyError(f"cache root is not a directory: {root}")
    if info.st_uid != uid:
        raise CacheSafetyError(
            f"cache root owner {info.st_uid} does not match required uid {uid}: {root}"
        )
    _reject_group_other_write(info, root)
    return root


def _guard_owned_directory(path: Path, uid: int) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise CacheSafetyError(f"required directory is missing: {path}") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise CacheSafetyError(f"expected a real directory, not a link: {path}")
    if info.st_uid != uid:
        raise CacheSafetyError(f"directory has wrong owner: {path}")
    _reject_group_other_write(info, path)
    return info


def _guard_owned_chain(root: Path, directory: Path, uid: int) -> None:
    try:
        directory.relative_to(root)
    except ValueError as error:
        raise CacheSafetyError(f"directory escapes guarded root: {directory}") from error
    current = directory
    while True:
        _guard_owned_directory(current, uid)
        if current == root:
            break
        current = current.parent


def _lstat_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _safe_open_regular(
    path: Path,
    uid: int,
    *,
    allowed_nlinks: tuple[int, ...] = (1,),
) -> tuple[int, os.stat_result]:
    _reject_symlink_components(path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise CacheSafetyError(f"cannot safely open regular file {path}: {error}") from error
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise CacheSafetyError(f"not a regular file: {path}")
        if info.st_uid != uid:
            raise CacheSafetyError(f"file has wrong owner: {path}")
        if info.st_nlink not in allowed_nlinks:
            raise CacheSafetyError(f"file has unexpected hard links ({info.st_nlink}): {path}")
        _reject_group_other_write(info, path)
        return fd, info
    except Exception:
        os.close(fd)
        raise


def _hash_fd(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _hash_regular_file(
    path: Path,
    uid: int,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    durable: bool = False,
    allowed_nlinks: tuple[int, ...] = (1,),
) -> tuple[int, str]:
    fd, before = _safe_open_regular(path, uid, allowed_nlinks=allowed_nlinks)
    try:
        if expected_size is not None and before.st_size != expected_size:
            raise CacheSafetyError(
                f"wrong payload size for {path}: {before.st_size}, expected {expected_size}"
            )
        with os.fdopen(fd, "rb", closefd=False) as handle:
            digest = _hash_fd(handle)
        if durable:
            # The pinned vLLM writer atomically renames but does not fsync.  A
            # sealed generation must force every payload to stable storage before
            # its durable manifest/ref can claim crash recovery.
            os.fsync(fd)
        after = os.fstat(fd)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise CacheSafetyError(f"file changed while hashing: {path}")
        if expected_sha256 is not None and digest != expected_sha256:
            raise CacheSafetyError(
                f"payload SHA-256 mismatch for {path}: {digest}, expected {expected_sha256}"
            )
        return before.st_size, digest
    finally:
        os.close(fd)


def _read_json(
    path: Path,
    uid: int,
    *,
    max_bytes: int = MAX_METADATA_JSON_BYTES,
) -> dict[str, Any]:
    fd, info = _safe_open_regular(path, uid)
    try:
        if info.st_size > max_bytes:
            raise CacheSafetyError(f"JSON file exceeds {max_bytes} bytes: {path}")
        chunks: list[bytes] = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                raise CacheSafetyError(f"short read while loading JSON: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.fstat(fd).st_size != info.st_size:
            raise CacheSafetyError(f"JSON file changed while reading: {path}")
    finally:
        os.close(fd)
    try:
        value = json.loads(b"".join(chunks), object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CacheSafetyError(f"invalid JSON in {path}: {error}") from error
    return _as_object(value, str(path))


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mkdir_owned(path: Path, root: Path, uid: int) -> None:
    if path == root:
        _guard_owned_directory(path, uid)
        return
    try:
        path.relative_to(root)
    except ValueError as error:
        raise CacheSafetyError(f"directory escapes cache root: {path}") from error
    parent = path.parent
    _mkdir_owned(parent, root, uid)
    try:
        path.mkdir(mode=0o700)
        _fsync_directory(parent)
    except FileExistsError:
        pass
    _guard_owned_directory(path, uid)


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise CacheSafetyError("short write while publishing metadata")
        offset += written


def _write_temp_file(parent: Path, payload: bytes, mode: int, uid: int) -> Path:
    _guard_owned_directory(parent, uid)
    temporary = parent / f".publish-{uuid.uuid4().hex}.tmp"
    fd = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        mode,
    )
    try:
        _write_all(fd, payload)
        os.fsync(fd)
        os.fchmod(fd, mode)
    except Exception:
        os.close(fd)
        temporary.unlink(missing_ok=True)
        raise
    os.close(fd)
    return temporary


def _publish_immutable(path: Path, document: dict[str, Any], root: Path, uid: int) -> None:
    _mkdir_owned(path.parent, root, uid)
    payload = _pretty_bytes(document)
    try:
        path.lstat()
    except FileNotFoundError:
        pass
    else:
        _, existing_sha = _hash_regular_file(path, uid)
        if existing_sha != _sha256_bytes(payload):
            raise CacheSafetyError(f"immutable metadata collision: {path}")
        return
    temporary = _write_temp_file(path.parent, payload, 0o400, uid)
    try:
        # All writers hold the metadata flock.  Same-directory rename avoids the
        # hard-link crash window where a published file could retain link count 2.
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_replace(path: Path, document: dict[str, Any], root: Path, uid: int) -> None:
    _mkdir_owned(path.parent, root, uid)
    try:
        existing = path.lstat()
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if not stat.S_ISREG(existing.st_mode) or existing.st_uid != uid or existing.st_nlink != 1:
            raise CacheSafetyError(f"refusing to replace unsafe metadata path: {path}")
        _reject_group_other_write(existing, path)
    temporary = _write_temp_file(path.parent, _pretty_bytes(document), 0o600, uid)
    try:
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _with_document_id(document: dict[str, Any], field: str) -> dict[str, Any]:
    if field in document:
        raise CacheSafetyError(f"document already contains {field}")
    result = dict(document)
    result[field] = _sha256_bytes(_canonical_bytes(document))
    return result


def _verify_document_id(document: dict[str, Any], field: str, label: str) -> str:
    claimed = _require_sha256(document.get(field), f"{label}.{field}")
    unsigned = dict(document)
    del unsigned[field]
    actual = _sha256_bytes(_canonical_bytes(unsigned))
    if actual != claimed:
        raise CacheSafetyError(f"{label} {field} mismatch: {claimed} != {actual}")
    return claimed


def _metadata_root(root: Path) -> Path:
    return root / METADATA_DIRECTORY


def _open_lifecycle_lock(root: Path, uid: int) -> tuple[int, Path]:
    metadata = _metadata_root(root)
    _mkdir_owned(metadata, root, uid)
    path = metadata / LIFECYCLE_LOCK_NAME
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != uid or info.st_nlink != 1:
            raise CacheSafetyError(f"unsafe metadata lock: {path}")
        _reject_group_other_write(info, path)
        return fd, path
    except Exception:
        os.close(fd)
        raise


@contextmanager
def lifecycle_shared_lock(
    root: Path,
    *,
    expected_uid: int | None = None,
) -> Iterable[Path]:
    """Hold the lock a launcher must retain for the complete engine lifetime."""

    uid = os.getuid() if expected_uid is None else expected_uid
    root = _guard_root(root, expected_uid=uid)
    fd, path = _open_lifecycle_lock(root, uid)
    locked = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CacheSafetyError("cache lifecycle is exclusively locked") from error
        locked = True
        yield path
    finally:
        if locked:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@contextmanager
def _exclusive_lock(root: Path, uid: int) -> Iterable[None]:
    fd, path = _open_lifecycle_lock(root, uid)
    locked = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CacheSafetyError(
                f"cache lifecycle lock is held by a launcher or another operation: {path}"
            ) from error
        locked = True
        yield
    finally:
        if locked:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _validate_binding_record(name: str, value: Any) -> dict[str, Any]:
    binding = _as_object(value, f"bindings.{name}")
    _require_exact_keys(
        binding,
        {"evidence_path", "evidence_sha256", "identity"},
        f"bindings.{name}",
    )
    identity = binding["identity"]
    if not isinstance(identity, str) or not identity or "\x00" in identity:
        raise CacheSafetyError(f"bindings.{name}.identity must be a nonempty string")
    evidence_text = binding["evidence_path"]
    if not isinstance(evidence_text, str):
        raise CacheSafetyError(f"bindings.{name}.evidence_path must be a string")
    evidence_path = Path(evidence_text)
    if not evidence_path.is_absolute():
        raise CacheSafetyError(f"bindings.{name}.evidence_path must be absolute")
    expected = _require_sha256(binding["evidence_sha256"], f"bindings.{name}.evidence_sha256")
    return {
        "evidence_path": str(evidence_path),
        "evidence_sha256": expected,
        "identity": identity,
    }


def _validate_binding(name: str, value: Any, uid: int) -> dict[str, Any]:
    binding = _validate_binding_record(name, value)
    evidence_path = Path(binding["evidence_path"])
    _reject_symlink_components(evidence_path)
    _hash_regular_file(
        evidence_path,
        uid,
        expected_sha256=binding["evidence_sha256"],
    )
    return binding


def _validate_bindings_record(value: Any) -> dict[str, Any]:
    bindings = _as_object(value, "bindings")
    _require_exact_keys(
        bindings,
        {"draft_model", "offload_abi_sha256", "runtime", "target_model", "tokenizer"},
        "bindings",
    )
    result: dict[str, Any] = {
        name: _validate_binding_record(name, bindings[name])
        for name in ("runtime", "target_model", "draft_model", "tokenizer")
    }
    result["offload_abi_sha256"] = _require_sha256(
        bindings["offload_abi_sha256"], "bindings.offload_abi_sha256"
    )
    return result


def _validate_bindings(value: Any, uid: int) -> dict[str, Any]:
    bindings = _validate_bindings_record(value)
    return {
        **{
            name: _validate_binding(name, bindings[name], uid)
            for name in ("runtime", "target_model", "draft_model", "tokenizer")
        },
        "offload_abi_sha256": bindings["offload_abi_sha256"],
    }


def _historical_producer_by_artifact_id(artifact_id: Any) -> HistoricalFixedSlotProducer:
    if not isinstance(artifact_id, str):
        raise CacheSafetyError("historical fixed-slot artifact_id must be a string")
    matches = tuple(
        producer
        for producer in HISTORICAL_FIXED_SLOT_PRODUCERS
        if producer.artifact_id == artifact_id
    )
    if len(matches) != 1:
        raise CacheSafetyError(f"unqualified historical fixed-slot producer: {artifact_id!r}")
    return matches[0]


def _historical_producer_by_selection_sha256(
    selection_sha256: str,
) -> HistoricalFixedSlotProducer:
    matches = tuple(
        producer
        for producer in HISTORICAL_FIXED_SLOT_PRODUCERS
        if producer.original_selection_sha256 == selection_sha256
    )
    if len(matches) != 1:
        raise CacheSafetyError(
            "original fixed-slot selection is not in the reviewed historical producer allowlist"
        )
    return matches[0]


def _validate_historical_runtime_evidence(
    *,
    bindings: Mapping[str, Any],
    producer: HistoricalFixedSlotProducer,
    uid: int,
) -> None:
    """Authenticate the complete historical runtime plus reviewed critical members."""

    runtime_binding = _as_object(bindings.get("runtime"), "bindings.runtime")
    if runtime_binding.get("evidence_sha256") != producer.runtime_evidence_sha256:
        raise CacheSafetyError("historical producer runtime evidence is not the qualified artifact")
    if bindings.get("offload_abi_sha256") != producer.offload_abi_sha256:
        raise CacheSafetyError(
            "historical producer offload ABI differs from the qualified artifact"
        )
    evidence_path = Path(runtime_binding["evidence_path"])
    evidence = _read_json(evidence_path, uid)
    _require_exact_keys(
        evidence,
        {
            "identity",
            "kind",
            "offload_abi_evidence_path",
            "offload_abi_sha256",
            "python_implementation",
            "python_version",
            "schema",
            "sources",
            "vllm_version",
        },
        "historical runtime evidence",
    )
    if (
        evidence["schema"] != "urn:qwen-r9700:runtime-binding-evidence:v1"
        or evidence["kind"] != "runtime"
        or evidence["identity"] != runtime_binding["identity"]
        or evidence["offload_abi_sha256"] != producer.offload_abi_sha256
    ):
        raise CacheSafetyError("historical runtime evidence identity or ABI mismatch")
    sources = evidence["sources"]
    if not isinstance(sources, list):
        raise CacheSafetyError("historical runtime evidence sources must be a list")
    source_hashes: dict[str, str] = {}
    for index, raw_source in enumerate(sources):
        source = _as_object(raw_source, f"historical runtime sources[{index}]")
        _require_exact_keys(
            source,
            {"name", "sha256", "size"},
            f"historical runtime sources[{index}]",
        )
        name = source["name"]
        if not isinstance(name, str) or not name or "\x00" in name:
            raise CacheSafetyError("historical runtime source name is invalid")
        if name in source_hashes:
            raise CacheSafetyError(f"duplicate historical runtime source: {name}")
        source_hashes[name] = _require_sha256(source["sha256"], f"historical runtime source {name}")
        if (
            not isinstance(source["size"], int)
            or isinstance(source["size"], bool)
            or source["size"] < 0
        ):
            raise CacheSafetyError(f"historical runtime source size is invalid: {name}")
    for name, expected_sha256 in producer.critical_sources:
        if source_hashes.get(name) != expected_sha256:
            raise CacheSafetyError(f"historical critical source mismatch: {name}")


def _validate_payload_provenance(
    value: Any,
    *,
    profile: CacheProfile,
    bindings: Mapping[str, Any],
    request: Mapping[str, Any],
    captured_at: str,
    source_catalog_sha256: str,
    label: str,
    uid: int | None = None,
    original_selection_canonical_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind payload bytes to the live request and semantics that produced them.

    Runtime/model hashes describe the code currently loading a snapshot.  They
    do not prove that existing target, recurrent, and draft payloads were made
    by that code.  This independently content-addressed receipt is emitted by
    the drained source request and must travel unchanged into the generation
    manifest.  It prevents a metadata-only reseal from silently relabelling old
    DFlash state as output of a corrected producer.
    """

    expected_contract = profile.payload_producer_contract
    if expected_contract is None:
        raise CacheSafetyError(f"{label} is forbidden for a profile without producer semantics")
    provenance = _as_object(value, label)
    schema = provenance.get("schema")
    live_schema = schema == FIXED_SLOT_PRODUCER_SCHEMA
    historical_schema = schema == HISTORICAL_FIXED_SLOT_PRODUCER_SCHEMA
    if not live_schema and not historical_schema:
        raise CacheSafetyError(f"unsupported fixed-slot payload producer schema in {label}")
    document_id_field = "receipt_sha256" if live_schema else "attestation_sha256"
    expected_keys = {
        "bindings_sha256",
        "captured_at",
        "contract",
        "prompt_token_ids_sha256",
        "request_id",
        "schema",
        "source_catalog_sha256",
        document_id_field,
    }
    if historical_schema:
        expected_keys |= {
            "artifact_id",
            "auxiliary_evidence_sha256",
            "critical_sources",
            "original_selection_canonical_sha256",
            "original_selection_sha256",
            "runtime_evidence_sha256",
        }
    _require_exact_keys(provenance, expected_keys, label)
    if provenance["contract"] != expected_contract:
        raise CacheSafetyError(f"fixed-slot payload producer contract drift in {label}")
    _verify_document_id(provenance, document_id_field, label)
    expected_bindings_sha256 = _sha256_bytes(_canonical_bytes(bindings))
    if provenance["bindings_sha256"] != expected_bindings_sha256:
        raise CacheSafetyError(f"fixed-slot payload producer bindings differ in {label}")
    if provenance["source_catalog_sha256"] != source_catalog_sha256:
        raise CacheSafetyError(f"fixed-slot payload producer catalog differs in {label}")
    if provenance["request_id"] != request["request_id"]:
        raise CacheSafetyError(f"fixed-slot payload producer request differs in {label}")
    if provenance["prompt_token_ids_sha256"] != request["prompt_token_ids_sha256"]:
        raise CacheSafetyError(f"fixed-slot payload producer prompt differs in {label}")
    if provenance["captured_at"] != captured_at:
        raise CacheSafetyError(f"fixed-slot payload producer timestamp differs in {label}")
    if historical_schema:
        if uid is None:
            raise CacheSafetyError("historical fixed-slot provenance requires owner validation")
        producer = _historical_producer_by_artifact_id(provenance["artifact_id"])
        expected_historical = {
            "auxiliary_evidence_sha256": dict(producer.auxiliary_evidence_sha256),
            "critical_sources": dict(producer.critical_sources),
            "original_selection_canonical_sha256": (producer.original_selection_canonical_sha256),
            "original_selection_sha256": producer.original_selection_sha256,
            "runtime_evidence_sha256": producer.runtime_evidence_sha256,
        }
        for field, expected in expected_historical.items():
            if provenance[field] != expected:
                raise CacheSafetyError(f"historical fixed-slot {field} mismatch in {label}")
        if producer.contract != expected_contract:
            raise CacheSafetyError("historical producer is not compatible with this cache profile")
        if (
            original_selection_canonical_sha256 is not None
            and original_selection_canonical_sha256 != producer.original_selection_canonical_sha256
        ):
            raise CacheSafetyError(
                "historical provenance does not describe this selection document"
            )
        _validate_historical_runtime_evidence(
            bindings=bindings,
            producer=producer,
            uid=uid,
        )
    return dict(provenance)


def _validate_selection_document(
    selection_value: Any,
    uid: int,
    profile: CacheProfile,
    *,
    allow_missing_provenance: bool = False,
) -> dict[str, Any]:
    selection = _as_object(selection_value, "selection")
    expected_selection_keys = {
        "bindings",
        "block_hashes",
        "branch",
        "captured_at",
        "generation",
        "namespace",
        "parent_manifest_sha256",
        "request",
        "schema",
        "session_id",
    }
    if profile.payload_producer_contract is not None:
        fixed_slot_keys = expected_selection_keys | {"source_catalog_sha256"}
        if not allow_missing_provenance:
            fixed_slot_keys.add("payload_provenance")
        _require_exact_keys(
            selection,
            fixed_slot_keys,
            "selection",
        )
        _require_sha256(selection["source_catalog_sha256"], "source_catalog_sha256")
    else:
        _require_exact_keys(selection, expected_selection_keys, "selection")
    if selection["schema"] != SELECTION_SCHEMA:
        raise CacheSafetyError(f"unsupported selection schema: {selection['schema']!r}")
    _require_identifier(selection["session_id"], "session_id")
    _require_identifier(selection["branch"], "branch")
    generation = selection["generation"]
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
        raise CacheSafetyError("generation must be a positive integer")
    parent = selection["parent_manifest_sha256"]
    if parent is not None:
        _require_sha256(parent, "parent_manifest_sha256")
    captured_at = selection["captured_at"]
    if not isinstance(captured_at, str) or not captured_at.endswith("Z"):
        raise CacheSafetyError("captured_at must be an explicit UTC timestamp ending in Z")

    request = _as_object(selection["request"], "request")
    _require_exact_keys(
        request,
        {
            "hash_aligned_tokens",
            "prompt_token_ids_sha256",
            "prompt_tokens",
            "replay_boundary_tokens",
            "request_id",
            "store_drained",
        },
        "request",
    )
    expected_request = {
        "prompt_tokens": profile.prompt_tokens,
        "hash_aligned_tokens": profile.hash_aligned_tokens,
        "replay_boundary_tokens": profile.replay_boundary_tokens,
        "store_drained": True,
    }
    for key, expected in expected_request.items():
        if request[key] != expected:
            raise CacheSafetyError(f"request.{key}={request[key]!r}, expected {expected!r}")
    _require_identifier(request["request_id"], "request.request_id")
    _require_sha256(request["prompt_token_ids_sha256"], "request.prompt_token_ids_sha256")

    namespace = _as_object(selection["namespace"], "namespace")
    _require_exact_keys(
        namespace,
        {"config_relative_path", "config_sha256", "rank"},
        "namespace",
    )
    relative_config = _validate_relative_path(
        namespace["config_relative_path"], "namespace.config_relative_path"
    )
    if PurePosixPath(relative_config).name != "config.json":
        raise CacheSafetyError("namespace config path must end in config.json")
    _require_sha256(namespace["config_sha256"], "namespace.config_sha256")
    if namespace["rank"] != profile.rank:
        raise CacheSafetyError(f"namespace rank must be {profile.rank}")

    block_hashes = selection["block_hashes"]
    if not isinstance(block_hashes, list) or len(block_hashes) != profile.full_attention_pages:
        raise CacheSafetyError(
            f"block_hashes must contain exactly {profile.full_attention_pages} page hashes"
        )
    for index, block_hash in enumerate(block_hashes):
        _require_sha256(block_hash, f"block_hashes[{index}]")
    if len(set(block_hashes)) != len(block_hashes):
        raise CacheSafetyError("block_hashes must be unique across physical page ordinals")

    original_selection = dict(selection)
    original_selection.pop("payload_provenance", None)
    original_selection_canonical_sha256 = _sha256_bytes(_canonical_bytes(original_selection))
    selection = dict(selection)
    selection["bindings"] = _validate_bindings(selection["bindings"], uid)
    if profile.payload_producer_contract is not None and "payload_provenance" in selection:
        selection["payload_provenance"] = _validate_payload_provenance(
            selection["payload_provenance"],
            profile=profile,
            bindings=selection["bindings"],
            request=request,
            captured_at=captured_at,
            source_catalog_sha256=selection["source_catalog_sha256"],
            label="selection.payload_provenance",
            uid=uid,
            original_selection_canonical_sha256=original_selection_canonical_sha256,
        )
    return selection


def _load_and_validate_selection(
    selection_path: Path,
    uid: int,
    profile: CacheProfile,
) -> dict[str, Any]:
    _reject_symlink_components(selection_path)
    return _validate_selection_document(_read_json(selection_path, uid), uid, profile)


def attest_historical_fixed_slot_selection(
    root: Path,
    original_selection_path: Path,
    *,
    apply: bool = False,
    confirm_root: str | None = None,
    engine_stopped: bool = False,
    proc_root: Path = Path("/proc"),
    expected_uid: int | None = None,
    profile: CacheProfile = FIXED_SLOT_60K_PROFILE,
) -> dict[str, Any]:
    """Create an immutable derived selection for one exact pre-receipt producer.

    The original selection and payloads are never changed.  Eligibility comes
    exclusively from the source-controlled allowlist above and the complete
    historical runtime-evidence hash; this function cannot bless a new producer.
    """

    if (
        profile.state_contract != "fixed-slot-fp32-gdn-conv-v1"
        or profile.payload_producer_contract not in HISTORICAL_FIXED_SLOT_ATTESTABLE_CONTRACTS
    ):
        raise CacheSafetyError(
            "historical attestation is restricted to reviewed fixed-slot contracts"
        )
    uid = os.getuid() if expected_uid is None else expected_uid
    root = _guard_root(root, expected_uid=uid)
    _reject_symlink_components(original_selection_path)
    _, original_selection_sha256 = _hash_regular_file(original_selection_path, uid)
    producer = _historical_producer_by_selection_sha256(original_selection_sha256)
    if producer.contract != profile.payload_producer_contract:
        raise CacheSafetyError("historical producer is not compatible with this cache profile")
    original_document = _read_json(original_selection_path, uid)
    if "payload_provenance" in original_document:
        raise CacheSafetyError(
            "historical attestation requires an unmodified pre-receipt selection"
        )
    canonical_sha256 = _sha256_bytes(_canonical_bytes(original_document))
    if canonical_sha256 != producer.original_selection_canonical_sha256:
        raise CacheSafetyError("historical original selection canonical SHA-256 mismatch")
    selection = _validate_selection_document(
        original_document,
        uid,
        profile,
        allow_missing_provenance=True,
    )
    if selection["source_catalog_sha256"] != producer.source_catalog_sha256:
        raise CacheSafetyError("historical selection source catalog is not qualified")
    if selection["bindings"]["offload_abi_sha256"] != producer.offload_abi_sha256:
        raise CacheSafetyError("historical selection offload ABI is not qualified")
    _validate_historical_runtime_evidence(
        bindings=selection["bindings"],
        producer=producer,
        uid=uid,
    )

    auxiliary_hashes: dict[str, str] = {}
    for relative_name, expected_sha256 in producer.auxiliary_evidence_sha256:
        evidence_path = original_selection_path.parent / relative_name
        _, actual_sha256 = _hash_regular_file(
            evidence_path,
            uid,
            expected_sha256=expected_sha256,
        )
        auxiliary_hashes[relative_name] = actual_sha256

    provenance = _with_document_id(
        {
            "artifact_id": producer.artifact_id,
            "auxiliary_evidence_sha256": auxiliary_hashes,
            "bindings_sha256": _sha256_bytes(_canonical_bytes(original_document["bindings"])),
            "captured_at": original_document["captured_at"],
            "contract": producer.contract,
            "critical_sources": dict(producer.critical_sources),
            "original_selection_canonical_sha256": canonical_sha256,
            "original_selection_sha256": original_selection_sha256,
            "prompt_token_ids_sha256": original_document["request"]["prompt_token_ids_sha256"],
            "request_id": original_document["request"]["request_id"],
            "runtime_evidence_sha256": producer.runtime_evidence_sha256,
            "schema": HISTORICAL_FIXED_SLOT_PRODUCER_SCHEMA,
            "source_catalog_sha256": producer.source_catalog_sha256,
        },
        "attestation_sha256",
    )
    derived_selection = dict(original_document)
    derived_selection["payload_provenance"] = provenance
    _validate_selection_document(derived_selection, uid, profile)
    output_path = (
        _metadata_root(root)
        / "historical-selections"
        / f"{producer.original_selection_sha256}.json"
    )
    result = {
        "action": "attest-historical-fixed-slot",
        "apply": apply,
        "artifact_id": producer.artifact_id,
        "attestation_sha256": provenance["attestation_sha256"],
        "original_selection_sha256": original_selection_sha256,
        "output_selection": str(output_path),
        "runtime_evidence_sha256": producer.runtime_evidence_sha256,
    }
    if not apply:
        return result
    require_apply_safety(
        root,
        confirm_root=confirm_root,
        engine_stopped=engine_stopped,
        proc_root=proc_root,
        expected_uid=uid,
    )
    with _exclusive_lock(root, uid):
        _publish_immutable(output_path, derived_selection, root, uid)
        _load_and_validate_selection(output_path, uid, profile)
    return result


def _validate_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise CacheSafetyError(f"{label} must be a nonempty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or str(path) != value:
        raise CacheSafetyError(f"{label} is not a canonical relative path: {value!r}")
    return value


def _namespace_base_name(config: Mapping[str, Any]) -> str:
    model_name = config.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise CacheSafetyError("namespace config model_name is missing")
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    return f"{model_name.replace('/', '_')}_{digest}"


def _role_for_group(group_id: int, profile: CacheProfile) -> str:
    if group_id in profile.mamba_groups:
        return "mamba"
    if group_id in profile.full_attention_groups:
        return "full_attention"
    if group_id in profile.sliding_window_groups:
        return "sliding_window"
    raise CacheSafetyError(f"unknown cache group {group_id}")


def _validate_namespace_config(
    config: dict[str, Any],
    selection: dict[str, Any],
    profile: CacheProfile,
) -> str:
    required = {
        "blocks_per_file": 1,
        "dcp_size": 1,
        "dtype": "fp8_e4m3",
        "inference_engine": "vllm",
        "parallel_agnostic": False,
        "pcp_size": 1,
        "pp_size": 1,
        "tokens_per_hash": profile.tokens_per_hash,
        "tp_size": 1,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise CacheSafetyError(
                f"namespace config {key}={config.get(key)!r}, expected {expected!r}"
            )
    target_identity = selection["bindings"]["target_model"]["identity"]
    if config.get("model_name") != target_identity:
        raise CacheSafetyError(
            "target model binding does not exactly match namespace config model_name"
        )
    groups = config.get("kv_cache_groups")
    if not isinstance(groups, list) or len(groups) != len(profile.all_groups):
        raise CacheSafetyError(
            f"namespace must contain exactly {len(profile.all_groups)} cache groups"
        )
    if tuple(profile.all_groups) != tuple(range(len(groups))):
        raise CacheSafetyError("profile group ids must form one exact zero-based sequence")
    actual_layer_names: list[str] = []
    for group_id, raw_group in enumerate(groups):
        group = _as_object(raw_group, f"kv_cache_groups[{group_id}]")
        if group.get("tokens_per_block") != profile.page_tokens:
            raise CacheSafetyError(f"cache group {group_id} has the wrong page size")
        names = group.get("layer_names")
        if not isinstance(names, list) or len(names) != 1 or not isinstance(names[0], str):
            raise CacheSafetyError(f"cache group {group_id} must bind exactly one layer")
        actual_layer_names.append(names[0])
        role = _role_for_group(group_id, profile)
        if role == "mamba" and ".linear_attn" not in names[0]:
            raise CacheSafetyError(f"Mamba group {group_id} is not a linear_attn layer")
        if role == "full_attention" and not names[0].startswith("language_model.model.layers."):
            raise CacheSafetyError(f"full-attention group {group_id} is not a target layer")
        if role == "sliding_window" and not names[0].startswith("model.layers."):
            raise CacheSafetyError(f"sliding-window group {group_id} is not a draft layer")
        if role != "mamba" and not names[0].endswith(".self_attn.attn"):
            raise CacheSafetyError(f"attention group {group_id} has an unexpected layer name")
    if len(set(actual_layer_names)) != len(actual_layer_names):
        raise CacheSafetyError("namespace cache-group layer names must be unique")
    if (
        profile.group_layer_names is not None
        and tuple(actual_layer_names) != profile.group_layer_names
    ):
        raise CacheSafetyError("namespace cache groups do not match the exact qualified layer map")
    return _namespace_base_name(config)


def _blob_relative_path(base_name: str, rank: int, group_id: int, block_hash: str) -> str:
    return f"{base_name}_r{rank}/{block_hash[:3]}/{block_hash[3:5]}_g{group_id}/{block_hash}.bin"


def expected_blob_specs(
    base_name: str,
    block_hashes: Sequence[str],
    profile: CacheProfile = PRODUCTION_PROFILE,
) -> tuple[BlobSpec, ...]:
    """Return the exact minimal raw-page set for one P-8 generation."""

    if len(block_hashes) != profile.full_attention_pages:
        raise CacheSafetyError("block hash count does not match the profile")
    roles = (
        (
            "mamba",
            profile.mamba_groups,
            range(
                profile.full_attention_pages - profile.mamba_retained_pages,
                profile.full_attention_pages,
            ),
        ),
        ("full_attention", profile.full_attention_groups, range(profile.full_attention_pages)),
        (
            "sliding_window",
            profile.sliding_window_groups,
            range(
                profile.full_attention_pages - profile.sliding_window_retained_pages,
                profile.full_attention_pages,
            ),
        ),
    )
    result = []
    for role, groups, ordinals in roles:
        for group_id in groups:
            for page_ordinal in ordinals:
                block_hash = block_hashes[page_ordinal]
                result.append(
                    BlobSpec(
                        relative_path=_blob_relative_path(
                            base_name, profile.rank, group_id, block_hash
                        ),
                        group_id=group_id,
                        role=role,
                        page_ordinal=page_ordinal,
                        block_hash=block_hash,
                    )
                )
    if len(result) != profile.blob_count:
        raise AssertionError("profile blob arithmetic drifted")
    paths = {record.relative_path for record in result}
    if len(paths) != len(result):
        raise CacheSafetyError("profile generated duplicate payload paths")
    return tuple(sorted(result, key=lambda item: item.relative_path))


def _clone_or_copy_fd(source_fd: int, destination_fd: int) -> str:
    """Clone one payload when supported, otherwise copy it without a shell helper."""

    try:
        fcntl.ioctl(destination_fd, _FICLONE, source_fd)
        return "reflink"
    except OSError as error:
        if error.errno not in {
            errno.EINVAL,
            errno.ENOTTY,
            errno.EOPNOTSUPP,
            errno.EXDEV,
        }:
            raise
    os.ftruncate(destination_fd, 0)
    os.lseek(source_fd, 0, os.SEEK_SET)
    os.lseek(destination_fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(source_fd, 8 * 1024 * 1024)
        if not chunk:
            break
        _write_all(destination_fd, chunk)
    return "copy"


def _publish_imported_blob(
    source: Path,
    destination: Path,
    root: Path,
    uid: int,
    *,
    expected_size: int,
    expected_sha256: str,
) -> str:
    """Publish a copied payload without ever replacing an existing cache key."""

    _mkdir_owned(destination.parent, root, uid)
    if _lstat_exists(destination):
        _hash_regular_file(
            destination,
            uid,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            durable=True,
        )
        return "existing"

    source_fd, source_before = _safe_open_regular(source, uid)
    temporary = destination.parent / f".import-{uuid.uuid4().hex}.tmp"
    destination_fd = -1
    method = ""
    try:
        if source_before.st_size != expected_size:
            raise CacheSafetyError(
                f"wrong source payload size for {source}: "
                f"{source_before.st_size}, expected {expected_size}"
            )
        destination_fd = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        method = _clone_or_copy_fd(source_fd, destination_fd)
        os.fsync(destination_fd)
        os.fchmod(destination_fd, 0o600)
        copied = os.fstat(destination_fd)
        if copied.st_size != expected_size or not stat.S_ISREG(copied.st_mode):
            raise CacheSafetyError(f"imported payload has the wrong shape: {temporary}")
        os.close(destination_fd)
        destination_fd = -1

        source_after = os.fstat(source_fd)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(source_before, field) != getattr(source_after, field) for field in stable_fields
        ):
            raise CacheSafetyError(f"source payload changed while importing: {source}")

        try:
            os.link(temporary, destination, follow_symlinks=False)
            _fsync_directory(destination.parent)
        except FileExistsError:
            _hash_regular_file(
                destination,
                uid,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
                durable=True,
            )
            return "existing"
        finally:
            if _lstat_exists(temporary):
                temporary.unlink()
                _fsync_directory(destination.parent)

        _hash_regular_file(
            destination,
            uid,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            durable=True,
        )
        return method
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)
        os.close(source_fd)
        if _lstat_exists(temporary):
            temporary.unlink()
            _fsync_directory(destination.parent)


def import_fixed_slot_generation(
    root: Path,
    source_root: Path,
    selection_path: Path,
    *,
    apply: bool = False,
    confirm_root: str | None = None,
    engine_stopped: bool = False,
    proc_root: Path = Path("/proc"),
    expected_uid: int | None = None,
    profile: CacheProfile = FIXED_SLOT_60K_PROFILE,
    fault: FaultHook | None = None,
) -> dict[str, Any]:
    """Promote a complete captured generation into the isolated fixed-slot root."""

    if profile.state_contract != "fixed-slot-fp32-gdn-conv-v1":
        raise CacheSafetyError("captured-source import is restricted to the fixed-slot profile")
    uid = os.getuid() if expected_uid is None else expected_uid
    root = _guard_root(root, expected_uid=uid)
    source_root = _guard_root(source_root, expected_uid=uid)
    if root == source_root:
        raise CacheSafetyError("source and fixed-slot roots must be different directories")
    if apply:
        root = require_apply_safety(
            root,
            confirm_root=confirm_root,
            engine_stopped=engine_stopped,
            proc_root=proc_root,
            expected_uid=uid,
        )

    def perform() -> dict[str, Any]:
        selection = _load_and_validate_selection(selection_path, uid, profile)
        expected_catalog_sha256 = selection.get("source_catalog_sha256")
        if expected_catalog_sha256 is None:
            raise CacheSafetyError(
                "fixed-slot captured-source import requires source_catalog_sha256"
            )
        target_config, target_config_sha, base_name = _config_for_selection(
            root,
            selection,
            uid,
            profile,
            durable=apply,
        )
        config_relative = selection["namespace"]["config_relative_path"]
        source_config_path = _resolve_under(source_root, config_relative)
        _guard_owned_chain(source_root, source_config_path.parent, uid)
        source_config = _read_json(source_config_path, uid)
        _, source_config_sha = _hash_regular_file(
            source_config_path,
            uid,
            expected_sha256=target_config_sha,
            durable=apply,
        )
        if source_config != target_config or source_config_sha != target_config_sha:
            raise CacheSafetyError("captured source and fixed-slot namespace configs differ")
        if _validate_namespace_config(source_config, selection, profile) != base_name:
            raise CacheSafetyError("captured source namespace base name differs")

        records: list[tuple[BlobSpec, str]] = []
        target_existing = 0
        for spec in expected_blob_specs(base_name, selection["block_hashes"], profile):
            source_path = _resolve_under(source_root, spec.relative_path)
            _guard_owned_chain(source_root, source_path.parent, uid)
            _, source_digest = _hash_regular_file(
                source_path,
                uid,
                expected_size=profile.page_bytes,
                durable=apply,
            )
            destination = _resolve_under(root, spec.relative_path)
            if _lstat_exists(destination):
                _, target_digest = _hash_regular_file(
                    destination,
                    uid,
                    expected_size=profile.page_bytes,
                    expected_sha256=source_digest,
                    durable=apply,
                )
                if target_digest != source_digest:
                    raise CacheSafetyError(f"captured-source collision: {destination}")
                target_existing += 1
            records.append((spec, source_digest))

        catalog = "".join(f"{spec.relative_path}\0{digest}\n" for spec, digest in records).encode()
        source_catalog_sha256 = _sha256_bytes(catalog)
        if source_catalog_sha256 != expected_catalog_sha256:
            raise CacheSafetyError(
                "captured source catalog SHA-256 mismatch: "
                f"{source_catalog_sha256}, expected {expected_catalog_sha256}"
            )
        result: dict[str, Any] = {
            "action": "import-fixed-slot",
            "apply": apply,
            "copied_files": 0 if not apply else profile.blob_count - target_existing,
            "expected_files": profile.blob_count,
            "payload_bytes": profile.payload_bytes,
            "reused_files": target_existing,
            "source_catalog_sha256": source_catalog_sha256,
            "source_root": str(source_root),
            "target_root": str(root),
        }
        if not apply:
            return result

        methods = {"copy": 0, "existing": 0, "reflink": 0}
        for index, (spec, digest) in enumerate(records):
            method = _publish_imported_blob(
                _resolve_under(source_root, spec.relative_path),
                _resolve_under(root, spec.relative_path),
                root,
                uid,
                expected_size=profile.page_bytes,
                expected_sha256=digest,
            )
            methods[method] += 1
            if fault is not None:
                fault(f"after_import_blob:{index}")
        result["methods"] = methods
        result["copied_files"] = methods["copy"] + methods["reflink"]
        result["reused_files"] = methods["existing"]
        return result

    if not apply:
        return perform()
    with _exclusive_lock(root, uid):
        return perform()


def _validate_blob_relative_path(relative_path: str) -> re.Match[str]:
    _validate_relative_path(relative_path, "blob relative_path")
    match = _BLOB_RE.fullmatch(relative_path)
    if match is None:
        raise CacheSafetyError(f"path does not match the pinned FileMapper ABI: {relative_path}")
    block_hash = match.group("block_hash")
    if match.group("prefix1") != block_hash[:3] or match.group("prefix2") != block_hash[3:5]:
        raise CacheSafetyError(f"hash fan-out directories disagree with filename: {relative_path}")
    return match


def _resolve_under(root: Path, relative_path: str) -> Path:
    relative = _validate_relative_path(relative_path, "relative_path")
    result = root.joinpath(*PurePosixPath(relative).parts)
    try:
        result.relative_to(root)
    except ValueError as error:
        raise CacheSafetyError(f"path escapes cache root: {relative_path}") from error
    return result


def _profile_document(profile: CacheProfile) -> dict[str, Any]:
    document = {
        "file_mapper_abi": FILE_MAPPER_ABI,
        "group_layer_names": (
            list(profile.group_layer_names) if profile.group_layer_names is not None else None
        ),
        "group_ids": {
            "full_attention": list(profile.full_attention_groups),
            "mamba": list(profile.mamba_groups),
            "sliding_window": list(profile.sliding_window_groups),
        },
        "hash_aligned_tokens": profile.hash_aligned_tokens,
        "page_bytes": profile.page_bytes,
        "page_tokens": profile.page_tokens,
        "profile": profile.name,
        "prompt_tokens": profile.prompt_tokens,
        "rank": profile.rank,
        "replay_boundary_tokens": profile.replay_boundary_tokens,
        "retained_pages_per_group": {
            "full_attention": profile.full_attention_pages,
            "mamba": profile.mamba_retained_pages,
            "sliding_window": profile.sliding_window_retained_pages,
        },
        "tokens_per_hash": profile.tokens_per_hash,
    }
    if profile.state_contract is not None:
        document["state_contract"] = {
            "canonical_gdn_dtype": "float32",
            "canonical_gdn_groups": list(profile.mamba_groups),
            "convolution_history": "included-in-authenticated-mamba-payload",
            "draft_kv": "authenticated-fp8-e4m3-pages",
            "generation_identity": "manifest-generation-plus-payload-sha256",
            "logical_restore_boundary": profile.replay_boundary_tokens,
            "mamba_cache_mode": "none",
            "name": profile.state_contract,
            "physical_block_ids": "ephemeral-never-authoritative",
            "scale_identity": "runtime-model-binding-plus-namespace-config",
            "target_kv": "authenticated-fp8-e4m3-pages",
        }
    if profile.payload_producer_contract is not None:
        document["payload_producer_contract"] = profile.payload_producer_contract
    return document


def _config_for_selection(
    root: Path,
    selection: dict[str, Any],
    uid: int,
    profile: CacheProfile,
    *,
    durable: bool,
) -> tuple[dict[str, Any], str, str]:
    relative = selection["namespace"]["config_relative_path"]
    config_path = _resolve_under(root, relative)
    _guard_owned_chain(root, config_path.parent, uid)
    config = _read_json(config_path, uid)
    _, config_sha = _hash_regular_file(
        config_path,
        uid,
        expected_sha256=selection["namespace"]["config_sha256"],
        durable=durable,
    )
    base_name = _validate_namespace_config(config, selection, profile)
    if PurePosixPath(relative).parent.name != base_name:
        raise CacheSafetyError(
            "namespace config directory does not match FileMapper's canonical base-path hash"
        )
    return config, config_sha, base_name


def _build_manifest(
    root: Path,
    selection_path: Path,
    uid: int,
    profile: CacheProfile,
    *,
    durable: bool,
) -> dict[str, Any]:
    selection = _load_and_validate_selection(selection_path, uid, profile)
    config, config_sha, base_name = _config_for_selection(
        root,
        selection,
        uid,
        profile,
        durable=durable,
    )
    records: list[BlobRecord] = []
    for spec in expected_blob_specs(base_name, selection["block_hashes"], profile):
        _validate_blob_relative_path(spec.relative_path)
        path = _resolve_under(root, spec.relative_path)
        _guard_owned_chain(root, path.parent, uid)
        size, digest = _hash_regular_file(
            path,
            uid,
            expected_size=profile.page_bytes,
            durable=durable,
        )
        records.append(
            BlobRecord(
                relative_path=spec.relative_path,
                group_id=spec.group_id,
                role=spec.role,
                page_ordinal=spec.page_ordinal,
                block_hash=spec.block_hash,
                size=size,
                payload_sha256=digest,
            )
        )
    if (
        len(records) != profile.blob_count
        or sum(item.size for item in records) != profile.payload_bytes
    ):
        raise CacheSafetyError(
            "sealed payload does not meet the exact profile count and byte target"
        )
    if durable:
        directories = {root}
        for record in records:
            current = _resolve_under(root, record.relative_path).parent
            while True:
                directories.add(current)
                if current == root:
                    break
                current = current.parent
        config_parent = _resolve_under(
            root,
            selection["namespace"]["config_relative_path"],
        ).parent
        directories.add(config_parent)
        for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            _guard_owned_directory(directory, uid)
            _fsync_directory(directory)
    document = {
        "bindings": selection["bindings"],
        "blobs": [record.as_dict() for record in records],
        "captured_at": selection["captured_at"],
        "generation": selection["generation"],
        "geometry": _profile_document(profile),
        "namespace": {
            "base_name": base_name,
            "config": config,
            "config_relative_path": selection["namespace"]["config_relative_path"],
            "config_sha256": config_sha,
        },
        "parent_manifest_sha256": selection["parent_manifest_sha256"],
        "request": selection["request"],
        "schema": MANIFEST_SCHEMA,
        # The runtime capture is the sealing event.  Keeping this deterministic
        # makes a retry after either publication boundary exactly idempotent.
        "sealed_at": selection["captured_at"],
        "session": {
            "branch": selection["branch"],
            "session_id": selection["session_id"],
        },
        "summary": {
            "blob_count": len(records),
            "payload_bytes": sum(item.size for item in records),
        },
    }
    if profile.payload_producer_contract is not None:
        document["payload_provenance"] = selection["payload_provenance"]
        document["source_catalog_sha256"] = selection["source_catalog_sha256"]
    return _with_document_id(document, "manifest_sha256")


def _ref_path(root: Path, session_id: str, branch: str) -> Path:
    _require_identifier(session_id, "session_id")
    _require_identifier(branch, "branch")
    return _metadata_root(root) / "refs" / session_id / f"{branch}.json"


def _manifest_path(root: Path, manifest_sha256: str) -> Path:
    _require_sha256(manifest_sha256, "manifest_sha256")
    return _metadata_root(root) / "manifests" / f"{manifest_sha256}.json"


def _validate_ref(document: dict[str, Any], label: str) -> dict[str, Any]:
    _require_exact_keys(
        document,
        {
            "branch",
            "generation",
            "manifest_sha256",
            "schema",
            "session_id",
            "updated_at",
        },
        label,
    )
    if document["schema"] != REF_SCHEMA:
        raise CacheSafetyError(f"unsupported ref schema in {label}")
    _require_identifier(document["session_id"], f"{label}.session_id")
    _require_identifier(document["branch"], f"{label}.branch")
    generation = document["generation"]
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
        raise CacheSafetyError(f"invalid ref generation in {label}")
    _require_sha256(document["manifest_sha256"], f"{label}.manifest_sha256")
    if not isinstance(document["updated_at"], str) or not document["updated_at"].endswith("Z"):
        raise CacheSafetyError(f"invalid ref timestamp in {label}")
    return document


def _read_current_ref(root: Path, session_id: str, branch: str, uid: int) -> dict[str, Any] | None:
    path = _ref_path(root, session_id, branch)
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    _guard_owned_chain(root, path.parent, uid)
    ref = _validate_ref(_read_json(path, uid), str(path))
    if ref["session_id"] != session_id or ref["branch"] != branch:
        raise CacheSafetyError(f"ref identity disagrees with path: {path}")
    return ref


def _read_validated_manifest(
    root: Path,
    manifest_sha256: str,
    uid: int,
    profile: CacheProfile,
) -> tuple[dict[str, Any], tuple[BlobRecord, ...]]:
    path = _manifest_path(root, manifest_sha256)
    _guard_owned_chain(root, path.parent, uid)
    document = _read_json(path, uid)
    actual_sha, blobs = _validate_manifest(document, str(path), profile, uid=uid)
    if actual_sha != manifest_sha256:
        raise CacheSafetyError(f"manifest filename/content hash mismatch: {path}")
    return document, blobs


def _authenticate_manifest_ancestry(
    root: Path,
    head_sha256: str,
    uid: int,
    profile: CacheProfile,
) -> tuple[dict[str, Any], tuple[BlobRecord, ...], tuple[str, ...]]:
    """Authenticate an unbroken same-session/branch chain down to generation one."""

    head, head_blobs = _read_validated_manifest(root, head_sha256, uid, profile)
    # The manifest cryptographically records each identity, while this check
    # proves the current evidence files still realize those exact bindings.
    _validate_bindings(head["bindings"], uid)
    expected_session = head["session"]
    expected_generation = head["generation"]
    invariant = {
        "bindings": head["bindings"],
        "geometry": head["geometry"],
        "namespace_base_name": head["namespace"]["base_name"],
        "namespace_config_sha256": head["namespace"]["config_sha256"],
    }
    ancestry = []
    seen: set[str] = set()
    current = head
    current_sha = head_sha256
    while True:
        if current_sha in seen:
            raise CacheSafetyError(f"manifest ancestry cycle at {current_sha}")
        seen.add(current_sha)
        ancestry.append(current_sha)
        if current["session"] != expected_session:
            raise CacheSafetyError("manifest ancestry crosses a session or branch")
        if current["generation"] != expected_generation:
            raise CacheSafetyError("manifest ancestry generation is not contiguous")
        current_invariant = {
            "bindings": current["bindings"],
            "geometry": current["geometry"],
            "namespace_base_name": current["namespace"]["base_name"],
            "namespace_config_sha256": current["namespace"]["config_sha256"],
        }
        if current_invariant != invariant:
            raise CacheSafetyError(
                "manifest ancestry changes ABI/model/tokenizer/namespace binding"
            )
        parent_sha = current["parent_manifest_sha256"]
        if expected_generation == 1:
            if parent_sha is not None:
                raise CacheSafetyError("generation-one manifest has a parent")
            break
        if parent_sha is None:
            raise CacheSafetyError("manifest ancestry terminates before generation one")
        expected_generation -= 1
        current_sha = parent_sha
        current, _ = _read_validated_manifest(root, current_sha, uid, profile)
    return head, head_blobs, tuple(ancestry)


def _validate_ref_head(
    root: Path,
    ref: dict[str, Any],
    uid: int,
    profile: CacheProfile,
) -> tuple[dict[str, Any], tuple[BlobRecord, ...], tuple[str, ...]]:
    head, blobs, ancestry = _authenticate_manifest_ancestry(
        root,
        ref["manifest_sha256"],
        uid,
        profile,
    )
    if (
        head["session"]
        != {
            "branch": ref["branch"],
            "session_id": ref["session_id"],
        }
        or head["generation"] != ref["generation"]
    ):
        raise CacheSafetyError("ref and authenticated head identity disagree")
    config_path = _resolve_under(root, head["namespace"]["config_relative_path"])
    _guard_owned_chain(root, config_path.parent, uid)
    current_config = _read_json(config_path, uid)
    if current_config != head["namespace"]["config"]:
        raise CacheSafetyError("current namespace config content differs from head manifest")
    _hash_regular_file(
        config_path,
        uid,
        expected_sha256=head["namespace"]["config_sha256"],
    )
    return head, blobs, ancestry


def _validate_parent(manifest: dict[str, Any], current: dict[str, Any] | None) -> bool:
    generation = manifest["generation"]
    parent = manifest["parent_manifest_sha256"]
    if current is None:
        if generation != 1 or parent is not None:
            raise CacheSafetyError("a new branch must start at generation 1 with a null parent")
        return False
    if (
        generation == current["generation"]
        and manifest["manifest_sha256"] == current["manifest_sha256"]
    ):
        return True
    if generation != current["generation"] + 1:
        raise CacheSafetyError("generation must advance the current branch by exactly one")
    if parent != current["manifest_sha256"]:
        raise CacheSafetyError("parent_manifest_sha256 does not match the current branch head")
    return False


def _ref_document(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "branch": manifest["session"]["branch"],
        "generation": manifest["generation"],
        "manifest_sha256": manifest["manifest_sha256"],
        "schema": REF_SCHEMA,
        "session_id": manifest["session"]["session_id"],
        "updated_at": _utc_now(),
    }


def seal_generation(
    root: Path,
    selection_path: Path,
    *,
    apply: bool = False,
    confirm_root: str | None = None,
    engine_stopped: bool = False,
    proc_root: Path = Path("/proc"),
    expected_uid: int | None = None,
    profile: CacheProfile = PRODUCTION_PROFILE,
    fault: FaultHook | None = None,
) -> dict[str, Any]:
    """Validate and optionally publish one immutable generation plus branch ref."""

    uid = os.getuid() if expected_uid is None else expected_uid
    root = _guard_root(root, expected_uid=uid)
    if apply:
        root = require_apply_safety(
            root,
            confirm_root=confirm_root,
            engine_stopped=engine_stopped,
            proc_root=proc_root,
            expected_uid=uid,
        )

    def perform() -> dict[str, Any]:
        manifest = _build_manifest(
            root,
            selection_path,
            uid,
            profile,
            durable=apply,
        )
        if apply and fault is not None:
            fault("after_payload_durability")
        session = manifest["session"]
        current = _read_current_ref(root, session["session_id"], session["branch"], uid)
        current_head: dict[str, Any] | None = None
        if current is not None:
            current_head, _, _ = _validate_ref_head(root, current, uid, profile)
        idempotent = _validate_parent(manifest, current)
        if idempotent and current_head != manifest:
            raise CacheSafetyError(
                "idempotent seal selection does not exactly match the authenticated head manifest"
            )
        if current_head is not None and not idempotent:
            for key in ("bindings", "geometry"):
                if manifest[key] != current_head[key]:
                    raise CacheSafetyError(f"new generation changes the branch {key} binding")
            for key in ("base_name", "config", "config_relative_path", "config_sha256"):
                if manifest["namespace"][key] != current_head["namespace"][key]:
                    raise CacheSafetyError(
                        f"new generation changes the branch namespace {key} binding"
                    )
        result = {
            "action": "seal",
            "apply": apply,
            "branch": session["branch"],
            "generation": manifest["generation"],
            "idempotent": idempotent,
            "manifest_sha256": manifest["manifest_sha256"],
            "payload_bytes": manifest["summary"]["payload_bytes"],
            "payload_files": manifest["summary"]["blob_count"],
            "session_id": session["session_id"],
        }
        if not apply or idempotent:
            return result
        manifest_path = _manifest_path(root, manifest["manifest_sha256"])
        _publish_immutable(manifest_path, manifest, root, uid)
        if fault is not None:
            fault("after_manifest_publish")
        ref_path = _ref_path(root, session["session_id"], session["branch"])
        _publish_replace(ref_path, _ref_document(manifest), root, uid)
        if fault is not None:
            fault("after_ref_publish")
        return result

    if not apply:
        return perform()
    with _exclusive_lock(root, uid):
        return perform()


def _parse_manifest_blob(raw: Any, profile: CacheProfile, label: str) -> BlobRecord:
    blob = _as_object(raw, label)
    _require_exact_keys(
        blob,
        {
            "block_hash",
            "group_id",
            "page_ordinal",
            "payload_sha256",
            "relative_path",
            "role",
            "size",
        },
        label,
    )
    relative = _validate_relative_path(blob["relative_path"], f"{label}.relative_path")
    match = _validate_blob_relative_path(relative)
    block_hash = _require_sha256(blob["block_hash"], f"{label}.block_hash")
    if match.group("block_hash") != block_hash:
        raise CacheSafetyError(f"{label} block hash disagrees with its path")
    group_id = blob["group_id"]
    if not isinstance(group_id, int) or isinstance(group_id, bool):
        raise CacheSafetyError(f"{label}.group_id must be an integer")
    if int(match.group("group")) != group_id:
        raise CacheSafetyError(f"{label} group id disagrees with its path")
    role = blob["role"]
    if role != _role_for_group(group_id, profile):
        raise CacheSafetyError(f"{label} role disagrees with its group")
    page_ordinal = blob["page_ordinal"]
    if not isinstance(page_ordinal, int) or isinstance(page_ordinal, bool):
        raise CacheSafetyError(f"{label}.page_ordinal must be an integer")
    if blob["size"] != profile.page_bytes:
        raise CacheSafetyError(f"{label} size is not the exact profile page size")
    digest = _require_sha256(blob["payload_sha256"], f"{label}.payload_sha256")
    return BlobRecord(relative, group_id, role, page_ordinal, block_hash, blob["size"], digest)


def _validate_manifest(
    document: dict[str, Any],
    label: str,
    profile: CacheProfile,
    *,
    uid: int,
) -> tuple[str, tuple[BlobRecord, ...]]:
    expected_manifest_keys = {
        "bindings",
        "blobs",
        "captured_at",
        "generation",
        "geometry",
        "manifest_sha256",
        "namespace",
        "parent_manifest_sha256",
        "request",
        "schema",
        "sealed_at",
        "session",
        "summary",
    }
    if profile.payload_producer_contract is not None:
        expected_manifest_keys |= {"payload_provenance", "source_catalog_sha256"}
    _require_exact_keys(document, expected_manifest_keys, label)
    if document["schema"] != MANIFEST_SCHEMA:
        raise CacheSafetyError(f"unsupported manifest schema in {label}")
    digest = _verify_document_id(document, "manifest_sha256", label)
    if document["geometry"] != _profile_document(profile):
        raise CacheSafetyError(f"manifest geometry drift in {label}")
    bindings = _validate_bindings_record(document["bindings"])
    for timestamp_key in ("captured_at", "sealed_at"):
        timestamp = document[timestamp_key]
        if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
            raise CacheSafetyError(f"invalid {timestamp_key} in {label}")
    if document["sealed_at"] != document["captured_at"]:
        raise CacheSafetyError(f"nondeterministic seal timestamp in {label}")
    session = _as_object(document["session"], f"{label}.session")
    _require_exact_keys(session, {"branch", "session_id"}, f"{label}.session")
    _require_identifier(session["session_id"], f"{label}.session_id")
    _require_identifier(session["branch"], f"{label}.branch")
    generation = document["generation"]
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
        raise CacheSafetyError(f"invalid generation in {label}")
    parent = document["parent_manifest_sha256"]
    if parent is not None:
        _require_sha256(parent, f"{label}.parent_manifest_sha256")
    if (generation == 1) != (parent is None):
        raise CacheSafetyError(f"generation/parent lineage mismatch in {label}")
    request = _as_object(document["request"], f"{label}.request")
    _require_exact_keys(
        request,
        {
            "hash_aligned_tokens",
            "prompt_token_ids_sha256",
            "prompt_tokens",
            "replay_boundary_tokens",
            "request_id",
            "store_drained",
        },
        f"{label}.request",
    )
    request_expectations = {
        "hash_aligned_tokens": profile.hash_aligned_tokens,
        "prompt_tokens": profile.prompt_tokens,
        "replay_boundary_tokens": profile.replay_boundary_tokens,
        "store_drained": True,
    }
    for key, expected_value in request_expectations.items():
        if request[key] != expected_value:
            raise CacheSafetyError(f"request contract drift for {key} in {label}")
    _require_identifier(request["request_id"], f"{label}.request.request_id")
    _require_sha256(
        request["prompt_token_ids_sha256"],
        f"{label}.request.prompt_token_ids_sha256",
    )
    if profile.payload_producer_contract is not None:
        source_catalog_sha256 = _require_sha256(
            document["source_catalog_sha256"],
            f"{label}.source_catalog_sha256",
        )
        _validate_payload_provenance(
            document["payload_provenance"],
            profile=profile,
            bindings=bindings,
            request=request,
            captured_at=document["captured_at"],
            source_catalog_sha256=source_catalog_sha256,
            label=f"{label}.payload_provenance",
            uid=uid,
        )
    blobs_raw = document["blobs"]
    if not isinstance(blobs_raw, list) or len(blobs_raw) != profile.blob_count:
        raise CacheSafetyError(f"{label} does not contain exactly {profile.blob_count} blobs")
    blobs = tuple(
        _parse_manifest_blob(raw, profile, f"{label}.blobs[{index}]")
        for index, raw in enumerate(blobs_raw)
    )
    if len({blob.relative_path for blob in blobs}) != len(blobs):
        raise CacheSafetyError(f"duplicate blob paths in {label}")
    summary = _as_object(document["summary"], f"{label}.summary")
    if summary != {"blob_count": profile.blob_count, "payload_bytes": profile.payload_bytes}:
        raise CacheSafetyError(f"summary drift in {label}")
    namespace = _as_object(document["namespace"], f"{label}.namespace")
    _require_exact_keys(
        namespace,
        {"base_name", "config", "config_relative_path", "config_sha256"},
        f"{label}.namespace",
    )
    _require_sha256(namespace["config_sha256"], f"{label}.namespace.config_sha256")
    config = _as_object(namespace["config"], f"{label}.namespace.config")
    base_name = _validate_namespace_config(config, {"bindings": bindings}, profile)
    if namespace["base_name"] != base_name:
        raise CacheSafetyError(f"namespace base-name hash mismatch in {label}")
    config_relative = _validate_relative_path(
        namespace["config_relative_path"],
        f"{label}.namespace.config_relative_path",
    )
    if config_relative != f"{base_name}/config.json":
        raise CacheSafetyError(f"namespace config path mismatch in {label}")
    first_full_group = profile.full_attention_groups[0]
    ordinal_hashes = {
        blob.page_ordinal: blob.block_hash
        for blob in blobs
        if blob.role == "full_attention" and blob.group_id == first_full_group
    }
    if set(ordinal_hashes) != set(range(profile.full_attention_pages)):
        raise CacheSafetyError(f"full-attention page chain is incomplete in {label}")
    expected_specs = expected_blob_specs(
        base_name,
        [ordinal_hashes[ordinal] for ordinal in range(profile.full_attention_pages)],
        profile,
    )
    expected = {
        (item.relative_path, item.group_id, item.role, item.page_ordinal, item.block_hash)
        for item in expected_specs
    }
    actual = {
        (item.relative_path, item.group_id, item.role, item.page_ordinal, item.block_hash)
        for item in blobs
    }
    if actual != expected:
        raise CacheSafetyError(f"blob membership does not match exact retained geometry in {label}")
    return digest, blobs


def _iter_ref_paths(root: Path, uid: int) -> tuple[Path, ...]:
    refs_root = _metadata_root(root) / "refs"
    try:
        refs_root.lstat()
    except FileNotFoundError:
        return ()
    _guard_owned_chain(root, refs_root, uid)
    result = []
    for session_entry in sorted(os.scandir(refs_root), key=lambda entry: entry.name):
        session_path = Path(session_entry.path)
        if session_entry.is_symlink() or not session_entry.is_dir(follow_symlinks=False):
            raise CacheSafetyError(f"unsafe entry below refs: {session_path}")
        _require_identifier(session_entry.name, "ref session directory")
        _guard_owned_directory(session_path, uid)
        for branch_entry in sorted(os.scandir(session_path), key=lambda entry: entry.name):
            path = Path(branch_entry.path)
            if branch_entry.is_symlink() or not branch_entry.is_file(follow_symlinks=False):
                raise CacheSafetyError(f"unsafe entry below ref session: {path}")
            if _PUBLISH_TEMP_RE.fullmatch(path.name) is not None:
                temporary_fd, _ = _safe_open_regular(path, uid)
                os.close(temporary_fd)
                # A same-directory ref replacement may leave this owner-only
                # temporary file after power loss.  It never becomes a live ref.
                continue
            if path.suffix != ".json":
                raise CacheSafetyError(f"unexpected ref file: {path}")
            _require_identifier(path.stem, "ref branch filename")
            result.append(path)
    return tuple(result)


def _load_active_state(
    root: Path,
    uid: int,
    profile: CacheProfile,
    *,
    verify_payloads: bool,
) -> ActiveState:
    refs = []
    manifests = []
    physical: dict[str, BlobRecord] = {}
    logical_count = 0
    logical_bytes = 0
    for ref_path in _iter_ref_paths(root, uid):
        ref = _validate_ref(_read_json(ref_path, uid), str(ref_path))
        expected_ref_path = _ref_path(root, ref["session_id"], ref["branch"])
        if expected_ref_path != ref_path:
            raise CacheSafetyError(f"ref identity disagrees with path: {ref_path}")
        manifest, blobs, _ = _validate_ref_head(root, ref, uid, profile)
        for blob in blobs:
            previous = physical.get(blob.relative_path)
            if previous is not None and previous != blob:
                raise CacheSafetyError(
                    f"shared blob has conflicting metadata: {blob.relative_path}"
                )
            physical[blob.relative_path] = blob
        logical_count += len(blobs)
        logical_bytes += sum(blob.size for blob in blobs)
        refs.append(ref)
        manifests.append(manifest)
    if verify_payloads:
        for relative, blob in sorted(physical.items()):
            _hash_regular_file(
                _resolve_under(root, relative),
                uid,
                expected_size=blob.size,
                expected_sha256=blob.payload_sha256,
            )
    return ActiveState(
        refs=tuple(refs),
        manifests=tuple(manifests),
        blobs=physical,
        logical_blob_count=logical_count,
        logical_payload_bytes=logical_bytes,
    )


def _scan_payload_files(root: Path, uid: int, profile: CacheProfile) -> dict[str, int]:
    result: dict[str, int] = {}
    metadata = _metadata_root(root)

    def walk(directory: Path) -> None:
        _guard_owned_directory(directory, uid)
        for entry in sorted(os.scandir(directory), key=lambda item: item.name):
            path = Path(entry.path)
            if path == metadata:
                if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                    raise CacheSafetyError(f"unsafe metadata root: {path}")
                continue
            if entry.is_symlink():
                raise CacheSafetyError(f"symlink below cache root is forbidden: {path}")
            if entry.is_dir(follow_symlinks=False):
                walk(path)
                continue
            if not entry.is_file(follow_symlinks=False):
                raise CacheSafetyError(f"special file below cache root is forbidden: {path}")
            info = entry.stat(follow_symlinks=False)
            if info.st_uid != uid:
                raise CacheSafetyError(f"file below cache root has wrong owner: {path}")
            _reject_group_other_write(info, path)
            relative = path.relative_to(root).as_posix()
            if path.name.endswith(".tmp") or ".tmp" in path.suffixes:
                raise CacheSafetyError(f"in-flight temporary payload exists: {path}")
            if path.suffix == ".bin":
                _validate_blob_relative_path(relative)
                if info.st_size != profile.page_bytes:
                    raise CacheSafetyError(
                        f"payload has wrong fixed-page size {info.st_size}: {path}"
                    )
                result[relative] = info.st_size

    walk(root)
    return result


def verify_active_generations(
    root: Path,
    *,
    expected_uid: int | None = None,
    profile: CacheProfile = PRODUCTION_PROFILE,
) -> dict[str, Any]:
    uid = os.getuid() if expected_uid is None else expected_uid
    root = _guard_root(root, expected_uid=uid)
    active = _load_active_state(root, uid, profile, verify_payloads=True)
    scanned = _scan_payload_files(root, uid, profile)
    missing = sorted(set(active.blobs) - set(scanned))
    if missing:
        raise CacheSafetyError(
            f"active payload files disappeared during verification: {missing[:3]}"
        )
    return {
        "action": "verify",
        "active_refs": len(active.refs),
        "deduplicated_bytes": active.logical_payload_bytes
        - sum(blob.size for blob in active.blobs.values()),
        "logical_payload_bytes": active.logical_payload_bytes,
        "logical_payload_files": active.logical_blob_count,
        "physical_active_payload_bytes": sum(blob.size for blob in active.blobs.values()),
        "physical_active_payload_files": len(active.blobs),
        "total_namespace_payload_bytes": sum(scanned.values()),
        "total_namespace_payload_files": len(scanned),
    }


def _root_fingerprint(root: Path, uid: int) -> dict[str, Any]:
    info = _guard_owned_directory(root, uid)
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "path": str(root),
        "uid": uid,
    }


def _new_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _run_directory(root: Path, run_id: str) -> Path:
    if _RUN_ID_RE.fullmatch(run_id) is None:
        raise CacheSafetyError(f"invalid quarantine run id: {run_id!r}")
    return _metadata_root(root) / "quarantine" / run_id


def _plan_path(root: Path, run_id: str) -> Path:
    return _run_directory(root, run_id) / "plan.json"


def _incomplete_run_ids(root: Path, uid: int) -> tuple[str, ...]:
    quarantine = _metadata_root(root) / "quarantine"
    try:
        quarantine.lstat()
    except FileNotFoundError:
        return ()
    _guard_owned_chain(root, quarantine, uid)
    result = []
    for entry in sorted(os.scandir(quarantine), key=lambda item: item.name):
        path = Path(entry.path)
        if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
            raise CacheSafetyError(f"unsafe quarantine entry: {path}")
        if _RUN_ID_RE.fullmatch(entry.name) is None:
            raise CacheSafetyError(f"unexpected quarantine run directory: {path}")
        _guard_owned_directory(path, uid)
        permitted = {
            "complete.json",
            "deferred.json",
            "payload",
            "plan.json",
            "purged.json",
            "purging.json",
        }
        for child in os.scandir(path):
            child_path = Path(child.path)
            if child.is_symlink():
                raise CacheSafetyError(f"symlink in quarantine metadata: {child_path}")
            if _PUBLISH_TEMP_RE.fullmatch(child.name) is not None:
                temporary_fd, _ = _safe_open_regular(child_path, uid)
                os.close(temporary_fd)
                continue
            if child.name not in permitted:
                raise CacheSafetyError(f"unexpected quarantine metadata: {child_path}")
            if child.name == "payload":
                if not child.is_dir(follow_symlinks=False):
                    raise CacheSafetyError(f"quarantine payload is not a directory: {child_path}")
                _guard_owned_directory(child_path, uid)
            else:
                marker_fd, _ = _safe_open_regular(child_path, uid)
                os.close(marker_fd)
        plan_path = path / "plan.json"
        complete_path = path / "complete.json"
        deferred_path = path / "deferred.json"
        purging_path = path / "purging.json"
        purged_path = path / "purged.json"
        if not _lstat_exists(plan_path):
            if any(
                _lstat_exists(candidate)
                for candidate in (
                    path / "payload",
                    complete_path,
                    deferred_path,
                    purging_path,
                    purged_path,
                )
            ):
                raise CacheSafetyError(f"quarantine data exists without a durable plan: {path}")
            # Power may fail after the run directory is fsynced but before the
            # immutable plan rename.  No payload move can have happened yet.
            continue
        plan = _validate_plan(_read_json(plan_path, uid), root, uid, entry.name)
        _scan_quarantine_payload_tree(root, entry.name, uid, plan)
        if _lstat_exists(deferred_path):
            _validate_deferred_marker(_read_json(deferred_path, uid), plan)
        if not _lstat_exists(complete_path):
            if _lstat_exists(purging_path) or _lstat_exists(purged_path):
                raise CacheSafetyError(f"purge marker exists before quarantine completion: {path}")
            result.append(entry.name)
            continue
        _validate_complete_marker(_read_json(complete_path, uid), plan)
        if _lstat_exists(deferred_path):
            raise CacheSafetyError(f"completed quarantine still has a deferred marker: {path}")
        if _lstat_exists(purging_path):
            _validate_purging_marker(_read_json(purging_path, uid), plan)
        if _lstat_exists(purged_path):
            if not _lstat_exists(purging_path):
                raise CacheSafetyError(f"purged marker exists without purging marker: {path}")
            _validate_purged_marker(_read_json(purged_path, uid), plan)
    return tuple(result)


def _candidate_records(
    root: Path,
    uid: int,
    candidates: Sequence[str],
    sizes: Mapping[str, int],
    *,
    durable: bool,
) -> list[dict[str, Any]]:
    result = []
    for relative in candidates:
        path = _resolve_under(root, relative)
        size, digest = _hash_regular_file(
            path,
            uid,
            expected_size=sizes[relative],
            durable=durable,
        )
        result.append(
            {
                "payload_sha256": digest,
                "relative_path": relative,
                "size": size,
            }
        )
    return result


def _quarantine_plan(
    root: Path,
    uid: int,
    active: ActiveState,
    scanned: Mapping[str, int],
    run_id: str,
    *,
    durable: bool,
    max_plan_bytes: int,
) -> dict[str, Any]:
    if max_plan_bytes < 1 or max_plan_bytes > MAX_METADATA_JSON_BYTES:
        raise CacheSafetyError(f"max_plan_bytes must be between 1 and {MAX_METADATA_JSON_BYTES}")
    candidates = sorted(set(scanned) - set(active.blobs))
    records = _candidate_records(
        root,
        uid,
        candidates,
        scanned,
        durable=durable,
    )
    if durable and records:
        directories = {root}
        for record in records:
            current = _resolve_under(root, record["relative_path"]).parent
            while True:
                directories.add(current)
                if current == root:
                    break
                current = current.parent
        for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            _guard_owned_directory(directory, uid)
            _fsync_directory(directory)
    document = {
        "active_manifest_sha256": sorted(
            manifest["manifest_sha256"] for manifest in active.manifests
        ),
        "candidates": records,
        "created_at": _utc_now(),
        "root": _root_fingerprint(root, uid),
        "run_id": run_id,
        "schema": QUARANTINE_PLAN_SCHEMA,
        "summary": {
            "payload_bytes": sum(record["size"] for record in records),
            "payload_files": len(records),
        },
    }
    plan = _with_document_id(document, "plan_sha256")
    encoded_size = len(_pretty_bytes(plan))
    if encoded_size > max_plan_bytes:
        raise CacheSafetyError(
            f"quarantine plan would be {encoded_size} bytes, exceeding the bounded "
            f"limit {max_plan_bytes}; no run was created"
        )
    return plan


def _validate_plan(document: dict[str, Any], root: Path, uid: int, run_id: str) -> dict[str, Any]:
    _require_exact_keys(
        document,
        {
            "active_manifest_sha256",
            "candidates",
            "created_at",
            "plan_sha256",
            "root",
            "run_id",
            "schema",
            "summary",
        },
        "quarantine plan",
    )
    if document["schema"] != QUARANTINE_PLAN_SCHEMA or document["run_id"] != run_id:
        raise CacheSafetyError("quarantine plan identity mismatch")
    _verify_document_id(document, "plan_sha256", "quarantine plan")
    if document["root"] != _root_fingerprint(root, uid):
        raise CacheSafetyError("quarantine plan belongs to a different cache root")
    if not isinstance(document["created_at"], str) or not document["created_at"].endswith("Z"):
        raise CacheSafetyError("quarantine plan has an invalid creation timestamp")
    active_manifests = document["active_manifest_sha256"]
    if not isinstance(active_manifests, list) or active_manifests != sorted(set(active_manifests)):
        raise CacheSafetyError("active manifest hashes must be a sorted unique list")
    for manifest_sha in active_manifests:
        _require_sha256(manifest_sha, "quarantine active manifest SHA-256")
    candidates = document["candidates"]
    if not isinstance(candidates, list):
        raise CacheSafetyError("quarantine candidates must be a list")
    seen: set[str] = set()
    total = 0
    for index, raw in enumerate(candidates):
        record = _as_object(raw, f"candidates[{index}]")
        _require_exact_keys(
            record,
            {"payload_sha256", "relative_path", "size"},
            f"candidates[{index}]",
        )
        relative = _validate_relative_path(record["relative_path"], "candidate relative_path")
        _validate_blob_relative_path(relative)
        if relative in seen:
            raise CacheSafetyError(f"duplicate quarantine candidate: {relative}")
        seen.add(relative)
        if (
            not isinstance(record["size"], int)
            or isinstance(record["size"], bool)
            or record["size"] < 0
        ):
            raise CacheSafetyError(f"invalid candidate size: {relative}")
        _require_sha256(record["payload_sha256"], "candidate payload_sha256")
        total += record["size"]
    if document["summary"] != {"payload_bytes": total, "payload_files": len(candidates)}:
        raise CacheSafetyError("quarantine plan summary mismatch")
    return document


def _validate_complete_marker(document: dict[str, Any], plan: dict[str, Any]) -> None:
    _require_exact_keys(
        document,
        {"completed_at", "plan_sha256", "run_id", "schema", "summary"},
        "quarantine completion marker",
    )
    if (
        document["schema"] != QUARANTINE_COMPLETE_SCHEMA
        or document["run_id"] != plan["run_id"]
        or document["plan_sha256"] != plan["plan_sha256"]
        or document["summary"] != plan["summary"]
        or not isinstance(document["completed_at"], str)
        or not document["completed_at"].endswith("Z")
    ):
        raise CacheSafetyError("quarantine completion marker does not match its plan")


def _validate_purging_marker(document: dict[str, Any], plan: dict[str, Any]) -> None:
    _require_exact_keys(
        document,
        {"plan_sha256", "run_id", "schema", "started_at"},
        "quarantine purging marker",
    )
    if (
        document["schema"] != PURGING_SCHEMA
        or document["run_id"] != plan["run_id"]
        or document["plan_sha256"] != plan["plan_sha256"]
        or not isinstance(document["started_at"], str)
        or not document["started_at"].endswith("Z")
    ):
        raise CacheSafetyError("quarantine purging marker does not match its plan")


def _validate_purged_marker(document: dict[str, Any], plan: dict[str, Any]) -> None:
    _require_exact_keys(
        document,
        {"completed_at", "plan_sha256", "run_id", "schema", "summary"},
        "quarantine purged marker",
    )
    if (
        document["schema"] != PURGED_SCHEMA
        or document["run_id"] != plan["run_id"]
        or document["plan_sha256"] != plan["plan_sha256"]
        or document["summary"] != plan["summary"]
        or not isinstance(document["completed_at"], str)
        or not document["completed_at"].endswith("Z")
    ):
        raise CacheSafetyError("quarantine purged marker does not match its plan")


def _validate_deferred_marker(document: dict[str, Any], plan: dict[str, Any]) -> None:
    _require_exact_keys(
        document,
        {
            "active_manifest_sha256",
            "active_relative_paths",
            "checked_at",
            "plan_sha256",
            "run_id",
            "schema",
        },
        "quarantine deferred marker",
    )
    paths = document["active_relative_paths"]
    manifests = document["active_manifest_sha256"]
    if not isinstance(paths, list) or not paths or paths != sorted(set(paths)):
        raise CacheSafetyError("deferred active paths must be a nonempty sorted unique list")
    planned_paths = {record["relative_path"] for record in plan["candidates"]}
    for relative in paths:
        _validate_blob_relative_path(relative)
        if relative not in planned_paths:
            raise CacheSafetyError("deferred active path is absent from its quarantine plan")
    if not isinstance(manifests, list) or not manifests or manifests != sorted(set(manifests)):
        raise CacheSafetyError("deferred manifest hashes must be a nonempty sorted unique list")
    for manifest_sha in manifests:
        _require_sha256(manifest_sha, "deferred active manifest SHA-256")
    if (
        document["schema"] != DEFERRED_SCHEMA
        or document["run_id"] != plan["run_id"]
        or document["plan_sha256"] != plan["plan_sha256"]
        or not isinstance(document["checked_at"], str)
        or not document["checked_at"].endswith("Z")
    ):
        raise CacheSafetyError("quarantine deferred marker does not match its plan")


def _quarantine_blob_path(root: Path, run_id: str, relative: str) -> Path:
    return _run_directory(root, run_id) / "payload" / PurePosixPath(relative)


def _scan_quarantine_payload_tree(
    root: Path,
    run_id: str,
    uid: int,
    plan: Mapping[str, Any],
) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    """Enumerate only planned payload names and recoverable move temporaries."""

    payload_root = _run_directory(root, run_id) / "payload"
    if not _lstat_exists(payload_root):
        return (), ()
    _guard_owned_chain(root, payload_root, uid)
    planned = {record["relative_path"] for record in plan["candidates"]}
    planned_parents = {PurePosixPath(relative).parent.as_posix() for relative in planned}
    temporary_paths: list[Path] = []
    payload_paths: list[str] = []

    def walk(directory: Path) -> None:
        _guard_owned_directory(directory, uid)
        for entry in sorted(os.scandir(directory), key=lambda item: item.name):
            path = Path(entry.path)
            if entry.is_symlink():
                raise CacheSafetyError(f"symlink in quarantine payload tree: {path}")
            if entry.is_dir(follow_symlinks=False):
                walk(path)
                continue
            if not entry.is_file(follow_symlinks=False):
                raise CacheSafetyError(f"special file in quarantine payload tree: {path}")
            if _MOVE_TEMP_RE.fullmatch(entry.name) is not None:
                parent_relative = path.parent.relative_to(payload_root).as_posix()
                if parent_relative not in planned_parents:
                    raise CacheSafetyError(f"unattributable move temporary: {path}")
                temporary_fd, _ = _safe_open_regular(path, uid)
                os.close(temporary_fd)
                temporary_paths.append(path)
                continue
            relative = path.relative_to(payload_root).as_posix()
            _validate_blob_relative_path(relative)
            if relative not in planned:
                raise CacheSafetyError(f"unplanned file in quarantine payload tree: {path}")
            payload_paths.append(relative)

    walk(payload_root)
    return tuple(temporary_paths), tuple(payload_paths)


def _copy_unlink_durable(
    source: Path,
    destination: Path,
    root: Path,
    uid: int,
    record: Mapping[str, Any],
    *,
    fault: FaultHook | None,
) -> None:
    """Publish a durable destination before unlinking and syncing the source."""

    if source == destination:
        raise CacheSafetyError(f"refusing a self move: {source}")
    _guard_owned_chain(root, source.parent, uid)
    _mkdir_owned(destination.parent, root, uid)
    if _lstat_exists(destination):
        raise CacheSafetyError(f"destination already exists: {destination}")
    source_fd, before = _safe_open_regular(source, uid)
    if before.st_size != record["size"]:
        os.close(source_fd)
        raise CacheSafetyError(f"source size changed before durable copy: {source}")
    temporary = destination.parent / f".move-{uuid.uuid4().hex}.tmp"
    try:
        destination_fd = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
    except Exception:
        os.close(source_fd)
        raise
    published = False
    try:
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(source_fd, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            _write_all(destination_fd, chunk)
            copied += len(chunk)
        if copied != record["size"] or digest.hexdigest() != record["payload_sha256"]:
            raise CacheSafetyError(
                f"source changed or failed SHA-256 during durable copy: {source}"
            )
        after = os.fstat(source_fd)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise CacheSafetyError(f"source changed during durable copy: {source}")
        os.fsync(destination_fd)
        os.fchmod(destination_fd, 0o600)
        os.close(destination_fd)
        destination_fd = -1
        temporary.replace(destination)
        published = True
        # Destination entry durability must precede any source removal.
        _fsync_directory(destination.parent)
        _hash_regular_file(
            destination,
            uid,
            expected_size=record["size"],
            expected_sha256=record["payload_sha256"],
        )
        if fault is not None:
            fault(f"after_destination_publish:{record['relative_path']}")
        source.unlink()
        _fsync_directory(source.parent)
        if fault is not None:
            fault(f"after_source_unlink:{record['relative_path']}")
    finally:
        os.close(source_fd)
        if destination_fd >= 0:
            os.close(destination_fd)
        if not published:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            else:
                _fsync_directory(temporary.parent)


def _validate_candidate_locations(
    source: Path,
    destination: Path,
    root: Path,
    uid: int,
    record: Mapping[str, Any],
) -> tuple[bool, bool]:
    """Authenticate either name, including a verified two-name crash state."""

    source_exists = _lstat_exists(source)
    destination_exists = _lstat_exists(destination)
    if not source_exists and not destination_exists:
        raise CacheSafetyError(
            f"candidate is missing from active and quarantine paths: {record['relative_path']}"
        )
    if source_exists:
        _guard_owned_chain(root, source.parent, uid)
    if destination_exists:
        _guard_owned_chain(root, destination.parent, uid)
    if source_exists and destination_exists:
        source_info = source.lstat()
        destination_info = destination.lstat()
        same_inode = (
            source_info.st_dev == destination_info.st_dev
            and source_info.st_ino == destination_info.st_ino
        )
        allowed_links = (2,) if same_inode else (1,)
        if same_inode and (source_info.st_nlink != 2 or destination_info.st_nlink != 2):
            raise CacheSafetyError(
                "two-name hard-link recovery has an unexpected link count: "
                f"{record['relative_path']}"
            )
        for path in (source, destination):
            _hash_regular_file(
                path,
                uid,
                expected_size=record["size"],
                expected_sha256=record["payload_sha256"],
                allowed_nlinks=allowed_links,
            )
        return True, True
    current = source if source_exists else destination
    _hash_regular_file(
        current,
        uid,
        expected_size=record["size"],
        expected_sha256=record["payload_sha256"],
    )
    return source_exists, destination_exists


def _remove_redundant_name(path: Path, uid: int) -> None:
    _guard_owned_directory(path.parent, uid)
    path.unlink()
    _fsync_directory(path.parent)


def _complete_quarantine(
    root: Path,
    uid: int,
    plan: dict[str, Any],
    *,
    fault: FaultHook | None,
) -> None:
    run_id = plan["run_id"]
    for record in plan["candidates"]:
        relative = record["relative_path"]
        source = _resolve_under(root, relative)
        destination = _quarantine_blob_path(root, run_id, relative)
        _copy_unlink_durable(
            source,
            destination,
            root,
            uid,
            record,
            fault=fault,
        )
        if fault is not None:
            fault(f"after_quarantine_move:{relative}")
    for record in plan["candidates"]:
        relative = record["relative_path"]
        source_exists, destination_exists = _validate_candidate_locations(
            _resolve_under(root, relative),
            _quarantine_blob_path(root, run_id, relative),
            root,
            uid,
            record,
        )
        if source_exists or not destination_exists:
            raise CacheSafetyError(
                f"quarantine cannot complete while an active name remains: {relative}"
            )
    complete = {
        "completed_at": _utc_now(),
        "plan_sha256": plan["plan_sha256"],
        "run_id": run_id,
        "schema": QUARANTINE_COMPLETE_SCHEMA,
        "summary": plan["summary"],
    }
    _publish_immutable(_run_directory(root, run_id) / "complete.json", complete, root, uid)


def garbage_collect(
    root: Path,
    *,
    apply: bool = False,
    confirm_root: str | None = None,
    engine_stopped: bool = False,
    proc_root: Path = Path("/proc"),
    allow_empty_mark_set: bool = False,
    expected_uid: int | None = None,
    profile: CacheProfile = PRODUCTION_PROFILE,
    run_id: str | None = None,
    fault: FaultHook | None = None,
    max_plan_bytes: int = MAX_METADATA_JSON_BYTES,
) -> dict[str, Any]:
    """Mark all live refs and optionally quarantine every unreferenced blob."""

    uid = os.getuid() if expected_uid is None else expected_uid
    root = _guard_root(root, expected_uid=uid)
    if apply:
        root = require_apply_safety(
            root,
            confirm_root=confirm_root,
            engine_stopped=engine_stopped,
            proc_root=proc_root,
            expected_uid=uid,
        )

    def perform() -> dict[str, Any]:
        incomplete = _incomplete_run_ids(root, uid)
        if incomplete:
            raise CacheSafetyError(
                "incomplete quarantine must be recovered before new GC: " + ", ".join(incomplete)
            )
        active = _load_active_state(root, uid, profile, verify_payloads=True)
        if not active.refs and not allow_empty_mark_set:
            raise CacheSafetyError(
                "active mark set is empty; pass --allow-empty-mark-set only when that is "
                "intentional"
            )
        scanned = _scan_payload_files(root, uid, profile)
        missing = sorted(set(active.blobs) - set(scanned))
        if missing:
            raise CacheSafetyError(f"active blobs are missing: {missing[:3]}")
        selected_run_id = run_id or _new_run_id()
        if _RUN_ID_RE.fullmatch(selected_run_id) is None:
            raise CacheSafetyError(f"invalid quarantine run id: {selected_run_id}")
        plan = _quarantine_plan(
            root,
            uid,
            active,
            scanned,
            selected_run_id,
            durable=apply,
            max_plan_bytes=max_plan_bytes,
        )
        result = {
            "action": "gc",
            "active_payload_bytes": sum(blob.size for blob in active.blobs.values()),
            "active_payload_files": len(active.blobs),
            "active_refs": len(active.refs),
            "apply": apply,
            "quarantine_bytes": plan["summary"]["payload_bytes"],
            "quarantine_files": plan["summary"]["payload_files"],
            "plan_bytes": len(_pretty_bytes(plan)),
            "run_id": selected_run_id,
        }
        if not apply or not plan["candidates"]:
            return result
        run_directory = _run_directory(root, selected_run_id)
        try:
            run_directory.lstat()
        except FileNotFoundError:
            _mkdir_owned(run_directory, root, uid)
        else:
            raise CacheSafetyError(f"quarantine run already exists: {run_directory}")
        _publish_immutable(_plan_path(root, selected_run_id), plan, root, uid)
        if fault is not None:
            fault("after_quarantine_plan")
        _complete_quarantine(root, uid, plan, fault=fault)
        return result

    if not apply:
        return perform()
    with _exclusive_lock(root, uid):
        return perform()


def recover_quarantine(
    root: Path,
    run_id: str,
    *,
    apply: bool = False,
    confirm_root: str | None = None,
    engine_stopped: bool = False,
    proc_root: Path = Path("/proc"),
    expected_uid: int | None = None,
    profile: CacheProfile = PRODUCTION_PROFILE,
    fault: FaultHook | None = None,
) -> dict[str, Any]:
    """Reconcile a crash-interrupted quarantine against the current live mark set."""

    uid = os.getuid() if expected_uid is None else expected_uid
    root = _guard_root(root, expected_uid=uid)
    if apply:
        root = require_apply_safety(
            root,
            confirm_root=confirm_root,
            engine_stopped=engine_stopped,
            proc_root=proc_root,
            expected_uid=uid,
        )

    def perform() -> dict[str, Any]:
        _guard_owned_chain(root, _run_directory(root, run_id), uid)
        plan = _validate_plan(_read_json(_plan_path(root, run_id), uid), root, uid, run_id)
        temporary_paths, _ = _scan_quarantine_payload_tree(root, run_id, uid, plan)
        complete_path = _run_directory(root, run_id) / "complete.json"
        deferred_path = _run_directory(root, run_id) / "deferred.json"
        if _lstat_exists(complete_path):
            if temporary_paths:
                raise CacheSafetyError("completed quarantine contains move temporaries")
            if _lstat_exists(deferred_path):
                raise CacheSafetyError("completed quarantine still has a deferred marker")
            _validate_complete_marker(_read_json(complete_path, uid), plan)
            active = _load_active_state(root, uid, profile, verify_payloads=True)
            planned = {record["relative_path"] for record in plan["candidates"]}
            marked_candidates = sorted(planned & set(active.blobs))
            if marked_candidates:
                raise CacheSafetyError(
                    "completed quarantine has candidates in the current mark set: "
                    f"{marked_candidates[:3]}"
                )
            for record in plan["candidates"]:
                source_exists, destination_exists = _validate_candidate_locations(
                    _resolve_under(root, record["relative_path"]),
                    _quarantine_blob_path(root, run_id, record["relative_path"]),
                    root,
                    uid,
                    record,
                )
                if source_exists or not destination_exists:
                    raise CacheSafetyError(
                        "completed quarantine no longer has an exclusively quarantined "
                        f"candidate: {record['relative_path']}"
                    )
            return {
                "action": "recover",
                "already_complete": True,
                "apply": apply,
                "complete": True,
                "deferred_active_files": 0,
                "restore_files": 0,
                "run_id": run_id,
                "temporary_files": 0,
                "quarantine_files": len(plan["candidates"]),
                "two_name_files": 0,
            }
        if _lstat_exists(deferred_path):
            _validate_deferred_marker(_read_json(deferred_path, uid), plan)
        active = _load_active_state(root, uid, profile, verify_payloads=False)
        marked = set(active.blobs)
        planned = {record["relative_path"] for record in plan["candidates"]}
        for relative, active_blob in active.blobs.items():
            if relative in planned:
                continue
            _hash_regular_file(
                _resolve_under(root, relative),
                uid,
                expected_size=active_blob.size,
                expected_sha256=active_blob.payload_sha256,
            )
        dispositions: list[tuple[dict[str, Any], Path, Path, str, bool, bool]] = []
        for record in plan["candidates"]:
            relative = record["relative_path"]
            if relative in marked:
                active_blob = active.blobs[relative]
                if (
                    active_blob.size != record["size"]
                    or active_blob.payload_sha256 != record["payload_sha256"]
                ):
                    raise CacheSafetyError(
                        f"current ref conflicts with quarantine metadata: {relative}"
                    )
            source = _resolve_under(root, relative)
            destination = _quarantine_blob_path(root, run_id, relative)
            source_exists, destination_exists = _validate_candidate_locations(
                source,
                destination,
                root,
                uid,
                record,
            )
            desired = "active" if relative in marked else "quarantine"
            dispositions.append(
                (record, source, destination, desired, source_exists, destination_exists)
            )
        restore_count = sum(
            1
            for _, _, _, desired, source_exists, _ in dispositions
            if desired == "active" and not source_exists
        )
        quarantine_count = sum(
            1
            for _, _, _, desired, _, destination_exists in dispositions
            if desired == "quarantine" and not destination_exists
        )
        two_name_count = sum(
            1
            for _, _, _, _, source_exists, destination_exists in dispositions
            if source_exists and destination_exists
        )
        deferred_paths = sorted(planned & marked)
        result = {
            "action": "recover",
            "already_complete": False,
            "apply": apply,
            "complete": False,
            "deferred_active_files": len(deferred_paths),
            "quarantine_files": quarantine_count,
            "restore_files": restore_count,
            "run_id": run_id,
            "temporary_files": len(temporary_paths),
            "two_name_files": two_name_count,
        }
        if not apply:
            return result
        for temporary_path in temporary_paths:
            _remove_redundant_name(temporary_path, uid)
            if fault is not None:
                fault(f"after_recovery_temp_unlink:{temporary_path.name}")
        for record, source, destination, desired, source_exists, destination_exists in dispositions:
            if source_exists and destination_exists:
                redundant = destination if desired == "active" else source
                _remove_redundant_name(redundant, uid)
            elif desired == "active" and not source_exists:
                _copy_unlink_durable(
                    destination,
                    source,
                    root,
                    uid,
                    record,
                    fault=fault,
                )
            elif desired == "quarantine" and not destination_exists:
                _copy_unlink_durable(
                    source,
                    destination,
                    root,
                    uid,
                    record,
                    fault=fault,
                )
            if fault is not None:
                fault(f"after_recovery_move:{record['relative_path']}")
        _load_active_state(root, uid, profile, verify_payloads=True)
        run_directory = _run_directory(root, run_id)
        if deferred_paths:
            for record in plan["candidates"]:
                if record["relative_path"] not in marked:
                    continue
                source_exists, destination_exists = _validate_candidate_locations(
                    _resolve_under(root, record["relative_path"]),
                    _quarantine_blob_path(root, run_id, record["relative_path"]),
                    root,
                    uid,
                    record,
                )
                if not source_exists or destination_exists:
                    raise CacheSafetyError(
                        "deferred candidate is not restored exclusively active: "
                        f"{record['relative_path']}"
                    )
            deferred = {
                "active_manifest_sha256": sorted(
                    manifest["manifest_sha256"] for manifest in active.manifests
                ),
                "active_relative_paths": deferred_paths,
                "checked_at": _utc_now(),
                "plan_sha256": plan["plan_sha256"],
                "run_id": run_id,
                "schema": DEFERRED_SCHEMA,
            }
            _publish_replace(deferred_path, deferred, root, uid)
            return result
        for record in plan["candidates"]:
            source_exists, destination_exists = _validate_candidate_locations(
                _resolve_under(root, record["relative_path"]),
                _quarantine_blob_path(root, run_id, record["relative_path"]),
                root,
                uid,
                record,
            )
            if source_exists or not destination_exists:
                raise CacheSafetyError(
                    "quarantine recovery cannot complete with an active candidate: "
                    f"{record['relative_path']}"
                )
        if _lstat_exists(deferred_path):
            _validate_deferred_marker(_read_json(deferred_path, uid), plan)
            deferred_path.unlink()
            _fsync_directory(run_directory)
        complete = {
            "completed_at": _utc_now(),
            "plan_sha256": plan["plan_sha256"],
            "run_id": run_id,
            "schema": QUARANTINE_COMPLETE_SCHEMA,
            "summary": plan["summary"],
        }
        _publish_immutable(complete_path, complete, root, uid)
        result["complete"] = True
        return result

    if not apply:
        return perform()
    with _exclusive_lock(root, uid):
        return perform()


def purge_quarantine(
    root: Path,
    run_id: str,
    *,
    apply: bool = False,
    confirm_root: str | None = None,
    engine_stopped: bool = False,
    proc_root: Path = Path("/proc"),
    expected_uid: int | None = None,
    profile: CacheProfile = PRODUCTION_PROFILE,
    fault: FaultHook | None = None,
) -> dict[str, Any]:
    """Permanently unlink only payloads from a completed quarantine run."""

    uid = os.getuid() if expected_uid is None else expected_uid
    root = _guard_root(root, expected_uid=uid)
    if apply:
        root = require_apply_safety(
            root,
            confirm_root=confirm_root,
            engine_stopped=engine_stopped,
            proc_root=proc_root,
            expected_uid=uid,
        )

    def perform() -> dict[str, Any]:
        run_directory = _run_directory(root, run_id)
        _guard_owned_chain(root, run_directory, uid)
        plan = _validate_plan(_read_json(_plan_path(root, run_id), uid), root, uid, run_id)
        complete = _read_json(run_directory / "complete.json", uid)
        _validate_complete_marker(complete, plan)
        temporary_paths, _ = _scan_quarantine_payload_tree(root, run_id, uid, plan)
        if temporary_paths:
            raise CacheSafetyError("completed quarantine contains move temporaries")
        deferred_path = run_directory / "deferred.json"
        if _lstat_exists(deferred_path):
            _validate_deferred_marker(_read_json(deferred_path, uid), plan)
            raise CacheSafetyError("deferred quarantine cannot be purged")
        active = _load_active_state(root, uid, profile, verify_payloads=True)
        marked_candidates = sorted(
            record["relative_path"]
            for record in plan["candidates"]
            if record["relative_path"] in active.blobs
        )
        if marked_candidates:
            raise CacheSafetyError(
                f"quarantine became referenced and cannot be purged: {marked_candidates[:3]}"
            )
        purging_path = run_directory / "purging.json"
        purged_path = run_directory / "purged.json"
        if _lstat_exists(purged_path):
            if not _lstat_exists(purging_path):
                raise CacheSafetyError("purged quarantine has no purging marker")
            _validate_purging_marker(_read_json(purging_path, uid), plan)
            _validate_purged_marker(_read_json(purged_path, uid), plan)
            return {
                "action": "purge",
                "already_purged": True,
                "apply": apply,
                "purge_bytes": plan["summary"]["payload_bytes"],
                "purge_files": plan["summary"]["payload_files"],
                "run_id": run_id,
            }
        purging = _lstat_exists(purging_path)
        if purging:
            _validate_purging_marker(_read_json(purging_path, uid), plan)
        existing = []
        for record in plan["candidates"]:
            relative = record["relative_path"]
            if _lstat_exists(_resolve_under(root, relative)):
                raise CacheSafetyError(f"quarantined payload reappeared in active tree: {relative}")
            path = _quarantine_blob_path(root, run_id, relative)
            if not _lstat_exists(path):
                if purging:
                    continue
                raise CacheSafetyError(f"quarantine payload is missing before purge: {relative}")
            _hash_regular_file(
                path,
                uid,
                expected_size=record["size"],
                expected_sha256=record["payload_sha256"],
            )
            existing.append((record, path))
        result = {
            "action": "purge",
            "already_purged": False,
            "apply": apply,
            "purge_bytes": plan["summary"]["payload_bytes"],
            "purge_files": plan["summary"]["payload_files"],
            "run_id": run_id,
        }
        if not apply:
            return result
        if not purging:
            marker = {
                "plan_sha256": plan["plan_sha256"],
                "run_id": run_id,
                "schema": PURGING_SCHEMA,
                "started_at": _utc_now(),
            }
            _publish_immutable(purging_path, marker, root, uid)
        for record, path in existing:
            path.unlink()
            _fsync_directory(path.parent)
            if fault is not None:
                fault(f"after_purge_unlink:{record['relative_path']}")
        purged = {
            "completed_at": _utc_now(),
            "plan_sha256": plan["plan_sha256"],
            "run_id": run_id,
            "schema": PURGED_SCHEMA,
            "summary": plan["summary"],
        }
        _publish_immutable(purged_path, purged, root, uid)
        return result

    if not apply:
        return perform()
    with _exclusive_lock(root, uid):
        return perform()


def retire_ref(
    root: Path,
    session_id: str,
    branch: str,
    *,
    apply: bool = False,
    confirm_root: str | None = None,
    engine_stopped: bool = False,
    proc_root: Path = Path("/proc"),
    expected_uid: int | None = None,
    fault: FaultHook | None = None,
) -> dict[str, Any]:
    """Move one branch ref into recoverable retired metadata before a later GC."""

    uid = os.getuid() if expected_uid is None else expected_uid
    root = _guard_root(root, expected_uid=uid)
    if apply:
        root = require_apply_safety(
            root,
            confirm_root=confirm_root,
            engine_stopped=engine_stopped,
            proc_root=proc_root,
            expected_uid=uid,
        )

    def perform() -> dict[str, Any]:
        ref_path = _ref_path(root, session_id, branch)
        ref = _read_current_ref(root, session_id, branch, uid)
        if ref is None:
            raise CacheSafetyError(f"live ref does not exist: {ref_path}")
        ref_sha256 = _sha256_bytes(_canonical_bytes(ref))
        retired = _metadata_root(root) / "retired-refs" / session_id / branch / f"{ref_sha256}.json"
        result = {
            "action": "retire-ref",
            "apply": apply,
            "branch": branch,
            "manifest_sha256": ref["manifest_sha256"],
            "retired_path": retired.relative_to(root).as_posix(),
            "session_id": session_id,
        }
        if not apply:
            return result
        # Publish and sync the recoverable destination before removing the live
        # name. A crash in between intentionally leaves two independently
        # authenticated copies; retrying the same retirement resolves that state.
        _publish_immutable(retired, ref, root, uid)
        _hash_regular_file(
            retired,
            uid,
            expected_sha256=_sha256_bytes(_pretty_bytes(ref)),
        )
        if fault is not None:
            fault("after_retired_ref_publish")
        if _validate_ref(_read_json(ref_path, uid), str(ref_path)) != ref:
            raise CacheSafetyError("live ref changed while its retirement was being published")
        ref_path.unlink()
        _fsync_directory(ref_path.parent)
        if fault is not None:
            fault("after_live_ref_unlink")
        return result

    if not apply:
        return perform()
    with _exclusive_lock(root, uid):
        return perform()


def find_engine_processes(proc_root: Path = Path("/proc")) -> tuple[tuple[int, str], ...]:
    """Return processes whose comm or argv identifies vLLM API/EngineCore."""

    result = []
    unreadable = []
    try:
        entries = tuple(proc_root.iterdir())
    except OSError as error:
        raise CacheSafetyError(f"cannot inspect process table {proc_root}: {error}") from error
    for entry in entries:
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
            comm_path = entry / "comm"
            comm = comm_path.read_text(encoding="utf-8", errors="replace").strip()
        except FileNotFoundError:
            continue
        except PermissionError:
            unreadable.append(entry.name)
            continue
        normalized = cmdline.replace(b"\x00", b" ")
        if any(marker in normalized for marker in _ENGINE_CMDLINE_MARKERS) or any(
            comm.startswith(marker) for marker in _ENGINE_COMM_MARKERS
        ):
            result.append((int(entry.name), normalized.decode(errors="replace") or comm))
    if unreadable:
        raise CacheSafetyError(
            "cannot prove engine is stopped because process entries are unreadable: "
            + ", ".join(sorted(unreadable)[:8])
        )
    return tuple(sorted(result))


def require_apply_safety(
    root: Path,
    *,
    confirm_root: str | None,
    engine_stopped: bool,
    proc_root: Path = Path("/proc"),
    expected_uid: int | None = None,
) -> Path:
    """Require an exact root acknowledgement and independently empty engine scan."""

    root = _guard_root(root, expected_uid=expected_uid)
    if not engine_stopped:
        raise CacheSafetyError("--apply requires the explicit --engine-stopped acknowledgement")
    if confirm_root != str(root):
        raise CacheSafetyError(f"--confirm-root must exactly equal the resolved root: {root}")
    engines = find_engine_processes(proc_root)
    if engines:
        summary = ", ".join(f"pid={pid} {command[:100]}" for pid, command in engines)
        raise CacheSafetyError(f"vLLM processes are still running: {summary}")
    return root


HELP = """NAME
    qwen-250k-cache - seal and compact the exact Qwen 249,957-token P-8 disk cache

SYNOPSIS
    qwen-250k-cache attest-historical-fixed-slot --profile fixed-slot-60k --root PATH
        --original-selection FILE [--apply SAFETY_OPTIONS]
    qwen-250k-cache import-fixed-slot --profile fixed-slot-60k --root PATH --source-root PATH
        --selection FILE [--apply SAFETY_OPTIONS]
    qwen-250k-cache seal --profile PROFILE --root PATH --selection FILE [--apply SAFETY_OPTIONS]
    qwen-250k-cache verify --profile PROFILE --root PATH
    qwen-250k-cache gc --profile PROFILE --root PATH [--allow-empty-mark-set]
        [--apply SAFETY_OPTIONS]
    qwen-250k-cache recover --profile PROFILE --root PATH --run-id ID [--apply SAFETY_OPTIONS]
    qwen-250k-cache purge --profile PROFILE --root PATH --run-id ID [--apply SAFETY_OPTIONS]
    qwen-250k-cache retire-ref --profile PROFILE --root PATH --session-id ID --branch NAME
        [--apply SAFETY_OPTIONS]

DESCRIPTION
    Seals immutable, ABI/model/tokenizer-bound session generations and performs
    stopped-engine mark/sweep over vLLM's content-keyed .bin payloads. Every
    mutating command is a dry run unless --apply is supplied. GC first moves
    unreferenced payloads into an auditable quarantine; purge is a separate step.

OPTIONS
    --root PATH
        Existing vLLM filesystem-tier root. Symlinks and wrong-owner entries fail.
    --profile PROFILE
        Exact geometry: 250k (default) or fixed-slot-60k. Different profiles must
        use different roots; manifests cannot cross profile geometry.
    --selection FILE
        Runtime-produced, drained P-8 key-chain selection for import or seal.
    --original-selection FILE
        Exact immutable pre-receipt fixed-slot selection. Only a source-controlled,
        hash-pinned historical producer can be attested; arbitrary files fail closed.
    --source-root PATH
        Read-only captured source namespace for import-fixed-slot. It must contain
        the identical namespace config and every selected authenticated payload.
    --run-id ID
        Exact quarantine run identifier reported by gc.
    --session-id ID, --branch NAME
        Exact live reference to retire before a later mark/sweep.
    --allow-empty-mark-set
        Permit gc to quarantine every payload when no active refs exist.
    --apply
        Perform the mutation. Omission is always a dry run.
    --engine-stopped
        Explicit acknowledgement required with --apply; /proc is also scanned.
    --confirm-root PATH
        Must byte-for-byte equal the normalized cache root with --apply.
    -h, --help
        Show this help text and exit.

OPERATION
    attest-historical-fixed-slot authenticates the original selection, complete runtime
    evidence, reviewed critical source hashes and preserved launch evidence, then emits
    a separate immutable derived selection. It never edits or relabels the original.
    import-fixed-slot authenticates all 665 source and destination payloads,
    then atomically reflinks or copies only missing fixed-slot files without
    replacing an existing content key. It never advances a session reference.
    seal hashes all selected payloads (2,553 for 250k; 665 for fixed-slot-60k),
    publishes an immutable manifest, then atomically advances one branch ref.
    verify authenticates every active payload
    and complete manifest ancestry. gc marks the union of all branch refs, bounds
    its durable plan to 16 MiB, and publishes each unmarked quarantine destination
    before unlinking its source. recover authenticates one- or two-name crash states
    and defers completion for newly active pages. purge validates a completed
    quarantine again before unlinking only its quarantined payloads.

EXAMPLES
    UV_CACHE_DIR=/data/.cache/uv uv run python scripts/qwen-250k-cache \\
        attest-historical-fixed-slot --profile fixed-slot-60k \\
        --root /path/to/dflash-agent262-fixed-slot-v1 \\
        --original-selection /path/to/preserved/selection.json
    UV_CACHE_DIR=/data/.cache/uv uv run python scripts/qwen-250k-cache \\
        import-fixed-slot --profile fixed-slot-60k \\
        --root /path/to/dflash-agent262-fixed-slot-v1 \\
        --source-root /path/to/captured-dflash-agent262-v1 \\
        --selection selection.json
    UV_CACHE_DIR=/data/.cache/uv uv run python scripts/qwen-250k-cache \\
        seal --root /path/to/dflash-agent262-v1 --selection selection.json
    UV_CACHE_DIR=/data/.cache/uv uv run python scripts/qwen-250k-cache \\
        gc --root /path/to/dflash-agent262-v1
    UV_CACHE_DIR=/data/.cache/uv uv run python scripts/qwen-250k-cache \\
        gc --root /path/to/dflash-agent262-v1 --apply --engine-stopped \\
        --confirm-root /path/to/dflash-agent262-v1

FILES
    ROOT/.qwen-250k-cache-v1/LIFECYCLE.lock
        Shared for the full engine lifetime; exclusive for every apply operation.
    ROOT/.qwen-250k-cache-v1/manifests/
        Immutable generation manifests.
    ROOT/.qwen-250k-cache-v1/historical-selections/
        Immutable, allowlisted attestations of preserved pre-receipt selections.
    ROOT/.qwen-250k-cache-v1/refs/
        Atomic live session/branch heads.
    ROOT/.qwen-250k-cache-v1/quarantine/
        Durable plans and quarantined payloads; never active vLLM storage.

PATHS
    Payload paths must exactly match vLLM FileMapper's rank/hash/group ABI. Paths
    are canonical relative POSIX names and may not contain links or traversal.

SECURITY NOTES
    Apply requires the exact root, an explicit stopped-engine acknowledgement, and
    an independent /proc scan plus the exclusive lifecycle lock. Owner, permission,
    regular-file, link-count, size, SHA-256, ancestry, namespace, model, tokenizer,
    ABI, and generation checks all fail closed. Import never trusts file names alone:
    source bytes are hashed and a conflicting destination aborts without replacement.
    Historical attestation cannot add producers at runtime and never claims the current
    artifact generated old payload bytes.

EXIT STATUS
    0 on success; 2 for invalid input, unsafe state, corruption, or a failed guard.

AUTHORS
    qwen-r9700 lab contributors.
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--help", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    def common(name: str) -> argparse.ArgumentParser:
        child = subparsers.add_parser(name, add_help=False)
        child.add_argument("-h", "--help", action="store_true")
        child.add_argument("--root", type=Path, required=False)
        child.add_argument("--profile", choices=tuple(PROFILES), default="250k")
        return child

    def mutating(name: str) -> argparse.ArgumentParser:
        child = common(name)
        child.add_argument("--apply", action="store_true")
        child.add_argument("--engine-stopped", action="store_true")
        child.add_argument("--confirm-root")
        return child

    seal = mutating("seal")
    seal.add_argument("--selection", type=Path)
    attest = mutating("attest-historical-fixed-slot")
    attest.add_argument("--original-selection", type=Path)
    import_fixed = mutating("import-fixed-slot")
    import_fixed.add_argument("--selection", type=Path)
    import_fixed.add_argument("--source-root", type=Path)
    common("verify")
    gc = mutating("gc")
    gc.add_argument("--allow-empty-mark-set", action="store_true")
    for name in ("recover", "purge"):
        mutating(name).add_argument("--run-id")
    retire = mutating("retire-ref")
    retire.add_argument("--session-id")
    retire.add_argument("--branch")
    return parser


def _required(value: Any, option: str) -> Any:
    if value is None:
        raise CacheSafetyError(f"{option} is required")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments == ["--help"] or arguments == ["-h"]:
        print(HELP, end="")
        return 0
    parser = _build_parser()
    try:
        args = parser.parse_args(arguments)
        if args.help:
            print(HELP, end="")
            return 0
        root = _required(args.root, "--root")
        profile = PROFILES[args.profile]
        if args.command == "attest-historical-fixed-slot":
            result = attest_historical_fixed_slot_selection(
                root,
                _required(args.original_selection, "--original-selection"),
                apply=args.apply,
                confirm_root=args.confirm_root,
                engine_stopped=args.engine_stopped,
                profile=profile,
            )
        elif args.command == "import-fixed-slot":
            result = import_fixed_slot_generation(
                root,
                _required(args.source_root, "--source-root"),
                _required(args.selection, "--selection"),
                apply=args.apply,
                confirm_root=args.confirm_root,
                engine_stopped=args.engine_stopped,
                profile=profile,
            )
        elif args.command == "seal":
            result = seal_generation(
                root,
                _required(args.selection, "--selection"),
                apply=args.apply,
                confirm_root=args.confirm_root,
                engine_stopped=args.engine_stopped,
                profile=profile,
            )
        elif args.command == "verify":
            result = verify_active_generations(root, profile=profile)
        elif args.command == "gc":
            result = garbage_collect(
                root,
                apply=args.apply,
                confirm_root=args.confirm_root,
                engine_stopped=args.engine_stopped,
                allow_empty_mark_set=args.allow_empty_mark_set,
                profile=profile,
            )
        elif args.command == "recover":
            result = recover_quarantine(
                root,
                _required(args.run_id, "--run-id"),
                apply=args.apply,
                confirm_root=args.confirm_root,
                engine_stopped=args.engine_stopped,
                profile=profile,
            )
        elif args.command == "purge":
            result = purge_quarantine(
                root,
                _required(args.run_id, "--run-id"),
                apply=args.apply,
                confirm_root=args.confirm_root,
                engine_stopped=args.engine_stopped,
                profile=profile,
            )
        elif args.command == "retire-ref":
            result = retire_ref(
                root,
                _required(args.session_id, "--session-id"),
                _required(args.branch, "--branch"),
                apply=args.apply,
                confirm_root=args.confirm_root,
                engine_stopped=args.engine_stopped,
            )
        else:
            raise CacheSafetyError("a command is required")
    except (CacheSafetyError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
