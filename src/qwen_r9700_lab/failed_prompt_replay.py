"""Capture and compare exact-token replays of preserved Qwen failures."""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import os
import re
import shlex
import stat
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "urn:qwen-r9700:failed-prompt-replay:v1"
CHAT_STREAM_SCHEMA = "urn:qwen-r9700:failed-prompt-chat-stream:v1"
COMPARISON_SCHEMA = "urn:qwen-r9700:failed-prompt-replay-comparison:v1"
TRANSPORT_CONTRACT_SCHEMA = "urn:qwen-r9700:failed-prompt-transport-contract:v1"
TOKEN_MAGIC = b"QWENSTG1"
TOKEN_TRAILER_BYTES = hashlib.sha256().digest_size
TOKEN_DIGEST_DOMAIN = b"qwen-r9700-token-ids-u32be-v1\0"
MAX_PROMPT_TOKENS = 253_792
MAX_OUTPUT_TOKENS = 8_192
PRODUCTION_CHAT_MAX_OUTPUT_TOKENS = 32_768
PRODUCTION_XHIGH_THINKING_TOKEN_BUDGET = 16_384
BF16_ORACLE_MAX_MODEL_LEN = 65_536
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TRANSPORT_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
MIN_FRESH_CACHE_SALT_CHARACTERS = 32
IDENTITY_CHAT_TEMPLATE = "{{ messages[0]['content'] }}"
QWEN_TOOL_CALL_START_TOKEN_ID = 248_058
QWEN_TOOL_CALL_END_TOKEN_ID = 248_059
REPETITION_DETECTION = {
    "max_pattern_size": 8,
    "min_pattern_size": 1,
    "min_count": 7,
}


def _validate_cache_salt(value: str | None, *, fresh_state: bool) -> str | None:
    """Require an isolated cache namespace for every claimed fresh-state arm.

    Without a salt, vLLM's ordinary or external prefix cache can silently seed
    one replay while another arm cold-fills.  That changes recurrent prefill
    state and can manufacture a false optimized-versus-serial divergence.
    """

    if value is None:
        if fresh_state:
            raise ReplayError(
                "cache-salt is required for a fresh-state capture so prefix reuse "
                "cannot contaminate the oracle"
            )
        return None
    value = value.strip()
    if not value:
        raise ReplayError("cache-salt must be nonempty when supplied")
    if fresh_state and len(value) < MIN_FRESH_CACHE_SALT_CHARACTERS:
        raise ReplayError(
            "fresh-state cache-salt must contain at least "
            f"{MIN_FRESH_CACHE_SALT_CHARACTERS} characters"
        )
    return value

TARGET_ONLY_OVERRIDES = {
    "QWEN_BOUNDED_REPETITION_ESCAPE": "0",
    # Production M8 correctness overlays deliberately fail closed when the
    # target-only oracle is selected.  Disable both the feature and its
    # required-mode latch so a command derived from the current production
    # stack is a genuine serial target control rather than an invalid hybrid.
    "QWEN_DFLASH_CONTEXT_KV_SERIAL_EXACT": "0",
    "QWEN_DFLASH_CONTEXT_KV_SERIAL_EXACT_REQUIRED": "0",
    "QWEN_DFLASH_FIXED_SLOT_TRUSTED_REPLAY": "0",
    "QWEN_DFLASH_GREEDY_LOOP_ESCAPE": "0",
    "QWEN_DFLASH_GREEDY_M8_VERIFIER": "0",
    "QWEN_DFLASH_WHOLE_MODEL_ACCEPTED_REPLAY": "0",
    "QWEN_FIXED_SLOT_TARGET_ONLY_ORACLE": "1",
    "QWEN_FULL_ATTENTION_M8_EXACT_K": "0",
    "QWEN_FULL_ATTENTION_M8_EXACT_K_REQUIRED": "0",
    "QWEN_FULL_ATTENTION_M8_ROW_EXACT": "0",
    "QWEN_FULL_ATTENTION_M8_ROW_EXACT_REQUIRED": "0",
    "QWEN_GDN_FIXED_SLOT_CACHED_ACCEPTED_COMMIT": "0",
    "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED": "0",
    "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_COMMIT": "0",
    "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_VERIFY": "0",
    "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_VERIFY_CONV": "0",
    "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_VERIFY_RECURRENCE": "0",
    "QWEN_GDN_FIXED_SLOT_SERIAL_CONV_M8": "0",
    "QWEN_GDN_FIXED_SLOT_SERIAL_CONV_M8_CROSSCHECK": "0",
    "QWEN_GDN_TRANSACTION_BULK": "0",
    "QWEN_LM_HEAD_PREFIX_SERIAL_M8": "0",
}

# The selected Hauhau flat candidate currently consists of six authenticated
# post-foundation components.  A restored target-only control must not execute
# those M8-only import hooks: several deliberately reject target-only mode and
# Python's sitecustomize loader otherwise prints the exception and continues
# with the ambient installed runner.  Strip only this exact outer-to-inner
# shape, leaving the authenticated Hauhau foundation and its fixed-slot
# target-only transport intact.
FLAT_M8_COMPONENT_PYTHONPATH_PREFIX = (
    "06-lm-head",
    "05-partial-width",
    "04-fixed-slot",
    "03-parser",
    "02-context-kv",
    "01-exact-k",
)


def _without_flat_m8_component_prefix(pythonpath: str) -> str:
    parts = pythonpath.split(":")
    count = len(FLAT_M8_COMPONENT_PYTHONPATH_PREFIX)
    if len(parts) <= count:
        return pythonpath
    observed = tuple(Path(part).name for part in parts[:count])
    if observed != FLAT_M8_COMPONENT_PYTHONPATH_PREFIX:
        return pythonpath
    component_parents = {str(Path(part).parent) for part in parts[:count]}
    if len(component_parents) != 1 or Path(next(iter(component_parents))).name != "components":
        raise ReplayError("flat M8 component PYTHONPATH prefix is not one bounded chain")
    foundation = ":".join(parts[count:])
    if not foundation or not foundation.startswith("/"):
        raise ReplayError("flat M8 component chain lacks an absolute foundation")
    return foundation

FRESH_TARGET_ONLY_OVERRIDES = {
    # A fresh-state target arm has neither a DFlash runner nor a KV connector.
    # Disable every fixed-slot/DFlash lifecycle latch so the exact prompt is
    # evaluated once by the target model and cannot read or publish cache state.
    "QWEN_D7_FUSED_GREEDY_COUNTS": "0",
    "QWEN_D7_RETAINED_MICROS": "0",
    "QWEN_DFLASH2_SELECTOR_OVERRIDE": "0",
    "QWEN_DFLASH_D7_C1_DRAFT_GRAPH": "0",
    "QWEN_DFLASH_D7_C1_DRAFT_GRAPH_REQUIRED": "0",
    "QWEN_DFLASH_PERSISTENT_EAGLE_TAIL": "0",
    "QWEN_DFLASH_PERSISTENT_KV": "0",
    "QWEN_DFLASH_RAW_FILL_MODE_NONE": "0",
    "QWEN_DFLASH_ROCM_OFFLOADING_CONNECTOR": "0",
    "QWEN_DFLASH_ROCM_REJECTION_TORCH_FALLBACK": "0",
    "QWEN_FIXED_SLOT_CORRECTED_GENERATION": "0",
    "QWEN_FIXED_SLOT_GENERIC_PREFIX_BLOCK": "0",
    "QWEN_FIXED_SLOT_GENERIC_PREFIX_BLOCK_REQUIRED": "0",
    "QWEN_FIXED_SLOT_ROLLING_CHECKPOINT": "0",
    "QWEN_FIXED_SLOT_ORPHAN_FINALIZER": "0",
    "QWEN_FIXED_SLOT_PINNED_REPLAY_ONLY": "0",
    "QWEN_FIXED_SLOT_SHORT_ROOT_CHECKPOINT": "0",
    "QWEN_FIXED_SLOT_SNAPSHOT_EXPORT": "0",
    "QWEN_FIXED_SLOT_TARGET_ONLY_ORACLE": "0",
    "QWEN_GDN_ALLOW_UNMASKED_MTP": "0",
    "QWEN_GDN_BA_GROUPED_PREFIX_EXACT": "0",
    "QWEN_GDN_BA_SERIAL_ROW_EXACT": "0",
    "QWEN_GDN_METADATA_M8_FAST": "0",
    "QWEN_GDN_METADATA_M8_FAST_REQUIRED": "0",
    "QWEN_GDN_RECOVERSSM": "0",
    "QWEN_GDN_RECOVERSSM_FIXED_SLOT_TRUSTED_REPLAY": "0",
    "QWEN_GDN_RECOVERSSM_REQUIRED": "0",
    "QWEN_GDN_RECOVERSSM_TRUSTED_REPLAY": "0",
    "QWEN_HAUHAU_PARTIAL_WIDTH_RUNTIME": "0",
    "QWEN_LIVE62_GDN_D7_DISCRIMINATOR": "0",
    "QWEN_LM_HEAD_DIRECT_M8": "0",
    "QWEN_LM_HEAD_DIRECT_M8_REQUIRED": "0",
    "QWEN_LM_HEAD_M8_ROW_EXACT": "0",
    "QWEN_LM_HEAD_M8_ROW_EXACT_REQUIRED": "0",
    "QWEN_LM_HEAD_PREFIX_SERIAL_M8": "0",
    "QWEN_LM_HEAD_SERIAL_M8": "0",
    "QWEN_QUEST_M8_DUALPHASE": "0",
    "QWEN_QUEST_M8_ROW_LOCAL": "0",
    # The cached-GEMM selector is specialized to DFlash's compact 1,648-token
    # cache blocks.  A connector-free target cache uses ordinary 16-token
    # blocks, so retain the general compiled Quest96 selector instead.
    "QWEN_QUEST_CACHED_GEMM_SELECTOR": "0",
    "QWEN_W4_M8_FAST_DISPATCH": "0",
    "VLLM_DFLASH_250K_EMPTY_CACHE": "0",
    "VLLM_DFLASH_COMPACT_KV_GROUPS": "0",
}

FRESH_TARGET_FORBIDDEN_ACTIVE_MARKERS = (
    "DFLASH",
    "FIXED_SLOT",
    "_M8",
    "_D7",
    "PARTIAL_WIDTH",
    "RECOVERSSM",
)

DENSE_OVERRIDES = {
    "QWEN_CODING_TURBO_QUEST96": "0",
    "QWEN_QUEST_ATTENTION": "0",
    "QWEN_QUEST_COMPILED_ATTENTION": "0",
    "QWEN_QUEST_COMPILED_ATTENTION_REQUIRED": "0",
    "QWEN_QUEST_GROUPED_GQA": "0",
    "QWEN_QUEST_NATIVE_GQA": "0",
    "QWEN_QUEST_Q16_DIRECT": "0",
    "QWEN_QUEST_Q1_CONTEXT_DIRECT": "0",
    "QWEN_QUEST_Q1_DIRECT": "0",
    "QWEN_QUEST_REQUIRE_COMPILED": "0",
    "QWEN_QUEST_REQUIRE_DIRECT": "0",
    "QWEN_QUEST_WMMA_FP8": "0",
}

M8_SERIAL_OVERRIDES = {
    "QWEN_BOUNDED_REPETITION_ESCAPE": "0",
    "QWEN_DFLASH_GREEDY_LOOP_ESCAPE": "0",
    "QWEN_GDN_FIXED_SLOT_CACHED_ACCEPTED_COMMIT": "0",
    "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED": "0",
    "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_COMMIT": "0",
    "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_VERIFY": "0",
    "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_VERIFY_CONV": "0",
    "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_VERIFY_RECURRENCE": "0",
    "QWEN_GDN_FIXED_SLOT_SERIAL_CONV_M8": "0",
    "QWEN_GDN_FIXED_SLOT_SERIAL_CONV_M8_CROSSCHECK": "0",
    # The production bulk transaction deliberately rejects the serial M8
    # oracle.  The model runner's built-in one-token commit path is the
    # qualification oracle for this arm.
    "QWEN_GDN_TRANSACTION_BULK": "0",
}

STANDARD_M8_OVERRIDES = {
    "QWEN_BOUNDED_REPETITION_ESCAPE": "0",
    "QWEN_DFLASH_FIXED_SLOT_TRUSTED_REPLAY": "0",
    "QWEN_DFLASH_GREEDY_LOOP_ESCAPE": "0",
    "QWEN_DFLASH_WHOLE_MODEL_ACCEPTED_REPLAY": "0",
    "QWEN_FIXED_SLOT_TARGET_ONLY_ORACLE": "0",
    "QWEN_GDN_RECOVERSSM_FIXED_SLOT_TRUSTED_REPLAY": "0",
}

PRODUCTION_M8_OVERRIDES = {
    # Preserve the deployed FP8 fixed-slot, trusted replay, GDN transaction,
    # Quest96, W4A16, and D7/M8 path byte-for-byte.  Only safety-net token
    # mutation is disabled so this arm can expose the original divergence.
    "QWEN_BOUNDED_REPETITION_ESCAPE": "0",
    "QWEN_DFLASH_GREEDY_LOOP_ESCAPE": "0",
}

STANDARD_M8_SERIAL_BA_OVERRIDES = {
    # Diagnostic-only negative-control arm.  The first version of the layer
    # capture retained a view into a reusable W4A16 output workspace and hashed
    # it after a later projection had overwritten the storage.  A synchronous
    # capture plus this arm disproved B/A grouping as the ef044 token divergence:
    # serial row projection still diverges at the same token.
    "QWEN_GDN_BA_BATCHED_EXACT": "0",
    "QWEN_GDN_BA_BATCHED_CROSSCHECK": "0",
    "QWEN_GDN_BA_GROUPED_PREFIX_EXACT": "0",
    "QWEN_GDN_BA_GROUPED_PREFIX_CROSSCHECK": "0",
    "QWEN_GDN_BA_SERIAL_ROW_EXACT": "1",
}

STANDARD_M8_GDN_CROSSCHECK_OVERRIDES = {
    # Qualification-only bitwise oracles.  They execute the production M8
    # convolution/recurrence and the corresponding ordered serial-M1 reference
    # from independent scratch at the exact failed-prompt boundary.
    "QWEN_GDN_FIXED_SLOT_SERIAL_CONV_M8_CROSSCHECK": "1",
    "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_CROSSCHECK": "1",
}

M8_LM_HEAD_OVERRIDES = {
    "production": {},
    "direct": {
        "QWEN_LM_HEAD_W4_TOPK": "0",
        "QWEN_LM_HEAD_W4_TOPK_REQUIRED": "0",
    },
    "grouped": {
        "QWEN_LM_HEAD_DIRECT_M8": "0",
        "QWEN_LM_HEAD_DIRECT_M8_REQUIRED": "0",
        "QWEN_LM_HEAD_W4_TOPK": "0",
        "QWEN_LM_HEAD_W4_TOPK_REQUIRED": "0",
    },
    "exact-m1": {},
}


class ReplayError(RuntimeError):
    """A replay input, response, or comparison violated its contract."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _token_digest(tokens: Sequence[int]) -> str:
    digest = hashlib.sha256(TOKEN_DIGEST_DOMAIN)
    for token in tokens:
        digest.update(struct.pack(">I", token))
    return digest.hexdigest()


def read_qwenstg1(path: Path) -> tuple[list[int], bytes]:
    """Read one stable, private QWENSTG1 vector and verify both digests."""

    path = path.expanduser().absolute()
    try:
        before = path.lstat()
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise ReplayError(f"cannot read token vector: {error}") from error
    stable = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if (
        not stable
        or not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_nlink != 1
    ):
        raise ReplayError("token vector must be one stable owned 0600 regular file")
    minimum = len(TOKEN_MAGIC) + 8 + TOKEN_TRAILER_BYTES
    if len(payload) < minimum or payload[:8] != TOKEN_MAGIC:
        raise ReplayError("token vector is not QWENSTG1")
    count = struct.unpack(">Q", payload[8:16])[0]
    if not 1 <= count <= MAX_PROMPT_TOKENS:
        raise ReplayError("token-vector count is out of bounds")
    expected_size = 16 + count * 4 + TOKEN_TRAILER_BYTES
    if len(payload) != expected_size:
        raise ReplayError("token-vector length does not match its header")
    body, trailer = payload[:-TOKEN_TRAILER_BYTES], payload[-TOKEN_TRAILER_BYTES:]
    if hashlib.sha256(body).digest() != trailer:
        raise ReplayError("token-vector trailer digest is invalid")
    tokens = list(struct.unpack(f">{count}I", payload[16:-TOKEN_TRAILER_BYTES]))
    if any(token >= 253_952 for token in tokens):
        raise ReplayError("token vector contains an out-of-vocabulary token")
    return tokens, payload


def _stable_private_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    path = path.expanduser().absolute()
    try:
        before = path.lstat()
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise ReplayError(f"cannot read {label}: {error}") from error
    stable = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if (
        not stable
        or not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_nlink != 1
    ):
        raise ReplayError(f"{label} must be one stable owned 0600 regular file")
    if not 1 <= len(payload) <= 1024 * 1024:
        raise ReplayError(f"{label} size is out of bounds")
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ReplayError(f"{label} is invalid JSON: {error}") from error
    if not isinstance(document, dict):
        raise ReplayError(f"{label} must contain one JSON object")
    return document, payload


def _load_transport_contract(
    path: Path,
    expected_sha256: str,
    *,
    prompt_token_count: int,
    prompt_token_ids_sha256: str,
    prompt_tokens: list[int],
    token_file_sha256: str,
) -> dict[str, Any]:
    """Authenticate one fixed-slot transport contract against its durable head."""

    if SHA256_RE.fullmatch(expected_sha256) is None:
        raise ReplayError("transport-contract-sha256 must be 64 lowercase hexadecimal characters")
    contract, payload = _stable_private_json(path, "transport contract")
    observed_sha256 = _sha256(payload)
    if observed_sha256 != expected_sha256:
        raise ReplayError("transport contract SHA-256 differs from the pinned value")
    if set(contract) != {"kv_transfer_params", "schema", "source_head", "state_contract"}:
        raise ReplayError("transport contract keys differ from the exact schema")
    if contract.get("schema") != TRANSPORT_CONTRACT_SCHEMA:
        raise ReplayError("transport contract schema is unsupported")
    state_contract = contract.get("state_contract")
    if state_contract not in {
        "fixed-slot-resume-authenticated-prefix-v1",
        "fixed-slot-resume-exact-token-vector-v1",
    }:
        raise ReplayError("transport contract state mode is unsupported")

    transfer = contract.get("kv_transfer_params")
    if not isinstance(transfer, dict) or set(transfer) != {"qwen_250k_cache"}:
        raise ReplayError("transport contract lacks one exact qwen_250k_cache object")
    cache = transfer.get("qwen_250k_cache")
    if not isinstance(cache, dict) or set(cache) != {
        "branch",
        "checkpoint_tokens",
        "manifest_sha256",
        "mode",
        "session_id",
    }:
        raise ReplayError("qwen_250k_cache keys differ from the exact resume contract")
    if cache.get("mode") != "resume":
        raise ReplayError("transport contract must use fixed-slot resume mode")
    if cache.get("checkpoint_tokens") is not None:
        raise ReplayError("full fixed-slot resume requires null checkpoint_tokens")
    for field in ("branch", "session_id"):
        value = cache.get(field)
        if not isinstance(value, str) or TRANSPORT_IDENTIFIER.fullmatch(value) is None:
            raise ReplayError(f"qwen_250k_cache.{field} is invalid")
    manifest_sha256 = cache.get("manifest_sha256")
    if not isinstance(manifest_sha256, str) or SHA256_RE.fullmatch(manifest_sha256) is None:
        raise ReplayError("qwen_250k_cache.manifest_sha256 is invalid")

    source = contract.get("source_head")
    expected_source_keys = {
        "generation",
        "head_prefix_token_ids_sha256",
        "path",
        "prompt_token_count",
        "prompt_token_file_sha256",
        "prompt_token_ids_sha256",
        "sha256",
    }
    if state_contract == "fixed-slot-resume-authenticated-prefix-v1":
        expected_source_keys.add("prompt_token_file_path")
    if not isinstance(source, dict) or set(source) != expected_source_keys:
        raise ReplayError("transport contract source_head keys differ from the exact schema")
    head_path_text = source.get("path")
    if not isinstance(head_path_text, str) or not Path(head_path_text).is_absolute():
        raise ReplayError("transport contract source_head.path must be absolute")
    head_sha256 = source.get("sha256")
    if not isinstance(head_sha256, str) or SHA256_RE.fullmatch(head_sha256) is None:
        raise ReplayError("transport contract source_head.sha256 is invalid")
    head, head_payload = _stable_private_json(Path(head_path_text), "transport source head")
    if _sha256(head_payload) != head_sha256:
        raise ReplayError("transport source head SHA-256 differs from the contract")
    if (
        head.get("schema") != "urn:qwen-r9700:authenticated-snapshot-head:v1"
        or head.get("generation") != source.get("generation")
        or head.get("session_id") != cache["session_id"]
        or head.get("branch") != cache["branch"]
        or head.get("manifest_sha256") != manifest_sha256
        or head.get("prompt_tokens") != source.get("prompt_token_count")
        or head.get("prefix_token_ids_sha256") != source.get("head_prefix_token_ids_sha256")
        or head.get("settled_token_file_sha256") != source.get("prompt_token_file_sha256")
    ):
        raise ReplayError("transport source head differs from the bound resume contract")
    if state_contract == "fixed-slot-resume-exact-token-vector-v1":
        if (
            source.get("prompt_token_count") != prompt_token_count
            or source.get("prompt_token_file_sha256") != token_file_sha256
            or source.get("prompt_token_ids_sha256") != prompt_token_ids_sha256
        ):
            raise ReplayError("transport contract is bound to a different exact prompt vector")
    else:
        source_token_path_text = source.get("prompt_token_file_path")
        if not isinstance(source_token_path_text, str) or not Path(
            source_token_path_text
        ).is_absolute():
            raise ReplayError("transport source prompt_token_file_path must be absolute")
        source_tokens, source_token_payload = read_qwenstg1(Path(source_token_path_text))
        source_count = source.get("prompt_token_count")
        if (
            not isinstance(source_count, int)
            or source_count <= 0
            or source_count > prompt_token_count
            or len(source_tokens) != source_count
            or _sha256(source_token_payload) != source.get("prompt_token_file_sha256")
            or _token_digest(source_tokens) != source.get("prompt_token_ids_sha256")
        ):
            raise ReplayError("transport source prefix vector authentication failed")
        if prompt_tokens[:source_count] != source_tokens:
            raise ReplayError("transport source head is not an exact prefix of the request")
    return {
        "contract_file": str(path.expanduser().absolute()),
        "contract_file_sha256": observed_sha256,
        "kv_transfer_params": transfer,
        "source_head_file": head_path_text,
        "source_head_file_sha256": head_sha256,
        "state_contract": state_contract,
        "transport_body_sha256": _sha256(_canonical(transfer)),
    }


def _endpoint_path(base_url: str, path: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ReplayError("base URL must be absolute HTTP(S)")
    if parsed.query or parsed.fragment:
        raise ReplayError("base URL must not contain a query or fragment")
    if not path.startswith("/"):
        raise ReplayError("endpoint path must be absolute")
    return f"{base_url.rstrip('/')}{path}"


def _endpoint(base_url: str) -> str:
    return _endpoint_path(base_url, "/v1/completions")


def _request_headers(api_key_env: str, request_id: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {os.environ.get(api_key_env, 'EMPTY')}",
        "Content-Type": "application/json",
        "X-Request-Id": request_id,
    }


def _post_json(
    *,
    api_key_env: str,
    base_url: str,
    path: str,
    payload: Mapping[str, Any],
    request_id: str,
    timeout: float,
) -> tuple[int, dict[str, str], bytes, dict[str, Any]]:
    body = _canonical(payload)
    request = urllib.request.Request(
        _endpoint_path(base_url, path),
        data=body,
        method="POST",
        headers=_request_headers(api_key_env, request_id),
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            headers = dict(response.headers.items())
            response_body = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ReplayError(f"{path} returned HTTP {error.code}: {detail[:1000]}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise ReplayError(f"{path} request failed: {error}") from error
    try:
        response_json = json.loads(response_body)
    except json.JSONDecodeError as error:
        raise ReplayError(f"{path} returned invalid JSON: {error}") from error
    if not isinstance(response_json, dict):
        raise ReplayError(f"{path} did not return a JSON object")
    return status, headers, response_body, response_json


def _extract_tools(prompt_text: str) -> list[dict[str, Any]]:
    """Recover the exact tool schemas already embedded in a Pi provider prompt."""

    opening = "<tools>"
    closing = "</tools>"
    start = prompt_text.find(opening)
    if start < 0:
        raise ReplayError("decoded prompt lacks a <tools> block")
    end = prompt_text.find(closing, start + len(opening))
    if end < 0:
        raise ReplayError("decoded prompt has an unterminated <tools> block")
    region = prompt_text[start + len(opening) : end]
    decoder = json.JSONDecoder()
    cursor = 0
    tools: list[dict[str, Any]] = []
    while cursor < len(region):
        while cursor < len(region) and region[cursor].isspace():
            cursor += 1
        if cursor == len(region):
            break
        try:
            value, cursor = decoder.raw_decode(region, cursor)
        except json.JSONDecodeError as error:
            raise ReplayError(f"decoded prompt has an invalid tool schema: {error}") from error
        if not isinstance(value, dict):
            raise ReplayError("decoded prompt tool schema is not an object")
        function = value.get("function")
        if (
            value.get("type") != "function"
            or not isinstance(function, dict)
            or not isinstance(function.get("name"), str)
            or not function["name"]
            or not isinstance(function.get("parameters"), dict)
        ):
            raise ReplayError("decoded prompt contains a malformed function tool")
        tools.append(value)
    if not tools:
        raise ReplayError("decoded prompt has an empty <tools> block")
    names = [tool["function"]["name"] for tool in tools]
    if len(names) != len(set(names)):
        raise ReplayError("decoded prompt contains duplicate tool names")
    return tools


def _strict_outcome_schema(value: Any, *, path: str = "schema") -> Any:
    """Mirror Pi's JSON-outcome schema normalization exactly."""

    if isinstance(value, bool):
        return value
    if not isinstance(value, dict):
        raise ReplayError(f"{path} is not a JSON-schema object")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key == "additionalProperties":
            continue
        if key in {"properties", "$defs", "definitions"}:
            if not isinstance(item, dict):
                raise ReplayError(f"{path}.{key} is not an object")
            result[key] = {
                name: _strict_outcome_schema(child, path=f"{path}.{key}.{name}")
                for name, child in item.items()
            }
        elif key in {"items", "contains", "not", "if", "then", "else"}:
            result[key] = _strict_outcome_schema(item, path=f"{path}.{key}")
        elif key in {"allOf", "anyOf", "oneOf", "prefixItems"}:
            if not isinstance(item, list):
                raise ReplayError(f"{path}.{key} is not an array")
            result[key] = [
                _strict_outcome_schema(child, path=f"{path}.{key}[{index}]")
                for index, child in enumerate(item)
            ]
        else:
            result[key] = item
    if value.get("type") == "object" or isinstance(value.get("properties"), dict):
        additional = value.get("additionalProperties")
        if additional is not None and additional is not False:
            raise ReplayError(f"{path} permits unbounded additional object properties")
        result["additionalProperties"] = False
    return result


def _json_outcome_response_format(tools: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the byte-semantic equivalent of qwen-json-outcome-router.mjs."""

    branches: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, tool in enumerate(tools):
        function = tool.get("function")
        if tool.get("type") != "function" or not isinstance(function, dict):
            raise ReplayError(f"tools[{index}] is not a function tool")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ReplayError(f"tools[{index}] is not a named function tool")
        if name in names:
            raise ReplayError(f"tool name {name} is duplicated")
        names.add(name)
        parameters = _strict_outcome_schema(
            function.get("parameters", {"type": "object", "properties": {}}),
            path=f"tools[{index}].parameters",
        )
        branches.append(
            {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "const": "tool"},
                    "name": {"type": "string", "const": name},
                    "arguments": parameters,
                },
                "required": ["kind", "name", "arguments"],
                "additionalProperties": False,
            }
        )
    branches.append(
        {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "const": "final"},
                "answer": {"type": "string", "minLength": 1},
            },
            "required": ["kind", "answer"],
            "additionalProperties": False,
        }
    )
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "qwen_pi_outcome",
            "strict": True,
            "schema": {"oneOf": branches},
        },
    }


def _parse_sse(body: bytes) -> tuple[list[dict[str, Any]], int, list[str]]:
    """Parse an SSE response while leaving the byte-for-byte body untouched."""

    events: list[dict[str, Any]] = []
    done_markers = 0
    seen_done = False
    errors: list[str] = []
    blocks = re.split(rb"\r?\n\r?\n", body)
    for block_index, block in enumerate(blocks):
        if not block:
            continue
        data_lines: list[bytes] = []
        for line in block.splitlines():
            if line.startswith(b"data:"):
                value = line[5:]
                if value.startswith(b" "):
                    value = value[1:]
                data_lines.append(value)
            elif line and not line.startswith(b":"):
                errors.append(f"event {block_index} contains an unsupported SSE field")
        if not data_lines:
            continue
        data = b"\n".join(data_lines)
        if data == b"[DONE]":
            done_markers += 1
            seen_done = True
            continue
        if seen_done:
            errors.append(f"event {block_index} appeared after [DONE]")
        try:
            value = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            errors.append(f"event {block_index} is invalid JSON: {error}")
            continue
        if not isinstance(value, dict):
            errors.append(f"event {block_index} is not a JSON object")
            continue
        events.append(value)
    if done_markers != 1:
        errors.append(f"expected one [DONE] marker, observed {done_markers}")
    return events, done_markers, errors


def _append_optional_text(parts: list[str], value: Any, field: str, errors: list[str]) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        errors.append(f"delta {field} is not a string")
        return
    parts.append(value)


def _aggregate_chat_stream(
    events: Sequence[Mapping[str, Any]], expected_prompt_tokens: Sequence[int]
) -> dict[str, Any]:
    completion_ids: list[int] = []
    prompt_ids: list[int] | None = None
    prompt_text: str | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_deltas: list[dict[str, Any]] = []
    tools_by_index: dict[int, dict[str, Any]] = {}
    finish_reasons: list[Any] = []
    stop_reasons: list[Any] = []
    usages: list[dict[str, Any]] = []
    response_ids: set[str] = set()
    errors: list[str] = []

    for event_index, event in enumerate(events):
        response_id = event.get("id")
        if isinstance(response_id, str):
            response_ids.add(response_id)
        event_prompt_ids = event.get("prompt_token_ids")
        if event_prompt_ids is not None:
            if not isinstance(event_prompt_ids, list) or not all(
                type(token) is int for token in event_prompt_ids
            ):
                errors.append(f"event {event_index} has invalid prompt_token_ids")
            elif prompt_ids is not None:
                errors.append("prompt_token_ids appeared in more than one SSE event")
            else:
                prompt_ids = event_prompt_ids
        event_prompt_text = event.get("prompt_text")
        if event_prompt_text is not None:
            if not isinstance(event_prompt_text, str):
                errors.append(f"event {event_index} has non-string prompt_text")
            elif prompt_text is not None:
                errors.append("prompt_text appeared in more than one SSE event")
            else:
                prompt_text = event_prompt_text
        usage = event.get("usage")
        if usage is not None:
            if isinstance(usage, dict):
                usages.append(usage)
            else:
                errors.append(f"event {event_index} has invalid usage")
        choices = event.get("choices")
        if not isinstance(choices, list):
            errors.append(f"event {event_index} lacks a choices list")
            continue
        if not choices:
            continue
        if len(choices) != 1 or not isinstance(choices[0], dict):
            errors.append(f"event {event_index} does not contain exactly one choice")
            continue
        choice = choices[0]
        if choice.get("index") not in (None, 0):
            errors.append(f"event {event_index} has a nonzero choice index")
        token_ids = choice.get("token_ids")
        if token_ids is not None:
            if not isinstance(token_ids, list) or not all(
                type(token) is int for token in token_ids
            ):
                errors.append(f"event {event_index} has invalid completion token IDs")
            else:
                completion_ids.extend(token_ids)
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            finish_reasons.append(finish_reason)
            stop_reasons.append(choice.get("stop_reason"))
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            errors.append(f"event {event_index} has an invalid delta")
            continue
        _append_optional_text(content_parts, delta.get("content"), "content", errors)
        # Match Pi's provider adapter: compatible servers use any of these names,
        # and a few mirror the same fragment into more than one field.  Preserve
        # exactly one nonempty reasoning fragment per event rather than silently
        # losing vLLM's usual ``reasoning_content`` stream or double-counting it.
        reasoning_fields = ("reasoning_content", "reasoning", "reasoning_text")
        populated_reasoning = [
            (field, delta.get(field))
            for field in reasoning_fields
            if delta.get(field) not in (None, "")
        ]
        for field, value in populated_reasoning:
            if not isinstance(value, str):
                errors.append(f"delta {field} is not a string")
        string_reasoning = [
            value for _field, value in populated_reasoning if isinstance(value, str)
        ]
        if string_reasoning:
            reasoning_parts.append(string_reasoning[0])
            if any(value != string_reasoning[0] for value in string_reasoning[1:]):
                errors.append("delta contains conflicting reasoning fields")
        calls = delta.get("tool_calls")
        if calls is None:
            continue
        if not isinstance(calls, list):
            errors.append(f"event {event_index} has invalid tool_calls")
            continue
        for call in calls:
            if not isinstance(call, dict) or type(call.get("index")) is not int:
                errors.append(f"event {event_index} has a malformed tool-call delta")
                continue
            tool_deltas.append(call)
            index = call["index"]
            aggregate = tools_by_index.setdefault(
                index, {"arguments": "", "id": "", "index": index, "name": "", "type": ""}
            )
            for field in ("id", "type"):
                value = call.get(field)
                if value is not None:
                    if isinstance(value, str):
                        aggregate[field] += value
                    else:
                        errors.append(f"tool-call delta {field} is not a string")
            function = call.get("function")
            if function is not None:
                if not isinstance(function, dict):
                    errors.append("tool-call delta function is not an object")
                    continue
                for source, target in (("name", "name"), ("arguments", "arguments")):
                    value = function.get(source)
                    if value is not None:
                        if isinstance(value, str):
                            aggregate[target] += value
                        else:
                            errors.append(f"tool-call delta function.{source} is not a string")

    if prompt_ids is None:
        errors.append("stream did not return prompt_token_ids")
    elif list(expected_prompt_tokens) != prompt_ids:
        errors.append("stream prompt_token_ids differ from the exact source vector")
    if len(finish_reasons) != 1:
        errors.append(f"expected one finish reason, observed {len(finish_reasons)}")
    if len(response_ids) != 1:
        errors.append(f"expected one response ID, observed {len(response_ids)}")
    if usages:
        usage = usages[-1]
        if usage.get("prompt_tokens") != len(expected_prompt_tokens):
            errors.append("usage prompt_tokens differs from the exact source vector")
        if usage.get("completion_tokens") != len(completion_ids):
            errors.append("usage completion_tokens differs from streamed completion token IDs")
    else:
        usage = None
        errors.append("stream did not return usage")

    parsed_tools = [tools_by_index[index] for index in sorted(tools_by_index)]
    for tool in parsed_tools:
        try:
            json.loads(tool["arguments"])
        except json.JSONDecodeError:
            tool["arguments_valid_json"] = False
        else:
            tool["arguments_valid_json"] = True
    return {
        "completion_token_ids": completion_ids,
        "content": "".join(content_parts),
        "errors": errors,
        "finish_reason": finish_reasons[0] if len(finish_reasons) == 1 else None,
        "prompt_text": prompt_text,
        "reasoning": "".join(reasoning_parts),
        "response_id": next(iter(response_ids)) if len(response_ids) == 1 else None,
        "stop_reason": stop_reasons[0] if len(stop_reasons) == 1 else None,
        "tool_call_deltas": tool_deltas,
        "tool_calls": parsed_tools,
        "usage": usage,
    }


def _raw_tool_delimiters(
    completion_ids: Sequence[int],
) -> tuple[list[int], list[int], bool]:
    starts: list[int] = []
    ends: list[int] = []
    active = False
    valid = True
    for index, token in enumerate(completion_ids):
        if token == QWEN_TOOL_CALL_START_TOKEN_ID:
            starts.append(index)
            if active:
                valid = False
            active = True
        elif token == QWEN_TOOL_CALL_END_TOKEN_ID:
            ends.append(index)
            if not active:
                valid = False
            active = False
    return starts, ends, valid and not active


def _classify_chat_outcome(
    completion_ids: Sequence[int],
    aggregate: Mapping[str, Any],
    allowed_tool_names: Sequence[str],
) -> str:
    starts, ends, delimiters_complete = _raw_tool_delimiters(completion_ids)
    parsed_tools = aggregate.get("tool_calls")
    if aggregate.get("errors"):
        return "invalid_stream_contract"
    if isinstance(parsed_tools, list) and parsed_tools:
        parsed_valid = all(
            tool.get("index") == index
            and isinstance(tool.get("id"), str)
            and bool(tool["id"])
            and tool.get("type") == "function"
            and isinstance(tool.get("name"), str)
            and tool["name"] in allowed_tool_names
            and tool.get("arguments_valid_json") is True
            and isinstance(tool.get("arguments"), str)
            and isinstance(json.loads(tool["arguments"]), dict)
            for index, tool in enumerate(parsed_tools)
        )
        if not delimiters_complete or len(starts) != len(ends):
            return "parsed_incomplete_tool_call"
        if (
            not parsed_valid
            or len(starts) != len(parsed_tools)
            or aggregate.get("finish_reason") != "tool_calls"
        ):
            return "parsed_invalid_tool_call"
        return "parsed_structured_tool_call"
    if not delimiters_complete or len(starts) != len(ends):
        return "parser_withheld_incomplete_tool_call"
    if starts:
        return "parser_withheld_complete_tool_call"
    if aggregate.get("finish_reason") == "stop":
        return "model_stop_without_structured_tool_call"
    return "no_structured_tool_call"


def _periodic_suffix(tokens: Sequence[int], max_pattern: int = 64) -> dict[str, Any] | None:
    """Return the strongest exact periodic suffix, if at least four copies exist."""

    best: tuple[int, int] | None = None
    for pattern_size in range(1, min(max_pattern, len(tokens) // 4) + 1):
        pattern = list(tokens[-pattern_size:])
        copies = 1
        cursor = len(tokens) - 2 * pattern_size
        while cursor >= 0 and list(tokens[cursor : cursor + pattern_size]) == pattern:
            copies += 1
            cursor -= pattern_size
        if copies >= 4 and (best is None or copies * pattern_size > best[0] * best[1]):
            best = pattern_size, copies
    if best is None:
        return None
    pattern_size, copies = best
    return {
        "copies": copies,
        "pattern_size": pattern_size,
        "pattern_token_ids": list(tokens[-pattern_size:]),
        "repeated_suffix_tokens": copies * pattern_size,
    }


def _publish_create_only(path: Path, document: Mapping[str, Any]) -> None:
    path = path.expanduser().absolute()
    parent = path.parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if parent.is_symlink() or stat.S_IMODE(parent.stat().st_mode) & 0o077:
        raise ReplayError("output parent must be a real private directory")
    payload = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ReplayError(f"output is create-only: {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _durable_stream_response(
    response: Any,
    path: Path,
    *,
    chunk_bytes: int = 16 * 1024,
) -> bytes:
    """Journal an HTTP response incrementally and retain it even after failure.

    The failed-prompt capture is itself diagnostic evidence.  A later
    detokenization failure, backend crash, timeout, or operator interruption must
    not erase the raw SSE bytes that already crossed the wire.
    """

    path = path.expanduser().absolute()
    parent = path.parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if parent.is_symlink() or stat.S_IMODE(parent.stat().st_mode) & 0o077:
        raise ReplayError("wire-evidence parent must be a real private directory")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ReplayError(f"wire evidence is create-only: {path}") from error
    directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

    payload = bytearray()
    try:
        with os.fdopen(descriptor, "wb") as handle:
            read_chunk = getattr(response, "read1", response.read)
            while True:
                chunk = read_chunk(chunk_bytes)
                if not chunk:
                    break
                payload.extend(chunk)
                handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
    except BaseException:
        # Deliberately do not unlink: the partial wire stream is the most useful
        # artifact after a transport or backend failure.
        raise
    return bytes(payload)


def _derive_oracle_command(
    payload: bytes,
    *,
    dense: bool,
    bf16_kv: bool,
    m8_serial: bool,
    standard_m8: bool = False,
    production_m8: bool = False,
    standard_m8_serial_ba: bool = False,
    m8_lm_head: str = "production",
    exact_m1_root: Path | None = None,
    exact_m1_binder_sha256: str | None = None,
    exact_m1_site_sha256: str | None = None,
    qualification_cache_root: Path | None = None,
    fresh_target: bool = False,
    max_num_batched_tokens: int | None = None,
    quest_rowwise_q1_control: bool = False,
) -> bytes:
    """Derive one direct oracle command from a pinned production command."""

    try:
        text = payload.decode("utf-8")
        arguments = shlex.split(text)
    except (UnicodeDecodeError, ValueError) as error:
        raise ReplayError(f"base command is invalid: {error}") from error
    if len(text.splitlines()) != 1 or arguments[:3] != ["exec", "/usr/bin/env", "-i"]:
        raise ReplayError("base command must be one exec /usr/bin/env -i line")
    environment_end = 3
    environment: dict[str, str] = {}
    environment_order: list[str] = []
    while environment_end < len(arguments):
        argument = arguments[environment_end]
        if "=" not in argument:
            break
        name, value = argument.split("=", 1)
        if ENVIRONMENT_NAME.fullmatch(name) is None:
            break
        if name in environment:
            raise ReplayError(f"base command duplicates environment variable {name}")
        environment[name] = value
        environment_order.append(name)
        environment_end += 1
    if m8_lm_head not in M8_LM_HEAD_OVERRIDES:
        raise ReplayError(f"unknown M8 LM-head arm: {m8_lm_head}")
    selected_m8_arms = sum((m8_serial, standard_m8, production_m8))
    if selected_m8_arms > 1:
        raise ReplayError("M8-serial, standard-M8, and production-M8 arms are mutually exclusive")
    if standard_m8_serial_ba and not standard_m8:
        raise ReplayError("serial-row B/A is valid only for the standard-M8 arm")
    if fresh_target and (selected_m8_arms or bf16_kv or qualification_cache_root is not None):
        raise ReplayError(
            "fresh target is mutually exclusive with M8, BF16-KV, and qualification-cache arms"
        )
    if not m8_serial and m8_lm_head != "production":
        raise ReplayError("M8 LM-head arms require the M8-serial oracle")
    if quest_rowwise_q1_control and not selected_m8_arms:
        raise ReplayError("Quest rowwise-Q1 control requires an M8 arm")
    overrides = dict(
        M8_SERIAL_OVERRIDES
        if m8_serial
        else STANDARD_M8_OVERRIDES
        if standard_m8
        else PRODUCTION_M8_OVERRIDES
        if production_m8
        else TARGET_ONLY_OVERRIDES
    )
    if m8_serial:
        overrides.update(M8_LM_HEAD_OVERRIDES[m8_lm_head])
    if standard_m8_serial_ba:
        overrides.update(STANDARD_M8_SERIAL_BA_OVERRIDES)
    if m8_lm_head == "exact-m1":
        if (
            exact_m1_root is None
            or not exact_m1_root.is_absolute()
            or SHA256_RE.fullmatch(exact_m1_binder_sha256 or "") is None
            or SHA256_RE.fullmatch(exact_m1_site_sha256 or "") is None
        ):
            raise ReplayError(
                "exact-M1 arm requires an absolute root and pinned binder/site hashes"
            )
        chained_sha = environment.get("QWEN_LIVE62_HAUHAU_DELTA_AGGRESSIVE_SITE_SHA256", "")
        runner_sha = environment.get("QWEN_LM_HEAD_DIRECT_M8_RUNNER_SHA256", "")
        qwen_sha = environment.get("QWEN_LM_HEAD_DIRECT_M8_QWEN35_SHA256", "")
        pythonpath = environment.get("PYTHONPATH", "")
        if (
            SHA256_RE.fullmatch(chained_sha) is None
            or SHA256_RE.fullmatch(runner_sha) is None
            or SHA256_RE.fullmatch(qwen_sha) is None
            or not pythonpath.startswith("/")
        ):
            raise ReplayError("base command lacks pinned exact-M1 chain identities")
        chained_root = Path(pythonpath.split(":", 1)[0])
        dynamic = {
            "PYTHONPATH": f"{exact_m1_root}:{pythonpath}",
            "QWEN_LM_HEAD_EXACT_M1_ORACLE": "1",
            "QWEN_LM_HEAD_EXACT_M1_ORACLE_REQUIRED": "1",
            "QWEN_LM_HEAD_EXACT_M1_ORACLE_BINDER_SHA256": exact_m1_binder_sha256 or "",
            "QWEN_LM_HEAD_EXACT_M1_ORACLE_SITE_SHA256": exact_m1_site_sha256 or "",
            "QWEN_LM_HEAD_EXACT_M1_ORACLE_RUNNER_SHA256": runner_sha,
            "QWEN_LM_HEAD_EXACT_M1_ORACLE_QWEN35_SHA256": qwen_sha,
            "QWEN_LM_HEAD_EXACT_M1_ORACLE_CHAINED_SITE": str(chained_root / "sitecustomize.py"),
            "QWEN_LM_HEAD_EXACT_M1_ORACLE_CHAINED_SITE_SHA256": chained_sha,
        }
        for name, value in dynamic.items():
            if name not in environment:
                environment_order.append(name)
            environment[name] = value
    if fresh_target:
        overrides.update(FRESH_TARGET_ONLY_OVERRIDES)
        pythonpath = environment.get("PYTHONPATH")
        if pythonpath is None:
            raise ReplayError("fresh target base command lacks PYTHONPATH")
        pythonpath_parts = pythonpath.split(":")
        if len(pythonpath_parts) < 4 or any(
            not part.startswith("/") for part in pythonpath_parts[-4:]
        ):
            raise ReplayError("fresh target PYTHONPATH lacks the four absolute base components")
        # The production prefix is a chain of M8-only import-time overlays that
        # deliberately rejects a target-only process.  The final four entries
        # are the authenticated release attention module, Quest sources,
        # installed-runtime support, and repository support package.  Retain
        # that base suffix while excluding only the M8 candidate wrappers.
        environment["PYTHONPATH"] = ":".join(pythonpath_parts[-4:])
    elif not selected_m8_arms:
        pythonpath = environment.get("PYTHONPATH")
        if pythonpath is not None:
            environment["PYTHONPATH"] = _without_flat_m8_component_prefix(pythonpath)
    if dense:
        overrides.update(DENSE_OVERRIDES)
    for name, value in overrides.items():
        if name not in environment:
            environment_order.append(name)
        environment[name] = value
    if quest_rowwise_q1_control:
        name = "QWEN_QUEST_M8_ROWWISE_Q1_CONTROL"
        if name not in environment:
            environment_order.append(name)
        environment[name] = "1"
    if fresh_target:
        active_candidate_latches = sorted(
            name
            for name, value in environment.items()
            if value == "1"
            and any(marker in name for marker in FRESH_TARGET_FORBIDDEN_ACTIVE_MARKERS)
        )
        if active_candidate_latches:
            raise ReplayError(
                "fresh target retains active candidate-only latches: "
                + ", ".join(active_candidate_latches)
            )
    executable_arguments = arguments[environment_end:]
    if max_num_batched_tokens is not None:
        if max_num_batched_tokens < 1:
            raise ReplayError("maximum batched-token override must be positive")
        option = "--max-num-batched-tokens"
        option_count = executable_arguments.count(option)
        if option_count != 1:
            raise ReplayError(
                "base command must contain exactly one --max-num-batched-tokens option"
            )
        option_index = executable_arguments.index(option)
        if option_index + 1 >= len(executable_arguments):
            raise ReplayError("base command has no maximum batched-token value")
        executable_arguments[option_index + 1] = str(max_num_batched_tokens)
    try:
        kv_option = executable_arguments.index("--kv-transfer-config")
    except ValueError as error:
        raise ReplayError("base command lacks --kv-transfer-config") from error
    if kv_option + 1 >= len(executable_arguments):
        raise ReplayError("base command has no KV-transfer configuration value")
    try:
        kv_transfer = json.loads(executable_arguments[kv_option + 1])
    except json.JSONDecodeError as error:
        raise ReplayError(f"KV-transfer configuration is invalid: {error}") from error
    extra = kv_transfer.get("kv_connector_extra_config")
    if not isinstance(extra, dict):
        raise ReplayError("KV-transfer configuration lacks extra configuration")
    if m8_serial:
        extra.update(
            {
                "fixed_slot_m8_serial_oracle": True,
                "fixed_slot_target_only_oracle": False,
                "fixed_slot_trusted_replay": True,
                "fixed_slot_whole_model_replay": True,
            }
        )
    elif standard_m8:
        extra.update(
            {
                "fixed_slot_m8_serial_oracle": False,
                "fixed_slot_target_only_oracle": False,
                "fixed_slot_trusted_replay": False,
                "fixed_slot_whole_model_replay": False,
            }
        )
    elif production_m8:
        required_production_extra = {
            "fixed_slot_m8_serial_oracle": False,
            "fixed_slot_target_only_oracle": False,
            "fixed_slot_trusted_replay": True,
            "fixed_slot_whole_model_replay": True,
        }
        observed_production_extra = {
            name: extra.get(name) for name in required_production_extra
        }
        if observed_production_extra != required_production_extra:
            raise ReplayError(
                "base command lacks the exact production M8 fixed-slot contract"
            )
    else:
        extra.update(
            {
                "fixed_slot_m8_serial_oracle": False,
                "fixed_slot_target_only_oracle": True,
                "fixed_slot_trusted_replay": False,
                "fixed_slot_whole_model_replay": False,
            }
        )
    if qualification_cache_root is not None:
        if bf16_kv:
            raise ReplayError(
                "a qualification cache root is not valid for the connector-free BF16 arm"
            )
        root = qualification_cache_root.expanduser()
        if not root.is_absolute() or root == Path("/") or ".." in root.parts:
            raise ReplayError("qualification cache root must be a bounded absolute path")
        snapshot_root = environment.get("QWEN_FIXED_SLOT_SNAPSHOT_ROOT")
        if not snapshot_root or not snapshot_root.startswith("/"):
            raise ReplayError("base command lacks an absolute fixed-slot snapshot root")
        tiers = extra.get("secondary_tiers")
        if not isinstance(tiers, list) or len(tiers) != 1 or not isinstance(tiers[0], dict):
            raise ReplayError("KV-transfer configuration has an unexpected tier layout")
        tier_root = tiers[0].get("root_dir")
        if not isinstance(tier_root, str) or not tier_root.startswith("/"):
            raise ReplayError("KV-transfer filesystem tier lacks an absolute root")
        isolated_snapshot_root = root / "fixed-slot"
        isolated_tier_root = root / "offload"
        if str(isolated_snapshot_root) == snapshot_root or str(isolated_tier_root) == tier_root:
            raise ReplayError("qualification cache root aliases a production cache namespace")
        environment["QWEN_FIXED_SLOT_SNAPSHOT_ROOT"] = str(isolated_snapshot_root)
        tiers[0]["root_dir"] = str(isolated_tier_root)
    executable_arguments[kv_option + 1] = json.dumps(
        kv_transfer, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    if fresh_target:
        try:
            speculative_option = executable_arguments.index("--speculative-config")
        except ValueError as error:
            raise ReplayError("base command lacks --speculative-config") from error
        if speculative_option + 1 >= len(executable_arguments):
            raise ReplayError("base command has no speculative configuration value")
        del executable_arguments[speculative_option : speculative_option + 2]
        kv_option = executable_arguments.index("--kv-transfer-config")
        del executable_arguments[kv_option : kv_option + 2]
        if "--enable-prefix-caching" not in executable_arguments:
            raise ReplayError("base command lacks --enable-prefix-caching")
        prefix_option = executable_arguments.index("--enable-prefix-caching")
        executable_arguments[prefix_option] = "--no-enable-prefix-caching"
    if bf16_kv:
        if "QWEN_FIXED_SLOT_SNAPSHOT_EXPORT" not in environment:
            raise ReplayError("base command lacks fixed-slot snapshot-export identity")
        environment["QWEN_FIXED_SLOT_SNAPSHOT_EXPORT"] = "0"
        try:
            dtype_option = executable_arguments.index("--kv-cache-dtype")
        except ValueError as error:
            raise ReplayError("base command lacks --kv-cache-dtype") from error
        if dtype_option + 1 >= len(executable_arguments):
            raise ReplayError("base command has no KV-cache dtype value")
        executable_arguments[dtype_option + 1] = "bfloat16"
        try:
            length_option = executable_arguments.index("--max-model-len")
        except ValueError as error:
            raise ReplayError("base command lacks --max-model-len") from error
        if length_option + 1 >= len(executable_arguments):
            raise ReplayError("base command has no maximum model length value")
        executable_arguments[length_option + 1] = str(BF16_ORACLE_MAX_MODEL_LEN)
        try:
            speculative_option = executable_arguments.index("--speculative-config")
        except ValueError as error:
            raise ReplayError("base command lacks --speculative-config") from error
        if speculative_option + 1 >= len(executable_arguments):
            raise ReplayError("base command has no speculative configuration value")
        try:
            speculative = json.loads(executable_arguments[speculative_option + 1])
        except json.JSONDecodeError as error:
            raise ReplayError(f"speculative configuration is invalid: {error}") from error
        if not isinstance(speculative, dict) or "max_model_len" not in speculative:
            raise ReplayError("speculative configuration lacks maximum model length")
        speculative["max_model_len"] = BF16_ORACLE_MAX_MODEL_LEN
        executable_arguments[speculative_option + 1] = json.dumps(
            speculative, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        snapshot_root = environment.get("QWEN_FIXED_SLOT_SNAPSHOT_ROOT")
        if not snapshot_root or not snapshot_root.startswith("/"):
            raise ReplayError("base command lacks an absolute fixed-slot snapshot root")
        environment["QWEN_FIXED_SLOT_SNAPSHOT_ROOT"] = f"{snapshot_root}-bf16-target-oracle-v1"
        tiers = extra.get("secondary_tiers")
        if not isinstance(tiers, list) or len(tiers) != 1 or not isinstance(tiers[0], dict):
            raise ReplayError("KV-transfer configuration has an unexpected tier layout")
        tier_root = tiers[0].get("root_dir")
        if not isinstance(tier_root, str) or not tier_root.startswith("/"):
            raise ReplayError("KV-transfer filesystem tier lacks an absolute root")
        tiers[0]["root_dir"] = f"{tier_root}-bf16-target-oracle-v1"
        target_only = not m8_serial and not standard_m8
        oracle_transport = "QWEN_HAUHAU_BF16_TARGET_ONLY_ORACLE"
        if oracle_transport not in environment:
            environment_order.append(oracle_transport)
        environment[oracle_transport] = "1" if target_only else "0"
        if "--enable-prefix-caching" not in executable_arguments:
            raise ReplayError("base command lacks --enable-prefix-caching")
        prefix_option = executable_arguments.index("--enable-prefix-caching")
        executable_arguments[prefix_option] = "--no-enable-prefix-caching"
        # Dense ROCm attention cannot be constructed with a KV connector.
        # The existing reference-GDN path also requires a fresh mode-none
        # cache. BF16 controls cannot publish snapshots, so the production
        # connector and prefix-cache path add no evidence to this arm.
        kv_option = executable_arguments.index("--kv-transfer-config")
        del executable_arguments[kv_option : kv_option + 2]
    rewritten = [
        "exec",
        "/usr/bin/env",
        "-i",
        *(f"{name}={environment[name]}" for name in environment_order),
        *executable_arguments,
    ]
    return (shlex.join(rewritten) + "\n").encode()


def _derive_standard_m8_serial_ba_command(payload: bytes) -> bytes:
    """Replace only the disproven grouped B/A schedule in a standard-M8 command."""

    try:
        text = payload.decode("utf-8")
        arguments = shlex.split(text)
    except (UnicodeDecodeError, ValueError) as error:
        raise ReplayError(f"base command is invalid: {error}") from error
    if len(text.splitlines()) != 1 or arguments[:3] != ["exec", "/usr/bin/env", "-i"]:
        raise ReplayError("base command must be one exec /usr/bin/env -i line")
    environment_end = 3
    environment: dict[str, str] = {}
    while environment_end < len(arguments):
        argument = arguments[environment_end]
        if "=" not in argument:
            break
        name, value = argument.split("=", 1)
        if ENVIRONMENT_NAME.fullmatch(name) is None:
            break
        if name in environment:
            raise ReplayError(f"base command duplicates environment variable {name}")
        environment[name] = value
        environment_end += 1
    required = {
        "QWEN_DFLASH_GREEDY_LOOP_ESCAPE": "0",
        "QWEN_DFLASH_GREEDY_M8_VERIFIER": "1",
        "QWEN_FIXED_SLOT_TARGET_ONLY_ORACLE": "0",
        "QWEN_GDN_BA_BATCHED_EXACT": "0",
        "QWEN_GDN_BA_BATCHED_CROSSCHECK": "0",
        "QWEN_GDN_BA_GROUPED_PREFIX_EXACT": "1",
        "QWEN_GDN_BA_GROUPED_PREFIX_CROSSCHECK": "0",
        "QWEN_GDN_BA_SERIAL_ROW_EXACT": "1",
    }
    mismatched = {
        name: environment.get(name)
        for name, expected in required.items()
        if environment.get(name) != expected
    }
    if mismatched:
        raise ReplayError(
            f"base command is not the pinned standard-M8 grouped-B/A arm: {mismatched}"
        )
    rewritten = list(arguments)
    replacements = STANDARD_M8_SERIAL_BA_OVERRIDES
    for index in range(3, environment_end):
        name, _value = rewritten[index].split("=", 1)
        if name in replacements:
            rewritten[index] = f"{name}={replacements[name]}"
    return (shlex.join(rewritten) + "\n").encode()


def _derive_standard_m8_gdn_crosscheck_command(payload: bytes) -> bytes:
    """Enable only the existing convolution and recurrence bitwise crosschecks."""

    try:
        text = payload.decode("utf-8")
        arguments = shlex.split(text)
    except (UnicodeDecodeError, ValueError) as error:
        raise ReplayError(f"base command is invalid: {error}") from error
    if len(text.splitlines()) != 1 or arguments[:3] != ["exec", "/usr/bin/env", "-i"]:
        raise ReplayError("base command must be one exec /usr/bin/env -i line")
    environment_end = 3
    environment: dict[str, str] = {}
    while environment_end < len(arguments):
        argument = arguments[environment_end]
        if "=" not in argument:
            break
        name, value = argument.split("=", 1)
        if ENVIRONMENT_NAME.fullmatch(name) is None:
            break
        if name in environment:
            raise ReplayError(f"base command duplicates environment variable {name}")
        environment[name] = value
        environment_end += 1
    required = {
        "QWEN_DFLASH_GREEDY_LOOP_ESCAPE": "0",
        "QWEN_DFLASH_GREEDY_M8_VERIFIER": "1",
        "QWEN_FIXED_SLOT_TARGET_ONLY_ORACLE": "0",
        "QWEN_GDN_FIXED_SLOT_CACHED_ACCEPTED_COMMIT": "1",
        "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED": "1",
        "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_VERIFY": "1",
        "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_VERIFY_CONV": "0",
        "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_VERIFY_RECURRENCE": "1",
        "QWEN_GDN_FIXED_SLOT_SERIAL_CONV_M8": "1",
    }
    mismatched = {
        name: environment.get(name)
        for name, expected in required.items()
        if environment.get(name) != expected
    }
    if mismatched:
        raise ReplayError(f"base command is not the pinned standard-M8 GDN arm: {mismatched}")
    rewritten = list(arguments)
    seen: set[str] = set()
    for index in range(3, environment_end):
        name, _value = rewritten[index].split("=", 1)
        if name in STANDARD_M8_GDN_CROSSCHECK_OVERRIDES:
            rewritten[index] = f"{name}={STANDARD_M8_GDN_CROSSCHECK_OVERRIDES[name]}"
            seen.add(name)
    missing = set(STANDARD_M8_GDN_CROSSCHECK_OVERRIDES) - seen
    rewritten[environment_end:environment_end] = [
        f"{name}={STANDARD_M8_GDN_CROSSCHECK_OVERRIDES[name]}" for name in sorted(missing)
    ]
    return (shlex.join(rewritten) + "\n").encode()


def derive_target_only_command(
    payload: bytes, *, dense: bool, bf16_kv: bool, fresh_target: bool = False
) -> bytes:
    """Derive a direct target-only oracle command from one pinned production command."""

    return _derive_oracle_command(
        payload,
        dense=dense,
        bf16_kv=bf16_kv,
        m8_serial=False,
        standard_m8=False,
        production_m8=False,
        standard_m8_serial_ba=False,
        m8_lm_head="production",
        fresh_target=fresh_target,
    )


def derive_command(args: argparse.Namespace) -> dict[str, Any]:
    try:
        base_payload = args.base_command.expanduser().read_bytes()
    except OSError as error:
        raise ReplayError(f"cannot read base command: {error}") from error
    observed_base_sha256 = _sha256(base_payload)
    if observed_base_sha256 != args.expected_base_sha256:
        raise ReplayError("base command SHA-256 differs from the pinned value")
    output_payload = _derive_oracle_command(
        base_payload,
        dense=args.dense,
        bf16_kv=args.bf16_kv,
        m8_serial=args.m8_serial,
        standard_m8=args.standard_m8,
        production_m8=args.production_m8,
        standard_m8_serial_ba=args.standard_m8_serial_ba,
        m8_lm_head=args.m8_lm_head,
        exact_m1_root=args.exact_m1_root,
        exact_m1_binder_sha256=args.exact_m1_binder_sha256,
        exact_m1_site_sha256=args.exact_m1_site_sha256,
        qualification_cache_root=args.qualification_cache_root,
        fresh_target=args.fresh_target,
        max_num_batched_tokens=args.max_num_batched_tokens,
        quest_rowwise_q1_control=args.quest_rowwise_q1_control,
    )
    output = args.output.expanduser().absolute()
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if output.parent.is_symlink() or stat.S_IMODE(output.parent.stat().st_mode) & 0o077:
        raise ReplayError("command output parent must be a real private directory")
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o700)
    except FileExistsError as error:
        raise ReplayError(f"output is create-only: {output}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(output_payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return {
        "base_command_sha256": observed_base_sha256,
        "bf16_kv": args.bf16_kv,
        "command_sha256": _sha256(output_payload),
        "dense": args.dense,
        "fresh_target": args.fresh_target,
        "oracle": (
            "m8-serial"
            if args.m8_serial
            else "standard-m8"
            if args.standard_m8
            else "production-m8"
            if args.production_m8
            else "target-only"
        ),
        "m8_lm_head": args.m8_lm_head,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "quest_rowwise_q1_control": args.quest_rowwise_q1_control,
        "standard_m8_serial_ba": args.standard_m8_serial_ba,
        "output": str(output),
    }


def derive_standard_m8_serial_ba_command(args: argparse.Namespace) -> dict[str, Any]:
    """Create an authenticated single-variable diagnostic command."""

    try:
        base_payload = args.base_command.expanduser().read_bytes()
    except OSError as error:
        raise ReplayError(f"cannot read base command: {error}") from error
    observed_base_sha256 = _sha256(base_payload)
    if observed_base_sha256 != args.expected_base_sha256:
        raise ReplayError("base command SHA-256 differs from the pinned value")
    output_payload = _derive_standard_m8_serial_ba_command(base_payload)
    output = args.output.expanduser().absolute()
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if output.parent.is_symlink() or stat.S_IMODE(output.parent.stat().st_mode) & 0o077:
        raise ReplayError("command output parent must be a real private directory")
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o700)
    except FileExistsError as error:
        raise ReplayError(f"output is create-only: {output}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(output_payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return {
        "base_command_sha256": observed_base_sha256,
        "changed_environment": STANDARD_M8_SERIAL_BA_OVERRIDES,
        "command_sha256": _sha256(output_payload),
        "output": str(output),
    }


def derive_standard_m8_gdn_crosscheck_command(args: argparse.Namespace) -> dict[str, Any]:
    """Create an authenticated standard-M8 GDN crosscheck command."""

    try:
        base_payload = args.base_command.expanduser().read_bytes()
    except OSError as error:
        raise ReplayError(f"cannot read base command: {error}") from error
    observed_base_sha256 = _sha256(base_payload)
    if observed_base_sha256 != args.expected_base_sha256:
        raise ReplayError("base command SHA-256 differs from the pinned value")
    output_payload = _derive_standard_m8_gdn_crosscheck_command(base_payload)
    output = args.output.expanduser().absolute()
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if output.parent.is_symlink() or stat.S_IMODE(output.parent.stat().st_mode) & 0o077:
        raise ReplayError("command output parent must be a real private directory")
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o700)
    except FileExistsError as error:
        raise ReplayError(f"output is create-only: {output}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(output_payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return {
        "base_command_sha256": observed_base_sha256,
        "changed_environment": STANDARD_M8_GDN_CROSSCHECK_OVERRIDES,
        "command_sha256": _sha256(output_payload),
        "output": str(output),
    }


def capture(args: argparse.Namespace) -> dict[str, Any]:
    tokens, vector_payload = read_qwenstg1(args.token_file)
    if not 1 <= args.max_tokens <= MAX_OUTPUT_TOKENS:
        raise ReplayError(f"max-tokens must be in 1..{MAX_OUTPUT_TOKENS}")
    if not args.arm_id or any(character.isspace() for character in args.arm_id):
        raise ReplayError("arm-id must be nonempty and contain no whitespace")
    if SHA256_RE.fullmatch(args.configuration_sha256) is None:
        raise ReplayError("configuration-sha256 must be 64 lowercase hexadecimal characters")
    requested_logprobs = getattr(args, "logprobs", None)
    if requested_logprobs is not None and not 1 <= requested_logprobs <= 20:
        raise ReplayError("logprobs must be in 1..20 when supplied")
    transport_path = getattr(args, "transport_contract", None)
    transport_sha256 = getattr(args, "transport_contract_sha256", None)
    if (transport_path is None) != (transport_sha256 is None):
        raise ReplayError(
            "transport-contract and transport-contract-sha256 must be supplied together"
        )
    source_file_sha256 = _sha256(vector_payload)
    source_token_ids_sha256 = _token_digest(tokens)
    transport = (
        _load_transport_contract(
            transport_path,
            transport_sha256,
            prompt_token_count=len(tokens),
            prompt_token_ids_sha256=source_token_ids_sha256,
            prompt_tokens=tokens,
            token_file_sha256=source_file_sha256,
        )
        if transport_path is not None and transport_sha256 is not None
        else None
    )
    cache_salt = _validate_cache_salt(args.cache_salt, fresh_state=transport is None)
    request_payload: dict[str, Any] = {
        "frequency_penalty": 0.0,
        "max_tokens": args.max_tokens,
        "model": args.model,
        "n": 1,
        "presence_penalty": 0.0,
        "prompt": tokens,
        "return_token_ids": True,
        "seed": 42,
        "stream": False,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
    }
    if cache_salt is not None:
        request_payload["cache_salt"] = cache_salt
    if transport is not None:
        request_payload["kv_transfer_params"] = transport["kv_transfer_params"]
    if requested_logprobs is not None:
        request_payload["logprobs"] = requested_logprobs
    request_body = _canonical(request_payload)
    semantic_request = dict(request_payload)
    semantic_request.pop("cache_salt", None)
    semantic_request.pop("kv_transfer_params", None)
    request = urllib.request.Request(
        _endpoint(args.base_url),
        data=request_body,
        method="POST",
        headers={
            "Authorization": f"Bearer {os.environ.get(args.api_key_env, 'EMPTY')}",
            "Content-Type": "application/json",
            "X-Request-Id": args.request_id,
        },
    )
    started = time.monotonic_ns()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            status = response.status
            response_headers = dict(response.headers.items())
            response_body = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ReplayError(f"endpoint returned HTTP {error.code}: {detail[:1000]}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise ReplayError(f"endpoint request failed: {error}") from error
    elapsed_ns = time.monotonic_ns() - started
    try:
        response_json = json.loads(response_body)
    except json.JSONDecodeError as error:
        raise ReplayError(f"endpoint returned invalid JSON: {error}") from error
    choices = response_json.get("choices") if isinstance(response_json, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ReplayError("response must contain exactly one choice")
    choice = choices[0]
    response_logprobs = choice.get("logprobs")
    if requested_logprobs is not None and not isinstance(response_logprobs, dict):
        raise ReplayError("response omitted requested completion logprobs")
    completion_ids = choice.get("token_ids")
    if not isinstance(completion_ids, list) or not all(
        type(token) is int for token in completion_ids
    ):
        raise ReplayError("response did not return exact integer completion token IDs")
    usage = response_json.get("usage")
    if (
        not isinstance(usage, dict)
        or usage.get("prompt_tokens") != len(tokens)
        or usage.get("completion_tokens") != len(completion_ids)
    ):
        raise ReplayError("response token usage differs from exact token vectors")
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "captured_at": datetime.now(UTC).isoformat(),
        "arm": {
            "id": args.arm_id,
            "configuration_sha256": args.configuration_sha256,
        },
        "source": {
            "token_file": str(args.token_file.expanduser().absolute()),
            "token_file_sha256": source_file_sha256,
            "prompt_token_count": len(tokens),
            "prompt_token_ids_sha256": source_token_ids_sha256,
        },
        "request": {
            "body_sha256": _sha256(request_body),
            "cache_salt": cache_salt,
            "endpoint": _endpoint(args.base_url),
            "logprobs": requested_logprobs,
            "request_id": args.request_id,
            "semantic_body_sha256": _sha256(_canonical(semantic_request)),
            "state_contract": (
                transport["state_contract"]
                if transport is not None
                else "fresh-state-exact-token-vector-v1"
            ),
            "transport": transport,
        },
        "response": {
            "body_base64": base64.b64encode(response_body).decode("ascii"),
            "body_sha256": _sha256(response_body),
            "completion_token_count": len(completion_ids),
            "completion_token_ids": completion_ids,
            "completion_token_ids_sha256": _token_digest(completion_ids),
            "elapsed_seconds": elapsed_ns / 1_000_000_000,
            "finish_reason": choice.get("finish_reason"),
            "http_status": status,
            "logprobs": response_logprobs,
            "periodic_suffix": _periodic_suffix(completion_ids),
            "response_headers": response_headers,
        },
    }
    fingerprint_source = dict(document)
    document["capture_sha256"] = _sha256(_canonical(fingerprint_source))
    _publish_create_only(args.output, document)
    return document


def capture_chat_stream(args: argparse.Namespace) -> dict[str, Any]:
    """Replay an exact provider prompt through the production chat parser.

    The preflight proves that detokenizing the preserved QWENSTG1 vector and
    rendering it through an identity chat template recreates exactly the same
    prompt token vector.  The streamed request then records both vLLM's parsed
    deltas and its otherwise-hidden raw completion token IDs.
    """

    tokens, vector_payload = read_qwenstg1(args.token_file)
    if not 1 <= args.max_tokens <= PRODUCTION_CHAT_MAX_OUTPUT_TOKENS:
        raise ReplayError(f"max-tokens must be in 1..{PRODUCTION_CHAT_MAX_OUTPUT_TOKENS}")
    if not 1 <= args.thinking_token_budget < args.max_tokens:
        raise ReplayError("thinking-token-budget must be positive and below max-tokens")
    if not args.arm_id or any(character.isspace() for character in args.arm_id):
        raise ReplayError("arm-id must be nonempty and contain no whitespace")
    if SHA256_RE.fullmatch(args.configuration_sha256) is None:
        raise ReplayError("configuration-sha256 must be 64 lowercase hexadecimal characters")
    cache_salt = args.cache_salt
    transport_path = args.transport_contract
    transport_sha256 = args.transport_contract_sha256
    if (transport_path is None) != (transport_sha256 is None):
        raise ReplayError(
            "transport-contract and transport-contract-sha256 must be supplied together"
        )
    source_file_sha256 = _sha256(vector_payload)
    source_token_ids_sha256 = _token_digest(tokens)
    transport = (
        _load_transport_contract(
            transport_path,
            transport_sha256,
            prompt_token_count=len(tokens),
            prompt_token_ids_sha256=source_token_ids_sha256,
            prompt_tokens=tokens,
            token_file_sha256=source_file_sha256,
        )
        if transport_path is not None and transport_sha256 is not None
        else None
    )
    cache_salt = _validate_cache_salt(cache_salt, fresh_state=transport is None)

    detokenize_payload = {"model": args.model, "tokens": tokens}
    (
        detokenize_status,
        detokenize_headers,
        detokenize_body,
        detokenize_json,
    ) = _post_json(
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        path="/detokenize",
        payload=detokenize_payload,
        request_id=f"{args.request_id}-prompt-detokenize",
        timeout=args.timeout,
    )
    prompt_text = detokenize_json.get("prompt")
    if not isinstance(prompt_text, str) or not prompt_text:
        raise ReplayError("/detokenize did not return a nonempty prompt")
    tools = _extract_tools(prompt_text)
    messages = [{"content": prompt_text, "role": "user"}]
    render_contract: dict[str, Any] = {
        "add_generation_prompt": False,
        "add_special_tokens": False,
        "chat_template": IDENTITY_CHAT_TEMPLATE,
        "continue_final_message": False,
        "messages": messages,
        "model": args.model,
        "tools": tools,
    }
    tokenize_payload = dict(render_contract)
    tokenize_status, tokenize_headers, tokenize_body, tokenize_json = _post_json(
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        path="/tokenize",
        payload=tokenize_payload,
        request_id=f"{args.request_id}-prompt-tokenize",
        timeout=args.timeout,
    )
    rendered_tokens = tokenize_json.get("tokens")
    if not isinstance(rendered_tokens, list) or not all(
        type(token) is int for token in rendered_tokens
    ):
        raise ReplayError("/tokenize did not return exact integer prompt token IDs")
    if rendered_tokens != tokens:
        shared = min(len(rendered_tokens), len(tokens))
        divergence = next(
            (index for index in range(shared) if rendered_tokens[index] != tokens[index]), shared
        )
        raise ReplayError(
            "identity chat rendering differs from the source vector "
            f"at token {divergence} (rendered={len(rendered_tokens)}, source={len(tokens)})"
        )

    request_payload: dict[str, Any] = {
        **render_contract,
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": True,
            "reasoning_effort": args.reasoning_effort,
        },
        "max_completion_tokens": args.max_tokens,
        "repetition_detection": dict(REPETITION_DETECTION),
        "return_prompt_text": True,
        "return_token_ids": True,
        "stream": True,
        "stream_options": {
            "continuous_usage_stats": True,
            "include_usage": True,
        },
        "temperature": 0.0,
        "thinking_token_budget": args.thinking_token_budget,
        "top_k": 1,
        "top_p": 1.0,
    }
    if cache_salt is not None:
        request_payload["cache_salt"] = cache_salt
    if transport is not None:
        request_payload["kv_transfer_params"] = transport["kv_transfer_params"]
    if getattr(args, "json_outcome_router", False):
        request_payload["response_format"] = _json_outcome_response_format(tools)
        request_payload["tool_choice"] = "none"
    request_body = _canonical(request_payload)
    semantic_request = dict(request_payload)
    semantic_request.pop("cache_salt", None)
    semantic_request.pop("kv_transfer_params", None)
    output = args.output.expanduser().absolute()
    request_evidence_path = Path(f"{output}.request.json")
    response_evidence_path = Path(f"{output}.response.sse")
    if output.exists() or request_evidence_path.exists() or response_evidence_path.exists():
        raise ReplayError("capture output and wire evidence paths must all be create-only")
    _publish_create_only(
        request_evidence_path,
        {
            "arm_id": args.arm_id,
            "body_base64": base64.b64encode(request_body).decode("ascii"),
            "body_sha256": _sha256(request_body),
            "configuration_sha256": args.configuration_sha256,
            "endpoint": _endpoint_path(args.base_url, "/v1/chat/completions"),
            "request_id": args.request_id,
            "semantic_body_sha256": _sha256(_canonical(semantic_request)),
            "source_token_file_sha256": source_file_sha256,
            "state_contract": (
                transport["state_contract"]
                if transport is not None
                else "fresh-state-exact-token-vector-v1"
            ),
            "transport": transport,
        },
    )
    chat_endpoint = _endpoint_path(args.base_url, "/v1/chat/completions")
    request = urllib.request.Request(
        chat_endpoint,
        data=request_body,
        method="POST",
        headers={
            **_request_headers(args.api_key_env, args.request_id),
            "Accept": "text/event-stream",
        },
    )
    started = time.monotonic_ns()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            status = response.status
            response_headers = dict(response.headers.items())
            response_body = _durable_stream_response(response, response_evidence_path)
    except urllib.error.HTTPError as error:
        try:
            detail_body = _durable_stream_response(error, response_evidence_path)
        except ReplayError:
            raise
        detail = detail_body.decode("utf-8", errors="replace")
        raise ReplayError(
            f"/v1/chat/completions returned HTTP {error.code}; raw response is preserved at "
            f"{response_evidence_path}: {detail[:1000]}"
        ) from error
    except (http.client.HTTPException, urllib.error.URLError, TimeoutError, OSError) as error:
        raise ReplayError(
            f"/v1/chat/completions request failed; partial raw response, if any, is preserved "
            f"at {response_evidence_path}: {error}"
        ) from error
    elapsed_ns = time.monotonic_ns() - started

    events, done_markers, sse_errors = _parse_sse(response_body)
    content_type = next(
        (value for key, value in response_headers.items() if key.lower() == "content-type"),
        "",
    )
    if not content_type.lower().startswith("text/event-stream"):
        sse_errors.append(f"unexpected response Content-Type: {content_type or '(missing)'}")
    aggregate = _aggregate_chat_stream(events, tokens)
    aggregate["errors"] = [*sse_errors, *aggregate["errors"]]
    if aggregate["prompt_text"] != prompt_text:
        aggregate["errors"].append("stream prompt_text differs from the preflight prompt")
    completion_ids = aggregate["completion_token_ids"]

    completion_detokenize_payload = {"model": args.model, "tokens": completion_ids}
    completion_detokenize_error: str | None = None
    completion_detokenize_status: int | None = None
    completion_detokenize_headers: dict[str, str] = {}
    completion_detokenize_body = b""
    raw_completion_text: str | None = None
    try:
        (
            completion_detokenize_status,
            completion_detokenize_headers,
            completion_detokenize_body,
            completion_detokenize_json,
        ) = _post_json(
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            path="/detokenize",
            payload=completion_detokenize_payload,
            request_id=f"{args.request_id}-completion-detokenize",
            timeout=args.timeout,
        )
        candidate_text = completion_detokenize_json.get("prompt")
        if isinstance(candidate_text, str):
            raw_completion_text = candidate_text
        else:
            completion_detokenize_error = "completion /detokenize did not return a prompt string"
    except ReplayError as error:
        # Raw SSE bytes and token IDs are primary evidence.  Detokenization is an
        # auxiliary rendering and must never make an otherwise valid capture vanish.
        completion_detokenize_error = str(error)

    tool_start_positions = [
        index
        for index, token in enumerate(completion_ids)
        if token == QWEN_TOOL_CALL_START_TOKEN_ID
    ]
    tool_end_positions = [
        index for index, token in enumerate(completion_ids) if token == QWEN_TOOL_CALL_END_TOKEN_ID
    ]
    tool_names = [tool["function"]["name"] for tool in tools]
    classification = _classify_chat_outcome(completion_ids, aggregate, tool_names)
    prompt_utf8 = prompt_text.encode()
    content_utf8 = aggregate["content"].encode()
    reasoning_utf8 = aggregate["reasoning"].encode()
    raw_completion_utf8 = raw_completion_text.encode() if raw_completion_text is not None else b""
    document: dict[str, Any] = {
        "schema": CHAT_STREAM_SCHEMA,
        "captured_at": datetime.now(UTC).isoformat(),
        "arm": {
            "id": args.arm_id,
            "configuration_sha256": args.configuration_sha256,
        },
        "source": {
            "token_file": str(args.token_file.expanduser().absolute()),
            "token_file_sha256": source_file_sha256,
            "prompt_token_count": len(tokens),
            "prompt_token_ids_sha256": source_token_ids_sha256,
        },
        "preflight": {
            "decoded_prompt_bytes": len(prompt_utf8),
            "decoded_prompt_utf8_sha256": _sha256(prompt_utf8),
            "detokenize": {
                "http_status": detokenize_status,
                "request_body_sha256": _sha256(_canonical(detokenize_payload)),
                "response_body_base64": base64.b64encode(detokenize_body).decode("ascii"),
                "response_body_sha256": _sha256(detokenize_body),
                "response_headers": detokenize_headers,
            },
            "exact_prompt_token_ids_verified": True,
            "identity_chat_template": IDENTITY_CHAT_TEMPLATE,
            "tokenize": {
                "http_status": tokenize_status,
                "request_body_sha256": _sha256(_canonical(tokenize_payload)),
                "response_body_base64": base64.b64encode(tokenize_body).decode("ascii"),
                "response_body_sha256": _sha256(tokenize_body),
                "response_headers": tokenize_headers,
            },
            "tool_names": tool_names,
            "tools_sha256": _sha256(_canonical(tools)),
        },
        "request": {
            # Keep the exact wire request, not merely a digest of fields that a
            # later version of this harness might reconstruct differently.
            # The enclosing capture is create-only and mode 0600 because this
            # body contains the preserved conversation.
            "body_base64": base64.b64encode(request_body).decode("ascii"),
            "body_sha256": _sha256(request_body),
            "cache_salt": cache_salt,
            "endpoint": chat_endpoint,
            "request_id": args.request_id,
            "semantic_body_sha256": _sha256(_canonical(semantic_request)),
            "state_contract": (
                transport["state_contract"]
                if transport is not None
                else "fresh-state-exact-token-vector-v1"
            ),
            "transport": transport,
            "wire_evidence_path": str(request_evidence_path),
            "wire_evidence_sha256": _sha256(request_evidence_path.read_bytes()),
        },
        "response": {
            "body_base64": base64.b64encode(response_body).decode("ascii"),
            "body_sha256": _sha256(response_body),
            "completion_token_count": len(completion_ids),
            "completion_token_ids": completion_ids,
            "completion_token_ids_sha256": _token_digest(completion_ids),
            "done_markers": done_markers,
            "elapsed_seconds": elapsed_ns / 1_000_000_000,
            "event_count": len(events),
            "finish_reason": aggregate["finish_reason"],
            "http_status": status,
            "parser": {
                "classification": classification,
                "content_base64": base64.b64encode(content_utf8).decode("ascii"),
                "content_utf8_sha256": _sha256(content_utf8),
                "errors": aggregate["errors"],
                "reasoning_base64": base64.b64encode(reasoning_utf8).decode("ascii"),
                "reasoning_utf8_sha256": _sha256(reasoning_utf8),
                "tool_call_deltas": aggregate["tool_call_deltas"],
                "tool_calls": aggregate["tool_calls"],
            },
            "periodic_suffix": _periodic_suffix(completion_ids),
            "raw_completion": {
                "detokenize_http_status": completion_detokenize_status,
                "detokenize_error": completion_detokenize_error,
                "detokenize_response_body_base64": base64.b64encode(
                    completion_detokenize_body
                ).decode("ascii"),
                "detokenize_response_body_sha256": _sha256(completion_detokenize_body),
                "detokenize_response_headers": completion_detokenize_headers,
                "text_base64": base64.b64encode(raw_completion_utf8).decode("ascii"),
                "text_bytes": len(raw_completion_utf8),
                "text_utf8_sha256": _sha256(raw_completion_utf8),
                "text_available": raw_completion_text is not None,
                "tool_call_end_positions": tool_end_positions,
                "tool_call_start_positions": tool_start_positions,
            },
            "response_headers": response_headers,
            "response_id": aggregate["response_id"],
            "stop_reason": aggregate["stop_reason"],
            "usage": aggregate["usage"],
            "wire_evidence_path": str(response_evidence_path),
            "wire_evidence_sha256": _sha256(response_evidence_path.read_bytes()),
        },
    }
    fingerprint_source = dict(document)
    document["capture_sha256"] = _sha256(_canonical(fingerprint_source))
    _publish_create_only(args.output, document)
    return document


def _load_capture(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.expanduser().read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayError(f"cannot load capture {path}: {error}") from error
    if not isinstance(document, dict) or document.get("schema") not in {
        SCHEMA,
        CHAT_STREAM_SCHEMA,
    }:
        raise ReplayError(f"not a failed-prompt replay capture: {path}")
    fingerprint = document.pop("capture_sha256", None)
    expected = _sha256(_canonical(document))
    document["capture_sha256"] = fingerprint
    if fingerprint != expected:
        raise ReplayError(f"capture fingerprint is invalid: {path}")
    return document


def _parser_semantic_signature(parser: Mapping[str, Any]) -> bytes:
    """Return the model-visible parser outcome without server-issued call IDs.

    Tool-call IDs are generated by the serving layer and SSE chunk boundaries
    are transport details.  Neither is emitted by the model, so comparing them
    would report a false semantic divergence for byte-identical completion
    token vectors.  The assembled call index, name, type, arguments and JSON
    validity remain part of the fail-closed semantic contract.
    """

    tool_calls = parser.get("tool_calls")
    if not isinstance(tool_calls, list) or not all(isinstance(call, dict) for call in tool_calls):
        raise ReplayError("capture parser has an invalid assembled tool-call ledger")
    normalized_calls = [
        {
            key: call.get(key)
            for key in ("arguments", "arguments_valid_json", "index", "name", "type")
        }
        for call in tool_calls
    ]
    return _canonical(
        {
            "classification": parser.get("classification"),
            "content_utf8_sha256": parser.get("content_utf8_sha256"),
            "reasoning_utf8_sha256": parser.get("reasoning_utf8_sha256"),
            "tool_calls": normalized_calls,
        }
    )


def _parser_generated_id_signature(parser: Mapping[str, Any]) -> bytes:
    """Return separately reported serving-generated tool-call identities."""

    tool_calls = parser.get("tool_calls")
    if not isinstance(tool_calls, list) or not all(isinstance(call, dict) for call in tool_calls):
        raise ReplayError("capture parser has an invalid assembled tool-call ledger")
    return _canonical([{"id": call.get("id"), "index": call.get("index")} for call in tool_calls])


def compare(paths: Sequence[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ReplayError("comparison requires at least two captures")
    captures = [_load_capture(path) for path in paths]
    capture_schema = captures[0]["schema"]
    source = captures[0]["source"]
    source_identity = {
        key: source[key]
        for key in (
            "prompt_token_count",
            "prompt_token_ids_sha256",
            "token_file_sha256",
        )
    }
    semantic_body_sha256 = captures[0]["request"]["semantic_body_sha256"]
    reference_state_contract = captures[0]["request"].get("state_contract")
    reference_transport = captures[0]["request"].get("transport")
    reference_transport_sha256 = (
        reference_transport.get("transport_body_sha256")
        if isinstance(reference_transport, dict)
        else None
    )
    vectors: list[list[int]] = []
    for path, capture_document in zip(paths, captures, strict=True):
        if capture_document["schema"] != capture_schema:
            raise ReplayError(f"captures use different schemas: {path}")
        candidate_source = capture_document["source"]
        candidate_identity = {key: candidate_source.get(key) for key in source_identity}
        if candidate_identity != source_identity:
            raise ReplayError(f"captures use different exact prompt vectors: {path}")
        if capture_document["request"]["semantic_body_sha256"] != semantic_body_sha256:
            raise ReplayError(f"captures use different semantic requests: {path}")
        if capture_schema == CHAT_STREAM_SCHEMA:
            response = capture_document["response"]
            parser = response.get("parser")
            parser_errors = parser.get("errors") if isinstance(parser, dict) else None
            if (
                response.get("http_status") != 200
                or response.get("done_markers") != 1
                or not isinstance(parser_errors, list)
                or parser_errors
            ):
                raise ReplayError(f"capture has an invalid SSE/parser contract: {path}")
        vectors.append(capture_document["response"]["completion_token_ids"])
    reference = vectors[0]
    reference_parser_signature: bytes | None = None
    reference_parser_generated_id_signature: bytes | None = None
    reference_terminal_signature: bytes | None = None
    if capture_schema == CHAT_STREAM_SCHEMA:
        reference_response = captures[0]["response"]
        reference_parser = reference_response["parser"]
        reference_parser_signature = _parser_semantic_signature(reference_parser)
        reference_parser_generated_id_signature = _parser_generated_id_signature(reference_parser)
        reference_terminal_signature = _canonical(
            {
                "finish_reason": reference_response["finish_reason"],
                "stop_reason": reference_response["stop_reason"],
            }
        )
    arms: list[dict[str, Any]] = []
    all_identical = True
    all_parser_outcomes_identical = True
    all_parser_generated_ids_identical = True
    all_state_contracts_identical = True
    all_terminal_outcomes_identical = True
    all_transport_contracts_identical = True
    for path, capture_document, candidate in zip(paths, captures, vectors, strict=True):
        shared = min(len(reference), len(candidate))
        divergence = next(
            (index for index in range(shared) if reference[index] != candidate[index]),
            shared if len(reference) != len(candidate) else None,
        )
        all_identical &= divergence is None
        parser_classification = None
        parser_outcome_identical = None
        parser_generated_ids_identical = None
        terminal_outcome_identical = None
        stop_reason = capture_document["response"].get("stop_reason")
        request_document = capture_document["request"]
        state_contract = request_document.get("state_contract")
        transport_document = request_document.get("transport")
        transport_sha256 = (
            transport_document.get("transport_body_sha256")
            if isinstance(transport_document, dict)
            else None
        )
        state_contract_identical = state_contract == reference_state_contract
        transport_contract_identical = transport_sha256 == reference_transport_sha256
        all_state_contracts_identical &= state_contract_identical
        all_transport_contracts_identical &= transport_contract_identical
        if capture_schema == CHAT_STREAM_SCHEMA:
            response = capture_document["response"]
            parser = response["parser"]
            parser_classification = parser["classification"]
            parser_outcome_identical = (
                _parser_semantic_signature(parser) == reference_parser_signature
            )
            parser_generated_ids_identical = (
                _parser_generated_id_signature(parser) == reference_parser_generated_id_signature
            )
            terminal_outcome_identical = (
                _canonical(
                    {
                        "finish_reason": response["finish_reason"],
                        "stop_reason": response["stop_reason"],
                    }
                )
                == reference_terminal_signature
            )
            all_parser_outcomes_identical &= parser_outcome_identical
            all_parser_generated_ids_identical &= parser_generated_ids_identical
            all_terminal_outcomes_identical &= terminal_outcome_identical
        arms.append(
            {
                "arm_id": capture_document["arm"]["id"],
                "capture_sha256": capture_document["capture_sha256"],
                "capture_path": str(path.expanduser().absolute()),
                "completion_tokens": len(candidate),
                "configuration_sha256": capture_document["arm"]["configuration_sha256"],
                "first_divergence_from_reference": divergence,
                "finish_reason": capture_document["response"]["finish_reason"],
                "parser_classification": parser_classification,
                "parser_generated_ids_identical": parser_generated_ids_identical,
                "parser_outcome_identical": parser_outcome_identical,
                "periodic_suffix": capture_document["response"]["periodic_suffix"],
                "state_contract": state_contract,
                "state_contract_identical": state_contract_identical,
                "stop_reason": stop_reason,
                "terminal_outcome_identical": terminal_outcome_identical,
                "token_ids_sha256": capture_document["response"]["completion_token_ids_sha256"],
                "transport_body_sha256": transport_sha256,
                "transport_contract_identical": transport_contract_identical,
            }
        )
    result: dict[str, Any] = {
        "schema": COMPARISON_SCHEMA,
        "capture_schema": capture_schema,
        "reference_arm": captures[0]["arm"]["id"],
        "source": source,
        "all_completion_token_ids_identical": all_identical,
        "all_parser_outcomes_identical": all_parser_outcomes_identical,
        "all_parser_generated_ids_identical": all_parser_generated_ids_identical,
        "all_state_contracts_identical": all_state_contracts_identical,
        "all_stream_contracts_valid": True,
        "all_terminal_outcomes_identical": all_terminal_outcomes_identical,
        "all_transport_contracts_identical": all_transport_contracts_identical,
        "all_observable_outcomes_identical": (
            all_identical and all_parser_outcomes_identical and all_terminal_outcomes_identical
        ),
        "arms": arms,
    }
    result["comparison_sha256"] = _sha256(_canonical(result))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen-failed-prompt-replay")
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    capture_parser.add_argument("--arm-id", required=True)
    capture_parser.add_argument("--base-url", required=True)
    capture_parser.add_argument("--cache-salt")
    capture_parser.add_argument("--configuration-sha256", required=True)
    capture_parser.add_argument(
        "--logprobs",
        type=int,
        help="record the top 1..20 completion logprobs for boundary diagnosis",
    )
    capture_parser.add_argument("--max-tokens", type=int, default=2048)
    capture_parser.add_argument("--model", required=True)
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.add_argument("--request-id", required=True)
    capture_parser.add_argument("--timeout", type=float, default=900.0)
    capture_parser.add_argument("--token-file", type=Path, required=True)
    capture_parser.add_argument("--transport-contract", type=Path)
    capture_parser.add_argument("--transport-contract-sha256")
    capture_parser.set_defaults(handler=lambda args: capture(args))
    chat_parser = subparsers.add_parser("capture-chat-stream")
    chat_parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    chat_parser.add_argument("--arm-id", required=True)
    chat_parser.add_argument("--base-url", required=True)
    chat_parser.add_argument("--cache-salt")
    chat_parser.add_argument("--configuration-sha256", required=True)
    chat_parser.add_argument(
        "--json-outcome-router",
        action="store_true",
        help="replay Pi's exact strict one-of tool/final JSON outcome contract",
    )
    chat_parser.add_argument("--max-tokens", type=int, default=PRODUCTION_CHAT_MAX_OUTPUT_TOKENS)
    chat_parser.add_argument("--model", required=True)
    chat_parser.add_argument("--output", type=Path, required=True)
    chat_parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        default="xhigh",
    )
    chat_parser.add_argument("--request-id", required=True)
    chat_parser.add_argument(
        "--thinking-token-budget",
        type=int,
        default=PRODUCTION_XHIGH_THINKING_TOKEN_BUDGET,
    )
    chat_parser.add_argument("--timeout", type=float, default=900.0)
    chat_parser.add_argument("--token-file", type=Path, required=True)
    chat_parser.add_argument("--transport-contract", type=Path)
    chat_parser.add_argument("--transport-contract-sha256")
    chat_parser.set_defaults(handler=lambda args: capture_chat_stream(args))
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("captures", nargs="+", type=Path)
    compare_parser.add_argument("--output", type=Path)
    compare_parser.set_defaults(handler=lambda args: compare(args.captures))
    derive_parser = subparsers.add_parser("derive-command")
    derive_parser.add_argument("--base-command", type=Path, required=True)
    derive_parser.add_argument("--bf16-kv", action="store_true")
    derive_parser.add_argument("--dense", action="store_true")
    derive_parser.add_argument("--expected-base-sha256", required=True)
    derive_parser.add_argument("--fresh-target", action="store_true")
    derive_parser.add_argument("--exact-m1-binder-sha256")
    derive_parser.add_argument("--exact-m1-root", type=Path)
    derive_parser.add_argument("--exact-m1-site-sha256")
    derive_parser.add_argument("--m8-serial", action="store_true")
    derive_parser.add_argument("--max-num-batched-tokens", type=int)
    derive_parser.add_argument("--production-m8", action="store_true")
    derive_parser.add_argument("--quest-rowwise-q1-control", action="store_true")
    derive_parser.add_argument("--qualification-cache-root", type=Path)
    derive_parser.add_argument("--standard-m8", action="store_true")
    derive_parser.add_argument("--standard-m8-serial-ba", action="store_true")
    derive_parser.add_argument(
        "--m8-lm-head",
        choices=tuple(M8_LM_HEAD_OVERRIDES),
        default="production",
    )
    derive_parser.add_argument("--output", type=Path, required=True)
    derive_parser.set_defaults(handler=derive_command)
    serial_ba_parser = subparsers.add_parser("derive-standard-m8-serial-ba-command")
    serial_ba_parser.add_argument("--base-command", type=Path, required=True)
    serial_ba_parser.add_argument("--expected-base-sha256", required=True)
    serial_ba_parser.add_argument("--output", type=Path, required=True)
    serial_ba_parser.set_defaults(handler=derive_standard_m8_serial_ba_command)
    gdn_crosscheck_parser = subparsers.add_parser("derive-standard-m8-gdn-crosscheck-command")
    gdn_crosscheck_parser.add_argument("--base-command", type=Path, required=True)
    gdn_crosscheck_parser.add_argument("--expected-base-sha256", required=True)
    gdn_crosscheck_parser.add_argument("--output", type=Path, required=True)
    gdn_crosscheck_parser.set_defaults(handler=derive_standard_m8_gdn_crosscheck_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = args.handler(args)
        if args.command == "compare" and args.output is not None:
            _publish_create_only(args.output, result)
        printable = result
        if args.command == "capture-chat-stream":
            printable = {
                "arm": result["arm"],
                "capture_sha256": result["capture_sha256"],
                "classification": result["response"]["parser"]["classification"],
                "completion_token_count": result["response"]["completion_token_count"],
                "finish_reason": result["response"]["finish_reason"],
                "output": str(args.output.expanduser().absolute()),
                "parser_errors": result["response"]["parser"]["errors"],
                "schema": result["schema"],
                "stop_reason": result["response"]["stop_reason"],
            }
        print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except ReplayError as error:
        print(f"qwen-failed-prompt-replay: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
