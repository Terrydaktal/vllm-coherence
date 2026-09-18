"""Render an authenticated fixed-slot-only external cache lookup overlay.

The live62 fixed-slot lane carries recurrent GDN/convolution state in its
authenticated snapshot payload.  A generic external prefix-cache hit carries
only the ordinary cache chunks and is therefore not a valid substitute.  This
overlay preserves explicitly authorized fixed-slot restores and returns a zero
external hit when no authorized boundary exists.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qwen_r9700_lab.full_attention_m8_row_exact_overlay import (
    RowExactOverlayError,
    _authenticate_chained_site,
    _digest,
    _environment_map,
    _find_chained_site,
    _split_command,
    _stable_private_file,
    _write_exclusive,
)

SCHEMA = "urn:qwen-r9700:fixed-slot-pinned-replay-overlay:v2"
TARGET_MODULE = "vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler"
KV_MANAGER_MODULE = "vllm.v1.core.kv_cache_manager"
ENABLE_ENV = "QWEN_FIXED_SLOT_GENERIC_PREFIX_BLOCK"
REQUIRED_ENV = "QWEN_FIXED_SLOT_GENERIC_PREFIX_BLOCK_REQUIRED"
SITE_SHA_ENV = "QWEN_FIXED_SLOT_GENERIC_PREFIX_BLOCK_SITE_SHA256"
RUNTIME_SHA_ENV = "QWEN_FIXED_SLOT_GENERIC_PREFIX_BLOCK_RUNTIME_SHA256"
KV_RUNTIME_SHA_ENV = "QWEN_FIXED_SLOT_RESIDENT_EVICT_RUNTIME_SHA256"
_OVERLAY_ENVIRONMENT = (
    ENABLE_ENV,
    REQUIRED_ENV,
    SITE_SHA_ENV,
    RUNTIME_SHA_ENV,
    KV_RUNTIME_SHA_ENV,
)


class FixedSlotPinnedReplayOverlayError(RowExactOverlayError):
    """The fixed-slot lookup overlay contract is invalid."""


def _option_values(argv: Sequence[str], option: str) -> list[str]:
    return [argv[index + 1] for index, token in enumerate(argv[:-1]) if token == option]


def _validate_base(
    environment: Mapping[str, str], argv: Sequence[str], *, target_oracle: bool
) -> None:
    required_environment = {
        "QWEN_FIXED_SLOT_PINNED_REPLAY_ONLY": "1",
        "QWEN_FIXED_SLOT_SNAPSHOT_EXPORT": "1",
        "QWEN_FIXED_SLOT_CORRECTED_GENERATION": "1",
        "QWEN_FIXED_SLOT_ROLLING_CHECKPOINT": "1",
        "QWEN_FIXED_SLOT_TARGET_ONLY_ORACLE": "1" if target_oracle else "0",
        "QWEN_DFLASH_GREEDY_M8_VERIFIER": "0" if target_oracle else "1",
    }
    if any(environment.get(name) != value for name, value in required_environment.items()):
        raise FixedSlotPinnedReplayOverlayError(
            "base command lacks the selected pinned fixed-slot execution identity"
        )
    if argv.count("--enable-prefix-caching") != 1:
        raise FixedSlotPinnedReplayOverlayError(
            "base command must enable prefix caching exactly once"
        )
    if _option_values(argv, "--max-num-seqs") != ["1"]:
        raise FixedSlotPinnedReplayOverlayError(
            "base command must isolate one request with --max-num-seqs 1"
        )
    if _option_values(argv, "--mamba-cache-mode") != ["none"]:
        raise FixedSlotPinnedReplayOverlayError(
            "base command must use one exact --mamba-cache-mode none"
        )
    transfer_values = _option_values(argv, "--kv-transfer-config")
    if len(transfer_values) != 1:
        raise FixedSlotPinnedReplayOverlayError(
            "base command must contain one exact --kv-transfer-config"
        )
    try:
        transfer = json.loads(transfer_values[0])
    except json.JSONDecodeError as error:
        raise FixedSlotPinnedReplayOverlayError(
            "base command KV transfer configuration is invalid JSON"
        ) from error
    extra = transfer.get("kv_connector_extra_config") if isinstance(transfer, dict) else None
    if (
        transfer.get("kv_connector") != "OffloadingConnector"
        or transfer.get("kv_role") != "kv_both"
        or not isinstance(extra, dict)
        or extra.get("offload_prompt_only") is not False
        or extra.get("fixed_slot_corrected_generation") is not True
        or extra.get("fixed_slot_target_only_oracle") is not target_oracle
        or extra.get("fixed_slot_trusted_replay") is target_oracle
        or extra.get("fixed_slot_whole_model_replay") is target_oracle
    ):
        raise FixedSlotPinnedReplayOverlayError(
            "base command lacks the complete fixed-slot connector contract"
        )


def _site_source(
    destination: Path,
    *,
    chained_site: Path,
    chained_site_sha256: str,
    chained_pythonpath: str,
    expected_runtime_sha256: str,
    expected_kv_runtime_sha256: str,
) -> bytes:
    outer_pythonpath = f"{destination}:{chained_pythonpath}"
    return f'''"""Authenticated fixed-slot-only external lookup bootstrap."""
import functools
import hashlib
import os
import runpy
import stat
import sys
from pathlib import Path

_ROOT = Path({str(destination)!r})
_SELF = _ROOT / "sitecustomize.py"
_CHAIN = Path({str(chained_site)!r})
_CHAIN_SHA256 = {chained_site_sha256!r}
_CHAIN_PYTHONPATH = {chained_pythonpath!r}
_OUTER_PYTHONPATH = {outer_pythonpath!r}
_TARGET = {TARGET_MODULE!r}
_KV_TARGET = {KV_MANAGER_MODULE!r}
_RUNTIME_SHA256 = {expected_runtime_sha256!r}
_KV_RUNTIME_SHA256 = {expected_kv_runtime_sha256!r}
_METHOD_MARKER = "_qwen_fixed_slot_generic_prefix_block_site_sha256"
_EVICT_MARKER = "_qwen_fixed_slot_resident_evict_site_sha256"

def _stable_digest(path, label, private=False):
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise RuntimeError(f"{{label}} identity is unsafe")
    if private and (before.st_uid != os.getuid() or stat.S_IMODE(before.st_mode) & 0o077):
        raise RuntimeError(f"{{label}} is not owned and private")
    payload = path.read_bytes()
    after = path.lstat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise RuntimeError(f"{{label}} changed during authentication")
    return hashlib.sha256(payload).hexdigest()

if os.environ.get({ENABLE_ENV!r}) != "1" or os.environ.get({REQUIRED_ENV!r}) != "1":
    raise RuntimeError("fixed-slot generic-prefix block identity is absent")
if os.environ.get({RUNTIME_SHA_ENV!r}) != _RUNTIME_SHA256:
    raise RuntimeError("fixed-slot scheduler runtime identity differs")
if os.environ.get({KV_RUNTIME_SHA_ENV!r}) != _KV_RUNTIME_SHA256:
    raise RuntimeError("fixed-slot KV-manager runtime identity differs")
if os.environ.get("QWEN_FIXED_SLOT_PINNED_REPLAY_ONLY") != "1":
    raise RuntimeError("fixed-slot generic-prefix block requires pinned replay only")
if os.environ.get("QWEN_FIXED_SLOT_SNAPSHOT_EXPORT") != "1":
    raise RuntimeError("fixed-slot snapshot export is absent")
if os.environ.get("PYTHONPATH") != _OUTER_PYTHONPATH:
    raise RuntimeError("fixed-slot generic-prefix block PYTHONPATH differs")
_SITE_SHA256 = _stable_digest(_SELF, "fixed-slot lookup site", private=True)
if _SITE_SHA256 != os.environ.get({SITE_SHA_ENV!r}):
    raise RuntimeError("fixed-slot lookup site SHA256 mismatch")
if _stable_digest(_CHAIN, "chained site", private=True) != _CHAIN_SHA256:
    raise RuntimeError("fixed-slot lookup chained-site SHA256 mismatch")
if _TARGET in sys.modules or _KV_TARGET in sys.modules:
    raise RuntimeError("fixed-slot runtime imported before replay-safety installation")

os.environ["PYTHONPATH"] = _CHAIN_PYTHONPATH
try:
    runpy.run_path(str(_CHAIN), run_name="_qwen_fixed_slot_lookup_chain")
    if os.environ.get("PYTHONPATH") != _CHAIN_PYTHONPATH:
        raise RuntimeError("chained site changed PYTHONPATH")
finally:
    os.environ["PYTHONPATH"] = _OUTER_PYTHONPATH
if _TARGET in sys.modules or _KV_TARGET in sys.modules:
    raise RuntimeError("chained site imported fixed-slot runtime before patch registration")

import combined_runtime_patch as retained  # noqa: E402

finder_type = getattr(retained, "_Finder", None)
patches = getattr(finder_type, "_PATCHES", None)
if not isinstance(finder_type, type) or type(patches) is not dict:
    raise RuntimeError("combined-runtime finder contract changed")
active_finders = [value for value in sys.meta_path if isinstance(value, finder_type)]
if len(active_finders) != 1:
    raise RuntimeError("combined-runtime finder is not installed exactly once")
previous_patch = patches.get(_TARGET)
if previous_patch is not None and not callable(previous_patch):
    raise RuntimeError("existing offloading-scheduler patch is not callable")
previous_marker = getattr(previous_patch, _METHOD_MARKER, None) if previous_patch else None
if previous_marker is not None and previous_marker != _SITE_SHA256:
    raise RuntimeError("a different fixed-slot lookup repair is registered")
previous_kv_patch = patches.get(_KV_TARGET)
if previous_kv_patch is not None and not callable(previous_kv_patch):
    raise RuntimeError("existing KV-manager patch is not callable")
previous_kv_marker = (
    getattr(previous_kv_patch, _EVICT_MARKER, None) if previous_kv_patch else None
)
if previous_kv_marker is not None and previous_kv_marker != _SITE_SHA256:
    raise RuntimeError("a different fixed-slot resident-eviction repair is registered")

def _patch_scheduler(module):
    source = Path(getattr(module, "__file__", ""))
    if (
        not source.is_absolute()
        or _stable_digest(source, "offloading scheduler runtime") != _RUNTIME_SHA256
    ):
        raise RuntimeError("offloading scheduler source SHA256 mismatch")
    if getattr(module, "__name__", None) != _TARGET:
        raise RuntimeError("offloading scheduler module identity changed")
    cls = getattr(module, "OffloadingConnectorScheduler", None)
    current = getattr(cls, "_lookup", None)
    current_marker = getattr(current, _METHOD_MARKER, None)
    if current_marker is not None:
        if current_marker != _SITE_SHA256:
            raise RuntimeError("scheduler already carries a different lookup repair")
        return
    if previous_patch is not None:
        previous_patch(module)
    cls = getattr(module, "OffloadingConnectorScheduler", None)
    original = getattr(cls, "_lookup", None)
    fixed_lookup = getattr(cls, "_lookup_fixed_slot_resume", None)
    original_stores = getattr(cls, "_build_store_jobs", None)
    original_partial_stores = getattr(cls, "_build_partial_tail_store_jobs", None)
    if (
        not isinstance(cls, type)
        or not callable(original)
        or not callable(fixed_lookup)
        or not callable(original_stores)
        or not callable(original_partial_stores)
    ):
        raise RuntimeError("offloading scheduler lookup ABI changed")

    @functools.wraps(original)
    def fixed_slot_only_lookup(self, req_status):
        config = getattr(self, "config", None)
        if not getattr(config, "qwen_fixed_slot_snapshot", False):
            raise RuntimeError("pinned fixed-slot lookup reached a non-fixed-slot scheduler")
        selection_runtime = getattr(self, "_qwen_250k_selection_runtime", None)
        if selection_runtime is None:
            raise RuntimeError("pinned fixed-slot lookup lacks the selection runtime")
        request = getattr(req_status, "req", None)
        request_id = getattr(request, "request_id", None)
        if type(request_id) is not str or not request_id:
            raise RuntimeError("pinned fixed-slot request identity is invalid")
        resident_attestation = getattr(request, "_qwen_fixed_slot_resident_attestation", None)
        boundary = (
            resident_attestation[3]
            if type(resident_attestation) is tuple and len(resident_attestation) == 4
            else selection_runtime.authorized_resume_boundary(request_id)
        )
        req_status.partial_tail_boundary = None
        if boundary is None:
            logger = getattr(module, "logger", None)
            if logger is not None:
                logger.info(
                    "Request %s: generic external prefix lookup blocked; "
                    "no authenticated fixed-slot boundary",
                    request_id,
                )
            return 0
        return fixed_lookup(self, req_status, boundary)

    def _authenticated_capture_owner(self, request_id, request):
        selection_runtime = getattr(self, "_qwen_250k_selection_runtime", None)
        directives = getattr(selection_runtime, "_directives", None)
        pending = getattr(selection_runtime, "_pending", None)
        if type(directives) is not dict or type(pending) is not dict:
            return False
        owner = directives.get(request_id)
        if owner is None:
            candidate = pending.get(request_id)
            owner = getattr(candidate, "directive", None)
        if owner is None:
            return False
        params = getattr(request, "kv_transfer_params", None)
        contract = params.get("qwen_250k_cache") if type(params) is dict else None
        if type(contract) is not dict or contract.get("mode") not in (
            "capture",
            "capture_checkpoint",
            "resume_checkpoint",
        ):
            return False
        return (
            type(request_id) is str
            and request_id
            and getattr(owner, "request_id", None) == request_id
            and getattr(owner, "session_id", None) == contract.get("session_id")
            and getattr(owner, "branch", None) == contract.get("branch")
        )

    def _all_store_requests_are_authenticated_captures(self, scheduler_output):
        request_ids = set(getattr(scheduler_output, "num_scheduled_tokens", ()))
        request_ids.update(getattr(scheduler_output, "finished_req_ids", ()) or ())
        if not request_ids:
            return False
        for request_id in request_ids:
            req_status = self._req_status.get(request_id)
            request = getattr(req_status, "req", None)
            if request is None or not _authenticated_capture_owner(self, request_id, request):
                return False
        return True

    @functools.wraps(original_stores)
    def fixed_slot_only_stores(self, scheduler_output):
        if not _all_store_requests_are_authenticated_captures(self, scheduler_output):
            return {{}}
        return original_stores(self, scheduler_output)

    @functools.wraps(original_partial_stores)
    def fixed_slot_only_partial_stores(self, scheduler_output):
        if not _all_store_requests_are_authenticated_captures(self, scheduler_output):
            return {{}}
        return original_partial_stores(self, scheduler_output)

    setattr(fixed_slot_only_lookup, _METHOD_MARKER, _SITE_SHA256)
    cls._lookup = fixed_slot_only_lookup
    cls._build_store_jobs = fixed_slot_only_stores
    cls._build_partial_tail_store_jobs = fixed_slot_only_partial_stores
    print(
        "[qwen-fixed-slot-pinned-replay] generic external prefix lookup blocked",
        flush=True,
    )

def _patch_kv_manager(module):
    source = Path(getattr(module, "__file__", ""))
    if (
        not source.is_absolute()
        or _stable_digest(source, "KV-manager runtime") != _KV_RUNTIME_SHA256
    ):
        raise RuntimeError("KV-manager source SHA256 mismatch")
    if getattr(module, "__name__", None) != _KV_TARGET:
        raise RuntimeError("KV-manager module identity changed")
    cls = getattr(module, "KVCacheManager", None)
    current = getattr(cls, "allocate_slots", None)
    current_marker = getattr(current, _EVICT_MARKER, None)
    if current_marker is not None:
        if current_marker != _SITE_SHA256:
            raise RuntimeError("KV manager already carries a different resident-eviction repair")
        return
    if previous_kv_patch is not None:
        previous_kv_patch(module)
    cls = getattr(module, "KVCacheManager", None)
    original = getattr(cls, "allocate_slots", None)
    if not isinstance(cls, type) or not callable(original):
        raise RuntimeError("KV-manager allocation ABI changed")

    @functools.wraps(original)
    def release_stale_resident_before_restore(
        self,
        request,
        num_new_tokens,
        num_new_computed_tokens=0,
        new_computed_blocks=None,
        num_lookahead_tokens=0,
        num_external_computed_tokens=0,
        delay_cache_blocks=False,
        num_encoder_tokens=0,
        full_sequence_must_fit=False,
        reserved_blocks=0,
        has_scheduled_reqs=True,
    ):
        # At this point an authenticated connector lookup has already returned a
        # positive external boundary.  A mismatched private resident/live head
        # cannot supply the earlier recurrent state, but its pins can consume the
        # lane's only Mamba slot and make allocate_slots return None forever.
        # Release only same-session/branch heads, and only when no exact resident
        # attestation was accepted for this request.
        params = getattr(request, "kv_transfer_params", None)
        cache = params.get("qwen_250k_cache") if type(params) is dict else None
        resident_attestation = getattr(
            request, "_qwen_fixed_slot_resident_attestation", None
        )
        authorized_restore = (
            delay_cache_blocks is True
            and type(num_external_computed_tokens) is int
            and num_external_computed_tokens > 0
            and num_new_computed_tokens == 0
            and resident_attestation is None
            and type(cache) is dict
            and cache.get("mode") in ("resume", "resume_checkpoint")
            and type(cache.get("session_id")) is str
            and type(cache.get("branch")) is str
        )
        if authorized_restore:
            session_id = cache["session_id"]
            branch = cache["branch"]
            resident_owner = getattr(self, "_fixed_slot_resident_owner", None)
            live_owner = getattr(self, "_fixed_slot_live_owner", None)
            owners = [owner for owner in (resident_owner, live_owner) if owner is not None]
            owners_match = bool(owners) and all(
                type(owner) is tuple
                and len(owner) >= 2
                and owner[0] == session_id
                and owner[1] == branch
                for owner in owners
            )
            if owners_match:
                resident_pins = list(
                    getattr(self, "_fixed_slot_resident_pins", ()) or ()
                )
                live_pins = list(getattr(self, "_fixed_slot_live_pins", ()) or ())
                pins = [*resident_pins, *live_pins]
                if pins:
                    self.block_pool.free_blocks(pins)
                self._fixed_slot_resident_owner = None
                self._fixed_slot_resident_blocks = {{}}
                self._fixed_slot_resident_hashes = {{}}
                self._fixed_slot_resident_sequences = {{}}
                self._fixed_slot_resident_pins = []
                self._fixed_slot_live_owner = None
                self._fixed_slot_live_sequences = {{}}
                self._fixed_slot_live_hashes = {{}}
                self._fixed_slot_live_pins = []
                logger = getattr(module, "logger", None)
                if logger is not None:
                    logger.info(
                        "Released stale same-branch fixed-slot heads before "
                        "authenticated external restore request=%s boundary=%d "
                        "resident_pins=%d live_pins=%d",
                        getattr(request, "request_id", ""),
                        num_external_computed_tokens,
                        len(resident_pins),
                        len(live_pins),
                    )
        return original(
            self,
            request,
            num_new_tokens,
            num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            num_lookahead_tokens=num_lookahead_tokens,
            num_external_computed_tokens=num_external_computed_tokens,
            delay_cache_blocks=delay_cache_blocks,
            num_encoder_tokens=num_encoder_tokens,
            full_sequence_must_fit=full_sequence_must_fit,
            reserved_blocks=reserved_blocks,
            has_scheduled_reqs=has_scheduled_reqs,
        )

    setattr(release_stale_resident_before_restore, _EVICT_MARKER, _SITE_SHA256)
    cls.allocate_slots = release_stale_resident_before_restore
    print(
        "[qwen-fixed-slot-pinned-replay] stale resident restore deadlock guard armed",
        flush=True,
    )

if previous_marker is None:
    setattr(_patch_scheduler, _METHOD_MARKER, _SITE_SHA256)
    patches[_TARGET] = _patch_scheduler
elif patches.get(_TARGET) is not previous_patch:
    raise RuntimeError("fixed-slot lookup finder identity changed")
if previous_kv_marker is None:
    setattr(_patch_kv_manager, _EVICT_MARKER, _SITE_SHA256)
    patches[_KV_TARGET] = _patch_kv_manager
elif patches.get(_KV_TARGET) is not previous_kv_patch:
    raise RuntimeError("fixed-slot KV-manager finder identity changed")
'''.encode()


def render(args: argparse.Namespace) -> Mapping[str, Any]:
    destination = args.destination.expanduser().absolute()
    target_oracle = bool(getattr(args, "target_oracle", False))
    if destination.exists() or destination.is_symlink():
        raise FixedSlotPinnedReplayOverlayError("destination is create-only")
    for label, value in (
        ("base command", args.expected_base_sha256),
        ("scheduler runtime", args.expected_scheduler_sha256),
        ("KV-manager runtime", args.expected_kv_manager_sha256),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise FixedSlotPinnedReplayOverlayError(f"expected {label} SHA256 is malformed")

    base_payload = _stable_private_file(args.base_command, "base command")
    base_sha256 = _digest(base_payload)
    if base_sha256 != args.expected_base_sha256:
        raise FixedSlotPinnedReplayOverlayError("base command SHA256 mismatch")
    environment, argv = _split_command(base_payload)
    env = _environment_map(environment)
    if any(name in env for name in _OVERLAY_ENVIRONMENT):
        raise FixedSlotPinnedReplayOverlayError(
            "base command already contains the fixed-slot lookup repair"
        )
    _validate_base(env, argv, target_oracle=target_oracle)

    chained_pythonpath = env.get("PYTHONPATH", "")
    chained_site = _find_chained_site(chained_pythonpath)
    chained_payload = _stable_private_file(chained_site, "chained sitecustomize")
    chained_sha256 = _digest(chained_payload)
    chained_claims = _authenticate_chained_site(env, chained_site, chained_sha256)

    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent_metadata = destination.parent.lstat()
    if (
        not stat.S_ISDIR(parent_metadata.st_mode)
        or destination.parent.is_symlink()
        or parent_metadata.st_uid != os.getuid()
        or stat.S_IMODE(parent_metadata.st_mode) & 0o077
    ):
        raise FixedSlotPinnedReplayOverlayError(
            "destination parent must be one owned private directory"
        )
    staging = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FixedSlotPinnedReplayOverlayError("staging destination already exists")
    staging.mkdir(mode=0o700)
    try:
        site_payload = _site_source(
            destination,
            chained_site=chained_site,
            chained_site_sha256=chained_sha256,
            chained_pythonpath=chained_pythonpath,
            expected_runtime_sha256=args.expected_scheduler_sha256,
            expected_kv_runtime_sha256=args.expected_kv_manager_sha256,
        )
        site_sha256 = _digest(site_payload)
        additions = {
            ENABLE_ENV: "1",
            REQUIRED_ENV: "1",
            SITE_SHA_ENV: site_sha256,
            RUNTIME_SHA_ENV: args.expected_scheduler_sha256,
            KV_RUNTIME_SHA_ENV: args.expected_kv_manager_sha256,
        }
        rewritten_environment = [
            token
            for token in environment
            if token.split("=", 1)[0] not in {"PYTHONPATH", *additions}
        ]
        rewritten_environment.append(f"PYTHONPATH={destination}:{chained_pythonpath}")
        rewritten_environment.extend(f"{name}={value}" for name, value in additions.items())
        command_payload = (
            shlex.join(["exec", "/usr/bin/env", "-i", *rewritten_environment, *argv]) + "\n"
        ).encode()
        manifest = {
            "schema": SCHEMA,
            "classification": (
                "diagnostic_target_oracle_state_safety_repair_candidate"
                if target_oracle
                else "production_state_safety_repair_candidate"
            ),
            "promotable": False,
            "base_command": {
                "path": str(args.base_command.expanduser().absolute()),
                "sha256": base_sha256,
            },
            "chained_site": {
                "path": str(chained_site),
                "sha256": chained_sha256,
                "authenticated_by_environment": chained_claims,
            },
            "command_sha256": _digest(command_payload),
            "environment_delta": {
                "PYTHONPATH": f"{destination}:{chained_pythonpath}",
                **additions,
            },
            "patch_contract": {
                "module": TARGET_MODULE,
                "method": "OffloadingConnectorScheduler._lookup",
                "authorized_fixed_slot_restore": "unchanged",
                "resident_fixed_slot_restore": "unchanged",
                "unauthorized_generic_external_hit": 0,
                "partial_tail_lookup_without_authorized_boundary": False,
                "unauthenticated_generic_store_jobs": 0,
                "authenticated_capture_store_jobs": "unchanged",
                "authenticated_capture_owner": "selection-runtime directive or pending candidate",
                "stale_same_branch_resident_release": (
                    "after authenticated external lookup and before allocation"
                ),
            },
            "scheduler_runtime_sha256": args.expected_scheduler_sha256,
            "kv_manager_runtime_sha256": args.expected_kv_manager_sha256,
            "site_sha256": site_sha256,
            "target_oracle": target_oracle,
        }
        manifest_payload = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
        _write_exclusive(staging / "sitecustomize.py", site_payload, 0o600)
        _write_exclusive(staging / "command.sh", command_payload, 0o700)
        _write_exclusive(staging / "manifest.json", manifest_payload, 0o600)
        if destination.exists() or destination.is_symlink():
            raise FixedSlotPinnedReplayOverlayError(
                "destination appeared during create-only rendering"
            )
        staging.rename(destination)
        return {**manifest, "manifest_sha256": _digest(manifest_payload)}
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-command", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--expected-scheduler-sha256", required=True)
    parser.add_argument("--expected-kv-manager-sha256", required=True)
    parser.add_argument("--target-oracle", action="store_true")
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = render(args)
    except (OSError, RowExactOverlayError) as error:
        print(f"fixed-slot-pinned-replay-overlay: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
