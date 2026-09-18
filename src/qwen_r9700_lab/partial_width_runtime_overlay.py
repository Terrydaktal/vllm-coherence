"""Render an authenticated Hauhau runtime overlay for speculative widths 1..8."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "urn:qwen-r9700:hauhau-partial-width-runtime-overlay:v6"
RUNNER_MODULE = "vllm.v1.worker.gpu.model_runner"
GDN_MODULE = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
QUEST_MODULE = "quest_vllm_attention"
REJECTION_MODULE = "vllm.v1.worker.gpu.spec_decode.rejection_sampler"
BULK_MODULE = "gdn_transaction_bulk"
ENABLE_ENV = "QWEN_HAUHAU_PARTIAL_WIDTH_RUNTIME"
SITE_SHA_ENV = "QWEN_HAUHAU_PARTIAL_WIDTH_RUNTIME_SITE_SHA256"
RUNNER_SHA_ENV = "QWEN_HAUHAU_PARTIAL_WIDTH_RUNTIME_RUNNER_SHA256"
GDN_SHA_ENV = "QWEN_HAUHAU_PARTIAL_WIDTH_RUNTIME_GDN_SHA256"
QUEST_SHA_ENV = "QWEN_HAUHAU_PARTIAL_WIDTH_RUNTIME_QUEST_SHA256"
REJECTION_SHA_ENV = "QWEN_HAUHAU_PARTIAL_WIDTH_RUNTIME_REJECTION_SHA256"
BULK_SHA_ENV = "QWEN_HAUHAU_PARTIAL_WIDTH_RUNTIME_BULK_SHA256"

# The packed multi-token recurrent kernel was introduced as a launch-coalescing
# optimization.  Exact c5 forced-token replay proved that its published state
# is not serial-M1 equivalent after a rejected speculative suffix.  Keep the
# already-qualified private M8 convolution, but force recurrence verification
# and accepted-prefix publication through the existing rowwise M1 primitive.
# These values are part of this overlay's authenticated semantic contract.
SERIAL_EQUIVALENT_GDN_ENV: dict[str, str] = {
    "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED": "0",
    "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_VERIFY": "0",
    "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_COMMIT": "0",
    "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_VERIFY_RECURRENCE": "0",
}


# These transformations are deliberately anchored to the authenticated combined
# postimage.  The canonical semantic source gained partial-width support after the
# assurance artifact was rendered; applying only the derivative-model renderer
# therefore produced a mixed runner/GDN ABI.  Every preimage must occur exactly
# once, every postimage is checked, and the rendered module is compiled before an
# immutable candidate may be published.
GDN_PARTIAL_WIDTH_PATCHES: tuple[tuple[str, bytes, bytes], ...] = (
    (
        "verification-width-state",
        b"""        self._recoverssm_m8_cached_commit_ready = False
        self._recoverssm_trusted_scratch_state: torch.Tensor | None = None
""",
        b"""        self._recoverssm_m8_cached_commit_ready = False
        self._recoverssm_cached_verification_width = 0
        self._recoverssm_trusted_scratch_state: torch.Tensor | None = None
""",
    ),
    (
        "accepted-transaction-width",
        b"""        Verification records the serial-equivalent post-convolution operands and
        all eight tiny convolution-state snapshots without mutating canonical
        state.  After sampling, replay only the accepted recurrent transitions in
        private scratch, then publish recurrent and convolution state together.
        No W4, attention, MLP, or whole-model replay is required.
        \"\"\"

        if not _GDN_FIXED_SLOT_CACHED_ACCEPTED_COMMIT:
            raise RuntimeError(\"cached M8 accepted-path commit is disabled\")
        if self._recoverssm_cached_transaction_prepared:
            raise RuntimeError(\"cached M8 accepted transaction is already prepared\")
        if not 1 <= accepted <= self._recoverssm_spec_width:
            raise RuntimeError(f\"cached M8 accepted count is invalid: {accepted}\")
""",
        b"""        Verification records the serial-equivalent post-convolution operands and
        every convolution-state snapshot for the actual one-to-eight-row target
        batch without mutating canonical state.  After sampling, replay only the
        accepted recurrent transitions in private scratch, then publish recurrent
        and convolution state together.
        No W4, attention, MLP, or whole-model replay is required.
        \"\"\"

        if not _GDN_FIXED_SLOT_CACHED_ACCEPTED_COMMIT:
            raise RuntimeError(\"cached M8 accepted-path commit is disabled\")
        if self._recoverssm_cached_transaction_prepared:
            raise RuntimeError(\"cached M8 accepted transaction is already prepared\")
        verification_width = self._recoverssm_cached_verification_width
        if not 1 <= verification_width <= self._recoverssm_spec_width:
            raise RuntimeError(
                \"cached M8 accepted path has no valid verification width: \"
                f\"{verification_width}\"
            )
        if not 1 <= accepted <= verification_width:
            raise RuntimeError(f\"cached M8 accepted count is invalid: {accepted}\")
""",
    ),
    (
        "rollback-retires-width",
        b"""        self._recoverssm_cached_transaction_prepared = False
        self._recoverssm_cached_transaction_published = False

    def qwen_finalize_fixed_slot_cached_accepted_transaction(self) -> None:
""",
        b"""        self._recoverssm_cached_transaction_prepared = False
        self._recoverssm_cached_transaction_published = False
        self._recoverssm_m8_cached_commit_ready = False
        self._recoverssm_cached_verification_width = 0

    def qwen_finalize_fixed_slot_cached_accepted_transaction(self) -> None:
""",
    ),
    (
        "finalize-retires-width",
        b"""        self._recoverssm_cached_transaction_prepared = False
        self._recoverssm_cached_transaction_published = False
        self._recoverssm_m8_cached_commit_ready = False

    def qwen_commit_fixed_slot_cached_accepted_path(self, accepted: int) -> None:
""",
        b"""        self._recoverssm_cached_transaction_prepared = False
        self._recoverssm_cached_transaction_published = False
        self._recoverssm_m8_cached_commit_ready = False
        self._recoverssm_cached_verification_width = 0

    def qwen_commit_fixed_slot_cached_accepted_path(self, accepted: int) -> None:
""",
    ),
    (
        "partial-convolution-width",
        b"""        width = self._recoverssm_spec_width
        if (
            mixed_qkv.size(0) != width
            or state_indices.dim() != 2
""",
        b"""        spec_width = self._recoverssm_spec_width
        width = mixed_qkv.size(0)
        if (
            not 1 <= width <= spec_width
            or state_indices.dim() != 2
""",
    ),
    (
        "partial-convolution-diagnostic",
        b"fixed-slot target convolution requires one complete M8 request: ",
        b"fixed-slot target convolution requires one partial or complete M8 request: ",
    ),
    (
        "best-first-requires-m8",
        b"""        scratch_index = self._recoverssm_trusted_scratch_indices.reshape(-1)[:1]
        if _GDN_BEST_FIRST_B7:
            from qwen_r9700_lab.dflash_b7_device_runtime import get_b7_device_round
""",
        b"""        scratch_index = self._recoverssm_trusted_scratch_indices.reshape(-1)[:1]
        if _GDN_BEST_FIRST_B7:
            if width != spec_width:
                raise RuntimeError(\"best-first B7 convolution requires exact M8\")
            from qwen_r9700_lab.dflash_b7_device_runtime import get_b7_device_round
""",
    ),
    (
        "serial-convolution-requires-m8",
        b"        if _GDN_FIXED_SLOT_SERIAL_CONV_M8:\n",
        b"        if _GDN_FIXED_SLOT_SERIAL_CONV_M8 and width == spec_width:\n",
    ),
    (
        "serial-convolution-output-width",
        b"""                if self._recoverssm_m8_replay_selected_row is not None:
                    self._recoverssm_m8_replay_conv_ready = True
            return self._recoverssm_trusted_current_postconv_mixed_qkv
        if _GDN_FIXED_SLOT_SERIAL_BATCHED_COMMIT:
""",
        b"""                if self._recoverssm_m8_replay_selected_row is not None:
                    self._recoverssm_m8_replay_conv_ready = True
            return self._recoverssm_trusted_current_postconv_mixed_qkv[:width]
        if _GDN_FIXED_SLOT_SERIAL_BATCHED_COMMIT:
""",
    ),
    (
        "partial-preconvolution-cache",
        b"""            self._recoverssm_trusted_prev_preconv_mixed_qkv.copy_(mixed_qkv)
        if _GDN_FIXED_SLOT_SERIAL_BATCHED_VERIFY_CONV:
""",
        b"""            self._recoverssm_trusted_prev_preconv_mixed_qkv[:width].copy_(mixed_qkv)
        if _GDN_FIXED_SLOT_SERIAL_BATCHED_VERIFY_CONV and width == spec_width:
""",
    ),
    (
        "batched-convolution-output-width",
        b"""                out=self._recoverssm_trusted_current_postconv_mixed_qkv,
            )
            return self._recoverssm_trusted_current_postconv_mixed_qkv
        for row in range(width):
""",
        b"""                out=self._recoverssm_trusted_current_postconv_mixed_qkv,
            )
            return self._recoverssm_trusted_current_postconv_mixed_qkv[:width]
        for row in range(width):
""",
    ),
    (
        "row-convolution-output-width",
        b"""                if capture_replay:
                    self._recoverssm_m8_replay_conv_ready = True
        return self._recoverssm_trusted_current_postconv_mixed_qkv

    def _forward_core_decode_spec_best_first_b7(
""",
        b"""                if capture_replay:
                    self._recoverssm_m8_replay_conv_ready = True
        return self._recoverssm_trusted_current_postconv_mixed_qkv[:width]

    def _forward_core_decode_spec_best_first_b7(
""",
    ),
    (
        "packed-partial-width",
        b"""        \"\"\"Verify the current M8 in private ordered packed-M1 scratch.

        Accepted-path commit is deliberately absent from the authoritative
        verification pass.  After sampling, the runner re-executes the exact
        complete M8 target shape in private scratch and selects only the
        sampler-approved row for canonical recurrent and convolution state.
        \"\"\"
""",
        b"""        \"\"\"Verify the current partial or complete M8 in ordered packed-M1 scratch.

        Accepted-path commit is deliberately absent from the authoritative
        verification pass.  After sampling, the runner prepares the exact
        accepted prefix in private scratch and publishes only the sampler-approved
        recurrent and convolution state.
        \"\"\"
""",
    ),
    (
        "packed-partial-width-validation",
        b"""        width = self._recoverssm_spec_width
        state = self.kv_cache[1]
""",
        b"""        spec_width = self._recoverssm_spec_width
        width = mixed_qkv.size(0)
        if (
            not 1 <= width <= spec_width
            or a.size(0) != width
            or b.size(0) != width
            or output_gate.size(0) != width
            or core_attn_out.size(0) < width
        ):
            raise RuntimeError(
                \"fixed-slot packed-M1 received an invalid partial M8 width: \"
                f\"mixed={mixed_qkv.size(0)} a={a.size(0)} b={b.size(0)} \"
                f\"gate={output_gate.size(0)} out={core_attn_out.size(0)}\"
            )
        state = self.kv_cache[1]
""",
    ),
    (
        "cache-actual-verification-width",
        b"""        if _GDN_FIXED_SLOT_CACHED_ACCEPTED_COMMIT:
            self._recoverssm_trusted_prev_a.copy_(a)
            self._recoverssm_trusted_prev_b.copy_(b)
            self._recoverssm_m8_cached_commit_ready = True
""",
        b"""        if _GDN_FIXED_SLOT_CACHED_ACCEPTED_COMMIT:
            self._recoverssm_trusted_prev_a[:width].copy_(a)
            self._recoverssm_trusted_prev_b[:width].copy_(b)
            self._recoverssm_cached_verification_width = width
            self._recoverssm_m8_cached_commit_ready = True
""",
    ),
    (
        "trusted-replay-partial-width",
        b"""        width = self._recoverssm_spec_width
        if mixed_qkv.size(0) != width or cu_seqlens.numel() != 2:
            raise RuntimeError(
                \"trusted RecoverSSM replay requires one complete M8 request: \"
""",
        b"""        spec_width = self._recoverssm_spec_width
        width = mixed_qkv.size(0)
        if not 1 <= width <= spec_width or cu_seqlens.numel() != 2:
            raise RuntimeError(
                \"trusted RecoverSSM replay requires one partial or complete M8 request: \"
""",
    ),
    (
        "trusted-replay-cache-width",
        b"""        self._recoverssm_trusted_prev_q.copy_(query)
        self._recoverssm_trusted_prev_k.copy_(key)
        self._recoverssm_trusted_prev_v.copy_(value)
        self._recoverssm_trusted_prev_a.copy_(a)
        self._recoverssm_trusted_prev_b.copy_(b)
""",
        b"""        self._recoverssm_trusted_prev_q[:, :width].copy_(query)
        self._recoverssm_trusted_prev_k[:, :width].copy_(key)
        self._recoverssm_trusted_prev_v[:, :width].copy_(value)
        self._recoverssm_trusted_prev_a[:width].copy_(a)
        self._recoverssm_trusted_prev_b[:width].copy_(b)
""",
    ),
)

RUNNER_RELATIVE = Path(
    "release/files/runtime-tools/combined-postimages/vllm/v1/worker/gpu/model_runner.py"
)
GDN_RELATIVE = Path(
    "release/files/runtime-tools/combined-postimages/vllm/model_executor/layers/"
    "mamba/gdn/qwen_gdn_linear_attn.py"
)
QUEST_RELATIVE = Path(
    "release/files/runtime-tools/combined-postimages/vllm/v1/attention/backends/"
    "quest_vllm_attention.py"
)
REJECTION_RELATIVE = Path(
    "release/files/runtime-tools/combined-postimages/vllm/v1/worker/gpu/"
    "spec_decode/rejection_sampler.py"
)


class PartialWidthOverlayError(RuntimeError):
    """The input identity or create-only output contract was violated."""


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def stable_private(path: Path, label: str) -> bytes:
    path = path.expanduser().absolute()
    try:
        before = path.lstat()
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise PartialWidthOverlayError(f"cannot read {label}: {error}") from error

    def identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_uid,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
        )

    if (
        identity(before) != identity(after)
        or not stat.S_ISREG(before.st_mode)
        or path.is_symlink()
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) & 0o077
    ):
        raise PartialWidthOverlayError(f"{label} must be one stable owned private file")
    return payload


def stable_owned_source(path: Path, label: str) -> bytes:
    """Read a stable owned source file that may be world-readable but not writable."""

    path = path.expanduser().absolute()
    try:
        before = path.lstat()
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise PartialWidthOverlayError(f"cannot read {label}: {error}") from error

    def identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_uid,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
        )

    if (
        identity(before) != identity(after)
        or not stat.S_ISREG(before.st_mode)
        or path.is_symlink()
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) & 0o022
    ):
        raise PartialWidthOverlayError(
            f"{label} must be one stable owned non-publicly-writable file"
        )
    return payload


def split_command(payload: bytes) -> tuple[list[str], list[str]]:
    try:
        tokens = shlex.split(payload.decode())
    except (UnicodeDecodeError, ValueError) as error:
        raise PartialWidthOverlayError(f"base command is invalid: {error}") from error
    if tokens[:3] != ["exec", "/usr/bin/env", "-i"]:
        raise PartialWidthOverlayError("base command must use exec /usr/bin/env -i")
    environment: list[str] = []
    index = 3
    while index < len(tokens) and "=" in tokens[index]:
        environment.append(tokens[index])
        index += 1
    names = [value.split("=", 1)[0] for value in environment]
    if not environment or index == len(tokens) or len(names) != len(set(names)):
        raise PartialWidthOverlayError("base command environment is invalid")
    return environment, tokens[index:]


def environment_map(environment: Sequence[str]) -> dict[str, str]:
    return dict(value.split("=", 1) for value in environment)


def load_hauhau_renderer(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("_qwen_hauhau_renderer", path)
    if spec is None or spec.loader is None:
        raise PartialWidthOverlayError("cannot load Hauhau renderer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authenticate_artifact(root: Path) -> tuple[bytes, bytes, bytes, bytes, dict[str, Any]]:
    root = root.expanduser().absolute()
    manifest_payload = stable_private(root / "release/manifest.json", "release manifest")
    try:
        manifest = json.loads(manifest_payload)
    except json.JSONDecodeError as error:
        raise PartialWidthOverlayError("release manifest is invalid JSON") from error
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, list):
        raise PartialWidthOverlayError("release manifest lacks a file catalog")
    by_path = {
        entry.get("path"): entry
        for entry in files
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    outputs: list[bytes] = []
    for relative, label in (
        (RUNNER_RELATIVE, "combined model runner"),
        (GDN_RELATIVE, "combined GDN runtime"),
        (QUEST_RELATIVE, "combined Quest runtime"),
        (REJECTION_RELATIVE, "combined rejection sampler"),
    ):
        manifest_relative = str(relative.relative_to("release/files"))
        entry = by_path.get(manifest_relative)
        payload = stable_private(root / relative, label)
        if (
            not isinstance(entry, dict)
            or entry.get("sha256") != digest(payload)
            or entry.get("bytes") not in (None, len(payload))
        ):
            raise PartialWidthOverlayError(f"release manifest does not bind {label}")
        outputs.append(payload)
    identity = {
        "manifest_sha256": digest(manifest_payload),
        "semantic_source_sha256": manifest.get("semantic_source_sha256"),
        "tree_sha256": manifest.get("tree_sha256"),
    }
    if not all(isinstance(value, str) and len(value) == 64 for value in identity.values()):
        raise PartialWidthOverlayError("release manifest identity is incomplete")
    return outputs[0], outputs[1], outputs[2], outputs[3], identity


def partial_width_contract_sha256() -> str:
    payload = b"\0".join(
        label.encode() + b"\0" + before + b"\0" + after
        for label, before, after in GDN_PARTIAL_WIDTH_PATCHES
    )
    return digest(payload)


def patch_qwen_gdn_partial_widths(source: bytes) -> bytes:
    """Apply the complete width-1..8 GDN contract to one known postimage."""

    rendered = source
    for label, before, after in GDN_PARTIAL_WIDTH_PATCHES:
        before_count = rendered.count(before)
        after_count = rendered.count(after)
        if before_count != 1 or after_count != 0:
            raise PartialWidthOverlayError(
                f"GDN partial-width patch {label} has incompatible counts: "
                f"before={before_count} after={after_count}"
            )
        rendered = rendered.replace(before, after, 1)
    for label, before, after in GDN_PARTIAL_WIDTH_PATCHES:
        if rendered.count(before) != 0 or rendered.count(after) != 1:
            raise PartialWidthOverlayError(
                f"GDN partial-width patch {label} failed its postcondition"
            )
    try:
        compile(rendered.decode(), "qwen_gdn_linear_attn.py", "exec")
    except (SyntaxError, UnicodeDecodeError) as error:
        raise PartialWidthOverlayError(
            f"rendered partial-width GDN does not compile: {error}"
        ) from error
    return rendered


def site_source(
    destination: Path,
    *,
    chained_site: Path,
    chained_site_sha: str,
    chained_pythonpath: str,
    old_runner_sha: str,
    old_gdn_sha: str,
    old_rejection_source: Path,
    old_rejection_sha: str,
    runner_sha: str,
    runner_bytes: int,
    gdn_sha: str,
    gdn_bytes: int,
    gdn_contract_sha: str,
    quest_sha: str,
    quest_bytes: int,
    rejection_sha: str,
    rejection_bytes: int,
    bulk_source: Path,
    bulk_sha: str,
    bulk_bytes: int,
) -> bytes:
    outer_pythonpath = f"{destination}:{chained_pythonpath}"
    return f'''"""Authenticated Hauhau speculative-width runtime bootstrap."""
import functools
import hashlib
import importlib.abc
import importlib.util
import os
import runpy
import stat
import sys
from pathlib import Path

_ROOT = Path({str(destination)!r})
_SELF = _ROOT / "sitecustomize.py"
_RUNNER = _ROOT / "model_runner.py"
_GDN = _ROOT / "qwen_gdn_linear_attn.py"
_QUEST = _ROOT / "quest_vllm_attention.py"
_REJECTION = _ROOT / "rejection_sampler.py"
_OLD_REJECTION = Path({str(old_rejection_source)!r})
_BULK = Path({str(bulk_source)!r})
_CHAIN = Path({str(chained_site)!r})
_CHAIN_SHA = {chained_site_sha!r}
_CHAIN_PYTHONPATH = {chained_pythonpath!r}
_OUTER_PYTHONPATH = {outer_pythonpath!r}
_RUNNER_MODULE = {RUNNER_MODULE!r}
_GDN_MODULE = {GDN_MODULE!r}
_OLD_RUNNER_SHA = {old_runner_sha!r}
_OLD_GDN_SHA = {old_gdn_sha!r}
_RUNNER_SHA = {runner_sha!r}
_RUNNER_BYTES = {runner_bytes}
_GDN_SHA = {gdn_sha!r}
_GDN_BYTES = {gdn_bytes}
_GDN_CONTRACT_SHA = {gdn_contract_sha!r}
_QUEST_SHA = {quest_sha!r}
_QUEST_BYTES = {quest_bytes}
_REJECTION_SHA = {rejection_sha!r}
_REJECTION_BYTES = {rejection_bytes}
_OLD_REJECTION_SHA = {old_rejection_sha!r}
_BULK_SHA = {bulk_sha!r}
_BULK_BYTES = {bulk_bytes}
_RUNNER_SHA_ENVS = (
    "QWEN_LM_HEAD_DIRECT_M8_RUNNER_SHA256",
    "QWEN_LM_HEAD_W4_TOPK_RUNNER_SHA256",
)

def _stable(path, expected_sha, expected_bytes=None, public_read=False):
    before = path.lstat()
    payload = path.read_bytes()
    after = path.lstat()
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_mode, value.st_uid,
        value.st_nlink, value.st_size, value.st_mtime_ns,
    )
    if (
        identity(before) != identity(after)
        or not stat.S_ISREG(before.st_mode)
        or path.is_symlink()
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) & (0o022 if public_read else 0o077)
        or (expected_bytes is not None and len(payload) != expected_bytes)
        or hashlib.sha256(payload).hexdigest() != expected_sha
    ):
        raise RuntimeError("partial-width runtime identity differs")
    return payload

if os.environ.get({ENABLE_ENV!r}) != "1":
    raise RuntimeError("partial-width runtime identity is absent")
if os.environ.get("PYTHONPATH") != _OUTER_PYTHONPATH:
    raise RuntimeError("partial-width runtime PYTHONPATH differs")
_site_sha = os.environ.get({SITE_SHA_ENV!r}, "")
if len(_site_sha) != 64:
    raise RuntimeError("partial-width runtime site declaration differs")
_stable(_SELF, _site_sha)
_runner_payload = _stable(_RUNNER, _RUNNER_SHA, _RUNNER_BYTES)
_gdn_payload = _stable(_GDN, _GDN_SHA, _GDN_BYTES)
_stable(_QUEST, _QUEST_SHA, _QUEST_BYTES)
_rejection_payload = _stable(_REJECTION, _REJECTION_SHA, _REJECTION_BYTES)
_stable(_OLD_REJECTION, _OLD_REJECTION_SHA)
_stable(_BULK, _BULK_SHA, _BULK_BYTES, public_read=True)
_stable(_CHAIN, _CHAIN_SHA)
if os.environ.get({RUNNER_SHA_ENV!r}) != _RUNNER_SHA:
    raise RuntimeError("partial-width runner declaration differs")
if os.environ.get({GDN_SHA_ENV!r}) != _GDN_SHA:
    raise RuntimeError("partial-width GDN declaration differs")
if os.environ.get({QUEST_SHA_ENV!r}) != _QUEST_SHA:
    raise RuntimeError("partial-width Quest declaration differs")
if os.environ.get({REJECTION_SHA_ENV!r}) != _REJECTION_SHA:
    raise RuntimeError("partial-width rejection-sampler declaration differs")
if os.environ.get({BULK_SHA_ENV!r}) != _BULK_SHA:
    raise RuntimeError("partial-width bulk-transaction declaration differs")
if (
    _RUNNER_MODULE in sys.modules
    or _GDN_MODULE in sys.modules
    or {QUEST_MODULE!r} in sys.modules
    or {REJECTION_MODULE!r} in sys.modules
):
    raise RuntimeError("partial-width target module imported before bootstrap")

for name in _RUNNER_SHA_ENVS:
    inherited = os.environ.get(name)
    if inherited == _RUNNER_SHA:
        os.environ[name] = _OLD_RUNNER_SHA
os.environ["PYTHONPATH"] = _CHAIN_PYTHONPATH
try:
    runpy.run_path(str(_CHAIN), run_name="_qwen_partial_width_chained_site")
finally:
    os.environ["PYTHONPATH"] = _OUTER_PYTHONPATH
if (
    _RUNNER_MODULE in sys.modules
    or _GDN_MODULE in sys.modules
    or {QUEST_MODULE!r} in sys.modules
    or {REJECTION_MODULE!r} in sys.modules
):
    raise RuntimeError("partial-width target module imported during chained bootstrap")

bulk = sys.modules.get({BULK_MODULE!r})
if (
    bulk is None
    or Path(getattr(bulk, "__file__", "")) != _BULK
    or not callable(getattr(bulk, "_patch", None))
    or not callable(getattr(bulk, "_validate_runner", None))
    or not callable(getattr(bulk, "_build_coordinator", None))
):
    raise RuntimeError("authenticated bulk-transaction module identity differs")
previous_bulk_patch = bulk._patch

def _patch_bulk_partial_widths(module):
    previous_bulk_patch(module)
    cls = getattr(module, "GPUModelRunner", None)
    installed = getattr(cls, "_qwen_replay_sampled_accepted_path", None)
    original = getattr(installed, "__wrapped__", None)
    if not isinstance(cls, type) or not callable(installed) or not callable(original):
        raise RuntimeError("bulk-transaction replay wrapper ABI differs")

    # Remove only the authenticated bulk wrapper that ``previous_bulk_patch``
    # just installed.  Its underlying runner method already implements the
    # complete prepare/publish/rollback lifecycle and widths 1..8; with the
    # four serial-batched selectors disabled it invokes the target's exact
    # one-row recurrent primitive for every accepted row.  Do not wrap or
    # approximate that path again here.
    cls._qwen_replay_sampled_accepted_path = original
    module._QWEN_GDN_TRANSACTION_BULK_INSTALLED = False
    module._QWEN_GDN_TRANSACTION_PARTIAL_WIDTH_ABI = _GDN_CONTRACT_SHA
    module._QWEN_GDN_SERIAL_M1_ACCEPTED_COMMIT = True
    print(
        "[qwen-hauhau-partial-width-runtime] serial-M1 GDN widths 1..8 active "
        f"contract={{_GDN_CONTRACT_SHA}}",
        flush=True,
    )

bulk._patch = _patch_bulk_partial_widths

# Retarget the authenticated Hauhau runner finder in place so the existing
# direct-M8 and W4-topK meta-path wrappers remain ahead of it.
runner_matches = []
for finder in sys.meta_path:
    if finder.__class__.__name__ != "_RunnerFinder" or not hasattr(finder, "_payload"):
        continue
    globals_map = finder.find_spec.__globals__
    if globals_map.get("_RUNNER_MODULE") != _RUNNER_MODULE:
        continue
    if globals_map.get("_PATCHED_RUNNER_SHA256") != _OLD_RUNNER_SHA:
        continue
    runner_matches.append((finder, globals_map))
if len(runner_matches) != 1:
    raise RuntimeError("authenticated Hauhau runner finder preimage differs")
runner_finder, runner_globals = runner_matches[0]
runner_globals["_PATCHED_RUNNER"] = _RUNNER
runner_globals["_PATCHED_RUNNER_SHA256"] = _RUNNER_SHA
runner_globals["_PATCHED_RUNNER_BYTES"] = _RUNNER_BYTES
runner_finder._payload = _runner_payload

for module_name in ("lm_head_m8_direct", "lm_head_w4_topk"):
    binder = sys.modules.get(module_name)
    if binder is None or binder._RUNNER_SHA256 != _OLD_RUNNER_SHA:
        raise RuntimeError("LM-head binder preimage differs")
    binder._RUNNER_PATH = _RUNNER
    binder._RUNNER_SHA256 = _RUNNER_SHA
    os.environ[binder._RUNNER_SHA_ENV] = _RUNNER_SHA

# Authenticate exactly the Hauhau identity-loader record for the GDN source,
# then supersede it with a loader whose code objects name the corrected source.
# The module spec retains the historical origin for fixed-slot source-catalog
# compatibility; Triton inspection follows function co_filename to _GDN.
gdn_matches = []
for finder in sys.meta_path:
    records = getattr(finder, "_records", None)
    if finder.__class__.__name__ != "_IdentityModuleFinder" or type(records) is not dict:
        continue
    record = records.get(_GDN_MODULE)
    if type(record) is tuple and len(record) == 8 and record[5] == _OLD_GDN_SHA:
        gdn_matches.append((finder, records, record))
if len(gdn_matches) != 1:
    raise RuntimeError("authenticated Hauhau GDN finder preimage differs")
gdn_finder, records, record = gdn_matches[0]

class _PartialWidthSourceLoader(importlib.abc.Loader):
    def __init__(self, payload, source, origin, expected_sha, expected_bytes):
        self.payload = payload
        self.source = source
        self.origin = origin
        self.expected_sha = expected_sha
        self.expected_bytes = expected_bytes

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        exec(compile(self.payload, str(self.source), "exec"), module.__dict__)
        _stable(self.source, self.expected_sha, self.expected_bytes)
        if Path(module.__file__) != self.origin:
            raise RuntimeError("partial-width replacement module origin differs")
        if self.source == _GDN:
            cls = getattr(module, "QwenGatedDeltaNetAttention", None)
            method = getattr(cls, "_forward_core_decode_spec_fixed_slot_conv_m1", None)
            if (
                not isinstance(cls, type)
                or not callable(method)
                or Path(method.__code__.co_filename) != _GDN
                or b"one complete M8 request" in self.payload
                or b"not 1 <= width <= spec_width" not in self.payload
                or b"self._recoverssm_cached_verification_width = width" not in self.payload
            ):
                raise RuntimeError("partial-width GDN executable postcondition differs")
            module._QWEN_HAUHAU_PARTIAL_WIDTH_GDN_ABI = _GDN_CONTRACT_SHA
            print(
                "[qwen-hauhau-partial-width-runtime] corrected GDN source active "
                f"contract={{_GDN_CONTRACT_SHA}}",
                flush=True,
            )

class _PartialWidthSourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, replacements):
        self.replacements = replacements

    def find_spec(self, fullname, path=None, target=None):
        replacement = self.replacements.get(fullname)
        if replacement is None:
            return None
        payload, source, origin, expected_sha, expected_bytes = replacement
        return importlib.util.spec_from_file_location(
            fullname,
            origin,
            loader=_PartialWidthSourceLoader(
                payload, source, origin, expected_sha, expected_bytes
            ),
        )

import combined_runtime_patch as retained
finder_type = getattr(retained, "_Finder", None)
if not isinstance(finder_type, type):
    raise RuntimeError("combined runtime finder type differs")
combined_finders = [item for item in sys.meta_path if isinstance(item, finder_type)]
if len(combined_finders) != 1:
    raise RuntimeError("combined runtime finder identity differs")
previous_rejection_patch = finder_type._PATCHES.get({REJECTION_MODULE!r})
if not callable(previous_rejection_patch):
    raise RuntimeError("retained rejection-sampler patch identity differs")

def _patch_rejection_sampler_compatible(module):
    previous_rejection_patch(module)
    cls = getattr(module, "RejectionSampler", None)
    patched = getattr(cls, "try_qwen_greedy_m8", None)
    original = getattr(patched, "__wrapped__", None)
    if not isinstance(cls, type) or not callable(patched) or not callable(original):
        raise RuntimeError("retained fused-count sampler wrapper ABI differs")

    @functools.wraps(original)
    def compatible(
        self,
        target_argmax,
        input_batch,
        *,
        synthetic_decode_warmup=False,
    ):
        if synthetic_decode_warmup:
            return original(
                self,
                target_argmax,
                input_batch,
                synthetic_decode_warmup=True,
            )
        return patched(self, target_argmax, input_batch)

    cls.try_qwen_greedy_m8 = compatible
    module._QWEN_PARTIAL_WIDTH_REJECTION_ABI = True

finder_type._PATCHES[{REJECTION_MODULE!r}] = _patch_rejection_sampler_compatible
replacements = {{
    _GDN_MODULE: (_gdn_payload, _GDN, record[1], _GDN_SHA, _GDN_BYTES),
    {REJECTION_MODULE!r}: (
        _rejection_payload,
        _REJECTION,
        _OLD_REJECTION,
        _REJECTION_SHA,
        _REJECTION_BYTES,
    ),
}}
# The derivative identity finder is deliberately at index zero and already owns
# the GDN module.  Appending this replacement after the combined-runtime finder
# would authenticate the new file while Python continued to execute the old GDN
# payload.  Install immediately before the exact authenticated owner and prove
# that priority before allowing startup to continue.
gdn_index = sys.meta_path.index(gdn_finder)
replacement_finder = _PartialWidthSourceFinder(replacements)
sys.meta_path.insert(gdn_index, replacement_finder)
if (
    sys.meta_path.index(replacement_finder) + 1 != sys.meta_path.index(gdn_finder)
    or replacement_finder.find_spec(_GDN_MODULE) is None
):
    raise RuntimeError("partial-width GDN replacement priority differs")

_stable(_SELF, _site_sha)
_stable(_RUNNER, _RUNNER_SHA, _RUNNER_BYTES)
_stable(_GDN, _GDN_SHA, _GDN_BYTES)
_stable(_QUEST, _QUEST_SHA, _QUEST_BYTES)
_stable(_REJECTION, _REJECTION_SHA, _REJECTION_BYTES)
print("[qwen-hauhau-partial-width-runtime] authenticated widths 1..8 armed", flush=True)
'''.encode()


def write_exclusive(path: Path, payload: bytes, mode: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def render(args: argparse.Namespace) -> Mapping[str, Any]:
    destination = args.destination.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise PartialWidthOverlayError("destination is create-only")
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_base_sha256):
        raise PartialWidthOverlayError("expected base SHA256 is malformed")
    base_payload = stable_private(args.base_command, "base command")
    if digest(base_payload) != args.expected_base_sha256:
        raise PartialWidthOverlayError("base command SHA256 mismatch")
    environment, argv = split_command(base_payload)
    values = environment_map(environment)
    if ENABLE_ENV in values:
        raise PartialWidthOverlayError("base command already contains this overlay")
    if values.get("QWEN_FIXED_SLOT_TARGET_ONLY_ORACLE") != "0":
        raise PartialWidthOverlayError("base command is not the optimized M8 arm")
    if values.get("QWEN_DFLASH_GREEDY_LOOP_ESCAPE") != "0":
        raise PartialWidthOverlayError("qualification requires loop escape disabled")
    pythonpath = values.get("PYTHONPATH", "")
    first_root = Path(pythonpath.split(":", 1)[0])
    chained_site = first_root / "sitecustomize.py"
    chained_payload = stable_private(chained_site, "chained site")
    chained_sha = digest(chained_payload)
    site_claims = [
        name
        for name, value in values.items()
        if name.endswith("SITE_SHA256") and value == chained_sha
    ]
    if not site_claims:
        raise PartialWidthOverlayError("base environment does not authenticate chained site")

    (
        source_runner,
        source_gdn,
        source_quest,
        source_rejection,
        artifact_identity,
    ) = authenticate_artifact(args.artifact_root)
    renderer = load_hauhau_renderer(args.hauhau_renderer)
    runner = renderer.BASE.patch_model_runner(source_runner)
    derivative_gdn = renderer.BASE.patch_qwen_gdn(source_gdn)
    gdn = patch_qwen_gdn_partial_widths(derivative_gdn)
    quest = source_quest
    old_runner_sha = args.expected_old_runner_sha256
    old_gdn_payload = stable_private(args.old_gdn_source, "Hauhau GDN preimage")
    old_gdn_sha = digest(old_gdn_payload)
    if not re.fullmatch(r"[0-9a-f]{64}", old_runner_sha):
        raise PartialWidthOverlayError("expected old runner SHA256 is malformed")
    if old_gdn_sha != args.expected_old_gdn_sha256:
        raise PartialWidthOverlayError("Hauhau GDN preimage SHA256 mismatch")
    old_rejection_payload = stable_private(
        args.old_rejection_sampler_source, "installed rejection-sampler preimage"
    )
    old_rejection_sha = digest(old_rejection_payload)
    if old_rejection_sha != args.expected_old_rejection_sampler_sha256:
        raise PartialWidthOverlayError("rejection-sampler preimage SHA256 mismatch")
    bulk_payload = stable_owned_source(
        args.gdn_transaction_bulk_source, "GDN bulk-transaction source"
    )
    bulk_sha = digest(bulk_payload)
    if bulk_sha != args.expected_gdn_transaction_bulk_sha256:
        raise PartialWidthOverlayError("GDN bulk-transaction source SHA256 mismatch")

    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent = destination.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or destination.parent.is_symlink()
        or parent.st_uid != os.getuid()
        or stat.S_IMODE(parent.st_mode) & 0o077
    ):
        raise PartialWidthOverlayError("destination parent must be owned and private")
    staging = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
    staging.mkdir(mode=0o700)
    try:
        site = site_source(
            destination,
            chained_site=chained_site,
            chained_site_sha=chained_sha,
            chained_pythonpath=pythonpath,
            old_runner_sha=old_runner_sha,
            old_gdn_sha=old_gdn_sha,
            old_rejection_source=args.old_rejection_sampler_source.expanduser().absolute(),
            old_rejection_sha=old_rejection_sha,
            runner_sha=digest(runner),
            runner_bytes=len(runner),
            gdn_sha=digest(gdn),
            gdn_bytes=len(gdn),
            gdn_contract_sha=partial_width_contract_sha256(),
            quest_sha=digest(quest),
            quest_bytes=len(quest),
            rejection_sha=digest(source_rejection),
            rejection_bytes=len(source_rejection),
            bulk_source=args.gdn_transaction_bulk_source.expanduser().absolute(),
            bulk_sha=bulk_sha,
            bulk_bytes=len(bulk_payload),
        )
        try:
            compile(site.decode(), "sitecustomize.py", "exec")
        except (SyntaxError, UnicodeDecodeError) as error:
            raise PartialWidthOverlayError(
                f"rendered partial-width bootstrap does not compile: {error}"
            ) from error
        additions = {
            ENABLE_ENV: "1",
            SITE_SHA_ENV: digest(site),
            RUNNER_SHA_ENV: digest(runner),
            GDN_SHA_ENV: digest(gdn),
            QUEST_SHA_ENV: digest(quest),
            REJECTION_SHA_ENV: digest(source_rejection),
            BULK_SHA_ENV: bulk_sha,
            **SERIAL_EQUIVALENT_GDN_ENV,
        }
        rewritten: list[str] = []
        for token in environment:
            name, value = token.split("=", 1)
            if name == "PYTHONPATH":
                value = f"{destination}:{value}"
            if name in SERIAL_EQUIVALENT_GDN_ENV:
                value = SERIAL_EQUIVALENT_GDN_ENV[name]
            rewritten.append(f"{name}={value}")
        rewritten_names = {token.split("=", 1)[0] for token in rewritten}
        rewritten.extend(
            f"{name}={value}"
            for name, value in additions.items()
            if name not in rewritten_names
        )
        command = (shlex.join(["exec", "/usr/bin/env", "-i", *rewritten, *argv]) + "\n").encode()
        manifest = {
            "schema": SCHEMA,
            "classification": "non_promotable_pending_gpu_qualification",
            "promotable": False,
            "artifact": artifact_identity,
            "base_command": {
                "path": str(args.base_command.expanduser().absolute()),
                "sha256": digest(base_payload),
            },
            "chained_site": {
                "path": str(chained_site),
                "sha256": chained_sha,
                "claims": site_claims,
            },
            "runtime": {
                "model_runner": {"bytes": len(runner), "sha256": digest(runner)},
                "qwen_gdn": {"bytes": len(gdn), "sha256": digest(gdn)},
                "quest": {"bytes": len(quest), "sha256": digest(quest)},
                "rejection_sampler": {
                    "bytes": len(source_rejection),
                    "sha256": digest(source_rejection),
                },
                "gdn_transaction_bulk": {
                    "bytes": len(bulk_payload),
                    "sha256": bulk_sha,
                },
            },
            "gdn_partial_width_contract_sha256": partial_width_contract_sha256(),
            "gdn_accepted_commit": {
                "bulk_transaction_active": False,
                "mode": "serial_m1_rowwise_private_prepare_atomic_publish",
                "environment": SERIAL_EQUIVALENT_GDN_ENV,
            },
            "supported_verification_widths": list(range(1, 9)),
            "site_sha256": digest(site),
            "command_sha256": digest(command),
        }
        write_exclusive(staging / "model_runner.py", runner, 0o600)
        write_exclusive(staging / "qwen_gdn_linear_attn.py", gdn, 0o600)
        write_exclusive(staging / "quest_vllm_attention.py", quest, 0o600)
        write_exclusive(staging / "rejection_sampler.py", source_rejection, 0o600)
        write_exclusive(staging / "sitecustomize.py", site, 0o600)
        write_exclusive(staging / "command.sh", command, 0o700)
        manifest_payload = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
        write_exclusive(staging / "manifest.json", manifest_payload, 0o600)
        staging.rename(destination)
        return {**manifest, "manifest_sha256": digest(manifest_payload)}
    except BaseException:
        if staging.exists() and staging.parent == destination.parent:
            shutil.rmtree(staging)
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base-command", type=Path, required=True)
    result.add_argument("--expected-base-sha256", required=True)
    result.add_argument("--artifact-root", type=Path, required=True)
    result.add_argument("--hauhau-renderer", type=Path, required=True)
    result.add_argument("--expected-old-runner-sha256", required=True)
    result.add_argument("--old-gdn-source", type=Path, required=True)
    result.add_argument("--expected-old-gdn-sha256", required=True)
    result.add_argument("--old-rejection-sampler-source", type=Path, required=True)
    result.add_argument("--expected-old-rejection-sampler-sha256", required=True)
    result.add_argument("--gdn-transaction-bulk-source", type=Path, required=True)
    result.add_argument("--expected-gdn-transaction-bulk-sha256", required=True)
    result.add_argument("--destination", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = render(parser().parse_args(argv))
    except (OSError, KeyError, PartialWidthOverlayError) as error:
        print(f"qwen-partial-width-runtime-overlay: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
