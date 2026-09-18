"""Render a qualification-only Hauhau BF16 oracle transport overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "urn:qwen-r9700:hauhau-bf16-oracle-transport:v1"
RUNNER_SHA256 = "18fc8aebc235a9ea4af0459393724c5ff941e93b6a78fc8d9e1bce22a52cd196"
RUNNER_MODULE = "vllm.v1.worker.gpu.model_runner"
ENV_NAME = "QWEN_HAUHAU_BF16_ORACLE_TRANSPORT"
TARGET_ONLY_ENV = "QWEN_HAUHAU_BF16_TARGET_ONLY_ORACLE"
ISOLATED_SUFFIX = "-bf16-target-oracle-v1"

HELPER_OLD = b'''def _qwen_fixed_slot_target_only_oracle_enabled(vllm_config: VllmConfig) -> bool:
    """Read the explicit API-to-EngineCore configuration transport."""
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None:
        return False
    extra_config = kv_transfer_config.kv_connector_extra_config
    if not isinstance(extra_config, dict):
        return False
    enabled = extra_config.get("fixed_slot_target_only_oracle", False)
    if not isinstance(enabled, bool):
        raise RuntimeError(
            "fixed_slot_target_only_oracle transport must be a JSON boolean"
        )
    return enabled
'''

HELPER_NEW = b'''def _qwen_fixed_slot_target_only_oracle_enabled(vllm_config: VllmConfig) -> bool:
    """Read the explicit API-to-EngineCore configuration transport."""
    bf16_oracle = os.environ.get("QWEN_HAUHAU_BF16_TARGET_ONLY_ORACLE", "0")
    if bf16_oracle not in ("0", "1"):
        raise RuntimeError("BF16 target-only oracle transport must be 0 or 1")
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None:
        if bf16_oracle == "1" and vllm_config.cache_config.cache_dtype != "bfloat16":
            raise RuntimeError("connector-free target-only oracle requires BF16 KV")
        return bf16_oracle == "1"
    if bf16_oracle != "0":
        raise RuntimeError("BF16 target-only oracle forbids a KV connector")
    extra_config = kv_transfer_config.kv_connector_extra_config
    if not isinstance(extra_config, dict):
        return False
    enabled = extra_config.get("fixed_slot_target_only_oracle", False)
    if not isinstance(enabled, bool):
        raise RuntimeError(
            "fixed_slot_target_only_oracle transport must be a JSON boolean"
        )
    return enabled
'''

TARGET_VALIDATE_OLD = b'''        if self._qwen_fixed_slot_target_only_oracle:
            kv_transfer_config = self.vllm_config.kv_transfer_config
            assert kv_transfer_config is not None
            extra_config = kv_transfer_config.kv_connector_extra_config
            assert isinstance(extra_config, dict)
            secondary_tiers = extra_config.get("secondary_tiers")
            corrected_generation = extra_config.get(
                "fixed_slot_corrected_generation", False
            )
            if not isinstance(corrected_generation, bool):
                raise RuntimeError(
                    "fixed_slot_corrected_generation transport must be a JSON boolean"
                )
            expected_root = (
                "/home/lewis/.local/share/qwen-r9700/kv-cache/"
                "dflash-agent262-hauhau-delta-autoround-v1"
                if corrected_generation
                else "/home/lewis/.local/share/qwen-r9700/kv-cache/"
                "dflash-agent262-fixed-slot-v1"
            )
            if (
                kv_transfer_config.kv_connector != "OffloadingConnector"
                or extra_config.get("spec_name") != "TieringOffloadingSpec"
                or not isinstance(secondary_tiers, list)
                or len(secondary_tiers) != 1
                or secondary_tiers[0].get("type") != "fs"
                or secondary_tiers[0].get("root_dir") != expected_root
            ):
                raise RuntimeError(
                    "fixed-slot target-only oracle requires the exact isolated "
                    "OffloadingConnector transport"
                )
'''

TARGET_VALIDATE_NEW = b'''        if self._qwen_fixed_slot_target_only_oracle:
            kv_transfer_config = self.vllm_config.kv_transfer_config
            bf16_oracle = os.environ.get(
                "QWEN_HAUHAU_BF16_TARGET_ONLY_ORACLE", "0"
            ) == "1"
            if bf16_oracle:
                if (
                    kv_transfer_config is not None
                    or self.cache_config.cache_dtype != "bfloat16"
                    or os.environ.get("QWEN_FIXED_SLOT_SNAPSHOT_EXPORT") != "0"
                ):
                    raise RuntimeError(
                        "BF16 target-only oracle requires connector-free fresh-only BF16 KV"
                    )
                extra_config = {}
            else:
                assert kv_transfer_config is not None
                extra_config = kv_transfer_config.kv_connector_extra_config
                assert isinstance(extra_config, dict)
                secondary_tiers = extra_config.get("secondary_tiers")
                corrected_generation = extra_config.get(
                    "fixed_slot_corrected_generation", False
                )
                if not isinstance(corrected_generation, bool):
                    raise RuntimeError(
                        "fixed_slot_corrected_generation transport must be a JSON boolean"
                    )
                expected_root = (
                    "/home/lewis/.local/share/qwen-r9700/kv-cache/"
                    "dflash-agent262-hauhau-delta-autoround-v1"
                    if corrected_generation
                    else "/home/lewis/.local/share/qwen-r9700/kv-cache/"
                    "dflash-agent262-fixed-slot-v1"
                )
                if (
                    kv_transfer_config.kv_connector != "OffloadingConnector"
                    or extra_config.get("spec_name") != "TieringOffloadingSpec"
                    or not isinstance(secondary_tiers, list)
                    or len(secondary_tiers) != 1
                    or secondary_tiers[0].get("type") != "fs"
                    or secondary_tiers[0].get("root_dir") != expected_root
                ):
                    raise RuntimeError(
                        "fixed-slot target-only oracle requires the exact isolated "
                        "OffloadingConnector transport"
                    )
'''

TARGET_OLD = b'''            expected_root = (
                "/home/lewis/.local/share/qwen-r9700/kv-cache/"
                "dflash-agent262-hauhau-delta-autoround-v1"
                if corrected_generation
                else "/home/lewis/.local/share/qwen-r9700/kv-cache/"
                "dflash-agent262-fixed-slot-v1"
            )
'''

TARGET_NEW = b'''            bf16_oracle_transport = self.cache_config.cache_dtype == "bfloat16"
            if bf16_oracle_transport and (
                not corrected_generation
                or os.environ.get("QWEN_FIXED_SLOT_SNAPSHOT_EXPORT") != "0"
            ):
                raise RuntimeError(
                    "BF16 target-only oracle requires corrected generation and disabled "
                    "snapshot export"
                )
            expected_root = (
                "/home/lewis/.local/share/qwen-r9700/kv-cache/"
                "dflash-agent262-hauhau-delta-autoround-v1"
                + ("-bf16-target-oracle-v1" if bf16_oracle_transport else "")
                if corrected_generation
                else "/home/lewis/.local/share/qwen-r9700/kv-cache/"
                "dflash-agent262-fixed-slot-v1"
            )
'''

WHOLE_OLD = b'''            spec_config = self.speculative_config
            if (
                self._qwen_fixed_slot_target_only_oracle
'''

WHOLE_NEW = b'''            spec_config = self.speculative_config
            bf16_oracle_transport = self.cache_config.cache_dtype == "bfloat16"
            if bf16_oracle_transport and os.environ.get(
                "QWEN_FIXED_SLOT_SNAPSHOT_EXPORT"
            ) != "0":
                raise RuntimeError(
                    "BF16 whole-model oracle requires disabled snapshot export"
                )
            expected_whole_model_root = (
                "/home/lewis/.local/share/qwen-r9700/kv-cache/"
                "dflash-agent262-hauhau-delta-autoround-v1"
                + ("-bf16-target-oracle-v1" if bf16_oracle_transport else "")
            )
            if (
                self._qwen_fixed_slot_target_only_oracle
'''

WHOLE_ROOT_OLD = b'''                or secondary_tiers[0].get("root_dir")
                != "/home/lewis/.local/share/qwen-r9700/kv-cache/"
                "dflash-agent262-hauhau-delta-autoround-v1"
'''

WHOLE_ROOT_NEW = b'''                or secondary_tiers[0].get("root_dir")
                != expected_whole_model_root
'''


class TransportOverlayError(RuntimeError):
    """The source, command, or destination violated the oracle contract."""


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def rewrite_runner(source: bytes) -> bytes:
    """Admit only the create-only BF16 oracle suffix in the two oracle paths."""

    if digest(source) != RUNNER_SHA256:
        raise TransportOverlayError("model-runner source SHA256 differs")
    replacements = (
        (HELPER_OLD, HELPER_NEW, "target-only transport helper"),
        (TARGET_VALIDATE_OLD, TARGET_VALIDATE_NEW, "target-only transport validation"),
        (WHOLE_OLD, WHOLE_NEW, "whole-model transport setup"),
        (WHOLE_ROOT_OLD, WHOLE_ROOT_NEW, "whole-model transport root"),
    )
    rewritten = source
    for old, new, label in replacements:
        if rewritten.count(old) != 1:
            raise TransportOverlayError(f"{label} preimage is not unique")
        rewritten = rewritten.replace(old, new)
    if any(old in rewritten for old, _new, _label in replacements):
        raise TransportOverlayError("model-runner transport postimage is invalid")
    compile(rewritten, "model_runner.py", "exec")
    return rewritten


def _split_command(payload: bytes) -> tuple[list[str], list[str]]:
    try:
        tokens = shlex.split(payload.decode())
    except (UnicodeDecodeError, ValueError) as error:
        raise TransportOverlayError(f"base command is invalid: {error}") from error
    if tokens[:3] != ["exec", "/usr/bin/env", "-i"]:
        raise TransportOverlayError("base command must use exec /usr/bin/env -i")
    index = 3
    environment: list[str] = []
    while index < len(tokens) and "=" in tokens[index]:
        environment.append(tokens[index])
        index += 1
    if not environment or index == len(tokens):
        raise TransportOverlayError("base command lacks environment or server argv")
    return environment, tokens[index:]


def _environment_map(environment: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in environment:
        name, value = token.split("=", 1)
        if not name or name in result:
            raise TransportOverlayError("base command has an invalid environment")
        result[name] = value
    return result


def _option(argv: Sequence[str], name: str) -> str:
    if argv.count(name) != 1:
        raise TransportOverlayError(f"base command must contain exactly one {name}")
    index = argv.index(name)
    if index + 1 == len(argv):
        raise TransportOverlayError(f"base command lacks the {name} value")
    return argv[index + 1]


def validate_bf16_command(environment: Mapping[str, str], argv: Sequence[str]) -> None:
    if _option(argv, "--kv-cache-dtype") != "bfloat16":
        raise TransportOverlayError("oracle command is not a BF16-KV arm")
    if environment.get("QWEN_FIXED_SLOT_SNAPSHOT_EXPORT") != "0":
        raise TransportOverlayError("BF16 oracle must disable snapshot export")
    if not environment.get("QWEN_FIXED_SLOT_SNAPSHOT_ROOT", "").endswith(ISOLATED_SUFFIX):
        raise TransportOverlayError("BF16 oracle snapshot root is not isolated")
    if "--kv-transfer-config" in argv:
        raise TransportOverlayError("BF16 oracle must not configure a KV connector")
    if "--enable-prefix-caching" in argv:
        raise TransportOverlayError("BF16 oracle must not enable prefix caching")
    if argv.count("--no-enable-prefix-caching") != 1:
        raise TransportOverlayError("BF16 oracle must explicitly disable prefix caching")
    if environment.get(TARGET_ONLY_ENV) not in ("0", "1"):
        raise TransportOverlayError("BF16 target-only identity must be 0 or 1")


def render_site(
    destination: Path,
    *,
    runner_sha256: str,
    runner_bytes: int,
    chained_site: Path,
    chained_site_sha256: str,
    chained_pythonpath: str,
) -> bytes:
    outer_pythonpath = f"{destination}:{chained_pythonpath}"
    return f'''"""Authenticated qualification-only Hauhau BF16 transport overlay."""
import hashlib
import importlib.machinery
import os
import runpy
import stat
import sys
from pathlib import Path

_ROOT = Path({str(destination)!r})
_SELF = _ROOT / "sitecustomize.py"
_RUNNER = _ROOT / "model_runner.py"
_RUNNER_SHA256 = {runner_sha256!r}
_RUNNER_BYTES = {runner_bytes!r}
_CHAIN = Path({str(chained_site)!r})
_CHAIN_SHA256 = {chained_site_sha256!r}
_CHAIN_PYTHONPATH = {chained_pythonpath!r}
_OUTER_PYTHONPATH = {outer_pythonpath!r}
_RUNNER_MODULE = {RUNNER_MODULE!r}
_INSTALLED_RUNNER_SHA256 = "62b160b5f043d697adcce7b9da71cf750f0d4cb519e08cb200c2fedf8eb30360"
_HAUHAU_RUNNER_SHA256 = {RUNNER_SHA256!r}
_RUNNER_SHA_ENVS = (
    "QWEN_LM_HEAD_DIRECT_M8_RUNNER_SHA256",
    "QWEN_LM_HEAD_W4_TOPK_RUNNER_SHA256",
)
_BF16_RUNTIME_ZERO_ENVS = (
    "QWEN_DFLASH_RAW_FILL_MODE_NONE",
    "QWEN_GDN_METADATA_M8_FAST",
    "QWEN_GDN_METADATA_M8_FAST_REQUIRED",
    "QWEN_GDN_TRANSACTION_BULK",
    "QWEN_GDN_FIXED_SLOT_CACHED_ACCEPTED_COMMIT",
    "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED",
    "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_COMMIT",
    "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_VERIFY",
    "QWEN_GDN_FIXED_SLOT_SERIAL_BATCHED_VERIFY_RECURRENCE",
    "QWEN_GDN_FIXED_SLOT_SERIAL_CONV_M8",
    "QWEN_GDN_RECOVERSSM",
    "QWEN_GDN_RECOVERSSM_REQUIRED",
    "QWEN_GDN_RECOVERSSM_TRUSTED_REPLAY",
    "QWEN_GDN_RECOVERSSM_FIXED_SLOT_TRUSTED_REPLAY",
    "QWEN_LM_HEAD_PREFIX_SERIAL_M8",
)
_BF16_REFERENCE_RUNTIME_ENVS = {{
    "QWEN_DFLASH_REFERENCE_GDN": "1",
    "QWEN_DFLASH_TARGET_KV_CACHE_DTYPE": "auto",
}}

def _stable_payload(path, expected_sha, expected_bytes=None):
    before = path.lstat()
    payload = path.read_bytes()
    after = path.lstat()
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_mode, value.st_uid,
        value.st_nlink, value.st_size, value.st_mtime_ns, value.st_ctime_ns,
    )
    if (
        identity(before) != identity(after)
        or not stat.S_ISREG(before.st_mode)
        or path.is_symlink()
        or before.st_uid != os.getuid()
        or before.st_nlink != 1
        or before.st_mode & 0o022
        or (expected_bytes is not None and len(payload) != expected_bytes)
        or hashlib.sha256(payload).hexdigest() != expected_sha
    ):
        raise RuntimeError("BF16 oracle transport file identity differs")
    return payload

if os.environ.get({ENV_NAME!r}) != "1":
    raise RuntimeError("BF16 oracle transport is not required")
if os.environ.get("PYTHONPATH") != _OUTER_PYTHONPATH:
    raise RuntimeError("BF16 oracle transport PYTHONPATH differs")
_declared_site = os.environ.get("QWEN_HAUHAU_BF16_ORACLE_TRANSPORT_SITE_SHA256", "")
_site_payload = _stable_payload(_SELF, _declared_site)
_runner_payload = _stable_payload(_RUNNER, _RUNNER_SHA256, _RUNNER_BYTES)
_stable_payload(_CHAIN, _CHAIN_SHA256)
if os.environ.get("QWEN_FIXED_SLOT_SNAPSHOT_EXPORT") != "0":
    raise RuntimeError("BF16 oracle transport requires disabled snapshot export")
if _RUNNER_MODULE in sys.modules:
    raise RuntimeError("model runner imported before BF16 oracle transport bootstrap")

for _name in _RUNNER_SHA_ENVS:
    if os.environ.get(_name) not in (
        _INSTALLED_RUNNER_SHA256, _HAUHAU_RUNNER_SHA256, _RUNNER_SHA256
    ):
        raise RuntimeError(f"BF16 oracle inherited {{_name}} identity differs")
    # A spawned EngineCore inherits the outer digest from its APIServer. Give
    # the immutable inner Hauhau layer its exact preimage for authentication;
    # this process is retargeted to the BF16 shadow again below.
    os.environ[_name] = _HAUHAU_RUNNER_SHA256
for _name in _BF16_RUNTIME_ZERO_ENVS:
    if os.environ.get(_name) not in ("0", "1"):
        raise RuntimeError(f"BF16 oracle inherited {{_name}} value differs")
    os.environ[_name] = "1"
for _name, _value in _BF16_REFERENCE_RUNTIME_ENVS.items():
    if os.environ.get(_name) not in (None, _value):
        raise RuntimeError(f"BF16 oracle inherited {{_name}} value differs")
    os.environ.pop(_name, None)
os.environ["PYTHONPATH"] = _CHAIN_PYTHONPATH
try:
    _chain_globals = runpy.run_path(
        str(_CHAIN), run_name="_qwen_hauhau_bf16_transport_chained_site"
    )
finally:
    os.environ["PYTHONPATH"] = _OUTER_PYTHONPATH
for _name in _BF16_RUNTIME_ZERO_ENVS:
    os.environ[_name] = "0"
for _name, _value in _BF16_REFERENCE_RUNTIME_ENVS.items():
    os.environ[_name] = _value
if _RUNNER_MODULE in sys.modules:
    raise RuntimeError("model runner imported during BF16 oracle transport bootstrap")

# The production metadata shortcut is authenticated above, but its exact
# one-column state-table contract is specific to the FP8/mode-none lane. Use
# the wrapped stock builder for this BF16 reference arm instead of weakening
# the production shortcut's fail-closed checks.
from vllm.v1.attention.backends import gdn_attn as _gdn_attn
_gdn_builder = _gdn_attn.GDNAttentionMetadataBuilder
_gdn_fast_build = _gdn_builder.build
_gdn_fast_source = Path(
    "/home/lewis/projects/qwen-r9700/artifacts/qualifications/"
    "20260831T-gdn-metadata-m8-fast-v2-create-only/gdn_metadata_m8_fast.py"
)
if (
    not getattr(_gdn_attn, "_QWEN_GDN_METADATA_M8_FAST_PATCHED", False)
    or not hasattr(_gdn_fast_build, "__wrapped__")
    or Path(_gdn_fast_build.__code__.co_filename).resolve() != _gdn_fast_source
    or _gdn_fast_build is _gdn_fast_build.__wrapped__
):
    raise RuntimeError("authenticated GDN metadata shortcut preimage differs")
_gdn_builder.build = _gdn_fast_build.__wrapped__
_gdn_attn._QWEN_GDN_METADATA_M8_FAST_PATCHED = False
_gdn_attn._QWEN_GDN_METADATA_M8_FAST_BF16_ORACLE_DISABLED = True

# The bulk accepted-path transaction is likewise specific to fixed-slot
# replay. Its authenticated finder has not imported the model runner yet, so
# remove exactly that deferred patch before loading the BF16 shadow.
_bulk_finders = [
    _finder
    for _finder in sys.meta_path
    if _finder.__class__.__name__ == "_Finder"
    and _finder.__class__.__module__ == "gdn_transaction_bulk"
]
if len(_bulk_finders) != 1:
    raise RuntimeError("authenticated bulk GDN finder preimage differs")
sys.meta_path.remove(_bulk_finders[0])

_old_runner = _chain_globals.get("_PATCHED_RUNNER")
_old_sha = _chain_globals.get("_PATCHED_RUNNER_SHA256")
_old_bytes = _chain_globals.get("_PATCHED_RUNNER_BYTES")
if (
    not isinstance(_old_runner, Path)
    or _old_sha != {RUNNER_SHA256!r}
    or _old_bytes != 119809
):
    raise RuntimeError("Hauhau chained runner identity differs")

# Retarget the authenticated Hauhau loader rather than bypassing it. This keeps
# the direct-M8 and W4-topK loader wrappers in their original order.
_chain_globals["_PATCHED_RUNNER"] = _RUNNER
_chain_globals["_PATCHED_RUNNER_SHA256"] = _RUNNER_SHA256
_chain_globals["_PATCHED_RUNNER_BYTES"] = _RUNNER_BYTES
_retargeted = 0
for _finder in sys.meta_path:
    if (
        _finder.__class__.__name__ != "_RunnerFinder"
        or _finder.__class__.__module__
        != "_qwen_hauhau_bf16_transport_chained_site"
    ):
        continue
    _globals = _finder.find_spec.__globals__
    if (
        _globals.get("_PATCHED_RUNNER") != _old_runner
        or _globals.get("_PATCHED_RUNNER_SHA256") != _old_sha
        or _globals.get("_PATCHED_RUNNER_BYTES") != _old_bytes
        or not hasattr(_finder, "_payload")
    ):
        raise RuntimeError("Hauhau model-runner finder preimage differs")
    _globals["_PATCHED_RUNNER"] = _RUNNER
    _globals["_PATCHED_RUNNER_SHA256"] = _RUNNER_SHA256
    _globals["_PATCHED_RUNNER_BYTES"] = _RUNNER_BYTES
    _finder._payload = _runner_payload
    _retargeted += 1
if _retargeted != 1:
    raise RuntimeError("Hauhau model-runner finder retarget count differs")

for _module_name in ("lm_head_m8_direct", "lm_head_w4_topk"):
    _module = sys.modules.get(_module_name)
    if _module is None:
        raise RuntimeError(f"{{_module_name}} binder is absent")
    _module._RUNNER_PATH = _RUNNER
    _module._RUNNER_SHA256 = _RUNNER_SHA256
    os.environ[_module._RUNNER_SHA_ENV] = _RUNNER_SHA256

_stable_payload(_SELF, _declared_site)
_stable_payload(_RUNNER, _RUNNER_SHA256, _RUNNER_BYTES)
print("[qwen-hauhau-bf16-oracle-transport] authenticated transport shadow armed", flush=True)
'''.encode()


def rewrite_command(
    payload: bytes,
    *,
    destination: Path,
    site_sha256: str,
    runner_sha256: str,
) -> bytes:
    environment, argv = _split_command(payload)
    values = _environment_map(environment)
    validate_bf16_command(values, argv)
    pythonpath = values.get("PYTHONPATH", "")
    chained_sha = values.get("QWEN_LIVE62_HAUHAU_DELTA_AGGRESSIVE_SITE_SHA256", "")
    if not pythonpath.startswith("/") or len(chained_sha) != 64:
        raise TransportOverlayError("base command lacks the authenticated Hauhau site chain")
    additions = {
        ENV_NAME: "1",
        "QWEN_HAUHAU_BF16_ORACLE_TRANSPORT_RUNNER_SHA256": runner_sha256,
        "QWEN_HAUHAU_BF16_ORACLE_TRANSPORT_SITE_SHA256": site_sha256,
    }
    if set(additions) & values.keys():
        raise TransportOverlayError("base command already contains BF16 transport identity")
    output: list[str] = []
    for token in environment:
        name, value = token.split("=", 1)
        if name == "PYTHONPATH":
            output.extend(f"{key}={item}" for key, item in additions.items())
            value = f"{destination}:{value}"
        output.append(f"{name}={value}")
    return (shlex.join(["exec", "/usr/bin/env", "-i", *output, *argv]) + "\n").encode()


def _write(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def render(
    *,
    runner_source: Path,
    base_command: Path,
    expected_base_sha256: str,
    destination: Path,
) -> Mapping[str, Any]:
    if destination.exists():
        raise TransportOverlayError("destination is create-only")
    source = runner_source.read_bytes()
    command_source = base_command.read_bytes()
    if digest(command_source) != expected_base_sha256:
        raise TransportOverlayError("base command SHA256 differs")
    environment, argv = _split_command(command_source)
    values = _environment_map(environment)
    validate_bf16_command(values, argv)
    chained_root = Path(values["PYTHONPATH"].split(":", 1)[0])
    chained_site = chained_root / "sitecustomize.py"
    chained_sha = values["QWEN_LIVE62_HAUHAU_DELTA_AGGRESSIVE_SITE_SHA256"]
    runner = rewrite_runner(source)
    site = render_site(
        destination,
        runner_sha256=digest(runner),
        runner_bytes=len(runner),
        chained_site=chained_site,
        chained_site_sha256=chained_sha,
        chained_pythonpath=values["PYTHONPATH"],
    )
    command = rewrite_command(
        command_source,
        destination=destination,
        site_sha256=digest(site),
        runner_sha256=digest(runner),
    )
    destination.mkdir(mode=0o700, parents=True)
    _write(destination / "model_runner.py", runner, 0o600)
    _write(destination / "sitecustomize.py", site, 0o600)
    _write(destination / "command.sh", command, 0o700)
    manifest = {
        "schema": SCHEMA,
        "source_runner_sha256": digest(source),
        "runner_sha256": digest(runner),
        "site_sha256": digest(site),
        "base_command_sha256": digest(command_source),
        "command_sha256": digest(command),
        "contract": {
            "bf16_only": True,
            "connector_free": True,
            "isolated_cache_suffix": ISOLATED_SUFFIX,
            "snapshot_export_disabled": True,
            "hauhau_loader_and_binders_preserved": True,
        },
    }
    manifest_payload = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    _write(destination / "manifest.json", manifest_payload, 0o600)
    return {**manifest, "manifest_sha256": digest(manifest_payload)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-command", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--runner-source", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = render(
            runner_source=args.runner_source,
            base_command=args.base_command,
            expected_base_sha256=args.expected_base_sha256,
            destination=args.destination,
        )
    except (TransportOverlayError, OSError, KeyError, json.JSONDecodeError) as error:
        print(f"qwen-bf16-oracle-transport-overlay: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
