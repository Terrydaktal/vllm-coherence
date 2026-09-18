"""Manifest-bound token-prefix validation for fixed-slot resume clients."""

from __future__ import annotations

import hashlib
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

TOKEN_DIGEST_DOMAIN = b"qwen-r9700-token-ids-u32be-v1\x00"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SnapshotPrefixError(ValueError):
    """A resume manifest or live token prefix does not satisfy its binding."""


@dataclass(frozen=True, slots=True)
class ResumePrefixContract:
    """The exact token slice a client must validate before server admission."""

    expected_sha256: str
    validation_tokens: int
    replay_boundary_tokens: int
    manifest_prompt_tokens: int
    legacy_full_prompt_fallback: bool

    @property
    def mutable_tail_start(self) -> int:
        return self.validation_tokens

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Hash uint32 token IDs with the runtime's domain-separated encoding."""

    digest = hashlib.sha256()
    digest.update(TOKEN_DIGEST_DOMAIN)
    digest.update(struct.pack(">Q", len(token_ids)))
    for index, token_id in enumerate(token_ids):
        if type(token_id) is not int or not 0 <= token_id <= 0xFFFFFFFF:
            raise SnapshotPrefixError(f"token ID {index} is not uint32")
        digest.update(struct.pack(">I", token_id))
    return digest.hexdigest()


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SnapshotPrefixError(f"{label} is not a lowercase SHA-256")
    return value


def resume_prefix_contract(
    manifest: Mapping[str, Any],
    *,
    allow_legacy_full_prompt: bool = False,
) -> ResumePrefixContract:
    """Extract the authenticated client-side validation contract.

    New manifests bind exactly the reusable prefix through ``R``.  A legacy
    manifest has no such digest, so it is accepted only in the stricter mode:
    the client must reproduce the complete old prompt through ``P``.  This
    permits a safe one-time upgrade without pretending a missing prefix digest
    can authorize a changed replay tail.
    """

    request = manifest.get("request")
    if not isinstance(request, Mapping):
        raise SnapshotPrefixError("manifest request binding is missing")
    prompt_tokens = request.get("prompt_tokens")
    replay_boundary = request.get("replay_boundary_tokens")
    if type(prompt_tokens) is not int or prompt_tokens <= 0:
        raise SnapshotPrefixError("manifest prompt token count is invalid")
    if type(replay_boundary) is not int or replay_boundary <= 0 or replay_boundary >= prompt_tokens:
        raise SnapshotPrefixError("manifest replay boundary is invalid")
    prefix_digest = request.get("replay_prefix_token_ids_sha256")
    if prefix_digest is None:
        if not allow_legacy_full_prompt:
            raise SnapshotPrefixError(
                "rolling resume manifest lacks replay_prefix_token_ids_sha256"
            )
        return ResumePrefixContract(
            expected_sha256=_sha256(
                request.get("prompt_token_ids_sha256"),
                "legacy full-prompt digest",
            ),
            validation_tokens=prompt_tokens,
            replay_boundary_tokens=replay_boundary,
            manifest_prompt_tokens=prompt_tokens,
            legacy_full_prompt_fallback=True,
        )
    return ResumePrefixContract(
        expected_sha256=_sha256(prefix_digest, "replay-prefix digest"),
        validation_tokens=replay_boundary,
        replay_boundary_tokens=replay_boundary,
        manifest_prompt_tokens=prompt_tokens,
        legacy_full_prompt_fallback=False,
    )


def validate_resume_prefix(
    token_ids: Sequence[int],
    contract: ResumePrefixContract,
) -> None:
    """Reject a token mismatch locally, before submitting it to EngineCore."""

    if len(token_ids) < contract.validation_tokens:
        raise SnapshotPrefixError(
            "live prompt is shorter than the manifest-bound validation prefix"
        )
    actual = token_ids_sha256(token_ids[: contract.validation_tokens])
    if actual != contract.expected_sha256:
        raise SnapshotPrefixError("live prompt differs before the authenticated resume boundary")
