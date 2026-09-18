"""Authenticate assurance/release execution equivalence into one trace fragment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qwen_r9700_lab import assurance_instrumentation as instrumentation
from qwen_r9700_lab import assurance_provider_producer as provider
from qwen_r9700_lab import coding_turbo_artifacts

PRODUCER_KIND = "release_controller"
INSTRUMENTATION_SITE = "assurance.release.differential"
EXPECTED_SCOPE = (
    "assurance_release_equivalence",
    None,
    None,
    None,
    instrumentation.UNIT_PHASES["assurance_release_equivalence"],
    None,
)


class ReleaseProducerError(RuntimeError):
    """Release equivalence evidence is absent, unauthenticated, or divergent."""


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReleaseProducerError(f"{label} must be a lowercase SHA-256")
    return value


def _fragment_context() -> tuple[dict[str, Any], str]:
    if os.environ.get(instrumentation.FULL_ASSURANCE_ENABLE_ENV) != "1":
        raise ReleaseProducerError("release fragment was not explicitly enabled")
    missing = [
        name
        for name in instrumentation.FULL_ASSURANCE_ENVIRONMENT
        if not os.environ.get(name)
    ]
    if missing:
        raise ReleaseProducerError(f"release fragment environment is incomplete: {missing}")
    if os.environ[instrumentation.FULL_ASSURANCE_PRODUCER_KIND_ENV] != PRODUCER_KIND:
        raise ReleaseProducerError("release fragment producer kind differs")
    own_path = Path(__file__).resolve(strict=True)
    own_payload = provider._read_stable_owned_file(own_path, "release producer source")
    own_sha256 = hashlib.sha256(own_payload).hexdigest()
    if own_sha256 != os.environ[instrumentation.FULL_ASSURANCE_PRODUCER_SHA_ENV]:
        raise ReleaseProducerError("release producer source binding differs")
    header = instrumentation.normalize_header(
        provider._load_json(
            Path(os.environ[instrumentation.FULL_ASSURANCE_HEADER_ENV]),
            "release campaign header",
        )
    )
    scopes_document = provider._load_json(
        Path(os.environ[instrumentation.FULL_ASSURANCE_SCOPES_ENV]),
        "release runtime scopes",
    )
    if not isinstance(scopes_document, dict) or not isinstance(
        scopes_document.get("scopes"), list
    ):
        raise ReleaseProducerError("release runtime scope document is malformed")
    expected = instrumentation.runtime_scopes_document(header, [EXPECTED_SCOPE])
    if scopes_document != expected:
        raise ReleaseProducerError("release runtime scope identity differs")
    return header, own_sha256


def _capture_evidence(
    path: Path,
    file_sha256: str,
    configuration_sha256: str,
    header: dict[str, Any],
    final_tool_name: str,
) -> dict[str, Any]:
    capture = provider._load_json(
        path,
        "release differential chat capture",
        expected_sha256=_require_digest(file_sha256, "release differential capture"),
    )
    return provider.build_evidence(
        capture,
        header=header,
        expected_configuration_sha256=_require_digest(
            configuration_sha256, "release differential configuration"
        ),
        final_tool_name=final_tool_name,
    )


def build_evidence(
    *,
    artifact_root: Path,
    expected_equivalence_sha256: str,
    header: dict[str, Any],
    assurance: dict[str, Any],
    release: dict[str, Any],
    serial: dict[str, Any],
) -> dict[str, Any]:
    """Verify artifact purity plus exact serial/assurance/release output identity."""

    equivalence_path = artifact_root / "equivalence.json"
    equivalence_payload = provider._read_stable_owned_file(
        equivalence_path,
        "assurance/release equivalence certificate",
        expected_sha256=_require_digest(
            expected_equivalence_sha256, "assurance/release equivalence certificate"
        ),
    )
    try:
        equivalence = json.loads(equivalence_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseProducerError(
            "assurance/release equivalence certificate is invalid"
        ) from error
    verified = coding_turbo_artifacts.verify_artifacts(artifact_root)
    if (
        verified.get("verified") is not True
        or verified.get("semantic_source_sha256") != header["semantic_source_sha256"]
        or equivalence.get("semantic_source_sha256") != header["semantic_source_sha256"]
    ):
        raise ReleaseProducerError("artifact pair differs from the campaign semantic source")
    assurance_manifest = artifact_root / "assurance" / "manifest.json"
    release_manifest = artifact_root / "release" / "manifest.json"
    if (
        hashlib.sha256(assurance_manifest.read_bytes()).hexdigest()
        != header["assurance_manifest_sha256"]
        or hashlib.sha256(release_manifest.read_bytes()).hexdigest()
        != header["release_manifest_sha256"]
    ):
        raise ReleaseProducerError("artifact manifests differ from the campaign header")

    required = {
        "raw_token_ids_sha256",
        "parser_input_sha256",
        "parser_output_sha256",
        "finish_reason",
        "tool_ledger_sha256",
        "outcome",
    }
    signatures = {
        label: {key: evidence.get(key) for key in required}
        for label, evidence in (
            ("assurance", assurance),
            ("release", release),
            ("serial", serial),
        )
    }
    if signatures["assurance"] != signatures["serial"]:
        raise ReleaseProducerError("instrumented assurance output differs from serial reference")
    if signatures["release"] != signatures["serial"]:
        raise ReleaseProducerError(
            "instrumentation-free release output differs from serial reference"
        )

    regions = equivalence.get("assurance_regions")
    if not isinstance(regions, list) or not regions:
        raise ReleaseProducerError("equivalence certificate has no removed instrumentation regions")
    return {
        "semantic_source_sha256": header["semantic_source_sha256"],
        "assurance_binary_sha256": verified["assurance_tree_sha256"],
        "release_binary_sha256": verified["release_tree_sha256"],
        "assurance_trace_sha256": assurance["capture_sha256"],
        "release_output_sha256": release["raw_token_ids_sha256"],
        "serial_reference_sha256": serial["raw_token_ids_sha256"],
        "instrumentation_removed_sha256": provider._sha(regions),
        "passed": True,
        "assurance_capture_sha256": assurance["capture_sha256"],
        "release_capture_sha256": release["capture_sha256"],
        "serial_capture_sha256": serial["capture_sha256"],
        "semantic_output_signature_sha256": provider._sha(signatures["serial"]),
    }


def produce(args: argparse.Namespace) -> dict[str, Any]:
    header, own_sha256 = _fragment_context()
    captures = {
        label: _capture_evidence(
            getattr(args, f"{label}_capture"),
            getattr(args, f"{label}_capture_sha256"),
            getattr(args, f"{label}_configuration_sha256"),
            header,
            args.final_tool_name,
        )
        for label in ("assurance", "release", "serial")
    }
    evidence = build_evidence(
        artifact_root=args.artifact_root,
        expected_equivalence_sha256=args.expected_equivalence_sha256,
        header=header,
        assurance=captures["assurance"],
        release=captures["release"],
        serial=captures["serial"],
    )
    writer = instrumentation.FullAssuranceFragmentWriter(
        Path(os.environ[instrumentation.FULL_ASSURANCE_ROOT_ENV]),
        header,
        fragment_id=os.environ[instrumentation.FULL_ASSURANCE_FRAGMENT_ID_ENV],
        scopes=(EXPECTED_SCOPE,),
        producer_kind=PRODUCER_KIND,
        producer_sha256=own_sha256,
        sync_interval=1,
    )
    try:
        writer.append(
            unit="assurance_release_equivalence",
            phase=instrumentation.UNIT_PHASES["assurance_release_equivalence"],
            position=None,
            layer_index=None,
            row=None,
            evidence=evidence,
        )
        return writer.finalize(
            device_synchronize_count=0,
            installed_instrumentation_sites=(INSTRUMENTATION_SITE,),
        )
    except BaseException:
        writer.abort()
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-assurance-release-producer",
        description="Verify assurance/release artifact and execution equivalence.",
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--expected-equivalence-sha256", required=True)
    for label in ("assurance", "release", "serial"):
        parser.add_argument(f"--{label}-capture", type=Path, required=True)
        parser.add_argument(f"--{label}-capture-sha256", required=True)
        parser.add_argument(f"--{label}-configuration-sha256", required=True)
    parser.add_argument("--final-tool-name", default="qwen_final_answer")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = produce(args)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        coding_turbo_artifacts.ArtifactSafetyError,
        instrumentation.InstrumentationError,
        provider.ProviderProducerError,
        ReleaseProducerError,
    ) as error:
        print(f"qwen-assurance-release-producer: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
