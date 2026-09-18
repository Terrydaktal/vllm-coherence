"""Offline entry point for portable backend diagnostic evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen_r9700_lab.diagnostic_adapters import inventory, require_capability
from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    compare_traces,
    execution_manifest,
    private_json,
    write_private,
)

HELP = """NAME
    qwen-diagnostics - inspect reusable inference diagnostics and compare evidence

SYNOPSIS
    qwen-diagnostics inventory [--root PATH]
    qwen-diagnostics capability ADAPTER CAPABILITY
    qwen-diagnostics manifest --spec FILE --output FILE
    qwen-diagnostics compare --left FILE --right FILE --output FILE
    qwen-diagnostics prove-helpers --output DIRECTORY

DESCRIPTION
    Reuse backend-independent contracts with versioned native adapters. Missing
    capabilities remain explicit. A finite comparison is not a formal proof.

OPTIONS
    --root PATH       Project containing the existing diagnostic implementations.
    --spec FILE       Private JSON with semantics, files and unavailable groups.
    --left FILE       Private, authenticated reference trace.
    --right FILE      Private, authenticated candidate trace.
    --output FILE     Create-only private evidence output; never overwritten.

OPERATION
    Inventory reads source metadata. Manifest hashes explicitly named artifacts
    once; requests can reference that manifest hash. Compare validates identities,
    input, coverage and boundary hashes. No command launches or modifies a backend.

EXAMPLES
    qwen-diagnostics inventory
    qwen-diagnostics capability radiance-0.28-qwen3-next-v1 native_full_state
    qwen-diagnostics compare --left reference.json --right candidate.json --output result.json

FILES
    Input JSON and output evidence are owner-only files. Raw tensors and tokens
    belong in private diagnostic storage; the comparison prints summary fields.

PATHS
    Artifact paths in a manifest specification are relative to that specification.

SECURITY NOTES
    Explicit artifact lists avoid scanning chats or credentials. Hashing a file
    does not prove that a running process loaded it. Native attestation is separate.

EXIT STATUS
    0  Operation succeeded; compared boundaries match if comparing.
    1  Compared boundaries differ or contain observed nonfinite values.
    2  Unsupported capability, incomplete evidence or invalid input.

AUTHORS
    Terrydaktal and contributors.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("inventory")
    listing.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    capability = commands.add_parser("capability")
    capability.add_argument("adapter")
    capability.add_argument("capability")
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--spec", required=True, type=Path)
    manifest.add_argument("--output", required=True, type=Path)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--left", required=True, type=Path)
    comparison.add_argument("--right", required=True, type=Path)
    comparison.add_argument("--output", required=True, type=Path)
    proofs = commands.add_parser("prove-helpers")
    proofs.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            result = inventory(args.root)
            print(json.dumps(result, indent=2))
            return 2 if result["missing_source_files"] else 0
        if args.command == "capability":
            require_capability(args.adapter, args.capability)
            print(json.dumps({"implemented": True, "qualification_inherited": False}))
            return 0
        if args.command == "prove-helpers":
            from qwen_r9700_lab.conformance_proofs import run_obligations

            result = run_obligations(args.output)
            print(json.dumps(result))
            return 0 if result["all_expected_results"] else 2
        if args.command == "manifest":
            spec = private_json(args.spec)
            if set(spec) != {"semantics", "files", "unavailable"}:
                raise DiagnosticError("manifest specification fields are incomplete")
            files = {
                group: {name: args.spec.parent / path for name, path in members.items()}
                for group, members in spec["files"].items()
            }
            result = execution_manifest(spec["semantics"], files, spec["unavailable"])
        else:
            result = compare_traces(private_json(args.left), private_json(args.right))
        write_private(args.output, result)
        print(json.dumps({key: value for key, value in result.items() if key != "artifacts"}))
        return int(result.get("equal_observed_boundaries") is False)
    except (DiagnosticError, OSError, ValueError, TypeError, KeyError) as error:
        # Even parse errors can quote private input. Expose a class, never input.
        print(json.dumps({"status": "invalid_or_unsupported", "error_type": type(error).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
