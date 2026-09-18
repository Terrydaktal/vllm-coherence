"""Command-line interface for the lab's reproducibility layer."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from qwen_r9700_lab.config import (
    ConfigurationError,
    all_config_paths,
    find_project_root,
    validate_config,
    validate_profiles,
)
from qwen_r9700_lab.manifests import capture_environment, capture_model, prepare_run


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _help_description(name: str, synopsis: str, description: str) -> str:
    return f"""NAME
    {name}

SYNOPSIS
    {synopsis}

DESCRIPTION
    {description}

OPTIONS
    The accepted options are listed below."""


def _help_epilog(operation: str, examples: str) -> str:
    return f"""OPERATION
    {operation}

EXAMPLES
{examples}

FILES
    configs/**/*.json define hardware, models, engines, and context profiles.
    schemas/*.json define the validation contracts.

PATHS
    Relative input paths are resolved from the current directory. Manifest records use absolute
    source paths. No model location is assumed.

SECURITY NOTES
    Manifests record host names, local paths, bounded command output, and an allowlist of
    accelerator-related environment variables. They do not collect the general environment.
    Existing outputs are never replaced; new manifests are made read-only after an atomic link.

EXIT STATUS
    0 on success; 2 for arguments, invalid input, failed probes required by the operation, or an
    existing output path.

AUTHORS
    Terrydaktal <9lewis9@gmail.com>"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-r9700-lab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_help_description(
            "qwen-r9700-lab - capture reproducible Qwen/R9700 evidence",
            "qwen-r9700-lab COMMAND [OPTIONS]",
            "Validate lab configuration and create immutable evidence manifests. This tool does "
            "not run inference.",
        ),
        epilog=_help_epilog(
            "Select one validation or manifest operation. Capture environment before model, then "
            "prepare a run from both immutable inputs.",
            "    qwen-r9700-lab validate-config\n"
            "    qwen-r9700-lab validate-profiles\n"
            "    qwen-r9700-lab capture-environment --help",
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate-config",
        help="validate configuration JSON (all checked-in configs when paths are omitted)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_help_description(
            "qwen-r9700-lab validate-config - validate typed lab configuration",
            "qwen-r9700-lab validate-config [PATH ...]",
            "Validate selected JSON configuration files, or all checked-in configs when no path "
            "is supplied.",
        ),
        epilog=_help_epilog(
            "Loads each object's kind, applies its JSON Schema, and enforces profile semantics.",
            "    qwen-r9700-lab validate-config\n"
            "    qwen-r9700-lab validate-config configs/models/*.json",
        ),
    )
    validate.add_argument("paths", nargs="*", type=_path)

    profiles = subparsers.add_parser(
        "validate-profiles",
        help="validate profile JSON and cross-profile uniqueness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_help_description(
            "qwen-r9700-lab validate-profiles - validate context profiles as a set",
            "qwen-r9700-lab validate-profiles [PATH ...]",
            "Validate profile schemas, native/YaRN invariants, and unique profile names.",
        ),
        epilog=_help_epilog(
            "Uses every checked-in profile when paths are omitted; it never writes files.",
            "    qwen-r9700-lab validate-profiles\n"
            "    qwen-r9700-lab validate-profiles configs/profiles/fast.json",
        ),
    )
    profiles.add_argument("paths", nargs="*", type=_path)

    environment = subparsers.add_parser(
        "capture-environment",
        help="capture an immutable host and accelerator-stack manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_help_description(
            "qwen-r9700-lab capture-environment - capture target-host evidence",
            "qwen-r9700-lab capture-environment --hardware-config PATH --output PATH",
            "Capture bounded system, PCI, ROCm, Vulkan, compiler, per-BDF DRM, and full-size ReBAR "
            "evidence against the expected hardware declaration.",
        ),
        epilog=_help_epilog(
            "Validates the hardware config, runs read-only probes, then atomically creates one "
            "content-addressed manifest.",
            "    qwen-r9700-lab capture-environment \\\n"
            "        --hardware-config configs/hardware/r9700-single.json \\\n"
            "        --output artifacts/manifests/environment.json",
        ),
    )
    environment.add_argument("--hardware-config", required=True, type=_path)
    environment.add_argument("--output", required=True, type=_path)

    model = subparsers.add_parser(
        "capture-model",
        help="hash a downloaded model tree into an immutable manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_help_description(
            "qwen-r9700-lab capture-model - inventory exact checkpoint files",
            "qwen-r9700-lab capture-model --model-config PATH --model-dir PATH --output PATH",
            "Hash every included file in a downloaded checkpoint and bind it to its model "
            "declaration.",
        ),
        epilog=_help_epilog(
            "Excludes declared transient paths, hashes files in stable path order, and atomically "
            "creates one content-addressed manifest.",
            "    qwen-r9700-lab capture-model \\\n"
            "        --model-config configs/models/frozenlock-qwen3.8-27b-autoround.json \\\n"
            "        --model-dir /models/Qwen3.8-27B-int4-AutoRound \\\n"
            "        --output artifacts/manifests/model.json",
        ),
    )
    model.add_argument("--model-config", required=True, type=_path)
    model.add_argument("--model-dir", required=True, type=_path)
    model.add_argument("--output", required=True, type=_path)

    run = subparsers.add_parser(
        "prepare-run",
        help="bind evidence and policy into a run manifest without invoking a backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_help_description(
            "qwen-r9700-lab prepare-run - declare a reproducible future run",
            "qwen-r9700-lab prepare-run --environment-manifest PATH --model-manifest PATH "
            "--engine-config PATH --profile-config PATH --reasoning-effort EFFORT --output PATH",
            "Bind verified environment/model inputs to independent engine, context, and reasoning "
            "choices without launching inference.",
        ),
        epilog=_help_epilog(
            "Verifies manifest IDs, validates selected configs, checks model-format compatibility, "
            "and creates a prepared run manifest with backend_invoked=false.",
            "    qwen-r9700-lab prepare-run \\\n"
            "        --environment-manifest artifacts/manifests/environment.json \\\n"
            "        --model-manifest artifacts/manifests/model.json \\\n"
            "        --engine-config configs/engines/rocm-hip-development.json \\\n"
            "        --profile-config configs/profiles/fast.json \\\n"
            "        --reasoning-effort xhigh --output artifacts/manifests/run.json",
        ),
    )
    run.add_argument("--environment-manifest", required=True, type=_path)
    run.add_argument("--model-manifest", required=True, type=_path)
    run.add_argument("--engine-config", required=True, type=_path)
    run.add_argument("--profile-config", required=True, type=_path)
    run.add_argument(
        "--reasoning-effort",
        required=True,
        choices=("off", "low", "medium", "xhigh"),
    )
    run.add_argument(
        "--preserve-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="preserve prior reasoning in thinking modes (default: true)",
    )
    run.add_argument("--output", required=True, type=_path)
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = find_project_root()
        if args.command == "validate-config":
            paths = args.paths or all_config_paths(root)
            if not paths:
                raise ConfigurationError("no configuration files were selected")
            for path in paths:
                validate_config(path, root)
                print(f"valid: {path}")
        elif args.command == "validate-profiles":
            paths = args.paths or sorted((root / "configs" / "profiles").glob("*.json"))
            validate_profiles(paths, root)
            for path in paths:
                print(f"valid profile: {path}")
        elif args.command == "capture-environment":
            manifest = capture_environment(args.hardware_config, args.output, root)
            print(f"created {args.output}: {manifest['manifest_id']}")
        elif args.command == "capture-model":
            manifest = capture_model(args.model_config, args.model_dir, args.output, root)
            print(f"created {args.output}: {manifest['manifest_id']}")
        elif args.command == "prepare-run":
            manifest = prepare_run(
                environment_manifest=args.environment_manifest,
                model_manifest=args.model_manifest,
                engine_config=args.engine_config,
                profile_config=args.profile_config,
                reasoning_effort=args.reasoning_effort,
                preserve_thinking=args.preserve_thinking,
                output=args.output,
                project_root=root,
            )
            print(f"created {args.output}: {manifest['manifest_id']}")
        else:  # pragma: no cover - argparse enforces the command set
            raise AssertionError(f"unhandled command: {args.command}")
    except ConfigurationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


def main() -> None:
    raise SystemExit(run())
