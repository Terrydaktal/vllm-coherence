"""Create a separate verification build from an immutable measured worker and selected GPU kernels."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import authenticate, seal

SOURCE = Path(__file__).resolve().parent


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def amend(parent, output, *, norm=None, attention=None, qk_norm=None, draft_context=None):
    if not any((norm, attention, qk_norm, draft_context)):
        raise ValueError(
            "select at least one independently qualified numerical correction"
        )
    if output.exists():
        raise FileExistsError(output)
    receipt = json.loads((parent / "manifest.json").read_bytes())
    authenticate(receipt)
    if "numerical_parent" in receipt:
        raise ValueError(
            "start with the existing measured parent, not a previous amendment"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        stage = Path(staging) / "worker"
        stage.mkdir(mode=0o700)
        for name, digest in receipt["files"].items():
            source = parent / name
            if (
                not source.resolve().is_relative_to(parent.resolve())
                or sha(source) != digest
            ):
                raise ValueError(f"parent source differs from its receipt: {name}")
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        (stage / "fourteen_retained.py").rename(stage / "fourteen_base_retained.py")
        shutil.copyfile(
            SOURCE / "retained_numerical_repairs.py", stage / "fourteen_retained.py"
        )
        corrections = {}
        if norm:
            metadata = json.loads((norm / "build.json").read_bytes())
            if (
                metadata.get("max_norm_rows") != 4096
                or sha(norm / "candidate.so") != metadata["binary_sha256"]
            ):
                raise ValueError("wide input norm does not match its 4,096-row build")
            folder = stage / "wide-norm-build"
            folder.mkdir()
            for name in ("candidate.so", "build.json"):
                shutil.copyfile(norm / name, folder / name)
            shutil.copyfile(SOURCE / "wide_input_norm.py", stage / "wide_input_norm.py")
            corrections["wide_input_norm"] = metadata["binary_sha256"]
        if attention:
            metadata = json.loads((attention / "build.json").read_bytes())
            authenticate(metadata)
            source = SOURCE / "attention_runtime_splits.py"
            if (
                sha(attention / "runtime-splits.so") != metadata["binary"]
                or sha(source) != metadata["generator"]
            ):
                raise ValueError(
                    "shared attention kernel and generator do not match their build"
                )
            folder = stage / "attention-runtime-splits"
            folder.mkdir()
            for name in ("runtime-splits.so", "build.json"):
                shutil.copyfile(attention / name, folder / name)
            for name in (
                "attention_runtime_splits.py",
                "install_attention_runtime_splits.py",
            ):
                shutil.copyfile(SOURCE / name, stage / name)
            corrections["attention_runtime_splits"] = metadata["binary"]
        if qk_norm:
            from wide_qk_norm import BINARY_SHA256, KERNEL_SHA256

            tested = json.loads(qk_norm.read_bytes())
            authenticate(tested)
            if (
                tested.get("passed") is not True
                or tested["source_sha256"] != KERNEL_SHA256
                or tested["binary_sha256"] != BINARY_SHA256
            ):
                raise ValueError("Q/K normalization does not match its verified operator run")
            shutil.copyfile(SOURCE / "wide_qk_norm.py", stage / "wide_qk_norm.py")
            shutil.copyfile(qk_norm, stage / "qk-norm-qualification.json")
            corrections["wide_qk_norm"] = BINARY_SHA256
        if draft_context:
            tested = json.loads(draft_context.read_bytes())
            authenticate(tested)
            module = SOURCE / "dflash_context_projection.py"
            if (
                tested.get("passed") is not True
                or tested.get("tile") != [64, 128, 64]
                or tested.get("source_sha256") != sha(module)
            ):
                raise ValueError("draft context projection differs from its qualification")
            shutil.copyfile(module, stage / module.name)
            shutil.copyfile(draft_context, stage / "draft-context-qualification.json")
            corrections["draft_context_projection"] = tested["source_sha256"]
        binding = dict(receipt)
        binding.pop("sha256")
        binding["numerical_parent"] = receipt["sha256"]
        binding["numerical_corrections"] = corrections
        binding["selection"] = [*receipt["selection"], *corrections]
        binding["files"] = {
            str(p.relative_to(stage)): sha(p)
            for p in sorted(stage.rglob("*"))
            if p.is_file()
        }
        binding = seal(binding)
        (stage / "manifest.json").write_text(json.dumps(binding, indent=2) + "\n")
        stage.rename(output)
    return binding


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wide-norm", type=Path)
    parser.add_argument("--attention-splits", type=Path)
    parser.add_argument("--qk-norm-receipt", type=Path)
    parser.add_argument("--draft-context-receipt", type=Path)
    args = parser.parse_args()
    print(
        amend(
            args.parent,
            args.output,
            norm=args.wide_norm,
            attention=args.attention_splits,
            qk_norm=args.qk_norm_receipt,
            draft_context=args.draft_context_receipt,
        )["sha256"]
    )
