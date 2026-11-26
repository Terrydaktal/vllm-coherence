"""Build the isolated R9700 serving and numerical qualification overlay."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from build_prefill_bank_norm import bank_kernel
from drafter_operator_candidates import write_draft_head
from materialize_snapshot_candidates import build as build_cpu
from mxfp4_prefill_tiles import build as build_tiles

from qwen_r9700_lab.diagnostic_contract import seal

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[1]
MODULES = (
    "cache_buffer_views",
    "dflash_prompt_lookup",
    "dflash_prompt_lookup_kernels",
    "dflash_proposal_temperature",
    "embedding_readback_candidate",
    "fourteen_retained",
    "fourteen_worker",
    "gdn_prefill_direct",
    "head_live_rows",
    "mxfp4_fold_precision",
    "mxfp4_prefill_candidate",
    "mxfp4_prefill_tiles",
    "prefill_bank_norm",
    "prefill_conv_slot",
    "snapshot_cpu_candidates",
)
SELECTION = (
    "startup_embedding_one_readback",
    "target_live_rows",
    "draft_live_rows",
    "mxfp4_preprocess_96MiB",
    "mxfp4_slab_ahead",
    "mxfp4_thin_and_shape_tn",
    "gdn_direct_scan",
    "gdn_direct_output_projection",
    "gdn_bank",
    "gdn_direct_convolution_slot",
    "gdn_bias",
    "cpu_cache_and_lineage",
    "dflash_streak_lookup",
    "dflash_proposal_temperature_0.55",
)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_binding(output, cache_source, drafter):
    overlay = {}
    for stem in ("radiance_cache", "radiance_prefix_lineage"):
        overlay[f"cpu/{stem}.py"] = {
            "path": f"qwen_{stem}.py",
            "preimage": digest(cache_source / f"{stem}.py"),
        }
    binding = seal(
        {
            "schema": "coherence-source-bound-combined-experiment-v1",
            "overlay": overlay,
            "drafter_head_source": digest(drafter),
            "files": {
                str(p.relative_to(output)): digest(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
                and p.name != "manifest.json"
                and "__pycache__" not in p.parts
            },
            "selection": list(SELECTION),
        }
    )
    (output / "manifest.json").write_text(json.dumps(binding, indent=2) + "\n")
    return binding


def build(output, cache_source):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("build into a new, empty private output directory")
    drafter = REPO / "radiance_drafthead.py"
    for name in MODULES:
        shutil.copy2(SOURCE / f"{name}.py", output)
    build_tiles(REPO, output)
    bank = output / "direct-build"
    bank.mkdir()
    (bank / "prefill_indexed_kernel.py").write_text(
        bank_kernel((SOURCE / "prepared_prefill_scan.py").read_text())
    )
    write_draft_head(drafter, output / "radiance_drafthead_candidate.py")
    build_cpu(cache_source, output / "cpu")
    return write_binding(output, cache_source, drafter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cache-source", type=Path, default=REPO / "src" / "qwen_r9700_lab"
    )
    args = parser.parse_args()
    print(build(args.output, args.cache_source)["sha256"])
