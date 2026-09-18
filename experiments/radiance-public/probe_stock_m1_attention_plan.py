"""CPU-only exhaustive comparison of query split plans with the pinned native split law."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from stock_m1_attention import m1_splits, split_groups

from qwen_r9700_lab.diagnostic_contract import seal, write_private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    # This executable invokes only the exported host-side scratch-size function.
    os.environ.update(HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="")
    import torch  # isort: skip; resolves the binary library dependencies
    import r4d

    native = []
    for context in range(1, 253793):
        size = r4d.attn_decode_h256_gqa6_scratch_bytes(1, 1, 24, 4, 256, context, 0)
        count, remainder = divmod(size, 24 * (256 * 2 + 8))
        if remainder or count != m1_splits(context):
            raise ValueError(f"native M1 split-law difference at context={context}")
        native.append(count)
    cases = 0
    for width in range(1, 9):
        for bound in range(width, 253793):
            groups = split_groups(width, bound)
            unpacked, cursor = [], 0
            for start, stop, splits in groups:
                if start != cursor or not start < stop <= width:
                    raise ValueError("query grouping omitted, duplicated or reordered a row")
                unpacked.extend([splits] * (stop - start))
                cursor = stop
            if cursor != width or unpacked != native[bound - width : bound]:
                raise ValueError(f"query grouping mismatch at bound={bound}, width={width}")
            cases += 1
    report = seal(
        {
            "status": "TESTED",
            "native_contexts": len(native),
            "query_groupings": cases,
            "native_splits_sha256": hashlib.sha256(bytes(native)).hexdigest(),
            "library_sha256": hashlib.sha256(Path(r4d.__file__).read_bytes()).hexdigest(),
            "adapter_sha256": hashlib.sha256(
                Path(__file__).with_name("stock_m1_attention.py").read_bytes()
            ).hexdigest(),
            "torch_version": torch.__version__,
            "gpu_used": False,
            "scope": "All 1..253792 context bounds and legal query widths 1..8; split plan only.",
            "formal_backend_equivalence": "UNPROVED",
        }
    )
    write_private(args.output / "probe-result.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
