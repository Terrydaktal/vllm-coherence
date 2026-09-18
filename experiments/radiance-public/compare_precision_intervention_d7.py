"""Compare a compiler rounding intervention using saved evidence; no GPU execution."""

import argparse
import json
from pathlib import Path

from compare_execution_modes_d7 import load

from qwen_r9700_lab.conformance_precision_intervention import compare_precision_intervention
from qwen_r9700_lab.diagnostic_contract import private_json, write_private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("left", "right", "left-rows", "right-rows", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = compare_precision_intervention(
        load(args.left),
        load(args.right),
        private_json(args.left_rows),
        private_json(args.right_rows),
    )
    write_private(args.output, result)
    print(json.dumps({k: result[k] for k in ("status", "decode", "prefill")}))


if __name__ == "__main__":
    main()
