"""CPU-only output comparison for the explicitly declared SiLU intervention."""

import argparse
import hashlib
import json
from pathlib import Path

from compare_execution_modes_d7 import load

from qwen_r9700_lab.conformance_silu_intervention import compare_silu_intervention
from qwen_r9700_lab.diagnostic_contract import private_json, write_private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "left",
        "right",
        "left-rows",
        "right-rows",
        "base-driver",
        "experiment-driver",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = compare_silu_intervention(
        load(args.left),
        load(args.right),
        private_json(args.left_rows),
        private_json(args.right_rows),
        base_driver=hashlib.sha256(args.base_driver.read_bytes()).hexdigest(),
        experiment_driver=hashlib.sha256(args.experiment_driver.read_bytes()).hexdigest(),
    )
    write_private(args.output, result)
    print(
        json.dumps(
            {"status": result["status"], "decode": result["decode"], "prefill": result["prefill"]}
        )
    )


if __name__ == "__main__":
    main()
