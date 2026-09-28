"""Create a hashed, isolated wider-prefill candidate; never edits a release.

Only the admitted row bounds change. The experiment still needs freshly built
kernels, native boundary tests, full-model replay and cache/lifecycle checks.
"""

import argparse
import hashlib
import json
from pathlib import Path

REPLACEMENTS = {
    "prefill_attention_alignment.py": (
        ("a->q_len > 2048", "a->q_len > 4096"),
        ("q.shape[0] <= 2048", "q.shape[0] <= 4096"),
    ),
    "prefill_gemm_alignment.py": (
        ("M > 2048", "M > 4096"),
        ("m <= 2048", "m <= 4096"),
    ),
    "prefill_alignment_runtime.py": (
        ("plan[0][2] <= 2048", "plan[0][2] <= 4096"),
        ("m <= 2048", "m <= 4096"),
        ("at most 2048 prefill rows", "at most 4096 prefill rows"),
        (
            '        calls["attention_prefill"] += 1',
            '        calls["attention_prefill"] += 1\n'
            '        widths = calls.setdefault("prefill_rows", {})\n'
            '        widths[str(width)] = widths.get(str(width), 0) + 1',
        ),
    ),
    "prepared_prefill_scan.py": (
        ("rows <= 2048", "rows <= 4096"),
        ("mixed.shape[0] <= 2048", "mixed.shape[0] <= 4096"),
    ),
    "prefill_dynamic_conv.py": (
        ("rows <= 2048", "rows <= 4096"),
    ),
}


def prepare(source, output):
    output.mkdir(mode=0o700)
    records = {}
    for path in source.glob("*.py"):
        raw = path.read_bytes()
        text = raw.decode()
        for before, after in REPLACEMENTS.get(path.name, ()):
            if text.count(before) != 1:
                raise ValueError(
                    f"wide-prefill source anchor changed: {path.name}: {before}"
                )
            text = text.replace(before, after)
        (output / path.name).write_text(text)
        records[path.name] = {
            "parent_sha256": hashlib.sha256(raw).hexdigest(),
            "candidate_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    if not set(REPLACEMENTS).issubset(records):
        raise ValueError("missing prefill source")
    (output / "experiment.json").write_text(
        json.dumps(
            {
                "status": "UNTESTED",
                "changes": REPLACEMENTS,
                "files": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
