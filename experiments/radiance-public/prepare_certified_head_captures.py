"""Convert explicitly supplied synthetic head captures to a CPU-only pilot input.

Run in the existing CPU container with Torch installed; this does not import a
serving backend, decode tokens, or discover/read session transcripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch

    def forbid_gpu(*_args, **_kwargs):
        raise RuntimeError("GPU access is forbidden by this CPU capture converter")

    torch.cuda._lazy_init = forbid_gpu
    torch.set_num_threads(2)
    manifest = json.loads(args.manifest.read_text())
    expected = {Path(key).name: value for key, value in manifest["files"].items()}
    if len(expected) != len(manifest["files"]):
        raise ValueError("capture basenames are ambiguous")
    hidden, exact, labels = [], [], []
    for name, identity in sorted(expected.items()):
        path = args.captures / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != identity["sha256"] or path.stat().st_size != identity["size"]:
                raise ValueError(f"capture identity mismatch: {name}")
            stream.seek(0)
            data = torch.load(stream, map_location="cpu", weights_only=True)
        x, y = data["hidden"], data["exact"]
        if x.dtype != torch.bfloat16 or x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
            raise ValueError("capture dtype/shape unsupported")
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("capture contains nonfinite values")
        if not torch.equal(y.bfloat16().float(), y.float()):
            raise ValueError("reference capture is not a BF16-valued head output")
        hidden.append(x.float().numpy())
        exact.append(y.float().numpy())
        labels.extend({"capture": name, "row": i} for i in range(len(x)))
    if torch.cuda.is_initialized():
        raise RuntimeError("unexpected GPU initialization")
    with args.output.open("xb") as stream:
        np.savez(stream, hidden=np.concatenate(hidden), reference=np.concatenate(exact))
    with args.output.open("rb") as stream:
        output_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    receipt = {
        "scope": manifest["scope"],
        "gpu_used": False,
        "source_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "output_sha256": output_sha,
        "rows": labels,
        "converter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"captures": len(expected), "rows": len(labels), "gpu_used": False}))


if __name__ == "__main__":
    main()
