"""Verify existing BF16 gate tensors against bounded ranges of public HF shards.

Downloads only headers and 96 small gate tensors, never the full model. The
received tensor bytes are hashed and discarded; only public provenance remains.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import struct
import time
import urllib.request
from pathlib import Path

MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def verify(reference_report, output):
    base = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/"
    with urllib.request.urlopen(base + "model.safetensors.index.json", timeout=30) as response:
        raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("unexpected model index size")
    index = json.loads(raw)["weight_map"]
    report = json.loads(reference_report.read_text())
    gates = report["gates"]
    if len(gates) != 96 or any(not re.fullmatch(
        r"model\.language_model\.layers\.\d+\.linear_attn\.in_proj_[ab]\.weight", x["name"]
    ) for x in gates):
        raise ValueError("unexpected gate reference set")

    def read_range(filename, begin, end):
        if not re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", filename):
            raise ValueError("unexpected public shard filename")
        size = end - begin + 1
        if not 0 < size <= 1_000_000:
            raise ValueError("unexpected tensor/header range size")
        url = base + filename + f"?qwen-range={begin}-{end}"
        request = urllib.request.Request(url, headers={"Range": f"bytes={begin}-{end}"})
        with urllib.request.urlopen(request, timeout=45) as response:
            expected = f"bytes {begin}-{end}/"
            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                raise ValueError("server did not honor the bounded range")
            data = response.read(size + 1)
            if len(data) != size:
                raise ValueError("unexpected public range length")
        return data

    def header(filename):
        length = struct.unpack("<Q", read_range(filename, 0, 7))[0]
        if not 0 < length < 1_000_000:
            raise ValueError("unexpected safetensors header length")
        return filename, (8 + length, json.loads(read_range(filename, 8, length + 7)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        headers = dict(executor.map(header, sorted({index[g["name"]] for g in gates})))

    def check(gate):
        name = gate["name"]
        filename = index[name]
        offset, metadata = headers[filename]
        tensor = metadata[name]
        if tensor["dtype"] != "BF16" or tensor["shape"] != [48, 5120]:
            raise ValueError("unexpected official gate precision/shape")
        begin, end = tensor["data_offsets"]
        if end - begin != 48 * 5120 * 2:
            raise ValueError("unexpected official gate byte count")
        data = read_range(filename, offset + begin, offset + end - 1)
        digest = hashlib.sha256(data).hexdigest()
        return {"name": name, "sha256": digest, "bytes": len(data),
                "matches_preserved_reference": digest == gate["reference_sha256"]}

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for row in executor.map(check, gates):
            rows.append(row)
            if len(rows) % 16 == 0:
                print(json.dumps({"verified_tensors": len(rows), "matching": sum(x["matches_preserved_reference"] for x in rows)}), flush=True)
    result = {"at": time.time(), "model": MODEL, "revision": REVISION,
              "index_sha256": hashlib.sha256(raw).hexdigest(),
              "all_match": all(row["matches_preserved_reference"] for row in rows),
              "tensor_bytes_read": sum(row["bytes"] for row in rows), "gates": rows}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "gates"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    verify(args.reference_report, args.output)
