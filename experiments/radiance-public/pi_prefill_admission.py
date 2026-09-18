"""Fail closed when the selected norm/scan evidence or source has changed."""

import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_report(entry):
    if digest(entry["qualification"]) != entry["qualification_sha256"]:
        raise ValueError("prefill qualification changed")
    if not entry.get("sources"):
        raise ValueError("prefill qualification has no source binding")
    for name, expected in entry["sources"].items():
        if Path(name).name != name or digest(Path(__file__).with_name(name)) != expected:
            raise ValueError("prefill implementation changed")
    return json.loads(Path(entry["qualification"]).read_text())


def gdn_evidence(entry):
    r = checked_report(entry)
    if (
        r.get("status") != "SAMPLE_CHECKED"
        or r.get("sites") != 48
        or r.get("input_rows_per_site", 0) < 1000
        or not r.get("negative_control_detected")
        or r.get("source_sha256") != entry["sources"].get("stock_gdn_norm_quant.py")
    ):
        raise ValueError("GDN norm/quant qualification is incomplete")
    actual = {(c["site"], c["width"]): c["matching_rows"] for c in r["checks"] if "site" in c}
    if any(actual.get((site, width), 0) < 1000 for site in range(48) for width in (1, 8)):
        raise ValueError("GDN norm/quant requires 1000 exact rows at all 48 sites")
    widths = {c["prefill_width"]: c["equal"] for c in r["checks"] if "prefill_width" in c}
    if any(widths.get(width) is not True for width in (9, 320, 1000, 1648, 2048)):
        raise ValueError("GDN norm/quant prefill comparison is incomplete")
    return r


def scan_evidence(entry):
    r = checked_report(entry)
    if r.get("status") != "SAMPLE_CHECKED" or not r.get("negative_control_detected"):
        raise ValueError("spatial scan qualification failed")
    if r.get("sources") != entry["sources"]:
        raise ValueError("spatial scan qualification source differs")
    actual = {(c["tokens"], c["tile"]): c["exact"] for c in r["checks"]}
    if any(
        actual.get((rows, tile)) is not True
        for rows in (1, 8, 64, 320, 1000, 1648, 2048)
        for tile in (8, 16, 32, "selected")
    ):
        raise ValueError("spatial scan qualification is incomplete")
    return r
