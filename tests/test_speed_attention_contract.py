"""Finite input-format certificate for the guarded attention experiment."""

import importlib.util
import inspect
import re
import struct
from pathlib import Path


def test_every_bf16_pattern_admitted_by_generated_gate_is_exact_normal_half():
    path = (
        Path(__file__).parents[1]
        / "experiments/radiance-public/probe_attention_memory_tuning.py"
    )
    spec = importlib.util.spec_from_file_location("attention_speed_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = inspect.getsource(module.guarded_half_queries)
    # Extract the actual native gate's constants, rather than proving an
    # unrelated intended-behaviour model of that gate.
    bounds = re.findall(r"low >= (0x[0-9a-f]+)u && low <= (0x[0-9a-f]+)u", source)
    assert len(bounds) == 1
    low, high = (int(value, 16) for value in bounds[0])
    assert f"high >= {low:#x}u && high <= {high:#x}u" in source
    admitted = 0
    for bits in range(1 << 16):
        magnitude = bits & 0x7FFF
        if magnitude != 0 and not low <= magnitude <= high:
            continue
        reference_bits = struct.pack("<I", bits << 16)
        value = struct.unpack("<f", reference_bits)[0]
        half_bytes = struct.pack("<e", value)
        half_bits = struct.unpack("<H", half_bytes)[0]
        # No half subnormal can reach the fast path, independently of the
        # GPU's denormal mode. Signs of zero also survive bit-exactly.
        assert half_bits & 0x7FFF == 0 or 0 < (half_bits >> 10) & 31 < 31
        restored = struct.pack("<f", struct.unpack("<e", half_bytes)[0])
        assert restored == reference_bits
        admitted += 1
    assert admitted == 7682
    # Overflow, NaNs, infinities and half subnormals remain outside admission.
    for rejected in (0x4780, 0x7F80, 0x7FC0, 0x3800, 0x0001):
        assert not low <= rejected <= high
