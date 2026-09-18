"""Experimental post-scan correction for the pinned GDN decay-clamping defect.

The original kernel remains unchanged. Only affected sequence/head pairs are
recomputed by a verified bounded recurrence on the same GPU stream. This adds no
text inspection, generation interruption, sampling change or host/GPU sync.
"""

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

PREIMAGE = "f6675ecfcc0bba7c8f2b77db91438e6f6888a251d7140b443d51f3c21aa2dbe6"
SOURCE_SHA256 = "174152c14314ab28e5f6ab7cb400b7d04f9a6d71df7534d4d4ad5b0fc46a1695"
LIBRARY_SHA256 = "8b04f207868a4e800f5865e248a0f18396c1ec75dfb1fbd3ed696db97d28243a"


def build(source, output):
    """Build the qualified arithmetic in the pinned image, with a stable HIP ID.

    Run at installation/startup only, never on a model request. Compilation uses
    no GPU. Verify the exact resulting binary before publishing it.
    """
    source, output = Path(source), Path(output)
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("GDN correction source does not match its qualified version")
    if output.exists() and hashlib.sha256(output.read_bytes()).hexdigest() == LIBRARY_SHA256:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qwen-gdn-build-", dir=output.parent) as temporary:
        candidate = Path(temporary) / "gdn_extreme_decay_reference.so"
        subprocess.run([
            "/opt/rocm/bin/hipcc", "-O3", "-std=c++17", "--offload-arch=gfx1201",
            "-shared", "-fPIC", "-Wall", "-Wextra", "-cuid=qwen_gdn_" + SOURCE_SHA256,
            str(source), "-o", str(candidate),
        ], check=True, timeout=120, capture_output=True)
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != LIBRARY_SHA256:
            raise ValueError("GDN correction build does not match the qualified binary")
        os.replace(candidate, output)
    return output


def install(package, library_path, expected_library_sha256):
    path = Path(package) / "radiance_gdn.py"
    library_path = Path(library_path)
    if hashlib.sha256(library_path.read_bytes()).hexdigest() != expected_library_sha256:
        raise ValueError("GDN correction library hash mismatch")
    source = path.read_text()
    appendix = f'''

# Verified numerical correction for extreme decay spans; no generation policy.
import ctypes as _qwen_gdn_ctypes
_qwen_gdn_library = _qwen_gdn_ctypes.CDLL({str(library_path)!r})
_qwen_gdn_repair = _qwen_gdn_library.qwen_gdn_extreme_decay_reference
_qwen_gdn_repair.argtypes = ([_qwen_gdn_ctypes.c_void_p] * 9
    + [_qwen_gdn_ctypes.c_int] * 4 + [_qwen_gdn_ctypes.c_float] * 2
    + [_qwen_gdn_ctypes.c_void_p])
_qwen_gdn_repair.restype = _qwen_gdn_ctypes.c_int
_qwen_gdn_original_scan = _CHUNK_SCAN


def _qwen_gdn_corrected_scan(q, k, v, matrix, g, beta, initial, output, final,
                             cu, sequences, heads, query_heads, key_width,
                             value_width, chunk, scale, stream):
    if key_width != 128 or value_width != 128 or chunk != 64:
        raise ValueError('GDN numerical correction received unsupported geometry')
    _qwen_gdn_original_scan(q, k, v, matrix, g, beta, initial, output, final,
                            cu, sequences, heads, query_heads, key_width,
                            value_width, chunk, scale, stream)
    result = _qwen_gdn_repair(q, k, v, g, beta, initial, output, final, cu,
                              sequences, heads, query_heads, chunk, scale,
                              128.0, stream)
    if result:
        raise RuntimeError('GDN numerical correction failed to launch: ' + str(result))


if _CHUNK_SCAN is not None:
    _CHUNK_SCAN = _qwen_gdn_corrected_scan
'''
    original = source[:-len(appendix)] if source.endswith(appendix) else source
    if hashlib.sha256(original.encode()).hexdigest() != PREIMAGE:
        raise ValueError("GDN source does not match the qualified preimage")
    updated = original + appendix
    compile(updated, str(path), "exec")
    if source != updated:
        path.write_text(updated)
    return {"source_sha256": hashlib.sha256(updated.encode()).hexdigest(),
            "library_sha256": expected_library_sha256,
            "correction": "bounded_fp32_recurrence_for_extreme_decay",
            "host_gpu_synchronization": False}
