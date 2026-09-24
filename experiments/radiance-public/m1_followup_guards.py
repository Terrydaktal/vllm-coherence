"""CPU admission repairs for packed GDN state and OCP/FNUZ cache formats."""

import hashlib

PACKED_SHA256 = "cf367195e14880a17d3a06e0b34cfb9f67a96c46a08438b9e91e1459029473b4"
PACKED_OLD = """    if initial_state.stride(-1) != 1:
        raise ValueError("`initial_state` must be contiguous in the last dim.")
    HV, V, K = initial_state.shape[-3:]"""
PACKED_NEW = """    HV, V, K = initial_state.shape[-3:]
    # The kernel accepts an outer slot stride but addresses each slot as [HV,V,K].
    if (
        tuple(initial_state.stride()[-3:]) != (V * K, K, 1)
        or initial_state.stride(0) < HV * V * K
    ):
        raise ValueError("`initial_state` requires packed non-overlapping [HV,V,K] slots.")"""


def patch_packed(source):
    original = source.replace(PACKED_NEW, PACKED_OLD)
    if hashlib.sha256(original.encode()).hexdigest() != PACKED_SHA256:
        raise ValueError("GDN layout repair requires the pinned stable-softplus source")
    if original.count(PACKED_OLD) != 1:
        raise ValueError("GDN layout admission anchor changed")
    return original.replace(PACKED_OLD, PACKED_NEW)


def normalize_selector_dtype(value):
    # FNUZ has a different bias and NaN encoding. No OCP kernel may accept it.
    if str(value) in ("torch.float8_e4m3fnuz", "float8_e4m3fnuz"):
        return "fp8_e4m3fnuz"
    return value


def guard_selector(r4d):
    if getattr(r4d, "_coherence_fp8_format_guard", False):
        return
    for name in ("select", "explain"):
        original = getattr(r4d, name)

        def wrapped(op, _original=original, **geometry):
            return _original(
                op,
                **{
                    key: normalize_selector_dtype(value)
                    for key, value in geometry.items()
                },
            )

        setattr(r4d, name, wrapped)
    r4d._coherence_fp8_format_guard = True


def validate_cache_dtype(dtype, query_dtype, out_dtype):
    if str(dtype) not in ("torch.float8_e4m3fn", "torch.uint8", "torch.bfloat16"):
        raise RuntimeError("R4D cache must be OCP E4M3 (or its uint8 storage) or BF16")
    if str(query_dtype) != "torch.bfloat16" or out_dtype != query_dtype:
        raise RuntimeError("R4D query/output must be BF16")
