import hashlib
import sys

import numpy as np
import pytest

from qwen_r9700_lab.conformance_radiance import gather_hybrid, gather_paged, install, verify_sources
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal


def test_worker_exception_preserves_original_cause_before_transport_translation(tmp_path):
    from qwen_r9700_lab.conformance_radiance import diagnosed_native_call
    from qwen_r9700_lab.diagnostic_contract import authenticate, private_json

    plan, binding = seal({"fixture": "plan"}), seal({"fixture": "binding"})
    failure = DiagnosticError("native committed position/prefix is invalid")

    def method(*, fail=False):
        if fail:
            raise failure
        return "original result"

    call = diagnosed_native_call(method, tmp_path, plan, binding, "sample_tokens")
    assert call() == "original result"
    assert not (tmp_path / "worker-error.json").exists()
    with pytest.raises(DiagnosticError) as caught:
        call(fail=True)
    assert caught.value is failure
    document = private_json(tmp_path / "worker-error.json")
    authenticate(document)
    assert document["plan"] == plan["sha256"]
    assert document["binding"] == binding["sha256"]
    assert document["type"] == "DiagnosticError"
    assert document["message"] == str(failure)
    assert "raise failure" in document["traceback"]
    first = (tmp_path / "worker-error.json").read_bytes()
    failure = RuntimeError("secondary teardown failure")
    with pytest.raises(RuntimeError):
        call(fail=True)
    assert (tmp_path / "worker-error.json").read_bytes() == first


@pytest.mark.parametrize("accepted", range(1, 9))
@pytest.mark.parametrize("dim_first", [False, True])
def test_physical_state_versions_and_conv_windows(accepted, dim_first):
    table = np.asarray([11, 7, 0, 14, 1, 6, 3, 9, 2], np.int32)
    states = np.arange(16 * 2 * 3 * 4, dtype=np.float32).reshape(16, 2, 3, 4)
    conv = np.arange(16 * 5 * 10, dtype=np.float32).reshape(16, 5, 10)
    stored = conv if dim_first else conv.transpose(0, 2, 1)
    state, history = gather_hybrid(
        stored,
        states,
        table,
        running_column=1,
        accepted_count=accepted,
        history_width=3,
        dim_first=dim_first,
    )
    np.testing.assert_array_equal(state, states[table[accepted]])
    np.testing.assert_array_equal(history, conv[7, :, accepted - 1 : accepted + 2])
    # Mutating rejected banks cannot affect a selected committed version.
    copy = states.copy()
    for bank in set(table.tolist()) - {int(table[accepted])}:
        copy[bank] = -1000
    selected, _ = gather_hybrid(
        stored,
        copy,
        table,
        running_column=1,
        accepted_count=accepted,
        history_width=3,
        dim_first=dim_first,
    )
    np.testing.assert_array_equal(selected, state)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_r4d_page_addresses_are_not_part_of_logical_equality(dtype):
    logical = np.arange(17 * 2 * 8, dtype=dtype).reshape(17, 2, 8)
    results = []
    for table in (np.asarray([4, 1, 3]), np.asarray([0, 2, 1])):
        cache = np.zeros((5, 2, 8, 8), dtype)
        for token in range(17):
            cache[table[token // 8], :, token % 8, :] = logical[token]
        results.append(gather_paged(cache, table, 17, block_size=8, kv_heads=2, head_dim=4))
    for a, b in zip(results[0], results[1], strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(results[0][0], logical[..., :4])
    np.testing.assert_array_equal(results[0][1], logical[..., 4:])


@pytest.mark.parametrize(
    "table", [np.asarray([1, 1]), np.asarray([-1, 0]), np.asarray([100, 0]), np.asarray([1])]
)
def test_bad_page_ownership_is_not_numerical_tolerance(table):
    with pytest.raises(DiagnosticError):
        gather_paged(
            np.zeros((3, 2, 8, 8), np.uint8), table, 10, block_size=8, kv_heads=2, head_dim=4
        )


def test_source_binding_and_unarmed_install_fail_without_gpu_imports(tmp_path, monkeypatch):
    path = tmp_path / "native.py"
    path.write_text("original source")
    binding = seal(
        {
            "schema": "urn:qwen:radiance-native-binding:v1",
            "files": {"native.py": hashlib.sha256(path.read_bytes()).hexdigest()},
        }
    )
    assert verify_sources(tmp_path, binding) == binding["sha256"]
    path.write_text("future optimization")
    with pytest.raises(DiagnosticError, match="source changed"):
        verify_sources(tmp_path, binding)
    before = set(sys.modules)
    monkeypatch.delenv("QWEN_CONFORMANCE_GPU", raising=False)
    with pytest.raises(DiagnosticError, match="not armed"):
        install(None, "unused", "unused", "unused")
    assert not ({"torch", "vllm", "triton"} & (set(sys.modules) - before))


@pytest.mark.parametrize(
    "dtype,words,kv_encoding",
    [
        ("torch.bfloat16", [0x8000, 0x7FC1, 0x3F80], None),
        ("torch.float8_e4m3fn", [128, 127, 56], None),
        ("torch.uint8", [128, 127, 56], "fp8_e4m3fn"),
    ],
)
def test_native_storage_export_preserves_bits_without_float_conversion(
    monkeypatch, dtype, words, kv_encoding
):
    from types import SimpleNamespace

    from qwen_r9700_lab.conformance_radiance import as_cpu

    # A CPU stand-in implements only the storage-export tensor surface. Importing
    # or using actual Torch/HIP is forbidden for this test.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(uint8="u1", uint16="<u2"))
    storage = np.array(words, dtype="<u2" if "bfloat16" in dtype else "u1")

    class Tensor:
        def __init__(self):
            self.dtype = dtype
            self.encoding = None

        def detach(self):
            return self

        def view(self, encoding):
            self.encoding = encoding
            return self

        def contiguous(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return storage.view(self.encoding)

        def float(self):
            pytest.fail("storage capture rounded floating-point state")

    result = as_cpu(Tensor(), storage=True, **({"kv_encoding": kv_encoding} if kv_encoding else {}))
    assert result.tobytes() == storage.tobytes()
    assert not np.shares_memory(result, storage)


@pytest.mark.parametrize(
    "cache_dtype,storage_dtype,reference_encoding,valid",
    [
        ("fp8", "torch.uint8", "fp8_e4m3fn", True),
        ("fp8_e4m3", "torch.uint8", "fp8_e4m3fn", True),
        ("fp8", "torch.float8_e4m3fn", "fp8_e4m3fn", True),
        ("auto", "torch.bfloat16", "bf16", True),
        ("bfloat16", "torch.bfloat16", "bf16", True),
        ("fp8_e5m2", "torch.uint8", "fp8_e4m3fn", False),
        ("fp8", "torch.uint8", "bf16", False),
        ("auto", "torch.uint8", "fp8_e4m3fn", False),
        ("fp8", "torch.float8_e4m3fnuz", "fp8_e4m3fn", False),
        ("fp8_per_token_head", "torch.uint8", "fp8_e4m3fn", False),
        ("auto", "torch.float16", "bf16", False),
    ],
)
def test_native_kv_encoding_distinguishes_format_from_backing_bytes(
    cache_dtype, storage_dtype, reference_encoding, valid
):
    from qwen_r9700_lab.conformance_radiance import validate_native_kv_encoding

    if valid:
        validate_native_kv_encoding(cache_dtype, storage_dtype, reference_encoding)
    else:
        with pytest.raises(DiagnosticError, match="native KV encoding"):
            validate_native_kv_encoding(cache_dtype, storage_dtype, reference_encoding)
