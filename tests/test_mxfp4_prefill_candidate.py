"""The GPU test uses synthetic projections on a separately provided R9700."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "radiance-public"))
import mxfp4_prefill_candidate as candidate

torch = pytest.importorskip("torch")


def test_packed_layout_decodes_all_bits_and_nibbles_in_lane_order():
    # Basis words expose any collision, rotation across a word boundary, or sign loss.
    words = [1 << n for n in range(32)] + [b * 0x01010101 for b in range(256)]
    original = torch.tensor(
        [list(w.to_bytes(4, "little")) for w in words], dtype=torch.uint8
    )
    packed = candidate.pack_words(original, chunk_bytes=28)
    for w, encoded in zip(words, packed.tolist(), strict=True):
        psw = int.from_bytes(encoded, "little")
        rotated = (psw >> 4) | ((psw & 15) << 28)
        # These are the two half-word WMMA fragment inputs consumed by the kernel.
        decoded = []
        for half in (psw, rotated):
            for j in range(4):
                byte = (half >> (8 * j)) & 255
                decoded.append((byte & 7) | ((byte >> 4) & 8))
        assert decoded == [(w >> (4 * j)) & 15 for j in range(8)]
    assert packed.data_ptr() != original.data_ptr()


def test_signed_exponents_and_sentinels():
    refs = torch.tensor([120, 179, 44], dtype=torch.uint8)
    differences = torch.arange(-6, 9).unsqueeze(1).expand(-1, 3)
    scales = (refs - differences).to(torch.uint8)
    before = scales.clone()
    assert torch.equal(
        candidate.fold_indices(scales, refs), (differences + 6).to(torch.uint8)
    )
    assert torch.equal(scales, before)
    assert candidate.fold_indices(scales, refs[:2]) is None
    assert (
        candidate.fold_indices(scales, torch.tensor([0, 179, 44], dtype=torch.uint8))
        is None
    )
    for invalid in (111, 127, 255):
        scales[0, 0] = invalid
        assert candidate.fold_indices(scales, refs) is None


def test_native_generator_keeps_unmodified_control_and_fails_closed(tmp_path):
    from mxfp4_fold_precision import patched_sources

    hip = (ROOT / "radiance_mxfp4_fp8.hip").read_text()
    wrapper = (ROOT / "radiance_mxfp4.py").read_text()
    repaired = patched_sources(hip, wrapper)[0]
    begin = repaired.index("template <int TN, bool WPERM, int LBK = BK>")
    end = repaired.index(
        "// ---------------------------------------------------------------- decode path"
    )
    source = candidate.source_for_candidate(ROOT)
    assert (
        repaired[begin:end].replace(
            "radiance_mxfp4_fp8_gemm_atiled", "coherence_control_atiled"
        )
        in source
    )
    assert "if (k0 + LBK < K) fetch_weights(k0 + LBK);" in source
    (tmp_path / "radiance_mxfp4_fp8.hip").write_text(hip + "\n")
    (tmp_path / "radiance_mxfp4.py").write_text(wrapper)
    with pytest.raises(ValueError, match="pinned"):
        candidate.source_for_candidate(tmp_path)


@pytest.mark.skipif(
    not os.getenv("QWEN_MXFP4_PREFILL_TEST_BUILD"),
    reason="requires an isolated R9700 and candidate build directory",
)
def test_r9700_every_arm_matches_corrected_control():
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1201"
    launch = candidate.checked_launch(os.environ["QWEN_MXFP4_PREFILL_TEST_BUILD"])
    rng = torch.Generator(device="cuda").manual_seed(873849)
    make = {"device": "cuda", "dtype": torch.uint8, "generator": rng}
    for n, k in sorted(candidate.SHAPES):
        w = torch.randint(256, (n, k // 2), **make)
        refs = torch.randint(121, 130, (n,), **make)
        idx = torch.randint(15, (k // 32, n), **make)
        ws = (refs.to(torch.int16) + 6 - idx).to(torch.uint8)
        packed, folded = candidate.pack_words(w), candidate.fold_indices(ws, refs)
        for m, tn in ((65, 2), (256, 2), (2049, 4), (3296, 4)):
            # Physical tile layout is arbitrary; all arms receive the identical bytes.
            a = torch.randint(0, 120, (((m + 15) // 16) * 16, k), **make)
            scale = torch.rand(m, device="cuda", generator=rng)
            control = None
            for mode in range(4):
                weight, exponents = (packed, folded) if mode & 1 else (w, ws)
                out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
                stream = torch.cuda.current_stream().cuda_stream
                assert (
                    launch(
                        *(
                            t.data_ptr()
                            for t in (a, weight, exponents, refs, scale, out)
                        ),
                        m,
                        n,
                        k,
                        stream,
                        tn,
                        mode,
                    )
                    == 0
                )
                if mode == 0:
                    control = out
                else:
                    assert torch.equal(
                        out.view(torch.int16), control.view(torch.int16)
                    ), (m, n, mode)


@pytest.mark.parametrize(
    "preprocess,slab,mode",
    [(False, False, 0), (True, False, 1), (False, True, 0), (True, True, 3)],
)
def test_serving_dispatch_requires_preprocessed_weights_and_checks_all_source_pointers(
    preprocess, slab, mode
):
    ref_calls, native = [], []
    usage = {"preprocessed_calls": 0, "slab_ahead_calls": 0}
    w = torch.zeros(1)
    s = torch.zeros(1)
    key = 11, 12, 13, 16480, 5120
    op = candidate._make_dispatch(
        lambda *args: ref_calls.append(args),
        lambda *args: native.append(args) or 0,
        {key: (w, s) if preprocess else None},
        usage,
        slab,
        2048,
    )
    args = [10, 11, 12, 13, 14, 15, 2048, 16480, 5120, 0]
    for index, replacement in ((6, 64), (6, 4097), (1, 99), (2, 99), (3, 0), (8, 128)):
        changed = list(args)
        changed[index] = replacement
        op(*changed)
        assert ref_calls[-1] == tuple(changed)
    assert not native
    op(*args)
    if mode:
        assert native[-1][-2:] == (4, mode)
        assert native[-1][1:3] == (w.data_ptr(), s.data_ptr())
        assert usage == {
            "preprocessed_calls": int(preprocess),
            "slab_ahead_calls": int(slab),
        }
    else:
        assert ref_calls[-1] == tuple(args) and not native
        assert usage == {"preprocessed_calls": 0, "slab_ahead_calls": 0}
