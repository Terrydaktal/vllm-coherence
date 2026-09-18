# Regression adapted from the upstream PR recorded in ../manifest.json.
# This module is collected only by the explicit isolated-GPU probe runner.
import os
import pytest
if os.environ.get("QWEN_UPSTREAM_GPU_TESTS") != "1":
    pytest.skip("requires the isolated GPU qualification runner", allow_module_level=True)
import torch
import triton
import triton.language as tl

def is_hip():
    return torch.version.hip is not None

@pytest.fixture
def device():
    if not is_hip():
        pytest.fail("requires the pinned AMD runtime")
    return "cuda:0"
def _issue_11378_reference(topk_indices, is_valid, t2r, block_table, block_size):
    nt, topk = topk_indices.shape
    lens = torch.zeros(nt, dtype=torch.int32)
    indptr = torch.zeros(nt + 1, dtype=torch.int32)
    ragged = torch.full((nt * topk, ), -2, dtype=torch.int32)
    row_len = block_table.shape[1]
    running = 0
    for t in range(nt):
        idx = topk_indices[t]
        cnt = int((idx >= 0).sum().item())
        if is_valid[t].item() == 0:
            cnt = 0
        lens[t] = cnt
        req = int(t2r[t].item())
        for o in range(cnt):
            v = int(idx[o].item())
            if v >= 0 and (v // block_size) < row_len:
                bnum = int(block_table[req, v // block_size].item())
                ragged[running + o] = bnum * block_size + v % block_size
        running += cnt
        indptr[t + 1] = running
    return ragged, indptr, lens


def _issue_11378_run(pack_kernel, topk_indices, is_valid, t2r, block_table, block_size, static, num_warps):
    nt, topk = topk_indices.shape
    dev = topk_indices.device
    lens = torch.empty(nt, dtype=torch.int32, device=dev)
    indptr = torch.empty(nt + 1, dtype=torch.int32, device=dev)
    ragged = torch.full((nt * topk, ), -2, dtype=torch.int32, device=dev)
    pack_kernel[(1, )](topk_indices, topk_indices.stride(0), is_valid, t2r, block_table, block_table.stride(0), lens,
                       indptr, ragged, nt, block_size, topk=topk, TOPK_PAD=triton.next_power_of_2(topk), STATIC=static,
                       num_warps=num_warps)
    return ragged.cpu(), indptr.cpu(), lens.cpu()


@triton.jit
def _issue_11378_pack_kernel(topk_indices_ptr, topk_stride, is_valid_ptr, t2r_ptr, block_table_ptr, bt_stride, lens_ptr,
                             indptr_ptr, ragged_ptr, num_tokens, block_size, topk: tl.constexpr, TOPK_PAD: tl.constexpr,
                             STATIC: tl.constexpr):
    offs = tl.arange(0, TOPK_PAD)
    omask = offs < topk
    running = tl.zeros((), dtype=tl.int32)
    tl.store(indptr_ptr + 0, 0)
    if STATIC:
        for t in tl.static_range(0, 64):
            tt = tl.minimum(t, num_tokens - 1)
            active = t < num_tokens
            idx = tl.load(topk_indices_ptr + tt * topk_stride + offs, mask=omask & active, other=-1)
            valid_tok = tl.load(is_valid_ptr + tt)
            cnt = tl.sum((idx >= 0).to(tl.int32), axis=0)
            cnt = tl.where(valid_tok != 0, cnt, 0)
            tl.store(lens_ptr + tt, cnt, mask=active)
            out_len = cnt
            req = tl.load(t2r_ptr + tt)
            pmask = omask & (offs < out_len)
            vald = pmask & (idx >= 0)
            bidx = idx // block_size
            bnum = tl.load(block_table_ptr + req * bt_stride + bidx, mask=vald, other=0)
            slot = tl.where(vald, bnum * block_size + idx % block_size, -1)
            tl.store(ragged_ptr + running + offs, slot, mask=pmask)
            running = running + cnt
            tl.store(indptr_ptr + tt + 1, running, mask=active)
    else:
        for t in range(0, num_tokens):
            idx = tl.load(topk_indices_ptr + t * topk_stride + offs, mask=omask, other=-1)
            valid_tok = tl.load(is_valid_ptr + t)
            cnt = tl.sum((idx >= 0).to(tl.int32), axis=0)
            cnt = tl.where(valid_tok != 0, cnt, 0)
            tl.store(lens_ptr + t, cnt)
            out_len = cnt
            req = tl.load(t2r_ptr + t)
            pmask = omask & (offs < out_len)
            vald = pmask & (idx >= 0)
            bidx = idx // block_size
            bnum = tl.load(block_table_ptr + req * bt_stride + bidx, mask=vald, other=0)
            slot = tl.where(vald, bnum * block_size + idx % block_size, -1)
            tl.store(ragged_ptr + running + offs, slot, mask=pmask)
            running = running + cnt
            tl.store(indptr_ptr + t + 1, running)


@pytest.mark.parametrize("static", [False, True])
def test_for_runtime_trip_count_masked_pack(device, static):
    if not is_hip():
        pytest.skip("AMD runtime loop regression")
    # More warps fault on the affected compiler and poison the HIP context.
    num_warps = 1
    nt, topk, block_size, seq_len, nreq = 6, 512, 32, 11271, 3
    idx = torch.full((nt, topk), -1, dtype=torch.int32)
    for t in range(nt):
        nval = topk - (t % 3) * 7
        vals = torch.arange(nval, dtype=torch.int32) - nval + 1 + seq_len - 1
        vals = vals.clamp(min=0)
        if t % 2 == 1:
            vals[::5] = -1
        idx[t, :nval] = vals
    t2r = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32)
    row_len = (seq_len + block_size - 1) // block_size
    tab = torch.arange(100, 100 + nreq * row_len, dtype=torch.int32)
    tab = tab.reshape(nreq, row_len).contiguous()
    valid = torch.ones(nt, dtype=torch.int32)
    valid[1::2] = 0

    r_ref, i_ref, l_ref = _issue_11378_reference(idx, valid, t2r, tab, block_size)

    idx_d = idx.to(device)
    t2r_d = t2r.to(device)
    tab_d = tab.to(device)
    valid_d = valid.to(device)
    r_gpu, i_gpu, l_gpu = _issue_11378_run(_issue_11378_pack_kernel, idx_d, valid_d, t2r_d, tab_d, block_size, static,
                                           num_warps)
    total = int(i_ref[-1])
    torch.testing.assert_close(l_gpu, l_ref, atol=0, rtol=0)
    torch.testing.assert_close(i_gpu, i_ref, atol=0, rtol=0)
    torch.testing.assert_close(r_gpu[:total], r_ref[:total], atol=0, rtol=0)


