"""Exercise the actual generated target attention split boundaries without a GPU."""

import hashlib
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "experiments/radiance-public"
PARENT = b"""template<int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT>
    const int ntl  = r4d_attn_tiles(ctx, TILE);
    const int pair = 0;
    const int query_tps = r4d_attn_tps(query_ctx, TILE, splits);
    const int u = r4d_attn_used(ctx, tile, splits);
"""


def test_tampered_source_is_rejected(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    import attention_runtime_splits as kernel

    with pytest.raises(ValueError, match="unexpected shared target attention"):
        kernel.source_for_candidate(PARENT)


@pytest.mark.parametrize("variant", ["combined", "shifts"])
def test_generated_schedule_preserves_native_tiles_at_every_short_prefix(
    tmp_path, monkeypatch, variant
):
    monkeypatch.syspath_prepend(str(ROOT))
    import attention_runtime_splits as kernel

    monkeypatch.setattr(kernel, "PARENT_SOURCE", hashlib.sha256(PARENT).hexdigest())
    source = kernel.optimized_source(PARENT, variant)
    law = source[: source.index("template<int")].replace(
        "__device__ __forceinline__", "static"
    )
    partition = source[
        source.index("    const int ntl") : source.index("    const int pair")
    ]
    local, merge = source[source.index("    const int query_tps") :].split(";\n", 1)
    local += ";"
    program = (
        r"""
#include <algorithm>
#include <cassert>
#include <vector>
using std::min;
using std::max;
constexpr int ceildiv(int n,int d) {return (n+d-1)/d;}
int r4d_attn_tiles(int ctx,int tile){return ceildiv(ctx,tile);}
int r4d_attn_tps(int ctx,int tile,int splits){return ceildiv(ceildiv(ctx,tile),splits);}
int r4d_attn_used(int ctx,int tile,int splits){return ceildiv(ceildiv(ctx,tile),r4d_attn_tps(ctx,tile,splits));}
int serial_splits(int tiles){ int n=32; while (n>16 && n>tiles/2) n/=2; return min(n,tiles); }
"""
        + law
        + r"""
void check(int prefix,int sp){
    const int first_ctx=prefix+1,ctx=prefix+8,splits=32,TILE=16;
    for(int position=first_ctx;position<=ctx;++position){
        int ctx=position, tile=16;
"""
        + merge
        + r"""
        int reference_tiles=ceildiv(position,TILE);
        assert(u==ceildiv(reference_tiles,ceildiv(reference_tiles,serial_splits(reference_tiles))));
    }
    // A missing physical group must have no active query in the oracle.
    bool any=false;
    for(int p=first_ctx;p<=ctx;++p){int t=ceildiv(p,TILE); any |= sp*ceildiv(t,serial_splits(t))<t;}
    bool ran=false;
    [&]{
"""
        + partition
        + r"""
    ran=true;
    for(int query_ctx=first_ctx;query_ctx<=ctx;++query_ctx){
        int query_ntl=ceildiv(query_ctx,TILE);
"""
        + local
        + r"""
        int tiles=ceildiv(query_ctx,TILE);
        int oracle_tps=ceildiv(tiles,serial_splits(tiles));
        assert(oracle_tps==query_tps);
        int lo=sp*query_tps, hi=min(lo+query_tps,tiles);
        if(lo>=hi) continue;
        // Group starts with empty registers, resets at a later start and emits at
        // the earlier/later end exactly as the two HIP write_partial sites do.
        int start=t_lo, writes=0;
        for(int t=t_lo;t<t_hi;++t){
            if(last_t_lo!=t_lo && t==last_t_lo && lo==last_t_lo) start=t;
            if(first_t_hi!=t_hi && t+1==first_t_hi && hi==first_t_hi){assert(start==lo);++writes;}
        }
        if(hi==t_hi){assert(start==lo);++writes;}
        assert(writes==1);
    }
    }();
    assert(!any||ran);
}
int main(){
    for(int p=0;p<=2100;++p)for(int s=0;s<32;++s)check(p,s);
    for(int p:{59999,60000,199999,200000,327944})for(int s=0;s<32;++s)check(p,s);
}
"""
    )
    path = tmp_path / "schedule.cpp"
    path.write_text(program)
    executable = tmp_path / "schedule"
    subprocess.run(
        ["c++", "-std=c++17", "-O2", str(path), "-o", str(executable)], check=True
    )
    subprocess.run([str(executable)], check=True)


def test_installer_requires_selected_parent_and_refuses_stacking(monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(ROOT))
    import install_attention_runtime_splits as candidate

    parent = object()
    replacement = object()
    attention = SimpleNamespace(
        manifest={"files": {"candidate.so": candidate.PARENT_BINARY}}, launch=parent
    )
    worker = SimpleNamespace(
        _qwen_performance_repairs=SimpleNamespace(attention=attention)
    )
    monkeypatch.setattr(candidate, "load", lambda directory: replacement)
    monkeypatch.setenv("QWEN_SPEED_TARGET_ATTN_BUILD", "/other")
    with pytest.raises(ValueError, match="independent candidate"):
        candidate.install(worker, tmp_path)
    assert attention.launch is parent
    monkeypatch.delenv("QWEN_SPEED_TARGET_ATTN_BUILD")
    candidate.install(worker, tmp_path)
    assert (
        attention.launch is replacement and attention._runtime_splits_original is parent
    )
    with pytest.raises(ValueError, match="already installed"):
        candidate.install(worker, tmp_path)
