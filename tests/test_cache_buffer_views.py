import hashlib
import runpy
import sys
from pathlib import Path

import pytest
from test_radiance_chat_cache import BLOCK, load_tier, store

from qwen_r9700_lab import radiance_cache as cache


@pytest.mark.parametrize("corrupt", (False, True))
@pytest.mark.parametrize("compressed", (False, True))
def test_tier_verifies_buffer_view_before_touching_destination(
    monkeypatch, tmp_path, corrupt, compressed
):
    module = load_tier(monkeypatch)
    monkeypatch.setitem(sys.modules, "qwen_radiance_chat_tier", module)
    adapters = Path(__file__).resolve().parents[1] / "experiments/radiance-public"
    monkeypatch.syspath_prepend(str(adapters))
    cls = runpy.run_path(str(adapters / "cache_buffer_view_tier.py"))[
        "BufferViewChatFileSystemTierManager"
    ]
    assert issubclass(cls, module.ChatFileSystemTierManager)
    tier = cls.__new__(cls)
    current = store(tmp_path)
    key = bytes.fromhex("ab" * 32) + bytes(4)
    name = module.object_key(key)
    raw = (
        BLOCK
        if compressed
        else hashlib.shake_256(b"raw snapshot fixture").digest(len(BLOCK))
    )
    assert current.write(name, memoryview(raw))
    path = current.path(name)
    assert cache.HEADER.unpack_from(path.read_bytes())[0] == (
        cache.COMPRESSED if compressed else cache.RAW
    )
    destination = bytearray(len(raw))
    tier._primary_kv_view = memoryview(destination)
    tier._block_size = len(raw)
    tier._ram_blocks = lambda *_: {}
    if corrupt:
        damaged = bytearray(path.read_bytes())
        damaged[cache.HEADER.size - 1] ^= 1
        path.write_bytes(damaged)

    stock_decode = cache.decode_block
    seen = []

    def decode(data, expected_size):
        seen.append(data)
        return stock_decode(data, expected_size)

    monkeypatch.setattr(cache, "decode_block", decode)
    if corrupt:
        with pytest.raises(cache.SnapshotIntegrityError, match="checksum"):
            tier._load({"store": current}, [key], [0])
        assert destination == bytes(len(raw))
        assert not path.exists()
    else:
        tier._load({"store": current}, [key], [0])
        assert destination == raw
    assert len(seen) == 1
    assert isinstance(seen[0], memoryview)
    assert seen[0].readonly and type(seen[0].obj) is bytes
