"""Run real vLLM filesystem thread-pool store/restart/load checks without a GPU."""

# Install the runtime modules before importing them below.
# ruff: noqa: E402

import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from patch_chat_snapshot import install

source = Path(__file__).parent
package = Path("/opt/vllm/lib/python3.12/site-packages")
install(package, source / "radiance_cache.py", source / "radiance_chat_tier.py")

from qwen_radiance_cache import ChatStore, report
from qwen_radiance_chat_tier import ChatFileSystemTierManager, object_key
from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.tiering.base import JobMetadata

config = SimpleNamespace(
    groups=[SimpleNamespace(tokens_per_block=1648, layer_names=["test.self_attn"])],
    model=SimpleNamespace(name="qualification", dtype="fp8"),
    cache=SimpleNamespace(tokens_per_hash=1648),
    parallel=SimpleNamespace(
        tp_size=1, pp_size=1, pcp_size=1, dcp_size=1, rank=0, is_parallelism_agnostic=False
    ),
    replicated_layout=False,
)
spec = SimpleNamespace(config=config, blocks_per_chunk=1)
original = np.arange(2 * 1024 * 1024, dtype=np.uint8).reshape(2, -1)
buffer = original.copy()
key = make_offload_key(hashlib.sha256(b"qualification-key").digest(), 6)
chat = {
    "id": hashlib.sha256(b"qualification-chat").hexdigest(),
    "generation": hashlib.sha256(b"initial").hexdigest(),
    "title": "Qualification",
}


def make_tier(root):
    return ChatFileSystemTierManager(
        offloading_spec=spec,
        primary_kv_view=memoryview(buffer),
        tier_type="qwen_chat_fs",
        root_dir=str(root),
        n_read_threads=2,
        n_write_threads=2,
        control_directory=str(root / "control"),
        tail_status_path=str(root / "tail-status.json"),
    )


with tempfile.TemporaryDirectory(prefix="qwen-chat-qualification-") as directory:
    root = Path(directory)
    tier = make_tier(root)
    request = ReqContext("first", {"qwen_chat": chat})
    tier.on_new_request(request)
    tier._chat_requests[request.req_id].update(
        head=([object_key(key)], 1648),
        tail_keys={object_key(key)},
        force_flush=False,
    )
    tier.submit_store(JobMetadata(1, [key], np.array([0]), False, request))
    tier.on_request_finished(request)
    tier.drain_jobs()
    result = tier.get_finished_jobs()
    assert len(result) == 1 and result[0].success
    tier._publish_ready()
    assert not ChatStore(root, chat).path(object_key(key)).exists()
    assert tier._force_identity(chat)["status"] == "flushed"
    assert report(root)["chats"][0]["status"] == "ready"
    tier.shutdown()

    # A fresh manager and a zeroed CPU arena must restore exactly from Zstd.
    buffer[:] = 0
    tier = make_tier(root)
    request = ReqContext("restart", {"qwen_chat": chat})
    tier.on_new_request(request)
    assert tier._lookup_manager.batch_lookup([key], request) == [True]
    tier.submit_load(JobMetadata(2, [key], np.array([1]), True, request))
    tier.drain_jobs()
    result = tier.get_finished_jobs()
    assert len(result) == 1 and result[0].success
    assert np.array_equal(buffer[1], original[0])

    old_generation = ChatStore(root, chat).generation
    next_chat = {**chat, "generation": hashlib.sha256(b"compacted").hexdigest()}
    retired = ChatStore(root, next_chat).activate()
    assert retired["retained_file_bytes"] > 0
    assert old_generation.exists()
    assert tier._lookup_manager.batch_lookup([key], request) == [False]
    tier.on_request_finished(request)
    tier.shutdown()

    # The prior generation remains a valid fallback until its replacement has
    # been written, verified, and atomically published.
    buffer[:] = original
    tier = make_tier(root)
    request = ReqContext("compacted", {"qwen_chat": next_chat})
    tier.on_new_request(request)
    tier._chat_requests[request.req_id].update(
        head=([object_key(key)], 1648),
        tail_keys={object_key(key)},
        force_flush=True,
    )
    tier.submit_store(JobMetadata(3, [key], np.array([0]), False, request))
    tier.on_request_finished(request)
    tier.drain_jobs()
    result = tier.get_finished_jobs()
    assert len(result) == 1 and result[0].success
    tier._publish_ready()
    assert not old_generation.exists()
    tier.shutdown()
    print(
        json.dumps(
            {
                "real_vllm_threads": "pass",
                "ram_tail_then_explicit_flush": "pass",
                "restart_byte_equality": "pass",
                "verified_successor_retirement": "pass",
            }
        )
    )
