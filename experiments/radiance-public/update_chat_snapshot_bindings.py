"""Refresh the chat-storage ABI and launcher hashes after reviewing source edits."""

import hashlib
import json
import re
from pathlib import Path

from patch_draft_head_initialization import POSTIMAGE as DRAFT_POSTIMAGE
from patch_draft_head_initialization import PREIMAGE as DRAFT_PREIMAGE
from patch_draft_head_initialization import UPSTREAM as DRAFT_UPSTREAM
from patch_gdn_extreme_decay import LIBRARY_SHA256, PREIMAGE, SOURCE_SHA256

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads((BASE / "snapshot-abi.json").read_text())
profile = json.loads((BASE / "runtime-radiance-1.0.16.json").read_text())
prior_path = BASE / "snapshot-abi-chat-cache-v1.json"
prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
manifest["schema"] = "qwen-radiance-durable-hybrid-snapshot-abi-v4"
patch_sha = sha(BASE / "patch_streaming_snapshot.py")
manifest["runtime"] = {
    "image": profile["image"],
    "image_digest": profile["image_digest"],
    "image_id": profile["image_id"],
    "source_commit": profile["source_commit"],
    "package_versions": profile["package_versions"],
    "kernel_hashes": profile["kernel_hashes"],
    "kernel_environment": profile["kernel_environment"],
    "deterministic_offload_hashes": {"PYTHONHASHSEED": "0"},
    "bounded_sliding_lookup_and_settled_tail_patch": {
        "path": "patch_streaming_snapshot.py",
        "sha256": patch_sha,
    },
    "release_files": {
        name: sha(BASE / name)
        for name in (
            "bootstrap_radiance_release.py",
            "qualified_speed_release.py",
            "qualified-speed-release.json",
            "speed_candidate_worker.py",
            "runtime-radiance-1.0.16.json",
            "radiance_request_guard.py",
            "patch_verify_head_memory.py",
            "patch_dflash_sampling_rng.py",
            "patch_draft_head_initialization.py",
            "patch_gdn_initial_prefill.py",
            "patch_gdn_extreme_decay.py",
            "gdn_extreme_decay_reference.hip",
            "qwen-fixed-v22.3.jinja",
        )
    },
}
manifest["runtime"]["draft_head_initialization"] = {
    "installer": "patch_draft_head_initialization.py",
    "upstream_commit": DRAFT_UPSTREAM,
    "source_preimage_sha256": DRAFT_PREIMAGE,
    "source_postimage_sha256": DRAFT_POSTIMAGE,
    "contract": (
        "Always defer INT2 draft-head packing until the real shared lm_head is passed to first use; "
        "unchanged quantizer, target arithmetic and snapshot data layout."
    ),
    "verification": (
        "CPU lifecycle and installed-source regressions; no new GPU throughput or "
        "full-model numerical qualification claimed."
    ),
}
manifest["runtime"]["qualified_speed"] = json.loads(
    (BASE / "qualified-speed-release.json").read_text()
)
if profile.get("optimized_d7"):
    manifest["runtime"]["optimized_d7"] = profile["optimized_d7"]
    for name in ("optimized-release.json", "optimized_pi_release.py"):
        manifest["runtime"]["release_files"][name] = sha(BASE / name)
    if profile["optimized_d7"].get("m1_arithmetic"):
        for name in (
            "m1_arithmetic_release.py",
            "mxfp4_fold_precision.py",
            "patch_gdn_stable_softplus.py",
            "probe_m1_arithmetic_repairs.py",
            "stock_gdn_scan_kernel.py",
        ):
            manifest["runtime"]["release_files"][name] = sha(BASE / name)
    if profile["optimized_d7"].get("attention_precision"):
        for name in profile["optimized_d7"]["attention_precision"]["sources"]:
            manifest["runtime"]["release_files"][name] = sha(BASE / name)
    if profile["optimized_d7"].get("target_head", {}).get("mode") in (
        "global256",
        "global512",
    ):
        manifest["runtime"]["release_files"]["radiance_verifyhead_global.py"] = sha(
            BASE / "radiance_verifyhead_global.py"
        )
rocr_path = BASE / "rocr-poll-backoff/runtime.json"
if rocr_path.exists():
    # Keep the already-qualified CPU backoff when refreshing scheduler bindings.
    manifest["runtime"]["rocr_poll_backoff"] = json.loads(rocr_path.read_text())
    manifest["runtime"]["release_files"]["rocr-poll-backoff/runtime.json"] = sha(
        rocr_path
    )
manifest["runtime"]["draft_sampling"] = {
    "installer": "patch_dflash_sampling_rng.py",
    "upstream_pr": "https://github.com/vllm-project/vllm/pull/54282",
    "upstream_commit": "fe755c88995ad468882517b6c4bdd60138d46a3a",
    "contract": (
        "independent proposal and residual Gumbel streams using a 1<<30 proposal "
        "position salt; unchanged greedy sampling, KV computation and snapshot layout"
    ),
}
prefill_math = {
    "implementation": "gdn_bounded_recurrence_extreme_decay_v1",
    "decay_span_threshold": 128,
    "source_sha256": SOURCE_SHA256,
    "binary_sha256": LIBRARY_SHA256,
}
manifest["runtime"]["gdn_prefill_correction"] = {
    **prefill_math,
    "installer": "patch_gdn_extreme_decay.py",
    "source_preimage_sha256": PREIMAGE,
    "host_gpu_synchronization": False,
}
manifest["runtime"]["chat_storage"] = {
    "format": "qwen-chat-cache-v1",
    "tier": "qwen_chat_fs",
    "modules": {
        p.name: sha(p)
        for p in (
            ROOT / "src/qwen_r9700_lab/radiance_cache.py",
            ROOT / "src/qwen_r9700_lab/radiance_memory.py",
            BASE / "radiance_chat_tier.py",
            BASE / "radiance_fair_scheduler.py",
            BASE / "patch_chat_snapshot.py",
        )
    },
}
manifest["runtime"]["memory_report"] = {
    "schema": "urn:qwen-r9700:radiance-memory:v1",
    "interval_seconds": 1,
    "directory": "/dev/shm/qwen-radiance-memory-v1",
    "worker_preimage_sha256": profile["source_preimages"][
        "vllm/v1/worker/gpu_worker.py"
    ],
    "contract": (
        "one shared CPU counter sampler; bounded startup/on-request storage inventory; "
        "no GPU work, tensor values, synchronization, cache clearing, or peak resets"
    ),
    "allocation_map": {
        "schema": "urn:qwen-r9700:radiance-allocation-map:v1",
        "request_interval_seconds": 10,
        "contract": (
            "on-request allocator snapshot with include_traces=False; numeric segment/block sizes; "
            "best-effort fixed-root storage owners; no addresses, trace history or tensor values; "
            "separate private/default/unknown pools; bounded analysis and overwritten tmpfs output"
        ),
    },
    "compatible_runtime_abis": [
        "a0fbc562276e24fcce8ddf48d71634920ec722e27dedeb3b1e78995a3da34832",
        # Candidate-only M1 bootstrap support does not alter the current serving
        # profile. A future M1 deployment changes data_abi and cannot reuse this
        # running predecessor even when its runtime identifier is listed here.
        "2fcc0356f7108572673b38e95c067cfa6c657b0ae0229b3a32ce256f76a6ad01",
    ],
}
manifest["runtime"]["buffered_tool_usage"] = {
    "installer": "patch_chat_snapshot.py",
    "serving_preimage_sha256": profile["source_preimages"][
        "vllm/entrypoints/openai/chat_completion/serving.py"
    ],
    "contract": "continuous per-choice usage during parser buffering; preserve raw-token privacy",
}
manifest["runtime"]["fair_scheduler"] = {
    "class": "qwen_radiance_fair_scheduler.FairScheduler",
    "policy": (
        "uninterrupted responses; least-recently-served waiting chat at completion; "
        "handover overlaps client-side tool execution; reserve selected response through "
        "cache admission until completion or cancellation; swaps alone do not count as service; "
        "2-second grace for completed tool calls; 30-second queue-age override at next boundary; "
        "equal priorities preserve this policy; higher priority 1 reserves the whole answer; "
        "higher priority 2 parks a response at the next safe synchronous step"
    ),
    "maximum_cached_chats": 2,
    "pinned_host_page_policy": (
        "MADV_NOHUGEPAGE on complete anonymous backing mappings after pinned allocation "
        "and before DMA; prevent khugepaged HSA queue invalidation; retain allocator ownership "
        "and existing snapshot data ABI"
    ),
    "runner_state_slots": 2,
    "maximum_dispatched_sequences": 1,
    "answer_priority": {
        "levels": [0, 1, 2],
        "default": 0,
        "lease_seconds": 60,
        "heartbeat_seconds": 10,
        "release": "Pi agent_settled, shutdown, session switch or lease expiry",
        "control": "bounded HTTP request and engine utility IPC; no inference or prompt changes",
        "state": "park without vLLM preemption; retain V2 request slot and bank allocator",
    },
    "tool_handover": {
        "grace_seconds": 2,
        "maximum_deferral_seconds": 30,
        "outcome_acknowledgement_timeout_seconds": 0.25,
        "contract": (
            "parser outcome through local engine IPC; no text; no blocking of client stream"
        ),
        "engine_core_preimage_sha256": (
            profile["source_preimages"]["vllm/v1/engine/core.py"]
        ),
    },
    "handover": (
        "flush pending stores; discard a superseded same-chat generation in place; "
        "otherwise swap packed GPU state; retain V2 request state"
    ),
    "residency_telemetry": (
        "content-free chat and generation hashes, useful and allocated pinned bytes, "
        "and separate allocation and transfer timings"
    ),
    "request_phase_telemetry": {
        "schema": "urn:qwen-r9700:request-phases:v2",
        "counter_interval_seconds": 0.5,
        "transitions": (
            "immediate tmpfs publication; shared inotify/SSH feed; local file notifications"
        ),
        "retention": (
            "last numeric response timings plus cumulative generation-round and "
            "draft/accepted counters per generation; at most 16 generations"
        ),
        "contract": (
            "no token IDs, text, tensors, GPU synchronization, or extra model requests; "
            "round duration is content-free wall time between consecutive generation "
            "updates for one request"
        ),
    },
}
manifest["serving"]["max_num_seqs"] = 2
manifest["storage"]["secondary_tier"] = "qwen_chat_fs"
manifest["serving"]["effective_chat_template_sha256"] = profile["chat_template_sha256"]
manifest["serving"]["cudagraph_mode"] = "PIECEWISE"
manifest["serving"]["target_verify_head"] = (
    profile["kernel_environment"]["RADIANCE_VERIFY_HEAD"] == "1"
)
manifest["serving"]["draft_rerank"] = 80
manifest["serving"]["r4d_attention_fp8"] = 0
# A new vLLM/kernel stack must not reinterpret old binary state. Keep this data
# contract independent of telemetry/installer source hashes for future repairs.
data_contract = {
    "schema": "qwen-radiance-release-data-v1",
    "image_digest": profile["image_digest"],
    "kernel_environment": profile["kernel_environment"],
    "target": manifest["target"],
    "drafter": manifest["drafter"],
    "serving": manifest["serving"],
    "layout": "qwen-chat-cache-v1; complete-window-plus-eagle-page; retained-mamba-pages=3",
}
previous_data_abi = hashlib.sha256(
    json.dumps(data_contract, sort_keys=True).encode()
).hexdigest()
data_contract["prefill_math"] = prefill_math
if profile.get("optimized_d7"):
    data_contract["optimized_arithmetic"] = profile["optimized_d7"]["arithmetic"]
    from optimized_pi_release import compatible_output_head_contract

    effective_data_contract = data_contract
    data_contract = compatible_output_head_contract(
        prior.get("storage", {}).get("data_contract"), effective_data_contract
    )
    if data_contract != effective_data_contract:
        manifest["storage"]["output_head_compatibility"] = (
            "Retain the prior canonical KV contract: only output-head selection changed; "
            "the backbone, state layout and processed-prefix identity are unchanged. "
            "The effective output head is recorded in runtime.optimized_d7.target_head."
        )
data_abi = hashlib.sha256(
    json.dumps(data_contract, sort_keys=True).encode()
).hexdigest()
manifest["storage"]["data_abi"] = data_abi
manifest["storage"]["data_contract"] = data_contract
prior_storage = prior.get("storage", {})
manifest["storage"]["previous_data_abi"] = (
    prior_storage.get("previous_data_abi", previous_data_abi)
    if prior_storage.get("data_abi") == data_abi
    else prior_storage.get("data_abi", previous_data_abi)
)
manifest["storage"]["migration"] = (
    "retain previous ABI for rollback; rebuild each resumed chat once"
)
manifest["storage"]["recurrent_publication_contract"] = (
    "complete-window-plus-eagle-page; retain-three-prior-aligned-mamba-states-until-offload; "
    "buffer-at-most-15-changing-blocks-in-RAM"
)
manifest["storage"]["recurrent_publication"] = (
    "immutable-full-attention-write-once; 8192-token-interval settled-tail journal"
)
manifest["storage"]["chat_snapshots"] = {
    "isolation": "session identity and compaction generation cache_salt",
    "compression": "lossless Zstd level 1; raw fallback; SHA256 verification",
    "retention": (
        "latest complete head plus its prior verified fallback until a successor is published"
    ),
    "failed_publication": (
        "retain previous head; collect abandoned writes; persist failure metadata"
    ),
    "garbage_collection": "durable intent; retry failed deletion; recover on engine startup",
    "compaction": (
        "force old-generation tail flush before commit; activate successor; retain prior complete "
        "generation until successor publication"
    ),
    "tail_journal": (
        "at most 15 changing blocks per chat in system RAM; 8192-token automatic flush; "
        "6-GiB/5-chat bound"
    ),
    "forced_tail_flush": "RAM eviction, clean backend shutdown, compaction, and explicit request",
    "stale_generation": (
        "retain isolated cache salt; bypass durable tier without reactivation or engine failure"
    ),
    "publication_transaction": (
        "previous disk head remains authoritative through tail write and full-head verification"
    ),
    "write_accounting": (
        "persistent per-chat lower-bound counters plus backing-drive SMART telemetry"
    ),
    "legacy": "unlabelled requests bypass the durable snapshot tier",
}
manifest_path = BASE / "snapshot-abi-chat-cache-v1.json"
if profile.get("optimized_d7", {}).get(
    "attention_boundary_qualification"
) and not profile["optimized_d7"].get("m1_arithmetic"):
    manifest["storage"]["attention_boundary_compatibility"] = (
        "Retain the declared corrected M1 arithmetic and existing nine-slot state layout. "
        "The shared M8 attention traversal changes work sharing across page boundaries; "
        "exact sampled operator output and natural Pi output agreement are bound in "
        "runtime.optimized_d7.attention_boundary_qualification. Existing processed-prefix "
        "snapshots remain compatible; this is sampled evidence, not a universal equivalence proof."
    )
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
abi = sha(manifest_path)
launcher = BASE / "launch_public_clean_snapshot_server.sh"
text = launcher.read_text()
text = re.sub(r"readonly abi_id=[0-9a-f]{64}", f"readonly abi_id={abi}", text)
text = re.sub(r"readonly data_abi=[0-9a-f]{64}", f"readonly data_abi={data_abi}", text)
text = re.sub(
    r"readonly expected_patch_sha256=[0-9a-f]{64}",
    f"readonly expected_patch_sha256={patch_sha}",
    text,
)
launcher.write_text(text)
frontend = ROOT / "scripts/pi-remote-qwen-radiance"
text = frontend.read_text()
for key, value in {
    "SNAPSHOT_ABI": abi,
    "SNAPSHOT_DATA_ABI": data_abi,
    "IMAGE_ID": profile["image_id"],
    "SNAPSHOT_PATCH_SHA256": patch_sha,
    "REMOTE_LAUNCHER_SHA256": sha(launcher),
}.items():
    text = re.sub(rf"readonly {key}=[0-9a-f]{{64}}", f"readonly {key}={value}", text)
text = re.sub(
    r"readonly REMOTE_ROOT=\S+",
    "readonly REMOTE_ROOT=/home/lewis/projects/r9700-radiance-1.0.16-20260913",
    text,
)
frontend.write_text(text)
print(abi)
