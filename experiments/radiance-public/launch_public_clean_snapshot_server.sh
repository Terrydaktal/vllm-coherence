#!/usr/bin/env bash

set -euo pipefail

readonly launcher_root=/home/lewis/projects/r9700-radiance-1.0.16-20260913
readonly cache_root=/home/lewis/.cache/qwen-radiance-public-clean-snapshot-v1
readonly abi_id=39ba7b315d31edeb05c2fe31f05989920ffd22d6b25c325d3d276ae244a680ab
readonly data_abi=d5ca655a9121c9207dd638fa2ed927b8f15ad16ccc407e33c22c4f8cf10f396f
readonly snapshot_root="${cache_root}/snapshots/${abi_id}"
readonly expected_patch_sha256=d3f67e813275bf8331e2d586e74c6e466307d39f0953396431c8c00c501259dd
readonly model_id=qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate
readonly container_name=${QWEN_QUALIFICATION_CONTAINER:-qwen38-27b-uncensored-mxfp4-public-snapshot-candidate}
readonly port=${QWEN_QUALIFICATION_PORT:-8080}
readonly image=docker.io/magiccodingman/vllm-radiance@sha256:83a9dc02a8f8e75aabe81366d36ebaa2e35fcbe181cacf8e8e0a4cef4ebccbcc
readonly required_shm_bytes=19327352832
readonly required_snapshot_free_bytes=12884901888
readonly optimized_root=/home/lewis/.local/share/qwen-r9700/optimized-pi/20260924-norm-consistency-v2

cd "$launcher_root"

[[ $(sha256sum radiance-vllm-mxfp4/patch_streaming_snapshot.py | awk '{print $1}') == "$expected_patch_sha256" ]] || {
	printf 'snapshot runtime patch differs from the snapshot ABI\n' >&2
	exit 1
}
[[ -f $snapshot_root/abi.json && ! -L $snapshot_root/abi.json ]] || {
	printf 'snapshot ABI manifest is missing or unsafe: %s\n' "$snapshot_root/abi.json" >&2
	exit 1
}
[[ $(sha256sum "$snapshot_root/abi.json" | awk '{print $1}') == "$abi_id" ]] || {
	printf 'snapshot ABI manifest hash differs from its namespace\n' >&2
	exit 1
}
[[ $(jq -r '.storage.data_abi' "$snapshot_root/abi.json") == "$data_abi" &&
-d $cache_root/snapshots/$data_abi/data && ! -L $cache_root/snapshots/$data_abi/data ]] || {
	printf 'compatible snapshot data namespace is missing or unsafe\n' >&2
	exit 1
}

while IFS=$'\t' read -r module expected; do
	path="radiance-vllm-mxfp4/$module"
	[[ -f $path && ! -L $path && $(sha256sum "$path" | awk '{print $1}') == "$expected" ]] || {
		printf 'chat snapshot module differs from its ABI: %s\n' "$path" >&2
		exit 1
	}
done < <(jq -r '(.runtime.chat_storage.modules + .runtime.release_files) | to_entries[] | [.key, .value] | @tsv' "$snapshot_root/abi.json")

[[ -d $optimized_root && ! -L $optimized_root &&
	$(sha256sum "$optimized_root/optimized-release.json" | awk '{print $1}') == $(jq -r '.optimized_d7.manifest_sha256' radiance-vllm-mxfp4/runtime-radiance-1.0.16.json) ]] || {
	printf 'optimized serving payload is missing or differs from its qualified manifest\n' >&2
	exit 1
}

rocr_sha256=$(jq -er '.runtime.rocr_poll_backoff.library_sha256' "$snapshot_root/abi.json")
[[ $rocr_sha256 =~ ^[0-9a-f]{64}$ ]] || {
	printf 'invalid ROCr polling-backoff library identity\n' >&2
	exit 1
}
rocr_library="/home/lewis/.local/share/qwen-r9700/overlays/rocr-poll-backoff/$rocr_sha256/libhsa-runtime64.so.1.21.0"
[[ -f $rocr_library && ! -L $rocr_library && $(sha256sum "$rocr_library" | awk '{print $1}') == "$rocr_sha256" ]] || {
	printf 'ROCr polling-backoff library is missing or differs from its pinned build\n' >&2
	exit 1
}

[[ $(podman image inspect --format '{{.Id}}' "$image") == $(jq -r '.runtime.image_id' "$snapshot_root/abi.json") ]] || {
	printf 'installed release image differs from the snapshot ABI\n' >&2
	exit 1
}
if podman container exists "$container_name"; then
	printf 'backend container already exists: %s\n' "$container_name" >&2
	exit 1
fi

# The offload arena is disposable process memory, not durable snapshot data.
# A killed container can leave its named mmap behind; reusing or populating it
# without an owner check either attaches to stale state or fails only after the
# model has loaded. Runtime-only ABI changes create a new engine ID, so retire
# every exact, unowned arena for this lane rather than checking only the new ID.
shopt -s nullglob
offload_mmaps=(/dev/shm/vllm_offload_qwen-radiance-public-clean-*.mmap)
shopt -u nullglob
((${#offload_mmaps[@]} <= 8)) || {
	printf 'too many Radiance offload arenas to retire safely: %s\n' "${#offload_mmaps[@]}" >&2
	exit 1
}
for offload_mmap in "${offload_mmaps[@]}"; do
	[[ ${offload_mmap##*/} =~ ^vllm_offload_qwen-radiance-public-clean-[0-9a-f]{16}\.mmap$ &&
		-f $offload_mmap && ! -L $offload_mmap ]] || {
		printf 'offload mmap is not an exact regular lane file: %s\n' "$offload_mmap" >&2
		exit 1
	}
	[[ $(stat -c '%a:%u:%h' -- "$offload_mmap") == "600:$(id -u):1" ]] || {
		printf 'offload mmap ownership/mode/link contract is invalid: %s\n' "$offload_mmap" >&2
		exit 1
	}
	[[ -z $(fuser "$offload_mmap" 2>/dev/null || true) ]] || {
		printf 'offload mmap is still owned by a live process: %s\n' "$offload_mmap" >&2
		exit 1
	}
	[[ -z $(find /proc/[0-9]*/fd -lname "$offload_mmap" -print -quit 2>/dev/null || true) ]] || {
		printf 'offload mmap still has a live file descriptor: %s\n' "$offload_mmap" >&2
		exit 1
	}
	rm -- "$offload_mmap"
	printf 'retired unowned transient offload arena: %s\n' "$offload_mmap" >&2
done

for status_file in /dev/shm/qwen-radiance-fair-public-scheduler.json /dev/shm/qwen-radiance-fair-public-worker.json /dev/shm/qwen-radiance-snapshot-tail.json; do
	if [[ -e $status_file || -L $status_file ]]; then
		[[ -f $status_file && ! -L $status_file ]] || {
			printf 'scheduler status path is unsafe: %s\n' "$status_file" >&2
			exit 1
		}
		rm -- "$status_file"
	fi
done

# A just-stopped container can release its unlinked mmap pages a fraction after
# podman returns. Give tmpfs a bounded settle window before declaring capacity
# failure; the normal already-free path exits this loop on its first sample.
available_shm_bytes=0
for _ in {1..100}; do
	available_shm_bytes=$(df -B1 --output=avail /dev/shm | awk 'NR == 2 {print $1}')
	[[ $available_shm_bytes =~ ^[0-9]+$ ]] || {
		printf 'cannot determine available /dev/shm capacity\n' >&2
		exit 1
	}
	((available_shm_bytes >= required_shm_bytes)) && break
	sleep 0.1
done
if ((available_shm_bytes < required_shm_bytes)); then
	printf 'insufficient /dev/shm before model load: available=%s required=%s\n' \
		"$available_shm_bytes" "$required_shm_bytes" >&2
	find /dev/shm -maxdepth 1 -type f -name 'vllm_offload_*.mmap' \
		-printf 'existing offload arena: %p (%s bytes)\n' >&2
	exit 1
fi

available_snapshot_bytes=$(df -B1 --output=avail "$snapshot_root" | awk 'NR == 2 {print $1}')
[[ $available_snapshot_bytes =~ ^[0-9]+$ ]] || {
	printf 'cannot determine available snapshot filesystem capacity\n' >&2
	exit 1
}
if ((available_snapshot_bytes < required_snapshot_free_bytes)); then
	printf 'insufficient durable snapshot headroom before model load: available=%s required=%s root=%s\n' \
		"$available_snapshot_bytes" "$required_snapshot_free_bytes" "$snapshot_root" >&2
	exit 1
fi
printf 'durable snapshot headroom accepted: available=%s required=%s root=%s\n' \
	"$available_snapshot_bytes" "$required_snapshot_free_bytes" "$snapshot_root" >&2

kv_transfer_config=$(jq -cn \
	--arg engine_id "qwen-radiance-public-clean-${abi_id:0:16}" \
	--arg root_dir "/cache/snapshots/${data_abi}/data" \
	'{
      kv_connector: "OffloadingConnector",
      engine_id: $engine_id,
      kv_role: "kv_both",
      kv_load_failure_policy: "fail",
      kv_connector_extra_config: {
        spec_name: "TieringOffloadingSpec",
	        cpu_bytes_to_use: 19327352832,
        offload_prompt_only: false,
        blocks_per_chunk: 1,
        eviction_policy: "lru",
        snapshot_settled_tail_only: true,
	        secondary_tiers: [{
	          type: "qwen_chat_fs",
	          root_dir: $root_dir,
	          n_read_threads: 8,
	          n_write_threads: 8,
	          tail_flush_tokens: 8192,
	          tail_ram_max_bytes: 6442450944,
	          tail_ram_max_chats: 5,
	          tail_block_limit: 15,
	          control_directory: "/dev/shm/qwen-radiance-snapshot-control-v1",
	          tail_status_path: "/dev/shm/qwen-radiance-snapshot-tail.json"
	        }]
      }
	    }')
fair_config=$(jq -cn '{qwen_fair:{policy:"response_boundary",tool_grace_seconds:2,max_tool_deferral_seconds:30,max_cached_chats:2,status_path:"/dev/shm/qwen-radiance-fair-public"}}')

declare -a release_environment=()
while IFS=$'\t' read -r key value; do
	release_environment+=(-e "$key=$value")
done < <(jq -r '.kernel_environment | to_entries[] | [.key, .value] | @tsv' radiance-vllm-mxfp4/runtime-radiance-1.0.16.json)
speculative_config='{"method":"dflash","model":"/models/Qwen3.8-27B-DFlash2-FP8","num_speculative_tokens":7,"attention_backend":"TRITON_ATTN","disable_padded_drafter_batch":true,"draft_sample_method":"probabilistic"}'
# The authenticated worker creates its receipt once. Each server lifetime needs
# a distinct record while retaining the same compilation-cache namespace.
read -r startup_id </proc/sys/kernel/random/uuid
readonly startup_id

# Use the complete, pinned release. Never replay the old vLLM copies or compile
# old kernels over this image. Only the authenticated local integration is added.
# Allow EngineCore to drain and publish pending snapshot tails before the
# container runtime's independent kill deadline. vLLM defaults to immediate abort.
exec podman run --rm --pull=never --name "$container_name" --privileged --ipc=host --network=host \
	--stop-timeout 90 \
	--device /dev/kfd --device /dev/dri --group-add keep-groups \
	--security-opt seccomp=unconfined --cap-add SYS_PTRACE \
	-e PYTHONHASHSEED=0 -e ROCR_VISIBLE_DEVICES=0 -e HIP_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 \
	-e QWEN_RADIANCE_CACHE_ABI="$data_abi" \
	-e QWEN_ROUND_EVENT_STATUS_PATH=/dev/shm/qwen-radiance-fair-public \
	-e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
	-e VLLM_ROCM_USE_AITER_MHA=0 -e VLLM_ROCM_USE_AITER_MLA=0 -e VLLM_ROCM_USE_AITER_MOE=0 \
	-e VLLM_ROCM_USE_AITER_LINEAR=0 -e VLLM_ROCM_USE_AITER_FP8BMM=0 \
	-e VLLM_ROCM_USE_AITER_FP4BMM=0 -e VLLM_ROCM_USE_AITER_RMSNORM=0 \
	-e NCCL_PROTO=Simple -e R4D_ATTN_FP8=0 -e RADIANCE_WEIGHT_QUANTIZATION=auto \
	-e RADIANCE_RUN_BWTEST=0 -e RADIANCE_BANNER_PLAIN=1 \
	-e VLLM_CACHE_ROOT="/cache/runtime/${abi_id}/vllm" \
	-e TORCHINDUCTOR_CACHE_DIR="/cache/runtime/${abi_id}/inductor" \
	-e TRITON_CACHE_DIR="/cache/runtime/${abi_id}/triton" \
	-e QWEN_OPTIMIZED_STARTUP_RECEIPT="/cache/runtime/${abi_id}/optimized-startup-${startup_id}.json" \
	-e AITER_ROOT_DIR=/cache/runtime-1.0.16/aiter -e TRITON_CACHE_AUTOTUNING=1 \
	"${release_environment[@]}" \
	-v /home/lewis/.cache/huggingface:/root/.cache/huggingface:ro \
	-v /home/lewis/models-radiance:/models:ro \
	-v "$cache_root":/cache \
	-v "$launcher_root/radiance-vllm-mxfp4":/patches:ro \
	-v "$optimized_root":/qualification:ro \
	-v "$rocr_library":/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1.21.0:ro \
	--entrypoint /opt/vllm/bin/python "$image" /patches/bootstrap_radiance_release.py \
	/models/Qwen3.8-27B-Uncensored-MXFP4-awq --served-model-name "$model_id" \
	--host 0.0.0.0 --port "$port" --kv-cache-dtype fp8 --tensor-parallel-size 1 \
	--shutdown-timeout 60 \
	--gpu-memory-utilization 0.97 --kv-cache-memory 10000000000 \
	--max-model-len 253792 --max-num-seqs 2 --max-num-batched-tokens 2048 \
	--attention-backend R4D --speculative-config "$speculative_config" \
	--no-async-scheduling --language-model-only --skip-mm-profiling \
	--scheduler-cls qwen_radiance_fair_scheduler.FairScheduler --additional-config "$fair_config" \
	--worker-cls optimized_d7_worker.OptimizedWorker \
	--kv-transfer-config "$kv_transfer_config" \
	--middleware qwen_radiance_request_guard.require_snapshot_abi \
	--enable-prefix-caching --mamba-cache-mode align --enable-auto-tool-choice \
	--tool-call-parser qwen3_xml --reasoning-parser qwen3 \
	--enable-per-request-metrics --enable-force-include-usage --enable-prompt-tokens-details \
	--override-generation-config '{"temperature":1,"top_p":0.95,"top_k":20}' \
	--chat-template /patches/qwen-fixed-v22.3.jinja \
	--default-chat-template-kwargs '{"reasoning_effort":"xhigh"}' \
	--compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8]}'
