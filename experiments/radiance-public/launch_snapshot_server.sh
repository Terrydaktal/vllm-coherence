#!/usr/bin/env bash

set -euo pipefail

readonly launcher_root=/home/lewis/projects/r9700-public/qwen3.6-vllm-gfx1201-launchers
readonly cache_root=/home/lewis/.cache/qwen-radiance-public-w4a8-093
readonly abi_id=a766d9bb6bf521bf77c7007fdbcc0f7706c896ef0b07565e27a50879d2893abd
readonly snapshot_root="${cache_root}/snapshots/${abi_id}"
readonly expected_abi_sha256=a766d9bb6bf521bf77c7007fdbcc0f7706c896ef0b07565e27a50879d2893abd
readonly model_id=qwen3.8-27b-uncensored-mxfp4-experimental
readonly container_name=qwen38-27b-uncensored-mxfp4-experimental

cd "$launcher_root"

actual_abi_sha256=$(sha256sum "$snapshot_root/abi.json" | awk '{print $1}')
if [[ "$actual_abi_sha256" != "$expected_abi_sha256" ]]; then
	printf 'snapshot ABI mismatch: got %s, expected %s\n' \
		"$actual_abi_sha256" "$expected_abi_sha256" >&2
	exit 1
fi

kv_transfer_config=$(jq -cn \
	--arg engine_id "qwen-radiance-snapshot-${abi_id:0:16}" \
	--arg root_dir "/cache/snapshots/${abi_id}/data" \
	'{
    kv_connector: "OffloadingConnector",
    engine_id: $engine_id,
    kv_role: "kv_both",
    kv_load_failure_policy: "fail",
    kv_connector_extra_config: {
      spec_name: "TieringOffloadingSpec",
      cpu_bytes_to_use: 10737418240,
      offload_prompt_only: false,
	      blocks_per_chunk: 1,
	      eviction_policy: "lru",
	      snapshot_settled_tail_only: true,
	      secondary_tiers: [{
        type: "fs",
        root_dir: $root_dir,
        n_read_threads: 8,
        n_write_threads: 8
      }]
    }
  }')

exec env \
	PYTHONHASHSEED=0 \
	MODELS=/home/lewis/models-radiance \
	SNAP=/home/lewis/models-radiance/Qwen3.8-27B-Uncensored-MXFP4-awq \
	DRAFTER=/home/lewis/models-radiance/Qwen3.8-27B-DFlash2-FP8 \
	NAME="$container_name" \
	SERVED="$model_id" \
	PORT=8080 \
	MAXLEN=253792 \
	MAXSEQS=1 \
	CHUNK=2048 \
	KV_MEM=10000000000 \
	CACHE="$cache_root" \
	SKIPMMPROF=1 \
	CAPTURE_SIZES='[1,2,4,8]' \
	REASONING_EFFORT=xhigh \
	EXTRA="--language-model-only --kv-transfer-config $kv_transfer_config" \
	bash ./startup-qwen3.8-27b-mxfp4.sh
