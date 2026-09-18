#!/usr/bin/env bash

set -euo pipefail

readonly launcher_root=/home/lewis/projects/r9700-public/qwen3.6-vllm-gfx1201-launchers
readonly cache_root=/home/lewis/.cache/qwen-radiance-public-w4a8-093
readonly model_id=qwen3.8-27b-uncensored-mxfp4-experimental
readonly container_name=qwen38-27b-uncensored-mxfp4-nosnapshot

cd "$launcher_root"

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
	RADIANCE_DFLASH_SMALL_WIDTH_COMPILE="${RADIANCE_DFLASH_SMALL_WIDTH_COMPILE:-0}" \
	RADIANCE_DFLASH_SUPPRESS_PROPOSALS="${RADIANCE_DFLASH_SUPPRESS_PROPOSALS:-0}" \
	RADIANCE_DFLASH_SUPPRESS_TARGET_AUX="${RADIANCE_DFLASH_SUPPRESS_TARGET_AUX:-0}" \
	RADIANCE_TARGET_ONLY_GEOMETRY="${RADIANCE_TARGET_ONLY_GEOMETRY:-0}" \
	RADIANCE_GDN_STATE_TRACE="${RADIANCE_GDN_STATE_TRACE:-0}" \
	RADIANCE_GDN_STATE_TRACE_ARM="${RADIANCE_GDN_STATE_TRACE_ARM:-unnamed}" \
	EXTRA="${EXTRA_OVERRIDE:---language-model-only}" \
	bash ./startup-qwen3.8-27b-mxfp4.sh
