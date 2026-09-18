#!/usr/bin/env bash
set -euo pipefail

# Run inside the pinned image without GPU devices. Supply the build tools on
# PATH and the image's rocm_sysdeps pkg-config directory through PKG_CONFIG_PATH.
recipe=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_root=$(realpath -- "${1:?provide the patched ROCr source directory}")
build_root=$(realpath -m -- "${2:?provide the build output directory}")
readonly expected_runtime=46ea7afeb0910a3c6b99e00e1aef29c8e20043532d3325b70f7206e914f4976b
readonly expected_header=2dcf4c597ca89721583185f75f0d4746e6a124edcb54d721df5d3e5ab90725b8
[[ $(sha256sum "$source_root/runtime/hsa-runtime/core/runtime/runtime.cpp" | awk '{print $1}') == "$expected_runtime" ]]
[[ $(sha256sum "$source_root/runtime/hsa-runtime/core/util/poll_backoff.h" | awk '{print $1}') == "$expected_header" ]]

# Keep $ORIGIN literal for the ELF loader.
# shellcheck disable=SC2016
cmake -S "$recipe" -B "$build_root" -G Ninja \
	-DROCR_SOURCE="$source_root" \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
	-DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
	-DCMAKE_ASM_COMPILER=/opt/rocm/llvm/bin/clang \
	'-DCMAKE_PREFIX_PATH=/opt/rocm/core-7.14;/opt/rocm/core-7.14/lib/rocm_sysdeps' \
	-DCMAKE_INSTALL_PREFIX=/opt/rocm/core-7.14 \
	'-DCMAKE_INSTALL_RPATH=$ORIGIN;$ORIGIN/rocm_sysdeps/lib' \
	-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
	-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	-DPKG_CONFIG_EXECUTABLE="$(command -v pkgconf)" \
	-DClang_DIR="$recipe/toolchain/clang" \
	-DLLVM_DIR="$recipe/toolchain/llvm"
cmake --build "$build_root" --parallel 6
ctest --test-dir "$build_root" --output-on-failure
