cmake_minimum_required(VERSION 3.20)
project(qwen_rocr_poll_backoff LANGUAGES C CXX ASM)

# Build ROCr and its statically linked thunk from the same pinned source.
# ROCR_SOURCE is the verified 2b22ab0195cc source plus the isolated backoff patch.
set(BUILD_SHARED_LIBS ON)
set(BUILD_THUNK_VIRTIO OFF)
add_subdirectory("${ROCR_SOURCE}/libhsakmt" thunk)
set(VERSION_MAJOR 1)
set(VERSION_MINOR 21)
set(VERSION_PATCH 0)
set(SO_VERSION_STRING "1.21.0")
set(PACKAGE_VERSION_STRING "1.21.0")
set(VERSION_JOB "qwen-poll-backoff")
set(VERSION_HASH "2b22ab01-46558b7a")
add_subdirectory("${ROCR_SOURCE}/runtime/hsa-runtime" rocr)

enable_testing()
add_executable(poll_backoff_test poll_backoff_test.cpp)
target_compile_features(poll_backoff_test PRIVATE cxx_std_17)
target_include_directories(poll_backoff_test PRIVATE "${ROCR_SOURCE}/runtime/hsa-runtime")
add_test(NAME poll_backoff COMMAND poll_backoff_test)

# Run manually in the isolated qualification window; never as a CPU-only test.
add_executable(async_event_smoke async_event_smoke.cpp)
target_compile_features(async_event_smoke PRIVATE cxx_std_17)
target_include_directories(async_event_smoke PRIVATE "/opt/rocm/core-7.14/include")
target_link_libraries(async_event_smoke PRIVATE hsa-runtime64 pthread)
