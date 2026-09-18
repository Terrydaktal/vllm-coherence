// SPDX-License-Identifier: Apache-2.0
//
// Shape-specialized executed path from vLLM csrc/rocm/skinny_gemms.cu at
// d626108b1841888ec90aced33367149a6bbc7e4b.  The arithmetic body and launch
// geometry match wvSplitK for BF16 weight [96,5120], input [4,5120], no bias,
// gfx1201, and the runtime-reported 32 CUs. The only intentional change is accepting the output
// address rather than allocating a Tensor.  The component gate compares every
// physical BF16 result bit against the installed operator before promotion.

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "gdn_ba_m4_pair_kernel.cu is qualified only for gfx1201"
#endif

namespace {

constexpr int kThreads = 32;
constexpr int kWavesPerGroup = 16;
constexpr int kYTile = 1;
constexpr int kAChunk = 8;
constexpr int kUnroll = 4;
constexpr int kBatchRows = 4;
constexpr int kHiddenSize = 5120;
constexpr int kOutputSize = 96;
constexpr int kCuCount = 32;
constexpr int kActiveWaves = 4;
constexpr int kMaxLdsElements = (64 * 1024) / 2;

template <typename T>
__device__ __forceinline__ T loadnt(T* address) {
    return __builtin_nontemporal_load(address);
}

__device__ __forceinline__ unsigned int min__(uint32_t left, uint32_t right) {
    return min(left, right);
}

// This is the gfx1x BF16 branch of upstream DOT2C. Keeping its two-element
// product, intra-pair add, and accumulator add in this order is the M4 contract.
#define DOT2C(V0, V2, V3)                                                     \
    {                                                                         \
        float2 product =                                                      \
            __bfloat1622float2(*((__hip_bfloat162*)(&(V2)))) *                \
            __bfloat1622float2(*((__hip_bfloat162*)(&(V3))));                 \
        V0 += (product.x + product.y);                                        \
    }

using Scalar8 =
    __attribute__((__vector_size__((kAChunk / 2) * sizeof(float)))) float;

union BigType {
    __hip_bfloat16 h[kAChunk];
    float f[kAChunk / 2];
    float2 f2[kAChunk / 4];
    double d[kAChunk / 4];
    Scalar8 h8;
};

__global__ void __launch_bounds__(kWavesPerGroup * kThreads)
    gdn_ba_wvsplitk_m4_kernel(
        const int K,
        const int Kbp,
        const int Kap,
        const int M,
        const __hip_bfloat16* B,
        const __hip_bfloat16* __restrict__ A,
        __hip_bfloat16* C,
        const int active_waves,
        const int cu_count) {
    __shared__ __hip_bfloat16 s[kMaxLdsElements];

    // Exact upstream wvSplitK_hf_sml_ activation staging for N=4.
    for (uint32_t k = (threadIdx.y * kThreads + threadIdx.x) * kAChunk;
         k < min__(Kap * kBatchRows, kMaxLdsElements);
         k += kThreads * kWavesPerGroup * kAChunk) {
        *((BigType*)(&s[k])) = *((BigType*)(&A[k]));
    }
    __syncthreads();

    if (threadIdx.y >= active_waves) {
        return;
    }

    uint32_t m =
        (blockIdx.x * active_waves + (threadIdx.y % active_waves)) * kYTile;

    while (m < M) {
        float sum[kBatchRows][kYTile] = {};

        for (uint32_t k1 = 0; k1 < K;
             k1 += kThreads * kAChunk * kUnroll) {
            BigType big_a[kBatchRows][kUnroll] = {};
            BigType big_b[kYTile][kUnroll];

#pragma unroll
            for (uint32_t k2 = 0; k2 < kUnroll; ++k2) {
                const uint32_t k = k1 + k2 * kThreads * kAChunk;
                const uint32_t lane_k = k + threadIdx.x * kAChunk;
                const __hip_bfloat16* weight = &B[min__(lane_k, K - kAChunk)];
                for (int y = 0; y < kYTile; ++y) {
                    big_b[y][k2].h8 =
                        loadnt((Scalar8*)(&weight[min__(y + m, M - 1) * Kbp]));
                }
            }

#pragma unroll
            for (uint32_t k2 = 0; k2 < kUnroll; ++k2) {
                const uint32_t k = k1 + k2 * kThreads * kAChunk;
                const uint32_t lane_k = k + threadIdx.x * kAChunk;
                if (lane_k >= K) {
                    break;
                }
                for (int n = 0; n < kBatchRows; ++n) {
                    big_a[n][k2] = *((const BigType*)(&(s[lane_k + Kap * n])));
                }
            }

            for (uint32_t k2 = 0; k2 < kUnroll; ++k2) {
                for (uint32_t n = 0; n < kBatchRows; ++n) {
                    for (int y = 0; y < kYTile; ++y) {
                        for (uint32_t pair = 0; pair < kAChunk / 2; ++pair) {
                            DOT2C(sum[n][y], big_a[n][k2].f[pair],
                                  big_b[y][k2].f[pair])
                        }
                    }
                }
            }
        }
        __builtin_amdgcn_sched_barrier(0);

        for (int n = 0; n < kBatchRows; ++n) {
            for (int y = 0; y < kYTile; ++y) {
                sum[n][y] += __builtin_amdgcn_mov_dpp(
                    sum[n][y], 0x118, 0xf, 0xf, 1);
                sum[n][y] += __builtin_amdgcn_mov_dpp(
                    sum[n][y], 0x114, 0xf, 0xf, 1);
                sum[n][y] += __builtin_amdgcn_mov_dpp(
                    sum[n][y], 0x112, 0xf, 0xf, 1);
                sum[n][y] += __builtin_amdgcn_mov_dpp(
                    sum[n][y], 0x111, 0xf, 0xf, 1);
                sum[n][y] += __shfl_xor(sum[n][y], 16);
            }
        }

        if (threadIdx.x == (kThreads - 1)) {
            for (int n = 0; n < kBatchRows; ++n) {
                for (int y = 0; y < kYTile; ++y) {
                    C[m + y + n * M] = __float2bfloat16(sum[n][y]);
                }
            }
        }
        m += cu_count * active_waves * kYTile;
    }
}

void launch_one_m4(
    const __hip_bfloat16* hidden,
    const __hip_bfloat16* weight,
    __hip_bfloat16* output,
    hipStream_t stream) {
    const dim3 grid(kCuCount);
    const dim3 block(kThreads, kWavesPerGroup);
    gdn_ba_wvsplitk_m4_kernel<<<grid, block, 0, stream>>>(
        kHiddenSize,
        kHiddenSize,
        kHiddenSize,
        kOutputSize,
        weight,
        hidden,
        output,
        kActiveWaves,
        kCuCount);
}

}  // namespace

void launch_gdn_ba_m4_pair_wvsplitk(
    const torch::Tensor& hidden_states,
    const torch::Tensor& weight,
    const torch::Tensor& output) {
    const auto* hidden = reinterpret_cast<const __hip_bfloat16*>(
        hidden_states.data_ptr<at::BFloat16>());
    const auto* weights =
        reinterpret_cast<const __hip_bfloat16*>(weight.data_ptr<at::BFloat16>());
    auto* result =
        reinterpret_cast<__hip_bfloat16*>(output.data_ptr<at::BFloat16>());
    const hipStream_t stream = at::cuda::getCurrentCUDAStream();

    // Preserve the authenticated operation sequence exactly: M4 rows 0..3,
    // followed by M4 rows 4..7. Both write directly into final M8 storage.
    launch_one_m4(hidden, weights, result, stream);
    launch_one_m4(
        hidden + kBatchRows * kHiddenSize,
        weights,
        result + kBatchRows * kOutputSize,
        stream);
}

#undef DOT2C
