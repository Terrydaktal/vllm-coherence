// SPDX-License-Identifier: BSD-3-Clause
// Reduction order derived from stock_m1_gemma_norm.hip / PyTorch contributors.
// TP1 fusion adapted from GGZ14's FP8 stream, retaining our corrected arithmetic
// and the native vLLM quantizer's division, clamp, and OCP conversion.
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <stdint.h>

static __device__ __forceinline__ float widen(uint16_t x) {
  return __uint_as_float(static_cast<uint32_t>(x) << 16);
}
static __device__ __forceinline__ uint16_t narrow(float x) {
  uint32_t b = __float_as_uint(x);
  if ((b & 0x7fffffffU) > 0x7f800000U) return (b >> 16) | 0x40U;
  return (b + 0x7fffU + ((b >> 16) & 1U)) >> 16;
}
static __device__ __forceinline__ unsigned char quant(float x, float scale) {
  const float r = fmaxf(-448.f, fminf(x / scale, 448.f));
  return __builtin_amdgcn_cvt_pk_fp8_f32(r, r, 0u, false) & 255u;
}

template <int THREADS>
static __device__ __forceinline__ float max_row(float x, float *partial) {
  const int tid = threadIdx.x;
  for (int d = 16; d; d >>= 1) x = fmaxf(x, __shfl_down(x, d, 32));
  if (!(tid & 31)) partial[tid >> 5] = x;
  __syncthreads();
  if (tid < 32) {
    x = tid < THREADS / 32 ? partial[tid] : 0.f;
    for (int d = 16; d; d >>= 1) x = fmaxf(x, __shfl_down(x, d, 32));
    if (!tid) partial[0] = fmaxf(x / 448.f, 1.f / (448.f * 512.f));
  }
  __syncthreads();
  return partial[0];
}

template <bool RESIDUAL, int THREADS>
__global__ __launch_bounds__(THREADS) void norm_quant(
    const uint16_t *x, const uint16_t *residual, const uint16_t *weight,
    unsigned char *q, float *scales, uint16_t *carry,
    long sx, long sr, float eps) {
  constexpr int WIDTH = 5120;
  const int tid = threadIdx.x, row = blockIdx.x;
  __shared__ float partial[THREADS];
  __shared__ float inverse;
  float accum[4] = {0, 0, 0, 0};
  float kept[(WIDTH / 4 + THREADS - 1) / THREADS][4];
  for (int vector = tid; vector < WIDTH / 4; vector += THREADS) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int col = vector * 4 + j;
      float z = widen(x[row * sx + col]);
      if (RESIDUAL) z += widen(residual[row * sr + col]);
      kept[vector / THREADS][j] = z;
      accum[j] += z * z;
    }
  }
  float total = ((accum[0] + accum[1]) + accum[2]) + accum[3];
  partial[tid] = total;
  __syncthreads();
  if (tid < 32) {
    float values[THREADS / 32];
#pragma unroll
    for (int wave = 0; wave < THREADS / 32; ++wave) values[wave] = partial[tid + wave * 32];
#pragma unroll
    for (int d = THREADS / 64; d; d >>= 1) {
#pragma unroll
      for (int wave = 0; wave < d; ++wave) values[wave] += values[wave + d];
    }
    total = values[0];
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) total += __shfl_down(total, d, 32);
    if (!tid) inverse = static_cast<float>(::rsqrt(static_cast<double>(total * (1.f / WIDTH) + eps)));
  }
  __syncthreads();
  float amax = 0;
  for (int vector = tid; vector < WIDTH / 4; vector += THREADS) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int col = vector * 4 + j;
      const float z = kept[vector / THREADS][j];
      if (RESIDUAL) carry[row * WIDTH + col] = narrow(z);
      const float y = widen(narrow((z * inverse) * (widen(weight[col]) + 1.f)));
      kept[vector / THREADS][j] = y;
      amax = fmaxf(amax, fabsf(y));
    }
  }
  const float scale = max_row<THREADS>(amax, partial);
  if (!tid) scales[row] = scale;
  for (int vector = tid; vector < WIDTH / 4; vector += THREADS) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
      q[row * WIDTH + vector * 4 + j] = quant(kept[vector / THREADS][j], scale);
  }
}

// The eager/compiled contract rounds SiLU to BF16 before multiplication.
// Full native qualification determines whether its exp agrees for this domain.
__global__ __launch_bounds__(1024) void silu_quant(
    const uint16_t *gu, unsigned char *q, float *scales, long stride) {
  constexpr int WIDTH = 17408, THREADS = 1024;
  const int tid = threadIdx.x, row = blockIdx.x;
  __shared__ float partial[THREADS];
  float kept[17];
  float amax = 0;
#pragma unroll
  for (int j = 0; j < 17; ++j) {
    const int col = tid + j * THREADS;
    const float x = widen(gu[row * stride + col]);
    const float sigmoid = 1.f / (1.f + expf(-x));
    const float act = widen(narrow(x * sigmoid));
    const float y = widen(narrow(act * widen(gu[row * stride + WIDTH + col])));
    kept[j] = y;
    amax = fmaxf(amax, fabsf(y));
  }
  const float scale = max_row<THREADS>(amax, partial);
  if (!tid) scales[row] = scale;
#pragma unroll
  for (int j = 0; j < 17; ++j) q[row * WIDTH + tid + j * THREADS] = quant(kept[j], scale);
}

extern "C" int qwen_stock_fp8_norm(const void *x, const void *r, const void *w,
    void *q, void *s, void *carry, int rows, long sx, long sr, float eps, void *stream) {
  if (rows < 1 || rows > 2048 || sx < 5120 || (r && sr < 5120) || !x || !w || !q || !s || (r && !carry))
    return static_cast<int>(hipErrorInvalidValue);
  // PyTorch's vectorized row reduction uses 512 lanes for M1 and
  // 64/32 lanes for batches with 8/16 or more outputs. Corrected decode
  // intentionally retains M1 arithmetic through eight positions.
#define LAUNCH_NORM(T) \
  if (r) hipLaunchKernelGGL((norm_quant<true, T>), dim3(rows), dim3(T), 0, static_cast<hipStream_t>(stream), \
    (const uint16_t*)x, (const uint16_t*)r, (const uint16_t*)w, (unsigned char*)q, \
    (float*)s, (uint16_t*)carry, sx, sr, eps); \
  else hipLaunchKernelGGL((norm_quant<false, T>), dim3(rows), dim3(T), 0, static_cast<hipStream_t>(stream), \
    (const uint16_t*)x, nullptr, (const uint16_t*)w, (unsigned char*)q, \
    (float*)s, nullptr, sx, sr, eps)
  if (rows <= 8) { LAUNCH_NORM(512); }
  else if (rows < 16) { LAUNCH_NORM(64); }
  else { LAUNCH_NORM(32); }
#undef LAUNCH_NORM
  return static_cast<int>(hipGetLastError());
}

extern "C" int qwen_stock_fp8_silu(const void *x, void *q, void *s, int rows, long stride, void *stream) {
  if (rows < 1 || rows > 8 || stride < 34816 || !x || !q || !s)
    return static_cast<int>(hipErrorInvalidValue);
  hipLaunchKernelGGL(silu_quant, dim3(rows), dim3(1024), 0, static_cast<hipStream_t>(stream),
    (const uint16_t*)x, (unsigned char*)q, (float*)s, stride);
  return static_cast<int>(hipGetLastError());
}
