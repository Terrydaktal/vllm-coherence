// Experimental numerical repair for extreme GDN prefill decay spans.
// The fast scan splits bounded decay products into huge/tiny factors and clamps
// them. Recompute only affected (sequence, head) pairs using the defining FP32
// recurrence, where exp(per-token negative decay) is always bounded by one.
// Inputs q/k are already normalized BF16, exactly as passed to the chunk scan.
#include <hip/hip_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ float from_bf16(uint16_t value) {
  return __uint_as_float(static_cast<uint32_t>(value) << 16);
}

static __device__ __forceinline__ uint16_t to_bf16(float value) {
  const uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7fffffffU) > 0x7f800000U)
    return static_cast<uint16_t>((bits >> 16) | 0x40U);
  return static_cast<uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

__global__ __launch_bounds__(256) void repair_extreme_decay(
    const uint16_t *__restrict__ q, const uint16_t *__restrict__ k,
    const uint16_t *__restrict__ v, const float *__restrict__ cumulative,
    const float *__restrict__ beta, const float *__restrict__ initial,
    uint16_t *__restrict__ output, float *__restrict__ final,
    const int *__restrict__ cu, int heads, int query_heads, int chunk,
    float scale, float threshold) {
  const int sequence = blockIdx.x, head = blockIdx.y;
  const int begin = cu[sequence], end = cu[sequence + 1];
  const int thread = threadIdx.x;
  __shared__ int required;
  if (thread == 0) {
    required = 0;
    for (int first = begin; first < end; first += chunk) {
      const int last = min(first + chunk, end) - 1;
      const float span = fabsf(cumulative[(size_t)last * heads + head] -
                               cumulative[(size_t)first * heads + head]);
      if (span > threshold) {
        required = 1;
        break;
      }
    }
  }
  __syncthreads();
  if (!required || begin == end)
    return;
  const int row = thread >> 1, half = thread & 1, start_k = half * 64;
  const int query_head = head / (heads / query_heads);
  const size_t state_offset = ((size_t)sequence * heads + head) * 128 * 128 +
                              row * 128 + start_k;
  float state[64];
#pragma unroll
  for (int j = 0; j < 64; ++j)
    state[j] = initial[state_offset + j];
  for (int token = begin; token < end; ++token) {
    const size_t gate_offset = (size_t)token * heads + head;
    const float previous = ((token - begin) % chunk == 0)
                               ? 0.0f : cumulative[gate_offset - heads];
    const float decay = expf(cumulative[gate_offset] - previous);
    const size_t qk_offset = ((size_t)token * query_heads + query_head) * 128 + start_k;
    float keys[64], prediction = 0.0f;
#pragma unroll
    for (int j = 0; j < 64; ++j) {
      keys[j] = from_bf16(k[qk_offset + j]);
      state[j] *= decay;
      prediction = fmaf(state[j], keys[j], prediction);
    }
    prediction += __shfl_xor(prediction, 1, 32);
    const float residual = beta[gate_offset] *
        (from_bf16(v[gate_offset * 128 + row]) - prediction);
    float value = 0.0f;
#pragma unroll
    for (int j = 0; j < 64; ++j) {
      state[j] = fmaf(residual, keys[j], state[j]);
      value = fmaf(state[j], from_bf16(q[qk_offset + j]), value);
    }
    value += __shfl_xor(value, 1, 32);
    if (half == 0)
      output[gate_offset * 128 + row] = to_bf16(value * scale);
  }
#pragma unroll
  for (int j = 0; j < 64; ++j)
    final[state_offset + j] = state[j];
}

extern "C" int qwen_gdn_extreme_decay_reference(
    const void *q, const void *k, const void *v, const void *g, const void *beta,
    const void *initial, void *output, void *final, const void *cu,
    int sequences, int heads, int query_heads, int chunk, float scale,
    float threshold, void *stream) {
  if (sequences <= 0 || query_heads <= 0 || heads % query_heads || chunk != 64)
    return static_cast<int>(hipErrorInvalidValue);
  repair_extreme_decay<<<dim3(sequences, heads), 256, 0, (hipStream_t)stream>>>(
      (const uint16_t *)q, (const uint16_t *)k, (const uint16_t *)v,
      (const float *)g, (const float *)beta, (const float *)initial,
      (uint16_t *)output, (float *)final, (const int *)cu,
      heads, query_heads, chunk, scale, threshold);
  return static_cast<int>(hipGetLastError());
}
