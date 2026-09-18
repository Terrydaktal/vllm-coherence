// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <string>

namespace {

constexpr int64_t kRows = 8;
constexpr int64_t kGroupRows = 4;
constexpr int64_t kHiddenSize = 5120;
constexpr int64_t kOutputSize = 96;

bool byte_ranges_overlap(const torch::Tensor& left, const torch::Tensor& right) {
    const auto left_begin = reinterpret_cast<std::uintptr_t>(left.data_ptr());
    const auto right_begin = reinterpret_cast<std::uintptr_t>(right.data_ptr());
    const auto left_end = left_begin + static_cast<std::uintptr_t>(left.nbytes());
    const auto right_end = right_begin + static_cast<std::uintptr_t>(right.nbytes());
    return left_begin < right_end && right_begin < left_end;
}

}  // namespace

void launch_gdn_ba_m4_pair_wvsplitk(
    const torch::Tensor& hidden_states,
    const torch::Tensor& weight,
    const torch::Tensor& output);

torch::Tensor gdn_ba_m4_pair_wvsplitk_out(
    torch::Tensor hidden_states,
    torch::Tensor weight,
    torch::Tensor output) {
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be on ROCm");
    TORCH_CHECK(weight.is_cuda(), "weight must be on ROCm");
    TORCH_CHECK(output.is_cuda(), "output must be on ROCm");
    TORCH_CHECK(
        hidden_states.scalar_type() == torch::kBFloat16,
        "hidden_states must be BF16");
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "weight must be BF16");
    TORCH_CHECK(output.scalar_type() == torch::kBFloat16, "output must be BF16");
    TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(
        hidden_states.dim() == 2 && hidden_states.size(0) == kRows &&
            hidden_states.size(1) == kHiddenSize,
        "hidden_states must have shape [8, 5120]");
    TORCH_CHECK(
        weight.dim() == 2 && weight.size(0) == kOutputSize &&
            weight.size(1) == kHiddenSize,
        "weight must have shape [96, 5120]");
    TORCH_CHECK(
        output.dim() == 2 && output.size(0) == kRows &&
            output.size(1) == kOutputSize,
        "output must have shape [8, 96]");
    TORCH_CHECK(
        hidden_states.device() == weight.device() && weight.device() == output.device(),
        "all tensors must be on the same ROCm device");
    TORCH_CHECK(
        !byte_ranges_overlap(output, hidden_states),
        "output must not overlap hidden_states");
    TORCH_CHECK(!byte_ranges_overlap(output, weight), "output must not overlap weight");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    const std::string architecture = properties->gcnArchName;
    TORCH_CHECK(
        architecture.find("gfx1201") != std::string::npos,
        "candidate is qualified only for gfx1201");
    TORCH_CHECK(
        properties->multiProcessorCount == 32,
        "candidate is qualified only for the runtime-reported 32-CU R9700 geometry");

    // The launcher performs two instances of the exact installed wvSplitK M4
    // arithmetic, targeting rows [0,4) and [4,8) of the caller-owned result.
    // Do not substitute a generic M8 GEMM: v363 already disproved its parity.
    launch_gdn_ba_m4_pair_wvsplitk(hidden_states, weight, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "gdn_ba_m4_pair_wvsplitk_out",
        &gdn_ba_m4_pair_wvsplitk_out,
        "Exact gfx1201 BF16 GDN B/A wvSplitK M4+M4 projection (out parameter)");
}
