// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace mga {

// FP32 scalar reference for the processed Humming representation.
// processed_weight and exp_offsets_logical use logical row-major layouts;
// token_scales already includes activation dequant, expert residual, and 64.
void launch_reference_low_latency_mxfp4_fp8(
    const __nv_fp8_e4m3* acts,
    const uint8_t* processed_weight,
    const uint8_t* exp_offsets_logical,
    const float* token_scales,
    const int32_t* token_to_expert,
    __nv_bfloat16* outs,
    int G,
    int N_orig,
    int K,
    int M_total,
    cudaStream_t stream);

}  // namespace mga
