// SPDX-License-Identifier: Apache-2.0
#include "low_latency_grouped_gemm/include/mxfp4_reference_kernel.h"

#include <cstdint>

namespace mga {
namespace {

__device__ inline float decode_e2m1(uint8_t code) {
    constexpr float kMagnitude[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float magnitude = kMagnitude[code & 0x7];
    return (code & 0x8) ? -magnitude : magnitude;
}

__device__ inline uint8_t load_e2m1(
    const uint8_t* weight,
    int n,
    int k,
    int K) {
    const uint8_t packed =
        weight[static_cast<size_t>(n) * K / 2 + k / 2];
    return (k & 1) ? packed >> 4 : packed & 0xf;
}

__device__ float reference_dot(
    const __nv_fp8_e4m3* acts,
    const uint8_t* processed_weight,
    const uint8_t* exp_offsets,
    int token,
    int expert,
    int n,
    int N,
    int K) {
    const uint8_t* weight_g =
        processed_weight + static_cast<size_t>(expert) * N * K / 2;
    const uint8_t* offsets_g =
        exp_offsets + static_cast<size_t>(expert) * N * K / 32;
    float accum = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float a =
            static_cast<float>(acts[static_cast<size_t>(token) * K + k]);
        const float fp4 = decode_e2m1(load_e2m1(weight_g, n, k, K));
        const int offset =
            offsets_g[static_cast<size_t>(n) * K / 32 + k / 32];
        const float temporary_weight = ldexpf(fp4, offset - 6);
        accum += a * temporary_weight;
    }
    return accum;
}

__global__ void reference_kernel(
    const __nv_fp8_e4m3* acts,
    const uint8_t* processed_weight,
    const uint8_t* exp_offsets,
    const float* token_scales,
    const int32_t* token_to_expert,
    __nv_bfloat16* outs,
    int N,
    int K,
    int M_total) {
    const int token = blockIdx.x;
    const int n = blockIdx.y * blockDim.x + threadIdx.x;
    if (token >= M_total || n >= N) return;
    const int expert = token_to_expert[token];
    if (expert < 0) return;
    const float accum = reference_dot(
        acts, processed_weight, exp_offsets, token, expert, n, N, K);
    outs[static_cast<size_t>(token) * N + n] =
        __float2bfloat16(accum * token_scales[token]);
}

}  // namespace

void launch_reference_low_latency_mxfp4_fp8(
    const __nv_fp8_e4m3* acts,
    const uint8_t* processed_weight,
    const uint8_t* exp_offsets_logical,
    const float* token_scales,
    const int32_t* token_to_expert,
    __nv_bfloat16* outs,
    int,
    int N_orig,
    int K,
    int M_total,
    cudaStream_t stream) {
    if (M_total == 0) return;
    constexpr int kThreads = 64;
    reference_kernel<<<
        dim3(M_total, (N_orig + kThreads - 1) / kThreads),
        kThreads,
        0,
        stream>>>(acts,
                  processed_weight,
                  exp_offsets_logical,
                  token_scales,
                  token_to_expert,
                  outs,
                  N_orig,
                  K,
                  M_total);
}

}  // namespace mga
