// SPDX-License-Identifier: Apache-2.0
// fp32 reference for the grouped GEMM W4A8.  One CUDA thread per (m, n).
// Uses the same input layouts as the production kernel (W_orig is the
// PRE-reordered K-contig layout — *not* the wgmma-friendly reorder).
#include "low_latency_grouped_gemm/include/reference_kernel.h"

#include <cstdint>

namespace mga {

namespace {

__device__ inline int unpack_int4_sym_dev(const uint8_t* W, int n, int k, int K) {
    uint8_t byte = W[(size_t)n * (K / 2) + (k / 2)];
    uint8_t nib  = (k & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
    return (nib & 0x8) ? (int(nib) - 16) : int(nib);
}

__global__ void reference_grouped_gemm_kernel(const __nv_fp8_e4m3*  acts,
                                              const uint8_t*        W_orig,
                                              const __nv_bfloat16*  scales,
                                              const int*            token_to_expt,
                                              float                 act_scale,
                                              __nv_bfloat16*        outs,
                                              int G, int N_orig, int K,
                                              int M_total,
                                              int scale_group_size) {
    int m = blockIdx.x;
    int n = blockIdx.y * blockDim.x + threadIdx.x;
    if (m >= M_total || n >= N_orig) return;

    int g = token_to_expt[m];
    if (g < 0) return;

    const int n_kgroup = (K + scale_group_size - 1) / scale_group_size;
    const uint8_t*       W_g = W_orig + (size_t)g * N_orig * K / 2;
    const __nv_bfloat16* S_g = scales  + (size_t)g * n_kgroup * N_orig;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        int q = unpack_int4_sym_dev(W_g, n, k, K);
        float a = static_cast<float>(acts[(size_t)m * K + k]);
        float s = static_cast<float>(
            S_g[(size_t)(k / scale_group_size) * N_orig + n]);
        acc += a * static_cast<float>(q) * s;
    }
    outs[(size_t)m * N_orig + n] = __float2bfloat16(acc * act_scale);
}

}  // namespace

void launch_reference_grouped_gemm(const __nv_fp8_e4m3*  acts,
                                   const uint8_t*        W_orig,
                                   const __nv_bfloat16*  scales,
                                   const int*            token_to_expt,
                                   float                 act_scale,
                                   __nv_bfloat16*        outs,
                                   int G, int N_orig, int K, int M_total,
                                   cudaStream_t stream,
                                   int scale_group_size) {
    if (M_total == 0) return;
    dim3 block(64);
    dim3 grid(M_total, (N_orig + 63) / 64);
    reference_grouped_gemm_kernel<<<grid, block, 0, stream>>>(
        acts, W_orig, scales, token_to_expt, act_scale, outs,
        G, N_orig, K, M_total, scale_group_size);
}

}  // namespace mga
