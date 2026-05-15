// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace mga {

// Low-latency int4-weight/fp8-activation grouped GEMM path adapted for
// the target MoE shape.
//
// This path keeps packed int4 weights.  It uses a lightweight CTA mapping
// derived from the original GEMM implementation instead of the persistent
// WGMMA/TMA grouped kernel.  K must be aligned to 64; K % 128 == 64 uses
// a one-unroll tail path for the final scale group.
struct LowLatencyGroupedGemmLaunchOpts {
    int G;
    int N_orig;              // output dimension per expert
    int K;                   // compute K, must be multiple of 64
    int scale_group_size = 128; // supported: 64 or 128; K must be a multiple
    int max_M_g;             // rectangular fallback only; dsched may pass 0
    float act_scale;

    // Device tensors.
    const __nv_fp8_e4m3* acts;          // [M_total, K] after any K padding
    const uint8_t*       w_interleaved; // [G, N_orig, K] packed/interleaved int4
    const __nv_bfloat16* scales_padded; // [G, N_orig, padded(K/group_size,8)]
    const int32_t*       token_counts;  // [G]
    const int32_t*       expert_offsets;// [G+1], output/activation row offsets
    int32_t*             tile_experts;  // optional [num_token_tiles], CTA-local order preferred
    int32_t*             tile_n;        // optional [num_token_tiles], token tile id
    int                  num_token_tiles = 0;
    int32_t*             num_token_tiles_device = nullptr;
    int                  tile_schedule_capacity = 0;
    int                  persistent_ctas = 0;
    bool                 build_device_schedule = false;
    __nv_bfloat16*       outs;          // [M_total, N_orig]

    cudaStream_t stream;
};

inline int low_latency_grouped_gemm_aligned_k(int K) {
    return ((K + 63) / 64) * 64;
}

inline int low_latency_grouped_gemm_scale_groups_per_row(
    int K,
    int scale_group_size = 128) {
    return (K + scale_group_size - 1) / scale_group_size;
}

inline int low_latency_grouped_gemm_padded_sfs_per_row(
    int K,
    int scale_group_size = 128) {
    return ((low_latency_grouped_gemm_scale_groups_per_row(
                 K, scale_group_size) + 7) /
            8) *
           8;
}

inline size_t low_latency_grouped_gemm_interleaved_weight_bytes(int G, int N_orig, int K) {
    return static_cast<size_t>(G) * N_orig * K / 2;
}

inline size_t low_latency_grouped_gemm_padded_scale_bytes(
    int G,
    int N_orig,
    int K,
    int scale_group_size = 128) {
    return static_cast<size_t>(G) * N_orig *
           low_latency_grouped_gemm_padded_sfs_per_row(K, scale_group_size) *
           sizeof(__nv_bfloat16);
}

// Device-side preprocessing.  Input weights are packed int4 in
// [G, N_orig, K/2] row-major layout.  Input scales are transposed to
// [G, N_orig, K/group_size] before calling this function.
void launch_low_latency_grouped_gemm_interleave(
    const uint8_t*       w_orig_padded,
    const __nv_bfloat16* scales_n_major,
    uint8_t*             w_interleaved,
    __nv_bfloat16*       scales_padded,
    int                  G,
    int                  N_orig,
    int                  K,
    cudaStream_t         stream,
    int                  scale_group_size = 128);

inline int low_latency_grouped_gemm_tile_schedule_capacity_from_total_tokens(int M_total) {
    return M_total > 0 ? M_total : 1;
}

inline int low_latency_grouped_gemm_tile_schedule_capacity(int G, int max_M_g) {
    return G * ((max_M_g + 7) / 8);
}

void launch_low_latency_grouped_gemm_build_tile_schedule(
    const int32_t* token_counts,
    int G,
    int tile_schedule_capacity,
    int32_t* tile_experts,
    int32_t* tile_n,
    int32_t* num_token_tiles,
    cudaStream_t stream);

void launch_low_latency_grouped_gemm(const LowLatencyGroupedGemmLaunchOpts& opts);

}  // namespace mga
