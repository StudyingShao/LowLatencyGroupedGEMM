// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace mga {

// Independent low-latency MXFP4-weight / per-token-FP8-activation grouped
// GEMM path. The mainloop expands processed E2M1 weights to temporary E4M3
// register fragments and executes native SM90 E4M3 x E4M3 WGMMA.
struct LowLatencyMxfp4Fp8LaunchOpts {
    int G;
    int N_orig;
    int K;       // positive multiple of 64
    int max_M_g; // rectangular fallback only; compact schedules may pass 0

    // Device tensors.
    const __nv_fp8_e4m3* acts;              // [M_total, K]
    const uint8_t* w_interleaved;           // packed E2M1, kernel K32/row-tile layout
    const uint8_t* exp_offsets_interleaved; // uint8, kernel K32/row-tile layout
    const float* token_scales;               // [M_total], act_dequant*residual*64
    const int32_t* token_counts;             // [G]
    const int32_t* expert_offsets;           // [G+1]

    // Compact schedule. Choose exactly one of these contracts:
    //   1. Host-known count: set num_token_tiles > 0 and provide both arrays.
    //   2. Device count: provide num_token_tiles_device, both arrays, and a
    //      positive persistent_ctas value. If build_device_schedule is true,
    //      also provide capacity; the launcher resets the counter on stream
    //      before running the builder.
    // If neither count is provided, max_M_g selects the rectangular fallback.
    int32_t* tile_experts;                   // [tile_schedule_capacity]
    int32_t* tile_n;                         // [tile_schedule_capacity]
    int num_token_tiles = 0;
    int32_t* num_token_tiles_device = nullptr; // one device int32 counter
    int tile_schedule_capacity = 0;            // required by device builder
    int persistent_ctas = 0;                   // device-schedule grid size
    bool build_device_schedule = false;        // build from token_counts
    __nv_bfloat16* outs;                     // [M_total, N_orig]

    cudaStream_t stream;
};

inline size_t low_latency_mxfp4_fp8_weight_bytes(int G, int N_orig, int K) {
    return static_cast<size_t>(G) * N_orig * K / 2;
}

inline size_t low_latency_mxfp4_fp8_scale_bytes(int G, int N_orig, int K) {
    return static_cast<size_t>(G) * N_orig * K / 32;
}

// Split raw E8M0 scales [G,N,K/32] into:
//   exp_offsets_logical : uint8 [G,N,K/32], values in [1,12]
//   delta_offsets       : uint8 [G,N,K/32], payload rewrite workspace
//   expert_residual     : float [G], 2^(expert_floor-128)
void launch_low_latency_mxfp4_fp8_preprocess_scales(
    const uint8_t* raw_e8m0_scales,
    uint8_t* exp_offsets_logical,
    uint8_t* delta_offsets,
    float* expert_residual,
    int G,
    int N_orig,
    int K,
    cudaStream_t stream);

// Apply the exact Humming E2M1 payload rewrite LUT.  Both weights use logical
// row-major [G,N,K/2] packing; delta_offsets is [G,N,K/32].
void launch_low_latency_mxfp4_fp8_rewrite_payload(
    const uint8_t* raw_weight,
    const uint8_t* delta_offsets,
    uint8_t* processed_weight,
    int G,
    int N_orig,
    int K,
    cudaStream_t stream);

// Convert logical processed weight/offset tensors into the kernel's
// K32-major mainloop layout.  Each full record covers 128 output rows for one
// K32 slice; an odd final 64-row tile uses a compact half-record.  Offset
// storage remains one byte per logical output-row/K32 group.
void launch_low_latency_mxfp4_fp8_interleave(
    const uint8_t* processed_weight,
    const uint8_t* exp_offsets_logical,
    uint8_t* w_interleaved,
    uint8_t* exp_offsets_interleaved,
    int G,
    int N_orig,
    int K,
    cudaStream_t stream);

// Convenience preprocessing pipeline.  The three caller-owned logical
// workspaces keep every stage independently testable and avoid hidden
// allocations:
//   delta_workspace     [G,N,K/32]
//   processed_workspace [G,N,K/2]
//   exp_offsets_logical [G,N,K/32]
void launch_low_latency_mxfp4_fp8_preprocess_weight(
    const uint8_t* raw_weight,
    const uint8_t* raw_e8m0_scales,
    uint8_t* delta_workspace,
    uint8_t* processed_workspace,
    uint8_t* w_interleaved,
    uint8_t* exp_offsets_logical,
    uint8_t* exp_offsets_interleaved,
    float* expert_residual,
    int G,
    int N_orig,
    int K,
    cudaStream_t stream);

// Build the epilogue multiplier in compact grouped-token order:
// token_scales[token] = activation_dequant[token] * residual[expert] * 64.
void launch_low_latency_mxfp4_fp8_combine_token_scales(
    const float* activation_dequant,
    const float* expert_residual,
    const int32_t* expert_offsets,
    float* token_scales,
    int G,
    cudaStream_t stream);

void launch_low_latency_mxfp4_fp8(
    const LowLatencyMxfp4Fp8LaunchOpts& opts);

}  // namespace mga
