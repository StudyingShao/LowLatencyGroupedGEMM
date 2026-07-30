// SPDX-License-Identifier: Apache-2.0
#include "low_latency_grouped_gemm/include/low_latency_mxfp4_fp8.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "low_latency_grouped_gemm/include/low_latency_grouped_gemm.h"
#include "low_latency_grouped_gemm/include/low_latency_mxfp4_fp8_kernel.cuh"

namespace mga {
namespace {

using Kernel =
    low_latency_mxfp4_fp8_detail::LowLatencyMxfp4Fp8Kernel;
using Params = Kernel::Params;

[[noreturn]] void abort_mxfp4(const char* msg) {
    std::fprintf(stderr,
                 "[low_latency_mxfp4_fp8] invalid launch options: %s\n",
                 msg);
    std::abort();
}

void check_cuda(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        std::fprintf(stderr,
                     "[low_latency_mxfp4_fp8] %s: %s\n",
                     msg,
                     cudaGetErrorString(error));
        std::abort();
    }
}

void validate_shape(int G, int N_orig, int K) {
    if (G < 0) abort_mxfp4("G must be non-negative");
    if (N_orig <= 0 || N_orig % 64 != 0) {
        abort_mxfp4("N_orig must be a positive multiple of 64");
    }
    if (!Kernel::can_implement(K)) {
        abort_mxfp4("K must be a positive multiple of 64");
    }
}

__constant__ uint8_t kHummingRewriteLut[256 * 16];

uint32_t float_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float float_from_bits(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint8_t quantize_e2m1_like_humming(double value) {
    const float rounded_input = static_cast<float>(value);
    const uint32_t bits = float_bits(rounded_input);
    constexpr uint32_t kMask = 0x81c00000U;
    const uint32_t rz_bits = bits & kMask;
    const uint32_t ru_bits = (bits + 0x00200000U) & kMask;
    const double rz = static_cast<double>(float_from_bits(rz_bits));
    const double ru = static_cast<double>(float_from_bits(ru_bits));
    const uint32_t rounded =
        std::fabs(value - rz) >= std::fabs(value - ru) ? ru_bits : rz_bits;
    return static_cast<uint8_t>(
        ((rounded & 0x80000000U) >> 28U) |
        ((rounded & 0x01c00000U) >> 22U));
}

const std::array<uint8_t, 256 * 16>& humming_rewrite_lut() {
    static const std::array<uint8_t, 256 * 16> lut = [] {
        std::array<uint8_t, 256 * 16> result{};
        for (uint32_t delta = 0; delta < 256; ++delta) {
            const uint32_t scale_bits =
                0x3f800000U - (delta << 23U);
            const double scale =
                static_cast<double>(float_from_bits(scale_bits));
            for (uint32_t code = 0; code < 16; ++code) {
                uint8_t normalized =
                    static_cast<uint8_t>(code == 8 ? 0 : code);
                if (delta != 0) {
                    const uint32_t value_bits =
                        ((normalized & 0x8U) << 28U) |
                        ((normalized & 0x7U) << 22U);
                    const double value =
                        static_cast<double>(float_from_bits(value_bits)) *
                        scale;
                    normalized = quantize_e2m1_like_humming(value);
                }
                result[delta * 16 + code] = normalized;
            }
        }
        return result;
    }();
    return lut;
}

__global__ void preprocess_scales_kernel(
    const uint8_t* raw,
    uint8_t* exp_offsets,
    uint8_t* delta_offsets,
    float* residual,
    size_t scales_per_expert) {
    const int g = blockIdx.x;
    int local_min = 255;
    int local_max = 0;
    const uint8_t* raw_g =
        raw + static_cast<size_t>(g) * scales_per_expert;

    for (size_t i = threadIdx.x; i < scales_per_expert; i += blockDim.x) {
        const int value = raw_g[i];
        local_min = value < local_min ? value : local_min;
        local_max = value > local_max ? value : local_max;
    }

    __shared__ int mins[256];
    __shared__ int maxs[256];
    __shared__ int floor_exp;
    mins[threadIdx.x] = local_min;
    maxs[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            mins[threadIdx.x] =
                mins[threadIdx.x] < mins[threadIdx.x + stride]
                    ? mins[threadIdx.x]
                    : mins[threadIdx.x + stride];
            maxs[threadIdx.x] =
                maxs[threadIdx.x] > maxs[threadIdx.x + stride]
                    ? maxs[threadIdx.x]
                    : maxs[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const int range = min(maxs[0] - mins[0], 11);
        floor_exp = maxs[0] - range;
        residual[g] = exp2f(static_cast<float>(floor_exp - 128));
    }
    __syncthreads();

    uint8_t* exp_g =
        exp_offsets + static_cast<size_t>(g) * scales_per_expert;
    uint8_t* delta_g =
        delta_offsets + static_cast<size_t>(g) * scales_per_expert;
    for (size_t i = threadIdx.x; i < scales_per_expert; i += blockDim.x) {
        const int value = raw_g[i];
        const int clamped = value > floor_exp ? value : floor_exp;
        delta_g[i] = static_cast<uint8_t>(clamped - value);
        exp_g[i] = static_cast<uint8_t>((clamped - floor_exp + 1) & 0xf);
    }
}

__global__ void rewrite_payload_kernel(
    const uint8_t* raw_weight,
    const uint8_t* delta_offsets,
    uint8_t* processed_weight,
    size_t bytes_per_expert,
    size_t scales_per_expert) {
    const int g = blockIdx.y;
    const size_t expert_base = static_cast<size_t>(g) * bytes_per_expert;
    for (size_t byte_idx =
             static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         byte_idx < bytes_per_expert;
         byte_idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const uint8_t packed = raw_weight[expert_base + byte_idx];
        const uint8_t delta =
            delta_offsets[static_cast<size_t>(g) * scales_per_expert +
                          byte_idx / 16];
        const uint8_t lo =
            kHummingRewriteLut[static_cast<int>(delta) * 16 +
                               (packed & 0xf)];
        const uint8_t hi =
            kHummingRewriteLut[static_cast<int>(delta) * 16 +
                               (packed >> 4)];
        processed_weight[expert_base + byte_idx] =
            static_cast<uint8_t>(lo | (hi << 4));
    }
}

__device__ __forceinline__ uint32_t
preprocess_fp4x8_signs_for_fp8(uint32_t fp4x8) {
    const uint32_t em = fp4x8 & 0x77777777U;
    const uint32_t signs =
        ((fp4x8 & 0x00000008U) << 4U) |
        ((fp4x8 & 0x00000080U) << 8U) |
        ((fp4x8 & 0x00000800U) << 12U) |
        ((fp4x8 & 0x00008000U) << 16U) |
        ((fp4x8 & 0x00080000U) >> 16U) |
        ((fp4x8 & 0x00800000U) >> 12U) |
        ((fp4x8 & 0x08000000U) >> 8U) |
        ((fp4x8 & 0x80000000U) >> 4U);
    return em | signs;
}

__global__ void interleave_kernel(
    const uint8_t* processed_weight,
    const uint8_t* exp_offsets_logical,
    uint8_t* w_interleaved,
    uint8_t* exp_offsets_interleaved,
    int G,
    int N,
    int K) {
    const int tid = threadIdx.x;
    for (int g = blockIdx.y; g < G; g += gridDim.y) {
        for (int row_tile = blockIdx.x;
             row_tile < N / 64;
             row_tile += gridDim.x) {
            const int t0 = tid & 3;
            const int t1 = (tid >> 2) & 7;
            const int t2 = tid >> 5;
            const int row0 = row_tile * 64 + t1 + t2 * 16;
            const int row1 = row0 + 8;
            const size_t weight_expert_base =
                static_cast<size_t>(g) * N * K / 2;
            const size_t scale_expert_base =
                static_cast<size_t>(g) * N * K / 32;
            const int k32_count = K / 32;

            for (int k32_idx = 0; k32_idx < k32_count; ++k32_idx) {
                const int k_base = k32_idx * 32 + t0 * 4;
                const uint8_t* row0_ptr =
                    processed_weight + weight_expert_base +
                    static_cast<size_t>(row0) * K / 2;
                const uint8_t* row1_ptr =
                    processed_weight + weight_expert_base +
                    static_cast<size_t>(row1) * K / 2;
                const uint32_t row0_logical =
                    static_cast<uint32_t>(
                        *reinterpret_cast<const uint16_t*>(
                            row0_ptr + k_base / 2)) |
                    (static_cast<uint32_t>(
                         *reinterpret_cast<const uint16_t*>(
                             row0_ptr + k_base / 2 + 8))
                     << 16U);
                const uint32_t row1_logical =
                    static_cast<uint32_t>(
                        *reinterpret_cast<const uint16_t*>(
                            row1_ptr + k_base / 2)) |
                    (static_cast<uint32_t>(
                         *reinterpret_cast<const uint16_t*>(
                             row1_ptr + k_base / 2 + 8))
                     << 16U);
                const uint64_t physical =
                    static_cast<uint64_t>(preprocess_fp4x8_signs_for_fp8(
                        row0_logical)) |
                    (static_cast<uint64_t>(preprocess_fp4x8_signs_for_fp8(
                         row1_logical))
                     << 32U);
                const int tile64_count = N / 64;
                const int full_pair_count = tile64_count / 2;
                size_t dst = 0;
                if (row_tile < full_pair_count * 2) {
                    const int pair = row_tile / 2;
                    const int part = row_tile & 1;
                    dst = weight_expert_base +
                          (static_cast<size_t>(pair) * k32_count +
                           k32_idx) * 128 * 16 +
                          static_cast<size_t>(tid) * 16 + part * 8;
                } else {
                    dst = weight_expert_base +
                          static_cast<size_t>(full_pair_count) *
                              k32_count * 128 * 16 +
                          (static_cast<size_t>(k32_idx) * 128 + tid) * 8;
                }
                *reinterpret_cast<uint64_t*>(w_interleaved + dst) = physical;

                if (tid < 32) {
                    const int scale_group = tid;
                    const int scale_row0_local =
                        (scale_group & 7) + (scale_group >> 3) * 16;
                    const int scale_row1_local = scale_row0_local + 8;
                    const size_t logical_idx0 =
                        scale_expert_base +
                        static_cast<size_t>(row_tile * 64 +
                                            scale_row0_local) *
                            k32_count +
                        k32_idx;
                    const size_t logical_idx1 =
                        scale_expert_base +
                        static_cast<size_t>(row_tile * 64 +
                                            scale_row1_local) *
                            k32_count +
                        k32_idx;
                    size_t interleaved_idx = 0;
                    if (row_tile < full_pair_count * 2) {
                        const int pair = row_tile / 2;
                        const int part = row_tile & 1;
                        interleaved_idx =
                            scale_expert_base +
                            (static_cast<size_t>(pair) * k32_count +
                             k32_idx) * 128 +
                            static_cast<size_t>(scale_group) * 4 +
                            part * 2;
                    } else {
                        interleaved_idx =
                            scale_expert_base +
                            static_cast<size_t>(full_pair_count) *
                                k32_count * 128 +
                            static_cast<size_t>(k32_idx) * 64 +
                            static_cast<size_t>(scale_group) * 2;
                    }
                    exp_offsets_interleaved[interleaved_idx] =
                        exp_offsets_logical[logical_idx0];
                    exp_offsets_interleaved[interleaved_idx + 1] =
                        exp_offsets_logical[logical_idx1];
                }
            }
        }
    }
}

__global__ void combine_token_scales_kernel(
    const float* activation_dequant,
    const float* residual,
    const int32_t* expert_offsets,
    float* token_scales) {
    const int g = blockIdx.x;
    const int begin = expert_offsets[g];
    const int end = expert_offsets[g + 1];
    const float expert_scale = residual[g] * 64.0f;
    for (int token = begin + threadIdx.x;
         token < end;
         token += blockDim.x) {
        token_scales[token] =
            activation_dequant[token] * expert_scale;
    }
}

Params make_params(const LowLatencyMxfp4Fp8LaunchOpts& opts) {
    Params params{};
    params.M = opts.N_orig;
    params.N = opts.token_counts;
    params.K = opts.K;
    params.batch_count = opts.G;
    params.ptr_A = opts.w_interleaved;
    params.ptr_B =
        reinterpret_cast<const low_latency_mxfp4_fp8_detail::ElementB*>(
            opts.acts);
    params.ptr_exp_offsets = opts.exp_offsets_interleaved;
    params.ptr_token_scales = opts.token_scales;
    params.ptr_D = opts.outs;
    params.offsets = opts.expert_offsets;
    params.tile_experts = opts.tile_experts;
    params.tile_n = opts.tile_n;
    params.num_token_tiles = opts.num_token_tiles;
    params.num_token_tiles_device = opts.num_token_tiles_device;
    return params;
}

}  // namespace

void launch_low_latency_mxfp4_fp8_preprocess_scales(
    const uint8_t* raw_e8m0_scales,
    uint8_t* exp_offsets_logical,
    uint8_t* delta_offsets,
    float* expert_residual,
    int G,
    int N_orig,
    int K,
    cudaStream_t stream) {
    validate_shape(G, N_orig, K);
    if (G == 0) return;
    if (!raw_e8m0_scales || !exp_offsets_logical ||
        !delta_offsets || !expert_residual) {
        abort_mxfp4("null scale-preprocess tensor");
    }
    const size_t scales_per_expert =
        static_cast<size_t>(N_orig) * K / 32;
    preprocess_scales_kernel<<<G, 256, 0, stream>>>(
        raw_e8m0_scales,
        exp_offsets_logical,
        delta_offsets,
        expert_residual,
        scales_per_expert);
}

void launch_low_latency_mxfp4_fp8_rewrite_payload(
    const uint8_t* raw_weight,
    const uint8_t* delta_offsets,
    uint8_t* processed_weight,
    int G,
    int N_orig,
    int K,
    cudaStream_t stream) {
    validate_shape(G, N_orig, K);
    if (G == 0) return;
    if (!raw_weight || !delta_offsets || !processed_weight) {
        abort_mxfp4("null payload-rewrite tensor");
    }
    const auto& lut = humming_rewrite_lut();
    check_cuda(cudaMemcpyToSymbolAsync(kHummingRewriteLut,
                                       lut.data(),
                                       lut.size(),
                                       0,
                                       cudaMemcpyHostToDevice,
                                       stream),
               "failed to upload Humming rewrite LUT");
    const size_t bytes_per_expert =
        static_cast<size_t>(N_orig) * K / 2;
    const size_t scales_per_expert =
        static_cast<size_t>(N_orig) * K / 32;
    constexpr int kThreads = 256;
    const int blocks =
        static_cast<int>(
            std::min<size_t>((bytes_per_expert + kThreads - 1) / kThreads,
                             4096));
    rewrite_payload_kernel<<<dim3(blocks, G), kThreads, 0, stream>>>(
        raw_weight,
        delta_offsets,
        processed_weight,
        bytes_per_expert,
        scales_per_expert);
}

void launch_low_latency_mxfp4_fp8_interleave(
    const uint8_t* processed_weight,
    const uint8_t* exp_offsets_logical,
    uint8_t* w_interleaved,
    uint8_t* exp_offsets_interleaved,
    int G,
    int N_orig,
    int K,
    cudaStream_t stream) {
    validate_shape(G, N_orig, K);
    if (G == 0) return;
    if (!processed_weight || !exp_offsets_logical ||
        !w_interleaved || !exp_offsets_interleaved) {
        abort_mxfp4("null interleave tensor");
    }
    const dim3 block(128, 1, 1);
    const dim3 grid(std::min(N_orig / 64, 1024), std::min(G, 1024), 1);
    interleave_kernel<<<grid, block, 0, stream>>>(
        processed_weight,
        exp_offsets_logical,
        w_interleaved,
        exp_offsets_interleaved,
        G,
        N_orig,
        K);
}

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
    cudaStream_t stream) {
    launch_low_latency_mxfp4_fp8_preprocess_scales(
        raw_e8m0_scales,
        exp_offsets_logical,
        delta_workspace,
        expert_residual,
        G,
        N_orig,
        K,
        stream);
    launch_low_latency_mxfp4_fp8_rewrite_payload(
        raw_weight,
        delta_workspace,
        processed_workspace,
        G,
        N_orig,
        K,
        stream);
    launch_low_latency_mxfp4_fp8_interleave(
        processed_workspace,
        exp_offsets_logical,
        w_interleaved,
        exp_offsets_interleaved,
        G,
        N_orig,
        K,
        stream);
}

void launch_low_latency_mxfp4_fp8_combine_token_scales(
    const float* activation_dequant,
    const float* expert_residual,
    const int32_t* expert_offsets,
    float* token_scales,
    int G,
    cudaStream_t stream) {
    if (G < 0) abort_mxfp4("G must be non-negative");
    if (G == 0) return;
    if (!activation_dequant || !expert_residual ||
        !expert_offsets || !token_scales) {
        abort_mxfp4("null token-scale tensor");
    }
    combine_token_scales_kernel<<<G, 256, 0, stream>>>(
        activation_dequant,
        expert_residual,
        expert_offsets,
        token_scales);
}

void launch_low_latency_mxfp4_fp8(
    const LowLatencyMxfp4Fp8LaunchOpts& opts) {
    validate_shape(opts.G, opts.N_orig, opts.K);
    if (opts.G == 0) return;
    if (!opts.acts || !opts.w_interleaved ||
        !opts.exp_offsets_interleaved || !opts.token_scales ||
        !opts.token_counts || !opts.expert_offsets || !opts.outs) {
        abort_mxfp4("null GEMM tensor");
    }

    const bool host_scheduled = opts.num_token_tiles > 0;
    const bool device_scheduled = opts.num_token_tiles_device != nullptr;
    if (!host_scheduled && !device_scheduled && opts.max_M_g <= 0) {
        abort_mxfp4("max_M_g must be positive for rectangular launch");
    }
    if (host_scheduled && device_scheduled) {
        abort_mxfp4("host and device schedules are mutually exclusive");
    }
    if ((host_scheduled || device_scheduled) &&
        (!opts.tile_experts || !opts.tile_n)) {
        abort_mxfp4("compact schedule requires tile arrays");
    }

    const Params params = make_params(opts);
    const dim3 block(
        low_latency_mxfp4_fp8_detail::kThreadsPerRow,
        low_latency_mxfp4_fp8_detail::kThreadCount /
            low_latency_mxfp4_fp8_detail::kThreadsPerRow,
        1);
    const int row_tiles = (opts.N_orig + 127) / 128;

    if (device_scheduled) {
        if (opts.persistent_ctas <= 0) {
            abort_mxfp4("device schedule requires persistent_ctas");
        }
        if (opts.build_device_schedule) {
            if (opts.tile_schedule_capacity <= 0) {
                abort_mxfp4("device schedule capacity is too small");
            }
            launch_low_latency_grouped_gemm_build_tile_schedule(
                opts.token_counts,
                opts.G,
                opts.tile_schedule_capacity,
                opts.tile_experts,
                opts.tile_n,
                opts.num_token_tiles_device,
                opts.stream);
        }
        if (opts.N_orig == 2560 && opts.K == 4096) {
            low_latency_mxfp4_fp8_detail::
                low_latency_mxfp4_fp8_device_schedule_kernel<
                    2560, 4096><<<
                    opts.persistent_ctas, block, 0, opts.stream>>>(
                    params, row_tiles);
        } else if (opts.N_orig == 4096 && opts.K == 1280) {
            low_latency_mxfp4_fp8_detail::
                low_latency_mxfp4_fp8_device_schedule_kernel<
                    4096, 1280><<<
                    opts.persistent_ctas, block, 0, opts.stream>>>(
                    params, row_tiles);
        } else {
            low_latency_mxfp4_fp8_detail::
                low_latency_mxfp4_fp8_device_schedule_kernel<><<<
                    opts.persistent_ctas, block, 0, opts.stream>>>(
                    params, row_tiles);
        }
        return;
    }

    const dim3 grid(
        row_tiles,
        host_scheduled ? 1 : (opts.max_M_g + 7) / 8,
        host_scheduled ? opts.num_token_tiles : opts.G);
    if (opts.N_orig == 2560 && opts.K == 4096) {
        low_latency_mxfp4_fp8_detail::
            low_latency_mxfp4_fp8_kernel<2560, 4096><<<
                grid, block, 0, opts.stream>>>(params);
    } else if (opts.N_orig == 4096 && opts.K == 1280) {
        low_latency_mxfp4_fp8_detail::
            low_latency_mxfp4_fp8_kernel<4096, 1280><<<
                grid, block, 0, opts.stream>>>(params);
    } else {
        low_latency_mxfp4_fp8_detail::low_latency_mxfp4_fp8_kernel<><<<
            grid, block, 0, opts.stream>>>(params);
    }
}

}  // namespace mga
