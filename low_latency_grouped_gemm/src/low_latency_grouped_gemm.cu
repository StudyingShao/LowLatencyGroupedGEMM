// SPDX-License-Identifier: Apache-2.0
#include "low_latency_grouped_gemm/include/low_latency_grouped_gemm.h"

#include <cstdio>
#include <cstdlib>

#include "low_latency_grouped_gemm/include/low_latency_grouped_gemm_kernel.cuh"

namespace mga {
namespace {

[[noreturn]] void abort_low_latency_grouped_gemm_args(const char* msg) {
    std::fprintf(stderr, "[low_latency_grouped_gemm] invalid launch options: %s\n", msg);
    std::abort();
}

void check_cuda_launch_setup(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[low_latency_grouped_gemm] %s: %s\n", msg, cudaGetErrorString(err));
        std::abort();
    }
}

void validate_common(int G, int N_orig, int K, int scale_group_size) {
    if (G < 0) abort_low_latency_grouped_gemm_args("G must be non-negative");
    if (N_orig <= 0) abort_low_latency_grouped_gemm_args("N_orig must be positive");
    if (N_orig % 64 != 0) {
        abort_low_latency_grouped_gemm_args("N_orig must be a multiple of 64");
    }
    if (!low_latency_grouped_gemm_detail::LowLatencyGroupedGemmKernel::can_implement(K)) {
        abort_low_latency_grouped_gemm_args("K must be a positive multiple of 64");
    }
    if (scale_group_size != 64 && scale_group_size != 128) {
        abort_low_latency_grouped_gemm_args("scale_group_size must be 64 or 128");
    }
    if (K % scale_group_size != 0) {
        abort_low_latency_grouped_gemm_args(
            "K must be a multiple of scale_group_size");
    }
}

using LowLatencyKernel =
    low_latency_grouped_gemm_detail::LowLatencyGroupedGemmKernel;
using LowLatencyParams = LowLatencyKernel::Params;

struct LowLatencyDispatchArgs {
    const LowLatencyParams& params;
    dim3 grid;
    dim3 block;
    int row_tiles;
    cudaStream_t stream;
};

bool use_full_k_prefetch_a(int K) {
    return (K % 128) == 0 && K >= 512;
}

LowLatencyParams make_low_latency_params(
    const LowLatencyGroupedGemmLaunchOpts& opts) {
    LowLatencyParams params{};
    params.M = opts.N_orig;
    params.N = opts.token_counts;
    params.K = opts.K;
    params.batch_count = opts.G;
    params.alpha = opts.act_scale;
    params.ptr_A =
        reinterpret_cast<const low_latency_grouped_gemm_detail::ElementA*>(
            opts.w_interleaved);
    params.ptr_B =
        reinterpret_cast<const low_latency_grouped_gemm_detail::ElementB*>(
            opts.acts);
    params.ptr_D = opts.outs;
    params.batch_stride_A = static_cast<int64_t>(opts.N_orig) * opts.K;
    params.ptr_SFA =
        reinterpret_cast<const low_latency_grouped_gemm_detail::ElementSF*>(
            opts.scales_padded);
    params.offsets = opts.expert_offsets;
    params.tile_experts = opts.tile_experts;
    params.tile_n = opts.tile_n;
    params.num_token_tiles = opts.num_token_tiles;
    params.num_token_tiles_device = opts.num_token_tiles_device;
    return params;
}

template <bool kDeviceSchedule, int kScaleGroupSize>
void launch_full_k_dynamic(const LowLatencyDispatchArgs& args) {
    if constexpr (kDeviceSchedule) {
        low_latency_grouped_gemm_detail::
            low_latency_grouped_gemm_w4a8_device_schedule_kernel<
                kScaleGroupSize><<<
                args.grid, args.block, 0, args.stream>>>(
                args.params, args.row_tiles);
    } else {
        low_latency_grouped_gemm_detail::
            low_latency_grouped_gemm_w4a8_device_kernel<kScaleGroupSize><<<
                args.grid, args.block, 0, args.stream>>>(args.params);
    }
}

template <bool kDeviceSchedule, int kScaleGroupSize>
void launch_full_k_prefetch_dynamic(const LowLatencyDispatchArgs& args) {
    if constexpr (kDeviceSchedule) {
        low_latency_grouped_gemm_detail::
            low_latency_grouped_gemm_w4a8_prefetch_a_device_schedule_kernel<
                kScaleGroupSize><<<args.grid, args.block, 0, args.stream>>>(
                args.params, args.row_tiles);
    } else {
        low_latency_grouped_gemm_detail::
            low_latency_grouped_gemm_w4a8_prefetch_a_device_kernel<
                kScaleGroupSize><<<
                args.grid, args.block, 0, args.stream>>>(args.params);
    }
}

template <bool kDeviceSchedule, int kScaleGroupSize, int kKConst>
void launch_full_k_prefetch_const(const LowLatencyDispatchArgs& args) {
    if constexpr (kDeviceSchedule) {
        low_latency_grouped_gemm_detail::
            low_latency_grouped_gemm_w4a8_prefetch_a_const_device_schedule_kernel<
                kScaleGroupSize, kKConst><<<
                args.grid, args.block, 0, args.stream>>>(
                args.params, args.row_tiles);
    } else {
        low_latency_grouped_gemm_detail::
            low_latency_grouped_gemm_w4a8_prefetch_a_const_device_kernel<
                kScaleGroupSize, kKConst><<<
                args.grid, args.block, 0, args.stream>>>(args.params);
    }
}

template <bool kDeviceSchedule, int kScaleGroupSize>
bool launch_full_k_prefetch_const_if_supported(
    int K,
    const LowLatencyDispatchArgs& args) {
    switch (K) {
        case 512:
            launch_full_k_prefetch_const<kDeviceSchedule, kScaleGroupSize, 512>(
                args);
            return true;
        case 768:
            launch_full_k_prefetch_const<kDeviceSchedule, kScaleGroupSize, 768>(
                args);
            return true;
        case 1024:
            launch_full_k_prefetch_const<kDeviceSchedule, kScaleGroupSize, 1024>(
                args);
            return true;
        case 2048:
            launch_full_k_prefetch_const<kDeviceSchedule, kScaleGroupSize, 2048>(
                args);
            return true;
        case 4096:
            launch_full_k_prefetch_const<kDeviceSchedule, kScaleGroupSize, 4096>(
                args);
            return true;
        case 8192:
            launch_full_k_prefetch_const<kDeviceSchedule, kScaleGroupSize, 8192>(
                args);
            return true;
        default:
            return false;
    }
}

template <bool kDeviceSchedule, int kScaleGroupSize>
void launch_tail64_dynamic(const LowLatencyDispatchArgs& args) {
    if constexpr (kDeviceSchedule) {
        low_latency_grouped_gemm_detail::
            low_latency_grouped_gemm_w4a8_tail64_device_schedule_kernel<
                kScaleGroupSize><<<args.grid, args.block, 0, args.stream>>>(
                args.params, args.row_tiles);
    } else {
        low_latency_grouped_gemm_detail::
            low_latency_grouped_gemm_w4a8_tail64_device_kernel<
                kScaleGroupSize><<<
                args.grid, args.block, 0, args.stream>>>(args.params);
    }
}

template <bool kDeviceSchedule, int kScaleGroupSize, int kKConst>
void launch_tail64_const(const LowLatencyDispatchArgs& args) {
    if constexpr (kDeviceSchedule) {
        low_latency_grouped_gemm_detail::
            low_latency_grouped_gemm_w4a8_tail64_const_device_schedule_kernel<
                kScaleGroupSize, kKConst><<<
                args.grid, args.block, 0, args.stream>>>(
                args.params, args.row_tiles);
    } else {
        low_latency_grouped_gemm_detail::
            low_latency_grouped_gemm_w4a8_tail64_const_device_kernel<
                kScaleGroupSize, kKConst><<<
                args.grid, args.block, 0, args.stream>>>(args.params);
    }
}

template <bool kDeviceSchedule, int kScaleGroupSize>
bool launch_tail64_const_if_supported(int K,
                                      const LowLatencyDispatchArgs& args) {
    switch (K) {
        case 64:
            launch_tail64_const<kDeviceSchedule, kScaleGroupSize, 64>(args);
            return true;
        case 192:
            launch_tail64_const<kDeviceSchedule, kScaleGroupSize, 192>(args);
            return true;
        case 320:
            launch_tail64_const<kDeviceSchedule, kScaleGroupSize, 320>(args);
            return true;
        case 448:
            launch_tail64_const<kDeviceSchedule, kScaleGroupSize, 448>(args);
            return true;
        case 576:
            launch_tail64_const<kDeviceSchedule, kScaleGroupSize, 576>(args);
            return true;
        case 704:
            launch_tail64_const<kDeviceSchedule, kScaleGroupSize, 704>(args);
            return true;
        case 832:
            launch_tail64_const<kDeviceSchedule, kScaleGroupSize, 832>(args);
            return true;
        case 960:
            launch_tail64_const<kDeviceSchedule, kScaleGroupSize, 960>(args);
            return true;
        default:
            return false;
    }
}

template <bool kDeviceSchedule, int kScaleGroupSize>
void launch_low_latency_compute(int K, const LowLatencyDispatchArgs& args) {
    if ((K % 128) == 0) {
        if (use_full_k_prefetch_a(K)) {
            if (!launch_full_k_prefetch_const_if_supported<kDeviceSchedule,
                                                           kScaleGroupSize>(
                    K, args)) {
                launch_full_k_prefetch_dynamic<kDeviceSchedule,
                                               kScaleGroupSize>(args);
            }
            return;
        }
        launch_full_k_dynamic<kDeviceSchedule, kScaleGroupSize>(args);
        return;
    }

    if (!launch_tail64_const_if_supported<kDeviceSchedule, kScaleGroupSize>(
            K, args)) {
        launch_tail64_dynamic<kDeviceSchedule, kScaleGroupSize>(args);
    }
}

template <bool kDeviceSchedule>
void launch_low_latency_compute(int K,
                                int scale_group_size,
                                const LowLatencyDispatchArgs& args) {
    switch (scale_group_size) {
        case 64:
            launch_low_latency_compute<kDeviceSchedule, 64>(K, args);
            return;
        case 128:
            launch_low_latency_compute<kDeviceSchedule, 128>(K, args);
            return;
        default:
            abort_low_latency_grouped_gemm_args(
                "scale_group_size must be 64 or 128");
    }
}

}  // namespace

void launch_low_latency_grouped_gemm_interleave(
    const uint8_t*       w_orig_padded,
    const __nv_bfloat16* scales_n_major,
    uint8_t*             w_interleaved,
    __nv_bfloat16*       scales_padded,
    int                  G,
    int                  N_orig,
    int                  K,
    cudaStream_t         stream,
    int                  scale_group_size) {
    validate_common(G, N_orig, K, scale_group_size);
    if (G == 0) return;
    if (!w_orig_padded) abort_low_latency_grouped_gemm_args("w_orig_padded is null");
    if (!scales_n_major) abort_low_latency_grouped_gemm_args("scales_n_major is null");
    if (!w_interleaved) abort_low_latency_grouped_gemm_args("w_interleaved is null");
    if (!scales_padded) abort_low_latency_grouped_gemm_args("scales_padded is null");

    dim3 block(4, 64, 1);
    dim3 grid(N_orig < 1024 ? N_orig : 1024,
              G < 1024 ? G : 1024,
              1);
    switch (scale_group_size) {
        case 64:
            low_latency_grouped_gemm_detail::
                low_latency_grouped_gemm_interleave_kernel<64><<<
                    grid, block, 0, stream>>>(
                    reinterpret_cast<low_latency_grouped_gemm_detail::ElementA*>(
                        w_interleaved),
                    reinterpret_cast<
                        const low_latency_grouped_gemm_detail::ElementA*>(
                        w_orig_padded),
                    reinterpret_cast<low_latency_grouped_gemm_detail::ElementSF*>(
                        scales_padded),
                    reinterpret_cast<
                        const low_latency_grouped_gemm_detail::ElementSF*>(
                        scales_n_major),
                    G,
                    N_orig,
                    K);
            return;
        case 128:
            low_latency_grouped_gemm_detail::
                low_latency_grouped_gemm_interleave_kernel<128><<<
                    grid, block, 0, stream>>>(
                    reinterpret_cast<low_latency_grouped_gemm_detail::ElementA*>(
                        w_interleaved),
                    reinterpret_cast<
                        const low_latency_grouped_gemm_detail::ElementA*>(
                        w_orig_padded),
                    reinterpret_cast<low_latency_grouped_gemm_detail::ElementSF*>(
                        scales_padded),
                    reinterpret_cast<
                        const low_latency_grouped_gemm_detail::ElementSF*>(
                        scales_n_major),
                    G,
                    N_orig,
                    K);
            return;
        default:
            abort_low_latency_grouped_gemm_args(
                "scale_group_size must be 64 or 128");
    }
}

void launch_low_latency_grouped_gemm_build_tile_schedule(
    const int32_t* token_counts,
    int G,
    int tile_schedule_capacity,
    int32_t* tile_experts,
    int32_t* tile_n,
    int32_t* num_token_tiles,
    cudaStream_t stream) {
    if (G < 0) abort_low_latency_grouped_gemm_args("G must be non-negative");
    if (G == 0) return;
    if (tile_schedule_capacity <= 0) {
        abort_low_latency_grouped_gemm_args("tile_schedule_capacity must be positive");
    }
    if (!token_counts) abort_low_latency_grouped_gemm_args("token_counts is null");
    if (!tile_experts) abort_low_latency_grouped_gemm_args("tile_experts is null");
    if (!tile_n) abort_low_latency_grouped_gemm_args("tile_n is null");
    if (!num_token_tiles) abort_low_latency_grouped_gemm_args("num_token_tiles is null");

    check_cuda_launch_setup(
        cudaMemsetAsync(num_token_tiles, 0, sizeof(int32_t), stream),
        "failed to clear device tile count");
    constexpr int kBuildThreads = 128;
    const int build_blocks = (G + kBuildThreads - 1) / kBuildThreads;
    low_latency_grouped_gemm_detail::low_latency_grouped_gemm_build_tile_schedule_kernel<<<
        build_blocks, kBuildThreads, 0, stream>>>(
            token_counts,
            G,
            tile_schedule_capacity,
            tile_experts,
            tile_n,
            num_token_tiles);
}

void launch_low_latency_grouped_gemm(const LowLatencyGroupedGemmLaunchOpts& opts) {
    validate_common(opts.G, opts.N_orig, opts.K, opts.scale_group_size);
    if (opts.G == 0) return;
    if (!opts.acts) abort_low_latency_grouped_gemm_args("acts is null");
    if (!opts.w_interleaved) abort_low_latency_grouped_gemm_args("w_interleaved is null");
    if (!opts.scales_padded) abort_low_latency_grouped_gemm_args("scales_padded is null");
    if (!opts.token_counts) abort_low_latency_grouped_gemm_args("token_counts is null");
    if (!opts.expert_offsets) abort_low_latency_grouped_gemm_args("expert_offsets is null");
    if (!opts.outs) abort_low_latency_grouped_gemm_args("outs is null");

    const bool scheduled = opts.num_token_tiles > 0;
    const bool device_scheduled = opts.num_token_tiles_device != nullptr;
    if (!scheduled && !device_scheduled && opts.max_M_g <= 0) {
        abort_low_latency_grouped_gemm_args("max_M_g must be positive for rectangular launch");
    }
    if (scheduled && device_scheduled) {
        abort_low_latency_grouped_gemm_args("host and device schedules are mutually exclusive");
    }
    if (scheduled && (!opts.tile_experts || !opts.tile_n)) {
        abort_low_latency_grouped_gemm_args(
            "scheduled low_latency_grouped_gemm requires tile_experts and tile_n");
    }

    LowLatencyParams params = make_low_latency_params(opts);
    dim3 block(
        low_latency_grouped_gemm_detail::kThreadsPerRow,
        low_latency_grouped_gemm_detail::kThreadCount /
            low_latency_grouped_gemm_detail::kThreadsPerRow,
        1);
    const int row_tiles = (opts.N_orig / 4 + block.y - 1) / block.y;
    if (device_scheduled) {
        if (!opts.tile_experts || !opts.tile_n) {
            abort_low_latency_grouped_gemm_args(
                "device-scheduled low_latency_grouped_gemm requires tile arrays");
        }
        if (opts.persistent_ctas <= 0) {
            abort_low_latency_grouped_gemm_args(
                "device-scheduled low_latency_grouped_gemm requires persistent_ctas");
        }
        if (opts.build_device_schedule) {
            if (opts.tile_schedule_capacity <= 0) {
                abort_low_latency_grouped_gemm_args("device schedule capacity is too small");
            }
            launch_low_latency_grouped_gemm_build_tile_schedule(opts.token_counts,
                                                 opts.G,
                                                 opts.tile_schedule_capacity,
                                                 opts.tile_experts,
                                                 opts.tile_n,
                                                 opts.num_token_tiles_device,
                                                 opts.stream);
        }
        dim3 grid(opts.persistent_ctas, 1, 1);
        LowLatencyDispatchArgs dispatch{
            params, grid, block, row_tiles, opts.stream};
        launch_low_latency_compute<true>(
            opts.K, opts.scale_group_size, dispatch);
        return;
    }

    dim3 grid((opts.N_orig / 4 + block.y - 1) / block.y,
              scheduled ? 1 : (opts.max_M_g + 7) / 8,
              scheduled ? opts.num_token_tiles : opts.G);
    LowLatencyDispatchArgs dispatch{
        params, grid, block, row_tiles, opts.stream};
    launch_low_latency_compute<false>(opts.K, opts.scale_group_size, dispatch);
}

}  // namespace mga
