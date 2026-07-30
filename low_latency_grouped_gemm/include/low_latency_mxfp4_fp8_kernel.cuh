// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "cute/arch/mma_sm90_gmma.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"
#include "cute/tensor.hpp"
#include "cutlass/array.h"
#include "cutlass/arch/cache_operation.h"
#include "cutlass/arch/memory.h"
#include "cutlass/arch/mma.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

namespace mga::low_latency_mxfp4_fp8_detail {

using ElementB = cutlass::float_e4m3_t;
using ElementC = __nv_bfloat16;

constexpr int kElementsPerAccess = 16;
constexpr int kThreadCount = 128;
constexpr int kThreadsPerRow = 8;
constexpr int kMmaK = 32;
constexpr int kBStages = 8;

using FragmentPackedA = cutlass::Array<uint8_t, kElementsPerAccess / 2>;
using FragmentPackedAPair = cutlass::Array<uint8_t, kElementsPerAccess>;
using FragmentB = cutlass::Array<ElementB, kElementsPerAccess>;
using FragmentC = cutlass::Array<float, 4>;
using SmemLayoutBAtom =
    cute::SM90::GMMA::Layout_K_INTER_Atom<ElementB>;
using SmemLayoutB = decltype(cute::tile_to_shape(
    SmemLayoutBAtom{}, cute::Shape<cute::_8, cute::_32>{}));
constexpr int kSmemBElements = cute::cosize_v<SmemLayoutB>;

struct Fp8x8Raw {
    uint32_t low;
    uint32_t high;
};

CUTLASS_DEVICE
uint32_t prmt(uint32_t hi, uint32_t lo, uint32_t selector) {
    uint32_t result;
    asm volatile("prmt.b32 %0, %1, %2, %3;"
                 : "=r"(result)
                 : "r"(lo), "r"(hi), "r"(selector));
    return result;
}

CUTLASS_DEVICE
void load_lastuse_16(FragmentPackedAPair& dst, const void* ptr) {
    uint4& data = reinterpret_cast<uint4&>(dst);
    asm volatile(
        "ld.global.lu.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
        : "l"(ptr));
}

CUTLASS_DEVICE
void load_lastuse_8(FragmentPackedA& dst, const void* ptr) {
    uint2& data = reinterpret_cast<uint2&>(dst);
    asm volatile(
        "ld.global.lu.v2.u32 {%0, %1}, [%2];"
        : "=r"(data.x), "=r"(data.y)
        : "l"(ptr));
}

CUTLASS_DEVICE
void load_cached_16(FragmentB& dst, const void* ptr) {
    uint4& data = reinterpret_cast<uint4&>(dst);
    asm volatile(
        "ld.global.L2::128B.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
        : "l"(ptr));
}

// Exact Humming conversion for the sign-preprocessed physical weight layout.
// The low three E2M1 bits remain in logical nibble order; signs for outputs
// 0..3 are already in byte bit 7 and signs for outputs 4..7 in byte bit 3.
CUTLASS_DEVICE
Fp8x8Raw e2m1x8_to_scaled_e4m3x8(uint32_t fp4_raw,
                                  uint32_t exp_offset) {
    const uint32_t em_selector = fp4_raw & 0x77777777U;
    constexpr uint32_t kCodes0To3Bias = 0x0c080000U;
    constexpr uint32_t kCodes4To7Bias = 0x1c181410U;
    const uint32_t lut_low =
        exp_offset * 0x08080800U + kCodes0To3Bias;
    const uint32_t lut_high =
        exp_offset * 0x08080808U + kCodes4To7Bias;

    const uint32_t low_em = prmt(lut_high, lut_low, em_selector);
    const uint32_t high_em =
        prmt(lut_high, lut_low, em_selector >> 16U);

    const uint32_t low_signs = fp4_raw & 0x80808080U;
    const uint32_t high_signs =
        (fp4_raw << 4U) & 0x80808080U;

    return {low_em | low_signs, high_em | high_signs};
}

CUTLASS_DEVICE
cutlass::Array<ElementB, kElementsPerAccess>
e2m1x16_to_scaled_e4m3x16(const FragmentPackedA& packed,
                          uint32_t exp_offset) {
    cutlass::Array<ElementB, kElementsPerAccess> result;
    const uint32_t* src = reinterpret_cast<const uint32_t*>(&packed);
    uint32_t* dst = reinterpret_cast<uint32_t*>(&result);
    const Fp8x8Raw first =
        e2m1x8_to_scaled_e4m3x8(src[0], exp_offset);
    const Fp8x8Raw second =
        e2m1x8_to_scaled_e4m3x8(src[1], exp_offset);
    dst[0] = first.low;
    dst[1] = first.high;
    dst[2] = second.low;
    dst[3] = second.high;
    return result;
}

struct LowLatencyMxfp4Fp8Kernel {
    struct Params {
        int32_t M;
        const int32_t* N;
        int32_t K;
        int32_t batch_count;

        const uint8_t* ptr_A;
        const ElementB* ptr_B;
        const uint8_t* ptr_exp_offsets;
        const float* ptr_token_scales;
        ElementC* ptr_D;

        const int32_t* offsets;
        const int32_t* tile_experts;
        const int32_t* tile_n;
        int32_t num_token_tiles;
        const int32_t* num_token_tiles_device;
    };

    using WGMMA_64x8x32_F32E4M3E4M3 =
        cute::SM90::GMMA::MMA_64x8x32_F32E4M3E4M3_RS_TN<>;

    static bool can_implement(int K) {
        return K > 0 && K % 64 == 0;
    }

    template <int kMConst = -1, int kKConst = -1>
    CUTLASS_DEVICE
    void compute_tile(Params const& params,
                      int row_tile,
                      int batch_idx,
                      int n_tile,
                      bool scheduled,
                      ElementB* smem_b_storage) {
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int token_count = params.N[batch_idx];
        constexpr bool kHasMConst = kMConst > 0;
        constexpr bool kHasKConst = kKConst > 0;
        constexpr bool kAlwaysSecondTile =
            kHasMConst && (kMConst % 128 == 0);
        const int matrix_m = kHasMConst ? kMConst : params.M;
        const int matrix_k = kHasKConst ? kKConst : params.K;

        if (!scheduled && n_tile >= (token_count + 7) / 8) {
            return;
        }

        const int token_base = params.offsets[batch_idx];
        const size_t weight_expert_stride =
            static_cast<size_t>(matrix_m) * matrix_k / 2;
        const size_t offset_expert_stride =
            static_cast<size_t>(matrix_m) * matrix_k / 32;
        const int k32_count = matrix_k / kMmaK;
        const size_t weight_tile128_stride =
            static_cast<size_t>(k32_count) * kThreadCount *
            sizeof(FragmentPackedAPair);
        const int tile64_base = row_tile * 2;
        const int full_pair_count = matrix_m / 128;
        const bool has_second_tile =
            kAlwaysSecondTile || tile64_base * 64 + 64 < matrix_m;
        const uint8_t* ptr_A_tile =
            params.ptr_A +
            static_cast<size_t>(batch_idx) * weight_expert_stride +
            static_cast<size_t>(row_tile) * weight_tile128_stride;
        const uint8_t* ptr_exp_tile =
            params.ptr_exp_offsets +
            static_cast<size_t>(batch_idx) * offset_expert_stride +
            static_cast<size_t>(has_second_tile ? row_tile
                                                : full_pair_count) *
                k32_count * 128;

        FragmentC accum0;
        FragmentC accum1;
        accum0.clear();
        accum1.clear();
        const int t0 = tid & 3;
        const int t1 = (tid >> 2) & 7;
        const int t2 = tid >> 5;
        const int load_stage = tid >> 4;
        const int load_slot = tid & 15;
        const int token_local = load_slot >> 1;
        const int k_local = (load_slot & 1) * kElementsPerAccess;
        auto smem_b = cute::make_tensor(
            cute::make_smem_ptr(
                smem_b_storage + load_stage * kSmemBElements),
            SmemLayoutB{});
        *reinterpret_cast<uint4*>(&smem_b(token_local, k_local)) =
            make_uint4(0U, 0U, 0U, 0U);
        __syncthreads();

        for (int k32_base = 0;
             k32_base < k32_count;
             k32_base += kBStages) {
            const int stage_count =
                min(kBStages, k32_count - k32_base);
            if (load_stage < stage_count) {
                const int token = n_tile * 8 + token_local;
                if (token < token_count) {
                    FragmentB packed_b;
                    load_cached_16(
                        packed_b,
                        params.ptr_B +
                            static_cast<size_t>(token_base + token) *
                                matrix_k +
                            (k32_base + load_stage) * kMmaK +
                            k_local);
                    auto smem_b = cute::make_tensor(
                        cute::make_smem_ptr(
                            smem_b_storage +
                            load_stage * kSmemBElements),
                        SmemLayoutB{});
                    *reinterpret_cast<uint4*>(
                        &smem_b(token_local, k_local)) =
                        *reinterpret_cast<const uint4*>(&packed_b);
                }
            }
            __syncthreads();

            CUTLASS_PRAGMA_UNROLL
            for (int stage = 0; stage < kBStages; ++stage) {
                if (stage < stage_count) {
                    const int k32_idx = k32_base + stage;
                    FragmentPackedAPair packed_a;
                    packed_a.clear();
                    if (has_second_tile) {
                        load_lastuse_16(
                            packed_a,
                            ptr_A_tile +
                                (static_cast<size_t>(k32_idx) *
                                     kThreadCount +
                                 tid) * sizeof(FragmentPackedAPair));
                    } else {
                        FragmentPackedA packed_tail;
                        load_lastuse_8(
                            packed_tail,
                            ptr_A_tile +
                                (static_cast<size_t>(k32_idx) *
                                     kThreadCount +
                                 tid) * sizeof(FragmentPackedA));
                        *reinterpret_cast<uint64_t*>(&packed_a) =
                            *reinterpret_cast<const uint64_t*>(&packed_tail);
                    }

                    uint32_t packed_exp = 0;
                    if (has_second_tile) {
                        packed_exp = *reinterpret_cast<const uint32_t*>(
                            ptr_exp_tile +
                            static_cast<size_t>(k32_idx) * 128 +
                            static_cast<size_t>(tid >> 2) * 4);
                    } else {
                        packed_exp = *reinterpret_cast<const uint16_t*>(
                            ptr_exp_tile +
                            static_cast<size_t>(k32_idx) * 64 +
                            static_cast<size_t>(tid >> 2) * 2);
                    }
                    const uint32_t exp00 = packed_exp & 0xffU;
                    const uint32_t exp01 = (packed_exp >> 8U) & 0xffU;
                    const uint32_t exp10 = (packed_exp >> 16U) & 0xffU;
                    const uint32_t exp11 = packed_exp >> 24U;

                    const uint32_t* packed_u32 =
                        reinterpret_cast<const uint32_t*>(&packed_a);
                    const Fp8x8Raw fp8_00 =
                        e2m1x8_to_scaled_e4m3x8(
                            packed_u32[0], exp00);
                    const Fp8x8Raw fp8_01 =
                        e2m1x8_to_scaled_e4m3x8(
                            packed_u32[1], exp01);
                    Fp8x8Raw fp8_10{};
                    Fp8x8Raw fp8_11{};
                    if (has_second_tile) {
                        fp8_10 = e2m1x8_to_scaled_e4m3x8(
                            packed_u32[2], exp10);
                        fp8_11 = e2m1x8_to_scaled_e4m3x8(
                            packed_u32[3], exp11);
                    }
                    auto smem_b = cute::make_tensor(
                        cute::make_smem_ptr(
                            smem_b_storage +
                            stage * kSmemBElements),
                        SmemLayoutB{});
                    const uint64_t desc_b =
                        cute::SM90::GMMA::make_gmma_desc<
                            cute::SM90::GMMA::Major::K>(smem_b);

                    cute::warpgroup_arrive();
                    WGMMA_64x8x32_F32E4M3E4M3::fma(
                        fp8_00.low,
                        fp8_01.low,
                        fp8_00.high,
                        fp8_01.high,
                        desc_b,
                        accum0[0],
                        accum0[1],
                        accum0[2],
                        accum0[3],
                        cute::SM90::GMMA::ScaleOut::One);
                    if (has_second_tile) {
                        WGMMA_64x8x32_F32E4M3E4M3::fma(
                            fp8_10.low,
                            fp8_11.low,
                            fp8_10.high,
                            fp8_11.high,
                            desc_b,
                            accum1[0],
                            accum1[1],
                            accum1[2],
                            accum1[3],
                            cute::SM90::GMMA::ScaleOut::One);
                    }
                    constexpr int kWgmmaK32PerGroup = 4;
                    if ((stage + 1) % kWgmmaK32PerGroup == 0 ||
                        stage + 1 == stage_count) {
                        cute::warpgroup_commit_batch();
                        if (stage + 1 == stage_count) {
                            cute::warpgroup_wait<0>();
                        } else {
                            cute::warpgroup_wait<1>();
                        }
                    }
                }
            }
            __syncthreads();
        }

        const int row0 = row_tile * 128 + t1 + t2 * 16;
        const int row1 = row0 + 8;
        const int row2 = row0 + 64;
        const int row3 = row1 + 64;
        const int n_CD = n_tile * 8 + t0 * 2;
        if (n_CD < token_count) {
            const float scale0 =
                params.ptr_token_scales[token_base + n_CD];
            ElementC* out0 =
                params.ptr_D +
                static_cast<size_t>(token_base + n_CD) * matrix_m;
            out0[row0] = __float2bfloat16(accum0[0] * scale0);
            out0[row1] = __float2bfloat16(accum0[2] * scale0);
            if (has_second_tile) {
                out0[row2] = __float2bfloat16(accum1[0] * scale0);
                out0[row3] = __float2bfloat16(accum1[2] * scale0);
            }

            if (n_CD + 1 < token_count) {
                const float scale1 =
                    params.ptr_token_scales[token_base + n_CD + 1];
                ElementC* out1 = out0 + matrix_m;
                out1[row0] = __float2bfloat16(accum0[1] * scale1);
                out1[row1] = __float2bfloat16(accum0[3] * scale1);
                if (has_second_tile) {
                    out1[row2] = __float2bfloat16(accum1[1] * scale1);
                    out1[row3] = __float2bfloat16(accum1[3] * scale1);
                }
            }
        }
    }

    template <int kMConst = -1, int kKConst = -1>
    CUTLASS_DEVICE
    void operator()(Params const& params, ElementB* smem_b) {
        int batch_idx = blockIdx.z;
        int n_tile = blockIdx.y;
        const bool scheduled = params.num_token_tiles > 0;
        if (scheduled) {
            const int task = blockIdx.z;
            if (task >= params.num_token_tiles) {
                return;
            }
            batch_idx = params.tile_experts[task];
            n_tile = params.tile_n[task];
        } else if (batch_idx >= params.batch_count) {
            return;
        }
        compute_tile<kMConst, kKConst>(
            params,
            blockIdx.x,
            batch_idx,
            n_tile,
            scheduled,
            smem_b);
    }

    template <int kMConst = -1, int kKConst = -1>
    CUTLASS_DEVICE
    void persistent_device_schedule(Params const& params,
                                    int row_tiles,
                                    ElementB* smem_b) {
        const int effective_row_tiles =
            kMConst > 0 ? (kMConst + 127) / 128 : row_tiles;
        const int num_token_tiles = *params.num_token_tiles_device;
        const int total_tasks = num_token_tiles * effective_row_tiles;
        const int tasks_per_cta = total_tasks / gridDim.x;
        const int extra_tasks = total_tasks % gridDim.x;
        const int cta = blockIdx.x;
        const int task_count =
            tasks_per_cta + static_cast<int>(cta < extra_tasks);
        const int task_begin =
            cta * tasks_per_cta + min(cta, extra_tasks);
        int row_tile = task_begin % effective_row_tiles;
        int token_task = task_begin / effective_row_tiles;
        for (int local_task = 0;
             local_task < task_count;
             ++local_task) {
            const int batch_idx = params.tile_experts[token_task];
            const int n_tile = params.tile_n[token_task];
            compute_tile<kMConst, kKConst>(
                params,
                row_tile,
                batch_idx,
                n_tile,
                true,
                smem_b);
            ++row_tile;
            if (row_tile == effective_row_tiles) {
                row_tile = 0;
                ++token_task;
            }
        }
    }

};

template <int kMConst = -1, int kKConst = -1>
__global__ __maxnreg__(255)
void low_latency_mxfp4_fp8_kernel(
    LowLatencyMxfp4Fp8Kernel::Params params) {
    __shared__ __align__(16)
        uint8_t smem_b_raw[kBStages * kSmemBElements];
    LowLatencyMxfp4Fp8Kernel op;
    op.operator()<kMConst, kKConst>(
        params, reinterpret_cast<ElementB*>(smem_b_raw));
}

template <int kMConst = -1, int kKConst = -1>
__global__ __maxnreg__(255)
void low_latency_mxfp4_fp8_device_schedule_kernel(
    LowLatencyMxfp4Fp8Kernel::Params params,
    int row_tiles) {
    __shared__ __align__(16)
        uint8_t smem_b_raw[kBStages * kSmemBElements];
    LowLatencyMxfp4Fp8Kernel op;
    op.persistent_device_schedule<kMConst, kKConst>(
        params, row_tiles, reinterpret_cast<ElementB*>(smem_b_raw));
}

}  // namespace mga::low_latency_mxfp4_fp8_detail
