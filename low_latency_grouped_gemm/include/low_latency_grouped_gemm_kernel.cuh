// SPDX-License-Identifier: BSD-3-Clause
//
// This file is adapted from the local CUTLASS GEMM prototype in
// MaloGEMM_For_Sync/cutlass/include/cutlass/gemm/kernel/gemv.h.
// It keeps the same lightweight CTA/data mapping but writes directly to the
// the grouped-MoE output layout [M_total, N_orig].
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "cutlass/array.h"
#include "cutlass/arch/cache_operation.h"
#include "cutlass/arch/memory.h"
#include "cutlass/arch/mma.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

namespace mga::low_latency_grouped_gemm_detail {

using ElementA = cutlass::int4b_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = __nv_bfloat16;
using ElementSF = cutlass::bfloat16_t;

constexpr int kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int kThreadCount = 128;
constexpr int kThreadsPerRow = 8;
constexpr int kPackedElementsA = 2;
constexpr int kUnroll = 2;
constexpr int kInterleaveBlockK = 4 * kElementsPerAccess;

using FragmentA = cutlass::Array<ElementA, kElementsPerAccess>;
using FragmentB = cutlass::Array<ElementB, kElementsPerAccess>;
using FragmentArrayA = cutlass::Array<FragmentA, kUnroll>;
using FragmentArrayB = cutlass::Array<FragmentB, kUnroll>;
using FragmentArrayC = cutlass::Array<float, 4>;
using FragmentCompute = cutlass::Array<cutlass::half_t, kElementsPerAccess>;

static_assert(kInterleaveBlockK == 64,
              "each unroll covers one 64-K scale group");
static_assert(kUnroll * kInterleaveBlockK == 128,
              "two unrolls cover one 128-K scale group");

template <int kScaleGroupSize>
CUTLASS_GLOBAL
void low_latency_grouped_gemm_interleave_kernel(ElementA* A_interleaved,
                                 const ElementA* A,
                                 ElementSF* SFA_padded,
                                 const ElementSF* SFA,
                                 int B,
                                 int M,
                                 int K) {
    static_assert(kScaleGroupSize == 64 || kScaleGroupSize == 128,
                  "supported scale group sizes are 64 and 128");

    for (int b = blockIdx.y; b < B; b += gridDim.y) {
        for (int m = blockIdx.x; m < M; m += gridDim.x) {
            for (int k = threadIdx.y * kInterleaveBlockK;
                 k < K;
                 k += blockDim.y * kInterleaveBlockK) {
                if (k + threadIdx.x * kElementsPerAccess >= K) {
                    break;
                }

                const uint8_t* src = reinterpret_cast<const uint8_t*>(A);
                uint8_t* dst = reinterpret_cast<uint8_t*>(A_interleaved);

                src += static_cast<size_t>(b) * M * K / kPackedElementsA;
                dst += static_cast<size_t>(b) * M * K / kPackedElementsA;

                src += static_cast<size_t>(m) * K / kPackedElementsA;
                dst += static_cast<size_t>(m / 2) * K * 2 / kPackedElementsA +
                       static_cast<size_t>(m % 2) * kInterleaveBlockK / kPackedElementsA;

                src += k / kPackedElementsA +
                       threadIdx.x * kElementsPerAccess / kPackedElementsA;
                dst += k * 2 / kPackedElementsA +
                       threadIdx.x * kElementsPerAccess / kPackedElementsA;

                constexpr int kNumU32 =
                    cutlass::sizeof_bits<FragmentA>::value / 32;
                cutlass::Array<uint32_t, kNumU32> reordered;
                reordered.fill(0);

                const uint32_t* src_u32 = reinterpret_cast<const uint32_t*>(src);
                constexpr int kMap[8] = {0, 4, 1, 5, 2, 6, 3, 7};

                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < kNumU32; ++i) {
                    uint32_t value = src_u32[i];
                    CUTLASS_PRAGMA_UNROLL
                    for (int j = 0; j < 8; ++j) {
                        uint32_t int4_value = (value >> (j * 4)) & 0xF;
                        reordered[i] |= int4_value << (kMap[j] * 4);
                    }
                }

                *reinterpret_cast<FragmentA*>(dst) =
                    *reinterpret_cast<FragmentA*>(&reordered);

                if ((k % kScaleGroupSize) == 0 && threadIdx.x == 0) {
                    const int sfs_per_row =
                        (K + kScaleGroupSize - 1) / kScaleGroupSize;
                    const int padded_sfs_per_row = (sfs_per_row + 7) / 8 * 8;

                    const ElementSF* s_src =
                        SFA + (static_cast<size_t>(b) * M + m) * sfs_per_row +
                        k / kScaleGroupSize;
                    ElementSF* s_dst =
                        SFA_padded +
                        (static_cast<size_t>(b) * M + m) * padded_sfs_per_row +
                        k / kScaleGroupSize;
                    *s_dst = *s_src;
                }
            }
        }
    }
}

struct LowLatencyGroupedGemmKernel {
    struct Params {
        int32_t M;
        const int32_t* N;
        int32_t K;
        int32_t batch_count;
        float alpha;

        const ElementA* ptr_A;
        const ElementB* ptr_B;
        ElementC* ptr_D;

        int64_t batch_stride_A;
        const ElementSF* ptr_SFA;
        const int32_t* offsets;
        const int32_t* tile_experts;
        const int32_t* tile_n;
        int32_t num_token_tiles;
        const int32_t* num_token_tiles_device;
    };

    using MMA_16x8x16_F32F16F16 = cutlass::arch::Mma<
        cutlass::gemm::GemmShape<16, 8, 16>,
        32,
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::ColumnMajor,
        float,
        cutlass::layout::RowMajor,
        cutlass::arch::OpMultiplyAdd>;

    static bool can_implement(int K) {
        return (K > 0) && (K % (4 * kElementsPerAccess) == 0);
    }

    // Direct scale loads keep one path for short and long K and avoid the
    // old shared-scale staging limit at 32 scale groups.
    template <bool kAlwaysSecondUnroll, int kKConst = -1>
    CUTLASS_DEVICE
    static void load_a_fragments(Params const& params,
                                 const ElementA* ptr_A,
                                 int unroll_col_k,
                                 bool has_second_unroll,
                                 FragmentArrayA& frag_array_A_row0,
                                 FragmentArrayA& frag_array_A_row1) {
        constexpr int tileA_k = kThreadsPerRow * kElementsPerAccess;
        const int row_stride_a =
            ((kKConst > 0) ? kKConst : params.K) * 2 / kPackedElementsA;

        cutlass::arch::global_load<
            FragmentA,
            sizeof(FragmentA),
            cutlass::arch::CacheOperation::LastUse>(
            frag_array_A_row0[0],
            ptr_A + unroll_col_k / kPackedElementsA,
            true);
        cutlass::arch::global_load<
            FragmentA,
            sizeof(FragmentA),
            cutlass::arch::CacheOperation::LastUse>(
            frag_array_A_row1[0],
            ptr_A + unroll_col_k / kPackedElementsA + row_stride_a,
            true);

        if (kAlwaysSecondUnroll || has_second_unroll) {
            const int unroll_col_k_ = unroll_col_k + tileA_k;
            cutlass::arch::global_load<
                FragmentA,
                sizeof(FragmentA),
                cutlass::arch::CacheOperation::LastUse>(
                frag_array_A_row0[1],
                ptr_A + unroll_col_k_ / kPackedElementsA,
                true);
            cutlass::arch::global_load<
                FragmentA,
                sizeof(FragmentA),
                cutlass::arch::CacheOperation::LastUse>(
                frag_array_A_row1[1],
                ptr_A + unroll_col_k_ / kPackedElementsA + row_stride_a,
                true);
        }
    }

    template <bool kAlwaysSecondUnroll>
    CUTLASS_DEVICE
    static void load_b_fragments(const ElementB* ptr_B,
                                 int unroll_col_k,
                                 bool has_second_unroll,
                                 FragmentArrayB& frag_array_B) {
        constexpr int tileA_k = kThreadsPerRow * kElementsPerAccess;

        cutlass::arch::global_load<
            FragmentB,
            sizeof(FragmentB),
            cutlass::arch::CacheOperation::Always>(
            frag_array_B[0],
            ptr_B + unroll_col_k / 2,
            true);

        if (kAlwaysSecondUnroll || has_second_unroll) {
            const int unroll_col_k_ = unroll_col_k + tileA_k;
            cutlass::arch::global_load<
                FragmentB,
                sizeof(FragmentB),
                cutlass::arch::CacheOperation::Always>(
                frag_array_B[1],
                ptr_B + unroll_col_k_ / 2,
                true);
        }
    }

    template <bool kAlwaysSecondUnroll, int kKConst = -1>
    CUTLASS_DEVICE
    static void load_fragments(Params const& params,
                               const ElementA* ptr_A,
                               const ElementB* ptr_B,
                               int unroll_col_k,
                               bool has_second_unroll,
                               FragmentArrayA& frag_array_A_row0,
                               FragmentArrayA& frag_array_A_row1,
                               FragmentArrayB& frag_array_B) {
        load_a_fragments<kAlwaysSecondUnroll, kKConst>(params,
                                                      ptr_A,
                                                      unroll_col_k,
                                                      has_second_unroll,
                                                      frag_array_A_row0,
                                                      frag_array_A_row1);
        load_b_fragments<kAlwaysSecondUnroll>(ptr_B,
                                             unroll_col_k,
                                             has_second_unroll,
                                             frag_array_B);
    }

    template <bool kFullK, int kScaleGroupSize, int kScaleIdx = -1>
    CUTLASS_DEVICE
    static void compute_fragments(int unroll_col_k,
                                  bool has_second_unroll,
                                  const FragmentArrayA& frag_array_A_row0,
                                  const FragmentArrayA& frag_array_A_row1,
                                  const FragmentArrayB& frag_array_B,
                                  const ElementSF* g_SFA_row0,
                                  const ElementSF* g_SFA_row1,
                                  FragmentArrayC& frag_mma_c) {
        static_assert(kScaleGroupSize == 64 || kScaleGroupSize == 128,
                      "supported scale group sizes are 64 and 128");

        cutlass::NumericArrayConverter<
            cutlass::half_t,
            ElementA,
            kElementsPerAccess,
            cutlass::FloatRoundStyle::round_to_nearest>
            srcA_converter;
        cutlass::NumericArrayConverter<
            cutlass::half_t,
            ElementB,
            kElementsPerAccess,
            cutlass::FloatRoundStyle::round_to_nearest>
            srcB_converter;
        MMA_16x8x16_F32F16F16 mma_op;

        if constexpr (kScaleGroupSize == 128) {
            FragmentArrayC frag_mma_accum;
            frag_mma_accum.clear();

            CUTLASS_PRAGMA_UNROLL
            for (int unroll_idx = 0; unroll_idx < kUnroll; ++unroll_idx) {
                if (kFullK || unroll_idx == 0 || has_second_unroll) {
                    FragmentCompute fragA_compute_row0 =
                        srcA_converter(frag_array_A_row0[unroll_idx]);
                    FragmentCompute fragA_compute_row1 =
                        srcA_converter(frag_array_A_row1[unroll_idx]);
                    FragmentCompute fragB_compute =
                        srcB_converter(frag_array_B[unroll_idx]);

                    CUTLASS_PRAGMA_UNROLL
                    for (int e = 0; e < kElementsPerAccess; e += 4) {
                        cutlass::Array<cutlass::half_t, 8> frag_mma_a;
                        cutlass::Array<cutlass::half_t, 4> frag_mma_b;

                        uint32_t* mma_2xfp16_A =
                            reinterpret_cast<uint32_t*>(&frag_mma_a);
                        uint32_t* mma_2xfp16_B =
                            reinterpret_cast<uint32_t*>(&frag_mma_b);

                        const uint32_t* frag_2xfp16_A_row0 =
                            reinterpret_cast<const uint32_t*>(
                                &(fragA_compute_row0.data()[e]));
                        const uint32_t* frag_2xfp16_A_row1 =
                            reinterpret_cast<const uint32_t*>(
                                &(fragA_compute_row1.data()[e]));
                        const uint32_t* frag_2xfp16_B =
                            reinterpret_cast<const uint32_t*>(
                                &(fragB_compute.data()[e]));

                        mma_2xfp16_A[0] = frag_2xfp16_A_row0[0];
                        mma_2xfp16_A[1] = frag_2xfp16_A_row1[0];
                        mma_2xfp16_A[2] = frag_2xfp16_A_row0[1];
                        mma_2xfp16_A[3] = frag_2xfp16_A_row1[1];
                        mma_2xfp16_B[0] = frag_2xfp16_B[0];
                        mma_2xfp16_B[1] = frag_2xfp16_B[1];

                        mma_op(frag_mma_accum,
                               frag_mma_a,
                               frag_mma_b,
                               frag_mma_accum);
                    }
                }
            }

            const int scale_idx =
                (kScaleIdx >= 0)
                    ? kScaleIdx
                    : unroll_col_k / (2 * 128);
            const ElementSF SFA_row0 = g_SFA_row0[scale_idx];
            const ElementSF SFA_row1 = g_SFA_row1[scale_idx];
            frag_mma_c[0] += frag_mma_accum[0] * float(SFA_row0);
            frag_mma_c[1] += frag_mma_accum[1] * float(SFA_row0);
            frag_mma_c[2] += frag_mma_accum[2] * float(SFA_row1);
            frag_mma_c[3] += frag_mma_accum[3] * float(SFA_row1);
        } else {
            const int scale_idx_base =
                (kScaleIdx >= 0)
                    ? kScaleIdx
                    : unroll_col_k / (2 * 64);
            cutlass::Array<ElementSF, kUnroll> SFA_row0;
            cutlass::Array<ElementSF, kUnroll> SFA_row1;
            if (kFullK || has_second_unroll) {
                cutlass::arch::global_load<
                    cutlass::Array<ElementSF, kUnroll>,
                    sizeof(cutlass::Array<ElementSF, kUnroll>),
                    cutlass::arch::CacheOperation::Always>(
                    SFA_row0,
                    g_SFA_row0 + scale_idx_base,
                    true);
                cutlass::arch::global_load<
                    cutlass::Array<ElementSF, kUnroll>,
                    sizeof(cutlass::Array<ElementSF, kUnroll>),
                    cutlass::arch::CacheOperation::Always>(
                    SFA_row1,
                    g_SFA_row1 + scale_idx_base,
                    true);
            } else {
                SFA_row0[0] = g_SFA_row0[scale_idx_base];
                SFA_row1[0] = g_SFA_row1[scale_idx_base];
            }

            CUTLASS_PRAGMA_UNROLL
            for (int unroll_idx = 0; unroll_idx < kUnroll; ++unroll_idx) {
                if (kFullK || unroll_idx == 0 || has_second_unroll) {
                    FragmentArrayC frag_mma_accum;
                    frag_mma_accum.clear();

                    FragmentCompute fragA_compute_row0 =
                        srcA_converter(frag_array_A_row0[unroll_idx]);
                    FragmentCompute fragA_compute_row1 =
                        srcA_converter(frag_array_A_row1[unroll_idx]);
                    FragmentCompute fragB_compute =
                        srcB_converter(frag_array_B[unroll_idx]);

                    CUTLASS_PRAGMA_UNROLL
                    for (int e = 0; e < kElementsPerAccess; e += 4) {
                        cutlass::Array<cutlass::half_t, 8> frag_mma_a;
                        cutlass::Array<cutlass::half_t, 4> frag_mma_b;

                        uint32_t* mma_2xfp16_A =
                            reinterpret_cast<uint32_t*>(&frag_mma_a);
                        uint32_t* mma_2xfp16_B =
                            reinterpret_cast<uint32_t*>(&frag_mma_b);

                        const uint32_t* frag_2xfp16_A_row0 =
                            reinterpret_cast<const uint32_t*>(
                                &(fragA_compute_row0.data()[e]));
                        const uint32_t* frag_2xfp16_A_row1 =
                            reinterpret_cast<const uint32_t*>(
                                &(fragA_compute_row1.data()[e]));
                        const uint32_t* frag_2xfp16_B =
                            reinterpret_cast<const uint32_t*>(
                                &(fragB_compute.data()[e]));

                        mma_2xfp16_A[0] = frag_2xfp16_A_row0[0];
                        mma_2xfp16_A[1] = frag_2xfp16_A_row1[0];
                        mma_2xfp16_A[2] = frag_2xfp16_A_row0[1];
                        mma_2xfp16_A[3] = frag_2xfp16_A_row1[1];
                        mma_2xfp16_B[0] = frag_2xfp16_B[0];
                        mma_2xfp16_B[1] = frag_2xfp16_B[1];

                        mma_op(frag_mma_accum,
                               frag_mma_a,
                               frag_mma_b,
                               frag_mma_accum);
                    }

                    const ElementSF SFA0 = SFA_row0[unroll_idx];
                    const ElementSF SFA1 = SFA_row1[unroll_idx];
                    frag_mma_c[0] += frag_mma_accum[0] * float(SFA0);
                    frag_mma_c[1] += frag_mma_accum[1] * float(SFA0);
                    frag_mma_c[2] += frag_mma_accum[2] * float(SFA1);
                    frag_mma_c[3] += frag_mma_accum[3] * float(SFA1);
                }
            }
        }
    }

    template <int kKConst, int kOffset, int kScaleGroupSize>
    CUTLASS_DEVICE
    static void compute_tail64_const_fragments(
        Params const& params,
        const ElementA* ptr_A,
        const ElementB* ptr_B,
        FragmentArrayA& frag_array_A_row0,
        FragmentArrayA& frag_array_A_row1,
        FragmentArrayB& frag_array_B,
        const ElementSF* g_SFA_row0,
        const ElementSF* g_SFA_row1,
        FragmentArrayC& frag_mma_c) {
        static_assert(kKConst > 0 && (kKConst % 128) == 64,
                      "constant tail64 path requires K % 128 == 64");
        constexpr int tileA_k = kThreadsPerRow * kElementsPerAccess;
        constexpr int unroll_tile_k = kUnroll * tileA_k;
        constexpr int tail_col_k = kKConst * 2 - tileA_k;
        static_assert(tail_col_k >= 0, "K must be at least 64");

        if constexpr (kOffset < tail_col_k) {
            load_fragments<true, kKConst>(params,
                                          ptr_A,
                                          ptr_B,
                                          kOffset,
                                          true,
                                          frag_array_A_row0,
                                          frag_array_A_row1,
                                          frag_array_B);
            compute_fragments<true,
                              kScaleGroupSize,
                              kOffset / (2 * kScaleGroupSize)>(
                kOffset,
                true,
                frag_array_A_row0,
                frag_array_A_row1,
                frag_array_B,
                g_SFA_row0,
                g_SFA_row1,
                frag_mma_c);
            compute_tail64_const_fragments<kKConst,
                                           kOffset + unroll_tile_k,
                                           kScaleGroupSize>(
                params,
                ptr_A,
                ptr_B,
                frag_array_A_row0,
                frag_array_A_row1,
                frag_array_B,
                g_SFA_row0,
                g_SFA_row1,
                frag_mma_c);
        } else {
            static_assert(kOffset == tail_col_k,
                          "tail64 const recursion must end at the 64-wide tail");
            load_fragments<false, kKConst>(params,
                                           ptr_A,
                                           ptr_B,
                                           kOffset,
                                           false,
                                           frag_array_A_row0,
                                           frag_array_A_row1,
                                           frag_array_B);
            compute_fragments<false,
                              kScaleGroupSize,
                              kOffset / (2 * kScaleGroupSize)>(
                kOffset,
                false,
                frag_array_A_row0,
                frag_array_A_row1,
                frag_array_B,
                g_SFA_row0,
                g_SFA_row1,
                frag_mma_c);
        }
    }

    template <bool kTail64,
              bool kPrefetchA = false,
              int kKConst = -1,
              int kScaleGroupSize = 128>
    CUTLASS_DEVICE
    void compute_tile(Params const& params,
                      int row_tile,
                      int batch_idx,
                      int n_tile,
                      bool scheduled) {
        const int idx_col_k = threadIdx.x;
        const int idx_row_m = 4 * (row_tile * blockDim.y + threadIdx.y);
        const int N = params.N[batch_idx];
        constexpr bool kHasKConst = kKConst > 0;
        int effective_M = params.M;
        int effective_K = params.K;
        int64_t effective_batch_stride_A = params.batch_stride_A;
        int padded_sfs_per_row = 0;

        if constexpr (kHasKConst) {
            effective_K = kKConst;
            effective_batch_stride_A =
                static_cast<int64_t>(effective_M) * kKConst;
            padded_sfs_per_row =
                (((kKConst + kScaleGroupSize - 1) / kScaleGroupSize) + 7) /
                8 * 8;
        } else {
            padded_sfs_per_row =
                (((params.K + kScaleGroupSize - 1) / kScaleGroupSize) + 7) /
                8 * 8;
        }
        const int K_A_split = effective_K * 2;

        if (!scheduled && n_tile >= (N + 7) / 8) {
            return;
        }

        if (idx_row_m >= effective_M) {
            return;
        }

        const int token_base = params.offsets[batch_idx];

        const ElementA* ptr_A =
            params.ptr_A +
            static_cast<size_t>(batch_idx) * effective_batch_stride_A /
                kPackedElementsA;
        const ElementB* ptr_B =
            params.ptr_B + static_cast<size_t>(token_base) * effective_K;
        ElementC* ptr_D =
            params.ptr_D + static_cast<size_t>(token_base) * effective_M;

        ptr_A += idx_col_k * kElementsPerAccess / kPackedElementsA;
        ptr_B += (idx_col_k % 4) * kElementsPerAccess;

        ptr_A += static_cast<size_t>(idx_row_m) * effective_K /
                 kPackedElementsA;
        ptr_D += idx_row_m + idx_col_k / 4;

        const int n_B =
            (threadIdx.y % 4) * 2 + idx_col_k / 4 + 8 * n_tile;
        if (n_B < N) {
            ptr_B += static_cast<size_t>(n_B) * effective_K;
        }

        const int n_CD = (idx_col_k % 4) * 2 + 8 * n_tile;
        ptr_D += static_cast<size_t>(n_CD) * effective_M;

        const ElementSF* g_SFA_row0 =
            params.ptr_SFA +
            static_cast<size_t>(batch_idx) * effective_M *
                padded_sfs_per_row +
            static_cast<size_t>(idx_row_m + idx_col_k / 4) *
                padded_sfs_per_row;
        const ElementSF* g_SFA_row1 =
            g_SFA_row0 + 2 * padded_sfs_per_row;

        FragmentArrayC frag_mma_c;
        frag_mma_c.clear();

        FragmentArrayA frag_array_A_row0;
        FragmentArrayA frag_array_A_row1;
        FragmentArrayB frag_array_B;

        constexpr int tileA_k = kThreadsPerRow * kElementsPerAccess;
        constexpr int unroll_tile_k = kUnroll * tileA_k;

        if constexpr (!kTail64 && kPrefetchA) {
            FragmentArrayA frag_next_A_row0;
            FragmentArrayA frag_next_A_row1;

            load_fragments<true, kKConst>(params,
                                          ptr_A,
                                          ptr_B,
                                          0,
                                          true,
                                          frag_array_A_row0,
                                          frag_array_A_row1,
                                          frag_array_B);

            for (int unroll_col_k = 0;
                 unroll_col_k < K_A_split;
                 unroll_col_k += unroll_tile_k) {
                const int next_col_k = unroll_col_k + unroll_tile_k;
                const bool has_next = next_col_k < K_A_split;
                if (has_next) {
                    load_a_fragments<true, kKConst>(params,
                                                    ptr_A,
                                                    next_col_k,
                                                    true,
                                                    frag_next_A_row0,
                                                    frag_next_A_row1);
                }

                compute_fragments<true, kScaleGroupSize>(unroll_col_k,
                                                         true,
                                                         frag_array_A_row0,
                                                         frag_array_A_row1,
                                                         frag_array_B,
                                                         g_SFA_row0,
                                                         g_SFA_row1,
                                                         frag_mma_c);

                if (has_next) {
                    frag_array_A_row0 = frag_next_A_row0;
                    frag_array_A_row1 = frag_next_A_row1;
                    load_b_fragments<true>(ptr_B,
                                           next_col_k,
                                           true,
                                           frag_array_B);
                }
            }
        } else if constexpr (kTail64) {
            if constexpr (kKConst > 0) {
                compute_tail64_const_fragments<kKConst, 0, kScaleGroupSize>(
                    params,
                    ptr_A,
                    ptr_B,
                    frag_array_A_row0,
                    frag_array_A_row1,
                    frag_array_B,
                    g_SFA_row0,
                    g_SFA_row1,
                    frag_mma_c);
            } else {
                for (int unroll_col_k = 0;
                     unroll_col_k < K_A_split;
                     unroll_col_k += unroll_tile_k) {
                    const bool has_second_unroll =
                        unroll_col_k + tileA_k < K_A_split;
                    load_fragments<false>(params,
                                          ptr_A,
                                          ptr_B,
                                          unroll_col_k,
                                          has_second_unroll,
                                          frag_array_A_row0,
                                          frag_array_A_row1,
                                          frag_array_B);
                    compute_fragments<false, kScaleGroupSize>(
                        unroll_col_k,
                        has_second_unroll,
                        frag_array_A_row0,
                        frag_array_A_row1,
                        frag_array_B,
                        g_SFA_row0,
                        g_SFA_row1,
                        frag_mma_c);
                }
            }
        } else {
            for (int unroll_col_k = 0;
                 unroll_col_k < K_A_split;
                 unroll_col_k += unroll_tile_k) {
                load_fragments<true>(params,
                                     ptr_A,
                                     ptr_B,
                                     unroll_col_k,
                                     true,
                                     frag_array_A_row0,
                                     frag_array_A_row1,
                                     frag_array_B);
                compute_fragments<true, kScaleGroupSize>(unroll_col_k,
                                                         true,
                                                         frag_array_A_row0,
                                                         frag_array_A_row1,
                                                         frag_array_B,
                                                         g_SFA_row0,
                                                         g_SFA_row1,
                                                         frag_mma_c);
            }
        }

        if (n_CD < N) {
            *ptr_D = __float2bfloat16(frag_mma_c[0] * params.alpha);
            *(ptr_D + 2) =
                __float2bfloat16(frag_mma_c[2] * params.alpha);

            if (n_CD + 1 < N) {
                *(ptr_D + effective_M) =
                    __float2bfloat16(frag_mma_c[1] * params.alpha);
                *(ptr_D + effective_M + 2) =
                    __float2bfloat16(frag_mma_c[3] * params.alpha);
            }
        }
    }

    template <bool kTail64,
              bool kPrefetchA = false,
              int kKConst = -1,
              int kScaleGroupSize = 128>
    CUTLASS_DEVICE
    void operator()(Params const& params) {
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

        compute_tile<kTail64, kPrefetchA, kKConst, kScaleGroupSize>(
            params, blockIdx.x, batch_idx, n_tile, scheduled);
    }

    template <bool kTail64,
              bool kPrefetchA = false,
              int kKConst = -1,
              int kScaleGroupSize = 128>
    CUTLASS_DEVICE
    void persistent_device_schedule(Params const& params, int row_tiles) {
        const int num_token_tiles = *params.num_token_tiles_device;
        const int total_tasks = num_token_tiles * row_tiles;
        for (int task = blockIdx.x; task < total_tasks; task += gridDim.x) {
            const int row_tile = task % row_tiles;
            const int token_task = task / row_tiles;
            const int batch_idx = params.tile_experts[token_task];
            const int n_tile = params.tile_n[token_task];
            compute_tile<kTail64, kPrefetchA, kKConst, kScaleGroupSize>(
                params, row_tile, batch_idx, n_tile, true);
        }
    }
};

__global__
void low_latency_grouped_gemm_build_tile_schedule_kernel(const int32_t* token_counts,
                                          int G,
                                          int tile_schedule_capacity,
                                          int32_t* tile_experts,
                                          int32_t* tile_n,
                                          int32_t* num_token_tiles) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= G) return;

    const int count = token_counts[g];
    const int tiles = (count + 7) / 8;
    if (tiles <= 0) return;

    const int base = atomicAdd(num_token_tiles, tiles);
    if (base + tiles > tile_schedule_capacity) {
        asm volatile("trap;");
        return;
    }
    for (int t = 0; t < tiles; ++t) {
        tile_experts[base + t] = g;
        tile_n[base + t] = t;
    }
}

template <int kScaleGroupSize>
__global__ __maxnreg__(255)
void low_latency_grouped_gemm_w4a8_device_kernel(
    LowLatencyGroupedGemmKernel::Params params) {
    LowLatencyGroupedGemmKernel op;
    op.operator()<false, false, -1, kScaleGroupSize>(params);
}

template <int kScaleGroupSize>
__global__ __maxnreg__(255)
void low_latency_grouped_gemm_w4a8_prefetch_a_device_kernel(
    LowLatencyGroupedGemmKernel::Params params) {
    LowLatencyGroupedGemmKernel op;
    op.operator()<false, true, -1, kScaleGroupSize>(params);
}

template <int kScaleGroupSize, int kKConst>
__global__ __maxnreg__(255)
void low_latency_grouped_gemm_w4a8_prefetch_a_const_device_kernel(
    LowLatencyGroupedGemmKernel::Params params) {
    LowLatencyGroupedGemmKernel op;
    op.operator()<false, true, kKConst, kScaleGroupSize>(params);
}

template <int kScaleGroupSize>
__global__ __maxnreg__(255)
void low_latency_grouped_gemm_w4a8_tail64_device_kernel(
    LowLatencyGroupedGemmKernel::Params params) {
    LowLatencyGroupedGemmKernel op;
    op.operator()<true, false, -1, kScaleGroupSize>(params);
}

template <int kScaleGroupSize, int kKConst>
__global__ __maxnreg__(255)
void low_latency_grouped_gemm_w4a8_tail64_const_device_kernel(
    LowLatencyGroupedGemmKernel::Params params) {
    LowLatencyGroupedGemmKernel op;
    op.operator()<true, false, kKConst, kScaleGroupSize>(params);
}

template <int kScaleGroupSize>
__global__ __maxnreg__(255)
void low_latency_grouped_gemm_w4a8_device_schedule_kernel(
    LowLatencyGroupedGemmKernel::Params params, int row_tiles) {
    LowLatencyGroupedGemmKernel op;
    op.persistent_device_schedule<false, false, -1, kScaleGroupSize>(
        params, row_tiles);
}

template <int kScaleGroupSize>
__global__ __maxnreg__(255)
void low_latency_grouped_gemm_w4a8_prefetch_a_device_schedule_kernel(
    LowLatencyGroupedGemmKernel::Params params, int row_tiles) {
    LowLatencyGroupedGemmKernel op;
    op.persistent_device_schedule<false, true, -1, kScaleGroupSize>(
        params, row_tiles);
}

template <int kScaleGroupSize, int kKConst>
__global__ __maxnreg__(255)
void low_latency_grouped_gemm_w4a8_prefetch_a_const_device_schedule_kernel(
    LowLatencyGroupedGemmKernel::Params params, int row_tiles) {
    LowLatencyGroupedGemmKernel op;
    op.persistent_device_schedule<false, true, kKConst, kScaleGroupSize>(
        params, row_tiles);
}

template <int kScaleGroupSize>
__global__ __maxnreg__(255)
void low_latency_grouped_gemm_w4a8_tail64_device_schedule_kernel(
    LowLatencyGroupedGemmKernel::Params params, int row_tiles) {
    LowLatencyGroupedGemmKernel op;
    op.persistent_device_schedule<true, false, -1, kScaleGroupSize>(
        params, row_tiles);
}

template <int kScaleGroupSize, int kKConst>
__global__ __launch_bounds__(kThreadCount, 8)
void low_latency_grouped_gemm_w4a8_tail64_const_device_schedule_kernel(
    LowLatencyGroupedGemmKernel::Params params, int row_tiles) {
    LowLatencyGroupedGemmKernel op;
    op.persistent_device_schedule<true, false, kKConst, kScaleGroupSize>(
        params, row_tiles);
}

}  // namespace mga::low_latency_grouped_gemm_detail
