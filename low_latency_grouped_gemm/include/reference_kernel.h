// SPDX-License-Identifier: Apache-2.0
// fp32 self-rolled reference for the grouped GEMM W4A8.
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace mga {

// Compute outs[m, n] = bf16(act_scale * Σ_k acts[m, k]
//                                              * scales[g_of_m, k/group, n]
//                                              * int4_sym(W[g_of_m][k, n]))
// for every token m in any expert (with M_g > 0).
//
// All inputs match the exact layouts used by the test driver (same as the
// production kernel inputs):
//   acts          : fp8[M_total, K]              row-major
//   W_orig        : int4[G, N_orig, K], packed   2 nibbles/byte along K
//   scales        : bf16[G, K/group, N_orig]     row-major
//   token_to_expt : int32[M_total]   token -> expert id (pre-built on host)
//   outs          : bf16[M_total, N_orig]
void launch_reference_grouped_gemm(const __nv_fp8_e4m3*  acts,
                                   const uint8_t*        W_orig,
                                   const __nv_bfloat16*  scales,
                                   const int*            token_to_expt,
	                                   float                 act_scale,
	                                   __nv_bfloat16*        outs,
	                                   int G, int N_orig, int K, int M_total,
	                                   cudaStream_t stream,
	                                   int scale_group_size = 128);

}  // namespace mga
