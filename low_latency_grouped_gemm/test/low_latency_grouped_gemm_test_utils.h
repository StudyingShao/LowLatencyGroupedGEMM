// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "low_latency_grouped_gemm/include/low_latency_grouped_gemm.h"
#include "low_latency_grouped_gemm/test/token_profiles.h"

#define LLGG_CUDA_CHECK(expr)                                                \
    do {                                                                   \
        cudaError_t err__ = (expr);                                        \
        if (err__ != cudaSuccess) {                                        \
            std::fprintf(stderr,                                           \
                         "CUDA error %s at %s:%d (%s)\n",                 \
                         cudaGetErrorString(err__),                        \
                         __FILE__,                                        \
                         __LINE__,                                        \
                         #expr);                                          \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

namespace mga::low_latency_grouped_gemm_test {

constexpr int kFC1N = 384;
constexpr int kFC1K = 4096;
constexpr int kFC2N = 4096;
constexpr int kFC2K = 192;

struct ProblemRun {
    std::string name;
    int N_orig;
    int K;
};

inline std::vector<ProblemRun> default_problem_runs() {
    return {
        {"FC1", kFC1N, kFC1K},
        {"FC2", kFC2N, kFC2K},
    };
}

inline std::vector<ProblemRun> problem_runs_for_shape(const char* shape) {
    if (!shape ||
        std::strcmp(shape, "default") == 0 ||
        std::strcmp(shape, "target192") == 0) {
        return default_problem_runs();
    }
    if (std::strcmp(shape, "moe128") == 0) {
        return {
            {"FC1", 1024, 4096},
            {"FC2", 4096, 512},
        };
    }
    if (std::strcmp(shape, "moe256_h7168_i3072") == 0) {
        return {
            {"FC1", 6144, 7168},
            {"FC2", 7168, 3072},
        };
    }
    std::fprintf(stderr,
                 "Unknown shape preset '%s' "
                 "(allowed: default, moe128, moe256_h7168_i3072)\n",
                 shape);
    std::exit(1);
}

inline uint8_t pack_int4_pair(int lo, int hi) {
    return static_cast<uint8_t>((lo & 0xF) | ((hi & 0xF) << 4));
}

inline std::vector<uint8_t>
gen_int4_weights(int G, int N, int K, std::mt19937& rng) {
    std::uniform_int_distribution<int> d(-8, 7);
    std::vector<uint8_t> W(static_cast<size_t>(G) * N * (K / 2));
    for (uint8_t& v : W) {
        v = pack_int4_pair(d(rng), d(rng));
    }
    return W;
}

inline std::vector<__nv_bfloat16>
gen_scales(int G,
           int N,
           int K,
           std::mt19937& rng,
           int scale_group_size = 128) {
    std::uniform_real_distribution<float> d(0.05f, 0.5f);
    const int n_kg =
        mga::low_latency_grouped_gemm_scale_groups_per_row(K, scale_group_size);
    std::vector<__nv_bfloat16> S(static_cast<size_t>(G) * n_kg * N);
    for (auto& s : S) s = __float2bfloat16(d(rng));
    return S;
}

inline std::vector<__nv_fp8_e4m3>
gen_acts(int M_total, int K, std::mt19937& rng) {
    std::uniform_real_distribution<float> d(-1.5f, 1.5f);
    std::vector<__nv_fp8_e4m3> A(static_cast<size_t>(M_total) * K);
    for (auto& a : A) a = __nv_fp8_e4m3(d(rng));
    return A;
}

inline std::vector<int32_t> counts_from_offsets(const std::vector<int>& offsets) {
    std::vector<int32_t> counts(offsets.size() - 1);
    for (size_t g = 0; g < counts.size(); ++g) {
        counts[g] = offsets[g + 1] - offsets[g];
    }
    return counts;
}

inline std::vector<int32_t> offsets_i32(const std::vector<int>& offsets) {
    std::vector<int32_t> out(offsets.size());
    for (size_t i = 0; i < offsets.size(); ++i) {
        out[i] = static_cast<int32_t>(offsets[i]);
    }
    return out;
}

inline std::vector<int> token_to_expert(const std::vector<int>& offsets) {
    const int M_total = offsets.empty() ? 0 : offsets.back();
    std::vector<int> tok(M_total, -1);
    for (int g = 0; g + 1 < static_cast<int>(offsets.size()); ++g) {
        for (int m = offsets[g]; m < offsets[g + 1]; ++m) {
            tok[m] = g;
        }
    }
    return tok;
}

struct TokenTile {
    int32_t expert;
    int32_t tile_n;
};

inline void append_expert_major_tiles(const std::vector<int>& offsets,
                                      std::vector<TokenTile>& tiles) {
    auto emit_expert = [&](int g) {
        if (g < 0 || g + 1 >= static_cast<int>(offsets.size())) {
            return;
        }
        const int count = offsets[g + 1] - offsets[g];
        for (int t = 0; t < (count + 7) / 8; ++t) {
            tiles.push_back({static_cast<int32_t>(g), static_cast<int32_t>(t)});
        }
    };

    for (size_t g = 0; g + 1 < offsets.size(); ++g) {
        emit_expert(static_cast<int>(g));
    }
}

inline int schedule_stream_count(int row_tiles,
                                 int persistent_ctas,
                                 int token_tiles) {
    if (token_tiles <= 0) return 1;
    if (row_tiles <= 0 || persistent_ctas <= 0) return 1;
    const int streams = (persistent_ctas + row_tiles - 1) / row_tiles;
    return std::min(token_tiles, std::max(1, streams));
}

inline void store_token_tiles(const std::vector<TokenTile>& tiles,
                              std::vector<int32_t>& tile_experts,
                              std::vector<int32_t>& tile_n) {
    tile_experts.resize(tiles.size());
    tile_n.resize(tiles.size());
    for (size_t i = 0; i < tiles.size(); ++i) {
        tile_experts[i] = tiles[i].expert;
        tile_n[i] = tiles[i].tile_n;
    }
}

inline void make_cta_local_token_tiles(const std::vector<TokenTile>& expert_major,
                                       int row_tiles,
                                       int persistent_ctas,
                                       std::vector<int32_t>& tile_experts,
                                       std::vector<int32_t>& tile_n) {
    const int token_tiles = static_cast<int>(expert_major.size());
    const int streams =
        schedule_stream_count(row_tiles, persistent_ctas, token_tiles);
    tile_experts.resize(expert_major.size());
    tile_n.resize(expert_major.size());

    int src = 0;
    for (int stream = 0; stream < streams; ++stream) {
        for (int dst = stream; dst < token_tiles; dst += streams) {
            tile_experts[dst] = expert_major[src].expert;
            tile_n[dst] = expert_major[src].tile_n;
            ++src;
        }
    }
}

inline void make_warp_swizzle_token_tiles(const std::vector<int>& offsets,
                                          std::vector<int32_t>& tile_experts,
                                          std::vector<int32_t>& tile_n) {
    tile_experts.clear();
    tile_n.clear();
    auto emit_expert = [&](int g) {
        if (g < 0 || g + 1 >= static_cast<int>(offsets.size())) {
            return;
        }
        const int count = offsets[g + 1] - offsets[g];
        for (int t = 0; t < (count + 7) / 8; ++t) {
            tile_experts.push_back(static_cast<int32_t>(g));
            tile_n.push_back(static_cast<int32_t>(t));
        }
    };

    constexpr int kWarp = 32;
    constexpr int kChunkOrder[] = {0, 128, 64, 160, 96, 32};
    for (int chunk_base : kChunkOrder) {
        for (int lane = 0; lane < kWarp; ++lane) {
            emit_expert(chunk_base + lane);
        }
    }
    for (int chunk_base = 192;
         chunk_base + 1 < static_cast<int>(offsets.size());
         chunk_base += kWarp) {
        for (int lane = 0; lane < kWarp; ++lane) {
            emit_expert(chunk_base + lane);
        }
    }
}

inline void make_token_tiles(const std::vector<int>& offsets,
                             std::vector<int32_t>& tile_experts,
                             std::vector<int32_t>& tile_n,
                             int row_tiles = 0,
                             int persistent_ctas = 0) {
    tile_experts.clear();
    tile_n.clear();

    std::vector<TokenTile> expert_major;
    append_expert_major_tiles(offsets, expert_major);

    const char* order_env = std::getenv("LOW_LATENCY_GROUPED_GEMM_SCHEDULE_ORDER");
    const char* order = order_env ? order_env : "cta_local";
    if (std::strcmp(order, "expert") == 0 ||
        std::strcmp(order, "expert_major") == 0) {
        store_token_tiles(expert_major, tile_experts, tile_n);
        return;
    }

    if (std::strcmp(order, "cta_local") == 0 ||
        std::strcmp(order, "cta") == 0) {
        // Persistent CTAs consume token tasks at a stride of roughly
        // persistent_ctas / row_tiles.  Transposing an expert-major list over
        // that stream count makes each CTA's private task sequence reuse the
        // same or adjacent experts as often as possible.
        make_cta_local_token_tiles(
            expert_major, row_tiles, persistent_ctas, tile_experts, tile_n);
        return;
    }

    if (std::strcmp(order, "warp_swizzle") == 0) {
        make_warp_swizzle_token_tiles(offsets, tile_experts, tile_n);
        return;
    }

    std::fprintf(stderr,
                 "Unknown schedule order '%s' "
                 "(allowed: expert, cta_local, warp_swizzle)\n",
                 order);
    std::exit(1);
}

inline int max_tokens_per_expert(const std::vector<int>& offsets) {
    int max_mg = 0;
    for (size_t g = 0; g + 1 < offsets.size(); ++g) {
        max_mg = std::max(max_mg, offsets[g + 1] - offsets[g]);
    }
    return max_mg;
}

inline int current_sm_count() {
    int device = 0;
    int sm_count = 0;
    LLGG_CUDA_CHECK(cudaGetDevice(&device));
    LLGG_CUDA_CHECK(cudaDeviceGetAttribute(
        &sm_count, cudaDevAttrMultiProcessorCount, device));
    return sm_count;
}

inline int default_persistent_ctas(int N, int K_compute, int M_total) {
    if (const char* env = std::getenv("LOW_LATENCY_GROUPED_GEMM_PERSISTENT_CTAS")) {
        const int value = std::atoi(env);
        if (value > 0) return value;
    }
    const int row_tiles = (N + 63) / 64;
    const int upper_bound_tasks = M_total * row_tiles;
    if (upper_bound_tasks <= 0) return 1;
    if (K_compute <= 256 && N >= 2048) {
        return std::min(upper_bound_tasks, 4096);
    }
    return std::min(upper_bound_tasks, current_sm_count() * 8);
}

inline std::vector<uint8_t>
pad_int4_weights_k(const std::vector<uint8_t>& W,
                   int G,
                   int N,
                   int K,
                   int K_compute) {
    std::vector<uint8_t> out(static_cast<size_t>(G) * N * (K_compute / 2), 0);
    const size_t src_row_bytes = K / 2;
    const size_t dst_row_bytes = K_compute / 2;
    for (int g = 0; g < G; ++g) {
        for (int n = 0; n < N; ++n) {
            std::memcpy(out.data() + (static_cast<size_t>(g) * N + n) * dst_row_bytes,
                        W.data() + (static_cast<size_t>(g) * N + n) * src_row_bytes,
                        src_row_bytes);
        }
    }
    return out;
}

inline std::vector<__nv_fp8_e4m3>
pad_acts_k(const std::vector<__nv_fp8_e4m3>& A,
           int rows,
           int K,
           int K_compute) {
    std::vector<__nv_fp8_e4m3> out(static_cast<size_t>(rows) * K_compute,
                                   __nv_fp8_e4m3(0.0f));
    for (int m = 0; m < rows; ++m) {
        std::memcpy(out.data() + static_cast<size_t>(m) * K_compute,
                    A.data() + static_cast<size_t>(m) * K,
                    static_cast<size_t>(K) * sizeof(__nv_fp8_e4m3));
    }
    return out;
}

inline std::vector<__nv_bfloat16>
transpose_scales_for_low_latency_grouped_gemm(const std::vector<__nv_bfloat16>& S,
                               int G,
                               int N,
                               int K,
                               int K_compute,
                               int scale_group_size = 128) {
    const int n_kg =
        mga::low_latency_grouped_gemm_scale_groups_per_row(K, scale_group_size);
    const int n_kg_compute =
        mga::low_latency_grouped_gemm_scale_groups_per_row(K_compute,
                                                           scale_group_size);
    std::vector<__nv_bfloat16> out(static_cast<size_t>(G) * N * n_kg_compute,
                                   __float2bfloat16(0.0f));
    for (int g = 0; g < G; ++g) {
        for (int n = 0; n < N; ++n) {
            for (int kg = 0; kg < n_kg_compute; ++kg) {
                if (kg < n_kg) {
                    out[(static_cast<size_t>(g) * N + n) * n_kg_compute + kg] =
                        S[(static_cast<size_t>(g) * n_kg + kg) * N + n];
                }
            }
        }
    }
    return out;
}

inline size_t bytes_per_iter_low_latency_grouped_gemm(const ProblemRun& p,
                                       int active_experts,
                                       int M_total,
                                       int K_compute,
                                       int scale_group_size = 128) {
    const size_t bytes_weights =
        static_cast<size_t>(active_experts) * p.N_orig * K_compute / 2;
    const size_t bytes_scales =
        static_cast<size_t>(active_experts) * p.N_orig *
        mga::low_latency_grouped_gemm_scale_groups_per_row(
            K_compute, scale_group_size) *
        sizeof(__nv_bfloat16);
    const size_t bytes_acts =
        static_cast<size_t>(M_total) * K_compute;
    const size_t bytes_outs =
        static_cast<size_t>(M_total) * p.N_orig * sizeof(__nv_bfloat16);
    return bytes_weights + bytes_scales + bytes_acts + bytes_outs;
}

inline double tflops_per_iter(int M_total, int N, int K) {
    return 2.0 * static_cast<double>(M_total) * N * K / 1e12;
}

}  // namespace mga::low_latency_grouped_gemm_test
