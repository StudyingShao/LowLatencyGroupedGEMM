// SPDX-License-Identifier: Apache-2.0
// Standalone benchmark for the best packed-int4/fp8 low_latency_grouped_gemm path.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "low_latency_grouped_gemm/include/low_latency_grouped_gemm.h"
#include "low_latency_grouped_gemm/test/low_latency_grouped_gemm_test_utils.h"

namespace {

using mga::low_latency_grouped_gemm_test::ProblemRun;

enum class ScheduleMode {
    HostCompact,
    DeviceCompactPersistent,
};

const char* schedule_mode_name(ScheduleMode mode, bool build_each) {
    if (mode == ScheduleMode::HostCompact) return "host";
    return build_each ? "dsched_build_each" : "dsched_prebuilt";
}

ScheduleMode parse_mode(const char* s) {
    if (!s || std::strcmp(s, "dsched") == 0 ||
        std::strcmp(s, "low_latency_grouped_gemm_dsched") == 0) {
        return ScheduleMode::DeviceCompactPersistent;
    }
    if (std::strcmp(s, "host") == 0 ||
        std::strcmp(s, "low_latency_grouped_gemm") == 0) {
        return ScheduleMode::HostCompact;
    }
    std::fprintf(stderr,
                 "Unknown benchmark mode '%s' (allowed: dsched, host)\n",
                 s);
    std::exit(1);
}

void bench_problem(const ProblemRun& p,
                   int G,
                   int active_experts,
                   const std::vector<int>& offsets,
                   int M_total,
                   const std::vector<__nv_fp8_e4m3>& h_acts,
                   const std::vector<uint8_t>& h_W_orig,
                   const std::vector<__nv_bfloat16>& h_scales,
                   float act_scale,
	                   int warmup,
	                   int iters,
	                   ScheduleMode mode,
	                   int scale_group_size) {
    using namespace mga::low_latency_grouped_gemm_test;

    const int K_compute = mga::low_latency_grouped_gemm_aligned_k(p.K);
    const int max_M_g = max_tokens_per_expert(offsets);
    const bool host_scheduled = mode == ScheduleMode::HostCompact;
    const bool device_scheduled = mode == ScheduleMode::DeviceCompactPersistent;
    const bool build_each =
        device_scheduled && (std::getenv("LOW_LATENCY_GROUPED_GEMM_BUILD_ONCE") == nullptr);
    const int row_tiles = (p.N_orig + 63) / 64;
    const int persistent_ctas =
        device_scheduled
            ? default_persistent_ctas(p.N_orig, K_compute, M_total)
            : 0;

    std::printf("\n=== [%s bench] G=%d N=%d K=%d K_compute=%d M_total=%d "
                "active_g=%d mode=%s scale_group_size=%d ===\n",
                p.name.c_str(),
                G,
                p.N_orig,
                p.K,
                K_compute,
                M_total,
                active_experts,
                schedule_mode_name(mode, build_each),
                scale_group_size);

    std::vector<int32_t> h_counts = counts_from_offsets(offsets);
    std::vector<int32_t> h_offsets_i32 = offsets_i32(offsets);
    std::vector<int32_t> h_tile_experts;
    std::vector<int32_t> h_tile_n;
    make_token_tiles(
        offsets, h_tile_experts, h_tile_n, row_tiles, persistent_ctas);
    const int h_num_tiles = static_cast<int>(h_tile_experts.size());

    std::vector<uint8_t> h_W_padded =
        pad_int4_weights_k(h_W_orig, G, p.N_orig, p.K, K_compute);
    std::vector<__nv_fp8_e4m3> h_acts_padded =
        pad_acts_k(h_acts, M_total, p.K, K_compute);
    std::vector<__nv_bfloat16> h_scales_n_major =
        transpose_scales_for_low_latency_grouped_gemm(
            h_scales, G, p.N_orig, p.K, K_compute, scale_group_size);

    const int n_kg_compute =
        mga::low_latency_grouped_gemm_scale_groups_per_row(K_compute,
                                                           scale_group_size);
    const size_t bytes_acts = static_cast<size_t>(M_total) * K_compute;
    const size_t bytes_W =
        mga::low_latency_grouped_gemm_interleaved_weight_bytes(G, p.N_orig, K_compute);
    const size_t bytes_scales_n_major =
        static_cast<size_t>(G) * p.N_orig * n_kg_compute *
        sizeof(__nv_bfloat16);
    const size_t bytes_scales_padded =
        mga::low_latency_grouped_gemm_padded_scale_bytes(
            G, p.N_orig, K_compute, scale_group_size);
    const size_t bytes_out =
        static_cast<size_t>(M_total) * p.N_orig * sizeof(__nv_bfloat16);

    __nv_fp8_e4m3* d_acts = nullptr;
    uint8_t* d_W_padded = nullptr;
    uint8_t* d_W_interleaved = nullptr;
    __nv_bfloat16* d_scales_n_major = nullptr;
    __nv_bfloat16* d_scales_padded = nullptr;
    __nv_bfloat16* d_out = nullptr;
    int32_t* d_counts = nullptr;
    int32_t* d_offsets = nullptr;
    int32_t* d_tile_experts = nullptr;
    int32_t* d_tile_n = nullptr;
    int32_t* d_num_token_tiles = nullptr;

    const int schedule_capacity =
        device_scheduled
            ? mga::low_latency_grouped_gemm_tile_schedule_capacity_from_total_tokens(M_total)
            : h_num_tiles;

    LLGG_CUDA_CHECK(cudaMalloc(&d_acts, bytes_acts));
    LLGG_CUDA_CHECK(cudaMalloc(&d_W_padded, bytes_W));
    LLGG_CUDA_CHECK(cudaMalloc(&d_W_interleaved, bytes_W));
    LLGG_CUDA_CHECK(cudaMalloc(&d_scales_n_major, bytes_scales_n_major));
    LLGG_CUDA_CHECK(cudaMalloc(&d_scales_padded, bytes_scales_padded));
    LLGG_CUDA_CHECK(cudaMalloc(&d_out, bytes_out));
    LLGG_CUDA_CHECK(cudaMalloc(&d_counts, G * sizeof(int32_t)));
    LLGG_CUDA_CHECK(cudaMalloc(&d_offsets, (G + 1) * sizeof(int32_t)));
    LLGG_CUDA_CHECK(cudaMalloc(&d_tile_experts,
                             schedule_capacity * sizeof(int32_t)));
    LLGG_CUDA_CHECK(cudaMalloc(&d_tile_n,
                             schedule_capacity * sizeof(int32_t)));
    if (device_scheduled) {
        LLGG_CUDA_CHECK(cudaMalloc(&d_num_token_tiles, sizeof(int32_t)));
    }

    LLGG_CUDA_CHECK(cudaMemcpy(d_acts, h_acts_padded.data(), bytes_acts,
                             cudaMemcpyHostToDevice));
    LLGG_CUDA_CHECK(cudaMemcpy(d_W_padded, h_W_padded.data(), bytes_W,
                             cudaMemcpyHostToDevice));
    LLGG_CUDA_CHECK(cudaMemcpy(d_scales_n_major, h_scales_n_major.data(),
                             bytes_scales_n_major, cudaMemcpyHostToDevice));
    LLGG_CUDA_CHECK(cudaMemcpy(d_counts, h_counts.data(),
                             G * sizeof(int32_t), cudaMemcpyHostToDevice));
    LLGG_CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets_i32.data(),
                             (G + 1) * sizeof(int32_t),
                             cudaMemcpyHostToDevice));
    if (!build_each) {
        LLGG_CUDA_CHECK(cudaMemcpy(d_tile_experts, h_tile_experts.data(),
                                 h_tile_experts.size() * sizeof(int32_t),
                                 cudaMemcpyHostToDevice));
        LLGG_CUDA_CHECK(cudaMemcpy(d_tile_n, h_tile_n.data(),
                                 h_tile_n.size() * sizeof(int32_t),
                                 cudaMemcpyHostToDevice));
    }
    if (device_scheduled && !build_each) {
        LLGG_CUDA_CHECK(cudaMemcpy(d_num_token_tiles, &h_num_tiles,
                                 sizeof(int32_t), cudaMemcpyHostToDevice));
    }
    LLGG_CUDA_CHECK(cudaMemset(d_out, 0, bytes_out));

    mga::launch_low_latency_grouped_gemm_interleave(d_W_padded,
                                     d_scales_n_major,
                                     d_W_interleaved,
                                     d_scales_padded,
                                     G,
                                     p.N_orig,
                                     K_compute,
                                     0,
                                     scale_group_size);
    LLGG_CUDA_CHECK(cudaDeviceSynchronize());

    mga::LowLatencyGroupedGemmLaunchOpts opts{};
    opts.G = G;
    opts.N_orig = p.N_orig;
    opts.K = K_compute;
    opts.scale_group_size = scale_group_size;
    opts.max_M_g = host_scheduled ? max_M_g : 0;
    opts.act_scale = act_scale;
    opts.acts = d_acts;
    opts.w_interleaved = d_W_interleaved;
    opts.scales_padded = d_scales_padded;
    opts.token_counts = d_counts;
    opts.expert_offsets = d_offsets;
    opts.tile_experts = d_tile_experts;
    opts.tile_n = d_tile_n;
    opts.num_token_tiles = host_scheduled ? h_num_tiles : 0;
    opts.num_token_tiles_device = device_scheduled ? d_num_token_tiles : nullptr;
    opts.tile_schedule_capacity = device_scheduled ? schedule_capacity : 0;
    opts.persistent_ctas = persistent_ctas;
    opts.build_device_schedule = build_each;
    opts.outs = d_out;
    opts.stream = 0;

    for (int i = 0; i < warmup; ++i) {
        mga::launch_low_latency_grouped_gemm(opts);
    }
    LLGG_CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    LLGG_CUDA_CHECK(cudaEventCreate(&start));
    LLGG_CUDA_CHECK(cudaEventCreate(&stop));
    LLGG_CUDA_CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < iters; ++i) {
        mga::launch_low_latency_grouped_gemm(opts);
    }
    LLGG_CUDA_CHECK(cudaEventRecord(stop, 0));
    LLGG_CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    LLGG_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    LLGG_CUDA_CHECK(cudaDeviceSynchronize());

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        std::fprintf(stderr,
                     "kernel launch error: %s\n",
                     cudaGetErrorString(launch_err));
        std::exit(1);
    }

    const double us = static_cast<double>(ms) * 1000.0 / std::max(iters, 1);
    const size_t bytes_iter =
        bytes_per_iter_low_latency_grouped_gemm(
            p, active_experts, M_total, K_compute, scale_group_size);
    const double gb_per_s = static_cast<double>(bytes_iter) / us / 1e3;
    const double tflops = tflops_per_iter(M_total, p.N_orig, K_compute) /
                          (us * 1e-6);

    std::printf("[%s] mode=%s iters=%d duration=%.4f us bytes/iter=%.2f MB\n"
                "      BW(actual): %.1f GB/s\n"
                "      Tensor(fp16 HMMA, compute K): %.2f TFLOPS\n",
                p.name.c_str(),
                schedule_mode_name(mode, build_each),
                iters,
                us,
                static_cast<double>(bytes_iter) / 1e6,
                gb_per_s,
                tflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_acts);
    cudaFree(d_W_padded);
    cudaFree(d_W_interleaved);
    cudaFree(d_scales_n_major);
    cudaFree(d_scales_padded);
    cudaFree(d_out);
    cudaFree(d_counts);
    cudaFree(d_offsets);
    cudaFree(d_tile_experts);
    cudaFree(d_tile_n);
    if (d_num_token_tiles) cudaFree(d_num_token_tiles);
}

}  // namespace

int main(int argc, char** argv) {
    // Args: iters [warmup] [profile] [mode] [shape] [scale_group_size]
    const int iters = argc >= 2 ? std::atoi(argv[1]) : 500;
    const int warmup = argc >= 3 ? std::atoi(argv[2]) : 20;
    const char* profile = argc >= 4 ? argv[3] : "target192";
    const char* mode_arg = argc >= 5 ? argv[4] : "dsched";
    const char* shape_arg = argc >= 6 ? argv[5] : "default";
    const int scale_group_size = argc >= 7 ? std::atoi(argv[6]) : 64;
    if (scale_group_size != 64 && scale_group_size != 128) {
        std::fprintf(stderr,
                     "Unknown scale group size '%s' (allowed: 64, 128)\n",
                     argc >= 7 ? argv[6] : "");
        return 1;
    }
    const ScheduleMode mode = parse_mode(mode_arg);

    LLGG_CUDA_CHECK(cudaFree(0));

    std::vector<int> token_counts = mga::test::make_token_counts(profile);
    mga::test::validate_token_profile(profile, token_counts);
    std::vector<int> offsets = mga::test::make_offsets(token_counts);
    const mga::test::TokenProfileStats stats =
        mga::test::compute_token_profile_stats(token_counts);

    std::printf("Token distribution: profile=%s G=%d active=%d M_total=%d "
                "max_M_g=%d\n",
                profile,
                stats.G,
                stats.active,
                stats.M_total,
                stats.max_mg);
    std::printf("Bench: warmup=%d iters=%d mode=%s shape=%s "
                "scale_group_size=%d\n",
                warmup,
                iters,
                schedule_mode_name(
                    mode,
                    mode == ScheduleMode::DeviceCompactPersistent &&
                        std::getenv("LOW_LATENCY_GROUPED_GEMM_BUILD_ONCE") == nullptr),
                shape_arg,
                scale_group_size);

    std::mt19937 rng(0xC0FFEEu);
    const std::vector<ProblemRun> problems =
        mga::low_latency_grouped_gemm_test::problem_runs_for_shape(shape_arg);
    for (const ProblemRun& p : problems) {
        auto W = mga::low_latency_grouped_gemm_test::gen_int4_weights(
            stats.G, p.N_orig, p.K, rng);
        auto S = mga::low_latency_grouped_gemm_test::gen_scales(
            stats.G, p.N_orig, p.K, rng, scale_group_size);
        auto A = mga::low_latency_grouped_gemm_test::gen_acts(stats.M_total, p.K, rng);
        bench_problem(p,
                      stats.G,
                      stats.active,
                      offsets,
                      stats.M_total,
                      A,
                      W,
                      S,
	                      0.5f,
	                      warmup,
	                      iters,
	                      mode,
	                      scale_group_size);
    }

    return 0;
}
