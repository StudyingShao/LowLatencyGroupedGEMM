// SPDX-License-Identifier: Apache-2.0
// Standalone correctness test for the best packed-int4/fp8 low_latency_grouped_gemm path.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "low_latency_grouped_gemm/include/low_latency_grouped_gemm.h"
#include "low_latency_grouped_gemm/include/reference_kernel.h"
#include "low_latency_grouped_gemm/test/low_latency_grouped_gemm_test_utils.h"

namespace {

using mga::low_latency_grouped_gemm_test::ProblemRun;

enum class ScheduleMode {
    HostCompact,
    DeviceCompactPrebuilt,
    DeviceCompactBuildEach,
};

const char* schedule_mode_name(ScheduleMode mode) {
    switch (mode) {
        case ScheduleMode::HostCompact: return "host";
        case ScheduleMode::DeviceCompactPrebuilt: return "dsched_prebuilt";
        case ScheduleMode::DeviceCompactBuildEach: return "dsched_build_each";
    }
    return "unknown";
}

struct ErrStats {
    float max_abs = 0.0f;
    float max_abs_ref = 0.0f;
    int n = 0;
    int n_fail = 0;
    int worst_idx = -1;
    float worst_test = 0.0f;
    float worst_ref = 0.0f;
    float max_rel_in_fail = 0.0f;
};

constexpr float kRelTol = 5e-2f;
constexpr float kAtolFactor = 1e-2f;

ErrStats compare_bf16(const __nv_bfloat16* test,
                      const __nv_bfloat16* ref,
                      int n) {
    ErrStats s{};
    s.n = n;
    for (int i = 0; i < n; ++i) {
        s.max_abs_ref = std::max(s.max_abs_ref, std::fabs(static_cast<float>(ref[i])));
    }
    const float atol = kAtolFactor * std::max(s.max_abs_ref, 1e-6f);
    for (int i = 0; i < n; ++i) {
        const float a = static_cast<float>(test[i]);
        const float b = static_cast<float>(ref[i]);
        const float abs_err = std::fabs(a - b);
        const float rel_err = abs_err / (std::fabs(b) + 1e-6f);
        if (abs_err > s.max_abs) {
            s.max_abs = abs_err;
        }
        if (abs_err > atol && rel_err > kRelTol) {
            ++s.n_fail;
            if (rel_err > s.max_rel_in_fail) {
                s.max_rel_in_fail = rel_err;
                s.worst_idx = i;
                s.worst_test = a;
                s.worst_ref = b;
            }
        }
    }
    return s;
}

bool run_problem(const ProblemRun& p,
                 int G,
                 const std::vector<int>& offsets,
                 int M_total,
                 const std::vector<__nv_fp8_e4m3>& h_acts,
	                 const std::vector<uint8_t>& h_W_orig,
	                 const std::vector<__nv_bfloat16>& h_scales,
	                 float act_scale,
	                 ScheduleMode mode,
	                 int scale_group_size) {
    using namespace mga::low_latency_grouped_gemm_test;

    const int K_compute = mga::low_latency_grouped_gemm_aligned_k(p.K);
    const int max_M_g = max_tokens_per_expert(offsets);
    const bool host_scheduled = mode == ScheduleMode::HostCompact;
    const bool device_scheduled = mode != ScheduleMode::HostCompact;
    const bool build_schedule = mode == ScheduleMode::DeviceCompactBuildEach;
    const int row_tiles = (p.N_orig + 63) / 64;
    const int persistent_ctas =
        device_scheduled
            ? default_persistent_ctas(p.N_orig, K_compute, M_total)
            : 0;

    std::printf("\n=== [%s] G=%d N=%d K=%d K_compute=%d M_total=%d mode=%s "
                "scale_group_size=%d ===\n",
                p.name.c_str(),
                G,
                p.N_orig,
                p.K,
                K_compute,
                M_total,
                schedule_mode_name(mode),
                scale_group_size);

    std::vector<int> h_tok_expt = token_to_expert(offsets);
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

    const int n_kg =
        mga::low_latency_grouped_gemm_scale_groups_per_row(p.K,
                                                           scale_group_size);
    const int n_kg_compute =
        mga::low_latency_grouped_gemm_scale_groups_per_row(K_compute,
                                                           scale_group_size);
    const size_t bytes_acts_ref = static_cast<size_t>(M_total) * p.K;
    const size_t bytes_acts_padded = static_cast<size_t>(M_total) * K_compute;
    const size_t bytes_W_orig = static_cast<size_t>(G) * p.N_orig * p.K / 2;
    const size_t bytes_W_padded =
        mga::low_latency_grouped_gemm_interleaved_weight_bytes(G, p.N_orig, K_compute);
    const size_t bytes_scales =
        static_cast<size_t>(G) * n_kg * p.N_orig * sizeof(__nv_bfloat16);
    const size_t bytes_scales_n_major =
        static_cast<size_t>(G) * p.N_orig * n_kg_compute * sizeof(__nv_bfloat16);
    const size_t bytes_scales_padded =
        mga::low_latency_grouped_gemm_padded_scale_bytes(
            G, p.N_orig, K_compute, scale_group_size);
    const size_t bytes_out =
        static_cast<size_t>(M_total) * p.N_orig * sizeof(__nv_bfloat16);

    __nv_fp8_e4m3* d_acts_ref = nullptr;
    __nv_fp8_e4m3* d_acts_padded = nullptr;
    uint8_t* d_W_orig = nullptr;
    uint8_t* d_W_padded = nullptr;
    uint8_t* d_W_interleaved = nullptr;
    __nv_bfloat16* d_scales = nullptr;
    __nv_bfloat16* d_scales_n_major = nullptr;
    __nv_bfloat16* d_scales_padded = nullptr;
    __nv_bfloat16* d_out_test = nullptr;
    __nv_bfloat16* d_out_ref = nullptr;
    int* d_tok_expt = nullptr;
    int32_t* d_counts = nullptr;
    int32_t* d_offsets = nullptr;
    int32_t* d_tile_experts = nullptr;
    int32_t* d_tile_n = nullptr;
    int32_t* d_num_token_tiles = nullptr;

    const int schedule_capacity =
        device_scheduled
            ? mga::low_latency_grouped_gemm_tile_schedule_capacity_from_total_tokens(M_total)
            : h_num_tiles;

    LLGG_CUDA_CHECK(cudaMalloc(&d_acts_ref, bytes_acts_ref));
    LLGG_CUDA_CHECK(cudaMalloc(&d_acts_padded, bytes_acts_padded));
    LLGG_CUDA_CHECK(cudaMalloc(&d_W_orig, bytes_W_orig));
    LLGG_CUDA_CHECK(cudaMalloc(&d_W_padded, bytes_W_padded));
    LLGG_CUDA_CHECK(cudaMalloc(&d_W_interleaved, bytes_W_padded));
    LLGG_CUDA_CHECK(cudaMalloc(&d_scales, bytes_scales));
    LLGG_CUDA_CHECK(cudaMalloc(&d_scales_n_major, bytes_scales_n_major));
    LLGG_CUDA_CHECK(cudaMalloc(&d_scales_padded, bytes_scales_padded));
    LLGG_CUDA_CHECK(cudaMalloc(&d_out_test, bytes_out));
    LLGG_CUDA_CHECK(cudaMalloc(&d_out_ref, bytes_out));
    LLGG_CUDA_CHECK(cudaMalloc(&d_tok_expt, M_total * sizeof(int)));
    LLGG_CUDA_CHECK(cudaMalloc(&d_counts, G * sizeof(int32_t)));
    LLGG_CUDA_CHECK(cudaMalloc(&d_offsets, (G + 1) * sizeof(int32_t)));
    LLGG_CUDA_CHECK(cudaMalloc(&d_tile_experts,
                             schedule_capacity * sizeof(int32_t)));
    LLGG_CUDA_CHECK(cudaMalloc(&d_tile_n,
                             schedule_capacity * sizeof(int32_t)));
    if (device_scheduled) {
        LLGG_CUDA_CHECK(cudaMalloc(&d_num_token_tiles, sizeof(int32_t)));
    }

    LLGG_CUDA_CHECK(cudaMemcpy(d_acts_ref, h_acts.data(), bytes_acts_ref,
                             cudaMemcpyHostToDevice));
    LLGG_CUDA_CHECK(cudaMemcpy(d_acts_padded, h_acts_padded.data(), bytes_acts_padded,
                             cudaMemcpyHostToDevice));
    LLGG_CUDA_CHECK(cudaMemcpy(d_W_orig, h_W_orig.data(), bytes_W_orig,
                             cudaMemcpyHostToDevice));
    LLGG_CUDA_CHECK(cudaMemcpy(d_W_padded, h_W_padded.data(), bytes_W_padded,
                             cudaMemcpyHostToDevice));
    LLGG_CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(), bytes_scales,
                             cudaMemcpyHostToDevice));
    LLGG_CUDA_CHECK(cudaMemcpy(d_scales_n_major, h_scales_n_major.data(),
                             bytes_scales_n_major, cudaMemcpyHostToDevice));
    LLGG_CUDA_CHECK(cudaMemcpy(d_tok_expt, h_tok_expt.data(),
                             M_total * sizeof(int), cudaMemcpyHostToDevice));
    LLGG_CUDA_CHECK(cudaMemcpy(d_counts, h_counts.data(),
                             G * sizeof(int32_t), cudaMemcpyHostToDevice));
    LLGG_CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets_i32.data(),
                             (G + 1) * sizeof(int32_t),
                             cudaMemcpyHostToDevice));
    if (!build_schedule) {
        LLGG_CUDA_CHECK(cudaMemcpy(d_tile_experts, h_tile_experts.data(),
                                 h_tile_experts.size() * sizeof(int32_t),
                                 cudaMemcpyHostToDevice));
        LLGG_CUDA_CHECK(cudaMemcpy(d_tile_n, h_tile_n.data(),
                                 h_tile_n.size() * sizeof(int32_t),
                                 cudaMemcpyHostToDevice));
    }
    if (device_scheduled && !build_schedule) {
        LLGG_CUDA_CHECK(cudaMemcpy(d_num_token_tiles, &h_num_tiles,
                                 sizeof(int32_t), cudaMemcpyHostToDevice));
    }
    LLGG_CUDA_CHECK(cudaMemset(d_out_test, 0, bytes_out));
    LLGG_CUDA_CHECK(cudaMemset(d_out_ref, 0, bytes_out));

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

    mga::launch_reference_grouped_gemm(d_acts_ref,
                                       d_W_orig,
                                       d_scales,
                                       d_tok_expt,
                                       act_scale,
                                       d_out_ref,
                                       G,
                                       p.N_orig,
                                       p.K,
                                       M_total,
                                       0,
                                       scale_group_size);

    mga::LowLatencyGroupedGemmLaunchOpts opts{};
    opts.G = G;
    opts.N_orig = p.N_orig;
    opts.K = K_compute;
    opts.scale_group_size = scale_group_size;
    opts.max_M_g = host_scheduled ? max_M_g : 0;
    opts.act_scale = act_scale;
    opts.acts = d_acts_padded;
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
    opts.build_device_schedule = build_schedule;
    opts.outs = d_out_test;
    opts.stream = 0;
    mga::launch_low_latency_grouped_gemm(opts);
    LLGG_CUDA_CHECK(cudaDeviceSynchronize());

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        std::fprintf(stderr,
                     "kernel launch error: %s\n",
                     cudaGetErrorString(launch_err));
        std::exit(1);
    }

    std::vector<__nv_bfloat16> h_test(static_cast<size_t>(M_total) * p.N_orig);
    std::vector<__nv_bfloat16> h_ref(static_cast<size_t>(M_total) * p.N_orig);
    LLGG_CUDA_CHECK(cudaMemcpy(h_test.data(), d_out_test, bytes_out,
                             cudaMemcpyDeviceToHost));
    LLGG_CUDA_CHECK(cudaMemcpy(h_ref.data(), d_out_ref, bytes_out,
                             cudaMemcpyDeviceToHost));

    ErrStats es = compare_bf16(
        h_test.data(), h_ref.data(), static_cast<int>(h_ref.size()));
    const float atol = kAtolFactor * std::max(es.max_abs_ref, 1e-6f);
    const bool pass = es.n_fail == 0;
    std::printf("[%s] max|ref|=%.4f max_abs_err=%.6f n_fail=%d/%d "
                "atol=%.4g rtol=%.2g %s\n",
                p.name.c_str(),
                es.max_abs_ref,
                es.max_abs,
                es.n_fail,
                es.n,
                atol,
                kRelTol,
                pass ? "PASS" : "FAIL");
    if (!pass) {
        std::printf("  worst mismatch: idx=%d test=%.6f ref=%.6f rel=%.4g\n",
                    es.worst_idx,
                    es.worst_test,
                    es.worst_ref,
                    es.max_rel_in_fail);
    }

    cudaFree(d_acts_ref);
    cudaFree(d_acts_padded);
    cudaFree(d_W_orig);
    cudaFree(d_W_padded);
    cudaFree(d_W_interleaved);
    cudaFree(d_scales);
    cudaFree(d_scales_n_major);
    cudaFree(d_scales_padded);
    cudaFree(d_out_test);
    cudaFree(d_out_ref);
    cudaFree(d_tok_expt);
    cudaFree(d_counts);
    cudaFree(d_offsets);
    cudaFree(d_tile_experts);
    cudaFree(d_tile_n);
    if (d_num_token_tiles) cudaFree(d_num_token_tiles);
    return pass;
}

ScheduleMode parse_mode(int argc, char** argv) {
    if (argc < 3 || std::strcmp(argv[2], "dsched") == 0 ||
        std::strcmp(argv[2], "prebuilt") == 0) {
        return ScheduleMode::DeviceCompactPrebuilt;
    }
    if (std::strcmp(argv[2], "build_each") == 0 ||
        std::strcmp(argv[2], "each") == 0) {
        return ScheduleMode::DeviceCompactBuildEach;
    }
    if (std::strcmp(argv[2], "host") == 0 ||
        std::strcmp(argv[2], "low_latency_grouped_gemm") == 0) {
        return ScheduleMode::HostCompact;
    }
    std::fprintf(stderr,
                 "Unknown mode '%s' (allowed: dsched, build_each, host)\n",
                 argv[2]);
    std::exit(1);
}

int parse_scale_group_size(int argc, char** argv) {
    if (argc < 4) return 64;
    const int value = std::atoi(argv[3]);
    if (value == 64 || value == 128) return value;
    std::fprintf(stderr,
                 "Unknown scale group size '%s' (allowed: 64, 128)\n",
                 argv[3]);
    std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
    LLGG_CUDA_CHECK(cudaFree(0));

    const char* profile = argc >= 2 ? argv[1] : "target192";
    ScheduleMode mode = parse_mode(argc, argv);
    const int scale_group_size = parse_scale_group_size(argc, argv);
    const char* shape_arg = argc >= 5 ? argv[4] : "default";

    std::vector<int> token_counts = mga::test::make_token_counts(profile);
    mga::test::validate_token_profile(profile, token_counts);
    std::vector<int> offsets = mga::test::make_offsets(token_counts);
    const mga::test::TokenProfileStats stats =
        mga::test::compute_token_profile_stats(token_counts);

    std::printf("\n=== low_latency_grouped_gemm correctness: profile=%s mode=%s "
                "scale_group_size=%d shape=%s ===\n",
                profile,
                schedule_mode_name(mode),
                scale_group_size,
                shape_arg);
    std::printf("Token distribution: G=%d active=%d M_total=%d max_M_g=%d\n",
                stats.G,
                stats.active,
                stats.M_total,
                stats.max_mg);

    std::mt19937 rng(0xC0FFEEu);
    bool all_pass = true;
    for (const ProblemRun& p :
         mga::low_latency_grouped_gemm_test::problem_runs_for_shape(shape_arg)) {
        auto W = mga::low_latency_grouped_gemm_test::gen_int4_weights(
            stats.G, p.N_orig, p.K, rng);
        auto S = mga::low_latency_grouped_gemm_test::gen_scales(
            stats.G, p.N_orig, p.K, rng, scale_group_size);
        auto A = mga::low_latency_grouped_gemm_test::gen_acts(stats.M_total, p.K, rng);
        all_pass = run_problem(p,
                               stats.G,
                               offsets,
                               stats.M_total,
                               A,
                               W,
                               S,
                               0.5f,
                               mode,
                               scale_group_size) && all_pass;
    }

    std::printf("\n=== %s ===\n", all_pass ? "OVERALL PASS" : "OVERALL FAIL");
    return all_pass ? 0 : 1;
}
