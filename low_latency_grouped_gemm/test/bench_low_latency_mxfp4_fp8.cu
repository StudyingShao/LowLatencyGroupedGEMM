// SPDX-License-Identifier: Apache-2.0
// MXFP4xFP8 validation/latency matrix:
//   M={4,8,16,32,44,48,64}, E=128, top-k=6
//   FC1 N=2560 K=4096; FC2 N=4096 K=1280.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "low_latency_grouped_gemm/include/low_latency_grouped_gemm.h"
#include "low_latency_grouped_gemm/include/low_latency_mxfp4_fp8.h"
#include "low_latency_grouped_gemm/include/mxfp4_reference_kernel.h"
#include "low_latency_grouped_gemm/test/low_latency_grouped_gemm_test_utils.h"

namespace {

constexpr int kExperts = 128;
constexpr int kTopK = 6;
constexpr int kBenchmarkM[] = {4, 8, 16, 32, 44, 48, 64};
// Match the INT4xFP8 direct-GEMM correctness policy.
constexpr float kCorrectnessAtolFactor = 1e-2f;
constexpr float kCorrectnessRtol = 5e-2f;
constexpr size_t kCorrectnessMaxBad = 0;

template <class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count = 0) : count_(count) {
        if (count_) LLGG_CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }
    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return count_; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

__device__ __forceinline__ uint32_t mix_bits(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

__global__ void init_weight_kernel(uint8_t* weight, size_t count) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const uint32_t bits = mix_bits(static_cast<uint32_t>(i));
        weight[i] = static_cast<uint8_t>((bits & 0xf) |
                                         (((bits >> 8) & 0xf) << 4));
    }
}

__global__ void init_scale_kernel(uint8_t* scales, size_t count) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        // A 13-exponent span exercises expert-level clamping and payload
        // rewrite (Humming keeps only 11 exponent steps).
        scales[i] = static_cast<uint8_t>(120 + (i % 13));
    }
}

__global__ void init_acts_kernel(__nv_fp8_e4m3* acts, size_t count) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const uint32_t bits = mix_bits(static_cast<uint32_t>(i + 17));
        const float value = (static_cast<int>(bits % 25) - 12) * 0.125f;
        acts[i] = __nv_fp8_e4m3(value);
    }
}

__global__ void init_activation_scales_kernel(float* scales, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) scales[i] = 0.01f + (i % 7) * 0.001f;
}

void init_bytes(uint8_t* ptr, size_t count, bool scale) {
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>(std::min<size_t>(
        (count + kThreads - 1) / kThreads, 4096));
    if (scale) {
        init_scale_kernel<<<blocks, kThreads>>>(ptr, count);
    } else {
        init_weight_kernel<<<blocks, kThreads>>>(ptr, count);
    }
}

struct Routing {
    std::vector<int32_t> counts;
    std::vector<int32_t> offsets;
    std::vector<int32_t> token_to_expert;
};

Routing make_routing(int M) {
    Routing routing;
    const int total = M * kTopK;
    routing.counts.assign(kExperts, 0);
    // Deterministic balanced input keeps the active expert count at
    // min(M * top-k, E); the kernel itself has no balance requirement.
    for (int routed_token = 0; routed_token < total; ++routed_token) {
        ++routing.counts[routed_token % kExperts];
    }
    routing.offsets.resize(kExperts + 1, 0);
    for (int g = 0; g < kExperts; ++g) {
        routing.offsets[g + 1] = routing.offsets[g] + routing.counts[g];
    }
    if (routing.offsets.back() != total) std::abort();
    routing.token_to_expert.resize(total);
    for (int g = 0; g < kExperts; ++g) {
        std::fill(routing.token_to_expert.begin() + routing.offsets[g],
                  routing.token_to_expert.begin() + routing.offsets[g + 1],
                  g);
    }
    return routing;
}

struct ErrorStats {
    float max_abs = 0.0f;
    float max_abs_ref = 0.0f;
    float atol = 0.0f;
    float mean_abs = 0.0f;
    float p95_abs = 0.0f;
    float p99_abs = 0.0f;
    size_t failures = 0;
    size_t total = 0;
};

ErrorStats compare(const std::vector<__nv_bfloat16>& got,
                   const std::vector<__nv_bfloat16>& ref) {
    ErrorStats stats;
    stats.total = got.size();
    for (const auto& value : ref) {
        stats.max_abs_ref =
            std::max(stats.max_abs_ref, std::fabs(static_cast<float>(value)));
    }
    stats.atol = kCorrectnessAtolFactor * std::max(stats.max_abs_ref, 1e-6f);
    std::vector<float> abs_errors;
    abs_errors.reserve(got.size());
    double abs_sum = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float a = static_cast<float>(got[i]);
        const float b = static_cast<float>(ref[i]);
        const float abs_error = std::fabs(a - b);
        const float rel_error = abs_error / (std::fabs(b) + 1e-6f);
        abs_errors.push_back(abs_error);
        abs_sum += abs_error;
        stats.max_abs = std::max(stats.max_abs, abs_error);
        if (!std::isfinite(a) || !std::isfinite(b) ||
            (abs_error > stats.atol && rel_error > kCorrectnessRtol)) {
            if (stats.failures++ < 4) {
                std::fprintf(stderr,
                             "sample mismatch idx=%zu got=%g ref=%g "
                             "abs=%g atol=%g rel=%g rtol=%g\n",
                             i,
                             a,
                             b,
                             abs_error,
                             stats.atol,
                             rel_error,
                             kCorrectnessRtol);
            }
        }
    }
    if (!abs_errors.empty()) {
        std::sort(abs_errors.begin(), abs_errors.end());
        stats.mean_abs =
            static_cast<float>(abs_sum / static_cast<double>(abs_errors.size()));
        stats.p95_abs = abs_errors[static_cast<size_t>(
            0.95 * static_cast<double>(abs_errors.size() - 1))];
        stats.p99_abs = abs_errors[static_cast<size_t>(
            0.99 * static_cast<double>(abs_errors.size() - 1))];
    }
    return stats;
}

struct Options {
    int warmup = 20;
    int iterations = 50;
    int ctas_per_sm = 4;
    int persistent_ctas = 0;
    int only_m = 0;
    bool check = true;
};

struct ShapeResult {
    bool correct;
};

template <class Launch>
float measure_cuda_average_us(Launch&& launch,
                              const Options& options,
                              cudaStream_t stream = 0) {
    for (int i = 0; i < options.warmup; ++i) launch();
    LLGG_CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    LLGG_CUDA_CHECK(cudaEventCreate(&start));
    LLGG_CUDA_CHECK(cudaEventCreate(&stop));
    LLGG_CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < options.iterations; ++i) {
        launch();
    }
    LLGG_CUDA_CHECK(cudaEventRecord(stop, stream));
    LLGG_CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    LLGG_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    LLGG_CUDA_CHECK(cudaEventDestroy(start));
    LLGG_CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms * 1000.0f / static_cast<float>(options.iterations);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            options.warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            options.iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--ctas-per-sm") == 0 &&
                   i + 1 < argc) {
            options.ctas_per_sm = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--persistent-ctas") == 0 &&
                   i + 1 < argc) {
            options.persistent_ctas = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
            options.only_m = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--no-check") == 0) {
            options.check = false;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--warmup N] [--iters N] "
                         "[--ctas-per-sm N] [--persistent-ctas N] "
                         "[--m M] [--no-check]\n",
                         argv[0]);
            std::exit(2);
        }
    }
    if (options.warmup < 0 || options.iterations <= 0 ||
        options.ctas_per_sm <= 0 || options.persistent_ctas < 0 ||
        options.only_m < 0) {
        std::exit(2);
    }
    if (options.only_m > 0) {
        bool found = false;
        for (int M : kBenchmarkM) found = found || M == options.only_m;
        if (!found) {
            std::fprintf(stderr, "--m must be one of 4,8,16,32,44,48,64\n");
            std::exit(2);
        }
    }
    return options;
}

class BenchmarkStage {
public:
    BenchmarkStage(const char* name, int N, int K)
        : name_(name),
          N_(N),
          K_(K),
          max_tokens_(64 * kTopK),
          weight_count_(mga::low_latency_mxfp4_fp8_weight_bytes(
              kExperts, N, K)),
          scale_count_(mga::low_latency_mxfp4_fp8_scale_bytes(
              kExperts, N, K)),
          raw_weight_(weight_count_),
          raw_scales_(scale_count_),
          delta_(scale_count_),
          processed_(weight_count_),
          interleaved_(weight_count_),
          logical_offsets_(scale_count_),
          interleaved_offsets_(scale_count_),
          residual_(kExperts),
          acts_(static_cast<size_t>(max_tokens_) * K),
          activation_scales_(max_tokens_),
          token_scales_(max_tokens_),
          counts_(kExperts),
          expert_offsets_(kExperts + 1),
          token_to_expert_(max_tokens_),
          tile_experts_(max_tokens_),
          tile_n_(max_tokens_),
          num_tiles_(1),
          output_(static_cast<size_t>(max_tokens_) * N),
          reference_(static_cast<size_t>(max_tokens_) * N) {
        init_bytes(raw_weight_.get(), weight_count_, false);
        init_bytes(raw_scales_.get(), scale_count_, true);
        mga::launch_low_latency_mxfp4_fp8_preprocess_weight(
            raw_weight_.get(),
            raw_scales_.get(),
            delta_.get(),
            processed_.get(),
            interleaved_.get(),
            logical_offsets_.get(),
            interleaved_offsets_.get(),
            residual_.get(),
            kExperts,
            N_,
            K_,
            0);
        LLGG_CUDA_CHECK(cudaDeviceSynchronize());
    }

    ShapeResult run_shape(int M, const Options& options) {
        const int total = M * kTopK;
        const Routing routing = make_routing(M);
        copy(routing.counts, counts_.get());
        copy(routing.offsets, expert_offsets_.get());
        copy(routing.token_to_expert, token_to_expert_.get());

        constexpr int kThreads = 256;
        const size_t act_count = static_cast<size_t>(total) * K_;
        const int act_blocks = static_cast<int>(std::min<size_t>(
            (act_count + kThreads - 1) / kThreads, 4096));
        init_acts_kernel<<<act_blocks, kThreads>>>(acts_.get(), act_count);
        init_activation_scales_kernel<<<
            (total + kThreads - 1) / kThreads, kThreads>>>(
            activation_scales_.get(), total);
        mga::launch_low_latency_mxfp4_fp8_combine_token_scales(
            activation_scales_.get(),
            residual_.get(),
            expert_offsets_.get(),
            token_scales_.get(),
            kExperts,
            0);

        int sm_count = 0;
        LLGG_CUDA_CHECK(cudaDeviceGetAttribute(
            &sm_count, cudaDevAttrMultiProcessorCount, 0));
        const int requested_ctas =
            options.persistent_ctas > 0
                ? options.persistent_ctas
                : sm_count * options.ctas_per_sm;
        const int persistent_ctas = std::max(1, requested_ctas);
        mga::LowLatencyMxfp4Fp8LaunchOpts launch{};
        launch.G = kExperts;
        launch.N_orig = N_;
        launch.K = K_;
        launch.max_M_g = 0;
        launch.acts = acts_.get();
        launch.w_interleaved = interleaved_.get();
        launch.exp_offsets_interleaved = interleaved_offsets_.get();
        launch.token_scales = token_scales_.get();
        launch.token_counts = counts_.get();
        launch.expert_offsets = expert_offsets_.get();
        launch.tile_experts = tile_experts_.get();
        launch.tile_n = tile_n_.get();
        launch.num_token_tiles_device = num_tiles_.get();
        launch.tile_schedule_capacity = max_tokens_;
        launch.persistent_ctas = persistent_ctas;
        launch.build_device_schedule = false;
        launch.outs = output_.get();
        launch.stream = 0;

        launch_ = launch;

        bool correct = true;
        ErrorStats errors;
        if (options.check) {
            launch_builder_main();
            mga::launch_reference_low_latency_mxfp4_fp8(
                acts_.get(),
                processed_.get(),
                logical_offsets_.get(),
                token_scales_.get(),
                token_to_expert_.get(),
                reference_.get(),
                kExperts,
                N_,
                K_,
                total,
                0);
            LLGG_CUDA_CHECK(cudaDeviceSynchronize());
            const size_t output_values = static_cast<size_t>(total) * N_;
            std::vector<__nv_bfloat16> got(output_values);
            std::vector<__nv_bfloat16> ref(output_values);
            LLGG_CUDA_CHECK(cudaMemcpy(got.data(),
                                       output_.get(),
                                       output_values * sizeof(__nv_bfloat16),
                                       cudaMemcpyDeviceToHost));
            LLGG_CUDA_CHECK(cudaMemcpy(ref.data(),
                                       reference_.get(),
                                       output_values * sizeof(__nv_bfloat16),
                                       cudaMemcpyDeviceToHost));
            errors = compare(got, ref);
            correct = errors.failures <= kCorrectnessMaxBad;
        }

        // Build once outside the timed region to report the actual compact
        // schedule size. The timed full path rebuilds it every iteration.
        launch_builder();
        LLGG_CUDA_CHECK(cudaDeviceSynchronize());
        int32_t num_tiles = 0;
        LLGG_CUDA_CHECK(cudaMemcpy(&num_tiles,
                                   num_tiles_.get(),
                                   sizeof(num_tiles),
                                   cudaMemcpyDeviceToHost));
        std::printf(
            "%s M=%-2d T=%-3d E=%d topk=%d N=%-4d K=%-4d "
            "tiles=%-3d ctas=%-4d schedule=generic check=%s",
            name_,
            M,
            total,
            kExperts,
            kTopK,
            N_,
            K_,
            num_tiles,
            persistent_ctas,
            options.check ? (correct ? "PASS" : "FAIL") : "SKIP");
        if (options.check) {
            std::printf(
                " max_abs=%g max_abs_ref=%g mean_abs=%g p95_abs=%g "
                "p99_abs=%g bad=%zu/%zu max_bad=%zu atol=%g "
                "atol_factor=%g rtol=%g\n",
                errors.max_abs,
                errors.max_abs_ref,
                errors.mean_abs,
                errors.p95_abs,
                errors.p99_abs,
                errors.failures,
                errors.total,
                kCorrectnessMaxBad,
                errors.atol,
                kCorrectnessAtolFactor,
                kCorrectnessRtol);
        } else {
            std::printf("\n");
        }
        return {correct};
    }

    void launch_builder() {
        mga::launch_low_latency_grouped_gemm_build_tile_schedule(
            counts_.get(),
            kExperts,
            max_tokens_,
            tile_experts_.get(),
            tile_n_.get(),
            num_tiles_.get(),
            launch_.stream);
    }

    void launch_builder_main() {
        auto combined_launch = launch_;
        combined_launch.build_device_schedule = true;
        mga::launch_low_latency_mxfp4_fp8(combined_launch);
    }

private:
    static void copy(const std::vector<int32_t>& source, int32_t* destination) {
        if (source.empty()) return;
        LLGG_CUDA_CHECK(cudaMemcpy(destination,
                                   source.data(),
                                   source.size() * sizeof(int32_t),
                                   cudaMemcpyHostToDevice));
    }

    const char* name_;
    int N_;
    int K_;
    int max_tokens_;
    size_t weight_count_;
    size_t scale_count_;
    DeviceBuffer<uint8_t> raw_weight_;
    DeviceBuffer<uint8_t> raw_scales_;
    DeviceBuffer<uint8_t> delta_;
    DeviceBuffer<uint8_t> processed_;
    DeviceBuffer<uint8_t> interleaved_;
    DeviceBuffer<uint8_t> logical_offsets_;
    DeviceBuffer<uint8_t> interleaved_offsets_;
    DeviceBuffer<float> residual_;
    DeviceBuffer<__nv_fp8_e4m3> acts_;
    DeviceBuffer<float> activation_scales_;
    DeviceBuffer<float> token_scales_;
    DeviceBuffer<int32_t> counts_;
    DeviceBuffer<int32_t> expert_offsets_;
    DeviceBuffer<int32_t> token_to_expert_;
    DeviceBuffer<int32_t> tile_experts_;
    DeviceBuffer<int32_t> tile_n_;
    DeviceBuffer<int32_t> num_tiles_;
    DeviceBuffer<__nv_bfloat16> output_;
    DeviceBuffer<__nv_bfloat16> reference_;
    mga::LowLatencyMxfp4Fp8LaunchOpts launch_{};
};

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    LLGG_CUDA_CHECK(cudaFree(0));
    int device = 0;
    cudaDeviceProp properties{};
    LLGG_CUDA_CHECK(cudaGetDevice(&device));
    LLGG_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    std::printf("GPU: %s, MXFP4xFP8 full GPU-path benchmark\n",
                properties.name);
    std::printf("warmup=%d iterations=%d full_check=%s "
                "routing=balanced schedule=generic\n",
                options.warmup,
                options.iterations,
                options.check ? "on" : "off");
    bool all_correct = true;
    std::vector<int> selected_m;
    for (int M : kBenchmarkM) {
        if (options.only_m == 0 || options.only_m == M) selected_m.push_back(M);
    }
    BenchmarkStage fc1("FC1", 2560, 4096);
    BenchmarkStage fc2("FC2", 4096, 1280);
    std::printf(
        "\nPaired FC1+FC2 full-path timing: each loop iteration "
        "contains FC1(reset+builder+main) and FC2(reset+builder+main); "
        "reduction=one batch CUDA event elapsed / iterations\n");
    for (int M : selected_m) {
        const ShapeResult fc1_result = fc1.run_shape(M, options);
        const ShapeResult fc2_result = fc2.run_shape(M, options);
        all_correct = fc1_result.correct && fc2_result.correct && all_correct;

        const float builder_main_us = measure_cuda_average_us(
            [&] {
                fc1.launch_builder_main();
                fc2.launch_builder_main();
            },
            options);
        std::printf("M=%-2d paired_builder_main_us=%.3f\n",
                    M,
                    builder_main_us);
    }
    std::printf("RESULT=%s\n", all_correct ? "PASS" : "FAIL");
    return all_correct ? 0 : 1;
}
