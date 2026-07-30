// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "low_latency_grouped_gemm/include/low_latency_mxfp4_fp8.h"
#include "low_latency_grouped_gemm/include/low_latency_mxfp4_fp8_kernel.cuh"
#include "low_latency_grouped_gemm/include/mxfp4_reference_kernel.h"
#include "low_latency_grouped_gemm/test/low_latency_grouped_gemm_test_utils.h"

namespace {

using namespace mga::low_latency_grouped_gemm_test;

constexpr float kCorrectnessAtolFactor = 1e-2f;
constexpr float kCorrectnessRtol = 5e-2f;

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
    void from_host(const std::vector<T>& values) {
        if (values.size() != count_) std::abort();
        LLGG_CUDA_CHECK(cudaMemcpy(
            ptr_, values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> to_host() const {
        std::vector<T> values(count_);
        LLGG_CUDA_CHECK(cudaMemcpy(
            values.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

uint32_t bits_from_float(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float float_from_bits(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint8_t host_quant_e2m1(double value) {
    const uint32_t bits = bits_from_float(static_cast<float>(value));
    constexpr uint32_t kMask = 0x81c00000U;
    const uint32_t rz_bits = bits & kMask;
    const uint32_t ru_bits = (bits + 0x00200000U) & kMask;
    const double rz = float_from_bits(rz_bits);
    const double ru = float_from_bits(ru_bits);
    const uint32_t rounded =
        std::fabs(value - rz) >= std::fabs(value - ru) ? ru_bits : rz_bits;
    return static_cast<uint8_t>(
        ((rounded & 0x80000000U) >> 28U) |
        ((rounded & 0x01c00000U) >> 22U));
}

std::array<uint8_t, 256 * 16> make_host_lut() {
    std::array<uint8_t, 256 * 16> lut{};
    for (uint32_t delta = 0; delta < 256; ++delta) {
        const double scale =
            float_from_bits(0x3f800000U - (delta << 23U));
        for (uint32_t code = 0; code < 16; ++code) {
            uint8_t normalized = static_cast<uint8_t>(code == 8 ? 0 : code);
            if (delta) {
                const uint32_t value_bits =
                    ((normalized & 8U) << 28U) |
                    ((normalized & 7U) << 22U);
                normalized = host_quant_e2m1(
                    static_cast<double>(float_from_bits(value_bits)) * scale);
            }
            lut[delta * 16 + code] = normalized;
        }
    }
    return lut;
}

struct HostPreprocess {
    std::vector<uint8_t> offsets;
    std::vector<uint8_t> deltas;
    std::vector<float> residual;
    std::vector<uint8_t> processed;
    std::vector<uint8_t> interleaved_weight;
    std::vector<uint8_t> interleaved_offsets;
};

__host__ __device__
uint32_t host_preprocess_fp4x8_signs_for_fp8(uint32_t fp4x8) {
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

HostPreprocess host_preprocess(const std::vector<uint8_t>& raw_weight,
                               const std::vector<uint8_t>& raw_scales,
                               int G,
                               int N,
                               int K) {
    HostPreprocess out;
    const size_t scale_count =
        mga::low_latency_mxfp4_fp8_scale_bytes(G, N, K);
    const size_t weight_count =
        mga::low_latency_mxfp4_fp8_weight_bytes(G, N, K);
    const size_t scales_per_expert = static_cast<size_t>(N) * K / 32;
    const size_t weights_per_expert = static_cast<size_t>(N) * K / 2;
    out.offsets.resize(scale_count);
    out.deltas.resize(scale_count);
    out.residual.resize(G);
    out.processed.resize(weight_count);
    out.interleaved_weight.resize(weight_count);
    out.interleaved_offsets.resize(scale_count);
    const auto lut = make_host_lut();

    for (int g = 0; g < G; ++g) {
        const auto begin =
            raw_scales.begin() + static_cast<size_t>(g) * scales_per_expert;
        const auto end = begin + scales_per_expert;
        const int emin = *std::min_element(begin, end);
        const int emax = *std::max_element(begin, end);
        const int floor_exp = emax - std::min(emax - emin, 11);
        out.residual[g] = std::exp2(static_cast<float>(floor_exp - 128));
        for (size_t i = 0; i < scales_per_expert; ++i) {
            const int value = begin[i];
            const int clamped = std::max(value, floor_exp);
            const size_t idx = static_cast<size_t>(g) * scales_per_expert + i;
            out.deltas[idx] = static_cast<uint8_t>(clamped - value);
            out.offsets[idx] =
                static_cast<uint8_t>((clamped - floor_exp + 1) & 0xf);
        }
        for (size_t i = 0; i < weights_per_expert; ++i) {
            const uint8_t packed =
                raw_weight[static_cast<size_t>(g) * weights_per_expert + i];
            const uint8_t delta =
                out.deltas[static_cast<size_t>(g) * scales_per_expert +
                           i / 16];
            const uint8_t lo = lut[delta * 16 + (packed & 0xf)];
            const uint8_t hi = lut[delta * 16 + (packed >> 4)];
            out.processed[static_cast<size_t>(g) * weights_per_expert + i] =
                static_cast<uint8_t>(lo | (hi << 4));
        }
    }

    for (int g = 0; g < G; ++g) {
        const size_t wb = static_cast<size_t>(g) * N * K / 2;
        const size_t sb = static_cast<size_t>(g) * N * K / 32;
        const int k32_count = K / 32;
        for (int row_tile = 0; row_tile < N / 64; ++row_tile) {
            for (int k32_idx = 0; k32_idx < k32_count; ++k32_idx) {
                for (int tid = 0; tid < 128; ++tid) {
                    const int t0 = tid & 3;
                    const int t1 = (tid >> 2) & 7;
                    const int t2 = tid >> 5;
                    const int row0 = row_tile * 64 + t1 + t2 * 16;
                    const int row1 = row0 + 8;
                    const int k_base = k32_idx * 32 + t0 * 4;
                    uint16_t row0_lo = 0;
                    uint16_t row0_hi = 0;
                    uint16_t row1_lo = 0;
                    uint16_t row1_hi = 0;
                    std::memcpy(
                        &row0_lo,
                        out.processed.data() + wb +
                            static_cast<size_t>(row0) * K / 2 + k_base / 2,
                        2);
                    std::memcpy(
                        &row0_hi,
                        out.processed.data() + wb +
                            static_cast<size_t>(row0) * K / 2 + k_base / 2 + 8,
                        2);
                    std::memcpy(
                        &row1_lo,
                        out.processed.data() + wb +
                            static_cast<size_t>(row1) * K / 2 + k_base / 2,
                        2);
                    std::memcpy(
                        &row1_hi,
                        out.processed.data() + wb +
                            static_cast<size_t>(row1) * K / 2 + k_base / 2 + 8,
                        2);
                    const uint32_t row0_logical =
                        static_cast<uint32_t>(row0_lo) |
                        (static_cast<uint32_t>(row0_hi) << 16U);
                    const uint32_t row1_logical =
                        static_cast<uint32_t>(row1_lo) |
                        (static_cast<uint32_t>(row1_hi) << 16U);
                    const uint64_t physical =
                        static_cast<uint64_t>(
                            host_preprocess_fp4x8_signs_for_fp8(
                                row0_logical)) |
                        (static_cast<uint64_t>(
                             host_preprocess_fp4x8_signs_for_fp8(
                                 row1_logical))
                         << 32U);
                    const int tile64_count = N / 64;
                    const int full_pair_count = tile64_count / 2;
                    size_t dst = 0;
                    if (row_tile < full_pair_count * 2) {
                        const int pair = row_tile / 2;
                        const int part = row_tile & 1;
                        dst = wb +
                              (static_cast<size_t>(pair) * k32_count +
                               k32_idx) * 128 * 16 +
                              static_cast<size_t>(tid) * 16 + part * 8;
                    } else {
                        dst = wb +
                              static_cast<size_t>(full_pair_count) *
                                  k32_count * 128 * 16 +
                              (static_cast<size_t>(k32_idx) * 128 + tid) * 8;
                    }
                    std::memcpy(out.interleaved_weight.data() + dst,
                                &physical,
                                8);
                }
                for (int scale_group = 0; scale_group < 32; ++scale_group) {
                    const int row0_local =
                        (scale_group & 7) + (scale_group >> 3) * 16;
                    const int row1_local = row0_local + 8;
                    const size_t logical0 =
                        sb + static_cast<size_t>(row_tile * 64 + row0_local) *
                                 k32_count +
                        k32_idx;
                    const size_t logical1 =
                        sb + static_cast<size_t>(row_tile * 64 + row1_local) *
                                 k32_count +
                        k32_idx;
                    const int tile64_count = N / 64;
                    const int full_pair_count = tile64_count / 2;
                    size_t interleaved = 0;
                    if (row_tile < full_pair_count * 2) {
                        const int pair = row_tile / 2;
                        const int part = row_tile & 1;
                        interleaved =
                            sb +
                            (static_cast<size_t>(pair) * k32_count +
                             k32_idx) * 128 +
                            static_cast<size_t>(scale_group) * 4 + part * 2;
                    } else {
                        interleaved =
                            sb + static_cast<size_t>(full_pair_count) *
                                     k32_count * 128 +
                            static_cast<size_t>(k32_idx) * 64 +
                            static_cast<size_t>(scale_group) * 2;
                    }
                    out.interleaved_offsets[interleaved] =
                        out.offsets[logical0];
                    out.interleaved_offsets[interleaved + 1] =
                        out.offsets[logical1];
                }
            }
        }
    }
    return out;
}

template <class T>
bool check_exact(const char* name,
                 const std::vector<T>& got,
                 const std::vector<T>& expected) {
    if (got.size() != expected.size()) return false;
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::memcmp(&got[i], &expected[i], sizeof(T)) != 0) {
            std::fprintf(stderr,
                         "%s mismatch at %zu: got=%g expected=%g\n",
                         name,
                         i,
                         static_cast<double>(got[i]),
                         static_cast<double>(expected[i]));
            return false;
        }
    }
    return true;
}

__global__ void converter_cases_kernel(uint8_t* output) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 12 * 16) return;
    const uint32_t offset = idx / 16 + 1;
    const uint32_t code = idx % 16;
    const uint32_t packed = host_preprocess_fp4x8_signs_for_fp8(
        code * 0x11111111U);
    const auto converted =
        mga::low_latency_mxfp4_fp8_detail::
            e2m1x8_to_scaled_e4m3x8(packed, offset);
    uint32_t* out = reinterpret_cast<uint32_t*>(output + idx * 8);
    out[0] = converted.low;
    out[1] = converted.high;
}

bool test_converter() {
    DeviceBuffer<uint8_t> output(12 * 16 * 8);
    converter_cases_kernel<<<1, 256>>>(output.get());
    LLGG_CUDA_CHECK(cudaDeviceSynchronize());
    const auto got = output.to_host();
    for (int offset = 1; offset <= 12; ++offset) {
        for (int code = 0; code < 16; ++code) {
            const int magnitude = code & 7;
            const uint8_t em[8] = {
                0,
                static_cast<uint8_t>(offset * 8),
                static_cast<uint8_t>(offset * 8 + 0x08),
                static_cast<uint8_t>(offset * 8 + 0x0c),
                static_cast<uint8_t>(offset * 8 + 0x10),
                static_cast<uint8_t>(offset * 8 + 0x14),
                static_cast<uint8_t>(offset * 8 + 0x18),
                static_cast<uint8_t>(offset * 8 + 0x1c)};
            const uint8_t expected =
                static_cast<uint8_t>(em[magnitude] |
                                     ((code & 8) ? 0x80 : 0));
            const int case_idx = (offset - 1) * 16 + code;
            for (int lane = 0; lane < 8; ++lane) {
                if (got[case_idx * 8 + lane] != expected) {
                    std::fprintf(stderr,
                                 "converter mismatch offset=%d code=%d lane=%d "
                                 "got=0x%02x expected=0x%02x\n",
                                 offset,
                                 code,
                                 lane,
                                 got[case_idx * 8 + lane],
                                 expected);
                    return false;
                }
            }
        }
    }
    std::printf("[PASS] exhaustive E2M1->scaled-E4M3 converter\n");
    return true;
}

bool test_preprocess() {
    constexpr int G = 3;
    constexpr int N = 64;
    constexpr int K = 64;
    const size_t weight_count =
        mga::low_latency_mxfp4_fp8_weight_bytes(G, N, K);
    const size_t scale_count =
        mga::low_latency_mxfp4_fp8_scale_bytes(G, N, K);
    std::vector<uint8_t> raw_weight(weight_count);
    std::vector<uint8_t> raw_scales(scale_count);
    for (size_t i = 0; i < raw_weight.size(); ++i) {
        raw_weight[i] = static_cast<uint8_t>(i * 73 + 8);
    }
    const size_t spe = static_cast<size_t>(N) * K / 32;
    for (int g = 0; g < G; ++g) {
        for (size_t i = 0; i < spe; ++i) {
            raw_scales[static_cast<size_t>(g) * spe + i] =
                static_cast<uint8_t>(
                    g == 0 ? 120 + i % 12 :
                    g == 1 ? 90 + i % 41 :
                             127 + i % 4);
        }
    }
    const HostPreprocess expected =
        host_preprocess(raw_weight, raw_scales, G, N, K);

    DeviceBuffer<uint8_t> d_raw_weight(weight_count);
    DeviceBuffer<uint8_t> d_raw_scales(scale_count);
    DeviceBuffer<uint8_t> d_delta(scale_count);
    DeviceBuffer<uint8_t> d_processed(weight_count);
    DeviceBuffer<uint8_t> d_interleaved(weight_count);
    DeviceBuffer<uint8_t> d_offsets(scale_count);
    DeviceBuffer<uint8_t> d_offsets_interleaved(scale_count);
    DeviceBuffer<float> d_residual(G);
    d_raw_weight.from_host(raw_weight);
    d_raw_scales.from_host(raw_scales);

    mga::launch_low_latency_mxfp4_fp8_preprocess_weight(
        d_raw_weight.get(),
        d_raw_scales.get(),
        d_delta.get(),
        d_processed.get(),
        d_interleaved.get(),
        d_offsets.get(),
        d_offsets_interleaved.get(),
        d_residual.get(),
        G,
        N,
        K,
        0);
    LLGG_CUDA_CHECK(cudaDeviceSynchronize());

    const bool ok =
        check_exact("delta", d_delta.to_host(), expected.deltas) &&
        check_exact("offset", d_offsets.to_host(), expected.offsets) &&
        check_exact("residual", d_residual.to_host(), expected.residual) &&
        check_exact("processed", d_processed.to_host(), expected.processed) &&
        check_exact("interleaved weight",
                    d_interleaved.to_host(),
                    expected.interleaved_weight) &&
        check_exact("interleaved offset",
                    d_offsets_interleaved.to_host(),
                    expected.interleaved_offsets);
    if (ok) std::printf("[PASS] bit-exact Humming preprocessing/interleave\n");
    return ok;
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

ErrorStats compare_outputs(const std::vector<__nv_bfloat16>& got,
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
            if (stats.failures++ < 8) {
                std::fprintf(stderr,
                             "GEMM mismatch at %zu: got=%g ref=%g "
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

enum class ScheduleMode {
    HostCompact,
    DeviceCompactPrebuilt,
    DeviceCompactBuildEach,
    Rectangular,
};

const char* schedule_mode_name(ScheduleMode mode) {
    switch (mode) {
        case ScheduleMode::HostCompact: return "host-schedule";
        case ScheduleMode::DeviceCompactPrebuilt:
            return "device-schedule-prebuilt";
        case ScheduleMode::DeviceCompactBuildEach:
            return "device-schedule-build";
        case ScheduleMode::Rectangular: return "rectangular";
    }
    return "unknown";
}

bool test_grouped_gemm_case(
    const char* profile,
    int N,
    int K,
    const std::vector<int>& offsets,
    ScheduleMode mode) {
    const int G = static_cast<int>(offsets.size()) - 1;
    const bool device_schedule =
        mode == ScheduleMode::DeviceCompactPrebuilt ||
        mode == ScheduleMode::DeviceCompactBuildEach;
    const bool build_device_schedule =
        mode == ScheduleMode::DeviceCompactBuildEach;
    const bool rectangular = mode == ScheduleMode::Rectangular;
    const int M_total = offsets.back();
    const auto counts = counts_from_offsets(offsets);
    const auto offsets32 = offsets_i32(offsets);
    const auto token_expert_int = token_to_expert(offsets);
    std::vector<int32_t> token_expert(
        token_expert_int.begin(), token_expert_int.end());
    std::vector<int32_t> tile_experts;
    std::vector<int32_t> tile_n;
    make_token_tiles(offsets, tile_experts, tile_n);

    std::mt19937 rng(20260727);
    std::uniform_int_distribution<int> nibble(0, 15);
    std::uniform_int_distribution<int> exponent(118, 132);
    std::uniform_real_distribution<float> activation(-1.5f, 1.5f);
    std::uniform_real_distribution<float> dequant(0.01f, 0.04f);
    const size_t weight_count =
        mga::low_latency_mxfp4_fp8_weight_bytes(G, N, K);
    const size_t scale_count =
        mga::low_latency_mxfp4_fp8_scale_bytes(G, N, K);
    std::vector<uint8_t> raw_weight(weight_count);
    std::vector<uint8_t> raw_scales(scale_count);
    std::vector<__nv_fp8_e4m3> acts(static_cast<size_t>(M_total) * K);
    std::vector<float> activation_scales(M_total);
    for (auto& byte : raw_weight) {
        byte = static_cast<uint8_t>(nibble(rng) | (nibble(rng) << 4));
    }
    for (auto& value : raw_scales) {
        value = static_cast<uint8_t>(exponent(rng));
    }
    for (auto& value : acts) value = __nv_fp8_e4m3(activation(rng));
    for (auto& value : activation_scales) value = dequant(rng);

    DeviceBuffer<uint8_t> d_raw_weight(weight_count);
    DeviceBuffer<uint8_t> d_raw_scales(scale_count);
    DeviceBuffer<uint8_t> d_delta(scale_count);
    DeviceBuffer<uint8_t> d_processed(weight_count);
    DeviceBuffer<uint8_t> d_interleaved(weight_count);
    DeviceBuffer<uint8_t> d_offsets_logical(scale_count);
    DeviceBuffer<uint8_t> d_offsets_interleaved(scale_count);
    DeviceBuffer<float> d_residual(G);
    DeviceBuffer<__nv_fp8_e4m3> d_acts(acts.size());
    DeviceBuffer<float> d_activation_scales(M_total);
    DeviceBuffer<float> d_token_scales(M_total);
    DeviceBuffer<int32_t> d_counts(G);
    DeviceBuffer<int32_t> d_offsets(G + 1);
    DeviceBuffer<int32_t> d_token_expert(M_total);
    DeviceBuffer<int32_t> d_tile_experts(tile_experts.size());
    DeviceBuffer<int32_t> d_tile_n(tile_n.size());
    DeviceBuffer<int32_t> d_num_tiles(device_schedule ? 1 : 0);
    DeviceBuffer<__nv_bfloat16> d_got(static_cast<size_t>(M_total) * N);
    DeviceBuffer<__nv_bfloat16> d_ref(static_cast<size_t>(M_total) * N);

    d_raw_weight.from_host(raw_weight);
    d_raw_scales.from_host(raw_scales);
    d_acts.from_host(acts);
    d_activation_scales.from_host(activation_scales);
    d_counts.from_host(counts);
    d_offsets.from_host(offsets32);
    d_token_expert.from_host(token_expert);
    d_tile_experts.from_host(tile_experts);
    d_tile_n.from_host(tile_n);
    if (device_schedule) {
        const std::vector<int32_t> ntiles = {
            build_device_schedule ? 0 : static_cast<int32_t>(tile_experts.size())};
        d_num_tiles.from_host(ntiles);
    }

    mga::launch_low_latency_mxfp4_fp8_preprocess_weight(
        d_raw_weight.get(),
        d_raw_scales.get(),
        d_delta.get(),
        d_processed.get(),
        d_interleaved.get(),
        d_offsets_logical.get(),
        d_offsets_interleaved.get(),
        d_residual.get(),
        G,
        N,
        K,
        0);
    mga::launch_low_latency_mxfp4_fp8_combine_token_scales(
        d_activation_scales.get(),
        d_residual.get(),
        d_offsets.get(),
        d_token_scales.get(),
        G,
        0);
    mga::launch_reference_low_latency_mxfp4_fp8(
        d_acts.get(),
        d_processed.get(),
        d_offsets_logical.get(),
        d_token_scales.get(),
        d_token_expert.get(),
        d_ref.get(),
        G,
        N,
        K,
        M_total,
        0);

    mga::LowLatencyMxfp4Fp8LaunchOpts opts{};
    opts.G = G;
    opts.N_orig = N;
    opts.K = K;
    opts.max_M_g = rectangular ? max_tokens_per_expert(offsets) : 0;
    opts.acts = d_acts.get();
    opts.w_interleaved = d_interleaved.get();
    opts.exp_offsets_interleaved = d_offsets_interleaved.get();
    opts.token_scales = d_token_scales.get();
    opts.token_counts = d_counts.get();
    opts.expert_offsets = d_offsets.get();
    opts.tile_experts = d_tile_experts.get();
    opts.tile_n = d_tile_n.get();
    opts.outs = d_got.get();
    opts.stream = 0;
    if (rectangular) {
        opts.tile_experts = nullptr;
        opts.tile_n = nullptr;
    } else if (device_schedule) {
        opts.num_token_tiles_device = d_num_tiles.get();
        opts.persistent_ctas = 8;
        opts.build_device_schedule = build_device_schedule;
        opts.tile_schedule_capacity =
            static_cast<int>(tile_experts.size());
    } else {
        opts.num_token_tiles = static_cast<int>(tile_experts.size());
    }
    LLGG_CUDA_CHECK(cudaMemset(d_got.get(),
                               0x7f,
                               static_cast<size_t>(M_total) * N *
                                   sizeof(__nv_bfloat16)));
    mga::launch_low_latency_mxfp4_fp8(opts);
    LLGG_CUDA_CHECK(cudaDeviceSynchronize());

    bool schedule_correct = true;
    if (build_device_schedule) {
        const int built_tiles = d_num_tiles.to_host().front();
        const int expected_tiles =
            static_cast<int>(tile_experts.size());
        schedule_correct = built_tiles == expected_tiles;
        if (!schedule_correct) {
            std::fprintf(stderr,
                         "device builder produced %d tiles; expected %d\n",
                         built_tiles, expected_tiles);
        }
    }

    const ErrorStats errors =
        compare_outputs(d_got.to_host(), d_ref.to_host());
    std::printf(
        "[%s/%s] G=%d N=%d K=%d tokens=%d max_per_expert=%d tiles=%zu "
        "max_abs=%g max_abs_ref=%g mean_abs=%g p95_abs=%g p99_abs=%g "
        "bad=%zu/%zu atol=%g atol_factor=%g rtol=%g\n",
        profile,
        schedule_mode_name(mode),
        G,
        N,
        K,
        M_total,
        max_tokens_per_expert(offsets),
        tile_experts.size(),
        errors.max_abs,
        errors.max_abs_ref,
        errors.mean_abs,
        errors.p95_abs,
        errors.p99_abs,
        errors.failures,
        errors.total,
        errors.atol,
        kCorrectnessAtolFactor,
        kCorrectnessRtol);
    return schedule_correct && errors.failures == 0;
}

}  // namespace

int main() {
    bool ok = true;
    ok = test_converter() && ok;
    ok = test_preprocess() && ok;

    const std::vector<int> smoke_offsets = {0, 3, 3, 8, 9};
    const std::vector<int> skew_offsets = {0, 17, 26, 27, 27};
    ok = test_grouped_gemm_case(
             "smoke", 64, 64, smoke_offsets, ScheduleMode::HostCompact) &&
         ok;
    ok = test_grouped_gemm_case(
             "smoke",
             64,
             64,
             smoke_offsets,
             ScheduleMode::DeviceCompactPrebuilt) &&
         ok;
    ok = test_grouped_gemm_case(
             "smoke",
             64,
             64,
             smoke_offsets,
             ScheduleMode::DeviceCompactBuildEach) &&
         ok;
    ok = test_grouped_gemm_case(
             "smoke", 64, 64, smoke_offsets, ScheduleMode::Rectangular) &&
         ok;

    ok = test_grouped_gemm_case(
             "skew", 64, 64, skew_offsets, ScheduleMode::HostCompact) &&
         ok;
    ok = test_grouped_gemm_case(
             "skew",
             64,
             64,
             skew_offsets,
             ScheduleMode::DeviceCompactPrebuilt) &&
         ok;
    ok = test_grouped_gemm_case(
             "skew",
             64,
             64,
             skew_offsets,
             ScheduleMode::DeviceCompactBuildEach) &&
         ok;
    ok = test_grouped_gemm_case(
             "skew", 64, 64, skew_offsets, ScheduleMode::Rectangular) &&
         ok;
    ok = test_grouped_gemm_case(
             "skew-tail",
             192,
             192,
             skew_offsets,
             ScheduleMode::DeviceCompactBuildEach) &&
         ok;
    ok = test_grouped_gemm_case(
             "skew-fc1",
             2560,
             4096,
             skew_offsets,
             ScheduleMode::DeviceCompactBuildEach) &&
         ok;
    ok = test_grouped_gemm_case(
             "skew-fc2",
             4096,
             1280,
             skew_offsets,
             ScheduleMode::DeviceCompactBuildEach) &&
         ok;

    std::printf("%s low_latency_mxfp4_fp8 tests\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
