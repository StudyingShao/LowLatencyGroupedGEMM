#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_root}/build}"
iters="${ITERS:-1000}"
warmup="${WARMUP:-50}"
jobs="${BUILD_JOBS:-$(nproc)}"
scale_group_size="${SCALE_GROUP_SIZE:-64}"
default_bench_gpu="${LOW_LATENCY_GROUPED_GEMM_BENCH_GPU:-3}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${default_bench_gpu}"
fi
bench_bin="${build_dir}/low_latency_grouped_gemm/bench_low_latency_grouped_gemm"
log_dir="${LOG_DIR:-${build_dir}/low_latency_grouped_gemm/bench_logs}"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="${LOG_FILE:-${log_dir}/latest_shapes_${timestamp}.log}"
if [[ -n "${CLOCK_GPU_INDEX:-}" ]]; then
    clock_gpu_index="${CLOCK_GPU_INDEX}"
elif [[ "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+$ ]]; then
    clock_gpu_index="${CUDA_VISIBLE_DEVICES}"
else
    clock_gpu_index="${default_bench_gpu}"
fi
clock_sample_interval="${CLOCK_SAMPLE_INTERVAL:-0.25}"

mkdir -p "${log_dir}"
exec > >(tee "${log_file}") 2>&1

sanitize_label() {
    local value="$1"
    value="${value//[^[:alnum:]_.-]/_}"
    value="${value##_}"
    value="${value%%_}"
    printf '%s' "${value:-case}"
}

summarize_clock_samples() {
    local sample_file="$1"

    awk -F',' '
        NR == 1 { next }
        {
            for (i = 1; i <= NF; ++i) {
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
            }
            clk = $3 + 0
            pwr = $4 + 0
            ++n
            clk_sum += clk
            pwr_sum += pwr
            if (n == 1 || clk < clk_min) clk_min = clk
            if (n == 1 || clk > clk_max) clk_max = clk
            if (n == 1 || pwr > pwr_max) pwr_max = pwr
            if ($6 == "Active") ++sw_power_cap
            if ($7 == "Active" || $8 == "Active" || $9 == "Active") ++hw_or_thermal
        }
        END {
            if (n == 0) {
                print "clock_samples=0"
            } else {
                printf("clock_samples=%d sm_clock_mhz=min/avg/max %.0f/%.1f/%.0f power_w=avg/max %.1f/%.1f sw_power_cap_active=%d hw_or_thermal_slowdown_active=%d\n",
                       n, clk_min, clk_sum / n, clk_max, pwr_sum / n, pwr_max,
                       sw_power_cap, hw_or_thermal)
            }
        }
    ' "${sample_file}"
}

run_with_clock_sampling() {
    local label="$1"
    shift

    local safe_label
    safe_label="$(sanitize_label "${label}")"
    local sample_file="${log_dir}/${timestamp}_${safe_label}_gpu${clock_gpu_index}_clock.csv"
    local bench_tmp="${log_dir}/${timestamp}_${safe_label}_bench.tmp"

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi not found; running benchmark without clock sampling"
        "$@"
        return
    fi

    echo "clock_sample_file=${sample_file}"
    echo "clock_sample_interval_s=${clock_sample_interval}"
    echo "clock_gpu_index=${clock_gpu_index}"
    echo "timestamp,index,clocks.sm_mhz,power.draw_w,utilization.gpu_pct,sw_power_cap,hw_slowdown,hw_thermal_slowdown,hw_power_brake_slowdown,temperature.gpu_c" \
        > "${sample_file}"

    "$@" > "${bench_tmp}" 2>&1 &
    local bench_pid=$!

    while kill -0 "${bench_pid}" 2>/dev/null; do
        printf '%s,' "$(date +%H:%M:%S.%3N)" >> "${sample_file}"
        nvidia-smi -i "${clock_gpu_index}" \
            --query-gpu=index,clocks.sm,power.draw,utilization.gpu,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.hw_power_brake_slowdown,temperature.gpu \
            --format=csv,noheader,nounits >> "${sample_file}" || true
        sleep "${clock_sample_interval}"
    done

    local status=0
    wait "${bench_pid}" || status=$?
    cat "${bench_tmp}"
    rm -f "${bench_tmp}"

    echo
    summarize_clock_samples "${sample_file}"
    echo "clock_sample_file=${sample_file}"

    return "${status}"
}

echo "repo_root=${repo_root}"
echo "build_dir=${build_dir}"
echo "bench_bin=${bench_bin}"
echo "iters=${iters}"
echo "warmup=${warmup}"
echo "scale_group_size=${scale_group_size}"
echo "default_bench_gpu=${default_bench_gpu}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "log_file=${log_file}"
echo "clock_gpu_index=${clock_gpu_index}"
echo "clock_sample_interval=${clock_sample_interval}"
echo

if [[ -d "${repo_root}/.git" && -f "${repo_root}/.gitmodules" ]]; then
    git -C "${repo_root}" submodule update --init --recursive third_party/cutlass
fi

cmake -S "${repo_root}" -B "${build_dir}"
cmake --build "${build_dir}" -j "${jobs}" --target bench_low_latency_grouped_gemm

run_case() {
    local label="$1"
    local profile="$2"
    local shape="$3"

    echo
    echo "============================================================"
    echo "${label}"
    echo "profile=${profile} shape=${shape} scale_group_size=${scale_group_size}"
    echo "============================================================"
    run_with_clock_sampling "${label}" \
        env LOW_LATENCY_GROUPED_GEMM_BUILD_ONCE=1 \
        "${bench_bin}" "${iters}" "${warmup}" "${profile}" dsched "${shape}" \
            "${scale_group_size}"
}

run_case "target192: G=192, FC1 N=384 K=4096, FC2 N=4096 K=192" \
    "target192" \
    "default"

for m_per_expert in 1 2 4 8 12 18 22; do
    run_case "uniform128: G=128, M_per_expert=${m_per_expert}, FC1 N=1024 K=4096, FC2 N=4096 K=512" \
        "uniform128_m${m_per_expert}" \
        "moe128"
done

echo
echo "Done. Full log: ${log_file}"
