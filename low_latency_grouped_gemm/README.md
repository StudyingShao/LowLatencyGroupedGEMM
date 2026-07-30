# Standalone low_latency_grouped_gemm

This directory contains two standalone low-latency grouped GEMM paths for
Hopper GPUs:

- packed INT4 weight x FP8 activation;
- MXFP4 weight x per-token FP8 activation.

## Packed INT4 weight x FP8 activation

The packed INT4 path uses a compact 8-token device schedule.  An upstream
permute or scheduler should produce:

```text
token_counts[g]
expert_offsets[g]
tile_experts[t]
tile_n[t]
num_token_tiles_device
```

Weight scales are per-output-channel along K.  The launch option
`scale_group_size` selects one BF16 scale per 64 or 128 K values.  The bundled
`target192` profile uses `scale_group_size=64`, so FC2 `K=192` has three
complete scale groups.  Inputs must satisfy
`K_compute % scale_group_size == 0`.

For prebuilt schedules, emit token tiles in the cache-friendly `cta_local`
order used by the benchmark harness:

```text
expert_major = [(expert, tile_n) ...], with tile_n ascending inside each expert
streams      = clamp(ceil(persistent_ctas / row_tiles), 1, num_token_tiles)
schedule[s + k * streams] = next(expert_major)
```

This is a generic CTA-local transpose.  The persistent kernel consumes linear
tasks with `task += gridDim.x`, so positions separated by roughly
`persistent_ctas / row_tiles` become one CTA's private sequence.  The transpose
makes that private sequence reuse the same or adjacent experts as often as
possible.

The packed INT4 flow diagram is:

```text
LOW_LATENCY_GROUPED_GEMM_FLOW.html
```

## Build

From the repository root:

```bash
cmake -S . -B build
cmake --build build -j --target test_low_latency_grouped_gemm bench_low_latency_grouped_gemm
```

The standalone package vendors the required CUTLASS headers under
`third_party/cutlass/include`.  To use a different CUTLASS checkout, configure
with `-DLLGG_CUTLASS_ROOT=/path/to/cutlass`.

## Correctness

```bash
./build/low_latency_grouped_gemm/test_low_latency_grouped_gemm target192 dsched 64
```

## MXFP4 weight x FP8 activation

The MXFP4 path is additive and does not change the existing W4A8 API.  The
public interface is `include/low_latency_mxfp4_fp8.h`; the mainloop expands
processed E2M1 weights to E4M3 register fragments and executes native Hopper
E4M3 x E4M3 WGMMA.  Activation dequantization is per routed token and output is
BF16.

This is a CUDA C++ kernel-launch API rather than a framework grouped-GEMM
operator: the caller owns the device tensors, routing metadata, schedule
storage, and preprocessing workspaces.  Operands are staged with explicit
global loads (`ld.global`) and register/shared-memory movement; this path does
not use TMA.

### Input contract

- CUDA architecture: Hopper `sm_90a`.
- Raw weight: packed E2M1 `[G,N,K/2]`, with adjacent K values in one byte.
- Raw weight scale: E8M0 `[G,N,K/32]`.
- Activation: E4M3 `[M_total,K]`, compacted in expert-major routed-token order.
- Routing: `token_counts[G]` and its prefix sum `expert_offsets[G+1]`.
- Output: BF16 `[M_total,N]` in the same compact routed-token order.
- `N` must be a positive multiple of 64; `K` must be a positive multiple of 64.

The runtime expert distribution may be arbitrary, including empty experts and
experts with multiple 8-token tiles.

### Weight preprocessing

Static weights are prepared once with
`launch_low_latency_mxfp4_fp8_preprocess_weight`:

1. split E8M0 scales into a K32 exponent offset and one FP32 residual per
   expert;
2. apply the E2M1 payload rewrite;
3. interleave weight and offset data into the kernel's K32-major row-tile
   layout.

The caller owns three temporary logical workspaces:

```text
delta_workspace      uint8 [G,N,K/32]
processed_workspace  uint8 [G,N,K/2]
exp_offsets_logical  uint8 [G,N,K/32]
```

Use `low_latency_mxfp4_fp8_weight_bytes` and
`low_latency_mxfp4_fp8_scale_bytes` for `w_interleaved` and
`exp_offsets_interleaved`, respectively.  The persistent
`expert_residual` buffer contains one FP32 value per expert.
For each invocation,
`launch_low_latency_mxfp4_fp8_combine_token_scales` builds
`activation_dequant[token] * expert_residual[expert] * 64` in compact token
order.

### Schedule and launch

`LowLatencyMxfp4Fp8LaunchOpts` supports three generic routing modes:

- caller-provided compact schedule with a host-known `num_token_tiles`;
- device compact schedule using `num_token_tiles_device`;
- rectangular fallback using `max_M_g`.

For the normal device-builder path, provide `tile_experts`, `tile_n`, a device
counter, `tile_schedule_capacity`, and `persistent_ctas`, then set
`build_device_schedule=true`.  The launcher enqueues the counter reset, schedule
builder, and main kernel on `opts.stream`.  A capacity of
`max(M_total,1)` entries is a safe upper bound.  The benchmark defaults to four
persistent CTAs per SM;
applications may tune this value for their target GPU and workload.

The FC1 `N=2560,K=4096` and FC2 `N=4096,K=1280` dispatches have compile-time
shape specialization, but they use the same routing contract as all other
supported shapes.

### Build and correctness

```bash
cmake --build build -j --target \
  test_low_latency_mxfp4_fp8 bench_low_latency_mxfp4_fp8
./build/low_latency_grouped_gemm/test_low_latency_mxfp4_fp8
```

The test covers exhaustive E2M1 conversion, bit-exact preprocessing and
interleave, host/device/rectangular schedules, empty experts, skewed routing,
multiple token tiles, N tails, and the FC1/FC2 target shapes.  GEMM output is
checked against an independent FP32 reference with the same policy as the
existing INT4xFP8 test:

```text
atol = 0.01 * max(max_abs_reference, 1e-6)
rtol = 0.05
bad  = (abs_error > atol) && (relative_error > rtol)
allowed bad elements = 0
```

### MXFP4 benchmark

The benchmark matrix is:

- `M={4,8,16,32,44,48,64}`, `E=128`, `top-k=6`;
- FC1 `N=2560,K=4096`;
- FC2 `N=4096,K=1280`.

Run correctness and a short timing pass with:

```bash
./build/low_latency_grouped_gemm/bench_low_latency_mxfp4_fp8 \
  --warmup 2 --iters 5
```

For a performance run on one M value:

```bash
./build/low_latency_grouped_gemm/bench_low_latency_mxfp4_fp8 \
  --m 48 --warmup 20 --iters 50 --no-check
```

One CUDA-event interval encloses the complete iteration loop, and elapsed time
is divided by `iters`.  Every iteration includes FC1 and FC2, each with device
counter reset, schedule builder, and main kernel.  Balanced routing is used
only to make benchmark input deterministic; it is not a kernel requirement.

#### NSYS kernel breakdown

Capture and export one M value with:

```bash
mkdir -p build/nsys

nsys profile \
  --trace=cuda \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output=build/nsys/mxfp4_m48 \
  ./build/low_latency_grouped_gemm/bench_low_latency_mxfp4_fp8 \
    --m 48 --warmup 20 --iters 50 --no-check

nsys export \
  --type sqlite \
  --force-overwrite=true \
  --output=build/nsys/mxfp4_m48.sqlite \
  build/nsys/mxfp4_m48.nsys-rep
```

Use the final 50 calls in the trace.  Each call must have this GPU-operation
order:

```text
FC1 memset, FC1 builder, FC1 main,
FC2 memset, FC2 builder, FC2 main
```

For the four-kernel table, sum the two builders and two mains within each call.
For complete GPU work, also include the two memsets.  Compute each per-call sum
first, then take the median of the 50 sums; do not add independently reduced
per-kernel medians.

## Packed INT4 benchmark

Run the bundled shape sweep with explicit GPU selection:

```bash
CUDA_VISIBLE_DEVICES=0 CLOCK_GPU_INDEX=0 \
  ./low_latency_grouped_gemm/bench_latest_shapes.sh
```

The script builds `bench_low_latency_grouped_gemm`, writes a timestamped log
under `build/low_latency_grouped_gemm/bench_logs/`, records clock and power
samples, and runs:

- `target192`: FC1 `N=384,K=4096`, FC2 `N=4096,K=192`;
- `uniform128_m{1,2,4,8,12,18,22}`: FC1 `N=1024,K=4096`,
  FC2 `N=4096,K=512`.

The script defaults to `SCALE_GROUP_SIZE=64`.  Override it only when every K in
the selected profile is divisible by the requested scale-group size.
`CLOCK_GPU_INDEX` is the physical `nvidia-smi` index and may differ from the
logical CUDA device index.

Prebuilt device schedule:

```bash
CUDA_VISIBLE_DEVICES=0 \
LOW_LATENCY_GROUPED_GEMM_BUILD_ONCE=1 \
./build/low_latency_grouped_gemm/bench_low_latency_grouped_gemm \
  1000 50 target192 dsched default 64
```

Build the device schedule inside each call:

```bash
CUDA_VISIBLE_DEVICES=0 \
./build/low_latency_grouped_gemm/bench_low_latency_grouped_gemm \
  1000 50 target192 dsched default 64
```

Host compact schedule:

```bash
CUDA_VISIBLE_DEVICES=0 \
./build/low_latency_grouped_gemm/bench_low_latency_grouped_gemm \
  1000 50 target192 host default 64
```

For reproducible comparisons, record load-time clocks, power, utilization, and
hardware/thermal slowdown flags together with the latency log.

## Packed INT4 shape notes

Bundled `target192` profile:

| GEMM | `N_orig` | `K` |
|---|---:|---:|
| FC1 | 384 | 4096 |
| FC2 | 4096 | 192 |

The implementation is not hard-bound to these exact shapes:

- full-K path uses register-level weight prefetch when
  `K_compute % 128 == 0 && K_compute >= 512`;
- `K_compute % 128 == 64` uses the tail64 K-const family;
- `K_compute` must be aligned to 64 and must be a multiple of
  `scale_group_size`;
- supported scale group sizes are 64 and 128.
