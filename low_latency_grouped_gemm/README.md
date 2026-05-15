# Standalone low_latency_grouped_gemm

This directory contains the current best packed-int4 weight / fp8 activation
GEMM implementation extracted from `grouped_gemm/`.

It is intentionally independent from the experimental WGMMA/TMA implementation:

- no `grouped_gemm_kernel.cu`
- no TMA descriptor code
- no old `TileTask` / `prepare_schedule_kernel`
- no offline fp8-expanded weight path

The production path is `low_latency_grouped_gemm_dsched` with a compact 8-token device
schedule.  Upstream permute should produce:

```text
token_counts[g]
expert_offsets[g]
tile_experts[t]
tile_n[t]
num_token_tiles_device
```

Weight scales are per-output-channel along K.  The launch option
`scale_group_size` selects one bf16 scale per 64 or 128 K values.  The current
production target uses `scale_group_size=64` so FC2 `K=192` has three complete
scale groups.  All production inputs must satisfy `K_compute % scale_group_size
== 0`.

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

The standalone flow diagram is:

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

## Benchmark

Run the current production-path sweep directly:

```bash
./low_latency_grouped_gemm/bench_latest_shapes.sh
```

The script builds `bench_low_latency_grouped_gemm`, writes a timestamped log
under `build/low_latency_grouped_gemm/bench_logs/`, records one clock/power
sample file per benchmark case, and runs:

- `target192`: FC1 `N=384 K=4096`, FC2 `N=4096 K=192`;
- `uniform128_m{1,2,4,8,12,18,22}` with FC1 `N=1024 K=4096`
  and FC2 `N=4096 K=512`.

`bench_latest_shapes.sh` defaults to physical GPU 3 for both CUDA execution and
clock sampling (`CUDA_VISIBLE_DEVICES=3`, `CLOCK_GPU_INDEX=3` when not
overridden) and defaults to `SCALE_GROUP_SIZE=64`.  Override the scale group
only for shapes whose K values are all multiples of the requested scale group
size.

For performance comparisons, keep the generated clock sample CSVs with the
benchmark log.  They record load-time `clocks.sm`, `power.draw`,
`utilization.gpu`, `clocks_event_reasons.sw_power_cap`, and hardware/thermal
slowdown flags.  Use `CLOCK_GPU_INDEX=<physical nvidia-smi index>` when the
benchmark runs on a GPU other than the script's default physical GPU 3.

Prebuilt schedule, matching the production target:

```bash
LOW_LATENCY_GROUPED_GEMM_BUILD_ONCE=1 \
CUDA_VISIBLE_DEVICES=3 \
./build/low_latency_grouped_gemm/bench_low_latency_grouped_gemm 1000 50 target192 dsched default 64
```

If running the benchmark directly instead of through `bench_latest_shapes.sh`,
also record load-time clock and power state; otherwise treat the number as
exploratory only.

Build schedule inside each GEMM call:

```bash
CUDA_VISIBLE_DEVICES=3 \
./build/low_latency_grouped_gemm/bench_low_latency_grouped_gemm 1000 50 target192 dsched default 64
```

This mode is a convenience/debug path.  The production latency target is the
prebuilt schedule path above, where the upstream permute/pre-scheduler emits
the CTA-local order.

Host compact ideal upper bound:

```bash
CUDA_VISIBLE_DEVICES=3 \
./build/low_latency_grouped_gemm/bench_low_latency_grouped_gemm 1000 50 target192 host default 64
```

Latest standalone target192 snapshot with `scale_group_size=64` on physical GPU
3:

```text
path                                      FC1_us     FC2_us   total_us
dsched prebuilt, CTA-local schedule       65.2266    37.0732  102.2998
```

## Shape Notes

Current target:

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
