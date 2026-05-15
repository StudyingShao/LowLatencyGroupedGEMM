# low_latency_grouped_gemm Standalone

Standalone extraction of MaloGEMM's current low-latency grouped GEMM path.

The implementation is under:

```text
low_latency_grouped_gemm/
```

This package is independent from the rest of MaloGEMM. It vendors the CUTLASS
headers needed by the kernel under:

```text
third_party/cutlass/include
```

## Build

```bash
cmake -S . -B build
cmake --build build -j --target test_low_latency_grouped_gemm bench_low_latency_grouped_gemm
```

By default the build targets Hopper `sm_90a`. Override if needed:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=90a
```

If you want to use an external CUTLASS checkout instead of the vendored headers:

```bash
cmake -S . -B build \
  -DLLGG_CUTLASS_ROOT=/path/to/cutlass
```

## Correctness

```bash
CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 \
  ./build/low_latency_grouped_gemm/test_low_latency_grouped_gemm \
  target192 dsched 64
```

## Benchmark

Production-style prebuilt device schedule:

```bash
CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 LOW_LATENCY_GROUPED_GEMM_BUILD_ONCE=1 \
  ./build/low_latency_grouped_gemm/bench_low_latency_grouped_gemm \
  1000 50 target192 dsched default 64
```

Full local sweep with clock sampling:

```bash
CUDA_VISIBLE_DEVICES=3 CLOCK_GPU_INDEX=3 \
  ./low_latency_grouped_gemm/bench_latest_shapes.sh
```

## Current Reference Number

Latest MaloGEMM snapshot on physical GPU 3, `scale_group_size=64`:

```text
profile / shape       FC1 us    FC2 us    total us
target192 / default   65.2266   37.0732   102.2998
```
