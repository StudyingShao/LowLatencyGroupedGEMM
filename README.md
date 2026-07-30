# low_latency_grouped_gemm Standalone

Standalone low-latency grouped GEMM kernels for Hopper GPUs.  The package
contains the existing packed-INT4 weight / FP8 activation path and an additive
MXFP4 weight / per-token FP8 activation path.

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
cmake --build build -j --target \
  test_low_latency_grouped_gemm bench_low_latency_grouped_gemm \
  test_low_latency_mxfp4_fp8 bench_low_latency_mxfp4_fp8
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

Packed INT4 x FP8:

```bash
./build/low_latency_grouped_gemm/test_low_latency_grouped_gemm \
  target192 dsched 64
```

MXFP4 x FP8:

```bash
./build/low_latency_grouped_gemm/test_low_latency_mxfp4_fp8
```

## Benchmark

Packed INT4 x FP8 with a prebuilt device schedule:

```bash
LOW_LATENCY_GROUPED_GEMM_BUILD_ONCE=1 \
  ./build/low_latency_grouped_gemm/bench_low_latency_grouped_gemm \
  1000 50 target192 dsched default 64
```

MXFP4 x FP8, including counter reset, device schedule builder, and main kernel:

```bash
./build/low_latency_grouped_gemm/bench_low_latency_mxfp4_fp8 \
  --warmup 20 --iters 50
```

Run one MXFP4 M value without the correctness pass:

```bash
./build/low_latency_grouped_gemm/bench_low_latency_mxfp4_fp8 \
  --m 48 --warmup 20 --iters 50 --no-check
```

See [`low_latency_grouped_gemm/README.md`](low_latency_grouped_gemm/README.md)
for tensor layouts, schedule contracts, supported shapes, and timing semantics.
