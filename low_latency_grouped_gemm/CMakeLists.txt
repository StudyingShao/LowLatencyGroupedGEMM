# Standalone best W4A8 low-latency grouped GEMM implementation.

if(NOT DEFINED LLGG_CUTLASS_INC)
    set(LLGG_CUTLASS_INC "${CMAKE_SOURCE_DIR}/third_party/cutlass/include")
endif()

find_package(CUDAToolkit REQUIRED)

add_library(low_latency_grouped_gemm_lib STATIC
    src/low_latency_grouped_gemm.cu
    src/low_latency_mxfp4_fp8.cu
    src/mxfp4_reference_kernel.cu
    src/reference_kernel.cu
)
set_target_properties(low_latency_grouped_gemm_lib PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(low_latency_grouped_gemm_lib PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${LLGG_CUTLASS_INC}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_options(low_latency_grouped_gemm_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --expt-relaxed-constexpr>
)

add_executable(test_low_latency_grouped_gemm test/test_low_latency_grouped_gemm.cu)
set_target_properties(test_low_latency_grouped_gemm PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)
target_include_directories(test_low_latency_grouped_gemm PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${LLGG_CUTLASS_INC}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_options(test_low_latency_grouped_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
)
target_link_libraries(test_low_latency_grouped_gemm PRIVATE low_latency_grouped_gemm_lib)

add_executable(test_low_latency_mxfp4_fp8 test/test_low_latency_mxfp4_fp8.cu)
set_target_properties(test_low_latency_mxfp4_fp8 PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)
target_include_directories(test_low_latency_mxfp4_fp8 PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${LLGG_CUTLASS_INC}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_options(test_low_latency_mxfp4_fp8 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --expt-relaxed-constexpr>
)
target_link_libraries(test_low_latency_mxfp4_fp8 PRIVATE low_latency_grouped_gemm_lib)

add_executable(bench_low_latency_grouped_gemm test/bench_low_latency_grouped_gemm.cu)
set_target_properties(bench_low_latency_grouped_gemm PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)
target_include_directories(bench_low_latency_grouped_gemm PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${LLGG_CUTLASS_INC}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_options(bench_low_latency_grouped_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
)
target_link_libraries(bench_low_latency_grouped_gemm PRIVATE low_latency_grouped_gemm_lib)

add_executable(bench_low_latency_mxfp4_fp8 test/bench_low_latency_mxfp4_fp8.cu)
set_target_properties(bench_low_latency_mxfp4_fp8 PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)
target_include_directories(bench_low_latency_mxfp4_fp8 PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${LLGG_CUTLASS_INC}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_options(bench_low_latency_mxfp4_fp8 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --expt-relaxed-constexpr>
)
target_link_libraries(bench_low_latency_mxfp4_fp8 PRIVATE low_latency_grouped_gemm_lib)
