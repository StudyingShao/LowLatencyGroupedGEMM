cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 90a)
endif()

project(low_latency_grouped_gemm_standalone LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(LLGG_CUTLASS_ROOT
    "${CMAKE_SOURCE_DIR}/third_party/cutlass"
    CACHE PATH "Path to a CUTLASS checkout")
set(LLGG_CUTLASS_INC "${LLGG_CUTLASS_ROOT}/include"
    CACHE PATH "Path to CUTLASS include directory")

if(NOT EXISTS "${LLGG_CUTLASS_INC}/cutlass/cutlass.h")
    message(FATAL_ERROR
        "CUTLASS headers not found at ${LLGG_CUTLASS_INC}. "
        "This standalone package vendors CUTLASS headers under "
        "third_party/cutlass/include, or configure with "
        "-DLLGG_CUTLASS_ROOT=/path/to/cutlass.")
endif()

add_subdirectory(low_latency_grouped_gemm)
