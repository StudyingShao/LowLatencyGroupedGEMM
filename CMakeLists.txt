# 
# Copyright (c) 2017 - 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.8)

# set(CMAKE_VERBOSE_MAKEFILE ON)

project(CutlassWrapper LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
if (CUDA_VERSION VERSION_GREATER_EQUAL 12.0 AND NOT CUDA_COMPILER MATCHES "[Cc]lang")
  list(APPEND CUTLASS_NVCC_ARCHS_SUPPORTED 100)
endif()
if (CUDA_VERSION VERSION_GREATER_EQUAL 12.3 AND NOT CUDA_COMPILER MATCHES "[Cc]lang")
  list(APPEND CUTLASS_NVCC_ARCHS_SUPPORTED 100a 101)
endif()
if (CUDA_VERSION VERSION_GREATER_EQUAL 12.4 AND NOT CUDA_COMPILER MATCHES "[Cc]lang")
  list(APPEND CUTLASS_NVCC_ARCHS_SUPPORTED 104)
endif()
if (CUDA_VERSION VERSION_GREATER_EQUAL 12.5 AND NOT CUDA_COMPILER MATCHES "[Cc]lang")
  list(APPEND CUTLASS_NVCC_ARCHS_SUPPORTED 101a 104a)
endif()

message (STATUS "NVIDIA CUDA Version: ${CUDA_VERSION}")




find_package(PythonInterp QUIET REQUIRED)

# set (CMAKE_BUILD_TYPE "Debug")
set (CMAKE_BUILD_TYPE "Release")
   
# CUDAToolkit_ROOT
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
message (STATUS "NVIDIA CUDA Path: ${CUDA_PATH}")

set(CUTLASS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/cutlass")
set(FasterTransformer_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/FasterTransformer")

set(CUTLASS_PATH ${CUTLASS_ROOT}/include)
set(CUTLAS_UTILS_PATH ${CUTLASS_ROOT}/tools/util/include)

list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

set(GPU_ARCHS "" CACHE STRING
  "List of GPU architectures (semicolon-separated) to be compiled for.")

if("${GPU_ARCHS}" STREQUAL "")
	# set(GPU_ARCHS "100a")
	set(GPU_ARCHS "90a")
endif()

foreach(arch ${GPU_ARCHS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch},code=sm_${arch}")
	if(SM STREQUAL 70 OR SM STREQUAL 75)
    set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
	endif()
endforeach()

# message(STATUS ${CMAKE_CUDA_FLAGS})

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")	
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall -lineinfo -lcublas")
# -Xcompiler: Specify options directly to the compiler/preprocessor.
# -Wall: Produce warning messages about a number of things that are legal but dubious.

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G")


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_STANDARD STREQUAL "17")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -use_fast_math -res-usage")

set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${CUDA_PATH}/include
)

set(COMMON_LIB_DIRS
  ${CUDA_PATH}/lib64
)

list(APPEND COMMON_HEADER_DIRS  ${CUTLASS_PATH})
list(APPEND COMMON_HEADER_DIRS  ${CUTLAS_UTILS_PATH})
list(APPEND COMMON_HEADER_DIRS  ${FasterTransformer_ROOT})


include_directories(
  ${COMMON_HEADER_DIRS}
)

link_directories(
  ${COMMON_LIB_DIRS}
)

if(NOT DEFINED LIB_INSTALL_PATH)
	set(LIB_INSTALL_PATH ${CMAKE_SOURCE_DIR})
endif()

# set(test
#     fp4_gemv.cu
# )
# add_executable(fp4_Gemv
#   ${test}
# )
# target_link_libraries(fp4_Gemv
#   -lcublas
#   -lcudart
#   ${CMAKE_THREAD_LIBS_INIT}
# )


set(test
    grouped_gemm_unit_test.cu
)
add_executable(grouped_gemm_unit_test
  ${test}
)
target_link_libraries(grouped_gemm_unit_test
  -lcublas
  -lcudart
  ${CMAKE_THREAD_LIBS_INIT}
)

