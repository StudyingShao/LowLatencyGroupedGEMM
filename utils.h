#pragma once

#include <curand_kernel.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Init Data
///
///////////////////////////////////////////////////////////////////////////////////////////////////


template<
    typename ElementA, 
    typename ElementSF, 
    typename ElementB,
    int      SFVecSize,
    bool     DEBUG_INPUT_A,
    bool     DEBUG_INPUT_SFA,
    bool     DEBUG_INPUT_B>
__global__ void init_data_kernel(ElementA* d_A, ElementSF* d_SFA, ElementB* d_B, int B, int M, int N, int K) {
    
    curandState_t rng_state;

    int seq_idx = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
    curand_init(threadIdx.x, seq_idx, 0, &rng_state);

    for (int b = blockIdx.x; b < B; b+=gridDim.x) {
        for (int m = blockIdx.y; m < M; m+=gridDim.y) {
            for (int k = threadIdx.x; k < K; k+=blockDim.x) {
                if (cutlass::sizeof_bits<ElementA>::value == 4) {

                    // Init A
                    uint8_t *ptr_A = reinterpret_cast<uint8_t *>(d_A);
                    if (k % 2 == 0) {
                        if constexpr (DEBUG_INPUT_A) {
                            ptr_A[b * M * K / 2 + m * K / 2 + k / 2] = uint8_t(1 | (1 << 4));
                        }
                        else {
                            ptr_A[b * M * K / 2 + m * K / 2 + k / 2] = uint8_t(curand(&rng_state) % 16 | ((curand(&rng_state) % 16) << 4));
                        }
                    }

                    // Init A scale
                    if (k % SFVecSize == 0) {
                        if constexpr (DEBUG_INPUT_SFA) {
                            d_SFA[(b * M * K + m * K + k) / SFVecSize] = ElementSF(1);
                        }
                        else {
                            d_SFA[(b * M * K + m * K + k) / SFVecSize] = ElementSF(int(curand(&rng_state) % 17) - 8);
                        }
                    }
                }
                else {
                    if constexpr (DEBUG_INPUT_A) {
                        d_A[b * M * K + m * K + k] = ElementA(1);
                    }
                    else {
                        d_A[b * M * K + m * K + k] = ElementA(curand(&rng_state) % 16);
                    }
                }
            }
        }
    }

    // Init B - separate loop for better memory access pattern
    for (int b = blockIdx.x; b < B; b+=gridDim.x) {
        for (int n = blockIdx.z; n < N; n+=gridDim.z) {
            for (int k = threadIdx.x; k < K; k+=blockDim.x) {
                // Init B
                if constexpr (DEBUG_INPUT_B) {
                    d_B[b * N * K + n * K + k] = ElementB(1);
                }
                else {
                    d_B[b * N * K + n * K + k] = ElementB(curand(&rng_state) % 16);
                }
            }
        }
    }
}
    

template<
    typename ElementA, 
    typename ElementSF, 
    typename ElementB,
    int      SFVecSize,
    bool     DEBUG_INPUT_A,
    bool     DEBUG_INPUT_SFA,
    bool     DEBUG_INPUT_B>
void init_data(
    ElementA* h_A,
    ElementSF* h_SFA,
    ElementB* h_B,
    ElementA* d_A,
    ElementSF* d_SFA,
    ElementB* d_B,
    int B, 
    int M, 
    int N, 
    int K) {

    dim3 grid(B, M, N);
    dim3 block(128);

    init_data_kernel<
        ElementA, 
        ElementSF, 
        ElementB, 
        SFVecSize, 
        DEBUG_INPUT_A, 
        DEBUG_INPUT_SFA, 
        DEBUG_INPUT_B><<<grid, block>>>(d_A, d_SFA, d_B, B, M, N, K);

    cudaMemcpy(h_A, d_A, B * M * K * cutlass::sizeof_bits<ElementA>::value / 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, B * K * N * sizeof(ElementB), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_SFA, d_SFA, B * M * K / SFVecSize * sizeof(ElementSF), cudaMemcpyDeviceToHost);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Reference GEMM
///
///////////////////////////////////////////////////////////////////////////////////////////////////


template<
    typename ElementA, 
    typename ElementSF, 
    typename ElementB,
    typename ElementC,
    int      SFVecSize,
    int      kSplitKSlices>
__global__ void ref_gemm_kernel(
    ElementA* d_A,
    ElementSF* d_SFA,
    ElementB* d_B,
    ElementC* d_C_ref,
    int B, 
    int M, 
    int N, 
    int K) {

    float alpha = 1.0f;

    // int4 lookup table
    float lut[] = { 0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0};

    for (int b = blockIdx.x; b < B; b+=gridDim.x) {
        for (int m = blockIdx.y; m < M; m+=gridDim.y) {
            for (int n = threadIdx.x; n < N; n+=blockDim.x) {
                float accu[kSplitKSlices] = {0};
                for (int k = 0; k < K; k++) {

                    int k_slice_id = k / (K / kSplitKSlices);

                    if (cutlass::sizeof_bits<ElementA>::value == 4) {
                        uint8_t *ptr = reinterpret_cast<uint8_t *>(d_A);
                        uint8_t packed_val = ptr[b * M * K / 2 + m * K / 2 + k / 2];

                        int idx;
                        if (k % 2 == 0) {
                            idx = int(packed_val & 0x0F);
                        }
                        else {
                            idx = int((packed_val >> 4) & 0x0F);
                        }
    
                        ElementB val_B = d_B[b * N * K + n * K + k];
                        ElementSF val_SFA = d_SFA[(b * M * K + m * K + k) / SFVecSize];
    
                        accu[k_slice_id] += cutlass::half_t(lut[idx]) * cutlass::half_t(val_B) * float(val_SFA);
                    }
                    else {
                        ElementA val_A = d_A[b * M * K + m * K + k];
                        ElementB val_B = d_B[b * N * K + n * K + k];
    
                        accu[k_slice_id] += cutlass::half_t(val_A) * cutlass::half_t(val_B);
                    }
                }

                ElementC accum = ElementC(0);
                for (int k_slice_id = 0; k_slice_id < kSplitKSlices; k_slice_id++) {
                    accum += ElementC(alpha * accu[k_slice_id]);
                }

                d_C_ref[b * M * N + n * M + m] = ElementC(accum);
            }
        }
    }
}


template<
    typename ElementA, 
    typename ElementSF, 
    typename ElementB,
    typename ElementC,
    int      SFVecSize,
    int      kSplitKSlices>
void ref_gemm(
    ElementC* h_C_ref,
    ElementA* d_A,
    ElementSF* d_SFA,
    ElementB* d_B,
    int B, 
    int M, 
    int N, 
    int K) {

    ElementC *d_C_ref;
    cudaMalloc((void **)&d_C_ref, B * M * N * sizeof(ElementC));

    dim3 grid(B, M);
    dim3 block(128);

    ref_gemm_kernel<
        ElementA, 
        ElementSF, 
        ElementB, 
        ElementC, 
        SFVecSize,
        kSplitKSlices><<<grid, block>>>(d_A, d_SFA, d_B, d_C_ref, B, M, N, K);

    cudaMemcpy(h_C_ref, d_C_ref, B * M * N * sizeof(ElementC), cudaMemcpyDeviceToHost);
}