#include <iostream>
#include "cutlass/numeric_types.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/array.h"

#include "cutlass/device_kernel.h"

#include "cutlass/gemm/device/gemv.h"
#include "cutlass/gemm/kernel/gemv.h"

const bool PRINT_INPUT = false;
const bool DEBUG_INTERLEAVE = false;
const bool PRINT_OUTPUT = false;
const bool DEBUG_INPUT_A = false;
const bool DEBUG_INPUT_B = false;

void test_fp4_gemv(int B, int M, int N, int K, int warm_up_runs, int runs)
{
    using ElementA = cutlass::int4b_t;
    // using ElementA = cutlass::float_e2m1_t;
    // using ElementA = cutlass::float_e4m3_t;
    
    using ElementB = cutlass::float_e4m3_t;
    
    using ElementC = __nv_bfloat16;
    // using ElementC = float;
    // using ElementSF = cutlass::float_e4m3_t;
    using ElementSF = cutlass::half_t;
    // using ElementSF = unsigned int;
    
    const int kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementB>::value;
    // const int kElementsPerAccess = 8;
    const int kThreadCount = 128;
    const int kThreadsPerRow = 8;
    const int kSplitKSlices = 1;
    
    const int SFVecSize = 128;

    // host buffer
    ElementA* h_A;
    ElementB* h_B;
    ElementC *h_C;
    ElementC *h_C_ref;
    int32_t  *h_N;
    // cutlass::array<int32_t, B> h_N;

    h_A     = (ElementA*) malloc(B * M * K * cutlass::sizeof_bits<ElementA>::value / 8);
    h_B     = (ElementB*) malloc(B * N * K * sizeof(ElementB));
    h_C     = (ElementC*) malloc(B * M * N * sizeof(ElementC));
    h_C_ref = (ElementC*) malloc(B * M * N * sizeof(ElementC));
    h_N          = (int32_t*) malloc(B * sizeof(int32_t));

    // device buffer
    ElementA *d_A;
    ElementA *d_A_interleaved;
    ElementB *d_B;
    ElementC *d_C;
    int32_t *d_N;

    cudaMalloc((void **)&d_A, B * M * K * cutlass::sizeof_bits<ElementA>::value / 8);
    cudaMalloc((void **)&d_A_interleaved, B * M * K * cutlass::sizeof_bits<ElementA>::value / 8);
    cudaMalloc((void **)&d_B, B * N * K * sizeof(ElementB));
    cudaMalloc((void **)&d_C, B * M * N * sizeof(ElementC));
    cudaMalloc((void **)&d_N, B * sizeof(int32_t));

    for(int i = 0; i < B; i++)
    {
        h_N[i] = N;
    }
    cudaMemcpy(d_N, h_N, B * sizeof(int32_t), cudaMemcpyHostToDevice);


    // scale factor
    ElementSF *h_fp8_SF_A;
    ElementSF *d_fp8_SF_A;

    h_fp8_SF_A = (ElementSF*) malloc(B * M * K / SFVecSize * sizeof(ElementSF));
    cudaMalloc((void **)&d_fp8_SF_A, B * M * K / SFVecSize * sizeof(ElementSF));

    srand(0);

    for (int b = 0; b < B; ++b) {
        for (int k = 0; k < K; ++k) {

            // Init A
            for (int m = 0; m < M; ++m) {
                if (cutlass::sizeof_bits<ElementA>::value == 4) {
                    uint8_t *ptr = reinterpret_cast<uint8_t *>(h_A);

                    if (k % 2 == 0) {
                        if constexpr (DEBUG_INPUT_A) {
                            ptr[b * M * K / 2 + m * K / 2 + k / 2] = uint8_t(1 | (1 << 4));
                        }
                        else {
                            ptr[b * M * K / 2 + m * K / 2 + k / 2] = uint8_t(rand() % 16 | ((rand() % 16) << 4));
                        }
                    }
                
                    // Init A scale
                    if (k % SFVecSize == 0) {
                        h_fp8_SF_A[(b * M * K + m * K + k) / SFVecSize] = ElementSF((rand() % 9) - 16);
                    }
                
                }
                else {
                    if constexpr (DEBUG_INPUT_A) {
                        h_A[b * M * K + m * K + k] = ElementA(1);
                    }
                    else {
                        h_A[b * M * K + m * K + k] = ElementA(rand() % 16);
                    }
                }
            }            
            
            // Init B
            for (int n = 0; n < N; ++n) {
                if constexpr (DEBUG_INPUT_B) {
                    h_B[b * N * K + n * K + k] = ElementB(1);
                }
                else{
                    h_B[b * N * K + n * K + k] = ElementB(rand() % 16);
                }
            }
        }
    }

    if(PRINT_INPUT)
    {
        using Fragment_temp = cutlass::Array<ElementA, 2>;
        // print input matrix A
        printf("---------------- input matrix A ----------------\n");
        for (int m = 0; m < 20; ++m) {
            for (int k = 0; k < 64; k+=2) {
                Fragment_temp temp_A = *reinterpret_cast<Fragment_temp *>(&h_A[m * K + k]);

                printf("%d:%.0f  \n", k, float(temp_A[0]));
                printf("%d:%.0f  \n", k + 1, float(temp_A[1]));
            }
            printf("\n");
        }
        // print input vector B
        printf("---------------- input vector B ----------------\n");
        for (int k = 0; k < 64; k+=2) {
            Fragment_temp temp_B = *reinterpret_cast<Fragment_temp *>(&h_B[k / 2]);
            printf("%d:%.0f  ", k, float(temp_B[0]));
            printf("%d:%.0f  ", k, float(temp_B[1]));
        }
        printf("\n");
    }

    // copy input tensor from host to device
    cudaMemcpy(d_A, h_A, B * M * K * cutlass::sizeof_bits<ElementA>::value / 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, B * K * N * sizeof(ElementB), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fp8_SF_A, h_fp8_SF_A, B * M * K / SFVecSize * sizeof(ElementSF), cudaMemcpyHostToDevice);

    using LayoutA = cutlass::layout::RowMajor;
    using ElementAccumulator = float;

    // LinearCombination: Applies a linear combination operator to an array of elements.
    // D = alpha * accumulator + beta * source + uniform
    using Epilogue = cutlass::epilogue::thread::LinearCombination<
        ElementC,           // Data type used to load and store tensors
        4,                  // Number of elements computed per operation.
        ElementAccumulator, // Accumulator data type
        ElementAccumulator>;   // Data type used to compute linear combination

    using cutlass_Gemv_kernel = cutlass::gemm::kernel::Gemv<
                                    ElementA, LayoutA, ElementB, ElementC, 
                                    ElementAccumulator, Epilogue, 
                                    kElementsPerAccess, kThreadCount, kThreadsPerRow, kSplitKSlices,
                                    ElementSF, SFVecSize>;
    using cutlass_Gemv_device = cutlass::gemm::device::Gemv<cutlass_Gemv_kernel>;

    
    //////////////////////////////////////////////////////////////////////////////////////////////
    // query occupancy after setting smem size
    int max_active_blocks = -1;
    cudaError_t result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks,
        cutlass::device_kernel<cutlass_Gemv_kernel>,
        cutlass_Gemv_kernel::kThreadCount,
        0);

    if (cudaSuccess != result) {
        result = cudaGetLastError();
        std::cout << "  cudaOccupancyMaxActiveBlocksPerMultiprocessor() returned error: "
        << cudaGetErrorString(result) << std::endl;
    }

    int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    
    float CTA_count = B * (M / 64) * kSplitKSlices;
    float wave_capacity = max_active_blocks * sm_count;
    float wave_num = CTA_count / wave_capacity;

    printf("maximum_active_blocks %d sm_count %d wave_num %f\n", max_active_blocks, sm_count, wave_num);
    //////////////////////////////////////////////////////////////////////////////////////////////

    cutlass_Gemv_kernel::matrix_A_interleave<ElementA, kElementsPerAccess>(d_A_interleaved, d_A, B, M, K);


    if (DEBUG_INTERLEAVE) {
        // print input matrix A interleaved
        ElementA *h_A_interleaved;
        h_A_interleaved = (ElementA*) malloc(B * M * K * cutlass::sizeof_bits<ElementA>::value / 8);
        cudaMemcpy(h_A_interleaved, d_A_interleaved, B * M * K * cutlass::sizeof_bits<ElementA>::value / 8, cudaMemcpyDeviceToHost);
        
        printf("---------------- input matrix A----------------\n");
        for (int b = 0; b < B; b++) {
            printf("---- batch %d ----\n", b);
            for (int m = 0; m < M; m++) {
                printf("m %d:   ", m);
                for (int k = 0; k < K; k++) {
                    printf("%d:%f  ", k, float(h_A[b * M * K + m * K + k]));
                }
                printf("\n");
            }
        }

        printf("---------------- input matrix A interleaved----------------\n");
        for (int b = 0; b < B; b++) {
            printf("---- batch %d ----\n", b);
            for (int m = 0; m < M / 2; m++) {
                printf("m %d:   ", m);
                for (int k = 0; k < K * 2; k++) {
                    printf("%d:%f  ", k, float(h_A_interleaved[b * M * K + m * K * 2 + k]));
                }
                printf("\n");
            }
        }
    }
    
    

    cutlass_Gemv_device gemv;

    ElementAccumulator alpha = ElementAccumulator(1);
    ElementAccumulator beta = ElementAccumulator(0);


    cutlass_Gemv_device::Arguments args{
        M,              // M
        d_N,            // N
        K,              // K
        N,              // max_N
        B,              // batch count
        {alpha, beta},
        {(ElementA *)d_A_interleaved, K},
        (ElementB *)d_B,
        d_C,
        d_C,
        M * K,          // batch_stride_A, M x K
        N * K,          // batch_stride_B, N x K
        M * N,          // batch_stride_C, M x N
        M * N,          // batch_stride_D, M x N
        d_fp8_SF_A,
    };

    auto can_implement = gemv.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess) {
        std::string err_msg =
            "cutlass gemv kernel will fail for params. Error: " + std::string(cutlassGetStatusString(can_implement));
        throw std::runtime_error("[CUTLASS Error][jiangs] " + err_msg);
    }

    auto init_status = gemv.initialize(args);
    if (init_status != cutlass::Status::kSuccess) {
        std::string err_msg = "Failed to initialize cutlass gemv. Error: "
                              + std::string(cutlassGetStatusString(init_status));
        throw std::runtime_error("[CUTLASS Error][jiangs] " + err_msg);
    }

    cudaEvent_t _event_start_;
    cudaEvent_t _event_end_;
    float _event_time_;
    cudaEventCreate(&_event_start_);
    cudaEventCreate(&_event_end_);
    
    cutlass::Status run_status = cutlass::Status::kInvalid;
    
    for (int i = 0; i < warm_up_runs; i++)
    {
        run_status = gemv.run(nullptr);
    }
    
    cudaEventRecord(_event_start_);
    for (int i = 0; i < runs; i++)
    {
        run_status = gemv.run(nullptr);
    }
    cudaEventRecord(_event_end_);

    cudaEventSynchronize(_event_end_);
    cudaEventElapsedTime(&_event_time_, _event_start_, _event_end_);
    float _event_time_once_ = _event_time_ / runs;
    printf("%10.3fus\n", _event_time_once_ * 1000);
    
    float bytes_of_A = float(B) * float(M) * float(K) * cutlass::sizeof_bits<ElementA>::value / 8;
    float bytes_of_B = float(B) * float(N) * float(K) * cutlass::sizeof_bits<ElementB>::value / 8;

    float GB_total = (bytes_of_A + bytes_of_B) / 1024 / 1024 / 1024;
    
    float LDG_throughput = GB_total / _event_time_once_ * 1E3;
    printf("GEMM Shape B %d M %d N %d K %d, MEM size = %f GB, LDG_throughput %f GB/s MEM SOL %.2f %%\t", B, M, N, K, GB_total, LDG_throughput, LDG_throughput / 4000 * 100);
    cudaDeviceSynchronize();
    printf("string: %s\n", cudaGetErrorString(cudaGetLastError()));

    if (run_status != cutlass::Status::kSuccess) {
        std::string err_msg =
            "Failed to run cutlass gemv. Error: " + std::string(cutlassGetStatusString(run_status));
        throw std::runtime_error("[CUTLASS Error][jiangs] " + err_msg);
    }

    cudaMemcpy(h_C, d_C, B * M * N * sizeof(ElementC), cudaMemcpyDeviceToHost);
    
    // result validation

    float lut[16];

    if constexpr (cute::is_same_v<ElementA, cutlass::float_e2m1_t>) {
        // fp4 e2m1
        float tmp[] = {0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0, 
                    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};
        memcpy(lut, tmp, sizeof(lut));
    }
    else if constexpr (cute::is_same_v<ElementA, cutlass::int4b_t>) {
        // int4
        float tmp[] = { 0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,
                    -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0};
        memcpy(lut, tmp, sizeof(lut));
    }

    for (int b = 0; b < B; b++) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float accu[kSplitKSlices] = {0};
                for (int k = 0; k < K; k++) {

                    int k_slice_id = k / (K / kSplitKSlices);

                    if (cutlass::sizeof_bits<ElementA>::value == 4) {
                        uint8_t *ptr = reinterpret_cast<uint8_t *>(h_A);
                        uint8_t packed_val = ptr[b * M * K / 2 + m * K / 2 + k / 2];

                        int idx;
                        if (k % 2 == 0) {
                            idx = int(packed_val & 0x0F);
                        }
                        else {
                            idx = int((packed_val >> 4) & 0x0F);
                        }
    
                        ElementB val_B = h_B[b * N * K + n * K + k];

                        ElementSF val_SFA = h_fp8_SF_A[(b * M * K + m * K + k) / SFVecSize];
    
                        accu[k_slice_id] += cutlass::half_t(lut[idx]) * cutlass::half_t(val_B) * float(val_SFA);
                    }
                    else {
                        ElementA val_A = h_A[b * M * K + m * K + k];
                        ElementB val_B = h_B[b * N * K + n * K + k];
    
                        accu[k_slice_id] += cutlass::half_t(val_A) * cutlass::half_t(val_B);
                    }
                }

                ElementC accum = ElementC(0);
                for (int k_slice_id = 0; k_slice_id < kSplitKSlices; k_slice_id++) {
                    accum += ElementC(alpha * accu[k_slice_id]);
                }

                h_C_ref[b * M * N + n * M + m] = ElementC(accum);
            }
        }
    }

    if (PRINT_OUTPUT) {
        const int PRINT_NUM = 50;
        // const int PRINT_NUM = 9999999999999;

        printf("---------------- output vector C ----------------\n");
        printf("out  ref\n");
        for (int b = 0; b < B; ++b) {
            for (int n = 0; n < N; ++n) {
                for (int m = 0; m < min(PRINT_NUM, M); ++m) {
                    float temp1 = float(h_C[b * M * N + n * M + m]);
                    float temp2 = float(h_C_ref[b * M * N + n * M + m]);
                    printf("index b %d n %d m %d   %f  out: %f, ref: %f\n", b, n, m, temp1 - temp2, temp1, temp2);
                }
            }
        }
    }

    float max_abs_error = 0.0f;
    for (int b = 0; b < B; ++b) {
        for (int n = 0; n < N; ++n) {
            for (int m = 0; m < M; ++m) {
                float temp1 = float(h_C[b * M * N + n * M + m]);
                float temp2 = float(h_C_ref[b * M * N + n * M + m]);
    
                max_abs_error = max(max_abs_error, abs(temp1 - temp2));
            }
        }
    }
    printf("max_abs_error = %f\n", max_abs_error);
}


int main(int argc, char *argv[])
{    
    int B = std::stoi(argv[1]);
    int M = std::stoi(argv[2]);
    int N = std::stoi(argv[3]);
    int K = std::stoi(argv[4]);
    int warm_up_runs = std::stoi(argv[5]);
    int runs = std::stoi(argv[6]);

    printf("GEMV B %d M %d N %d K %d warm_up_runs %d runs %d\n", B, M, N, K, warm_up_runs, runs);

    test_fp4_gemv(B, M, N, K, warm_up_runs, runs);
    
    cudaDeviceSynchronize();
    fflush(stdout);

    return 0;
}