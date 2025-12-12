#include <iostream>
#include "cutlass/numeric_types.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/array.h"

#include "cutlass/gemm/device/gemv.h"
#include "cutlass/gemm/kernel/gemv.h"

bool PRINT_INPUT = false;
bool DEBUG_INTERLEAVE = false;
bool PRINT_OUTPUT = false;


void test_fp4_gemv(int B, int M, int N, int K, int warm_up_runs, int runs, int split_k_slices)
{
    // using ElementA = cutlass::float_e2m1_t;
    // using ElementB = cutlass::float_e2m1_t;
    
    // using ElementA = cutlass::half_t;
    // using ElementB = cutlass::half_t;
    // using ElementC = float;

    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementC = cutlass::bfloat16_t;
    using ElementSF = cutlass::float_e4m3_t;
    // using ElementSF = unsigned int;
    const int SFVecSize = 16;

    // host buffer
    ElementA* h_A;
    ElementB* h_B;
    ElementC *h_C;
    ElementC *h_C_ref;
    int32_t  *h_N;
    // cutlass::array<int32_t, B> h_N;

    h_A     = (ElementA*) malloc(B * M * K * sizeof(ElementA));
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

    cudaMalloc((void **)&d_A, B * M * K * sizeof(ElementA));
    cudaMalloc((void **)&d_A_interleaved, B * M * K * sizeof(ElementA));
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
    ElementSF *h_fp8_SF_B;
    ElementSF *d_fp8_SF_A;
    ElementSF *d_fp8_SF_B;

    int SFBlocksByK = (K / SFVecSize + 3) / 4; // K contribution
    int SFBlocksByM = (M + 127) / 128;         // M contribution
    int SFBlocksByN = (N + 127) / 128;         // N contribution
    // printf("SFBlocksByK = %d, SFBlocksByM = %d, SFBlocksByN = %d\n", SFBlocksByK, SFBlocksByM, SFBlocksByN);

    h_fp8_SF_A = (ElementSF*) malloc(SFBlocksByK * SFBlocksByM * 512 * sizeof(ElementSF));
    h_fp8_SF_B = (ElementSF*) malloc(SFBlocksByK * SFBlocksByN * 512 * sizeof(ElementSF));
    cudaMalloc((void **)&d_fp8_SF_A, SFBlocksByK * SFBlocksByM * 512 * sizeof(ElementSF));
    cudaMalloc((void **)&d_fp8_SF_B, SFBlocksByK * SFBlocksByN * 512 * sizeof(ElementSF));

    srand(0);


    for (int b = 0; b < B; ++b) {
        for (int k = 0; k < K; ++k) {
            for (int n = 0; n < N; ++n) {
                h_B[b * N * K + n * K + k] = ElementB(rand() % 16);
            }

            for (int m = 0; m < M; ++m) {
                h_A[b * M * K + m * K + k] = ElementA(rand() % 16);
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
                Fragment_temp temp_A = *reinterpret_cast<Fragment_temp *>(&h_A[m * K / 2 + k / 2]);

                printf("%d:%f  ", k, float(temp_A[0]));
                printf("%d:%f  ", k + 1, float(temp_A[1]));
            }
            printf("\n");
        }
        // print input vector B
        printf("---------------- input vector B ----------------\n");
        for (int k = 0; k < 64; k+=2) {
            Fragment_temp temp_B = *reinterpret_cast<Fragment_temp *>(&h_B[k / 2]);
            printf("%d:%f  ", k, float(temp_B[0]));
            printf("%d:%f  ", k, float(temp_B[1]));
        }
        printf("\n");
    }

    // copy input tensor from host to device
    cudaMemcpy(d_A, h_A, B * M * K * sizeof(ElementA), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, B * K * N * sizeof(ElementB), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fp8_SF_A, h_fp8_SF_A, SFBlocksByK * SFBlocksByM * 512 * sizeof(ElementSF), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fp8_SF_B, h_fp8_SF_B, SFBlocksByK * SFBlocksByN * 512 * sizeof(ElementSF), cudaMemcpyHostToDevice);

    using LayoutA = cutlass::layout::RowMajor;
    using ElementAccumulator = float;
    const int kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementA>::value;
    const int kThreadCount = 128;
    const int kThreadsPerRow = 8;

    // LinearCombination: Applies a linear combination operator to an array of elements.
    // D = alpha * accumulator + beta * source + uniform
    using Epilogue = cutlass::epilogue::thread::LinearCombination<
        ElementC,           // Data type used to load and store tensors
        4,                  // Number of elements computed per operation.
        ElementAccumulator, // Accumulator data type
        ElementAccumulator>;   // Data type used to compute linear combination

    using cutlass_Gemv_kernel = cutlass::gemm::kernel::Gemv<
                                    ElementA, LayoutA, ElementB, ElementC, 
                                    ElementAccumulator, Epilogue, kElementsPerAccess, kThreadCount, kThreadsPerRow, 
                                    ElementSF, SFVecSize>;
    using cutlass_Gemv_device = cutlass::gemm::device::Gemv<cutlass_Gemv_kernel>;

    cutlass_Gemv_kernel::matrix_A_interleave(d_A_interleaved, d_A, B, M, K);


    if (DEBUG_INTERLEAVE) {
        // print input matrix A interleaved
        ElementA *h_A_interleaved;
        h_A_interleaved = (ElementA*) malloc(B * M * K * sizeof(ElementA));
        cudaMemcpy(h_A_interleaved, d_A_interleaved, B * M * K * sizeof(ElementA), cudaMemcpyDeviceToHost);
        
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
        d_fp8_SF_B
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
    
    cutlass::Status run_status;
    
    for (int i = 0; i < warm_up_runs; i++)
    {
        run_status = gemv.run(nullptr);
    }
    
    cudaEventRecord(_event_start_);
    for (int i = 0; i < runs; i++)
    {
        // if (split_k_slices > 1) {
        //     cudaMemset(d_C, 0, B * M * N * sizeof(ElementC));
        // }
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
    printf("GEMM Shape B %d M %d N %d K %d, MEM size = %f GB, LDG_throughput %f GB/s MEM SOL %.2f \%\t", B, M, N, K, GB_total, LDG_throughput, LDG_throughput / 4000 * 100);
    cudaDeviceSynchronize();
    printf("string: %s\n", cudaGetErrorString(cudaGetLastError()));

    if (run_status != cutlass::Status::kSuccess) {
        std::string err_msg =
            "Failed to run cutlass gemv. Error: " + std::string(cutlassGetStatusString(run_status));
        throw std::runtime_error("[CUTLASS Error][jiangs] " + err_msg);
    }

    cudaMemcpy(h_C, d_C, B * M * N * sizeof(ElementC), cudaMemcpyDeviceToHost);
    
    // result validation
    for (int b = 0; b < B; b++) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float accu = 0.f;
                for (int k = 0; k < K; k++) {
    
                    ElementA val_A = h_A[b * M * K + m * K + k];
                    ElementB val_B = h_B[b * N * K + n * K + k];

                    accu += float(val_A) * float(val_B);
                }
                h_C_ref[b * M * N + n * M + m] = ElementC(alpha * accu);
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
    int split_k_slices = std::stoi(argv[7]);

    printf("GEMV B %d M %d N %d K %d warm_up_runs %d runs %d split_k_slices %d\n", B, M, N, K, warm_up_runs, runs, split_k_slices);

    test_fp4_gemv(B, M, N, K, warm_up_runs, runs, split_k_slices);
    
    cudaDeviceSynchronize();
    fflush(stdout);

    return 0;
}