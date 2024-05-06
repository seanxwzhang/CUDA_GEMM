#include <stdio.h>
#include "utils.cuh"
#include "kernel.cuh"

#define WARP_SIZE 32

float get_sec() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return (1e6 * time.tv_sec + time.tv_usec);
}

float cpu_elapsed_time(float &beg, float &end) {
    return 1.0e-6 * (end - beg);
}

void cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
};

void CudaDeviceInfo() {
    int deviceId;

    cudaGetDevice(&deviceId);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    /*
   * There should be no need to modify the output string below.
   */

    printf("Device ID: %d\n\
       *Number of SMs: %d\n\
       Compute Capability Major: %d\n\
       Compute Capability Minor: %d\n\
       memoryBusWidth: %d\n\
       *maxThreadsPerBlock: %d\n\
       maxThreadsPerMultiProcessor: %d\n\
       *totalGlobalMem: %zuM\n\
       sharedMemPerBlock: %zuKB\n\
       *sharedMemPerMultiprocessor: %zuKB\n\
       totalConstMem: %zuKB\n\
       *multiProcessorCount: %d\n\
       *Warp Size: %d\n",
           deviceId,
           props.multiProcessorCount,
           props.major,
           props.minor,
           props.memoryBusWidth,
           props.maxThreadsPerBlock,
           props.maxThreadsPerMultiProcessor,
           props.totalGlobalMem / 1024 / 1024,
           props.sharedMemPerBlock / 1024,
           props.sharedMemPerMultiprocessor / 1024,
           props.totalConstMem / 1024,
           props.multiProcessorCount,
           props.warpSize);
};

void randomize_matrix(float *mat, int N) {
    // NOTICE: 使用gettimeofdays替代srand((unsigned)time(NULL));time精度过低，产生相同随机数
    struct timeval time;
    gettimeofday(&time, NULL);
    srand(time.tv_usec);
    for (int i = 0; i < N; i++) {
        float tmp = (float) (rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i] = tmp;
    }
}

void copy_matrix(float *src, float *dest, int N) {
    int i;
    for (i = 0; src + i && dest + i && i < N; i++)
        *(dest + i) = *(src + i);
    if (i != N)
        printf("copy failed at %d while there are %d elements in total.\n", i, N);
}

void print_matrix(const float *A, int M, int N) {
    int i;
    printf("[");
    for (i = 0; i < M * N; i++) {
        if ((i + 1) % N == 0)
            printf("%5.2f ", A[i]);
        else
            printf("%5.2f, ", A[i]);
        if ((i + 1) % N == 0) {
            if (i + 1 < M * N)
                printf(";\n");
        }
    }
    printf("]\n");
}

bool verify_matrix(float *mat1, float *mat2, int N) {
    double diff = 0.0;
    int i;
    for (i = 0; mat1 + i && mat2 + i && i < N; i++) {
        diff = fabs((double) mat1[i] - (double) mat2[i]);
        if (diff > 1e-2) {
            printf("error. %5.2f,%5.2f,%d\n", mat1[i], mat2[i], i);
            return false;
        }
    }
    return true;
}

#define CEIL_DIV(M, N) int(((M) + (N)-1) / (N))

void test_cublas(cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    //cublas列主序计算：https://www.cnblogs.com/cuancuancuanhao/p/7763256.html
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
}

void test_cublasSgemmEx(cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
               N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void test_mysgemm_v1(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 blockDim(32, 32);
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    mysgemm_v1<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void test_mysgemm_v2(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 blockDim(1024);
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    mysgemm_v2<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void test_mysgemm_v3(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 blockDim(512);
    dim3 gridDim(CEIL_DIV(M, 64), CEIL_DIV(N, 64));
    mysgemm_v3<64, 64, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void test_mysgemm_v4(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 blockDim(256);
    dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
    mysgemm_v4<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void test_mysgemm_v5(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 blockDim(256);
    dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
    mysgemm_v5<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

//void test_mysgemm_v6(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
//    dim3 blockDim(4);
//    dim3 gridDim(CEIL_DIV(M, 8), CEIL_DIV(N, 8));
//    mysgemm_v6<8, 8, 4, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
//}

void test_mysgemm_v6(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 blockDim(256);
    dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
    mysgemm_v6<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void test_mysgemm_v7(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 blockDim(256);
    dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
    mysgemm_v7<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}


// my baseline
void test_mysgemm_v8(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
    dim3 blockDim(16, 16); // 2D CTA
    mysgemm_v8<128, 128, 8, 8, 8, 256><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// autotune
void test_mysgemm_v9(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
    dim3 blockDim(16, 16); // 2D CTA, 2 warps per block
    mysgemm_v9<128, 128, 8, 8, 8, 256><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void test_mysgemm_v10(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int WM = 64;
    constexpr int WN = 64;
    constexpr int TM = 8;
    constexpr int TN = 4; // we want this to be 4 to avoid bank conflict naturally 
    constexpr int warps_per_block = BM / WM * BN / WN;
    constexpr int NUM_THREADS = warps_per_block * WARP_SIZE;
    constexpr int loads_per_iter = 4 * BM / WM * BN / WN * WARP_SIZE;
    constexpr int lda_m_stride = loads_per_iter / BK;
    constexpr int ldb_k_stride = loads_per_iter / BN;


    // we are distributing threads in row-major order
    constexpr int WM_SUBTILE = 32; // the size of an interation/subtile in the M dimension, 4 threads * 4 floats/per thread
    constexpr int WN_SUBTILE = WARP_SIZE * TN * TM / WM_SUBTILE; // the size of an interation/subtile in the N dimension, 8 threads * 4 floats/per thread
    constexpr int m_subtiles = WM / WM_SUBTILE; // number of subtiles in the M dimension
    constexpr int n_subtiles = WN / WN_SUBTILE; // number of subtiles in the N dimension

    static_assert(WARP_SIZE == WN_SUBTILE / TN * WM_SUBTILE / TM);
    static_assert(loads_per_iter % BK == 0, "BK must be divisible by loads_per_iter");
    static_assert(loads_per_iter % BN == 0, "BN must be divisible by loads_per_iter");


    dim3 gridDim(CEIL_DIV(M, BN), CEIL_DIV(N, BM));
    dim3 blockDim(WARP_SIZE * warps_per_block); // 1D CTA
    mysgemm_v10<BM, BN, BK, WM, WN, TM, TN, WM_SUBTILE, WN_SUBTILE, NUM_THREADS, lda_m_stride, ldb_k_stride, m_subtiles, n_subtiles><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}


void test_mysgemm_v11(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
// This kernel implments split-K algorithm to further increase the utilization of SMs
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 4;
    constexpr int SPLIT = 16;
    constexpr int WM = 64;
    constexpr int WN = 64;
    constexpr int TM = 8;
    constexpr int TN = 4; 

    constexpr int warps_per_block = BM / WM * BN / WN;
    constexpr int NUM_THREADS = warps_per_block * WARP_SIZE;
    constexpr int loads_per_iter = 4 * BM / WM * BN / WN * WARP_SIZE;
    constexpr int lda_m_stride = loads_per_iter / BK;
    constexpr int ldb_k_stride = loads_per_iter / BN;

    constexpr int WM_SUBTILE = 32;
    constexpr int WN_SUBTILE = WARP_SIZE * TN * TM / WM_SUBTILE; 
    constexpr int m_subtiles = WM / WM_SUBTILE;
    constexpr int n_subtiles = WN / WN_SUBTILE;

    assert(K % BK == 0);
    static_assert(WARP_SIZE == WN_SUBTILE / TN * WM_SUBTILE / TM);
    static_assert(loads_per_iter % BK == 0, "BK must be divisible by loads_per_iter");
    static_assert(loads_per_iter % BN == 0, "BN must be divisible by loads_per_iter");
    

    // create temporary C
    float *tC = nullptr;
    cudaCheck(cudaMalloc((void **)&tC, sizeof(float) * M * N * SPLIT), __FILE__, __LINE__);

    dim3 gridDim(CEIL_DIV(M, BN), CEIL_DIV(N, BM), SPLIT);
    dim3 blockDim(WARP_SIZE * warps_per_block); // 1D CTA
    mysgemm_v11<BM, BN, BK, SPLIT, WM, WN, TM, TN, WM_SUBTILE, WN_SUBTILE, NUM_THREADS, lda_m_stride, ldb_k_stride, m_subtiles, n_subtiles><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, tC, C);

    constexpr int blockSize = 1024; // number of threads in the CTA
    constexpr int gridSize = 1024;
    constexpr int smem_elements = 128; // how many 
    const int iterations = M * N / gridSize / smem_elements; // number of "big" iterations every block needs to handle
    constexpr int stages = 2;

    assert((M * N) % gridSize == 0);
    assert((M * N / gridSize) % smem_elements == 0); // don't want to handle this edge case
    assert(stages < iterations); // it doesn't make sense to have more stages than iterations
    // I adopt the following reduction scheme:
    // for block in grid:
    //      for i in iterations:
    //          1. fetch smem_elements*SPLIT elements from from gmem->smem, using producer and consumer pipelines
    //          2. carry out reduction:
    //              2.1 break each block into groups of SPLIT/2 (for reduction)
    //              2.2 each group carries out the paralle reduction with step complexity of log(SPLIT)
    //              2.3 the first thread in each group stores the result back to smem
    //              2.4 kick off an async smem->gmem copy, don't wait for it
    //          3. wait for next phase of smem to be ready 

    dim3 reduceGrid(gridSize);
    dim3 reduceBlock(blockSize);
    constexpr int smem_size = smem_elements * SPLIT * sizeof(float) * stages;
    constexpr int group_num = blockSize / SPLIT;
    constexpr int reduction_iters = smem_elements / group_num; // reduction iterations per smem_elements
    reduce_k<SPLIT, smem_elements, stages, reduction_iters><<<reduceGrid, reduceBlock, smem_size>>>(M, N, K, tC, C, iterations);
}


void runSgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  // Settings for A100
  // const uint K10_NUM_THREADS = 128;
  // const uint K10_BN = 128;
  // const uint K10_BM = 64;
  // const uint K10_BK = 16;
  // const uint K10_WN = 64;
  // const uint K10_WM = 32;
  // const uint K10_WNITER = 1;
  // const uint K10_TN = 4;
  // const uint K10_TM = 4;
  // Settings for A6000
  const uint K10_NUM_THREADS = 128;
  const uint K10_BN = 128;
  const uint K10_BM = 128;
  const uint K10_BK = 16;
  const uint K10_WN = 64;
  const uint K10_WM = 64;
  const uint K10_WNITER = 4;
  const uint K10_TN = 4;
  const uint K10_TM = 8;
  dim3 blockDim(K10_NUM_THREADS);

  constexpr uint NUM_WARPS = K10_NUM_THREADS / 32;

  // warptile in threadblocktile
  static_assert((K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0));
  static_assert((K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((K10_WM * K10_WN) % (WARP_SIZE * K10_TM * K10_TN * K10_WNITER) ==
                0);
  constexpr uint K10_WMITER =
      (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);
  // warpsubtile in warptile
  static_assert((K10_WM % K10_WMITER == 0) and (K10_WN % K10_WNITER == 0));

  static_assert((K10_NUM_THREADS * 4) % K10_BK == 0,
                "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((K10_NUM_THREADS * 4) % K10_BN == 0,
                "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(K10_BN % (16 * K10_TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K10_BM % (16 * K10_TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(M, K10_BM));
  sgemmWarptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
                  K10_TN, K10_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void test_kernel(int kernel_num, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C,
                 cublasHandle_t handle) {
    switch (kernel_num) {
        case 0:
            // test_cublas(handle, M, N, K, alpha, A, B, beta, C);
            test_cublasSgemmEx(handle, M, N, K, alpha, A, B, beta, C);
            break;
        case 1:
            test_mysgemm_v1(M, N, K, alpha, A, B, beta, C);
            break;
        case 2:
            test_mysgemm_v2(M, N, K, alpha, A, B, beta, C);
            break;
        case 3:
            test_mysgemm_v3(M, N, K, alpha, A, B, beta, C);
            break;
        case 4:
            test_mysgemm_v4(M, N, K, alpha, A, B, beta, C);
            break;
        case 5:
            test_mysgemm_v5(M, N, K, alpha, A, B, beta, C);
            break;
        case 6:
            test_mysgemm_v6(M, N, K, alpha, A, B, beta, C);
            break;
        case 7:
            test_mysgemm_v7(M, N, K, alpha, A, B, beta, C);
            break;
        case 8:
            test_mysgemm_v8(M, N, K, alpha, A, B, beta, C);
            break;
        case 9:
            test_mysgemm_v9(M, N, K, alpha, A, B, beta, C);
            break;
        case 10:
            test_mysgemm_v10(M, N, K, alpha, A, B, beta, C);
            break;
        case 11:
            test_mysgemm_v11(M, N, K, alpha, A, B, beta, C);
            break;
        case 99:
            runSgemmWarptiling(M, N, K, alpha, A, B, beta, C);
            break;
        case 100:
            test_cublasSgemmEx(handle, M, N, K, alpha, A, B, beta, C);
            break;
        default:
            break;
    }
}