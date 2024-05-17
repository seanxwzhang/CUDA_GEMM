#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda/pipeline>
#include "kernel/kernel_11.cuh"

void cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
};

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))
#define CEIL_DIV(M, N) int(((M) + (N)-1) / (N))

namespace cg = cooperative_groups;


int main() {
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int K = 4096;

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

    float * A = (float *) malloc(M * K * sizeof(float));
    float * B = (float *) malloc(N * K * sizeof(float));
    float * C = (float *) malloc(M * N * sizeof(float));
    float * Result = (float *) malloc(M * N * SPLIT * sizeof(float));
    float alpha = 1.0; float beta = 0.5;

    for (int i = 0; i < M * K; i++) A[i] = (i / K) % 2 == 0 ? 1.0 : -1.0;
    for (int i = 0; i < N * K; i++) B[i] = (i % N) % 2 == 0 ? 1.0 : -1.0;
    for (int i = 0; i < M * N; i++) C[i] = 0.5;

    float * dA, * dB, * dC, *dResult;
    cudaCheck(cudaMalloc(&dA, M * K * sizeof(float)));
    cudaCheck(cudaMalloc(&dB, N * K * sizeof(float)));
    cudaCheck(cudaMalloc(&dC, M * N * sizeof(float)));
    cudaCheck(cudaMalloc(&dResult, M * N * SPLIT * sizeof(float)));

    cudaCheck(cudaMemcpy(dA, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, N * K * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, C, M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 gridDim(CEIL_DIV(M, BN), CEIL_DIV(N, BM), SPLIT);
    dim3 blockDim(WARP_SIZE * warps_per_block); // 1D CTA
    mysgemm_v11<BM, BN, BK, SPLIT, WM, WN, TM, TN, WM_SUBTILE, WN_SUBTILE, NUM_THREADS, lda_m_stride, ldb_k_stride, m_subtiles, n_subtiles><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dResult, dC);


    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());

    cudaCheck(cudaMemcpy(Result, dResult, M * N * SPLIT * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N * SPLIT; i++) {
        int x = (i % (N * SPLIT)) / SPLIT;
        int y = (i / (N * SPLIT));
        float expected = std::pow(-1, x % 2) * std::pow(-1, y % 2) * (float)(K / SPLIT);
        if (i % SPLIT == 0) {
            expected += 0.25;
        }
        if (Result[i] != expected) {
            std::cout << "Error at " << i << ": expected " << expected << ", received "<< Result[i] << std::endl;
            break;
        }
    }

    constexpr int blockSize = 1024; // number of threads in the CTA
    constexpr int gridSize = 1024;
    constexpr int smem_elements = 32; // number of final element a block processes
    const int iterations = M * N / gridSize / smem_elements; // number of "big" iterations every block needs to handle
    constexpr int stages = 5;

    dim3 reduceGrid(gridSize);
    dim3 reduceBlock(blockSize);
    constexpr int smem_size = smem_elements * SPLIT * sizeof(float) * stages;
    constexpr int group_num = blockSize / SPLIT;
    constexpr int reduction_iters = smem_elements / group_num; 

    reduce_k<SPLIT, smem_elements, stages, reduction_iters><<<reduceGrid, reduceBlock, smem_size>>>(M, N, dResult, dC, iterations);

    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());
    free(A); free(B);
    cudaFree(dA); cudaFree(dB);
    cudaCheck(cudaMemcpy(C, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; i++) {
        int x = i % N;
        int y = i / N;
        float expected = std::pow(-1, x % 2) * std::pow(-1, y % 2) * (float)K + 0.25;
        if (C[i] != expected) {
            std::cout << "Error at " << i << ": expected " << expected << ", received "<< C[i] << std::endl;
            break;
        }
    }


    free(C); free(Result);
    cudaFree(dC); cudaFree(dResult);
    return 0;
}