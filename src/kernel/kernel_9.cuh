#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef FETCH_FLOAT4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#endif
#ifndef DIV_UP
#define DIV_UP(m, n) ((m + n - 1) / n)
#endif
#define GROUP_SIZE 8
#define WARP_SIZE 32


namespace kernel9 {

template<int BM, int BN, int BK>
__device__ __forceinline__ void gmem_to_smem(float *A, float *B, float smem_a[][BK][BM], float smem_b[][BK][BN], float ldreg_a[][4], float ldreg_b[][4], int a_smem_rounds, int a_stride, int a_smem_x, int a_smem_y, int b_smem_rounds, int b_stride, int b_smem_y, int b_smem_x, int phase)
{
#pragma unroll // A: global -> reg buffer
    for (int i = 0; i < a_smem_rounds; ++i)
    {
        FETCH_FLOAT4(ldreg_a[i]) = FETCH_FLOAT4(A[i * a_stride]);
        // int bank_id = a_smem_y;
        // int bank_row = tid * BK / 128;
        // int swizzled_a_smem_y = a_smem_y + bank_row * WARP_SIZE * 4 / BK;

        smem_a[phase][a_smem_x][a_smem_y + i * a_stride] = ldreg_a[i][0];
        smem_a[phase][a_smem_x + 1][a_smem_y + i * a_stride] = ldreg_a[i][1];
        smem_a[phase][a_smem_x + 2][a_smem_y + i * a_stride] = ldreg_a[i][2];
        smem_a[phase][a_smem_x + 3][a_smem_y + i * a_stride] = ldreg_a[i][3];
    }
#pragma unroll // B: global -> reg buffer
    for (int i = 0; i < b_smem_rounds; ++i)
    {
        FETCH_FLOAT4(ldreg_b[i]) = FETCH_FLOAT4(B[i * b_stride]);
        FETCH_FLOAT4(smem_b[phase][b_smem_y][b_smem_x + i * b_stride]) = FETCH_FLOAT4(ldreg_b[i]);
    }
}

__device__ __forceinline__ void gmem_to_reg(float *A, float *B, float ldreg_a[][4], float ldreg_b[][4], int a_smem_rounds, int a_stride, int b_smem_rounds, int b_stride)
{
#pragma unroll // A: global -> reg buffer
    for (int i = 0; i < a_smem_rounds; ++i)
    {
        FETCH_FLOAT4(ldreg_a[i]) = FETCH_FLOAT4(A[i * a_stride]);
    }
#pragma unroll // B: global -> reg buffer
    for (int i = 0; i < b_smem_rounds; ++i)
    {
        FETCH_FLOAT4(ldreg_b[i]) = FETCH_FLOAT4(B[i * b_stride]);
    }
}

template<int BM, int BN, int BK>
__device__ __forceinline__ void reg_to_smem(float smem_a[][BK][BM], float smem_b[][BK][BN], float ldreg_a[][4], float ldreg_b[][4], int a_smem_rounds, int a_stride, int a_smem_x, int a_smem_y, int b_smem_rounds, int b_stride, int b_smem_y, int b_smem_x, int phase)
{
#pragma unroll // A: reg buffer -> smem
    for (int i = 0; i < a_smem_rounds; ++i)
    { // note that this is uncoalesce memory write, and only 4 floats * 4 byte/float = 16 bytes per write
        smem_a[phase][a_smem_x][a_smem_y + i * a_stride] = ldreg_a[i][0];
        smem_a[phase][a_smem_x + 1][a_smem_y + i * a_stride] = ldreg_a[i][1];
        smem_a[phase][a_smem_x + 2][a_smem_y + i * a_stride] = ldreg_a[i][2];
        smem_a[phase][a_smem_x + 3][a_smem_y + i * a_stride] = ldreg_a[i][3];
    }
#pragma unroll // B: reg buffer -> smem
    for (int i = 0; i < b_smem_rounds; ++i)
    {
        FETCH_FLOAT4(smem_b[phase][b_smem_y][b_smem_x + i * b_stride]) = FETCH_FLOAT4(ldreg_b[i]);
    }
}

template<int BM, int BN, int BK, int TM, int TN>
__device__ __forceinline__ void smem_to_frag(float frag_a[][TM], float frag_b[][TN], float smem_a[][BK][BM], float smem_b[][BK][BN], int frag_phase, int smem_phase, int bk)
{
#pragma unroll 
    for (int i = 0; i < TM; i += 4)
    {
        int tmp = (threadIdx.y * TM + i);
        tmp = ((tmp / WARP_SIZE) ^ ((tmp % WARP_SIZE) / 4)) % 2 * 4;
        FETCH_FLOAT4(frag_a[frag_phase][tmp]) = FETCH_FLOAT4(smem_a[smem_phase][bk][threadIdx.y * TM + tmp]);
    }
#pragma unroll
    for (int i = 0; i < TN; i += 4)
    {
        int tmp = (threadIdx.x * TN + i);
        tmp = ((tmp / WARP_SIZE) ^ ((tmp % WARP_SIZE) / 4)) % 2 * 4;
        FETCH_FLOAT4(frag_b[frag_phase][tmp]) = FETCH_FLOAT4(smem_b[smem_phase][bk][threadIdx.x * TN + tmp]);
    }
}

} // namespace kernel 9

// This function assumes B is already transposed
template <const int BM,
          const int BN,
          const int BK,
          const int TM,
          const int TN,
          const int THREAD_NUMS>
__global__ void __launch_bounds__(THREAD_NUMS, 2) mysgemm_v9(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    constexpr int threads_per_block = BM / TM * BN / TN;
    constexpr int a_ele_per_thread_smem = BM * BK / threads_per_block;
    constexpr int b_ele_per_thread_smem = BK * BN / threads_per_block;
    constexpr int a_smem_rounds = a_ele_per_thread_smem / 4;
    constexpr int b_smem_rounds = b_ele_per_thread_smem / 4;
    constexpr int a_threads_per_row_per_round = BK / 4;
    int a_stride = threads_per_block / a_threads_per_row_per_round * K;
    constexpr int b_threads_per_row_per_round = BN / 4;
    int b_stride = threads_per_block / b_threads_per_row_per_round * N;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // int lane_id = tid % 32;
    int a_smem_x = (tid % a_threads_per_row_per_round) * 4;
    int a_smem_y = tid / a_threads_per_row_per_round;
    int b_smem_x = (tid % b_threads_per_row_per_round) * 4;
    int b_smem_y = tid / b_threads_per_row_per_round;

    static_assert((BM * BK) % threads_per_block == 0);
    static_assert((BK * BN) % threads_per_block == 0);
    static_assert(a_ele_per_thread_smem % 4 == 0);
    static_assert(b_ele_per_thread_smem % 4 == 0);
    static_assert(BK % 4 == 0);
    static_assert(BN % 4 == 0);
    static_assert(threads_per_block / a_threads_per_row_per_round >= 1); // at least cover a row per round
    static_assert(threads_per_block / b_threads_per_row_per_round >= 1); // at least cover a row per round
    static_assert(TN % 4 == 0); // at least 4 elements per thread and TN is a multiple of 4

    float accum[TM][TN] = {0.};

    __shared__ float smem_a[2][BK][BM]; // a transposed version of A block
    __shared__ float smem_b[2][BK][BN];

    // register for loading from global mem to smem
    float ldreg_a[a_smem_rounds][4];
    float ldreg_b[b_smem_rounds][4];

    // fragment/register for computation
    float frag_a[2][TM];
    float frag_b[2][TN];

    // move A to thread start
    A = &A[by * BM * K + a_smem_y * K + a_smem_x];
    B = &B[b_smem_y * N + bx * BN + b_smem_x];

    // 1.1 fetch from global to smem, use register as buffer
    kernel9::gmem_to_smem<BM, BN, BK>(A, B, smem_a, smem_b, ldreg_a, ldreg_b, a_smem_rounds, a_stride, a_smem_x, a_smem_y, b_smem_rounds, b_stride, b_smem_y, b_smem_x, 0);
    __syncthreads(); // need the sync such that the following fragment can be obtained

    // 1.2 load 0 round of smem->frag
    kernel9::smem_to_frag<BM, BN, BK, TM, TN>(frag_a, frag_b, smem_a, smem_b, 0, 0, 0); // load first batch of frag from first block of smem
    int smem_write_index = 1; // next index of smems to write to
    int smem_read_index; // read is current write

    // 2. start the blockwise loop
    for (int k = 0; k < K / BK ; ++k)
    {
        // 2.0 fetch from global to smem, use register as buffer
        if (k + 1 < K / BK) {
            A += BK; // every iteration, A moves BK to the right
            B += N * BK; // every iteration, B moves BK * N down
            kernel9::gmem_to_reg(A, B, ldreg_a, ldreg_b, a_smem_rounds, a_stride, b_smem_rounds, b_stride); // only load to reg, this is non-blocking
        }
        // 2.1 use the frag already loaded to compute the outer product, note that we do register prefetching here

        smem_read_index = smem_write_index ^ 1;
#pragma unroll
        for (int b_k = 1; b_k < BK; ++b_k) // load one sub row at a time from smem to frag
        {
            kernel9::smem_to_frag<BM, BN, BK, TM, TN>(frag_a, frag_b, smem_a, smem_b, b_k % 2, smem_read_index, b_k);
#pragma unroll
            for (int i = 0; i < TM; ++i)
            { // outer product for the previous prefetched frag
#pragma unroll
                for (int j = 0; j < TN; ++j)
                {
                    accum[i][j] += frag_a[(b_k - 1) % 2][i] * frag_b[(b_k - 1) % 2][j];
                }
            }
        }
        // 2.2 if there's next block, start loading from reg to smem
        if (k + 1 < K / BK) {
            kernel9::reg_to_smem<BM, BN, BK>(smem_a, smem_b, ldreg_a, ldreg_b, a_smem_rounds, a_stride, a_smem_x, a_smem_y, b_smem_rounds, b_stride, b_smem_y, b_smem_x, smem_write_index);
            __syncthreads();
            // prefetch a round of fragments from the current write, this will be blocking
            kernel9::smem_to_frag<BM, BN, BK, TM, TN>(frag_a, frag_b, smem_a, smem_b, 0, smem_write_index, 0);
            smem_write_index ^= 1; // update next write
        }
#pragma unroll
        for (int i = 0; i < TM; ++i) 
        { // one last round of outer product because we have only done BK - 1 products
#pragma unroll
            for (int j = 0; j < TN; ++j)
            {
                accum[i][j] += frag_a[(BK - 1) % 2][i] * frag_b[(BK - 1) % 2][j];
            }
        }
    }

    // 3. put the accumulate value down to C
    // move C to thread tile start
    C = &C[(by * BM + threadIdx.y * TM) * N + bx * BN + threadIdx.x * TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TM; j += 4) {
            float4 tmp = FETCH_FLOAT4(C[i * N + j]);
            tmp.x = alpha * accum[i][j] + beta * tmp.x;
            tmp.y = alpha * accum[i][j + 1] + beta * tmp.y;
            tmp.z = alpha * accum[i][j + 2] + beta * tmp.z;
            tmp.w = alpha * accum[i][j + 3] + beta * tmp.w;
            FETCH_FLOAT4(C[i * N + j]) = tmp;
        }
    }
}

