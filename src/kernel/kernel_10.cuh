#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#ifndef FETCH_FLOAT4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#endif
#ifndef FETCH_FLOAT4_CONST
#define FETCH_FLOAT4_CONST(pointer) (reinterpret_cast<const float4 *>(&(pointer))[0])
#endif
#define GROUP_SIZE 8
#define WARP_SIZE 32


namespace kernel10 {
    template<int BM, int BN, int BK, int lda_m_stride, int ldb_k_stride>
    __device__ __forceinline__ void gmem_to_smem(const float * A, const float * B, int M, int N, int K, float * smem_a, float * smem_b)
    {
        // #pragma unroll // A: global -> reg buffer
        for (uint i = 0; i + lda_m_stride <= BM; i += lda_m_stride)
        {
            const float4 tmp = FETCH_FLOAT4_CONST(A[i * K]);
            smem_a[i] = tmp.x;
            smem_a[BM + i] = tmp.y;
            smem_a[2 * BM + i] = tmp.z;
            smem_a[3 * BM + i] = tmp.w;
        }
        // #pragma unroll // B: global -> reg buffer
        for (uint i = 0; i + ldb_k_stride <= BK; i += ldb_k_stride)
        {
            FETCH_FLOAT4(smem_b[i * BN]) = FETCH_FLOAT4_CONST(B[i * N]);
        }
    }

    template <const int BM,
              const int BN,
              const int BK,
              const int WM,
              const int WN,
              const int TM,
              const int TN,
              const int WM_SUBTILE,
              const int WN_SUBTILE,
              const int m_subtiles,
              const int n_subtiles>
    __device__ __forceinline__ void warp_matmul(const float *smem_a, const float *smem_b, float *acc, float *frag_a, float *frag_b) {
        // #pragma unroll
        for (uint k = 0; k < BK; ++k) { 
            // #pragma unroll
            for (uint i = 0; i < m_subtiles; ++i) {
                // #pragma unroll
                // for (uint m = 0; m < TM; m+=4) {
                //     FETCH_FLOAT4(frag_a[i * TM + m]) = FETCH_FLOAT4_CONST(smem_a[k * BM + i * WM_SUBTILE + m]);
                // }
                // #pragma unroll
                for (uint m = 0; m < TM; m+=1) {
                    frag_a[i * TM + m] = smem_a[k * BM + i * WM_SUBTILE + m];
                }
            }
            // #pragma unroll
            for (uint i = 0; i < n_subtiles; ++i) {
                // #pragma unroll
                // for (uint n = 0; n < TN; n+=4) {
                //     FETCH_FLOAT4(frag_b[i * TN + n]) = FETCH_FLOAT4_CONST(smem_b[k * BN + i * WN_SUBTILE + n]);
                // }
                #pragma unroll
                for (uint n = 0; n < TN; n+=1) {
                    frag_b[i * TN + n] = smem_b[k * BN + i * WN_SUBTILE + n];
                }
            }
            // #pragma unroll
            for (uint i = 0; i < m_subtiles; ++i) {
                // #pragma unroll
                for (uint j = 0; j < n_subtiles; ++j) {
                    // #pragma unroll
                    for (uint m = 0; m < TM; ++m) {
                        // #pragma unroll
                        for (uint n = 0; n < TN; ++n) {
                            acc[(i * TM + m) * n_subtiles * TN + j * TN + n] += frag_a[i * TM + m] * frag_b[j * TN + n];
                        }
                    }
                }
            }
        }
    }

} // namespace kernel 10

// WARP tiling without double cache, performing C = alpha * A * B + beta * C
template <const int BM,
          const int BN,
          const int BK,
          const int WM,
          const int WN,
          const int TM,
          const int TN,
          const int WM_SUBTILE,
          const int WN_SUBTILE,
          const int NUM_THREADS,
          const int lda_m_stride,
          const int ldb_k_stride,
          const int m_subtiles,
          const int n_subtiles
          >
__global__ void __launch_bounds__(NUM_THREADS, 3) mysgemm_v10(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    // every thread loads 4 floats at a time in stride-fashion
    const uint warp_m_offset = (threadIdx.x / WARP_SIZE) / (BN / WN) * WM;
    const uint warp_n_offset = (threadIdx.x / WARP_SIZE) % (BN / WN) * WN;
    const uint m_idx_a = threadIdx.x * 4 / BK;
    const uint k_idx_a = threadIdx.x % (BK / 4) * 4;
    const uint k_idx_b = threadIdx.x * 4 / BN;
    const uint n_idx_b = threadIdx.x % (BN / 4) * 4;
    const uint subtile_idx_m = (threadIdx.x % WARP_SIZE) / (WN_SUBTILE / TN) * TM;
    const uint subtile_idx_n = (threadIdx.x % WARP_SIZE) % (WN_SUBTILE / TN) * TN;

    static_assert(lda_m_stride > 0, "lda_m_stride must be positive to ensure uniform strides");
    static_assert(ldb_k_stride > 0, "ldb_k_stride must be positive to ensure uniform strides");

    // declare shared memory
    __shared__ float smem_a[BK * BM]; // transposed
    __shared__ float smem_b[BK * BN];

    A += blockIdx.y * BM * K + m_idx_a * K + k_idx_a;
    B += blockIdx.x * BN + k_idx_b * N + n_idx_b;
    // move C to the warp start
    C += (blockIdx.y * BM + warp_m_offset + subtile_idx_m) * N  + blockIdx.x * BN + warp_n_offset + subtile_idx_n;

    // move A and B to thread start for loading, this has nothing to do with warps

    // declare accumulators
    float acc[m_subtiles * n_subtiles * TM * TN] = {0.};

    // declare fragments
    float frag_a[m_subtiles * TM] = {0.};
    float frag_b[n_subtiles * TN] = {0.};

    
    // #pragma unroll
    for (uint k = 0; k < K; k += BK) {
        kernel10::gmem_to_smem<BM, BN, BK, lda_m_stride, ldb_k_stride>(A, B, M, N, K, smem_a + k_idx_a * BM + m_idx_a, smem_b + k_idx_b * BN + n_idx_b);
        __syncthreads();
        // compute the warp level matmul
        kernel10::warp_matmul<BM, BN, BK, WM, WN, TM, TN, WM_SUBTILE, WN_SUBTILE, m_subtiles, n_subtiles>(smem_a + warp_m_offset + subtile_idx_m, smem_b + + warp_n_offset + subtile_idx_n, acc, frag_a, frag_b);
        A += BK;
        B += BK * N;
        __syncthreads();
    }

    // reduce

    for (uint i = 0; i < m_subtiles; ++i) {
        for (uint j = 0; j < n_subtiles; ++j) {
            // move C to the subtile start
            float *C_subtile = C + i * WM_SUBTILE * N + j * WN_SUBTILE;
            // #pragma unroll
            for (uint m = 0; m < TM; m += 1) {
                // #pragma unroll
                for (uint n = 0; n < TN; n += 4) {
                    float4 tmp = FETCH_FLOAT4(
                        C_subtile[m * N + n]);
                    const int acc_offset = (i * TM + m) * n_subtiles * TN + j * TN + n;
                    tmp.x = alpha * acc[acc_offset] + beta * tmp.x;
                    tmp.y = alpha * acc[acc_offset + 1] + beta * tmp.y;
                    tmp.z = alpha * acc[acc_offset + 2] + beta * tmp.z;
                    tmp.w = alpha * acc[acc_offset + 3] + beta * tmp.w;
                    FETCH_FLOAT4(C_subtile[m * N + n]) = tmp;
                }
            }
        }
    }
}

