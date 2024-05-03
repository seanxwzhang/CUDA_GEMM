#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef FETCH_FLOAT4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#endif
#ifndef FETCH_FLOAT4_CONST
#define FETCH_FLOAT4_CONST(pointer) (reinterpret_cast<const float4 *>(&(pointer))[0])
#endif
#define GROUP_SIZE 8
#define WARP_SIZE 32


namespace kernel10 {
    template<int BM, int BN, int BK, int lda_m_stride, int ldb_k_stride, int lda_rounds, int ldb_rounds>
    __device__ __forceinline__ void gmem_to_smem(const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ smem_a, float * __restrict__ smem_b, int sa_m_offset, int sa_k_offset, int sb_k_offset, int sb_n_offset, int lda_stride, int ldb_stride)
    {
        #pragma unroll // A: global -> reg buffer
        for (int i = 0; i < lda_rounds; ++i)
        {
            const float4 tmp = FETCH_FLOAT4_CONST(A[i * lda_stride]);
            smem_a[sa_k_offset * BM + sa_m_offset + i * lda_m_stride] = tmp.x;
            smem_a[(sa_k_offset + 1) * BM + sa_m_offset + i * lda_m_stride] = tmp.x;
            smem_a[(sa_k_offset + 2) * BM + sa_m_offset + i * lda_m_stride] = tmp.x;
            smem_a[(sa_k_offset + 3) * BM + sa_m_offset + i * lda_m_stride] = tmp.x;
        }
        #pragma unroll // B: global -> reg buffer
        for (int i = 0; i < ldb_rounds; ++i)
        {
            FETCH_FLOAT4(smem_b[sb_k_offset * BN + sb_n_offset + i * ldb_k_stride]) = FETCH_FLOAT4_CONST(B[i * ldb_stride]);
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
              const int warp_m_subtiles,
              const int warp_n_subtiles>
    __device__ __forceinline__ void warp_matmul(const float *smem_a, const float *smem_b, float *acc, float *frag_a, float *frag_b, int warp_m_offset, int warp_n_offset) {
        // load A from smem for this warp, for all subtiles
        int lane_id = threadIdx.x % WARP_SIZE;
        int subtile_m_offset = lane_id / (WN_SUBTILE / TN) * TM;
        int subtile_n_offset = lane_id % (WN_SUBTILE / TN) * TN;
        #pragma unroll
        for (int k = 0; k < BK; ++k) { 
            #pragma unroll
            for (int i = 0; i < warp_m_subtiles; ++i) {
                for (int m = 0; m < TM; m += 4) {
                    FETCH_FLOAT4(frag_a[i * TM + m]) = FETCH_FLOAT4_CONST(smem_a[k * BM + warp_m_offset + i * WM_SUBTILE + subtile_m_offset + m]);
                }
            }
            #pragma unroll
            for (int i = 0; i < warp_n_subtiles; ++i) {
                for (int n = 0; n < TN; n += 4) {
                    FETCH_FLOAT4(frag_b[i * TN + n]) = FETCH_FLOAT4_CONST(smem_b[k * BN + warp_n_offset + i * WN_SUBTILE + subtile_n_offset + n]);
                }
            }
            #pragma unroll
            for (int i = 0; i < warp_m_subtiles; ++i) {
                #pragma unroll
                for (int j = 0; j < warp_n_subtiles; ++j) {
                    #pragma unroll
                    for (int m = 0; m < TM; m += 1) {
                        #pragma unroll
                        for (int n = 0; n < TN; n += 1) {
                            acc[(i * TM + m) * warp_n_subtiles * TN + j * TN + n] += frag_a[i * TM + m] * frag_b[j * TN + n];
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
          const int WN_SUBTILE
          >
__global__ void mysgemm_v10(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    // every thread loads 4 floats at a time in stride-fashion
    constexpr int threads_per_block = BM / WM * BN / WN * WARP_SIZE;
    constexpr int loads_per_iter = 4 * threads_per_block;
    constexpr int lda_m_stride = loads_per_iter / BK;
    constexpr int ldb_k_stride = loads_per_iter / BN;
    constexpr int lda_rounds = BM * BK / loads_per_iter;
    constexpr int ldb_rounds = BN * BK / loads_per_iter;
    constexpr int warp_m_subtiles = WM / WM_SUBTILE; // number of subtiles in the M dimension
    constexpr int warp_n_subtiles = WN / WN_SUBTILE; // number of subtiles in the N dimension
    int lda_stride = lda_m_stride * K;
    int ldb_stride = ldb_k_stride * N;
    int warp_m_offset = (threadIdx.x / WARP_SIZE) / (BN / WN) * WM;
    int warp_n_offset = (threadIdx.x / WARP_SIZE) % (BN / WN) * WN;

    static_assert(lda_m_stride > 0, "lda_m_stride must be positive to ensure uniform strides");
    // declare shared memory
    __shared__ float smem_a[BK * BM]; // transposed
    __shared__ float smem_b[BK * BN];

    // declare accumulators
    float acc[warp_m_subtiles * warp_n_subtiles * TM * TN] = {0.};

    // declare fragments
    float frag_a[warp_m_subtiles * TM] = {0.};
    float frag_b[warp_n_subtiles * TN] = {0.};

    // move A and B to thread start for loading, this has nothing to do with warps
    int sa_m_offset = threadIdx.x * 4 / BK;
    int sa_k_offset = threadIdx.x % (BK / 4) * 4;
    int sb_k_offset = threadIdx.x * 4 / BN;
    int sb_n_offset = threadIdx.x % (BN / 4) * 4;
    A += blockIdx.y * BM * K + sa_m_offset * K + sa_k_offset;
    B += (threadIdx.x * 4 / BN) * N + sa_k_offset * N + sa_m_offset;
    
    // #pragma unroll
    for (int k = 0; k < K; k += BK) {
        kernel10::gmem_to_smem<BM, BN, BK, lda_m_stride, ldb_k_stride, lda_rounds, ldb_rounds>(A, B, smem_a, smem_b, sa_m_offset, sa_k_offset, sb_k_offset, sb_n_offset, lda_stride, ldb_stride);
        __syncthreads();
        // compute the warp level matmul
        kernel10::warp_matmul<BM, BN, BK, WM, WN, TM, TN, WM_SUBTILE, WN_SUBTILE, warp_m_subtiles, warp_n_subtiles>(smem_a, smem_b, acc, frag_a, frag_b, warp_m_offset, warp_n_offset);
        __syncthreads();
        A += BK;
        B += BK * N;
    }

    // reduce
    int lane_id = threadIdx.x % WARP_SIZE;
    int subtile_m_offset = lane_id / (WN_SUBTILE / TN) * TM;
    int subtile_n_offset = lane_id % (WN_SUBTILE / TN) * TN;

    // move C to the warp start
    C += (blockIdx.y * BM + warp_m_offset) * N  + blockIdx.x * BN + warp_n_offset;
    #pragma unroll
    for (int i = 0; i < warp_m_subtiles; ++i) {
        #pragma unroll
        for (int j = 0; j < warp_n_subtiles; ++j) {
            // move C to the subtile start
            float *C_subtile = C + i * WM_SUBTILE * N + j * WN_SUBTILE;
            #pragma unroll
            for (int m = 0; m < TM; m += 1) {
                #pragma unroll
                for (int n = 0; n < TM; n += 4) {
                    float4 tmp = FETCH_FLOAT4(C_subtile[(subtile_m_offset + m) * N + subtile_n_offset + n]);
                    const int acc_offset = (i * TM + m) * warp_n_subtiles * TN + j * TN + n;
                    tmp.x = beta * tmp.x + alpha * acc[acc_offset];
                    tmp.y = beta * tmp.y + alpha * acc[acc_offset + 1];
                    tmp.z = beta * tmp.z + alpha * acc[acc_offset + 2];
                    tmp.w = beta * tmp.w + alpha * acc[acc_offset + 3];
                    FETCH_FLOAT4(C_subtile[(subtile_m_offset + m) * N + subtile_n_offset + n]) = tmp;
                }
            }
        }
    }
}

