#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda/barrier>
#include <cuda/pipeline>

namespace cg = cooperative_groups;

#ifndef FETCH_FLOAT4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#endif
#ifndef FETCH_FLOAT4_CONST
#define FETCH_FLOAT4_CONST(pointer) (reinterpret_cast<const float4 *>(&(pointer))[0])
#endif
#ifndef DIV_UP
#define DIV_UP(m, n) (((m) + (n) - 1) / (n))
#endif
#define GROUP_SIZE 8
#define WARP_SIZE 32


namespace kernel11 {
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
    __device__ void warp_matmul(const float *smem_a, const float *smem_b, float *acc, float *frag_a, float *frag_b, int warp_m_offset, int subtile_idx_m, int warp_n_offset, int subtile_idx_n) {
        smem_a += warp_m_offset + subtile_idx_m;
        smem_b += warp_n_offset + subtile_idx_n;
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
                // #pragma unroll
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

} // namespace kernel 11

// WARP tiling without double cache, performing C = alpha * A * B + beta * C
template <const int BM,
          const int BN,
          const int BK,
          const int SPLIT,
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
__global__ void __launch_bounds__(NUM_THREADS, 2) mysgemm_v11(int M, int N, int K, float alpha, float *A, float *B, float beta, float *tC, float *C)
{
    // The strided split K can be visualized as follows:
    // ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┐
    // │        │        │        │        │        │        │        │
    // │ split0 │ split1 │ split0 │ split1 │ split0 │ split1 │ split0 │
    // │        │        │        │        │        │        │        │
    // │ block0 │ block1 │ block2 │ block3 │ block4 │ block5 │ block6 │
    // │        │        │        │        │        │        │        │
    // └────────┴────────┴────────┴────────┴────────┴────────┴────────┘
    // The reason for strided splits is that different splits handle BKs in a strided fashion to improve L2 cache hit rate.
    // Note that there might be remainder blocks left causing imbalanced processing across CTAs, this can be handled via stream-K (https://arxiv.org/pdf/2301.03598), but here we'll just ignore (the imbalance) and process it anyway.
    // To assist reduction, it's better to store the output from different splits together:
    //  ┌─────────────────────┐                     
    //  │    unit0 - split0   │                     
    //  ├─────────────────────┤                     
    //  │    unit0 - split1   │                     
    //  ├─────────────────────┤                     
    //  │    unit1 - split0   │                     
    //  ├─────────────────────┤                     
    //  │    unit1 - split0   │                     
    //  └─────────────────────┘                     
                              

    const uint iters_per_split = DIV_UP(K, BK) / SPLIT; // number of BKs a split handles (at least)
    const uint last_iter_splits = DIV_UP(K, BK) % SPLIT;

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

    // move A and B to thread start for loading, this has nothing to do with warps
    A += blockIdx.y * BM * K + m_idx_a * K + k_idx_a + blockIdx.z * BK;
    B += blockIdx.x * BN + k_idx_b * N + n_idx_b + blockIdx.z * BK * N;
    // move tC to the warp start, tC is the temporary gmem to store splits results
    tC += ((blockIdx.y * BM + warp_m_offset + subtile_idx_m) * N  + blockIdx.x * BN + warp_n_offset + subtile_idx_n) * SPLIT;
    // move C to the warp start as well
    C += (blockIdx.y * BM + warp_m_offset + subtile_idx_m) * N  + blockIdx.x * BN + warp_n_offset + subtile_idx_n;

    // declare accumulators
    float acc[m_subtiles * n_subtiles * TM * TN] = {0.};

    // declare fragments
    float frag_a[m_subtiles * TM] = {0.};
    float frag_b[n_subtiles * TN] = {0.};

    
    // #pragma unroll
    for (uint it = 0; it < iters_per_split; ++it) {
        kernel11::gmem_to_smem<BM, BN, BK, lda_m_stride, ldb_k_stride>(A, B, M, N, K, smem_a + k_idx_a * BM + m_idx_a, smem_b + k_idx_b * BN + n_idx_b);
        __syncthreads();
        // compute the warp level matmul
        kernel11::warp_matmul<BM, BN, BK, WM, WN, TM, TN, WM_SUBTILE, WN_SUBTILE, m_subtiles, n_subtiles>(smem_a, smem_b, acc, frag_a, frag_b, warp_m_offset, subtile_idx_m, warp_n_offset, subtile_idx_n);
        A += BK * SPLIT;
        B += BK * SPLIT * N;
        __syncthreads();
    }

    if (last_iter_splits > 0 && blockIdx.z < last_iter_splits) { // process last iteration
        kernel11::gmem_to_smem<BM, BN, BK, lda_m_stride, ldb_k_stride>(A, B, M, N, K, smem_a + k_idx_a * BM + m_idx_a, smem_b + k_idx_b * BN + n_idx_b);
        __syncthreads();
        // compute the warp level matmul
        kernel11::warp_matmul<BM, BN, BK, WM, WN, TM, TN, WM_SUBTILE, WN_SUBTILE, m_subtiles, n_subtiles>(smem_a, smem_b, acc, frag_a, frag_b, warp_m_offset, subtile_idx_m, warp_n_offset, subtile_idx_n);
        __syncthreads();
    }

    // epilogue: reduce to (temporary) gmem
    for (uint i = 0; i < m_subtiles; ++i) {
        for (uint j = 0; j < n_subtiles; ++j) {
            // move C to the subtile start
            float *C_subtile = C + (i * WM_SUBTILE * N + j * WN_SUBTILE);
            float *tC_subtile = tC + (i * WM_SUBTILE * N + j * WN_SUBTILE) * SPLIT;
            // #pragma unroll
            for (uint m = 0; m < TM; m += 1) {
                // #pragma unroll
                for (uint n = 0; n < TN; n += 4) {
                    const int acc_offset = (i * TM + m) * n_subtiles * TN + j * TN + n;
                    if (blockIdx.z == 0) { // only the first block in that split should accumulate from original C matrix
                        float4 tmp = FETCH_FLOAT4(
                        C_subtile[m * N + n]);
                        tmp.x = alpha * acc[acc_offset] + beta * tmp.x;
                        tmp.y = alpha * acc[acc_offset + 1] + beta * tmp.y;
                        tmp.z = alpha * acc[acc_offset + 2] + beta * tmp.z;
                        tmp.w = alpha * acc[acc_offset + 3] + beta * tmp.w;
                        
                        tC_subtile[m * N * SPLIT + n * SPLIT] = tmp.x;
                        tC_subtile[m * N * SPLIT + (n + 1) * SPLIT] = tmp.y;
                        tC_subtile[m * N * SPLIT + (n + 2) * SPLIT] = tmp.z;
                        tC_subtile[m * N * SPLIT + (n + 3) * SPLIT] = tmp.w;
                    } else {
                        tC_subtile[m * N * SPLIT + n * SPLIT + blockIdx.z] = alpha * acc[acc_offset];
                        tC_subtile[m * N * SPLIT + (n + 1) * SPLIT + blockIdx.z] = alpha * acc[acc_offset + 1];
                        tC_subtile[m * N * SPLIT + (n + 2) * SPLIT + blockIdx.z] = alpha * acc[acc_offset + 2];
                        tC_subtile[m * N * SPLIT + (n + 3) * SPLIT + blockIdx.z] = alpha * acc[acc_offset + 3];
                    }
                }
            }
        }
    }
}


template <int SPLIT,
          int smem_elements,
          int stages,
          int reduction_iters>
__global__ void reduce_k(const int M, const int N, const int K, float* __restrict__ tC, float* __restrict__ C, const int block_iters) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block(); // data is loaded using block as a group
    auto tile = cg::tiled_partition<SPLIT>(block); // data is reduced using tile as a group

    extern __shared__ float smem[];
    uint smem_stage_offsets[stages];
    for (int s = 0; s < stages; ++s) smem_stage_offsets[s] = s * smem_elements * SPLIT;

    uint gmem_init_offset = blockIdx.x * smem_elements * SPLIT;
    uint gmem_stride = gridDim.x * smem_elements * SPLIT;
    uint smem_offset = tile.meta_group_rank() * SPLIT + tile.thread_rank();
    uint smem_stride = tile.meta_group_size() * SPLIT;

    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages
    > shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);

    for (uint reduce_iter = 0, fetch_iter = 0; reduce_iter < block_iters; ++reduce_iter) {
        for (; fetch_iter < block_iters && fetch_iter < (reduce_iter + stages); ++fetch_iter) {
            pipeline.producer_acquire();
            uint shared_idx = fetch_iter % stages;
            cuda::memcpy_async(block,
                               smem + smem_stage_offsets[shared_idx],
                               tC + gmem_init_offset + gmem_stride * fetch_iter,
                               sizeof(float) * smem_elements * SPLIT,
                               pipeline);
            pipeline.producer_commit();
        }
        pipeline.consumer_wait();
        float element[reduction_iters] = {0.0f};
        for (; smem_offset < smem_elements * SPLIT; smem_offset += smem_stride) {
            uint element_idx = smem_offset / smem_stride;
            element[element_idx] = smem[smem_offset];
            element[element_idx] = cg::reduce(tile, element[element_idx], cg::plus<float>());
            if (tile.thread_rank() == 0) {
                C[blockIdx.x * smem_elements + smem_offset / SPLIT] = element[element_idx]; // copy to global memory
            }
        }
        pipeline.consumer_release();
    }
}