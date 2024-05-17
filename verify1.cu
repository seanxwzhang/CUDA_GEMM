#include <iostream>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda/pipeline>

void cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
};

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

namespace cg = cooperative_groups;

template <int SPLIT,
          int smem_elements,
          int stages,
          int reduction_iters>
__global__ void reduce_k(const int M, const int N, float* __restrict__ tC, float* __restrict__ C, const int block_iters) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block(); // data is loaded using block as a group
    auto tile = cg::tiled_partition<SPLIT>(block); // data is reduced using tile as a group

    extern __shared__ float smem[];
    uint smem_stage_offsets[stages];
    float sum[reduction_iters] = {0.0f};
    for (int s = 0; s < stages; ++s) smem_stage_offsets[s] = s * smem_elements * SPLIT;

    uint gmem_init_offset = blockIdx.x * smem_elements * SPLIT;
    uint gmem_stride = gridDim.x * smem_elements * SPLIT;
    uint smem_stride = tile.meta_group_size() * SPLIT;

    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages
    > shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);

    for (uint reduce_iter = 0, fetch_iter = 0; reduce_iter < block_iters; ++reduce_iter) {
        uint smem_offset = tile.meta_group_rank() * SPLIT + tile.thread_rank();
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
        for (; smem_offset < smem_elements * SPLIT; smem_offset += smem_stride) {
            uint element_idx = smem_offset / smem_stride;
            sum[element_idx] = smem[smem_offset];
            sum[element_idx] = cg::reduce(tile, sum[element_idx], cg::plus<float>());
            if (tile.thread_rank() == 0) {
                uint output_offset = blockIdx.x * smem_elements + gridDim.x * smem_elements * reduce_iter + smem_offset / SPLIT;
                C[output_offset] = sum[element_idx]; // copy to global memory
            }
        }
        __syncthreads();
        pipeline.consumer_release();
    }
}


int main() {
    constexpr int N = 4096;
    constexpr int M = 4096;
    constexpr int SPLIT = 32;
    constexpr int blockSize = 512;
    constexpr int gridSize = 512;
    constexpr int smem_elements = 128;
    constexpr int iterations = M * N / gridSize / smem_elements;
    constexpr int stages = 2;
    constexpr int smem_size = smem_elements * SPLIT * sizeof(float) * stages;
    constexpr int group_num = blockSize / SPLIT;
    constexpr int reduction_iters = smem_elements / group_num;

    float * mat = (float *) malloc(M * N * SPLIT * sizeof(float));
    float * output = (float *) malloc(M * N * sizeof(float));
    for (int i = 0; i < M * N * SPLIT; i++) {
        if (i % SPLIT == 0) {
            mat[i] = 0.5;
        } else {
            mat[i] = 1.0;
        }
    }

    float *dmat = nullptr, *doutput = nullptr;
    cudaCheck(cudaMalloc(&dmat, M * N * SPLIT * sizeof(float)));
    cudaCheck(cudaMalloc(&doutput, M * N * sizeof(float)));
    cudaCheck(cudaMemcpy(dmat, mat, M * N * SPLIT * sizeof(float), cudaMemcpyHostToDevice));

    dim3 reduceGrid(gridSize);
    dim3 reduceBlock(blockSize);
    reduce_k<SPLIT, smem_elements, stages, reduction_iters><<<reduceGrid, reduceBlock, smem_size>>>(M, N, dmat, doutput, iterations);

    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(output, doutput, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; i++) {
        if (output[i] != (float)SPLIT - 0.5) {
            std::cout << "Error at " << i << ": expected " << (float)SPLIT - 0.5 << ", received "<< output[i] << std::endl;
            break;
        }
    }

    free(mat);
    free(output);
    cudaFree(dmat);
    cudaFree(doutput);
    return 0;
}