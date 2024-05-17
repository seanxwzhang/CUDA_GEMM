#include <iostream>

// Device code

__global__ void MyKernel(int *d, int *a, int *b)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d[idx] = a[idx] * b[idx];
}

// Host code
int main()
{
    int numBlocks;        // Occupancy in terms of active blocks
    int blockSize = 32;

    // These variables are used to convert occupancy to warps
    int device;
    cudaDeviceProp prop;
    int activeWarps;
    int maxWarps;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
        MyKernel,
        blockSize,
        0);

    activeWarps = numBlocks * blockSize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    std::cout << "MaxWarps: " << maxWarps << std::endl;
    std::cout << "prop.maxThreadsPerBlock: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "prop.sharedMemPerBlock: " << prop.sharedMemPerBlock << std::endl;
    std::cout << "prop.multiProcessorCount: " << prop.multiProcessorCount << std::endl;
    std::cout << "prop.memoryClockRate: " << prop.memoryClockRate << std::endl;
    std::cout << "prop.memoryBusWidth: " << prop.memoryBusWidth << std::endl;
    std::cout << "prop.l2CacheSize: " << prop.l2CacheSize << std::endl;
    std::cout << "prop.maxThreadsPerMultiProcessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "prop.sharedMemPerMultiprocessor: " << prop.sharedMemPerMultiprocessor << std::endl;
    std::cout << "prop.maxBlocksPerMultiProcessor: " << prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;

    return 0;
}