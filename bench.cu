template <typename T>

__global__ void k(volatile T * __restrict__ d1, volatile T * __restrict__ d2, const int loops, const int ds){

  for (int i = 0; i < loops; i++)
    for (int j = threadIdx.x+blockDim.x*blockIdx.x; j < ds; j += gridDim.x*blockDim.x)
      if (i&1) d1[j] = d2[j];
      else d2[j] = d1[j];
}
const int dsize = 1048576*128;
const int iter = 64;
int main(){

  int *d;
  cudaMalloc(&d, dsize);
  // case 1: 32MB copy, should exceed L2 cache on V100
  int csize = 1048576*8;
  k<<<128, 1024>>>(d, d+csize, iter, csize);
  // case 2: 2MB copy, should fit in L2 cache on V100
  csize = 1048576/2;
  k<<<128, 1024>>>(d, d+csize, iter, csize);
  cudaDeviceSynchronize();
}