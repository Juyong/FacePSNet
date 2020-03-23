#ifndef CUDA_CONFIG_H_
#define CUDA_CONFIG_H_

#include <cuda.h>
#include <cuda_runtime.h>

// CUDA: use 1024 threads per block
const int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N)
{
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                                   \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
         i += blockDim.x * gridDim.x)

#endif // CUDA_CONFIG_H_