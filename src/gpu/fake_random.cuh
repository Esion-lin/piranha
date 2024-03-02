#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <curand.h>
#include <curand_kernel.h>
#include "DeviceData.h"
#include "../globals.h"
#include "../util/util.cuh"



namespace kernel {
__device__ curandState globalState;

// 初始化全局随机数生成器状态

__global__ void initializeRandomGenerator() {
    curand_init(8989, 0, 0, &globalState);
}
template<typename T>
__global__ void generateRandomNumbers(T *randomNumbers, int data_len1, int data_len2) {
    int COL = blockIdx.y*blockDim.y+threadIdx.y;
    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
    if (ROW < data_len2 && COL < data_len1){
        randomNumbers[ROW * data_len1 + COL] = curand(&globalState);
    }
    // 使用全局状态生成随机数
}
}
namespace gpu {
template<typename T, typename I>
void generateRandomNumbers(
        DeviceData<T, I> *randomNumbers,
        size_t data_len1, size_t data_len2) {



    // printf("ringMultiplication: %dx%d\n", data_len, degree);

    dim3 threadsPerBlock(data_len2, data_len1);
    dim3 blocksPerGrid(1, 1);

    if (data_len2 > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(data_len2)/double(threadsPerBlock.x));
    }
    
    if (data_len1 > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.y = ceil(double(data_len1)/double(threadsPerBlock.y));
    }

    //std::cout << "rows " << rows << " shared " << shared << " cols " << cols << std::endl;
    //std::cout << "grid x = " << blocksPerGrid.x << " y = " << blocksPerGrid.y << " threads x = " << threadsPerBlock.x << " y = " << threadsPerBlock.y << std::endl;

    kernel::generateRandomNumbers<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&randomNumbers->begin()[0]),
        data_len1, data_len2
    );

    cudaThreadSynchronize();
}
}