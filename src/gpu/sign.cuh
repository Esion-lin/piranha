#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <curand.h>
#include "DeviceData.h"
#include "../globals.h"
#include "../util/util.cuh"

namespace kernel {

template<typename T>
__global__ void bit_operator(T *r_1, T *r_2, uint8_t *bits, 
        int data_len1, int bitsize) {
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;
    
    T highestbit = (((T)1)<<(sizeof(T)*8 - 1));
    T nhighestbit = ~(((T)1)<<(sizeof(T)*8 - 1));
    if (ROW < data_len1 && COL < bitsize) {
        bits[ROW * bitsize + COL] = ((((highestbit - 1 - ((- r_1[ROW] - r_2[ROW])&nhighestbit))<<1) + 1)>>COL) & 1;
    }
}
template<typename T, uint32_t modp, uint8_t select>
__global__ void bit_operator_for_m(uint8_t *rb, T *m, uint8_t *bits, 
        int data_len1, int bitsize) {
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;
    uint8_t* prex_sum = (uint8_t*)malloc(data_len1*bitsize);
    T nhighestbit = ~(((T)1)<<(sizeof(T)*8 - 1));
    if (ROW < data_len1 && COL < bitsize) {
        bits[ROW * bitsize + COL] = (rb[ROW * bitsize + COL] + modp - 2*((((m[ROW]&nhighestbit)<<1)>>COL) & 1)*rb[ROW * bitsize + COL] + ((((m[ROW]&nhighestbit)<<1)>>COL) & 1)*select) % modp;
        for(int k = 0; k < COL; k++){
            prex_sum[ROW * bitsize + COL] += bits[ROW * bitsize + COL] 
        }
    }
}

template<typename T>
__global__ void msb(T *r_1, T *r_2, uint8_t *msb, 
        int data_len1, int data_len2) {
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < data_len1 && COL < data_len2) {
        msb[ROW * data_len2 + COL] = ((- r_1[ROW * data_len2 + COL] - r_2[ROW * data_len2 + COL]) >> (sizeof(T)*8 - 1)) & 1;
    }

}

template<typename T>
__global__ void calculate_gamma(T *z_p, uint8_t *delta, T *r_z, T *gamma,
        int data_len1, int data_len2) {
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < data_len1 && COL < data_len2) {
        //if P2, += delta
        gamma[ROW * data_len2 + COL] = z_p[ROW * data_len2 + COL] - 2*z_p[ROW * data_len2 + COL]*delta[ROW * data_len2 + COL] + r_z[ROW * data_len2 + COL] + delta[ROW * data_len2 + COL];
    }

}

}
namespace gpu {
template<typename T, typename I>
void bit_operator(
        const DeviceData<T, I> *r_1, const DeviceData<T, I> *r_2, DeviceData<uint8_t> *bits,
        size_t data_len, size_t bitsize) {



    // printf("ringMultiplication: %dx%d\n", data_len, degree);

    dim3 threadsPerBlock(bitsize, data_len);
    dim3 blocksPerGrid(1, 1);

    if (bitsize > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(bitsize)/double(threadsPerBlock.x));
    }
    
    if (data_len > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.y = ceil(double(data_len)/double(threadsPerBlock.y));
    }

    //std::cout << "rows " << rows << " shared " << shared << " cols " << cols << std::endl;
    //std::cout << "grid x = " << blocksPerGrid.x << " y = " << blocksPerGrid.y << " threads x = " << threadsPerBlock.x << " y = " << threadsPerBlock.y << std::endl;

    kernel::bit_operator<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&r_1->begin()[0]),
        thrust::raw_pointer_cast(&r_2->begin()[0]),
        thrust::raw_pointer_cast(&bits->begin()[0]),
        data_len, bitsize
    );

    cudaThreadSynchronize();
}


template<typename T, typename I>
void msb(
        const DeviceData<T, I> *r_1, const DeviceData<T, I> *r_2, DeviceData<uint8_t> *bits,
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

    kernel::msb<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&r_1->begin()[0]),
        thrust::raw_pointer_cast(&r_2->begin()[0]),
        thrust::raw_pointer_cast(&bits->begin()[0]),
        data_len1, data_len2
    );

    cudaThreadSynchronize();
}

template<typename T, typename I>
void calculate_gamma(
        const DeviceData<T, I> *z_p, const DeviceData<T, I> *delta, DeviceData<uint8_t> *r_z, DeviceData<uint8_t> *gamma,
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

    kernel::calculate_gamma<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&z_p->begin()[0]),
        thrust::raw_pointer_cast(&delta->begin()[0]),
        thrust::raw_pointer_cast(&r_z->begin()[0]),
        thrust::raw_pointer_cast(&gamma->begin()[0]),
        data_len1, data_len2
    );

    cudaThreadSynchronize();
}
}