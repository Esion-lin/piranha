#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <thrust/device_vector.h>

#include "DeviceData.h"
#include "../globals.h"
#include "../util/util.cuh"

namespace kernel {

template<typename T>
__global__ void ringMultiplication(T *a, T *b, T *c,
        int data_len, int degree) {
            //mode X^degree
    int COL = blockIdx.y*blockDim.y+threadIdx.y;
    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
    

    if (ROW < data_len && COL < degree) {
        for(int k = 0; k < COL; k++){
            int a_idx = ROW * degree + COL - k;
            int b_idx = ROW * degree + k;
            c[ROW * degree + COL] += a[a_idx] * b[b_idx];
        }
    }
}
template<typename T>
__global__ void ringDotProduct(T *a, T *b, T *c,
        int data_len, int degree) {
            //mode X^degree
    int COL = blockIdx.y*blockDim.y+threadIdx.y;
    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
    if (ROW < data_len && COL < degree) {
        for(int k = 0; k < COL; k++){
            int a_idx = ROW * degree + COL - k;
            int b_idx = ROW * degree + k;
            c[COL] += a[a_idx] * b[b_idx];
        }
    }
}
template<typename T>
__global__ void mssOffline(T *r_x, T *r_y, T *r_a, T*gamma, T*ss,
        int data_len, int degree) {
            //mode X^degree
    int COL = blockIdx.y*blockDim.y+threadIdx.y;
    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
    int bias = data_len * degree;
    if (ROW < data_len && COL < degree) {
        for(int k = 0; k < COL; k++){
            for(int w = 0; w < k; w++){
                int a_idx = ROW * degree + k - w;
                int b_idx = ROW * degree + w;
                int c_idx = ROW * degree + COL - k;
                gamma[COL] += r_x[a_idx] * r_y[b_idx] * r_a[c_idx];
                
            }
            int a_idx = ROW * degree + COL - k;
            int b_idx = ROW * degree + k;
            ss[ROW * degree + COL] += r_x[a_idx] * r_y[b_idx];
            ss[bias + ROW * degree + COL] += r_x[a_idx] * r_a[b_idx];
            ss[2*bias + ROW * degree + COL] += r_a[a_idx] * r_y[b_idx];

        }
    }
}
template<typename T>
__global__ void mssOnline(const T *ss, const T *m_x, const T *m_y, T *m_a, T*gamma, T*output,
        int data_len, int degree) {
    //ss : joint item
    int COL = blockIdx.y*blockDim.y+threadIdx.y;
    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
    int bias = data_len * degree;
    if (ROW < data_len && COL < degree) {
        for(int k = 0; k < COL; k++){
            
            int a_idx = ROW * degree + COL - k;
            int b_idx = ROW * degree + k;
            output[COL] += m_x[a_idx] * m_y[b_idx] + ss[a_idx] * m_a[b_idx] + ss[bias+a_idx] * m_y[b_idx] + ss[2*bias+a_idx] * m_x[b_idx];

        }
    }
}

// template<typename T>
// __global__ void ringInterpolation(T *a, T *b, T *c, T * delta,
//         int data_len, int degree) {
//             //mode X^degree
//     int COL = blockIdx.y*blockDim.y+threadIdx.y;
//     int ROW = blockIdx.x*blockDim.x+threadIdx.x;
//     if (ROW < data_len && COL < degree) {
//         for(int k = 0; k < COL; k++){
//             int a_idx = ROW * degree + COL - k;
//             int b_idx = ROW * degree + k;
//             c[COL] += a[a_idx] * b[b_idx];
//         }
//     }
// }

}

namespace gpu {
template<typename T, typename I>
void ringMultiplication(
        const DeviceData<T, I> *a, const DeviceData<T, I> *b, DeviceData<T, I> *c,
        size_t data_len, size_t degree) {



    printf("ringMultiplication: %dx%d\n", data_len, degree);

    dim3 threadsPerBlock(degree, data_len);
    dim3 blocksPerGrid(1, 1);

    if (degree > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(degree)/double(threadsPerBlock.x));
    }
    
    if (data_len > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.y = ceil(double(data_len)/double(threadsPerBlock.y));
    }

    //std::cout << "rows " << rows << " shared " << shared << " cols " << cols << std::endl;
    //std::cout << "grid x = " << blocksPerGrid.x << " y = " << blocksPerGrid.y << " threads x = " << threadsPerBlock.x << " y = " << threadsPerBlock.y << std::endl;

    kernel::ringMultiplication<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&a->begin()[0]),
        thrust::raw_pointer_cast(&b->begin()[0]),
        thrust::raw_pointer_cast(&c->begin()[0]),
        data_len, degree
    );

    cudaThreadSynchronize();
}
template<typename T, typename I>
void ringDot(
        const DeviceData<T, I> *a, const DeviceData<T, I> *b, DeviceData<T, I> *c,
        size_t data_len, size_t degree) {



    printf("ringDot: %dx%d\n", data_len, degree);

    dim3 threadsPerBlock(degree, data_len);
    dim3 blocksPerGrid(1, 1);

    if (degree > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(degree)/double(threadsPerBlock.x));
    }
    
    if (data_len > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.y = ceil(double(data_len)/double(threadsPerBlock.y));
    }

    //std::cout << "rows " << rows << " shared " << shared << " cols " << cols << std::endl;
    //std::cout << "grid x = " << blocksPerGrid.x << " y = " << blocksPerGrid.y << " threads x = " << threadsPerBlock.x << " y = " << threadsPerBlock.y << std::endl;

    kernel::ringMultiplication<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&a->begin()[0]),
        thrust::raw_pointer_cast(&b->begin()[0]),
        thrust::raw_pointer_cast(&c->begin()[0]),
        data_len, degree
    );

    cudaThreadSynchronize();
}

template<typename T, typename I>
void mssOffline(
        const DeviceData<T, I> *r_x, const DeviceData<T, I> *r_y, DeviceData<T, I> *r_a, DeviceData<T, I> *gamma, DeviceData<T, I> *ss,
        size_t data_len, size_t degree) {



    printf("mssOffline: %dx%d\n", data_len, degree);

    dim3 threadsPerBlock(degree, data_len);
    dim3 blocksPerGrid(1, 1);

    if (degree > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(degree)/double(threadsPerBlock.x));
    }
    
    if (data_len > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.y = ceil(double(data_len)/double(threadsPerBlock.y));
    }

    //std::cout << "rows " << rows << " shared " << shared << " cols " << cols << std::endl;
    //std::cout << "grid x = " << blocksPerGrid.x << " y = " << blocksPerGrid.y << " threads x = " << threadsPerBlock.x << " y = " << threadsPerBlock.y << std::endl;

    kernel::mssOffline<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&r_x->begin()[0]),
        thrust::raw_pointer_cast(&r_y->begin()[0]),
        thrust::raw_pointer_cast(&r_a->begin()[0]),
        thrust::raw_pointer_cast(&gamma->begin()[0]),
        thrust::raw_pointer_cast(&ss->begin()[0]),
        data_len, degree
    );

    cudaThreadSynchronize();
}
template<typename T, typename I>
void mssOnline(
        const DeviceData<T, I> *ss, const DeviceData<T, I> *m_x, const DeviceData<T, I> *m_y,  DeviceData<T, I> *m_a, DeviceData<T, I> *gamma, DeviceData<T, I> *output,
        size_t data_len, size_t degree) {



    printf("mssOffline: %dx%d\n", data_len, degree);

    dim3 threadsPerBlock(degree, data_len);
    dim3 blocksPerGrid(1, 1);

    if (degree > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(degree)/double(threadsPerBlock.x));
    }
    
    if (data_len > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.y = ceil(double(data_len)/double(threadsPerBlock.y));
    }

    //std::cout << "rows " << rows << " shared " << shared << " cols " << cols << std::endl;
    //std::cout << "grid x = " << blocksPerGrid.x << " y = " << blocksPerGrid.y << " threads x = " << threadsPerBlock.x << " y = " << threadsPerBlock.y << std::endl;

    kernel::mssOnline<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&ss->begin()[0]),
        thrust::raw_pointer_cast(&m_x->begin()[0]),
        thrust::raw_pointer_cast(&m_y->begin()[0]),
        thrust::raw_pointer_cast(&m_a->begin()[0]),
        thrust::raw_pointer_cast(&gamma->begin()[0]),
        thrust::raw_pointer_cast(&output->begin()[0]),
        data_len, degree
    );

    cudaThreadSynchronize();
}
}