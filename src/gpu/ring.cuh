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
__global__ void ringExpend(T *x, T *output,
        int data_len1, int data_len2, int degree) {
            //mode X^degree
    int COL = blockIdx.y*blockDim.y+threadIdx.y;
    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
    

    if (ROW < data_len1 && COL < data_len2) {
        output[degree*(ROW * data_len1 + COL)] = x[ROW * data_len1 + COL];
    }
}

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

template<typename T>
__global__ void mssOnline_1d(const T *m_x, const T *m_y, const T *r_x, T *r_y, T*rr, T*rz, T*output_1,
        int data_len1, int data_len2) {
    //ss : joint item
    int COL = blockIdx.y*blockDim.y+threadIdx.y;
    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
    if (ROW < data_len1 && COL < data_len2){
        output_1[ROW * data_len2 + COL] = m_x[ROW * data_len2 + COL] * r_y[ROW * data_len2 + COL] + m_y[ROW * data_len2 + COL]*r_x[ROW * data_len2 + COL] + rr[ROW * data_len2 + COL] - rz[ROW * data_len2 + COL];
    }
}

template<typename T>
__global__ void ringLineInterpolation(T *x, T *f0, T *f1, T *f2,
        int data_len, int degree) {
            //mode X^degree
    int COL = blockIdx.y*blockDim.y+threadIdx.y;
    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
    if (ROW < data_len/2 && COL < degree) {
        f0[ROW * degree + COL] = x[2*ROW * degree + COL];
        f1[ROW * degree + COL] = x[(2*ROW + 1) * degree + COL];
        f2[ROW * degree + COL] = 2*f0[ROW * degree + COL] - f1[ROW * degree + COL];
        //f2
    }
}

template<typename T>
__global__ void ringInterpolation_1d(T *delta, T *f1, T *f2, T *g1, T *g2, T *f, T *g, 
        int data_len, int degree) {
            //mode X^degree
    int COL = blockIdx.y*blockDim.y+threadIdx.y;
    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
    if (ROW < data_len && COL < degree) {
        for(int k = 0; k < COL; k++){
            int a_idx = ROW * degree + COL - k;
            int b_idx = ROW * degree + k;
            if(COL - k == 0){
                f[ROW * degree + COL] = delta[ COL - k] * f2[b_idx] - (delta[ COL - k] - 1) * f1[b_idx];
                g[ROW * degree + COL] = delta[ COL - k] * g2[b_idx] - (delta[ COL - k] - 1) * g1[b_idx];
            }else{
                f[ROW * degree + COL] = delta[ COL - k] * f2[b_idx] - delta[ COL - k] * f1[b_idx];
                g[ROW * degree + COL] = delta[ COL - k] * g2[b_idx] - delta[ COL - k] * g1[b_idx];
            }
        }
        //f2
    }
}

template<typename T>
__global__ void ringInterpolation_2d(T *delta, T *h1, T *h2, T *h3, T *h, 
        int data_len, int degree) {
            //mode X^degree
    int COL = blockIdx.y*blockDim.y+threadIdx.y;
    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
    T delt1, delt2, delt3;

    if (ROW < data_len && COL < degree) {
        for(int k = 0; k < COL; k++){
            for(int w = 0; w < k; w++){
                int a_idx = ROW * degree + k - w;
                int b_idx = ROW * degree + w;
                int c_idx = ROW * degree + COL - k;
                if(COL - k == 0){
                    delt1 = delta[ COL - k] - 1;
                }else{
                    delt1 = delta[ COL - k];
                }
                if(w == 0){
                    delt2 = delta[ w] - 2;
                    delt3 = 2 - delta[ w];
                }else{
                    delt2 = delta[ w];
                    delt3 = - delta[w];
                }
                h[ROW * degree + COL] = delt1 * delt2 * h1[a_idx] 
                    + (2 * delta[ COL - k]) * delt3 * h2[a_idx]
                    + delt1 * delta[ w] * h2[a_idx];
               
            }
        }
        //f2
    }
}
}

namespace gpu {

template<typename T, typename I>
void ringExpend(
        const DeviceData<T, I> *x, DeviceData<T, I> *output,
        size_t data_len1, size_t data_len2, size_t degree) {



    // printf("ringExpend: %dx%d\n", data_len1, data_len2);

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

    kernel::ringExpend<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&x->begin()[0]),
        thrust::raw_pointer_cast(&output->begin()[0]),
        data_len1, data_len2, degree
    );

    cudaThreadSynchronize();
}


template<typename T, typename I>
void ringMultiplication(
        const DeviceData<T, I> *a, const DeviceData<T, I> *b, DeviceData<T, I> *c,
        size_t data_len, size_t degree) {



    // printf("ringMultiplication: %dx%d\n", data_len, degree);

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



    // printf("ringDot: %dx%d\n", data_len, degree);

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



    // printf("mssOffline: %dx%d\n", data_len, degree);

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



    // printf("mssOnline: %dx%d\n", data_len, degree);

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

template<typename T, typename I>
void mssOnline_1d(
        const DeviceData<T, I> *m_x, const DeviceData<T, I> *m_y, const DeviceData<T, I> *r_x,  DeviceData<T, I> *r_y, DeviceData<T, I> *rr, DeviceData<T, I> *rz, DeviceData<T, I> *output_1,
        size_t data_len1, size_t data_len2) {



    // printf("mssonline_id: %dx%d\n", data_len1, data_len2);

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

    kernel::mssOnline_1d<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&m_y->begin()[0]),
        thrust::raw_pointer_cast(&m_y->begin()[0]),
        thrust::raw_pointer_cast(&r_x->begin()[0]),
        thrust::raw_pointer_cast(&r_y->begin()[0]),
        thrust::raw_pointer_cast(&rr->begin()[0]),
        thrust::raw_pointer_cast(&rz->begin()[0]),
        thrust::raw_pointer_cast(&output_1->begin()[0]),
        data_len1, data_len2
    );

    cudaThreadSynchronize();
}

template<typename T, typename I>
void ringLineInterpolation(
        const DeviceData<T, I> *x, const DeviceData<T, I> *f0, DeviceData<T, I> *f1, DeviceData<T, I> *f2,
        size_t data_len, size_t degree) {
    // printf("ringLineInterpolation: %dx%d\n", data_len, degree);

    dim3 threadsPerBlock(degree, data_len);
    dim3 blocksPerGrid(1, 1);

    if (degree > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(degree)/double(threadsPerBlock.x));
    }
    
    if (data_len > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.y = ceil(double(data_len/2)/double(threadsPerBlock.y));
    }

    //std::cout << "rows " << rows << " shared " << shared << " cols " << cols << std::endl;
    //std::cout << "grid x = " << blocksPerGrid.x << " y = " << blocksPerGrid.y << " threads x = " << threadsPerBlock.x << " y = " << threadsPerBlock.y << std::endl;

    kernel::ringLineInterpolation<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&x->begin()[0]),
        thrust::raw_pointer_cast(&f0->begin()[0]),
        thrust::raw_pointer_cast(&f1->begin()[0]),
        thrust::raw_pointer_cast(&f2->begin()[0]),
        data_len, degree
    );

    cudaThreadSynchronize();
}

template<typename T, typename I>
void ringInterpolation_1d(
        const DeviceData<T, I> *delta, const DeviceData<T, I> *f1, DeviceData<T, I> *f2, DeviceData<T, I> *g1, DeviceData<T, I> *g2,DeviceData<T, I> *f,DeviceData<T, I> *g,
        size_t data_len, size_t degree) {

    // printf("ringInterpolation_1d: %dx%d\n", data_len, degree);

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

    kernel::ringInterpolation_1d<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&delta->begin()[0]),
        thrust::raw_pointer_cast(&f1->begin()[0]),
        thrust::raw_pointer_cast(&f2->begin()[0]),
        thrust::raw_pointer_cast(&g1->begin()[0]),
        thrust::raw_pointer_cast(&g2->begin()[0]),
        thrust::raw_pointer_cast(&f->begin()[0]),
        thrust::raw_pointer_cast(&g->begin()[0]),
        data_len, degree
    );

    cudaThreadSynchronize();
}


template<typename T, typename I>
void ringInterpolation_2d(
        const DeviceData<T, I> *delta, const DeviceData<T, I> *h1, DeviceData<T, I> *h2, DeviceData<T, I> *h3, DeviceData<T, I> *h,
        size_t data_len, size_t degree) {

    // printf("ringInterpolation_2d: %dx%d\n", data_len, degree);

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

    kernel::ringInterpolation_2d<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&delta->begin()[0]),
        thrust::raw_pointer_cast(&h1->begin()[0]),
        thrust::raw_pointer_cast(&h2->begin()[0]),
        thrust::raw_pointer_cast(&h3->begin()[0]),
        thrust::raw_pointer_cast(&h->begin()[0]),
        data_len, degree
    );

    cudaThreadSynchronize();
}

}