#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h> 
#include "cuda_utils.h"

__global__ void vectorAdd(const float *a, const float *b,float *c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x; 

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() { 
    const int size = 500000000;  
    const int bytes = size * sizeof(float); // around 0.5GB for each array 
    float *a_h = (float*)malloc(bytes);
    float *b_h = (float*)malloc(bytes);
    float *c_h = (float*)malloc(bytes);

    for (int i=0; i < size; i++) {
        a_h[i] = i * 1.23; 
        b_h[i] = i * 2.12;
    }

    // allocating memory for host arrays 
    float *a_d, *b_d, *c_d; 
    cudaCheckError(cudaMalloc((void**)&a_d, bytes));
    cudaCheckError(cudaMalloc((void**)&b_d, bytes));
    cudaCheckError(cudaMalloc((void**)&c_d, bytes));

    cudaEvent_t start, stop, startKernel, stopKernel; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&startKernel);
    cudaEventCreate(&stopKernel);
    
    // copy to vram 
    cudaEventRecord(start, 0);
    cudaCheckError(cudaMemcpy(a_d, a_h, bytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(b_d, b_h, bytes, cudaMemcpyHostToDevice));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 
    float memcpyHtoDTime = 0;
    cudaEventElapsedTime(&memcpyHtoDTime, start, stop);
    printf("Host to Device memcpy time: %f ms\n", memcpyHtoDTime);

    int threadsPerBlock = 256;
    int BlocksPerGrid = (size + threadsPerBlock -1) / threadsPerBlock;

    // launching the kernel 
    cudaEventRecord(startKernel, 0);
    vectorAdd<<<BlocksPerGrid, threadsPerBlock>>>(a_d, b_d, c_d, size);
    cudaEventRecord(stopKernel, 0);
    cudaEventSynchronize(stopKernel);
    float kernel_time = 0; 
    cudaEventElapsedTime(&kernel_time, startKernel, stopKernel);
    printf("Kernel operation time: %f ms\n", kernel_time);
    cudaCheckError(cudaGetLastError());

    cudaCheckError(cudaDeviceSynchronize());
    
    cudaEventRecord(start, 0);
    cudaCheckError(cudaMemcpy(c_h, c_d, bytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float memcpyDtoHTime = 0;
    cudaEventElapsedTime(&memcpyDtoHTime, start, stop); 
    printf("Elapsed time copying from Device to Host: %f ms\n", memcpyDtoHTime);

    
    
    printf("The addition was done successfully\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(startKernel);
    cudaEventDestroy(stopKernel);
    cudaCheckError(cudaFree(a_d));
    cudaCheckError(cudaFree(b_d));
    cudaCheckError(cudaFree(c_d));

    free(a_h);
    free(b_h);
    free(c_h);


}