#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__
void vecElementwiseMul(const float *A, const float *B, float *C, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if (idx < n) {
        C[idx] = A[idx] * B[idx];
    }
}


int main() {
    int size = 102400; 
    int bytes = size * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i=0; i < size; ++i){
        h_A[i] = 3.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C; 
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(256);
    int numBlocks = (size + block.x - 1) / block.x;
    dim3 grid(numBlocks);


    vecElementwiseMul<<<grid, block>>>(d_A, d_B, d_C, size);
    
    cudaEvent_t start, stop; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    vecElementwiseMul<<<grid, block>>>(d_A, d_B, d_C, size);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop); 
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    for (int i=0; i < size; ++i){
        if (h_C[i] != 6.0f){
            printf("Error in computation of the issue in C[%d]= %f\n", i, h_C[i]);
            break;
        }
    }

    printf("Time taken: %f ms\n", time);
     

}