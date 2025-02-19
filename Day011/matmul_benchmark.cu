#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__
void matrixMul(const float *A, const float *B, float *C, int A_height, int B_width, int common_dim) {
    int col = blockIdx.x * blockDim.x + threadIdx.x ;
    int row = blockIdx.y * blockDim.y + threadIdx.y ;

    if (row < A_height && col < B_width) {
        float acc = 0.0f;
        for (int k=0; k < common_dim; ++k) {
            acc += A[row * common_dim + k] * B[k * B_width + col];
        }
        C[row * B_width + col] = acc;
    } 
}

int main() {
    // Matrix dimensions
    const int A_height = 1024;
    const int A_width = 1024;  // This is also B_height
    const int B_width = 1024;
    
    // Allocate host memory
    float *h_A = (float*)malloc(A_height * A_width * sizeof(float));
    float *h_B = (float*)malloc(A_width * B_width * sizeof(float));
    float *h_C = (float*)malloc(A_height * B_width * sizeof(float));
    
    // Initialize matrices with random values
    for (int i = 0; i < A_height * A_width; i++) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < A_width * B_width; i++) h_B[i] = rand() / (float)RAND_MAX;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A_height * A_width * sizeof(float));
    cudaMalloc(&d_B, A_width * B_width * sizeof(float));
    cudaMalloc(&d_C, A_height * B_width * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, A_height * A_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, A_width * B_width * sizeof(float), cudaMemcpyHostToDevice);
    
    // Test different block sizes
    dim3 blockSizes[] = {
        dim3(8, 8),
        dim3(16, 16),
        dim3(32, 32)
    };
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("Matrix multiplication benchmark (%dx%d * %dx%d):\n", 
           A_height, A_width, A_width, B_width);
    
    for (const auto& block : blockSizes) {
        // Calculate grid dimensions
        dim3 grid(
            (B_width + block.x - 1) / block.x,
            (A_height + block.y - 1) / block.y
        );
        
        // Warm-up run
        matrixMul<<<grid, block>>>(d_A, d_B, d_C, A_height, B_width, A_width);
        
        // Timing run
        cudaEventRecord(start);
        
        // Run kernel multiple times for better timing accuracy
        for (int i = 0; i < 10; i++) {
            matrixMul<<<grid, block>>>(d_A, d_B, d_C, A_height, B_width, A_width);
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        printf("Block size: %dx%d, Grid size: %dx%d, Average time: %.3f ms\n",
               block.x, block.y, grid.x, grid.y, milliseconds / 10.0f);
    }
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}