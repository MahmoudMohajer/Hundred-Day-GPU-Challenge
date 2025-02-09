#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define HEIGHT 1024 
#define WIDTH 1024

__global__
void matrixAdd(const float *A, const float *B, float *C, int width, int height) { 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y; 

    if (row < height && col < width) {
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int size = HEIGHT * WIDTH; 
    int bytes = size * sizeof(float);

    float *A_h = (float*)malloc(bytes);
    float *B_h = (float*)malloc(bytes);
    float *C_h = (float*)malloc(bytes);

    for (int i=0; i < size; i++) {
        A_h[i] = 1.0f;
        B_h[i] = 2.0f;
    }

    float *A_d, *B_d, *C_d; 
    cudaMalloc((void**)&A_d, bytes);
    cudaMalloc((void**)&B_d, bytes);
    cudaMalloc((void**)&C_d, bytes);

    cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice);

    int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y -1) / block.y); 

    matrixAdd<<<grid, block>>>(A_d, B_d, C_d, WIDTH, HEIGHT);

    cudaDeviceSynchronize(); 

    cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost);

    for (int i=0; i < size; i++){
        if (C_h[i] != 3.0f) {
            printf("Error at index %d: %f\n", i, C_h[i]);
            break;
        }
    }
    printf("Matrix addtion was done successfully!\n");

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    free(A_h);
    free(B_h);
    free(C_h);

}