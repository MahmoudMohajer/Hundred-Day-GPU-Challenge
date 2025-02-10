#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define WIDTH 1024
#define HEIGHT 1024 

__device__
int index_finder(int row, int col, int width) {
    return row * width + col;
}

__global__
void MatrixAdd(const float *A, const float *B, float *C, int width, int height) { 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height){
        int idx = index_finder(row, col, width);
        C[idx] = A[idx] + B[idx];
    }
}

int main() { 
    int size = WIDTH * HEIGHT; 
    int bytes = size * sizeof(float);

    float *A_h = (float*)malloc(bytes);
    float *B_h = (float*)malloc(bytes);
    float *C_h = (float*)malloc(bytes);

    for (int i = 0; i < size; i++) {
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
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

    MatrixAdd<<<grid, block>>>(A_d, B_d, C_d, WIDTH, HEIGHT); 

    cudaDeviceSynchronize();

    cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++)
    {
        if (C_h[i] != 3.0f)
        {
            printf("Error in computation of matrix addtion at %d = %f\n", i, C_h[i]);
            return 1;
        }
        
        
    }
    printf("Matrix addtion was done successfully\n");

    
}