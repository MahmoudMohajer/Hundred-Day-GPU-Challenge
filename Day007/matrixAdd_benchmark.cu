#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define WIDTH 10240
#define HEIGHT 10240

__device__
int find_index(int row, int col, int width) {
    return row * width + col;
}

__global__
void matrixAdd(const float *A, const float *B, float *C, int width, int height){
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 

    if (row < height && col < width){
        int idx = find_index(row, col, width);
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int size = WIDTH * HEIGHT;
    int bytes = size * sizeof(float);

    float *A_h = (float*)malloc(bytes);
    float *B_h = (float*)malloc(bytes);
    float *C_h = (float*)malloc(bytes);

    for (int i=0; i < size; i++){
        A_h[i] = 1.0f;
        B_h[i] = 2.0f;
    }

    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, bytes);
    cudaMalloc((void**)&B_d, bytes);
    cudaMalloc((void**)&C_d, bytes);

    cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice);

    // Experiment with different block sizes
    int BLOCK_SIZES[] = {8, 16, 32};
    int num_block_sizes = sizeof(BLOCK_SIZES) / sizeof(BLOCK_SIZES[0]);

    for (int i = 0; i < num_block_sizes; i++) {
        int BLOCK_SIZE = BLOCK_SIZES[i];
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Record the start event
        cudaEventRecord(start);

        // Launch the kernel
        matrixAdd<<<grid, block>>>(A_d, B_d, C_d, WIDTH, HEIGHT);

        // Record the stop event
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Calculate the elapsed time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Check for errors in kernel launch
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch error: %s\n", cudaGetErrorString(err));
            continue;
        }

        cudaDeviceSynchronize();

        cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost);

        // Verify correctness
        int correct = 1;
        for (int i=0; i < size; i++){
            if(C_h[i] != 3.0f){
                correct = 0;
                break;
            }
        }

        if (correct) {
            printf("Matrix Addition with BLOCK_SIZE %d was done successfully in %0.2f ms\n", BLOCK_SIZE, milliseconds);
        } else {
            printf("Matrix Addition with BLOCK_SIZE %d failed\n", BLOCK_SIZE);
        }

        // Destroy CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}