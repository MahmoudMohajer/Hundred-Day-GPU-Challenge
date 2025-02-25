#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \
    do \
    { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "Error in %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
} while (0);

__global__
void sumArray(float *array, float *result, int n) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n) {
       temp[tid] = array[idx];   
    } else {
        temp[tid] = 0.0f;
    }

    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride){
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = temp[0];
    }
}

int main() {
    int n = 1 << 28; 
    int bytes = n * sizeof(float); 

    float *h_a = (float*)malloc(bytes);
    float h_r; 
    for(int i=0; i < n; i++) h_a[i] = 1.0f;

    float *d_a, *d_r; 
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_r, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    int block_size = 256;
    int shared_memory = block_size * sizeof(float);
    dim3 block(block_size);

    int current_size = n;
    float *input = d_a;
    float *output = d_r;

    while (current_size > 1) {
        int num_blocks = (current_size + block_size - 1) / block_size;
        dim3 grid(num_blocks); 
        sumArray<<<grid, block, shared_memory>>>(input, output, current_size);
        CUDA_CHECK(cudaGetLastError());

        float *temp = input;
        input = output;
        output = temp;

        current_size = num_blocks;
    }


    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize()); 

    CUDA_CHECK(cudaMemcpy(&h_r, input, sizeof(float), cudaMemcpyDeviceToHost));

    printf("Result of sum reduction: %f\n", h_r);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_r));
    free(h_a);
}
    