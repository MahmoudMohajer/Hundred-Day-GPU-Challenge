#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)\
    do \
    { \
        cudaError_t error = call; \
        if (error != cudaSuccess) \
        { \
            fprintf(stderr, "Error at %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
} while (0);

__global__ 
void sumArray(float *input, float *result, int n) {
    extern __shared__ float sdata[]; 
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    sdata[tid] = input[idx] * (idx < n); 

    __syncthreads(); 
    for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
        int offset = (tid < stride) * stride; 
        sdata[tid] +=  sdata[tid + offset] * (offset != 0);

        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

int main() {
    int n = 1 << 28; 
    int bytes = n * sizeof(float); 

    float *h_input = (float*)malloc(bytes); 
    float h_result;

    for (int i=0; i < n; i++) h_input[i] = 1.0f;

    float *d_input, *d_result; 
    CUDA_CHECK(cudaMalloc((void**)&d_input, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_result, bytes));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    int block_size = 256; 
    int current_size = n; 
    int shared_memory = block_size * sizeof(float);
    float *input = d_input;
    float *result = d_result;
    
    while (current_size != 1) {
        int num_blocks = (current_size + block_size - 1) / block_size;
        dim3 grid(num_blocks);
        sumArray<<<grid, block_size, shared_memory>>>(input, result, current_size);
        CUDA_CHECK(cudaGetLastError());

        float *temp = input;
        input = result; 
        result = temp; 

        current_size = num_blocks;
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_result, input, sizeof(float), cudaMemcpyDeviceToHost));

    printf("the Sum is %f \n", h_result);

    return 0;

}