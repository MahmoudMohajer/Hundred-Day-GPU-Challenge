#include <stdio.h>
#include <stdlib.h>

#define cudaCheckError(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, \
                                                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0);
    


__global__ void vectorAdd(const float *a, const float *b,float *c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x; 

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() { 
    const int size = 1024; 
    const int bytes = size * sizeof(float);
    float a_h[size];
    float b_h[size];
    float c_h[size];

    for (int i=0; i < size; i++) {
        a_h[i] = i * 1.23; 
        b_h[i] = i * 2.12;
    }

    // allocating memory for host arrays 
    float *a_d, *b_d, *c_d; 
    cudaCheckError(cudaMalloc((void**)&a_d, bytes));
    cudaCheckError(cudaMalloc((void**)&b_d, bytes));
    cudaCheckError(cudaMalloc((void**)&c_d, bytes));

    // copy to vram 
    cudaCheckError(cudaMemcpy(a_d, a_h, bytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(b_d, b_h, bytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int BlocksPerGrid = (size + threadsPerBlock -1) / threadsPerBlock;

    // launching the kernel 
    vectorAdd<<<BlocksPerGrid, threadsPerBlock>>>(a_d, b_d, c_d, size);
    cudaCheckError(cudaGetLastError());

    cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaMemcpy(c_h, c_d, bytes, cudaMemcpyDeviceToHost));

    cudaCheckError(cudaFree(a_d));
    cudaCheckError(cudaFree(b_d));
    cudaCheckError(cudaFree(c_d));

    for(int i = 0; i < size; i++) {
        if (c_h[i] != a_h[i] + b_h[i]){
            printf("The vector addition is wrong! index: %d a=%0.2f b=%0.2f c=%0.2f\n", i, a_h[i], b_h[i], c_h[i]);
            return 1;
        }
    }
    printf("The addition was done successfully\n");



}