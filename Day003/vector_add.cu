#include <stdio.h>
#include <stdlib.h>

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
    cudaMalloc((void**)&a_d, bytes);
    cudaMalloc((void**)&b_d, bytes);
    cudaMalloc((void**)&c_d, bytes);

    // copy to vram 
    cudaMemcpy(a_d, a_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int BlocksPerGrid = (size + threadsPerBlock -1) / threadsPerBlock;

    // launching the kernel 
    vectorAdd<<<BlocksPerGrid, threadsPerBlock>>>(a_d, b_d, c_d, size);

    cudaDeviceSynchronize();

    cudaMemcpy(c_h, c_d, bytes, cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    for(int i = 0; i < size; i++) {
        if (c_h[i] != a_h[i] + b_h[i]){
            printf("The vector addition is wrong! index: %d a=%0.2f b=%0.2f c=%0.2f\n", i, a_h[i], b_h[i], c_h[i]);
            return 1;
        }
    }
    printf("The addition was done successfully\n");



}