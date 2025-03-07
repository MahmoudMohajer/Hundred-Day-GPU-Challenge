#include <stdio.h>
#include <stdlib.h>

__global__ 
void naiveMatrixMul(const float *A, const float *B, float *C, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < width && col < width) {
        float acc = 0; 
        for (int k=0; k < width; k ++) {
            acc += A[row * k + col] * B[k * width + col];
        }
        C[row * width + col] = acc;
    }
}

int main() {
    int width = 16; 
    int size = width * width; 
    int bytes = size * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    for (int i = 0; i < size; i++)
    {
        h_a[i] = 3.2f;
        h_b[i] = 1.2f;
    }

    float *d_a, *d_b, *d_c; 
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice); 

    int BLOCK_SIZE = 16;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (width + threadsPerBlock.y - 1) / threadsPerBlock.y); 
    naiveMatrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, width);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost); 

    for (int i = 0 ; i < 10; i++){
        printf("%0.2f\t", h_c[i]);
    }
    printf("\n");
    
}