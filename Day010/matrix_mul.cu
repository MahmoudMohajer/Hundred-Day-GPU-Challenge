#include <stdio.h>
#include <stdlib.h>

__global__
void matrixMul(const float *A, const float *B, float *C, int dim) { //assuiming square matrices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < dim && row < dim) {
       float cVal = 0;
       for(int k = 0; k < dim; ++k){
            cVal += A[row*dim+k] * B[k * dim+col];
       } 
       C[row * dim + col] = cVal;
    }

}  

int main() {
    float A[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float B[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float C[9] = {0};

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, 9 * sizeof(float));
    cudaMalloc(&d_B, 9 * sizeof(float));
    cudaMalloc(&d_C, 9 * sizeof(float));

    cudaMemcpy(d_A, A, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, 9 * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 block(3, 3);
    dim3 grid((9 + block.x - 1) / block.x, (9 + block.y - 1) / block.y);

    matrixMul<<<grid, block>>>(d_A, d_B, d_C, 3);

    cudaMemcpy(C, d_C, 9 * sizeof(float), cudaMemcpyDeviceToHost);
    
    for(int i = 0; i < 9; i++){
        printf("%f ", C[i]);
    }
    printf("\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}