#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \
    do \
    { \
        cudaError_t error = call; \
        if (error != cudaSuccess) \
        { \
            fprintf(stderr, "Error at the %s:%d is %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
} while (0);

#define TILE_SIZE 16 

__global__
void naive_mm(float *a, float *b, float *c, int w) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < w && col < w) {
       float acc = 0;
       for (int k = 0; k < w; k++)
       {
        acc += a[row * w + k] * b[k * w + col]; 
       }
       c[row * w + col] = acc;
    }
}

__global__ 
void tiled_mm(float *A, float *B, float *C, int N) {
    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE + 1]; // Add 1-element padding
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + tx;
    int row = blockIdx.y * TILE_SIZE + ty;
    
    float Cvalue = 0.0f;

    for (int t = 0; t < N/TILE_SIZE; t++) {
        // Coalesced global loads remain the same
        Asub[ty][tx] = A[row * N + (t * TILE_SIZE + tx)];
        Bsub[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        
        __syncthreads();

        // Bank conflict-free access to Bsub
        for (int k = 0; k < TILE_SIZE; k++) {
            Cvalue += Asub[ty][k] * Bsub[k][tx]; // Now using padded Bsub
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = Cvalue;
    }
}

void verify_result(float* h_a, float* h_b, float* h_c, int width) {
    float max_rel_error = 0.0f;
    int errors = 0;
    const int samples = 1000; // Check 1000 random elements

    for (int n = 0; n < samples; n++) {
        int i = rand() % width;
        int j = rand() % width;
        
        float expected = 0.0f;
        for (int k = 0; k < width; k++) {
            expected += h_a[i * width + k] * h_b[k * width + j];
        }
        
        float computed = h_c[i * width + j];
        float abs_error = fabs(computed - expected);
        float rel_error = abs_error / fabs(expected);
        
        if (rel_error > 1e-5f) { // 0.001% relative error
            if (errors < 10) {
                printf("Sample %d [%d][%d]: Rel error %.2e (expected %.2f, got %.2f)\n",
                       n, i, j, rel_error, expected, computed);
            }
            errors++;
            max_rel_error = fmaxf(max_rel_error, rel_error);
        }
    }

    if (errors == 0) {
        printf("Validation passed: All %d samples within 0.001%% relative error\n", samples);
    } else {
        printf("Validation issues: %d/%d samples exceeded 0.001%% rel error (max: %.2e)\n",
               errors, samples, max_rel_error);
    }
}


int main() {
    int N = 1 << 10; 
    int size = N * N;
    int bytes = size * sizeof(float); 
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = i + j * 0.005f; // Initialize h_A
            h_B[i * N + j] = i * j * 0.001f; // Initialize h_B
        }
    }
   

    float *d_A, *d_B, *d_C; 
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes)); 
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes)); 
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes)); 
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE); 
    dim3 grid(N / TILE_SIZE, N / TILE_SIZE); 

    // Create CUDA events for timing
    cudaEvent_t start_tiled, stop_tiled, start_naive, stop_naive;
    CUDA_CHECK(cudaEventCreate(&start_tiled));
    CUDA_CHECK(cudaEventCreate(&stop_tiled));
    CUDA_CHECK(cudaEventCreate(&start_naive));
    CUDA_CHECK(cudaEventCreate(&stop_naive));

    // Benchmark tiled kernel
    CUDA_CHECK(cudaEventRecord(start_tiled));
    tiled_mm<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop_tiled));
    CUDA_CHECK(cudaEventSynchronize(stop_tiled));
    
    float tiled_time;
    CUDA_CHECK(cudaEventElapsedTime(&tiled_time, start_tiled, stop_tiled));
    printf("\nTiled Kernel Execution Time: %.2f ms\n", tiled_time);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    verify_result(h_A, h_B, h_C, N); 

    dim3 block_naive(16, 16); 
    dim3 grid_naive((N + block_naive.x - 1) / block_naive.x, (N + block_naive.y - 1) / block_naive.y); 

    // Benchmark naive kernel
    CUDA_CHECK(cudaEventRecord(start_naive));
    naive_mm<<<grid_naive, block_naive>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop_naive));
    CUDA_CHECK(cudaEventSynchronize(stop_naive));
    
    float naive_time;
    CUDA_CHECK(cudaEventElapsedTime(&naive_time, start_naive, stop_naive));
    printf("Naive Kernel Execution Time: %.2f ms\n", naive_time);
    printf("Speedup: %.2fx\n", naive_time / tiled_time);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    verify_result(h_A, h_B, h_C, N); 

    // Cleanup events
    CUDA_CHECK(cudaEventDestroy(start_tiled));
    CUDA_CHECK(cudaEventDestroy(stop_tiled));
    CUDA_CHECK(cudaEventDestroy(start_naive));
    CUDA_CHECK(cudaEventDestroy(stop_naive));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);
}