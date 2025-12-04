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

// ============================================================================
// DAY 019: LOOP UNROLLED TILED MATRIX MULTIPLICATION
// ============================================================================
// This kernel adds manual loop unrolling to the inner computation loop
// to reduce loop overhead and increase instruction-level parallelism (ILP).
// The #pragma unroll directive hints the compiler to unroll the loop,
// reducing branch instructions and enabling better instruction scheduling.
// ============================================================================

__global__ 
void tiled_mm_unrolled(float *A, float *B, float *C, int N) {
    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE + 1]; // Padding for bank conflict avoidance
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + tx;
    int row = blockIdx.y * TILE_SIZE + ty;
    
    float Cvalue = 0.0f;

    int numTiles = N / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Coalesced global loads
        int aCol = t * TILE_SIZE + tx;
        int bRow = t * TILE_SIZE + ty;
        
        Asub[ty][tx] = A[row * N + aCol];
        Bsub[ty][tx] = B[bRow * N + col];
        
        __syncthreads();

        // Unrolled inner loop - 16 iterations unrolled manually
        // This reduces loop overhead and allows better instruction pipelining
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            Cvalue += Asub[ty][k] * Bsub[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = Cvalue;
    }
}

// ============================================================================
// DAY 019 BONUS: FULLY MANUALLY UNROLLED TILED MATRIX MULTIPLICATION
// ============================================================================
// This version manually unrolls the inner loop without relying on compiler
// pragmas. This can sometimes provide better performance guarantees and
// allows for explicit register allocation optimization.
// ============================================================================

__global__ 
void tiled_mm_fully_unrolled(float *A, float *B, float *C, int N) {
    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE + 1];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + tx;
    int row = blockIdx.y * TILE_SIZE + ty;
    
    float Cvalue = 0.0f;

    int numTiles = N / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Coalesced global loads
        int aCol = t * TILE_SIZE + tx;
        int bRow = t * TILE_SIZE + ty;
        
        Asub[ty][tx] = A[row * N + aCol];
        Bsub[ty][tx] = B[bRow * N + col];
        
        __syncthreads();

        // Fully manual unrolling of 16 iterations
        Cvalue += Asub[ty][0]  * Bsub[0][tx];
        Cvalue += Asub[ty][1]  * Bsub[1][tx];
        Cvalue += Asub[ty][2]  * Bsub[2][tx];
        Cvalue += Asub[ty][3]  * Bsub[3][tx];
        Cvalue += Asub[ty][4]  * Bsub[4][tx];
        Cvalue += Asub[ty][5]  * Bsub[5][tx];
        Cvalue += Asub[ty][6]  * Bsub[6][tx];
        Cvalue += Asub[ty][7]  * Bsub[7][tx];
        Cvalue += Asub[ty][8]  * Bsub[8][tx];
        Cvalue += Asub[ty][9]  * Bsub[9][tx];
        Cvalue += Asub[ty][10] * Bsub[10][tx];
        Cvalue += Asub[ty][11] * Bsub[11][tx];
        Cvalue += Asub[ty][12] * Bsub[12][tx];
        Cvalue += Asub[ty][13] * Bsub[13][tx];
        Cvalue += Asub[ty][14] * Bsub[14][tx];
        Cvalue += Asub[ty][15] * Bsub[15][tx];

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
    int N = 1 << 10; // 1024x1024 matrix
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
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float elapsed_time;

    printf("\n=== Day 019: Loop Unrolling Optimization ===\n");
    printf("Matrix size: %d x %d\n\n", N, N);

    // ========================================================================
    // Benchmark 1: Naive kernel (baseline)
    // ========================================================================
    dim3 block_naive(16, 16); 
    dim3 grid_naive((N + block_naive.x - 1) / block_naive.x, (N + block_naive.y - 1) / block_naive.y); 

    CUDA_CHECK(cudaEventRecord(start));
    naive_mm<<<grid_naive, block_naive>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("1. Naive Kernel:                %.3f ms\n", elapsed_time);
    float naive_time = elapsed_time;

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    verify_result(h_A, h_B, h_C, N); 

    // ========================================================================
    // Benchmark 2: Tiled kernel (Day 018)
    // ========================================================================
    CUDA_CHECK(cudaEventRecord(start));
    tiled_mm<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("\n2. Tiled Kernel (Day 018):      %.3f ms\n", elapsed_time);
    float tiled_time = elapsed_time;

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    verify_result(h_A, h_B, h_C, N); 

    // ========================================================================
    // Benchmark 3: Pragma unrolled kernel (Day 019)
    // ========================================================================
    CUDA_CHECK(cudaEventRecord(start));
    tiled_mm_unrolled<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("\n3. Pragma Unrolled (Day 019):   %.3f ms\n", elapsed_time);
    float pragma_time = elapsed_time;

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    verify_result(h_A, h_B, h_C, N); 

    // ========================================================================
    // Benchmark 4: Fully manual unrolled kernel (Day 019 Bonus)
    // ========================================================================
    CUDA_CHECK(cudaEventRecord(start));
    tiled_mm_fully_unrolled<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("\n4. Fully Unrolled (Day 019):    %.3f ms\n", elapsed_time);
    float manual_time = elapsed_time;

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    verify_result(h_A, h_B, h_C, N); 

    // ========================================================================
    // Summary
    // ========================================================================
    printf("\n=== Performance Summary ===\n");
    printf("Speedup (Tiled vs Naive):           %.2fx\n", naive_time / tiled_time);
    printf("Speedup (Pragma Unrolled vs Naive): %.2fx\n", naive_time / pragma_time);
    printf("Speedup (Manual Unrolled vs Naive): %.2fx\n", naive_time / manual_time);
    printf("Speedup (Pragma vs Tiled):          %.2fx\n", tiled_time / pragma_time);
    printf("Speedup (Manual vs Tiled):          %.2fx\n", tiled_time / manual_time);

    // Cleanup events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
