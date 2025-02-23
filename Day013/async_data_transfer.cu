#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Vector addition kernel with increased computation
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = a[idx] + b[idx];
        for (int i = 0; i < 1000; i++) {  // Heavy computation
            sum += sinf(sum) * cosf(sum);
        }
        c[idx] = sum;
    }
}

int main() {
    const int N = 1 << 28;  // 268,435,456 elements
    const int STREAMS = 4;  // Number of streams
    const int SEGMENT_SIZE = N / STREAMS;
    const int BYTES = SEGMENT_SIZE * sizeof(float);

    // Host arrays (pinned memory)
    float *h_a, *h_b, *h_c;
    CUDA_CHECK(cudaHostAlloc((void**)&h_a, N * sizeof(float), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_b, N * sizeof(float), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_c, N * sizeof(float), cudaHostAllocDefault));

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // Device arrays
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    // Create streams
    cudaStream_t streams[STREAMS];
    for (int i = 0; i < STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // Timing variables
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start time
    CUDA_CHECK(cudaEventRecord(start));

    // Launch kernels and transfers using streams
    int threadsPerBlock = 256;
    int blocksPerGrid = (SEGMENT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    for (int i = 0; i < STREAMS; i++) {
        int offset = i * SEGMENT_SIZE;
        
        // Asynchronous memory transfers
        CUDA_CHECK(cudaMemcpyAsync(&d_a[offset], &h_a[offset], BYTES,
                                 cudaMemcpyHostToDevice, streams[i]));
        CUDA_CHECK(cudaMemcpyAsync(&d_b[offset], &h_b[offset], BYTES,
                                 cudaMemcpyHostToDevice, streams[i]));
        
        // Kernel launch
        vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>
            (&d_a[offset], &d_b[offset], &d_c[offset], SEGMENT_SIZE);
        
        // Asynchronous memory transfer back to host
        CUDA_CHECK(cudaMemcpyAsync(&h_c[offset], &d_c[offset], BYTES,
                                 cudaMemcpyDeviceToHost, streams[i]));
    }

    // Record stop time
    CUDA_CHECK(cudaEventRecord(stop));

    // Synchronize all streams
    for (int i = 0; i < STREAMS; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    // Calculate and print execution time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Async Execution time: %.2f ms\n", milliseconds);

    // Cleanup
    for (int i = 0; i < STREAMS; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c));

    return 0;
}