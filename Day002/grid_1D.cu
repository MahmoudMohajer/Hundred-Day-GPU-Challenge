#include <stdio.h>

__global__ void printIndices() {
    int threadId = threadIdx.x;
    int blockId = blockIdx.x;

    printf("Block %d, Thread %d\n", blockId, threadId);
}

int main() {
    int numBlocks = 2;
    int threadsPerBlock = 4;

    printIndices<<<numBlocks, threadsPerBlock>>>();

    cudaDeviceSynchronize();

}