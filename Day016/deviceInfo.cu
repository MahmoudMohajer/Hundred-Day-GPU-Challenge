#include <stdio.h>

int main() {
    int deviceCount; 
    cudaGetDeviceCount(&deviceCount);
    printf("Number of devices: %d \n", deviceCount);
    cudaDeviceProp devProp; 
    for (int i=0; i < deviceCount; i++) {
        cudaGetDeviceProperties(&devProp, i);
        printf("Maximum threads per SM: %d\n", devProp.maxThreadsPerMultiProcessor);
        printf("Maximum threads per block: %d\n", devProp.maxThreadsPerBlock);
        printf("Maximum Warp Size: %d\n", devProp.warpSize);
        printf("Max threads per dim (x, y, z)(%d, %d, %d)\n", devProp.maxThreadsDim[0],
                devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
        printf("Max threads per dim (x, y, z)(%d, %d, %d)\n", devProp.maxGridSize[0],
                devProp.maxGridSize[1], devProp.maxGridSize[2]);
        printf("Registers per Block: %d\n", devProp.regsPerBlock);

    }
}