#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define cudaCheckError(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, \
                                                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0);

#endif // CUDA_UTILS_H 