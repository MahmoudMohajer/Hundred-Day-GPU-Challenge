# Day 017 - Naive Matrix Multiplication in CUDA

This program implements a basic matrix multiplication algorithm using CUDA. While this implementation is straightforward, it serves as a baseline for understanding more optimized approaches.

## Program Overview

The program `naive_MM.cu` performs matrix multiplication C = A × B with the following characteristics:
- Square matrices of size 16×16
- Simple thread-per-element computation model
- No shared memory optimization
- Basic row-column multiplication algorithm

## Implementation Details

### Kernel Function
```cuda
__global__ void naiveMatrixMul(const float *A, const float *B, float *C, int width)
```

The kernel:
1. Assigns each thread to compute one element of the output matrix
2. Each thread:
   - Calculates its position in the output matrix using block and thread indices
   - Performs dot product of a row from matrix A and a column from matrix B
   - Stores the result in the corresponding position in matrix C

### Memory Management
- Allocates memory on both host and device
- Uses standard cudaMalloc and cudaMemcpy operations
- Matrices are stored in row-major format

### Grid and Block Configuration
- Uses 16×16 thread blocks
- Grid size is calculated to cover the entire matrix
- Block size matches the matrix dimension for this simple case

## Limitations of Naive Implementation

1. **Memory Access Pattern**
   - Non-coalesced memory accesses
   - High global memory bandwidth usage
   - No data reuse

2. **Performance**
   - Each element requires width number of global memory accesses
   - No use of shared memory or cache
   - Poor arithmetic intensity

## Potential Optimizations

Future improvements could include:
- Using shared memory to reduce global memory accesses
- Tiling the matrices for better cache utilization
- Employing memory coalescing techniques
- Loop unrolling for better instruction-level parallelism
- Using more efficient matrix storage formats

## Usage

To compile and run:
```bash
nvcc naive_MM.cu -o naive_MM
./naive_MM
```

The program will output the first 10 elements of the resulting matrix C for verification.
