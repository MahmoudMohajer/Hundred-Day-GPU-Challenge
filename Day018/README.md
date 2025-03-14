# Day 18: Tiled Matrix Multiplication

This implementation improves upon Day 17's naive matrix multiplication by using shared memory tiling, a crucial optimization technique for matrix operations on GPUs.

## Key Improvements

1. **Shared Memory Usage**: 
   - Implements tiling using shared memory to reduce global memory accesses
   - Uses shared memory tiles of size 16x16
   - Added padding to avoid bank conflicts in shared memory access

2. **Memory Access Pattern**:
   - Coalesced global memory loads for better memory throughput
   - Bank conflict-free shared memory access patterns
   - Reduced global memory traffic by reusing data in shared memory

3. **Performance Gains**:
   - Achieves ~1.2-1.3x speedup over the naive implementation
   - For 1024x1024 matrices:
     - Tiled version: ~2.13-2.36 ms
     - Naive version: ~2.78-2.79 ms

## Implementation Details

### Tiled Algorithm
1. Each thread block loads a tile of input matrices into shared memory
2. Threads in a block cooperate to compute a portion of the output matrix
3. The process repeats for all tiles needed to compute the final result

### Key Features
- Uses CUDA events for precise timing measurements
- Includes validation against CPU computation
- Implements error checking using CUDA_CHECK macro
- Supports arbitrary matrix sizes (powers of 2)

## Validation
- Performs random sampling of 1000 elements
- Validates results against CPU computation
- Ensures relative error is within 0.001%

## Usage
Compile with CUDA compiler:
```bash
nvcc tiled_mm.cu -o tiled_mm
```

Run the executable:
```bash
./tiled_mm
```

The program will output execution times for both tiled and naive implementations, along with the speedup factor and validation results.
