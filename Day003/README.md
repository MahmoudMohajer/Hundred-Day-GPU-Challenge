# Day 003 - Vector Addition in CUDA

## Objective
Implement a parallel vector addition program using CUDA that demonstrates basic memory management and parallel computation concepts.

## Concepts Covered
- CUDA Memory Management (Host and Device)
- Memory Transfer between CPU and GPU
- Parallel Vector Operations
- Error Checking in CUDA
- Grid and Block Dimension Calculations

## Key Components
1. Memory Operations:
   - Host memory allocation (CPU)
   - Device memory allocation (GPU) using cudaMalloc
   - Memory transfers using cudaMemcpy
   - Proper memory cleanup

2. Kernel Implementation:
   - Vector addition kernel function
   - Thread index calculation
   - Boundary checking for array access

3. Grid Configuration:
   - Block size optimization (256 threads per block)
   - Grid size calculation based on input size
   - Handling arbitrary vector sizes

## Expected Output
The program performs element-wise addition of two float vectors and verifies the results. It will output:
- Success message if the addition is correct
- Error message with details if any mismatch is found

## Key Learning Points
1. Understanding CUDA memory hierarchy
2. Managing data transfer between host and device
3. Implementing basic parallel algorithms
4. Calculating appropriate grid and block dimensions
5. Error handling in CUDA programs

## Building and Running
1. Compile with nvcc:
   ```bash
   nvcc vector_add.cu -o vector_add
   ```
2. Run the executable:
   ```bash
   ./vector_add
   ```

## Notes
- This example serves as a foundation for more complex parallel algorithms
- Demonstrates proper CUDA programming practices including memory management
- Shows how to handle array bounds checking in parallel operations
- Introduces basic performance considerations in CUDA programming
