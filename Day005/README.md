# Day 005 - 2D Matrix Addition in CUDA

## Objective
Implement parallel matrix addition using CUDA, demonstrating the use of 2D grid and block structures for handling matrix operations efficiently.

## Concepts Covered
- 2D Grid and Block Organization
- Matrix Operations in CUDA
- 2D Thread Indexing
- Memory Management for Matrices
- Row-Major Matrix Layout

## Key Components
1. Matrix Configuration:
   - 1024 x 1024 matrix dimensions
   - Total elements: ~1 million
   - Row-major memory layout

2. Thread Organization:
   - 2D Block Structure (16x16 threads)
   - 2D Grid Configuration
   - Efficient thread mapping to matrix elements

3. Memory Operations:
   - Host memory allocation for matrices
   - Device memory management
   - Matrix data transfers

## Expected Output
The program performs element-wise addition of two matrices (A + B = C) where:
- Matrix A is initialized with all 1.0
- Matrix B is initialized with all 2.0
- Expected result C should contain all 3.0

## Key Learning Points
1. Understanding 2D thread organization in CUDA
2. Managing large matrix data structures
3. Calculating appropriate grid and block dimensions for 2D problems
4. Implementing boundary checks for matrix operations
5. Row-major indexing in matrix computations

## Building and Running
1. Compile with nvcc:
   ```bash
   nvcc matrix_addition.cu -o matrix_addition
   ```
2. Run the executable:
   ```bash
   ./matrix_addition
   ```

## Notes
- Uses 16x16 thread blocks for optimal performance
- Demonstrates proper error checking for matrix operations
- Shows how to map 2D thread structure to matrix elements
- Important foundation for more complex matrix operations
- Memory layout consideration is crucial for performance
