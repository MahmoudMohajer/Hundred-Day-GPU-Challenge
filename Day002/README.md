# Day 002 - 1D Grid and Block Indexing in CUDA

## Objective
Create a CUDA program that demonstrates the concept of 1D grid and block indexing by printing thread and block indices for each thread in the grid.

## Concepts Covered
- 1D Grid Structure in CUDA
- Block and Thread Indexing
- Built-in CUDA Variables:
  - `blockIdx.x`: Block index in the grid
  - `threadIdx.x`: Thread index within a block

## Expected Output
The program will print the following information for each thread:
- Block Index (blockIdx.x)
- Thread Index (threadIdx.x)

## Key Learning Points
1. Understanding how CUDA organizes threads in a 1D grid structure
2. Learning to calculate global thread indices
3. Visualizing the relationship between blocks and threads
4. Working with CUDA's built-in indexing variables

## Example Calculation
For a grid with:
- 2 blocks
- 4 threads per block


## Building and Running
1. Compile with nvcc:
   ```bash
   nvcc grid_1D.cu -o grid_1D
   ```
2. Run the executable:
   ```bash
   ./grid_1D
   ```

## Notes
- This example serves as a foundation for understanding more complex grid structures in CUDA
- Understanding thread indexing is crucial for parallel algorithm development
- The concepts learned here will be essential for future GPU programming challenges
