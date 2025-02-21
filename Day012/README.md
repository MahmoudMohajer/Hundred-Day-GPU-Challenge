# Day 012 - Vector Element-wise Multiplication with Performance Measurement

## Objective
Implement and measure the performance of parallel vector element-wise multiplication using CUDA, demonstrating efficient 1D grid/block organization and basic performance profiling.

## Concepts Covered
- 1D Vector Operations in CUDA
- Element-wise Multiplication
- Basic Performance Measurement
- CUDA Event-based Timing
- Error Checking in CUDA

## Key Components
1. **Vector Operation Implementation:**
   - Element-wise multiplication kernel
   - 1D thread/block organization
   - Automatic grid size calculation
   - Boundary checking

2. **Data Management:**
   - Large vector size (102,400 elements)
   - Host and device memory allocation
   - Efficient memory transfers
   - Result verification

3. **Performance Measurement:**
   - CUDA Events for timing
   - Single kernel timing
   - Device synchronization
   - Error checking implementation

## Key Learning Points
1. Efficient 1D thread organization
2. Vector operation parallelization
3. Basic performance measurement
4. Result verification techniques
5. Memory management for vectors

## Building and Running
1. Compile with nvcc:
   ```bash
   nvcc vec_elementwise_mul.cu -o vec_elementwise_mul
   ```

2. Run the program:
   ```bash
   ./vec_elementwise_mul
   ```

## Implementation Details
The implementation features:
- Vector size of 102,400 elements
- Block size of 256 threads
- Automatic grid size calculation
- Result verification (expected value: 6.0)
- Performance timing using CUDA Events

## Notes
- Uses 1D grid and block configuration
- Block size fixed at 256 threads
- Input vectors initialized with constant values (A=3.0, B=2.0)
- Includes error checking for computation accuracy
- Measures kernel execution time only

## Results
Performance metrics for vector multiplication:
- Vector Size: 102,400 elements
- Block Size: 256 threads
- Grid Size: Automatically calculated based on vector size
- Memory Required: ~1.2 MB (3 vectors × 102,400 × 4 bytes)
- Execution Time: Measured in milliseconds
- Result Verification: All elements should equal 6.0 (3.0 × 2.0)

## Performance Considerations
- Memory transfer overhead not included in timing
- Block size of 256 chosen for good occupancy
- Grid size automatically adjusts to vector size
- Error checking adds minimal overhead
- Single kernel timing without averaging
- Simple computation allows focus on parallelization concepts
