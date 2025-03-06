# Day 016 - CUDA Device Properties

This program demonstrates how to query and display various CUDA device properties using the CUDA Runtime API.

## Program Overview

The program `deviceInfo.cu` queries important hardware characteristics of available CUDA devices, including:
- Number of CUDA-capable devices
- Maximum threads per Streaming Multiprocessor (SM)
- Maximum threads per block
- Warp size
- Maximum thread dimensions
- Maximum grid dimensions
- Available registers per block

## Output Example

```bash
Number of devices: 1 
Maximum threads per SM: 1536
Maximum threads per block: 1024
Maximum Warp Size: 32
Max threads per dim (x, y, z)(1024, 1024, 64)
Max threads per dim (x, y, z)(2147483647, 65535, 65535)
Registers per Block: 65536
```

## Understanding the Output

1. **Number of devices**: Shows how many CUDA-capable GPUs are available in the system
2. **Maximum threads per SM**: Maximum number of threads that can run simultaneously on a single Streaming Multiprocessor
3. **Maximum threads per block**: Maximum number of threads that can be launched in a single thread block
4. **Maximum Warp Size**: Number of threads in a warp (group of threads that execute in SIMD fashion)
5. **Max threads per dim**: Maximum number of threads allowed in each dimension (x, y, z) of a thread block
6. **Max grid size**: Maximum size of the grid in each dimension (x, y, z)
7. **Registers per Block**: Number of 32-bit registers available per block

## Key Concepts

- **Thread Block**: A 3D group of threads that can cooperate and share resources
- **Grid**: A 3D collection of thread blocks
- **Warp**: The basic unit of thread execution in CUDA (32 threads)
- **SM (Streaming Multiprocessor)**: Hardware unit that executes one or more thread blocks

This information is crucial for optimizing CUDA programs as it helps in:
- Choosing appropriate grid and block dimensions
- Understanding hardware limitations
- Planning resource utilization
- Optimizing kernel launch configurations
