# Hundred-Day-GPU-Challenge

This repository documents my journey through 100 days of GPU programming challenges, focusing on CUDA and parallel computing concepts.

## Progress Tracking

| Day | Date | Topic/Challenge | Status | Notes |
|-----|------|----------------|--------|-------|
| [001](./Day001/hello_world.cu) |2025-02-05 | Hello World CUDA | Done | Basic CUDA program setup |
| [002](./Day002/grid_1D.cu) |2025-02-06 | Grid 1D CUDA | Done |  |
| [003](./Day003/vector_add.cu) |2025-02-07 | Vector Addition | Done |  |
| [004](./Day004/benchmarking.cu) |2025-02-08 | Performance Benchmarking | Done | Memory transfer vs kernel execution analysis |
| [005](./Day005/README.md) |2025-02-09 | 2D Matrix Addition | Done |  |
| [006](./Day006/README.md) |2025-02-10 | Modular Matrix Addition | Done | using device functions |
| [007](./Day007/README.md) |2025-02-11 | Matrix Addition Performance Analysis | Done | Block size optimization (8x8, 16x16, 32x32) with Nsight profiling |
| 008 | 2025-02-15 | Image Brightness Adjustment using CUDA | Done | CUDA-based image processing with brightness adjustment, host/device memory management, and PGM file I/O |
| [009](./Day009/README.md) | 2025-02-16 | Image Blur using CUDA | Done | Parallel box blur implementation with configurable kernel size, boundary handling, and PGM image processing |
| [010](./Day010/README.md) | 2025-02-17 | Matrix Multiplication using CUDA | Done | Basic parallel matrix multiplication with 2D grid/block organization |
| [011](./Day011/README.md) | 2025-02-19 | Matrix Multiplication Performance Benchmarking | Done | Systematic performance analysis with different block sizes (8x8 to 32x32) on 1024x1024 matrices |
| [012](./Day012/README.md) | 2025-02-21 | Vector Element-wise Multiplication | Done | 1D parallel implementation with performance measurement and result verification |
| [013](./Day013/README.md) | 2025-02-23 | Synchronous vs Asynchronous Data Transfer | Done | Performance comparison between sync and async data transfer with CUDA streams |
| [014](./Day014/README.md) | 2025-02-26 | CUDA Thread Synchronization and Parallel Reduction | Done | Parallel reduction implementation with thread synchronization, shared memory, and multiple-pass reduction for large arrays |
| [015](./Day015/README.md) | 2025-03-02 | Branchless CUDA Parallel Reduction | Done | Performance comparison between branching and branchless implementations, demonstrating that eliminating branches isn't always beneficial |
| [016](./Day016/README.md) | 2025-03-06 | CUDA Device Properties | Done | Querying and displaying GPU hardware characteristics using CUDA Runtime API |
| [017](./Day017/README.md) | 2025-03-07 | Naive Matrix Multiplication | Done | Basic CUDA matrix multiplication implementation without optimizations |
| [018](./Day018/README.md) | 2025-03-14 | Tiled Matrix Multiplication | Done | Optimized matrix multiplication using shared memory tiling, achieving 1.3x speedup |
| 019 | | | | |
| 020 | | | | |

*Note: Table will be updated daily with new entries and progress.*

## Repository Structure
- Each day's work is stored in its own directory (e.g., `Day001`, `Day002`, etc.)
- Each directory contains the source code and documentation for that day's challenge