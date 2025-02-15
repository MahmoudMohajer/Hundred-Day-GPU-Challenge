# Day 009 - Image Blur using CUDA

## Objective
Implement a parallel image blurring algorithm using CUDA that performs a box blur on PGM images. This project demonstrates neighborhood operations, boundary handling, and image processing fundamentals using GPU acceleration.

## Concepts Covered
- Neighborhood Operations in CUDA
- Image Convolution Principles
- PGM Image File Handling
- Dynamic Blur Size Configuration
- Boundary Pixel Management

## Key Components
1. **Blur Kernel Implementation:**
   - Configurable blur window size
   - Proper boundary checking
   - Average calculation for each pixel
   - Parallel processing per pixel

2. **Memory Management:**
   - Host and device memory allocation
   - Image data transfer
   - Proper cleanup of resources

3. **Image Processing:**
   - PGM format handling
   - Pixel value averaging
   - Dynamic blur size from command line

## Key Learning Points
1. Understanding neighborhood operations in parallel computing
2. Managing boundary conditions in image processing
3. Implementing averaging operations efficiently
4. Handling variable-sized convolution windows
5. Processing grayscale image data in parallel

## Building and Running
1. Compile with nvcc:
   ```bash
   nvcc blur_image_kernel.cu -o blur_image_kernel
   ```

2. Run the program:
   ```bash
   ./blur_image_kernel input.pgm output.pgm blur_size
   ```
   - `input.pgm`: Source image in PGM format
   - `output.pgm`: Destination for blurred image
   - `blur_size`: Integer determining the blur window size (e.g., 1 for 3x3, 2 for 5x5)

## Implementation Details
The blur kernel processes each pixel in parallel, where:
- Each thread handles one output pixel
- The blur window size is (2 * blur_size + 1) × (2 * blur_size + 1)
- Boundary checking ensures valid memory access
- Final value is the average of all pixels in the blur window

## Notes
- Only binary PGM (P5) format is supported
- Maximum pixel value must be 255
- Blur size affects both performance and blur intensity
- Edge pixels are handled properly with reduced window size
- Uses 16×16 thread blocks for optimal performance

## Results
The program produces a blurred version of the input image, where:
- Larger blur_size values create more pronounced blurring
- Edge pixels are properly handled
- Processing is performed in parallel for efficiency

![original image](picture.png)
*Figure 1: Original image before blur application.*

![blurred image](./blured_picture.png)
*Figure 2: Blurred image output from CUDA program.*

## Performance Considerations
- Thread block size of 16×16 balances occupancy and performance
- Memory transfers are minimized to essential operations
- Boundary checking adds some overhead but ensures correctness
- Larger blur windows increase computational complexity
