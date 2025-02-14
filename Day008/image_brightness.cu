#include <stdio.h>
#include <stdlib.h>

__global__
void brightness_adjust(const unsigned char *input, unsigned char *output, int width, int height, int brightness_offset){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width){
        int idx = row * width + col;
        int pixel = input[idx] + brightness_offset;
        if (pixel > 255) pixel = 255;
        if (pixel < 0) pixel = 0;
        output[idx] = (unsigned char) pixel;
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s input.pgm output.pgm\n", argv[0]);
        return 1;
    }

    FILE *fp = fopen(argv[1], "rb");
    if (!fp) {
        perror("Error opening input file\n");
        return 1;
    }

    char format[3];
    if (fscanf(fp, "%2s", format) != 1) {
        fprintf(stderr, "Error reading PGM format\n");
        fclose(fp);
        return 1;
    }
    if (format[0] != 'P' || format[1] != '5') {
        fprintf(stderr, "Only binary PGM (P5) is supported\n");
        fclose(fp);
        return 1;
    }

    int width, height, maxval;
    fscanf(fp, "%d %d",&width, &height);
    fscanf(fp, "%d", &maxval);

    if (maxval != 255) {
        fprintf(stderr, "Only maxval of 255 is supported\n");
        fclose(fp);
        return 1;
    }

    int size = width * height;
    unsigned char *h_input = (unsigned char *)malloc(size);
    if (!h_input){
        fprintf(stderr, "couldn't allocate memory for host input\n");
        fclose(fp);
        return 1;
    } 

    if (fread(h_input, 1, size, fp) != size) {
        fprintf(stderr, "The elements are less than than expected size\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);

    unsigned char *h_output = (unsigned char *)malloc(size);
    if (!h_output){
        fprintf(stderr, "couldn't allocate memory for host output\n");
        free(h_input);
        return 1;
    }

    unsigned char *d_input, *d_output; 
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice); 

    int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y); 

    int brightness_offset = atoi(argv[3]);
    brightness_adjust<<<grid, block>>>(d_input, d_output, width, height, brightness_offset); 

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost); 

    fp = fopen(argv[2], "wb");
    if (!fp) {
        perror("Error opening output file");
        free(h_input);
        free(h_output);
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    } 

    fprintf(fp, "P5\n%d %d\n255\n", width, height);
    fwrite(h_output, 1, size, fp);
    fclose(fp);

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}