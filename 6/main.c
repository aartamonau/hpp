#include <wb.h>

// Check ec2-174-129-21-232.compute-1.amazonaws.com:8080/mp/6 for more information


#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

#define MASK_WIDTH  5
#define MASK_RADIUS (MASK_WIDTH/2)

#define TILE_WIDTH 32
#define CHANNELS 3

#define EXT_TILE_WIDTH (TILE_WIDTH + 2 * MASK_RADIUS)

//@@ INSERT CODE HERE

__global__ void convolution(float *input, float *output,
                            int width, int height,
                            const float * __restrict__ mask)
{
    int bw = blockDim.x;
    int bh = blockDim.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int sx;
    int sy;
    int six;

    int tile_x;
    int tile_y;

    float __shared__ tile[EXT_TILE_WIDTH][EXT_TILE_WIDTH][CHANNELS];

#define x(dx) ((bx + (dx)) * bw + tx)
#define y(dy) ((by + (dy)) * bh + ty)
#define source_ix(x, y) (((y) * width + (x)) * CHANNELS)

#define read(dx, dy)                                   \
    do {                                               \
        sx = x(dx);                                    \
        sy = y(dy);                                    \
                                                       \
        tile_x = MASK_RADIUS + tx + (dx) * TILE_WIDTH; \
        tile_y = MASK_RADIUS + ty + (dy) * TILE_WIDTH; \
                                                       \
        if (sx < width && sy < height) {               \
            six = source_ix(sx, sy);                   \
                                                       \
            tile[tile_x][tile_y][0] = input[six];      \
            tile[tile_x][tile_y][1] = input[six + 1];  \
            tile[tile_x][tile_y][2] = input[six + 2];  \
        } else {                                       \
            tile[tile_x][tile_y][0] = 0;               \
            tile[tile_x][tile_y][1] = 0;               \
            tile[tile_x][tile_y][2] = 0;               \
        }                                              \
                                                       \
    } while (0)

    /* top halo */
    read(0, -1);

    /* bottom halo */
    read(0, 1);

    /* left halo */
    read(-1, 0);

    /* right halo */
    read(1, 0);

    /* top-left halo */
    read(-1, -1);

    /* top-right halo */
    read(1, -1);

    /* bottom-left halo */
    read(-1, 1);

    /* bottom-right halo */
    read(1, 1);

    /* tile */
    read(0, 0);

    __syncthreads();

    float p[CHANNELS] = { 0 };
    float m;

    for (int i = -MASK_RADIUS; i <= MASK_RADIUS; ++i) {
        for (int j = -MASK_RADIUS; j <= MASK_RADIUS; ++j) {
            for (int c = 0; c < CHANNELS; ++c) {
                int mi = i + MASK_RADIUS;
                int mj = j + MASK_RADIUS;

                m = mask[(mi + mj * MASK_WIDTH) * CHANNELS + c];
                p[c] += tile[tx + i][ty + j][c] * m;
            }
        }
    }

    sx = x(0);
    sy = y(0);
    tile_x = MASK_RADIUS + tx;
    tile_y = MASK_RADIUS + ty;

    if (sx < width && sy < height) {
        six = source_ix(sx, sy);

        output[six] = tile[tile_x][tile_y][0];
        output[six + 1] = tile[tile_x][tile_y][1];
        output[six + 2] = tile[tile_x][tile_y][2];
    }
}


int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    wbLog(TRACE, "Image dimensions: ", imageWidth, "x", imageHeight);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");

    dim3 dimGrid(TILE_WIDTH, TILE_WIDTH);
    dim3 dimBlock(1 + (imageWidth - 1) / TILE_WIDTH,
                  1 + (imageHeight - 1) / TILE_WIDTH);

    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
    convolution<<< dimGrid, dimBlock >>>(deviceInputImageData,
                                         deviceOutputImageData,
                                         imageWidth, imageHeight, deviceMaskData);
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
