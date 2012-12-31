#include <wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

#define BLOCK_SIZE 16
#define TILE_SIZE (BLOCK_SIZE)

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float * B, float * C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    float partial = 0;

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int tileRow = threadIdx.y;
    int tileCol = threadIdx.x;

    int tileASrcRow = row;
    int tileASrcCol = tileCol;
    int tileBSrcRow = tileRow;
    int tileBSrcCol = col;

    int tile_count = 1 + (numAColumns - 1) / TILE_SIZE;

    while (tile_count--) {
        if (tileASrcRow < numARows && tileASrcCol < numAColumns) {
            tileA[tileRow][tileCol] = A[tileASrcRow * numAColumns + tileASrcCol];
        } else {
            tileA[tileRow][tileCol] = 0;
        }

        if (tileBSrcRow < numBRows && tileBSrcCol < numBColumns) {
            tileB[tileRow][tileCol] = B[tileBSrcRow * numBColumns + tileBSrcCol];
        } else {
            tileB[tileRow][tileCol] = 0;
        }
        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            partial += tileA[tileRow][i] * tileB[i][tileCol];
        }

        tileASrcCol += TILE_SIZE;
        tileBSrcRow += TILE_SIZE;

        __syncthreads();
    }

    if ((row < numCRows) && (col < numCColumns)) {
        C[row * numCColumns + col] = partial;
    }
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");

    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);

    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;

    //@@ Allocate the hostC matrix
    hostC = (float *) malloc(numCRows * numCColumns * sizeof(float));
    if (hostC == NULL) {
        wbLog(ERROR, "Could not allocate hostC");
        return -1;
    }

    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    wbCheck( cudaMalloc(&deviceA, sizeof(float) * numARows * numAColumns) );
    wbCheck( cudaMalloc(&deviceB, sizeof(float) * numBRows * numBColumns) );
    wbCheck( cudaMalloc(&deviceC, sizeof(float) * numCRows * numCColumns) );

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    wbCheck( cudaMemcpy(deviceA, hostA, sizeof(float) * numARows * numAColumns,
                        cudaMemcpyHostToDevice) );
    wbCheck( cudaMemcpy(deviceB, hostB, sizeof(float) * numBRows * numBColumns,
                        cudaMemcpyHostToDevice) );
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 dimGrid(((numCColumns - 1) / BLOCK_SIZE) + 1, ((numCRows - 1) / BLOCK_SIZE) + 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    matrixMultiplyShared<<< dimGrid, dimBlock >>>(deviceA, deviceB, deviceC,
                                                      numARows, numAColumns,
                                                      numBRows, numBColumns,
                                                      numCRows, numCColumns);

    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    wbCheck( cudaMemcpy(hostC, deviceC,
                        sizeof(float) * numCRows * numCColumns,
                        cudaMemcpyDeviceToHost) );

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    wbCheck( cudaFree(deviceA) );
    wbCheck( cudaFree(deviceB) );
    wbCheck( cudaFree(deviceC) );

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}
