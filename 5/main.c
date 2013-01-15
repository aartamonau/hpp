// MP 5 Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}
// Due Tuesday, January 22, 2013 at 11:59 p.m. PST

#include <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

__global__ void scan(float * input, float * output, int len) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here
    __shared__ float tile[BLOCK_SIZE * 2];

    int tix = threadIdx.x;
    int ix = 2 * (tix + blockDim.x * blockIdx.x);
    int stride;

    tile[2 * tix] = (ix < len) ? input[ix] : 0;
    tile[2 * tix + 1] = (ix + 1 < len) ? input[ix + 1] : 0;

    for (stride = 1; stride <= blockDim.x; stride <<= 1) {
        __syncthreads();

        if ((tix + 1) % stride == 0) {
            tile[2 * tix + 1] += tile[2 * tix + 1 - stride];
        }
    }

    for (stride = blockDim.x; stride > 0; stride >>= 1) {
        __syncthreads();

        if (((tix + 1) % stride == 0) && ((2 * (tix - stride) + 1) >= 0)) {
            tile[2 * tix + 1 - stride] += tile[2 * (tix - stride) + 1];
        }
    }

    output[ix] = (ix < len) ? tile[2 * tix] : 0;
    output[ix + 1] = (ix + 1 < len) ? tile[2 * tix + 1] : 0;
}

__global__ void completeScan(float * output, float * increments, int len) {
    __shared__ float inc;

    int tix = threadIdx.x;
    int ix = 2 * (tix + blockDim.x * blockIdx.x);

    if (tix == 0) {
        inc = increments[blockIdx.x];
    }

    __syncthreads();

    if (ix < len) {
        output[ix] += inc;
    }

    if (ix + 1 < len) {
        output[ix + 1] += inc;
    }
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numElements; // number of elements in the list
    int numBlocks;
    float * hostIncrements;
    float * deviceIncrements;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    numBlocks = (numElements + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceIncrements, numBlocks*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceInput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here

    dim3 dimGrid(numBlocks);
    dim3 dimBlock(BLOCK_SIZE);

    wbTime_start(Compute, "Performing CUDA computation (1st phase)");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce

    scan<<< dimGrid, dimBlock >>>(deviceInput, deviceOutput, numElements);

    cudaDeviceSynchronize();

    wbTime_stop(Compute, "Performing CUDA computation (1st phase)");

    wbTime_start(Copy, "Copying output memory to the CPU (interim)");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU (interim)");

    hostIncrements = (float *) malloc(sizeof(float) * numBlocks);
    hostIncrements[0] = 0;

    int i;
    int j;
    for (i = 2 * BLOCK_SIZE - 1, j = 1; i < numElements; i += 2 * BLOCK_SIZE, ++j) {
        hostIncrements[j] = hostIncrements[j - 1] + hostOutput[i];
    }

    wbTime_start(Copy, "Copying increments to GPU");
    wbCheck(cudaMemcpy(deviceIncrements, hostIncrements,
                       numBlocks * sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(Copy, "Copying increments to GPU");

    wbTime_start(Compute, "Performing CUDA computation (2st phase)");

    completeScan<<< dimGrid, dimBlock >>>(deviceOutput, deviceIncrements, numElements);

    cudaDeviceSynchronize();

    wbTime_stop(Compute, "Performing CUDA computation (2st phase)");

    wbTime_start(Copy, "Copying output memory to the CPU (final)");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU (final)");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}
