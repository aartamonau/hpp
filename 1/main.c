// MP 1
#include <wb.h>

#define CHECK(code) \
do { \
  cudaError_t ret = (code); \
  if (ret != cudaSuccess) { \
    wbLog(ERROR, "%s::%s Unexpected error %s", __func__, __LINE__, cudaGetErrorString(ret)); \
  } \
} while (0)

#define ROWS_SHIFT 2
#define ROWS (1 << ROWS_SHIFT)
#define THREADS 256
#define BLOCKS(n) ((((n) - 1) / (THREADS * ROWS)) + 1)

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
    int i = (blockDim.x * blockIdx.x + threadIdx.x) * ROWS;

    if (i >= len) {
        return;
    }

    in1 = in1 + i;
    in2 = in2 + i;
    out = out + i;

    switch (ROWS - 1 - (i & ((1 << ROWS_SHIFT) - 1))) {
    case 7:
        *out++ = *in1++ + *in2++;
    case 6:
        *out++ = *in1++ + *in2++;
    case 5:
        *out++ = *in1++ + *in2++;
    case 4:
        *out++ = *in1++ + *in2++;
    case 3:
        *out++ = *in1++ + *in2++;
    case 2:
        *out++ = *in1++ + *in2++;
    case 1:
        *out++ = *in1++ + *in2++;
    case 0:
        *out = *in1 + *in2;
    }
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    CHECK( cudaMalloc(&deviceInput1, sizeof(float) * inputLength * 3) );
    deviceInput2 = deviceInput1 + inputLength;
    deviceOutput = deviceInput2 + inputLength;

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    CHECK( cudaMemcpy(deviceInput1, hostInput1, sizeof(float) * inputLength, cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(deviceInput2, hostInput2, sizeof(float) * inputLength, cudaMemcpyHostToDevice) );

    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 dimGrid(BLOCKS(inputLength));
    dim3 dimBlock(THREADS);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    vecAdd<<< dimGrid, dimBlock >>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    CHECK( cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * inputLength, cudaMemcpyDeviceToHost) );

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    CHECK( cudaFree(deviceInput1) );

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}
