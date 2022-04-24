/**
 * atax.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#include "../../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

/* Problem size. */
// 47600: ~110% usage.
// 46500: ~105% usage.
// 45400-45450: ~100% usage. 45400: no eviction, 45450: eviction.
#define NX 45500
#define NY 45500

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

#ifndef M_PI
#define M_PI 3.14159
#endif

#define ENABLE_CPU 0

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;


#if ENABLE_CPU
void init_array(DATA_TYPE *x, DATA_TYPE *A,
                DATA_TYPE *x_gpu, DATA_TYPE *A_gpu) {
#else
void init_array(DATA_TYPE *x_gpu, DATA_TYPE *A_gpu) {
#endif
  size_t i, j;

  for (i = 0; i < NX; i++) {
#if ENABLE_CPU
    x[i] = i * M_PI;
#endif
    x_gpu[i] = i * M_PI;
    for (j = 0; j < NY; j++) {
#if ENABLE_CPU
      A[i*NY + j] = ((DATA_TYPE) i*(j)) / NX;
#endif
      A_gpu[i*NY + j] = ((DATA_TYPE) i*(j)) / NX;
    }
  }
}


void compareResults(DATA_TYPE *z, DATA_TYPE *z_outputFromGpu) {
  size_t i, fail;
  fail = 0;

  for (i = 0; i < NY; i++) {
    if (percentDiff(z[i], z_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
      fail++;
    }
  }

  // print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %ld\n",
         PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init() {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
  printf("setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
  cudaSetDevice(GPU_DEVICE);
}


__global__ void atax_kernel1(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *tmp) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < NX) {
    size_t j;
    for (j = 0; j < NY; j++) {
      tmp[i] += A[i * NY + j] * x[j];
    }
  }
}

__global__ void atax_kernel2(DATA_TYPE *A, DATA_TYPE *y, DATA_TYPE *tmp) {
  size_t j = blockIdx.x * blockDim.x + threadIdx.x;

  if (j < NY) {
    size_t i;
    for (i = 0; i < NX; i++) {
      y[j] += A[i * NY + j] * tmp[i];
    }
  }
}

void atax_cpu(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp) {
  size_t i, j;

  for (i = 0; i < NY; i++) {
    y[i] = 0;
  }

  for (i = 0; i < NX; i++) {
    tmp[i] = 0;

    for (j = 0; j < NY; j++) {
      tmp[i] = tmp[i] + A[i * NY + j] * x[j];
    }

    for (j = 0; j < NY; j++) {
      y[j] = y[j] + A[i * NY + j] * tmp[i];
    }
  }
}


void ataxGpu(DATA_TYPE* A_gpu, DATA_TYPE* x_gpu, DATA_TYPE* y_gpu,
             DATA_TYPE* tmp_gpu) {
  double t_start, t_end;

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid1((size_t)(ceil(((float)NX) / ((float)block.x))), 1);
  dim3 grid2((size_t)(ceil(((float)NY) / ((float)block.x))), 1);

  t_start = rtclock();
  atax_kernel1<<<grid1, block>>>(A_gpu, x_gpu, tmp_gpu);
  cudaDeviceSynchronize();
  atax_kernel2<<<grid2, block>>>(A_gpu, y_gpu, tmp_gpu);
  cudaDeviceSynchronize();
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
}


int main(int argc, char** argv) {
  DATA_TYPE *A_gpu;
  DATA_TYPE *x_gpu;
  DATA_TYPE *y_gpu;
  DATA_TYPE *tmp_gpu;

#if ENABLE_CPU
  DATA_TYPE* A;
  DATA_TYPE* x;
  DATA_TYPE* y;
  DATA_TYPE* tmp;

  A = (DATA_TYPE*)malloc(NX * NY * sizeof(DATA_TYPE));
  x = (DATA_TYPE*)malloc(NY * sizeof(DATA_TYPE));
  y = (DATA_TYPE*)malloc(NY * sizeof(DATA_TYPE));
  tmp = (DATA_TYPE*)malloc(NX * sizeof(DATA_TYPE));
#endif

  cudaMallocManaged(&A_gpu, sizeof(DATA_TYPE) * NX * NY);
  cudaMallocManaged(&x_gpu, sizeof(DATA_TYPE) * NY);
  cudaMallocManaged(&y_gpu, sizeof(DATA_TYPE) * NY);
  cudaMallocManaged(&tmp_gpu, sizeof(DATA_TYPE) * NX);

#if ENABLE_CPU
  init_array(x, A, x_gpu, A_gpu);
#else
  init_array(x_gpu, A_gpu);
#endif

  GPU_argv_init();
  ataxGpu(A_gpu, x_gpu, y_gpu, tmp_gpu);

#if ENABLE_CPU
  double t_start, t_end;

  t_start = rtclock();
  atax_cpu(A, x, y, tmp);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(y, y_gpu);

  free(A);
  free(x);
  free(y);
  free(tmp);
#endif

  cudaFree(A_gpu);
  cudaFree(x_gpu);
  cudaFree(y_gpu);
  cudaFree(tmp_gpu);

  return 0;
}
