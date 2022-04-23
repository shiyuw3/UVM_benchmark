#include <iostream>
#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ~100% usage: 7e8, ~105% usage: 7.35e8
#define N (735000000)

__global__ void kernel(float* input, float* output, float* table, size_t size) {
  int x_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (x_id > size || x_id % 100 != 0)
    return;

  float in_f = input[x_id];
  int in_i = (int)(floor(in_f));
  int table_index = (int)((in_f - float(in_i)) *( (float)(N) ));
  float* t = table + table_index;
  output[table_index] = t[0] * in_f;
}

int main(void) {
  float *input, *output, *table;

  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&input, N * sizeof(float));
  cudaMallocManaged(&output, N * sizeof(float));
  cudaMallocManaged(&table, N * sizeof(float));

  // Initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    input[i] = static_cast <float>(rand()) / static_cast<float>(RAND_MAX);
    table[i] = ((float)(i));
  }

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsed_time = 0.0f;

  cudaEventRecord(start, 0);

  kernel<<<numBlocks, blockSize>>>(input, output, table, N);
  cudaDeviceSynchronize();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  printf("Elapsed time: %f s\n", elapsed_time / 1000);

  // for (int i = 0; i < N; i++) {
  //   if(output[i] != 0) {
  //     printf("-%d %lf-\n", i, output[i]);
  //   }
  // }

  return 0;
}
