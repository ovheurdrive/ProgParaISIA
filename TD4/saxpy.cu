#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"
#include "omp.h"
#include "device_launch_parameters.h"

using namespace std;

__global__ void gpu_saxpy(int n, float a, float *x, float *y, float *s)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) s[i] = a*x[i] + y[i];
}

void cpu_saxpy(int n, float a, float *x, float *y, float *s)
{
  #pragma omp parallel  for num_threads(8) 
  for (int i=0; i<n; i++)
  {
      s[i] = a*x[i] + y[i];
  }
}

void cpu_saxpy_mono(int n, float a, float *x, float *y, float *s)
{
  for (int i=0; i<n; i++)
  {
      s[i] = a*x[i] + y[i];
  }
}


int main(void)
{
  unsigned long int N = 4096*4096;
  float *x, *y, *s_cpu, *s_gpu, *d_x, *d_y, *d_s;
  
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));
  s_gpu = (float*)malloc(N*sizeof(float));
  s_cpu = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));
  cudaMalloc(&d_s, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 16M elements (CPU)
  cpu_saxpy_mono(N, 2.0f, x, y, s_cpu);  

  // Perform SAXPY on 16M elements (GPU)
  int k = 32;
  gpu_saxpy<<<(N+k)/k, k>>>(N, 2.0f, d_x, d_y, d_s);  
  
  // Perform SAXPY on 16M elements (CPU)
  cpu_saxpy(N, 2.0f, x, y, s_cpu);  

  cudaMemcpy(s_gpu, d_s, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = std::max(maxError, s_cpu[i]-s_gpu[i]);
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_s);
  free(x);
  free(y);
  free(s_cpu);
  free(s_gpu);
}

