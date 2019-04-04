#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"
#include "omp.h"
#include "device_launch_parameters.h"

#include "1-dotp/dotp.cuh"
#include "2-sobel/sobel.cuh"
#include "3-transpo/transpo.cuh"
#include "4-histo/histo.cuh"

#include "utils.h"

#include "container.cuh"
#include "ios.cuh"
#include "parameters.cuh"

void run_exercice1(float* vec1, float* vec2, int size, int k, int iter);
void run_exercice2(u_char** Source, unsigned width, unsigned height, int iter);
void run_exercice3(u_char** Source, unsigned width, unsigned height, int iter);
void run_exercice4(u_char** Source, unsigned height, unsigned width, int iter);