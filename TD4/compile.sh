#!/bin/bash
nvcc -Xcompiler -fopenmp -O3 -o tdcuda.run tdcuda.cu
