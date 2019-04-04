#include <iostream>
#include <omp.h>
#include <float.h>
#include <random>
#include <chrono>

#define VEC_SIZE 1024*1024
#define WIDTH 1024
#define MAX_ITER 10000

void exercice_1(void);
void exercice2_omp(double* vec1, double* vec2, double* res);
void exercice2_seq(double* vec1, double* vec2, double* res);
double exercice3_omp_no_red(double* vec1, double* vec2);
double exercice3_omp_red(double* vec1, double* vec2);
double exercice3_seq(double* vec1, double* vec2);
void exercice4_omp(double* vec, double* res);
void exercice4_seq(double* vec, double* res);

double rand_double();