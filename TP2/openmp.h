#include <iostream>
#include <omp.h>
#include <float.h>
#include <random>
#include <chrono>

#include "container.hpp"
#include "global_parameters.hpp"
#include "ios.hpp"
#include "csv.h"

#define VEC_SIZE 1024*1024*16
#define DIV_MAX 1024*16
#define DIV_MIN 1
#define POW_MAX 15
#define MAX_ITER 1000

int run_exercices(int data_size, int n_threads, double p_stats[][4][POW_MAX][3] , int iter);

void exercice1_omp(double* vec1, double* vec2, double* res, int size);
void exercice1_seq(double* vec1, double* vec2, double* res, int size);
double exercice2_omp_no_red(double* vec1, double* vec2, int size);
double exercice2_omp_red(double* vec1, double* vec2, int size, bool imbalanced, int iter);
double exercice2_seq(double* vec1, double* vec2, int size, bool imbalanced);
void exercice4_omp(u_char **Source, u_char **Resultat, unsigned int height, unsigned int width);
void exercice4_seq(u_char **Source, u_char **Resultat, unsigned int height, unsigned int width);
void exercice5_seq(int* list, int* even_sublist, int* counter, int size);
void exercice5_omp(int* list, int* even_sublist, int* counter, int n_threads, int size);

double rand_double();
int rand_int();