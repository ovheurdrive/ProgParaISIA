#include <iostream>
#include <float.h>
#include <immintrin.h>
#include <random>
#include <chrono>

#define VEC_SIZE 1048576
#define MAX_ITER 1000

typedef struct minmax {
    float min;
    float max;
} minmax;

void exercice1_vec(float* vec1, float* vec2, float* vec3, float* res);
void exercice1_scal(float* vec1, float* vec2, float* vec3, float* res);
float exercice2_vec(float* vec1, float* vec2);
float exercice2_2_vec(float* vec1, float* vec2);
float exercice2_scal(float* vec1, float* vec2);
minmax exercice3_vec(float* vec);
minmax exercice3_scal(float* vec);
void exercice4_vec(float* src, float* res);
void exercice4_scal(float* src, float* res);
void exercice5_vec(float* src, float* res);
void exercice5_vec_bis(float* src, float* res);
void exercice5_scal(float* src, float* res);

float rand_float();