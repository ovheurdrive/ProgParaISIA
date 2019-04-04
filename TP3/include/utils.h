#include <iostream>
#include <random>

enum g_type {
    g_int,
    g_float
};

typedef struct timer{
    double cpu_time;
    double gpu_time_kernel;
    double gpu_time_total;
} timer;

void display_vec(void* vec, int size, enum g_type id);
void display_timer(timer times);
void timer_avg(timer* p_timer, int iter, int size);
float rand_float();