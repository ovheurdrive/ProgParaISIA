#include <stdio.h>
#include <string.h>

#define VEC_SIZE 1024*1024*16
#define DIV_MAX 1024*16
#define DIV_MIN 1
#define POW_MAX 15

void create_stats_csv(double stats[4][POW_MAX][3]);