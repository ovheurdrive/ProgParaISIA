#include "SIMD.h"

void test(){
    __m256 a = _mm256_set1_ps(3);
    a = a - 1;
    a = a + 8;
    __m256 b = _mm256_set_ps(1,2,3,5,8,9,7,7);
    __m256 c = _mm256_set_ps(8,7,5,57,9,5,4,8);
    a = a * b;
    a = a / c;
    float d[8];
    _mm256_storeu_ps(&d[0], a);
    for(int i = 0; i < 8; i++)
        printf("%f\n", d[i]);
}

int main(){
    test();
}