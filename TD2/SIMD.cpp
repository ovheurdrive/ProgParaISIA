#include "SIMD.h"

int main(){
    time_t seconds = time(NULL);
    time_t error_seed = 1549875661;
    srand(seconds);
    printf("Time Seed: %ld\n", seconds);
    // std::cout.setf(std::ios::fixed, std::ios::floatfield);
    // std::cout.setf(std::ios::showpoint);

    // Setup
    float* vec1 = (float*)malloc(VEC_SIZE*sizeof(float));
    float* vec2 = (float*)malloc(VEC_SIZE*sizeof(float));
    float* vec3 = (float*)malloc(VEC_SIZE*sizeof(float));
    float* res1 = (float*)malloc(VEC_SIZE*sizeof(float));
    float* res2 = (float*)malloc(VEC_SIZE*sizeof(float));
    float* res7 = (float*)malloc(VEC_SIZE*sizeof(float));
    float* res8 = (float*)malloc(VEC_SIZE*sizeof(float));
    float* res9 = (float*)malloc(VEC_SIZE*sizeof(float));
    float* res10 = (float*)malloc(VEC_SIZE*sizeof(float));

    for( int i = 0; i < VEC_SIZE; i++) {
        vec1[i] = rand_float();
        vec2[i] = rand_float();
        vec3[i] = rand_float();
    }
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    /* Exercice 1 */

    std::cout << "Exercice 1:\n====================\n";

    t0 = std::chrono::high_resolution_clock::now();
    for( int j = 0; j< MAX_ITER; j++){
        exercice1_vec(vec1,vec2,vec3,res1);
    }
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Vectorial Time: " <<  std::chrono::duration<double>(t1-t0).count()/(MAX_ITER*VEC_SIZE) << std::endl;

    t0 = std::chrono::high_resolution_clock::now();
    for( int j = 0; j< MAX_ITER; j++){
        exercice1_scal(vec1,vec2,vec3,res2);
    }
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Scalar Time: " <<  std::chrono::duration<double>(t1-t0).count()/(MAX_ITER*VEC_SIZE) << std::endl << std::endl;
    for( int j = 0; j< MAX_ITER; j++){
        if (res1[j] != res2[j]){
            std::cout << "Error, vec different" << std::endl << std::endl;
            break;
        }
    }

    /* Exercice 2 */

    std::cout << "Exercice 2:\n====================\n";
    float res3 = 0;
    float res4 = 0;
    float res3bis = 0;
    t0 = std::chrono::high_resolution_clock::now();
    for( int j = 0; j< MAX_ITER; j++){
        res3 = exercice2_vec(vec1,vec2);
    }
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Vectorial Time: " <<  std::chrono::duration<double>(t1-t0).count()/(MAX_ITER*VEC_SIZE) << " res: " << res3 << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    for( int j = 0; j< MAX_ITER; j++){
        res3bis = exercice2_2_vec(vec1,vec2);
    }
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Vectorial Time (using dp): " <<  std::chrono::duration<double>(t1-t0).count()/(MAX_ITER*VEC_SIZE) << " res: " << res3bis << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    for( int j = 0; j< MAX_ITER; j++){
        res4 = exercice2_scal(vec1,vec2);
    }
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Scalar Time: " <<  std::chrono::duration<double>(t1-t0).count()/(MAX_ITER*VEC_SIZE) <<  " res: " << res4 << std::endl << std::endl;
    if(res3 != res4){
        std::cout << "Error, dot product different" << std::endl<< std::endl;
    }

    /* Exercice 3 */

    std::cout << "Exercice 3:\n====================\n";
    minmax res5 = { 0, 0 };
    minmax res6 = { 0, 0 };
    t0 = std::chrono::high_resolution_clock::now();
    for( int j = 0; j< MAX_ITER; j++){
        res5 = exercice3_vec(vec1);
    }
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Vectorial Time: " <<  std::chrono::duration<double>(t1-t0).count()/(MAX_ITER*VEC_SIZE) << " min: " << res5.min << " max: " << res5.max << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    for( int j = 0; j< MAX_ITER; j++){
        res6 = exercice3_scal(vec1);
    }
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Scalar Time: " <<  std::chrono::duration<double>(t1-t0).count()/(MAX_ITER*VEC_SIZE) << " min: " << res6.min << " max: " << res6.max << std::endl << std::endl;
    if(res5.max != res6.max || res5.min != res6.min){
        std::cout << "Error, min max different" << std::endl << std::endl;
    }

    /* Exercice 4 */
    std::cout << "Exercice 4:\n====================\n";
    t0 = std::chrono::high_resolution_clock::now();
    for( int j = 0; j< MAX_ITER; j++){
        exercice4_vec(vec1,res7);
    }
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Vectorial Time: " <<  std::chrono::duration<double>(t1-t0).count()/(MAX_ITER*VEC_SIZE) << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    for( int j = 0; j< MAX_ITER; j++){
        exercice4_scal(vec1,res8);
    }
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Scalar Time: " <<  std::chrono::duration<double>(t1-t0).count()/(MAX_ITER*VEC_SIZE) << std::endl << std::endl;
    for( int j = 0; j< VEC_SIZE; j++){
        if (res7[j] != res8[j]){
            std::cout << "Error, images different" <<  std::endl << std::endl;
            break;
        }
    }
    
    /* Exercice 5 */
    std::cout << "Exercice 5:\n====================\n";
    t0 = std::chrono::high_resolution_clock::now();
    for( int j = 0; j< MAX_ITER; j++){
        exercice5_vec_bis(vec1,res9);
    }
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Vectorial Time: " <<  std::chrono::duration<double>(t1-t0).count()/(MAX_ITER*VEC_SIZE) << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    for( int j = 0; j< MAX_ITER; j++){
        exercice5_scal(vec1,res10);
    }
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Scalar Time: " <<  std::chrono::duration<double>(t1-t0).count()/(MAX_ITER*VEC_SIZE) << std::endl;
    for( int j = 0; j< VEC_SIZE; j++){
        if (res9[j] != res10[j]){
            std::cout << res9[j] << " " << res10[j] << " " << j << std::endl;
            std::cout << "Error, vec different" <<  std::endl << std::endl;
        }
    }


    free(vec1);
    free(vec2);
    free(vec3);
    free(res1);
    free(res2);
    free(res7);
    free(res8);
    free(res9);
    free(res10);
    return 0;
}

void exercice1_vec(float* vec1, float* vec2, float* vec3, float* res){
    __m256 d = _mm256_set1_ps(3);
    for( int i = 0; i < VEC_SIZE; i+=8){
        __m256 a = _mm256_loadu_ps(&vec1[i]);
        __m256 b = _mm256_loadu_ps(&vec2[i]);
        __m256 c = _mm256_loadu_ps(&vec3[i]);

        __m256 s = _mm256_add_ps(a,b);
        s = _mm256_add_ps(s,c);
        s = _mm256_div_ps(s,d);

        _mm256_storeu_ps(&res[i], s);
    }
}

void exercice1_scal(float* vec1, float* vec2, float* vec3, float* res){
    for ( int i = 0; i < VEC_SIZE; i++){
        res[i] = (vec1[i] + vec2[i] + vec3[i])/3;
    }
}

float exercice2_vec(float* vec1, float* vec2){
    float res = 0;
    for( int i = 0; i < VEC_SIZE; i+=8){
        __m256 a = _mm256_loadu_ps(&vec1[i]);
        __m256 b = _mm256_loadu_ps(&vec2[i]);

        __m256 s = _mm256_mul_ps(a,b);

        s = _mm256_hadd_ps(s, s);
        s = _mm256_hadd_ps(s, s);

        __m128 hi = _mm256_extractf128_ps(s,0);
        __m128 lo = _mm256_extractf128_ps(s,1);
        
        res += _mm_cvtss_f32(hi) + _mm_cvtss_f32(lo);
    }
    return res;
}

float exercice2_2_vec(float* vec1, float* vec2){
    float res = 0;
    __m256 resv = _mm256_set_ps(0, 0, 0, 0, 0, 0, 0, 0); 

    for (int i = 0; i < VEC_SIZE; i+=8) {    
        __m256 a = _mm256_loadu_ps(&vec1[i]);
        __m256 b = _mm256_loadu_ps(&vec2[i]);

        __m256 dotprod = _mm256_dp_ps(a, b, 241);

        resv = _mm256_add_ps(resv, dotprod);
    }    
    float t[8];
    _mm256_storeu_ps(t, resv);
    res = t[0] + t[4];
    return res;
}

float exercice2_scal(float* vec1, float* vec2){
    float res = 0;
    for ( int i = 0; i < VEC_SIZE; i++){
        res += vec1[i] * vec2[i];
    }
    return res;
}

minmax exercice3_vec(float* vec){
    minmax res = { 1000, -1000 };

    __m256 cmp_max = _mm256_loadu_ps(&vec[0]);
    __m256 cmp_min = _mm256_loadu_ps(&vec[0]);

    for ( int i = 8; i < VEC_SIZE; i +=8 ){    
        __m256 val = _mm256_loadu_ps(&vec[i]);
        cmp_max = _mm256_max_ps(cmp_max, val);
        cmp_min = _mm256_min_ps(cmp_min, val);
    }
    
    float minv[8];
    float maxv[8];
    _mm256_storeu_ps(&minv[0], cmp_min);
    _mm256_storeu_ps(&maxv[0], cmp_max);

    for (int i = 0; i < 8; i++ ) {
        if (minv[i] < res.min){
            res.min = minv[i];
        }
        if (maxv[i] > res.max){
            res.max = maxv[i];
        }
    }

    return res; 
}


minmax exercice3_scal(float* vec){
    minmax res = { 1000, -1000 };
    for (int i = 0; i < VEC_SIZE; i++ ) {
        if (vec[i] < res.min){
            res.min = vec[i];
        }
        if (vec[i] > res.max){
            res.max = vec[i];
        }
    }
    return res;
}

void exercice4_vec(float* src, float* res){
    __m256 a = _mm256_set_ps(src[6], src[5], src[4], src[3], src[2], src[1], src[0], 0.0);
    __m256 b = _mm256_loadu_ps(&src[0]);
    __m256 c = _mm256_loadu_ps(&src[1]);
    __m256 d = _mm256_set_ps(4,4,4,4,4,4,4,3);
    
    __m256 r = _mm256_add_ps(_mm256_add_ps(a,c), _mm256_add_ps(b,b));
    r = _mm256_div_ps(r,d);
    _mm256_storeu_ps(&res[0], r);

    for(int i = 8; i < VEC_SIZE-8; i+=8){
        a = _mm256_loadu_ps(&src[i-1]);
        b = _mm256_loadu_ps(&src[i]);
        c = _mm256_loadu_ps(&src[i+1]);
        d = _mm256_set1_ps(4);

        r = _mm256_add_ps(_mm256_add_ps(a,c), _mm256_add_ps(b,b));
        r = _mm256_div_ps(r,d);
        _mm256_storeu_ps(&res[i], r);
    }

    a = _mm256_loadu_ps(&src[VEC_SIZE-9]);
    b = _mm256_loadu_ps(&src[VEC_SIZE-8]);
    c = _mm256_set_ps(0.0, src[VEC_SIZE-1], src[VEC_SIZE-2], src[VEC_SIZE-3], src[VEC_SIZE-4], src[VEC_SIZE-5], src[VEC_SIZE-6], src[VEC_SIZE-7]);
    d = _mm256_set_ps(3,4,4,4,4,4,4,4);

    r = _mm256_add_ps(_mm256_add_ps(a,c), _mm256_add_ps(b,b));
    r = _mm256_div_ps(r,d);
    _mm256_storeu_ps(&res[VEC_SIZE-8], r);

}

void exercice4_scal(float* src, float* res){
    res[0] = (2*src[0] + src[1])/3;
    for ( int i = 1; i < VEC_SIZE-1; i++){
        res[i] = (src[i-1] + 2*src[i] + src[i+1])/4;
    }
    res[VEC_SIZE-1] = (src[VEC_SIZE-2] + 2*src[VEC_SIZE-1])/3;
}

void exercice5_vec(float* src, float* res){
    for( int i = 0; i < VEC_SIZE; i+=8){
        __m256 v = _mm256_loadu_ps(&src[i]);
        v = _mm256_andnot_ps(_mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)), v);
        _mm256_storeu_ps(&res[i], v);
    }
}

void exercice5_vec_bis(float* src, float* res){
    for( int i = 0; i < VEC_SIZE; i+=8){
        __m256 v = _mm256_loadu_ps(&src[i]);
        v = _mm256_max_ps(-v, v);
        _mm256_storeu_ps(&res[i], v);
    }
}

void exercice5_scal(float* src, float* res){
    for(int i = 0; i < VEC_SIZE; i++){
        int casted = *(int*)(&src[i]);
        casted &= 0x7fffffff;
        res[i] = *(float*)&casted;
    }
}


float rand_float(){
    return (float)((rand() % 360) - 180.0);
}