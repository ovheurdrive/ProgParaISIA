
#include "openmp.h"

int main(){
    // Setup
    double* vec1 = (double*)malloc(VEC_SIZE*sizeof(double));
    double* vec2 = (double*)malloc(VEC_SIZE*sizeof(double));
    double* res1 = (double*)calloc(VEC_SIZE, sizeof(double));
    double* res2 = (double*)calloc(VEC_SIZE, sizeof(double));
    double* res6 = (double*)calloc(VEC_SIZE, sizeof(double));
    double* res7 = (double*)calloc(VEC_SIZE, sizeof(double));

    for( int i = 0; i < VEC_SIZE; i++) {
        vec1[i] = rand_double();
        vec2[i] = rand_double();
    }
    double start, end;

    /* Exercice 1 */
    std::cout << "Exercice 1:\n====================\n";
    exercice_1();
    std::cout << std::endl << std::endl;

    /* Exercice 2 */

    std::cout << "Exercice 2:\n====================\n";

    start = omp_get_wtime();
    for( int j = 0; j< MAX_ITER; j++){
        exercice2_omp(vec1,vec2,res1);
    }
    end = omp_get_wtime();
    std::cout << "OMP Time: " <<  (end-start)/(MAX_ITER) << std::endl;

    start = omp_get_wtime();
    for( int j = 0; j< MAX_ITER; j++){
        exercice2_seq(vec1,vec2,res2);
    }
    end = omp_get_wtime();
    std::cout << "Seq Time: " <<  (end-start)/(MAX_ITER) << std::endl << std::endl;
    for(int i = 0; i< VEC_SIZE; i++){
        if(res1[i] != res2[i]){
            std::cout << "Vector different" << std::endl;
            return 1;
        }
    }
    std::cout << "OK" << std::endl;

    /* Exercice 3 */

    std::cout << "Exercice 3:\n====================\n";
    double res3 = 0;
    double res4 = 0;
    double res5 = 0;
    start = omp_get_wtime();
    for( int j = 0; j< MAX_ITER; j++){
        res3 = exercice3_omp_no_red(vec1,vec2);
    }
    end = omp_get_wtime();
    std::cout << "OMP Time ( no reduction ): " <<  (end-start)/(MAX_ITER) << std::endl;

    start = omp_get_wtime();
    for( int j = 0; j< MAX_ITER; j++){
        res4 = exercice3_omp_red(vec1,vec2);
    }
    end = omp_get_wtime();
    std::cout << "OMP Time ( with reduction ): " <<  (end-start)/(MAX_ITER) << std::endl;

    start = omp_get_wtime();
    for( int j = 0; j< MAX_ITER; j++){
        res5 = exercice3_seq(vec1,vec2);
    }
    end = omp_get_wtime();
    std::cout << "Seq Time: " <<  (end-start)/(MAX_ITER) << std::endl << std::endl;
    if(res3 != res5 || res4 != res5){
        std::cout << "Dot prod different" << std::endl;
        return 1;
    }
    std::cout << "OK" << std::endl;

    /* Exercice 4 */

    std::cout << "Exercice 4:\n====================\n";

    start = omp_get_wtime();
    for( int j = 0; j< MAX_ITER; j++){
        exercice4_omp(vec1, res7);
    }
    end = omp_get_wtime();
    std::cout << "OMP Time: " <<  (end-start)/(MAX_ITER) << std::endl;

    start = omp_get_wtime();
    for( int j = 0; j< MAX_ITER; j++){
        exercice4_seq(vec1,res6);
    }
    end = omp_get_wtime();
    std::cout << "Seq Time: " <<  (end-start)/(MAX_ITER) << std::endl << std::endl;
    for(int i = 0; i< (WIDTH-1)*(WIDTH-1); i++){
        if(res6[i] != res7[i]){
            std::cout << "Vector different" << std::endl;
            return 1;
        }
    }
    std::cout << "OK" << std::endl;

    free(vec1);
    free(vec2);
    free(res1);
    free(res2);
    free(res6);
    free(res7);
    return 0;
}

void exercice_1(){
    #pragma omp parallel
    {
        int i = omp_get_thread_num();
        std::cout << "Thread " << i << std::endl;
    }

    #pragma omp parallel
    {
        int i = omp_get_thread_num();
        #pragma omp critical
        std::cout << "Thread " << i << std::endl;
    }
}

void exercice2_omp(double* vec1, double* vec2, double* res){
    omp_set_num_threads(4);
    #pragma omp parallel for
    for ( int i = 0; i < VEC_SIZE; i++){
        res[i] = (vec1[i] + vec2[i])/2;
    }
}

void exercice2_seq(double* vec1, double* vec2, double* res){
    for ( int i = 0; i < VEC_SIZE; i++){
        res[i] = (vec1[i] + vec2[i])/2;
    }
}

double exercice3_omp_no_red(double* vec1, double* vec2){
    double res[8] = {0};
    #pragma omp parallel for
    for ( int k = 0; k < 8; k++ ){
        for ( int i = 0; i < VEC_SIZE/8; i++){
            res[k] += vec1[i+k*VEC_SIZE/8] * vec2[i+k*VEC_SIZE/8];
        }
    }
    return res[0]+res[1]+res[2]+res[3]+res[4]+res[5]+res[6]+res[7];
    
}

double exercice3_omp_red(double* vec1, double* vec2){
    double res = 0;
    #pragma omp parallel
    {
        #pragma omp for reduction(+:res) schedule(auto)
        for ( int i = 0; i < VEC_SIZE; i++){
                res += vec1[i] * vec2[i];
        }
    }
    return res;
}

double exercice3_seq(double* vec1, double* vec2){
    double res = 0;
    for ( int i = 0; i < VEC_SIZE; i++){
        res += vec1[i] * vec2[i];
    }
    return res;
}

void exercice4_omp(double* vec, double* res){
    double kernely[3][3] = {{1,1,1},{0,0,0},{-1,-1,-1}};
    double kernelx[3][3] = {{1,0,-1},{1,0,-1},{1,0,-1}};
    #pragma omp parallel for
    for(int x = 1; x<WIDTH-1; x++){
        for(int y = 1; y<WIDTH-1; y++){
            for(int a = 0; a < 3; a++)
            {
                for(int b = 0; b < 3; b++)
                {            
                    int xn = x + a - 1;
                    int yn = y + b - 1;

                    int index = xn + yn * WIDTH;
                    res[index] += vec[index] * kernelx[a][b];
                }
            }
        }
    }
}

void exercice4_seq(double* vec, double* res){
    double kernely[3][3] = {{1,1,1},{0,0,0},{-1,-1,-1}};
    double kernelx[3][3] = {{1,0,-1},{1,0,-1},{1,0,-1}};
    for(int x = 1; x<WIDTH-1; x++){
        for(int y = 1; y<WIDTH-1; y++){
            for(int a = 0; a < 3; a++)
            {
                for(int b = 0; b < 3; b++)
                {            
                    int xn = x + a - 1;
                    int yn = y + b - 1;

                    int index = xn + yn * WIDTH;
                    res[index] += vec[index] * kernelx[a][b];
                }
            }
        }
    }
}

double rand_double(){
    return (double)((rand() % 360) - 180.0);
}