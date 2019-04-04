#include "histo.cuh"

__global__ void gpu_histo_kernel_naive(u_char** Source, int (*res)[256], unsigned height, unsigned width){

}

__global__ void gpu_histo_kernel_shared(u_char** Source, int (*res)[256], unsigned height, unsigned width){

}

void cpu_histo(u_char** Source, int (*res)[256], unsigned height, unsigned width){
    #pragma omp parallel for num_threads(8)
    for( int i = 0; i < height; i++){
        for( int j = 0; j < width; j++){
            (*res)[Source[i][j]]++;
        }
    }
}