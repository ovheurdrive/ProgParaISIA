#include "dotp.cuh"

__global__ void gpu_dotp_kernel(int size, float* vec1, float* vec2, float* res){

    float cache = 0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i < size ){
        cache = vec1[i]*vec2[i];
    }

    atomicAdd(res, cache);
}

float cpu_dotp(float* vec1, float* vec2, int size){
    int res = 0;
    #pragma omp parallel num_threads(8)
    {
        #pragma for reduction(+:res) schedule(auto)
        for( int i = 0; i < size; i++ ){
            res+= vec1[i]*vec2[i];
        }
    }
    return res;
}