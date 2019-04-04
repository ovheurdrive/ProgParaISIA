#include "tdcuda.cuh"

using namespace std;

int main(int argc, char** argv){
    srand(time(NULL));
    run_exercices();

    return 0;
}

int run_exercices(){
    float *v1, *v2, *s_cpu, *s_gpu, *d_v1, *d_v2, *d_s;
    double start, end;
    double gpu_time, cpu_time;

    int N = VEC_SIZE/DIV_MAX;

    v1 = (float*)malloc(N*sizeof(float));
    v2 = (float*)malloc(N*sizeof(float));
    s_gpu = (float*)malloc(N*sizeof(float));
    s_cpu = (float*)malloc(N*sizeof(float));

    cudaMalloc(&d_v1, N*sizeof(float)); 
    cudaMalloc(&d_v2, N*sizeof(float));
    cudaMalloc(&d_s, N*sizeof(float));

    for (int i = 0; i < N; i++) {
        v1[i] = rand_float();
        v2[i] = rand_float();
    }


    // Exercice 2
    cout << "Exercice 2\n==============================" << endl;
    start = omp_get_wtime();
    for( int j = 0; j< MAX_ITER; j++){
        exercice2_cpu(2.0f,v1,v2,s_cpu,VEC_SIZE/DIV_MAX);
    }
    end = omp_get_wtime();
    cpu_time = (end-start)/(MAX_ITER);
    std::cout << "Cpu Time: " << cpu_time << std::endl;
    cudaMemcpy(d_v1, v1, (VEC_SIZE/DIV_MAX)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2, v2, (VEC_SIZE/DIV_MAX)*sizeof(float), cudaMemcpyHostToDevice);
    start = omp_get_wtime();
    for( int j = 0; j< MAX_ITER; j++){
        exercice2_cuda_nomemcpy(d_v1,d_v2,d_s,32,VEC_SIZE/DIV_MAX);
    }
    end = omp_get_wtime();
    cudaMemcpy(s_gpu, d_s, (VEC_SIZE/DIV_MAX)*sizeof(float), cudaMemcpyDeviceToHost);
    gpu_time = (end-start)/(MAX_ITER);
    std::cout << "Gpu Time (without memcpy): " << gpu_time << " Acceleration Ratio: " << cpu_time/gpu_time  << std::endl;

    start = omp_get_wtime();
    for( int j = 0; j< MAX_ITER; j++){
        exercice2_cuda_withmemcpy(d_v1,v1,d_v2,v2,d_s,s_gpu,32,VEC_SIZE/DIV_MAX);
    }
    end = omp_get_wtime();
    gpu_time = (end-start)/(MAX_ITER);
    std::cout << "Gpu Time: " << gpu_time << " Acceleration Ratio: " << cpu_time/gpu_time  << std::endl << std::endl;

    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_s);
    free(v1);
    free(v2);
    free(s_cpu);
    free(s_gpu);

    return 0;
}


void exercice2_cuda_withmemcpy(float* d_x, float* x, float* d_y, float* y, float* d_s, float* s, int k, int size){
    cudaMemcpy(d_x, x, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size*sizeof(float), cudaMemcpyHostToDevice);
    saxpy_kernel<<<(size+k)/k, k>>>(size, 2.0f, d_x, d_y, d_s);
    cudaMemcpy(s, d_s, size*sizeof(float), cudaMemcpyDeviceToHost);
}

void exercice2_cuda_nomemcpy(float* d_x, float* d_y, float* d_s, int k, int size){
    saxpy_kernel<<<(size+k)/k, k>>>(size, 2.0f, d_x, d_y, d_s);
    cudaDeviceSynchronize();
}

void exercice2_cpu(float a, float* x, float* y, float* s, int size){
    #pragma omp parallel  for num_threads(8) 
    for (int i=0; i<size; i++)
    {
        s[i] = a*x[i] + y[i];
    }
}


__global__ void saxpy_kernel(int n, float a, float *v1, float *v2, float *s){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if ( i < n ) s[i] = a*v1[i] + v2[i];
}

__global__ void mean_kernel(int n, float* v1, float* v2, float* res){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if( i < n ) res[i] = (v1[i] + v2[i])/2;
}

float rand_float(){
    return (float)((rand() % 360) - 180.0);
}