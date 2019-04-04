#include "main.cuh"

int main(int argc, char** argv){
    std::string image_filename("images/Carre.pgm");
    int iter = 100;
    if ( argc > 2 ) { 
        image_filename = argv[1];
        iter = atoi(argv[2]);
    }
    srand(time(NULL));

    unsigned width, height;
    get_source_params(image_filename, &height, &width);
    std::cout << "Image Dimensions: " <<  width << "x" << height << std::endl;
    u_char **Source;
    image<u_char> imgSource(height, width, &Source);

    auto fail = init_source_image(Source, image_filename, height, width);
    if (fail) {
        std::cout << "Chargement impossible de l'image" << std::endl;
        return 0;
    }

    int size = 1024*1024;
    float *vec1, *vec2;
    vec1 = (float*)malloc(size*sizeof(float));
    vec2 = (float*)malloc(size*sizeof(float));

    for (int i = 0; i < size; i++) {
        vec1[i] = rand_float();
        vec2[i] = rand_float();
    }

    // Exercice 1
    std::cout << "Exercice 1 : Dot Product\n====================================" << std::endl;
    run_exercice1(vec1,vec2, size, BLOCKDIM_X, iter);

    std::cout << "\nExercice 2 : Sobel Filter\n====================================" << std::endl;
    run_exercice2(Source, width, height, iter);

    std::cout << "\nExercice 3 : Matrix Transposition\n====================================" << std::endl;
    run_exercice3(Source, width, height, iter);

    std::cout << "\nExercice 4: Image Histogram\n====================================" << std::endl;
    run_exercice4(Source, height, width, iter);

    free(vec1);
    free(vec2);

}

void run_exercice1(float* vec1, float* vec2, int size, int k, int iter){
    timer avg_times = { 0 };
    timer times = { 0 };
    float res_cpu = 0;
    float res_gpu = 0;

    float *d_vec1CUDA, *d_vec2CUDA, *d_resCUDA;

    for( int i = 0; i < iter; i++){
        cudaMalloc(&d_vec1CUDA, size*sizeof(float)); 
        cudaMalloc(&d_vec2CUDA, size*sizeof(float));
        cudaMalloc(&d_resCUDA, sizeof(float));
        // GPU Benchmark
        times.gpu_time_total = omp_get_wtime();
        cudaMemcpy(d_vec1CUDA, vec1, size*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vec2CUDA, vec2, size*sizeof(float), cudaMemcpyHostToDevice);
        
        // Kernel benchmark
        times.gpu_time_kernel = omp_get_wtime();
        gpu_dotp_kernel<<<(size+k)/k, k>>>(size, d_vec1CUDA, d_vec2CUDA, d_resCUDA);
        times.gpu_time_kernel = omp_get_wtime() - times.gpu_time_kernel;
        avg_times.gpu_time_kernel += times.gpu_time_kernel;
        cudaMemcpy(&res_gpu, d_resCUDA, sizeof(float), cudaMemcpyDeviceToHost);
        times.gpu_time_total = omp_get_wtime() - times.gpu_time_total;
        avg_times.gpu_time_total += times.gpu_time_total;

        // CPU Benchmark
        times.cpu_time =  omp_get_wtime();
        res_cpu = cpu_dotp(vec1,vec2,size);
        times.cpu_time = omp_get_wtime() - times.cpu_time;
        avg_times.cpu_time += times.cpu_time;
    }
    // Check
    if(res_cpu != res_gpu){
        std::cout << "Cpu dot prod: " << res_cpu << std::endl;
        std::cout << "Gpu dot prod: " << res_gpu << std::endl;
        std::cout << "Absolute Error: " << fabs(res_cpu-res_gpu) << std::endl;
        std::cout << "Error: dot product result different" << std::endl;
    }
    timer_avg(&avg_times, iter, size);
    display_timer(avg_times);
}

void run_exercice2(u_char** Source, unsigned width, unsigned height, int iter){
    u_char** ResultatCPU, **ResultatGPU, **ResultatGPUShared;
    u_char *d_ResultatCUDAShared, *d_ResultatCUDA, *d_SourceCUDA;

    image<u_char> imgResultatCPU(height, width, &ResultatCPU);
    image<u_char> imgResultatGPU(height, width, &ResultatGPU);
    image<u_char> imgResultatGPUShared(height, width, &ResultatGPUShared);
    
    timer times_naive = { 0 };
    timer times_shared = { 0 };
    timer avg_times_naive = { 0 };
    timer avg_times_shared = { 0 };

    for( int i = 0; i < iter; i++){
        cudaMalloc(&d_SourceCUDA, height*width*sizeof(u_char));    
        cudaMalloc(&d_ResultatCUDA, height*width*sizeof(u_char));    
        cudaMalloc(&d_ResultatCUDAShared, height*width*sizeof(u_char)); 

        dim3 threads(BLOCKDIM_X,BLOCKDIM_Y);
        dim3 blocks(width/BLOCKDIM_X,height/BLOCKDIM_Y);

        // GPU Naive Benchmark
        times_naive.gpu_time_total = omp_get_wtime();
        cudaMemcpy(d_SourceCUDA, Source[0], height*width*sizeof(u_char), cudaMemcpyHostToDevice);
        times_naive.gpu_time_kernel = omp_get_wtime();
        gpu_sobel_kernel_naive<<<blocks,threads>>>(d_SourceCUDA, d_ResultatCUDA, width, height);
        times_naive.gpu_time_kernel = omp_get_wtime() - times_naive.gpu_time_kernel;
        avg_times_naive.gpu_time_kernel += times_naive.gpu_time_kernel;
        cudaMemcpy(ResultatGPU[0], d_ResultatCUDA, height*width*sizeof(u_char), cudaMemcpyDeviceToHost);
        times_naive.gpu_time_total = omp_get_wtime() - times_naive.gpu_time_total;
        avg_times_naive.gpu_time_total += times_naive.gpu_time_total;

        dim3 blocks2(width/(BLOCKDIM_X-2),height/(BLOCKDIM_Y-2));

        // GPU Shared Benchmark
        times_shared.gpu_time_total = omp_get_wtime();
        cudaMemcpy(d_SourceCUDA, Source[0], height*width*sizeof(u_char), cudaMemcpyHostToDevice);
        times_shared.gpu_time_kernel = omp_get_wtime();
        gpu_sobel_kernel_shared<<<blocks2,threads>>>(d_SourceCUDA, d_ResultatCUDAShared, width, height);
        times_shared.gpu_time_kernel = omp_get_wtime() - times_shared.gpu_time_kernel;
        avg_times_shared.gpu_time_kernel += times_shared.gpu_time_kernel;
        cudaMemcpy(ResultatGPUShared[0], d_ResultatCUDAShared, height*width*sizeof(u_char), cudaMemcpyDeviceToHost);
        times_shared.gpu_time_total = omp_get_wtime() - times_shared.gpu_time_total;
        avg_times_shared.gpu_time_total += times_shared.gpu_time_total;

        // CPU Benchmark
        times_naive.cpu_time = omp_get_wtime();
        cpu_sobel(Source, ResultatCPU, width, height);
        times_naive.cpu_time = omp_get_wtime() - times_naive.cpu_time;
        times_shared.cpu_time = times_naive.cpu_time;
        avg_times_naive.cpu_time += times_naive.cpu_time;
        avg_times_shared.cpu_time += times_shared.cpu_time;
    }

    std::string image_filename=std::string("images/Resultats/Sobel_cpu.pgm");
    save_gray_level_image(&imgResultatCPU, image_filename, height, width);
    image_filename=std::string("images/Resultats/Sobel_gpu.pgm");
    save_gray_level_image(&imgResultatGPU, image_filename, height, width);
    image_filename=std::string("images/Resultats/Sobel_gpu_shared.pgm");
    save_gray_level_image(&imgResultatGPUShared, image_filename, height, width);

    std::cout << std::endl << "Naive GPU Algorithm (not coalescent):" << std::endl;
    timer_avg(&avg_times_naive, iter, height*width);
    display_timer(avg_times_naive);
    std::cout << std::endl << "Shared GPU Algorithm (coalescent):" << std::endl;
    timer_avg(&avg_times_shared, iter, height*width);
    display_timer(avg_times_shared);

}

void run_exercice3(u_char** Source, unsigned width, unsigned height, int iter){
    u_char** ResultatCPU, **ResultatGPU, **ResultatGPUShared;
    u_char *d_ResultatCUDAShared, *d_ResultatCUDA, *d_SourceCUDA;

    image<u_char> imgResultatCPU(width, height, &ResultatCPU);
    image<u_char> imgResultatGPU(width, height, &ResultatGPU);
    image<u_char> imgResultatGPUShared(width, height, &ResultatGPUShared);

    timer times_naive = { 0 };
    timer times_shared = { 0 };
    timer avg_times_naive = { 0 };
    timer avg_times_shared = { 0 };

    for( int i = 0; i < iter; i++){
        cudaMalloc(&d_SourceCUDA, height*width*sizeof(u_char));    
        cudaMalloc(&d_ResultatCUDA, height*width*sizeof(u_char));    
        cudaMalloc(&d_ResultatCUDAShared, height*width*sizeof(u_char));

        dim3 threads(BLOCKDIM_X,BLOCKDIM_Y);
        dim3 blocks(width/BLOCKDIM_X,height/BLOCKDIM_Y);
        
        // GPU Naive Benchmark
        times_naive.gpu_time_total = omp_get_wtime();
        cudaMemcpy(d_SourceCUDA, Source[0], height*width*sizeof(u_char), cudaMemcpyHostToDevice);
        times_naive.gpu_time_kernel = omp_get_wtime();
        gpu_transpo_kernel_naive<<<blocks,threads>>>(d_SourceCUDA, d_ResultatCUDA, width, height);
        times_naive.gpu_time_kernel = omp_get_wtime() - times_naive.gpu_time_kernel;
        avg_times_naive.gpu_time_kernel += times_naive.gpu_time_kernel;
        cudaMemcpy(ResultatGPU[0], d_ResultatCUDA, height*width*sizeof(u_char), cudaMemcpyDeviceToHost);
        times_naive.gpu_time_total = omp_get_wtime() - times_naive.gpu_time_total;
        avg_times_naive.gpu_time_total += times_naive.gpu_time_total;

        dim3 blocks2(width/(BLOCKDIM_X-2),height/(BLOCKDIM_Y-2));

        // GPU Shared Benchmark
        times_shared.gpu_time_total = omp_get_wtime();
        cudaMemcpy(d_SourceCUDA, Source[0], height*width*sizeof(u_char), cudaMemcpyHostToDevice);
        times_shared.gpu_time_kernel = omp_get_wtime();
        gpu_transpo_kernel_shared<<<blocks2,threads>>>(d_SourceCUDA, d_ResultatCUDAShared, width, height);
        times_shared.gpu_time_kernel = omp_get_wtime() - times_shared.gpu_time_kernel;
        avg_times_shared.gpu_time_kernel += times_shared.gpu_time_kernel;
        cudaMemcpy(ResultatGPUShared[0], d_ResultatCUDAShared, height*width*sizeof(u_char), cudaMemcpyDeviceToHost);
        times_shared.gpu_time_total = omp_get_wtime() - times_shared.gpu_time_total;
        avg_times_shared.gpu_time_total += times_shared.gpu_time_total;

        // CPU Benchmark
        times_naive.cpu_time = omp_get_wtime();
        cpu_transpo(Source, ResultatCPU, width, height);
        times_naive.cpu_time = omp_get_wtime() - times_naive.cpu_time;
        times_shared.cpu_time = times_naive.cpu_time;
        avg_times_naive.cpu_time += times_naive.cpu_time;
        avg_times_shared.cpu_time += times_shared.cpu_time;
    }

    std::string image_filename=std::string("images/Resultats/Transpo_cpu.pgm");
    save_gray_level_image(&imgResultatCPU, image_filename, width, height);
    image_filename=std::string("images/Resultats/Transpo_gpu.pgm");
    save_gray_level_image(&imgResultatGPU, image_filename, width, height);
    image_filename=std::string("images/Resultats/Transpo_gpu_shared.pgm");
    save_gray_level_image(&imgResultatGPUShared, image_filename, width, height);

    std::cout << std::endl << "Naive GPU Algorithm (not coalescent):" << std::endl;
    timer_avg(&avg_times_naive, iter, height*width);
    display_timer(avg_times_naive);
    std::cout << std::endl << "Shared GPU Algorithm (coalescent):" << std::endl;
    timer_avg(&avg_times_shared, iter, height*width);
    display_timer(avg_times_shared);
}

void run_exercice4(u_char** Source, unsigned height, unsigned width, int iter) {
    timer times = { 0 };
    int resCPU[256] = { 0 };
    int resGPU[256] = { 0 };
    int resGPUShared[256] = { 0 };

    u_char* d_SourceCUDA;
    int* d_resCUDA, *d_resCUDAShared;

    cudaMalloc(&d_SourceCUDA, height*width*sizeof(u_char));    
    cudaMalloc(&d_resCUDA, height*width*sizeof(u_char));    
    cudaMalloc(&d_resCUDAShared, height*width*sizeof(u_char));

    times.cpu_time = omp_get_wtime();
    cpu_histo(Source, &resCPU, height, width);
    times.cpu_time = omp_get_wtime() - times.cpu_time;

    display_timer(times);
    display_vec(resCPU,256,g_int);
    display_vec(resGPU,256,g_int);
    display_vec(resGPUShared,256,g_int);
}