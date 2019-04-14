#include "utils.h"

void display_vec(void* vec, int size, enum g_type id){
    switch(id){
        case g_float:
            std::cout << "[";
            for(int i = 0; i < size-1; i++)
                std::cout << ((float*)vec)[i] << ", ";
            std::cout << ((float*)vec)[size-1] << "]" << std::endl;
            break;
        case g_int:
            std::cout << "[";
            for(int i = 0; i < size-1; i++)
                std::cout << ((int*)vec)[i] << ", ";
            std::cout << ((int*)vec)[size-1] << "]" << std::endl;
            break;
        default:
            std::cout << "Unsupported vec type" << std::endl;
            break;
    }
    
}

void display_timer(timer times){
    // Cpu time
    std::cout << "Cpu time: " << times.cpu_time << std::endl;
    
    // Gpu kernel time
    std::cout << "Gpu time (kernel): " << times.gpu_time_kernel;
    std::cout << " Acceleration Ratio: " << times.cpu_time/times.gpu_time_kernel;
    std::cout << std::endl;
    
    // Gpu total time
    std::cout << "Gpu time (total): " << times.gpu_time_total;
    std::cout << " Acceleration Ratio: " << times.cpu_time/times.gpu_time_total;
    std::cout << std::endl;
}

void timer_avg(timer* p_timer, int iter, int size){
    p_timer->gpu_time_kernel /= iter;
    p_timer->gpu_time_total /= iter;
    p_timer->cpu_time /= iter;
    p_timer->gpu_time_kernel /= size;
    p_timer->gpu_time_total /= size;
    p_timer->cpu_time /= size;
}

float rand_float(){
    return (float)((rand() % 360) - 180.0);
}