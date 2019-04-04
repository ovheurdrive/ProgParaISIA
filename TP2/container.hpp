#pragma once

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <memory>
#include "global_parameters.hpp"
#include <omp.h>

template<typename T>
class image
{
public:
	image();
    image(unsigned int height, unsigned int width, T ***ptr);
    void init_table_2D();
	~image();


    std::size_t nbBytes;
    std::size_t nbLines;
    T *DataStorage;
    T **handler;

    unsigned int width;
    unsigned int height;

    
};

template<typename T> 
image<T>::image(){}

template<typename T> 
image<T>::image(unsigned int height, unsigned int width, T ***ptr){
    image::width = width;
    image::height = height;
    nbBytes = (std::size_t)(width * height * sizeof(T)); //image size in memory 
	nbLines = (std::size_t)((height) * sizeof(T*)); //line indexes size in memory 
    DataStorage  = (T *)std::malloc(nbBytes); // The whole image
    handler = (T **)std::malloc(nbLines); // The line indexes

    for (auto i = 0; i < height; i++) { // affect each index with the start adress of its corresponding line
        handler[i] = &DataStorage[0+i*width];
    }
    *ptr = handler;
}

template<typename T> 
void image<T>::init_table_2D() {
    auto PU = omp_get_num_threads();
    #pragma omp parallel for
    for (auto k = 0; k < PU; k++) {
    	auto delta = k*height/PU;
	    for (auto i = 0; i < height/PU; i++) {
		    for (auto j = 0; j < width; j++) {
			    handler[delta+i][j] = static_cast<u_char>(0);
		    }
	    }
    }
}

template<typename T> 
image<T>::~image()
{
    std::free(DataStorage);
    std::free(handler);
}
