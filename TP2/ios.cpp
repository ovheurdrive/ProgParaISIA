#include "ios.hpp"
#include <iostream> // cout, cerr
#include <fstream> // ifstream
#include <sstream> // stringstream

#include "pixmap_io.hpp"

void get_source_params(std::string filename, unsigned int *height, unsigned int *width) {
    load_pixmap((char*)filename.c_str(), (int *)width, (int *)height);
}


int init_source_image(u_char **Source, std::string filename, int height, int width) {
    unsigned char *data;
    data = load_pixmap((char*)filename.c_str(), &width, &height);

    for (auto i = 0; i < height; i++) {
        for (auto j = 0; j < width; j++) {
            Source[i][j] = (char)data[i * width + j];
        }
    }   
    return 0;
}


int save_gray_level_image(image<u_char> *img_to_save, std::string filename, int height, int width) {

    auto nbBytes = (std::size_t)(width * height * 1); //image size in memory 
    unsigned char* data;
    data = (unsigned char *)std::malloc(nbBytes);
    
    for (auto i = 0; i < height; i++) {
        for (auto j = 0; j < width; j++) {
            data[i*width+j] =  img_to_save->handler[i][j];
        }
    }
    
    store_pixmap((char*)filename.c_str(), data, width, height);
}

