#pragma once

#include <vector>
#include "container.cuh"

void get_source_params(std::string filename, unsigned int *height, unsigned int *width);
int init_source_image(u_char **Source, std::string filename, int height, int width) ;
void save_gray_level_image(image<u_char> *img_to_save, std::string filename, int height, int width);
// int save_gray_level_image(image<u_char> *img_to_save, std::string filename);