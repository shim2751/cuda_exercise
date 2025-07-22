
#ifndef CONV_H
#define CONV_H

#define FILTER_RADIUS 7
#define BLOCK_SIZE 32 
#define IN_TILE_WIDTH 32 
#define OUT_TILE_WIDTH (IN_TILE_WIDTH - 2*FILTER_RADIUS)

// Convolution kernel types
typedef enum {
    CONV_BASIC = 0,
    CONV_CONSTANT_MEM = 1,
    CONV_TILED = 2,
    CONV_CACHED_TILED = 3
} conv_kernel_t;

// Unified launch function
void launch_convolution2D(float* N_h, float* F_h, float* P_h, 
                         int r, int width, int height, conv_kernel_t kernel_type);

#endif