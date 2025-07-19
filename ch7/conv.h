
#ifndef CONV_H
#define CONV_H

#define FILTER_RADIUS 2
#define IN_TILE_WIDTH 32
#define OUT_TILE_WIDTH (IN_TILE_WIDTH - 2*FILTER_RADIUS)

void launch_convolution2D_basic(float* h_N, float* h_F, float* h_P, 
                         int r, int width, int height);

void launch_convolution2D_constant_mem(float* N_h, float* F_h, float* P_h, 
                        int r, int width, int height);

#endif