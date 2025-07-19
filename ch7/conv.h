
#ifndef CONV_H
#define CONV_H


void launch_convolution2D_basic(float* h_N, float* h_F, float* h_P, 
                         int r, int width, int height);

#define FILTER_RADIUS 2
__constant__ float F[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

#endif