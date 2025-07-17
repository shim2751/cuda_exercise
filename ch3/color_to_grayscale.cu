#include "ch3.h"

__global__
void colorToGrayscaleKernel(unsigned char* Pin, unsigned char* Pout, int width, int height){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if(col < width && row < height){
        int grayOffset = width*row + col;

        int rgbOffset = grayOffset * CHANNEL;
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset+1];
        unsigned char b = Pin[rgbOffset+2];
        
        Pout[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

void color_to_grayscale(unsigned char* Pin, unsigned char* Pout, int width, int height){
    unsigned char* Pin_d, *Pout_d;
    int size = width*height*sizeof(unsigned char);

    cudaMalloc((void **) &Pin_d, size*CHANNEL);
    cudaMalloc((void **) &Pout_d, size);
    
    cudaMemcpy(Pin_d, Pin, size*CHANNEL, cudaMemcpyHostToDevice);

    dim3 grid_dim(ceil(width/16.0), ceil(height/16.0), 1);
    dim3 block_dim(16, 16, 1);
    colorToGrayscaleKernel<<<grid_dim, block_dim>>>(Pin_d, Pout_d, width, height);

    cudaMemcpy(Pout, Pout_d, size, cudaMemcpyDeviceToHost);

    cudaFree(Pin_d);
    cudaFree(Pout_d);
}