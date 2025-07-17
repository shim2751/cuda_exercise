#include "ch3.h"

__global__
void imageBlurKernel(unsigned char* Pin, unsigned char* Pout, int width, int height, int r){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int ch = threadIdx.z;

    if(col < width && row < height){
        int pixelVal = 0; 
        int pixelNum = 0;
        for(int i=-r; i<r+1; i++){
            for(int j=-r; j<r+1; j++){
                int curCol = col + i;
                int curRow = row + j;
                if(0 < curCol && curCol < width && 0 < curRow && curRow < height){
                    pixelVal += Pin[(width * curRow + curCol) * CHANNEL + ch];
                    pixelNum++;
                }
            }
        }
        Pout[(width*row+col)*CHANNEL + ch] = pixelVal/pixelNum;
    }
    
}

void image_blur(unsigned char* Pin, unsigned char* Pout, int width, int height, int radius){
    unsigned char* Pin_d, *Pout_d;
    int size = width*height*sizeof(unsigned char)*CHANNEL;

    cudaMalloc((void **) &Pin_d, size);
    cudaMalloc((void **) &Pout_d, size);
    
    cudaMemcpy(Pin_d, Pin, size, cudaMemcpyHostToDevice);

    dim3 grid_dim(ceil(width/16.0), ceil(height/16.0), 1);
    dim3 block_dim(16, 16, 3);
    imageBlurKernel<<<grid_dim, block_dim>>>(Pin_d, Pout_d, width, height, radius);

    cudaMemcpy(Pout, Pout_d, size, cudaMemcpyDeviceToHost);

    cudaFree(Pin_d);
    cudaFree(Pout_d);
}