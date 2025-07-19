
#include "conv.h"
__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

__global__
void convolution2D_basic_kernel(float* N, float* F, float* P, int r, int width, int height){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int inRow, inCol;

    if (row >= height || col >= width) return;

    float Pval = 0.0;
    for(int i=0; i<2*r+1; i++){
        for(int j=0; j<2*r+1; j++){
            inRow = row-r+i;
            inCol = col-r+j;
            if (inRow >= 0 && inRow<height && inCol >= 0 && inCol < width)
                Pval += N[inRow*width + inCol] * F[i*(2*r+1) + j];
        }
    }
    P[row * width + col] = Pval;
}

__global__
void convolution2D_constant_mem(float* N, float* P, int width, int height){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int inRow, inCol;

    if (row >= height || col >= width) return;

    float Pval = 0.0;
    for(int i=0; i<2*FILTER_RADIUS+1; i++){
        for(int j=0; j<2*FILTER_RADIUS+1; j++){
            inRow = row-FILTER_RADIUS+i;
            inCol = col-FILTER_RADIUS+j;
            if (inRow >= 0 && inRow<height && inCol >= 0 && inCol < width)
                Pval += N[inRow*width + inCol] * F_c[i][j];
        }
    }
    P[row * width + col] = Pval;
}
__global__
void convolution2D_tiled(float* N, float* P, int width, int height){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float N_s[IN_TILE_WIDTH][IN_TILE_WIDTH];
    if(row >= 0 && row < height && col >= 0 && col < width)
        N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
    else
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads();

    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    if(row >= 0 && row < height && col >= 0 && col < width){
        if(tileRow >= 0 && tileRow < OUT_TILE_WIDTH && tileCol >= 0 && tileCol < OUT_TILE_WIDTH){
            float Pval = 0.0;
            for(int i=0; i<2*FILTER_RADIUS+1; i++){
                for(int j=0; j<2*FILTER_RADIUS+1; j++){
                    Pval += N_s[tileRow+i][tileCol+j] * F_c[i][j];                        
                }
            }
            P[row * width + col] = Pval;
        }
    }

}

void launch_convolution2D_basic(float* N_h, float* F_h, float* P_h, 
                        int r, int width, int height) {
   int size = width * height * sizeof(float);
   int filter_size = (2*r+1) * (2*r+1) * sizeof(float);
   
   // Device memory allocation
   float *N_d, *F_d, *P_d;
   cudaMalloc(&N_d, size);
   cudaMalloc(&F_d, filter_size);
   cudaMalloc(&P_d, size);
   
   // Copy to device
   cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);
   cudaMemcpy(F_d, F_h, filter_size, cudaMemcpyHostToDevice);
   
   // Launch kernel
   dim3 grid_dim(ceil(width/16.0), ceil(height/16.0), 1);
   dim3 block_dim(16, 16, 1);

   convolution2D_basic_kernel<<<grid_dim, block_dim>>>(N_d, F_d, P_d, r, width, height);
   
   // Copy result back
   cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);
   
   // Cleanup
   cudaFree(N_d);
   cudaFree(F_d);
   cudaFree(P_d);
}

void launch_convolution2D_constant_mem(float* N_h, float* F_h, float* P_h, 
                        int r, int width, int height) {
   int size = width * height * sizeof(float);
   int filter_size = (2*FILTER_RADIUS+1) * (2*FILTER_RADIUS+1) * sizeof(float);
   
   // Device memory allocation
   float *N_d, *P_d;
   cudaMalloc(&N_d, size);
   cudaMalloc(&P_d, size);
   
   // Copy to device
   cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);
   
   cudaMemcpyToSymbol(F_c, F_h, filter_size);

   // Launch kernel
   dim3 grid_dim(ceil(width/16.0), ceil(height/16.0), 1);
   dim3 block_dim(16, 16, 1);

   convolution2D_constant_mem<<<grid_dim, block_dim>>>(N_d, P_d, width, height);
   
   // Copy result back
   cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);
   
   // Cleanup
   cudaFree(N_d);
   cudaFree(P_d);
}

void launch_convolution2D_tiled(float* N_h, float* F_h, float* P_h, 
                        int r, int width, int height) {
   int size = width * height * sizeof(float);
   int filter_size = (2*FILTER_RADIUS+1) * (2*FILTER_RADIUS+1) * sizeof(float);
   
   // Device memory allocation
   float *N_d, *P_d;
   cudaMalloc(&N_d, size);
   cudaMalloc(&P_d, size);
   
   // Copy to device
   cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);
   
   cudaMemcpyToSymbol(F_c, F_h, filter_size);

   // Launch kernel
   dim3 grid_dim(ceil(width/16.0), ceil(height/16.0), 1);
   dim3 block_dim(16, 16, 1);

   convolution2D_tiled<<<grid_dim, block_dim>>>(N_d, P_d, width, height);
   
   // Copy result back
   cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);
   
   // Cleanup
   cudaFree(N_d);
   cudaFree(P_d);
}