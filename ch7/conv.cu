
#include "conv.h"
#include <cstdio>
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
void convolution2D_constant_mem_kernel(float* N, float* P, int width, int height){
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
void convolution2D_tiled_constant_mem_kernel(float* N, float* P, int width, int height){
    int col = OUT_TILE_WIDTH * blockIdx.x + threadIdx.x - FILTER_RADIUS;
    int row = OUT_TILE_WIDTH * blockIdx.y + threadIdx.y - FILTER_RADIUS;

    __shared__ float N_s[IN_TILE_WIDTH][IN_TILE_WIDTH];
    if(row >= 0 && row < height && col >= 0 && col < width)
        N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
    else
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads();

    if(row >= 0 && row < height && col >= 0 && col < width){
        int tileCol = threadIdx.x - FILTER_RADIUS;
        int tileRow = threadIdx.y - FILTER_RADIUS;
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

__global__
void convolution2D_cached_tiled_constant_mem_kernel(float* N, float* P, int width, int height){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float N_s[IN_TILE_WIDTH][IN_TILE_WIDTH];
    if(row < height && col < width)
        N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
    else
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads();

    if(row >= 0 && row < height && col >= 0 && col < width){
        float Pval = 0.0;
        for(int fRow=0; fRow<2*FILTER_RADIUS+1; fRow++){
            for(int fCol=0; fCol<2*FILTER_RADIUS+1; fCol++){
                int iCol = threadIdx.x - FILTER_RADIUS + fCol;
                int iRow = threadIdx.y - FILTER_RADIUS + fRow;

                if (iCol >= 0 && iCol < IN_TILE_WIDTH && iRow >= 0 && iRow < IN_TILE_WIDTH ){
                    Pval += N_s[iRow][iCol] * F_c[fRow][fCol];      
                }
                else{
                    iCol = col - FILTER_RADIUS + fCol;
                    iRow = row - FILTER_RADIUS + fRow;
                    if (iCol >= 0 && iCol < width && iRow >= 0 && iRow < height )
                        Pval += N[iRow*width + iCol] * F_c[fRow][fCol];  
                }
            }
        }
        P[row * width + col] = Pval;
    }
}

// Unified launch function
void launch_convolution2D(float* N_h, float* F_h, float* P_h, 
                         int r, int width, int height, conv_kernel_t kernel_type) {
    int size = width * height * sizeof(float);
    int filter_size = (2*r+1) * (2*r+1) * sizeof(float);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Device memory allocation
    float *N_d, *F_d, *P_d;
    cudaMalloc(&N_d, size);
    if (kernel_type == CONV_BASIC) {
        cudaMalloc(&F_d, filter_size);
    }
    cudaMalloc(&P_d, size);
    
    // Copy to device
    cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);
    if (kernel_type == CONV_BASIC) {
        cudaMemcpy(F_d, F_h, filter_size, cudaMemcpyHostToDevice);
    } else {
        // Copy filter to constant memory for other kernels
        cudaMemcpyToSymbol(F_c, F_h, filter_size);
    }
    
    // Record start time
    cudaEventRecord(start);
    
    // Launch appropriate kernel based on type
    switch(kernel_type) {
        case CONV_BASIC: {
            dim3 grid_dim(ceil(width/(float)BLOCK_SIZE), ceil(height/(float)BLOCK_SIZE), 1);
            dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);
            convolution2D_basic_kernel<<<grid_dim, block_dim>>>(N_d, F_d, P_d, r, width, height);
            break;
        }
        case CONV_CONSTANT_MEM: {
            dim3 grid_dim(ceil(width/(float)BLOCK_SIZE), ceil(height/(float)BLOCK_SIZE), 1);
            dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);
            convolution2D_constant_mem_kernel<<<grid_dim, block_dim>>>(N_d, P_d, width, height);
            break;
        }
        case CONV_TILED: {
            dim3 grid_dim(ceil(width/(float)OUT_TILE_WIDTH), ceil(height/(float)OUT_TILE_WIDTH), 1);
            dim3 block_dim(IN_TILE_WIDTH, IN_TILE_WIDTH, 1);
            convolution2D_tiled_constant_mem_kernel<<<grid_dim, block_dim>>>(N_d, P_d, width, height);
            break;
        }
        case CONV_CACHED_TILED: {
            dim3 grid_dim(ceil(width/(float)IN_TILE_WIDTH), ceil(height/(float)IN_TILE_WIDTH), 1);
            dim3 block_dim(IN_TILE_WIDTH, IN_TILE_WIDTH, 1);
            convolution2D_cached_tiled_constant_mem_kernel<<<grid_dim, block_dim>>>(N_d, P_d, width, height);
            break;
        }
        default: {
            dim3 grid_dim(ceil(width/(float)BLOCK_SIZE), ceil(height/(float)BLOCK_SIZE), 1);
            dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);
            convolution2D_basic_kernel<<<grid_dim, block_dim>>>(N_d, F_d, P_d, r, width, height);
            break;
        }
    }
    
    // Record stop time
    cudaEventRecord(stop);
    
    // Wait for kernel to complete
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Print timing information
    const char* kernel_names[] = {
        "Basic GPU",
        "Constant Memory",
        "Tiled",
        "Cached Tiled"
    };
    printf("[%s] Kernel execution time: %.3f ms\n", 
           kernel_names[kernel_type], milliseconds);
    
    // Copy result back
    cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(N_d);
    if (kernel_type == CONV_BASIC) {
        cudaFree(F_d);
    }
    cudaFree(P_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}