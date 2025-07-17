#include "ch5.h"

__global__
void matrixMulKernel_ch3(float* M, float* N, float* P, int width){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    
    if(col < width && row < width) {
        float Pval = 0;
        for(int i=0; i < width; i++){
            Pval += M[row*width + i] * N[i*width + col];
        }
        P[row*width+col] = Pval;
    }
}

void matrix_mul_ch3(float* M, float* N, float* P, int width){
    float *M_d, *N_d, *P_d;
    int size = width * width * sizeof(float);

    cudaMalloc(&M_d, size);
    cudaMalloc(&N_d, size);
    cudaMalloc(&P_d, size);

    cudaMemcpy(M_d, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N, size, cudaMemcpyHostToDevice);

    dim3 grid_dim(ceil(width/16.0), ceil(width/16.0), 1);
    dim3 block_dim(16, 16, 1);

    matrixMulKernel_ch3<<<grid_dim, block_dim>>>(M_d, N_d, P_d, width);

    cudaMemcpy(P, P_d, size, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}

__global__
void matrixMulKernel_ch5(float* M, float* N, float* P, int width){
    // assert(blockDim.x == TILE_WIDTH && blockDim.y == TILE_WIDTH);

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int col = blockDim.x * bx + tx;
    int row = blockDim.y * by + ty;
    
    float Pvalue = 0;

    for (int ph=0; ph<ceil(width/(float)TILE_WIDTH); ph++){
        if (row < width && (ph * TILE_WIDTH + tx) < width)
            Mds[ty][tx] = M[row * width + ph * TILE_WIDTH + tx];    // M[row][ph*TILE_WIDTH+tx]
        else
            Mds[ty][tx] = 0.0f;
        if (col < width && (ph*TILE_WIDTH+ty) < width)
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + col];  // N[ph*TILE_WIDTH+ty][col]
        else
            Nds[ty][tx] = 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    // threads that are outside the boundary also participate in shared memory loading => Cannot wrap the first for loop
    // Also __syncthreads() cause problem if wrap the first for loop
    if(col < width && row < width)
        P[row*width + col] = Pvalue;
}

void matrix_mul_ch5(float* M, float* N, float* P, int width){
    float *M_d, *N_d, *P_d;
    int size = width * width * sizeof(float);

    cudaMalloc(&M_d, size);
    cudaMalloc(&N_d, size);
    cudaMalloc(&P_d, size);

    cudaMemcpy(M_d, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N, size, cudaMemcpyHostToDevice);

    dim3 grid_dim(ceil(width/16.0), ceil(width/16.0), 1);
    dim3 block_dim(16, 16, 1);

    matrixMulKernel_ch5<<<grid_dim, block_dim>>>(M_d, N_d, P_d, width);

    cudaMemcpy(P, P_d, size, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}

__global__
void matrixMulKernel_ch6(float* M, float* N, float* P, int width){
    // assert(blockDim.x == TILE_WIDTH && blockDim.y == TILE_WIDTH);

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int col = blockDim.x * bx + tx;
    int row = blockDim.y * by + ty;
    
    float Pvalue = 0;

    for (int ph=0; ph<ceil(width/(float)TILE_WIDTH); ph++){
        if (row < width && (ph * TILE_WIDTH + tx) < width)
            Mds[ty][tx] = M[row * width + ph * TILE_WIDTH + tx];    // M[row][ph*TILE_WIDTH+tx]
        else
            Mds[ty][tx] = 0.0f;
        if (col < width && (ph*TILE_WIDTH+ty) < width)
            // transpose tx, ty => N[ph*TILE_WIDTH+tx ][blockDim.x * bx + ty]
            Nds[tx][ty] = N[(ph * TILE_WIDTH + tx) * width + blockDim.x * bx + ty];
        else
            Nds[tx][ty] = 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    // threads that are outside the boundary also participate in shared memory loading => Cannot wrap the first for loop
    // Also __syncthreads() cause problem if wrap the first for loop
    if(col < width && row < width)
        P[row*width + col] = Pvalue;
}

void matrix_mul_ch6(float* M, float* N, float* P, int width){
    float *M_d, *N_d, *P_d;
    int size = width * width * sizeof(float);

    cudaMalloc(&M_d, size);
    cudaMalloc(&N_d, size);
    cudaMalloc(&P_d, size);

    cudaMemcpy(M_d, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N, size, cudaMemcpyHostToDevice);

    dim3 grid_dim(ceil(width/16.0), ceil(width/16.0), 1);
    dim3 block_dim(16, 16, 1);

    matrixMulKernel_ch6<<<grid_dim, block_dim>>>(M_d, N_d, P_d, width);

    cudaMemcpy(P, P_d, size, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}