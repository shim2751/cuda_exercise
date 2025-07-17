__global__
void matrixMulKernel(float* M, float* N, float* P, int width){
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

void matrix_mul(float* M, float* N, float* P, int width){
    float *M_d, *N_d, *P_d;
    int size = width * width * sizeof(float);

    cudaMalloc(&M_d, size);
    cudaMalloc(&N_d, size);
    cudaMalloc(&P_d, size);

    cudaMemcpy(M_d, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N, size, cudaMemcpyHostToDevice);

    dim3 block_dim(ceil(width/16.0), ceil(width/16.0), 1);
    dim3 grid_dim(16, 16, 1);

    matrixMulKernel<<<grid_dim, block_dim>>>(M_d, N_d, P_d, width);

    cudaMemcpy(P, P_d, size, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}
