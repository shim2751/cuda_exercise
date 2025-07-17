__global__
void matMulKernel(float* A, float* B, float* C, int width){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    
    if(col < width && row < width) {
        float Pval = 0;
        for(int i=0; i < width; i++){
            Pval += A[row*width + i] * B[i*width + col];
        }
        C[row*width+col] = Pval;
    }
}

void matrix_mul(float* A, float* B, float* C, int width){
    float *A_d, *B_d, *C_d;
    int size = width * width * sizeof(float);

    cudaMalloc(&A_d, size);
    cudaMalloc(&B_d, size);
    cudaMalloc(&C_d, size);

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    dim3 block_dim(ceil(width/16.0), ceil(width/16.0), 1);
    dim3 grid_dim(16, 16, 1);

    matMulKernel<<<grid_dim, block_dim>>>(A_d, B_d, C_d, width);

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}