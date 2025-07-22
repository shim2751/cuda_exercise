__global__
void stencil_kernel(float* in, float* out, unsigned int N){
    unsigned int i = blockIdx.z*blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x*blockDim.x + threadIdx.x;

    if(i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        out[i*N*N + j*N + k] = c0*in[i*N*N + j*N + k]
                               + c1*in[i*N*N + j*N + (k - 1)]
                               + c2*in[i*N*N + j*N + (k + 1)]
                               + c3*in[i*N*N + (j - 1)*N + k]
                               + c4*in[i*N*N + (j + 1)*N + k]
                               + c5*in[(i - 1)*N*N + j*N + k]
                               + c6*in[(i + 1)*N*N + j*N + k];
    }

}

__global__
void stencil_sm_kernel(float* in, float* out, unsigned int N){
    unsigned int i = blockIdx.z*OUT_TILE_DIM + threadIdx.z - 1;
    unsigned int j = blockIdx.y*OUT_TILE_DIM + threadIdx.y - 1;
    unsigned int k = blockIdx.x*OUT_TILE_DIM + threadIdx.x - 1;

    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    if(i>=0 && i < N && j>=0 && j < N && k>=0 && k < N )
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i*N*N + j*N + k];
    __syncthreads();
    
    if(i >= 1 && i < N-1 && j >= 1 && j < N-1 && k >= 1 && k < N-1) {
        if(threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM-1 && threadIdx.y >= 1
            && threadIdx.y<IN_TILE_DIM-1 && threadIdx.x>=1 && threadIdx.x<IN_TILE_DIM-1){

            out[i*N*N + j*N + k] = c0*in_s[threadIdx.z][threadIdx.y][threadIdx.x]
                                + c1*in_s[threadIdx.z][threadIdx.y][threadIdx.x-1]
                                + c2*in_s[threadIdx.z][threadIdx.y][threadIdx.x+1]
                                + c3*in_s[threadIdx.z][threadIdx.y-1][threadIdx.x]
                                + c4*in_s[threadIdx.z][threadIdx.y+1][threadIdx.x]
                                + c5*in_s[threadIdx.z-1][threadIdx.y][threadIdx.x]
                                + c6*in_s[threadIdx.z+1][threadIdx.y][threadIdx.x];
        }
    }

}

__global__
void stencil_th_coarsening_kernel(float* in, float* out, unsigned int N){       //block 3dim , thread 2dim
    unsigned int iStr = blockIdx.z * OUT_TILE_DIM_C;
    unsigned int j = blockIdx.y * OUT_TILE_DIM_C + threadIdx.y - 1;
    unsigned int k = blockIdx.x * OUT_TILE_DIM_C + threadIdx.x - 1;

    __shared__ float inPrev_s[IN_TILE_DIM_C][IN_TILE_DIM_C]; //T^3 => 3T^2
    __shared__ float inCur_s[IN_TILE_DIM_C][IN_TILE_DIM_C];
    __shared__ float inNext_s[IN_TILE_DIM_C][IN_TILE_DIM_C];

    if(iStr >= 0 && iStr < N && j>=0 && j < N && k>=0 && k < N ){
        if(iStr-1 >= 0 && iStr-1 < N)
            inPrev_s[threadIdx.y][threadIdx.x] = in[j*N + k];
        inCur_s[threadIdx.y][threadIdx.x] = in[1*N*N + j*N + k];
    }

    for(int i=iStr; i<iStr+OUT_TILE_DIM_C; i++){
        if(i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) 
            inNext_s[threadIdx.y][threadIdx.x] = in[(i+1)*N*N + j*N + k];
        __syncthreads();

        out[i*N*N + j*N + k] = c0*inCur_s[threadIdx.y][threadIdx.x]
                            + c1*inCur_s[threadIdx.y][threadIdx.x-1]
                            + c2*inCur_s[threadIdx.y][threadIdx.x+1]
                            + c3*inCur_s[threadIdx.y-1][threadIdx.x]
                            + c4*inCur_s[threadIdx.y+1][threadIdx.x]
                            + c5*inPrev_s[threadIdx.y][threadIdx.x]
                            + c6*inNext_s[threadIdx.y][threadIdx.x];
        __syncthreads();
        inPrev_s[threadIdx.y][threadIdx.x] = inCur_s[threadIdx.y][threadIdx.x];
        inCur_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
    }
}


__global__
void stencil_reg_tile_kernel(float* in, float* out, unsigned int N){       //block 3dim , thread 2dim
    unsigned int iStr = blockIdx.z * OUT_TILE_DIM_C;
    unsigned int j = blockIdx.y * OUT_TILE_DIM_C + threadIdx.y - 1;
    unsigned int k = blockIdx.x * OUT_TILE_DIM_C + threadIdx.x - 1;

    __shared__ float inCur_s[IN_TILE_DIM_C][IN_TILE_DIM_C];
    float inPrev; 
    float inCur;
    float inNext;

    if(iStr >= 0 && iStr < N && j>=0 && j < N && k>=0 && k < N ){
        if(iStr-1 >= 0 && iStr-1 < N)
            inPrev = in[j*N + k];
        inCur = in[1*N*N + j*N + k];
        inCur_s[threadIdx.y][threadIdx.x] = inCur;
    }

    for(int i=iStr; i<iStr+OUT_TILE_DIM_C; i++){
        if(i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) 
            inNext = in[(i+1)*N*N + j*N + k];
        __syncthreads();

        out[i*N*N + j*N + k] = c0*inCur
                            + c1*inCur_s[threadIdx.y][threadIdx.x-1]
                            + c2*inCur_s[threadIdx.y][threadIdx.x+1]
                            + c3*inCur_s[threadIdx.y-1][threadIdx.x]
                            + c4*inCur_s[threadIdx.y+1][threadIdx.x]
                            + c5*inPrev
                            + c6*inNext;
        __syncthreads();
        inPrev = inCur_s[threadIdx.y][threadIdx.x];
        inCur = inNext;
        inCur_s[threadIdx.y][threadIdx.x] = inNext;
    }
}