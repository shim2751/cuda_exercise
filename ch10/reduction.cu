__global__
void sum_reduction_kernel(float* input, float* output){
    unsigned int i = 2 * threadIdx.x;
    
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        if(i % stride == 0)
            input[i] += input[i + stride];
        __syncthreads();
    }
    if(threadIdx.x == 0)
        *output = input[0];
}

__global__
void sum_reduction_cntl_div_kernel(float* input, float* output){
    unsigned int i = threadIdx.x;
    
    for(unsigned int stride = blockDim.x; stride > 0; stride /= 2){
        if (threadIdx.x < stride)
            input[i] += input[i + stride];
        __syncthreads();
    }
    if(threadIdx.x == 0)
        *output = input[0];
}

__global__
void sum_reduction_sm_kernel(float* input, float* output){
    unsigned int i = threadIdx.x;
    __shared__ float input_s[BLOCK_DIM];

    input_s[i] = input[i] + input[i+BLOCK_DIM];
    for(unsigned int stride = blockDim.x/2; stride > 0; stride /= 2){
        __syncthreads();
        if (threadIdx.x < stride)
            input_s[i] += input_s[i + stride];
    }
    if(threadIdx.x == 0)
        *output = input_s[0];
}

__global__
void sum_reduction_hierarchical_kernel(float* input, float* output){
    
    unsigned int seg = 2 * blockDim.x * blockIdx.x;
    __shared__ float input_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;
    unsigned int i = seg + t;

    input_s[t] = input[i] + input[i+BLOCK_DIM];
    for(unsigned int stride = blockDim.x/2; stride > 0; stride /= 2){
        __syncthreads();
        if (t < stride)
            input_s[t] += input_s[t + stride];
    }
    if(t == 0)
        atomicAdd(output, input_s[0]);
}

__global__
void sum_reduction_coarsened_kernel(float* input, float* output){
    unsigned int seg = CFACTOR * 2 * blockDim.x * blockIdx.x;
    __shared__ float input_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;
    unsigned int i = seg + t;
    
    float sum = 0;
    for(unsigned int tile = 1; tile < CFACTOR * 2; tile++){
        sum += input[i + tile*BLOCK_DIM];
    }
    input_s[t] = sum;
    for(unsigned int stride = blockDim.x/2; stride > 0; stride /= 2){
        __syncthreads();
        if (t < stride)
            input_s[t] += input_s[t + stride];
    }
    if(t == 0)
        atomicAdd(output, input_s[0]);
}