__global__
void scan_sequential_kernel(float* input, float* output, unsigned int N){
    float output[0] = input[0];
    for(int i = 1; i < N; i++){
        output[i] = output[i-1] + input[i];
    }
}

__global__
void scan_kogge_stone_kernel(float* input, float* output, unsigned int N){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float output_s[SECTION_SIZE];
    if(i < N)
        output_s[threadIdx.x] = input[i];
    else
        output_s[threadIdx.x] = 0.;

    for(int stride = 1; stride < N; stride *= 2){
        __syncthreads();
        float tmp;
        if (threadIdx.x >= stride)
            tmp = output_s[threadIdx.x] + output_s[threadIdx.x-stride];
        __syncthreads();
        if (threadIdx.x >= stride)
            output_s[threadIdx.x] = tmp;
    }
    if(i < N)
        output[i] = output_s[threadIdx.x];
}

__global__
void scan_brent_kung_kernel(float* input, float* output, unsigned int N){
    // each thread handles two elements
    unsigned int i = 2 * blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float output_s[SECTION_SIZE];
    if(i < N)
        output_s[threadIdx.x] = input[i];
    if(i + blockDim.x < N)
        output_s[threadIdx.x + blockDim.x] = input[i + blockDim.x];

    for(int stride = 1; stride <= blockDim.x; stride *= 2){
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
        if(index < SECTION_SIZE)
            output_s[index] += output_s[index - stride];
    }
    for(int stride = SECTION_SIZE / 4; stride > 0; stride /= 2){
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
        if(index + stride < SECTION_SIZE)
            output_s[index + stride] += output_s[index];
    }
    __syncthreads();
    if(i < N)
        output[i] = output_s[threadIdx.x];
    if(i + blockDim.x < N)
        output[i + blockDim.x] = output_s[threadIdx.x + blockDim.x];
}

__global__
void scan_coarsened_kernel(float* input, float* output, unsigned int N){
    // Phase 1: Sequential scan within sub-sections
    float seq_sum[SUBSEC_SIZE];

    seq_sum[0] = input[0];
    for(int i = 1; i < SUBSEC_SIZE; i++){
        seq_sum[i] = seq_sum[i-1] + input[i];
    }
    __syncthreads();
    // Phase 2: Parallel scan across sub-sections
    unsigned int num_subsec = (SECTION_SIZE + SUBSEC_SIZE - 1) / SUBSEC_SIZE;
    __shared__ float output_s[num_subsec];

    output_s[threadIdx.x] = seq_sum[SUBSEC_SIZE-1];

    for(int stride = 1; stride <= blockDim.x; stride *= 2){
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
        if(index < num_subsec)
            output_s[index] += output_s[index - stride];
    }
    for(int stride = num_subsec / 4; stride > 0; stride /= 2){
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
        if(index + stride < num_subsec)
            output_s[index + stride] += output_s[index];
    }
    __syncthreads();
    // Phase 3: Final output computation
    unsigned int idx = 2 * blockDim.x * blockIdx.x + threadIdx.x;

    if(threadIdx.x > 0){
        for(int j = 0; j < SUBSEC_SIZE; j++){
            unsigned int idx = 2 * blockDim.x * blockIdx.x + threadIdx.x * SUBSEC_SIZE + j;
            if(idx < N)
                output[idx] = seq_sum[j] + output_s[threadIdx.x-1];
        }
    }
}