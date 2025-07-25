#include "scan.h"
#include <cstdio>

__global__
void scan_sequential_kernel(float* input, float* output, unsigned int N){
    output[0] = input[0];
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
    unsigned int idx = SECTION_SIZE * blockIdx.x + threadIdx.x * SUBSEC_SIZE;
    seq_sum[0] = input[idx];
    for(int j = 1; j < SUBSEC_SIZE; j++){
        seq_sum[j] = seq_sum[j-1] + input[idx+j];
    }
    __syncthreads();
    // Phase 2: Parallel scan across sub-sections
    const unsigned int num_subsec = SECTION_SIZE / SUBSEC_SIZE;
    __shared__ float scan_s[num_subsec];

    if (threadIdx.x == num_subsec - 1) 
        scan_s[0] = 0.0f;
    else 
        scan_s[threadIdx.x+1] = seq_sum[SUBSEC_SIZE-1];

    for(int stride = 1; stride < num_subsec; stride *= 2){
        __syncthreads();
        float tmp;
        if (threadIdx.x >= stride)
            tmp = scan_s[threadIdx.x] + scan_s[threadIdx.x-stride];
        __syncthreads();
        if (threadIdx.x >= stride)
            scan_s[threadIdx.x] = tmp;
    }
    __syncthreads();
    // Phase 3: Final output computation
    for(int j = 0; j < SUBSEC_SIZE; j++){
        if(idx < N)
            output[idx+j] = seq_sum[j] + scan_s[threadIdx.x];
    }
}
__global__
void scan_coarsened_segment_kernel(float* input, float* output, float* S, unsigned int N){
    // Phase 1: Sequential scan within sub-sections
    float seq_sum[SUBSEC_SIZE];
    unsigned int idx = SECTION_SIZE * blockIdx.x + threadIdx.x * SUBSEC_SIZE;
    seq_sum[0] = input[idx];
    for(int j = 1; j < SUBSEC_SIZE; j++){
        seq_sum[j] = seq_sum[j-1] + input[idx+j];
    }
    __syncthreads();
    // Phase 2: Parallel scan across sub-sections
    const unsigned int num_subsec = SECTION_SIZE / SUBSEC_SIZE;
    __shared__ float scan_s[num_subsec];

    if (threadIdx.x == num_subsec - 1) 
        scan_s[0] = 0.0f;
    else 
        scan_s[threadIdx.x+1] = seq_sum[SUBSEC_SIZE-1];

    for(int stride = 1; stride < num_subsec; stride *= 2){
        __syncthreads();
        float tmp;
        if (threadIdx.x >= stride)
            tmp = scan_s[threadIdx.x] + scan_s[threadIdx.x-stride];
        __syncthreads();
        if (threadIdx.x >= stride)
            scan_s[threadIdx.x] = tmp;
    }
    __syncthreads();
    // Phase 3: Final output computation
    for(int j = 0; j < SUBSEC_SIZE; j++){
        if(idx < N)
            output[idx+j] = seq_sum[j] + scan_s[threadIdx.x];
    }
    __syncthreads();

    // 4. The last thread in the block stores the block's total sum into the S array
    if (threadIdx.x == blockDim.x - 1) {
        S[blockIdx.x] = seq_sum[SUBSEC_SIZE-1] + scan_s[threadIdx.x];
    }

}

__global__ void kernel_scan_sums(float *S, unsigned int num_sums) {
    const unsigned int num_subsec = SECTION_SIZE / SUBSEC_SIZE;
    __shared__ float scan_s[num_subsec];

    // Load data from the S array into shared memory
    if (threadIdx.x < num_sums) {
        scan_s[threadIdx.x] = S[threadIdx.x];
    }
    __syncthreads();

    // Scan the sums array using the Kogge-Stone algorithm.
    // This scan is in-place and will overwrite the values in the S array.
    for (unsigned int stride = 1; stride < num_sums; stride *= 2) {
        __syncthreads();
        float temp = 0.0f;
        if (threadIdx.x >= stride) {
            temp = scan_s[threadIdx.x] + scan_s[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride) {
            scan_s[threadIdx.x] = temp;
        }
    }
    __syncthreads();

    // Store the scanned results back into the S array
    if (threadIdx.x < num_sums) {
        S[threadIdx.x] = scan_s[threadIdx.x];
    }
}

__global__ void kernel_add_sums(float *output, float *S, unsigned int N) {
    // Calculate global index
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Exclude the first block (blockIdx.x == 0) as there is no preceding sum to add
    if (blockIdx.x > 0 && i < N) {
        // Add the cumulative sum of all previous blocks to each element of output
        output[i] += S[blockIdx.x - 1];
    }
}



// Unified launch function
void launch_scan(float* input_h, float* output_h, 
                 unsigned int length, scan_kernel_t kernel_type) {
    int input_size = length * sizeof(float);
    int output_size = length * sizeof(float);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Device memory allocation
    float *input_d, *output_d;
    cudaMalloc(&input_d, input_size);
    cudaMalloc(&output_d, output_size);
    
    // Copy input data to device
    cudaMemcpy(input_d, input_h, input_size, cudaMemcpyHostToDevice);
    cudaMemset(output_d, 0, output_size);
    
    // Record start time
    cudaEventRecord(start);
    
    // Launch appropriate kernel based on type
    switch(kernel_type) {
        case SCAN_SEQUENTIAL: {
            // Sequential scan - single thread
            dim3 grid_dim(1);
            dim3 block_dim(1);
            scan_sequential_kernel<<<grid_dim, block_dim>>>(input_d, output_d, length);
            break;
        }
        case SCAN_KOGGE_STONE: {
            // Kogge-Stone scan - single block
            if (length > SECTION_SIZE) {
                printf("Warning: Kogge-Stone scan limited to %d elements, truncating input\n", 
                       SECTION_SIZE);
            }
            
            dim3 grid_dim(1);
            dim3 block_dim(length < SECTION_SIZE ? length : SECTION_SIZE);
            scan_kogge_stone_kernel<<<grid_dim, block_dim>>>(input_d, output_d, length);
            break;
        }
        case SCAN_BRENT_KUNG: {
            // Brent-Kung scan - single block, each thread handles 2 elements
            if (length > SECTION_SIZE) {
                printf("Warning: Brent-Kung scan limited to %d elements, truncating input\n", 
                       SECTION_SIZE);
            }
            
            dim3 grid_dim(1);
            dim3 block_dim(SECTION_SIZE / 2);
            scan_brent_kung_kernel<<<grid_dim, block_dim>>>(input_d, output_d, length);
            break;
        }
        case SCAN_COARSENED: {
            // Coarsened scan - can handle multiple blocks
            int threads_per_block = SECTION_SIZE / SUBSEC_SIZE;
            
            dim3 grid_dim(1);
            dim3 block_dim(threads_per_block);
            scan_coarsened_kernel<<<grid_dim, block_dim>>>(input_d, output_d, length);
            break;
        }
        default: {
            // Default to sequential scan
            dim3 grid_dim(1);
            dim3 block_dim(1);
            scan_sequential_kernel<<<grid_dim, block_dim>>>(input_d, output_d, length);
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
        "Sequential Scan",
        "Kogge-Stone Scan",
        "Brent-Kung Scan",
        "Coarsened Scan"
    };
    printf("[%s] Kernel execution time: %.6f ms\n", 
           kernel_names[kernel_type], milliseconds);
    
    // Copy result back to host
    cudaMemcpy(output_h, output_d, output_size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(input_d);
    cudaFree(output_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}