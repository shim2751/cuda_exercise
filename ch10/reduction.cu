
#include "reduction.h"
#include <cstdio>

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
    for(unsigned int tile = 0; tile < CFACTOR * 2; tile++){
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

// Unified launch function
void launch_reduction(float* input_h, float* output_h, 
                     unsigned int length, reduction_kernel_t kernel_type) {
    int input_size = length * sizeof(float);
    int output_size = sizeof(float);
    
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
        case REDUCTION_BASIC: {
            // Basic reduction - single block only
            int elements_per_block = BLOCK_DIM * 2;
            if (length > elements_per_block) {
                printf("Warning: Basic reduction limited to %d elements, truncating input\n", 
                       elements_per_block);
            }
            
            dim3 grid_dim(1);
            dim3 block_dim(BLOCK_DIM);
            sum_reduction_kernel<<<grid_dim, block_dim>>>(input_d, output_d);
            break;
        }
        case REDUCTION_CONTROL_DIVERGENCE: {
            // Control divergence optimized - single block only
            int elements_per_block = BLOCK_DIM * 2;
            if (length > elements_per_block) {
                printf("Warning: Control divergence reduction limited to %d elements, truncating input\n", 
                       elements_per_block);
            }
            
            dim3 grid_dim(1);
            dim3 block_dim(BLOCK_DIM);
            sum_reduction_cntl_div_kernel<<<grid_dim, block_dim>>>(input_d, output_d);
            break;
        }
        case REDUCTION_SHARED_MEMORY: {
            // Shared memory optimized - single block only
            int elements_per_block = BLOCK_DIM * 2;
            if (length > elements_per_block) {
                printf("Warning: Shared memory reduction limited to %d elements, truncating input\n", 
                       elements_per_block);
            }
            
            dim3 grid_dim(1);
            dim3 block_dim(BLOCK_DIM);
            sum_reduction_sm_kernel<<<grid_dim, block_dim>>>(input_d, output_d);
            break;
        }
        case REDUCTION_HIERARCHICAL: {
            // Hierarchical reduction - multiple blocks
            int elements_per_block = BLOCK_DIM * 2;
            int num_blocks = (length + elements_per_block - 1) / elements_per_block;
            
            dim3 grid_dim(num_blocks);
            dim3 block_dim(BLOCK_DIM);
            sum_reduction_hierarchical_kernel<<<grid_dim, block_dim>>>(input_d, output_d);
            break;
        }
        case REDUCTION_COARSENED: {
            // Coarsened reduction - multiple blocks with thread coarsening
            int elements_per_block = BLOCK_DIM * CFACTOR * 2;
            int num_blocks = (length + elements_per_block - 1) / elements_per_block;
            
            dim3 grid_dim(num_blocks);
            dim3 block_dim(BLOCK_DIM);
            sum_reduction_coarsened_kernel<<<grid_dim, block_dim>>>(input_d, output_d);
            break;
        }
        default: {
            // Default to basic reduction
            dim3 grid_dim(1);
            dim3 block_dim(BLOCK_DIM);
            sum_reduction_kernel<<<grid_dim, block_dim>>>(input_d, output_d);
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
        "Basic Reduction",
        "Control Divergence Optimized",
        "Shared Memory Optimized",
        "Hierarchical Reduction",
        "Coarsened Reduction"
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