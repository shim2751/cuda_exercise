#include "histogram.h"
#include <cstdio>

__global__
void histo_kernel(char* data, unsigned int length, unsigned int * histo){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length){
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < 26){
            atomicAdd(&(histo[pos/4]),1);
        }
    }
}

__global__
void histo_private_w_GM_kernel(char* data, unsigned int length, unsigned int * histo){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length){
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < 26){
            atomicAdd(&(histo[NUM_BINS * blockIdx.x + pos/4]),1);
        }
    }

    //block 0 already stored proper place.
    if(blockIdx.x > 0){
        __syncthreads();
        // each thread could work multiple times if blockDim.x is smaller than # of bin elements.
        for(unsigned int bin=threadIdx.x; bin < NUM_BINS; bin += blockDim.x){
            int binVal = histo[NUM_BINS * blockIdx.x + bin];
            if(binVal > 0)
                atomicAdd(&(histo[bin]),binVal);
        }
    }
}

__global__
void histo_private_w_SM_kernel(char* data, unsigned int length, unsigned int * histo){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ unsigned int histo_s[NUM_BINS];
    for(unsigned int bin=threadIdx.x; bin < NUM_BINS; bin += blockDim.x){
        histo_s[bin] = 0u;
    }
    __syncthreads();
    if (i < length){
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < 26){
            atomicAdd(&(histo_s[pos/4]),1);
        }
    }

    __syncthreads();
    // each thread could work multiple times if blockDim.x is smaller than # of bin elements.
    for(unsigned int bin=threadIdx.x; bin < NUM_BINS; bin += blockDim.x){
        int binVal = histo_s[bin];
        if (binVal > 0)
            atomicAdd(&(histo[bin]),binVal);
    }
}

__global__
void histo_coarsening_contiguous_kernel(char* data, unsigned int length, unsigned int * histo){
    __shared__ unsigned int histo_s[NUM_BINS];
    for(unsigned int bin=threadIdx.x; bin < NUM_BINS; bin += blockDim.x){
        histo_s[bin] = 0u;
    }
    __syncthreads();
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    for(unsigned int i=tid*CFACTOR; i<min((tid+1)*CFACTOR, length); ++i) {
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < 26) {
            atomicAdd(&(histo_s[pos/4]), 1);
        }
    }

    __syncthreads();
    // each thread could work multiple times if blockDim.x is smaller than # of bin elements.
    for(unsigned int bin=threadIdx.x; bin < NUM_BINS; bin += blockDim.x){
        int binVal = histo_s[bin];
        if (binVal > 0)
            atomicAdd(&(histo[bin]),binVal);
    }
}

__global__
void histo_coarsening_interleaved_kernel(char* data, unsigned int length, unsigned int * histo){
    __shared__ unsigned int histo_s[NUM_BINS];
    for(unsigned int bin=threadIdx.x; bin < NUM_BINS; bin += blockDim.x){
        histo_s[bin] = 0u;
    }
    __syncthreads();
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    for(unsigned int i=tid; i<length; i+=blockDim.x*gridDim.x) {
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < 26) {
            atomicAdd(&(histo_s[pos/4]), 1);
        }
    }

    __syncthreads();
    // each thread could work multiple times if blockDim.x is smaller than # of bin elements.
    for(unsigned int bin=threadIdx.x; bin < NUM_BINS; bin += blockDim.x){
        int binVal = histo_s[bin];
        if (binVal > 0 )
            atomicAdd(&(histo[bin]),binVal);
    }
}

__global__
void histo_aggregated_kernel(char* data, unsigned int length, unsigned int * histo){
    __shared__ unsigned int histo_s[NUM_BINS];
    for(unsigned int bin=threadIdx.x; bin < NUM_BINS; bin += blockDim.x){
        histo_s[bin] = 0u;
    }
    __syncthreads();
    unsigned int accumulator = 0;
    int prev_pos = 0;
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    for(unsigned int i=tid; i<length; i+=blockDim.x*gridDim.x) {
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < 26){
            int bin = pos/4;
            if (prev_pos == bin)
                accumulator++;
            else{
                if (accumulator > 0){
                    atomicAdd(&(histo_s[prev_pos]), accumulator);
                }
                accumulator = 1;
                prev_pos = bin;
            }
        }
    }
    if (accumulator > 0){
        atomicAdd(&(histo_s[prev_pos]), accumulator);
    }

    __syncthreads();
    // each thread could work multiple times if blockDim.x is smaller than # of bin elements.
    for(unsigned int bin=threadIdx.x; bin < NUM_BINS; bin += blockDim.x){
        int binVal = histo_s[bin];
        if (binVal > 0 )
            atomicAdd(&(histo[bin]),binVal);
    }
}
// Unified launch function
void launch_histogram(char* data_h, unsigned int* histo_h, 
                     unsigned int length, histogram_kernel_t kernel_type) {
    int data_size = length * sizeof(char);
    int histo_size = NUM_BINS * sizeof(unsigned int);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Device memory allocation
    char *data_d;
    unsigned int *histo_d;
    cudaMalloc(&data_d, data_size);
    
    // Copy input data to device
    cudaMemcpy(data_d, data_h, data_size, cudaMemcpyHostToDevice);
    
    // Record start time
    cudaEventRecord(start);
    
    // Launch appropriate kernel based on type
    switch(kernel_type) {
        case HISTOGRAM_BASIC: {
            cudaMalloc(&histo_d, histo_size);
            cudaMemset(histo_d, 0, histo_size);
            
            dim3 grid_dim((length + BLOCK_SIZE - 1) / BLOCK_SIZE);
            dim3 block_dim(BLOCK_SIZE);
            histo_kernel<<<grid_dim, block_dim>>>(data_d, length, histo_d);
            break;
        }
        case HISTOGRAM_PRIVATIZED_GLOBAL: {
            int num_blocks = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;
            cudaMalloc(&histo_d, histo_size * num_blocks);
            cudaMemset(histo_d, 0, histo_size * num_blocks);
            
            dim3 grid_dim(num_blocks);
            dim3 block_dim(BLOCK_SIZE);
            histo_private_w_GM_kernel<<<grid_dim, block_dim>>>(data_d, length, histo_d);
            break;
        }
        case HISTOGRAM_PRIVATIZED_SHARED: {
            cudaMalloc(&histo_d, histo_size);
            cudaMemset(histo_d, 0, histo_size);
            
            dim3 grid_dim((length + BLOCK_SIZE - 1) / BLOCK_SIZE);
            dim3 block_dim(BLOCK_SIZE);
            histo_private_w_SM_kernel<<<grid_dim, block_dim>>>(data_d, length, histo_d);
            break;
        }
        case HISTOGRAM_COARSENED_CONTIGUOUS: {
            int num_blocks = (length + (BLOCK_SIZE * CFACTOR) - 1) / (BLOCK_SIZE * CFACTOR);
            cudaMalloc(&histo_d, histo_size);
            cudaMemset(histo_d, 0, histo_size);
            
            dim3 grid_dim(num_blocks);
            dim3 block_dim(BLOCK_SIZE);
            histo_coarsening_contiguous_kernel<<<grid_dim, block_dim>>>(data_d, length, histo_d);
            break;
        }
        case HISTOGRAM_COARSENED_INTERLEAVED: {
            int num_blocks = (length + (BLOCK_SIZE * CFACTOR) - 1) / (BLOCK_SIZE * CFACTOR);
            cudaMalloc(&histo_d, histo_size);
            cudaMemset(histo_d, 0, histo_size);
            
            dim3 grid_dim(num_blocks);
            dim3 block_dim(BLOCK_SIZE);
            histo_coarsening_interleaved_kernel<<<grid_dim, block_dim>>>(data_d, length, histo_d);
            break;
        }
        case HISTOGRAM_AGGREGATED: {
            cudaMalloc(&histo_d, histo_size);
            cudaMemset(histo_d, 0, histo_size);
            
            dim3 grid_dim((length + BLOCK_SIZE - 1) / BLOCK_SIZE);
            dim3 block_dim(BLOCK_SIZE);
            histo_aggregated_kernel<<<grid_dim, block_dim>>>(data_d, length, histo_d);
            break;
        }
        default: {
            cudaMalloc(&histo_d, histo_size);
            cudaMemset(histo_d, 0, histo_size);
            
            dim3 grid_dim((length + BLOCK_SIZE - 1) / BLOCK_SIZE);
            dim3 block_dim(BLOCK_SIZE);
            histo_kernel<<<grid_dim, block_dim>>>(data_d, length, histo_d);
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
        "Basic Atomic",
        "Privatized (Global Memory)",
        "Privatized (Shared Memory)",
        "Coarsened Contiguous",
        "Coarsened Interleaved",
        "Aggregated"
    };
    printf("[%s] Kernel execution time: %.6f ms\n", 
           kernel_names[kernel_type], milliseconds);
    
    // Copy result back to host
    cudaMemcpy(histo_h, histo_d, histo_size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(data_d);
    cudaFree(histo_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}