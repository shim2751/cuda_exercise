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
        if(idx+j < N)
            output[idx+j] = seq_sum[j] + scan_s[threadIdx.x];
    }
}
__global__
void scan_coarsened_segment_kernel(float* input, float* output, float* S, unsigned int N){
    // Phase 1: Sequential scan within sub-sections
    float seq_sum[SUBSEC_SIZE];
    unsigned int idx = SUBSEC_SIZE * blockDim.x * blockIdx.x + threadIdx.x * SUBSEC_SIZE;

    if (idx >= N) return;
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
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float output_s[SECTION_SIZE];

    if(i < num_sums)
        output_s[threadIdx.x] = S[i];
    else
        output_s[threadIdx.x] = 0.;

    for(int stride = 1; stride < num_sums; stride *= 2){
        __syncthreads();
        float tmp;
        if (threadIdx.x >= stride)
            tmp = output_s[threadIdx.x] + output_s[threadIdx.x-stride];
        __syncthreads();
        if (threadIdx.x >= stride)
            output_s[threadIdx.x] = tmp;
    }
    if(i < num_sums)
        S[i] = output_s[threadIdx.x];
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


__global__
void scan_coarsened_segment_domino_kernel(float* input, float* output, float* scan_value, unsigned int* flags, unsigned int* blockCounter,  unsigned int N){
    __shared__ unsigned int bid_s;
    if(threadIdx.x == 0) {
        bid_s = atomicAdd(blockCounter, 1);
    }
    __syncthreads();
    unsigned int bid = bid_s;

    // Phase 1: Sequential scan within sub-sections
    float seq_sum[SUBSEC_SIZE];
    unsigned int idx = SUBSEC_SIZE * blockDim.x * bid + threadIdx.x * SUBSEC_SIZE;

    if (idx >= N) return;
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
    
    float previous_sum;
    if(threadIdx.x == blockDim.x - 1) {
        if(bid != 0)
            while(atomicAdd(&flags[bid], 0) == 0) {}
        scan_value[bid+1] = scan_value[bid] + seq_sum[SUBSEC_SIZE-1] + scan_s[threadIdx.x];
        __threadfence();
        atomicAdd(&flags[bid+1], 1);
    }
    __syncthreads();

    previous_sum = scan_value[bid];
    // Phase 3: Final output computation
    for(int j = 0; j < SUBSEC_SIZE; j++){
        if(idx+j < N)
            output[idx+j] = seq_sum[j] + scan_s[threadIdx.x] + previous_sum;
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
    
    // Launch appropriate kernel based on type
    // Timing events are now placed inside each case to accurately measure kernel execution time
    switch(kernel_type) {
        case SCAN_SEQUENTIAL: {
            // Sequential scan - single thread
            dim3 grid_dim(1);
            dim3 block_dim(1);
            
            cudaEventRecord(start);
            scan_sequential_kernel<<<grid_dim, block_dim>>>(input_d, output_d, length);
            cudaEventRecord(stop);
            break;
        }
        case SCAN_KOGGE_STONE: {
            // Kogge-Stone scan - single block            
            dim3 grid_dim(1);
            dim3 block_dim(SECTION_SIZE);

            cudaEventRecord(start);
            scan_kogge_stone_kernel<<<grid_dim, block_dim>>>(input_d, output_d, length);
            cudaEventRecord(stop);
            break;
        }
        case SCAN_BRENT_KUNG: {
            // Brent-Kung scan - single block, each thread handles 2 elements            
            dim3 grid_dim(1);
            dim3 block_dim(SECTION_SIZE / 2);

            cudaEventRecord(start);
            scan_brent_kung_kernel<<<grid_dim, block_dim>>>(input_d, output_d, length);
            cudaEventRecord(stop);
            break;
        }
        case SCAN_COARSENED: {
            // Coarsened scan - can handle multiple blocks
            int threads_per_block = SECTION_SIZE / SUBSEC_SIZE;
            
            dim3 grid_dim(1);
            dim3 block_dim(threads_per_block);

            cudaEventRecord(start);
            scan_coarsened_kernel<<<grid_dim, block_dim>>>(input_d, output_d, length);
            cudaEventRecord(stop);
            break;
        }
        case SCAN_SEGMENTED: {
            // Calculate number of blocks needed
            int threads_per_block = SECTION_SIZE / SUBSEC_SIZE;
            int num_blocks = ceil(length / (float)(SECTION_SIZE));
            printf("Segmented scan: %d elements, %d blocks, %d threads per block\n",
                   length, num_blocks, threads_per_block);
            
            // Allocate memory for block sums array S
            float *S_d;
            cudaMalloc(&S_d, num_blocks * sizeof(float));
            
            // Record start time before kernel launches
            cudaEventRecord(start);
            
            // Step 1: Perform segmented scan on each block and store block sums
            dim3 grid_dim1(num_blocks);
            dim3 block_dim1(threads_per_block);
            scan_coarsened_segment_kernel<<<grid_dim1, block_dim1>>>(input_d, output_d, S_d, length);
            
            // Step 2: Scan the block sums array (if more than one block)
            if (num_blocks > 1) {
                // The sync is necessary to ensure step 1 is complete before step 2
                cudaDeviceSynchronize(); 
                
                dim3 grid_dim2(1);
                dim3 block_dim2(num_blocks);
                kernel_scan_sums<<<grid_dim2, block_dim2>>>(S_d, num_blocks);
                
                // The sync is necessary to ensure step 2 is complete before step 3
                cudaDeviceSynchronize(); 
                
                // Step 3: Add scanned block sums to all elements in their respective blocks
                dim3 grid_dim3(num_blocks);
                dim3 block_dim3(SECTION_SIZE);
                kernel_add_sums<<<grid_dim3, block_dim3>>>(output_d, S_d, length);
            }
            
            // Record stop time after all kernel launches for this case
            cudaEventRecord(stop);
            
            // Wait for all kernels to complete before freeing S_d
            cudaEventSynchronize(stop);

            // Cleanup S array
            cudaFree(S_d);
            break;
        }
        case DOMINO_SCAN_SEGMENTED: {
            // Calculate number of blocks needed
            int threads_per_block = SECTION_SIZE / SUBSEC_SIZE;
            int num_blocks = ceil(length / (float)(SECTION_SIZE));
            printf("Segmented scan: %d elements, %d blocks, %d threads per block\n",
                   length, num_blocks, threads_per_block);
            
            // Allocate memory for block sums array S and other auxiliary arrays
            float *S_d;
            unsigned int *flags_d;
            unsigned int *blockCounter_d;
            
            cudaMalloc(&S_d, (num_blocks+1) * sizeof(float));
            cudaMalloc(&flags_d, (num_blocks+1) * sizeof(unsigned int));
            cudaMalloc(&blockCounter_d, sizeof(unsigned int));

            // Initialize 
            cudaMemset(S_d, 0, (num_blocks+1) * sizeof(float));
            cudaMemset(flags_d, 0, (num_blocks+1) * sizeof(unsigned int));
            cudaMemset(blockCounter_d, 0, sizeof(unsigned int));

            // Record start time before the kernel launch
            cudaEventRecord(start);
            
            // Perform segmented scan using domino kernel
            dim3 grid_dim1(num_blocks);
            dim3 block_dim1(threads_per_block);            
            scan_coarsened_segment_domino_kernel<<<grid_dim1, block_dim1>>>(input_d, output_d, S_d, flags_d, blockCounter_d, length);
            
            // Record stop time after the kernel launch
            cudaEventRecord(stop);
            
            // Wait for the kernel to complete before freeing memory
            cudaEventSynchronize(stop);

            cudaFree(S_d);
            cudaFree(flags_d);
            cudaFree(blockCounter_d);
            break;
        }
        default: {
            // Default to sequential scan
            dim3 grid_dim(1);
            dim3 block_dim(1);

            cudaEventRecord(start);
            scan_sequential_kernel<<<grid_dim, block_dim>>>(input_d, output_d, length);
            cudaEventRecord(stop);
            break;
        }
    }
    
    // Wait for the recorded events to complete
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Print timing information
    const char* kernel_names[] = {
        "Sequential Scan",
        "Kogge-Stone Scan",
        "Brent-Kung Scan",
        "Coarsened Scan",
        "Segmented Scan",
        "Domino Segmented Scan"
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