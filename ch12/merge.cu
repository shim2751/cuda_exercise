#include "merge.h"
#include <cstdio>
#include <cmath>

__device__
void merge_sequential(int* A, int m, int* B, int n, int* C) {
    int i = 0, j = 0, k = 0;
    while(i < m && j < n){
        if(A[i] < B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    if(i == m){
        while(j < n) {
            C[k++] = B[j++];
        }
    } else {
        while(i < m) {
            C[k++] = A[i++];
        }
    }
}

__device__
int co_rank(int k, int* A, int m, int* B, int n){
    int i = k < m ? k : m; // min(k, m)  k가 m보다 작으면 C는 A의 최대 k개의 원소를 포함
    int j = k - i;
    int i_low = 0 > (k-n) ? 0 : k - n; // max(0, k-n) n이 k보다 작으면 C는 A의 최소 k-n개의 원소를 포함
    int j_low = 0 > (k-m) ? 0 : k - m; // max(0, k-m) m이 k보다 작으면 C는 B의 최소 k-m개의 원소를 포함
    int delta;
    bool active = true;
    //until B[j-1] < A[i] && A[i-1] < B[j]
    while(active){
        if(i > 0 && j < n && A[i-1] >= B[j]){
            delta = ((i - i_low + 1) >> 1); 
            j_low = j;
            i -= delta;
            j += delta;
        }
        else if (j > 0 && i < m && B[j-1] >= A[i]){
            delta = ((j - j_low + 1) >> 1);
            i_low = i;
            j -= delta;
            i += delta;
        }
        else {
            active = false;
        }
    }
    return i;
}

__global__
void merge_basic_kerner(int* A, int m, int* B, int n, int* C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int elementsPerThread = ((m+n) + blockDim.x*gridDim.x - 1)/(blockDim.x*gridDim.x);
    int k_curr = tid*elementsPerThread; // start output index
    int k_next = min((tid+1)*elementsPerThread, m+n); // end output index
    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;
    merge_sequential(&A[i_curr], i_next-i_curr, &B[j_curr], j_next-j_curr, &C[k_curr]);
}

__global__
void merge_tiled_kernel(int* A, int m, int* B, int n, int* C, int tile_size) {
    // Part 1: Identifying block-level output and input subarrays.
    extern __shared__ int shareAB[];
    int* A_s = &shareAB[0];
    int* B_s = &shareAB[tile_size];
    
    int elementsPerBlock = ceilf((m + n) / gridDim.x);
    int C_curr = blockIdx.x * elementsPerBlock; // start output index
    int C_next = min((blockIdx.x + 1) * elementsPerBlock, m + n); // end output index
    
    if(threadIdx.x == 0){
        A_s[0] = co_rank(C_curr, A, m, B, n);
        A_s[1] = co_rank(C_next, A, m, B, n);
    }

    __syncthreads();
    int A_curr = A_s[0];
    int A_next = A_s[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;

    __syncthreads();
    // Part 2: Loading A and B elements into the shared memory.
    int counter = 0;                                           //iteration counter
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = ceilf((C_length)/tile_size);          //total iteration
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;
    while(counter < total_iteration){
        /* loading tile-size A and B elements into shared memory */
        for(int i=0; i<tile_size; i+=blockDim.x){
            if( i + threadIdx.x < A_length - A_consumed) {
                A_s[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x ];
            }
        }
        for(int i=0; i<tile_size; i+=blockDim.x) {
            if(i + threadIdx.x  < B_length - B_consumed) {
                B_s[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
            }
        }
        __syncthreads();

        //Part 3: All threads merge their individual subarrays in parallel.
        int c_curr = threadIdx.x  * (tile_size/blockDim.x);
        int c_next = (threadIdx.x+1) * (tile_size/blockDim.x);
        c_curr = (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;
        c_next = (c_next <= C_length - C_completed) ? c_next : C_length - C_completed;
        /* find co-rank for c_curr and c_next */
        int a_curr = co_rank(c_curr, A_s, min(tile_size, A_length-A_consumed),
                                    B_s, min(tile_size, B_length-B_consumed));
        int b_curr = c_curr - a_curr;
        int a_next = co_rank(c_next, A_s, min(tile_size, A_length-A_consumed),
                                    B_s, min(tile_size, B_length-B_consumed));
        int b_next = c_next - a_next;

        /* All threads call the sequential merge function */
        merge_sequential (A_s+a_curr, a_next-a_curr, B_s+b_curr, b_next-b_curr,
                            C+C_curr+C_completed+c_curr);
        /* Update the number of A and B elements that have been consumed thus far */
        counter ++;
        C_completed += tile_size;
        A_consumed += co_rank(tile_size,  A_s, tile_size, B_s, tile_size);
        B_consumed = C_completed - A_consumed;
        __syncthreads();
    }

}
__device__
int co_rank_circular(int k, int* A, int m, int* B, int n, int A_s_start, int B_s_start, int tile_size) {
    int i = k < m ? k : m;  // i = min(k,m)
    int j = k - i;
    int i_low = 0 > (k-n) ? 0 : k-n; // i_low = max(0, k-n)
    int j_low = 0 > (k-m) ? 0 : k-m; // j_low = max(0,k-m)
    int delta;
    bool active = true;
    while(active) {
        int i_cir = (A_s_start+i) % tile_size;
        int i_m_1_cir = (A_s_start+i-1) % tile_size;
        int j_cir = (B_s_start+j) % tile_size;
        int j_m_1_cir = (B_s_start+i-1) % tile_size;
        if (i > 0 && j < n && A[i_m_1_cir] > B[j_cir]) {
            delta = ((i - i_low +1) >> 1) ; // ceil(i-i_low)/2)
            j_low = j;
            i = i - delta;
            j = j + delta;
        } else if (j > 0 && i < m && B[j_m_1_cir] >= A[i_cir]) {
            delta = ((j - j_low +1) >> 1) ;
            i_low = i;
            i = i + delta;
            j = j - delta;
        } else {
            active = false;
        }
    }
    return i;
}

__device__
void merge_sequential_circular(int *A, int m, int *B, int n, int *C, int A_s_start, int B_s_start, int tile_size) {
    int i = 0;  //virtual index into A
    int j = 0;  //virtual index into B
    int k = 0; //virtual index into C
    while ((i < m) && (j < n)) {
        int i_cir = (A_s_start + i) % tile_size;
        int j_cir = (B_s_start + j) % tile_size;
        if (A[i_cir] <= B[j_cir]) {
            C[k++] = A[i_cir]; i++;
        } else {
            C[k++] = B[j_cir]; j++;
        }
    }
    if (i == m) { //done with A[] handle remaining B[]
        for (; j < n; j++) {
            int j_cir = (B_s_start + j) % tile_size;
            C[k++] = B[j_cir];
        }
    } else { //done with B[], handle remaining A[]
        for (; i <m; i++) {
            int i_cir = (A_s_start + i) % tile_size;
            C[k++] = A[i_cir];
        }
    }
}

__global__
void merge_tiled_circular_kernel(int* A, int m, int* B, int n, int* C, int tile_size) {
    // Part 1: Identifying block-level output and input subarrays.
    extern __shared__ int shareAB[];
    int* A_s = &shareAB[0];
    int* B_s = &shareAB[tile_size];
    
    int elementsPerBlock = ceilf((m + n) / gridDim.x);
    int C_curr = blockIdx.x * elementsPerBlock; // start output index
    int C_next = min((blockIdx.x + 1) * elementsPerBlock, m + n); // end output index
    
    if(threadIdx.x == 0){
        A_s[0] = co_rank(C_curr, A, m, B, n);
        A_s[1] = co_rank(C_next, A, m, B, n);
    }

    __syncthreads();
    int A_curr = A_s[0];
    int A_next = A_s[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;

    __syncthreads();
    // Part 2: Loading A and B elements into the shared memory.
    int counter = 0;                                           //iteration counter
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = ceilf((C_length)/tile_size);          //total iteration
    int C_completed = 0;
    int A_s_start = 0; 
    int B_s_start = 0; 
    int A_consumed = 0;
    int B_consumed = 0;
    int A_s_consumed = tile_size;
    int B_s_consumed = tile_size;
    while(counter < total_iteration){
        /* loading tile-size A and B elements into shared memory */
        for(int i=0; i<A_s_consumed; i+=blockDim.x){
            if( i + threadIdx.x < A_length - A_consumed && i + threadIdx.x < A_s_consumed) {
                A_s[A_s_start + (tile_size - A_s_consumed) + i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x ];
            }
        }
        for(int i=0; i<tile_size; i+=blockDim.x) {
            if(i + threadIdx.x  < B_length - B_consumed && i + threadIdx.x < B_s_consumed) {
                B_s[B_s_start + (tile_size - B_s_consumed) + i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
            }
        }
        __syncthreads();

        //Part 3: All threads merge their individual subarrays in parallel.
        int c_curr = threadIdx.x * (tile_size/blockDim.x);
        int c_next = (threadIdx.x+1) * (tile_size/blockDim.x);

        c_curr = (c_curr <= C_length-C_completed) ? c_curr : C_length-C_completed;
        c_next = (c_next <= C_length-C_completed) ? c_next : C_length-C_completed;
        /* find co-rank for c_curr and c_next */
        int a_curr = co_rank_circular(c_curr,
                        A_s, min(tile_size, A_length-A_consumed),
                        B_s, min(tile_size, B_length-B_consumed),
                        A_s_start, B_s_start, tile_size);
        int b_curr = c_curr - a_curr;
        int a_next = co_rank_circular(c_next,
                        A_s, min(tile_size, A_length-A_consumed),
                        B_s, min(tile_size, B_length-B_consumed),
                        A_s_start, B_s_start, tile_size);

        int b_next = c_next - a_next;
        /* All threads call the circular-buffer version of the sequential merge function */
        merge_sequential_circular( A_s, a_next-a_curr,
                        B_s, b_next-b_curr,  C+C_curr+C_completed+c_curr,
                        A_s_start+a_curr, B_s_start+b_curr, tile_size);

        /* Figure out the work has been done */
        counter ++;
        A_s_consumed = co_rank_circular(min(tile_size,C_length-C_completed),
                        A_s, min(tile_size, A_length-A_consumed),
                        B_s, min(tile_size, B_length-B_consumed),
                        A_s_start, B_s_start, tile_size);

        B_s_consumed = min(tile_size, C_length-C_completed) - A_s_consumed;
        A_consumed += A_s_consumed;
        C_completed += min(tile_size, C_length-C_completed);
        B_consumed = C_completed - A_consumed;

        A_s_start = (A_s_start + A_s_consumed) % tile_size;
        B_s_start = (B_s_start + B_s_consumed) % tile_size;
        __syncthreads();

        __syncthreads();
    }

}


// Unified launch function for merge kernels
void launch_merge(int* A_h, int m, int* B_h, int n, int* C_h, merge_kernel_t kernel_type) {
    int A_size = m * sizeof(int);
    int B_size = n * sizeof(int);
    int C_size = (m + n) * sizeof(int);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Device memory allocation
    int *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, A_size);
    cudaMalloc(&B_d, B_size);
    cudaMalloc(&C_d, C_size);
    
    // Copy input data to device
    cudaMemcpy(A_d, A_h, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, B_size, cudaMemcpyHostToDevice);
    cudaMemset(C_d, 0, C_size);
    
    // Launch appropriate kernel based on type
    switch(kernel_type) {
        case MERGE_BASIC: {
            // Basic parallel merge with co-rank
            int threads_per_block = 256;
            int num_blocks = 1;
            
            printf("Basic merge: %d blocks, %d threads per block\n", num_blocks, threads_per_block);
            
            dim3 grid_dim(num_blocks);
            dim3 block_dim(threads_per_block);
            
            cudaEventRecord(start);
            merge_basic_kerner<<<grid_dim, block_dim>>>(A_d, m, B_d, n, C_d);
            cudaEventRecord(stop);
            break;
        }
        case MERGE_TILED: {
            // Tiled merge with shared memory
            int threads_per_block = 128;
            int num_blocks = ceil((m + n) / (float)(TILE_SIZE * 2)); // Conservative estimate
            num_blocks = max(1, num_blocks);
            
            printf("Tiled merge: %d blocks, %d threads per block, tile size: %d\n", 
                   num_blocks, threads_per_block, TILE_SIZE);
            
            dim3 grid_dim(num_blocks);
            dim3 block_dim(threads_per_block);
            
            // Shared memory size: TILE_SIZE for A + TILE_SIZE for B
            int shared_mem_size = 2 * TILE_SIZE * sizeof(int);
            
            cudaEventRecord(start);
            merge_tiled_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
                A_d, m, B_d, n, C_d, TILE_SIZE);
            cudaEventRecord(stop);
            break;
        }
        case MERGE_TILED_CIRCULAR: {
            // Tiled merge with circular buffer
            int threads_per_block = 128;
            int num_blocks = ceil((m + n) / (float)(TILE_SIZE * 2)); // Conservative estimate
            num_blocks = max(1, num_blocks);
            
            printf("Tiled circular merge: %d blocks, %d threads per block, tile size: %d\n", 
                   num_blocks, threads_per_block, TILE_SIZE);
            
            dim3 grid_dim(num_blocks);
            dim3 block_dim(threads_per_block);
            
            // Shared memory size: TILE_SIZE for A + TILE_SIZE for B
            int shared_mem_size = 2 * TILE_SIZE * sizeof(int);
            
            cudaEventRecord(start);
            merge_tiled_circular_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
                A_d, m, B_d, n, C_d, TILE_SIZE);
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
        "Basic Parallel Merge", 
        "Tiled Merge",
        "Tiled Circular Merge"
    };
    printf("[%s] Kernel execution time: %.6f ms\n", 
           kernel_names[kernel_type], milliseconds);
    
    // Copy result back to host
    cudaMemcpy(C_h, C_d, C_size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}