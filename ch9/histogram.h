#ifndef HISTOGRAM_H
#define HISTOGRAM_H

// Histogram configuration
#define NUM_BINS 7          // Number of histogram bins (a-d, e-h, i-l, m-p, q-t, u-x, y-z)
#define BLOCK_SIZE 256      // Threads per block
#define CFACTOR 4 // Elements per thread for coarsening

// Histogram kernel types (following PDF chapter 9 progression)
typedef enum {
    HISTOGRAM_BASIC = 0,                    // Basic atomic operations (Fig 9.6)
    HISTOGRAM_PRIVATIZED_GLOBAL = 1,        // Privatization in global memory (Fig 9.9)
    HISTOGRAM_PRIVATIZED_SHARED = 2,        // Privatization in shared memory (Fig 9.10)
    HISTOGRAM_COARSENED_CONTIGUOUS = 3,     // Thread coarsening - contiguous partitioning (Fig 9.12)
    HISTOGRAM_COARSENED_INTERLEAVED = 4,    // Thread coarsening - interleaved partitioning (Fig 9.14)
    HISTOGRAM_AGGREGATED = 5                // Aggregation optimization (Fig 9.15)
} histogram_kernel_t;

// Unified launch function
void launch_histogram(char* data_h, unsigned int* histo_h, 
                     unsigned int length, histogram_kernel_t kernel_type);

#endif