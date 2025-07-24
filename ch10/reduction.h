#ifndef REDUCTION_H
#define REDUCTION_H

// Reduction configuration
#define BLOCK_DIM 256       // Threads per block (must match kernel expectations)
#define CFACTOR 4          // Elements per thread for coarsening

// Reduction kernel types (following PDF chapter 10 progression)
typedef enum {
    REDUCTION_BASIC = 0,                    // Basic reduction with control divergence
    REDUCTION_CONTROL_DIVERGENCE = 1,       // Improved control divergence handling
    REDUCTION_SHARED_MEMORY = 2,            // Using shared memory optimization
    REDUCTION_HIERARCHICAL = 3,             // Hierarchical reduction across blocks
    REDUCTION_COARSENED = 4                 // Thread coarsening optimization
} reduction_kernel_t;

// Unified launch function
void launch_reduction(float* input_h, float* output_h, 
                     unsigned int length, reduction_kernel_t kernel_type);

#endif