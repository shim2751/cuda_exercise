#ifndef MERGE_H
#define MERGE_H

// Merge configuration
#define TILE_SIZE 32       // Elements per tile for shared memory

// Merge kernel types
typedef enum {
    MERGE_BASIC = 0,        // Basic parallel merge with co-rank
    MERGE_TILED = 1,        // Tiled merge with shared memory
    MERGE_TILED_CIRCULAR = 2 // Tiled merge with circular buffer
} merge_kernel_t;

// Unified launch function
void launch_merge(int* A_h, int m, int* B_h, int n, int* C_h, merge_kernel_t kernel_type);

#endif