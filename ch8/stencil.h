#ifndef STENCIL_H
#define STENCIL_H

// Stencil constants (assuming 7-point stencil coefficients)
#define BLOCK_SIZE 8
#define IN_TILE_DIM 8 
#define OUT_TILE_DIM (IN_TILE_DIM - 2)
#define IN_TILE_DIM_C 8 
#define OUT_TILE_DIM_C (IN_TILE_DIM_C - 2)

// Stencil kernel types
typedef enum {
    STENCIL_BASIC = 0,
    STENCIL_SHARED_MEMORY = 1,
    STENCIL_THREAD_COARSENING = 2,
    STENCIL_REGISTER_TILING = 3
} stencil_kernel_t;


// Unified launch function
void launch_stencil(float* in_h, float* out_h, unsigned int N, stencil_kernel_t kernel_type);

#endif