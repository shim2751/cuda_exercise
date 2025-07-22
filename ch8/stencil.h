#ifndef STENCIL_H
#define STENCIL_H

// Stencil constants (assuming 7-point stencil coefficients)
#define BLOCK_SIZE 8
#define OUT_TILE_DIM 6
#define IN_TILE_DIM (OUT_TILE_DIM + 2)
#define OUT_TILE_DIM_C 4
#define IN_TILE_DIM_C (OUT_TILE_DIM_C + 2)

// Stencil kernel types
typedef enum {
    STENCIL_BASIC = 0,
    STENCIL_SHARED_MEMORY = 1,
    STENCIL_THREAD_COARSENING = 2,
    STENCIL_REGISTER_TILING = 3
} stencil_kernel_t;

// Stencil coefficients (to be defined in implementation)
extern __constant__ float c0, c1, c2, c3, c4, c5, c6;

// Unified launch function
void launch_stencil(float* in_h, float* out_h, unsigned int N, stencil_kernel_t kernel_type);

#endif