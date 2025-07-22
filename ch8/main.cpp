#include "stencil.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void stencil_cpu(float* in, float* out, int N) {
    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {
            for (int k = 1; k < N-1; k++) {
                out[i*N*N + j*N + k] = -6*in[i*N*N + j*N + k]
                    + in[i*N*N + j*N + (k-1)] + in[i*N*N + j*N + (k+1)]
                    + in[i*N*N + (j-1)*N + k] + in[i*N*N + (j+1)*N + k]
                    + in[(i-1)*N*N + j*N + k] + in[(i+1)*N*N + j*N + k];
            }
        }
    }
}

bool check_results(float* cpu, float* gpu, int N) {
    for (int i = 0; i < N*N*N; i++) {
        if (fabsf(cpu[i] - gpu[i]) > 1e-4) return false;
    }
    return true;
}

int main() {
    const int N = 64;
    int size = N * N * N * sizeof(float);
    
    float *input = (float*)malloc(size);
    float *cpu_out = (float*)malloc(size);
    float *gpu_out = (float*)malloc(size);
    
    // Initialize random input
    srand(123);
    for (int i = 0; i < N*N*N; i++) input[i] = rand() % 100;
    
    // CPU test
    clock_t start = clock();
    stencil_cpu(input, cpu_out, N);
    double cpu_time = (double)(clock() - start) / CLOCKS_PER_SEC * 1000;
    printf("CPU: %.2f ms\n", cpu_time);
    
    // GPU tests
    stencil_kernel_t kernels[] = {STENCIL_BASIC, STENCIL_SHARED_MEMORY, 
                                  STENCIL_THREAD_COARSENING, STENCIL_REGISTER_TILING};
    const char* names[] = {"Basic", "Shared", "Coarsening", "Register"};
    
    for (int i = 0; i < 4; i++) {
        launch_stencil(input, gpu_out, N, kernels[i]);
        printf("%s: %s\n", names[i], check_results(cpu_out, gpu_out, N) ? "✓" : "✗");
    }
    
    free(input); free(cpu_out); free(gpu_out);
    return 0;
}