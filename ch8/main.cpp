#include "stencil.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <random>


void initialize_matrix(float* matrix, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < size; i++) {
        matrix[i] = dis(gen);
    }
}

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
        if (fabsf(cpu[i] - gpu[i]) > 1e-3) return false;
    }
    return true;
}

int main() {
    const int N = 256;
    int size = N * N * N;
    
    float* input = new float[size];
    float* cpu_out = new float[size];
    
    // Initialize random input
    initialize_matrix(input, size);
    
    // CPU test
    clock_t start = clock();
    stencil_cpu(input, cpu_out, N);
    double cpu_time = (double)(clock() - start) / CLOCKS_PER_SEC * 1000;
    printf("CPU: %.2f ms\n", cpu_time);
    
    // GPU tests
    stencil_kernel_t kernels[] = {STENCIL_BASIC, STENCIL_SHARED_MEMORY, 
                                  STENCIL_THREAD_COARSENING, STENCIL_REGISTER_TILING};
    
    for (int i = 0; i < 4; i++) {
        float* gpu_out = new float[size]();
        launch_stencil(input, gpu_out, N, kernels[i]);
        printf("Correct: %s\n", check_results(cpu_out, gpu_out, N) ? "✓" : "✗");
        delete[] gpu_out;
    }
    
    delete[] input; delete[] cpu_out; 
    return 0;
}