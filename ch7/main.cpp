#include <iostream>
#include <vector>
#include <cmath>
#include "conv.h"
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <random>

void initialize_matrix(float* matrix, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < size; i++) {
        matrix[i] = dis(gen);
    }
}

// CPU reference implementation
void convolution2D_cpu(float* N, float* F, float* P, int r, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float Pval = 0.0;
            for(int i = 0; i < 2*r+1; i++){
                for(int j = 0; j < 2*r+1; j++){
                    int inRow = row - r + i;
                    int inCol = col - r + j;
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                        Pval += N[inRow * width + inCol] * F[i * (2*r+1) + j];
                }
            }
            P[row * width + col] = Pval;
        }
    }
}

float time_kernel(void (*kernel_func)(float*, float*, float*, int, int, int), 
                  float* N, float* F, float* P, int r, int width, int height) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm up
    kernel_func(N, F, P, r, width, height);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    kernel_func(N, F, P, r, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return elapsed_time;
}

int main() {
    const int width = 1024, height = 1024, r = FILTER_RADIUS;
    const int size = width * height;
    const int filter_size = (2*r+1) * (2*r+1);
    
    float* h_N = new float[size];
    float* h_F = new float[filter_size]; 

    float* h_P_cpu = new float[size];  
    float* h_P1 = new float[size];
    float* h_P2 = new float[size];
    float* h_P3 = new float[size];
    float* h_P4 = new float[size];

    // Initialize data
    initialize_matrix(h_N, size);
    initialize_matrix(h_F, filter_size);

    convolution2D_cpu(h_N, h_F, h_P_cpu, r, width, height);
    // Time GPU kernels
    float basic_time = time_kernel(launch_convolution2D_basic, h_N, h_F, h_P1, r, width, height);
    float constant_time = time_kernel(launch_convolution2D_constant_mem, h_N, h_F, h_P2, r, width, height);
    float tiled_time = time_kernel(launch_convolution2D_tiled, h_N, h_F, h_P3, r, width, height);
    float cached_tiled_time = time_kernel(launch_convolution2D_cached_tiled, h_N, h_F, h_P4, r, width, height);

    // Results
    printf("Basic GPU:        %.3f ms\n", basic_time);
    printf("Constant Memory:  %.3f ms\n", constant_time);
    printf("Tiled:            %.3f ms\n", tiled_time);
    printf("Cached Tiled:     %.3f ms\n\n", cached_tiled_time);

    printf("Speedup (const):  %.2fx\n", basic_time / constant_time);
    printf("Speedup (tiled):  %.2fx\n", basic_time / tiled_time);
    printf("Speedup (cached_tiled):  %.2fx\n\n", basic_time / cached_tiled_time);
    
    // Verify with CPU
    bool results[4] = {true, true, true, true};
    std::vector<float*> outputs = {h_P1, h_P2, h_P3, h_P4};
    const char* names[] = {"Basic", "Constant", "Tiled", "Cached Tiled"};
    
    for (int k = 0; k < 4; k++) {
        for (int i = 0; i < size; i++) {
            if (fabs(outputs[k][i] - h_P_cpu[i]) > 1e-4) {
                results[k] = false;
                break;
            }
        }
        printf("%s vs CPU: %s\n", names[k], results[k] ? "PASSED" : "FAILED");
    }
    
    delete[] h_N;
    delete[] h_F;
    delete[] h_P1;
    delete[] h_P2;
    delete[] h_P3;
    delete[] h_P4;
    delete[] h_P_cpu;
    return 0;
}