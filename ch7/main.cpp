#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <random>
#include "conv.h"

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

bool check_results(float* cpu, float* gpu, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(cpu[i] - gpu[i]) > 1e-4) {
            return false;
        }
    }
    return true;
}

int main() {
    const int width = 1024, height = 1024, r = FILTER_RADIUS;
    const int size = width * height;
    const int filter_size = (2*r+1) * (2*r+1);
    
    float* N_h = new float[size];
    float* F_h = new float[filter_size]; 
    float* cpu_out = new float[size];  

    // Initialize data
    initialize_matrix(N_h, size);
    initialize_matrix(F_h, filter_size);

    // CPU computation for verification
    clock_t start = clock();
    convolution2D_cpu(N_h, F_h, cpu_out, r, width, height);
    double cpu_time = (double)(clock() - start) / CLOCKS_PER_SEC * 1000;
    printf("CPU: %.2f ms\n\n", cpu_time);

    // GPU kernel tests using unified launch function
    conv_kernel_t kernels[] = {CONV_BASIC, CONV_CONSTANT_MEM, CONV_TILED, CONV_CACHED_TILED};
    const char* names[] = {"Basic", "Constant Memory", "Tiled", "Cached Tiled"};

    // Run all kernels and collect results
    for (int i = 0; i < 4; i++) {
        float* gpu_out = new float[size]();
        // Launch kernel (timing is done inside the function)
        launch_convolution2D(N_h, F_h, gpu_out, r, width, height, kernels[i]);
        printf("Correct: %s\n", check_results(cpu_out, gpu_out, size) ? "✓" : "✗");
        delete[] gpu_out;
    }
    
    // Cleanup
    delete[] N_h;
    delete[] F_h;
    delete[] cpu_out;
    
    return 0;
}