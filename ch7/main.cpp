#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include "conv.h"

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

int main() {
    const int width = 4096, height = 4096, r = 2;
    const int size = width * height;
    const int filter_size = (2*r+1) * (2*r+1);
    
    // Memory allocation
    std::vector<float> h_N(size), h_F(filter_size), h_P1(size), h_P2(size), h_P3(size), h_P4(size), h_P_cpu(size);
    
    // Initialize data
    for (int i = 0; i < size; i++) h_N[i] = i % 100;
    for (int i = 0; i < filter_size; i++) h_F[i] = 1.0f / filter_size;  // Average filter
    
    // CPU computation
    convolution2D_cpu(h_N.data(), h_F.data(), h_P_cpu.data(), r, width, height);
    
    // Warm up
    launch_convolution2D_basic(h_N.data(), h_F.data(), h_P1.data(), r, width, height);
    launch_convolution2D_constant_mem(h_N.data(), h_F.data(), h_P2.data(), r, width, height);
    launch_convolution2D_tiled(h_N.data(), h_F.data(), h_P3.data(), r, width, height);
    launch_convolution2D_cached_tiled(h_N.data(), h_F.data(), h_P4.data(), r, width, height);
    
    // Timing basic version
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        launch_convolution2D_basic(h_N.data(), h_F.data(), h_P1.data(), r, width, height);
    }
    auto basic_time = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - start).count() / 10;
    
    // Timing constant memory version
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        launch_convolution2D_constant_mem(h_N.data(), h_F.data(), h_P2.data(), r, width, height);
    }
    auto constant_time = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - start).count() / 10;
    
    // Timing tiled version
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        launch_convolution2D_tiled(h_N.data(), h_F.data(), h_P3.data(), r, width, height);
    }
    auto tiled_time = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - start).count() / 10;
    
    // Timing cached tiled version
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        launch_convolution2D_cached_tiled(h_N.data(), h_F.data(), h_P4.data(), r, width, height);
    }
    auto cached_tiled_time = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - start).count() / 10;

    // Results
    printf("Basic GPU:        %.3f ms\n", basic_time);
    printf("Constant Memory:  %.3f ms\n", constant_time);
    printf("Tiled:            %.3f ms\n", tiled_time);
    printf("Speedup (const):  %.2fx\n", basic_time / constant_time);
    printf("Speedup (tiled):  %.2fx\n", basic_time / tiled_time);
    printf("Speedup (cached_tiled):  %.2fx\n", basic_time / cached_tiled_time);
    
    // Verify with CPU
    bool basic_correct = true, constant_correct = true, tiled_correct = true, cached_tiled_correct = true;
    for (int i = 0; i < size; i++) {
        if (fabs(h_P1[i] - h_P_cpu[i]) > 1e-5) basic_correct = false;
        if (fabs(h_P2[i] - h_P_cpu[i]) > 1e-5) constant_correct = false;
        if (fabs(h_P3[i] - h_P_cpu[i]) > 1e-5) tiled_correct = false;
        if (fabs(h_P4[i] - h_P_cpu[i]) > 1e-5) cached_tiled_correct = false;
    }
    printf("Basic vs CPU:     %s\n", basic_correct ? "PASSED" : "FAILED");
    printf("Constant vs CPU:  %s\n", constant_correct ? "PASSED" : "FAILED");
    printf("Tiled vs CPU:     %s\n", tiled_correct ? "PASSED" : "FAILED");
    printf("Cached tiled vs CPU:     %s\n", cached_tiled_correct ? "PASSED" : "FAILED");
    
    return 0;
}