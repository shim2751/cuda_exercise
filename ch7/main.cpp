#include <iostream>
#include <vector>
#include <cmath>
#include "conv.h"

// CPU reference implementation
void convolution2D_cpu(float* N, float* F, float* P, int r, int width, int height) {
    int inRow, inCol;
    
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float Pval = 0.0;
            for(int i = 0; i < 2*r+1; i++){
                for(int j = 0; j < 2*r+1; j++){
                    inRow = row - r + i;
                    inCol = col - r + j;
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                        Pval += N[inRow * width + inCol] * F[i * (2*r+1) + j];
                }
            }
            P[row * width + col] = Pval;
        }
    }
}

int main() {
    // Test parameters
    const int width = 8, height = 8, r = 1;
    const int size = width * height;
    const int filter_size = (2*r+1) * (2*r+1);
    
    // Host memory allocation
    std::vector<float> h_N(size), h_F(filter_size), h_P_gpu(size), h_P_cpu(size);
    
    // Initialize input image
    for (int i = 0; i < size; i++) {
        h_N[i] = i % 10;
    }
    
    // Initialize filter (edge detection)
    float filter[] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
    for (int i = 0; i < filter_size; i++) {
        h_F[i] = filter[i];
    }
    
    // GPU computation
    launch_convolution2D_basic(h_N.data(), h_F.data(), h_P_gpu.data(), r, width, height);
    
    // CPU reference
    convolution2D_cpu(h_N.data(), h_F.data(), h_P_cpu.data(), r, width, height);
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < size; i++) {
        float error = fabs(h_P_gpu[i] - h_P_cpu[i]);
        if (error > 1e-5) {
            correct = false;
        }
    }
    
    std::cout << "Verification: " << (correct ? "PASSED" : "FAILED") << std::endl;
    
    return 0;
}
