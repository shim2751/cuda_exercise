#include <iostream>
#include <chrono>
#include <random>
#include <cmath>
#include <cuda_runtime.h>

#include "ch5.h"

bool verify_result(float* A, float* B, int size, float tolerance = 1e-3) {
    for (int i = 0; i < size; i++) {
        if (fabs(A[i] - B[i]) > tolerance) return false;
    }
    return true;
}

void initialize_matrix(float* matrix, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < size; i++) {
        matrix[i] = dis(gen);
    }
}

int main() {
    int width = 544;
    int size = width * width;
    
    float* A = new float[size];
    float* B = new float[size];
    float* C_ch3 = new float[size];
    float* C_ch5 = new float[size];
    
    initialize_matrix(A, size);
    initialize_matrix(B, size);
    
    // CH3 성능 측정
    auto start = std::chrono::high_resolution_clock::now();
    matrix_mul_ch3(A, B, C_ch3, width);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto ch3_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // CH5 성능 측정
    start = std::chrono::high_resolution_clock::now();
    matrix_mul_ch5(A, B, C_ch5, width);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto ch5_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Matrix size: " << width << "x" << width << std::endl;
    std::cout << "CH3 time: " << ch3_time.count() << " μs" << std::endl;
    std::cout << "CH5 time: " << ch5_time.count() << " μs" << std::endl;
    std::cout << "Speedup: " << (double)ch3_time.count() / ch5_time.count() << "x" << std::endl;
    
    // 정확성 검증
    if (verify_result(C_ch3, C_ch5, size)) {
        std::cout << "✓ Results match" << std::endl;
    } else {
        std::cout << "✗ Results differ" << std::endl;
    }
    
    delete[] A;
    delete[] B;
    delete[] C_ch3;
    delete[] C_ch5;
    
    return 0;
}