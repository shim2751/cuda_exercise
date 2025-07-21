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

// ch7/main.cpp의 time_kernel 함수를 참고하여 수정한 시간 측정 함수
float time_kernel(void (*kernel_func)(float*, float*, float*, int),
                  float* A, float* B, float* C, int width, int iterations = 10) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm up
    kernel_func(A, B, C, width);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel_func(A, B, C, width);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsed_time / iterations;
}


int main() {
    int width = 1280;
    int size = width * width;

    float* A = new float[size];
    float* B = new float[size];
    float* C_ch3 = new float[size];
    float* C_ch5 = new float[size];
    float* C_ch6 = new float[size];

    initialize_matrix(A, size);
    initialize_matrix(B, size);

    // CH3 성능 측정
    float ch3_time = time_kernel(matrix_mul_ch3, A, B, C_ch3, width);

    // CH5 성능 측정
    float ch5_time = time_kernel(matrix_mul_ch5, A, B, C_ch5, width);

    // CH6 성능 측정
    float ch6_time = time_kernel(matrix_mul_ch6, A, B, C_ch6, width);

    std::cout << "Matrix size: " << width << "x" << width << std::endl;
    std::cout << "CH3 time: " << ch3_time << " ms" << std::endl;
    std::cout << "CH5 time: " << ch5_time << " ms" << std::endl;
    std::cout << "CH6 time: " << ch6_time << " ms" << std::endl;
    std::cout << "CH5 vs CH3 speedup: " << ch3_time / ch5_time << "x" << std::endl;
    std::cout << "CH6 vs CH3 speedup: " << ch3_time / ch6_time << "x" << std::endl;
    std::cout << "CH6 vs CH5 speedup: " << ch5_time / ch6_time << "x" << std::endl;

    // 정확성 검증
    // CH5, CH6의 결과를 CH3와 비교하기 위해 CH3를 다시 실행하여 C_ch3를 채웁니다.
    matrix_mul_ch3(A, B, C_ch3, width);
    bool ch5_matches = verify_result(C_ch3, C_ch5, size);
    bool ch6_matches = verify_result(C_ch3, C_ch6, size);

    std::cout << "\n=== Results Verification ===" << std::endl;
    std::cout << "CH5 vs CH3: " << (ch5_matches ? "✓ Match" : "✗ Differ") << std::endl;
    std::cout << "CH6 vs CH3: " << (ch6_matches ? "✓ Match" : "✗ Differ") << std::endl;

    delete[] A;
    delete[] B;
    delete[] C_ch3;
    delete[] C_ch5;
    delete[] C_ch6;

    return 0;
}