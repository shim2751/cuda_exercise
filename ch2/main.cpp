// main.cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "vec_add.h"

int main() {
    const int n = 1000;  // 벡터 크기
    
    // 호스트 메모리 할당
    std::vector<float> A(n), B(n), C(n);
    
    // 데이터 초기화
    for (int i = 0; i < n; i++) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(i * 2);
    }
    
    std::cout << "Vector size: " << n << std::endl;
    
    // 시간 측정 시작
    auto start = std::chrono::high_resolution_clock::now();
    
    // CUDA 벡터 덧셈 수행
    vec_add(A.data(), B.data(), C.data(), n);
    
    // CUDA 동기화
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "CUDA execution time: " << duration.count() << " microseconds" << std::endl;
    
    // 결과 검증 (처음 10개 원소만)
    std::cout << "\nFirst 10 results:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "C[" << i << "] = " << C[i] 
                  << " (expected: " << A[i] + B[i] << ")" << std::endl;
    }
    
    // 전체 결과 검증
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (abs(C[i] - (A[i] + B[i])) > 1e-5) {
            correct = false;
            std::cout << "Error at index " << i << ": " 
                      << C[i] << " != " << A[i] + B[i] << std::endl;
            break;
        }
    }
    
    if (correct) {
        std::cout << "\nAll results are correct!" << std::endl;
    } else {
        std::cout << "\nSome results are incorrect!" << std::endl;
    }
    
    return 0;
}