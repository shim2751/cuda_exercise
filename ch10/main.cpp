#include "reduction.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <random>

void initialize_data(float* data, int length) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 10.0f);
    
    for (int i = 0; i < length; i++) {
        data[i] = dis(gen);
    }
}

float reduction_cpu(float* data, unsigned int length) {
    float sum = 0.0f;
    for (unsigned int i = 0; i < length; i++) {
        sum += data[i];
    }
    return sum;
}

bool check_results(float cpu_result, float gpu_result, float tolerance = 1e-3) {
    float diff = fabs(cpu_result - gpu_result);
    float relative_error = diff / fabs(cpu_result);
    
    if (relative_error > tolerance) {
        printf("Mismatch: CPU=%.6f, GPU=%.6f, Relative Error=%.6f\n", 
               cpu_result, gpu_result, relative_error);
        return false;
    }
    return true;
}

int main() {
    const unsigned int length = 1024 * 1024;  // 1M elements
    
    float* input_data = new float[length];
    float gpu_result = 0.0f;
    
    // Initialize random data
    initialize_data(input_data, length);
    
    printf("Processing %u float elements...\n\n", length);
    
    // CPU computation for verification
    clock_t start = clock();
    float cpu_result = reduction_cpu(input_data, length);
    double cpu_time = (double)(clock() - start) / CLOCKS_PER_SEC * 1000;
    printf("CPU: %.2f ms, Result: %.6f\n\n", cpu_time, cpu_result);
    
    // GPU kernel tests
    reduction_kernel_t kernels[] = {
        REDUCTION_BASIC,
        REDUCTION_CONTROL_DIVERGENCE,
        REDUCTION_SHARED_MEMORY,
        REDUCTION_HIERARCHICAL,
        REDUCTION_COARSENED
    };
    
    const char* names[] = {
        "Basic Reduction",
        "Control Divergence Optimized",
        "Shared Memory Optimized",
        "Hierarchical Reduction",
        "Coarsened Reduction"
    };
    
    // Run all kernels and collect results
    for (int i = 0; i < 5; i++) {
        gpu_result = 0.0f;
        
        printf("Testing %s:\n", names[i]);
        
        // Launch kernel (timing is done inside the function)
        launch_reduction(input_data, &gpu_result, length, kernels[i]);
        
        // Verify correctness
        printf("GPU Result: %.6f\n", gpu_result);
        printf("Correct: %s\n", check_results(cpu_result, gpu_result) ? "✓" : "✗");
        printf("\n");
    }
    
    // Cleanup
    delete[] input_data;
    
    return 0;
}