#include "scan.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <random>

void initialize_data(float* data, int length) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 5.0f);
    
    for (int i = 0; i < length; i++) {
        data[i] = dis(gen);
    }
}

void scan_cpu(float* input, float* output, unsigned int length) {
    output[0] = input[0];
    for (unsigned int i = 1; i < length; i++) {
        output[i] = output[i-1] + input[i];
    }
}

bool check_results(float* cpu_result, float* gpu_result, unsigned int length, float tolerance = 1e-3) {
    for (unsigned int i = 0; i < length; i++) {
        float diff = fabs(cpu_result[i] - gpu_result[i]);
        float relative_error = diff / fabs(cpu_result[i]);
        
        if (relative_error > tolerance) {
            printf("Mismatch at index %u: CPU=%.6f, GPU=%.6f, Relative Error=%.6f\n", 
                   i, cpu_result[i], gpu_result[i], relative_error);
            return false;
        }
    }
    return true;
}

int main() {
    const unsigned int length = SECTION_SIZE;  // Use section size for testing
    
    float* input_data = new float[length];
    float* cpu_output = new float[length];
    float* gpu_output = new float[length];
    
    // Initialize random data
    initialize_data(input_data, length);
    
    printf("Processing %u float elements (prefix sum)...\n\n", length);
    
    // CPU computation for verification
    clock_t start = clock();
    scan_cpu(input_data, cpu_output, length);
    double cpu_time = (double)(clock() - start) / CLOCKS_PER_SEC * 1000;
    printf("CPU: %.2f ms, Final sum: %.6f\n\n", cpu_time, cpu_output[length-1]);
    
    // GPU kernel tests
    scan_kernel_t kernels[] = {
        SCAN_SEQUENTIAL,
        SCAN_KOGGE_STONE,
        SCAN_BRENT_KUNG,
        SCAN_COARSENED
    };
    
    const char* names[] = {
        "Sequential Scan",
        "Kogge-Stone Scan",
        "Brent-Kung Scan", 
        "Coarsened Scan"
    };
    
    // Run all kernels and collect results
    for (int i = 0; i < 4; i++) {
        // Clear GPU output buffer
        for (unsigned int j = 0; j < length; j++) {
            gpu_output[j] = 0.0f;
        }
        
        printf("Testing %s:\n", names[i]);
        
        // Launch kernel (timing is done inside the function)
        launch_scan(input_data, gpu_output, length, kernels[i]);
        
        // Verify correctness
        printf("GPU Final sum: %.6f\n", gpu_output[length-1]);
        printf("Correct: %s\n", check_results(cpu_output, gpu_output, length) ? "✓" : "✗");
        printf("\n");
    }
    
    // Print sample of results for verification
    printf("Sample results:\n");
    printf("Index\tInput\tCPU\tGPU (last kernel)\n");
    for (int i = 26; i < 36 && i < length; i++) {
        printf("%d\t%.2f\t%.2f\t%.2f\n", 
               i, input_data[i], cpu_output[i], gpu_output[i]);
    }
    
    // Cleanup
    delete[] input_data;
    delete[] cpu_output;
    delete[] gpu_output;
    
    return 0;
}