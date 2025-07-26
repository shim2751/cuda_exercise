#include "merge.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <random>

void initialize_sorted_data(int* data, int length, int start_val = 1) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(1, 10);
    
    data[0] = start_val;
    for (int i = 1; i < length; i++) {
        data[i] = data[i-1] + dis(gen);
    }
}

void merge_cpu(int* A, int m, int* B, int n, int* C) {
    int i = 0, j = 0, k = 0;
    while(i < m && j < n){
        if(A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    while(i < m) {
        C[k++] = A[i++];
    }
    while(j < n) {
        C[k++] = B[j++];
    }
}

bool check_results(int* cpu_result, int* gpu_result, int length) {
    for (int i = 0; i < length; i++) {
        if (cpu_result[i] != gpu_result[i]) {
            printf("Mismatch at index %d: CPU=%d, GPU=%d\n", 
                   i, cpu_result[i], gpu_result[i]);
            return false;
        }
    }
    return true;
}

void print_array(int* array, int length, const char* name, int max_print = 10) {
    printf("%s: [", name);
    int print_count = length < max_print ? length : max_print;
    for (int i = 0; i < print_count; i++) {
        printf("%d", array[i]);
        if (i < print_count - 1) printf(", ");
    }
    if (length > max_print) printf(", ...");
    printf("]\n");
}

int main() {
    const int m = 256;  // Size of array A
    const int n = 256;  // Size of array B
    const int total_size = m + n;

    int* A = new int[m];
    int* B = new int[n];
    int* cpu_result = new int[total_size];
    int* gpu_result = new int[total_size];
    
    // Initialize sorted arrays
    initialize_sorted_data(A, m, 1);
    initialize_sorted_data(B, n, 2);
    
    printf("Merging two sorted arrays:\n");
    printf("Array A size: %d, Array B size: %d, Output size: %d\n\n", m, n, total_size);
    
    // Print sample of input arrays
    print_array(A, m, "Array A");
    print_array(B, n, "Array B");
    printf("\n");
    
    // CPU computation for verification
    clock_t start = clock();
    merge_cpu(A, m, B, n, cpu_result);
    double cpu_time = (double)(clock() - start) / CLOCKS_PER_SEC * 1000;
    printf("CPU: %.2f ms\n", cpu_time);
    print_array(cpu_result, total_size, "CPU Result");
    printf("\n");
    
    // GPU kernel tests
    merge_kernel_t kernels[] = {
        MERGE_BASIC,
        MERGE_TILED,
        MERGE_TILED_CIRCULAR
    };
    
    const char* names[] = {
        "Basic Parallel Merge",
        "Tiled Merge",
        "Tiled Circular Merge"
    };
    
    // Run all kernels and collect results
    for (int i = 0; i < 3; i++) {
        // Clear GPU output buffer
        for (int j = 0; j < total_size; j++) {
            gpu_result[j] = 0;
        }
        
        printf("Testing %s:\n", names[i]);
        
        // Launch kernel (timing is done inside the function)
        launch_merge(A, m, B, n, gpu_result, kernels[i]);
        
        // Verify correctness
        printf("Results match CPU: %s\n", check_results(cpu_result, gpu_result, total_size) ? "✓" : "✗");
        print_array(gpu_result, total_size, "GPU Result");
        printf("\n");
    }
    
    // Test with larger arrays
    printf("=== Testing with Larger Arrays ===\n");
    const int large_m = 1024 * 2;
    const int large_n = 1024 * 3;
    const int large_total = large_m + large_n;
    
    int* large_A = new int[large_m];
    int* large_B = new int[large_n];
    int* large_cpu = new int[large_total];
    int* large_gpu = new int[large_total];
    
    initialize_sorted_data(large_A, large_m, 1);
    initialize_sorted_data(large_B, large_n, 2);
    
    printf("Large test: A=%d elements, B=%d elements, Total=%d elements\n", 
           large_m, large_n, large_total);
    
    // CPU reference
    start = clock();
    merge_cpu(large_A, large_m, large_B, large_n, large_cpu);
    cpu_time = (double)(clock() - start) / CLOCKS_PER_SEC * 1000;
    printf("CPU: %.2f ms\n", cpu_time);
    
    // Test tiled versions with larger data
    merge_kernel_t large_kernels[] = {MERGE_TILED, MERGE_TILED_CIRCULAR};
    const char* large_names[] = {"Tiled Merge", "Tiled Circular Merge"};
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < large_total; j++) {
            large_gpu[j] = 0;
        }
        
        printf("\nTesting %s with large arrays:\n", large_names[i]);
        launch_merge(large_A, large_m, large_B, large_n, large_gpu, large_kernels[i]);
        printf("Results match CPU: %s\n", 
               check_results(large_cpu, large_gpu, large_total) ? "✓" : "✗");
    }
    
    // Cleanup
    delete[] A;
    delete[] B;
    delete[] cpu_result;
    delete[] gpu_result;
    delete[] large_A;
    delete[] large_B;
    delete[] large_cpu;
    delete[] large_gpu;
    
    return 0;
}