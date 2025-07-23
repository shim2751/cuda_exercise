#include "histogram.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <random>
#include <string.h>

void initialize_data(char* data, int length) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 25);  // a-z range
    
    for (int i = 0; i < length; i++) {
        data[i] = 'a' + dis(gen);
    }
    data[length] = '\0';  // null terminate
}

void histogram_cpu(char* data, unsigned int* histo, unsigned int length) {
    // Initialize histogram to zero
    for (int i = 0; i < NUM_BINS; i++) {
        histo[i] = 0;
    }
    
    // Count occurrences
    for (unsigned int i = 0; i < length; i++) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            histo[alphabet_position / 4]++;
        }
    }
}

bool check_results(unsigned int* cpu, unsigned int* gpu, int bins) {
    for (int i = 0; i < bins; i++) {
        if (cpu[i] != gpu[i]) {
            printf("Mismatch at bin %d: CPU=%u, GPU=%u\n", i, cpu[i], gpu[i]);
            return false;
        }
    }
    return true;
}

void print_histogram(unsigned int* histo, const char* label) {
    printf("%s Histogram:\n", label);
    const char* bin_labels[] = {"a-d", "e-h", "i-l", "m-p", "q-t", "u-x", "y-z"};
    for (int i = 0; i < NUM_BINS; i++) {
        printf("  %s: %u\n", bin_labels[i], histo[i]);
    }
    printf("\n");
}

int main() {
    const unsigned int length = 1024 * 1024;  // 1M characters
    
    char* data = new char[length + 1];
    unsigned int* cpu_histo = new unsigned int[NUM_BINS];
    
    // Initialize random text data
    initialize_data(data, length);
    
    printf("Processing %u characters...\n\n", length);
    
    // CPU computation for verification
    clock_t start = clock();
    histogram_cpu(data, cpu_histo, length);
    double cpu_time = (double)(clock() - start) / CLOCKS_PER_SEC * 1000;
    printf("CPU: %.2f ms\n", cpu_time);
    print_histogram(cpu_histo, "CPU");
    
    // GPU kernel tests
    histogram_kernel_t kernels[] = {
        HISTOGRAM_BASIC, 
        HISTOGRAM_PRIVATIZED_GLOBAL,
        HISTOGRAM_PRIVATIZED_SHARED, 
        HISTOGRAM_COARSENED_CONTIGUOUS,
        HISTOGRAM_COARSENED_INTERLEAVED,
        HISTOGRAM_AGGREGATED
    };
    
    const char* names[] = {
        "Basic Atomic",
        "Privatized (Global Memory)",
        "Privatized (Shared Memory)", 
        "Coarsened Contiguous",
        "Coarsened Interleaved",
        "Aggregated"
    };
    
    // Run all kernels and collect results
    for (int i = 0; i < 6; i++) {
        unsigned int* gpu_histo = new unsigned int[NUM_BINS]();
        
        // Launch kernel (timing is done inside the function)
        launch_histogram(data, gpu_histo, length, kernels[i]);
        
        // Verify correctness
        printf("Correct: %s\n", check_results(cpu_histo, gpu_histo, NUM_BINS) ? "✓" : "✗");
        
        delete[] gpu_histo;
        printf("\n");
    }
    
    // Cleanup
    delete[] data;
    delete[] cpu_histo;
    
    return 0;
}