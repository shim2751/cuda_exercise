#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <cmath>

#include "ch3.h"

int main() {
    int width = 4, height = 4;
    
    // Simple 4x4 RGB image (red, green, blue, white)
    unsigned char colorImage[48] = {
        255,0,0,  0,255,0,  0,0,255,  255,255,255,  // first row
        255,0,0,  0,255,0,  0,0,255,  255,255,255,  // second row  
        255,0,0,  0,255,0,  0,0,255,  255,255,255,  // third row
        255,0,0,  0,255,0,  0,0,255,  255,255,255   // fourth row
    };
    
    unsigned char grayImage[16];
    unsigned char blurredImage[48];
    
    // GPU grayscale conversion
    color_to_grayscale(colorImage, grayImage, width, height);
    
    // GPU image blur
    int radius = 1;
    image_blur(colorImage, blurredImage, width, height, radius);
    
    // Grayscale result verification
    printf("=== Grayscale Result Verification ===\n");
    unsigned char expectedGray[16];
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            int idx = (i * width + j) * 3;
            unsigned char r = colorImage[idx];
            unsigned char g = colorImage[idx+1];
            unsigned char b = colorImage[idx+2];
            expectedGray[i * width + j] = (unsigned char)(0.21f*r + 0.71f*g + 0.07f*b);
        }
    }
    
    bool grayCorrect = true;
    for(int i = 0; i < 16; i++) {
        if(grayImage[i] != expectedGray[i]) {
            grayCorrect = false;
            break;
        }
    }
    printf("Grayscale result: %s\n", grayCorrect ? "Correct" : "Error");
    
    // Blur result verification
    printf("\n=== Blur Result Verification ===\n");
    unsigned char expectedBlur[48];
    
    // CPU blur implementation for verification
    for(int row = 0; row < height; row++) {
        for(int col = 0; col < width; col++) {
            for(int ch = 0; ch < 3; ch++) {
                int pixelVal = 0;
                int pixelNum = 0;
                
                // Apply blur with radius 1
                for(int i = -radius; i < radius+1; i++) {
                    for(int j = -radius; j < radius+1; j++) {
                        int curRow = row + i;
                        int curCol = col + j;
                        
                        if(curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                            pixelVal += colorImage[(curRow * width + curCol) * 3 + ch];
                            pixelNum++;
                        }
                    }
                }
                
                expectedBlur[(row * width + col) * 3 + ch] = pixelVal / pixelNum;
            }
        }
    }
    
    // Check if blur results match
    bool blurCorrect = true;
    for(int i = 0; i < 48; i++) {
        if(blurredImage[i] != expectedBlur[i]) {
            blurCorrect = false;
            break;
        }
    }
    printf("Blur result: %s\n", blurCorrect ? "Correct" : "Error");
    
    // Matrix multiplication verification
    printf("\n=== Matrix Multiplication Verification ===\n");
    
    int matWidth = 4;
    float A[16] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    
    float B[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    
    float C_gpu[16];
    float C_cpu[16];
    
    // Initialize result matrices to zero
    memset(C_gpu, 0, sizeof(float) * 16);
    memset(C_cpu, 0, sizeof(float) * 16);
    
    // GPU matrix multiplication
    matrix_mul(A, B, C_gpu, matWidth);
    
    // CPU matrix multiplication for verification
    for(int i = 0; i < matWidth; i++) {
        for(int j = 0; j < matWidth; j++) {
            C_cpu[i * matWidth + j] = 0;
            for(int k = 0; k < matWidth; k++) {
                C_cpu[i * matWidth + j] += A[i * matWidth + k] * B[k * matWidth + j];
            }
        }
    }
    
    // Compare results
    bool matMulCorrect = true;
    const float epsilon = 1e-5f;
    for(int i = 0; i < 16; i++) {
        if(fabs(C_gpu[i] - C_cpu[i]) > epsilon) {
            matMulCorrect = false;
            break;
        }
    }
    
    printf("Matrix multiplication result: %s\n", matMulCorrect ? "Correct" : "Error");
    
    // Print matrix A
    printf("\nMatrix A:\n");
    for(int i = 0; i < matWidth; i++) {
        for(int j = 0; j < matWidth; j++) {
            printf("%6.1f ", A[i * matWidth + j]);
        }
        printf("\n");
    }
    
    // Print matrix B
    printf("\nMatrix B (Identity):\n");
    for(int i = 0; i < matWidth; i++) {
        for(int j = 0; j < matWidth; j++) {
            printf("%6.1f ", B[i * matWidth + j]);
        }
        printf("\n");
    }
    
    // Print results comparison
    printf("\nMatrix C (GPU vs CPU):\n");
    for(int i = 0; i < matWidth; i++) {
        for(int j = 0; j < matWidth; j++) {
            printf("%6.1f/%6.1f ", C_gpu[i * matWidth + j], C_cpu[i * matWidth + j]);
        }
        printf("\n");
    }
    
    // Results output for image processing
    printf("\nOriginal RGB image:\n");
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            int idx = (i * width + j) * 3;
            printf("(%3d,%3d,%3d) ", colorImage[idx], colorImage[idx+1], colorImage[idx+2]);
        }
        printf("\n");
    }
    
    printf("\nGrayscale result (GPU vs CPU):\n");
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            int idx = i * width + j;
            printf("%3d/%3d ", grayImage[idx], expectedGray[idx]);
        }
        printf("\n");
    }
    
    printf("\nr=%d Blur result (GPU vs CPU):\n", radius);
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            int idx = (i * width + j) * 3;
            printf("(%3d,%3d,%3d)/(%3d,%3d,%3d) ", 
                   blurredImage[idx], blurredImage[idx+1], blurredImage[idx+2],
                   expectedBlur[idx], expectedBlur[idx+1], expectedBlur[idx+2]);
        }
        printf("\n");
    }

    return 0;
}