#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    int width = 4, height = 4;
    
    // 간단한 4x4 RGB 이미지 (빨강, 초록, 파랑, 흰색)
    unsigned char colorImage[48] = {
        255,0,0,  0,255,0,  0,0,255,  255,255,255,  // 첫 번째 행
        255,0,0,  0,255,0,  0,0,255,  255,255,255,  // 두 번째 행  
        255,0,0,  0,255,0,  0,0,255,  255,255,255,  // 세 번째 행
        255,0,0,  0,255,0,  0,0,255,  255,255,255   // 네 번째 행
    };
    
    unsigned char grayImage[16];
    
    // GPU로 변환
    colorToGrayscale(colorImage, grayImage, width, height);
    
    // 결과 출력
    printf("원본 RGB 이미지:\n");
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            int idx = (i * width + j) * 3;
            printf("(%3d,%3d,%3d) ", colorImage[idx], colorImage[idx+1], colorImage[idx+2]);
        }
        printf("\n");
    }
    
    printf("\n그레이스케일 결과:\n");
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            int idx = i * width + j;
            printf("%3d ", grayImage[idx]);
        }
        printf("\n");
    }
    
    return 0;
}
