NVCC = nvcc
TARGET = bin/ch3

$(TARGET): color_to_grayscale.cu image_blur.cu matrix_mul.cu main.cpp
	mkdir -p bin
	$(NVCC) -O3 -arch=sm_75 color_to_grayscale.cu image_blur.cu matrix_mul.cu main.cpp -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: clean