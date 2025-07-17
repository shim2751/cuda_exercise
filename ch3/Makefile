NVCC = nvcc
TARGET = bin/color_to_grayscale

$(TARGET): color_to_grayscale.cu main.cpp
	mkdir -p bin
	$(NVCC) -O3 -arch=sm_75 color_to_grayscale.cu main.cpp -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: clean