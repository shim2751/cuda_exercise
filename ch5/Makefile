NVCC = nvcc
TARGET = bin/ch5
SOURCES = matrix_mul.cu main.cpp

$(TARGET): $(SOURCES)
	mkdir -p bin
	$(NVCC) -O3 -arch=sm_75 $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: clean